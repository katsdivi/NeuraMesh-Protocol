//
//  VaultServer.swift
//  NMP — Future Plan #3: the weight vault (server side)
//
//  A tiny HTTP file server the COORDINATOR runs (it holds the full model).
//  A peer that holds no local model streams ONLY its assigned layers:
//
//      GET /vault?start=<a>&end=<b>   → a valid per-shard GGUF (blocks [a,b))
//      GET /vault/manifest           → {modelTag,hiddenSize,layerCount} JSON
//
//  Slices are produced in-process by NMPGGUFSlicer and cached by range, so
//  repeated joins (and re-shards) are cheap. Trusted-LAN only, like the
//  dashboard — no TLS/auth (see CLAUDE.md security stance).
//
//  Callback style, no async (house rule): NWListener + NWConnection.
//

import Foundation
import Network

public final class NMPVaultServer {

    public enum VaultError: Error, Sendable {
        case alreadyRunning
        case bindFailed(String)
        case startTimeout
    }

    public var onDiagnostic: ((String) -> Void)?
    public private(set) var boundPort: UInt16 = 0

    private let modelPath: String
    private let modelTag: String
    private let queue = DispatchQueue(label: "nmp.vault.server")
    private var listener: NWListener?

    /// Cached slice bytes by "start_end" (produced once, reused across joins).
    private var sliceCache: [String: Data] = [:]
    private let cacheLock = NSLock()

    public init(modelPath: String, modelTag: String) {
        self.modelPath = (modelPath as NSString).expandingTildeInPath
        self.modelTag = modelTag
    }

    /// Start on `port` (0 ⇒ an OS-chosen port; read `boundPort` after). Binds
    /// all interfaces so LAN peers can reach it.
    public func start(port: UInt16 = 0) throws {
        guard listener == nil else { throw VaultError.alreadyRunning }
        let nwPort = NWEndpoint.Port(rawValue: port) ?? .any
        let listener = try NWListener(using: .tcp, on: nwPort)
        let ready = DispatchSemaphore(value: 0)
        var failure: Error?

        listener.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                self?.boundPort = listener.port?.rawValue ?? port
                ready.signal()
            case .failed(let error):
                failure = VaultError.bindFailed(String(describing: error))
                ready.signal()
            default: break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }
        listener.start(queue: queue)
        self.listener = listener

        guard ready.wait(timeout: .now() + 5) == .success else {
            listener.cancel(); self.listener = nil
            throw VaultError.startTimeout
        }
        if let failure { listener.cancel(); self.listener = nil; throw failure }
        onDiagnostic?("vault serving \(modelTag) on port \(boundPort)")
    }

    public func stop() {
        listener?.cancel()
        listener = nil
    }

    // MARK: - Connection handling

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        readRequest(connection, buffer: Data())
    }

    /// Accumulate until the header terminator, then dispatch one request.
    private func readRequest(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) {
            [weak self] data, _, isComplete, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            if let terminator = Self.headerEnd(in: buffer) {
                self.handle(request: buffer.subdata(in: 0..<terminator), on: connection)
                return
            }
            if error != nil || isComplete || buffer.count > 64 * 1024 {
                self.respond(connection, status: "400 Bad Request", body: Data("bad request\n".utf8))
                return
            }
            self.readRequest(connection, buffer: buffer)
        }
    }

    private static func headerEnd(in data: Data) -> Int? {
        let sep = Data("\r\n\r\n".utf8)
        return data.range(of: sep)?.upperBound
    }

    private func handle(request: Data, on connection: NWConnection) {
        guard let line = String(data: request, encoding: .utf8)?
            .split(separator: "\r\n").first,
              line.hasPrefix("GET ") else {
            respond(connection, status: "400 Bad Request", body: Data("GET only\n".utf8))
            return
        }
        // "GET /vault?start=0&end=12 HTTP/1.1"
        let target = line.dropFirst(4).split(separator: " ").first.map(String.init) ?? "/"
        let path = target.split(separator: "?").first.map(String.init) ?? target

        if path == "/vault/manifest" {
            serveManifest(connection)
            return
        }
        if path == "/vault" {
            let params = Self.query(target)
            guard let start = params["start"].flatMap({ Int($0) }),
                  let end = params["end"].flatMap({ Int($0) }), start >= 0, end > start else {
                respond(connection, status: "400 Bad Request",
                        body: Data("need ?start=<a>&end=<b>\n".utf8))
                return
            }
            serveSlice(start: start, end: end, on: connection)
            return
        }
        respond(connection, status: "404 Not Found", body: Data("not found\n".utf8))
    }

    private static func query(_ target: String) -> [String: String] {
        guard let q = target.split(separator: "?").dropFirst().first else { return [:] }
        var out: [String: String] = [:]
        for pair in q.split(separator: "&") {
            let kv = pair.split(separator: "=", maxSplits: 1)
            if kv.count == 2 { out[String(kv[0])] = String(kv[1]) }
        }
        return out
    }

    private func serveManifest(_ connection: NWConnection) {
        do {
            let gguf = try NMPGGUFModel.load(path: modelPath)
            let json: [String: Any] = [
                "modelTag": modelTag,
                "hiddenSize": gguf.hiddenSize ?? 0,
                "layerCount": gguf.layerCount ?? 0,
            ]
            let body = try JSONSerialization.data(withJSONObject: json)
            respond(connection, status: "200 OK", body: body,
                    contentType: "application/json")
        } catch {
            respond(connection, status: "500 Internal Server Error",
                    body: Data("manifest failed: \(error)\n".utf8))
        }
    }

    private func serveSlice(start: Int, end: Int, on connection: NWConnection) {
        let key = "\(start)_\(end)"
        cacheLock.lock()
        let cached = sliceCache[key]
        cacheLock.unlock()
        if let cached {
            respond(connection, status: "200 OK", body: cached,
                    contentType: "application/octet-stream")
            return
        }
        do {
            let slice = try NMPGGUFSlicer.sliceData(
                modelPath: modelPath, start: start, end: end)
            cacheLock.lock(); sliceCache[key] = slice; cacheLock.unlock()
            onDiagnostic?("served shard [\(start),\(end)) — "
                          + "\(slice.count / 1_048_576) MB (of the full model)")
            respond(connection, status: "200 OK", body: slice,
                    contentType: "application/octet-stream")
        } catch {
            respond(connection, status: "500 Internal Server Error",
                    body: Data("slice failed: \(error)\n".utf8))
        }
    }

    private func respond(_ connection: NWConnection, status: String, body: Data,
                         contentType: String = "text/plain; charset=utf-8") {
        var header = "HTTP/1.1 \(status)\r\n"
        header += "Content-Length: \(body.count)\r\n"
        header += "Content-Type: \(contentType)\r\n"
        header += "Connection: close\r\n\r\n"
        var payload = Data(header.utf8)
        payload.append(body)
        connection.send(content: payload, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }
}
