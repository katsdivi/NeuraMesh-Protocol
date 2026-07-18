//
//  main.swift
//  nmp-memory-peer — distributed-memory peer CLI (hackathon build, NEW)
//
//  Runs ONE memory peer: joins the NMP mesh over the existing encrypted
//  transport, holds erasure-coded shards of other peers' memories, and
//  exposes a tiny trusted-LAN HTTP control surface the demo harness drives.
//
//  Usage:
//    nmp-memory-peer --config path/to/config.json
//
//  Config is written by scripts/setup_memory_mesh.sh (per peer). It carries
//  this device's NMP identity, its roster of peers, the K-of-N scheme, and
//  the LOCAL Supermemory base URL + API key (never hardcoded; never a cloud
//  endpoint — NMPSupermemoryConfig rejects non-localhost URLs).
//
//  Control API (HTTP on controlPort, loopback only — same trusted-LAN
//  stance as the dashboard: no TLS, no auth, never port-forward):
//    GET  /                      → help
//    GET  /status                → node + link + supermemory status
//    GET  /memories              → memories this peer knows about (index)
//    POST /remember  {content,title}
//                                → seal, shard K-of-N, distribute; receipt
//    GET  /recall?q=<query>      → semantic recall + reconstruction, or an
//                                  explicit quorum-failure error
//    GET  /health                → 200 "ok" (liveness for the harness)
//

import Foundation
import Network
import NMP

// MARK: - Arguments

var configPath: String?
var useLocalStore = false
var localStoreDir: String?
do {
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let flag = it.next() {
        switch flag {
        case "--config": configPath = it.next()
        case "--local-store":
            // On-device backend (NMPLocalMemoryStore) instead of Supermemory —
            // the SAME code path the iPhone runs. Optional dir override.
            useLocalStore = true
            localStoreDir = it.next()
        case "--help", "-h":
            print("""
            usage: nmp-memory-peer --config path/to/config.json [--local-store [dir]]

            Runs one distributed-memory peer. See scripts/setup_memory_mesh.sh
            to generate per-peer config.json files for a 3-peer K=2 demo.

            --local-store [dir]  Back this peer with the on-device native store
                                 (NMPLocalMemoryStore: file blobs + on-device
                                 embeddings, no Supermemory server) — the exact
                                 backend an iPhone uses. Defaults to a
                                 'localstore' folder next to the config file.
            """)
            exit(0)
        default:
            FileHandle.standardError.write(Data("unknown flag \(flag)\n".utf8))
            exit(2)
        }
    }
}
guard let configPath else {
    FileHandle.standardError.write(Data("--config is required (see --help)\n".utf8))
    exit(2)
}

// MARK: - Node

let config: NMPMemoryPeerConfig
do {
    config = try NMPMemoryPeerConfig.load(path: configPath)
} catch {
    FileHandle.standardError.write(Data("config error: \(error)\n".utf8))
    exit(1)
}

func stamp() -> String {
    let f = DateFormatter(); f.dateFormat = "HH:mm:ss.SSS"
    return f.string(from: Date())
}

let node: NMPMemoryMeshNode
do {
    if useLocalStore {
        // On-device backend — identical to what the iPhone runs. No server.
        let dir = localStoreDir.map { ($0 as NSString).expandingTildeInPath }
            ?? ((configPath as NSString).deletingLastPathComponent + "/localstore")
        let store = NMPLocalMemoryStore(directory: URL(fileURLWithPath: dir))
        node = try NMPMemoryMeshNode(config: config, store: store)
        print("[mem \(stamp())] backend: on-device native store at \(dir)")
    } else {
        // Mac path: back the node with this device's localhost Supermemory
        // server. (iOS injects NMPLocalMemoryStore via init(config:store:).)
        node = try NMPMemoryMeshNode.withSupermemory(config: config)
    }
} catch {
    FileHandle.standardError.write(Data("failed to build node: \(error)\n".utf8))
    exit(1)
}
node.onStatus = { print("[mem \(stamp())] \($0)") }
node.onDiagnostic = { print("[mem \(stamp())] (diag) \($0)") }

do {
    try node.start()
} catch {
    FileHandle.standardError.write(Data("failed to start node: \(error)\n".utf8))
    exit(1)
}

// MARK: - Control HTTP server (loopback, trusted-LAN only)

/// Minimal blocking-per-connection HTTP/1.1 server. Callback style (house
/// rule): NWListener + NWConnection, same shape as NMPVaultServer.
final class ControlServer {
    private let port: UInt16
    private let queue = DispatchQueue(label: "nmp.memory.control")
    private var listener: NWListener?
    let handle: (_ method: String, _ path: String, _ query: [String: String],
                 _ body: Data, _ respond: @escaping (Int, Any) -> Void) -> Void

    init(port: UInt16,
         handle: @escaping (String, String, [String: String], Data,
                            @escaping (Int, Any) -> Void) -> Void) {
        self.port = port
        self.handle = handle
    }

    func start() throws {
        // Bind loopback explicitly — this surface is never for the LAN.
        let params = NWParameters.tcp
        params.requiredLocalEndpoint = NWEndpoint.hostPort(
            host: "127.0.0.1", port: NWEndpoint.Port(rawValue: port)!)
        let listener = try NWListener(using: params)
        listener.newConnectionHandler = { [weak self] c in self?.accept(c) }
        listener.start(queue: queue)
        self.listener = listener
    }

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        read(connection, buffer: Data())
    }

    private func read(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) {
            [weak self] data, _, isComplete, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            if let headerEnd = buffer.range(of: Data("\r\n\r\n".utf8))?.upperBound {
                let header = buffer.subdata(in: 0..<headerEnd)
                let contentLength = Self.contentLength(header)
                let bodySoFar = buffer.count - headerEnd
                if bodySoFar >= contentLength {
                    let body = buffer.subdata(in: headerEnd..<(headerEnd + contentLength))
                    self.dispatch(header: header, body: body, on: connection)
                    return
                }
            }
            if error != nil || isComplete || buffer.count > (1 << 21) {
                self.respond(connection, status: 400, body: Data("bad request\n".utf8))
                return
            }
            self.read(connection, buffer: buffer)
        }
    }

    private static func contentLength(_ header: Data) -> Int {
        guard let text = String(data: header, encoding: .utf8) else { return 0 }
        for line in text.split(separator: "\r\n") where
            line.lowercased().hasPrefix("content-length:") {
            return Int(line.split(separator: ":")[1]
                .trimmingCharacters(in: .whitespaces)) ?? 0
        }
        return 0
    }

    private func dispatch(header: Data, body: Data, on connection: NWConnection) {
        guard let line = String(data: header, encoding: .utf8)?
            .split(separator: "\r\n").first else {
            respond(connection, status: 400, body: Data("bad request\n".utf8)); return
        }
        let parts = line.split(separator: " ")
        guard parts.count >= 2 else {
            respond(connection, status: 400, body: Data("bad request\n".utf8)); return
        }
        let method = String(parts[0])
        let target = String(parts[1])
        let path = target.split(separator: "?").first.map(String.init) ?? target
        var query: [String: String] = [:]
        if let q = target.split(separator: "?").dropFirst().first {
            for pair in q.split(separator: "&") {
                let kv = pair.split(separator: "=", maxSplits: 1)
                let key = String(kv[0])
                let value = kv.count > 1 ? String(kv[1]) : ""
                query[key] = value.removingPercentEncoding ?? value
            }
        }
        handle(method, path, query, body) { [weak self] status, object in
            guard let self else { return }
            let data = (try? JSONSerialization.data(
                withJSONObject: object, options: [.prettyPrinted, .sortedKeys]))
                ?? Data("{}".utf8)
            self.respond(connection, status: status, body: data,
                         contentType: "application/json")
        }
    }

    private func respond(_ connection: NWConnection, status: Int, body: Data,
                         contentType: String = "text/plain; charset=utf-8") {
        let reason = [200: "OK", 400: "Bad Request", 404: "Not Found",
                      500: "Internal Server Error", 503: "Service Unavailable"][status]
            ?? "Status"
        var head = "HTTP/1.1 \(status) \(reason)\r\n"
        head += "Content-Length: \(body.count)\r\n"
        head += "Content-Type: \(contentType)\r\n"
        head += "Connection: close\r\n\r\n"
        var payload = Data(head.utf8); payload.append(body)
        connection.send(content: payload, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }
}

let help: [String: Any] = [
    "peer": "nmp-memory-peer",
    "peerID": Int(config.peerID),
    "endpoints": [
        "GET /status", "GET /memories", "POST /remember {content,title}",
        "GET /recall?q=…", "GET /health",
    ],
]

let control = ControlServer(port: config.controlPort) { method, path, query, body, respond in
    switch (method, path) {
    case ("GET", "/"):
        respond(200, help)

    case ("GET", "/health"):
        respond(200, ["ok": true, "peerID": Int(config.peerID)])

    case ("GET", "/status"):
        node.status { respond(200, $0) }

    case ("GET", "/memories"):
        node.listMemories { respond(200, ["memories": $0]) }

    case ("POST", "/remember"):
        let json = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any]
        guard let content = json?["content"] as? String, !content.isEmpty else {
            respond(400, ["error": "POST body needs a non-empty 'content' string"])
            return
        }
        let title = (json?["title"] as? String) ?? ""
        node.createMemory(content: content, title: title) { result in
            switch result {
            case .success(let receipt): respond(200, receipt)
            case .failure(let error):
                respond(error.httpStatus, ["error": error.label])
            }
        }

    case ("GET", "/recall"):
        guard let q = query["q"], !q.isEmpty else {
            respond(400, ["error": "need ?q=<query>"])
            return
        }
        node.recall(query: q) { result in
            switch result {
            case .success(let answer): respond(200, answer)
            case .failure(let error):
                respond(error.httpStatus, ["error": error.label,
                                           "query": q])
            }
        }

    default:
        respond(404, ["error": "no route \(method) \(path)"])
    }
}

do {
    try control.start()
    print("[mem \(stamp())] control API on http://127.0.0.1:\(config.controlPort) "
          + "(loopback only)")
} catch {
    FileHandle.standardError.write(Data("control server failed: \(error)\n".utf8))
    exit(1)
}

print("[mem \(stamp())] nmp-memory-peer \(config.peerID) ready — Ctrl+C to stop")
dispatchMain()
