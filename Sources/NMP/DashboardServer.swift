//
//  DashboardServer.swift
//  NMP — Phase 6
//
//  The testing dashboard's backend: a minimal HTTP + WebSocket server on
//  localhost, built directly on NWListener — same Apple-native,
//  zero-dependency rule as the rest of NMP. One port serves both the
//  static dashboard page (GET /) and the live update stream (GET /ws,
//  RFC 6455 upgrade).
//
//  Scope: a LOCAL testing tool. It binds loopback-reachable TCP with no
//  TLS and no auth — never expose it beyond the machine running the mesh.
//
//  Outbound messages (JSON over WebSocket text frames):
//    peer_update         — per-peer latency / load / assignment / liveness
//    inference_progress  — pipeline progress percent + stage label
//    packet_event        — FEC/NACK recovery events (the packet log)
//    mesh_event          — failover, re-shard, benchmark milestones
//    benchmark_result    — one scenario's summary numbers
//    loss_rate           — echo of the currently configured loss rate
//
//  Inbound control messages: set_loss_rate {rate}, inject_peer_drop,
//  start_benchmark, reset_metrics — surfaced via `onControl`.
//

import Foundation
import Network
import CryptoKit

// MARK: - WebSocket wire helpers (RFC 6455)

enum NMPWebSocketOpcode: UInt8 {
    case continuation = 0x0
    case text = 0x1
    case binary = 0x2
    case close = 0x8
    case ping = 0x9
    case pong = 0xA
}

struct NMPWebSocketFrame: Equatable {
    var opcode: NMPWebSocketOpcode
    var payload: Data
}

enum NMPWebSocketCodec {

    static let handshakeGUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    /// Sec-WebSocket-Accept = base64(SHA1(clientKey + GUID)).
    static func acceptKey(forClientKey key: String) -> String {
        let digest = Insecure.SHA1.hash(data: Data((key + handshakeGUID).utf8))
        return Data(digest).base64EncodedString()
    }

    /// Server→client frame: FIN set, never masked (RFC 6455 §5.1).
    static func encodeFrame(opcode: NMPWebSocketOpcode, payload: Data) -> Data {
        var out = Data()
        out.append(0x80 | opcode.rawValue)
        switch payload.count {
        case 0..<126:
            out.append(UInt8(payload.count))
        case 126...0xFFFF:
            out.append(126)
            out.appendBigEndian(UInt16(payload.count))
        default:
            out.append(127)
            out.appendBigEndian(UInt64(payload.count))
        }
        out.append(payload)
        return out
    }

    /// Decodes one frame from the front of `buffer`, consuming its bytes.
    /// Returns nil when the buffer holds only a partial frame. Client
    /// frames are masked per spec; unmasked frames are tolerated.
    static func decodeFrame(from buffer: inout Data) -> NMPWebSocketFrame? {
        let bytes = Data(buffer)
        guard bytes.count >= 2 else { return nil }
        guard let opcode = NMPWebSocketOpcode(rawValue: bytes[0] & 0x0F) else {
            // Unknown opcode: drop the connection's buffer (protocol error).
            buffer.removeAll()
            return nil
        }
        let masked = bytes[1] & 0x80 != 0
        var length = Int(bytes[1] & 0x7F)
        var cursor = 2
        if length == 126 {
            guard bytes.count >= 4 else { return nil }
            length = Int(bytes.readBigEndianUInt16(at: 2))
            cursor = 4
        } else if length == 127 {
            guard bytes.count >= 10 else { return nil }
            let full = bytes.readBigEndianUInt64(at: 2)
            guard full <= 1 << 20 else {         // 1 MB cap: control traffic only
                buffer.removeAll()
                return nil
            }
            length = Int(full)
            cursor = 10
        }
        var maskKey: [UInt8] = []
        if masked {
            guard bytes.count >= cursor + 4 else { return nil }
            maskKey = [bytes[cursor], bytes[cursor + 1], bytes[cursor + 2], bytes[cursor + 3]]
            cursor += 4
        }
        guard bytes.count >= cursor + length else { return nil }

        var payload = bytes.subdata(in: cursor..<cursor + length)
        if masked {
            for i in 0..<payload.count {
                payload[i] ^= maskKey[i % 4]
            }
        }
        buffer.removeSubrange(0..<(cursor + length))
        return NMPWebSocketFrame(opcode: opcode, payload: payload)
    }
}

// MARK: - Dashboard server

public final class NMPDashboardServer {

    public enum ControlMessage: Equatable, Sendable {
        case setLossRate(Double)
        case injectPeerDrop
        case startBenchmark
        case resetMetrics
    }

    public enum ServerError: Error {
        case alreadyRunning
        case bindFailed(String)
        case startTimeout
    }

    /// Control messages from any connected dashboard. Fires on the
    /// server's queue.
    public var onControl: ((ControlMessage) -> Void)?
    public var onDiagnostic: ((String) -> Void)?

    /// Actual bound port (differs from the requested one when passing 0).
    public private(set) var boundPort: UInt16 = 0
    /// Currently connected WebSocket client count (for tests/CLI status).
    public var clientCount: Int {
        queue.sync { clients.values.filter(\.isWebSocket).count }
    }

    private let queue = DispatchQueue(label: "nmp.dashboard.server")
    private var listener: NWListener?
    private var clients: [ObjectIdentifier: Client] = [:]

    private final class Client {
        let connection: NWConnection
        var buffer = Data()
        var isWebSocket = false
        init(connection: NWConnection) { self.connection = connection }
    }

    public init() {}

    deinit { stopNow() }

    // MARK: Lifecycle

    /// Binds and blocks until the listener is ready (or throws). Pass
    /// port 0 for an ephemeral port (tests); read `boundPort` after.
    public func start(port: UInt16 = 8080) throws {
        guard listener == nil else { throw ServerError.alreadyRunning }
        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            throw ServerError.bindFailed("invalid port \(port)")
        }
        let listener = try NWListener(using: .tcp, on: nwPort)
        let ready = DispatchSemaphore(value: 0)
        var failure: Error?

        listener.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                self?.boundPort = listener.port?.rawValue ?? port
                ready.signal()
            case .failed(let error):
                failure = ServerError.bindFailed(String(describing: error))
                ready.signal()
            default:
                break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in
            self?.queue.async { self?.accept(connection) }
        }
        listener.start(queue: queue)
        self.listener = listener

        guard ready.wait(timeout: .now() + 5) == .success else {
            listener.cancel()
            self.listener = nil
            throw ServerError.startTimeout
        }
        if let failure {
            listener.cancel()
            self.listener = nil
            throw failure
        }
        onDiagnostic?("dashboard listening on http://localhost:\(boundPort)")
    }

    public func stop() {
        queue.async { [self] in stopNow() }
    }

    private func stopNow() {
        listener?.cancel()
        listener = nil
        for client in clients.values { client.connection.cancel() }
        clients.removeAll()
    }

    // MARK: Broadcast API

    /// Sends a raw JSON string to every connected WebSocket client.
    public func broadcast(_ json: String) {
        queue.async { [self] in
            let frame = NMPWebSocketCodec.encodeFrame(
                opcode: .text, payload: Data(json.utf8))
            for client in clients.values where client.isWebSocket {
                client.connection.send(content: frame,
                                       completion: .contentProcessed { _ in })
            }
        }
    }

    private func broadcast(object: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: object),
              let json = String(data: data, encoding: .utf8) else { return }
        broadcast(json)
    }

    public func updatePeerState(peerID: UInt32, name: String, latencyMS: Int,
                                loadPercent: Int, assigned: String, alive: Bool) {
        broadcast(object: [
            "type": "peer_update",
            "peerID": String(peerID, radix: 16),
            "name": name,
            "latencyMS": latencyMS,
            "loadPercent": loadPercent,
            "assigned": assigned,
            "alive": alive,
        ])
    }

    public func updateInferenceProgress(progress: Double, stage: String) {
        broadcast(object: [
            "type": "inference_progress",
            "progress": Int((progress * 100).rounded()),
            "stage": stage,
        ])
    }

    public func reportPacketEvent(_ event: NMPPacketEvent, peerID: UInt32) {
        let (kind, sequences): (String, [UInt32])
        switch event {
        case .fecRecovered(let seq): (kind, sequences) = ("fec_recovered", [seq])
        case .nackSent(let seqs): (kind, sequences) = ("nack_sent", seqs)
        case .retransmitted(let seq): (kind, sequences) = ("retransmitted", [seq])
        case .unrecoverableLoss(let seqs): (kind, sequences) = ("unrecoverable_loss", seqs)
        }
        broadcast(object: [
            "type": "packet_event",
            "event": kind,
            "seq": sequences.map(Int.init),
            "peer": String(peerID, radix: 16),
        ])
    }

    public func reportMeshEvent(_ message: String) {
        broadcast(object: ["type": "mesh_event", "message": message])
    }

    public func reportBenchmarkResult(_ result: NMPBenchmarkResult) {
        broadcast(object: [
            "type": "benchmark_result",
            "name": result.name,
            "p50_ms": result.stats.p50 * 1000,
            "p95_ms": result.stats.p95 * 1000,
            "p99_ms": result.stats.p99 * 1000,
            "avg_ms": result.stats.average * 1000,
            "throughput": result.throughputTokensPerSecond,
            "notes": result.notes,
        ])
    }

    public func reportLossRate(_ rate: Double) {
        broadcast(object: ["type": "loss_rate", "rate": rate])
    }

    // MARK: Connection handling (all on `queue`)

    private func accept(_ connection: NWConnection) {
        let client = Client(connection: connection)
        clients[ObjectIdentifier(connection)] = client
        connection.stateUpdateHandler = { [weak self] state in
            if case .failed = state {
                self?.queue.async { self?.drop(connection) }
            } else if case .cancelled = state {
                self?.queue.async { self?.drop(connection) }
            }
        }
        connection.start(queue: queue)
        receive(on: client)
    }

    private func drop(_ connection: NWConnection) {
        clients.removeValue(forKey: ObjectIdentifier(connection))
    }

    private func receive(on client: Client) {
        client.connection.receive(minimumIncompleteLength: 1,
                                  maximumLength: 64 << 10) { [weak self, weak client] data, _, isComplete, error in
            guard let self, let client else { return }
            if let data, !data.isEmpty {
                client.buffer.append(data)
                self.processBuffer(of: client)
            }
            if isComplete || error != nil {
                client.connection.cancel()
                return
            }
            self.receive(on: client)
        }
    }

    private func processBuffer(of client: Client) {
        if client.isWebSocket {
            while let frame = NMPWebSocketCodec.decodeFrame(from: &client.buffer) {
                handleFrame(frame, from: client)
            }
        } else if let headerEnd = client.buffer.range(of: Data("\r\n\r\n".utf8)) {
            let head = String(decoding: Data(client.buffer[..<headerEnd.lowerBound]),
                              as: UTF8.self)
            client.buffer = Data(client.buffer[headerEnd.upperBound...])
            handleHTTPRequest(head: head, client: client)
        }
    }

    // MARK: HTTP

    private func handleHTTPRequest(head: String, client: Client) {
        let lines = head.split(separator: "\r\n", omittingEmptySubsequences: true)
        guard let requestLine = lines.first else {
            client.connection.cancel()
            return
        }
        let parts = requestLine.split(separator: " ")
        guard parts.count >= 2, parts[0] == "GET" else {
            respond(client, status: "405 Method Not Allowed", body: "GET only\n")
            return
        }
        let path = String(parts[1])

        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            headers[key] = value
        }

        if path == "/ws" {
            guard headers["upgrade"]?.lowercased() == "websocket",
                  let key = headers["sec-websocket-key"] else {
                respond(client, status: "400 Bad Request", body: "expected WebSocket upgrade\n")
                return
            }
            let accept = NMPWebSocketCodec.acceptKey(forClientKey: key)
            let response = "HTTP/1.1 101 Switching Protocols\r\n"
                + "Upgrade: websocket\r\n"
                + "Connection: Upgrade\r\n"
                + "Sec-WebSocket-Accept: \(accept)\r\n\r\n"
            client.connection.send(content: Data(response.utf8),
                                   completion: .contentProcessed { _ in })
            client.isWebSocket = true
            // Any frames that rode in behind the upgrade request.
            processBuffer(of: client)
            return
        }

        switch path {
        case "/", "/index.html", "/dashboard.html":
            respond(client, status: "200 OK", body: Self.dashboardHTML(),
                    contentType: "text/html; charset=utf-8")
        default:
            respond(client, status: "404 Not Found", body: "not found\n")
        }
    }

    private func respond(_ client: Client, status: String, body: String,
                         contentType: String = "text/plain; charset=utf-8") {
        let payload = Data(body.utf8)
        let head = "HTTP/1.1 \(status)\r\n"
            + "Content-Type: \(contentType)\r\n"
            + "Content-Length: \(payload.count)\r\n"
            + "Connection: close\r\n\r\n"
        var response = Data(head.utf8)
        response.append(payload)
        client.connection.send(content: response, completion: .contentProcessed { _ in
            client.connection.cancel()
        })
    }

    /// The dashboard page, from the SwiftPM resource bundle. A stub page
    /// is served if the bundle is unavailable (never a crash).
    static func dashboardHTML() -> String {
        if let url = Bundle.module.url(forResource: "dashboard", withExtension: "html"),
           let html = try? String(contentsOf: url, encoding: .utf8) {
            return html
        }
        return "<!DOCTYPE html><html><body><h1>NeuraMesh Dashboard</h1>"
            + "<p>dashboard.html resource missing from bundle</p></body></html>"
    }

    // MARK: WebSocket inbound

    private func handleFrame(_ frame: NMPWebSocketFrame, from client: Client) {
        switch frame.opcode {
        case .text:
            handleControlJSON(frame.payload)
        case .ping:
            let pong = NMPWebSocketCodec.encodeFrame(opcode: .pong, payload: frame.payload)
            client.connection.send(content: pong, completion: .contentProcessed { _ in })
        case .close:
            let close = NMPWebSocketCodec.encodeFrame(opcode: .close, payload: Data())
            client.connection.send(content: close, completion: .contentProcessed { _ in
                client.connection.cancel()
            })
        default:
            break
        }
    }

    private func handleControlJSON(_ payload: Data) {
        guard let object = try? JSONSerialization.jsonObject(with: payload) as? [String: Any],
              let type = object["type"] as? String else {
            onDiagnostic?("dashboard: unparseable control message")
            return
        }
        switch type {
        case "set_loss_rate":
            if let rate = object["rate"] as? Double {
                onControl?(.setLossRate(rate.clamped(to: 0...1)))
            }
        case "inject_peer_drop":
            onControl?(.injectPeerDrop)
        case "start_benchmark":
            onControl?(.startBenchmark)
        case "reset_metrics":
            onControl?(.resetMetrics)
        default:
            onDiagnostic?("dashboard: unknown control type '\(type)'")
        }
    }
}
