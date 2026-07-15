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
//  REST: POST /api/inference {prompt, max_tokens} runs a prompt through
//  the mesh via `onInferenceRequest` and returns generated text + measured
//  metrics as JSON. 503 when no handler is wired, 429 while a generation
//  is in flight. No CORS headers: the expected caller is a server-side
//  process (the NeuraMesh web app's API route), not a browser page.
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

    /// One parsed POST /api/inference request.
    public struct InferenceRequest: Equatable, Sendable {
        public let prompt: String
        public let maxTokens: Int
        /// Phase 9: {"enable_speculation": true} — served by the
        /// speculative path when the CLI wired one up.
        public let enableSpeculation: Bool
        /// Mesh 2.0: {"enable_comparison": true} — attach the protocol
        /// comparison (measured NMP + modeled TCP/QUIC) to the response.
        public let enableComparison: Bool

        public init(prompt: String, maxTokens: Int,
                    enableSpeculation: Bool = false,
                    enableComparison: Bool = false) {
            self.prompt = prompt
            self.maxTokens = maxTokens
            self.enableSpeculation = enableSpeculation
            self.enableComparison = enableComparison
        }
    }

    /// What the inference handler reports back; the server serializes it.
    public enum InferenceResponse {
        case success(NMPPromptInferenceService.GenerationResult)
        case failure(status: Int, message: String)
    }

    /// Control messages from any connected dashboard. Fires on the
    /// server's queue.
    public var onControl: ((ControlMessage) -> Void)?
    public var onDiagnostic: ((String) -> Void)?
    /// Handler for POST /api/inference. Fires on the server's queue; the
    /// reply callback may be invoked from any queue, exactly once.
    public var onInferenceRequest: ((InferenceRequest, @escaping (InferenceResponse) -> Void) -> Void)?

    // MARK: Mesh 2.0 web UI surface

    /// One parsed POST /api/benchmark/run request.
    public struct BenchmarkRequest: Equatable, Sendable {
        public let prompt: String
        public let maxTokens: Int
        public let runs: Int

        public init(prompt: String, maxTokens: Int, runs: Int) {
            self.prompt = prompt
            self.maxTokens = maxTokens
            self.runs = runs
        }
    }

    /// Why a benchmark run failed, surfaced verbatim to the API caller.
    public struct BenchmarkFailure: Error, Sendable {
        public let message: String
        public init(_ message: String) { self.message = message }
    }

    /// Handler for POST /api/benchmark/run: run `runs` sequential
    /// generations and return each result (or one failure). Fires on
    /// the server's queue; reply from any queue, exactly once.
    public var onBenchmarkRequest: ((BenchmarkRequest,
        @escaping (Result<[NMPPromptInferenceService.GenerationResult],
                          BenchmarkFailure>) -> Void) -> Void)?

    // MARK: Mesh 2.1 surface

    /// What POST /api/comparison/run hands back: one real generation plus
    /// the transport race that replayed its traffic pattern over real
    /// loopback sockets (see TransportRace.swift).
    public struct ComparisonRunOutcome {
        public let generation: NMPPromptInferenceService.GenerationResult
        public let race: NMPTransportRace.RaceResult
        public init(generation: NMPPromptInferenceService.GenerationResult,
                    race: NMPTransportRace.RaceResult) {
            self.generation = generation
            self.race = race
        }
    }

    /// Handler for POST /api/comparison/run. Fires on the server's
    /// queue; reply from any queue, exactly once.
    public var onComparisonRunRequest: ((InferenceRequest,
        @escaping (Result<ComparisonRunOutcome, BenchmarkFailure>) -> Void) -> Void)?

    /// Handler for GET /api/devices/metrics: compose the live resource
    /// picture (host sample + per-peer mesh facts). The server ships the
    /// object verbatim as JSON. 503 when unwired.
    public var onDeviceMetricsRequest: ((@escaping ([String: Any]) -> Void) -> Void)?

    /// Handler for POST /api/devices/<hexID>/allocate {"share": 0...1}.
    /// Reply with a human-readable summary of what the new share did
    /// (e.g. the re-planned layer spans) or a failure. Fires on the
    /// server's queue; reply from any queue, exactly once.
    public var onAllocationRequest: ((UInt32, Double,
        @escaping (Result<String, BenchmarkFailure>) -> Void) -> Void)?

    /// Handler for POST /api/mesh/objective {"objective": "..."}. Switches
    /// the sharding strategy and re-shards the live mesh; reply with a
    /// human summary. Fires on the server's queue; reply from any queue.
    public var onObjectiveRequest: ((String,
        @escaping (Result<String, BenchmarkFailure>) -> Void) -> Void)?

    /// Handler for GET /api/models: the installed models with compatibility
    /// flags (compatible / fits this host / recommended / active). The server
    /// ships the array verbatim as JSON under "models". 503 when unwired.
    public var onModelsListRequest: ((@escaping ([[String: Any]]) -> Void) -> Void)?

    /// Handler for POST /api/models/select {"path": "..."}. Switch the active
    /// model — the CLI relaunches itself onto it. Reply with a human summary
    /// (the UI shows a brief "reconnecting…" while the mesh restarts) or a
    /// failure reason (incompatible / won't fit / not found). Fires on the
    /// server's queue; reply from any queue, exactly once.
    public var onModelSelectRequest: ((String,
        @escaping (Result<String, BenchmarkFailure>) -> Void) -> Void)?

    /// Static facts about the running mesh, surfaced by GET /health and
    /// the UI's dashboard. Set once by the CLI after assembly.
    public struct MeshInfo: Sendable {
        public var engine = "reference"
        public var modelName = ""
        public var shardCount = 0
        public var wireFormat = "float32"
        public var speculationAvailable = false

        public init() {}
    }

    public var meshInfo: MeshInfo {
        get { queue.sync { storedMeshInfo } }
        set { queue.async { self.storedMeshInfo = newValue } }
    }

    /// Directory of the built web UI (web/ → Public/). When set, GET
    /// serves files from it with an index.html fallback for SPA routes;
    /// when nil, the legacy embedded dashboard page is served at /.
    public var publicDirectory: URL? {
        get { queue.sync { storedPublicDirectory } }
        set { queue.async { self.storedPublicDirectory = newValue } }
    }

    private var storedMeshInfo = MeshInfo()
    private var storedPublicDirectory: URL?

    /// Latest state per peer — the same facts `updatePeerState` pushes
    /// over the WebSocket, retained for GET /api/devices.
    struct PeerSnapshot {
        var name: String
        var latencyMS: Int
        var loadPercent: Int
        var assigned: String
        var alive: Bool
    }
    private var peerSnapshots: [UInt32: PeerSnapshot] = [:]

    /// Mesh 2.1: web clients (browsers) seen recently, keyed by
    /// address + User-Agent. HTTP connections are one-shot (Connection:
    /// close), so "connected" means: holding the WebSocket open, or any
    /// HTTP request within `webClientWindow` (the UI polls every 2-3 s).
    struct BrowserSighting {
        var userAgent: String
        var lastSeen: Date
        /// Currently holding /ws open (dropped WS connections flip this
        /// back to false and start the staleness clock).
        var webSocketCount = 0
    }
    private var browserSightings: [String: BrowserSighting] = [:]
    /// Seconds an HTTP-only client stays "connected" after its last
    /// request. 15 s ≈ five UI polling intervals.
    public static let webClientWindow: TimeInterval = 15

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
        /// Parsed request head while its body is still arriving.
        var pendingHead: String?
        var pendingBodyLength = 0
        /// Mesh 2.1: browserSightings key this client was counted under
        /// (set once its first request head is parsed).
        var sightingKey: String?
        init(connection: NWConnection) { self.connection = connection }

        /// The remote address, for the web-clients list ("who is looking
        /// at the dashboard"). IPv6 zone suffixes stripped for stability.
        var remoteAddress: String {
            guard case let .hostPort(host, _) = connection.endpoint else {
                return "unknown"
            }
            let raw: String
            switch host {
            case .ipv4(let address): raw = "\(address)"
            case .ipv6(let address): raw = "\(address)"
            case .name(let name, _): raw = name
            @unknown default: raw = "unknown"
            }
            return raw.split(separator: "%").first.map(String.init) ?? raw
        }
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
        queue.async { [self] in
            peerSnapshots[peerID] = PeerSnapshot(
                name: name, latencyMS: latencyMS, loadPercent: loadPercent,
                assigned: assigned, alive: alive)
        }
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

    // MARK: Mesh 2.1 broadcasts

    /// A generation began (any client, any device sees it start).
    public func reportGenerationStarted(prompt: String, maxTokens: Int,
                                        speculative: Bool) {
        broadcast(object: [
            "type": "generation_started",
            "prompt": prompt,
            "max_tokens": maxTokens,
            "speculative": speculative,
        ])
    }

    /// One confirmed token, streamed live to every open browser.
    public func reportGenerationToken(text: String, index: Int,
                                      count: Int, requested: Int) {
        broadcast(object: [
            "type": "generation_token",
            "text": text,
            "index": index,
            "count": count,
            "requested": requested,
        ])
    }

    /// The generation finished; ships the same metrics /api/inference
    /// returns so every browser converges on identical numbers.
    public func reportGenerationComplete(
        _ result: NMPPromptInferenceService.GenerationResult
    ) {
        let tokensPerSec = result.totalSeconds > 0
            ? Double(result.tokenCount) / result.totalSeconds : 0
        var object: [String: Any] = [
            "type": "generation_complete",
            "output": result.text,
            "token_count": result.tokenCount,
            "latency_ms": (result.totalSeconds * 1000 * 10).rounded() / 10,
            "tokens_per_sec": (tokensPerSec * 100).rounded() / 100,
            "network_payload_bytes": result.networkPayloadBytes,
            "round_trips": result.speculation?.meshRoundTrips
                ?? result.perTokenSeconds.count,
            "engine": result.engine,
        ]
        if let stats = result.speculation {
            object["acceptance_rate"] = (stats.acceptanceRate * 1000).rounded() / 1000
        }
        broadcast(object: object)
    }

    public func reportGenerationFailed(_ message: String) {
        broadcast(object: ["type": "generation_failed", "error": message])
    }

    /// A device's compute share changed (slider moved on SOME device —
    /// every other device's slider follows).
    public func reportAllocation(peerID: UInt32, share: Double) {
        broadcast(object: [
            "type": "allocation_update",
            "peer": String(peerID, radix: 16),
            "share": (share * 100).rounded() / 100,
        ])
    }

    /// Web-client count changed (a browser joined or left). Deliberately
    /// DELAYED: this fires on the accept/upgrade path, and CFNetwork's
    /// WebSocket client (URLSessionWebSocketTask, Safari) fails the
    /// handshake when a server frame shares a TCP segment with the 101
    /// response — the delay gives the upgrade its own segment. A count
    /// update is not latency-sensitive.
    private func broadcastWebClientCount() {
        queue.asyncAfter(deadline: .now() + 0.25) { [weak self] in
            guard let self else { return }
            self.broadcast(object: [
                "type": "client_update",
                "web_clients": self.activeWebClientsLocked().count,
            ])
        }
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
        guard let client = clients.removeValue(forKey: ObjectIdentifier(connection)) else {
            return
        }
        // A departing WebSocket client leaves immediately (HTTP-only
        // clients age out through the staleness window instead).
        if client.isWebSocket, let key = client.sightingKey {
            browserSightings[key]?.webSocketCount -= 1
            browserSightings[key]?.lastSeen = Date()
            broadcastWebClientCount()
        }
    }

    // MARK: Web client tracking (all on `queue`)

    private func recordSighting(client: Client, userAgent: String) {
        let key = client.remoteAddress + "|" + String(userAgent.prefix(64))
        let wasActive = activeWebClientsLocked().contains { $0.key == key }
        var sighting = browserSightings[key]
            ?? BrowserSighting(userAgent: userAgent, lastSeen: Date())
        sighting.lastSeen = Date()
        browserSightings[key] = sighting
        client.sightingKey = key
        if !wasActive { broadcastWebClientCount() }
    }

    private func markWebSocket(client: Client) {
        guard let key = client.sightingKey else { return }
        browserSightings[key]?.webSocketCount += 1
        broadcastWebClientCount()
    }

    /// Currently "connected" web clients: WebSocket holders plus anyone
    /// who made an HTTP request within the window. Also prunes entries
    /// dead for over five minutes so the map stays bounded.
    private func activeWebClientsLocked() -> [(key: String, sighting: BrowserSighting)] {
        let now = Date()
        browserSightings = browserSightings.filter {
            $0.value.webSocketCount > 0
                || now.timeIntervalSince($0.value.lastSeen) < 300
        }
        return browserSightings
            .filter {
                $0.value.webSocketCount > 0
                    || now.timeIntervalSince($0.value.lastSeen) < Self.webClientWindow
            }
            .sorted { $0.value.lastSeen > $1.value.lastSeen }
            .map { (key: $0.key, sighting: $0.value) }
    }

    /// Active web-client count (for tests/CLI status).
    public var webClientCount: Int {
        queue.sync { activeWebClientsLocked().count }
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
        } else {
            tryHandleHTTP(client)
        }
    }

    // MARK: HTTP

    /// 1 MB body cap — prompts are small; anything bigger is a bug or abuse.
    private static let maxBodyBytes = 1 << 20

    /// Dispatches a request once its head AND full body are buffered.
    private func tryHandleHTTP(_ client: Client) {
        // A head was parsed earlier; we were waiting on body bytes.
        if let head = client.pendingHead {
            guard client.buffer.count >= client.pendingBodyLength else { return }
            let body = client.buffer.prefix(client.pendingBodyLength)
            client.buffer = Data(client.buffer.dropFirst(client.pendingBodyLength))
            client.pendingHead = nil
            handleHTTPRequest(head: head, body: Data(body), client: client)
            return
        }
        guard let headerEnd = client.buffer.range(of: Data("\r\n\r\n".utf8)) else { return }
        let head = String(decoding: Data(client.buffer[..<headerEnd.lowerBound]),
                          as: UTF8.self)
        client.buffer = Data(client.buffer[headerEnd.upperBound...])

        let contentLength = Self.contentLength(inHead: head)
        guard contentLength <= Self.maxBodyBytes else {
            respond(client, status: "413 Payload Too Large", body: "body too large\n")
            return
        }
        if client.buffer.count >= contentLength {
            let body = client.buffer.prefix(contentLength)
            client.buffer = Data(client.buffer.dropFirst(contentLength))
            handleHTTPRequest(head: head, body: Data(body), client: client)
        } else {
            client.pendingHead = head
            client.pendingBodyLength = contentLength
        }
    }

    private static func contentLength(inHead head: String) -> Int {
        for line in head.split(separator: "\r\n").dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            if key == "content-length" {
                return Int(line[line.index(after: colon)...]
                    .trimmingCharacters(in: .whitespaces)) ?? 0
            }
        }
        return 0
    }

    private func handleHTTPRequest(head: String, body: Data, client: Client) {
        let lines = head.split(separator: "\r\n", omittingEmptySubsequences: true)
        guard let requestLine = lines.first else {
            client.connection.cancel()
            return
        }
        let parts = requestLine.split(separator: " ")
        guard parts.count >= 2 else {
            respond(client, status: "400 Bad Request", body: "malformed request line\n")
            return
        }
        let method = String(parts[0])
        let path = String(parts[1])

        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            headers[key] = value
        }

        // CORS preflight (a Vite dev server or another origin's page may
        // call the API cross-origin on the trusted LAN).
        if method == "OPTIONS" {
            respond(client, status: "204 No Content", body: "")
            return
        }

        // Mesh 2.1: every HTTP request marks its sender as a live web
        // client (the "does my iPhone show up?" fix).
        recordSighting(client: client,
                       userAgent: headers["user-agent"] ?? "unknown")

        if method == "POST" {
            if path.hasPrefix("/api/devices/") && path.hasSuffix("/allocate") {
                handleAllocatePOST(path: path, body: body, client: client)
                return
            }
            switch path {
            case "/api/inference":
                handleInferencePOST(body: body, client: client)
            case "/api/chat":
                handleChatPOST(body: body, client: client)
            case "/api/mesh/objective":
                handleObjectivePOST(body: body, client: client)
            case "/api/benchmark/run":
                handleBenchmarkPOST(body: body, client: client)
            case "/api/comparison":
                handleComparisonPOST(body: body, client: client)
            case "/api/comparison/run":
                handleComparisonRunPOST(body: body, client: client)
            case "/api/models/select":
                handleModelSelectPOST(body: body, client: client)
            default:
                respond(client, status: "404 Not Found", body: "not found\n")
            }
            return
        }
        guard method == "GET" else {
            respond(client, status: "405 Method Not Allowed", body: "GET or POST only\n")
            return
        }

        switch path {
        case "/health":
            handleHealthGET(client)
            return
        case "/api/devices":
            handleDevicesGET(client)
            return
        case "/api/devices/metrics":
            handleDeviceMetricsGET(client)
            return
        case "/api/clients":
            handleClientsGET(client)
            return
        case "/api/models":
            handleModelsGET(client)
            return
        default:
            break
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
            markWebSocket(client: client)
            // Any frames that rode in behind the upgrade request.
            processBuffer(of: client)
            return
        }

        // Static web UI (Mesh 2.0): serve from publicDirectory when set,
        // with index.html as the SPA fallback for non-file routes. The
        // legacy embedded page stays reachable at /legacy either way.
        if path == "/legacy" || path == "/dashboard.html" {
            respond(client, status: "200 OK", body: Self.dashboardHTML(),
                    contentType: "text/html; charset=utf-8")
            return
        }
        if let publicDirectory = storedPublicDirectory {
            serveStatic(path: path, from: publicDirectory, client: client)
            return
        }
        switch path {
        case "/", "/index.html":
            respond(client, status: "200 OK", body: Self.dashboardHTML(),
                    contentType: "text/html; charset=utf-8")
        default:
            respond(client, status: "404 Not Found", body: "not found\n")
        }
    }

    // MARK: Static files (SPA)

    private static let mimeTypes: [String: String] = [
        "html": "text/html; charset=utf-8",
        "js": "text/javascript; charset=utf-8",
        "css": "text/css; charset=utf-8",
        "json": "application/json",
        "svg": "image/svg+xml",
        "png": "image/png",
        "ico": "image/x-icon",
        "webp": "image/webp",
        "woff2": "font/woff2",
        "map": "application/json",
        "txt": "text/plain; charset=utf-8",
        "webmanifest": "application/manifest+json",
    ]

    private func serveStatic(path: String, from root: URL, client: Client) {
        // /api paths never fall through to the SPA.
        if path.hasPrefix("/api") {
            respondJSON(client, status: "404 Not Found", object: ["error": "not found"])
            return
        }
        let rawPath = path.split(separator: "?").first.map(String.init) ?? path
        let relative = rawPath == "/" ? "index.html"
            : String(rawPath.dropFirst().removingPercentEncoding ?? String(rawPath.dropFirst()))
        let candidate = root.appendingPathComponent(relative).standardizedFileURL

        // Traversal guard: the resolved file must stay inside the root.
        let rootPath = root.standardizedFileURL.path
        guard candidate.path == rootPath
                || candidate.path.hasPrefix(rootPath + "/") else {
            respond(client, status: "403 Forbidden", body: "forbidden\n")
            return
        }

        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(
            atPath: candidate.path, isDirectory: &isDirectory)
        let fileURL: URL
        if exists && !isDirectory.boolValue {
            fileURL = candidate
        } else {
            // SPA fallback: unknown (extension-less) routes get the app
            // shell; missing assets stay honest 404s.
            guard !relative.contains(".") else {
                respond(client, status: "404 Not Found", body: "not found\n")
                return
            }
            fileURL = root.appendingPathComponent("index.html")
        }

        guard let contents = try? Data(contentsOf: fileURL) else {
            respond(client, status: "404 Not Found", body: "not found\n")
            return
        }
        let contentType = Self.mimeTypes[fileURL.pathExtension.lowercased()]
            ?? "application/octet-stream"
        respondData(client, status: "200 OK", payload: contents,
                    contentType: contentType)
    }

    // MARK: GET /health + /api/devices

    private func handleHealthGET(_ client: Client) {
        let info = storedMeshInfo
        respondJSON(client, status: "200 OK", object: [
            "status": "ok",
            // Which machine the PWA just found ("Connected to <host>").
            "hostname": NMPLANIdentity.localHostname(),
            "mesh": [
                "engine": info.engine,
                "model": info.modelName,
                "shard_count": info.shardCount,
                "wire_format": info.wireFormat,
                "speculation_available": info.speculationAvailable,
                "peers": peerSnapshots.count,
                "peers_alive": peerSnapshots.values.filter(\.alive).count,
                "web_clients": activeWebClientsLocked().count,
            ],
        ])
    }

    // MARK: GET /api/clients + /api/devices/metrics (Mesh 2.1)

    private func handleClientsGET(_ client: Client) {
        let now = Date()
        let clients = activeWebClientsLocked().map { key, sighting -> [String: Any] in
            [
                "address": key.split(separator: "|").first.map(String.init) ?? "unknown",
                "user_agent": sighting.userAgent,
                "websocket": sighting.webSocketCount > 0,
                "seconds_since_seen":
                    (now.timeIntervalSince(sighting.lastSeen) * 10).rounded() / 10,
            ]
        }
        respondJSONArray(client, status: "200 OK", array: clients)
    }

    private func handleDeviceMetricsGET(_ client: Client) {
        guard let handler = onDeviceMetricsRequest else {
            respondJSON(client, status: "503 Service Unavailable",
                        object: ["error": "no device metrics source is wired to this server"])
            return
        }
        handler { [weak self, weak client] object in
            guard let self, let client else { return }
            self.queue.async {
                self.respondJSON(client, status: "200 OK", object: object)
            }
        }
    }

    // MARK: POST /api/devices/<hexID>/allocate

    private func handleObjectivePOST(body: Data, client: Client) {
        guard let handler = onObjectiveRequest else {
            respondJSON(client, status: "503 Service Unavailable",
                        object: ["error": "no objective handler is wired to this server"])
            return
        }
        guard let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let objective = object["objective"] as? String,
              !objective.isEmpty else {
            respondJSON(client, status: "400 Bad Request",
                        object: ["error": "body must be JSON with a non-empty 'objective'"])
            return
        }
        handler(objective) { [weak self, weak client] result in
            guard let self, let client else { return }
            self.queue.async {
                switch result {
                case .success(let summary):
                    self.respondJSON(client, status: "200 OK", object: [
                        "status": "ok",
                        "objective": objective,
                        "summary": summary,
                    ])
                case .failure(let failure):
                    self.respondJSON(client, status: "400 Bad Request",
                                     object: ["error": failure.message])
                }
            }
        }
    }

    // MARK: GET /api/models + POST /api/models/select

    private func handleModelsGET(_ client: Client) {
        guard let handler = onModelsListRequest else {
            respondJSON(client, status: "503 Service Unavailable",
                        object: ["error": "no model catalog is wired to this server"])
            return
        }
        handler { [weak self, weak client] models in
            guard let self, let client else { return }
            self.queue.async {
                self.respondJSON(client, status: "200 OK", object: ["models": models])
            }
        }
    }

    private func handleModelSelectPOST(body: Data, client: Client) {
        guard let handler = onModelSelectRequest else {
            respondJSON(client, status: "503 Service Unavailable",
                        object: ["error": "model switching is available only in the sharded engine (--engine llamaShard)"])
            return
        }
        guard let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let path = object["path"] as? String, !path.isEmpty else {
            respondJSON(client, status: "400 Bad Request",
                        object: ["error": "body must be JSON with a non-empty 'path'"])
            return
        }
        handler(path) { [weak self, weak client] result in
            guard let self, let client else { return }
            self.queue.async {
                switch result {
                case .success(let summary):
                    self.respondJSON(client, status: "200 OK", object: [
                        "status": "ok",
                        "path": path,
                        "summary": summary,
                        // The mesh relaunches onto the new model; the UI should
                        // show "reconnecting…" and poll /health until it's back.
                        "reconnecting": true,
                    ])
                case .failure(let failure):
                    self.respondJSON(client, status: "400 Bad Request",
                                     object: ["error": failure.message])
                }
            }
        }
    }

    private func handleAllocatePOST(path: String, body: Data, client: Client) {
        guard let handler = onAllocationRequest else {
            respondJSON(client, status: "503 Service Unavailable",
                        object: ["error": "no allocation handler is wired to this server"])
            return
        }
        // /api/devices/<hexID>/allocate
        let segments = path.split(separator: "/").map(String.init)
        guard segments.count == 4, let peerID = UInt32(segments[2], radix: 16) else {
            respondJSON(client, status: "400 Bad Request",
                        object: ["error": "path must be /api/devices/<hex peer id>/allocate"])
            return
        }
        guard let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let share = (object["share"] as? Double)
                ?? (object["share"] as? Int).map(Double.init),
              share > 0, share <= 1 else {
            respondJSON(client, status: "400 Bad Request",
                        object: ["error": "body must be JSON with 'share' in (0, 1]"])
            return
        }
        handler(peerID, share) { [weak self, weak client] result in
            guard let self, let client else { return }
            self.queue.async {
                switch result {
                case .success(let summary):
                    self.reportAllocation(peerID: peerID, share: share)
                    self.respondJSON(client, status: "200 OK", object: [
                        "status": "ok",
                        "peer": String(peerID, radix: 16),
                        "share": share,
                        "summary": summary,
                    ])
                case .failure(let failure):
                    self.respondJSON(client, status: "500 Internal Server Error",
                                     object: ["error": failure.message])
                }
            }
        }
    }

    // MARK: POST /api/comparison/run (measured transport race)

    private func handleComparisonRunPOST(body: Data, client: Client) {
        guard let handler = onComparisonRunRequest else {
            respondJSON(client, status: "503 Service Unavailable",
                        object: ["error": "no comparison pipeline is wired to this server"])
            return
        }
        guard let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let prompt = object["prompt"] as? String,
              !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            respondJSON(client, status: "400 Bad Request",
                        object: ["error": "body must be JSON with a non-empty 'prompt'"])
            return
        }
        let request = InferenceRequest(
            prompt: prompt,
            maxTokens: (object["max_tokens"] as? Int) ?? 32,
            enableSpeculation: (object["enable_speculation"] as? Bool) ?? false)

        handler(request) { [weak self, weak client] result in
            guard let self, let client else { return }
            self.queue.async {
                switch result {
                case .failure(let failure):
                    self.respondJSON(client, status: "500 Internal Server Error",
                                     object: ["error": failure.message])
                case .success(let outcome):
                    self.respondJSON(client, status: "200 OK",
                                     object: Self.comparisonRunJSON(outcome))
                }
            }
        }
    }

    /// Serializes a comparison run: the real generation, the measured
    /// transport race, and one "splice" projection per raced transport —
    /// the generation's wall clock with the NMP leg's measured transport
    /// time swapped for that leg's measured transport time. Every input
    /// to a splice is a measurement; the splice itself is arithmetic,
    /// and says so.
    static func comparisonRunJSON(_ outcome: ComparisonRunOutcome) -> [String: Any] {
        let generation = outcome.generation
        let race = outcome.race
        let generationMs = generation.totalSeconds * 1000
        let tokensPerSec: (Double) -> Double = { totalMs in
            totalMs > 0 ? Double(generation.tokenCount) / (totalMs / 1000) : 0
        }

        let projected = projectedJSON(race: race, generationMs: generationMs,
                                      tokenCount: generation.tokenCount)

        return [
            "note": "generation and every race leg are measured; each "
                + "projection is the measured generation wall clock with "
                + "NMP's measured transport time replaced by that leg's",
            "generation": [
                "output": generation.text,
                "token_count": generation.tokenCount,
                "latency_ms": round2(generationMs),
                "tokens_per_sec": round2(tokensPerSec(generationMs)),
                "network_payload_bytes": generation.networkPayloadBytes,
                "round_trips": generation.speculation?.meshRoundTrips
                    ?? generation.perTokenSeconds.count,
                "engine": generation.engine,
            ] as [String: Any],
            "race": race.asJSONObject,
            "projected": projected,
        ]
    }

    /// Whole-generation totals per raced transport: element 0 is the
    /// measured run; the rest swap NMP's measured transport time for
    /// each leg's measured transport time (arithmetic on measurements).
    static func projectedJSON(race: NMPTransportRace.RaceResult,
                              generationMs: Double,
                              tokenCount: Int) -> [[String: Any]] {
        let tokensPerSec: (Double) -> Double = { totalMs in
            totalMs > 0 ? Double(tokenCount) / (totalMs / 1000) : 0
        }
        return [[
            "name": "NMP (the run itself)",
            "total_ms": round2(generationMs),
            "tokens_per_sec": round2(tokensPerSec(generationMs)),
            "basis": "measured",
        ]] + race.legs.dropFirst().map { leg in
            let splicedMs = generationMs - race.nmp.totalMs + leg.totalMs
            return [
                "name": "\(leg.name) transport splice",
                "total_ms": round2(splicedMs),
                "tokens_per_sec": round2(tokensPerSec(splicedMs)),
                "basis": "measured splice",
            ]
        }
    }

    private func handleDevicesGET(_ client: Client) {
        let devices = peerSnapshots
            .sorted { $0.key < $1.key }
            .map { peerID, snapshot -> [String: Any] in
                [
                    "id": String(peerID, radix: 16),
                    "name": snapshot.name,
                    "latency_ms": snapshot.latencyMS,
                    "load_percent": snapshot.loadPercent,
                    "assigned": snapshot.assigned,
                    "alive": snapshot.alive,
                ]
            }
        respondJSONArray(client, status: "200 OK", array: devices)
    }

    // MARK: POST /api/benchmark/run

    private func handleBenchmarkPOST(body: Data, client: Client) {
        guard let handler = onBenchmarkRequest else {
            respondJSON(client, status: "503 Service Unavailable",
                        object: ["error": "no benchmark pipeline is wired to this server"])
            return
        }
        guard let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let prompt = object["prompt"] as? String,
              !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            respondJSON(client, status: "400 Bad Request",
                        object: ["error": "body must be JSON with a non-empty 'prompt'"])
            return
        }
        let request = BenchmarkRequest(
            prompt: prompt,
            maxTokens: (object["max_tokens"] as? Int) ?? 32,
            runs: min(max((object["runs"] as? Int) ?? 3, 1), 10))

        handler(request) { [weak self, weak client] result in
            guard let self, let client else { return }
            self.queue.async {
                switch result {
                case .failure(let failure):
                    self.respondJSON(client, status: "500 Internal Server Error",
                                     object: ["error": failure.message])
                case .success(let generations):
                    let latencies = generations.map { $0.totalSeconds * 1000 }
                    let throughputs = generations.map {
                        $0.totalSeconds > 0
                            ? Double($0.tokenCount) / $0.totalSeconds : 0
                    }
                    let meanLatency = latencies.reduce(0, +) / Double(max(1, latencies.count))
                    let variance = latencies
                        .map { ($0 - meanLatency) * ($0 - meanLatency) }
                        .reduce(0, +) / Double(max(1, latencies.count))
                    self.respondJSON(client, status: "200 OK", object: [
                        "prompt": request.prompt,
                        "avg_tokens_per_sec": Self.round2(
                            throughputs.reduce(0, +) / Double(max(1, throughputs.count))),
                        "avg_latency_ms": Self.round2(meanLatency),
                        "stddev_latency_ms": Self.round2(variance.squareRoot()),
                        "runs": generations.enumerated().map { index, generation in
                            [
                                "run": index + 1,
                                "tokens_per_sec": Self.round2(throughputs[index]),
                                "latency_ms": Self.round2(latencies[index]),
                                "token_count": generation.tokenCount,
                                "payload_bytes": generation.networkPayloadBytes,
                            ] as [String: Any]
                        },
                    ])
                }
            }
        }
    }

    // MARK: POST /api/comparison

    private func handleComparisonPOST(body: Data, client: Client) {
        guard let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let tokens = object["tokens"] as? Int, tokens > 0,
              let payloadBytes = object["payload_bytes"] as? Int,
              let roundTrips = object["round_trips"] as? Int,
              let measuredMs = object["measured_total_ms"] as? Double, measuredMs > 0
        else {
            respondJSON(client, status: "400 Bad Request", object: [
                "error": "body must be JSON with tokens, payload_bytes, "
                    + "round_trips, measured_total_ms (from a real run)",
            ])
            return
        }
        let inputs = NMPProtocolComparisonModel.Inputs(
            tokens: tokens, payloadBytes: payloadBytes, roundTrips: roundTrips,
            measuredTotalSeconds: measuredMs / 1000,
            lanRTTMs: (object["lan_rtt_ms"] as? Double) ?? 2.0,
            lossRate: (object["loss_rate"] as? Double) ?? 0.0)
        respondJSON(client, status: "200 OK", object: [
            "note": "NMP row is the measured run; TCP/QUIC rows are that run "
                + "re-priced with modeled transport costs (see assumptions)",
            "protocols": NMPProtocolComparisonModel.compare(inputs)
                .map(\.asJSONObject),
        ])
    }

    private static func round2(_ value: Double) -> Double {
        (value * 100).rounded() / 100
    }

    // MARK: POST /api/inference

    private func handleInferencePOST(body: Data, client: Client) {
        guard let handler = onInferenceRequest else {
            respondJSON(client, status: "503 Service Unavailable",
                        object: ["error": "no inference pipeline is wired to this server"])
            return
        }
        guard let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let prompt = object["prompt"] as? String,
              !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            respondJSON(client, status: "400 Bad Request",
                        object: ["error": "body must be JSON with a non-empty 'prompt'"])
            return
        }
        let maxTokens = (object["max_tokens"] as? Int) ?? 32
        let enableSpeculation = (object["enable_speculation"] as? Bool) ?? false
        let enableComparison = (object["enable_comparison"] as? Bool) ?? false

        handler(InferenceRequest(prompt: prompt, maxTokens: maxTokens,
                                 enableSpeculation: enableSpeculation,
                                 enableComparison: enableComparison)) { [weak self, weak client] response in
            guard let self, let client else { return }
            // The handler may reply from any queue; connection sends are
            // thread-safe, but client bookkeeping stays on our queue.
            self.queue.async {
                switch response {
                case .success(let result):
                    let object = self.generationJSON(result)
                    // Mesh 2.5: "Compare protocols" runs the MEASURED
                    // transport race on the generation's real traffic
                    // pattern — the modeled re-pricing is gone from this
                    // path (it survives only in the labeled what-if
                    // explorer, POST /api/comparison).
                    guard enableComparison, result.tokenCount > 0,
                          result.totalSeconds > 0 else {
                        self.respondJSON(client, status: "200 OK", object: object)
                        return
                    }
                    let plan = NMPTransportRace.Plan(
                        roundTrips: result.speculation?.meshRoundTrips
                            ?? result.perTokenSeconds.count,
                        payloadBytes: result.networkPayloadBytes)
                    NMPTransportRace.run(plan: plan) { [weak self, weak client] raced in
                        guard let self, let client else { return }
                        self.queue.async {
                            var object = object
                            switch raced {
                            case .success(let race):
                                object["transport_race"] = [
                                    "race": race.asJSONObject,
                                    "projected": Self.projectedJSON(
                                        race: race,
                                        generationMs: result.totalSeconds * 1000,
                                        tokenCount: result.tokenCount),
                                ] as [String: Any]
                            case .failure(let error):
                                object["transport_race_error"] =
                                    String(describing: error)
                            }
                            self.respondJSON(client, status: "200 OK",
                                             object: object)
                        }
                    }
                case .failure(let status, let message):
                    self.respondJSON(
                        client,
                        status: "\(status) \(Self.reasonPhrase(for: status))",
                        object: ["error": message])
                }
            }
        }
    }

    /// The JSON body every generation-returning route shares
    /// (/api/inference, /api/chat). Runs on `queue`.
    private func generationJSON(
        _ result: NMPPromptInferenceService.GenerationResult
    ) -> [String: Any] {
        let tokensPerSec = result.totalSeconds > 0
            ? Double(result.tokenCount) / result.totalSeconds : 0
        // A round trip = one full mesh pass; the speculative path counts
        // them explicitly, the plain path spends one per pass
        // (perTokenSeconds entries).
        let roundTrips = result.speculation?.meshRoundTrips
            ?? result.perTokenSeconds.count
        var object: [String: Any] = [
            "output": result.text,
            "token_count": result.tokenCount,
            "latency_ms": (result.totalSeconds * 1000 * 10).rounded() / 10,
            "tokens_per_sec": (tokensPerSec * 100).rounded() / 100,
            "network_payload_bytes": result.networkPayloadBytes,
            "shard_count": result.shardCount,
            "engine": result.engine,
            "round_trips": roundTrips,
            "wire_format": storedMeshInfo.wireFormat,
        ]
        if let stats = result.speculation {
            object["speculation"] = [
                "drafter": stats.drafterName,
                "mesh_round_trips": stats.meshRoundTrips,
                "drafted_tokens": stats.draftedTokens,
                "accepted_draft_tokens": stats.acceptedDraftTokens,
                "fallback_rounds": stats.fallbackRounds,
                "acceptance_rate":
                    (stats.acceptanceRate * 1000).rounded() / 1000,
                "tokens_per_round_trip":
                    (stats.tokensPerRoundTrip(tokenCount: result.tokenCount)
                     * 100).rounded() / 100,
            ]
        }
        return object
    }

    // MARK: POST /api/chat (Mesh 2.7)

    /// Chat = the same generation pipeline as /api/inference, with the
    /// prompt assembled server-side from a whole conversation
    /// (`{"messages":[{"role":"user","content":"…"}, …]}`) so the web UI
    /// and the peer app share one template per engine (NMPChatPrompt).
    /// The mesh stays stateless: clients resend the transcript each turn.
    private func handleChatPOST(body: Data, client: Client) {
        guard let handler = onInferenceRequest else {
            respondJSON(client, status: "503 Service Unavailable",
                        object: ["error": "no inference pipeline is wired to this server"])
            return
        }
        guard let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let rawMessages = object["messages"] as? [[String: Any]],
              !rawMessages.isEmpty else {
            respondJSON(client, status: "400 Bad Request",
                        object: ["error": "body must be JSON with a non-empty 'messages' array"])
            return
        }
        var messages: [NMPChatMessage] = []
        for raw in rawMessages {
            guard let roleRaw = raw["role"] as? String,
                  let role = NMPChatMessage.Role(rawValue: roleRaw),
                  let content = raw["content"] as? String else {
                respondJSON(client, status: "400 Bad Request",
                            object: ["error": "each message needs a role "
                                + "(system|user|assistant) and string content"])
                return
            }
            messages.append(NMPChatMessage(role: role, content: content))
        }
        guard messages.last?.role == .user,
              messages.last.map({ !$0.content.trimmingCharacters(
                in: .whitespacesAndNewlines).isEmpty }) == true else {
            respondJSON(client, status: "400 Bad Request",
                        object: ["error": "the last message must be a non-empty user turn"])
            return
        }
        let maxTokens = (object["max_tokens"] as? Int) ?? 64
        let enableSpeculation = (object["enable_speculation"] as? Bool) ?? false
        let prompt = NMPChatPrompt.format(messages: messages,
                                          engine: storedMeshInfo.engine,
                                          model: storedMeshInfo.modelName)

        handler(InferenceRequest(prompt: prompt, maxTokens: maxTokens,
                                 enableSpeculation: enableSpeculation)) { [weak self, weak client] response in
            guard let self, let client else { return }
            self.queue.async {
                switch response {
                case .success(let result):
                    var object = self.generationJSON(result)
                    // What the template produced — chat clients show only
                    // the transcript, so surface the real prompt for
                    // debugging/curiosity.
                    object["assembled_prompt_chars"] = prompt.count
                    self.respondJSON(client, status: "200 OK", object: object)
                case .failure(let status, let message):
                    self.respondJSON(
                        client,
                        status: "\(status) \(Self.reasonPhrase(for: status))",
                        object: ["error": message])
                }
            }
        }
    }

    private func respondJSON(_ client: Client, status: String, object: [String: Any]) {
        let body = (try? JSONSerialization.data(withJSONObject: object))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        respond(client, status: status, body: body, contentType: "application/json")
    }

    private static func reasonPhrase(for status: Int) -> String {
        switch status {
        case 400: return "Bad Request"
        case 429: return "Too Many Requests"
        case 503: return "Service Unavailable"
        default: return "Internal Server Error"
        }
    }

    private func respondJSONArray(_ client: Client, status: String, array: [Any]) {
        let body = (try? JSONSerialization.data(withJSONObject: array))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
        respond(client, status: status, body: body, contentType: "application/json")
    }

    private func respond(_ client: Client, status: String, body: String,
                         contentType: String = "text/plain; charset=utf-8") {
        respondData(client, status: status, payload: Data(body.utf8),
                    contentType: contentType)
    }

    private func respondData(_ client: Client, status: String, payload: Data,
                             contentType: String) {
        // Permissive CORS: the server is a trusted-LAN testing tool; the
        // UI may be served from a dev origin (Vite) during development.
        let head = "HTTP/1.1 \(status)\r\n"
            + "Content-Type: \(contentType)\r\n"
            + "Content-Length: \(payload.count)\r\n"
            + "Access-Control-Allow-Origin: *\r\n"
            + "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            + "Access-Control-Allow-Headers: Content-Type\r\n"
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
