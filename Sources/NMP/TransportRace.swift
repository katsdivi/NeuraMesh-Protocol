//
//  TransportRace.swift
//  NMP — Mesh 2.1
//
//  ACTUALLY-MEASURED protocol comparison. Mesh 2.0's comparison model
//  re-priced a measured NMP run with modeled TCP/QUIC constants; this
//  file replaces the guesswork for TCP with a real race:
//
//    take the traffic pattern of the generation that just ran
//    (round trips × payload bytes) and REPLAY it over real sockets —
//
//    NMP leg: production UDPListener/UDPTransport + PeerConnection.
//             Real Noise IK handshake, AES-256-GCM on every packet,
//             sequencing, FEC parity — the full stack, chunked into
//             1024-byte tensor-chunk-sized sends like real mesh traffic.
//
//    TCP leg: NWConnection/NWListener stream socket. Real 3-way
//             handshake, real kernel TCP, raw bytes (no framing, no
//             encryption).
//
//  Each round trip sends half the trip's bytes and waits for the far
//  side to echo them back — request out, response in, exactly the shape
//  of a mesh inference pass. Every number in the result is a wall-clock
//  measurement of real sockets on this machine.
//
//  WHAT THIS DOES AND DOESN'T CLAIM (stated in the API response too):
//  - Both legs run over loopback, so radio time is absent from BOTH —
//    the race isolates protocol/stack cost, not Wi-Fi.
//  - The TCP leg is PLAIN TCP. A production alternative would add TLS,
//    which costs more (handshake + per-byte crypto); racing plain TCP
//    is the conservative comparison — NMP is doing per-packet AEAD and
//    FEC while TCP does nothing, and still has to win.
//  - QUIC is NOT raced: Network.framework's QUIC requires a TLS
//    identity (a certificate), which a zero-dependency LAN tool can't
//    conjure honestly. QUIC stays in the labeled comparison MODEL.
//
//  Threading: callback style on private queues; `run` completion fires
//  exactly once (success or timeout). Legs run sequentially so they
//  never contend for the loopback path.
//

import Foundation
import Network

public enum NMPTransportRace {

    // MARK: Types

    public struct Plan: Sendable {
        /// Mesh round trips to replay (one request+response each).
        public var roundTrips: Int
        /// Total application bytes across the whole replay (both
        /// directions) — same accounting as GenerationResult's
        /// networkPayloadBytes.
        public var payloadBytes: Int
        /// Per-send chunk size on the NMP leg (mesh traffic ships
        /// NMPTensorChunk.defaultChunkBytes-sized packets).
        public var chunkBytes: Int

        public init(roundTrips: Int, payloadBytes: Int,
                    chunkBytes: Int = NMPTensorChunk.defaultChunkBytes) {
            self.roundTrips = max(1, roundTrips)
            self.payloadBytes = max(2, payloadBytes)
            self.chunkBytes = max(64, chunkBytes)
        }

        /// Bytes sent each way per round trip.
        var bytesPerDirection: Int { max(1, payloadBytes / roundTrips / 2) }
    }

    /// One transport's measured run. Everything here is a wall-clock
    /// measurement — there is no modeled field in this struct.
    public struct LegResult: Sendable {
        public let name: String
        public let transportDescription: String
        /// Connection setup: socket start → established/ready.
        public let handshakeMs: Double
        /// All round trips, first send → last echo byte.
        public let transferMs: Double
        public let roundTrips: Int
        /// Application bytes moved (both directions).
        public let bytesMoved: Int

        public var totalMs: Double { handshakeMs + transferMs }
        public var perTripMs: Double { transferMs / Double(max(1, roundTrips)) }

        public var asJSONObject: [String: Any] {
            [
                "name": name,
                "transport": transportDescription,
                "measured": true,
                "handshake_ms": round2(handshakeMs),
                "transfer_ms": round2(transferMs),
                "per_trip_ms": round2(perTripMs),
                "total_ms": round2(totalMs),
                "round_trips": roundTrips,
                "bytes_moved": bytesMoved,
            ]
        }
    }

    public struct RaceResult: Sendable {
        public let nmp: LegResult
        public let tcp: LegResult
        public let note = "both legs measured over loopback sockets on this "
            + "machine — same trip count and bytes; NMP carries Noise IK + "
            + "AES-256-GCM + FEC, the TCP leg is plain kernel TCP (no TLS). "
            + "QUIC is not raced (needs a TLS identity) and stays modeled."

        public var asJSONObject: [String: Any] {
            ["note": note, "legs": [nmp.asJSONObject, tcp.asJSONObject]]
        }
    }

    public enum RaceError: Error {
        case setupFailed(String)
        case timedOut(String)
    }

    // MARK: Entry

    /// Runs the NMP leg, then the TCP leg. Completion fires exactly once
    /// on an arbitrary queue.
    public static func run(
        plan: Plan, timeout: TimeInterval = 20,
        completion: @escaping (Result<RaceResult, RaceError>) -> Void
    ) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let nmp = try runNMPLeg(plan: plan, timeout: timeout)
                let tcp = try runTCPLeg(plan: plan, timeout: timeout)
                completion(.success(RaceResult(nmp: nmp, tcp: tcp)))
            } catch let error as RaceError {
                completion(.failure(error))
            } catch {
                completion(.failure(.setupFailed(String(describing: error))))
            }
        }
    }

    /// Blocking wrapper (call from a plain thread, never a mesh queue).
    public static func runSync(plan: Plan,
                               timeout: TimeInterval = 20) throws -> RaceResult {
        let done = DispatchSemaphore(value: 0)
        var outcome: Result<RaceResult, RaceError>?
        run(plan: plan, timeout: timeout) { result in
            outcome = result
            done.signal()
        }
        guard done.wait(timeout: .now() + timeout * 2 + 5) == .success,
              let outcome else {
            throw RaceError.timedOut("race did not finish")
        }
        return try outcome.get()
    }

    // MARK: NMP leg (blocking; runs on the global queue from `run`)

    static func runNMPLeg(plan: Plan, timeout: TimeInterval) throws -> LegResult {
        let listenerQueue = DispatchQueue(label: "nmp.race.udp.listener")
        let listener = try UDPListener(port: 0, queue: listenerQueue)

        let initiatorStatic = NoiseStaticKeyPair()
        let responderStatic = NoiseStaticKeyPair()

        // Responder: echo every decrypted payload straight back.
        var responder: PeerConnection?
        listener.onNewTransport = { transport, _ in
            guard responder == nil else { return }   // one race client
            do {
                let connection = try PeerConnection(
                    role: .responder,
                    config: PeerConnectionConfig(localPeerID: RaceIDs.responder),
                    transport: transport,
                    localStatic: responderStatic,
                    queue: DispatchQueue(label: "nmp.race.udp.responder"))
                connection.onPacket = { [weak connection] packet in
                    connection?.sendAsync(payload: packet.payload)
                }
                responder = connection
                connection.start()
            } catch {
                // Initiator's handshake wait will time out and report.
            }
        }

        let listenerReady = DispatchSemaphore(value: 0)
        listener.onStateChange = { state in
            if case .ready = state { listenerReady.signal() }
        }
        listener.start()
        guard listenerReady.wait(timeout: .now() + timeout) == .success,
              let port = listener.port else {
            listener.cancel()
            throw RaceError.setupFailed("UDP listener never became ready")
        }
        defer {
            responder?.close()
            listener.cancel()
        }

        let transport = UDPTransport(
            host: "127.0.0.1", port: port,
            queue: DispatchQueue(label: "nmp.race.udp.transport"))
        let initiatorQueue = DispatchQueue(label: "nmp.race.udp.initiator")
        let initiator = try PeerConnection(
            role: .initiator,
            config: PeerConnectionConfig(localPeerID: RaceIDs.initiator),
            transport: transport,
            localStatic: initiatorStatic,
            remoteStaticPublicKey: responderStatic.publicKeyData,
            queue: initiatorQueue)
        defer { initiator.close() }

        // Handshake: real Noise IK over real UDP loopback.
        let established = DispatchSemaphore(value: 0)
        initiator.onEstablished = { _, _ in established.signal() }
        let handshakeBegan = DispatchTime.now()
        initiator.start()
        guard established.wait(timeout: .now() + timeout) == .success else {
            throw RaceError.timedOut("NMP handshake")
        }
        let handshakeMs = elapsedMs(since: handshakeBegan)

        // Round trips: send bytesPerDirection in ≤chunkBytes packets,
        // wait for the same byte count echoed back.
        let perDirection = plan.bytesPerDirection
        let chunk = Data(repeating: 0xA5, count: min(plan.chunkBytes, perDirection))
        let tripDone = DispatchSemaphore(value: 0)
        let receivedLock = NSLock()
        var receivedThisTrip = 0
        initiator.onPacket = { packet in
            receivedLock.lock()
            let before = receivedThisTrip
            receivedThisTrip += packet.payload.count
            // Signal exactly once per trip — on the packet that crosses
            // the threshold, not on stragglers after it.
            let crossed = before < perDirection && receivedThisTrip >= perDirection
            receivedLock.unlock()
            if crossed { tripDone.signal() }
        }

        let transferBegan = DispatchTime.now()
        for _ in 0..<plan.roundTrips {
            receivedLock.lock()
            receivedThisTrip = 0
            receivedLock.unlock()

            var sent = 0
            while sent < perDirection {
                let size = min(chunk.count, perDirection - sent)
                initiator.sendAsync(payload: chunk.prefix(size))
                sent += size
            }
            guard tripDone.wait(timeout: .now() + timeout) == .success else {
                throw RaceError.timedOut(
                    "NMP echo round trip (\(perDirection) B/direction)")
            }
        }
        let transferMs = elapsedMs(since: transferBegan)

        return LegResult(
            name: "NMP",
            transportDescription: "UDP + Noise IK + AES-256-GCM + FEC "
                + "(production stack, loopback)",
            handshakeMs: handshakeMs,
            transferMs: transferMs,
            roundTrips: plan.roundTrips,
            bytesMoved: perDirection * 2 * plan.roundTrips)
    }

    // MARK: TCP leg (blocking; runs on the global queue from `run`)

    static func runTCPLeg(plan: Plan, timeout: TimeInterval) throws -> LegResult {
        let serverQueue = DispatchQueue(label: "nmp.race.tcp.server")
        // Bind BELOW the kernel's ephemeral range (49152+). An ephemeral
        // listener port lands where this process's earlier loopback
        // connections left TIME_WAIT four-tuples, and connects to such a
        // port fail EADDRINUSE deterministically once enough churn has
        // happened. A random port in 20000..<40000 is fresh by
        // construction; retry a few times in case one is genuinely taken.
        var listener: NWListener?
        for _ in 0..<8 {
            let parameters = NWParameters.tcp
            parameters.allowLocalEndpointReuse = true
            let candidate = NWEndpoint.Port(
                rawValue: UInt16.random(in: 20000..<40000))!
            if let bound = try? NWListener(using: parameters, on: candidate) {
                listener = bound
                break
            }
        }
        guard let listener else {
            throw RaceError.setupFailed(
                "TCP listener: no bindable port in 20000..<40000")
        }

        // Echo server: stream every received byte straight back.
        var serverSide: NWConnection?
        func pump(_ connection: NWConnection) {
            connection.receive(minimumIncompleteLength: 1,
                               maximumLength: 256 << 10) { data, _, isComplete, error in
                if let data, !data.isEmpty {
                    connection.send(content: data,
                                    completion: .contentProcessed { _ in })
                }
                guard !isComplete, error == nil else { return }
                pump(connection)
            }
        }
        listener.newConnectionHandler = { connection in
            guard serverSide == nil else {
                connection.cancel()
                return
            }
            serverSide = connection
            connection.start(queue: serverQueue)
            pump(connection)
        }

        let listenerReady = DispatchSemaphore(value: 0)
        listener.stateUpdateHandler = { state in
            if case .ready = state { listenerReady.signal() }
        }
        listener.start(queue: serverQueue)
        guard listenerReady.wait(timeout: .now() + timeout) == .success,
              let port = listener.port else {
            listener.cancel()
            throw RaceError.setupFailed("TCP listener never became ready")
        }
        defer {
            serverSide?.cancel()
            listener.cancel()
        }

        // Handshake: real TCP 3-way (start → .ready). A connect can land
        // in .waiting(EADDRINUSE) when the kernel's sequential ephemeral
        // port assignment collides with one of this process's TIME_WAIT
        // four-tuples (a process that has churned many loopback
        // connections — a test suite, a long-lived dashboard — hits this
        // deterministically), and Network.framework never retries the
        // bind on its own. A FRESH connection gets the next ephemeral
        // port, so retry with a new socket instead of waiting out a
        // wedged flow. Only clean attempts count toward the handshake
        // measurement.
        var client: NWConnection?
        var handshakeMs = 0.0
        var lastProblem = "no state change seen"
        let attempts = 4
        for attempt in 1...attempts {
            let candidate = NWConnection(host: "127.0.0.1", port: port, using: .tcp)
            let ready = DispatchSemaphore(value: 0)
            let stateLock = NSLock()
            var problem: String?
            candidate.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    ready.signal()
                case .failed(let error):
                    stateLock.lock()
                    problem = String(describing: error)
                    stateLock.unlock()
                    ready.signal()
                case .waiting(let error):
                    stateLock.lock()
                    problem = "waiting: \(error)"
                    stateLock.unlock()
                    // A waiting loopback connect will not self-heal; let
                    // the short wait below expire and retry fresh.
                default:
                    break
                }
            }
            let began = DispatchTime.now()
            candidate.start(queue: DispatchQueue(label: "nmp.race.tcp.client"))
            let attemptWindow = attempt < attempts ? min(timeout, 2.0) : timeout
            let outcome = ready.wait(timeout: .now() + attemptWindow)

            stateLock.lock()
            let seen = problem
            stateLock.unlock()
            if outcome == .success, seen == nil {
                handshakeMs = elapsedMs(since: began)
                client = candidate
                break
            }
            lastProblem = seen ?? "timed out with no state change"
            candidate.cancel()
        }
        guard let client else {
            throw RaceError.timedOut(
                "TCP connect to 127.0.0.1:\(port) after \(attempts) "
                    + "attempts — \(lastProblem)")
        }
        defer { client.cancel() }

        // Round trips: raw stream bytes, counted until fully echoed.
        let perDirection = plan.bytesPerDirection
        let payload = Data(repeating: 0x5A, count: perDirection)
        let tripDone = DispatchSemaphore(value: 0)
        let receivedLock = NSLock()
        var receivedThisTrip = 0

        func receiveLoop() {
            client.receive(minimumIncompleteLength: 1,
                           maximumLength: 256 << 10) { data, _, isComplete, error in
                if let data, !data.isEmpty {
                    receivedLock.lock()
                    let before = receivedThisTrip
                    receivedThisTrip += data.count
                    let crossed = before < perDirection
                        && receivedThisTrip >= perDirection
                    receivedLock.unlock()
                    if crossed { tripDone.signal() }
                }
                guard !isComplete, error == nil else { return }
                receiveLoop()
            }
        }
        receiveLoop()

        let transferBegan = DispatchTime.now()
        for _ in 0..<plan.roundTrips {
            receivedLock.lock()
            receivedThisTrip = 0
            receivedLock.unlock()

            client.send(content: payload, completion: .contentProcessed { _ in })
            guard tripDone.wait(timeout: .now() + timeout) == .success else {
                throw RaceError.timedOut(
                    "TCP echo round trip (\(perDirection) B/direction)")
            }
        }
        let transferMs = elapsedMs(since: transferBegan)

        return LegResult(
            name: "TCP",
            transportDescription: "kernel TCP stream, no TLS, no framing "
                + "(loopback)",
            handshakeMs: handshakeMs,
            transferMs: transferMs,
            roundTrips: plan.roundTrips,
            bytesMoved: perDirection * 2 * plan.roundTrips)
    }

    // MARK: Helpers

    static func elapsedMs(since began: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds
               - began.uptimeNanoseconds) / 1e6
    }
}

/// Fixed peer IDs for race connections (never collide with mesh IDs,
/// which the testbeds allocate from 1).
private enum RaceIDs {
    static let initiator: UInt32 = 0xACE0_0001
    static let responder: UInt32 = 0xACE0_0002
}

private func round2(_ value: Double) -> Double {
    (value * 100).rounded() / 100
}
