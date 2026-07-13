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
//    TLS leg (Mesh 2.5): kernel TCP + real TLS 1.3, using an ephemeral
//             self-signed P-256 identity generated in-process
//             (TLSIdentity.swift); the client pins the exact cert.
//
//    QUIC leg (Mesh 2.5): Network.framework QUIC (TLS 1.3 built in),
//             same pinned identity. What Mesh 2.1 called impossible
//             without dependencies just needed hand-rolled X.509.
//
//  Each round trip sends half the trip's bytes and waits for the far
//  side to echo them back — request out, response in, exactly the shape
//  of a mesh inference pass. Every number in the result is a wall-clock
//  measurement of real sockets on this machine.
//
//  WHAT THIS DOES AND DOESN'T CLAIM (stated in the API response too):
//  - All legs run over loopback, so radio time is absent from ALL —
//    the race isolates protocol/stack cost, not Wi-Fi.
//  - The plain-TCP leg stays in the race as the floor: NMP does
//    per-packet AEAD and FEC while plain TCP does nothing, and still
//    has to win. TLS 1.3 is the like-for-like encrypted comparison.
//  - Loss recovery is NOT modeled here: every leg binds in the
//    20000..<40000 band so scripts/loss_lab.sh (root, dummynet/pf) can
//    inject REAL loss on loopback — rerun the race under it and the
//    recovery cost shows up in measured transfer time.
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
        /// Per-send chunk size on the NMP leg. nil (the default) matches
        /// mesh traffic: the connection's link-adaptive recommendation
        /// (`PeerConnection.recommendedChunkBytes` — MTU-safe 1024 B on
        /// radio, the kernel datagram ceiling on loopback/wired).
        public var chunkBytes: Int?

        public init(roundTrips: Int, payloadBytes: Int,
                    chunkBytes: Int? = nil) {
            self.roundTrips = max(1, roundTrips)
            self.payloadBytes = max(2, payloadBytes)
            self.chunkBytes = chunkBytes.map { max(64, $0) }
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
        /// Every leg that ran, in race order (NMP always first). Every
        /// number in every leg is a wall-clock measurement.
        public let legs: [LegResult]
        public let note: String

        public var nmp: LegResult { legs[0] }

        public init(legs: [LegResult], skippedLegs: String? = nil) {
            self.legs = legs
            var note = "all legs measured over loopback sockets on this "
                + "machine — same trip count and bytes per leg; NMP carries "
                + "Noise IK + AES-256-GCM (its FEC parity and AWDL shaping "
                + "are radio-path features and stay out of wired/loopback "
                + "runs, exactly as in production); the TLS 1.3 and QUIC legs "
                + "use an ephemeral self-signed P-256 identity generated "
                + "in-process (the client pins the exact certificate). "
                + "Loopback isolates protocol/stack cost — radio time is "
                + "absent from every leg. For measured loss recovery, shape "
                + "the race ports first: scripts/loss_lab.sh."
            if let skippedLegs {
                note += " SKIPPED: \(skippedLegs)"
            }
            self.note = note
        }

        public var asJSONObject: [String: Any] {
            ["note": note, "legs": legs.map(\.asJSONObject)]
        }
    }

    public enum RaceError: Error {
        case setupFailed(String)
        case timedOut(String)
    }

    // MARK: Entry

    /// Runs the legs sequentially: NMP, plain TCP, TCP+TLS 1.3, QUIC.
    /// The TLS and QUIC legs need the ephemeral identity; if the keychain
    /// refuses to stage one (locked keychain, headless session) the race
    /// still returns NMP + TCP with the skip reason in the note — a
    /// partial measurement beats a model. Completion fires exactly once
    /// on an arbitrary queue.
    public static func run(
        plan: Plan, timeout: TimeInterval = 20,
        completion: @escaping (Result<RaceResult, RaceError>) -> Void
    ) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // Untimed warmup: the first leg to run otherwise pays
                // one-time process costs (allocator pools, framework init,
                // crypto setup) that later legs get for free — and NMP
                // always runs first. Measured: ~1.5 ms of NMP "handshake"
                // was warmup, not protocol.
                _ = try? runNMPLeg(
                    plan: Plan(roundTrips: 2, payloadBytes: 8_192),
                    timeout: timeout)

                var legs = [try runNMPLeg(plan: plan, timeout: timeout),
                            try runTCPLeg(plan: plan, timeout: timeout)]
                var skipped: String?
                #if os(macOS)
                do {
                    let identity = try NMPEphemeralTLSIdentity()
                    defer { identity.cleanup() }
                    legs.append(try runTLSLeg(plan: plan, timeout: timeout,
                                              identity: identity))
                    legs.append(try runQUICLeg(plan: plan, timeout: timeout,
                                               identity: identity))
                } catch {
                    skipped = "TLS 1.3 and QUIC legs — could not stage an "
                        + "ephemeral TLS identity or complete the leg "
                        + "(\(error)); nothing is modeled in their place."
                }
                #else
                skipped = "TLS 1.3 and QUIC legs — the race only runs them "
                    + "on macOS (the dashboard host)."
                #endif
                completion(.success(RaceResult(legs: legs, skippedLegs: skipped)))
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
        // Bind inside the race band (20000..<40000) so loss_lab.sh can
        // shape every leg with one port-range rule; fall back to an
        // ephemeral port rather than fail the leg.
        var boundListener: UDPListener?
        for _ in 0..<8 {
            let candidate = NWEndpoint.Port(
                rawValue: UInt16.random(in: 20000..<40000))!
            if let bound = try? UDPListener(port: candidate,
                                            queue: listenerQueue) {
                boundListener = bound
                break
            }
        }
        let listener = try boundListener
            ?? UDPListener(port: 0, queue: listenerQueue)

        let initiatorStatic = NoiseStaticKeyPair()
        let responderStatic = NoiseStaticKeyPair()

        // Responder: echo every decrypted payload straight back. The
        // connection shares the listener queue with its transport, exactly
        // like production peers (PeerNode) — a split pair pays two
        // cross-queue hops per datagram that production never pays.
        var responder: PeerConnection?
        listener.onNewTransport = { transport, _ in
            guard responder == nil else { return }   // one race client
            do {
                let connection = try PeerConnection(
                    role: .responder,
                    config: PeerConnectionConfig(localPeerID: RaceIDs.responder),
                    transport: transport,
                    localStatic: responderStatic,
                    queue: listenerQueue)
                // Echo like PeerShardEngine responds: accumulate the request
                // burst and answer with ONE burst when FLUSH (end-of-burst)
                // arrives — onPacket already runs on the connection queue,
                // so no extra hop and the reply's writes coalesce.
                var pendingEcho: [Data] = []
                connection.onPacket = { [weak connection] packet in
                    pendingEcho.append(packet.payload)
                    guard packet.header.flags.contains(.flush) else { return }
                    let burst = pendingEcho
                    pendingEcho.removeAll(keepingCapacity: true)
                    try? connection?.sendBurst(payloads: burst)
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

        // One queue for transport + connection, like production (PeerNode).
        let initiatorQueue = DispatchQueue(label: "nmp.race.udp.initiator")
        let transport = UDPTransport(
            host: "127.0.0.1", port: port,
            queue: initiatorQueue)
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
        let chunkBytes = plan.chunkBytes ?? initiator.recommendedChunkBytes
        let chunk = Data(repeating: 0xA5, count: min(chunkBytes, perDirection))
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

            // One burst per trip: single queue hop, coalesced transport
            // writes, FLUSH on the last chunk — exactly like
            // PeerShardEngine tensor traffic.
            var payloads: [Data] = []
            var sent = 0
            while sent < perDirection {
                let size = min(chunk.count, perDirection - sent)
                payloads.append(chunk.prefix(size))
                sent += size
            }
            initiator.sendBurstAsync(payloads: payloads)
            guard tripDone.wait(timeout: .now() + timeout) == .success else {
                throw RaceError.timedOut(
                    "NMP echo round trip (\(perDirection) B/direction)")
            }
        }
        let transferMs = elapsedMs(since: transferBegan)

        return LegResult(
            name: "NMP",
            transportDescription: "UDP + Noise IK + AES-256-GCM (production "
                + "stack, loopback: link-adaptive \(chunkBytes) B chunks; "
                + "FEC parity + AWDL shaping engage on radio paths only)",
            handshakeMs: handshakeMs,
            transferMs: transferMs,
            roundTrips: plan.roundTrips,
            bytesMoved: perDirection * 2 * plan.roundTrips)
    }

    // MARK: Stream legs (blocking; run on the global queue from `run`)

    static func runTCPLeg(plan: Plan, timeout: TimeInterval) throws -> LegResult {
        let parameters = { () -> NWParameters in
            let parameters = NWParameters.tcp
            parameters.allowLocalEndpointReuse = true
            return parameters
        }
        return try runStreamLeg(
            plan: plan, timeout: timeout, name: "TCP",
            transportDescription: "kernel TCP stream, no TLS, no framing "
                + "(loopback)",
            listenerParameters: parameters,
            clientParameters: { NWParameters.tcp })
    }

    #if os(macOS)
    static func runTLSLeg(plan: Plan, timeout: TimeInterval,
                          identity: NMPEphemeralTLSIdentity) throws -> LegResult {
        try runStreamLeg(
            plan: plan, timeout: timeout, name: "TCP+TLS 1.3",
            transportDescription: "kernel TCP + TLS 1.3, ephemeral "
                + "self-signed P-256 identity, certificate pinned (loopback)",
            listenerParameters: {
                let tls = NWProtocolTLS.Options()
                Self.configureServer(tls.securityProtocolOptions,
                                     identity: identity)
                let parameters = NWParameters(tls: tls)
                parameters.allowLocalEndpointReuse = true
                return parameters
            },
            clientParameters: {
                let tls = NWProtocolTLS.Options()
                Self.configureClient(tls.securityProtocolOptions,
                                     pinnedDER: identity.certificateDER)
                return NWParameters(tls: tls)
            })
    }

    static func runQUICLeg(plan: Plan, timeout: TimeInterval,
                           identity: NMPEphemeralTLSIdentity) throws -> LegResult {
        try runStreamLeg(
            plan: plan, timeout: timeout, name: "QUIC",
            transportDescription: "Network.framework QUIC (TLS 1.3 built "
                + "in, ALPN nmp-race), same pinned identity (loopback)",
            listenerParameters: {
                let quic = NWProtocolQUIC.Options(alpn: ["nmp-race"])
                Self.configureServer(quic.securityProtocolOptions,
                                     identity: identity)
                return NWParameters(quic: quic)
            },
            clientParameters: {
                let quic = NWProtocolQUIC.Options(alpn: ["nmp-race"])
                Self.configureClient(quic.securityProtocolOptions,
                                     pinnedDER: identity.certificateDER)
                return NWParameters(quic: quic)
            })
    }

    private static func configureServer(_ options: sec_protocol_options_t,
                                        identity: NMPEphemeralTLSIdentity) {
        sec_protocol_options_set_min_tls_protocol_version(options, .TLSv13)
        if let secIdentity = sec_identity_create(identity.identity) {
            sec_protocol_options_set_local_identity(options, secIdentity)
        }
    }

    /// Pin the exact generated certificate — byte equality, no policy
    /// evaluation. Nothing outside this process ever trusts this cert.
    private static func configureClient(_ options: sec_protocol_options_t,
                                        pinnedDER: Data) {
        sec_protocol_options_set_min_tls_protocol_version(options, .TLSv13)
        sec_protocol_options_set_verify_block(options, { _, secTrust, complete in
            let trust = sec_trust_copy_ref(secTrust).takeRetainedValue()
            var matches = false
            if let chain = SecTrustCopyCertificateChain(trust)
                as? [SecCertificate],
               let leaf = chain.first {
                matches = SecCertificateCopyData(leaf) as Data == pinnedDER
            }
            complete(matches)
        }, DispatchQueue(label: "nmp.race.tls.verify"))
    }
    #endif

    /// One echo leg over an NWListener/NWConnection pair — TCP, TCP+TLS,
    /// and QUIC all reduce to this: bind in the race band, connect,
    /// measure start → .ready as the handshake, then replay the plan's
    /// round trips and measure the transfer.
    static func runStreamLeg(
        plan: Plan, timeout: TimeInterval, name: String,
        transportDescription: String,
        listenerParameters: () -> NWParameters,
        clientParameters: () -> NWParameters
    ) throws -> LegResult {
        let serverQueue = DispatchQueue(label: "nmp.race.stream.server")
        // Bind BELOW the kernel's ephemeral range (49152+). An ephemeral
        // listener port lands where this process's earlier loopback
        // connections left TIME_WAIT four-tuples, and connects to such a
        // port fail EADDRINUSE deterministically once enough churn has
        // happened. A random port in 20000..<40000 is fresh by
        // construction; retry a few times in case one is genuinely taken.
        var listener: NWListener?
        for _ in 0..<8 {
            let candidate = NWEndpoint.Port(
                rawValue: UInt16.random(in: 20000..<40000))!
            if let bound = try? NWListener(using: listenerParameters(),
                                           on: candidate) {
                listener = bound
                break
            }
        }
        guard let listener else {
            throw RaceError.setupFailed(
                "\(name) listener: no bindable port in 20000..<40000")
        }

        // Echo server: stream every received byte straight back. Accept
        // and pump EVERY inbound connection — QUIC hands the listener two
        // (the connection plus the client's stream); cancelling the
        // "extra" one kills the stream that carries the data.
        let serverLock = NSLock()
        var serverSides: [NWConnection] = []
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
            serverLock.lock()
            serverSides.append(connection)
            serverLock.unlock()
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
            throw RaceError.setupFailed("\(name) listener never became ready")
        }
        defer {
            serverLock.lock()
            let connections = serverSides
            serverLock.unlock()
            connections.forEach { $0.cancel() }
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
            let candidate = NWConnection(host: "127.0.0.1", port: port,
                                         using: clientParameters())
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
            candidate.start(queue: DispatchQueue(label: "nmp.race.stream.client"))
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
                "\(name) connect to 127.0.0.1:\(port) after \(attempts) "
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
                    "\(name) echo round trip (\(perDirection) B/direction)")
            }
        }
        let transferMs = elapsedMs(since: transferBegan)

        return LegResult(
            name: name,
            transportDescription: transportDescription,
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
