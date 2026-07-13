//
//  PeerConnection.swift
//  NMP — Phases 1-4
//
//  One authenticated, encrypted NMP session with a single remote peer.
//  Owns: handshake state machine (Noise IK), retry/backoff, session crypto,
//  the send/receive paths, (Phase 2) NACK-only reliability — every sealed
//  datagram is kept in a 64-packet retransmit window, inbound sequence gaps
//  are NACKed on a timer, and NACKed sequences are resent verbatim — and
//  (Phase 3) XOR FEC + AWDL suppression: outbound DATA packets form parity
//  groups so a single loss per group is reconstructed receiver-side without
//  a NACK round trip, and inferred AWDL contention defers non-critical
//  traffic until the link calms.
//
//  Handshake framing (spec §1: "Unencrypted NMP header: peer ID, nonce seed").
//  The 20-byte NMP header has no nonce-seed field, so the 8-byte nonce seed is
//  carried as the FIRST 8 BYTES of the handshake packet payload, followed by
//  the Noise message (whose Noise-level payload is the capability
//  advertisement). This framing decision is documented in Phase1_Design.md.
//
//    HANDSHAKE_MSG1/2 payload := nonce_seed(8, BE) ‖ noise_message
//
//  Retry semantics (spec §1):
//    - Initiator resends the SAME message 1 bytes on timeout (Noise handshake
//      state cannot be rewound; fresh attempts would need a new HandshakeState).
//    - Backoff: 5s, 10s, 20s. Max 3 retries, then peer marked unreachable 60s.
//    - Responder caches message 2 and resends it if a duplicate message 1
//      arrives (covers the msg2-lost case).
//    - Responder times out if no message 1 arrives within 5s of start().
//

import Foundation
import CryptoKit

// MARK: - Configuration

public struct PeerConnectionConfig: Sendable {
    /// This device's 32-bit NMP peer ID.
    public var localPeerID: UInt32
    /// Initial handshake timeout (seconds). Spec: 5.
    public var handshakeTimeout: TimeInterval = 5
    /// Backoff schedule for initiator retries. Spec: 5, 10, 20.
    public var retryBackoff: [TimeInterval] = [5, 10, 20]
    /// Spec: after 3 failed retries, mark peer unreachable for 60s.
    public var unreachableCooldown: TimeInterval = 60
    /// Optional allowlist of trusted remote static public keys (32-byte raw).
    /// nil = accept any peer that completes Noise IK authentication.
    /// SECURITY: nil is only safe if static keys are provisioned exclusively
    /// to trusted devices. See Phase1_Design.md.
    public var authorizedStaticKeys: Set<Data>?
    /// Phase 2 NACK/retransmit tuning.
    public var reliability = NMPReliabilityConfig()
    /// Phase 3 XOR FEC tuning.
    public var fec = NMPFECConfig()
    /// Phase 3 AWDL contention detection + traffic shaping.
    public var awdl = NMPAWDLConfig()

    public init(localPeerID: UInt32) {
        self.localPeerID = localPeerID
    }
}

// MARK: - State

public enum PeerConnectionState: Equatable, Sendable {
    case idle
    /// attempt = number of message-1 transmissions so far (1-based).
    case handshaking(attempt: Int)
    case established
    /// Handshake failed after max retries; do not retry until `until`.
    case unreachable(until: Date)
    case closed
}

public enum PeerConnectionError: Error, Equatable, Sendable {
    case notEstablished
    case alreadyStarted
    case handshakeTimeout
    case unauthorizedPeer
    case peerUnreachable(until: Date)
    case noiseFailure(String)
}

/// Phase 6: structured loss-recovery events, one per recovery action the
/// connection takes. `onDiagnostic` carries the same information as prose;
/// this is the machine-readable feed the testing dashboard renders.
public enum NMPPacketEvent: Equatable, Sendable {
    /// A lost DATA packet was reconstructed from XOR parity — recovery
    /// without a round trip.
    case fecRecovered(sequence: UInt32)
    /// The receiver requested retransmission of these missing sequences.
    case nackSent(sequences: [UInt32])
    /// The sender re-sent this sequence verbatim in answer to a peer NACK.
    case retransmitted(sequence: UInt32)
    /// The reliability machinery gave up on these sequences (attempts
    /// exhausted or aged out of the retransmit window).
    case unrecoverableLoss(sequences: [UInt32])
}

// MARK: - PeerConnection

public final class PeerConnection {

    public enum Role: Sendable { case initiator, responder }

    // MARK: Callbacks (invoked on `queue`)

    /// Handshake finished; argument = remote capability advertisement bytes.
    public var onEstablished: ((_ remoteCapabilities: Data, _ remotePeerID: UInt32) -> Void)?
    /// A decrypted application packet arrived.
    public var onPacket: ((NMPPacket) -> Void)?
    /// Terminal failure (timeout after retries, unauthorized peer, transport death).
    public var onFailed: ((PeerConnectionError) -> Void)?
    /// Non-fatal diagnostics (dropped malformed/replayed packets, retries).
    public var onDiagnostic: ((String) -> Void)?
    /// Sequences abandoned by the reliability layer (NACK attempts exhausted
    /// or aged out of the retransmit window). Phase 3's FEC consumes this.
    public var onUnrecoverableLoss: ((_ sequences: [UInt32]) -> Void)?
    /// Phase 6: structured loss/recovery event stream (FEC reconstructions,
    /// NACKs sent, retransmits served). Consumed by the testing dashboard.
    public var onPacketEvent: ((NMPPacketEvent) -> Void)?

    // MARK: State

    public private(set) var state: PeerConnectionState = .idle
    public private(set) var remotePeerID: UInt32?

    // Phase 4: pushed by NMPPeerDiscoveryManager (election + TXT records).
    /// Whether the LOCAL device currently coordinates the mesh.
    public private(set) var isCoordinator = false
    /// Latest discovered capabilities for the REMOTE peer, if any.
    public private(set) var remoteCapabilities: NMPCapabilities?

    private let role: Role
    private let config: PeerConnectionConfig
    private let transport: NMPTransport
    private let queue: DispatchQueue
    private let localCapabilities: Data

    private var noise: NoiseIKHandshake?
    private var session: NMPSecureSession?
    private let localNonceSeed: UInt64

    // Initiator retry machinery.
    private var cachedMessage1Datagram: Data?
    private var retryTimer: DispatchSourceTimer?
    private var retryCount = 0

    // Responder: cache msg2 so a duplicated msg1 (initiator retry) can be
    // answered without touching completed Noise state.
    private var cachedMessage2Datagram: Data?
    private var seenMessage1Digest: Data?

    // Phase 2 reliability.
    private var retransmitBuffer = NMPRetransmitBuffer()
    private var lossTracker: NMPLossTracker
    private var nackTimer: DispatchSourceTimer?

    // Phase 3 FEC + AWDL suppression.
    private var fecBuilder: NMPFECGroupBuilder
    private var fecReceiver: NMPFECGroupReceiver
    private var awdlDetector: NMPAWDLDetector
    private var shaper: NMPTrafficShaper
    private var deferTimer: DispatchSourceTimer?

    /// AWDL contention shaping only makes sense on radio paths: loopback and
    /// wired Ethernet have no shared airtime to protect, so shaping there
    /// only adds latency. `.unknown` (in-memory test transports, unresolved
    /// paths) keeps shaping on — the conservative default.
    private var awdlShapingApplies: Bool {
        config.awdl.enabled && transport.linkKind != .wiredOrLoopback
    }

    /// FEC parity exists to absorb radio loss without a retransmit round
    /// trip. Loopback and wired Ethernet lose essentially nothing, so parity
    /// there is +25% packets for no recovery — skip it. The NACK path stays
    /// armed as the (never-exercised) safety net, and the receive side still
    /// consumes parity, so a mixed-classification pair interoperates.
    private var fecApplies: Bool {
        config.fec.enabled && transport.linkKind != .wiredOrLoopback
    }

    /// Payload bytes that pack a 1500-byte Wi-Fi MTU without IP
    /// fragmentation (a lost fragment kills the whole datagram on radio):
    /// 1350 + seal(36) + UDP/IP(28) = 1414, leaving 86 B of headroom for
    /// VPN/tunnel encapsulation on real networks.
    static let radioChunkBytes = 1350

    /// Largest payload worth putting in one packet on THIS link. Radio
    /// paths pack the MTU but never fragment; unknown paths keep the
    /// conservative 1024 B default. Loopback/wired paths take the kernel's
    /// datagram ceiling instead: fragmentation is loss-free there and
    /// per-datagram cost dominates. Valid once the connection is
    /// established.
    public var recommendedChunkBytes: Int {
        switch transport.linkKind {
        case .radio:
            return Self.radioChunkBytes
        case .unknown:
            return NMPTensorChunk.defaultChunkBytes
        case .wiredOrLoopback:
            guard let maxDatagram = transport.maxDatagramBytes else {
                return NMPTensorChunk.defaultChunkBytes
            }
            let sealedOverhead = NMPHeader.byteCount + NMPHeader.gcmTagByteCount
            return max(NMPTensorChunk.defaultChunkBytes,
                       min(maxDatagram - sealedOverhead, NMPHeader.maxPayloadLength))
        }
    }

    // Mesh 2.3: cumulative wire traffic — every datagram handed to /
    // received from the transport, handshake, NACKs, FEC parity and
    // retransmits included (this is what actually crossed the link).
    // Lock-protected: written on `queue`, read by dashboard pollers
    // from their own queues.
    private let trafficLock = NSLock()
    private var sentWireBytes: UInt64 = 0
    private var receivedWireBytes: UInt64 = 0

    // MARK: Init

    /// - Parameters:
    ///   - localStatic: this device's long-term Curve25519 key pair.
    ///   - remoteStaticPublicKey: required for `.initiator` (IK pre-message),
    ///     must be nil for `.responder`.
    ///   - localCapabilities: opaque capability advertisement carried in the
    ///     handshake payload (format defined in Phase 4).
    ///   - queue: serial queue owning all connection state and callbacks.
    public init(
        role: Role,
        config: PeerConnectionConfig,
        transport: NMPTransport,
        localStatic: NoiseStaticKeyPair,
        remoteStaticPublicKey: Data? = nil,
        localCapabilities: Data = Data(),
        queue: DispatchQueue
    ) throws {
        self.role = role
        self.config = config
        self.transport = transport
        self.queue = queue
        self.localCapabilities = localCapabilities
        self.lossTracker = NMPLossTracker(config: config.reliability)
        self.fecBuilder = NMPFECGroupBuilder(groupSize: config.fec.groupSize)
        self.fecReceiver = NMPFECGroupReceiver(pendingTimeout: config.fec.pendingTimeout)
        self.awdlDetector = NMPAWDLDetector(config: config.awdl)
        self.shaper = NMPTrafficShaper(capacity: config.awdl.maxDeferredPackets)
        self.localNonceSeed = UInt64.random(in: UInt64.min...UInt64.max)
        self.noise = try NoiseIKHandshake(
            role: role == .initiator ? .initiator : .responder,
            localStatic: localStatic,
            remoteStaticPublicKey: remoteStaticPublicKey
        )
        transport.onReceive = { [weak self] datagram in
            self?.queue.async { self?.handleDatagram(datagram) }
        }
        transport.onClosed = { [weak self] error in
            self?.queue.async {
                self?.fail(.noiseFailure("transport closed: \(String(describing: error))"))
            }
        }
    }

    // MARK: Wire traffic accounting (Mesh 2.3)

    /// Total bytes this connection has put on / taken off the wire, from
    /// any thread. The Devices panel diffs successive readings into live
    /// per-link throughput.
    public var trafficTotals: (sentBytes: UInt64, receivedBytes: UInt64) {
        trafficLock.lock()
        defer { trafficLock.unlock() }
        return (sentWireBytes, receivedWireBytes)
    }

    /// The single choke point for outbound datagrams: counts, then sends.
    private func transmit(_ datagram: Data) {
        trafficLock.lock()
        sentWireBytes &+= UInt64(datagram.count)
        trafficLock.unlock()
        transport.send(datagram)
    }

    // MARK: Lifecycle

    /// Starts the transport and, for initiators, sends handshake message 1.
    /// Responders arm a 5s timeout waiting for message 1.
    public func start() {
        queue.async { [self] in
            guard state == .idle else {
                onFailed?(.alreadyStarted)
                return
            }
            transport.start()
            switch role {
            case .initiator:
                sendMessage1()
            case .responder:
                state = .handshaking(attempt: 0)
                armTimer(after: config.handshakeTimeout) { [weak self] in
                    guard let self, case .handshaking = self.state else { return }
                    self.fail(.handshakeTimeout)
                }
            }
        }
    }

    public func close() {
        queue.async { [self] in
            retryTimer?.cancel(); retryTimer = nil
            nackTimer?.cancel(); nackTimer = nil
            deferTimer?.cancel(); deferTimer = nil
            transport.cancel()
            state = .closed
        }
    }

    // MARK: Sending application data

    /// Encrypts and sends an application packet, passing through Phase 3
    /// traffic shaping and FEC grouping. Returns the assigned sequence
    /// number, or nil if the packet was deferred by AWDL suppression (it
    /// will be sealed and sent when suppression clears or the defer delay
    /// expires; throws `deferralBufferFull` if the buffer is at capacity).
    @discardableResult
    public func send(
        packetType: NMPPacketType = .data,
        flags: NMPFlags = [],
        priority: NMPSendPriority = .normal,
        payload: Data
    ) throws -> UInt32? {
        // Serialized access: callers off-queue should dispatch onto `queue`.
        dispatchPrecondition(condition: .onQueue(queue))
        guard state == .established, session != nil else {
            throw PeerConnectionError.notEstablished
        }
        if awdlShapingApplies {
            syncSuppression(at: Self.monotonicNow())
            if shaper.shouldDefer(packetType: packetType, flags: flags, priority: priority) {
                try shaper.deferPacket(packetType: packetType, flags: flags, payload: payload)
                onDiagnostic?("AWDL suppression: deferred \(packetType) packet "
                              + "(\(shaper.deferredCount) buffered)")
                armDeferTimer()
                return nil
            }
        }
        return try sealAndTransmit(packetType: packetType, flags: flags, payload: payload)
    }

    /// Convenience for callers not already on the connection queue.
    public func sendAsync(
        packetType: NMPPacketType = .data,
        flags: NMPFlags = [],
        priority: NMPSendPriority = .normal,
        payload: Data,
        completion: ((Result<UInt32?, Error>) -> Void)? = nil
    ) {
        queue.async { [self] in
            do {
                let seq = try send(packetType: packetType, flags: flags,
                                   priority: priority, payload: payload)
                completion?(.success(seq))
            } catch {
                completion?(.failure(error))
            }
        }
    }

    /// Seals and sends several payloads as one burst with transport writes
    /// coalesced (`NMPTransport.batched`) and FLUSH on the last payload
    /// unless disabled — the burst IS the unit whose end expedites receiver
    /// gap detection. This is the preferred way to ship a chunked tensor.
    /// Callers must already be on the connection queue (like `send`);
    /// off-queue callers use `sendBurstAsync`. Throws the first send error.
    public func sendBurst(
        payloads: [Data],
        priority: NMPSendPriority = .normal,
        flushLast: Bool = true
    ) throws {
        dispatchPrecondition(condition: .onQueue(queue))
        var failure: Error?
        transport.batched {
            for (index, payload) in payloads.enumerated() {
                let flags: NMPFlags =
                    flushLast && index == payloads.count - 1 ? [.flush] : []
                do {
                    _ = try send(flags: flags, priority: priority,
                                 payload: payload)
                } catch {
                    failure = error
                    break
                }
            }
        }
        if let failure { throw failure }
    }

    /// Off-queue convenience for `sendBurst`: one hop onto the connection
    /// queue for the whole burst. `completion` gets the first error, or nil.
    public func sendBurstAsync(
        payloads: [Data],
        priority: NMPSendPriority = .normal,
        flushLast: Bool = true,
        completion: ((Error?) -> Void)? = nil
    ) {
        guard !payloads.isEmpty else { completion?(nil); return }
        queue.async { [self] in
            do {
                try sendBurst(payloads: payloads, priority: priority,
                              flushLast: flushLast)
                completion?(nil)
            } catch {
                completion?(error)
            }
        }
    }

    /// Seals and sends one packet, maintaining the retransmit window and the
    /// FEC group. The group-closing packet gets FEC_GROUP_END set BEFORE
    /// sealing (the header is AAD and cannot change afterward); the parity
    /// packet follows the closing packet immediately.
    private func sealAndTransmit(
        packetType: NMPPacketType,
        flags: NMPFlags,
        payload: Data
    ) throws -> UInt32 {
        guard let session else { throw PeerConnectionError.notEstablished }

        var flags = flags
        let fecEligible = fecApplies && packetType == .data
        let closesGroup = fecEligible && fecBuilder.willCloseGroup(flush: flags.contains(.flush))
        if closesGroup { flags.insert(.fecGroupEnd) }

        let (datagram, seq) = try session.seal(
            packetType: packetType,
            flags: flags,
            senderPeerID: config.localPeerID,
            payload: payload,
            timestampNanos: Self.nowNanos()
        )
        // Every sealed datagram (including NACKs and parity) enters the
        // retransmit window, so any sequence the peer sees a gap for can be
        // refilled.
        retransmitBuffer.store(sequence: seq, datagram: datagram)
        transmit(datagram)
        if awdlShapingApplies { awdlDetector.recordSent(at: Self.monotonicNow()) }

        if fecEligible,
           let parityPayload = fecBuilder.add(sequence: seq, payload: payload,
                                              closesGroup: closesGroup) {
            _ = try sealAndTransmit(packetType: .fecRecovery, flags: [],
                                    payload: parityPayload)
        }
        return seq
    }

    // MARK: - Handshake: initiator

    private func sendMessage1() {
        do {
            let datagram: Data
            if let cached = cachedMessage1Datagram {
                datagram = cached // retry: resend identical bytes
            } else {
                guard let noise else { throw PeerConnectionError.noiseFailure("no handshake state") }
                let noiseMsg = try noise.writeMessage1(payload: localCapabilities)
                datagram = try Self.encodeHandshakePacket(
                    type: .handshakeMsg1,
                    senderPeerID: config.localPeerID,
                    nonceSeed: localNonceSeed,
                    noiseMessage: noiseMsg
                )
                cachedMessage1Datagram = datagram
            }
            retryCount += 1 // transmission number (1 = initial send)
            state = .handshaking(attempt: retryCount)
            transmit(datagram)

            // Spec backoff: wait 5s → retry, 10s → retry, 20s → retry,
            // then wait once more and give up (initial + 3 retries total).
            let waitIndex = retryCount - 1
            let delay = waitIndex < config.retryBackoff.count
                ? config.retryBackoff[waitIndex]
                : (config.retryBackoff.last ?? config.handshakeTimeout)
            let moreRetriesAllowed = retryCount <= config.retryBackoff.count
            armTimer(after: delay) { [weak self] in
                guard let self, case .handshaking = self.state else { return }
                if moreRetriesAllowed {
                    self.onDiagnostic?("handshake msg1 retry #\(self.retryCount) after \(delay)s")
                    self.sendMessage1()
                } else {
                    self.markUnreachable()
                }
            }
        } catch {
            fail(.noiseFailure("writeMessage1: \(error)"))
        }
    }

    private func handleMessage2(header: NMPHeader, payload: Data) {
        guard role == .initiator else {
            onDiagnostic?("dropped: msg2 received but we are responder")
            return
        }
        guard case .handshaking = state else {
            onDiagnostic?("dropped: duplicate msg2 in state \(state)")
            return
        }
        do {
            let (remoteSeed, noiseMsg) = try Self.decodeHandshakePayload(payload)
            guard let noise else { throw PeerConnectionError.noiseFailure("no handshake state") }
            let remoteCapabilities = try noise.readMessage2(noiseMsg)
            let result = try noise.finalize()
            try authorize(result.remoteStaticPublicKey)
            establishSession(
                result: result,
                remoteSeed: remoteSeed,
                remoteID: header.senderPeerID,
                remoteCapabilities: remoteCapabilities
            )
        } catch let e as PeerConnectionError {
            fail(e)
        } catch {
            // Bad/forged msg2: drop and keep waiting; retry timer still armed.
            onDiagnostic?("dropped malformed msg2: \(error)")
        }
    }

    // MARK: - Handshake: responder

    private func handleMessage1(header: NMPHeader, payload: Data, raw: Data) {
        guard role == .responder else {
            onDiagnostic?("dropped: msg1 received but we are initiator")
            return
        }
        // Duplicate msg1 after completion → initiator missed msg2; resend it.
        if state == .established {
            if let digest = seenMessage1Digest, digest == Self.digest(raw),
               let cached = cachedMessage2Datagram {
                onDiagnostic?("duplicate msg1 → resending cached msg2")
                transmit(cached)
            } else {
                onDiagnostic?("dropped: unexpected msg1 while established")
            }
            return
        }
        guard case .handshaking = state else {
            onDiagnostic?("dropped: msg1 in state \(state)")
            return
        }
        do {
            let (remoteSeed, noiseMsg) = try Self.decodeHandshakePayload(payload)
            guard let noise else { throw PeerConnectionError.noiseFailure("no handshake state") }
            let remoteCapabilities = try noise.readMessage1(noiseMsg)

            let msg2 = try noise.writeMessage2(payload: localCapabilities)
            let result = try noise.finalize()
            try authorize(result.remoteStaticPublicKey)

            let datagram = try Self.encodeHandshakePacket(
                type: .handshakeMsg2,
                senderPeerID: config.localPeerID,
                nonceSeed: localNonceSeed,
                noiseMessage: msg2
            )
            cachedMessage2Datagram = datagram
            seenMessage1Digest = Self.digest(raw)

            retryTimer?.cancel(); retryTimer = nil
            transmit(datagram)
            establishSession(
                result: result,
                remoteSeed: remoteSeed,
                remoteID: header.senderPeerID,
                remoteCapabilities: remoteCapabilities
            )
        } catch let e as PeerConnectionError {
            fail(e)
        } catch {
            // Malformed or unauthenticated msg1: drop, keep listening until timeout.
            onDiagnostic?("dropped malformed msg1: \(error)")
        }
    }

    // MARK: - Common handshake plumbing

    private func authorize(_ remoteStatic: Data) throws {
        if let allow = config.authorizedStaticKeys, !allow.contains(remoteStatic) {
            throw PeerConnectionError.unauthorizedPeer
        }
    }

    private func establishSession(
        result: NoiseHandshakeResult,
        remoteSeed: UInt64,
        remoteID: UInt32,
        remoteCapabilities: Data
    ) {
        retryTimer?.cancel(); retryTimer = nil
        let keys = NMPSessionKeys(
            handshake: result,
            localNonceSeed: localNonceSeed,
            remoteNonceSeed: remoteSeed
        )
        session = NMPSecureSession(keys: keys)
        remotePeerID = remoteID
        noise = nil // handshake state no longer needed; drop key material refs
        state = .established
        onEstablished?(remoteCapabilities, remoteID)
    }

    private func markUnreachable() {
        retryTimer?.cancel(); retryTimer = nil
        let until = Date().addingTimeInterval(config.unreachableCooldown)
        state = .unreachable(until: until)
        onFailed?(.peerUnreachable(until: until))
    }

    private func fail(_ error: PeerConnectionError) {
        retryTimer?.cancel(); retryTimer = nil
        if state != .closed {
            state = .closed
            onFailed?(error)
        }
    }

    // MARK: - Receive dispatch

    private func handleDatagram(_ datagram: Data) {
        guard state != .closed else { return }
        trafficLock.lock()
        receivedWireBytes &+= UInt64(datagram.count)
        trafficLock.unlock()
        let header: NMPHeader
        do {
            header = try NMPPacketCodec.decodeHeader(datagram)
        } catch {
            onDiagnostic?("dropped malformed datagram (\(datagram.count)B): \(error)")
            return
        }

        if header.isEncrypted {
            guard state == .established, let session else {
                onDiagnostic?("dropped encrypted packet before session established")
                return
            }
            do {
                let packet = try session.open(datagram: datagram)
                let now = Self.monotonicNow()

                // AWDL: one-way delay sample (sender wall clock; the
                // detector only reacts to shifts, so skew is tolerable).
                if awdlShapingApplies {
                    let delta = Int64(bitPattern: Self.nowNanos() &- packet.header.timestampNanos)
                    awdlDetector.recordLatencySample(Double(delta) / 1e9, at: now)
                    syncSuppression(at: now)
                }

                // Reliability: track gaps across ALL authenticated sequences.
                let agedOut = lossTracker.observe(
                    sequence: packet.header.sequenceNumber, at: now)
                if !agedOut.isEmpty {
                    onDiagnostic?("gaps aged out of retransmit window: \(agedOut)")
                    onUnrecoverableLoss?(agedOut)
                    onPacketEvent?(.unrecoverableLoss(sequences: agedOut))
                }
                // FLUSH = end of burst; nothing behind it will fill gaps.
                if packet.header.flags.contains(.flush) {
                    lossTracker.expediteAll(at: now)
                }

                switch packet.header.packetType {
                case .nack:
                    handleNack(packet.payload, at: now)
                case .fecRecovery:
                    handleParity(packet.payload, at: now)
                case .data where config.fec.enabled:
                    let recoveries = fecReceiver.observeData(
                        sequence: packet.header.sequenceNumber,
                        payload: packet.payload, at: now)
                    onPacket?(packet)
                    deliverRecoveries(recoveries, at: now)
                default:
                    onPacket?(packet)
                }
                serviceNacks(at: now)
            } catch NMPCryptoError.replayDetected(let seq, let last) {
                onDiagnostic?("dropped replay/too-old seq=\(seq) highest=\(last)")
            } catch {
                onDiagnostic?("dropped undecryptable packet: \(error)")
            }
        } else {
            do {
                let packet = try NMPPacketCodec.decodePlaintextPacket(datagram)
                switch packet.header.packetType {
                case .handshakeMsg1:
                    handleMessage1(header: packet.header, payload: packet.payload, raw: datagram)
                case .handshakeMsg2:
                    handleMessage2(header: packet.header, payload: packet.payload)
                default:
                    onDiagnostic?("dropped plaintext packet of type \(packet.header.packetType)")
                }
            } catch {
                onDiagnostic?("dropped malformed handshake packet: \(error)")
            }
        }
    }

    // MARK: - Phase 2 reliability

    /// Peer reported missing sequences: resend the exact original bytes.
    /// (Verbatim resend is mandatory — the header is GCM AAD, so mutating it
    /// post-seal breaks the tag, and re-sealing under the same nonce with a
    /// different header is the GCM forbidden attack. See Reliability.swift.)
    private func handleNack(_ payload: Data, at now: TimeInterval) {
        let sequences: [UInt32]
        do {
            sequences = try NMPNackCodec.decode(payload)
        } catch {
            onDiagnostic?("dropped malformed NACK: \(error)")
            return
        }
        // Each NACKed sequence is a loss report for the AWDL detector. FEC-
        // recovered losses never reach this point — by design, only loss the
        // FEC layer failed to absorb counts toward suppression.
        if awdlShapingApplies {
            awdlDetector.recordLosses(sequences.count, at: now)
            syncSuppression(at: now)
        }
        for seq in sequences {
            if let datagram = retransmitBuffer.datagram(for: seq) {
                onDiagnostic?("retransmitting seq=\(seq) on NACK")
                onPacketEvent?(.retransmitted(sequence: seq))
                transmit(datagram)
            } else {
                onDiagnostic?("NACK for seq=\(seq) outside retransmit window; ignored")
            }
        }
    }

    /// Sends any due NACKs, reports abandoned sequences, and re-arms the
    /// timer for the next deadline. Runs on `queue`.
    private func serviceNacks(at now: TimeInterval) {
        let (toNack, gaveUp) = lossTracker.dueNacks(at: now)
        if !gaveUp.isEmpty {
            onDiagnostic?("NACK attempts exhausted for \(gaveUp)")
            onUnrecoverableLoss?(gaveUp)
            onPacketEvent?(.unrecoverableLoss(sequences: gaveUp))
        }
        if !toNack.isEmpty {
            do {
                _ = try send(packetType: .nack, payload: NMPNackCodec.encode(toNack))
                onPacketEvent?(.nackSent(sequences: toNack))
            } catch {
                onDiagnostic?("failed to send NACK for \(toNack): \(error)")
            }
        }
        armNackTimer()
    }

    private func armNackTimer() {
        nackTimer?.cancel(); nackTimer = nil
        guard state == .established, let deadline = lossTracker.nextDeadline else { return }
        let delay = max(0, deadline - Self.monotonicNow())
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + delay)
        timer.setEventHandler { [weak self] in
            guard let self, self.state == .established else { return }
            self.serviceNacks(at: Self.monotonicNow())
        }
        timer.resume()
        nackTimer = timer
    }

    // MARK: - Phase 3: FEC receive path

    private func handleParity(_ payload: Data, at now: TimeInterval) {
        guard config.fec.enabled else {
            onDiagnostic?("dropped FEC parity packet (FEC disabled)")
            return
        }
        do {
            let recoveries = try fecReceiver.observeParity(payload, at: now)
            deliverRecoveries(recoveries, at: now)
        } catch {
            onDiagnostic?("dropped malformed FEC parity packet: \(error)")
        }
    }

    /// Injects FEC-reconstructed packets into the normal delivery path: the
    /// replay window marks the sequence seen (a straggling original or a
    /// racing NACK retransmit is then dropped as a replay), the loss tracker
    /// cancels the pending NACK, and the application receives the payload
    /// under a synthesized header (timestamp 0 — the original header was
    /// lost with the packet).
    private func deliverRecoveries(_ recoveries: [NMPFECGroupReceiver.Recovery],
                                   at now: TimeInterval) {
        guard !recoveries.isEmpty, let session else { return }
        for recovery in recoveries {
            session.markSequenceSeen(recovery.sequence)
            lossTracker.markRecovered(recovery.sequence)
            onDiagnostic?("FEC recovered seq=\(recovery.sequence) "
                          + "(group 0x\(String(recovery.groupID, radix: 16)))")
            onPacketEvent?(.fecRecovered(sequence: recovery.sequence))
            let header = NMPHeader(
                isEncrypted: true,
                flags: [],
                packetType: .data,
                payloadLength: UInt16(clamping: recovery.payload.count),
                sequenceNumber: recovery.sequence,
                senderPeerID: remotePeerID ?? 0,
                timestampNanos: 0
            )
            onPacket?(NMPPacket(header: header, payload: recovery.payload))
        }
    }

    // MARK: - Phase 3: AWDL suppression

    /// Re-evaluates the detector and reacts to state flips: on engage, give
    /// unattempted NACK gaps extra grace (FEC groups get time to complete);
    /// on clear, release everything the shaper deferred.
    private func syncSuppression(at now: TimeInterval) {
        guard awdlDetector.updateState(at: now) else {
            shaper.suppressionActive = awdlDetector.suppressionActive
            return
        }
        shaper.suppressionActive = awdlDetector.suppressionActive
        if awdlDetector.suppressionActive {
            let pct = String(format: "%.1f", awdlDetector.currentLossRate * 100)
            onDiagnostic?("AWDL suppression engaged (loss rate \(pct)%)")
            lossTracker.postponeUnattempted(until: now + config.fec.pendingTimeout)
        } else {
            onDiagnostic?("AWDL suppression cleared")
            flushDeferred(reason: "suppression cleared")
        }
    }

    /// Backstop: deferred packets leave after `maxDeferDelay` even if the
    /// link never calms — model activations cannot wait forever.
    private func armDeferTimer() {
        guard deferTimer == nil, shaper.hasDeferred else { return }
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + config.awdl.maxDeferDelay)
        timer.setEventHandler { [weak self] in
            self?.flushDeferred(reason: "max defer delay reached")
        }
        timer.resume()
        deferTimer = timer
    }

    private func flushDeferred(reason: String) {
        deferTimer?.cancel(); deferTimer = nil
        guard shaper.hasDeferred, state == .established else { return }
        let packets = shaper.drain()
        onDiagnostic?("flushing \(packets.count) deferred packets (\(reason))")
        for packet in packets {
            do {
                _ = try sealAndTransmit(packetType: packet.packetType,
                                        flags: packet.flags,
                                        payload: packet.payload)
            } catch {
                onDiagnostic?("failed to send deferred packet: \(error)")
            }
        }
    }

    // MARK: - Phase 4: discovery integration

    /// Applies the latest election outcome and (optionally) the remote
    /// peer's discovered capabilities. Called by NMPPeerDiscoveryManager
    /// consumers; Phase 5 branches on `isCoordinator` for shard assignment.
    public func updateDiscoveryState(
        isCoordinator: Bool,
        remoteCapabilities: NMPCapabilities? = nil
    ) {
        queue.async { [self] in
            self.isCoordinator = isCoordinator
            if let remoteCapabilities {
                self.remoteCapabilities = remoteCapabilities
            }
        }
    }

    // MARK: - Handshake packet framing helpers

    static func encodeHandshakePacket(
        type: NMPPacketType,
        senderPeerID: UInt32,
        nonceSeed: UInt64,
        noiseMessage: Data
    ) throws -> Data {
        var payload = Data(capacity: 8 + noiseMessage.count)
        payload.appendBigEndian(nonceSeed)
        payload.append(noiseMessage)
        let header = NMPHeader(
            isEncrypted: false,
            flags: [],
            packetType: type,
            payloadLength: UInt16(clamping: payload.count),
            sequenceNumber: 0,
            senderPeerID: senderPeerID,
            timestampNanos: nowNanos()
        )
        return try NMPPacketCodec.encodePlaintextPacket(NMPPacket(header: header, payload: payload))
    }

    static func decodeHandshakePayload(_ payload: Data) throws -> (nonceSeed: UInt64, noiseMessage: Data) {
        guard payload.count > 8 else { throw NoiseError.malformedMessage }
        let bytes = Data(payload)
        let seed = bytes.readBigEndianUInt64(at: 0)
        return (seed, bytes.subdata(in: 8..<bytes.count))
    }

    // MARK: - Utilities

    private func armTimer(after seconds: TimeInterval, _ handler: @escaping () -> Void) {
        retryTimer?.cancel()
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + seconds)
        timer.setEventHandler(handler: handler)
        timer.resume()
        retryTimer = timer
    }

    /// Monotonic seconds for reliability deadlines (wall clock can jump).
    static func monotonicNow() -> TimeInterval {
        TimeInterval(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000
    }

    static func nowNanos() -> UInt64 {
        #if canImport(Darwin)
        return clock_gettime_nsec_np(CLOCK_REALTIME)
        #else
        return UInt64(Date().timeIntervalSince1970 * 1_000_000_000)
        #endif
    }

    private static func digest(_ data: Data) -> Data {
        Data(SHA256.hash(data: data))
    }
}
