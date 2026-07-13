//
//  FECIntegrationTests.swift
//  NMPTests — Phase 3
//
//  End-to-end FEC + AWDL suppression between two mock peers: single loss
//  recovered without any NACK round trip, NACK fallback when FEC cannot
//  help (parity lost, 2+ losses per group), recovery-rate measurement under
//  random loss, the Phase 2 vs Phase 3 recovery-latency comparison, and
//  traffic deferral under inferred contention.
//
//  Loss is injected per sequence number by decoding the plaintext header
//  (it is GCM AAD, readable without decryption). FEC-recovered deliveries
//  are identified by `header.timestampNanos == 0` — the original header was
//  lost with the packet, so PeerConnection synthesizes one.
//

import XCTest
@testable import NMP

final class FECIntegrationTests: XCTestCase {

    private struct Mesh {
        let initiator: PeerConnection
        let responder: PeerConnection
        let tInit: MockTransport
        let tResp: MockTransport
    }

    private func makeEstablishedMesh(
        fecEnabled: Bool = true,
        configure: ((inout PeerConnectionConfig) -> Void)? = nil
    ) throws -> Mesh {
        let (tInit, tResp) = MockTransport.pair()
        let sInit = NoiseStaticKeyPair()
        let sResp = NoiseStaticKeyPair()

        var cfgI = PeerConnectionConfig(localPeerID: 1)
        var cfgR = PeerConnectionConfig(localPeerID: 2)
        cfgI.fec.enabled = fecEnabled
        cfgR.fec.enabled = fecEnabled
        configure?(&cfgI)
        configure?(&cfgR)

        let initiator = try PeerConnection(
            role: .initiator, config: cfgI, transport: tInit,
            localStatic: sInit, remoteStaticPublicKey: sResp.publicKeyData,
            queue: DispatchQueue(label: "nmp.fec.init"))
        let responder = try PeerConnection(
            role: .responder, config: cfgR, transport: tResp,
            localStatic: sResp,
            queue: DispatchQueue(label: "nmp.fec.resp"))

        let ready = expectation(description: "established")
        ready.expectedFulfillmentCount = 2
        initiator.onEstablished = { _, _ in ready.fulfill() }
        responder.onEstablished = { _, _ in ready.fulfill() }
        responder.start()
        initiator.start()
        wait(for: [ready], timeout: 5)
        return Mesh(initiator: initiator, responder: responder,
                    tInit: tInit, tResp: tResp)
    }

    private static func header(of datagram: Data) -> NMPHeader? {
        try? NMPPacketCodec.decodeHeader(datagram)
    }

    // MARK: Single loss → FEC, no NACK

    func testSingleLossRecoveredViaFECWithoutNack() throws {
        let mesh = try makeEstablishedMesh()
        let lostSeq: UInt32 = 1

        // Black-hole seq 1 completely — if recovery happens, FEC did it.
        mesh.tInit.dropOutbound = { datagram in
            guard let h = Self.header(of: datagram) else { return false }
            return h.isEncrypted && h.packetType == .data && h.sequenceNumber == lostSeq
        }
        let retransmitLock = NSLock()
        var nackRetransmits = 0
        mesh.initiator.onDiagnostic = { message in
            if message.contains("retransmitting seq=") {
                retransmitLock.lock(); nackRetransmits += 1; retransmitLock.unlock()
            }
        }

        let recovered = expectation(description: "payload recovered via FEC")
        var dropTime: DispatchTime?
        var recoveredPayload: Data?
        mesh.responder.onPacket = { packet in
            if packet.header.sequenceNumber == lostSeq {
                XCTAssertEqual(packet.header.timestampNanos, 0,
                               "recovery must come from FEC (synthesized header)")
                recoveredPayload = packet.payload
                recovered.fulfill()
            }
        }

        dropTime = DispatchTime.now()
        for n in 0..<4 { // one full group; parity follows packet 4
            mesh.initiator.sendAsync(payload: Data("fec #\(n)".utf8))
        }
        wait(for: [recovered], timeout: 2)

        XCTAssertEqual(recoveredPayload, Data("fec #1".utf8))
        let ms = Double(DispatchTime.now().uptimeNanoseconds - dropTime!.uptimeNanoseconds) / 1e6
        print("[NMP] FEC end-to-end loss recovery time: \(String(format: "%.3f", ms)) ms")
        // Give any stray NACK a moment, then confirm none was needed.
        Thread.sleep(forTimeInterval: 0.1)
        retransmitLock.lock()
        XCTAssertEqual(nackRetransmits, 0, "FEC recovery must not trigger NACK retransmits")
        retransmitLock.unlock()
    }

    // MARK: FEC fallbacks to NACK

    func testParityLossFallsBackToNack() throws {
        let mesh = try makeEstablishedMesh()
        let lostSeq: UInt32 = 2

        // Drop data seq 2 (first transmission only) AND every parity packet:
        // FEC is blinded, Phase 2 must recover.
        let lock = NSLock()
        var droppedData = false
        mesh.tInit.dropOutbound = { datagram in
            guard let h = Self.header(of: datagram), h.isEncrypted else { return false }
            if h.packetType == .fecRecovery { return true }
            guard h.packetType == .data, h.sequenceNumber == lostSeq else { return false }
            lock.lock(); defer { lock.unlock() }
            if droppedData { return false }
            droppedData = true
            return true
        }

        let recovered = expectation(description: "payload recovered via NACK")
        mesh.responder.onPacket = { packet in
            if packet.header.sequenceNumber == lostSeq {
                XCTAssertNotEqual(packet.header.timestampNanos, 0,
                                  "recovery must be the verbatim retransmit, not FEC")
                recovered.fulfill()
            }
        }
        for n in 0..<6 {
            mesh.initiator.sendAsync(payload: Data("pl #\(n)".utf8))
        }
        wait(for: [recovered], timeout: 5)
    }

    func testTwoLossesInGroupFallBackToNack() throws {
        let mesh = try makeEstablishedMesh()
        let lostSeqs: Set<UInt32> = [1, 2] // two members of the first group

        let lock = NSLock()
        var dropped: Set<UInt32> = []
        mesh.tInit.dropOutbound = { datagram in
            guard let h = Self.header(of: datagram), h.isEncrypted,
                  h.packetType == .data, lostSeqs.contains(h.sequenceNumber) else { return false }
            lock.lock(); defer { lock.unlock() }
            return dropped.insert(h.sequenceNumber).inserted // first transmission only
        }

        let allRecovered = expectation(description: "both payloads recovered")
        var got: Set<UInt32> = []
        mesh.responder.onPacket = { packet in
            if lostSeqs.contains(packet.header.sequenceNumber) {
                got.insert(packet.header.sequenceNumber)
                if got == lostSeqs { allRecovered.fulfill() }
            }
        }
        for n in 0..<6 {
            mesh.initiator.sendAsync(payload: Data("dbl #\(n)".utf8))
        }
        wait(for: [allRecovered], timeout: 5)
    }

    func testInterleavedGroupsAllRecovered() throws {
        let mesh = try makeEstablishedMesh()
        // One loss in each of three consecutive groups. Data sequences with
        // parity interleaved: group1 = 0-3 (parity 4), group2 = 5-8
        // (parity 9), group3 = 10-13 (parity 14).
        let lostSeqs: Set<UInt32> = [1, 6, 12]
        mesh.tInit.dropOutbound = { datagram in
            guard let h = Self.header(of: datagram) else { return false }
            return h.isEncrypted && h.packetType == .data
                && lostSeqs.contains(h.sequenceNumber)
        }

        let total = 12
        let allDelivered = expectation(description: "all payloads delivered")
        var received: Set<Data> = []
        var fecRecoveries = 0
        let lock = NSLock()
        mesh.responder.onPacket = { packet in
            lock.lock()
            received.insert(packet.payload)
            if packet.header.timestampNanos == 0 { fecRecoveries += 1 }
            let done = received.count == total
            lock.unlock()
            if done { allDelivered.fulfill() }
        }
        for n in 0..<total {
            mesh.initiator.sendAsync(payload: Data("grp #\(n)".utf8))
        }
        wait(for: [allDelivered], timeout: 5)
        lock.lock()
        XCTAssertEqual(received, Set((0..<total).map { Data("grp #\($0)".utf8) }))
        XCTAssertEqual(fecRecoveries, 3, "each group's single loss must be FEC-recovered")
        lock.unlock()
    }

    // MARK: Recovery-rate measurement (success criterion: ≥80%)

    func testFECRecoveryRateUnder2PercentLoss() throws {
        let mesh = try makeEstablishedMesh()
        let total = 1000

        // Deterministic 2% loss over first transmissions of data AND parity
        // packets (parity is not privileged — it can be lost too).
        var rng = SplitMix64(seed: 0x5EED_CAFE)
        let lock = NSLock()
        var seenSeqs: Set<UInt32> = []
        var droppedData: Set<UInt32> = []
        mesh.tInit.dropOutbound = { datagram in
            guard let h = Self.header(of: datagram), h.isEncrypted,
                  h.packetType == .data || h.packetType == .fecRecovery else { return false }
            lock.lock(); defer { lock.unlock() }
            guard seenSeqs.insert(h.sequenceNumber).inserted else { return false } // retransmit
            guard rng.chance(0.02) else { return false }
            if h.packetType == .data { droppedData.insert(h.sequenceNumber) }
            return true
        }

        let allDelivered = expectation(description: "all payloads delivered")
        var received = 0
        var fecRecoveredSeqs: Set<UInt32> = []
        let rxLock = NSLock()
        mesh.responder.onPacket = { packet in
            rxLock.lock()
            received += 1
            if packet.header.timestampNanos == 0 {
                fecRecoveredSeqs.insert(packet.header.sequenceNumber)
            }
            let done = received == total
            rxLock.unlock()
            if done { allDelivered.fulfill() }
        }

        for n in 0..<total {
            mesh.initiator.sendAsync(payload: Data("bulk #\(n)".utf8))
        }
        wait(for: [allDelivered], timeout: 15)

        lock.lock(); rxLock.lock()
        let losses = droppedData.count
        let viaFEC = fecRecoveredSeqs.intersection(droppedData).count
        let rate = losses > 0 ? Double(viaFEC) / Double(losses) : 1
        print("[NMP] FEC recovery rate at 2% loss: \(viaFEC)/\(losses) "
              + "(\(String(format: "%.1f", rate * 100))%) recovered without NACK")
        XCTAssertGreaterThan(losses, 5, "loss injection must have actually fired")
        XCTAssertGreaterThanOrEqual(rate, 0.8,
            "≥80% of single-packet losses must be FEC-recovered")
        rxLock.unlock(); lock.unlock()
    }

    // MARK: Phase 2 vs Phase 3 recovery latency (success criterion: <50%)

    func testRecoveryLatencyFECVersusNack() throws {
        func measureRecovery(fecEnabled: Bool) throws -> Double {
            let mesh = try makeEstablishedMesh(fecEnabled: fecEnabled)
            let lostSeq: UInt32 = 3
            let lock = NSLock()
            var droppedOnce = false
            mesh.tInit.dropOutbound = { datagram in
                guard let h = Self.header(of: datagram), h.isEncrypted,
                      h.packetType == .data, h.sequenceNumber == lostSeq else { return false }
                lock.lock(); defer { lock.unlock() }
                if droppedOnce { return false }
                droppedOnce = true
                return true
            }
            let recovered = expectation(description: "payload recovered")
            var deliveredAt: DispatchTime?
            mesh.responder.onPacket = { packet in
                if packet.payload == Data("cmp #3".utf8), deliveredAt == nil {
                    deliveredAt = DispatchTime.now()
                    recovered.fulfill()
                }
            }
            let start = DispatchTime.now()
            for n in 0..<8 {
                mesh.initiator.sendAsync(payload: Data("cmp #\(n)".utf8))
            }
            wait(for: [recovered], timeout: 5)
            mesh.initiator.close(); mesh.responder.close()
            return Double(deliveredAt!.uptimeNanoseconds - start.uptimeNanoseconds) / 1e6
        }

        let nackMs = try measureRecovery(fecEnabled: false)
        let fecMs = try measureRecovery(fecEnabled: true)
        print("[NMP] recovery latency comparison: Phase 2 NACK \(String(format: "%.3f", nackMs)) ms "
              + "vs Phase 3 FEC \(String(format: "%.3f", fecMs)) ms "
              + "(\(String(format: "%.0f", fecMs / nackMs * 100))%)")
        XCTAssertLessThan(fecMs, nackMs * 0.5,
            "FEC recovery must be <50% of NACK recovery latency")
    }

    // MARK: AWDL suppression

    func testSuppressionDefersDataAndBackstopFlushes() throws {
        // FEC off so injected loss produces NACKs (the detector's signal);
        // short defer backstop so the test completes quickly.
        let mesh = try makeEstablishedMesh(fecEnabled: false) { cfg in
            cfg.awdl.minSendSamples = 10
            cfg.awdl.lossWindow = 0.5
            cfg.awdl.maxDeferDelay = 0.3
        }

        // Sustained heavy loss: drop 40% of first-transmission data packets.
        var rng = SplitMix64(seed: 42)
        let lock = NSLock()
        var seenSeqs: Set<UInt32> = []
        var dropping = true
        mesh.tInit.dropOutbound = { datagram in
            guard let h = Self.header(of: datagram), h.isEncrypted,
                  h.packetType == .data else { return false }
            lock.lock(); defer { lock.unlock() }
            guard dropping, seenSeqs.insert(h.sequenceNumber).inserted else { return false }
            return rng.chance(0.4)
        }

        let engaged = expectation(description: "suppression engaged")
        engaged.assertForOverFulfill = false
        mesh.initiator.onDiagnostic = { message in
            if message.contains("AWDL suppression engaged") { engaged.fulfill() }
        }

        // Stream packets with a small gap so NACKs flow back between sends.
        for n in 0..<40 {
            mesh.initiator.sendAsync(payload: Data("storm #\(n)".utf8))
            Thread.sleep(forTimeInterval: 0.002)
        }
        wait(for: [engaged], timeout: 5)

        // Under suppression a normal-priority send defers (returns nil)…
        lock.lock(); dropping = false; lock.unlock()
        let deferred = expectation(description: "probe deferred")
        let probePayload = Data("deferred probe".utf8)
        mesh.initiator.sendAsync(payload: probePayload) { result in
            if case .success(nil) = result { deferred.fulfill() }
        }
        wait(for: [deferred], timeout: 2)

        // …and still reaches the peer once the backstop flushes it.
        let delivered = expectation(description: "deferred probe delivered")
        delivered.assertForOverFulfill = false
        mesh.responder.onPacket = { packet in
            if packet.payload == probePayload { delivered.fulfill() }
        }
        wait(for: [delivered], timeout: 3)
    }

    func testNoSuppressionOnWiredOrLoopbackPath() throws {
        // Same loss storm as above, but the transport reports a path that
        // cannot experience AWDL contention — shaping must stay out of the
        // way entirely: no engagement, and sends never defer (non-nil seq).
        let mesh = try makeEstablishedMesh(fecEnabled: false) { cfg in
            cfg.awdl.minSendSamples = 10
            cfg.awdl.lossWindow = 0.5
        }
        mesh.tInit.linkKind = .wiredOrLoopback
        mesh.tResp.linkKind = .wiredOrLoopback

        var rng = SplitMix64(seed: 42)
        let lock = NSLock()
        var seenSeqs: Set<UInt32> = []
        mesh.tInit.dropOutbound = { datagram in
            guard let h = Self.header(of: datagram), h.isEncrypted,
                  h.packetType == .data else { return false }
            lock.lock(); defer { lock.unlock() }
            guard seenSeqs.insert(h.sequenceNumber).inserted else { return false }
            return rng.chance(0.4)
        }
        mesh.initiator.onDiagnostic = { message in
            XCTAssertFalse(message.contains("AWDL suppression engaged"),
                           "wired/loopback path must never engage suppression")
        }

        let allSent = expectation(description: "every send got a sequence")
        allSent.expectedFulfillmentCount = 40
        for n in 0..<40 {
            mesh.initiator.sendAsync(payload: Data("storm #\(n)".utf8)) { result in
                if case .success(.some) = result { allSent.fulfill() }
            }
            Thread.sleep(forTimeInterval: 0.002)
        }
        wait(for: [allSent], timeout: 5)
    }

    func testNoParityOnWiredOrLoopbackPath() throws {
        // FEC parity is a radio-loss feature; a wired/loopback sender must
        // not spend +25% packets on it. Headers are plaintext AAD, so the
        // wire can be audited without decrypting.
        let mesh = try makeEstablishedMesh(fecEnabled: true)
        mesh.tInit.linkKind = .wiredOrLoopback

        let delivered = expectation(description: "all payloads delivered")
        delivered.expectedFulfillmentCount = 8
        mesh.responder.onPacket = { _ in delivered.fulfill() }
        for n in 0..<8 {
            mesh.initiator.sendAsync(payload: Data("wired #\(n)".utf8))
        }
        wait(for: [delivered], timeout: 5)

        let parityCount = mesh.tInit.sentDatagrams
            .compactMap(Self.header(of:))
            .filter { $0.packetType == .fecRecovery }
            .count
        XCTAssertEqual(parityCount, 0,
                       "wired/loopback path must not emit FEC parity")
    }

    func testRecommendedChunkBytesFollowsLinkKind() throws {
        let mesh = try makeEstablishedMesh()

        // Unknown: conservative default, whatever the kernel allows.
        mesh.tInit.maxDatagramBytes = 9216
        XCTAssertEqual(mesh.initiator.recommendedChunkBytes,
                       NMPTensorChunk.defaultChunkBytes)

        // Radio: MTU-packed but never fragmenting.
        mesh.tInit.linkKind = .radio
        XCTAssertEqual(mesh.initiator.recommendedChunkBytes,
                       PeerConnection.radioChunkBytes)

        // Wired/loopback with a known ceiling: ceiling minus seal overhead.
        mesh.tInit.linkKind = .wiredOrLoopback
        XCTAssertEqual(mesh.initiator.recommendedChunkBytes,
                       9216 - NMPHeader.byteCount - NMPHeader.gcmTagByteCount)

        // Wired/loopback but ceiling unknown: stay at the safe default.
        mesh.tInit.maxDatagramBytes = nil
        XCTAssertEqual(mesh.initiator.recommendedChunkBytes,
                       NMPTensorChunk.defaultChunkBytes)

        // Never exceed the UInt16 payload-length field.
        mesh.tInit.maxDatagramBytes = 1 << 20
        XCTAssertEqual(mesh.initiator.recommendedChunkBytes,
                       NMPHeader.maxPayloadLength)
    }

    func testSendBurstDeliversInOrderWithFlushOnLast() throws {
        let mesh = try makeEstablishedMesh()
        let payloads = (0..<10).map { Data("burst #\($0)".utf8) }

        let all = expectation(description: "burst delivered in order")
        let lock = NSLock()
        var received: [(payload: Data, flushed: Bool)] = []
        mesh.responder.onPacket = { packet in
            lock.lock()
            received.append((packet.payload,
                             packet.header.flags.contains(.flush)))
            let count = received.count
            lock.unlock()
            if count == payloads.count { all.fulfill() }
        }
        mesh.initiator.sendBurstAsync(payloads: payloads) { error in
            XCTAssertNil(error)
        }
        wait(for: [all], timeout: 5)

        lock.lock(); defer { lock.unlock() }
        XCTAssertEqual(received.map(\.payload), payloads,
                       "burst must preserve submit order")
        XCTAssertEqual(received.map(\.flushed),
                       Array(repeating: false, count: payloads.count - 1) + [true],
                       "FLUSH must land on exactly the last payload")
    }

    func testCriticalDataBypassesSuppression() throws {
        let mesh = try makeEstablishedMesh(fecEnabled: false) { cfg in
            cfg.awdl.minSendSamples = 10
            cfg.awdl.lossWindow = 0.5
        }
        var rng = SplitMix64(seed: 7)
        let lock = NSLock()
        var seenSeqs: Set<UInt32> = []
        mesh.tInit.dropOutbound = { datagram in
            guard let h = Self.header(of: datagram), h.isEncrypted,
                  h.packetType == .data else { return false }
            lock.lock(); defer { lock.unlock() }
            guard seenSeqs.insert(h.sequenceNumber).inserted else { return false }
            return rng.chance(0.4)
        }
        let engaged = expectation(description: "suppression engaged")
        engaged.assertForOverFulfill = false
        mesh.initiator.onDiagnostic = { message in
            if message.contains("AWDL suppression engaged") { engaged.fulfill() }
        }
        for n in 0..<40 {
            mesh.initiator.sendAsync(payload: Data("storm #\(n)".utf8))
            Thread.sleep(forTimeInterval: 0.002)
        }
        wait(for: [engaged], timeout: 5)

        // Critical-priority data is sealed immediately (non-nil sequence).
        let sentNow = expectation(description: "critical send not deferred")
        mesh.initiator.sendAsync(priority: .critical,
                                 payload: Data("critical".utf8)) { result in
            if case .success(.some) = result { sentNow.fulfill() }
        }
        wait(for: [sentNow], timeout: 2)
    }
}

// MARK: - Deterministic RNG for loss injection

private struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func chance(_ p: Double) -> Bool {
        Double(next() >> 11) / Double(1 << 53) < p
    }
}
