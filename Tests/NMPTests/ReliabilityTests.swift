//
//  ReliabilityTests.swift
//  NMPTests — Phase 2
//
//  NACK-only reliability: NACK payload codec, retransmit ring buffer, loss
//  tracker state machine (deterministic, synthetic clock), and end-to-end
//  loss recovery between two mock peers (drop → gap → NACK → verbatim
//  retransmit → delivery), including the give-up path.
//

import XCTest
@testable import NMP

// MARK: - NACK codec

final class NackCodecTests: XCTestCase {

    func testRoundTrip() throws {
        let seqs: [UInt32] = [3, 7, 0xFFFF_FFFE]
        XCTAssertEqual(try NMPNackCodec.decode(NMPNackCodec.encode(seqs)), seqs)
    }

    func testEmptyListRoundTrip() throws {
        XCTAssertEqual(try NMPNackCodec.decode(NMPNackCodec.encode([])), [])
    }

    func testTruncatedPayloadThrows() {
        XCTAssertThrowsError(try NMPNackCodec.decode(Data([0x00])))
        XCTAssertThrowsError(try NMPNackCodec.decode(Data()))
    }

    func testCountLengthMismatchThrows() {
        var bad = Data()
        bad.appendBigEndian(UInt16(2))   // declares 2 sequences…
        bad.appendBigEndian(UInt32(9))   // …carries 1
        XCTAssertThrowsError(try NMPNackCodec.decode(bad)) {
            XCTAssertEqual($0 as? NMPReliabilityError, .malformedNack)
        }
    }

    func testSliceOffsetsHandled() throws {
        let wire = Data([0xEE, 0xEE]) + NMPNackCodec.encode([42])
        XCTAssertEqual(try NMPNackCodec.decode(wire.dropFirst(2)), [42])
    }
}

// MARK: - Retransmit buffer

final class RetransmitBufferTests: XCTestCase {

    func testStoreAndRetrieve() {
        var buf = NMPRetransmitBuffer()
        buf.store(sequence: 5, datagram: Data([5]))
        XCTAssertEqual(buf.datagram(for: 5), Data([5]))
        XCTAssertNil(buf.datagram(for: 4))
    }

    func testEvictionAfterWindowSize() {
        var buf = NMPRetransmitBuffer()
        let window = NMPReliabilityConfig.windowSize
        for seq in 0..<(window * 2) {
            buf.store(sequence: seq, datagram: Data([UInt8(seq & 0xFF)]))
        }
        // Everything within the last `window` sequences survives…
        for seq in window..<(window * 2) {
            XCTAssertEqual(buf.datagram(for: seq), Data([UInt8(seq & 0xFF)]))
        }
        // …everything older was evicted by its same-residue successor.
        for seq in 0..<window {
            XCTAssertNil(buf.datagram(for: seq))
        }
    }
}

// MARK: - Loss tracker (synthetic clock)

final class LossTrackerTests: XCTestCase {

    private var config: NMPReliabilityConfig {
        var c = NMPReliabilityConfig()
        c.reorderDelay = 0.010
        c.nackRetryInterval = 0.020
        c.maxNackAttempts = 2
        return c
    }

    func testInOrderArrivalTracksNothing() {
        var t = NMPLossTracker(config: config)
        for seq: UInt32 in 0...5 { t.observe(sequence: seq, at: 0) }
        XCTAssertTrue(t.missing.isEmpty)
        XCTAssertNil(t.nextDeadline)
    }

    func testGapDetectedAndNackedAfterReorderDelay() {
        var t = NMPLossTracker(config: config)
        t.observe(sequence: 0, at: 0)
        t.observe(sequence: 3, at: 0)   // 1 and 2 missing
        XCTAssertEqual(Set(t.missing.keys), [1, 2])

        // Inside the reorder grace window: nothing due yet.
        XCTAssertEqual(t.dueNacks(at: 0.005).nack, [])
        // After it: both due, in order.
        let due = t.dueNacks(at: 0.011)
        XCTAssertEqual(due.nack, [1, 2])
        XCTAssertEqual(due.gaveUp, [])
    }

    func testLateArrivalFillsGapBeforeNack() {
        var t = NMPLossTracker(config: config)
        t.observe(sequence: 0, at: 0)
        t.observe(sequence: 2, at: 0)
        t.observe(sequence: 1, at: 0.002) // reordered, arrives late
        XCTAssertTrue(t.missing.isEmpty)
        XCTAssertEqual(t.dueNacks(at: 1).nack, [])
    }

    func testRetryThenGiveUp() {
        var t = NMPLossTracker(config: config)
        t.observe(sequence: 0, at: 0)
        t.observe(sequence: 2, at: 0)   // 1 missing
        XCTAssertEqual(t.dueNacks(at: 0.010).nack, [1])   // attempt 1
        XCTAssertEqual(t.dueNacks(at: 0.015).nack, [])    // not due again yet
        XCTAssertEqual(t.dueNacks(at: 0.030).nack, [1])   // attempt 2 (max)
        let final = t.dueNacks(at: 0.050)                 // attempts exhausted
        XCTAssertEqual(final.nack, [])
        XCTAssertEqual(final.gaveUp, [1])
        XCTAssertTrue(t.missing.isEmpty)
    }

    func testExpediteAllMakesGapsImmediatelyDue() {
        var t = NMPLossTracker(config: config)
        t.observe(sequence: 0, at: 0)
        t.observe(sequence: 2, at: 0)
        t.expediteAll(at: 0.001) // FLUSH arrived
        XCTAssertEqual(t.dueNacks(at: 0.001).nack, [1])
    }

    func testFirstArrivalAboveZeroRecordsLeadingGap() {
        var t = NMPLossTracker(config: config)
        t.observe(sequence: 3, at: 0) // seq 0-2 never seen
        XCTAssertEqual(Set(t.missing.keys), [0, 1, 2])
    }

    func testGapDeeperThanWindowTracksOnlyTail() {
        var t = NMPLossTracker(config: config)
        t.observe(sequence: 0, at: 0)
        t.observe(sequence: 500, at: 0)
        // The sender's retransmit buffer holds the last 64 sequences
        // (437...500 once 500 is sealed), so only the 63 missing sequences
        // inside that range are worth NACKing.
        let window = NMPReliabilityConfig.windowSize
        XCTAssertEqual(t.missing.count, Int(window) - 1)
        XCTAssertEqual(t.missing.keys.min(), 500 - window + 1)
        XCTAssertEqual(t.missing.keys.max(), 499)
    }

    func testMarkRecoveredCancelsPendingNack() {
        // Phase 3: an FEC reconstruction cancels the gap's NACK.
        var t = NMPLossTracker(config: config)
        t.observe(sequence: 0, at: 0)
        t.observe(sequence: 2, at: 0)
        t.markRecovered(1)
        XCTAssertTrue(t.missing.isEmpty)
        XCTAssertEqual(t.dueNacks(at: 1).nack, [])
    }

    func testPostponeUnattemptedDelaysOnlyFirstNacks() {
        // Phase 3: contention inferred — un-NACKed gaps wait for FEC, gaps
        // already being retried keep their schedule.
        var t = NMPLossTracker(config: config)
        t.observe(sequence: 0, at: 0)
        t.observe(sequence: 2, at: 0)              // gap 1, due 0.010
        XCTAssertEqual(t.dueNacks(at: 0.010).nack, [1]) // attempt 1, due 0.030
        t.observe(sequence: 4, at: 0.011)          // gap 3, due 0.021
        t.postponeUnattempted(until: 0.100)
        XCTAssertEqual(t.dueNacks(at: 0.030).nack, [1]) // attempted gap unaffected
        XCTAssertEqual(t.dueNacks(at: 0.050).nack, [])  // gap 3 postponed
        XCTAssertEqual(t.dueNacks(at: 0.100).nack, [3])
    }

    func testGapAgedOutOfWindowIsAbandoned() {
        var t = NMPLossTracker(config: config)
        t.observe(sequence: 0, at: 0)
        t.observe(sequence: 2, at: 0)   // 1 missing
        // The stream races far ahead; seq 1 leaves the retransmit window.
        let aged = t.observe(sequence: 200, at: 0.001)
        XCTAssertTrue(aged.contains(1))
        XCTAssertFalse(t.missing.keys.contains(1))
    }
}

// MARK: - End-to-end loss recovery (mock transport)

final class ReliabilityEnd2EndTests: XCTestCase {

    private struct Mesh {
        let initiator: PeerConnection
        let responder: PeerConnection
        let tInit: MockTransport
        let tResp: MockTransport
    }

    /// Fast reliability timings so tests complete in tens of milliseconds.
    private static func fastReliability() -> NMPReliabilityConfig {
        var r = NMPReliabilityConfig()
        r.reorderDelay = 0.008
        r.nackRetryInterval = 0.020
        r.maxNackAttempts = 3
        return r
    }

    private func makeEstablishedMesh(
        initiatorReliability: NMPReliabilityConfig = fastReliability(),
        responderReliability: NMPReliabilityConfig = fastReliability()
    ) throws -> Mesh {
        let (tInit, tResp) = MockTransport.pair()
        let sInit = NoiseStaticKeyPair()
        let sResp = NoiseStaticKeyPair()

        var cfgI = PeerConnectionConfig(localPeerID: 1)
        cfgI.reliability = initiatorReliability
        var cfgR = PeerConnectionConfig(localPeerID: 2)
        cfgR.reliability = responderReliability
        // These tests exercise the Phase 2 NACK path in ISOLATION. With FEC
        // (Phase 3) enabled, parity packets would recover the injected
        // losses first and nothing here would test NACK. FEC-on loss
        // recovery is covered by FECIntegrationTests.
        cfgI.fec.enabled = false
        cfgR.fec.enabled = false

        let initiator = try PeerConnection(
            role: .initiator, config: cfgI, transport: tInit,
            localStatic: sInit, remoteStaticPublicKey: sResp.publicKeyData,
            queue: DispatchQueue(label: "nmp.rel.init"))
        let responder = try PeerConnection(
            role: .responder, config: cfgR, transport: tResp,
            localStatic: sResp,
            queue: DispatchQueue(label: "nmp.rel.resp"))

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

    /// Encrypted headers are plaintext AAD, so loss can be injected per
    /// sequence number without decrypting.
    private static func isDataPacket(_ datagram: Data, sequence: UInt32) -> Bool {
        guard let header = try? NMPPacketCodec.decodeHeader(datagram) else { return false }
        return header.isEncrypted && header.packetType == .data
            && header.sequenceNumber == sequence
    }

    func testLostPacketRecoveredViaNack() throws {
        let mesh = try makeEstablishedMesh()
        let lostSeq: UInt32 = 3
        let total = 10

        // Drop the first transmission of seq 3; let its retransmit through.
        let lock = NSLock()
        var droppedOnce = false
        var dropTime: DispatchTime?
        mesh.tInit.dropOutbound = { datagram in
            guard Self.isDataPacket(datagram, sequence: lostSeq) else { return false }
            lock.lock(); defer { lock.unlock() }
            if droppedOnce { return false }
            droppedOnce = true
            dropTime = DispatchTime.now()
            return true
        }

        let allDelivered = expectation(description: "all payloads delivered")
        var received: Set<Data> = []
        var recoveredAt: DispatchTime?
        mesh.responder.onPacket = { packet in
            if packet.header.sequenceNumber == lostSeq { recoveredAt = DispatchTime.now() }
            received.insert(packet.payload)
            if received.count == total { allDelivered.fulfill() }
        }

        for n in 0..<total {
            mesh.initiator.sendAsync(payload: Data("packet #\(n)".utf8))
        }
        wait(for: [allDelivered], timeout: 5)

        XCTAssertEqual(received, Set((0..<total).map { Data("packet #\($0)".utf8) }))
        if let dropTime, let recoveredAt {
            let ms = Double(recoveredAt.uptimeNanoseconds - dropTime.uptimeNanoseconds) / 1e6
            print("[NMP] NACK loss recovery time: \(String(format: "%.3f", ms)) ms")
            XCTAssertLessThan(ms, 100, "recovery took \(ms) ms (target <100ms)")
        } else {
            XCTFail("loss was never injected or never recovered")
        }
    }

    func testBurstLossRecovered() throws {
        let mesh = try makeEstablishedMesh()
        let lostSeqs: Set<UInt32> = [2, 3, 4]
        let total = 12

        let lock = NSLock()
        var droppedSeqs: Set<UInt32> = []
        mesh.tInit.dropOutbound = { datagram in
            guard let header = try? NMPPacketCodec.decodeHeader(datagram),
                  header.isEncrypted, header.packetType == .data,
                  lostSeqs.contains(header.sequenceNumber) else { return false }
            lock.lock(); defer { lock.unlock() }
            return droppedSeqs.insert(header.sequenceNumber).inserted // drop first try only
        }

        let allDelivered = expectation(description: "all payloads delivered")
        var received: Set<Data> = []
        mesh.responder.onPacket = { packet in
            received.insert(packet.payload)
            if received.count == total { allDelivered.fulfill() }
        }

        for n in 0..<total {
            mesh.initiator.sendAsync(payload: Data("burst #\(n)".utf8))
        }
        wait(for: [allDelivered], timeout: 5)
        XCTAssertEqual(received.count, total)
    }

    func testFlushExpeditesNackWhenNothingFollows() throws {
        // Make the plain reorder delay effectively infinite: only the FLUSH
        // path can trigger a NACK fast enough for this test to pass.
        var slow = Self.fastReliability()
        slow.reorderDelay = 30
        let mesh = try makeEstablishedMesh(responderReliability: slow)

        let lostSeq: UInt32 = 0
        let lock = NSLock()
        var droppedOnce = false
        mesh.tInit.dropOutbound = { datagram in
            guard Self.isDataPacket(datagram, sequence: lostSeq) else { return false }
            lock.lock(); defer { lock.unlock() }
            if droppedOnce { return false }
            droppedOnce = true
            return true
        }

        let recovered = expectation(description: "lost packet recovered")
        mesh.responder.onPacket = { packet in
            if packet.header.sequenceNumber == lostSeq { recovered.fulfill() }
        }

        mesh.initiator.sendAsync(payload: Data("lost".utf8))                  // dropped
        mesh.initiator.sendAsync(flags: [.flush], payload: Data("end".utf8)) // reveals gap, expedites
        wait(for: [recovered], timeout: 2)
    }

    func testUnrecoverableLossReported() throws {
        let mesh = try makeEstablishedMesh()
        let lostSeq: UInt32 = 1

        // Drop seq 1 EVERY time — retransmits included. NACK attempts must
        // exhaust and the loss must surface via onUnrecoverableLoss.
        mesh.tInit.dropOutbound = { Self.isDataPacket($0, sequence: lostSeq) }

        let reported = expectation(description: "unrecoverable loss reported")
        mesh.responder.onUnrecoverableLoss = { sequences in
            if sequences.contains(lostSeq) { reported.fulfill() }
        }

        mesh.initiator.sendAsync(payload: Data("seq0".utf8))
        mesh.initiator.sendAsync(payload: Data("gone".utf8))  // seq 1, black-holed
        mesh.initiator.sendAsync(payload: Data("seq2".utf8))
        wait(for: [reported], timeout: 5)
    }

    func testNackLossItselfIsRecovered() throws {
        // Even the NACK packet can be lost; the reveal is that the responder
        // re-NACKs on the retry interval and recovery still completes.
        let mesh = try makeEstablishedMesh()
        let lostSeq: UInt32 = 0

        let dataLock = NSLock()
        var droppedData = false
        mesh.tInit.dropOutbound = { datagram in
            guard Self.isDataPacket(datagram, sequence: lostSeq) else { return false }
            dataLock.lock(); defer { dataLock.unlock() }
            if droppedData { return false }
            droppedData = true
            return true
        }
        // Drop the responder's first NACK.
        let nackLock = NSLock()
        var droppedNack = false
        mesh.tResp.dropOutbound = { datagram in
            guard let header = try? NMPPacketCodec.decodeHeader(datagram),
                  header.isEncrypted, header.packetType == .nack else { return false }
            nackLock.lock(); defer { nackLock.unlock() }
            if droppedNack { return false }
            droppedNack = true
            return true
        }

        let recovered = expectation(description: "recovered despite lost NACK")
        mesh.responder.onPacket = { packet in
            if packet.header.sequenceNumber == lostSeq { recovered.fulfill() }
        }

        mesh.initiator.sendAsync(payload: Data("lost".utf8))
        mesh.initiator.sendAsync(payload: Data("follow".utf8))
        wait(for: [recovered], timeout: 5)
    }
}
