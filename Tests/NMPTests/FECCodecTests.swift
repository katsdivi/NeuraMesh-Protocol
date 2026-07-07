//
//  FECCodecTests.swift
//  NMPTests — Phase 3
//
//  XOR parity primitives, CRC32, the parity packet wire format, and the
//  sender/receiver group state machines. Includes the Phase 3 performance
//  gates: parity computation <100 µs per 4×1400 B group, reconstruction
//  <1 ms.
//

import XCTest
@testable import NMP

// MARK: - Codec

final class FECCodecTests: XCTestCase {

    private let payloads = [
        Data([0x01, 0x02, 0x03, 0x04]),
        Data([0x50, 0x60, 0x70, 0x80]),
        Data([0xAA, 0xBB, 0xCC, 0xDD]),
        Data([0xFF, 0x00, 0xFF, 0x00]),
    ]

    func testParityRoundTripEveryMissingIndex() throws {
        let parity = NMPFECCodec.computeParity(payloads)
        for missing in 0..<payloads.count {
            let surviving = payloads.enumerated()
                .filter { $0.offset != missing }.map(\.element)
            let recovered = try NMPFECCodec.reconstruct(
                parity: parity, surviving: surviving,
                missingLength: payloads[missing].count)
            XCTAssertEqual(recovered, payloads[missing], "missing index \(missing)")
        }
    }

    func testUnequalPayloadLengths() throws {
        let uneven = [
            Data([0x11, 0x22, 0x33]),
            Data(repeating: 0x44, count: 10),
            Data([0x55]),
            Data(repeating: 0x66, count: 7),
        ]
        let parity = NMPFECCodec.computeParity(uneven)
        XCTAssertEqual(parity.count, 10) // padded to longest member
        for missing in 0..<uneven.count {
            let surviving = uneven.enumerated()
                .filter { $0.offset != missing }.map(\.element)
            let recovered = try NMPFECCodec.reconstruct(
                parity: parity, surviving: surviving,
                missingLength: uneven[missing].count)
            XCTAssertEqual(recovered, uneven[missing], "missing index \(missing)")
        }
    }

    func testSingleMemberGroup() throws {
        // A flush after one packet produces a 1-member group; its parity IS
        // the payload, so losing the member is still recoverable.
        let solo = Data([9, 8, 7])
        let parity = NMPFECCodec.computeParity([solo])
        let recovered = try NMPFECCodec.reconstruct(
            parity: parity, surviving: [], missingLength: solo.count)
        XCTAssertEqual(recovered, solo)
    }

    func testCRC32KnownVector() {
        // Canonical IEEE 802.3 check value.
        XCTAssertEqual(NMPCRC32.checksum(Data("123456789".utf8)), 0xCBF4_3926)
    }

    func testParityPayloadWireRoundTrip() throws {
        let sequences: [UInt32] = [10, 11, 13, 14] // non-contiguous (NACK interleaved)
        let parity = NMPFECCodec.computeParity(payloads)
        let wire = NMPFECCodec.encodeParityPayload(
            sequences: sequences,
            payloadLengths: payloads.map(\.count),
            parity: parity)
        let decoded = try NMPFECCodec.decodeParityPayload(wire)
        XCTAssertEqual(decoded.sequences, sequences)
        XCTAssertEqual(decoded.lengths, payloads.map(\.count))
        XCTAssertEqual(decoded.parity, parity)
        XCTAssertEqual(decoded.groupID, NMPFECCodec.groupID(for: sequences))
    }

    func testDecodeRejectsMalformedPayloads() {
        // Truncated header.
        XCTAssertThrowsError(try NMPFECCodec.decodeParityPayload(Data([0, 0, 0])))
        // Count of zero / oversized count.
        XCTAssertThrowsError(try NMPFECCodec.decodeParityPayload(
            Data([0, 0, 0, 0, 0])))
        XCTAssertThrowsError(try NMPFECCodec.decodeParityPayload(
            Data([0, 0, 0, 0, 200])))
        // Corrupted group ID (fails the CRC cross-check).
        var wire = NMPFECCodec.encodeParityPayload(
            sequences: [1, 2, 3, 4], payloadLengths: [4, 4, 4, 4],
            parity: NMPFECCodec.computeParity(payloads))
        wire[0] ^= 0xFF
        XCTAssertThrowsError(try NMPFECCodec.decodeParityPayload(wire)) {
            XCTAssertEqual($0 as? NMPFECError, .groupIDMismatch)
        }
        // Declared length exceeding the parity body.
        let bad = NMPFECCodec.encodeParityPayload(
            sequences: [1], payloadLengths: [100], parity: Data([0x00]))
        XCTAssertThrowsError(try NMPFECCodec.decodeParityPayload(bad)) {
            XCTAssertEqual($0 as? NMPFECError, .lengthExceedsParity)
        }
    }

    func testSliceOffsetsHandled() throws {
        let wire = NMPFECCodec.encodeParityPayload(
            sequences: [7], payloadLengths: [2], parity: Data([1, 2]))
        let shifted = Data([0xEE, 0xEE]) + wire
        XCTAssertNoThrow(try NMPFECCodec.decodeParityPayload(shifted.dropFirst(2)))
    }

    // MARK: Performance gates (Phase 3 success criteria)

    func testParityComputationUnder100Microseconds() {
        let group = (0..<4).map { Data(repeating: UInt8($0), count: 1400) }
        _ = NMPFECCodec.computeParity(group) // warm up
        let iterations = 200
        let start = DispatchTime.now()
        for _ in 0..<iterations { _ = NMPFECCodec.computeParity(group) }
        let avgMicros = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds)
            / Double(iterations) / 1_000
        print("[NMP] FEC parity computation: \(String(format: "%.2f", avgMicros)) µs per 4×1400B group")
        XCTAssertLessThan(avgMicros, 100, "parity computation must stay under 100 µs")
    }

    func testReconstructionUnder1Millisecond() throws {
        let group = (0..<4).map { Data(repeating: UInt8($0 + 1), count: 1400) }
        let parity = NMPFECCodec.computeParity(group)
        let surviving = Array(group.dropFirst())
        _ = try NMPFECCodec.reconstruct(parity: parity, surviving: surviving,
                                        missingLength: 1400) // warm up
        let iterations = 200
        let start = DispatchTime.now()
        for _ in 0..<iterations {
            _ = try NMPFECCodec.reconstruct(parity: parity, surviving: surviving,
                                            missingLength: 1400)
        }
        let avgMs = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds)
            / Double(iterations) / 1_000_000
        print("[NMP] FEC reconstruction: \(String(format: "%.4f", avgMs)) ms per packet")
        XCTAssertLessThan(avgMs, 1, "reconstruction must stay under 1 ms")
    }
}

// MARK: - Sender group builder

final class FECGroupBuilderTests: XCTestCase {

    func testParityEmittedOnFourthPacket() throws {
        var builder = NMPFECGroupBuilder(groupSize: 4)
        var parityPayload: Data?
        for i in 0..<4 {
            let closes = builder.willCloseGroup(flush: false)
            XCTAssertEqual(closes, i == 3)
            parityPayload = builder.add(sequence: UInt32(i), payload: Data([UInt8(i)]),
                                        closesGroup: closes)
            if i < 3 { XCTAssertNil(parityPayload) }
        }
        let descriptor = try NMPFECCodec.decodeParityPayload(try XCTUnwrap(parityPayload))
        XCTAssertEqual(descriptor.sequences, [0, 1, 2, 3])
        XCTAssertEqual(descriptor.parity, NMPFECCodec.computeParity(
            (0..<4).map { Data([UInt8($0)]) }))
    }

    func testFlushClosesGroupEarly() throws {
        var builder = NMPFECGroupBuilder(groupSize: 4)
        XCTAssertNil(builder.add(sequence: 0, payload: Data([1]),
                                 closesGroup: builder.willCloseGroup(flush: false)))
        XCTAssertTrue(builder.willCloseGroup(flush: true))
        let parity = builder.add(sequence: 1, payload: Data([2]), closesGroup: true)
        let descriptor = try NMPFECCodec.decodeParityPayload(try XCTUnwrap(parity))
        XCTAssertEqual(descriptor.sequences, [0, 1])
        // Next group starts clean.
        XCTAssertFalse(builder.willCloseGroup(flush: false))
    }
}

// MARK: - Receiver group state

final class FECGroupReceiverTests: XCTestCase {

    private func parityPayload(sequences: [UInt32], payloads: [Data]) -> Data {
        NMPFECCodec.encodeParityPayload(
            sequences: sequences,
            payloadLengths: payloads.map(\.count),
            parity: NMPFECCodec.computeParity(payloads))
    }

    func testAllMembersPresentNoRecovery() throws {
        var receiver = NMPFECGroupReceiver(pendingTimeout: 0.05)
        let payloads = (0..<4).map { Data([UInt8($0 + 1)]) }
        for (i, payload) in payloads.enumerated() {
            XCTAssertEqual(receiver.observeData(sequence: UInt32(i), payload: payload, at: 0), [])
        }
        let recoveries = try receiver.observeParity(
            parityPayload(sequences: [0, 1, 2, 3], payloads: payloads), at: 0)
        XCTAssertEqual(recoveries, [])
    }

    func testSingleMissingMemberReconstructed() throws {
        var receiver = NMPFECGroupReceiver(pendingTimeout: 0.05)
        let payloads = (0..<4).map { Data([UInt8($0 + 1), UInt8($0 + 10)]) }
        for i in [0, 1, 3] { // seq 2 lost
            _ = receiver.observeData(sequence: UInt32(i), payload: payloads[i], at: 0)
        }
        let recoveries = try receiver.observeParity(
            parityPayload(sequences: [0, 1, 2, 3], payloads: payloads), at: 0)
        XCTAssertEqual(recoveries.count, 1)
        XCTAssertEqual(recoveries.first?.sequence, 2)
        XCTAssertEqual(recoveries.first?.payload, payloads[2])
    }

    func testParityArrivingBeforeDataStillRecovers() throws {
        var receiver = NMPFECGroupReceiver(pendingTimeout: 0.05)
        let payloads = (0..<4).map { Data([UInt8($0 * 3)]) }
        // Parity reordered ahead of everything.
        XCTAssertEqual(try receiver.observeParity(
            parityPayload(sequences: [0, 1, 2, 3], payloads: payloads), at: 0), [])
        _ = receiver.observeData(sequence: 0, payload: payloads[0], at: 0.001)
        _ = receiver.observeData(sequence: 1, payload: payloads[1], at: 0.002)
        // Third arrival leaves exactly one missing → reconstruction fires.
        let recoveries = receiver.observeData(sequence: 3, payload: payloads[3], at: 0.003)
        XCTAssertEqual(recoveries.first?.sequence, 2)
        XCTAssertEqual(recoveries.first?.payload, payloads[2])
    }

    func testTwoMissingMembersWaitThenExpire() throws {
        var receiver = NMPFECGroupReceiver(pendingTimeout: 0.05)
        let payloads = (0..<4).map { Data([UInt8($0)]) }
        _ = receiver.observeData(sequence: 0, payload: payloads[0], at: 0)
        _ = receiver.observeData(sequence: 1, payload: payloads[1], at: 0)
        // Seqs 2 and 3 missing: single parity cannot recover two losses.
        XCTAssertEqual(try receiver.observeParity(
            parityPayload(sequences: [0, 1, 2, 3], payloads: payloads), at: 0), [])
        // …but a NACK retransmit filling one of them unlocks the other.
        let recoveries = receiver.observeData(sequence: 2, payload: payloads[2], at: 0.01)
        XCTAssertEqual(recoveries.first?.sequence, 3)
    }

    func testStalePendingGroupExpires() throws {
        var receiver = NMPFECGroupReceiver(pendingTimeout: 0.05)
        let payloads = (0..<4).map { Data([UInt8($0)]) }
        _ = try receiver.observeParity(
            parityPayload(sequences: [0, 1, 2, 3], payloads: payloads), at: 0)
        XCTAssertEqual(receiver.expirePending(at: 0.01), []) // still fresh
        let expired = receiver.expirePending(at: 0.1)
        XCTAssertEqual(expired, [NMPFECCodec.groupID(for: [0, 1, 2, 3])])
        // Late data can no longer trigger recovery for the expired group.
        for i in [0, 1, 2] {
            XCTAssertEqual(receiver.observeData(sequence: UInt32(i),
                                                payload: payloads[i], at: 0.2), [])
        }
    }
}
