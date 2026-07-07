//
//  PacketCodecTests.swift
//  NMPTests — Phase 1
//

import XCTest
@testable import NMP

final class PacketCodecTests: XCTestCase {

    private func makeHeader(
        encrypted: Bool = true,
        flags: NMPFlags = [.flush],
        type: NMPPacketType = .data,
        payloadLength: UInt16 = 1234,
        seq: UInt32 = 0xDEADBEEF,
        peer: UInt32 = 0xCAFEBABE,
        ts: UInt64 = 1_720_000_000_123_456_789
    ) -> NMPHeader {
        NMPHeader(
            isEncrypted: encrypted, flags: flags, packetType: type,
            payloadLength: payloadLength, sequenceNumber: seq,
            senderPeerID: peer, timestampNanos: ts
        )
    }

    // MARK: Round-trips

    func testHeaderRoundTrip() throws {
        let header = makeHeader()
        let encoded = NMPPacketCodec.encodeHeader(header)
        XCTAssertEqual(encoded.count, NMPHeader.byteCount)
        let decoded = try NMPPacketCodec.decodeHeader(encoded)
        XCTAssertEqual(decoded, header)
    }

    func testHeaderRoundTripAllTypesAndFlags() throws {
        let flagSets: [NMPFlags] = [[], [.fecGroupEnd], [.retransmit], [.flush],
                                    [.fecGroupEnd, .retransmit, .flush]]
        for type in NMPPacketType.allCases {
            for flags in flagSets {
                let header = makeHeader(encrypted: !type.isHandshake, flags: flags, type: type)
                let decoded = try NMPPacketCodec.decodeHeader(NMPPacketCodec.encodeHeader(header))
                XCTAssertEqual(decoded, header, "type=\(type) flags=\(flags.rawValue)")
            }
        }
    }

    func testPlaintextPacketRoundTrip() throws {
        let payload = Data((0..<300).map { UInt8($0 & 0xFF) })
        let header = makeHeader(encrypted: false, flags: [], type: .handshakeMsg1,
                                payloadLength: UInt16(payload.count))
        let packet = NMPPacket(header: header, payload: payload)
        let wire = try NMPPacketCodec.encodePlaintextPacket(packet)
        let decoded = try NMPPacketCodec.decodePlaintextPacket(wire)
        XCTAssertEqual(decoded, packet)
    }

    func testBigEndianEncoding() {
        // Spot-check wire bytes: seq 0x01020304 must land big-endian at offset 4.
        let header = makeHeader(seq: 0x01020304)
        let encoded = NMPPacketCodec.encodeHeader(header)
        XCTAssertEqual([UInt8](encoded[4..<8]), [0x01, 0x02, 0x03, 0x04])
    }

    func testSliceOffsetsHandled() throws {
        // Decoding must work on Data slices with non-zero startIndex.
        let header = makeHeader()
        let padded = Data([0xAA, 0xBB]) + NMPPacketCodec.encodeHeader(header)
        let slice = padded.dropFirst(2)
        XCTAssertEqual(try NMPPacketCodec.decodeHeader(Data(slice)), header)
    }

    // MARK: Malformed input

    func testTruncatedHeaderThrows() {
        for n in 0..<NMPHeader.byteCount {
            XCTAssertThrowsError(try NMPPacketCodec.decodeHeader(Data(repeating: 0, count: n)))
        }
    }

    func testUnknownPacketTypeThrows() {
        var bytes = NMPPacketCodec.encodeHeader(makeHeader())
        bytes[1] = 0x7B // not a defined type
        XCTAssertThrowsError(try NMPPacketCodec.decodeHeader(bytes)) {
            XCTAssertEqual($0 as? NMPCodecError, .unknownPacketType(0x7B))
        }
    }

    func testReservedBitThrows() {
        var bytes = NMPPacketCodec.encodeHeader(makeHeader())
        bytes[0] |= 1 << 5 // R bit
        XCTAssertThrowsError(try NMPPacketCodec.decodeHeader(bytes)) {
            XCTAssertEqual($0 as? NMPCodecError, .reservedBitSet)
        }
    }

    func testReservedFlagBitsThrow() {
        var bytes = NMPPacketCodec.encodeHeader(makeHeader())
        bytes[0] |= 1 << 3 // reserved flag bit
        XCTAssertThrowsError(try NMPPacketCodec.decodeHeader(bytes)) {
            XCTAssertEqual($0 as? NMPCodecError, .reservedFlagBitsSet)
        }
    }

    func testUnsupportedVersionThrows() {
        var bytes = NMPPacketCodec.encodeHeader(makeHeader())
        bytes[0] |= 1 << 7 // V=1
        XCTAssertThrowsError(try NMPPacketCodec.decodeHeader(bytes)) {
            XCTAssertEqual($0 as? NMPCodecError, .unsupportedVersion(1))
        }
    }

    func testEncryptionBitMismatchThrows() {
        // DATA packet with T=0 must be rejected.
        var bytes = NMPPacketCodec.encodeHeader(makeHeader(encrypted: true, type: .data))
        bytes[0] &= ~(UInt8(1) << 6) // clear T
        XCTAssertThrowsError(try NMPPacketCodec.decodeHeader(bytes))

        // Handshake packet with T=1 must be rejected.
        var hs = NMPPacketCodec.encodeHeader(
            makeHeader(encrypted: false, flags: [], type: .handshakeMsg1))
        hs[0] |= UInt8(1) << 6 // set T
        XCTAssertThrowsError(try NMPPacketCodec.decodeHeader(hs))
    }

    func testPayloadLengthMismatchThrows() throws {
        let payload = Data(repeating: 0x42, count: 64)
        let header = makeHeader(encrypted: false, flags: [], type: .handshakeMsg1,
                                payloadLength: UInt16(payload.count))
        var wire = try NMPPacketCodec.encodePlaintextPacket(
            NMPPacket(header: header, payload: payload))
        wire.removeLast(10) // truncate the body
        XCTAssertThrowsError(try NMPPacketCodec.decodePlaintextPacket(wire)) {
            XCTAssertEqual($0 as? NMPCodecError,
                           .payloadLengthMismatch(declared: 64, available: 54))
        }
    }

    // MARK: Fuzz — decoder must never crash (Phase 1 success criterion)

    func testRandomBytesNeverCrashDecoder() {
        var rng = SystemRandomNumberGenerator()
        for _ in 0..<2000 {
            let len = Int.random(in: 0...256, using: &rng)
            let junk = Data((0..<len).map { _ in UInt8.random(in: .min ... .max, using: &rng) })
            _ = try? NMPPacketCodec.decodeHeader(junk)
            _ = try? NMPPacketCodec.decodePlaintextPacket(junk)
        }
        // Reaching here without a crash is the assertion.
    }
}
