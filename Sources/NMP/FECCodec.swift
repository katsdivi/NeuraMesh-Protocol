//
//  FECCodec.swift
//  NMP — Phase 3
//
//  XOR forward error correction primitives (spec roadmap item 3).
//
//  A group of N data packets (default 4) is protected by one parity packet
//  whose payload is the XOR of the group's payloads, zero-padded to the
//  longest member. Any single missing member can be reconstructed by XORing
//  the parity with the surviving members.
//
//  Parity packet payload (type 0x12 FEC_RECOVERY, encrypted like any data):
//
//      group_id (u32)             CRC32 over the member sequence numbers
//      count    (u8, 1...16)      number of data packets in the group
//      seqs     (count × u32)     member sequence numbers, send order
//      lengths  (count × u16)     original payload length of each member
//      parity   (max(lengths) B)  XOR of zero-padded member payloads
//
//  The member sequence numbers are listed EXPLICITLY (not base+count):
//  NACK and control packets share the per-direction sequence space, so a
//  group's sequences are not guaranteed contiguous. Lengths are carried
//  because members may have unequal payload sizes; reconstruction truncates
//  the padded XOR result back to the missing member's true length. The
//  group_id doubles as an integrity check (recomputed on decode).
//
//  All fields big-endian, consistent with the packet header codec.
//

import Foundation

// MARK: - Errors

public enum NMPFECError: Error, Equatable, Sendable {
    case truncated
    case invalidGroupSize
    case groupIDMismatch
    case lengthExceedsParity
}

// MARK: - CRC32 (IEEE 802.3, reflected, poly 0xEDB88320)

enum NMPCRC32 {
    private static let table: [UInt32] = (0..<256).map { i in
        var c = UInt32(i)
        for _ in 0..<8 { c = (c & 1) != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        return c
    }

    static func checksum(_ data: Data) -> UInt32 {
        var c: UInt32 = 0xFFFF_FFFF
        for byte in data {
            c = table[Int((c ^ UInt32(byte)) & 0xFF)] ^ (c >> 8)
        }
        return c ^ 0xFFFF_FFFF
    }
}

// MARK: - FEC codec

public enum NMPFECCodec {

    /// Hard cap on group size the wire format supports. Practical groups are
    /// 2-8; see Phase3_Design.md for the N=4 default tradeoff.
    public static let maxGroupSize = 16

    // MARK: XOR primitives

    /// `accumulator ^= other` over the overlapping prefix, 8 bytes at a time.
    /// Word-wise so a 4×1400 B group parity stays well under the 100 µs
    /// success criterion even in debug builds.
    static func xorInto(_ accumulator: inout Data, _ other: Data) {
        let n = Swift.min(accumulator.count, other.count)
        guard n > 0 else { return }
        accumulator.withUnsafeMutableBytes { (acc: UnsafeMutableRawBufferPointer) in
            other.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
                let a = acc.baseAddress!
                let b = src.baseAddress!
                var i = 0
                while i + 8 <= n {
                    let word = a.loadUnaligned(fromByteOffset: i, as: UInt64.self)
                        ^ b.loadUnaligned(fromByteOffset: i, as: UInt64.self)
                    a.storeBytes(of: word, toByteOffset: i, as: UInt64.self)
                    i += 8
                }
                while i < n {
                    let byte = a.load(fromByteOffset: i, as: UInt8.self)
                        ^ b.load(fromByteOffset: i, as: UInt8.self)
                    a.storeBytes(of: byte, toByteOffset: i, as: UInt8.self)
                    i += 1
                }
            }
        }
    }

    /// XOR of all payloads, zero-padded to the longest.
    public static func computeParity(_ payloads: [Data]) -> Data {
        let maxLen = payloads.map(\.count).max() ?? 0
        var parity = Data(repeating: 0, count: maxLen)
        for payload in payloads {
            xorInto(&parity, payload)
        }
        return parity
    }

    /// Recovers the one missing member: parity ⊕ all surviving payloads,
    /// truncated to the missing member's original length.
    public static func reconstruct(
        parity: Data,
        surviving: [Data],
        missingLength: Int
    ) throws -> Data {
        guard missingLength <= parity.count else { throw NMPFECError.lengthExceedsParity }
        var result = Data(parity) // rebase + copy
        for payload in surviving {
            xorInto(&result, payload)
        }
        return result.prefix(missingLength)
    }

    // MARK: Group identity

    public static func groupID(for sequences: [UInt32]) -> UInt32 {
        var bytes = Data(capacity: sequences.count * 4)
        for seq in sequences { bytes.appendBigEndian(seq) }
        return NMPCRC32.checksum(bytes)
    }

    // MARK: Wire format

    public struct ParityDescriptor: Equatable, Sendable {
        public let groupID: UInt32
        public let sequences: [UInt32]
        public let lengths: [Int]
        public let parity: Data
    }

    public static func encodeParityPayload(
        sequences: [UInt32],
        payloadLengths: [Int],
        parity: Data
    ) -> Data {
        precondition(sequences.count == payloadLengths.count)
        precondition((1...maxGroupSize).contains(sequences.count))
        var out = Data(capacity: 5 + sequences.count * 6 + parity.count)
        out.appendBigEndian(groupID(for: sequences))
        out.append(UInt8(sequences.count))
        for seq in sequences { out.appendBigEndian(seq) }
        for len in payloadLengths { out.appendBigEndian(UInt16(clamping: len)) }
        out.append(parity)
        return out
    }

    public static func decodeParityPayload(_ payload: Data) throws -> ParityDescriptor {
        let bytes = Data(payload) // rebase slice offsets
        guard bytes.count >= 5 else { throw NMPFECError.truncated }
        let declaredID = bytes.readBigEndianUInt32(at: 0)
        let count = Int(bytes[4])
        guard (1...maxGroupSize).contains(count) else { throw NMPFECError.invalidGroupSize }
        let headerLen = 5 + count * 6
        guard bytes.count >= headerLen else { throw NMPFECError.truncated }

        var sequences: [UInt32] = []
        sequences.reserveCapacity(count)
        for i in 0..<count {
            sequences.append(bytes.readBigEndianUInt32(at: 5 + i * 4))
        }
        var lengths: [Int] = []
        lengths.reserveCapacity(count)
        for i in 0..<count {
            lengths.append(Int(bytes.readBigEndianUInt16(at: 5 + count * 4 + i * 2)))
        }
        let parity = bytes.subdata(in: headerLen..<bytes.count)

        guard declaredID == groupID(for: sequences) else { throw NMPFECError.groupIDMismatch }
        guard lengths.allSatisfy({ $0 <= parity.count }) else { throw NMPFECError.lengthExceedsParity }
        return ParityDescriptor(groupID: declaredID, sequences: sequences,
                                lengths: lengths, parity: parity)
    }
}
