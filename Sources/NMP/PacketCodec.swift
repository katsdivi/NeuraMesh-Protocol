//
//  PacketCodec.swift
//  NMP — Phase 1
//
//  Encode/decode NMP packet headers per NMP_Specification.md §2.
//  All multi-byte fields are big-endian.
//
//  Wire layout (20-byte header):
//
//    byte 0      : V(1) | T(1) | R(1) | FLAGS(5)
//    byte 1      : PACKET_TYPE
//    bytes 2-3   : PAYLOAD_LENGTH (UInt16)
//    bytes 4-7   : SEQUENCE_NUMBER (UInt32)
//    bytes 8-11  : SENDER_PEER_ID (UInt32)
//    bytes 12-19 : TIMESTAMP (UInt64, nanoseconds since epoch)
//    bytes 20-.. : PAYLOAD (PAYLOAD_LENGTH bytes)
//    trailing    : GCM_TAG (16 bytes) — present only when T=1 (encrypted)
//

import Foundation

// MARK: - Packet types

public enum NMPPacketType: UInt8, CaseIterable, Sendable {
    case handshakeMsg1 = 0x00
    case handshakeMsg2 = 0x01
    case data          = 0x10
    case nack          = 0x11
    case fecRecovery   = 0x12
    case capabilityAdv = 0x13
    case shardAssign   = 0x14
    case ackRange      = 0x15
    case control       = 0xFF

    /// Handshake packets are the only unencrypted (T=0) types.
    public var isHandshake: Bool {
        self == .handshakeMsg1 || self == .handshakeMsg2
    }
}

// MARK: - Flags

public struct NMPFlags: OptionSet, Sendable {
    public let rawValue: UInt8
    public init(rawValue: UInt8) { self.rawValue = rawValue & 0x1F }

    /// Bit 0: last packet in an FEC group.
    public static let fecGroupEnd = NMPFlags(rawValue: 1 << 0)
    /// Bit 1: retransmit of a previous packet (diagnostics).
    public static let retransmit  = NMPFlags(rawValue: 1 << 1)
    /// Bit 2: receiver should process immediately, no buffering.
    public static let flush       = NMPFlags(rawValue: 1 << 2)
    // Bits 3-4 reserved, must be 0.

    static let reservedMask: UInt8 = 0b11000
    public var hasReservedBits: Bool { rawValue & Self.reservedMask != 0 }
}

// MARK: - Header

public struct NMPHeader: Equatable, Sendable {
    public static let byteCount = 20
    public static let gcmTagByteCount = 16
    public static let maxPayloadLength = Int(UInt16.max)
    /// Protocol version this implementation speaks. V is a single bit; 0 for now.
    public static let currentVersion: UInt8 = 0

    public var version: UInt8          // 1 bit on the wire; must be 0
    public var isEncrypted: Bool       // T bit: false = handshake, true = data/control
    public var flags: NMPFlags
    public var packetType: NMPPacketType
    public var payloadLength: UInt16
    public var sequenceNumber: UInt32
    public var senderPeerID: UInt32
    public var timestampNanos: UInt64

    public init(
        isEncrypted: Bool,
        flags: NMPFlags = [],
        packetType: NMPPacketType,
        payloadLength: UInt16,
        sequenceNumber: UInt32,
        senderPeerID: UInt32,
        timestampNanos: UInt64
    ) {
        self.version = Self.currentVersion
        self.isEncrypted = isEncrypted
        self.flags = flags
        self.packetType = packetType
        self.payloadLength = payloadLength
        self.sequenceNumber = sequenceNumber
        self.senderPeerID = senderPeerID
        self.timestampNanos = timestampNanos
    }
}

// MARK: - Packet

public struct NMPPacket: Equatable, Sendable {
    public var header: NMPHeader
    /// Decrypted (or plaintext) payload. For encrypted packets this is set
    /// after `NMPSecureSession.open` succeeds.
    public var payload: Data

    public init(header: NMPHeader, payload: Data) {
        self.header = header
        self.payload = payload
    }
}

// MARK: - Errors

public enum NMPCodecError: Error, Equatable, Sendable {
    case truncated(expectedAtLeast: Int, got: Int)
    case unsupportedVersion(UInt8)
    case reservedBitSet
    case reservedFlagBitsSet
    case unknownPacketType(UInt8)
    case payloadLengthMismatch(declared: Int, available: Int)
    case payloadTooLarge(Int)
    case encryptionBitMismatch(type: NMPPacketType, tBit: Bool)
}

// MARK: - Codec

public enum NMPPacketCodec {

    // MARK: Encode

    /// Encodes only the 20-byte header. Used both for wire output and as
    /// AAD input for AES-GCM (spec §3: AAD = everything before PAYLOAD).
    public static func encodeHeader(_ header: NMPHeader) -> Data {
        var out = Data(capacity: NMPHeader.byteCount)
        var byte0: UInt8 = 0
        byte0 |= (header.version & 0x1) << 7
        byte0 |= (header.isEncrypted ? 1 : 0) << 6
        // R bit (bit 5) always 0.
        byte0 |= header.flags.rawValue & 0x1F
        out.append(byte0)
        out.append(header.packetType.rawValue)
        out.appendBigEndian(header.payloadLength)
        out.appendBigEndian(header.sequenceNumber)
        out.appendBigEndian(header.senderPeerID)
        out.appendBigEndian(header.timestampNanos)
        return out
    }

    /// Encodes a full *unencrypted* packet (handshake traffic, T=0).
    /// Encrypted packets are assembled by `NMPSecureSession.seal`, which
    /// appends ciphertext + GCM tag after this header.
    public static func encodePlaintextPacket(_ packet: NMPPacket) throws -> Data {
        guard packet.payload.count <= NMPHeader.maxPayloadLength else {
            throw NMPCodecError.payloadTooLarge(packet.payload.count)
        }
        var header = packet.header
        header.payloadLength = UInt16(packet.payload.count)
        var out = encodeHeader(header)
        out.append(packet.payload)
        return out
    }

    // MARK: Decode

    /// Decodes and validates the 20-byte header from raw datagram bytes.
    public static func decodeHeader(_ data: Data) throws -> NMPHeader {
        guard data.count >= NMPHeader.byteCount else {
            throw NMPCodecError.truncated(expectedAtLeast: NMPHeader.byteCount, got: data.count)
        }
        // Work on a rebased copy so indices start at 0 regardless of the
        // incoming Data's slice offsets.
        let bytes = Data(data.prefix(NMPHeader.byteCount))

        let byte0 = bytes[0]
        let version = (byte0 >> 7) & 0x1
        guard version == NMPHeader.currentVersion else {
            throw NMPCodecError.unsupportedVersion(version)
        }
        let tBit = (byte0 >> 6) & 0x1 == 1
        let rBit = (byte0 >> 5) & 0x1
        guard rBit == 0 else { throw NMPCodecError.reservedBitSet }

        let flags = NMPFlags(rawValue: byte0 & 0x1F)
        // OptionSet init masks to 5 bits; validate reserved flag bits 3-4.
        guard byte0 & 0x1F & NMPFlags.reservedMask == 0 else {
            throw NMPCodecError.reservedFlagBitsSet
        }

        guard let type = NMPPacketType(rawValue: bytes[1]) else {
            throw NMPCodecError.unknownPacketType(bytes[1])
        }
        // T bit must be consistent with the packet type (spec §2).
        let expectedEncrypted = !type.isHandshake
        guard tBit == expectedEncrypted else {
            throw NMPCodecError.encryptionBitMismatch(type: type, tBit: tBit)
        }

        let payloadLength = bytes.readBigEndianUInt16(at: 2)
        let sequence = bytes.readBigEndianUInt32(at: 4)
        let peerID = bytes.readBigEndianUInt32(at: 8)
        let timestamp = bytes.readBigEndianUInt64(at: 12)

        var header = NMPHeader(
            isEncrypted: tBit,
            flags: flags,
            packetType: type,
            payloadLength: payloadLength,
            sequenceNumber: sequence,
            senderPeerID: peerID,
            timestampNanos: timestamp
        )
        header.version = version
        return header
    }

    /// Decodes a full *unencrypted* packet (T=0). Validates that the datagram
    /// contains exactly the declared payload.
    public static func decodePlaintextPacket(_ data: Data) throws -> NMPPacket {
        let header = try decodeHeader(data)
        guard !header.isEncrypted else {
            // Encrypted packets must go through NMPSecureSession.open.
            throw NMPCodecError.encryptionBitMismatch(type: header.packetType, tBit: true)
        }
        let body = Data(data.dropFirst(NMPHeader.byteCount))
        guard body.count == Int(header.payloadLength) else {
            throw NMPCodecError.payloadLengthMismatch(
                declared: Int(header.payloadLength), available: body.count)
        }
        return NMPPacket(header: header, payload: body)
    }
}

// MARK: - Big-endian helpers

extension Data {
    mutating func appendBigEndian(_ value: UInt16) {
        Swift.withUnsafeBytes(of: value.bigEndian) { append(contentsOf: $0) }
    }
    mutating func appendBigEndian(_ value: UInt32) {
        Swift.withUnsafeBytes(of: value.bigEndian) { append(contentsOf: $0) }
    }
    mutating func appendBigEndian(_ value: UInt64) {
        Swift.withUnsafeBytes(of: value.bigEndian) { append(contentsOf: $0) }
    }

    /// Readers assume `self` is rebased (startIndex == 0) and bounds-checked
    /// by the caller.
    func readBigEndianUInt16(at offset: Int) -> UInt16 {
        precondition(count >= offset + 2)
        return (UInt16(self[offset]) << 8) | UInt16(self[offset + 1])
    }
    func readBigEndianUInt32(at offset: Int) -> UInt32 {
        precondition(count >= offset + 4)
        var v: UInt32 = 0
        for i in 0..<4 { v = (v << 8) | UInt32(self[offset + i]) }
        return v
    }
    func readBigEndianUInt64(at offset: Int) -> UInt64 {
        precondition(count >= offset + 8)
        var v: UInt64 = 0
        for i in 0..<8 { v = (v << 8) | UInt64(self[offset + i]) }
        return v
    }
}
