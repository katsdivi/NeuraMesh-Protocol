//
//  NMPMemoryWire.swift
//  NMP — Memory mesh (hackathon build, NEW code)
//
//  Wire formats for the distributed-memory layer. Memory messages are
//  application payloads riding ordinary DATA packets over the EXISTING
//  Phase 1-3 transport (encrypted, sequenced, FEC-protected, NACK-repaired)
//  via PeerConnection.send / sendBurst — no new networking underneath.
//
//  Kind namespace: memory-mesh DATA payloads start at 0x20, well clear of
//  the compute mesh's NMPMeshMessageKind (0x01-0x08). A memory peer process
//  never attaches a PeerShardEngine, so the namespaces cannot collide in
//  one dispatcher; the disjoint ranges are for log readability and safety.
//
//  Framing, two layers (all multi-byte fields big-endian, house rule):
//
//  1. CHUNK (the only DATA payload shape this module sends):
//
//         kind        (u8)   0x20 MEM_CHUNK
//         transfer_id (u32)  per-link monotonically increasing
//         chunk_index (u16)
//         chunk_count (u16)  1...  (single-chunk messages use count = 1)
//         bytes       (...)  fragment of the assembled MESSAGE
//
//     A shard record easily exceeds one radio-safe packet payload, so
//     assembled messages are split at the APPLICATION layer (same rationale
//     as tensor chunking in ShardMessages.swift) and shipped with sendBurst.
//
//  2. MESSAGE (after reassembly):
//
//         inner_kind  (u8)   see NMPMemoryMessageKind
//         header_len  (u32)
//         header      (JSON, header_len bytes)  small metadata dictionary
//         body        (rest)  opaque binary (an encoded NMPMemoryShardRecord)
//
//     JSON for the metadata dictionary is a deliberate app-layer choice
//     (mirrors the dashboard's HTTP API); the hot path — shard bytes —
//     stays binary in `body`.
//

import Foundation

// MARK: - Kinds

/// First byte of every memory-mesh DATA payload.
public enum NMPMemoryWireKind: UInt8, Sendable {
    case chunk = 0x20
}

/// Inner message kinds (first byte of an assembled message).
public enum NMPMemoryMessageKind: UInt8, Sendable {
    /// Write path: "store this shard + replicated index entry". Body = record.
    case storeShard  = 0x01
    /// Receiver's verdict on storeShard. Header only.
    case storeAck    = 0x02
    /// Read path: "send me your shard of memoryID". Header only.
    case fetchShard  = 0x03
    /// Answer to fetchShard. Body = record when found.
    case fetchResult = 0x04
    /// Link liveness (kill/restart detection between fetches). Header only.
    case ping        = 0x05
    case pong        = 0x06
}

public enum NMPMemoryWireError: Error, Equatable, Sendable {
    case truncated
    case unknownKind(UInt8)
    case malformedHeader(String)
    case chunkMismatch(String)
}

// MARK: - Message

public struct NMPMemoryMessage {
    public let kind: NMPMemoryMessageKind
    /// Small JSON metadata dictionary (string keys; values JSON scalars/arrays).
    public let header: [String: Any]
    /// Opaque binary payload (an encoded shard record) — empty when unused.
    public let body: Data

    public init(kind: NMPMemoryMessageKind, header: [String: Any], body: Data = Data()) {
        self.kind = kind
        self.header = header
        self.body = body
    }

    public func encode() throws -> Data {
        let headerData = try JSONSerialization.data(withJSONObject: header)
        var out = Data(capacity: 5 + headerData.count + body.count)
        out.append(kind.rawValue)
        out.appendBigEndian(UInt32(headerData.count))
        out.append(headerData)
        out.append(body)
        return out
    }

    public static func decode(_ data: Data) throws -> NMPMemoryMessage {
        let bytes = Data(data) // rebase slice offsets
        guard bytes.count >= 5 else { throw NMPMemoryWireError.truncated }
        guard let kind = NMPMemoryMessageKind(rawValue: bytes[0]) else {
            throw NMPMemoryWireError.unknownKind(bytes[0])
        }
        let headerLen = Int(bytes.readBigEndianUInt32(at: 1))
        guard bytes.count >= 5 + headerLen else { throw NMPMemoryWireError.truncated }
        let headerData = bytes.subdata(in: 5..<(5 + headerLen))
        guard let header = try? JSONSerialization.jsonObject(with: headerData)
                as? [String: Any] else {
            throw NMPMemoryWireError.malformedHeader("header is not a JSON object")
        }
        let body = bytes.subdata(in: (5 + headerLen)..<bytes.count)
        return NMPMemoryMessage(kind: kind, header: header, body: body)
    }
}

// MARK: - Chunking

public enum NMPMemoryChunker {

    static let chunkHeaderBytes = 9 // kind(1) + transfer(4) + index(2) + count(2)

    /// Splits an encoded message into DATA payloads of at most `chunkBytes`
    /// of content each. `chunkBytes` should come from
    /// `PeerConnection.recommendedChunkBytes` minus nothing — the 9-byte
    /// chunk header is accounted for here.
    public static func split(message: Data, transferID: UInt32,
                             chunkBytes: Int) -> [Data] {
        let capacity = max(64, chunkBytes - chunkHeaderBytes)
        let count = max(1, (message.count + capacity - 1) / capacity)
        var payloads: [Data] = []
        payloads.reserveCapacity(count)
        for index in 0..<count {
            let start = index * capacity
            let end = Swift.min(start + capacity, message.count)
            var payload = Data(capacity: chunkHeaderBytes + (end - start))
            payload.append(NMPMemoryWireKind.chunk.rawValue)
            payload.appendBigEndian(transferID)
            payload.appendBigEndian(UInt16(index))
            payload.appendBigEndian(UInt16(count))
            if start < end { payload.append(message.subdata(in: start..<end)) }
            payloads.append(payload)
        }
        return payloads
    }
}

/// Per-link reassembler. NOT thread-safe — owned by one connection queue,
/// like everything else on a link.
public final class NMPMemoryReassembler {

    private struct Partial {
        var chunks: [Int: Data] = [:]
        var count: Int
        var firstSeen: TimeInterval
    }

    private var partials: [UInt32: Partial] = [:]
    private let staleAfter: TimeInterval

    public init(staleAfter: TimeInterval = 30) {
        self.staleAfter = staleAfter
    }

    /// Feeds one DATA payload. Returns the assembled message when this chunk
    /// completes a transfer, nil while more chunks are pending. Non-memory
    /// payloads (wrong kind byte) and malformed chunks throw.
    public func absorb(_ payload: Data,
                       now: TimeInterval = ProcessInfo.processInfo.systemUptime)
    throws -> NMPMemoryMessage? {
        let bytes = Data(payload)
        guard bytes.count >= NMPMemoryChunker.chunkHeaderBytes else {
            throw NMPMemoryWireError.truncated
        }
        guard bytes[0] == NMPMemoryWireKind.chunk.rawValue else {
            throw NMPMemoryWireError.unknownKind(bytes[0])
        }
        let transferID = bytes.readBigEndianUInt32(at: 1)
        let index = Int(bytes.readBigEndianUInt16(at: 5))
        let count = Int(bytes.readBigEndianUInt16(at: 7))
        guard count >= 1, index < count else {
            throw NMPMemoryWireError.chunkMismatch(
                "chunk \(index)/\(count) of transfer \(transferID)")
        }
        let content = bytes.subdata(in: NMPMemoryChunker.chunkHeaderBytes..<bytes.count)

        sweepStale(now: now)

        var partial = partials[transferID]
            ?? Partial(count: count, firstSeen: now)
        guard partial.count == count else {
            partials[transferID] = nil
            throw NMPMemoryWireError.chunkMismatch(
                "transfer \(transferID): count changed \(partial.count) → \(count)")
        }
        partial.chunks[index] = content
        guard partial.chunks.count == count else {
            partials[transferID] = partial
            return nil
        }
        partials[transferID] = nil
        var assembled = Data()
        for i in 0..<count {
            guard let piece = partial.chunks[i] else {
                throw NMPMemoryWireError.chunkMismatch(
                    "transfer \(transferID): missing chunk \(i)")
            }
            assembled.append(piece)
        }
        return try NMPMemoryMessage.decode(assembled)
    }

    /// Drops transfers that never completed (their sender died mid-burst).
    /// The transport's FEC/NACK make in-session loss effectively zero, so a
    /// stale partial means the LINK died — safe to forget.
    private func sweepStale(now: TimeInterval) {
        partials = partials.filter { now - $0.value.firstSeen < staleAfter }
    }
}
