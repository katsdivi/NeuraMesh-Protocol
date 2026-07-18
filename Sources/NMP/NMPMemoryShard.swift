//
//  NMPMemoryShard.swift
//  NMP — Memory mesh
//
//  K-of-N erasure coding + sealing for conversational memory blobs.
//
//  Generalizes the fixed 4+1 packet FEC grouping (FECCodec.swift/FECGroup.swift)
//  to a configurable K over arbitrary byte blobs: a blob is split into K equal
//  blocks (the last zero-padded) plus one XOR parity block, yielding N = K+1
//  self-describing shard records. ANY K of the N shards reconstruct the blob;
//  fewer than K fails loudly (never silent wrong output).
//
//  This module REUSES the pure static XOR primitives `NMPFECCodec.computeParity`
//  and `NMPFECCodec.reconstruct`. Those are stateless free functions, so calling
//  them here cannot affect the live packet-loss machinery in FECGroup.swift.
//
//  LIMITATION: XOR parity supports exactly ONE parity shard, so N must equal
//  K+1 — the scheme tolerates any SINGLE shard loss. Losing two or more shards
//  is unrecoverable and reported as `.insufficientShards`. True arbitrary-N
//  redundancy would need Reed-Solomon coding, which is out of scope here.
//
//  Shard record wire format (all multi-byte fields big-endian):
//
//      version     (u8)          = 1
//      id_len      (u8)          memoryID UTF-8 byte count (<= 255)
//      memory_id   (id_len B)    caller-chosen identifier, UTF-8
//      shard_index (u8)          0..<k = data shards, k = the parity shard
//      k           (u8)          shards required to reconstruct
//      n           (u8)          total shards produced (= k + 1)
//      blob_length (u32)         original blob byte count, pre-padding
//      payload_len (u32)         this shard's byte count = ceil(blob_length/k)
//      payload     (payload_len B)
//
//  Every record carries the full scheme so shards are individually
//  self-describing at rest and over the wire; decode rejects any record whose
//  byte count is not exactly header + payload_len (no trailing garbage).
//
//  Sealing (NMPMemorySeal) runs BEFORE sharding: LZFSE-compress, then
//  AES-256-GCM encrypt under a fresh random 256-bit key. Shards of ciphertext
//  are individually opaque, and GCM authentication makes reconstruction
//  tamper-evident: a wrong key or corrupted shard fails `open` loudly instead
//  of returning garbage. The compressed body is prefixed with a 1-byte flag
//  (0x01 = LZFSE, 0x00 = stored raw) because libcompression rejects empty
//  input; empty plaintext round-trips via the raw path.
//

import Foundation
import CryptoKit

// MARK: - Errors

public enum NMPMemoryShardError: Error, Equatable, Sendable {
    case invalidScheme(k: Int, n: Int)           // n != k+1, k < 1, or n > 16
    case insufficientShards(have: Int, needed: Int)
    case shardMismatch(String)                   // mixed memoryIDs/schemes/lengths, duplicate index, index out of range
    case truncatedRecord
    case unsupportedVersion(UInt8)
    case sealFailed(String)
    case openFailed(String)                      // wrong key, tampered ciphertext (GCM auth), corrupt LZFSE
}

// MARK: - Scheme

/// K-of-N scheme. XOR parity ⇒ n == k + 1 (tolerates any single shard loss).
public struct NMPMemoryShardScheme: Equatable, Sendable {
    public let k: Int   // shards required to reconstruct (data shard count)
    public let n: Int   // total shards produced = k + 1

    /// Throws `.invalidScheme` unless n == k+1, 1 <= k, and n <= 16 (the same
    /// group-size cap as the packet FEC wire format).
    public init(k: Int, n: Int) throws {
        guard k >= 1, n == k + 1, n <= NMPFECCodec.maxGroupSize else {
            throw NMPMemoryShardError.invalidScheme(k: k, n: n)
        }
        self.k = k
        self.n = n
    }
}

// MARK: - Shard record

/// One erasure-coded shard of one memory blob, self-describing on the wire.
public struct NMPMemoryShardRecord: Equatable, Sendable {
    public static let version: UInt8 = 1

    public let memoryID: String     // caller-chosen id, <= 255 UTF-8 bytes
    public let shardIndex: Int      // 0..<k = data shards, k = the parity shard
    public let k: Int
    public let n: Int
    public let blobLength: Int      // original blob byte count (pre-padding)
    public let payload: Data        // this shard's bytes; all n shards equal length = ceil(blobLength/k)

    public init(memoryID: String, shardIndex: Int, k: Int, n: Int,
                blobLength: Int, payload: Data) {
        precondition(memoryID.utf8.count <= 255, "memoryID must be <= 255 UTF-8 bytes")
        self.memoryID = memoryID
        self.shardIndex = shardIndex
        self.k = k
        self.n = n
        self.blobLength = blobLength
        self.payload = payload
    }

    /// Expected per-shard payload length for a given blob under a k-way split.
    static func blockLength(blobLength: Int, k: Int) -> Int {
        guard blobLength > 0, k > 0 else { return 0 }
        return (blobLength + k - 1) / k
    }

    // MARK: Wire format

    /// Versioned, big-endian; layout documented in the file header.
    public func encode() -> Data {
        let idBytes = Data(memoryID.utf8)
        var out = Data(capacity: 13 + idBytes.count + payload.count)
        out.append(Self.version)
        out.append(UInt8(idBytes.count))
        out.append(idBytes)
        out.append(UInt8(shardIndex))
        out.append(UInt8(k))
        out.append(UInt8(n))
        out.appendBigEndian(UInt32(blobLength))
        out.appendBigEndian(UInt32(payload.count))
        out.append(payload)
        return out
    }

    public static func decode(_ data: Data) throws -> NMPMemoryShardRecord {
        let bytes = Data(data) // rebase slice offsets
        guard bytes.count >= 2 else { throw NMPMemoryShardError.truncatedRecord }
        guard bytes[0] == version else {
            throw NMPMemoryShardError.unsupportedVersion(bytes[0])
        }
        let idLen = Int(bytes[1])
        let headerLen = 13 + idLen
        guard bytes.count >= headerLen else { throw NMPMemoryShardError.truncatedRecord }

        guard let memoryID = String(data: bytes.subdata(in: 2..<(2 + idLen)),
                                    encoding: .utf8) else {
            throw NMPMemoryShardError.shardMismatch("memoryID is not valid UTF-8")
        }
        let shardIndex = Int(bytes[2 + idLen])
        let k = Int(bytes[3 + idLen])
        let n = Int(bytes[4 + idLen])
        let blobLength = Int(bytes.readBigEndianUInt32(at: 5 + idLen))
        let payloadLen = Int(bytes.readBigEndianUInt32(at: 9 + idLen))
        guard bytes.count == headerLen + payloadLen else {
            throw NMPMemoryShardError.truncatedRecord
        }

        _ = try NMPMemoryShardScheme(k: k, n: n) // rejects malformed schemes
        guard (0...k).contains(shardIndex) else {
            throw NMPMemoryShardError.shardMismatch(
                "shardIndex \(shardIndex) out of range 0...\(k)")
        }
        guard payloadLen == blockLength(blobLength: blobLength, k: k) else {
            throw NMPMemoryShardError.shardMismatch(
                "payload length \(payloadLen) != ceil(\(blobLength)/\(k))")
        }

        return NMPMemoryShardRecord(
            memoryID: memoryID, shardIndex: shardIndex, k: k, n: n,
            blobLength: blobLength,
            payload: bytes.subdata(in: headerLen..<bytes.count))
    }
}

// MARK: - Shard codec

public enum NMPMemoryShardCodec {

    /// Splits blob into k equal blocks (last zero-padded) + 1 XOR parity block.
    /// Returns exactly scheme.n records, shardIndex 0...k. Empty blob allowed.
    public static func encode(memoryID: String, blob: Data,
                              scheme: NMPMemoryShardScheme) -> [NMPMemoryShardRecord] {
        let bytes = Data(blob) // rebase slice offsets
        let k = scheme.k
        let blockLen = NMPMemoryShardRecord.blockLength(blobLength: bytes.count, k: k)

        var blocks: [Data] = []
        blocks.reserveCapacity(k)
        for i in 0..<k {
            let start = Swift.min(i * blockLen, bytes.count)
            let end = Swift.min(start + blockLen, bytes.count)
            var block = bytes.subdata(in: start..<end)
            if block.count < blockLen {
                block.append(Data(repeating: 0, count: blockLen - block.count))
            }
            blocks.append(block)
        }
        let parity = NMPFECCodec.computeParity(blocks)

        return (0...k).map { index in
            NMPMemoryShardRecord(
                memoryID: memoryID, shardIndex: index, k: k, n: scheme.n,
                blobLength: bytes.count,
                payload: index < k ? blocks[index] : parity)
        }
    }

    /// Reconstructs the original blob from ANY k of the n records (order
    /// irrelevant, duplicates by index rejected). Missing one data shard ⇒
    /// recovered via parity XOR. Fewer than k distinct shards ⇒
    /// .insufficientShards — explicit failure, never silent wrong output.
    public static func reconstruct(records: [NMPMemoryShardRecord]) throws -> Data {
        guard let first = records.first else {
            throw NMPMemoryShardError.insufficientShards(have: 0, needed: 1)
        }
        let scheme = try NMPMemoryShardScheme(k: first.k, n: first.n)
        let k = scheme.k
        let blockLen = NMPMemoryShardRecord.blockLength(blobLength: first.blobLength, k: k)

        var blocks = [Data?](repeating: nil, count: k)
        var parity: Data?
        for record in records {
            guard record.memoryID == first.memoryID else {
                throw NMPMemoryShardError.shardMismatch(
                    "mixed memoryIDs: \(record.memoryID) vs \(first.memoryID)")
            }
            guard record.k == first.k, record.n == first.n,
                  record.blobLength == first.blobLength else {
                throw NMPMemoryShardError.shardMismatch(
                    "mixed schemes for memoryID \(first.memoryID)")
            }
            guard record.payload.count == blockLen else {
                throw NMPMemoryShardError.shardMismatch(
                    "payload length \(record.payload.count) != expected \(blockLen)")
            }
            guard (0...k).contains(record.shardIndex) else {
                throw NMPMemoryShardError.shardMismatch(
                    "shardIndex \(record.shardIndex) out of range 0...\(k)")
            }
            if record.shardIndex == k {
                guard parity == nil else {
                    throw NMPMemoryShardError.shardMismatch(
                        "duplicate shardIndex \(record.shardIndex)")
                }
                parity = record.payload
            } else {
                guard blocks[record.shardIndex] == nil else {
                    throw NMPMemoryShardError.shardMismatch(
                        "duplicate shardIndex \(record.shardIndex)")
                }
                blocks[record.shardIndex] = record.payload
            }
        }

        let have = blocks.compactMap { $0 }.count + (parity != nil ? 1 : 0)
        guard have >= k else {
            throw NMPMemoryShardError.insufficientShards(have: have, needed: k)
        }

        // With >= k distinct shards of k+1, at most one data block is missing.
        if let missing = blocks.firstIndex(where: { $0 == nil }) {
            let surviving = blocks.compactMap { $0 }
            do {
                blocks[missing] = try NMPFECCodec.reconstruct(
                    parity: parity!, surviving: surviving, missingLength: blockLen)
            } catch {
                throw NMPMemoryShardError.shardMismatch("parity XOR failed: \(error)")
            }
        }

        var blob = Data(capacity: k * blockLen)
        for block in blocks { blob.append(block!) }
        return Data(blob.prefix(first.blobLength))
    }
}

// MARK: - Sealing

/// Pre-shard sealing: LZFSE-compress then AES-256-GCM encrypt with a fresh
/// random 256-bit key. Shards of ciphertext are individually opaque at rest,
/// and GCM authentication makes reconstruction tamper-evident: a wrong or
/// corrupted shard fails open() loudly instead of returning garbage.
public enum NMPMemorySeal {

    private static let flagRaw: UInt8 = 0x00     // empty plaintext, stored as-is
    private static let flagLZFSE: UInt8 = 0x01   // LZFSE-compressed body

    public struct Sealed: Sendable {
        public let ciphertext: Data   // AES.GCM combined representation (nonce+ct+tag)
        public let key: Data          // 32 bytes; caller distributes/stores it
    }

    public static func seal(plaintext: Data) throws -> Sealed {
        var body = Data()
        if plaintext.isEmpty {
            // libcompression rejects empty input — store the empty body raw.
            body.append(flagRaw)
        } else {
            guard let packed = try? (plaintext as NSData).compressed(using: .lzfse) else {
                throw NMPMemoryShardError.sealFailed("LZFSE compression failed")
            }
            body.append(flagLZFSE)
            body.append(packed as Data)
        }

        let key = SymmetricKey(size: .bits256)
        do {
            let box = try AES.GCM.seal(body, using: key)
            guard let combined = box.combined else {
                throw NMPMemoryShardError.sealFailed("no combined GCM representation")
            }
            return Sealed(ciphertext: combined,
                          key: key.withUnsafeBytes { Data($0) })
        } catch let error as NMPMemoryShardError {
            throw error
        } catch {
            throw NMPMemoryShardError.sealFailed("AES-GCM seal failed: \(error)")
        }
    }

    public static func open(ciphertext: Data, key: Data) throws -> Data {
        guard key.count == 32 else {
            throw NMPMemoryShardError.openFailed("key must be 32 bytes, got \(key.count)")
        }
        let body: Data
        do {
            let box = try AES.GCM.SealedBox(combined: ciphertext)
            body = try AES.GCM.open(box, using: SymmetricKey(data: key))
        } catch {
            throw NMPMemoryShardError.openFailed("AES-GCM open failed (wrong key or tampered ciphertext)")
        }

        guard let flag = body.first else {
            throw NMPMemoryShardError.openFailed("empty sealed body")
        }
        let payload = body.dropFirst()
        switch flag {
        case flagRaw:
            return Data(payload)
        case flagLZFSE:
            guard let inflated = try? (Data(payload) as NSData).decompressed(using: .lzfse) else {
                throw NMPMemoryShardError.openFailed("corrupt LZFSE body")
            }
            return inflated as Data
        default:
            throw NMPMemoryShardError.openFailed("unknown body flag 0x\(String(flag, radix: 16))")
        }
    }
}
