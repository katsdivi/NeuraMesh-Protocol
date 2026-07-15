//
//  ShardMessages.swift
//  NMP — Phase 5
//
//  Wire formats for shard orchestration and pipelined inference. All
//  multi-byte fields big-endian, consistent with every NMP format.
//
//  Two carriage paths:
//
//  - SHARD_ASSIGN (packet type 0x14): the coordinator's layer-range
//    assignment. Its own packet type — it changes peer state.
//
//  - DATA packets with a 1-byte envelope kind: inference request/response
//    metadata, tensor chunks, shard acks, metrics. These are application
//    payloads riding the normal Phase 1-3 path (encrypted, sequenced,
//    FEC-protected, NACK-repaired).
//
//  CHUNKING: an activation tensor (hiddenSize × 4 bytes; 16 KB for a 7B
//  model's 4096-wide activations) exceeds a Wi-Fi MTU. IP fragmentation
//  under loss multiplies the effective loss rate (any lost fragment kills
//  the whole datagram, and one datagram would carry the whole tensor), so
//  tensors are chunked at the APPLICATION layer into ≤1024-byte pieces:
//  each chunk is its own NMP packet, individually FEC-grouped and
//  NACK-repairable. A lost chunk costs one chunk's recovery, not the
//  tensor.
//

import Foundation

// MARK: - Envelope kinds (first byte of DATA payloads in the mesh)

public enum NMPMeshMessageKind: UInt8, Sendable {
    case inferRequestMeta  = 0x01
    case tensorChunk       = 0x02
    case inferResponseMeta = 0x03
    case metrics           = 0x04
    case shardAck          = 0x05
    case resourceReport    = 0x06
    case ping              = 0x07
    case pong              = 0x08
}

public enum NMPShardCodecError: Error, Equatable, Sendable {
    case truncated(expectedAtLeast: Int, got: Int)
    case unsupportedVersion(UInt8)
    case unknownKind(UInt8)
    case wrongKind(expected: NMPMeshMessageKind, got: NMPMeshMessageKind)
    case invalidString
    case chunkMismatch(String)
    case tensorTooLarge(bytes: Int)
}

// MARK: - Float tensor <-> bytes

public enum NMPTensorCodec {
    /// Big-endian IEEE-754 bit patterns; exact round trip.
    public static func encode(_ floats: [Float]) -> Data {
        var out = Data(capacity: floats.count * 4)
        for value in floats { out.appendBigEndian(value.bitPattern) }
        return out
    }

    public static func decode(_ data: Data) throws -> [Float] {
        let bytes = Data(data)
        guard bytes.count % 4 == 0 else {
            throw NMPShardCodecError.truncated(
                expectedAtLeast: (bytes.count / 4 + 1) * 4, got: bytes.count)
        }
        var floats = [Float]()
        floats.reserveCapacity(bytes.count / 4)
        for i in stride(from: 0, to: bytes.count, by: 4) {
            floats.append(Float(bitPattern: bytes.readBigEndianUInt32(at: i)))
        }
        return floats
    }
}

// MARK: - SHARD_ASSIGN (packet type 0x14)

public struct NMPShardAssign: Equatable, Sendable {
    public static let version: UInt8 = 1

    /// Position of this shard in the pipeline (0-based).
    public var shardIndex: UInt16
    /// Total shards in the pipeline.
    public var pipelineLength: UInt16
    public var startLayer: UInt16
    /// Exclusive.
    public var endLayer: UInt16
    public var totalLayers: UInt16
    /// Activation width — peers validate against their loaded model.
    public var hiddenSize: UInt32
    /// Model identity ("llama-7B-q4_K_M", GGUF general.name, or a hash) so
    /// a peer with the wrong model refuses instead of computing garbage.
    public var modelTag: String
    /// Future Plan #3 (weight vault): "host:port" of the coordinator's HTTP
    /// slice server, so a peer holding no local model can stream ONLY its
    /// assigned layers. Empty ⇒ no vault (the peer must already hold the model).
    /// Encoded as a backward-compatible trailing field (older peers ignore it).
    public var vaultEndpoint: String

    public init(shardIndex: UInt16, pipelineLength: UInt16, startLayer: UInt16,
                endLayer: UInt16, totalLayers: UInt16, hiddenSize: UInt32,
                modelTag: String, vaultEndpoint: String = "") {
        self.shardIndex = shardIndex
        self.pipelineLength = pipelineLength
        self.startLayer = startLayer
        self.endLayer = endLayer
        self.totalLayers = totalLayers
        self.hiddenSize = hiddenSize
        self.modelTag = modelTag
        self.vaultEndpoint = vaultEndpoint
    }

    public func encode() -> Data {
        var out = Data()
        out.append(Self.version)
        out.appendBigEndian(shardIndex)
        out.appendBigEndian(pipelineLength)
        out.appendBigEndian(startLayer)
        out.appendBigEndian(endLayer)
        out.appendBigEndian(totalLayers)
        out.appendBigEndian(hiddenSize)
        let tag = Data(modelTag.utf8.prefix(255))
        out.append(UInt8(tag.count))
        out.append(tag)
        // Backward-compatible trailing field: vault endpoint. Older decoders
        // stop after the tag and ignore these bytes.
        let vault = Data(vaultEndpoint.utf8.prefix(255))
        out.append(UInt8(vault.count))
        out.append(vault)
        return out
    }

    public static func decode(_ data: Data) throws -> NMPShardAssign {
        let bytes = Data(data)
        guard bytes.count >= 16 else {
            throw NMPShardCodecError.truncated(expectedAtLeast: 16, got: bytes.count)
        }
        guard bytes[0] == version else {
            throw NMPShardCodecError.unsupportedVersion(bytes[0])
        }
        let tagLength = Int(bytes[15])
        guard bytes.count >= 16 + tagLength else {
            throw NMPShardCodecError.truncated(expectedAtLeast: 16 + tagLength, got: bytes.count)
        }
        guard let tag = String(data: bytes.subdata(in: 16..<16 + tagLength),
                               encoding: .utf8) else {
            throw NMPShardCodecError.invalidString
        }
        // Optional backward-compatible trailing field: vault endpoint.
        var vaultEndpoint = ""
        let vaultLenIndex = 16 + tagLength
        if bytes.count > vaultLenIndex {
            let vaultLength = Int(bytes[vaultLenIndex])
            if bytes.count >= vaultLenIndex + 1 + vaultLength, vaultLength > 0 {
                vaultEndpoint = String(
                    data: bytes.subdata(in: (vaultLenIndex + 1)..<(vaultLenIndex + 1 + vaultLength)),
                    encoding: .utf8) ?? ""
            }
        }
        return NMPShardAssign(
            shardIndex: bytes.readBigEndianUInt16(at: 1),
            pipelineLength: bytes.readBigEndianUInt16(at: 3),
            startLayer: bytes.readBigEndianUInt16(at: 5),
            endLayer: bytes.readBigEndianUInt16(at: 7),
            totalLayers: bytes.readBigEndianUInt16(at: 9),
            hiddenSize: bytes.readBigEndianUInt32(at: 11),
            modelTag: tag, vaultEndpoint: vaultEndpoint)
    }
}

// MARK: - Inference request/response metadata

/// kind(u8)=0x01 ‖ requestID(u32) ‖ startLayer(u16) ‖ endLayer(u16) ‖
/// totalBytes(u32) ‖ chunkCount(u16)
public struct NMPInferRequestMeta: Equatable, Sendable {
    public var requestID: UInt32
    public var startLayer: UInt16
    public var endLayer: UInt16
    public var totalBytes: UInt32
    public var chunkCount: UInt16

    public init(requestID: UInt32, startLayer: UInt16, endLayer: UInt16,
                totalBytes: UInt32, chunkCount: UInt16) {
        self.requestID = requestID
        self.startLayer = startLayer
        self.endLayer = endLayer
        self.totalBytes = totalBytes
        self.chunkCount = chunkCount
    }

    public func encode() -> Data {
        var out = Data([NMPMeshMessageKind.inferRequestMeta.rawValue])
        out.appendBigEndian(requestID)
        out.appendBigEndian(startLayer)
        out.appendBigEndian(endLayer)
        out.appendBigEndian(totalBytes)
        out.appendBigEndian(chunkCount)
        return out
    }

    public static func decode(_ data: Data) throws -> NMPInferRequestMeta {
        let bytes = try requireKind(.inferRequestMeta, data, minLength: 15)
        return NMPInferRequestMeta(
            requestID: bytes.readBigEndianUInt32(at: 1),
            startLayer: bytes.readBigEndianUInt16(at: 5),
            endLayer: bytes.readBigEndianUInt16(at: 7),
            totalBytes: bytes.readBigEndianUInt32(at: 9),
            chunkCount: bytes.readBigEndianUInt16(at: 13))
    }
}

/// kind(u8)=0x03 ‖ requestID(u32) ‖ status(u8) ‖ computeMicros(u32) ‖
/// totalBytes(u32) ‖ chunkCount(u16)
public struct NMPInferResponseMeta: Equatable, Sendable {
    public enum Status: UInt8, Sendable {
        case ok = 0
        case notAssigned = 1
        case computeFailed = 2
        case badRequest = 3
    }

    public var requestID: UInt32
    public var status: Status
    /// Peer-side pure compute time (excludes network), microseconds.
    public var computeMicros: UInt32
    public var totalBytes: UInt32
    public var chunkCount: UInt16

    public init(requestID: UInt32, status: Status, computeMicros: UInt32,
                totalBytes: UInt32, chunkCount: UInt16) {
        self.requestID = requestID
        self.status = status
        self.computeMicros = computeMicros
        self.totalBytes = totalBytes
        self.chunkCount = chunkCount
    }

    public func encode() -> Data {
        var out = Data([NMPMeshMessageKind.inferResponseMeta.rawValue])
        out.appendBigEndian(requestID)
        out.append(status.rawValue)
        out.appendBigEndian(computeMicros)
        out.appendBigEndian(totalBytes)
        out.appendBigEndian(chunkCount)
        return out
    }

    public static func decode(_ data: Data) throws -> NMPInferResponseMeta {
        let bytes = try requireKind(.inferResponseMeta, data, minLength: 16)
        guard let status = Status(rawValue: bytes[5]) else {
            throw NMPShardCodecError.chunkMismatch("unknown response status \(bytes[5])")
        }
        return NMPInferResponseMeta(
            requestID: bytes.readBigEndianUInt32(at: 1),
            status: status,
            computeMicros: bytes.readBigEndianUInt32(at: 6),
            totalBytes: bytes.readBigEndianUInt32(at: 10),
            chunkCount: bytes.readBigEndianUInt16(at: 14))
    }
}

/// kind(u8)=0x02 ‖ requestID(u32) ‖ chunkIndex(u16) ‖ payload bytes
public struct NMPTensorChunk: Equatable, Sendable {
    /// Data bytes per chunk. 1024 + envelope(7) + NMP header/tag(36) keeps
    /// each datagram comfortably inside a 1500-byte MTU — no IP
    /// fragmentation, so Phase 2/3 loss recovery operates per-chunk.
    /// This is the radio-path size; wired/loopback paths use
    /// `PeerConnection.recommendedChunkBytes` (kernel datagram ceiling)
    /// minus `envelopeBytes`.
    public static let defaultChunkBytes = 1024
    /// Wire envelope: kind(1) + requestID(4) + chunkIndex(2).
    public static let envelopeBytes = 7

    public var requestID: UInt32
    public var chunkIndex: UInt16
    public var payload: Data

    public init(requestID: UInt32, chunkIndex: UInt16, payload: Data) {
        self.requestID = requestID
        self.chunkIndex = chunkIndex
        self.payload = payload
    }

    public func encode() -> Data {
        var out = Data([NMPMeshMessageKind.tensorChunk.rawValue])
        out.appendBigEndian(requestID)
        out.appendBigEndian(chunkIndex)
        out.append(payload)
        return out
    }

    public static func decode(_ data: Data) throws -> NMPTensorChunk {
        let bytes = try requireKind(.tensorChunk, data, minLength: 7)
        return NMPTensorChunk(
            requestID: bytes.readBigEndianUInt32(at: 1),
            chunkIndex: bytes.readBigEndianUInt16(at: 5),
            payload: bytes.subdata(in: 7..<bytes.count))
    }

    /// Splits tensor bytes into send-ready chunks.
    public static func split(requestID: UInt32, tensorBytes: Data,
                             chunkBytes: Int = defaultChunkBytes) throws -> [NMPTensorChunk] {
        let size = max(1, chunkBytes)
        let count = tensorBytes.isEmpty ? 0 : (tensorBytes.count + size - 1) / size
        guard count <= Int(UInt16.max) else {
            throw NMPShardCodecError.tensorTooLarge(bytes: tensorBytes.count)
        }
        let rebased = Data(tensorBytes)
        return (0..<count).map { index in
            let lower = index * size
            let upper = Swift.min(lower + size, rebased.count)
            return NMPTensorChunk(requestID: requestID, chunkIndex: UInt16(index),
                                  payload: rebased.subdata(in: lower..<upper))
        }
    }
}

/// kind(u8)=0x05 ‖ shardIndex(u16) ‖ status(u8)
public struct NMPShardAck: Equatable, Sendable {
    public enum Status: UInt8, Sendable {
        case ready = 0
        case rejectedModelMismatch = 1
        case rejectedBadRange = 2
    }

    public var shardIndex: UInt16
    public var status: Status

    public init(shardIndex: UInt16, status: Status) {
        self.shardIndex = shardIndex
        self.status = status
    }

    public func encode() -> Data {
        var out = Data([NMPMeshMessageKind.shardAck.rawValue])
        out.appendBigEndian(shardIndex)
        out.append(status.rawValue)
        return out
    }

    public static func decode(_ data: Data) throws -> NMPShardAck {
        let bytes = try requireKind(.shardAck, data, minLength: 4)
        guard let status = Status(rawValue: bytes[3]) else {
            throw NMPShardCodecError.chunkMismatch("unknown ack status \(bytes[3])")
        }
        return NMPShardAck(shardIndex: bytes.readBigEndianUInt16(at: 1), status: status)
    }
}

/// kind(u8)=0x04 ‖ peerID(u32) ‖ latencyMicros(u32) ‖ memoryMB(u32) ‖ load(u8)
public struct NMPPeerMetrics: Equatable, Sendable {
    public var peerID: UInt32
    public var inferenceLatencyMicros: UInt32
    public var memoryUsageMB: UInt32
    public var currentLoadPercent: UInt8

    public init(peerID: UInt32, inferenceLatencyMicros: UInt32,
                memoryUsageMB: UInt32, currentLoadPercent: UInt8) {
        self.peerID = peerID
        self.inferenceLatencyMicros = inferenceLatencyMicros
        self.memoryUsageMB = memoryUsageMB
        self.currentLoadPercent = currentLoadPercent
    }

    public func encode() -> Data {
        var out = Data([NMPMeshMessageKind.metrics.rawValue])
        out.appendBigEndian(peerID)
        out.appendBigEndian(inferenceLatencyMicros)
        out.appendBigEndian(memoryUsageMB)
        out.append(currentLoadPercent)
        return out
    }

    public static func decode(_ data: Data) throws -> NMPPeerMetrics {
        let bytes = try requireKind(.metrics, data, minLength: 14)
        return NMPPeerMetrics(
            peerID: bytes.readBigEndianUInt32(at: 1),
            inferenceLatencyMicros: bytes.readBigEndianUInt32(at: 5),
            memoryUsageMB: bytes.readBigEndianUInt32(at: 9),
            currentLoadPercent: bytes[13])
    }
}

/// Mesh 2.3: a peer's own host resource sample, sent over the mesh so the
/// coordinator's Devices panel can show REAL per-device numbers. In-process
/// peers report the same host the coordinator sees (they share it — the
/// hostname match is how the UI knows to say so); a physical peer (second
/// Mac, iPhone app) reports its own kernel counters, which is the honest
/// per-device data this message exists for.
///
/// kind(u8)=0x06 ‖ version(u8) ‖ peerID(u32) ‖ ramTotalMB(u32) ‖
/// ramUsedMB(u32) ‖ processFootprintMB(u32) ‖ storageTotalMB(u32) ‖
/// storageFreeMB(u32) ‖ cpuPercentX10(u16) ‖ gpuPercentX10(u16) ‖
/// hostnameLen(u8) ‖ hostname(utf8)
///
/// cpu/gpuPercentX10 use 0xFFFF as "unavailable" (first CPU sample has no
/// tick delta to diff; GPU counters only exist on macOS).
public struct NMPPeerResourceReport: Equatable, Sendable {
    public static let version: UInt8 = 1
    private static let unavailable: UInt16 = 0xFFFF

    public var peerID: UInt32
    public var ramTotalMB: UInt32
    public var ramUsedMB: UInt32
    public var processFootprintMB: UInt32
    public var storageTotalMB: UInt32
    public var storageFreeMB: UInt32
    /// 0...100, nil = unavailable.
    public var cpuPercent: Double?
    /// 0...100, nil = unavailable (non-macOS, or no GPU counters).
    public var gpuPercent: Double?
    public var hostname: String

    public init(peerID: UInt32, ramTotalMB: UInt32, ramUsedMB: UInt32,
                processFootprintMB: UInt32, storageTotalMB: UInt32,
                storageFreeMB: UInt32, cpuPercent: Double?,
                gpuPercent: Double?, hostname: String) {
        self.peerID = peerID
        self.ramTotalMB = ramTotalMB
        self.ramUsedMB = ramUsedMB
        self.processFootprintMB = processFootprintMB
        self.storageTotalMB = storageTotalMB
        self.storageFreeMB = storageFreeMB
        self.cpuPercent = cpuPercent
        self.gpuPercent = gpuPercent
        self.hostname = hostname
    }

    public init(peerID: UInt32, sample: NMPHostResourceSample) {
        self.init(
            peerID: peerID,
            ramTotalMB: UInt32(clamping: sample.ramTotalBytes / (1 << 20)),
            ramUsedMB: UInt32(clamping: sample.ramUsedBytes / (1 << 20)),
            processFootprintMB: UInt32(clamping: sample.processFootprintBytes / (1 << 20)),
            storageTotalMB: UInt32(clamping: sample.storageTotalBytes / (1 << 20)),
            storageFreeMB: UInt32(clamping: sample.storageFreeBytes / (1 << 20)),
            cpuPercent: sample.cpuPercent,
            gpuPercent: sample.gpuPercent,
            hostname: sample.hostname)
    }

    private static func packPercent(_ value: Double?) -> UInt16 {
        guard let value else { return unavailable }
        let scaled = UInt16(clamping: Int((value * 10).rounded()))
        return scaled == unavailable ? unavailable - 1 : scaled
    }

    private static func unpackPercent(_ raw: UInt16) -> Double? {
        raw == unavailable ? nil : Double(raw) / 10
    }

    public func encode() -> Data {
        var out = Data([NMPMeshMessageKind.resourceReport.rawValue, Self.version])
        out.appendBigEndian(peerID)
        out.appendBigEndian(ramTotalMB)
        out.appendBigEndian(ramUsedMB)
        out.appendBigEndian(processFootprintMB)
        out.appendBigEndian(storageTotalMB)
        out.appendBigEndian(storageFreeMB)
        out.appendBigEndian(Self.packPercent(cpuPercent))
        out.appendBigEndian(Self.packPercent(gpuPercent))
        let name = Data(hostname.utf8.prefix(255))
        out.append(UInt8(name.count))
        out.append(name)
        return out
    }

    public static func decode(_ data: Data) throws -> NMPPeerResourceReport {
        let bytes = try requireKind(.resourceReport, data, minLength: 31)
        guard bytes[1] == version else {
            throw NMPShardCodecError.unsupportedVersion(bytes[1])
        }
        let nameLength = Int(bytes[30])
        guard bytes.count >= 31 + nameLength else {
            throw NMPShardCodecError.truncated(expectedAtLeast: 31 + nameLength,
                                               got: bytes.count)
        }
        guard let hostname = String(data: bytes.subdata(in: 31..<31 + nameLength),
                                    encoding: .utf8) else {
            throw NMPShardCodecError.invalidString
        }
        return NMPPeerResourceReport(
            peerID: bytes.readBigEndianUInt32(at: 2),
            ramTotalMB: bytes.readBigEndianUInt32(at: 6),
            ramUsedMB: bytes.readBigEndianUInt32(at: 10),
            processFootprintMB: bytes.readBigEndianUInt32(at: 14),
            storageTotalMB: bytes.readBigEndianUInt32(at: 18),
            storageFreeMB: bytes.readBigEndianUInt32(at: 22),
            cpuPercent: unpackPercent(bytes.readBigEndianUInt16(at: 26)),
            gpuPercent: unpackPercent(bytes.readBigEndianUInt16(at: 28)),
            hostname: hostname)
    }
}

/// Mesh 2.4: coordinator → peer keepalive. The activity-based health
/// monitor reads silence as death, but a pipeline stalled on ONE slow
/// stage (a backgrounded iPhone, a Wi-Fi hiccup) leaves every OTHER peer
/// silent too — pinging idle peers lets the alive ones answer instead of
/// being dropped for someone else's stall. The pong echoes the nonce;
/// receiving ANY authenticated packet already counts as activity, so the
/// pong needs no handling beyond arriving. Peers older than Mesh 2.4
/// ignore pings (unknown-kind diagnostic) and keep working — they just
/// stay droppable when idle, exactly as before.
///
/// kind(u8)=0x07/0x08 ‖ nonce(u64)
public struct NMPPeerPing: Equatable, Sendable {
    public var nonce: UInt64

    public init(nonce: UInt64 = UInt64.random(in: UInt64.min...UInt64.max)) {
        self.nonce = nonce
    }

    public func encode() -> Data {
        var out = Data([NMPMeshMessageKind.ping.rawValue])
        out.appendBigEndian(nonce)
        return out
    }

    public static func decode(_ data: Data) throws -> NMPPeerPing {
        let bytes = try requireKind(.ping, data, minLength: 9)
        return NMPPeerPing(nonce: bytes.readBigEndianUInt64(at: 1))
    }

    /// The echo a peer sends back.
    public func pongPayload() -> Data {
        var out = Data([NMPMeshMessageKind.pong.rawValue])
        out.appendBigEndian(nonce)
        return out
    }
}

// MARK: - Kind dispatch helper

public func NMPMeshMessageKindOf(_ data: Data) -> NMPMeshMessageKind? {
    guard let first = data.first else { return nil }
    return NMPMeshMessageKind(rawValue: first)
}

private func requireKind(_ kind: NMPMeshMessageKind, _ data: Data,
                         minLength: Int) throws -> Data {
    let bytes = Data(data)
    guard bytes.count >= minLength else {
        throw NMPShardCodecError.truncated(expectedAtLeast: minLength, got: bytes.count)
    }
    guard let got = NMPMeshMessageKind(rawValue: bytes[0]) else {
        throw NMPShardCodecError.unknownKind(bytes[0])
    }
    guard got == kind else {
        throw NMPShardCodecError.wrongKind(expected: kind, got: got)
    }
    return bytes
}

// MARK: - Tensor reassembly

/// Collects a request's metadata + chunks (any arrival order, duplicates
/// tolerated) and yields the complete tensor bytes once. One instance per
/// direction per connection.
public final class NMPTensorReassembler {

    private struct Pending {
        var expectedChunks: Int?
        var expectedBytes: Int?
        var chunks: [UInt16: Data] = [:]
        var receivedBytes = 0
    }

    /// Per-request budget guard: a hostile/buggy sender cannot balloon
    /// memory (default 64 MB ≈ a 16M-element f32 tensor).
    public var maxTensorBytes = 64 << 20

    private var pending: [UInt32: Pending] = [:]

    public init() {}

    /// Registers expectations from a request/response meta message.
    /// Returns the completed tensor if all chunks already arrived (meta
    /// can arrive after chunks under reordering).
    public func setExpectation(requestID: UInt32, chunkCount: Int,
                               totalBytes: Int) throws -> Data? {
        guard totalBytes <= maxTensorBytes else {
            throw NMPShardCodecError.tensorTooLarge(bytes: totalBytes)
        }
        var entry = pending[requestID] ?? Pending()
        entry.expectedChunks = chunkCount
        entry.expectedBytes = totalBytes
        pending[requestID] = entry
        return try completeIfReady(requestID)
    }

    /// Buffers one chunk. Returns the completed tensor when the last
    /// missing chunk lands, else nil.
    public func addChunk(_ chunk: NMPTensorChunk) throws -> Data? {
        var entry = pending[chunk.requestID] ?? Pending()
        if entry.chunks[chunk.chunkIndex] == nil {
            entry.receivedBytes += chunk.payload.count
            guard entry.receivedBytes <= maxTensorBytes else {
                pending.removeValue(forKey: chunk.requestID)
                throw NMPShardCodecError.tensorTooLarge(bytes: entry.receivedBytes)
            }
            entry.chunks[chunk.chunkIndex] = chunk.payload
        }
        pending[chunk.requestID] = entry
        return try completeIfReady(chunk.requestID)
    }

    /// Drops partial state for a request (timeout/failure path).
    public func abandon(requestID: UInt32) {
        pending.removeValue(forKey: requestID)
    }

    /// Drops partial state for every request older than `requestID` —
    /// valid because requests are strictly serial: seeing a newer one
    /// means the sender abandoned all older ones (Phase 6 stage retry).
    public func abandonOlder(than requestID: UInt32) {
        pending = pending.filter { $0.key >= requestID }
    }

    private func completeIfReady(_ requestID: UInt32) throws -> Data? {
        guard let entry = pending[requestID],
              let expectedChunks = entry.expectedChunks,
              entry.chunks.count == expectedChunks else { return nil }
        defer { pending.removeValue(forKey: requestID) }

        var assembled = Data(capacity: entry.expectedBytes ?? entry.receivedBytes)
        for index in 0..<expectedChunks {
            guard let piece = entry.chunks[UInt16(index)] else {
                throw NMPShardCodecError.chunkMismatch("hole at chunk \(index)")
            }
            assembled.append(piece)
        }
        if let expectedBytes = entry.expectedBytes, assembled.count != expectedBytes {
            throw NMPShardCodecError.chunkMismatch(
                "reassembled \(assembled.count)B, meta declared \(expectedBytes)B")
        }
        return assembled
    }
}
