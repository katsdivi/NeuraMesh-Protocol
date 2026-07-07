//
//  GGUF.swift
//  NMP — Phase 5
//
//  GGUF container parsing: header, metadata key/value store, tensor
//  directory. This is a REAL parser for the GGUF v2/v3 format (the format
//  llama.cpp writes) — point it at any .gguf file and it reports the
//  model's architecture, layer count, hidden size, head count, and the
//  full tensor table, without loading tensor data into memory.
//
//  IMPORTANT ENDIANNESS NOTE: GGUF files are LITTLE-endian (per the GGUF
//  spec) while NMP wire formats are big-endian. Do not mix the helpers.
//
//  Scope boundary (see Phase5_Design.md): this file parses the container.
//  Executing transformer layers from the quantized tensor data is the job
//  of an NMPShardComputeEngine implementation — the production path binds
//  llama.cpp behind that protocol; tests and the cross-device harness use
//  the deterministic reference engine in ComputeEngine.swift.
//

import Foundation

// MARK: - Errors

public enum NMPGGUFError: Error, Equatable, Sendable {
    case notGGUF(magic: UInt32)
    case unsupportedVersion(UInt32)
    case truncated(needed: Int, available: Int)
    case unknownValueType(UInt32)
    case invalidString
    case implausibleCount(String, UInt64)
    case missingMetadata(String)
}

// MARK: - Metadata values

/// One GGUF metadata value. Numeric widths are preserved so callers can
/// round-trip exactly; convenience accessors coerce where sensible.
public indirect enum NMPGGUFValue: Equatable, Sendable {
    case uint8(UInt8), int8(Int8)
    case uint16(UInt16), int16(Int16)
    case uint32(UInt32), int32(Int32)
    case uint64(UInt64), int64(Int64)
    case float32(Float), float64(Double)
    case bool(Bool)
    case string(String)
    case array([NMPGGUFValue])

    /// Widest-lossless integer view (nil for non-integer values).
    public var intValue: Int? {
        switch self {
        case .uint8(let v): return Int(v)
        case .int8(let v): return Int(v)
        case .uint16(let v): return Int(v)
        case .int16(let v): return Int(v)
        case .uint32(let v): return Int(v)
        case .int32(let v): return Int(v)
        case .uint64(let v): return v <= UInt64(Int.max) ? Int(v) : nil
        case .int64(let v): return Int(v)
        default: return nil
        }
    }

    public var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }
}

// MARK: - Tensor directory entry

public struct NMPGGUFTensorInfo: Equatable, Sendable {
    public let name: String
    /// Dimensions in GGUF order (ne[0] fastest-varying), 1–4 entries.
    public let dimensions: [UInt64]
    /// Raw ggml type id (0 = F32, 1 = F16, quantized types 2+; the id is
    /// preserved verbatim — interpretation belongs to the compute engine).
    public let ggmlTypeID: UInt32
    /// Byte offset of the tensor data, relative to the start of the
    /// (aligned) tensor-data section.
    public let offset: UInt64

    public var elementCount: UInt64 {
        dimensions.reduce(1) { $0 &* $1 }
    }
}

// MARK: - Model

public struct NMPGGUFModel: Sendable {
    public static let magic: UInt32 = 0x4655_4747 // "GGUF" read little-endian

    public let version: UInt32
    public let metadata: [String: NMPGGUFValue]
    public let tensors: [NMPGGUFTensorInfo]
    /// Byte offset where the aligned tensor-data section begins.
    public let tensorDataOffset: UInt64
    /// Alignment of the tensor-data section (`general.alignment`, default 32).
    public let alignment: UInt32

    // MARK: Convenience metadata accessors

    /// e.g. "llama", "qwen2", "phi3".
    public var architecture: String? {
        metadata["general.architecture"]?.stringValue
    }
    public var modelName: String? {
        metadata["general.name"]?.stringValue
    }
    /// Transformer block count — the unit ModelSharder distributes.
    public var layerCount: Int? {
        architectureScoped("block_count")?.intValue
    }
    /// Embedding width — the activation vector size peers exchange.
    public var hiddenSize: Int? {
        architectureScoped("embedding_length")?.intValue
    }
    public var attentionHeadCount: Int? {
        architectureScoped("attention.head_count")?.intValue
    }
    public var contextLength: Int? {
        architectureScoped("context_length")?.intValue
    }

    private func architectureScoped(_ suffix: String) -> NMPGGUFValue? {
        guard let arch = architecture else { return nil }
        return metadata["\(arch).\(suffix)"]
    }

    // MARK: Loading

    /// Parses the container header from a file. The file is memory-mapped,
    /// so multi-GB models cost only the header pages actually touched.
    public static func load(path: String) throws -> NMPGGUFModel {
        let data = try Data(contentsOf: URL(fileURLWithPath: path),
                            options: .mappedIfSafe)
        return try parse(data)
    }

    /// Parses a GGUF container from raw bytes (tests build tiny synthetic
    /// files in memory).
    public static func parse(_ data: Data) throws -> NMPGGUFModel {
        var cursor = GGUFCursor(Data(data))

        let magic = try cursor.readUInt32()
        guard magic == Self.magic else { throw NMPGGUFError.notGGUF(magic: magic) }
        let version = try cursor.readUInt32()
        guard version == 2 || version == 3 else {
            throw NMPGGUFError.unsupportedVersion(version)
        }
        let tensorCount = try cursor.readUInt64()
        let kvCount = try cursor.readUInt64()
        // Sanity bounds: a hostile header must not drive giant allocations.
        guard tensorCount <= 1_000_000 else {
            throw NMPGGUFError.implausibleCount("tensor_count", tensorCount)
        }
        guard kvCount <= 1_000_000 else {
            throw NMPGGUFError.implausibleCount("metadata_kv_count", kvCount)
        }

        var metadata: [String: NMPGGUFValue] = [:]
        metadata.reserveCapacity(Int(kvCount))
        for _ in 0..<kvCount {
            let key = try cursor.readString()
            let typeID = try cursor.readUInt32()
            metadata[key] = try cursor.readValue(typeID: typeID, depth: 0)
        }

        var tensors: [NMPGGUFTensorInfo] = []
        tensors.reserveCapacity(Int(tensorCount))
        for _ in 0..<tensorCount {
            let name = try cursor.readString()
            let dimCount = try cursor.readUInt32()
            guard dimCount >= 1, dimCount <= 4 else {
                throw NMPGGUFError.implausibleCount("n_dims", UInt64(dimCount))
            }
            var dims: [UInt64] = []
            for _ in 0..<dimCount { dims.append(try cursor.readUInt64()) }
            let typeID = try cursor.readUInt32()
            let offset = try cursor.readUInt64()
            tensors.append(NMPGGUFTensorInfo(
                name: name, dimensions: dims, ggmlTypeID: typeID, offset: offset))
        }

        let alignment = metadata["general.alignment"]?.intValue.map(UInt32.init) ?? 32
        let unaligned = UInt64(cursor.offset)
        let align = UInt64(max(alignment, 1))
        let dataOffset = (unaligned + align - 1) / align * align

        return NMPGGUFModel(
            version: version, metadata: metadata, tensors: tensors,
            tensorDataOffset: dataOffset, alignment: alignment)
    }
}

// MARK: - Little-endian parse cursor

private struct GGUFCursor {
    let data: Data
    private(set) var offset = 0

    init(_ data: Data) { self.data = data }

    private mutating func take(_ count: Int) throws -> Data {
        guard data.count - offset >= count else {
            throw NMPGGUFError.truncated(needed: count, available: data.count - offset)
        }
        defer { offset += count }
        return data.subdata(in: offset..<offset + count)
    }

    mutating func readUInt32() throws -> UInt32 {
        let bytes = try take(4)
        return bytes.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian
    }

    mutating func readUInt64() throws -> UInt64 {
        let bytes = try take(8)
        return bytes.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
    }

    mutating func readString() throws -> String {
        let length = try readUInt64()
        guard length <= 1 << 20 else {
            throw NMPGGUFError.implausibleCount("string_length", length)
        }
        guard let string = String(data: try take(Int(length)), encoding: .utf8) else {
            throw NMPGGUFError.invalidString
        }
        return string
    }

    /// GGUF value type ids (spec): 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32,
    /// 6 f32, 7 bool, 8 string, 9 array, 10 u64, 11 i64, 12 f64.
    mutating func readValue(typeID: UInt32, depth: Int) throws -> NMPGGUFValue {
        switch typeID {
        case 0: return .uint8(try take(1)[0])
        case 1: return .int8(Int8(bitPattern: try take(1)[0]))
        case 2:
            let raw = try take(2).withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) }
            return .uint16(raw.littleEndian)
        case 3:
            let raw = try take(2).withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) }
            return .int16(Int16(bitPattern: raw.littleEndian))
        case 4: return .uint32(try readUInt32())
        case 5: return .int32(Int32(bitPattern: try readUInt32()))
        case 6: return .float32(Float(bitPattern: try readUInt32()))
        case 7: return .bool(try take(1)[0] != 0)
        case 8: return .string(try readString())
        case 9:
            guard depth < 2 else { // spec allows nesting; cap it defensively
                throw NMPGGUFError.implausibleCount("array_depth", UInt64(depth))
            }
            let elementTypeID = try readUInt32()
            let count = try readUInt64()
            guard count <= 10_000_000 else {
                throw NMPGGUFError.implausibleCount("array_length", count)
            }
            var elements: [NMPGGUFValue] = []
            elements.reserveCapacity(Int(count))
            for _ in 0..<count {
                elements.append(try readValue(typeID: elementTypeID, depth: depth + 1))
            }
            return .array(elements)
        case 10: return .uint64(try readUInt64())
        case 11: return .int64(Int64(bitPattern: try readUInt64()))
        case 12: return .float64(Double(bitPattern: try readUInt64()))
        default:
            throw NMPGGUFError.unknownValueType(typeID)
        }
    }
}
