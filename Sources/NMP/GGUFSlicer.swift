//
//  GGUFSlicer.swift
//  NMP — Future Plan #3: the weight vault
//
//  Produces a SMALLER, still-valid GGUF that contains only the tensors one
//  shard needs, so a device stores ≈ only the bytes for its layers (not the
//  whole model). Today the shard shim `fread`s only its layers into RAM but
//  the offsets point into the full file, so every host must keep the whole
//  file on disk. A slice removes that: disk ≈ RAM.
//
//  The slice is engineered to load in the EXISTING shim with NO C changes:
//   • block tensors keep their GLOBAL names (blk.20.*, not renumbered) —
//     the shim's compute loop indexes blocks globally (`for l=start..<end`).
//   • `<arch>.block_count` is kept at the FULL N — the shim's want()/clamp and
//     `first`/`last` shard detection read it.
//   • only the WANTED tensors are included, mirroring the shim's want():
//       blk.[start,end).*  +  token_embd.weight (iff start==0)
//                          +  output_norm.weight, output.weight (iff end==N)
//   • the big tokenizer arrays are dropped (the shard never tokenizes) — the
//     coordinator keeps the full model for that.
//
//  GGUF is LITTLE-endian (see GGUF.swift). This writer mirrors that parser's
//  byte layout exactly; the full-range round-trip test + the shim's own gguf
//  loader validate it before any device ever sees a slice.
//

import Foundation

public enum NMPGGUFSlicerError: Error, Equatable, Sendable {
    case notLoadable(String)
    case missingBlockCount
    case emptyArrayType(String)
}

public enum NMPGGUFSlicer {

    /// Tokenizer arrays the shard never reads — dropped by default so slices
    /// don't carry the (multi-MB) vocab. Pass `dropKeys: []` for an exact
    /// metadata round-trip.
    public static let defaultDropKeys: Set<String> = [
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.merges",
    ]

    /// Slice `modelPath`'s layers [start, end) into a valid GGUF at `outputPath`.
    /// The source is memory-mapped, so only the selected tensor bytes are read.
    @discardableResult
    public static func slice(modelPath: String, start: Int, end: Int,
                             to outputPath: String,
                             dropKeys: Set<String> = defaultDropKeys) throws -> Int {
        let data = try sliceData(modelPath: modelPath, start: start, end: end,
                                 dropKeys: dropKeys)
        try data.write(to: URL(fileURLWithPath:
            (outputPath as NSString).expandingTildeInPath))
        return data.count
    }

    /// Build the slice in memory (used to stream a shard over HTTP without a
    /// temp file). The source file is mmapped; the returned Data is ≈ the
    /// shard's own weight bytes plus a small header.
    public static func sliceData(modelPath: String, start: Int, end: Int,
                                 dropKeys: Set<String> = defaultDropKeys) throws -> Data {
        let expanded = (modelPath as NSString).expandingTildeInPath
        let source = try Data(contentsOf: URL(fileURLWithPath: expanded),
                              options: .mappedIfSafe)
        let model = try NMPGGUFModel.parse(source)
        return try build(model: model, source: source, start: start, end: end,
                         dropKeys: dropKeys)
    }

    // MARK: - Core

    static func build(model: NMPGGUFModel, source: Data, start: Int, end: Int,
                      dropKeys: Set<String>) throws -> Data {
        guard let arch = model.architecture else {
            throw NMPGGUFSlicerError.notLoadable("general.architecture")
        }
        guard let n = model.layerCount else {
            throw NMPGGUFSlicerError.missingBlockCount
        }
        let clampedEnd = (end < 0 || end > n) ? n : end

        // Which tensors this shard needs — mirrors the shim's want().
        func wanted(_ name: String) -> Bool {
            if name == "token_embd.weight" { return start == 0 }
            if name == "output_norm.weight" { return clampedEnd == n }
            if name == "output.weight" { return clampedEnd == n }
            if let l = blockIndex(name) { return l >= start && l < clampedEnd }
            // Non-block, non-output tensors (e.g. a rare global norm) ride with
            // the first shard so nothing the graph might touch goes missing.
            return start == 0
        }
        let selected = model.tensors.filter { wanted($0.name) }

        // Exact byte size of each selected tensor (consecutive-offset deltas,
        // same rule as NMPGGUFModel.tensorByteSizes).
        let sizes = tensorByteSizes(model.tensors, fileBytes: source.count,
                                    dataOffset: Int(model.tensorDataOffset))

        // KVs to keep: everything except the dropped tokenizer arrays. Order is
        // irrelevant (GGUF looks up by name); block_count/alignment unchanged.
        let keptKV = model.metadata.filter { !dropKeys.contains($0.key) }

        let align = Int(max(model.alignment, 1))
        var out = Data()

        // --- header ---
        out.appU32(NMPGGUFModel.magic)
        out.appU32(model.version)
        out.appU64(UInt64(selected.count))
        out.appU64(UInt64(keptKV.count))

        // --- metadata KV ---
        for (key, value) in keptKV {
            out.appString(key)
            try out.appValue(value, keyForError: key)
        }

        // --- tensor info (new offsets, packed & aligned) ---
        var packedOffset = 0
        var newOffsets: [Int] = []
        newOffsets.reserveCapacity(selected.count)
        for tensor in selected {
            packedOffset = roundUp(packedOffset, align)
            newOffsets.append(packedOffset)
            packedOffset += sizes[tensor.name] ?? 0
        }
        for (i, tensor) in selected.enumerated() {
            out.appString(tensor.name)
            out.appU32(UInt32(tensor.dimensions.count))
            for d in tensor.dimensions { out.appU64(d) }
            out.appU32(tensor.ggmlTypeID)
            out.appU64(UInt64(newOffsets[i]))
        }

        // --- align to the tensor-data section ---
        padTo(&out, multiple: align)
        let dataSectionStart = out.count

        // --- tensor data (copy each selected tensor's bytes) ---
        for (i, tensor) in selected.enumerated() {
            // Pad to this tensor's aligned position within the data section.
            let target = dataSectionStart + newOffsets[i]
            if out.count < target { out.append(Data(count: target - out.count)) }
            let size = sizes[tensor.name] ?? 0
            let srcStart = Int(model.tensorDataOffset) + Int(tensor.offset)
            if size > 0 {
                out.append(source.subdata(in: srcStart ..< srcStart + size))
            }
        }
        return out
    }

    // MARK: - Helpers

    /// Global block index parsed from a `blk.<L>.*` name, else nil.
    static func blockIndex(_ name: String) -> Int? {
        guard name.hasPrefix("blk.") else { return nil }
        let rest = name.dropFirst(4)
        guard let dot = rest.firstIndex(of: ".") else { return nil }
        return Int(rest[..<dot])
    }

    static func tensorByteSizes(_ tensors: [NMPGGUFTensorInfo],
                                fileBytes: Int, dataOffset: Int) -> [String: Int] {
        let dataBytes = max(0, fileBytes - dataOffset)
        let sorted = tensors.sorted { $0.offset < $1.offset }
        var sizes: [String: Int] = [:]
        for (i, tensor) in sorted.enumerated() {
            let end = (i + 1 < sorted.count) ? Int(sorted[i + 1].offset) : dataBytes
            sizes[tensor.name] = max(0, end - Int(tensor.offset))
        }
        return sizes
    }

    static func roundUp(_ value: Int, _ multiple: Int) -> Int {
        multiple <= 1 ? value : (value + multiple - 1) / multiple * multiple
    }

    static func padTo(_ data: inout Data, multiple: Int) {
        let target = roundUp(data.count, multiple)
        if data.count < target { data.append(Data(count: target - data.count)) }
    }
}

// MARK: - Little-endian GGUF writer (mirrors GGUF.swift's cursor)

private extension Data {
    mutating func appU32(_ v: UInt32) {
        var le = v.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }
    mutating func appU64(_ v: UInt64) {
        var le = v.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }
    mutating func appString(_ s: String) {
        let bytes = Array(s.utf8)
        appU64(UInt64(bytes.count))
        append(contentsOf: bytes)
    }

    /// GGUF value type id (spec): 0 u8,1 i8,2 u16,3 i16,4 u32,5 i32,6 f32,
    /// 7 bool,8 string,9 array,10 u64,11 i64,12 f64.
    func typeID(of v: NMPGGUFValue) -> UInt32 {
        switch v {
        case .uint8: return 0
        case .int8: return 1
        case .uint16: return 2
        case .int16: return 3
        case .uint32: return 4
        case .int32: return 5
        case .float32: return 6
        case .bool: return 7
        case .string: return 8
        case .array: return 9
        case .uint64: return 10
        case .int64: return 11
        case .float64: return 12
        }
    }

    /// Append a KV value: type id (u32) then payload.
    mutating func appValue(_ v: NMPGGUFValue, keyForError: String) throws {
        appU32(typeID(of: v))
        try appPayload(v, keyForError: keyForError)
    }

    /// Append only the value payload (no leading type id) — used for both KV
    /// values and array elements, exactly as the parser consumes them.
    mutating func appPayload(_ v: NMPGGUFValue, keyForError: String) throws {
        switch v {
        case .uint8(let x): append(x)
        case .int8(let x): append(UInt8(bitPattern: x))
        case .uint16(let x):
            var le = x.littleEndian; Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
        case .int16(let x):
            var le = UInt16(bitPattern: x).littleEndian
            Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
        case .uint32(let x): appU32(x)
        case .int32(let x): appU32(UInt32(bitPattern: x))
        case .float32(let x): appU32(x.bitPattern)
        case .bool(let x): append(x ? 1 : 0)
        case .string(let s): appString(s)
        case .uint64(let x): appU64(x)
        case .int64(let x): appU64(UInt64(bitPattern: x))
        case .float64(let x): appU64(x.bitPattern)
        case .array(let elems):
            // Homogeneous per the spec; element type from the first element.
            // The parser drops an empty array's element-type id, so it's
            // unrecoverable — default to int32 (count 0, so no payload follows;
            // it re-parses as an empty array regardless). The shard never reads
            // array-valued metadata, so this is invisible to compute.
            appU32(elems.first.map { typeID(of: $0) } ?? 5)
            appU64(UInt64(elems.count))
            for e in elems { try appPayload(e, keyForError: keyForError) }
        }
    }
}
