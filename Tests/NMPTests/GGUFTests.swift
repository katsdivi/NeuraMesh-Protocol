//
//  GGUFTests.swift
//  NMPTests — Phase 5
//
//  GGUF container parsing against synthetic files built byte-by-byte
//  (little-endian, per the GGUF spec), plus malformed-input rejection.
//

import XCTest
@testable import NMP

// MARK: - Little-endian GGUF builder

struct GGUFBuilder {
    var data = Data()

    mutating func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    mutating func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    mutating func f32(_ v: Float) { u32(v.bitPattern) }
    mutating func str(_ s: String) {
        u64(UInt64(s.utf8.count))
        data.append(contentsOf: s.utf8)
    }
    mutating func kvString(_ key: String, _ value: String) {
        str(key); u32(8); str(value)
    }
    mutating func kvU32(_ key: String, _ value: UInt32) {
        str(key); u32(4); u32(value)
    }

    static func header(tensorCount: UInt64, kvCount: UInt64, version: UInt32 = 3) -> GGUFBuilder {
        var b = GGUFBuilder()
        b.u32(NMPGGUFModel.magic)
        b.u32(version)
        b.u64(tensorCount)
        b.u64(kvCount)
        return b
    }
}

final class GGUFTests: XCTestCase {

    /// A minimal llama-architecture GGUF: 5 metadata keys, 2 tensors.
    private func makeLlamaContainer() -> Data {
        var b = GGUFBuilder.header(tensorCount: 2, kvCount: 5)
        b.kvString("general.architecture", "llama")
        b.kvString("general.name", "TestLlama 7B")
        b.kvU32("llama.block_count", 32)
        b.kvU32("llama.embedding_length", 4096)
        b.kvU32("llama.attention.head_count", 32)

        // tensor 0: token_embd.weight, 2 dims [4096, 32000], type F16(1), offset 0
        b.str("token_embd.weight")
        b.u32(2); b.u64(4096); b.u64(32000)
        b.u32(1); b.u64(0)
        // tensor 1: blk.0.attn_q.weight, 2 dims, q4_K(12), offset 0x1000
        b.str("blk.0.attn_q.weight")
        b.u32(2); b.u64(4096); b.u64(4096)
        b.u32(12); b.u64(0x1000)
        return b.data
    }

    func testParsesLlamaMetadataAndTensors() throws {
        let model = try NMPGGUFModel.parse(makeLlamaContainer())

        XCTAssertEqual(model.version, 3)
        XCTAssertEqual(model.architecture, "llama")
        XCTAssertEqual(model.modelName, "TestLlama 7B")
        XCTAssertEqual(model.layerCount, 32)
        XCTAssertEqual(model.hiddenSize, 4096)
        XCTAssertEqual(model.attentionHeadCount, 32)

        XCTAssertEqual(model.tensors.count, 2)
        XCTAssertEqual(model.tensors[0].name, "token_embd.weight")
        XCTAssertEqual(model.tensors[0].dimensions, [4096, 32000])
        XCTAssertEqual(model.tensors[0].elementCount, 4096 * 32000)
        XCTAssertEqual(model.tensors[1].ggmlTypeID, 12)
        XCTAssertEqual(model.tensors[1].offset, 0x1000)

        // Data section aligned to the default 32.
        XCTAssertEqual(model.alignment, 32)
        XCTAssertEqual(model.tensorDataOffset % 32, 0)
    }

    func testParsesV2AndArraysAndScalarTypes() throws {
        var b = GGUFBuilder.header(tensorCount: 0, kvCount: 4, version: 2)
        b.kvString("general.architecture", "llama")
        // array of u32 (type 9, element type 4)
        b.str("llama.rope.dims"); b.u32(9); b.u32(4); b.u64(3)
        b.u32(10); b.u32(20); b.u32(30)
        // f32 scalar
        b.str("llama.rope.freq_base"); b.u32(6); b.f32(10000)
        // bool scalar
        b.str("general.quantized"); b.u32(7); b.data.append(1)

        let model = try NMPGGUFModel.parse(b.data)
        XCTAssertEqual(model.version, 2)
        XCTAssertEqual(model.metadata["llama.rope.dims"],
                       .array([.uint32(10), .uint32(20), .uint32(30)]))
        XCTAssertEqual(model.metadata["llama.rope.freq_base"], .float32(10000))
        XCTAssertEqual(model.metadata["general.quantized"], .bool(true))
    }

    func testRejectsWrongMagic() {
        var b = GGUFBuilder()
        b.u32(0xDEAD_BEEF); b.u32(3); b.u64(0); b.u64(0)
        XCTAssertThrowsError(try NMPGGUFModel.parse(b.data)) {
            guard case NMPGGUFError.notGGUF = $0 else {
                return XCTFail("expected .notGGUF, got \($0)")
            }
        }
    }

    func testRejectsUnsupportedVersionAndTruncation() {
        var v99 = GGUFBuilder.header(tensorCount: 0, kvCount: 0, version: 99)
        XCTAssertThrowsError(try NMPGGUFModel.parse(v99.data)) {
            XCTAssertEqual($0 as? NMPGGUFError, .unsupportedVersion(99))
        }
        _ = v99 // silence mutation warning

        let whole = makeLlamaContainer()
        for cut in [3, 11, 25, whole.count / 2] {
            XCTAssertThrowsError(try NMPGGUFModel.parse(whole.prefix(cut)),
                                 "cut at \(cut) must throw")
        }
    }

    func testRejectsHostileCounts() {
        // kv_count = 2^40: must be rejected up front, not allocated.
        var b = GGUFBuilder()
        b.u32(NMPGGUFModel.magic); b.u32(3); b.u64(0); b.u64(1 << 40)
        XCTAssertThrowsError(try NMPGGUFModel.parse(b.data)) {
            guard case NMPGGUFError.implausibleCount = $0 else {
                return XCTFail("expected .implausibleCount, got \($0)")
            }
        }
    }

    func testLoadsFromDiskAndFeedsEngine() throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("nmp-test-\(UInt32.random(in: 0...UInt32.max)).gguf").path
        try makeLlamaContainer().write(to: URL(fileURLWithPath: path))
        defer { try? FileManager.default.removeItem(atPath: path) }

        let model = try NMPGGUFModel.load(path: path)
        // The engine sizes itself from real container metadata.
        let engine = try NMPReferenceComputeEngine(gguf: model)
        XCTAssertEqual(engine.layerCount, 32)
        XCTAssertEqual(engine.hiddenSize, 4096)
    }
}
