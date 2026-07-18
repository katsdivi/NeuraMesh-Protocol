//
//  LlamaShardTests.swift
//  NMP — Phase 10 (TRUE cross-device layer sharding)
//
//  Proves the REAL sharded path end-to-end through the Swift layer: two (and
//  three) NMPLlamaShardComputeEngines, each partial-loading ONLY its block
//  range, chained exactly the way the pipeline chains stages (output residual
//  of one shard is the input of the next). The greedy token stream must equal
//  the whole-model llama.cpp output.
//
//  Gating: needs the ggml shard shim (scripts/setup_shard.sh →
//  Vendor/llama/libnmpshard.dylib) AND a Qwen2 GGUF; otherwise XCTSkip.
//  The PINNED golden stream below is Qwen2.5-0.5B for "The capital of France
//  is" (token ids [785,6722,315,9625,374]). For any OTHER model (e.g. the
//  tied-LM-head Qwen2.5-1.5B) the expected stream is derived live from
//  llama.cpp — the same oracle that produced the pin — so the whole class
//  runs bit-exact against upstream on both models.
//

import XCTest
@testable import NMP

/// Fixtures for the shard shim (independent of the llama.cpp shim — the
/// low-level tests need only libnmpshard + a model file).
enum ShardTestSupport {
    static let modelPath: String? = LlamaTestSupport.modelPath

    static var shimAvailable: Bool {
        NMPLlamaShardRuntime.locate() != nil && modelPath != nil
    }

    static func requireModelPath() throws -> String {
        guard NMPLlamaShardRuntime.locate() != nil else {
            throw XCTSkip("shard shim missing — run scripts/setup_shard.sh (needs `brew install ggml`)")
        }
        guard let modelPath else {
            throw XCTSkip("no GGUF model — set NMP_LLAMA_MODEL")
        }
        return modelPath
    }

    /// "The capital of France is" and its whole-model greedy continuation,
    /// "Paris. It is the largest city in" — the oracle from the C self-test.
    /// This pin is specific to qwen2.5-0.5b-instruct (hidden 896, 24 blocks).
    static let promptTokens: [Int32] = [785, 6722, 315, 9625, 374]
    static let goldenContinuation: [Int32] = [12095, 13, 1084, 374, 279, 7772, 3283, 304]

    /// The whole-model greedy continuation of `promptTokens` for WHATEVER
    /// model NMP_LLAMA_MODEL points at: the pinned 0.5B stream when it is the
    /// pinned fixture, else derived live from llama.cpp (nil when the llama
    /// shim can't load to derive it). Computed once per process.
    static let expectedContinuation: [Int32]? = {
        guard let modelPath else { return nil }
        if let gguf = try? NMPGGUFModel.load(path: modelPath),
           gguf.hiddenSize == 896, gguf.layerCount == 24 {
            return goldenContinuation   // the pinned 0.5B fixture
        }
        guard let oracle = LlamaTestSupport.fullModel else { return nil }
        var context = promptTokens
        var stream: [Int32] = []
        for _ in 0..<goldenContinuation.count {
            guard let top = try? oracle.decodeTopK(tokens: context, basePos: 0, k: 1).first
            else { return nil }
            stream.append(top.id)
            context.append(top.id)
        }
        return stream
    }()

    static func requireGolden() throws -> [Int32] {
        guard let stream = expectedContinuation else {
            throw XCTSkip("model is not the pinned 0.5B fixture and the llama.cpp shim "
                          + "is unavailable to derive its golden stream")
        }
        return stream
    }

    /// True when the model ships NO output.weight (tied word embeddings —
    /// e.g. the Qwen2.5-1.5B GGUF): the LM head IS token_embd.weight.
    static func isTiedLMHead(path: String) throws -> Bool {
        let gguf = try NMPGGUFModel.load(path: path)
        return !gguf.tensors.contains { $0.name == "output.weight" }
    }

    /// Drives a split plan the way the pipeline does WITH the KV cache:
    /// prefill the whole prompt at cache position 0, then feed just the newest
    /// token at the growing position each step, walking every engine in order.
    static func generate(engines: [(engine: NMPLlamaShardComputeEngine, start: Int, end: Int)],
                         prompt: [Int32], count: Int, hiddenSize: Int) throws -> [Int32] {
        var generated: [Int32] = []
        var nPast = 0
        var batch = prompt   // first pass carries the whole prompt
        for _ in 0..<count {
            var activation = try NMPLlamaWire.encode(
                NMPLlamaWire.Request(basePos: nPast, tokens: batch), width: hiddenSize)
            for stage in engines {
                activation = try stage.engine.runLayers(
                    start: stage.start, end: stage.end, input: activation)
            }
            let response = try NMPLlamaWire.decodeResponse(activation)
            let top = try XCTUnwrap(response.top?.id)
            generated.append(top)
            nPast += batch.count
            batch = [top]   // decode one token at a time
        }
        return generated
    }
}

final class LlamaShardTests: XCTestCase {

    // MARK: Partial load

    /// Each shard loads ONLY its blocks — neither holds the whole model.
    func testShardsPartialLoadTheirRangeOnly() throws {
        let path = try ShardTestSupport.requireModelPath()
        let full = try NMPLlamaShard(modelPath: path, start: 0, end: -1) // whole model
        let n = full.layerCount
        XCTAssertGreaterThan(n, 1)
        let split = n / 2

        let a = try NMPLlamaShard(modelPath: path, start: 0, end: split)
        let b = try NMPLlamaShard(modelPath: path, start: split, end: n)

        XCTAssertTrue(a.isFirst); XCTAssertFalse(a.isLast)
        XCTAssertFalse(b.isFirst); XCTAssertTrue(b.isLast)

        let fileSize = (try FileManager.default
            .attributesOfItem(atPath: path)[.size] as? Int) ?? Int.max
        // Honest proof: each shard's loaded weights are a strict subset.
        XCTAssertGreaterThan(a.bytesLoaded, 0)
        XCTAssertGreaterThan(b.bytesLoaded, 0)
        XCTAssertLessThan(a.bytesLoaded, fileSize)
        XCTAssertLessThan(b.bytesLoaded, fileSize)
    }

    // MARK: Golden streams (bit-exact vs whole model)

    func testTwoWaySplitMatchesWholeModel() throws {
        let path = try ShardTestSupport.requireModelPath()
        let golden = try ShardTestSupport.requireGolden()
        let engineA = try NMPLlamaShardComputeEngine(modelPath: path)
        let engineB = try NMPLlamaShardComputeEngine(modelPath: path)
        let n = engineA.layerCount
        let split = n / 2

        let out = try ShardTestSupport.generate(
            engines: [(engineA, 0, split), (engineB, split, n)],
            prompt: ShardTestSupport.promptTokens,
            count: golden.count,
            hiddenSize: engineA.hiddenSize)
        XCTAssertEqual(out, golden)

        // The engines really only loaded their halves.
        let fileSize = (try FileManager.default
            .attributesOfItem(atPath: path)[.size] as? Int) ?? Int.max
        XCTAssertLessThan(engineA.loadedBytes, fileSize)
        XCTAssertLessThan(engineB.loadedBytes, fileSize)
    }

    func testThreeWaySplitMatchesWholeModel() throws {
        let path = try ShardTestSupport.requireModelPath()
        let golden = try ShardTestSupport.requireGolden()
        let engineA = try NMPLlamaShardComputeEngine(modelPath: path)
        let engineB = try NMPLlamaShardComputeEngine(modelPath: path)
        let engineC = try NMPLlamaShardComputeEngine(modelPath: path)
        let n = engineA.layerCount
        let s1 = n / 3, s2 = (2 * n) / 3

        let out = try ShardTestSupport.generate(
            engines: [(engineA, 0, s1), (engineB, s1, s2), (engineC, s2, n)],
            prompt: ShardTestSupport.promptTokens,
            count: golden.count,
            hiddenSize: engineA.hiddenSize)
        XCTAssertEqual(out, golden)
    }

    func testSingleShardWholeModelMatches() throws {
        let path = try ShardTestSupport.requireModelPath()
        let golden = try ShardTestSupport.requireGolden()
        let engine = try NMPLlamaShardComputeEngine(modelPath: path)
        let n = engine.layerCount
        // start==0 && end==N: one shard, tokens straight to logits.
        let out = try ShardTestSupport.generate(
            engines: [(engine, 0, n)],
            prompt: ShardTestSupport.promptTokens,
            count: golden.count,
            hiddenSize: engine.hiddenSize)
        XCTAssertEqual(out, golden)
    }

    // MARK: KV cache — the wire hand-off shrinks during decode

    /// With the per-shard KV cache, a decode step ships only the NEW token's
    /// residual (n_embd), not the whole sequence (n_embd × T) — the bandwidth
    /// win that keeps per-token mesh round trips small.
    func testDecodeShrinksWireToOneTokenResidual() throws {
        let path = try ShardTestSupport.requireModelPath()
        let engine = try NMPLlamaShardComputeEngine(modelPath: path)
        let n = engine.layerCount
        let split = n / 2
        let hidden = engine.hiddenSize
        let prompt = ShardTestSupport.promptTokens

        // Prefill: the first shard emits the residual for the whole prompt.
        let prefillIn = try NMPLlamaWire.encode(
            NMPLlamaWire.Request(basePos: 0, tokens: prompt), width: hidden)
        let prefillOut = try engine.runLayers(start: 0, end: split, input: prefillIn)
        let prefill = try NMPLlamaShardWire.decodeShardResponse(prefillOut)
        XCTAssertEqual(prefill.hiddenState.count, hidden * prompt.count)

        // Decode one token at the next position: the residual is ONE position.
        let decodeIn = try NMPLlamaWire.encode(
            NMPLlamaWire.Request(basePos: prompt.count, tokens: [12095]), width: hidden)
        let decodeOut = try engine.runLayers(start: 0, end: split, input: decodeIn)
        let decode = try NMPLlamaShardWire.decodeShardResponse(decodeOut)
        XCTAssertEqual(decode.hiddenState.count, hidden,
                       "decode must ship a single-token residual, not the whole sequence")
    }

    // MARK: Falsification

    /// Zeroing the residual before the last shard MUST change the output —
    /// proof the downstream shard genuinely depends on the hand-off (the old
    /// fake, which re-ran from tokens, would have been unaffected).
    func testZeroedResidualDivergesFromGolden() throws {
        let path = try ShardTestSupport.requireModelPath()
        let golden = try ShardTestSupport.requireGolden()
        let engineA = try NMPLlamaShardComputeEngine(modelPath: path)
        let engineB = try NMPLlamaShardComputeEngine(modelPath: path)
        let n = engineA.layerCount
        let split = n / 2
        let hidden = engineA.hiddenSize

        let request = try NMPLlamaWire.encode(
            NMPLlamaWire.Request(basePos: 0, tokens: ShardTestSupport.promptTokens),
            width: hidden)
        let shardWire = try engineA.runLayers(start: 0, end: split, input: request)

        // Corrupt the residual to all zeros, keep the header/tokens intact.
        var corrupted = try NMPLlamaShardWire.decodeShardResponse(shardWire)
        let zeroed = NMPLlamaShardWire.ShardResponse(
            nextPos: corrupted.nextPos, tokens: corrupted.tokens,
            hiddenState: [Float](repeating: 0, count: corrupted.hiddenState.count))
        corrupted = zeroed
        let corruptedWire = try NMPLlamaShardWire.encode(corrupted, width: hidden)

        let goodOut = try engineB.runLayers(start: split, end: n, input: shardWire)
        let badOut = try engineB.runLayers(start: split, end: n, input: corruptedWire)
        let good = try NMPLlamaWire.decodeResponse(goodOut).top?.id
        let bad = try NMPLlamaWire.decodeResponse(badOut).top?.id
        XCTAssertEqual(good, golden.first)
        XCTAssertNotEqual(good, bad)
    }

    // MARK: Edge cases

    /// A sequence longer than the KV cache capacity must fail cleanly, not
    /// crash or emit garbage.
    func testMaxContextOverflowIsGracefulError() throws {
        let path = try ShardTestSupport.requireModelPath()
        let engine = try NMPLlamaShardComputeEngine(modelPath: path, maxContext: 8)
        let n = engine.layerCount
        // 20 tokens into an 8-slot cache — over capacity on the first pass.
        let longPrompt = [Int32](repeating: 374, count: 20)
        let request = try NMPLlamaWire.encode(
            NMPLlamaWire.Request(basePos: 0, tokens: longPrompt), width: engine.hiddenSize)
        XCTAssertThrowsError(try engine.runLayers(start: 0, end: n, input: request),
                             "over-capacity prompt must throw, not crash")
    }

    /// An extreme boundary — a single-layer first shard — is still bit-exact.
    func testSingleLayerFirstShardMatches() throws {
        let path = try ShardTestSupport.requireModelPath()
        let golden = try ShardTestSupport.requireGolden()
        let engineA = try NMPLlamaShardComputeEngine(modelPath: path)
        let engineB = try NMPLlamaShardComputeEngine(modelPath: path)
        let n = engineA.layerCount
        let out = try ShardTestSupport.generate(
            engines: [(engineA, 0, 1), (engineB, 1, n)],
            prompt: ShardTestSupport.promptTokens,
            count: golden.count,
            hiddenSize: engineA.hiddenSize)
        XCTAssertEqual(out, golden)
    }

    // MARK: Codec (needs the llama.cpp shim for the tokenizer)

    /// The shard-aware prompt codec feeds the WHOLE sequence each step and the
    /// full engine+codec text loop is deterministic (greedy). Needs both shims.
    func testShardCodecTextLoopDeterministic() throws {
        guard LlamaTestSupport.available else {
            throw XCTSkip("llama.cpp shim/model missing — needed for the tokenizer")
        }
        let path = try ShardTestSupport.requireModelPath()
        let vocab = try LlamaTestSupport.requireVocabOnly()
        let engineA = try NMPLlamaShardComputeEngine(modelPath: path)
        let engineB = try NMPLlamaShardComputeEngine(modelPath: path)
        let n = engineA.layerCount
        let split = n / 2

        func run() throws -> String {
            let codec = NMPLlamaShardPromptCodec(model: vocab)
            var activation = try codec.makeInitialInput(prompt: "The capital of France is")
            var tokens: [NMPGeneratedToken] = []
            for _ in 0..<6 {
                var a = activation
                a = try engineA.runLayers(start: 0, end: split, input: a)
                a = try engineB.runLayers(start: split, end: n, input: a)
                guard let token = try codec.extractToken(from: a, position: tokens.count) else { break }
                tokens.append(token)
                activation = try codec.makeNextInput(after: a, token: token, position: tokens.count)
            }
            return codec.render(tokens: tokens)
        }

        let first = try run()
        let second = try run()
        XCTAssertFalse(first.isEmpty)
        XCTAssertEqual(first, second, "greedy shard generation must be deterministic")
    }

    // MARK: Weight vault (Future Plan #3) — slice a model into per-shard GGUFs

    /// A full-range slice with nothing dropped round-trips: same tensors and
    /// same metadata. This validates the GGUF WRITER byte layout independently
    /// of any device (the risky part of the slicer).
    func testFullRangeSliceRoundTrips() throws {
        let path = try ShardTestSupport.requireModelPath()
        let original = try NMPGGUFModel.load(path: path)
        let n = try XCTUnwrap(original.layerCount)

        let out = NSTemporaryDirectory() + "nmp-slice-roundtrip-\(getpid()).gguf"
        defer { try? FileManager.default.removeItem(atPath: out) }
        try NMPGGUFSlicer.slice(modelPath: path, start: 0, end: n, to: out, dropKeys: [])

        let round = try NMPGGUFModel.load(path: out)
        XCTAssertEqual(Set(round.tensors.map(\.name)),
                       Set(original.tensors.map(\.name)),
                       "every tensor survives a full-range slice")
        let origByName = Dictionary(uniqueKeysWithValues: original.tensors.map { ($0.name, $0) })
        for t in round.tensors {
            let o = try XCTUnwrap(origByName[t.name])
            XCTAssertEqual(t.dimensions, o.dimensions, "\(t.name) dims")
            XCTAssertEqual(t.ggmlTypeID, o.ggmlTypeID, "\(t.name) type")
        }
        XCTAssertEqual(round.metadata, original.metadata, "metadata survives verbatim")
    }

    /// THE PROOF: two engines built from PER-SHARD SLICES (each holding only its
    /// layers) produce the exact whole-model greedy continuation — bit-identical
    /// to running from the full file. This is disk ≈ RAM: no slice holds it all.
    func testShardedGenerationFromSlicesMatchesFullModel() throws {
        let path = try ShardTestSupport.requireModelPath()
        let golden = try ShardTestSupport.requireGolden()
        // Tied-LM-head models: the TAIL slice carries token_embd.weight (it IS
        // the LM head) — NMPGGUFSlicer.wanted() mirrors the shim's tied-aware
        // want(), so this proof runs for tied and untied models alike.
        let n = try NMPLlamaShard(modelPath: path, start: 0, end: -1).layerCount
        let split = n / 2

        let dir = NSTemporaryDirectory()
        let loPath = dir + "nmp-vault-lo-\(getpid()).gguf"
        let hiPath = dir + "nmp-vault-hi-\(getpid()).gguf"
        defer {
            try? FileManager.default.removeItem(atPath: loPath)
            try? FileManager.default.removeItem(atPath: hiPath)
        }
        try NMPGGUFSlicer.slice(modelPath: path, start: 0, end: split, to: loPath)
        try NMPGGUFSlicer.slice(modelPath: path, start: split, end: n, to: hiPath)

        // Each slice is strictly smaller than the whole model — the disk win.
        let fullBytes = try XCTUnwrap(
            FileManager.default.attributesOfItem(atPath: path)[.size] as? Int)
        let loBytes = try XCTUnwrap(
            FileManager.default.attributesOfItem(atPath: loPath)[.size] as? Int)
        let hiBytes = try XCTUnwrap(
            FileManager.default.attributesOfItem(atPath: hiPath)[.size] as? Int)
        XCTAssertLessThan(loBytes, fullBytes, "lo slice must not hold the whole model")
        XCTAssertLessThan(hiBytes, fullBytes, "hi slice must not hold the whole model")

        let loEngine = try NMPLlamaShardComputeEngine(modelPath: loPath)
        let hiEngine = try NMPLlamaShardComputeEngine(modelPath: hiPath)
        // Slices preserve the FULL block_count, so the engines agree on N.
        XCTAssertEqual(loEngine.layerCount, n)
        XCTAssertEqual(hiEngine.layerCount, n)

        let got = try ShardTestSupport.generate(
            engines: [(loEngine, 0, split), (hiEngine, split, n)],
            prompt: ShardTestSupport.promptTokens,
            count: golden.count,
            hiddenSize: loEngine.hiddenSize)
        XCTAssertEqual(got, golden,
                       "sharded-from-slices output must match the whole-model oracle")
    }

    // MARK: BUG-1 regression — tied LM head / wide heads (Qwen2.5-1.5B class)

    /// BUG-1 (SEV-1): ANY generation on Qwen2.5-1.5B-Instruct segfaulted the
    /// whole mesh on the FIRST token. Root cause: tied-word-embedding GGUFs
    /// (Qwen2.5 ≤3B) ship NO `output.weight` — the LM head is
    /// `token_embd.weight` — and the shim handed ggml_mul_mat the missing
    /// tensor (NULL → EXC_BAD_ACCESS at 0x10, NULL->ne[0]). The fix loads and
    /// uses token_embd as the LM head (llama.cpp's own loader fallback) and
    /// validates the full tensor set at open. Gate: runs only when
    /// NMP_LLAMA_MODEL points at a >64-head_dim model (the 1.5B: head_dim
    /// 128); the 0.5B fixture (head_dim 64, output.weight present) skips.
    func testWideHeadTiedLMHeadModelGeneratesBitExact() throws {
        let path = try ShardTestSupport.requireModelPath()
        let gguf = try NMPGGUFModel.load(path: path)
        let heads = try XCTUnwrap(gguf.attentionHeadCount)
        let hidden = try XCTUnwrap(gguf.hiddenSize)
        let headDim = hidden / heads
        guard headDim > 64 else {
            throw XCTSkip("model head_dim \(headDim) ≤ 64 — point NMP_LLAMA_MODEL at "
                          + "Qwen2.5-1.5B-Instruct (head_dim 128) for this regression")
        }
        let tied = try ShardTestSupport.isTiedLMHead(path: path)
        let count = ShardTestSupport.goldenContinuation.count

        // 1) The whole-model single-shard path — BUG-1's exact crash site
        //    (LlamaShardEngine's isFirst && isLast local path) — generates.
        let whole = try NMPLlamaShardComputeEngine(modelPath: path)
        let n = whole.layerCount
        let wholeStream = try ShardTestSupport.generate(
            engines: [(whole, 0, n)], prompt: ShardTestSupport.promptTokens,
            count: count, hiddenSize: whole.hiddenSize)
        XCTAssertEqual(wholeStream.count, count)

        // 2) Greedy determinism: fresh engine, identical stream.
        let again = try NMPLlamaShardComputeEngine(modelPath: path)
        let rerun = try ShardTestSupport.generate(
            engines: [(again, 0, n)], prompt: ShardTestSupport.promptTokens,
            count: count, hiddenSize: again.hiddenSize)
        XCTAssertEqual(rerun, wholeStream, "greedy whole-model stream must be deterministic")

        // 3) A 2-way split — whose LAST shard must load the tied LM head
        //    WITHOUT holding block 0 — stays bit-exact vs the whole model.
        let engineA = try NMPLlamaShardComputeEngine(modelPath: path)
        let engineB = try NMPLlamaShardComputeEngine(modelPath: path)
        let split = n / 2
        let splitStream = try ShardTestSupport.generate(
            engines: [(engineA, 0, split), (engineB, split, n)],
            prompt: ShardTestSupport.promptTokens,
            count: count, hiddenSize: engineA.hiddenSize)
        XCTAssertEqual(splitStream, wholeStream,
                       "split stream must equal the whole-model stream (tied=\(tied))")

        // 4) Oracle: bit-exact vs llama.cpp itself, when its shim is loadable
        //    (the same bar the 0.5B pinned golden holds).
        if let expected = ShardTestSupport.expectedContinuation {
            XCTAssertEqual(wholeStream, expected,
                           "shard-shim greedy stream must be bit-exact vs llama.cpp")
        }

        // 5) Text sanity: the continuation of "The capital of France is"
        //    must be coherent, not the punctuation soup of a wrong LM head.
        if let vocab = LlamaTestSupport.vocabOnlyModel {
            var bytes = Data()
            for id in wholeStream { bytes.append(try vocab.pieceBytes(for: id)) }
            let text = String(decoding: bytes, as: UTF8.self)
            XCTAssertTrue(text.contains("Paris"),
                          "greedy continuation should mention Paris, got: \(text)")
        }
    }
}
