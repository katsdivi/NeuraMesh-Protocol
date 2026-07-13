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
//  The golden stream below is Qwen2.5-0.5B for "The capital of France is"
//  (token ids [785,6722,315,9625,374]); point NMP_LLAMA_MODEL at that model.
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
    static let promptTokens: [Int32] = [785, 6722, 315, 9625, 374]
    static let goldenContinuation: [Int32] = [12095, 13, 1084, 374, 279, 7772, 3283, 304]

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
        let engineA = try NMPLlamaShardComputeEngine(modelPath: path)
        let engineB = try NMPLlamaShardComputeEngine(modelPath: path)
        let n = engineA.layerCount
        let split = n / 2

        let out = try ShardTestSupport.generate(
            engines: [(engineA, 0, split), (engineB, split, n)],
            prompt: ShardTestSupport.promptTokens,
            count: ShardTestSupport.goldenContinuation.count,
            hiddenSize: engineA.hiddenSize)
        XCTAssertEqual(out, ShardTestSupport.goldenContinuation)

        // The engines really only loaded their halves.
        let fileSize = (try FileManager.default
            .attributesOfItem(atPath: path)[.size] as? Int) ?? Int.max
        XCTAssertLessThan(engineA.loadedBytes, fileSize)
        XCTAssertLessThan(engineB.loadedBytes, fileSize)
    }

    func testThreeWaySplitMatchesWholeModel() throws {
        let path = try ShardTestSupport.requireModelPath()
        let engineA = try NMPLlamaShardComputeEngine(modelPath: path)
        let engineB = try NMPLlamaShardComputeEngine(modelPath: path)
        let engineC = try NMPLlamaShardComputeEngine(modelPath: path)
        let n = engineA.layerCount
        let s1 = n / 3, s2 = (2 * n) / 3

        let out = try ShardTestSupport.generate(
            engines: [(engineA, 0, s1), (engineB, s1, s2), (engineC, s2, n)],
            prompt: ShardTestSupport.promptTokens,
            count: ShardTestSupport.goldenContinuation.count,
            hiddenSize: engineA.hiddenSize)
        XCTAssertEqual(out, ShardTestSupport.goldenContinuation)
    }

    func testSingleShardWholeModelMatches() throws {
        let path = try ShardTestSupport.requireModelPath()
        let engine = try NMPLlamaShardComputeEngine(modelPath: path)
        let n = engine.layerCount
        // start==0 && end==N: one shard, tokens straight to logits.
        let out = try ShardTestSupport.generate(
            engines: [(engine, 0, n)],
            prompt: ShardTestSupport.promptTokens,
            count: ShardTestSupport.goldenContinuation.count,
            hiddenSize: engine.hiddenSize)
        XCTAssertEqual(out, ShardTestSupport.goldenContinuation)
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
        XCTAssertEqual(good, ShardTestSupport.goldenContinuation.first)
        XCTAssertNotEqual(good, bad)
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
}
