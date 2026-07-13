//
//  LlamaChurnTests.swift
//  NMP — Phase 10 bulletproofing (devices going in and out of the mesh)
//
//  Proves the re-shard correctness path for the REAL shard engine:
//
//    1. When a peer's layer range changes (a re-shard), its per-shard KV cache
//       is stale — the shim REFUSES an incremental decode into a fresh/empty
//       cache (a hard error) rather than emitting garbage.
//    2. A from-scratch re-prefill of the whole sequence recovers, and the
//       continued generation is BIT-IDENTICAL to a run that never churned.
//
//  This is what makes "degrade when a device leaves, upgrade when it returns"
//  safe. Gated on the ggml shard shim + a model.
//

import XCTest
@testable import NMP

final class LlamaChurnTests: XCTestCase {

    /// One pipeline pass across an ordered set of shard engines at cache
    /// position `nPast`, returning the greedy next-token id.
    private func step(_ engines: [(engine: NMPLlamaShardComputeEngine, start: Int, end: Int)],
                      batch: [Int32], nPast: Int, hiddenSize: Int) throws -> Int32 {
        var activation = try NMPLlamaWire.encode(
            NMPLlamaWire.Request(basePos: nPast, tokens: batch), width: hiddenSize)
        for stage in engines {
            activation = try stage.engine.runLayers(
                start: stage.start, end: stage.end, input: activation)
        }
        return try XCTUnwrap(try NMPLlamaWire.decodeResponse(activation).top?.id)
    }

    private func freshEngines(path: String, count: Int) throws -> [NMPLlamaShardComputeEngine] {
        try (0..<count).map { _ in try NMPLlamaShardComputeEngine(modelPath: path) }
    }

    /// Re-shard mid-generation: after `churnAt` tokens, swap to a DIFFERENT
    /// plan (fresh, empty caches). Assert the stale cache is caught, then
    /// re-prefill and finish — the whole stream must equal the no-churn
    /// baseline.
    private func runChurn(path: String, total: Int, churnAt: Int,
                          plan1: (Int) -> [(Int, Int)],   // n -> [(start,end)]
                          plan2: (Int) -> [(Int, Int)]) throws {
        let probe = try NMPLlamaShardComputeEngine(modelPath: path)
        let n = probe.layerCount
        let hidden = probe.hiddenSize
        let prompt = ShardTestSupport.promptTokens

        // Baseline: one shard, the whole model — the ground truth stream.
        let baseline = try ShardTestSupport.generate(
            engines: [(probe, 0, n)], prompt: prompt, count: total, hiddenSize: hidden)

        // Churn run.
        let e1 = try freshEngines(path: path, count: plan1(n).count)
        let stage1 = zip(e1, plan1(n)).map { (engine: $0, start: $1.0, end: $1.1) }
        let e2 = try freshEngines(path: path, count: plan2(n).count)
        let stage2 = zip(e2, plan2(n)).map { (engine: $0, start: $1.0, end: $1.1) }

        var generated: [Int32] = []
        var nPast = 0
        var batch = prompt
        var engines = stage1
        for i in 0..<total {
            if i == churnAt {
                engines = stage2
                // (1) The new shards have empty caches: an incremental decode
                //     at nPast>0 MUST be refused, not silently wrong.
                XCTAssertThrowsError(
                    try step(stage2, batch: batch, nPast: nPast, hiddenSize: hidden),
                    "re-shard must invalidate the stale KV cache")
                // (2) Recover by re-prefilling the whole sequence at position 0.
                batch = prompt + generated
                nPast = 0
            }
            let top = try step(engines, batch: batch, nPast: nPast, hiddenSize: hidden)
            generated.append(top)
            nPast += batch.count
            batch = [top]
        }
        XCTAssertEqual(generated, baseline,
                       "generation across a re-shard must match the no-churn baseline")
    }

    // MARK: Drop / degrade (fewer shards, different boundary)

    func testReShardToDifferentSplitRecovers() throws {
        let path = try ShardTestSupport.requireModelPath()
        // 2-way at n/2 → 2-way at n/3 (a boundary move, like a re-balance).
        try runChurn(path: path, total: 10, churnAt: 5,
                     plan1: { n in [(0, n / 2), (n / 2, n)] },
                     plan2: { n in [(0, n / 3), (n / 3, n)] })
    }

    func testDegradeToSingleShardRecovers() throws {
        let path = try ShardTestSupport.requireModelPath()
        // 2-way → 1-way (a peer left; the survivor holds the whole model).
        try runChurn(path: path, total: 10, churnAt: 4,
                     plan1: { n in [(0, n / 2), (n / 2, n)] },
                     plan2: { n in [(0, n)] })
    }

    // MARK: Join / upgrade (more shards)

    func testUpgradeToMoreShardsRecovers() throws {
        let path = try ShardTestSupport.requireModelPath()
        // 1-way → 3-way (two peers joined; the model spreads out).
        try runChurn(path: path, total: 10, churnAt: 4,
                     plan1: { n in [(0, n)] },
                     plan2: { n in [(0, n / 3), (n / 3, 2 * n / 3), (2 * n / 3, n)] })
    }

    // MARK: Codec recovery contract (the Swift glue the service uses on retry)

    /// The shard codec's rebuildInput() must reconstruct a from-scratch prefill
    /// (basePos 0) of the WHOLE sequence so far — prompt + every accepted token.
    func testShardCodecRebuildInputReconstructsFullSequence() throws {
        guard LlamaTestSupport.available else {
            throw XCTSkip("llama.cpp shim/model missing — needed for the tokenizer")
        }
        let vocab = try LlamaTestSupport.requireVocabOnly()
        let width = vocab.hiddenSize
        let codec = NMPLlamaShardPromptCodec(model: vocab)

        let prompt = try vocab.tokenize("The capital of France is", addSpecial: true)
        _ = try codec.makeInitialInput(prompt: "The capital of France is")

        // Simulate two accepted tokens flowing back through the codec.
        var pos = prompt.count
        for id in [Int32(12095), Int32(13)] {
            let response = try NMPLlamaWire.encode(
                NMPLlamaWire.Response(nextPos: pos, candidates: [(id, 1.0)]), width: width)
            _ = try codec.makeNextInput(
                after: response, token: NMPGeneratedToken(index: Int(id), text: ""),
                position: pos)
            pos += 1
        }

        let rebuilt = try XCTUnwrap(codec.rebuildInput())
        let request = try NMPLlamaWire.decodeRequest(rebuilt)
        XCTAssertEqual(request.basePos, 0, "recovery must re-prefill from position 0")
        XCTAssertEqual(request.tokens, prompt + [12095, 13],
                       "recovery must carry the whole sequence so far")
    }
}
