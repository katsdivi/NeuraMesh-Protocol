//
//  LlamaTests.swift
//  NMP — Phase 8
//
//  Two tiers:
//
//  1. Wire-format tests — pure Swift, run everywhere (they pin the
//     token-state convention that llama shards and codecs speak).
//
//  2. Real-model tests — need the shim (scripts/setup_llama.sh) AND a
//     GGUF model; otherwise they XCTSkip. Point NMP_LLAMA_MODEL at any
//     chat model, e.g.:
//
//       NMP_LLAMA_MODEL=~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
//           swift test --filter Llama
//
//     A small model keeps the suite fast; the format/mesh behavior under
//     test is model-size-independent.
//

import XCTest
@testable import NMP

// MARK: - Shared real-model fixtures

/// Loads the model once per process (weights + a vocab-only twin) — the
/// real-model tests share these instead of re-reading GGUF per test.
enum LlamaTestSupport {
    static let modelPath: String? = {
        let candidates = [
            ProcessInfo.processInfo.environment["NMP_LLAMA_MODEL"],
            "~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "~/models/llama-2-7b-chat.Q4_K_M.gguf",
        ]
        for candidate in candidates {
            guard let candidate else { continue }
            let expanded = (candidate as NSString).expandingTildeInPath
            if FileManager.default.fileExists(atPath: expanded) { return expanded }
        }
        return nil
    }()

    static var available: Bool {
        NMPLlamaRuntime.locate() != nil && modelPath != nil
    }

    static let fullModel: NMPLlamaModel? = {
        guard available, let modelPath else { return nil }
        return try? NMPLlamaModel(modelPath: modelPath)
    }()

    static let vocabOnlyModel: NMPLlamaModel? = {
        guard available, let modelPath else { return nil }
        return try? NMPLlamaModel(modelPath: modelPath, vocabOnly: true)
    }()

    static func requireFullModel() throws -> NMPLlamaModel {
        guard available else {
            throw XCTSkip("llama shim or model missing — run scripts/setup_llama.sh "
                          + "and set NMP_LLAMA_MODEL")
        }
        guard let model = fullModel else {
            throw XCTSkip("model failed to load: \(modelPath ?? "?")")
        }
        return model
    }

    static func requireVocabOnly() throws -> NMPLlamaModel {
        guard available else {
            throw XCTSkip("llama shim or model missing — run scripts/setup_llama.sh "
                          + "and set NMP_LLAMA_MODEL")
        }
        guard let model = vocabOnlyModel else {
            throw XCTSkip("vocab-only load failed: \(modelPath ?? "?")")
        }
        return model
    }
}

// MARK: - Tier 1: wire format (always runs)

final class LlamaWireTests: XCTestCase {

    func testRequestRoundTrip() throws {
        let request = NMPLlamaWire.Request(basePos: 17, tokens: [1, 15043, 590, 338])
        let vector = try NMPLlamaWire.encode(request, width: 64)
        XCTAssertEqual(vector.count, 64)
        XCTAssertTrue(NMPLlamaWire.isRequest(vector))
        XCTAssertEqual(try NMPLlamaWire.decodeRequest(vector), request)
    }

    func testResponseRoundTripPreservesLogitsExactly() throws {
        let response = NMPLlamaWire.Response(
            nextPos: 21,
            candidates: [(310, 14.625), (29892, -0.03125), (0, -11.5)])
        let vector = try NMPLlamaWire.encode(response, width: 64)
        XCTAssertFalse(NMPLlamaWire.isRequest(vector))
        let decoded = try NMPLlamaWire.decodeResponse(vector)
        XCTAssertEqual(decoded, response)
        XCTAssertEqual(decoded.top?.id, 310)
    }

    func testRequestCapacityIsEnforced() {
        let tokens = [Int32](repeating: 7, count: 62)
        // 62 tokens need width 65; 64 must fail, 65 must fit.
        XCTAssertThrowsError(try NMPLlamaWire.encode(
            NMPLlamaWire.Request(basePos: 0, tokens: tokens), width: 64))
        XCTAssertNoThrow(try NMPLlamaWire.encode(
            NMPLlamaWire.Request(basePos: 0, tokens: tokens), width: 65))
    }

    func testDecodeRejectsWrongMagicAndCorruptCounts() throws {
        let plain = [Float](repeating: 0.5, count: 32)
        XCTAssertThrowsError(try NMPLlamaWire.decodeRequest(plain))
        XCTAssertThrowsError(try NMPLlamaWire.decodeResponse(plain))

        // A request whose count field exceeds the tensor is rejected, not read OOB.
        var corrupt = try NMPLlamaWire.encode(
            NMPLlamaWire.Request(basePos: 0, tokens: [1, 2, 3]), width: 32)
        corrupt[2] = 1_000
        XCTAssertThrowsError(try NMPLlamaWire.decodeRequest(corrupt))
    }

    func testValuesAboveFloatExactRangeAreRejected() {
        XCTAssertThrowsError(try NMPLlamaWire.encode(
            NMPLlamaWire.Request(basePos: 1 << 24, tokens: [1]), width: 32))
    }

    /// The wire vectors survive the mesh's actual tensor codec unchanged —
    /// this is the exact byte path a request rides inside NMP packets.
    func testWireSurvivesTensorCodec() throws {
        let request = NMPLlamaWire.Request(basePos: 3, tokens: [151644, 872, 198])
        let vector = try NMPLlamaWire.encode(request, width: 896)
        let decoded = try NMPTensorCodec.decode(NMPTensorCodec.encode(vector))
        XCTAssertEqual(decoded.map(\.bitPattern), vector.map(\.bitPattern))
        XCTAssertEqual(try NMPLlamaWire.decodeRequest(decoded), request)
    }
}

final class LlamaShardWireTests: XCTestCase {

    func testShardRequestRoundTrip() throws {
        let request = NMPLlamaShardWire.ShardRequest(basePos: 17, tokens: [1, 2, 3, 4], hiddenState: [0.1, -0.5, 0.99, -100.2])
        let vector = try NMPLlamaShardWire.encode(request, width: 64)
        XCTAssertEqual(vector.count, 64)
        XCTAssertTrue(NMPLlamaShardWire.isShardRequest(vector))
        let decoded = try NMPLlamaShardWire.decodeShardRequest(vector)
        XCTAssertEqual(decoded.basePos, request.basePos)
        XCTAssertEqual(decoded.tokens, request.tokens)
        XCTAssertEqual(decoded.hiddenState, request.hiddenState)
    }

    func testShardResponseRoundTrip() throws {
        let response = NMPLlamaShardWire.ShardResponse(nextPos: 21, tokens: [5, 6], hiddenState: [-0.0125, 12.34, 45.67])
        let vector = try NMPLlamaShardWire.encode(response, width: 64)
        XCTAssertEqual(vector.count, 64)
        XCTAssertFalse(NMPLlamaShardWire.isShardRequest(vector))
        XCTAssertTrue(NMPLlamaShardWire.isShardResponse(vector))
        let decoded = try NMPLlamaShardWire.decodeShardResponse(vector)
        XCTAssertEqual(decoded.nextPos, response.nextPos)
        XCTAssertEqual(decoded.tokens, response.tokens)
        XCTAssertEqual(decoded.hiddenState, response.hiddenState)
    }

    func testCapacityLimits() throws {
        let state = [Float](repeating: 1.0, count: 62)
        // Header is 3 + tokens (1) = 4 slots. Max hidden capacity is 64 - 4 = 60.
        // It should encode successfully but truncate the hidden state to 60.
        let request = NMPLlamaShardWire.ShardRequest(basePos: 0, tokens: [1], hiddenState: state)
        let vector = try NMPLlamaShardWire.encode(request, width: 64)
        XCTAssertEqual(vector.count, 64)
        
        let decoded = try NMPLlamaShardWire.decodeShardRequest(vector)
        XCTAssertEqual(decoded.hiddenState.count, 60)
        XCTAssertEqual(decoded.hiddenState, Array(state[..<60]))
    }
}

// MARK: - Tier 2: real model (skips without shim + model)

final class LlamaEngineTests: XCTestCase {

    func testModelLoadsAndReportsPlausibleShape() throws {
        let model = try LlamaTestSupport.requireFullModel()
        XCTAssertTrue(model.hasWeights)
        XCTAssertGreaterThan(model.layerCount, 0)
        XCTAssertGreaterThanOrEqual(model.hiddenSize, 256)
        XCTAssertGreaterThan(model.vocabSize, 1000)
        XCTAssertGreaterThan(model.contextSize, 0)
    }

    func testVocabOnlyTokenizesButCannotDecode() throws {
        let vocabOnly = try LlamaTestSupport.requireVocabOnly()
        let full = try LlamaTestSupport.requireFullModel()
        XCTAssertFalse(vocabOnly.hasWeights)

        // llama.cpp's vocab_only load skips hparams; the GGUF fallback must
        // recover the true shape — the coordinator sizes SHARD_ASSIGN off it.
        XCTAssertEqual(vocabOnly.layerCount, full.layerCount)
        XCTAssertEqual(vocabOnly.hiddenSize, full.hiddenSize)

        let tokens = try vocabOnly.tokenize("Hello, my name is")
        XCTAssertFalse(tokens.isEmpty)
        // Round trip through pieces reproduces the words.
        var bytes = Data()
        for token in tokens { bytes.append(try vocabOnly.pieceBytes(for: token)) }
        XCTAssertTrue(String(decoding: bytes, as: UTF8.self).contains("name"))

        XCTAssertThrowsError(try vocabOnly.decodeTopK(tokens: tokens, basePos: 0, k: 4)) {
            guard case NMPLlamaRuntimeError.weightsNotLoaded = $0 else {
                return XCTFail("expected weightsNotLoaded, got \($0)")
            }
        }
    }

    /// Same tokens, same position ⇒ bit-identical top-k — the property
    /// the mesh's cross-device identity guarantee rests on.
    func testDecodeIsDeterministic() throws {
        let model = try LlamaTestSupport.requireFullModel()
        let tokens = try model.tokenize("The capital of France is")
        let first = try model.decodeTopK(tokens: tokens, basePos: 0, k: 8)
        let second = try model.decodeTopK(tokens: tokens, basePos: 0, k: 8)
        XCTAssertEqual(first.map(\.id), second.map(\.id))
        XCTAssertEqual(first.map(\.logit), second.map(\.logit))
        // Sorted by logit descending.
        XCTAssertEqual(first.map(\.logit), first.map(\.logit).sorted(by: >))
    }

    func testEngineServesFullRangeAndRejectsPartial() throws {
        let model = try LlamaTestSupport.requireFullModel()
        let engine = NMPLlamaComputeEngine(model: model)

        let prompt = try model.tokenize("Hello")
        let input = try NMPLlamaWire.encode(
            NMPLlamaWire.Request(basePos: 0, tokens: prompt), width: engine.hiddenSize)

        let output = try engine.runLayers(start: 0, end: engine.layerCount, input: input)
        let response = try NMPLlamaWire.decodeResponse(output)
        XCTAssertEqual(response.nextPos, prompt.count)
        XCTAssertFalse(response.candidates.isEmpty)
        XCTAssertLessThan(response.top!.id, Int32(model.vocabSize))

        // When sharding is supported, sub-ranges are allowed. Otherwise they must throw.
        if model.runtime.supportsSharding {
            let output = try engine.runLayers(start: 0, end: engine.layerCount / 2, input: input)
            XCTAssertTrue(NMPLlamaShardWire.isShardResponse(output))
        } else {
            XCTAssertThrowsError(try engine.runLayers(
                start: 0, end: engine.layerCount / 2, input: input))
        }
        // Raw (non-token-state) activations are a hard error too.
        XCTAssertThrowsError(try engine.runLayers(
            start: 0, end: engine.layerCount,
            input: [Float](repeating: 0.5, count: engine.hiddenSize)))
    }
}

// MARK: - Tier 2: mesh integration (skips without shim + model)

final class LlamaMeshIntegrationTests: XCTestCase {

    private func generate(placement: NMPLlamaTestbed.Placement,
                          model: NMPLlamaModel,
                          prompt: String, tokens: Int) throws
        -> NMPPromptInferenceService.GenerationResult {
        let engine = NMPLlamaComputeEngine(model: model)
        let testbed = try NMPLlamaTestbed(
            engine: engine, modelTag: model.name, placement: placement)
        try testbed.startSync()

        let service = NMPPromptInferenceService(
            orchestrator: testbed.orchestrator,
            codec: NMPLlamaPromptCodec(model: model))
        let done = expectation(description: "generation \(placement)")
        var outcome: Result<NMPPromptInferenceService.GenerationResult,
                            NMPPromptInferenceService.ServiceError>?
        service.run(prompt: prompt, maxTokens: tokens) { result in
            outcome = result
            done.fulfill()
        }
        wait(for: [done], timeout: 300)
        return try XCTUnwrap(outcome).get()
    }

    /// The Phase 8 headline: REAL llama text out of a full mesh pass per
    /// token, and the remote-shard plan (full transport stack) produces
    /// text IDENTICAL to the local plan — same weights, greedy sampling.
    func testLocalAndRemoteShardProduceIdenticalRealText() throws {
        let model = try LlamaTestSupport.requireFullModel()
        let prompt = "The capital of France is"

        let local = try generate(placement: .local, model: model,
                                 prompt: prompt, tokens: 12)
        let remote = try generate(placement: .remotePeer, model: model,
                                  prompt: prompt, tokens: 12)

        XCTAssertEqual(local.engine, "llamaCpp")
        XCTAssertEqual(local.shardCount, 1)
        XCTAssertEqual(remote.shardCount, 1)
        XCTAssertGreaterThan(local.tokenCount, 0)
        XCTAssertFalse(local.text.isEmpty)
        XCTAssertEqual(local.text, remote.text,
                       "mesh transport must not change the token stream")
        // The remote plan moved real bytes over the (in-memory) stack.
        XCTAssertGreaterThan(remote.networkPayloadBytes, 0)
        XCTAssertEqual(local.networkPayloadBytes, 0)
    }

    /// EOS stops generation early instead of padding to maxTokens, and
    /// the pass that produced EOS is still accounted (it cost a full mesh
    /// round trip even though it emitted no token).
    func testGenerationStopsAtEndOfGeneration() throws {
        let model = try LlamaTestSupport.requireFullModel()
        let result = try generate(
            placement: .local, model: model,
            prompt: "Q: What is 2+2? A: 4", tokens: 64)
        XCTAssertLessThanOrEqual(result.tokenCount, 64)
        if result.tokenCount < 64 {
            // Stopped early ⇒ exactly one extra (EOS) pass was measured.
            XCTAssertEqual(result.perTokenSeconds.count, result.tokenCount + 1)
        } else {
            XCTAssertEqual(result.perTokenSeconds.count, result.tokenCount)
        }
    }

    func testShardedMeshProducesIdenticalText() throws {
        let model = try LlamaTestSupport.requireFullModel()
        let prompt = "The capital of France is"

        let local = try generate(placement: .local, model: model,
                                 prompt: prompt, tokens: 12)

        // For sharded mode, we need a factory to instantiate multiple model/engine copies.
        let engine = NMPLlamaComputeEngine(model: model)
        let testbed = try NMPLlamaTestbed(
            engine: engine, modelTag: model.name, placement: .sharded(shardCount: 2),
            engineFactory: {
                let m = try NMPLlamaModel(modelPath: LlamaTestSupport.modelPath!)
                return NMPLlamaComputeEngine(model: m)
            }
        )
        try testbed.startSync()

        let service = NMPPromptInferenceService(
            orchestrator: testbed.orchestrator,
            codec: NMPLlamaPromptCodec(model: model))
        let done = expectation(description: "generation sharded")
        var outcome: Result<NMPPromptInferenceService.GenerationResult,
                            NMPPromptInferenceService.ServiceError>?
        service.run(prompt: prompt, maxTokens: 12) { result in
            outcome = result
            done.fulfill()
        }
        wait(for: [done], timeout: 300)
        let shardedResult = try XCTUnwrap(outcome).get()

        XCTAssertEqual(shardedResult.shardCount, 2)
        XCTAssertEqual(local.text, shardedResult.text,
                       "sharded model execution must produce identical text to local baseline")
    }

    func testShardedWeightExecutionProducesIdenticalText() throws {
        let model = try LlamaTestSupport.requireFullModel()
        let sourcePath = LlamaTestSupport.modelPath!
        
        let fileManager = FileManager.default
        let part1Path = sourcePath.replacingOccurrences(of: ".gguf", with: "_part1.gguf")
        let part2Path = sourcePath.replacingOccurrences(of: ".gguf", with: "_part2.gguf")
        
        let layerCount = model.layerCount
        let middle = layerCount / 2
        
        // Helper to run python slice script
        func runSlicer(input: String, output: String, start: Int, end: Int) throws {
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            
            // Find current directory path to locate scripts/gguf_slice.py
            let currentDir = fileManager.currentDirectoryPath
            let scriptPath = "\(currentDir)/scripts/gguf_slice.py"
            
            task.arguments = ["python3", scriptPath, input, output, String(start), String(end)]
            try task.run()
            task.waitUntilExit()
            guard task.terminationStatus == 0 else {
                throw NSError(domain: "LlamaTests", code: Int(task.terminationStatus),
                              userInfo: [NSLocalizedDescriptionKey: "GGUF slice script failed with exit code \(task.terminationStatus)"])
            }
        }
        
        // Slicing model parts dynamically if not already cached
        if !fileManager.fileExists(atPath: part1Path) {
            print("Dynamic Test Setup: Slicing Part 1 (layers 0..\(middle))...")
            try runSlicer(input: sourcePath, output: part1Path, start: 0, end: middle)
        }
        if !fileManager.fileExists(atPath: part2Path) {
            print("Dynamic Test Setup: Slicing Part 2 (layers \(middle)..\(layerCount))...")
            try runSlicer(input: sourcePath, output: part2Path, start: middle, end: layerCount)
        }
        
        let prompt = "The capital of France is"
        let local = try generate(placement: .local, model: model,
                                 prompt: prompt, tokens: 12)
        
        // Print full model intermediate embeddings for prompt to compare
        let localCodec = NMPLlamaPromptCodec(model: model)
        let promptTokens = try model.tokenize(prompt, addSpecial: true)
        let fullEmbd = try model.decodeEmbedding(tokens: promptTokens, basePos: 0)
        let fullSumSq = fullEmbd.map { $0 * $0 }.reduce(0, +)
        print("DIAGNOSTIC: Full model layer 24 embedding sum of squares = \(fullSumSq)")
        print("DIAGNOSTIC: Full model layer 24 embedding first 5 floats = \(Array(fullEmbd.prefix(5)))")
        
        var callCount = 0
        let engineFactory = {
            defer { callCount += 1 }
            let path = callCount == 0 ? part1Path : part2Path
            let m = try NMPLlamaModel(modelPath: path)
            return NMPLlamaComputeEngine(model: m)
        }
        
        let testbed = try NMPLlamaTestbed(
            engine: NMPLlamaComputeEngine(model: model), // Host reference
            modelTag: model.name,
            placement: .sharded(shardCount: 2),
            engineFactory: engineFactory
        )
        try testbed.startSync()
        
        // Let's test Peer A directly first
        let peerAModel = try NMPLlamaModel(modelPath: part1Path)
        let peerAEmbd = try peerAModel.decodeEmbedding(tokens: promptTokens, basePos: 0)
        print("DIAGNOSTIC: Sliced Part 1 total embedding count = \(peerAEmbd.count)")
        for i in 0..<promptTokens.count {
            let slice = peerAEmbd[i*896 ..< (i+1)*896]
            let sumSq = slice.map { $0 * $0 }.reduce(0, +)
            let rms = sqrt(sumSq / 896.0)
            print("DIAGNOSTIC: Token \(i) RMS = \(rms)")
        }
        
        let service = NMPPromptInferenceService(
            orchestrator: testbed.orchestrator,
            codec: NMPLlamaPromptCodec(model: model))
        let done = expectation(description: "generation weight-sharded")
        var outcome: Result<NMPPromptInferenceService.GenerationResult,
                            NMPPromptInferenceService.ServiceError>?
        service.run(prompt: prompt, maxTokens: 12) { result in
            outcome = result
            done.fulfill()
        }
        wait(for: [done], timeout: 300)
        let shardedResult = try XCTUnwrap(outcome).get()
        
        XCTAssertEqual(shardedResult.shardCount, 2)
        print("DIAGNOSTIC: Local result: \(local.text)")
        print("DIAGNOSTIC: Sharded result: \(shardedResult.text)")
        XCTAssertEqual(local.text, shardedResult.text,
                       "weight-sharded model execution must produce identical text to local baseline")
    }
}
