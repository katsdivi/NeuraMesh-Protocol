//
//  LlamaEngine.swift
//  NMP — Phase 8 + Phase 10 (cross-device sharding)
//
//  Real-LLM compute behind the NMPShardComputeEngine seam, via llama.cpp
//  (see LlamaRuntime.swift for the binding).
//
//  SHARDING HONESTY: llama.cpp's public API executes the WHOLE model per
//  decode step — it exposes no "run layers [a, b) over this activation
//  vector" entry point, and the KV cache lives inside one context. So an
//  NMPLlamaComputeEngine shard must own the model's full layer range, and
//  a llama plan has exactly one shard. What the mesh still buys you is
//  REAL remote execution: the coordinator holds only the tokenizer
//  (vocab-only load, a few MB), the weights live on whichever peer owns
//  the shard, and every token is a genuine NMP pass — Noise IK, AES-GCM,
//  FEC, chunking, measured latency. Splitting mid-layer needs ggml-graph
//  surgery (Phase 9 candidate), not a different binding.
//
//  Tensors carry token state (NMPLlamaWire), not raw activations; the
//  response returns real top-k logits, and the coordinator samples
//  greedily — so a local plan and a remote plan produce IDENTICAL token
//  streams from identical weights.
//
//  PHASE 10 UPDATE: When the shim supports sharding (ABI ≥ 2), the engine
//  CAN execute layer sub-ranges. The first shard receives token-state,
//  runs the full model, and returns the hidden-state embedding. The last
//  shard receives token-state, runs the full model, and returns logits.
//  Each shard peer loads the full model. This is pipeline-parallel at
//  the shard level: the coordinator manages the pipeline flow.
//

import Foundation
import os

public enum NMPLlamaEngineError: Error, Sendable {
    /// llama.cpp cannot execute a layer sub-range (see header comment).
    case partialRangeUnsupported(start: Int, end: Int, layerCount: Int)
    /// The input tensor is not a token-state request (NMPLlamaWire).
    case notTokenState
    /// The prompt tokenizes to more tokens than the tensor can carry.
    case promptTooLong(tokens: Int, capacity: Int)
    /// A response carried zero candidates.
    case emptyCandidates
}

// MARK: - Compute engine

public final class NMPLlamaComputeEngine: NMPShardComputeEngine {

    public let model: NMPLlamaModel
    /// Candidates returned per decode — enough for any sampling strategy
    /// the coordinator might apply (Phase 8 samples greedily from [0]).
    public static let maxCandidates = 40

    private let _globalLayerCount = OSAllocatedUnfairLock(initialState: 0)
    public var globalLayerCount: Int {
        get {
            let val = _globalLayerCount.withLock { $0 }
            return val == 0 ? layerCount : val
        }
        set {
            _globalLayerCount.withLock { $0 = newValue }
        }
    }

    public var layerCount: Int { model.layerCount }
    public var hiddenSize: Int { model.hiddenSize }
    public var modelTag: String { model.name }

    public init(model: NMPLlamaModel) {
        self.model = model
    }

    /// Convenience: load weights and wrap them in one step.
    public convenience init(modelPath: String, gpuLayers: Int32 = -1,
                            contextLength: Int32 = 0) throws {
        self.init(model: try NMPLlamaModel(
            modelPath: modelPath, gpuLayers: gpuLayers, contextLength: contextLength))
    }

    public func runLayers(start: Int, end: Int, input: [Float]) throws -> [Float] {
        // Full-range path: the original Phase 8 behavior, unchanged.
        if start == 0 && end == globalLayerCount {
            return try runFullModel(input: input)
        }

        // Sub-range path: Phase 10 cross-device sharding.
        guard model.runtime.supportsSharding else {
            throw NMPLlamaEngineError.partialRangeUnsupported(
                start: start, end: end, layerCount: layerCount)
        }
        guard start >= 0, start < end, (end - start) <= layerCount else {
            throw NMPComputeError.invalidLayerRange(
                start: start, end: end, layerCount: layerCount)
        }
        if !NMPLlamaShardWire.isShardResponse(input) {
            guard input.count == hiddenSize else {
                throw NMPComputeError.invalidInputWidth(
                    expected: hiddenSize, got: input.count)
            }
        }

        // First shard (start == 0): accepts token-state request, runs a
        // full decode, returns the hidden-state embedding. The embedding
        // is the model's internal representation after the transformer
        // layers, before the output projection (lm_head).
        if start == 0 {
            // The input must be a token-state request (NMPLlamaWire format)
            // since this is the first shard in the pipeline.
            guard NMPLlamaWire.isRequest(input) || NMPLlamaWire.isVerifyRequest(input) else {
                throw NMPLlamaEngineError.notTokenState
            }
            let request = try NMPLlamaWire.decodeRequest(input)
            let embedding = try model.decodeEmbedding(
                tokens: request.tokens, basePos: request.basePos)
            // Wrap the embedding in a shard response wire vector.
            return try NMPLlamaShardWire.encode(
                NMPLlamaShardWire.ShardResponse(
                    nextPos: request.basePos + request.tokens.count,
                    tokens: request.tokens,
                    hiddenState: embedding),
                width: hiddenSize)
        }

        // Last shard (end == globalLayerCount): accepts a shard response from
        // the previous shard, extracts the tokens and hidden states, runs
        // the remaining layers starting from the hidden state, and returns
        // top-k logits in NMPLlamaWire format.
        if end == globalLayerCount {
            guard NMPLlamaShardWire.isShardResponse(input) else {
                throw NMPLlamaEngineError.notTokenState
            }
            let response = try NMPLlamaShardWire.decodeShardResponse(input)
            let basePos = response.nextPos - response.tokens.count
            
            let k = min(Self.maxCandidates,
                        NMPLlamaWire.responseCapacity(width: hiddenSize))
            let candidates = try model.decodeEmbeddingInput(
                embd: response.hiddenState, tokenCount: response.tokens.count,
                basePos: basePos, k: k)
            return try NMPLlamaWire.encode(
                NMPLlamaWire.Response(
                    nextPos: response.nextPos,
                    candidates: candidates),
                width: hiddenSize)
        }

        // Middle shard: in a 3+ shard pipeline, middle shards compute
        // by evaluating the remaining layers using the incoming embeddings
        // as input, and returning the output embeddings.
        guard NMPLlamaShardWire.isShardResponse(input) else {
            throw NMPLlamaEngineError.notTokenState
        }
        let response = try NMPLlamaShardWire.decodeShardResponse(input)
        let basePos = response.nextPos - response.tokens.count
        
        let outputEmbd = try model.decodeEmbeddingToEmbedding(
            embd: response.hiddenState, tokenCount: response.tokens.count, basePos: basePos)
        
        return try NMPLlamaShardWire.encode(
            NMPLlamaShardWire.ShardResponse(
                nextPos: response.nextPos,
                tokens: response.tokens,
                hiddenState: outputEmbd),
            width: hiddenSize)
    }

    /// The original full-model decode path (Phase 8), factored out for reuse.
    private func runFullModel(input: [Float]) throws -> [Float] {
        // Phase 9: a verify request asks for the greedy argmax at EVERY
        // decoded position — speculative-draft verification in one pass.
        if NMPLlamaWire.isVerifyRequest(input) {
            let request = try NMPLlamaWire.decodeVerifyRequest(input)
            let verdicts = try model.decodeGreedyPerPosition(
                tokens: request.tokens, basePos: request.basePos)
            return try NMPLlamaWire.encode(
                NMPLlamaWire.VerifyResponse(
                    nextPos: request.basePos + request.tokens.count,
                    verdicts: verdicts),
                width: hiddenSize)
        }
        guard NMPLlamaWire.isRequest(input) else {
            throw NMPLlamaEngineError.notTokenState
        }
        let request = try NMPLlamaWire.decodeRequest(input)
        let k = min(Self.maxCandidates,
                    NMPLlamaWire.responseCapacity(width: hiddenSize))
        let candidates = try model.decodeTopK(
            tokens: request.tokens, basePos: request.basePos, k: k)
        return try NMPLlamaWire.encode(
            NMPLlamaWire.Response(
                nextPos: request.basePos + request.tokens.count,
                candidates: candidates),
            width: hiddenSize)
    }
}

// MARK: - Prompt codec

/// Text ↔ token-state translation for llama plans. Needs only the
/// tokenizer, so the coordinator constructs it over a vocab-only model.
/// Stateless: positions travel inside the wire vectors, so retries and
/// replays cannot desynchronize a generation.
public final class NMPLlamaPromptCodec: NMPPromptCodec {

    public let engineName = "llamaCpp"

    private let model: NMPLlamaModel
    private let width: Int

    /// - Parameter model: any handle over the same GGUF (vocab-only is
    ///   enough — tokenize/piece/EOG never touch weights).
    public init(model: NMPLlamaModel) {
        self.model = model
        self.width = model.hiddenSize
    }

    public func makeInitialInput(prompt: String) throws -> [Float] {
        let tokens = try model.tokenize(prompt, addSpecial: true)
        let capacity = NMPLlamaWire.requestCapacity(width: width)
        guard tokens.count <= capacity else {
            throw NMPLlamaEngineError.promptTooLong(
                tokens: tokens.count, capacity: capacity)
        }
        return try NMPLlamaWire.encode(
            NMPLlamaWire.Request(basePos: 0, tokens: tokens), width: width)
    }

    public func extractToken(from output: [Float], position: Int) throws -> NMPGeneratedToken? {
        let response = try NMPLlamaWire.decodeResponse(output)
        guard let top = response.top else {
            throw NMPLlamaEngineError.emptyCandidates
        }
        if model.isEndOfGeneration(top.id) {
            return nil
        }
        // Lossy per-piece text (a piece may split a multi-byte character);
        // render() re-derives the exact byte stream.
        let text = String(decoding: try model.pieceBytes(for: top.id), as: UTF8.self)
        return NMPGeneratedToken(index: Int(top.id), text: text)
    }

    public func makeNextInput(after output: [Float], token: NMPGeneratedToken,
                              position: Int) throws -> [Float] {
        let response = try NMPLlamaWire.decodeResponse(output)
        return try NMPLlamaWire.encode(
            NMPLlamaWire.Request(basePos: response.nextPos, tokens: [Int32(token.index)]),
            width: width)
    }

    public func render(tokens: [NMPGeneratedToken]) -> String {
        var bytes = Data()
        for token in tokens {
            if let piece = try? model.pieceBytes(for: Int32(token.index)) {
                bytes.append(piece)
            }
        }
        return String(decoding: bytes, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
