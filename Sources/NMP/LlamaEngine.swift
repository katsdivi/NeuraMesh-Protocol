//
//  LlamaEngine.swift
//  NMP — Phase 8
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

import Foundation

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
        guard start == 0, end == layerCount else {
            throw NMPLlamaEngineError.partialRangeUnsupported(
                start: start, end: end, layerCount: layerCount)
        }
        guard input.count == hiddenSize else {
            throw NMPComputeError.invalidInputWidth(expected: hiddenSize, got: input.count)
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
