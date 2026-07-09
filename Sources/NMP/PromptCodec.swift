//
//  PromptCodec.swift
//  NMP — Phase 8
//
//  The text seam of prompt inference. NMPPromptInferenceService drives
//  the token loop (one full mesh pass per token); HOW text becomes the
//  first activation vector, how a token is read out of an output vector,
//  and how the next input is built are engine-specific — this protocol
//  captures exactly those steps.
//
//  - NMPReferencePromptCodec: the Phase 6 behavior, bit-for-bit (splitmix
//    embedding, argmax over the built-in vocabulary, mixed feedback).
//  - NMPLlamaPromptCodec (LlamaEngine.swift): real tokenizer, token-state
//    wire vectors, greedy sampling over real logits, EOS-aware.
//
//  Codecs are stateless across calls: everything the next step needs
//  travels inside the vectors themselves, so a retried or replayed pass
//  cannot desynchronize the generation.
//

import Foundation

/// One generated token: `index` in the engine's vocabulary, `text` as a
/// human-readable piece (may be lossy for multi-byte splits — `render`
/// re-derives exact text from indices).
public struct NMPGeneratedToken: Equatable, Sendable {
    public let index: Int
    public let text: String
    public init(index: Int, text: String) {
        self.index = index
        self.text = text
    }
}

public protocol NMPPromptCodec: AnyObject {
    /// Surfaced to API clients in GenerationResult.engine.
    var engineName: String { get }

    /// The first pipeline input for a prompt (throws when the prompt
    /// cannot be encoded, e.g. exceeds the tensor's token capacity).
    func makeInitialInput(prompt: String) throws -> [Float]

    /// Reads the generated token out of a pipeline output. `position` is
    /// the number of tokens generated so far. nil = end of generation
    /// (e.g. the model emitted EOS) — the loop stops early.
    func extractToken(from output: [Float], position: Int) throws -> NMPGeneratedToken?

    /// Builds the next pipeline input after `token` was extracted from
    /// `output` at `position`.
    func makeNextInput(after output: [Float], token: NMPGeneratedToken,
                       position: Int) throws -> [Float]

    /// Final text for the generated tokens.
    func render(tokens: [NMPGeneratedToken]) -> String
}

// MARK: - Reference codec (Phase 6 behavior, unchanged)

/// Deterministic pseudo-text codec for NMPReferenceComputeEngine — the
/// exact Phase 6 semantics, delegated to the static helpers that the
/// existing tests pin down.
public final class NMPReferencePromptCodec: NMPPromptCodec {

    public let engineName: String
    private let hiddenSize: Int

    public init(hiddenSize: Int, engineName: String = "reference") {
        self.hiddenSize = hiddenSize
        self.engineName = engineName
    }

    public func makeInitialInput(prompt: String) throws -> [Float] {
        NMPPromptInferenceService.embed(prompt: prompt, hiddenSize: hiddenSize)
    }

    public func extractToken(from output: [Float], position: Int) throws -> NMPGeneratedToken? {
        let sampled = NMPPromptInferenceService.sampleToken(from: output)
        return NMPGeneratedToken(index: sampled.index, text: sampled.word)
    }

    public func makeNextInput(after output: [Float], token: NMPGeneratedToken,
                              position: Int) throws -> [Float] {
        NMPPromptInferenceService.feedback(
            output: output, tokenIndex: token.index, position: position)
    }

    public func render(tokens: [NMPGeneratedToken]) -> String {
        NMPPromptInferenceService.render(words: tokens.map(\.text))
    }
}
