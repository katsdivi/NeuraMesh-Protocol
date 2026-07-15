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

    /// A from-scratch input that reconstructs the WHOLE generation so far, for
    /// recovery: when a pass fails (a dropped peer, a re-shard, a stale per-
    /// shard KV cache), retrying the same incremental input can't succeed —
    /// the caches must be refilled. Codecs that keep per-shard state (the real
    /// llama shard codec) return a full re-prefill here; others return nil and
    /// the retry just re-sends the same input. Default: nil.
    func rebuildInput() -> [Float]?

    /// Final text for the generated tokens.
    func render(tokens: [NMPGeneratedToken]) -> String
}

extension NMPPromptCodec {
    /// Default: no special recovery input (stateless / KV-cache-free codecs).
    public func rebuildInput() -> [Float]? { nil }
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

// MARK: - Chat prompt assembly (Mesh 2.7)

/// One turn of a chat conversation, as posted to POST /api/chat.
public struct NMPChatMessage: Equatable, Sendable {
    public enum Role: String, Sendable {
        case system, user, assistant
    }

    public let role: Role
    public let content: String

    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }
}

/// Folds a chat transcript into the single prompt string the token loop
/// consumes. The mesh itself is stateless across requests — the client
/// resends the whole conversation each turn, and the template makes the
/// engine treat it as dialogue.
///
/// Three templates, chosen by MODEL family first, engine second — the
/// template a model was trained on is a property of the model, not of
/// which engine happens to run it (the shard engine runs qwen2/qwen3,
/// the classic llama engine runs llama-2; both are "llama*" engines):
/// - qwen models get ChatML (`<|im_start|>role … <|im_end|>`). The
///   markers are real special tokens — the shim tokenizes with
///   parse_special, and generation stops at `<|im_end|>` (qwen's EOS).
/// - other llama-engine models get the Llama-2-chat instruction format
///   (`[INST] … [/INST]`, system folded into the first instruction via
///   `<<SYS>>`), which the validated llama-2-7b-chat model was trained
///   on. No literal `<s>` tokens — the tokenizer adds BOS itself.
/// - everything else (the reference engine) gets a plain transcript
///   (`User: … / Assistant: …`); the reference engine emits placeholder
///   vocabulary regardless, so the template only needs to be consistent.
public enum NMPChatPrompt {

    public static func format(messages: [NMPChatMessage],
                              engine: String,
                              model: String = "") -> String {
        if model.lowercased().contains("qwen") {
            return chatMLFormat(messages)
        }
        return engine.hasPrefix("llama")
            ? llamaChatFormat(messages)
            : transcriptFormat(messages)
    }

    /// ChatML, exactly as qwen2.5-instruct's chat template renders it —
    /// including the default system turn the official template injects
    /// when the client sends none.
    static func chatMLFormat(_ messages: [NMPChatMessage]) -> String {
        var out = ""
        if !messages.contains(where: { $0.role == .system }) {
            out += "<|im_start|>system\nYou are Qwen, created by Alibaba "
                 + "Cloud. You are a helpful assistant.<|im_end|>\n"
        }
        for message in messages {
            out += "<|im_start|>\(message.role.rawValue)\n"
                 + "\(message.content)<|im_end|>\n"
        }
        out += "<|im_start|>assistant\n"
        return out
    }

    static func llamaChatFormat(_ messages: [NMPChatMessage]) -> String {
        var system: String?
        var out = ""
        var pendingUser: String?

        func flushInstruction(_ user: String) {
            var instruction = user
            if let sys = system {
                instruction = "<<SYS>>\n\(sys)\n<</SYS>>\n\n\(user)"
                system = nil // only the first instruction carries it
            }
            out += out.isEmpty ? "[INST] \(instruction) [/INST]"
                               : " [INST] \(instruction) [/INST]"
        }

        for message in messages {
            switch message.role {
            case .system:
                system = message.content
            case .user:
                if let user = pendingUser { flushInstruction(user) }
                pendingUser = message.content
            case .assistant:
                if let user = pendingUser {
                    flushInstruction(user)
                    pendingUser = nil
                }
                out += " \(message.content)"
            }
        }
        if let user = pendingUser { flushInstruction(user) }
        return out
    }

    static func transcriptFormat(_ messages: [NMPChatMessage]) -> String {
        var lines: [String] = []
        for message in messages {
            switch message.role {
            case .system:    lines.append(message.content)
            case .user:      lines.append("User: \(message.content)")
            case .assistant: lines.append("Assistant: \(message.content)")
            }
        }
        lines.append("Assistant:")
        return lines.joined(separator: "\n")
    }
}
