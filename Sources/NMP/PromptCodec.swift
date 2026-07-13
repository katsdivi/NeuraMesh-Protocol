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
/// Two templates, chosen by engine:
/// - llama engines get the Llama-2-chat instruction format
///   (`[INST] … [/INST]`, system folded into the first instruction via
///   `<<SYS>>`), which the validated llama-2-7b-chat model was trained
///   on. No literal `<s>` tokens — the tokenizer adds BOS itself.
/// - everything else (the reference engine) gets a plain transcript
///   (`User: … / Assistant: …`); the reference engine emits placeholder
///   vocabulary regardless, so the template only needs to be consistent.
public enum NMPChatPrompt {

    public static func format(messages: [NMPChatMessage],
                              engine: String) -> String {
        engine.hasPrefix("llama")
            ? llamaChatFormat(messages)
            : transcriptFormat(messages)
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
