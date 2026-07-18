//
//  PromptInferenceService.swift
//  NMP — Phase 6+
//
//  Bridges text prompts onto the mesh: embeds a prompt into an activation
//  vector, then generates tokens autoregressively — every token is one
//  full pipelined pass through the live mesh (real transport, crypto,
//  FEC, and timings), with the output tensor fed back as the next input.
//
//  ENGINE HONESTY: token *quality* is bounded by the compute engine. With
//  NMPReferenceComputeEngine the decoded words are deterministic pseudo-
//  text (sampled from the output tensor via a built-in vocabulary), while
//  every latency/byte metric is genuinely measured. Phase 8 binds real
//  llama.cpp: pass an NMPLlamaPromptCodec (LlamaEngine.swift) and the
//  same loop generates real LLM text — the codec seam (PromptCodec.swift)
//  is the only thing that changes.
//
//  Threading: callback style on a private serial queue (spec rule: no
//  async/await). The orchestrator is strictly one-request-in-flight, so
//  the service holds a `busy` flag and rejects concurrent runs.
//

import Foundation

public final class NMPPromptInferenceService {

    // MARK: Types

    public struct GenerationResult: Sendable {
        /// Decoded text for the generated tokens.
        public let text: String
        public let tokenCount: Int
        /// Wall clock across all pipeline passes.
        public let totalSeconds: TimeInterval
        /// Application payload bytes moved over the mesh (all passes).
        public let networkPayloadBytes: Int
        /// Shards in the plan that served the final pass.
        public let shardCount: Int
        /// Per-token pipeline latency (per-ROUND-TRIP when speculation is
        /// active — one round trip can emit several tokens).
        public let perTokenSeconds: [TimeInterval]
        /// Compute engine identity, surfaced to API clients.
        public let engine: String
        /// Phase 9: draft/verify accounting; nil when speculation was off.
        public let speculation: NMPSpeculationStats?

        public init(text: String, tokenCount: Int, totalSeconds: TimeInterval,
                    networkPayloadBytes: Int, shardCount: Int,
                    perTokenSeconds: [TimeInterval], engine: String,
                    speculation: NMPSpeculationStats? = nil) {
            self.text = text
            self.tokenCount = tokenCount
            self.totalSeconds = totalSeconds
            self.networkPayloadBytes = networkPayloadBytes
            self.shardCount = shardCount
            self.perTokenSeconds = perTokenSeconds
            self.engine = engine
            self.speculation = speculation
        }
    }

    public enum ServiceError: Error, Equatable {
        /// A generation is already in flight (the pipeline is sequential).
        case busy
        case emptyPrompt
        case orchestration(NMPOrchestrationError)
        /// The codec rejected the prompt or a pipeline output (e.g. prompt
        /// exceeds tensor capacity, malformed token-state response).
        case codec(String)
    }

    /// Fires after every generated token: (tokensDone, tokensRequested).
    /// Invoked on the service queue.
    public var onProgress: ((Int, Int) -> Void)?

    /// Mesh 2.1: fires with each CONFIRMED token as it is generated —
    /// (token, tokensDone, tokensRequested) — so the dashboard can stream
    /// text to every connected browser in real time. Invoked on the
    /// service queue; keep the handler cheap (it sits inside the token
    /// loop between mesh passes).
    public var onToken: ((NMPGeneratedToken, Int, Int) -> Void)?

    // MARK: State

    private let queue = DispatchQueue(label: "nmp.prompt.inference")
    private let orchestrator: NMPInferenceOrchestrator
    private let codec: NMPPromptCodec
    private var busy = false

    /// Which surface owns the generation in flight ("inference" | "chat" |
    /// "benchmark" | "comparison") — set by `run(source:)` ONLY once the
    /// busy guard passes, so a rejected concurrent request never relabels
    /// someone else's stream. Queue-owned: read it from `onToken` /
    /// `onProgress` (both fire on the service queue), nowhere else.
    public private(set) var activeSource = "inference"

    /// Per-token mesh round-trip deadline. Each token is one pass, so this
    /// bounds how long ONE stalled remote stage locks the (sequential)
    /// pipeline before the pass fails and the retry/reshard path recovers.
    /// The orchestrator's own 30 s default is sized for a whole inference,
    /// not a single ~100 ms token: a real device (e.g. an iOS peer iOS is
    /// throttling) that keeps its heartbeat alive but stalls one response
    /// would otherwise hold `busy` — and 429 every other request — for the
    /// full 30 s. Kept generous vs. measured token latency (~90 ms) + link
    /// jitter so a merely-slow peer is retried, not abandoned. Default 30 s
    /// preserves existing (loopback) test timing; the dashboard lowers it
    /// for the radio mesh.
    public var perTokenTimeout: TimeInterval = 30

    /// Hard cap per request: each token is a full mesh pass, so this
    /// bounds worst-case request wall clock.
    public static let maxTokensPerRequest = 128

    /// Reference-engine convenience (Phase 6 behavior, unchanged).
    public convenience init(orchestrator: NMPInferenceOrchestrator, hiddenSize: Int,
                            engineName: String = "reference") {
        self.init(orchestrator: orchestrator,
                  codec: NMPReferencePromptCodec(hiddenSize: hiddenSize,
                                                 engineName: engineName))
    }

    /// Phase 8: any codec — NMPLlamaPromptCodec plugs real llama.cpp text
    /// into the same token loop.
    public init(orchestrator: NMPInferenceOrchestrator, codec: NMPPromptCodec) {
        self.orchestrator = orchestrator
        self.codec = codec
    }

    // MARK: API

    /// Generates up to `maxTokens` tokens for `prompt`. `completion` fires
    /// on the service queue exactly once. `source` labels the generation's
    /// streamed events (see `activeSource`).
    public func run(prompt: String, maxTokens: Int,
                    source: String = "inference",
                    completion: @escaping (Result<GenerationResult, ServiceError>) -> Void) {
        queue.async { [self] in
            guard !busy else {
                completion(.failure(.busy))
                return
            }
            let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                completion(.failure(.emptyPrompt))
                return
            }
            let initialInput: [Float]
            do {
                initialInput = try codec.makeInitialInput(prompt: trimmed)
            } catch {
                completion(.failure(.codec(String(describing: error))))
                return
            }
            busy = true
            activeSource = source
            let state = GenerationState(
                activations: initialInput,
                requested: min(max(maxTokens, 1), Self.maxTokensPerRequest),
                began: DispatchTime.now())
            step(state, completion: completion)
        }
    }

    // MARK: Token loop

    private final class GenerationState {
        var activations: [Float]
        let requested: Int
        let began: DispatchTime
        var tokens: [NMPGeneratedToken] = []
        var perTokenSeconds: [TimeInterval] = []
        var networkPayloadBytes = 0
        var shardCount = 0
        /// Mesh 2.4: a failover re-shard mid-generation fails the pass in
        /// flight (its stage metas were built from the outgoing plan — the
        /// peer rejects the stale range). The pass is idempotent (same
        /// activations in, same tensor out), so retrying against the NEW
        /// plan continues the generation instead of killing it. Budgeted:
        /// a mesh failing for real must still fail the request.
        var passRetriesRemaining = 2
        init(activations: [Float], requested: Int, began: DispatchTime) {
            self.activations = activations
            self.requested = requested
            self.began = began
        }
    }

    private func step(_ state: GenerationState,
                      completion: @escaping (Result<GenerationResult, ServiceError>) -> Void) {
        orchestrator.infer(input: state.activations,
                            stageTimeout: perTokenTimeout) { [weak self] result in
            guard let self else { return }
            self.queue.async {
                switch result {
                case .failure(let error):
                    guard state.passRetriesRemaining > 0 else {
                        self.busy = false
                        completion(.failure(.orchestration(error)))
                        return
                    }
                    state.passRetriesRemaining -= 1
                    // A failed pass often means the plan just changed under us
                    // (a peer dropped/joined → re-shard → fresh, empty per-shard
                    // KV caches). Retrying the incremental decode input would
                    // hit a stale cache; rebuild a from-scratch re-prefill of the
                    // whole sequence so every shard's cache refills consistently.
                    // Codecs without per-shard state return nil → resend as-is.
                    if let reprefill = self.codec.rebuildInput() {
                        state.activations = reprefill
                    }
                    // Brief pause lets an in-progress re-shard finish
                    // (SHARD_ASSIGN round is sub-ms in-process, one RTT
                    // per peer over Wi-Fi).
                    self.queue.asyncAfter(deadline: .now() + 0.3) {
                        self.step(state, completion: completion)
                    }
                case .success(let report):
                    state.perTokenSeconds.append(report.totalSeconds)
                    state.networkPayloadBytes += report.networkPayloadBytes
                    state.shardCount = report.perShard.count

                    let token: NMPGeneratedToken?
                    do {
                        token = try self.codec.extractToken(
                            from: report.output, position: state.tokens.count)
                    } catch {
                        self.busy = false
                        completion(.failure(.codec(String(describing: error))))
                        return
                    }
                    if let token {
                        state.tokens.append(token)
                        self.onToken?(token, state.tokens.count, state.requested)
                    }
                    self.onProgress?(state.tokens.count, state.requested)

                    // Stop on the requested count — or early when the codec
                    // signals end of generation (a real model emitting EOS).
                    if token == nil || state.tokens.count >= state.requested {
                        self.busy = false
                        let total = TimeInterval(
                            DispatchTime.now().uptimeNanoseconds
                                - state.began.uptimeNanoseconds) / 1e9
                        completion(.success(GenerationResult(
                            text: self.codec.render(tokens: state.tokens),
                            tokenCount: state.tokens.count,
                            totalSeconds: total,
                            networkPayloadBytes: state.networkPayloadBytes,
                            shardCount: state.shardCount,
                            perTokenSeconds: state.perTokenSeconds,
                            engine: self.codec.engineName)))
                    } else if let token {
                        // Autoregressive: the codec folds the sampled token
                        // into the next pipeline input (reference: mixed
                        // feedback; llama: token-state request at nextPos).
                        do {
                            state.activations = try self.codec.makeNextInput(
                                after: report.output, token: token,
                                position: state.tokens.count)
                        } catch {
                            self.busy = false
                            completion(.failure(.codec(String(describing: error))))
                            return
                        }
                        self.step(state, completion: completion)
                    }
                }
            }
        }
    }

    // MARK: Embedding / decoding (deterministic, reference engine)
    //
    // Used by NMPReferencePromptCodec (PromptCodec.swift); kept here as
    // statics because the Phase 6 tests pin their exact semantics.

    /// Folds the prompt's words into a hiddenSize activation vector via
    /// splitmix64 — same prompt, same vector, on every platform.
    static func embed(prompt: String, hiddenSize: Int) -> [Float] {
        var vector = [Float](repeating: 0, count: hiddenSize)
        let words = prompt.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber })
        for (position, word) in words.enumerated() {
            var seed: UInt64 = 0x9E37_79B9_7F4A_7C15 &+ UInt64(position)
            for byte in word.utf8 { seed = (seed &* 31) &+ UInt64(byte) }
            var rng = SplitMix64(seed: seed)
            for i in 0..<hiddenSize {
                vector[i] += (rng.nextUnitFloat() * 2 - 1) / Float(max(words.count, 1))
            }
        }
        // Empty-word edge case (e.g. punctuation-only prompt): seed constant.
        if words.isEmpty {
            var rng = SplitMix64(seed: 0xBEEF)
            for i in 0..<hiddenSize { vector[i] = rng.nextUnitFloat() * 2 - 1 }
        }
        return vector
    }

    /// Argmax over the output tensor, folded onto the vocabulary.
    static func sampleToken(from output: [Float]) -> (index: Int, word: String) {
        var bestIndex = 0
        var bestValue = -Float.infinity
        for (i, value) in output.enumerated() where value > bestValue {
            bestValue = value
            bestIndex = i
        }
        let index = bestIndex % vocabulary.count
        return (index, vocabulary[index])
    }

    /// Next-pass input: half hidden state, half the sampled token's
    /// (position-salted) embedding — deterministic, never fixed-point.
    static func feedback(output: [Float], tokenIndex: Int, position: Int) -> [Float] {
        var rng = SplitMix64(seed: 0xA5A5_0000
            &+ UInt64(tokenIndex) &* 0x9E37_79B9
            &+ UInt64(position))
        return output.map { 0.5 * $0 + 0.5 * (rng.nextUnitFloat() * 2 - 1) }
    }

    static func render(words: [String]) -> String {
        guard let first = words.first else { return "" }
        return ([first.prefix(1).uppercased() + first.dropFirst()] + words.dropFirst())
            .joined(separator: " ") + "."
    }

    /// Small fixed vocabulary for the reference engine's decode step. A
    /// real engine binding replaces this with the model's tokenizer.
    static let vocabulary: [String] = [
        "mesh", "peer", "layer", "shard", "token", "packet", "frame", "route",
        "signal", "stream", "vector", "tensor", "weight", "device", "network",
        "latency", "cipher", "relay", "beacon", "bridge", "buffer", "channel",
        "cluster", "compute", "core", "cycle", "data", "decode", "encode",
        "engine", "fabric", "flow", "gateway", "graph", "grid", "handshake",
        "hop", "index", "inference", "kernel", "link", "local", "loop",
        "matrix", "memory", "model", "node", "output", "path", "phase",
        "pipeline", "plan", "protocol", "pulse", "queue", "range", "recover",
        "remote", "reshape", "result", "runtime", "sample", "schedule",
        "segment", "sequence", "session", "socket", "spread", "stage",
        "state", "sync", "system", "thread", "trace", "transform", "transit",
        "transport", "value", "wave", "window", "across", "between", "over",
        "through", "under", "with", "the", "a", "each", "every", "and",
        "then", "now", "fast", "steady", "secure", "stable", "live", "runs",
        "moves", "sends", "binds", "splits", "joins", "maps", "holds",
        "carries", "returns", "settles", "aligns", "balances", "completes",
    ]
}
