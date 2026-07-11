//
//  SpeculativeDecoder.swift
//  NMP — Phase 9
//
//  Speculative decoding over the mesh. Phase 8's cost model: every token
//  is one full mesh round trip (~68 ms/token for 7B over UDP — mostly
//  peer compute, but one RTT + protocol overhead each). Phase 9 changes
//  the exchange rate: the coordinator DRAFTS d tokens cheaply and the
//  peer verifies the whole draft in ONE round trip.
//
//    round trip carries: [T, D₁ … D_d]  (T = last confirmed token)
//    peer returns: greedy argmax after every position (LlamaWire verify)
//    accept: longest prefix where Dᵢ == argmax after Dᵢ₋₁, plus one
//            BONUS token (the model's own prediction after the last
//            accepted draft) — up to d+1 tokens per round trip.
//
//  DETERMINISM: a draft is accepted only where it EQUALS the target
//  model's greedy argmax, and the bonus token IS the target's argmax —
//  so the emitted stream is token-for-token identical to Phase 8's
//  non-speculative greedy stream, no matter how bad the drafter is. A
//  wrong drafter costs round trips, never correctness.
//
//  Rejection is free: the next request's basePos simply trims the peer's
//  KV cache back to the last accepted position — the same idempotent
//  rewind Phase 8 built for loss-recovery retries.
//
//  Drafters:
//  - NMPPromptLookupDrafter (default): n-gram continuation lookup over
//    the generation's own context — no second model, no extra RAM
//    (llama.cpp ships the same idea as its "lookup" example). Shines on
//    repetitive/structured text, degrades to plain Phase 8 elsewhere.
//  - NMPLlamaDraftModelDrafter: a small same-vocabulary GGUF on the
//    coordinator (e.g. TinyLlama-1.1B drafting for Llama-2-7B) decodes
//    drafts greedily on local compute.
//
//  Threading: callback style on a private serial queue (spec rule: no
//  async/await), single-flight like NMPPromptInferenceService.
//

import Foundation

// MARK: - Tokenizer seam

/// The tokenizer surface the speculative service needs. NMPLlamaModel
/// satisfies it (vocab-only is enough); tests drive the service with a
/// toy tokenizer over a deterministic fake engine — no GGUF required.
public protocol NMPLlamaTokenizing: AnyObject {
    /// Wire tensor width (the model's hidden size for llama plans).
    var hiddenSize: Int { get }
    func tokenize(_ text: String, addSpecial: Bool) throws -> [Int32]
    func pieceBytes(for token: Int32) throws -> Data
    func isEndOfGeneration(_ token: Int32) -> Bool
}

extension NMPLlamaModel: NMPLlamaTokenizing {}

// MARK: - Drafter seam

public protocol NMPSpeculativeDrafter: AnyObject {
    /// Human-readable identity for logs/API responses.
    var drafterName: String { get }
    /// Proposes up to `count` continuation tokens for `context` (prompt +
    /// everything generated so far). Empty = no idea; the service falls
    /// back to a plain single-token round trip.
    func draft(context: [Int32], count: Int) -> [Int32]
}

// MARK: - Prompt-lookup drafter

/// Drafts by n-gram continuation: find the most recent earlier occurrence
/// of the context's tail n-gram and propose the tokens that followed it.
public final class NMPPromptLookupDrafter: NMPSpeculativeDrafter {

    public let drafterName = "prompt-lookup"
    private let maxNgram: Int
    private let minNgram: Int

    /// Longer n-grams are tried first (more specific match, higher
    /// acceptance); 2 as the floor avoids drafting off single-token
    /// coincidences.
    public init(maxNgram: Int = 3, minNgram: Int = 2) {
        precondition(minNgram >= 1 && maxNgram >= minNgram)
        self.maxNgram = maxNgram
        self.minNgram = minNgram
    }

    public func draft(context: [Int32], count: Int) -> [Int32] {
        guard count > 0 else { return [] }
        for n in stride(from: maxNgram, through: minNgram, by: -1) {
            guard context.count > n else { continue }
            let tail = Array(context.suffix(n))
            // Most recent earlier occurrence (recency ≈ topical relevance).
            var start = context.count - n - 1
            while start >= 0 {
                if Array(context[start..<start + n]) == tail {
                    let from = start + n
                    let to = min(from + count, context.count)
                    return Array(context[from..<to])
                }
                start -= 1
            }
        }
        return []
    }
}

// MARK: - Draft-model drafter

/// Greedy drafts from a small local model sharing the target's
/// vocabulary. The draft model keeps its own KV cache; a diverging
/// context is rewound via the same basePos trim the peer uses.
public final class NMPLlamaDraftModelDrafter: NMPSpeculativeDrafter {

    public let drafterName: String
    public let model: NMPLlamaModel
    /// Tokens the draft model's KV cache currently covers.
    private var cachedTokens: [Int32] = []

    /// - Parameter model: full (weights-loaded) handle of the DRAFT model.
    public init(model: NMPLlamaModel) throws {
        guard model.hasWeights else { throw NMPLlamaRuntimeError.weightsNotLoaded }
        self.model = model
        self.drafterName = "draft-model(\(model.name))"
    }

    public func draft(context: [Int32], count: Int) -> [Int32] {
        guard count > 0, !context.isEmpty else { return [] }

        // Reuse the KV prefix that still matches; re-decode the rest.
        var prefix = 0
        while prefix < min(cachedTokens.count, context.count),
              cachedTokens[prefix] == context[prefix] {
            prefix += 1
        }
        if prefix == context.count { prefix = context.count - 1 }

        var drafts: [Int32] = []
        var position = prefix
        var pending = Array(context[prefix...])
        do {
            for _ in 0..<count {
                guard let top = try model.decodeTopK(
                    tokens: pending, basePos: position, k: 1).first else { break }
                if model.isEndOfGeneration(top.id) { break }
                drafts.append(top.id)
                position += pending.count
                pending = [top.id]
            }
        } catch {
            cachedTokens = []
            return drafts
        }
        // The last draft was proposed but never decoded into the KV cache.
        cachedTokens = context + drafts.dropLast()
        return drafts
    }
}

// MARK: - Stats

/// What speculation actually bought during one generation.
public struct NMPSpeculationStats: Sendable {
    /// Mesh round trips spent (Phase 8 spends exactly `tokens` of them).
    public var meshRoundTrips = 0
    /// Draft tokens proposed / accepted across all rounds.
    public var draftedTokens = 0
    public var acceptedDraftTokens = 0
    /// Rounds where the drafter had no proposal (plain Phase 8 step).
    public var fallbackRounds = 0
    public var drafterName = ""

    public var acceptanceRate: Double {
        draftedTokens > 0 ? Double(acceptedDraftTokens) / Double(draftedTokens) : 0
    }

    /// Tokens emitted per round trip, the effective speedup over the
    /// one-token-per-round-trip Phase 8 loop.
    public func tokensPerRoundTrip(tokenCount: Int) -> Double {
        meshRoundTrips > 0 ? Double(tokenCount) / Double(meshRoundTrips) : 0
    }

    public init() {}
}

// MARK: - Speculative generation service

/// The llama token loop with draft/verify round trips. Mirrors
/// NMPPromptInferenceService's surface so the dashboard can swap between
/// them per request.
public final class NMPSpeculativeGenerationService {

    public typealias ServiceError = NMPPromptInferenceService.ServiceError

    /// Fires after every accepted token: (tokensDone, tokensRequested).
    public var onProgress: ((Int, Int) -> Void)?

    /// Mesh 2.1: fires with each CONFIRMED token (pending or accepted
    /// draft) — same contract as NMPPromptInferenceService.onToken, so
    /// the dashboard streams speculative and plain runs identically.
    public var onToken: ((NMPGeneratedToken, Int, Int) -> Void)?

    public static let maxTokensPerRequest = NMPPromptInferenceService.maxTokensPerRequest
    /// Recommended default (Part G answer #1).
    public static let defaultDepth = 4

    private let queue = DispatchQueue(label: "nmp.speculative.inference")
    private let orchestrator: NMPInferenceOrchestrator
    /// Tokenizer-side handle (vocab-only is enough).
    private let model: NMPLlamaTokenizing
    private let drafter: NMPSpeculativeDrafter
    private let depth: Int
    private let width: Int
    private var busy = false

    public init(orchestrator: NMPInferenceOrchestrator,
                model: NMPLlamaTokenizing,
                drafter: NMPSpeculativeDrafter = NMPPromptLookupDrafter(),
                depth: Int = NMPSpeculativeGenerationService.defaultDepth) {
        self.orchestrator = orchestrator
        self.model = model
        self.drafter = drafter
        self.depth = max(1, depth)
        self.width = model.hiddenSize
    }

    // MARK: API

    public func run(
        prompt: String, maxTokens: Int,
        completion: @escaping (Result<NMPPromptInferenceService.GenerationResult,
                                      ServiceError>) -> Void
    ) {
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
            let promptTokens: [Int32]
            do {
                promptTokens = try model.tokenize(trimmed, addSpecial: true)
                guard promptTokens.count <= NMPLlamaWire.requestCapacity(width: width) else {
                    throw NMPLlamaEngineError.promptTooLong(
                        tokens: promptTokens.count,
                        capacity: NMPLlamaWire.requestCapacity(width: width))
                }
            } catch {
                completion(.failure(.codec(String(describing: error))))
                return
            }
            busy = true
            var stats = NMPSpeculationStats()
            stats.drafterName = drafter.drafterName
            let state = State(
                promptTokens: promptTokens,
                requested: min(max(maxTokens, 1), Self.maxTokensPerRequest),
                stats: stats)

            // Prompt pass: a plain Phase 8 request — its argmax is the
            // first pending token.
            let initial: [Float]
            do {
                initial = try NMPLlamaWire.encode(
                    NMPLlamaWire.Request(basePos: 0, tokens: promptTokens),
                    width: width)
            } catch {
                busy = false
                completion(.failure(.codec(String(describing: error))))
                return
            }
            infer(initial, state: state, completion: completion) { [self] output, state in
                let response = try NMPLlamaWire.decodeResponse(output)
                guard let top = response.top else {
                    throw NMPLlamaEngineError.emptyCandidates
                }
                state.nextPos = response.nextPos
                state.pendingToken = top.id
                round(state, completion: completion)
            }
        }
    }

    /// Blocking wrapper (call from a plain thread, never a mesh queue).
    public func runSync(
        prompt: String, maxTokens: Int, timeout: TimeInterval = 600
    ) throws -> NMPPromptInferenceService.GenerationResult {
        let done = DispatchSemaphore(value: 0)
        var outcome: Result<NMPPromptInferenceService.GenerationResult, ServiceError>?
        run(prompt: prompt, maxTokens: maxTokens) { result in
            outcome = result
            done.signal()
        }
        guard done.wait(timeout: .now() + timeout) == .success, let outcome else {
            throw ServiceError.busy
        }
        return try outcome.get()
    }

    // MARK: Token loop (service queue)

    private final class State {
        let promptTokens: [Int32]
        let requested: Int
        let began = DispatchTime.now()
        var tokens: [NMPGeneratedToken] = []
        /// Target-model argmax awaiting emission (certain, not a draft).
        var pendingToken: Int32 = 0
        /// Peer KV position the next request continues from.
        var nextPos = 0
        var perRoundSeconds: [TimeInterval] = []
        var networkPayloadBytes = 0
        var shardCount = 0
        var stats: NMPSpeculationStats
        init(promptTokens: [Int32], requested: Int, stats: NMPSpeculationStats) {
            self.promptTokens = promptTokens
            self.requested = requested
            self.stats = stats
        }
    }

    private func round(
        _ state: State,
        completion: @escaping (Result<NMPPromptInferenceService.GenerationResult,
                                      ServiceError>) -> Void
    ) {
        // Emit the pending (certain) token.
        if !emit(state.pendingToken, state: state) {
            finish(state, completion: completion)
            return
        }
        guard state.tokens.count < state.requested else {
            finish(state, completion: completion)
            return
        }

        let remaining = state.requested - state.tokens.count
        let context = state.promptTokens + state.tokens.map { Int32($0.index) }
        let budget = min(depth, remaining,
                         NMPLlamaWire.verifyCapacity(width: width) - 1,
                         NMPLlamaWire.requestCapacity(width: width) - 1)
        let drafts = budget > 0 ? drafter.draft(context: context, count: budget) : []
        state.stats.draftedTokens += drafts.count

        do {
            if drafts.isEmpty {
                // Plain Phase 8 step: decode the pending token, sample next.
                state.stats.fallbackRounds += 1
                let request = try NMPLlamaWire.encode(
                    NMPLlamaWire.Request(basePos: state.nextPos,
                                         tokens: [state.pendingToken]),
                    width: width)
                infer(request, state: state, completion: completion) { [self] output, state in
                    let response = try NMPLlamaWire.decodeResponse(output)
                    guard let top = response.top else {
                        throw NMPLlamaEngineError.emptyCandidates
                    }
                    state.nextPos = response.nextPos
                    state.pendingToken = top.id
                    round(state, completion: completion)
                }
            } else {
                // Draft/verify: T + drafts in one round trip.
                let basePos = state.nextPos
                let request = try NMPLlamaWire.encodeVerify(
                    NMPLlamaWire.Request(basePos: basePos,
                                         tokens: [state.pendingToken] + drafts),
                    width: width)
                infer(request, state: state, completion: completion) { [self] output, state in
                    let response = try NMPLlamaWire.decodeVerifyResponse(output)
                    guard response.verdicts.count == drafts.count + 1 else {
                        throw NMPLlamaWireError.malformed(
                            "expected \(drafts.count + 1) verdicts, got "
                            + "\(response.verdicts.count)")
                    }
                    // verdicts[i] = target argmax after request token i.
                    var accepted = 0
                    var ended = false
                    while accepted < drafts.count,
                          drafts[accepted] == response.verdicts[accepted].id {
                        if !emit(drafts[accepted], state: state) {
                            ended = true
                            break
                        }
                        accepted += 1
                        if state.tokens.count >= state.requested { break }
                    }
                    state.stats.acceptedDraftTokens += accepted
                    // Confirmed decodes: T + accepted drafts. Anything the
                    // peer decoded beyond that is trimmed by basePos next
                    // round.
                    state.nextPos = basePos + 1 + accepted
                    if ended || state.tokens.count >= state.requested {
                        finish(state, completion: completion)
                        return
                    }
                    // Bonus token: the model's own argmax after the last
                    // accepted token — certain, becomes the next pending.
                    state.pendingToken = response.verdicts[accepted].id
                    round(state, completion: completion)
                }
            }
        } catch {
            busy = false
            completion(.failure(.codec(String(describing: error))))
        }
    }

    /// Appends one CONFIRMED token. false = EOS (do not append, stop).
    private func emit(_ token: Int32, state: State) -> Bool {
        if model.isEndOfGeneration(token) { return false }
        let text = (try? model.pieceBytes(for: token))
            .map { String(decoding: $0, as: UTF8.self) } ?? ""
        let generated = NMPGeneratedToken(index: Int(token), text: text)
        state.tokens.append(generated)
        onToken?(generated, state.tokens.count, state.requested)
        onProgress?(state.tokens.count, state.requested)
        return true
    }

    /// One mesh round trip; `handle` runs on the service queue and may
    /// throw codec errors.
    private func infer(
        _ input: [Float], state: State,
        completion: @escaping (Result<NMPPromptInferenceService.GenerationResult,
                                      ServiceError>) -> Void,
        handle: @escaping ([Float], State) throws -> Void
    ) {
        orchestrator.infer(input: input) { [weak self] result in
            guard let self else { return }
            self.queue.async {
                switch result {
                case .failure(let error):
                    self.busy = false
                    completion(.failure(.orchestration(error)))
                case .success(let report):
                    state.stats.meshRoundTrips += 1
                    state.perRoundSeconds.append(report.totalSeconds)
                    state.networkPayloadBytes += report.networkPayloadBytes
                    state.shardCount = report.perShard.count
                    do {
                        try handle(report.output, state)
                    } catch {
                        self.busy = false
                        completion(.failure(.codec(String(describing: error))))
                    }
                }
            }
        }
    }

    private func finish(
        _ state: State,
        completion: (Result<NMPPromptInferenceService.GenerationResult,
                            ServiceError>) -> Void
    ) {
        busy = false
        let total = TimeInterval(
            DispatchTime.now().uptimeNanoseconds
                - state.began.uptimeNanoseconds) / 1e9
        var bytes = Data()
        for token in state.tokens {
            if let piece = try? model.pieceBytes(for: Int32(token.index)) {
                bytes.append(piece)
            }
        }
        let text = String(decoding: bytes, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        completion(.success(NMPPromptInferenceService.GenerationResult(
            text: text,
            tokenCount: state.tokens.count,
            totalSeconds: total,
            networkPayloadBytes: state.networkPayloadBytes,
            shardCount: state.shardCount,
            perTokenSeconds: state.perRoundSeconds,
            engine: "llamaCpp+speculative",
            speculation: state.stats)))
    }
}
