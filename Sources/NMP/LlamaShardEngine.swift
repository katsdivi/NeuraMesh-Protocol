//
//  LlamaShardEngine.swift
//  NMP — Phase 10 (TRUE cross-device layer sharding)
//
//  The REAL sharded compute engine behind the NMPShardComputeEngine seam.
//  Where NMPLlamaComputeEngine (LlamaEngine.swift) is bounded by llama.cpp's
//  whole-model-per-decode API — so a llama plan is always ONE full-range
//  shard on one peer that must hold the whole model — this engine drives the
//  ggml graph-surgery shim (LlamaShardRuntime.swift) and genuinely executes
//  a layer SUB-RANGE while loading ONLY that range's weights. That is what
//  lets a model too big for any single device run split across the mesh.
//
//  Placement in the pipeline (decided by start/end vs the model's block
//  count N):
//
//    first  [0, k)   : token-state request  -> residual (n_embd × T)
//    middle [k, m)   : residual             -> residual (n_embd × T)
//    last   [m, N)   : residual             -> top-k logits
//    single [0, N)   : token-state request  -> top-k logits   (one peer)
//
//  Each peer runs its OWN engine and opens ONLY its assigned range: the
//  range is learned from the first runLayers call (SHARD_ASSIGN sets it),
//  the shard is opened once and cached, and a re-shard that changes the
//  range reopens it. No KV cache yet, so the residual carries every
//  position (n_embd × T) — NMPLlamaShardWire grows to fit it, and the mesh
//  must stay on a LOSSLESS activation format (.float32 / .zeroTrimmed) for
//  the hand-off to be bit-exact.
//

import Foundation

public enum NMPLlamaShardEngineError: Error, Sendable {
    /// The GGUF is missing block_count / embedding_length metadata.
    case missingMetadata(String)
    /// The first shard was handed something other than a token-state request.
    case notTokenState
    /// A non-first shard was handed something other than a shard response.
    case notShardResponse
    /// The prompt tokenizes to more than the request wire can carry.
    case promptTooLong(tokens: Int, capacity: Int)
    /// A last-shard decode returned zero candidates.
    case emptyCandidates
}

// MARK: - Compute engine

/// Real sharded llama compute for one peer. Holds the model path and the
/// whole-model shape (from GGUF metadata); opens exactly its assigned
/// `[start, end)` on demand and partial-loads only those blocks.
public final class NMPLlamaShardComputeEngine: NMPShardComputeEngine, NMPGlobalLayerAware {

    /// Candidates the last shard returns per decode (greedy uses [0]).
    public static let maxCandidates = 40

    public let modelPath: String
    public let layerCount: Int
    public let hiddenSize: Int
    /// `general.name` from GGUF (basename fallback) — the SHARD_ASSIGN tag.
    public let modelTag: String
    /// Per-shard KV cache capacity (prompt + generated tokens).
    public let maxContext: Int

    /// Total layers in the plan. Defaults to the model's block count and is
    /// overwritten by SHARD_ASSIGN.totalLayers (see PeerShardEngine), so the
    /// engine agrees with the coordinator on which shard is terminal.
    private let stateLock = NSLock()
    private var _globalLayerCount: Int
    public var globalLayerCount: Int {
        get { stateLock.lock(); defer { stateLock.unlock() }; return _globalLayerCount }
        set { stateLock.lock(); defer { stateLock.unlock() }; _globalLayerCount = newValue }
    }

    /// Cached open shard for the current assigned range (reopened on change).
    private var openShard: NMPLlamaShard?

    public init(modelPath: String, maxContext: Int = 4096) throws {
        let expanded = (modelPath as NSString).expandingTildeInPath
        let gguf = try NMPGGUFModel.load(path: expanded)
        guard let layers = gguf.layerCount else {
            throw NMPLlamaShardEngineError.missingMetadata("block_count")
        }
        guard let hidden = gguf.hiddenSize else {
            throw NMPLlamaShardEngineError.missingMetadata("embedding_length")
        }
        self.modelPath = expanded
        self.layerCount = layers
        self.hiddenSize = hidden
        self._globalLayerCount = layers
        // Cap the cache at the model's trained context when it is smaller.
        self.maxContext = min(maxContext, gguf.contextLength ?? maxContext)
        let metadataName = gguf.modelName
        self.modelTag = (metadataName?.isEmpty == false)
            ? metadataName! : (expanded as NSString).lastPathComponent
    }

    /// Bytes the currently-open shard loaded (0 before the first runLayers) —
    /// the honest proof it holds only its slice of the model.
    public var loadedBytes: Int {
        stateLock.lock(); defer { stateLock.unlock() }
        return openShard?.bytesLoaded ?? 0
    }

    /// Eagerly opens (partial-loads) the shard for [start, end) — the same
    /// shard runLayers will use — so `loadedBytes` reports the real footprint
    /// before the first inference (the dashboard shows it immediately).
    @discardableResult
    public func preload(start: Int, end: Int) throws -> Int {
        try shard(start: start, end: end).bytesLoaded
    }

    /// Returns the cached shard for [start, end), opening (and replacing any
    /// stale one) if the range changed. Serialized by `stateLock`.
    private func shard(start: Int, end: Int) throws -> NMPLlamaShard {
        stateLock.lock()
        defer { stateLock.unlock() }
        if let existing = openShard, existing.start == start, existing.end == end {
            return existing
        }
        let opened = try NMPLlamaShard(modelPath: modelPath, start: start, end: end,
                                       maxCtx: maxContext)
        openShard = opened  // ARC frees the previous shard (and its weights)
        return opened
    }

    public func runLayers(start: Int, end: Int, input: [Float]) throws -> [Float] {
        let n = globalLayerCount
        guard start >= 0, start < end, end <= n else {
            throw NMPComputeError.invalidLayerRange(start: start, end: end, layerCount: n)
        }
        let shard = try shard(start: start, end: end)
        let k = min(Self.maxCandidates, NMPLlamaWire.responseCapacity(width: hiddenSize))

        // The request's basePos IS the KV cache position (n_past): the tokens
        // it carries are the NEW positions to append to the cache.
        // Single shard: token-state request straight to top-k logits.
        if shard.isFirst && shard.isLast {
            let request = try decodeTokenRequest(input)
            let candidates = try shard.evalTokensToTopK(
                request.tokens, nPast: request.basePos, k: k)
            return try encodeLogits(nextPos: request.basePos + request.tokens.count,
                                    candidates: candidates)
        }

        // First shard: token-state request -> residual for the new positions.
        if shard.isFirst {
            let request = try decodeTokenRequest(input)
            let residual = try shard.evalTokensToHidden(
                request.tokens, nPast: request.basePos)
            return try NMPLlamaShardWire.encode(
                NMPLlamaShardWire.ShardResponse(
                    nextPos: request.basePos + request.tokens.count,
                    tokens: request.tokens,
                    hiddenState: residual),
                width: hiddenSize)
        }

        // Non-first shard: input is a residual carried in a shard response.
        // Its positions are [nextPos - tokenCount, nextPos), so n_past is the
        // former — the same cache position the earlier shards used.
        guard NMPLlamaShardWire.isShardResponse(input) else {
            throw NMPLlamaShardEngineError.notShardResponse
        }
        let response = try NMPLlamaShardWire.decodeShardResponse(input)
        let tokenCount = response.tokens.count
        let nPast = response.nextPos - tokenCount

        // Last shard: residual -> top-k logits.
        if shard.isLast {
            let candidates = try shard.evalHiddenToTopK(
                response.hiddenState, tokenCount: tokenCount, nPast: nPast, k: k)
            return try encodeLogits(nextPos: response.nextPos, candidates: candidates)
        }

        // Middle shard: residual -> residual.
        let outResidual = try shard.evalHiddenToHidden(
            response.hiddenState, tokenCount: tokenCount, nPast: nPast)
        return try NMPLlamaShardWire.encode(
            NMPLlamaShardWire.ShardResponse(
                nextPos: response.nextPos,
                tokens: response.tokens,
                hiddenState: outResidual),
            width: hiddenSize)
    }

    // MARK: Wire helpers

    private func decodeTokenRequest(_ input: [Float]) throws -> NMPLlamaWire.Request {
        guard NMPLlamaWire.isRequest(input) else {
            throw NMPLlamaShardEngineError.notTokenState
        }
        return try NMPLlamaWire.decodeRequest(input)
    }

    private func encodeLogits(nextPos: Int,
                              candidates: [(id: Int32, logit: Float)]) throws -> [Float] {
        try NMPLlamaWire.encode(
            NMPLlamaWire.Response(nextPos: nextPos, candidates: candidates),
            width: hiddenSize)
    }
}

// MARK: - Prompt codec

/// Text ↔ token-state translation for a REAL sharded plan. With the shard
/// shim's per-shard KV cache (ABI 2), this is incremental and stateless — the
/// prompt prefills the cache (basePos 0), then each step ships just the newest
/// token at the growing position. `basePos` is the cache length each shard
/// resumes from, so a retried/replayed pass overwrites the same slots and
/// cannot desynchronize the generation.
///
/// Needs only the tokenizer, so it is built over a vocab-only llama model
/// (the coordinator never loads weights).
public final class NMPLlamaShardPromptCodec: NMPPromptCodec {

    public let engineName = "llamaShard"

    private let model: NMPLlamaModel
    private let width: Int
    private let queue = DispatchQueue(label: "nmp.llama.shard.codec")
    /// The full sequence so far (prompt + every accepted token) — kept ONLY
    /// so rebuildInput() can re-prefill all shards after a churn event; the
    /// happy path stays incremental (one token per step).
    private var sequence: [Int32] = []

    public init(model: NMPLlamaModel) {
        self.model = model
        self.width = model.hiddenSize
    }

    public func makeInitialInput(prompt: String) throws -> [Float] {
        let tokens = try model.tokenize(prompt, addSpecial: true)
        let capacity = NMPLlamaWire.requestCapacity(width: width)
        guard tokens.count <= capacity else {
            throw NMPLlamaShardEngineError.promptTooLong(
                tokens: tokens.count, capacity: capacity)
        }
        queue.sync { sequence = tokens }
        // Prefill: the whole prompt at cache position 0.
        return try NMPLlamaWire.encode(
            NMPLlamaWire.Request(basePos: 0, tokens: tokens), width: width)
    }

    public func extractToken(from output: [Float], position: Int) throws -> NMPGeneratedToken? {
        let response = try NMPLlamaWire.decodeResponse(output)
        guard let top = response.top else {
            throw NMPLlamaShardEngineError.emptyCandidates
        }
        if model.isEndOfGeneration(top.id) { return nil }
        let text = String(decoding: try model.pieceBytes(for: top.id), as: UTF8.self)
        return NMPGeneratedToken(index: Int(top.id), text: text)
    }

    public func makeNextInput(after output: [Float], token: NMPGeneratedToken,
                              position: Int) throws -> [Float] {
        queue.sync { sequence.append(Int32(token.index)) }
        // Decode: just the new token at the cache position the shards reached.
        let response = try NMPLlamaWire.decodeResponse(output)
        return try NMPLlamaWire.encode(
            NMPLlamaWire.Request(basePos: response.nextPos, tokens: [Int32(token.index)]),
            width: width)
    }

    /// Recovery after churn: re-prefill EVERY shard with the whole sequence at
    /// cache position 0, so a re-shard (new ranges → fresh empty caches) or a
    /// stale cache refills consistently before the next token.
    public func rebuildInput() -> [Float]? {
        let full = queue.sync { sequence }
        guard !full.isEmpty,
              full.count <= NMPLlamaWire.requestCapacity(width: width) else { return nil }
        return try? NMPLlamaWire.encode(
            NMPLlamaWire.Request(basePos: 0, tokens: full), width: width)
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
