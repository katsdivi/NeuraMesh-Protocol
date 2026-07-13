//
//  LlamaShardRuntime.swift
//  NMP — Phase 10 (TRUE cross-device layer sharding)
//
//  Runtime binding to the ggml graph-surgery shard shim (libnmpshard.dylib,
//  built by scripts/setup_shard.sh). This is the REAL sharding path — unlike
//  the llama.cpp shim (LlamaRuntime.swift), which drives llama.cpp's
//  high-level API and can only run the WHOLE model per decode, this shim
//  builds the transformer forward directly in ggml, so a shard can:
//
//    • execute an arbitrary contiguous block range [start, end), and
//    • partial-load ONLY those blocks' weights (real RAM + compute cut).
//
//  That is what finally lets a model too big for any single device run split
//  across the mesh: Mac holds blocks 0..k, phone holds k..N, neither holds
//  the whole thing. Greedy output is bit-identical to the whole-model run
//  (verified against llama.cpp on Qwen2.5-0.5B — see LlamaShardTests).
//
//  Like the llama shim, the package deliberately does NOT link ggml: the
//  dylib is dlopen'd on demand and every symbol is scalar/pointer-only, so
//  machines without the `ggml` brew formula keep building and testing.
//
//  No KV cache yet: each eval reprocesses the whole sequence (correct but
//  O(n^2)); the residual that crosses the wire is therefore n_embd × T
//  floats, not a single n_embd vector. The KV-cache speed pass shrinks that
//  to n_embd per token later.
//
//  Thread-safety: a shard owns one ggml backend and rebuilds a graph per
//  eval — one serial user at a time. `eval` takes the handle's lock, the
//  same discipline the peer engines already follow (compute on the
//  connection queue, single-flight pipeline).
//

import Foundation

public enum NMPLlamaShardError: Error, Sendable {
    /// No shard shim dylib at $NMP_SHARD_LIB, Vendor/llama/, or ~/.nmp/.
    case libraryNotFound(searched: [String])
    case libraryUnloadable(path: String, reason: String)
    case symbolMissing(String)
    /// Shim ABI newer/older than this binary understands.
    case abiMismatch(found: Int32, expected: Int32)
    /// nmp_shard_open returned NULL (bad path / missing tensors / OOM).
    case openFailed(path: String, start: Int, end: Int)
    /// A shard eval returned a negative status.
    case evalFailed(status: Int32)
    /// eval() was called with arguments that don't match the shard's
    /// position (e.g. tokens for a non-first shard).
    case wrongShardInput(String)
}

// MARK: - Shim binding

/// dlopen/dlsym binding to libnmpshard.dylib. One process-wide instance is
/// plenty (`shared()`); instances are cheap and independent.
public final class NMPLlamaShardRuntime {

    static let expectedABI: Int32 = 1

    // Shim signatures (scalars and pointers only — see nmp_shard_shim.c).
    typealias AbiVersionFn = @convention(c) () -> Int32
    typealias OpenFn = @convention(c) (
        UnsafePointer<CChar>?, Int32, Int32) -> OpaquePointer?
    typealias ArchFn = @convention(c) (
        OpaquePointer?,
        UnsafeMutablePointer<Int32>?, UnsafeMutablePointer<Int32>?,
        UnsafeMutablePointer<Int32>?, UnsafeMutablePointer<Int32>?,
        UnsafeMutablePointer<Int32>?, UnsafeMutablePointer<Int32>?,
        UnsafeMutablePointer<Int32>?) -> Void
    typealias BytesFn = @convention(c) (OpaquePointer?) -> Int
    typealias EvalFn = @convention(c) (
        OpaquePointer?, UnsafePointer<Int32>?, UnsafePointer<Float>?, Int32,
        UnsafeMutablePointer<Float>?, Int32,
        UnsafeMutablePointer<Int32>?, UnsafeMutablePointer<Float>?) -> Int32
    typealias FreeFn = @convention(c) (OpaquePointer?) -> Void

    let openShard: OpenFn
    let arch: ArchFn
    let bytes: BytesFn
    let eval: EvalFn
    let freeShard: FreeFn

    public let libraryPath: String
    private let library: UnsafeMutableRawPointer

    /// Search order for the shard shim dylib (mirrors the llama shim).
    public static func candidatePaths() -> [String] {
        var paths: [String] = []
        if let override = ProcessInfo.processInfo.environment["NMP_SHARD_LIB"] {
            paths.append((override as NSString).expandingTildeInPath)
        }
        paths.append(FileManager.default.currentDirectoryPath
                     + "/Vendor/llama/libnmpshard.dylib")
        paths.append((NSHomeDirectory() as NSString)
                     .appendingPathComponent(".nmp/libnmpshard.dylib"))
        return paths
    }

    /// The first candidate path that exists, or nil — the cue for callers
    /// to fall back to the reference engine / single-peer llama plan.
    public static func locate() -> String? {
        candidatePaths().first { FileManager.default.fileExists(atPath: $0) }
    }

    private static let sharedLock = NSLock()
    private static var sharedInstance: NMPLlamaShardRuntime?

    /// Process-wide runtime (the dylib is never unloaded).
    public static func shared() throws -> NMPLlamaShardRuntime {
        sharedLock.lock()
        defer { sharedLock.unlock() }
        if let sharedInstance { return sharedInstance }
        let runtime = try NMPLlamaShardRuntime()
        sharedInstance = runtime
        return runtime
    }

    public init(libraryPath: String? = nil) throws {
        let path: String
        if let libraryPath {
            path = (libraryPath as NSString).expandingTildeInPath
        } else if let located = Self.locate() {
            path = located
        } else {
            throw NMPLlamaShardError.libraryNotFound(searched: Self.candidatePaths())
        }
        guard let handle = dlopen(path, RTLD_NOW | RTLD_LOCAL) else {
            let reason = dlerror().map { String(cString: $0) } ?? "dlopen failed"
            throw NMPLlamaShardError.libraryUnloadable(path: path, reason: reason)
        }
        self.library = handle
        self.libraryPath = path

        func symbol<T>(_ name: String, as type: T.Type) throws -> T {
            guard let raw = dlsym(handle, name) else {
                throw NMPLlamaShardError.symbolMissing(name)
            }
            return unsafeBitCast(raw, to: T.self)
        }

        let abiVersion = try symbol("nmp_shard_abi_version", as: AbiVersionFn.self)
        let abi = abiVersion()
        guard abi == Self.expectedABI else {
            throw NMPLlamaShardError.abiMismatch(found: abi, expected: Self.expectedABI)
        }

        openShard = try symbol("nmp_shard_open", as: OpenFn.self)
        arch = try symbol("nmp_shard_arch", as: ArchFn.self)
        bytes = try symbol("nmp_shard_bytes", as: BytesFn.self)
        eval = try symbol("nmp_shard_eval", as: EvalFn.self)
        freeShard = try symbol("nmp_shard_free", as: FreeFn.self)
    }
}

// MARK: - Shard handle

/// One opened shard: blocks `[start, end)` of a GGUF model, with ONLY those
/// blocks' weights loaded (plus token_embd on the first shard, output_norm +
/// lm_head on the last). Evaluate a sequence with `eval`.
public final class NMPLlamaShard {

    public let runtime: NMPLlamaShardRuntime
    public let modelPath: String

    /// Total transformer blocks in the whole model (N).
    public let layerCount: Int
    /// Hidden width (n_embd) — the residual carried per token.
    public let hiddenSize: Int
    public let headCount: Int
    public let headCountKV: Int
    public let feedForward: Int
    /// This shard's block range.
    public let start: Int
    public let end: Int
    /// Bytes this shard actually loaded (proof it doesn't hold the whole model).
    public let bytesLoaded: Int

    /// First shard reads token ids and runs token_embd.
    public var isFirst: Bool { start == 0 }
    /// Last shard runs output_norm + lm_head and returns logits.
    public var isLast: Bool { end == layerCount }

    private let handle: OpaquePointer
    private let lock = NSLock()

    /// Opens blocks [start, end) of the model, partial-loading their weights.
    /// `end < 0` (or > N) is clamped to the model's block count.
    public init(modelPath: String, start: Int, end: Int,
                runtime: NMPLlamaShardRuntime? = nil) throws {
        self.runtime = try runtime ?? NMPLlamaShardRuntime.shared()
        let expanded = (modelPath as NSString).expandingTildeInPath
        self.modelPath = expanded

        guard let opened = self.runtime.openShard(expanded, Int32(start), Int32(end)) else {
            throw NMPLlamaShardError.openFailed(path: expanded, start: start, end: end)
        }
        self.handle = opened

        var nLayer: Int32 = 0, nEmbd: Int32 = 0, nHead: Int32 = 0
        var nHeadKV: Int32 = 0, nFF: Int32 = 0, s: Int32 = 0, e: Int32 = 0
        self.runtime.arch(opened, &nLayer, &nEmbd, &nHead, &nHeadKV, &nFF, &s, &e)
        self.layerCount = Int(nLayer)
        self.hiddenSize = Int(nEmbd)
        self.headCount = Int(nHead)
        self.headCountKV = Int(nHeadKV)
        self.feedForward = Int(nFF)
        self.start = Int(s)
        self.end = Int(e)
        self.bytesLoaded = self.runtime.bytes(opened)
    }

    deinit {
        runtime.freeShard(handle)
    }

    // MARK: Evaluation

    /// First shard: run `tokens` (the whole sequence, since there is no KV
    /// cache) through token_embd + blocks [0, end), returning the residual
    /// hidden state for every position — `hiddenSize × tokens.count` floats,
    /// laid out position-major (position 0's n_embd floats, then position 1's,
    /// …). Precondition: `isFirst && !isLast`.
    public func evalTokensToHidden(_ tokens: [Int32]) throws -> [Float] {
        precondition(isFirst && !isLast, "evalTokensToHidden on a non-first or terminal shard")
        let count = tokens.count
        var out = [Float](repeating: 0, count: max(1, count * hiddenSize))
        try withLockedEval { eval, h in
            tokens.withUnsafeBufferPointer { tp in
                out.withUnsafeMutableBufferPointer { op in
                    eval(h, tp.baseAddress, nil, Int32(count), op.baseAddress, 0, nil, nil)
                }
            }
        }
        return out
    }

    /// Middle shard: run an incoming residual (`hiddenSize × tokenCount`
    /// floats) through blocks [start, end), returning the outgoing residual
    /// of the same shape. Precondition: `!isFirst && !isLast`.
    public func evalHiddenToHidden(_ hidden: [Float], tokenCount: Int) throws -> [Float] {
        precondition(!isFirst && !isLast, "evalHiddenToHidden on a boundary shard")
        try requireHiddenWidth(hidden, tokenCount: tokenCount)
        var out = [Float](repeating: 0, count: max(1, tokenCount * hiddenSize))
        try withLockedEval { eval, h in
            hidden.withUnsafeBufferPointer { ip in
                out.withUnsafeMutableBufferPointer { op in
                    eval(h, nil, ip.baseAddress, Int32(tokenCount), op.baseAddress, 0, nil, nil)
                }
            }
        }
        return out
    }

    /// Last shard: run an incoming residual through blocks [start, end),
    /// then output_norm + lm_head on the final position, returning the top-`k`
    /// (token id, logit) pairs sorted by logit descending. Precondition:
    /// `!isFirst && isLast`.
    public func evalHiddenToTopK(_ hidden: [Float], tokenCount: Int, k: Int)
        throws -> [(id: Int32, logit: Float)] {
        precondition(!isFirst && isLast, "evalHiddenToTopK on a first or non-terminal shard")
        try requireHiddenWidth(hidden, tokenCount: tokenCount)
        let kk = max(1, k)
        var ids = [Int32](repeating: 0, count: kk)
        var logits = [Float](repeating: 0, count: kk)
        try withLockedEval { eval, h in
            hidden.withUnsafeBufferPointer { ip in
                ids.withUnsafeMutableBufferPointer { idp in
                    logits.withUnsafeMutableBufferPointer { lp in
                        eval(h, nil, ip.baseAddress, Int32(tokenCount), nil,
                             Int32(kk), idp.baseAddress, lp.baseAddress)
                    }
                }
            }
        }
        return (0..<kk).map { (ids[$0], logits[$0]) }
    }

    /// Single shard (start == 0 && end == N): run the WHOLE model over
    /// `tokens` and return the top-`k` (id, logit) pairs at the last position.
    /// Precondition: `isFirst && isLast`.
    public func evalTokensToTopK(_ tokens: [Int32], k: Int)
        throws -> [(id: Int32, logit: Float)] {
        precondition(isFirst && isLast, "evalTokensToTopK on a split shard")
        let kk = max(1, k)
        var ids = [Int32](repeating: 0, count: kk)
        var logits = [Float](repeating: 0, count: kk)
        try withLockedEval { eval, h in
            tokens.withUnsafeBufferPointer { tp in
                ids.withUnsafeMutableBufferPointer { idp in
                    logits.withUnsafeMutableBufferPointer { lp in
                        eval(h, tp.baseAddress, nil, Int32(tokens.count), nil,
                             Int32(kk), idp.baseAddress, lp.baseAddress)
                    }
                }
            }
        }
        return (0..<kk).map { (ids[$0], logits[$0]) }
    }

    // MARK: Internals

    private func requireHiddenWidth(_ hidden: [Float], tokenCount: Int) throws {
        guard tokenCount > 0, hidden.count == tokenCount * hiddenSize else {
            throw NMPLlamaShardError.wrongShardInput(
                "residual \(hidden.count) floats != tokenCount \(tokenCount) × n_embd \(hiddenSize)")
        }
    }

    /// Serializes access to the ggml backend and maps a negative shim status
    /// onto an error.
    private func withLockedEval(
        _ body: (_ eval: NMPLlamaShardRuntime.EvalFn, _ handle: OpaquePointer) -> Int32
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        let status = body(runtime.eval, handle)
        guard status == 0 else {
            throw NMPLlamaShardError.evalFailed(status: status)
        }
    }
}
