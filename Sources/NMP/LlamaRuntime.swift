//
//  LlamaRuntime.swift
//  NMP — Phase 8
//
//  Runtime binding to the llama.cpp shim (libnmpllama.dylib, built by
//  scripts/setup_llama.sh). The package deliberately does NOT link
//  llama.cpp: the shim is dlopen'd on demand, every symbol has a
//  scalar/pointer-only signature (the shim absorbs llama.cpp's unstable
//  struct ABI), and machines without llama.cpp keep building and testing
//  with the reference engine.
//
//  NMPLlamaModel wraps one shim handle (model + context). Vocab-only
//  handles (coordinator side) can tokenize/detokenize but not decode;
//  full handles (the peer that owns the weights) do real forward passes.
//
//  Thread-safety: a handle must be used from one serial queue at a time —
//  the same discipline the rest of the mesh already follows (peer engines
//  compute on their connection queue; the pipeline is single-flight).
//

import Foundation

public enum NMPLlamaRuntimeError: Error, Sendable {
    /// No shim dylib at $NMP_LLAMA_LIB, Vendor/llama/, or ~/.nmp/.
    case libraryNotFound(searched: [String])
    case libraryUnloadable(path: String, reason: String)
    case symbolMissing(String)
    /// Shim ABI newer/older than this binary understands.
    case abiMismatch(found: Int32, expected: Int32)
    case openFailed(path: String, reason: String)
    /// A shim call returned a negative NMP_LLAMA_ERR_* status.
    case callFailed(function: String, status: Int32)
    /// Decode on a vocab-only handle (no weights loaded).
    case weightsNotLoaded
}

// MARK: - Shim binding

/// dlopen/dlsym binding to libnmpllama.dylib. One instance per process is
/// plenty (`shared()`), but instances are cheap and independent.
public final class NMPLlamaRuntime {

    static let expectedABI: Int32 = 1

    // Shim signatures (scalars and pointers only — see nmp_llama_shim.c).
    typealias AbiVersionFn = @convention(c) () -> Int32
    typealias OpenFn = @convention(c) (
        UnsafePointer<CChar>?, Int32, Int32, Int32,
        UnsafeMutablePointer<CChar>?, Int32) -> OpaquePointer?
    typealias CloseFn = @convention(c) (OpaquePointer?) -> Void
    typealias IntPropertyFn = @convention(c) (OpaquePointer?) -> Int32
    typealias NameFn = @convention(c) (
        OpaquePointer?, UnsafeMutablePointer<CChar>?, Int32) -> Int32
    typealias TokenizeFn = @convention(c) (
        OpaquePointer?, UnsafePointer<CChar>?, Int32,
        UnsafeMutablePointer<Int32>?, Int32) -> Int32
    typealias TokenTextFn = @convention(c) (
        OpaquePointer?, Int32, UnsafeMutablePointer<CChar>?, Int32) -> Int32
    typealias TokenFlagFn = @convention(c) (OpaquePointer?, Int32) -> Int32
    typealias DecodeTopKFn = @convention(c) (
        OpaquePointer?, UnsafePointer<Int32>?, Int32, Int32, Int32,
        UnsafeMutablePointer<Int32>?, UnsafeMutablePointer<Float>?) -> Int32

    let open: OpenFn
    let close: CloseFn
    let nLayer: IntPropertyFn
    let nEmbd: IntPropertyFn
    let nVocab: IntPropertyFn
    let nCtx: IntPropertyFn
    let hasWeights: IntPropertyFn
    let modelName: NameFn
    let tokenize: TokenizeFn
    let tokenText: TokenTextFn
    let tokenIsEOG: TokenFlagFn
    let decodeTopK: DecodeTopKFn

    public let libraryPath: String
    private let library: UnsafeMutableRawPointer

    /// Search order for the shim dylib.
    public static func candidatePaths() -> [String] {
        var paths: [String] = []
        if let override = ProcessInfo.processInfo.environment["NMP_LLAMA_LIB"] {
            paths.append((override as NSString).expandingTildeInPath)
        }
        paths.append(FileManager.default.currentDirectoryPath
                     + "/Vendor/llama/libnmpllama.dylib")
        paths.append((NSHomeDirectory() as NSString)
                     .appendingPathComponent(".nmp/libnmpllama.dylib"))
        return paths
    }

    /// The first candidate path that exists, or nil — the cue for callers
    /// to fall back to the reference engine.
    public static func locate() -> String? {
        candidatePaths().first { FileManager.default.fileExists(atPath: $0) }
    }

    private static let sharedLock = NSLock()
    private static var sharedInstance: NMPLlamaRuntime?

    /// Process-wide runtime (the dylib is never unloaded anyway).
    public static func shared() throws -> NMPLlamaRuntime {
        sharedLock.lock()
        defer { sharedLock.unlock() }
        if let sharedInstance { return sharedInstance }
        let runtime = try NMPLlamaRuntime()
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
            throw NMPLlamaRuntimeError.libraryNotFound(searched: Self.candidatePaths())
        }
        guard let handle = dlopen(path, RTLD_NOW | RTLD_LOCAL) else {
            let reason = dlerror().map { String(cString: $0) } ?? "dlopen failed"
            throw NMPLlamaRuntimeError.libraryUnloadable(path: path, reason: reason)
        }
        self.library = handle
        self.libraryPath = path

        func symbol<T>(_ name: String, as type: T.Type) throws -> T {
            guard let raw = dlsym(handle, name) else {
                throw NMPLlamaRuntimeError.symbolMissing(name)
            }
            return unsafeBitCast(raw, to: T.self)
        }

        let abiVersion = try symbol("nmp_llama_abi_version", as: AbiVersionFn.self)
        let abi = abiVersion()
        guard abi == Self.expectedABI else {
            throw NMPLlamaRuntimeError.abiMismatch(found: abi, expected: Self.expectedABI)
        }

        open = try symbol("nmp_llama_open", as: OpenFn.self)
        close = try symbol("nmp_llama_close", as: CloseFn.self)
        nLayer = try symbol("nmp_llama_n_layer", as: IntPropertyFn.self)
        nEmbd = try symbol("nmp_llama_n_embd", as: IntPropertyFn.self)
        nVocab = try symbol("nmp_llama_n_vocab", as: IntPropertyFn.self)
        nCtx = try symbol("nmp_llama_n_ctx", as: IntPropertyFn.self)
        hasWeights = try symbol("nmp_llama_has_weights", as: IntPropertyFn.self)
        modelName = try symbol("nmp_llama_model_name", as: NameFn.self)
        tokenize = try symbol("nmp_llama_tokenize", as: TokenizeFn.self)
        tokenText = try symbol("nmp_llama_token_text", as: TokenTextFn.self)
        tokenIsEOG = try symbol("nmp_llama_token_is_eog", as: TokenFlagFn.self)
        decodeTopK = try symbol("nmp_llama_decode_topk", as: DecodeTopKFn.self)
    }
}

// MARK: - Model handle

/// One loaded GGUF model (weights + context, or tokenizer-only).
public final class NMPLlamaModel {

    public let runtime: NMPLlamaRuntime
    public let modelPath: String
    public let isVocabOnly: Bool

    public let layerCount: Int
    public let hiddenSize: Int
    public let vocabSize: Int
    /// 0 in vocab-only mode.
    public let contextSize: Int
    /// `general.name` from the GGUF metadata (path basename fallback).
    public let name: String

    private let handle: OpaquePointer

    /// - Parameters:
    ///   - gpuLayers: layers offloaded to Metal; -1 = all (llama.cpp default).
    ///   - contextLength: 0 = the model's training context.
    ///   - vocabOnly: load tokenizer + metadata without weights — the
    ///     coordinator-side mode (a few MB instead of the full model).
    public init(modelPath: String, gpuLayers: Int32 = -1, contextLength: Int32 = 0,
                vocabOnly: Bool = false, runtime: NMPLlamaRuntime? = nil) throws {
        self.runtime = try runtime ?? NMPLlamaRuntime.shared()
        let expanded = (modelPath as NSString).expandingTildeInPath
        self.modelPath = expanded
        self.isVocabOnly = vocabOnly

        var errorBuffer = [CChar](repeating: 0, count: 256)
        guard let opened = self.runtime.open(
            expanded, gpuLayers, contextLength, vocabOnly ? 1 : 0,
            &errorBuffer, Int32(errorBuffer.count)) else {
            throw NMPLlamaRuntimeError.openFailed(
                path: expanded, reason: String(cString: errorBuffer))
        }
        handle = opened

        var layers = Int(self.runtime.nLayer(handle))
        var hidden = Int(self.runtime.nEmbd(handle))
        // vocab_only loads skip hparams in llama.cpp, reporting 0 layers ×
        // 0 hidden — but the coordinator needs the real shape to size
        // SHARD_ASSIGN. The Phase 5 GGUF parser reads it straight from the
        // container metadata (mmap'd, no tensor data touched).
        if vocabOnly, layers <= 0 || hidden <= 0,
           let gguf = try? NMPGGUFModel.load(path: expanded) {
            layers = gguf.layerCount ?? layers
            hidden = gguf.hiddenSize ?? hidden
        }
        layerCount = layers
        hiddenSize = hidden
        vocabSize = Int(self.runtime.nVocab(handle))
        contextSize = Int(self.runtime.nCtx(handle))

        var nameBuffer = [CChar](repeating: 0, count: 256)
        let nameLength = self.runtime.modelName(handle, &nameBuffer, Int32(nameBuffer.count))
        if nameLength > 0 {
            name = String(cString: nameBuffer)
        } else {
            name = (expanded as NSString).lastPathComponent
        }
    }

    deinit {
        runtime.close(handle)
    }

    public var hasWeights: Bool {
        runtime.hasWeights(handle) == 1
    }

    // MARK: Tokenizer

    public func tokenize(_ text: String, addSpecial: Bool = true) throws -> [Int32] {
        var tokens = [Int32](repeating: 0, count: text.utf8.count + 16)
        let count = runtime.tokenize(handle, text, addSpecial ? 1 : 0,
                                     &tokens, Int32(tokens.count))
        guard count >= 0 else {
            throw NMPLlamaRuntimeError.callFailed(function: "tokenize", status: count)
        }
        return Array(tokens.prefix(Int(count)))
    }

    /// Raw UTF-8 bytes of one token's piece. Pieces may split multi-byte
    /// characters — join bytes across tokens before decoding final text.
    public func pieceBytes(for token: Int32) throws -> Data {
        var buffer = [CChar](repeating: 0, count: 128)
        let written = runtime.tokenText(handle, token, &buffer, Int32(buffer.count))
        guard written >= 0 else {
            throw NMPLlamaRuntimeError.callFailed(function: "token_text", status: written)
        }
        return buffer.prefix(Int(written)).withUnsafeBytes { Data($0) }
    }

    public func isEndOfGeneration(_ token: Int32) -> Bool {
        runtime.tokenIsEOG(handle, token) == 1
    }

    // MARK: Forward pass

    /// One REAL forward pass over the whole model: trims the KV cache to
    /// `basePos`, decodes `tokens` at positions basePos…, and returns the
    /// top-k (token id, logit) pairs of the last position's logits,
    /// sorted by logit descending. Requires weights.
    public func decodeTopK(tokens: [Int32], basePos: Int,
                           k: Int) throws -> [(id: Int32, logit: Float)] {
        guard hasWeights else { throw NMPLlamaRuntimeError.weightsNotLoaded }
        let count = max(1, min(k, vocabSize))
        var ids = [Int32](repeating: 0, count: count)
        var logits = [Float](repeating: 0, count: count)
        let status = runtime.decodeTopK(handle, tokens, Int32(tokens.count),
                                        Int32(basePos), Int32(count), &ids, &logits)
        guard status > 0 else {
            throw NMPLlamaRuntimeError.callFailed(function: "decode_topk", status: status)
        }
        return (0..<Int(status)).map { (ids[$0], logits[$0]) }
    }
}
