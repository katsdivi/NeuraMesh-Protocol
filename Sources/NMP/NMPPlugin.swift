//
//  NMPPlugin.swift
//  NMP — Plugin seam (formalized)
//
//  The mesh's compute seam has always been `NMPShardComputeEngine`
//  (ComputeEngine.swift): everything above it — job placement, capacity
//  ceilings, churn-safe re-sharding, commit-on-ack, encrypted transport —
//  is engine-agnostic. This file names that seam "the NMP plugin interface"
//  and gives it ONE registry so the set of selectable plugins, their help
//  text, and their construction live in a single place instead of being
//  duplicated across the three CLIs.
//
//  This is a documentation/organization pass, NOT new capability:
//
//  - `NMPPlugin` is a typealias for the existing `NMPShardComputeEngine`.
//    A full rename would ripple through ~15 files and every test; the
//    typealias lets new code and docs speak "plugin" while the existing
//    Swift type name (and all its conformers) stay put. See
//    Docs/Plugin_Architecture.md for the rationale.
//
//  - `NMPPluginRegistry` is the single catalog. Adding a *pure-compute*
//    plugin (one that needs nothing beyond `NMPPluginContext` to build —
//    like the reference engine or the hashShard stub) is a one-file change
//    here. LLM plugins (llamaCpp / llamaShard) additionally need per-CLI
//    orchestration (a vocab-only tokenizer on the coordinator, a weight
//    vault on the dashboard); that wiring is flagged, not hidden.
//

import Foundation

// MARK: - Plugin alias

/// The NMP plugin interface. A plugin is any type that can run a job over an
/// assigned shard — today always a transformer layer sub-range, hence the
/// historical name `NMPShardComputeEngine`, which this aliases so existing
/// conformers and tests are untouched. New code should prefer `NMPPlugin`.
public typealias NMPPlugin = NMPShardComputeEngine

// MARK: - Construction context

/// Everything a *pure-compute* plugin needs to build itself from CLI flags.
/// LLM plugins that also need a tokenizer handle or a weight vault are built
/// by the CLIs directly (see Docs/Plugin_Architecture.md § "LLM-specific").
public struct NMPPluginContext {
    /// `--layers` (used only when no GGUF sizes the engine).
    public var layers: Int
    /// `--hidden` (used only when no GGUF sizes the engine).
    public var hiddenSize: Int
    /// `--gguf` / `--model`: sizes an engine from real model metadata.
    public var ggufPath: String?
    /// Default `--tag` if the plugin cannot derive one from a model file.
    public var modelTag: String
    /// `--slow` converted to seconds: artificial per-layer delay for demos.
    public var slowSecondsPerLayer: TimeInterval

    public init(layers: Int = 32, hiddenSize: Int = 4096, ggufPath: String? = nil,
                modelTag: String = "nmp-reference-model",
                slowSecondsPerLayer: TimeInterval = 0) {
        self.layers = layers
        self.hiddenSize = hiddenSize
        self.ggufPath = ggufPath
        self.modelTag = modelTag
        self.slowSecondsPerLayer = slowSecondsPerLayer
    }
}

/// A built plugin plus the model tag the mesh should advertise for it (a
/// GGUF-sized engine derives the tag from `general.name`, so the factory
/// returns it rather than making every caller re-read the file).
public struct NMPPluginInstance {
    public let engine: NMPPlugin
    public let modelTag: String
    public init(engine: NMPPlugin, modelTag: String) {
        self.engine = engine
        self.modelTag = modelTag
    }
}

// MARK: - Descriptor

/// One entry in the plugin catalog: the `--engine` id, its help text, and
/// (for pure-compute plugins) a factory. LLM plugins leave `makeGeneric` nil
/// because their construction is entangled with CLI-specific orchestration.
public struct NMPPluginDescriptor {
    /// The value passed to `--engine`.
    public let id: String
    /// One-line description shown in `--help`.
    public let summary: String
    /// True when the job is an LLM shard. Drives what the mesh/UI can assume
    /// (model/layer language, tensor-shaped activations). See the doc.
    public let isLLM: Bool
    /// True when the plugin cannot run without a `--model` / `--gguf` file.
    public let requiresModelFile: Bool
    /// Builds the plugin from generic context alone, or nil when the plugin
    /// needs bespoke CLI wiring (tokenizer, vault) the registry can't supply.
    public let makeGeneric: ((NMPPluginContext) throws -> NMPPluginInstance)?

    public init(id: String, summary: String, isLLM: Bool, requiresModelFile: Bool,
                makeGeneric: ((NMPPluginContext) throws -> NMPPluginInstance)?) {
        self.id = id
        self.summary = summary
        self.isLLM = isLLM
        self.requiresModelFile = requiresModelFile
        self.makeGeneric = makeGeneric
    }
}

// MARK: - Registry

/// The single source of truth for which plugins exist. To add a pure-compute
/// plugin, append one descriptor to `all` — nothing else in the CLIs changes.
public enum NMPPluginRegistry {

    /// The reference engine: deterministic, bit-exact cross-device, weightless.
    /// The correctness oracle behind most of the test suite.
    public static let reference = NMPPluginDescriptor(
        id: "reference",
        summary: "reference: deterministic bit-exact stand-in (no weights, shardable)",
        isLLM: false,
        requiresModelFile: false,
        makeGeneric: { ctx in
            if let path = ctx.ggufPath {
                let gguf = try NMPGGUFModel.load(path: path)
                let engine = try NMPReferenceComputeEngine(gguf: gguf)
                engine.simulatedSecondsPerLayer = ctx.slowSecondsPerLayer
                let tag = (gguf.modelName?.isEmpty == false) ? gguf.modelName! : ctx.modelTag
                return NMPPluginInstance(engine: engine, modelTag: tag)
            }
            let engine = NMPReferenceComputeEngine(
                layerCount: ctx.layers, hiddenSize: ctx.hiddenSize)
            engine.simulatedSecondsPerLayer = ctx.slowSecondsPerLayer
            return NMPPluginInstance(engine: engine, modelTag: ctx.modelTag)
        })

    /// Full-range llama.cpp: one peer holds the whole model, one full-range
    /// shard. Built per-CLI (needs a vocab-only tokenizer on the coordinator).
    public static let llamaCpp = NMPPluginDescriptor(
        id: "llamaCpp",
        summary: "llamaCpp: real llama.cpp, ONE full-range shard (peer holds whole model)",
        isLLM: true,
        requiresModelFile: true,
        makeGeneric: nil)

    /// TRUE cross-device layer sharding via ggml graph surgery: each peer
    /// loads only its assigned range. Built per-CLI (tokenizer + weight vault).
    public static let llamaShard = NMPPluginDescriptor(
        id: "llamaShard",
        summary: "llamaShard: real ggml sub-range sharding (each peer loads only its layers)",
        isLLM: true,
        requiresModelFile: false,   // vault mode streams slices, no local file
        makeGeneric: nil)

    /// Non-LLM SCAFFOLD (Plugin Architecture task 4): proves the seam is real
    /// for a job that isn't a transformer. Logic is TODO — see HashShardEngine.
    public static let hashShard = NMPPluginDescriptor(
        id: "hashShard",
        summary: "hashShard: non-LLM STUB (checksum job) — proves the seam, logic is TODO",
        isLLM: false,
        requiresModelFile: false,
        makeGeneric: { ctx in
            let engine = NMPHashShardComputeEngine(
                layerCount: ctx.layers, hiddenSize: ctx.hiddenSize)
            return NMPPluginInstance(engine: engine, modelTag: "nmp-hashshard-stub")
        })

    /// The catalog. Order = display order in `--help`.
    public static let all: [NMPPluginDescriptor] = [
        reference, llamaCpp, llamaShard, hashShard,
    ]

    /// Look up a descriptor by its `--engine` id.
    public static func descriptor(id: String) -> NMPPluginDescriptor? {
        all.first { $0.id == id }
    }

    /// `reference|llamaCpp|llamaShard|hashShard` — for CLI usage strings.
    public static var usageList: String {
        all.map(\.id).joined(separator: "|")
    }

    /// Multi-line help block listing every plugin and its summary.
    public static var helpBlock: String {
        all.map { "  \($0.id)\n      \($0.summary)" }.joined(separator: "\n")
    }
}
