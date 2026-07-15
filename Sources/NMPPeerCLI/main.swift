//
//  main.swift
//  nmp-peer — Phase 5 compute-peer CLI
//
//  Runs the same NMPPeerNode runtime the iOS app embeds: binds a UDP
//  listener, advertises over Bonjour (capabilities + port + static key),
//  accepts the coordinator's Noise IK handshake, serves its assigned
//  shard, and logs every inference it computes.
//
//  Usage:
//    swift run nmp-peer [--layers N] [--hidden N] [--gguf path]
//                       [--tag modelTag] [--slow msPerLayer]
//                       [--engine reference|llamaCpp] [--model path.gguf]
//                       [--gpu-layers N]
//
//  --gguf sizes the engine from a real model file's metadata (layer count
//  and hidden size are read from the container; compute itself is the
//  deterministic reference engine).
//  --engine llamaCpp (Phase 8) loads the model INTO llama.cpp via --model
//  and serves REAL forward passes for its assigned shard — which must be
//  the model's full layer range (see LlamaEngine.swift). Requires the
//  shim from scripts/setup_llama.sh.
//  --slow adds artificial per-layer delay to emulate a weaker device when
//  demoing load-balanced sharding on identical hardware (reference only).
//

import Foundation
import NMP

// MARK: - Arguments

struct PeerArguments {
    var layers = 32
    var hidden = 4096
    var ggufPath: String?
    var modelTag = "nmp-reference-model"
    var slowMillisPerLayer = 0.0
    var engineKind = "reference"
    var modelPath: String?
    var gpuLayers: Int32 = -1

    static func parse() -> PeerArguments {
        var arguments = PeerArguments()
        var iterator = CommandLine.arguments.dropFirst().makeIterator()
        while let flag = iterator.next() {
            let value = { iterator.next() }
            switch flag {
            case "--layers": arguments.layers = value().flatMap(Int.init) ?? arguments.layers
            case "--hidden": arguments.hidden = value().flatMap(Int.init) ?? arguments.hidden
            case "--gguf": arguments.ggufPath = value()
            case "--tag": arguments.modelTag = value() ?? arguments.modelTag
            case "--slow": arguments.slowMillisPerLayer = value().flatMap(Double.init) ?? 0
            case "--engine": arguments.engineKind = value() ?? arguments.engineKind
            case "--model": arguments.modelPath = value()
            case "--gpu-layers": arguments.gpuLayers = value().flatMap(Int32.init) ?? -1
            case "--help", "-h":
                print("""
                usage: nmp-peer [--layers N] [--hidden N] [--gguf path] \
                [--tag modelTag] [--slow msPerLayer] \
                [--engine reference|llamaCpp|llamaShard] [--model path.gguf] [--gpu-layers N]
                """)
                exit(0)
            default:
                FileHandle.standardError.write(Data("unknown flag \(flag)\n".utf8))
                exit(2)
            }
        }
        return arguments
    }
}

// MARK: - Engine setup

let arguments = PeerArguments.parse()
let engine: NMPShardComputeEngine
var modelTag = arguments.modelTag
/// Non-nil only for --engine llamaShard, so onServed can report loaded MB.
var llamaShardEngine: NMPLlamaShardComputeEngine?
/// Non-nil only in vault mode (--engine llamaShard with no --model).
var vaultShardEngine: NMPVaultShardComputeEngine?

if arguments.engineKind == "llamaShard" {
    // TRUE cross-device sharding: this peer loads ONLY the layer range the
    // coordinator assigns (ggml graph surgery), so a device too small for the
    // whole model still contributes. Needs the shard shim (scripts/setup_shard.sh).
    if let modelPath = arguments.modelPath ?? arguments.ggufPath {
        do {
            let shardEngine = try NMPLlamaShardComputeEngine(modelPath: modelPath)
            engine = shardEngine
            llamaShardEngine = shardEngine
            modelTag = shardEngine.modelTag
            print("[peer] llamaShard engine: \(shardEngine.modelTag) — "
                  + "\(shardEngine.layerCount) layers × \(shardEngine.hiddenSize) hidden; "
                  + "will partial-load ONLY the assigned range (ctx \(shardEngine.maxContext))")
        } catch {
            FileHandle.standardError.write(Data("""
            failed to start llamaShard engine: \(error)
            checklist: brew install ggml && scripts/setup_shard.sh, and --model must point at a .gguf file
            \n
            """.utf8))
            exit(1)
        }
    } else {
        // Future Plan #3: no local model — stream each assigned slice from the
        // coordinator's vault (disk ≈ RAM). Accept whatever model it serves.
        let vaultEngine = NMPVaultShardComputeEngine()
        engine = vaultEngine
        vaultShardEngine = vaultEngine
        modelTag = NMPVaultShardComputeEngine.wildcardModelTag
        print("[peer] llamaShard engine (VAULT mode): no local model — will stream "
              + "ONLY the assigned layers from the coordinator and cache them on disk")
    }
} else if arguments.engineKind == "llamaCpp" {
    guard let modelPath = arguments.modelPath else {
        FileHandle.standardError.write(Data("--engine llamaCpp requires --model path.gguf\n".utf8))
        exit(2)
    }
    do {
        let llamaEngine = try NMPLlamaComputeEngine(
            modelPath: modelPath, gpuLayers: arguments.gpuLayers)
        engine = llamaEngine
        modelTag = llamaEngine.modelTag
        print("[peer] llamaCpp engine: \(llamaEngine.modelTag) — "
              + "\(llamaEngine.layerCount) layers × \(llamaEngine.hiddenSize) hidden, "
              + "vocab \(llamaEngine.model.vocabSize), ctx \(llamaEngine.model.contextSize) "
              + "(mem \(NMPPeerShardEngine.residentMemoryMB()) MB)")
    } catch {
        FileHandle.standardError.write(Data("""
        failed to start llamaCpp engine: \(error)
        checklist: brew install llama.cpp && scripts/setup_llama.sh, and --model must point at a .gguf file
        \n
        """.utf8))
        exit(1)
    }
} else if let path = arguments.ggufPath {
    do {
        let gguf = try NMPGGUFModel.load(path: path)
        let referenceEngine = try NMPReferenceComputeEngine(gguf: gguf)
        referenceEngine.simulatedSecondsPerLayer = arguments.slowMillisPerLayer / 1000
        engine = referenceEngine
        if let name = gguf.modelName { modelTag = name }
        print("[peer] loaded GGUF: \(gguf.modelName ?? path) — "
              + "\(referenceEngine.layerCount) layers × \(referenceEngine.hiddenSize) hidden "
              + "(\(gguf.tensors.count) tensors, \(gguf.architecture ?? "?") arch)")
    } catch {
        FileHandle.standardError.write(Data("failed to load GGUF at \(path): \(error)\n".utf8))
        exit(1)
    }
} else {
    let referenceEngine = NMPReferenceComputeEngine(
        layerCount: arguments.layers, hiddenSize: arguments.hidden)
    referenceEngine.simulatedSecondsPerLayer = arguments.slowMillisPerLayer / 1000
    engine = referenceEngine
}

// MARK: - Run

let node = NMPPeerNode(engine: engine, modelTag: modelTag)

func stamp() -> String {
    let formatter = DateFormatter()
    formatter.dateFormat = "HH:mm:ss.SSS"
    return formatter.string(from: Date())
}

node.onStatus = { print("[peer \(stamp())] \($0)") }
node.onDiagnostic = { print("[peer \(stamp())] (diag) \($0)") }
node.onAssigned = { assign in
    if vaultShardEngine != nil {
        let cached = vaultShardEngine?.cachedSliceBytes ?? 0
        print("[peer \(stamp())] shard assigned: layers "
              + "\(assign.startLayer)..<\(assign.endLayer) of \(assign.totalLayers) "
              + String(format: "— streamed its slice from the vault (%.1f MB on disk, its range only)",
                       Double(cached) / 1_048_576))
        return
    }
    guard llamaShardEngine != nil else { return }
    print("[peer \(stamp())] shard assigned: layers "
          + "\(assign.startLayer)..<\(assign.endLayer) of \(assign.totalLayers) "
          + "— partial-loading only this range")
}
node.onServed = { requestID, layers, seconds in
    // For a real shard peer, report the MEASURED loaded weights (only this
    // range), the honest proof it doesn't hold the whole model.
    let loaded = llamaShardEngine?.loadedBytes ?? 0
    let vaultDisk = vaultShardEngine?.cachedSliceBytes ?? 0
    let loadedNote = loaded > 0
        ? String(format: ", loaded %.1f MB (its range only)", Double(loaded) / 1_048_576)
        : (vaultDisk > 0
           ? String(format: ", vault slice %.1f MB on disk (its range only)", Double(vaultDisk) / 1_048_576)
           : "")
    print("[peer \(stamp())] served request \(requestID): layers "
          + "\(layers.lowerBound)..<\(layers.upperBound) in "
          + String(format: "%.2f", seconds * 1000) + " ms "
          + "(mem \(NMPPeerShardEngine.residentMemoryMB()) MB\(loadedNote))")
}

do {
    try node.start()
} catch {
    FileHandle.standardError.write(Data("failed to start peer: \(error)\n".utf8))
    exit(1)
}

print("[peer] running — Ctrl+C to stop")
dispatchMain()
