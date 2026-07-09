//
//  main.swift
//  nmp-coordinator — Phase 5 coordinator + benchmark CLI
//
//  Discovers peers over Bonjour, dials each one (Noise IK over UDP, keys
//  from TXT records), shards the model proportionally to peer speed,
//  measures a single-device baseline, then runs the pipelined benchmark
//  and prints per-shard/per-run metrics.
//
//  Usage:
//    swift run nmp-coordinator [--peers N] [--layers N] [--hidden N]
//                              [--gguf path] [--tag modelTag]
//                              [--runs N] [--tokens N] [--wait seconds]
//
//  --peers   remote peers to wait for before sharding (default 1)
//  --runs    benchmark repetitions per token count (default 5)
//  --tokens  sequential activation passes per run, emulating token steps
//            (default 8)
//  --wait    discovery timeout in seconds (default 60)
//
//  Phase 8 — real LLM over the real mesh:
//    swift run nmp-coordinator --engine llamaCpp --model path.gguf
//                              [--prompt "..."] [--tokens N] [--runs N]
//
//  The coordinator loads only the TOKENIZER (vocab-only, a few MB); the
//  remote nmp-peer (--engine llamaCpp) owns the weights and serves the
//  model's full layer range — llama.cpp cannot execute layer sub-ranges,
//  so a llama plan is one full-range shard and "distributed" means real
//  remote execution over the real transport (see LlamaEngine.swift).
//  Every generated token is one full mesh round trip. Runs repeat the
//  same prompt; greedy sampling makes outputs bit-identical across runs
//  and identical to single-device output from the same weights.
//

import Foundation
import NMP

// MARK: - Arguments

struct CoordinatorArguments {
    var peers = 1
    var layers = 32
    var hidden = 4096
    var ggufPath: String?
    var modelTag = "nmp-reference-model"
    var runs = 5
    var tokens = 8
    var waitSeconds = 60.0
    /// Artificial per-layer compute (ms) to emulate 7B-scale work: with
    /// real models, per-layer time dwarfs a LAN RTT; the reference engine
    /// alone is so fast that network dominates and the mesh/baseline
    /// ratio is meaningless. Use the same value on the peers.
    var slowMillisPerLayer = 0.0
    var engineKind = "reference"
    var modelPath: String?
    var prompt = "Hello, my name is"

    static func parse() -> CoordinatorArguments {
        var arguments = CoordinatorArguments()
        var iterator = CommandLine.arguments.dropFirst().makeIterator()
        while let flag = iterator.next() {
            let value = { iterator.next() }
            switch flag {
            case "--peers": arguments.peers = value().flatMap(Int.init) ?? arguments.peers
            case "--layers": arguments.layers = value().flatMap(Int.init) ?? arguments.layers
            case "--hidden": arguments.hidden = value().flatMap(Int.init) ?? arguments.hidden
            case "--gguf": arguments.ggufPath = value()
            case "--tag": arguments.modelTag = value() ?? arguments.modelTag
            case "--runs": arguments.runs = value().flatMap(Int.init) ?? arguments.runs
            case "--tokens": arguments.tokens = value().flatMap(Int.init) ?? arguments.tokens
            case "--wait": arguments.waitSeconds = value().flatMap(Double.init) ?? arguments.waitSeconds
            case "--slow": arguments.slowMillisPerLayer = value().flatMap(Double.init) ?? 0
            case "--engine": arguments.engineKind = value() ?? arguments.engineKind
            case "--model": arguments.modelPath = value()
            case "--prompt": arguments.prompt = value() ?? arguments.prompt
            case "--help", "-h":
                print("""
                usage: nmp-coordinator [--peers N] [--layers N] [--hidden N] \
                [--gguf path] [--tag modelTag] [--runs N] [--tokens N] [--wait seconds] \
                [--engine reference|llamaCpp] [--model path.gguf] [--prompt "..."]
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

let arguments = CoordinatorArguments.parse()

// MARK: - Engine

let engine: NMPShardComputeEngine
var modelTag = arguments.modelTag
/// Set in llamaCpp mode: the coordinator's VOCAB-ONLY handle (tokenizer +
/// metadata, no weights — the remote peer owns those).
var llamaModel: NMPLlamaModel?

if arguments.engineKind == "llamaCpp" {
    guard let modelPath = arguments.modelPath else {
        FileHandle.standardError.write(Data("--engine llamaCpp requires --model path.gguf\n".utf8))
        exit(2)
    }
    do {
        let model = try NMPLlamaModel(modelPath: modelPath, vocabOnly: true)
        llamaModel = model
        engine = NMPLlamaComputeEngine(model: model)
        modelTag = model.name
        print("[coordinator] llamaCpp (vocab-only): \(model.name) — "
              + "\(model.layerCount) layers × \(model.hiddenSize) hidden, "
              + "vocab \(model.vocabSize)")
    } catch {
        FileHandle.standardError.write(Data("""
        failed to load llama tokenizer: \(error)
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
        print("[coordinator] loaded GGUF: \(gguf.modelName ?? path) — "
              + "\(referenceEngine.layerCount) layers × \(referenceEngine.hiddenSize) hidden")
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

// MARK: - Helpers

func ms(_ seconds: TimeInterval) -> String { String(format: "%.1f", seconds * 1000) }
func hex(_ id: UInt32) -> String { String(format: "%08x", id) }

/// Deterministic input vector (same one every run — comparable numbers).
func makeInput(width: Int) -> [Float] {
    var state: UInt64 = 0x5EED
    return (0..<width).map { _ in
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float(state >> 40) / Float(1 << 24) * 2 - 1
    }
}

// MARK: - Mesh assembly

let nodeQueue = DispatchQueue(label: "nmp.coordinator.main")
let node = NMPCoordinatorNode(engine: engine, modelTag: modelTag, queue: nodeQueue)
node.onStatus = { print("[coordinator] \($0)") }
node.onPeerMetrics = { metrics in
    print("[coordinator] metrics from \(hex(metrics.peerID)): "
          + "compute \(String(format: "%.1f", Double(metrics.inferenceLatencyMicros) / 1000)) ms, "
          + "mem \(metrics.memoryUsageMB) MB, load \(metrics.currentLoadPercent)%")
}

let assembled = DispatchSemaphore(value: 0)
node.onPeerReady = { capabilities in
    print("[coordinator] ready: \(capabilities.deviceName) "
          + "(\(hex(capabilities.peerID)), \(capabilities.computeClass.label), "
          + "\(capabilities.ramMB) MB RAM)")
    if node.readyPeers.count >= arguments.peers { assembled.signal() }
}

// start() asserts it runs on the node's queue.
nodeQueue.sync {
    do {
        try node.start()
    } catch {
        FileHandle.standardError.write(Data("failed to start coordinator: \(error)\n".utf8))
        exit(1)
    }
}

print("[coordinator] waiting for \(arguments.peers) peer(s), "
      + "timeout \(Int(arguments.waitSeconds)) s …")
guard assembled.wait(timeout: .now() + arguments.waitSeconds) == .success else {
    FileHandle.standardError.write(Data("""
    timed out: no mesh assembled. Checklist:
      - is nmp-peer (or the iPhone app) running on the same Wi-Fi/LAN?
      - did the device show the Local Network permission prompt and was it allowed?
      - does the network allow mDNS (some corporate/guest networks block it)?
    \n
    """.utf8))
    exit(1)
}

// MARK: - Phase 8: llama generation over the real mesh

/// Full-range shard on the first ready peer, then `runs` prompt
/// generations — every token one real mesh round trip. Never returns.
func runLlamaGenerations(model: NMPLlamaModel) -> Never {
    guard let shardPeerID = node.readyPeers.keys.sorted().first else {
        FileHandle.standardError.write(Data("no ready peer to place the llama shard on\n".utf8))
        exit(1)
    }
    let plan = [NMPShardPlanEntry.fullRange(peerID: shardPeerID,
                                            layerCount: engine.layerCount)]
    let assigned = DispatchSemaphore(value: 0)
    node.orchestrator.assignShards(plan) { result in
        if case .failure(let error) = result {
            FileHandle.standardError.write(Data("shard assignment failed: \(error)\n".utf8))
            exit(1)
        }
        assigned.signal()
    }
    assigned.wait()

    print("\n=== Llama plan (model '\(modelTag)') ===")
    print("  shard 0: layers 0..<\(engine.layerCount) (full model) → peer \(hex(shardPeerID))")
    print("  coordinator: tokenizer only (no weights loaded)")
    print("\n=== \(arguments.runs) generation(s): \"\(arguments.prompt)\" "
          + "(up to \(arguments.tokens) tokens) ===")

    let service = NMPPromptInferenceService(
        orchestrator: node.orchestrator,
        codec: NMPLlamaPromptCodec(model: model))

    var texts: [String] = []
    var bestTokensPerSecond = 0.0
    for run in 1...max(1, arguments.runs) {
        let done = DispatchSemaphore(value: 0)
        service.run(prompt: arguments.prompt, maxTokens: arguments.tokens) { result in
            switch result {
            case .failure(let error):
                FileHandle.standardError.write(Data("generation failed: \(error)\n".utf8))
                exit(1)
            case .success(let generation):
                let tokensPerSecond = Double(generation.tokenCount)
                    / max(generation.totalSeconds, 0.001)
                bestTokensPerSecond = max(bestTokensPerSecond, tokensPerSecond)
                texts.append(generation.text)
                let perToken = generation.perTokenSeconds.map { $0 * 1000 }
                print("  run \(run): \(generation.tokenCount) tokens in "
                      + "\(ms(generation.totalSeconds)) ms "
                      + "(\(String(format: "%.2f", tokensPerSecond)) tok/s, "
                      + "payload \(generation.networkPayloadBytes) B, "
                      + "per-token p50 ~\(String(format: "%.1f", perToken.sorted()[perToken.count / 2])) ms)")
                print("    → \(generation.text)")
            }
            done.signal()
        }
        done.wait()
    }

    print("\n=== Results ===")
    print("  engine: llamaCpp (real model, remote full-range shard)")
    print("  best throughput: \(String(format: "%.2f", bestTokensPerSecond)) tokens/s")
    let deterministic = Set(texts).count == 1
    print("  determinism: \(arguments.runs) runs "
          + (deterministic ? "IDENTICAL output ✓ (greedy sampling)" : "DIVERGED ✗"))
    print("  (single-device comparison: run the same model via "
          + "`nmp-dashboard --engine llamaCpp --placement local` and POST the same prompt)")
    exit(deterministic ? 0 : 1)
}

if let llamaModel {
    runLlamaGenerations(model: llamaModel)
}

// MARK: - Shard assignment

let planned = DispatchSemaphore(value: 0)
var shardPlan: [NMPShardPlanEntry] = []
node.planAndAssign { result in
    switch result {
    case .failure(let error):
        FileHandle.standardError.write(Data("shard assignment failed: \(error)\n".utf8))
        exit(1)
    case .success(let plan):
        shardPlan = plan
        planned.signal()
    }
}
planned.wait()

print("\n=== Shard plan (\(engine.layerCount) layers × \(engine.hiddenSize) hidden, "
      + "model '\(modelTag)') ===")
for entry in shardPlan {
    let who = entry.peerID == node.localPeerID ? "coordinator (local)" : "peer \(hex(entry.peerID))"
    print("  shard \(entry.shardIndex): layers "
          + "\(entry.startLayer)..<\(entry.endLayer) (\(entry.layerSpan)) → \(who)")
}

// MARK: - Single-device baseline

let baselineInput = makeInput(width: engine.hiddenSize)
var baselineOutput: [Float] = []
var baselineSeconds = TimeInterval.greatestFiniteMagnitude
for _ in 0..<max(1, arguments.runs) {
    let began = DispatchTime.now()
    for _ in 0..<arguments.tokens {
        baselineOutput = try! engine.runLayers(start: 0, end: engine.layerCount,
                                               input: baselineInput)
    }
    let elapsed = TimeInterval(DispatchTime.now().uptimeNanoseconds
                               - began.uptimeNanoseconds) / 1e9
    baselineSeconds = min(baselineSeconds, elapsed)
}
print("\n=== Baseline (coordinator alone) ===")
print("  \(arguments.tokens) tokens: \(ms(baselineSeconds)) ms  "
      + "(\(String(format: "%.1f", Double(arguments.tokens) / baselineSeconds)) tokens/s)")

// MARK: - Mesh benchmark

print("\n=== Mesh benchmark: \(arguments.runs) runs × \(arguments.tokens) tokens ===")
var meshBest = TimeInterval.greatestFiniteMagnitude
var lastOutput: [Float] = []
var totalNetworkBytes = 0

for run in 1...max(1, arguments.runs) {
    let runDone = DispatchSemaphore(value: 0)
    var runSeconds: TimeInterval = 0
    var tokensLeft = arguments.tokens
    var failed = false
    var shardLines: [String] = []

    func step() {
        node.orchestrator.infer(input: baselineInput) { result in
            switch result {
            case .failure(let error):
                FileHandle.standardError.write(Data("inference failed: \(error)\n".utf8))
                failed = true
                runDone.signal()
            case .success(let report):
                runSeconds += report.totalSeconds
                totalNetworkBytes += report.networkPayloadBytes
                lastOutput = report.output
                tokensLeft -= 1
                if tokensLeft == 0 {
                    shardLines = report.perShard.map { timing in
                        let who = timing.isLocal ? "local" : "peer \(hex(timing.peerID))"
                        return "    shard \(timing.shardIndex) (\(who), layers "
                            + "\(timing.layers.lowerBound)..<\(timing.layers.upperBound)): "
                            + "compute \(ms(timing.computeSeconds)) ms, "
                            + "stage \(ms(timing.stageSeconds)) ms"
                    }
                    runDone.signal()
                } else {
                    step()
                }
            }
        }
    }
    step()
    runDone.wait()
    if failed { exit(1) }

    meshBest = min(meshBest, runSeconds)
    print("  run \(run): \(ms(runSeconds)) ms  "
          + "(\(String(format: "%.1f", Double(arguments.tokens) / runSeconds)) tokens/s)")
    if run == arguments.runs {
        print("  last-run shard breakdown:")
        shardLines.forEach { print($0) }
    }
}

// MARK: - Verdict

print("\n=== Results ===")
let ratio = meshBest / baselineSeconds
print("  baseline (1 device) best: \(ms(baselineSeconds)) ms")
print("  mesh (\(shardPlan.count) shards)  best: \(ms(meshBest)) ms  "
      + "(\(String(format: "%.2f", ratio))× baseline)")
let outputBytes = lastOutput.count * 4
if outputBytes > 0 {
    let overhead = Double(totalNetworkBytes)
        / Double(arguments.runs * arguments.tokens * outputBytes)
    print("  network payload total: \(totalNetworkBytes) B "
          + "(\(String(format: "%.1f", overhead))× the output tensor per token)")
}
let correct = lastOutput.map(\.bitPattern) == baselineOutput.map(\.bitPattern)
print("  numerics: mesh output "
      + (correct ? "BIT-EXACT vs single device ✓" : "DIVERGED vs single device ✗"))
if !correct { exit(1) }

print("\n[coordinator] done.")
exit(0)
