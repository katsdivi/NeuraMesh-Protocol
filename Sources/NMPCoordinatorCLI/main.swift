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
            case "--help", "-h":
                print("""
                usage: nmp-coordinator [--peers N] [--layers N] [--hidden N] \
                [--gguf path] [--tag modelTag] [--runs N] [--tokens N] [--wait seconds]
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

let engine: NMPReferenceComputeEngine
var modelTag = arguments.modelTag
if let path = arguments.ggufPath {
    do {
        let gguf = try NMPGGUFModel.load(path: path)
        engine = try NMPReferenceComputeEngine(gguf: gguf)
        if let name = gguf.modelName { modelTag = name }
        print("[coordinator] loaded GGUF: \(gguf.modelName ?? path) — "
              + "\(engine.layerCount) layers × \(engine.hiddenSize) hidden")
    } catch {
        FileHandle.standardError.write(Data("failed to load GGUF at \(path): \(error)\n".utf8))
        exit(1)
    }
} else {
    engine = NMPReferenceComputeEngine(layerCount: arguments.layers, hiddenSize: arguments.hidden)
}
engine.simulatedSecondsPerLayer = arguments.slowMillisPerLayer / 1000

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
