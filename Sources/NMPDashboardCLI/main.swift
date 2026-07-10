//
//  main.swift
//  nmp-dashboard — Phase 6
//
//  Runs a live simulated mesh (coordinator + 3 shard peers over in-memory
//  links: real handshakes, encryption, FEC, NACK — loss injectors in the
//  datapath) and serves the testing dashboard on http://localhost:8080.
//
//  Dashboard controls:
//    loss slider       → steady loss on every link
//    inject peer drop  → silences a peer; failover re-shards survivors
//    start benchmark   → loss sweep, results stream into the event log
//    reset metrics     → clears injector counters
//
//  Usage:
//    swift run nmp-dashboard [port]              — live dashboard
//    swift run nmp-dashboard --benchmark [dir]   — headless comprehensive
//                                                  benchmark, CSVs to dir
//                                                  (default: Results/)
//
//  Phase 8 — real LLM engine:
//    swift run nmp-dashboard [port] --engine llamaCpp --model PATH
//                            [--gpu-layers N] [--placement local|remote]
//
//  llamaCpp mode serves REAL llama.cpp text through POST /api/inference.
//  --placement remote (default) puts the model behind an in-process link
//  running the full protocol stack (Noise IK, AES-GCM, FEC, NACK — the
//  loss slider works); --placement local is the single-device baseline.
//  Falls back to the reference mesh (with a warning) when the shim or
//  model is missing — see scripts/setup_llama.sh.
//

import Foundation
import NMP

// MARK: - Arguments

struct DashboardArguments {
    var port: UInt16 = 8080
    var engine = "reference"
    var modelPath: String?
    var gpuLayers: Int32 = -1
    var placement = NMPLlamaTestbed.Placement.remotePeer
    // Phase 9
    /// Automatic setup: benchmark → balanced shards → optimized wire format.
    var autoConfig = false
    /// Serve every /api/inference request speculatively (llamaCpp only);
    /// without the flag, {"enable_speculation": true} opts in per request.
    var speculation = false
    /// Small same-vocabulary GGUF drafting for the target model; without
    /// it speculation uses the prompt-lookup drafter.
    var draftModelPath: String?
    /// Probe passes for --auto-config benchmarking.
    var probePasses = 3

    static func parse() -> DashboardArguments {
        var arguments = DashboardArguments()
        var iterator = CommandLine.arguments.dropFirst().makeIterator()
        while let flag = iterator.next() {
            let value = { iterator.next() }
            switch flag {
            case "--engine": arguments.engine = value() ?? arguments.engine
            case "--model": arguments.modelPath = value()
            case "--gpu-layers": arguments.gpuLayers = value().flatMap(Int32.init) ?? -1
            case "--placement":
                arguments.placement = value().flatMap {
                    NMPLlamaTestbed.Placement(rawValue: $0 == "remote" ? "remotePeer" : $0)
                } ?? arguments.placement
            case "--auto-config": arguments.autoConfig = true
            case "--speculation": arguments.speculation = true
            case "--draft-model": arguments.draftModelPath = value()
            case "--probe-passes":
                arguments.probePasses = value().flatMap(Int.init) ?? arguments.probePasses
            case "--help", "-h":
                print("""
                usage: nmp-dashboard [port] [--engine reference|llamaCpp] \
                [--model path.gguf] [--gpu-layers N] [--placement local|remote] \
                [--auto-config] [--probe-passes N] [--speculation] [--draft-model path.gguf]
                """)
                exit(0)
            default:
                if let parsed = UInt16(flag) {
                    arguments.port = parsed
                } else {
                    FileHandle.standardError.write(Data("unknown flag \(flag)\n".utf8))
                    exit(2)
                }
            }
        }
        return arguments
    }
}

// MARK: - Headless benchmark mode

if CommandLine.arguments.dropFirst().first == "--benchmark" {
    let directory = URL(fileURLWithPath: CommandLine.arguments.count > 2
        ? CommandLine.arguments[2] : "Results")
    do {
        // 3 remote peers + coordinator, 4 KB activations — the same shape
        // the live dashboard runs.
        let benchTestbed = try NMPMeshTestbed(
            layerCount: 24, hiddenSize: 1024, remotePeerCount: 3)
        _ = try benchTestbed.startSync()
        let suite = NMPBenchmarkSuite(testbed: benchTestbed)
        _ = try suite.runComprehensive()
        try suite.exportCSV(to: directory)
        print("\nCSV exports written to \(directory.path)/")
        exit(0)
    } catch {
        FileHandle.standardError.write(Data("benchmark failed: \(error)\n".utf8))
        exit(1)
    }
}

let dashboardArguments = DashboardArguments.parse()
let port = dashboardArguments.port

// MARK: - Phase 8: llamaCpp engine mode

/// Real-LLM dashboard: single full-range llama shard (local or behind the
/// in-process link), POST /api/inference generates real text. Never
/// returns. See the header comment for what is and isn't available here.
func runLlamaDashboard(model: NMPLlamaModel, arguments: DashboardArguments) -> Never {
    let placement = arguments.placement
    let engine = NMPLlamaComputeEngine(model: model)
    print("[nmp-dashboard] llamaCpp engine: \(model.name) — "
          + "\(model.layerCount) layers × \(model.hiddenSize) hidden, "
          + "vocab \(model.vocabSize), ctx \(model.contextSize)")

    let testbed: NMPLlamaTestbed
    do {
        testbed = try NMPLlamaTestbed(
            engine: engine, modelTag: model.name, placement: placement)
        let plan = try testbed.startSync()
        let shard = plan[0]
        print("[nmp-dashboard] llama mesh live: layers "
              + "\(shard.startLayer)..<\(shard.endLayer) on "
              + (placement == .local ? "coordinator (local)" : "in-process peer (full stack)"))
    } catch {
        fatalError("llama mesh assembly failed: \(error)")
    }

    // Phase 9: auto-config for a llama plan. Layer sharding cannot apply
    // (one full-range shard by construction), but the wire format can:
    // token-state vectors are mostly zero padding, so zero-trim them.
    if arguments.autoConfig {
        let format = NMPAutoConfig.recommendedWireFormat(engineName: "llamaCpp")
        testbed.orchestrator.activationWireFormat = format
        print("[auto-config] llama plan is one full-range shard — layer sharding n/a")
        print("[auto-config] activation wire format → \(format.rawValue) "
              + "(lossless, ~99% smaller token-state messages)")
    }

    // Phase 9: speculative decoding. --speculation serves every request
    // speculatively; otherwise {"enable_speculation": true} opts in.
    var speculativeService: NMPSpeculativeGenerationService?
    if model.runtime.supportsSpeculation {
        var drafter: NMPSpeculativeDrafter = NMPPromptLookupDrafter()
        if let draftPath = arguments.draftModelPath {
            do {
                let draftModel = try NMPLlamaModel(modelPath: draftPath)
                if draftModel.vocabSize == model.vocabSize {
                    drafter = try NMPLlamaDraftModelDrafter(model: draftModel)
                    print("[nmp-dashboard] draft model: \(draftModel.name) "
                          + "(\(draftModel.layerCount) layers)")
                } else {
                    print("⚠️  draft model vocab \(draftModel.vocabSize) ≠ target "
                          + "\(model.vocabSize) — falling back to prompt-lookup drafting")
                }
            } catch {
                print("⚠️  draft model unavailable (\(error)) — "
                      + "falling back to prompt-lookup drafting")
            }
        }
        speculativeService = NMPSpeculativeGenerationService(
            orchestrator: testbed.orchestrator, model: model, drafter: drafter)
        print("[nmp-dashboard] speculation ready (\(drafter.drafterName), "
              + "depth \(NMPSpeculativeGenerationService.defaultDepth)"
              + (arguments.speculation ? ", default ON)" : ", per-request opt-in)"))
    } else if arguments.speculation {
        print("⚠️  --speculation needs a Phase 9 shim — rerun scripts/setup_llama.sh")
    }

    let server = NMPDashboardServer()
    server.onDiagnostic = { print("[nmp-dashboard] \($0)") }
    do {
        try server.start(port: port)
    } catch {
        fatalError("dashboard server failed to start: \(error)")
    }
    print("[nmp-dashboard] dashboard running at http://localhost:\(server.boundPort)")

    testbed.onPacketEvent = { event in
        server.reportPacketEvent(event, peerID: testbed.peerID)
    }
    testbed.onInferenceServed = { _, layers, seconds in
        server.updatePeerState(
            peerID: testbed.peerID, name: "llama-peer (in-process)",
            latencyMS: Int(seconds * 1000), loadPercent: 0,
            assigned: "layers \(layers.lowerBound)-\(layers.upperBound - 1)",
            alive: true)
    }
    server.updatePeerState(
        peerID: testbed.coordinatorID, name: "coordinator (tokenizer)",
        latencyMS: 0, loadPercent: 0,
        assigned: placement == .local ? "layers 0-\(engine.layerCount - 1)" : "—",
        alive: true)
    if placement == .remotePeer {
        server.updatePeerState(
            peerID: testbed.peerID, name: "llama-peer (in-process)",
            latencyMS: 0, loadPercent: 0,
            assigned: "layers 0-\(engine.layerCount - 1)", alive: true)
    }

    server.onControl = { control in
        switch control {
        case .setLossRate(let rate):
            guard placement == .remotePeer else {
                server.reportMeshEvent("loss injection needs --placement remote")
                return
            }
            testbed.setLossRate(rate)
            server.reportLossRate(rate)
            server.reportMeshEvent(String(format: "loss rate set to %.0f%%", rate * 100))
        case .injectPeerDrop:
            server.reportMeshEvent("peer drop n/a: a llama mesh has one shard "
                                   + "(the model is whole on one peer)")
        case .startBenchmark:
            server.reportMeshEvent("loss-sweep benchmark n/a with llamaCpp — "
                                   + "use POST /api/inference for timed generations")
        case .resetMetrics:
            server.reportMeshEvent("metrics reset")
        }
    }

    let promptService = NMPPromptInferenceService(
        orchestrator: testbed.orchestrator,
        codec: NMPLlamaPromptCodec(model: model))
    let reportProgress: (Int, Int) -> Void = { done, total in
        server.updateInferenceProgress(
            progress: Double(done) / Double(max(total, 1)),
            stage: "llama generation: token \(done)/\(total)")
    }
    promptService.onProgress = reportProgress
    speculativeService?.onProgress = reportProgress

    server.onInferenceRequest = { request, respond in
        let handleResult: (Result<NMPPromptInferenceService.GenerationResult,
                                  NMPPromptInferenceService.ServiceError>) -> Void = { result in
            switch result {
            case .success(let generation):
                var summary = String(
                    format: "🌐 llama inference done: %d tokens in %.1f ms "
                        + "(%.2f tok/s) across %d shard(s)",
                    generation.tokenCount, generation.totalSeconds * 1000,
                    Double(generation.tokenCount) / max(generation.totalSeconds, 0.001),
                    generation.shardCount)
                if let stats = generation.speculation {
                    summary += String(
                        format: " — speculative: %d round trip(s), %.2f tok/trip, "
                            + "%.0f%% draft acceptance",
                        stats.meshRoundTrips,
                        stats.tokensPerRoundTrip(tokenCount: generation.tokenCount),
                        stats.acceptanceRate * 100)
                }
                server.reportMeshEvent(summary)
                respond(.success(generation))
            case .failure(.busy):
                respond(.failure(status: 429, message: "an inference is already running — retry shortly"))
            case .failure(.emptyPrompt):
                respond(.failure(status: 400, message: "prompt is empty"))
            case .failure(.codec(let reason)):
                respond(.failure(status: 400, message: "prompt encoding failed: \(reason)"))
            case .failure(.orchestration(let error)):
                server.reportMeshEvent("🌐 llama inference failed: \(error)")
                respond(.failure(status: 500, message: "mesh orchestration failed: \(error)"))
            }
        }

        let speculate = request.enableSpeculation || dashboardArguments.speculation
        if speculate, let speculativeService {
            server.reportMeshEvent("🌐 llama inference (speculative): "
                                   + "up to \(request.maxTokens) token(s)")
            speculativeService.run(prompt: request.prompt,
                                   maxTokens: request.maxTokens,
                                   completion: handleResult)
        } else {
            if speculate {
                server.reportMeshEvent("speculation requested but unavailable "
                                       + "(rebuild the shim) — serving plain")
            }
            server.reportMeshEvent("🌐 llama inference: up to \(request.maxTokens) token(s)")
            promptService.run(prompt: request.prompt,
                              maxTokens: request.maxTokens,
                              completion: handleResult)
        }
    }

    print("[nmp-dashboard] Ctrl-C to stop")
    dispatchMain()
}

if dashboardArguments.engine == "llamaCpp" {
    guard let modelPath = dashboardArguments.modelPath else {
        FileHandle.standardError.write(Data("--engine llamaCpp requires --model path.gguf\n".utf8))
        exit(2)
    }
    do {
        let model = try NMPLlamaModel(
            modelPath: modelPath, gpuLayers: dashboardArguments.gpuLayers)
        runLlamaDashboard(model: model, arguments: dashboardArguments)
    } catch {
        print("⚠️  llamaCpp engine unavailable (\(error)) — falling back to reference mesh")
    }
}

// MARK: - Mesh

print("[nmp-dashboard] assembling mesh (coordinator + 3 peers)…")
let testbed: NMPMeshTestbed
do {
    // 2 ms/layer simulated compute so stage progress is visible by eye.
    testbed = try NMPMeshTestbed(
        layerCount: 24, hiddenSize: 1024, remotePeerCount: 3,
        simulatedSecondsPerLayer: 0.002)
    let plan: [NMPShardPlanEntry]
    if dashboardArguments.autoConfig {
        // Phase 9: benchmark-driven balanced shards + optimized wire format.
        let setup = NMPAutomaticMeshSetup(
            failover: testbed.failover, orchestrator: testbed.orchestrator,
            modelTag: testbed.modelTag, engineName: "reference")
        setup.onDiagnostic = { print("[auto-config] \($0)") }
        print("[auto-config] starting automatic mesh setup…")
        let report = try setup.runSync(
            probePasses: dashboardArguments.probePasses,
            makeProbeInput: { testbed.makeInput(seed: UInt64($0)) })
        plan = report.adaptive.plan
    } else {
        plan = try testbed.startSync()
    }
    print("[nmp-dashboard] mesh live, \(plan.count) shards: "
          + plan.map { "0x\(String($0.peerID, radix: 16))→L\($0.startLayer)..<\($0.endLayer)" }
                .joined(separator: " "))
} catch {
    fatalError("mesh assembly failed: \(error)")
}

// MARK: - Server

let server = NMPDashboardServer()
server.onDiagnostic = { print("[nmp-dashboard] \($0)") }
do {
    try server.start(port: port)
} catch {
    fatalError("dashboard server failed to start: \(error)")
}
print("[nmp-dashboard] dashboard running at http://localhost:\(server.boundPort)")

// MARK: - Wiring: mesh → dashboard

testbed.onPacketEvent = { peerID, event in
    server.reportPacketEvent(event, peerID: peerID)
}

let stateQueue = DispatchQueue(label: "nmp.dashboard.cli")
var latestMetrics: [UInt32: NMPPeerMetrics] = [:]
var lossRate = 0.0
var benchmarkRunning = false

testbed.orchestrator.onPeerMetrics = { metrics in
    stateQueue.async { latestMetrics[metrics.peerID] = metrics }
}

func assignmentLabel(for peerID: UInt32) -> String {
    guard let entry = testbed.failover.activePlan.first(where: { $0.peerID == peerID }) else {
        return "—"
    }
    return "layers \(entry.startLayer)-\(entry.endLayer - 1)"
}

func pushPeerStates() {
    stateQueue.async {
        let livePeerIDs = Set(testbed.failover.activePeers.map(\.peerID))
        for caps in testbed.failover.activePeers {
            let metrics = latestMetrics[caps.peerID]
            server.updatePeerState(
                peerID: caps.peerID,
                name: caps.deviceName,
                latencyMS: Int((metrics.map { Double($0.inferenceLatencyMicros) / 1000 }) ?? 0),
                loadPercent: Int(metrics?.currentLoadPercent ?? 0),
                assigned: assignmentLabel(for: caps.peerID),
                alive: true)
        }
        for (peerID, metrics) in latestMetrics where !livePeerIDs.contains(peerID) {
            server.updatePeerState(
                peerID: peerID,
                name: "testbed-\(String(peerID, radix: 16))",
                latencyMS: Int(Double(metrics.inferenceLatencyMicros) / 1000),
                loadPercent: 0, assigned: "—", alive: false)
        }
    }
}

testbed.failover.onPeerDropped = { peerID in
    server.reportMeshEvent("peer 0x\(String(peerID, radix: 16)) LOST — re-sharding…")
}
testbed.failover.onResharded = { plan, seconds in
    server.reportMeshEvent(String(
        format: "re-sharded to %d shard(s) in %.1f ms", plan.count, seconds * 1000))
    pushPeerStates()
}

// MARK: - Wiring: dashboard → mesh

server.onControl = { control in
    switch control {
    case .setLossRate(let rate):
        stateQueue.async { lossRate = rate }
        testbed.setLossRate(rate)
        server.reportLossRate(rate)
        server.reportMeshEvent(String(format: "loss rate set to %.0f%%", rate * 100))

    case .injectPeerDrop:
        guard let victim = testbed.remotePeers.last?.capabilities.peerID else {
            server.reportMeshEvent("no remote peer left to drop")
            return
        }
        server.reportMeshEvent("💣 injecting drop of peer 0x\(String(victim, radix: 16))")
        DispatchQueue.global().async {
            do {
                _ = try testbed.dropPeerSync(victim)
            } catch {
                server.reportMeshEvent("failover failed: \(error)")
            }
        }

    case .startBenchmark:
        stateQueue.async {
            guard !benchmarkRunning else { return }
            benchmarkRunning = true
            DispatchQueue.global().async {
                server.reportMeshEvent("📊 benchmark started (loss sweep, 8 tokens)")
                let suite = NMPBenchmarkSuite(testbed: testbed)
                suite.log = { server.reportMeshEvent($0) }
                do {
                    for rate in [0.0, 0.02, 0.05, 0.10, 0.15] {
                        let result = try suite.benchmark(
                            name: String(format: "loss %.0f%%", rate * 100),
                            generations: 5, tokens: 8, lossRate: rate)
                        server.reportBenchmarkResult(result)
                    }
                    server.reportMeshEvent("📊 benchmark complete")
                } catch {
                    server.reportMeshEvent("benchmark failed: \(error)")
                }
                stateQueue.async { benchmarkRunning = false }
                testbed.setLossRate(lossRate) // restore slider setting
            }
        }

    case .resetMetrics:
        for peer in testbed.remotePeers {
            peer.coordinatorInjector.reset()
            peer.peerInjector.reset()
        }
        stateQueue.async { latestMetrics.removeAll() }
        testbed.setLossRate(lossRate)
        server.reportMeshEvent("metrics reset")
    }
}

// MARK: - Wiring: REST inference → mesh

// POST /api/inference: prompt in, mesh-generated tokens out. The heartbeat
// loop below pauses while a request runs so its passes don't interleave
// with (and inflate) the request's per-token timings.
var apiInferenceRunning = false

let promptService = NMPPromptInferenceService(
    orchestrator: testbed.orchestrator, hiddenSize: testbed.hiddenSize)
promptService.onProgress = { done, total in
    server.updateInferenceProgress(
        progress: Double(done) / Double(max(total, 1)),
        stage: "API generation: token \(done)/\(total)")
}

server.onInferenceRequest = { request, respond in
    stateQueue.async { apiInferenceRunning = true }
    server.reportMeshEvent("🌐 API inference: up to \(request.maxTokens) token(s)")
    promptService.run(prompt: request.prompt, maxTokens: request.maxTokens) { result in
        stateQueue.async { apiInferenceRunning = false }
        switch result {
        case .success(let generation):
            server.reportMeshEvent(String(
                format: "🌐 API inference done: %d tokens in %.1f ms across %d shard(s)",
                generation.tokenCount, generation.totalSeconds * 1000,
                generation.shardCount))
            respond(.success(generation))
        case .failure(.busy):
            respond(.failure(status: 429, message: "an inference is already running — retry shortly"))
        case .failure(.emptyPrompt):
            respond(.failure(status: 400, message: "prompt is empty"))
        case .failure(.codec(let reason)):
            respond(.failure(status: 400, message: "prompt encoding failed: \(reason)"))
        case .failure(.orchestration(let error)):
            server.reportMeshEvent("🌐 API inference failed: \(error)")
            respond(.failure(status: 500, message: "mesh orchestration failed: \(error)"))
        }
    }
}

// MARK: - Background inference loop (the dashboard's heartbeat)

let inferenceQueue = DispatchQueue(label: "nmp.dashboard.inference")
func runInferenceLoop() {
    inferenceQueue.async {
        var generation = 0
        while true {
            var skip = false
            stateQueue.sync { skip = benchmarkRunning || apiInferenceRunning }
            if skip {
                Thread.sleep(forTimeInterval: 0.5)
                continue
            }
            let stages = testbed.failover.activePlan.count
            do {
                server.updateInferenceProgress(progress: 0, stage: "generation \(generation)…")
                // Short stage timeout: a freshly-killed peer costs ~2 s
                // (timeout + one retry) instead of freezing the dashboard.
                let report = try testbed.inferSync(
                    input: testbed.makeInput(seed: UInt64(generation)),
                    stageTimeout: 1.0)
                for (index, shard) in report.perShard.enumerated() {
                    server.updateInferenceProgress(
                        progress: Double(index + 1) / Double(max(1, stages)),
                        stage: "shard \(shard.shardIndex) on 0x\(String(shard.peerID, radix: 16)) — "
                            + String(format: "%.1f ms", shard.stageSeconds * 1000))
                }
                server.updateInferenceProgress(
                    progress: 1.0,
                    stage: String(format: "generation %d done in %.1f ms",
                                  generation, report.totalSeconds * 1000))
            } catch {
                server.reportMeshEvent("inference failed: \(error)")
                Thread.sleep(forTimeInterval: 1.0)
            }
            pushPeerStates()
            generation += 1
            Thread.sleep(forTimeInterval: 0.4)
        }
    }
}

// Health-based auto-failover runs alongside the manual drop button: the
// inference loop is the heartbeat traffic it feeds on.
testbed.failover.startHealthChecks()

pushPeerStates()
runInferenceLoop()
print("[nmp-dashboard] Ctrl-C to stop")
dispatchMain()
