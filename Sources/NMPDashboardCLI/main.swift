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
    var portExplicit = false
    var engine = "reference"
    var modelPath: String?
    var gpuLayers: Int32 = -1
    var placement = NMPLlamaTestbed.Placement.remotePeer
    // Mesh 2.0
    /// Serve the React web UI from Public/, advertise over Bonjour, and
    /// print the multi-device access banner (QR + hostname + IPs).
    /// Defaults the port to 3000 unless one is given explicitly.
    var ui = false
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
    // Mesh 2.8
    /// In-process simulated peers to seed the mesh with (demo). Real LAN
    /// peers (the iPhone app) auto-retire these when they join, so the
    /// mesh becomes genuinely cross-device.
    var simPeers = 3
    /// Declared model size in GB. The reference engine is weightless, so
    /// this is the *modeled* memory footprint used for capacity-aware
    /// sharding — set it to your target model (e.g. 9 for Qwen3-14B q4) to
    /// see layers forced across devices. 0 = unbounded (no capacity limit).
    var modelGB = 0.0

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
                arguments.placement = value().flatMap { val in
                    if val == "local" {
                        return .local
                    } else if val == "remote" || val == "remotePeer" {
                        return .remotePeer
                    } else if val == "sharded" {
                        return .sharded(shardCount: 2)
                    } else if val.hasPrefix("sharded:"), let count = Int(val.dropFirst(8)) {
                        return .sharded(shardCount: count)
                    }
                    return nil
                } ?? arguments.placement
            case "--auto-config": arguments.autoConfig = true
            case "--speculation": arguments.speculation = true
            case "--draft-model": arguments.draftModelPath = value()
            case "--probe-passes":
                arguments.probePasses = value().flatMap(Int.init) ?? arguments.probePasses
            case "--sim-peers":
                arguments.simPeers = value().flatMap(Int.init) ?? arguments.simPeers
            case "--model-gb":
                arguments.modelGB = value().flatMap(Double.init) ?? arguments.modelGB
            case "--ui": arguments.ui = true
            case "--help", "-h":
                print("""
                usage: nmp-dashboard [port] [--engine reference|llamaCpp] \
                [--model path.gguf] [--gpu-layers N] [--placement local|remote] \
                [--auto-config] [--probe-passes N] [--speculation] [--draft-model path.gguf] \
                [--ui]
                """)
                exit(0)
            default:
                if let parsed = UInt16(flag) {
                    arguments.port = parsed
                    arguments.portExplicit = true
                } else {
                    FileHandle.standardError.write(Data("unknown flag \(flag)\n".utf8))
                    exit(2)
                }
            }
        }
        if arguments.ui && !arguments.portExplicit {
            arguments.port = 3000
        }
        return arguments
    }
}

// MARK: - Mesh 2.0 web UI wiring (shared by both engine paths)

/// Bonjour advert must outlive this scope.
var webUIBroadcaster: NMPWebUIBroadcaster?

/// Locates the built React app (web/ → Public/): next to the CWD when
/// running from the package root, or alongside the binary.
func locatePublicDirectory() -> URL? {
    let candidates = [
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("Public"),
        Bundle.main.bundleURL.deletingLastPathComponent()
            .appendingPathComponent("Public"),
    ]
    return candidates.first {
        FileManager.default.fileExists(
            atPath: $0.appendingPathComponent("index.html").path)
    }
}

/// Sequential benchmark runs over the single-flight prompt service.
func wireBenchmark(server: NMPDashboardServer,
                   run: @escaping (String, Int,
                       @escaping (Result<NMPPromptInferenceService.GenerationResult,
                                         NMPPromptInferenceService.ServiceError>) -> Void) -> Void) {
    server.onBenchmarkRequest = { request, respond in
        var results: [NMPPromptInferenceService.GenerationResult] = []
        func step() {
            guard results.count < request.runs else {
                respond(.success(results))
                return
            }
            run(request.prompt, request.maxTokens) { result in
                switch result {
                case .success(let generation):
                    results.append(generation)
                    step()
                case .failure(let error):
                    respond(.failure(.init("run \(results.count + 1) failed: \(error)")))
                }
            }
        }
        step()
    }
}

/// Mesh 2.1: one host resource monitor for /api/devices/metrics. Primed
/// immediately so the first poll already has a CPU tick baseline.
let resourceMonitor = NMPResourceMonitor()
_ = resourceMonitor.sample()

/// Mesh 2.1: POST /api/comparison/run — one real generation, then the
/// transport race replaying its traffic pattern (round trips × payload)
/// over real loopback sockets: full NMP stack vs plain kernel TCP.
func wireComparisonRun(server: NMPDashboardServer,
                       run: @escaping (NMPDashboardServer.InferenceRequest,
                           @escaping (Result<NMPPromptInferenceService.GenerationResult,
                                             NMPPromptInferenceService.ServiceError>) -> Void) -> Void) {
    server.onComparisonRunRequest = { request, respond in
        server.reportMeshEvent("🏁 protocol race: generation, then measured "
                               + "NMP vs TCP transport replay")
        run(request) { result in
            switch result {
            case .failure(let error):
                respond(.failure(.init("generation failed: \(error)")))
            case .success(let generation):
                let trips = generation.speculation?.meshRoundTrips
                    ?? generation.perTokenSeconds.count
                guard generation.networkPayloadBytes >= 2, trips >= 1 else {
                    respond(.failure(.init(
                        "this run moved no mesh traffic (local placement?) — "
                            + "nothing to race")))
                    return
                }
                NMPTransportRace.run(plan: .init(
                    roundTrips: trips,
                    payloadBytes: generation.networkPayloadBytes)) { raceResult in
                    switch raceResult {
                    case .success(let race):
                        let legs = race.legs
                            .map { String(format: "%@ %.1f ms (handshake %.2f)",
                                          $0.name, $0.totalMs, $0.handshakeMs) }
                            .joined(separator: " vs ")
                        server.reportMeshEvent(
                            "🏁 race done over \(trips) trip(s) × "
                            + "\(generation.networkPayloadBytes) B: \(legs) "
                            + "— all measured")
                        respond(.success(.init(generation: generation, race: race)))
                    case .failure(let error):
                        respond(.failure(.init("transport race failed: \(error)")))
                    }
                }
            }
        }
    }
}

/// Serves Public/, registers the Bonjour advert, prints the banner.
func activateWebUI(server: NMPDashboardServer, meshSummary: [String]) {
    if let publicDirectory = locatePublicDirectory() {
        server.publicDirectory = publicDirectory
        print("[nmp-dashboard] web UI: \(publicDirectory.path)")
    } else {
        print("⚠️  Public/index.html not found — serving the legacy dashboard.")
        print("    Build the web UI first: cd web && npm install && npm run build")
    }
    let broadcaster = NMPWebUIBroadcaster(port: server.boundPort)
    broadcaster.onDiagnostic = { print("[nmp-dashboard] \($0)") }
    broadcaster.start()
    webUIBroadcaster = broadcaster
    print(NMPWebUIBanner.render(port: server.boundPort, meshSummary: meshSummary))
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
            engine: engine, modelTag: model.name, placement: placement,
            engineFactory: {
                guard let path = arguments.modelPath else {
                    throw NMPLlamaRuntimeError.weightsNotLoaded
                }
                let m = try NMPLlamaModel(modelPath: path)
                return NMPLlamaComputeEngine(model: m)
            }
        )
        let plan = try testbed.startSync()
        print("[nmp-dashboard] llama mesh live: \(plan.count) shard(s) active")
        for entry in plan {
            let desc = entry.peerID == testbed.coordinatorID ? "coordinator (local)" : "in-process peer-\(entry.shardIndex) (full stack)"
            print("  - shard \(entry.shardIndex): layers \(entry.startLayer)..<\(entry.endLayer) on \(desc)")
        }
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
    testbed.onInferenceServed = { peerID, layers, seconds in
        let name = peerID == testbed.coordinatorID ? "coordinator (local)" : "llama-peer-\(peerID) (in-process)"
        server.updatePeerState(
            peerID: peerID, name: name,
            latencyMS: Int(seconds * 1000), loadPercent: 0,
            assigned: "layers \(layers.lowerBound)-\(layers.upperBound - 1)",
            alive: true)
    }
    server.updatePeerState(
        peerID: testbed.coordinatorID, name: "coordinator (tokenizer)",
        latencyMS: 0, loadPercent: 0,
        assigned: "—",
        alive: true)
    for entry in testbed.plan {
        let name = entry.peerID == testbed.coordinatorID ? "coordinator (tokenizer + local compute)" : "llama-peer-\(entry.shardIndex) (in-process)"
        server.updatePeerState(
            peerID: entry.peerID, name: name,
            latencyMS: 0, loadPercent: 0,
            assigned: "layers \(entry.startLayer)-\(entry.endLayer - 1)",
            alive: true)
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

    // Mesh 2.1: stream every confirmed token to every open browser.
    let streamToken: (NMPGeneratedToken, Int, Int) -> Void = { token, count, requested in
        server.reportGenerationToken(text: token.text, index: token.index,
                                     count: count, requested: requested)
    }
    promptService.onToken = streamToken
    speculativeService?.onToken = streamToken

    // Mesh 2.1: in-flight flag for /api/devices/metrics.
    let llamaStateQueue = DispatchQueue(label: "nmp.dashboard.llama.state")
    var llamaGenerationRunning = false
    var llamaLastServedAt: Date?
    // Mesh 2.3: serve count + per-link throughput for the device panel.
    var llamaRequestsServed = 0
    var llamaTrafficBaseline: (sent: UInt64, received: UInt64, at: Date)?
    var llamaNetRates: (toDeviceBps: Int, fromDeviceBps: Int)?
    testbed.onInferenceServed = { [existing = testbed.onInferenceServed] id, layers, seconds in
        llamaStateQueue.async {
            llamaLastServedAt = Date()
            llamaRequestsServed += 1
        }
        existing?(id, layers, seconds)
    }

    server.onInferenceRequest = { request, respond in
        llamaStateQueue.async { llamaGenerationRunning = true }
        server.reportGenerationStarted(
            prompt: request.prompt, maxTokens: request.maxTokens,
            speculative: (request.enableSpeculation || dashboardArguments.speculation)
                && speculativeService != nil)
        let handleResult: (Result<NMPPromptInferenceService.GenerationResult,
                                  NMPPromptInferenceService.ServiceError>) -> Void = { result in
            llamaStateQueue.async { llamaGenerationRunning = false }
            if case .success(let generation) = result {
                server.reportGenerationComplete(generation)
            } else if case .failure(let error) = result {
                server.reportGenerationFailed(String(describing: error))
            }
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

    // Mesh 2.0: health/devices facts + benchmark + multi-device web UI.
    var meshInfo = NMPDashboardServer.MeshInfo()
    meshInfo.engine = "llamaCpp"
    meshInfo.modelName = model.name
    meshInfo.shardCount = 1
    meshInfo.wireFormat = testbed.orchestrator.activationWireFormat.rawValue
    meshInfo.speculationAvailable = speculativeService != nil
    server.meshInfo = meshInfo

    wireBenchmark(server: server) { prompt, maxTokens, completion in
        if dashboardArguments.speculation, let speculativeService {
            speculativeService.run(prompt: prompt, maxTokens: maxTokens,
                                   completion: completion)
        } else {
            promptService.run(prompt: prompt, maxTokens: maxTokens,
                              completion: completion)
        }
    }

    // Mesh 2.1: measured protocol race (generation + transport replay).
    wireComparisonRun(server: server) { request, completion in
        if request.enableSpeculation || dashboardArguments.speculation,
           let speculativeService {
            speculativeService.run(prompt: request.prompt,
                                   maxTokens: request.maxTokens,
                                   completion: completion)
        } else {
            promptService.run(prompt: request.prompt,
                              maxTokens: request.maxTokens,
                              completion: completion)
        }
    }

    // Mesh 2.1: live device metrics. Both mesh members are in-process,
    // so they genuinely share this host — the payload says so instead of
    // inventing per-peer hardware.
    server.onDeviceMetricsRequest = { respond in
        llamaStateQueue.async {
            let sample = resourceMonitor.sample()
            let now = Date()
            let computingNow = llamaGenerationRunning
                || (llamaLastServedAt.map { now.timeIntervalSince($0) < 1.0 } ?? false)

            // Live link throughput (remotePeer only) — same 0.5 s-window
            // diffing as the reference path.
            if let totals = testbed.wireTraffic {
                if let baseline = llamaTrafficBaseline {
                    let elapsed = now.timeIntervalSince(baseline.at)
                    if elapsed >= 0.5 {
                        llamaNetRates = (
                            toDeviceBps: Int(Double(totals.sentBytes - baseline.sent) / elapsed),
                            fromDeviceBps: Int(Double(totals.receivedBytes - baseline.received) / elapsed))
                        llamaTrafficBaseline = (totals.sentBytes, totals.receivedBytes, now)
                    }
                } else {
                    llamaTrafficBaseline = (totals.sentBytes, totals.receivedBytes, now)
                }
            }

            var peers: [[String: Any]] = [
                [
                    "id": String(testbed.coordinatorID, radix: 16),
                    "name": "coordinator (tokenizer)",
                    "alive": true,
                    "assigned": placement == .local
                        ? "layers 0-\(engine.layerCount - 1)" : "tokenizer only",
                    "compute_share": 1.0,
                    "computing": computingNow && placement == .local,
                    "is_coordinator": true,
                    "link": "local — no network hop",
                ],
            ]
            if placement == .remotePeer {
                var peer: [String: Any] = [
                    "id": String(testbed.peerID, radix: 16),
                    "name": "llama-peer (in-process)",
                    "alive": true,
                    "assigned": "layers 0-\(engine.layerCount - 1)",
                    "compute_share": 1.0,
                    "computing": computingNow,
                    "is_coordinator": false,
                    "link": "in-process link (full NMP stack: Noise IK, AES-GCM, FEC, NACK)",
                    "requests_served": llamaRequestsServed,
                ]
                if let totals = testbed.wireTraffic {
                    peer["wire_in_mb"] =
                        (Double(totals.sentBytes) / 1_048_576 * 100).rounded() / 100
                    peer["wire_out_mb"] =
                        (Double(totals.receivedBytes) / 1_048_576 * 100).rounded() / 100
                }
                if let rates = llamaNetRates {
                    peer["net_in_bytes_per_sec"] = rates.toDeviceBps
                    peer["net_out_bytes_per_sec"] = rates.fromDeviceBps
                }
                peers.append(peer)
            }
            respond([
                "host": sample.asJSONObject,
                "host_note": "all mesh peers run in-process — these are "
                    + "genuinely this machine's live kernel counters "
                    + "(watch process_footprint_mb and gpu_percent during "
                    + "a generation: llama.cpp computes on the GPU via Metal)",
                "generation_in_flight": llamaGenerationRunning,
                "allocation_supported": false,
                "allocation_note": "a llama plan is one full-range shard "
                    + "(llama.cpp cannot split layers), so there is nothing "
                    + "to re-balance — compute shares apply to the "
                    + "reference mesh",
                "peers": peers,
                "totals": [
                    "devices": peers.count,
                    "devices_alive": peers.count,
                    "layers_assigned": engine.layerCount,
                    "requests_served": llamaRequestsServed,
                    "net_bytes_per_sec": llamaNetRates.map { $0.toDeviceBps + $0.fromDeviceBps } ?? 0,
                    "generation_in_flight": llamaGenerationRunning,
                ] as [String: Any],
            ])
        }
    }

    server.onAllocationRequest = { _, _, respond in
        respond(.failure(.init(
            "allocation needs a multi-shard mesh — a llama plan is one "
                + "full-range shard by construction (llama.cpp cannot run "
                + "layer sub-ranges); run the reference mesh to see "
                + "allocation re-shard live")))
    }

    if dashboardArguments.ui {
        activateWebUI(server: server, meshSummary: [
            "Mesh: llamaCpp — \(model.name)",
            "Placement: " + (placement == .local
                ? "single device" : "remote shard (full protocol stack)"),
            "Wire format: \(meshInfo.wireFormat)"
                + (meshInfo.speculationAvailable ? ", speculation ready" : ""),
        ])
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

print("[nmp-dashboard] assembling mesh (coordinator + \(dashboardArguments.simPeers) "
      + "in-process peer(s); LAN peers join live via Bonjour and retire the "
      + "simulated ones)…")
let testbed: NMPMeshTestbed
do {
    // 2 ms/layer simulated compute so stage progress is visible by eye.
    // Mesh 2.4: the shape and tag MATCH the peer app / nmp-peer defaults
    // (32 × 4096, 'nmp-reference-model') so a discovered LAN peer accepts
    // the dashboard's SHARD_ASSIGN instead of rejecting on model mismatch.
    // heartbeatTimeout 12 s (not the 5 s default): with a phone in the
    // mesh, one slow Wi-Fi stage or a briefly backgrounded app must not
    // read as everyone else's death. Keepalive pings (Mesh 2.4) answer
    // for idle-but-alive peers within that window.
    testbed = try NMPMeshTestbed(
        layerCount: 32, hiddenSize: 4096, remotePeerCount: dashboardArguments.simPeers,
        modelTag: "nmp-reference-model",
        simulatedSecondsPerLayer: 0.002,
        heartbeatTimeout: 12)
    // Mesh 2.8: capacity-aware sharding. The reference engine is weightless,
    // so --model-gb declares a MODELED per-layer footprint (labeled as such)
    // — set it to your target model to see layers forced across devices.
    if dashboardArguments.modelGB > 0 {
        let bytesPerLayer = Int(dashboardArguments.modelGB * 1_000_000_000
                                / Double(testbed.layerCount))
        testbed.failover.bytesPerLayer = bytesPerLayer
        print(String(format: "[nmp-dashboard] modeled model footprint: %.1f GB "
                     + "(%d MB/layer) — capacity-aware sharding active",
                     dashboardArguments.modelGB, bytesPerLayer / 1_048_576))
    }
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

// Mesh 2.3/2.4: per-device telemetry, all measured.
// - latestResources: each peer's own kernel-counter sample, shipped over
//   the mesh (in-process peers report this same host — the hostname match
//   tells the UI to say so instead of pretending they are separate boxes;
//   a LAN peer reports its own device and gets real bars).
// - serveStats: requests actually served per peer. Counted from the
//   metrics message every shard engine sends after each serve — the one
//   signal that arrives identically from in-process AND LAN peers.
// - trafficBaselines/latestNetRates: coordinator-side wire totals per
//   link, diffed between polls into live bytes/sec.
var serveStats: [UInt32: (served: Int, lastComputeSeconds: Double, lastAt: Date)] = [:]
func recordServe(peerID: UInt32, computeSeconds: Double) {
    stateQueue.async {
        serveStats[peerID] = ((serveStats[peerID]?.served ?? 0) + 1,
                              computeSeconds, Date())
    }
}

testbed.orchestrator.onPeerMetrics = { metrics in
    stateQueue.async { latestMetrics[metrics.peerID] = metrics }
    recordServe(peerID: metrics.peerID,
                computeSeconds: Double(metrics.inferenceLatencyMicros) / 1e6)
}

var latestResources: [UInt32: (report: NMPPeerResourceReport, at: Date)] = [:]
testbed.orchestrator.onPeerResourceReport = { report in
    stateQueue.async { latestResources[report.peerID] = (report, Date()) }
}

var trafficBaselines: [UInt32: (sent: UInt64, received: UInt64, at: Date)] = [:]
var latestNetRates: [UInt32: (toDeviceBps: Int, fromDeviceBps: Int)] = [:]

// Mesh 2.4: coordinator-side connections to REAL LAN peers (iPhone app,
// nmp-peer on another Mac), keyed by peerID. Mutated on stateQueue.
var lanConnections: [UInt32: PeerConnection] = [:]

// Names of every peer ever seen, so a dropped peer can stay visible in
// the panel (greyed out) instead of vanishing without explanation.
var knownPeerNames: [UInt32: String] = [:]

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
            // A dropped LAN peer keeps its real device name ("iPhone 17
            // Pro"), not a synthetic testbed label — the card is how the
            // user recognizes which physical device left.
            server.updatePeerState(
                peerID: peerID,
                name: knownPeerNames[peerID]
                    ?? "testbed-\(String(peerID, radix: 16))",
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
// Mesh 2.1: stream every generated token to every open browser.
promptService.onToken = { token, count, requested in
    server.reportGenerationToken(text: token.text, index: token.index,
                                 count: count, requested: requested)
}

server.onInferenceRequest = { request, respond in
    stateQueue.async { apiInferenceRunning = true }
    server.reportMeshEvent("🌐 API inference: up to \(request.maxTokens) token(s)")
    server.reportGenerationStarted(prompt: request.prompt,
                                   maxTokens: request.maxTokens,
                                   speculative: false)
    promptService.run(prompt: request.prompt, maxTokens: request.maxTokens) { result in
        stateQueue.async { apiInferenceRunning = false }
        if case .success(let generation) = result {
            server.reportGenerationComplete(generation)
        } else if case .failure(let error) = result {
            server.reportGenerationFailed(String(describing: error))
        }
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

// MARK: - Mesh 2.0: web UI + health + benchmark (reference mesh)

var referenceMeshInfo = NMPDashboardServer.MeshInfo()
referenceMeshInfo.engine = "reference"
referenceMeshInfo.modelName = testbed.modelTag
referenceMeshInfo.shardCount = testbed.failover.activePlan.count
referenceMeshInfo.wireFormat = testbed.orchestrator.activationWireFormat.rawValue
server.meshInfo = referenceMeshInfo

// Benchmark runs pause the heartbeat loop exactly like API inference does.
wireBenchmark(server: server) { prompt, maxTokens, completion in
    stateQueue.async { apiInferenceRunning = true }
    promptService.run(prompt: prompt, maxTokens: maxTokens) { result in
        stateQueue.async { apiInferenceRunning = false }
        completion(result)
    }
}

// Mesh 2.1: measured protocol race. The heartbeat pauses for the whole
// run (generation + race) so its passes pollute neither measurement.
wireComparisonRun(server: server) { request, completion in
    stateQueue.async { apiInferenceRunning = true }
    promptService.run(prompt: request.prompt,
                      maxTokens: request.maxTokens) { result in
        // Heartbeat resumes when the API reply goes out; the race itself
        // is loopback-only but keeping the mesh quiet keeps timings clean.
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.1) {
            stateQueue.async { apiInferenceRunning = false }
        }
        completion(result)
    }
}

// Mesh 2.1/2.3: live device metrics — host kernel counters plus per-peer
// mesh facts (assignment, measured speed, compute share, liveness) plus
// per-device telemetry: peer-reported resources, live wire throughput,
// requests served, and mesh totals. Every number is measured; the labels
// say where it was measured.
server.onDeviceMetricsRequest = { respond in
    stateQueue.async {
        let sample = resourceMonitor.sample()
        let shares = testbed.orchestrator.computeShares
        let measured = testbed.orchestrator.measuredSecondsPerLayer
        let inFlight = apiInferenceRunning || benchmarkRunning
        let alivePeerIDs = Set(testbed.failover.activePeers.map(\.peerID))
        let planned = Dictionary(uniqueKeysWithValues:
            testbed.failover.activePlan.map { ($0.peerID, $0) })
        // Mesh 2.8: why each 0-layer device holds nothing, for the UI.
        let exclusionReasons = Dictionary(uniqueKeysWithValues:
            testbed.failover.activeExclusions.map { ($0.peerID, $0.reason) })
        var connections = Dictionary(uniqueKeysWithValues:
            testbed.remotePeers.map { ($0.capabilities.peerID, $0.coordinatorSide) })
        for (peerID, connection) in lanConnections {
            connections[peerID] = connection
        }
        let now = Date()

        // Refresh per-link throughput from the connections' wire totals.
        // Rates only update when ≥0.5 s has passed since the last baseline
        // (several browsers polling at once would otherwise shrink the
        // window into noise); between updates the last rate is served.
        for (peerID, connection) in connections {
            let totals = connection.trafficTotals
            guard let baseline = trafficBaselines[peerID] else {
                trafficBaselines[peerID] = (totals.sentBytes, totals.receivedBytes, now)
                continue
            }
            let elapsed = now.timeIntervalSince(baseline.at)
            guard elapsed >= 0.5 else { continue }
            latestNetRates[peerID] = (
                toDeviceBps: Int(Double(totals.sentBytes - baseline.sent) / elapsed),
                fromDeviceBps: Int(Double(totals.receivedBytes - baseline.received) / elapsed))
            trafficBaselines[peerID] = (totals.sentBytes, totals.receivedBytes, now)
        }

        var peers = testbed.failover.activePeers.map { caps -> [String: Any] in
            knownPeerNames[caps.peerID] = caps.deviceName
            let entry = planned[caps.peerID]
            let stats = serveStats[caps.peerID]
            let secondsSinceServe = stats.map { now.timeIntervalSince($0.lastAt) }
            var peer: [String: Any] = [
                "id": String(caps.peerID, radix: 16),
                "name": caps.deviceName,
                "alive": alivePeerIDs.contains(caps.peerID),
                "assigned": entry.map {
                    "layers \($0.startLayer)-\($0.endLayer - 1)"
                } ?? "0 shards — standing by",
                "layer_span": entry?.layerSpan ?? 0,
                "excluded": entry == nil,
                "exclusion_reason": entry == nil
                    ? (exclusionReasons[caps.peerID]
                        ?? "0 shards: not used by the current plan.")
                    : "",
                "compute_share": shares[caps.peerID] ?? 1.0,
                // Served a request in the last 1.5 s (the heartbeat runs a
                // pass every ~0.4 s, so an assigned live peer stays hot).
                "computing": entry != nil
                    && (secondsSinceServe.map { $0 < 1.5 } ?? false),
                "load_percent": Int(latestMetrics[caps.peerID]?.currentLoadPercent ?? 0),
                "is_coordinator": caps.peerID == testbed.coordinatorID,
                "link": lanConnections[caps.peerID] != nil
                    ? "Wi-Fi/LAN link — real UDP (Noise IK, AES-GCM, FEC, NACK)"
                    : connections[caps.peerID] != nil
                        ? "in-process link (full NMP stack: Noise IK, AES-GCM, FEC, NACK)"
                        : "local shard on the coordinator — no network hop",
            ]
            if let rate = measured[caps.peerID] {
                peer["measured_ms_per_layer"] = (rate * 1000 * 100).rounded() / 100
            }
            if let stats {
                peer["requests_served"] = stats.served
                peer["last_compute_ms"] =
                    (stats.lastComputeSeconds * 1000 * 100).rounded() / 100
                peer["seconds_since_active"] =
                    ((secondsSinceServe ?? 0) * 10).rounded() / 10
            }
            if let connection = connections[caps.peerID] {
                let totals = connection.trafficTotals
                // Device perspective: "in" = what the coordinator sent it.
                peer["wire_in_mb"] =
                    (Double(totals.sentBytes) / 1_048_576 * 100).rounded() / 100
                peer["wire_out_mb"] =
                    (Double(totals.receivedBytes) / 1_048_576 * 100).rounded() / 100
                if let rates = latestNetRates[caps.peerID] {
                    peer["net_in_bytes_per_sec"] = rates.toDeviceBps
                    peer["net_out_bytes_per_sec"] = rates.fromDeviceBps
                }
            }
            if let (report, at) = latestResources[caps.peerID] {
                var resources: [String: Any] = [
                    "hostname": report.hostname,
                    "same_host_as_coordinator": report.hostname == sample.hostname,
                    "age_seconds": (now.timeIntervalSince(at) * 10).rounded() / 10,
                    "ram_total_mb": Int(report.ramTotalMB),
                    "ram_used_mb": Int(report.ramUsedMB),
                    "ram_used_percent": report.ramTotalMB > 0
                        ? (Double(report.ramUsedMB) / Double(report.ramTotalMB)
                            * 1000).rounded() / 10 : 0,
                    "process_footprint_mb": Int(report.processFootprintMB),
                    "storage_total_gb":
                        (Double(report.storageTotalMB) / 1024 * 10).rounded() / 10,
                    "storage_free_gb":
                        (Double(report.storageFreeMB) / 1024 * 10).rounded() / 10,
                    "storage_used_percent": report.storageTotalMB > 0
                        ? (Double(report.storageTotalMB - report.storageFreeMB)
                            / Double(report.storageTotalMB) * 1000).rounded() / 10 : 0,
                ]
                if let cpu = report.cpuPercent {
                    resources["cpu_percent"] = (cpu * 10).rounded() / 10
                }
                if let gpu = report.gpuPercent {
                    resources["gpu_percent"] = (gpu * 10).rounded() / 10
                }
                peer["resources"] = resources
            }
            return peer
        }

        // Dropped peers stay visible (greyed out) instead of silently
        // vanishing — a peer the health monitor removed mid-session is a
        // fact the panel should show, not hide.
        for (peerID, name) in knownPeerNames where !alivePeerIDs.contains(peerID) {
            peers.append([
                "id": String(peerID, radix: 16),
                "name": name,
                "alive": false,
                "assigned": "dropped from the mesh",
                "layer_span": 0,
                "compute_share": 1.0,
                "computing": false,
                "is_coordinator": false,
                "link": "rejoins automatically when it reappears on the network",
            ])
        }

        let totalLayers = planned.values.reduce(0) { $0 + $1.layerSpan }
        let netToBps = latestNetRates.values.reduce(0) { $0 + $1.toDeviceBps }
        let netFromBps = latestNetRates.values.reduce(0) { $0 + $1.fromDeviceBps }
        let totals: [String: Any] = [
            "devices": peers.count,
            "devices_alive": alivePeerIDs.count,
            "layers_assigned": totalLayers,
            "requests_served": serveStats.values.reduce(0) { $0 + $1.served },
            "net_bytes_per_sec": netToBps + netFromBps,
            "generation_in_flight": inFlight,
        ]

        let objective = testbed.failover.shardingObjective
        let shortfall = testbed.failover.activeShortfall
        respond([
            "host": sample.asJSONObject,
            "host_note": "all mesh peers run in-process — these are "
                + "genuinely this machine's live kernel counters (GPU% is "
                + "the whole machine: the reference engine computes on CPU)",
            "generation_in_flight": inFlight,
            "allocation_supported": true,
            "allocation_note": "compute share re-plans the mesh: a device "
                + "at 50% is planned as half as fast and receives "
                + "proportionally fewer layers — watch 'assigned' change",
            // Mesh 2.8: the layer-distribution strategy, switchable live.
            "sharding_objective": objective.rawValue,
            "sharding_objective_label": objective.label,
            "sharding_objectives": NMPShardingObjective.allCases.map {
                ["value": $0.rawValue, "label": $0.label] as [String: Any]
            },
            "capacity_shortfall": shortfall,
            "capacity_note": shortfall > 0
                ? "⚠️ \(shortfall) layer(s) fit on no device — the model is "
                    + "too big for this mesh. Add a device or a smaller model."
                : "",
            "peers": peers,
            "totals": totals,
        ])
    }
}

// Mesh 2.1: the allocation slider. Sets the peer's compute share, then
// re-shards the live mesh through the normal SHARD_ASSIGN round — the
// layer spans that come back are the proof the allocation took effect.
server.onAllocationRequest = { peerID, share, respond in
    guard testbed.failover.activePeers.contains(where: { $0.peerID == peerID }) else {
        respond(.failure(.init(
            "unknown peer 0x\(String(peerID, radix: 16)) — see GET /api/devices")))
        return
    }
    testbed.orchestrator.setComputeShare(share, forPeer: peerID)
    server.reportMeshEvent(String(
        format: "⚖️ compute share for 0x%@ → %.0f%% — re-sharding…",
        String(peerID, radix: 16), share * 100))
    testbed.failover.replan(reason: "compute share change") { result in
        switch result {
        case .failure(let error):
            respond(.failure(.init("re-plan failed: \(error)")))
        case .success(let plan):
            let names = Dictionary(uniqueKeysWithValues:
                testbed.failover.activePeers.map { ($0.peerID, $0.deviceName) })
            let summary = plan.map {
                "\(names[$0.peerID] ?? "0x" + String($0.peerID, radix: 16)): "
                    + "L\($0.startLayer)-\($0.endLayer - 1)"
            }.joined(separator: ", ")
            server.reportMeshEvent("⚖️ re-sharded: \(summary)")
            pushPeerStates()
            respond(.success(summary))
        }
    }
}

// Mesh 2.8: switch the layer-distribution strategy live from the Devices
// tab. Capacity + Speed (default) spreads across the mesh; Pure Speed
// packs the fastest device (others stand by). Either way it re-shards
// through the normal SHARD_ASSIGN round.
server.onObjectiveRequest = { raw, respond in
    guard let objective = NMPShardingObjective(rawValue: raw) else {
        respond(.failure(.init("unknown objective '\(raw)' — expected one of: "
            + NMPShardingObjective.allCases.map(\.rawValue).joined(separator: ", "))))
        return
    }
    testbed.failover.shardingObjective = objective
    server.reportMeshEvent("⚖️ sharding objective → \(objective.label) — re-sharding…")
    testbed.failover.replan(reason: "objective change") { result in
        switch result {
        case .failure(let error):
            respond(.failure(.init("re-plan failed: \(error)")))
        case .success(let plan):
            let excluded = testbed.failover.activeExclusions.count
            var summary = "\(objective.label): \(plan.count) shard(s)"
            if excluded > 0 { summary += ", \(excluded) standing by" }
            pushPeerStates()
            respond(.success(summary))
        }
    }
}

// MARK: - Mesh 2.4: real LAN peers join the live dashboard mesh
//
// The dashboard browses for _neuramesh._tcp adverts (the iPhone peer app,
// `swift run nmp-peer` on another Mac), dials each one over REAL UDP —
// Noise IK with the static key from the peer's TXT record, the same
// trust-on-first-use model as nmp-coordinator — and joins it into the
// SAME failover mesh this panel displays. The join re-shards live on
// every open browser; the device's card shows its own reported
// RAM/CPU/storage, wire throughput, and serve counter. No flags, no
// configuration: open the peer app on the same Wi-Fi and watch it join.

let lanQueue = DispatchQueue(label: "nmp.dashboard.lan")
let lanStaticKeys = NoiseStaticKeyPair()
let lanBrowser = NMPBonjourBrowser()
let lanDiscovery = NMPPeerDiscoveryManager(
    localCapabilities: NMPSystemCapabilityProbe.measure(peerID: testbed.coordinatorID),
    publisher: nil, // browse-only: the dashboard dials, peers advertise
    source: lanBrowser,
    queue: lanQueue)
/// Peers being dialed or already joined (lanQueue-owned) — one dial per
/// advert, however often Bonjour re-announces it.
var lanDialing = Set<UInt32>()
/// Mesh 2.8: the moment a REAL device joins, retire the in-process
/// simulated peers so the mesh is genuinely cross-device (the phone
/// becomes a true ~50% member instead of competing with fast Mac-local
/// phantoms). One-shot, guarded by stateQueue.
var simulatedPeersRetired = false
func retireSimulatedPeersOnce() {
    stateQueue.async {
        guard !simulatedPeersRetired else { return }
        simulatedPeersRetired = true
        let victims = testbed.remotePeers.map(\.capabilities.peerID)
        guard !victims.isEmpty else { return }
        server.reportMeshEvent("🖥️ real device joined — retiring "
            + "\(victims.count) simulated in-process peer(s); the mesh is now "
            + "genuinely cross-device")
        DispatchQueue.global(qos: .userInitiated).async {
            for victim in victims {
                _ = try? testbed.dropPeerSync(victim)
            }
            pushPeerStates()
        }
    }
}
/// Per-peer re-dial backoff (lanQueue-owned). Doubles on every drop and
/// resets on a successful join: a transient drop recovers in seconds,
/// while TWO coordinators fighting over one peer (each steal fails the
/// other's session — the peer keeps one coordinator) back off instead of
/// stealing it back and forth every few seconds forever.
var lanRedialDelay: [UInt32: TimeInterval] = [:]

/// Dials one discovered peer. Runs on lanQueue.
func dialLANPeer(_ capabilities: NMPCapabilities) {
    let peerID = capabilities.peerID
    guard !lanDialing.contains(peerID) else { return }
    guard capabilities.udpPort != 0,
          let remoteStatic = capabilities.noiseStaticPublicKey else {
        print("[nmp-dashboard] LAN peer \(capabilities.deviceName) not dialable "
              + "(no port/key in TXT) — skipping")
        return
    }
    guard let endpoint = lanDiscovery.discoveredPeers[peerID]?.endpoint else { return }
    lanDialing.insert(peerID)

    server.reportMeshEvent("📱 found \(capabilities.deviceName) "
        + "(\(capabilities.computeClass.label), \(capabilities.ramMB) MB RAM) — dialing…")
    let connectionQueue = DispatchQueue(label: "nmp.dashboard.lan.\(peerID)")
    let transport = UDPTransport(endpoint: endpoint, queue: connectionQueue)
    do {
        let connection = try PeerConnection(
            role: .initiator,
            config: PeerConnectionConfig(localPeerID: testbed.coordinatorID),
            transport: transport,
            localStatic: lanStaticKeys,
            remoteStaticPublicKey: remoteStatic,
            queue: connectionQueue)

        connection.onEstablished = { _, remoteID in
            stateQueue.async { lanConnections[remoteID] = connection }
            server.reportMeshEvent("📱 \(capabilities.deviceName) connected "
                + "(Noise IK over Wi-Fi) — joining the mesh…")
            testbed.failover.handlePeerJoin(capabilities, connection: connection) { result in
                switch result {
                case .success(let plan):
                    server.reportMeshEvent("📱 \(capabilities.deviceName) joined — "
                        + "re-sharded to \(plan.count) shard(s)")
                    lanQueue.async { lanRedialDelay[peerID] = nil }
                    retireSimulatedPeersOnce()
                    pushPeerStates()
                case .failure(let error):
                    server.reportMeshEvent("📱 \(capabilities.deviceName) join failed: \(error)")
                    stateQueue.async { lanConnections.removeValue(forKey: remoteID) }
                    lanQueue.async { lanDialing.remove(peerID) }
                    connection.close()
                }
            }
        }
        connection.onFailed = { error in
            server.reportMeshEvent("📱 \(capabilities.deviceName): \(error)")
            dropLANPeer(peerID, reason: "connection failed")
        }
        connection.start()
    } catch {
        lanDialing.remove(peerID)
        print("[nmp-dashboard] failed to dial \(capabilities.deviceName): \(error)")
    }
}

func dropLANPeer(_ peerID: UInt32, reason: String) {
    lanQueue.async {
        lanDialing.remove(peerID)
        // Re-dial: the session can die while the advert lives on (another
        // coordinator stole the peer — one coordinator per peer, last
        // dial wins — or a transient Wi-Fi drop). If it is still
        // advertised, dial it again after a backoff instead of ignoring
        // it until the next Bonjour announcement.
        let delay = min(60, lanRedialDelay[peerID] ?? 4)
        lanRedialDelay[peerID] = delay * 2
        lanQueue.asyncAfter(deadline: .now() + delay) {
            if let capabilities = lanDiscovery.discoveredPeers[peerID]?.capabilities {
                dialLANPeer(capabilities)
            }
        }
    }
    stateQueue.async {
        guard lanConnections.removeValue(forKey: peerID) != nil else { return }
        guard testbed.failover.activePeers.contains(where: { $0.peerID == peerID }) else {
            return
        }
        server.reportMeshEvent(
            "📱 LAN peer 0x\(String(peerID, radix: 16)) \(reason) — re-sharding…")
        testbed.failover.handlePeerDrop(peerID, timeout: 5) { _ in
            pushPeerStates()
        }
    }
}

lanDiscovery.onPeerDiscovered = { capabilities in
    dialLANPeer(capabilities) // already on lanQueue
}

// The advert vanishing (app closed / left the Wi-Fi) is the discovery-
// level drop signal; the health monitor catches silent deaths too.
lanDiscovery.onPeerRemoved = { peerID in
    dropLANPeer(peerID, reason: "left the network")
}

do {
    try lanQueue.sync { try lanDiscovery.start() }
    print("[nmp-dashboard] browsing for LAN peers (_neuramesh._tcp) — "
          + "open the NeuraMeshPeer app or `swift run nmp-peer` to join one")
} catch {
    print("[nmp-dashboard] ⚠️ LAN peer discovery unavailable: \(error)")
}

if dashboardArguments.ui {
    activateWebUI(server: server, meshSummary: [
        "Mesh: reference engine — \(testbed.failover.activePlan.count) shard(s), "
            + "\(testbed.failover.activePeers.count) device(s)",
        "Wire format: \(referenceMeshInfo.wireFormat)",
    ])
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
                // Remote serves are recorded by the peer-side engines;
                // the coordinator's local shard only shows up here.
                for shard in report.perShard where shard.isLocal {
                    recordServe(peerID: shard.peerID,
                                computeSeconds: shard.computeSeconds)
                }
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
