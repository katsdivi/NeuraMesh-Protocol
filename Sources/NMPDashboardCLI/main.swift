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

// MARK: - Chat history

/// This device's on-disk chat store, under Application Support, stamped with
/// the LAN hostname so a future mesh-shared view can name whose chats these
/// are. Starts the inactivity sweep that packs quiet conversations to LZFSE.
/// Returns nil (chat stays ephemeral) only if Application Support is
/// unavailable.
func makeChatStore() -> NMPChatStore? {
    guard let base = try? FileManager.default.url(
        for: .applicationSupportDirectory, in: .userDomainMask,
        appropriateFor: nil, create: true) else { return nil }
    let dir = base.appendingPathComponent("NeuraMesh/chats", isDirectory: true)
    let store = NMPChatStore(directory: dir,
                             deviceName: NMPLANIdentity.localHostname())
    store.startSweeping()
    print("[nmp-dashboard] chat history at \(dir.path) "
          + "(quiet chats compress after \(Int(store.inactivityThreshold))s)")
    return store
}

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
                usage: nmp-dashboard [port] [--engine reference|llamaCpp|llamaShard] \
                [--model path.gguf] [--gpu-layers N] [--placement local|remote|sharded:N] \
                [--auto-config] [--probe-passes N] [--speculation] [--draft-model path.gguf] \
                [--ui]
                  llamaShard: TRUE cross-device sharding — each peer loads only its
                  layer range (needs scripts/setup_shard.sh); use --placement sharded:N
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

/// Sequential benchmark runs over the single-flight prompt service. Every
/// run is framed on the WebSocket (generation_started/complete, source
/// "benchmark" — BUG-16); busy contention answers 429, not 500 (BUG-5).
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
            server.reportGenerationStarted(
                prompt: request.prompt, maxTokens: request.maxTokens,
                speculative: false, source: "benchmark")
            run(request.prompt, request.maxTokens) { result in
                switch result {
                case .success(let generation):
                    server.reportGenerationComplete(generation, source: "benchmark")
                    results.append(generation)
                    step()
                case .failure(.busy):
                    server.reportGenerationFailed(
                        "an inference is already running", source: "benchmark")
                    respond(.failure(.init(
                        "an inference is already running — retry shortly",
                        status: 429)))
                case .failure(let error):
                    server.reportGenerationFailed(String(describing: error),
                                                  source: "benchmark")
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
        server.reportGenerationStarted(
            prompt: request.prompt, maxTokens: request.maxTokens,
            speculative: request.enableSpeculation, source: "comparison")
        run(request) { result in
            switch result {
            case .failure(.busy):
                // Contention is a normal state, not a server fault (BUG-5).
                server.reportGenerationFailed(
                    "an inference is already running", source: "comparison")
                respond(.failure(.init(
                    "an inference is already running — retry shortly",
                    status: 429)))
            case .failure(let error):
                server.reportGenerationFailed(String(describing: error),
                                              source: "comparison")
                respond(.failure(.init("generation failed: \(error)")))
            case .success(let generation):
                server.reportGenerationComplete(generation, source: "comparison")
                let trips = generation.speculation?.meshRoundTrips
                    ?? generation.perTokenSeconds.count
                guard generation.networkPayloadBytes >= 2, trips >= 1 else {
                    // Honest refusal (nothing to fabricate), but a client
                    // error class: the mesh's state has nothing to race
                    // (BUG-6). Still no invented numbers.
                    respond(.failure(.init(
                        "this run moved no mesh traffic (local placement?) — "
                            + "nothing to race",
                        status: 409)))
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

// MARK: - Headless transport-race benchmark
//
//   swift run nmp-dashboard --benchmark-race [dir] [trials] [condition]
//
// Races NMP vs TCP vs TCP+TLS 1.3 vs QUIC over `trials` runs per traffic
// shape and writes Results/transport_race_<condition>.csv. Under real loss:
//   sudo scripts/loss_lab.sh 2      # shape the 20000..40000 band
//   swift run nmp-dashboard --benchmark-race Results 30 loss2pct

if CommandLine.arguments.dropFirst().first == "--benchmark-race" {
    let args = Array(CommandLine.arguments.dropFirst(2))
    let directory = URL(fileURLWithPath: args.first ?? "Results")
    let trials = args.count > 1 ? (Int(args[1]) ?? 20) : 20
    let condition = args.count > 2 ? args[2] : "clean"
    do {
        print("[nmp-dashboard] transport race: \(trials) trials/shape, condition=\(condition)")
        let report = try NMPTransportRaceBenchmark.runAndExport(
            trials: trials, condition: condition, to: directory,
            progress: { print("  \($0)") })
        print("\n" + report.summaryLines.joined(separator: "\n"))
        print("\nCSV: \(directory.path)/transport_race_\(condition).csv")
        exit(0)
    } catch {
        FileHandle.standardError.write(Data("race benchmark failed: \(error)\n".utf8))
        exit(1)
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
    server.chatStore = makeChatStore()
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
    // Radio mesh: one stalled token must not lock the pipeline (and 429
    // every other request) for the orchestrator's 30 s whole-inference
    // default. 4 s is 30×+ over measured token latency yet below the 5 s
    // health-monitor drop, so a slow peer is retried and a dead one fails
    // fast either way.
    promptService.perTokenTimeout = 4
    let reportProgress: (Int, Int) -> Void = { done, total in
        server.updateInferenceProgress(
            progress: Double(done) / Double(max(total, 1)),
            stage: "llama generation: token \(done)/\(total)")
    }
    promptService.onProgress = reportProgress
    speculativeService?.onProgress = reportProgress

    // Mesh 2.1: stream every confirmed token to every open browser, labeled
    // with the surface that owns the generation (BUG-16). The plain service
    // stamps its own activeSource (onToken fires on its queue); the
    // speculative service has no source plumbing, so a queue-guarded label
    // set at each dispatch site covers it.
    let specSourceQueue = DispatchQueue(label: "nmp.dashboard.llama.specsource")
    var specSource = "inference"
    let setSpecSource: (String) -> Void = { label in
        specSourceQueue.async { specSource = label }
    }
    promptService.onToken = { [weak promptService] token, count, requested in
        server.reportGenerationToken(text: token.text, index: token.index,
                                     count: count, requested: requested,
                                     source: promptService?.activeSource ?? "inference")
    }
    speculativeService?.onToken = { token, count, requested in
        let label = specSourceQueue.sync { specSource }
        server.reportGenerationToken(text: token.text, index: token.index,
                                     count: count, requested: requested,
                                     source: label)
    }

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
                && speculativeService != nil,
            source: request.source)
        let handleResult: (Result<NMPPromptInferenceService.GenerationResult,
                                  NMPPromptInferenceService.ServiceError>) -> Void = { result in
            llamaStateQueue.async { llamaGenerationRunning = false }
            if case .success(let generation) = result {
                server.reportGenerationComplete(generation, source: request.source)
            } else if case .failure(let error) = result {
                server.reportGenerationFailed(String(describing: error),
                                              source: request.source)
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
            setSpecSource(request.source)
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
                              source: request.source,
                              completion: handleResult)
        }
    }

    // Mesh 2.0: health/devices facts + benchmark + multi-device web UI.
    var meshInfo = NMPDashboardServer.MeshInfo()
    meshInfo.engine = "llamaCpp"
    meshInfo.modelName = model.name
    meshInfo.modelDisplayName = model.name
    meshInfo.shardCount = 1
    meshInfo.wireFormat = testbed.orchestrator.activationWireFormat.rawValue
    meshInfo.speculationAvailable = speculativeService != nil
    // The model finished loading during NMPLlamaModel init and the testbed
    // is live — the mesh can genuinely generate from here (BUG-12).
    meshInfo.ready = true
    server.meshInfo = meshInfo

    wireBenchmark(server: server) { prompt, maxTokens, completion in
        if dashboardArguments.speculation, let speculativeService {
            setSpecSource("benchmark")
            speculativeService.run(prompt: prompt, maxTokens: maxTokens,
                                   completion: completion)
        } else {
            promptService.run(prompt: prompt, maxTokens: maxTokens,
                              source: "benchmark", completion: completion)
        }
    }

    // Mesh 2.1: measured protocol race (generation + transport replay).
    wireComparisonRun(server: server) { request, completion in
        if request.enableSpeculation || dashboardArguments.speculation,
           let speculativeService {
            setSpecSource("comparison")
            speculativeService.run(prompt: request.prompt,
                                   maxTokens: request.maxTokens,
                                   completion: completion)
        } else {
            promptService.run(prompt: request.prompt,
                              maxTokens: request.maxTokens,
                              source: "comparison", completion: completion)
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
                // BUG-20: "manual_mode" is the honest name; the old key is
                // kept for compat and mirrors it. Both false here: a llama
                // plan has nothing to re-balance.
                "manual_mode": false,
                "allocation_supported": false,
                "allocation_supported_note":
                    "deprecated name — same value as manual_mode",
                "allocation_note": "a llama plan is one full-range shard "
                    + "(llama.cpp cannot split layers), so there is nothing "
                    + "to re-balance — compute shares apply to the "
                    + "reference mesh",
                "peers": peers,
                "totals": [
                    "devices": peers.count,
                    "devices_alive": peers.count,
                    "devices_note": "coordinator + every mesh member "
                        + "(all in-process on this host)",
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

// MARK: - Phase 10: llamaShard engine mode (TRUE cross-device sharding)

/// The installed models in ~/models with the compatibility flags the web UI
/// needs to let you pick ANY model and honestly flag the ones that won't work:
///   • compatible — the shard shim runs its architecture (qwen2/qwen3 only)
///   • fits_host  — this Mac has the RAM to hold it (the in-process split loads
///                  the whole model across shards on THIS box)
///   • recommended — the highest-quality compatible model that fits
///   • active     — the model this mesh is currently serving
/// Caches the `~/models` scan. Parsing every GGUF header takes seconds, and
/// the Models tab hits it on every open — so serve the last scan instantly
/// and refresh in the background when it goes stale, instead of blocking the
/// request on disk each time. The catalog changes rarely (you add a model),
/// so a short TTL is plenty.
final class NMPModelScanCache {
    static let shared = NMPModelScanCache()
    private let queue = DispatchQueue(label: "nmp.dashboard.modelscan")
    private var cached: [NMPModelCandidate] = []
    private var lastScan = Date.distantPast
    private var refreshing = false
    private let ttl: TimeInterval = 30

    /// The last scan, immediately. Cold start scans synchronously (once);
    /// a warm-but-stale cache returns now and refreshes in the background.
    func candidates() -> [NMPModelCandidate] {
        queue.sync {
            if cached.isEmpty {
                cached = NMPModelCatalog.scan(directory: "~/models")
                lastScan = Date()
            } else if Date().timeIntervalSince(lastScan) > ttl {
                scheduleRefreshLocked()
            }
            return cached
        }
    }

    /// Fills the cache off the request path (call at startup) so even the
    /// first Models-tab open is instant. The scan runs on a background queue.
    func warm() {
        queue.async { [self] in
            if cached.isEmpty || Date().timeIntervalSince(lastScan) > ttl {
                scheduleRefreshLocked()
            }
        }
    }

    /// Must hold `queue`. Kicks a single background rescan.
    private func scheduleRefreshLocked() {
        guard !refreshing else { return }
        refreshing = true
        DispatchQueue.global(qos: .utility).async { [self] in
            let scan = NMPModelCatalog.scan(directory: "~/models")
            queue.async {
                self.cached = scan
                self.lastScan = Date()
                self.refreshing = false
            }
        }
    }
}

func shardModelCatalogJSON(currentPath: String) -> [[String: Any]] {
    let hostRAM = Int(NMPSystemCapabilityProbe.measure(peerID: 0x0000_0001).ramMB)
    let fitsRAM: (NMPModelCandidate) -> Bool = { Double($0.fileMB) <= Double(hostRAM) * 0.7 }
    let all = NMPModelScanCache.shared.candidates()   // cached; highest quality first
    let recommendedPath = all.first {
        $0.architecture.hasPrefix("qwen") && fitsRAM($0)
    }?.path
    let activeExpanded = (currentPath as NSString).expandingTildeInPath

    return all.map { c in
        let compatible = c.architecture.hasPrefix("qwen")
        let fits = fitsRAM(c)
        var notes: [String] = []
        if !compatible {
            notes.append("architecture ‘\(c.architecture)’ isn’t supported yet "
                         + "(the shard shim runs qwen2/qwen3 only)")
        }
        if !fits {
            notes.append("needs ~\(c.fileMB) MB in RAM; this host has \(hostRAM) MB "
                         + "— add a device or use a smaller quant")
        }
        return [
            "path": c.path,
            "name": c.name,
            "arch": c.architecture,
            "size_mb": c.fileMB,
            "params": c.totalParameters,
            "layers": c.layerCount,
            "bits_per_weight": (c.bitsPerWeight * 10).rounded() / 10,
            "compatible": compatible,
            "fits_host": fits,
            "usable": compatible && fits,
            "recommended": c.path == recommendedPath,
            "active": (c.path as NSString).expandingTildeInPath == activeExpanded,
            "note": notes.joined(separator: "; "),
        ]
    }
}

/// Re-exec THIS dashboard process onto a different --model. Keeps the same PID
/// (so the launching `start.sh` / terminal Ctrl-C still owns it) and the same
/// cwd (so the dlopen'd shim in Vendor/ still resolves). Used by the web UI's
/// model picker — the mesh restarts and the page's WebSocket reconnects.
enum NMPProcessRelaunch {
    static func relaunch(withModel modelPath: String) {
        var args = CommandLine.arguments
        if let i = args.firstIndex(of: "--model"), i + 1 < args.count {
            args[i + 1] = modelPath
        } else {
            args.append(contentsOf: ["--model", modelPath])
        }
        // execv wants a NULL-terminated C argv; strdup each (leaked, but we're
        // about to replace the whole image anyway).
        let cArgs: [UnsafeMutablePointer<CChar>?] = args.map { strdup($0) } + [nil]
        execv(args[0], cArgs)
        // Only reached if execv failed — stay honest and exit loudly.
        perror("nmp-dashboard: model-switch relaunch failed (execv)")
        exit(1)
    }
}

/// Real sharded dashboard: N in-process peers, each partial-loading ONLY its
/// layer range via the ggml graph-surgery shim, chained over the full NMP
/// stack. The Devices panel shows each peer's MEASURED layer range and loaded
/// MB (via NMPShardReport) — so "no single device holds the whole model" is
/// shown as fact. Needs the shard shim (scripts/setup_shard.sh) AND the llama
/// shim (for the tokenizer). Never returns.
func runLlamaShardDashboard(modelPath: String, selectionReason: String? = nil,
                            arguments: DashboardArguments) -> Never {
    // The coordinator runs the ggml shard engine on its OWN layer range AND
    // tokenizes; real LAN peers (a second Mac's nmp-peer, or the iPhone app)
    // join over UDP via Bonjour and are handed their own sub-ranges. The split
    // RE-SHARDS live on every join/leave, a peer with no local model streams
    // only its layers from the weight vault, and all of it is visible in the
    // browser (Mesh/Devices tabs). Needs the shard shim + the tokenizer shim.
    let expandedPath = (modelPath as NSString).expandingTildeInPath
    let coordinatorEngine: NMPLlamaShardComputeEngine
    let vocab: NMPLlamaModel
    do {
        coordinatorEngine = try NMPLlamaShardComputeEngine(modelPath: modelPath)
        vocab = try NMPLlamaModel(modelPath: modelPath, vocabOnly: true)
    } catch {
        fatalError("llamaShard mesh assembly failed: \(error) "
                   + "(needs scripts/setup_shard.sh + scripts/setup_llama.sh)")
    }
    let modelTag = coordinatorEngine.modelTag
    let layerCount = coordinatorEngine.layerCount
    print("[nmp-dashboard] llamaShard engine: \(modelTag) — "
          + "\(layerCount) layers × \(coordinatorEngine.hiddenSize) hidden; "
          + "real LAN peers join live and re-shard")

    // Honest per-peer loaded MB: the shim partial-loads exactly its blocks, so a
    // shard's RAM ≈ its layer span × the model's measured bytes/layer.
    let fullModelBytes = ((try? FileManager.default
        .attributesOfItem(atPath: expandedPath)[.size]) as? Int) ?? 0
    let perLayerBytes = NMPModelCatalog.candidate(path: expandedPath)?.bytesPerLayer
        ?? (layerCount > 0 ? fullModelBytes / layerCount : 0)

    let nodeQueue = DispatchQueue(label: "nmp.dashboard.shard.node")
    let node = NMPCoordinatorNode(engine: coordinatorEngine, modelTag: modelTag, queue: nodeQueue)
    node.modelBytesPerLayer = perLayerBytes // RAM ceilings for the auto planner
    node.orchestrator.activationWireFormat = .float32 // lossless residual hand-off
    node.onStatus = { print("[nmp-dashboard] \($0)") }

    let server = NMPDashboardServer()
    server.chatStore = makeChatStore()
    server.onDiagnostic = { print("[nmp-dashboard] \($0)") }
    do { try server.start(port: port) }
    catch { fatalError("dashboard server failed to start: \(error)") }
    print("[nmp-dashboard] dashboard running at http://localhost:\(server.boundPort)")

    // Weight vault (Future Plan #3): a peer with NO local model streams ONLY its
    // assigned layers from here (disk ≈ RAM) — the iPhone path.
    let vaultServer = NMPVaultServer(modelPath: expandedPath, modelTag: modelTag)
    vaultServer.onDiagnostic = { print("[nmp-dashboard] \($0)") }
    // Start the vault OFF the setup path — its NWListener can take a moment to
    // come up, and the UI must wire immediately. `vaultEndpoint` is set well
    // before any peer can realistically discover + join, and rides on each
    // SHARD_ASSIGN, so peers with no local model stream their layers from here.
    DispatchQueue.global(qos: .userInitiated).async {
        do {
            try vaultServer.start(port: 0)
            let endpoint = "\(NMPLANIdentity.localHostname()):\(vaultServer.boundPort)"
            node.orchestrator.vaultEndpoint = endpoint
            print("[nmp-dashboard] weight vault: http://\(endpoint)/vault — peers stream only their layers")
        } catch {
            print("[nmp-dashboard] vault unavailable (\(error)) — peers must hold the model locally")
        }
    }

    // Live split + loaded bytes + membership + adaptive choice, all
    // stateQueue-owned so the UI reads a consistent snapshot.
    let stateQueue = DispatchQueue(label: "nmp.dashboard.shard.state")
    let hostCaps = NMPSystemCapabilityProbe.measure(peerID: node.localPeerID)
    var currentPlan: [NMPShardPlanEntry] = []
    var loadedByPeer: [UInt32: Int] = [:]
    // The coordinator's MEASURED resident bytes and the range they were
    // measured for. Re-plans keep the measurement while the range is
    // unchanged instead of resetting to the modeled estimate (the
    // 39.3↔176.1 MB flip-flop, BUG-19). stateQueue-owned.
    var coordinatorMeasured: (range: Range<Int>, bytes: Int)?
    var peerNames: [UInt32: String] = [node.localPeerID: "coordinator (this Mac)"]
    var memberCaps: [UInt32: NMPCapabilities] = [hostCaps.peerID: hostCaps]
    var adaptiveChoice: (name: String, reason: String, decision: String, devices: Int)?
    // stateQueue-owned mirror of node.autoBalance, so the metrics handler
    // (server queue) reads the mode without racing the node's own queue.
    var autoBalanceMode = true
    // True while a generation is in flight, so the auto-rebalance timer never
    // re-shards mid-token (which would strand the pass on a stale range).
    var shardInferenceInFlight = false

    // The adaptive controller needs a catalog scan of ~/models, which parses
    // every GGUF there (seconds if a big model is present) — so build it in the
    // BACKGROUND and report once it is ready, rather than blocking the UI wire.
    var modelController: NMPAdaptiveModelController? // stateQueue-owned
    DispatchQueue.global(qos: .utility).async {
        let catalog = NMPModelCatalog.scan(directory: "~/models")
            .filter { $0.architecture.hasPrefix("qwen") }
        stateQueue.async {
            modelController = NMPAdaptiveModelController(catalog: catalog)
            reportAdaptive()
        }
    }

    func shortName(_ id: UInt32) -> String {
        peerNames[id] ?? "peer 0x\(String(id, radix: 16))"
    }

    // Push the current split to the Devices/Mesh tabs. MUST run on stateQueue.
    // Every in-mesh device gets a row derived from the CURRENT committed plan
    // — a 0-layer device shows an explicit exclusion, never its stale range
    // (BUG-9), and the coordinator never vanishes at 0 layers (BUG-8).
    func pushShardDevices() {
        var covered = Set<UInt32>()
        for device in NMPShardReport.devices(plan: currentPlan, loadedBytesByPeer: loadedByPeer) {
            let isCoord = device.peerID == node.localPeerID
            covered.insert(device.peerID)
            server.updatePeerState(
                peerID: device.peerID,
                name: isCoord ? "coordinator (tokenizer + shard)" : shortName(device.peerID),
                latencyMS: 0, loadPercent: 0,
                assigned: isCoord ? "\(device.summary) · tokenizer" : device.summary,
                alive: true)
        }
        if !covered.contains(node.localPeerID) {
            covered.insert(node.localPeerID)
            server.updatePeerState(
                peerID: node.localPeerID,
                name: "coordinator (tokenizer + shard)",
                latencyMS: 0, loadPercent: 0,
                assigned: "0 layers — tokenizer only this plan",
                alive: true)
        }
        for peerID in memberCaps.keys where !covered.contains(peerID) {
            server.updatePeerState(
                peerID: peerID, name: shortName(peerID),
                latencyMS: 0, loadPercent: 0,
                assigned: "0 layers — excluded from the current plan",
                alive: true)
        }
    }

    // Report which model the CURRENT real mesh would optimally run (surfaced in
    // the UI; switching is a click in the Models tab, never auto-disruptive).
    // MUST run on stateQueue (reads memberCaps, mutates adaptiveChoice).
    func reportAdaptive() {
        guard let modelController else { return }
        let members = Array(memberCaps.values)
        let word = members.count == 1 ? "device" : "devices"
        switch modelController.evaluate(mesh: members) {
        case .switchModel(_, let to):
            adaptiveChoice = (to.model.name, to.reason, "switch", members.count)
            if to.model.name != modelTag {
                server.reportMeshEvent("🧠 adaptive: \(members.count) real \(word) → "
                    + "\(to.model.name) now optimal (\(to.reason)) — switch in the Models tab")
            }
        case .reshard(let sel), .unchanged(let sel):
            adaptiveChoice = (sel.model.name, sel.reason, "fits", members.count)
        case .noModelFits:
            adaptiveChoice = nil
        }
    }

    // (Re)plan across the coordinator + all ready peers and push the new split
    // to the UI. Called on startup and on every real join/leave. The
    // coordinator's own engine re-learns its range from the SHARD_ASSIGN and
    // lazy-loads it on the next token (Phase A churn-safe re-prefill).
    func reshard(trigger: String, attempt: Int = 0) {
        node.planAndAssign { result in
            switch result {
            case .failure(let error):
                if case .assignmentSuperseded = error {
                    // Benign: a newer plan replaced this round mid-flight;
                    // the newer round reports its own outcome.
                    return
                }
                if case .assignmentRejected(let peerID, .rejectedModelMismatch) = error {
                    server.reportMeshEvent("⚠️ \(shortName(peerID)) holds a DIFFERENT "
                        + "model than this mesh (\(modelTag)) — it stays connected but "
                        + "shard-less. Select the same model on both devices, or remove "
                        + "the phone's local model so it streams layers from the vault.")
                } else {
                    server.reportMeshEvent("⚠️ re-shard failed (\(trigger)): \(error)")
                }
                // The orchestrator kept the previous plan, so inference
                // still runs — but a joined peer would stay shard-less
                // until the next join/leave. One delayed retry covers the
                // common transient (a slow first vault stream on a phone).
                // A rejection is deterministic (e.g. the peer holds a
                // DIFFERENT model) — retrying can't change its answer.
                if case .assignmentRejected = error { return }
                if attempt < 1 {
                    stateQueue.asyncAfter(deadline: .now() + 5) {
                        reshard(trigger: "\(trigger) · retry", attempt: attempt + 1)
                    }
                }
            case .success(let plan):
                applyShardPlan(plan, trigger: trigger)
            }
        }
    }

    // Fold a freshly-assigned plan into the UI state. MUST be called off the
    // node's completion (any queue); it hops to stateQueue itself. Shared by
    // join/leave reshards, manual allocation, the auto-balance toggle, and the
    // auto-rebalance tick so every path updates the mesh identically.
    func applyShardPlan(_ plan: [NMPShardPlanEntry], trigger: String) {
        stateQueue.async {
            currentPlan = plan
            loadedByPeer = Dictionary(uniqueKeysWithValues:
                plan.map { ($0.peerID, max(0, $0.layerSpan) * perLayerBytes) })
            // Keep the coordinator's MEASURED bytes while its range is
            // unchanged — never flip back to the estimate (BUG-19).
            if let mine = plan.first(where: { $0.peerID == node.localPeerID }),
               let measured = coordinatorMeasured,
               measured.range == mine.startLayer..<mine.endLayer {
                loadedByPeer[node.localPeerID] = measured.bytes
            }
            pushShardDevices()
            server.updateShardCount(plan.count)   // async — NOT a sync get-set (deadlocks)
            let split = plan.map {
                "L\($0.startLayer)–\($0.endLayer - 1)→\(shortName($0.peerID))"
            }.joined(separator: "  ")
            let honest = fullModelBytes > 0 && plan.count > 1
                ? " (no device holds all \(fullModelBytes / 1_048_576) MB)" : ""
            server.reportMeshEvent("🧩 re-sharded (\(trigger)) → "
                + "\(plan.count) shard(s): \(split)\(honest)")
            reportAdaptive()
        }
        // Replace the coordinator's estimate with its REAL loaded bytes:
        // partial-loading its range also brings token_embd (first shard),
        // output (last shard), or the whole model when it is the only
        // shard — so the number can't under-report what this Mac holds.
        if let mine = plan.first(where: { $0.peerID == node.localPeerID }) {
            DispatchQueue.global(qos: .userInitiated).async {
                _ = try? coordinatorEngine.preload(start: mine.startLayer, end: mine.endLayer)
                // A plan is committed and the coordinator's weights are
                // resident — the mesh can genuinely generate (BUG-12).
                server.setReady(true)
                let real = coordinatorEngine.loadedBytes
                guard real > 0 else { return }
                stateQueue.async {
                    coordinatorMeasured = (mine.startLayer..<mine.endLayer, real)
                    loadedByPeer[node.localPeerID] = real
                    pushShardDevices()
                }
            }
        } else {
            // All layers live on peers; the coordinator still tokenizes.
            // The plan is committed, so generation is served from here.
            server.setReady(true)
        }
    }

    node.onPeerReady = { caps in
        stateQueue.async {
            memberCaps[caps.peerID] = caps
            peerNames[caps.peerID] = caps.deviceName
        }
        server.reportMeshEvent("📱 \(caps.deviceName) joined "
            + "(\(caps.computeClass.label), \(caps.ramMB) MB) — re-sharding to include it")
        reshard(trigger: "join: \(caps.deviceName)")
    }
    node.onPeerLost = { id in
        stateQueue.async {
            memberCaps[id] = nil
            peerNames[id] = nil
        }
        // A departed peer's /api/devices row must go with it — rows derive
        // from CURRENT membership + plan, never linger stale (BUG-9).
        server.removePeerState(peerID: id)
        server.reportMeshEvent("📴 peer 0x\(String(id, radix: 16)) left — re-sharding")
        reshard(trigger: "leave: 0x\(String(id, radix: 16))")
    }
    // BUG-7 companion: setAutoBalance answers instantly with a STAGED plan
    // while the SHARD_ASSIGN round runs in the background (a peer may
    // vault-stream its new layers for ~30 s). The UI state moves only when
    // the round actually COMMITS — which lands here.
    node.onBackgroundReshard = { result in
        switch result {
        case .success(let plan):
            applyShardPlan(plan, trigger: "balance-mode re-shard committed")
        case .failure(.assignmentSuperseded):
            break // a newer plan took over mid-round — its own commit reports
        case .failure(let error):
            server.reportMeshEvent("⚠️ balance-mode re-shard failed in the "
                + "background: \(error) — the previous plan keeps serving")
        }
    }

    let promptService = NMPPromptInferenceService(
        orchestrator: node.orchestrator,
        codec: NMPLlamaShardPromptCodec(model: vocab))
    // See the llamaCpp path above: bound a stalled remote token at 4 s so a
    // throttled shard peer (e.g. an iOS app iOS is suspending) can't hold
    // the sequential pipeline — and 429 the whole mesh — for 30 s.
    promptService.perTokenTimeout = 4
    promptService.onProgress = { done, total in
        server.updateInferenceProgress(
            progress: Double(done) / Double(max(total, 1)),
            stage: "sharded generation: token \(done)/\(total)")
    }
    promptService.onToken = { [weak promptService] token, count, requested in
        // Fires on the service queue, where activeSource is owned (BUG-16).
        server.reportGenerationToken(text: token.text, index: token.index,
                                     count: count, requested: requested,
                                     source: promptService?.activeSource ?? "inference")
    }

    server.onInferenceRequest = { request, respond in
        let shards = stateQueue.sync { currentPlan.count }
        server.reportGenerationStarted(
            prompt: request.prompt, maxTokens: request.maxTokens,
            speculative: false, source: request.source)
        server.reportMeshEvent("🌐 sharded inference: up to \(request.maxTokens) token(s) "
                               + "across \(shards) shard(s)")
        stateQueue.async { shardInferenceInFlight = true }
        promptService.run(prompt: request.prompt, maxTokens: request.maxTokens,
                          source: request.source) { result in
            stateQueue.async { shardInferenceInFlight = false }
            switch result {
            case .success(let generation):
                server.reportGenerationComplete(generation, source: request.source)
                server.reportMeshEvent(String(
                    format: "🌐 sharded inference done: %d tokens in %.1f ms across %d shard(s)",
                    generation.tokenCount, generation.totalSeconds * 1000, generation.shardCount))
                respond(.success(generation))
            case .failure(.busy):
                respond(.failure(status: 429, message: "an inference is already running — retry shortly"))
            case .failure(.emptyPrompt):
                respond(.failure(status: 400, message: "prompt is empty"))
            case .failure(.codec(let reason)):
                respond(.failure(status: 400, message: "prompt encoding failed: \(reason)"))
            case .failure(.orchestration(let error)):
                server.reportGenerationFailed(String(describing: error),
                                              source: request.source)
                respond(.failure(status: 500, message: "mesh orchestration failed: \(error)"))
            }
        }
    }

    // Benchmark tab + the measured NMP-vs-TCP/QUIC transport race both drive the
    // SAME real sharded pipeline (they take a run-inference closure), so they
    // work unchanged in shard mode. (Packet-loss injection is NOT wired here: the
    // shard coordinator is a REAL UDP mesh, not the in-process loopback, so loss
    // testing is `sudo scripts/loss_lab.sh`, per Docs.)
    wireBenchmark(server: server) { prompt, maxTokens, completion in
        promptService.run(prompt: prompt, maxTokens: maxTokens,
                          source: "benchmark", completion: completion)
    }
    wireComparisonRun(server: server) { request, completion in
        promptService.run(prompt: request.prompt, maxTokens: request.maxTokens,
                          source: "comparison", completion: completion)
    }

    server.onDeviceMetricsRequest = { respond in
        let sample = resourceMonitor.sample()
        let (planSnap, loadedSnap, namesSnap, choiceSnap, autoSnap, membersSnap,
             coordMeasuredSnap, inFlightSnap) = stateQueue.sync {
            (currentPlan, loadedByPeer, peerNames, adaptiveChoice, autoBalanceMode,
             memberCaps, coordinatorMeasured, shardInferenceInFlight)
        }
        var peers: [[String: Any]] = []
        var planned = Set<UInt32>()
        for device in NMPShardReport.devices(plan: planSnap, loadedBytesByPeer: loadedSnap) {
            let isCoord = device.peerID == node.localPeerID
            planned.insert(device.peerID)
            // Honest footprint labeling (BUG-19): the coordinator's number
            // is MEASURED resident bytes once its preload finished (weights
            // for its range PLUS token_embd/output head); every remote
            // peer's is MODELED from its layer span × the file's bytes/layer.
            let coordMeasured = isCoord && coordMeasuredSnap != nil
                && coordMeasuredSnap!.range == device.startLayer..<device.endLayer
            peers.append([
                "id": String(device.peerID, radix: 16),
                "name": isCoord ? "coordinator (tokenizer + shard)"
                    : (namesSnap[device.peerID] ?? "peer 0x\(String(device.peerID, radix: 16))"),
                "alive": true,
                "assigned": isCoord ? "\(device.summary) · tokenizer" : device.summary,
                "layers_loaded": device.layerCount,
                "loaded_mb": device.loadedMB,
                "loaded_mb_basis": coordMeasured
                    ? "measured: resident weight bytes for its range incl. "
                        + "token_embd/output head (why it exceeds the plan's "
                        + "weights-only footprint_mb)"
                    : "modeled: layer span × the model file's bytes/layer "
                        + "(weights only)",
                "compute_share": Double(device.layerCount) / Double(max(1, layerCount)),
                "computing": false,
                "is_coordinator": isCoord,
                "link": isCoord ? "local — tokenizer + its own shard"
                    : "Wi-Fi/UDP (Noise IK, AES-GCM, FEC, NACK) — streams its layers from the vault",
            ])
        }
        // The coordinator card is ALWAYS present — a 0-layer coordinator
        // renders excluded-style like the phone does, instead of vanishing
        // from its own mesh (BUG-8).
        if !planned.contains(node.localPeerID) {
            planned.insert(node.localPeerID)
            peers.insert([
                "id": String(node.localPeerID, radix: 16),
                "name": "coordinator (tokenizer + shard)",
                "alive": true,
                "assigned": "0 layers · tokenizer",
                "layers_loaded": 0,
                "loaded_mb": 0,
                "loaded_mb_basis": "no layers this plan",
                "compute_share": 0.0,
                "computing": false,
                "is_coordinator": true,
                "excluded": true,
                "exclusion_reason": "the current plan gives the coordinator "
                    + "0 layers — it still tokenizes and coordinates",
                "link": "local — tokenizer, no shard this plan",
            ], at: 0)
        }
        // Connected peers that the current plan gives ZERO layers stay VISIBLE
        // (greyed, "excluded") instead of vanishing — auto mode drops a peer
        // whose round trip isn't worth its compute, and the operator must see
        // it's still in the mesh, just idle, not that it silently left.
        for (peerID, _) in membersSnap
        where peerID != node.localPeerID && !planned.contains(peerID) {
            peers.append([
                "id": String(peerID, radix: 16),
                "name": namesSnap[peerID] ?? "peer 0x\(String(peerID, radix: 16))",
                "alive": true,
                "assigned": "0 layers",
                "layers_loaded": 0,
                "loaded_mb": 0,
                "loaded_mb_basis": "no layers this plan",
                "compute_share": 0.0,
                "computing": false,
                "is_coordinator": false,
                "excluded": true,
                "exclusion_reason": autoSnap
                    ? "auto: its network round trip outweighs the compute it "
                        + "would offload — Mac-only is faster for this model"
                    : "manually set to 0% (excluded)",
                "link": "connected (Wi-Fi/UDP) — holding no layers this plan",
            ])
        }
        var payload: [String: Any] = [
            "host": sample.asJSONObject,
            "host_note": "the coordinator tokenizes AND holds a shard; every other device "
                + "holds ONLY its layer range (partial ggml load / vault stream), so once a "
                + "peer joins no single device holds the whole \(fullModelBytes / 1_048_576) MB model. "
                + "the coordinator's loaded_mb is MEASURED; a remote peer's is COMPUTED from its "
                + "layer range × the model's bytes/layer (see each card's loaded_mb_basis).",
            "generation_in_flight": inFlightSnap,
            // Auto mode owns the split (measured speed + capacity); manual mode
            // hands the sliders to the operator. So allocation is "supported"
            // exactly when auto is OFF — the UI gates the sliders on this.
            "auto_balance": autoSnap,
            // BUG-20: honest name for the same fact; allocation_supported is
            // kept for compat and mirrors it (deprecated).
            "manual_mode": !autoSnap,
            "allocation_supported": !autoSnap,
            "allocation_supported_note":
                "deprecated name — same value as manual_mode",
            "allocation_note": autoSnap
                ? "AUTO: minimizes measured per-token latency — Σ(compute) + "
                    + "each used peer's round trip — so a fast-but-far device is "
                    + "used only when its compute saving beats its hop (or "
                    + "capacity requires it). Re-shards on join/leave and as "
                    + "measurements converge. Switch off to allocate manually."
                : "MANUAL: set each device's compute share; 0% excludes it "
                    + "(Mac-only, no per-token round trip). Switch Auto on to "
                    + "rebalance for lowest measured latency.",
            "peers": peers,
            "totals": [
                "devices": peers.count,
                "devices_alive": peers.count,
                "devices_note": "coordinator + every connected peer "
                    + "(excluded 0-layer devices included) — same rows as "
                    + "peers[] and /api/devices",
                "layers_assigned": planSnap.reduce(0) { $0 + max(0, $1.layerSpan) },
            ] as [String: Any],
        ]
        if let choice = choiceSnap {
            payload["adaptive_model"] = [
                "name": choice.name, "reason": choice.reason,
                "decision": choice.decision, "devices": choice.devices,
            ] as [String: Any]
        }
        respond(payload)
    }

    // GET /api/models — the installed catalog with compatibility flags, so the
    // UI can offer any model and flag the ones that won't work here.
    NMPModelScanCache.shared.warm()   // fill off the request path so the tab is instant
    server.onModelsListRequest = { respond in
        respond(shardModelCatalogJSON(currentPath: modelPath))
    }
    // POST /api/models/select — validate then relaunch the mesh onto the model.
    // Accepts a path (the web UI) or a bare filename / GGUF model name (the
    // iPhone app, which knows models by name, not by this Mac's paths).
    server.onModelSelectRequest = { requestedPath, reply in
        var expanded = (requestedPath as NSString).expandingTildeInPath
        if !FileManager.default.fileExists(atPath: expanded),
           !requestedPath.contains("/") {
            let wanted = requestedPath.lowercased()
            let catalog = NMPModelCatalog.scan(directory: "~/models")
            if let match = catalog.first(where: {
                $0.name.lowercased() == wanted
                    || ($0.path as NSString).lastPathComponent.lowercased() == wanted
            }) {
                expanded = match.path
            } else {
                reply(.failure(NMPDashboardServer.BenchmarkFailure(
                    "this Mac has no ‘\(requestedPath)’ in ~/models — the coordinator "
                        + "needs the file (it tokenizes and serves the vault). Installed: "
                        + (catalog.isEmpty ? "none"
                            : catalog.map(\.name).joined(separator: ", ")),
                    status: 400)))
                return
            }
        }
        guard FileManager.default.fileExists(atPath: expanded),
              let candidate = NMPModelCatalog.candidate(path: expanded) else {
            reply(.failure(NMPDashboardServer.BenchmarkFailure(
                "no readable GGUF at \(requestedPath)", status: 400)))
            return
        }
        // BUG-2: a vault slice / split fragment parses as a valid GGUF but
        // is NOT a runnable model — the same metadata criterion that keeps
        // slices out of /api/models rejects them here.
        guard candidate.isCompleteModel else {
            reply(.failure(NMPDashboardServer.BenchmarkFailure(
                "‘\(candidate.name)’ is a vault slice — not a complete model "
                    + "(it carries only a layer range of its base model); "
                    + "select the full GGUF instead",
                status: 400)))
            return
        }
        guard candidate.architecture.hasPrefix("qwen") else {
            reply(.failure(NMPDashboardServer.BenchmarkFailure(
                "‘\(candidate.name)’ is a \(candidate.architecture) model — the shard "
                    + "shim runs qwen2/qwen3 only (a llama-arch variant is future work)",
                status: 400)))
            return
        }
        let hostRAM = Int(NMPSystemCapabilityProbe.measure(peerID: 0x0000_0001).ramMB)
        guard Double(candidate.fileMB) <= Double(hostRAM) * 0.7 else {
            reply(.failure(NMPDashboardServer.BenchmarkFailure(
                "‘\(candidate.name)’ needs ~\(candidate.fileMB) MB in RAM; this host "
                    + "has \(hostRAM) MB — add a device or pick a smaller quant",
                status: 400)))
            return
        }
        // Selecting the model that's already live is a no-op, not a
        // multi-second mesh restart.
        guard expanded != expandedPath else {
            reply(.success(.init(summary: "already active", reconnecting: false)))
            return
        }
        reply(.success(.init(
            summary: "switching to \(candidate.name) — the mesh restarts and the page "
                + "reconnects in a few seconds",
            reconnecting: true)))
        server.reportMeshEvent("🔄 switching model → \(candidate.name); relaunching mesh")
        // Let the HTTP response flush, release the listener, then re-exec this
        // process onto the new --model (same PID, so start.sh/Ctrl-C still own it).
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.8) {
            server.stop()
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.4) {
                NMPProcessRelaunch.relaunch(withModel: expanded)
            }
        }
    }

    var shardMeshInfo = NMPDashboardServer.MeshInfo()
    shardMeshInfo.engine = "llamaShard"
    shardMeshInfo.modelName = modelTag
    // Display name for saved-chat stamping (BUG-14): the catalog's name for
    // the active model (GGUF general.name), falling back to the engine tag.
    shardMeshInfo.modelDisplayName =
        NMPModelCatalog.candidate(path: expandedPath)?.name ?? modelTag
    shardMeshInfo.shardCount = max(1, stateQueue.sync { currentPlan.count })
    shardMeshInfo.wireFormat = node.orchestrator.activationWireFormat.rawValue
    // NOT ready yet: the startup re-shard + coordinator preload flip it via
    // server.setReady(true) in applyShardPlan (BUG-12).
    server.meshInfo = shardMeshInfo

    if dashboardArguments.ui {
        var summary = [
            "Mesh: llamaShard — \(modelTag) (REAL cross-device sharding)",
            "Coordinator tokenizes + holds a shard; LAN peers join live and re-shard",
            "Wire: float32 residual (lossless) — only the activation crosses the network",
        ]
        if let selectionReason { summary.append("Model choice: \(selectionReason)") }
        activateWebUI(server: server, meshSummary: summary)
    }
    if let selectionReason { server.reportMeshEvent("model choice — \(selectionReason)") }

    // Manual allocation (POST /api/devices/<id>/allocate): cap one device's
    // compute share and re-shard. Flips the mesh to MANUAL mode (auto off).
    // A 0 share drops the device to zero layers — Mac-only, no round trip.
    let hexName: (UInt32) -> String = { "0x" + String($0, radix: 16) }
    let planSummary: ([NMPShardPlanEntry]) -> String = { plan in
        plan.map { "\(hexName($0.peerID)): L\($0.startLayer)-\($0.endLayer - 1)" }
            .joined(separator: ", ")
    }
    server.onAllocationRequest = { peerID, share, respond in
        // BUG-10: an id that is not the coordinator or a connected member
        // is a 404, not a silently-accepted share for a ghost.
        let known = stateQueue.sync { memberCaps[peerID] != nil }
        guard known else {
            respond(.failure(.init("unknown device", status: 404)))
            return
        }
        node.setManualShare(peerID: peerID, share: share) { result in
            switch result {
            case .success(let plan):
                stateQueue.async { autoBalanceMode = false }
                applyShardPlan(plan, trigger: String(
                    format: "manual %.0f%% → %@", share * 100, hexName(peerID)))
                // Lead with the target device's own resulting assignment
                // (BUG-19's echo half), then the whole plan.
                let mine = plan.first { $0.peerID == peerID }
                let own = mine.map { "L\($0.startLayer)-\($0.endLayer - 1)" }
                    ?? "0 layers (excluded)"
                respond(.success("manual — \(hexName(peerID)): \(own); "
                                 + "plan: " + planSummary(plan)))
            case .failure(let error):
                respond(.failure(.init("re-shard failed: \(error)")))
            }
        }
    }

    // POST /api/mesh/objective {"objective": "speed" | "capacityThenSpeed"}
    // (BUG-11): applies the sharding objective through the node's existing
    // plan-strategy API — "speed" packs the fastest device (fewest hops),
    // "capacityThenSpeed" spreads across the fewest devices that hold the
    // model, balanced by speed (the sharder's .capacityThenSpeed objective,
    // which the node exposes as its `balanced` strategy).
    server.onObjectiveRequest = { raw, respond in
        let strategy: NMPCoordinatorNode.PlanStrategy?
        switch raw {
        case NMPShardingObjective.speed.rawValue: strategy = .speed
        case NMPShardingObjective.capacityThenSpeed.rawValue: strategy = .balanced
        default: strategy = nil
        }
        guard let strategy else {
            respond(.failure(.init("unknown objective '\(raw)' — expected one of: "
                + NMPShardingObjective.allCases.map(\.rawValue)
                    .joined(separator: ", "),
                status: 400)))
            return
        }
        node.setPlanStrategy(strategy) { result in
            switch result {
            case .success(let plan):
                stateQueue.async { autoBalanceMode = true }
                applyShardPlan(plan, trigger: "objective: \(raw)")
                respond(.success(planSummary(plan)))
            case .failure(let error):
                respond(.failure(.init("re-shard failed: \(error)")))
            }
        }
    }

    // POST /api/mesh/autobalance {"enabled": bool}. Auto = balance by measured
    // speed + capacity (and keep converging); manual hands over the sliders.
    // The node answers instantly with the STAGED plan (BUG-7: the assign
    // round may vault-stream for ~30 s); the UI/devices state is updated by
    // onBackgroundReshard when the round commits — never optimistically.
    server.onAutoBalanceRequest = { enabled, respond in
        node.setAutoBalance(enabled) { result in
            switch result {
            case .success(let plan):
                stateQueue.async { autoBalanceMode = enabled }
                respond(.success((enabled ? "auto — " : "manual — ")
                    + planSummary(plan)
                    + " (staged; re-shard applying in the background)"))
            case .failure(let error):
                respond(.failure(.init("re-shard failed: \(error)")))
            }
        }
    }

    // Auto-convergence: on a slow cadence, when idle and in auto mode, re-plan
    // from FRESH measurements and re-assign only on a material change — so the
    // split converges to speed-optimal after real traffic, then holds steady.
    let rebalanceTimer = DispatchSource.makeTimerSource(queue: stateQueue)
    rebalanceTimer.schedule(deadline: .now() + 20, repeating: 20)
    rebalanceTimer.setEventHandler {
        let inflight = shardInferenceInFlight   // read on stateQueue
        node.autoRebalanceTick(inflight: inflight) { result in
            if case .success(let plan)? = result {
                applyShardPlan(plan, trigger: "auto-rebalance by measured speed")
            }
        }
    }
    rebalanceTimer.resume()

    // GET /api/mesh/plans — preview the candidate splits (speed / balanced /
    // capacity) with each device's storage footprint + % of its RAM, so the
    // operator picks one instead of the mesh silently deciding — and can see
    // that no device would end up full.
    let strategyLabels: [NMPCoordinatorNode.PlanStrategy: (String, String)] = [
        .speed: ("Best for speed",
                 "fewest hops — minimizes measured per-token latency"),
        .balanced: ("Balanced",
                    "spread by measured speed across the mesh, capped by RAM"),
        .capacity: ("Best for capacity",
                    "even % load so no single device fills up"),
    ]
    server.onPlansRequest = { respond in
        let (names, members) = stateQueue.sync { (peerNames, memberCaps) }
        let coordFirst = members.sorted {
            ($0.key == node.localPeerID ? 0 : 1) < ($1.key == node.localPeerID ? 0 : 1)
        }
        node.candidatePlans { candidates in
            var plansJSON: [[String: Any]] = []
            for (strategy, plan) in candidates {
                let assigned = Dictionary(uniqueKeysWithValues:
                    plan.entries.map { ($0.peerID, $0.layerSpan) })
                var devices: [[String: Any]] = []
                var maxPct = 0.0
                for (peerID, caps) in coordFirst {
                    let layers = assigned[peerID] ?? 0
                    let footprintMB = perLayerBytes > 0
                        ? layers * perLayerBytes / 1_048_576 : 0
                    let ramMB = Int(caps.ramMB)
                    let pct = ramMB > 0 ? Double(footprintMB) / Double(ramMB) * 100 : 0
                    maxPct = max(maxPct, pct)
                    let isCoord = peerID == node.localPeerID
                    devices.append([
                        "id": String(peerID, radix: 16),
                        "name": isCoord ? "coordinator (this Mac)"
                            : (names[peerID] ?? "peer 0x\(String(peerID, radix: 16))"),
                        "layers": layers,
                        "footprint_mb": footprintMB,
                        "ram_mb": ramMB,
                        "percent": (pct * 10).rounded() / 10,
                        "is_coordinator": isCoord,
                        "excluded": layers == 0 && !isCoord,
                    ])
                }
                let (label, note) = strategyLabels[strategy]
                    ?? (strategy.rawValue, "")
                plansJSON.append([
                    "strategy": strategy.rawValue,
                    "label": label,
                    "note": note,
                    "fits": plan.capacityShortfall == 0,
                    "capacity_shortfall": plan.capacityShortfall,
                    "max_device_percent": (maxPct * 10).rounded() / 10,
                    "devices": devices,
                ])
            }
            respond(["current_strategy": node.planStrategy.rawValue,
                     // BUG-19: name what footprint_mb IS, so it can't be
                     // confused with the runtime loaded_mb in
                     // /api/devices/metrics (which adds embeddings/output
                     // head on the coordinator).
                     "footprint_note": "footprint_mb is modeled, weights-only: "
                        + "layer span × the model file's measured bytes/layer. "
                        + "Runtime residency is larger (token_embd/output head, "
                        + "KV cache) — see loaded_mb + loaded_mb_basis in "
                        + "/api/devices/metrics.",
                     "plans": plansJSON])
        }
    }

    // POST /api/mesh/strategy {"strategy": ...} — apply a previewed plan.
    server.onPlanStrategyRequest = { strategyRaw, respond in
        guard let strategy = NMPCoordinatorNode.PlanStrategy(rawValue: strategyRaw) else {
            respond(.failure(.init("unknown strategy '\(strategyRaw)' — "
                                   + "use speed | balanced | capacity")))
            return
        }
        node.setPlanStrategy(strategy) { result in
            switch result {
            case .success(let plan):
                stateQueue.async { autoBalanceMode = true }
                applyShardPlan(plan, trigger: "\(strategyRaw) plan")
                respond(.success(planSummary(plan)))
            case .failure(let error):
                respond(.failure(.init("re-shard failed: \(error)")))
            }
        }
    }

    // Assign the coordinator's OWN shard right away (no peer needed) so the mesh
    // is inferable immediately, THEN start Bonjour discovery in the background —
    // its first browse can take a few seconds and must not block the UI. Real
    // peers that join re-shard live (node.onPeerReady/onPeerLost → reshard).
    reshard(trigger: "startup")
    nodeQueue.async {
        do { try node.start() }
        catch { print("[nmp-dashboard] discovery failed to start: \(error) — LAN peers can't join") }
    }
    print("[nmp-dashboard] browsing for LAN shard peers (nmp-peer / iPhone app); "
          + "the coordinator holds all layers until one joins. Ctrl-C to stop")
    dispatchMain()
}

// Validate --engine against the ONE plugin registry (NMPPlugin.swift) before
// dispatch. Unknown ids fail fast; the hashShard scaffold is a peer-only stub
// with no dashboard orchestration, so it is rejected here rather than silently
// falling through to the reference mesh.
switch NMPPluginRegistry.descriptor(id: dashboardArguments.engine) {
case .none:
    FileHandle.standardError.write(Data("""
    unknown --engine '\(dashboardArguments.engine)'. available on the dashboard: \
    reference, llamaCpp, llamaShard.

    """.utf8))
    exit(2)
case .some(let descriptor) where descriptor.id == "hashShard":
    FileHandle.standardError.write(Data("""
    --engine hashShard is a non-LLM SCAFFOLD (see Docs/Plugin_Architecture.md); \
    it is selectable on nmp-peer only, not the dashboard.

    """.utf8))
    exit(2)
default:
    break
}

if dashboardArguments.engine == "llamaShard" {
    // Explicit --model wins; otherwise auto-select the OPTIMAL model from
    // ~/models for THIS host (the real "pick whatever fits" path).
    var modelPath = dashboardArguments.modelPath
    var selectionReason: String?
    if modelPath == nil {
        // The ggml shard shim implements qwen2/qwen3 blocks (NEOX RoPE, GQA);
        // only offer architectures it runs correctly. Other arches (e.g. llama
        // NORMAL RoPE) need a shim variant first.
        let catalog = NMPModelCatalog.scan(directory: "~/models")
            .filter { $0.architecture.hasPrefix("qwen") }
        let host = NMPSystemCapabilityProbe.measure(peerID: 0x0000_0001)
        guard !catalog.isEmpty,
              let pick = NMPModelSelector.pick(mesh: [host], catalog: catalog) else {
            let msg = "no model in ~/models fits this host (RAM \(host.ramMB) MB, "
                + "free disk \(host.storageFreeMB) MB) — pass --model path.gguf\n"
            FileHandle.standardError.write(Data(msg.utf8))
            exit(2)
        }
        modelPath = pick.model.path
        selectionReason = pick.reason
        print("[nmp-dashboard] auto-selected \(pick.model.name): \(pick.reason)")
    }
    runLlamaShardDashboard(modelPath: modelPath!, selectionReason: selectionReason,
                           arguments: dashboardArguments)
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
server.chatStore = makeChatStore()
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
// Mesh 2.1: stream every generated token to every open browser, labeled
// with the owning surface (fires on the service queue, where activeSource
// is owned — BUG-16).
promptService.onToken = { [weak promptService] token, count, requested in
    server.reportGenerationToken(text: token.text, index: token.index,
                                 count: count, requested: requested,
                                 source: promptService?.activeSource ?? "inference")
}

server.onInferenceRequest = { request, respond in
    stateQueue.async { apiInferenceRunning = true }
    server.reportMeshEvent("🌐 API inference: up to \(request.maxTokens) token(s)")
    server.reportGenerationStarted(prompt: request.prompt,
                                   maxTokens: request.maxTokens,
                                   speculative: false,
                                   source: request.source)
    promptService.run(prompt: request.prompt, maxTokens: request.maxTokens,
                      source: request.source) { result in
        stateQueue.async { apiInferenceRunning = false }
        if case .success(let generation) = result {
            server.reportGenerationComplete(generation, source: request.source)
        } else if case .failure(let error) = result {
            server.reportGenerationFailed(String(describing: error),
                                          source: request.source)
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
referenceMeshInfo.modelDisplayName = testbed.modelTag
referenceMeshInfo.shardCount = testbed.failover.activePlan.count
referenceMeshInfo.wireFormat = testbed.orchestrator.activationWireFormat.rawValue
// The simulated mesh assembled synchronously above — it can generate now.
referenceMeshInfo.ready = true
server.meshInfo = referenceMeshInfo

// MARK: - Phase C: adaptive model selection on live churn
//
// As REAL devices join and leave, re-pick the optimal model (storage + RAM
// aware) and report the decision. Driven ONLY by real device capabilities —
// the local host (measured) plus real LAN peers — never the synthetic
// in-process stand-ins, so the choice reflects the PHYSICAL mesh honestly.
// The shard shim runs qwen2/qwen3, so only those are offered as reachable
// targets (a llama-arch model would need a NORMAL-RoPE shim variant first).
let modelCatalog = NMPModelCatalog.scan(directory: "~/models")
    .filter { $0.architecture.hasPrefix("qwen") }
let hostCapabilities = NMPSystemCapabilityProbe.measure(peerID: testbed.coordinatorID)
let modelController = NMPAdaptiveModelController(catalog: modelCatalog)
/// Real LAN peers' capabilities (stateQueue-owned) — the honest membership.
var lanPeerCapabilities: [UInt32: NMPCapabilities] = [:]
/// Latest adaptive choice for the API/UI (stateQueue-owned).
var currentModelChoice: (name: String, reason: String, decision: String, devices: Int)?

/// Re-runs the storage+RAM-aware selector over the REAL mesh and reports what
/// changed. Safe to call from any queue (hops to stateQueue).
func reevaluateModelSelection(trigger: String) {
    guard !modelCatalog.isEmpty else { return }
    stateQueue.async {
        let mesh = [hostCapabilities] + Array(lanPeerCapabilities.values)
        let word = mesh.count == 1 ? "device" : "devices"
        switch modelController.evaluate(mesh: mesh) {
        case .switchModel(let from, let to):
            currentModelChoice = (to.model.name, to.reason, "switch", mesh.count)
            server.reportMeshEvent("🧠 adaptive model (\(trigger)): \(mesh.count) real "
                + "\(word) → \(to.model.name) [was \(from)] — \(to.reason)")
        case .reshard(let sel):
            currentModelChoice = (sel.model.name, sel.reason, "reshard", mesh.count)
            server.reportMeshEvent("🧠 adaptive model (\(trigger)): \(sel.model.name) "
                + "stays — re-split across \(mesh.count) \(word)")
        case .unchanged(let sel):
            currentModelChoice = (sel.model.name, sel.reason, "unchanged", mesh.count)
        case .noModelFits:
            currentModelChoice = nil
            server.reportMeshEvent("🧠 adaptive model (\(trigger)): no model in "
                + "~/models fits \(mesh.count) real \(word) — add a device or a model")
        }
    }
}
if modelCatalog.isEmpty {
    print("[nmp-dashboard] adaptive model selection idle — no qwen GGUF in ~/models")
} else {
    print("[nmp-dashboard] adaptive model selection: \(modelCatalog.count) candidate(s) "
          + "in ~/models, re-picked on every real join/leave")
}
reevaluateModelSelection(trigger: "startup")

// Benchmark runs pause the heartbeat loop exactly like API inference does.
wireBenchmark(server: server) { prompt, maxTokens, completion in
    stateQueue.async { apiInferenceRunning = true }
    promptService.run(prompt: prompt, maxTokens: maxTokens,
                      source: "benchmark") { result in
        stateQueue.async { apiInferenceRunning = false }
        completion(result)
    }
}

// Mesh 2.1: measured protocol race. The heartbeat pauses for the whole
// run (generation + race) so its passes pollute neither measurement.
wireComparisonRun(server: server) { request, completion in
    stateQueue.async { apiInferenceRunning = true }
    promptService.run(prompt: request.prompt,
                      maxTokens: request.maxTokens,
                      source: "comparison") { result in
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
            "devices_note": "devices counts every card in peers[] (incl. "
                + "dropped-but-remembered ones); devices_alive counts "
                + "current mesh members",
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
            // BUG-20: manual_mode is the honest name (the reference mesh
            // always accepts operator shares); allocation_supported is a
            // deprecated alias kept for compat.
            "manual_mode": true,
            "allocation_supported": true,
            "allocation_supported_note":
                "deprecated name — same value as manual_mode",
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
            // Phase C: the storage+RAM-aware model the REAL mesh would run,
            // re-picked on every physical join/leave (nil = no model fits).
            "adaptive_model": currentModelChoice.map {
                [
                    "name": $0.name,
                    "reason": $0.reason,
                    "decision": $0.decision,
                    "real_devices": $0.devices,
                ] as [String: Any]
            } ?? [
                "name": "",
                "reason": modelCatalog.isEmpty
                    ? "no qwen GGUF in ~/models"
                    : "no model fits the current real device(s)",
                "decision": "none",
                "real_devices": 0,
            ],
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
        // BUG-10: not a member of this mesh → 404, never a 200/500.
        respond(.failure(.init("unknown device", status: 404)))
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
                    stateQueue.async { lanPeerCapabilities[capabilities.peerID] = capabilities }
                    reevaluateModelSelection(trigger: "join: \(capabilities.deviceName)")
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
        lanPeerCapabilities.removeValue(forKey: peerID)
        guard testbed.failover.activePeers.contains(where: { $0.peerID == peerID }) else {
            return
        }
        server.reportMeshEvent(
            "📱 LAN peer 0x\(String(peerID, radix: 16)) \(reason) — re-sharding…")
        testbed.failover.handlePeerDrop(peerID, timeout: 5) { _ in
            pushPeerStates()
        }
        reevaluateModelSelection(trigger: "leave: 0x\(String(peerID, radix: 16))")
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
