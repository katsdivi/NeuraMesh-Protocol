//
//  AdaptiveSharding.swift
//  NMP — Phase 9
//
//  Benchmark-driven layer balancing. Phase 5's sharder already splits
//  layers proportionally to measured seconds-per-layer — but nothing ever
//  MEASURED before the first real inference, so a fresh mesh runs its
//  first plan on static class weights (high 4 : medium 2 : low 1) and
//  only converges after live traffic. Phase 9 closes that loop:
//
//    1. assign the naive plan,
//    2. drive a few real probe passes through the live mesh (every stage
//       is timed by the peers themselves — the same measurement path
//       production inference uses, so probes cost real work, not fake),
//    3. re-plan with the measurements and re-assign (one normal
//       SHARD_ASSIGN round, the exact machinery failover uses),
//    4. persist the per-device profile so the NEXT startup seeds the
//       sharder before the first plan and skips the probe phase.
//
//  Balance math: pipeline throughput is set by the SLOWEST stage, so the
//  optimizer's goal is minimizing max(stage seconds) — a plan where a
//  device 10% slower gets 10% fewer layers. `NMPShardBalance` reports the
//  achieved quality (efficiency 1.0 = every stage takes equally long).
//
//  Profiles are keyed by (modelTag, deviceName), NOT peerID — peer IDs
//  are handed out per session, device names survive restarts.
//
//  Threading: callback style on a private serial queue, same as every
//  other NMP component. `setupSync` blocks for CLI/test callers.
//

import Foundation

// MARK: - Balance report

/// How well a plan spreads wall-clock across the pipeline.
public struct NMPShardBalance: Sendable {

    public struct Stage: Sendable {
        public let entry: NMPShardPlanEntry
        /// span × measured seconds-per-layer (nil when never measured).
        public let estimatedSeconds: TimeInterval?
    }

    public let stages: [Stage]
    /// Estimated per-token pipeline latency: max stage seconds.
    public let pipelineSeconds: TimeInterval
    /// mean(stage) / max(stage): 1.0 = perfectly balanced, lower = the
    /// slowest stage is dragging idle time into every other device.
    public let balanceEfficiency: Double

    public static func evaluate(
        plan: [NMPShardPlanEntry],
        measuredSecondsPerLayer: [UInt32: Double]
    ) -> NMPShardBalance {
        // Unmeasured peers borrow the mean measured rate so a partially
        // profiled mesh still yields a usable (clearly labeled) estimate.
        let known = measuredSecondsPerLayer.values.filter { $0 > 0 }
        let fallback = known.isEmpty ? nil : known.reduce(0, +) / Double(known.count)

        let stages = plan.map { entry -> Stage in
            let rate = measuredSecondsPerLayer[entry.peerID] ?? fallback
            return Stage(entry: entry,
                         estimatedSeconds: rate.map { Double(entry.layerSpan) * $0 })
        }
        let seconds = stages.compactMap(\.estimatedSeconds)
        let maxStage = seconds.max() ?? 0
        let mean = seconds.isEmpty ? 0 : seconds.reduce(0, +) / Double(seconds.count)
        return NMPShardBalance(
            stages: stages,
            pipelineSeconds: maxStage,
            balanceEfficiency: maxStage > 0 ? mean / maxStage : 0)
    }

    /// Human-readable per-stage lines for logs/CLI.
    public func summaryLines(deviceNames: [UInt32: String] = [:]) -> [String] {
        var lines = stages.map { stage -> String in
            let name = deviceNames[stage.entry.peerID]
                ?? "peer 0x\(String(stage.entry.peerID, radix: 16))"
            let estimate = stage.estimatedSeconds
                .map { String(format: "%.1f ms", $0 * 1000) } ?? "unmeasured"
            return "\(name): layers \(stage.entry.startLayer)-\(stage.entry.endLayer - 1) "
                + "(\(stage.entry.layerSpan) layers, \(estimate))"
        }
        lines.append(String(format: "pipeline latency: %.1f ms per pass",
                            pipelineSeconds * 1000))
        lines.append(String(format: "balance efficiency: %.1f%%",
                            balanceEfficiency * 100))
        return lines
    }
}

// MARK: - Profile persistence

/// One model's measured per-device speeds.
public struct NMPShardingProfile: Codable, Equatable, Sendable {
    public var modelTag: String
    /// deviceName → measured seconds-per-layer.
    public var secondsPerLayerByDevice: [String: Double]
    public var updatedAt: Date

    public init(modelTag: String, secondsPerLayerByDevice: [String: Double],
                updatedAt: Date = Date()) {
        self.modelTag = modelTag
        self.secondsPerLayerByDevice = secondsPerLayerByDevice
        self.updatedAt = updatedAt
    }
}

/// JSON persistence for sharding profiles (default: ~/.nmp/, next to the
/// llama shim's fallback location).
public final class NMPShardingProfileStore {

    public let fileURL: URL

    public static func defaultURL() -> URL {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".nmp/neuramesh_sharding.json")
    }

    public init(fileURL: URL = NMPShardingProfileStore.defaultURL()) {
        self.fileURL = fileURL
    }

    /// All stored profiles ([] when the file is missing or unreadable —
    /// a corrupt cache must never block mesh startup).
    public func load() -> [NMPShardingProfile] {
        guard let data = try? Data(contentsOf: fileURL) else { return [] }
        return (try? Self.decoder().decode([NMPShardingProfile].self, from: data)) ?? []
    }

    public func profile(forModelTag modelTag: String) -> NMPShardingProfile? {
        load().first { $0.modelTag == modelTag }
    }

    /// Upserts by modelTag; device entries merge (new measurements win).
    public func save(_ profile: NMPShardingProfile) throws {
        var profiles = load()
        if let index = profiles.firstIndex(where: { $0.modelTag == profile.modelTag }) {
            var merged = profiles[index]
            merged.secondsPerLayerByDevice.merge(
                profile.secondsPerLayerByDevice) { _, new in new }
            merged.updatedAt = profile.updatedAt
            profiles[index] = merged
        } else {
            profiles.append(profile)
        }
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try Self.encoder().encode(profiles).write(to: fileURL, options: .atomic)
    }

    public func clear() {
        try? FileManager.default.removeItem(at: fileURL)
    }

    private static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    private static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}

// MARK: - Adaptive controller

/// Drives the benchmark → re-balance → persist loop over a live mesh.
public final class NMPAdaptiveShardingController {

    public enum SetupError: Error {
        case failover(NMPFailoverError)
        case probeFailed(NMPOrchestrationError)
        case timedOut
    }

    /// Everything one adaptive setup measured and decided.
    public struct SetupReport: Sendable {
        /// false when a persisted profile covered every peer and the
        /// probe phase was skipped.
        public let probed: Bool
        public let probePasses: Int
        public let probeSeconds: TimeInterval
        public let plan: [NMPShardPlanEntry]
        public let balance: NMPShardBalance
        public let measuredSecondsPerLayer: [UInt32: Double]
    }

    /// Diagnostics stream ("[auto-config] …" lines). Fires on the
    /// controller queue.
    public var onDiagnostic: ((String) -> Void)?

    private let failover: NMPFailoverOrchestrator
    private let orchestrator: NMPInferenceOrchestrator
    private let modelTag: String
    private let store: NMPShardingProfileStore
    private let queue = DispatchQueue(label: "nmp.adaptive.sharding")

    public init(failover: NMPFailoverOrchestrator,
                orchestrator: NMPInferenceOrchestrator,
                modelTag: String,
                store: NMPShardingProfileStore = NMPShardingProfileStore()) {
        self.failover = failover
        self.orchestrator = orchestrator
        self.modelTag = modelTag
        self.store = store
    }

    /// Full adaptive setup: seed from cache → naive plan → probe passes →
    /// re-balanced plan → persist. `makeProbeInput` supplies one activation
    /// vector per probe pass (probes are REAL pipeline passes). Completion
    /// fires on the controller queue.
    public func setup(
        probePasses: Int = 3,
        makeProbeInput: @escaping (Int) -> [Float],
        completion: @escaping (Result<SetupReport, SetupError>) -> Void
    ) {
        queue.async { [self] in
            let deviceNames = nameIndex()

            // 1. Seed the sharder from last session's profile.
            let cached = store.profile(forModelTag: modelTag)?
                .secondsPerLayerByDevice ?? [:]
            var seeded: [UInt32: Double] = [:]
            for (peerID, name) in deviceNames {
                if let rate = cached[name] { seeded[peerID] = rate }
            }
            let fullyCached = !deviceNames.isEmpty
                && seeded.count == deviceNames.count
            if !seeded.isEmpty {
                orchestrator.seedMeasurements(seeded)
                diagnose("profile cache: \(seeded.count)/\(deviceNames.count) "
                         + "device(s) known for '\(modelTag)'")
            }

            // 2. First plan — measured where seeded, class weights elsewhere.
            assignPlan { [self] result in
                switch result {
                case .failure(let error):
                    completion(.failure(.failover(error)))
                case .success(let initialPlan):
                    if fullyCached || probePasses <= 0 {
                        diagnose(fullyCached
                            ? "probe phase skipped (profile cache is complete)"
                            : "probe phase skipped (0 passes requested)")
                        finish(plan: initialPlan, probed: false, passes: 0,
                               probeSeconds: 0, deviceNames: deviceNames,
                               completion: completion)
                        return
                    }
                    // 3. Probe: real pipeline passes fill in measurements.
                    diagnose("probing \(deviceNames.count) device(s), "
                             + "\(probePasses) pass(es)…")
                    let began = DispatchTime.now()
                    runProbes(remaining: probePasses, pass: 0,
                              makeProbeInput: makeProbeInput) { [self] probeResult in
                        if case .failure(let error) = probeResult {
                            // Recommended fallback: a failed probe keeps the
                            // naive plan instead of failing the mesh.
                            diagnose("probe failed (\(error)) — keeping naive plan")
                            finish(plan: initialPlan, probed: false, passes: 0,
                                   probeSeconds: 0, deviceNames: deviceNames,
                                   completion: completion)
                            return
                        }
                        let probeSeconds = TimeInterval(
                            DispatchTime.now().uptimeNanoseconds
                                - began.uptimeNanoseconds) / 1e9
                        // 4. Re-plan with measurements and re-assign.
                        assignPlan { [self] replanned in
                            switch replanned {
                            case .failure(let error):
                                completion(.failure(.failover(error)))
                            case .success(let balancedPlan):
                                finish(plan: balancedPlan, probed: true,
                                       passes: probePasses,
                                       probeSeconds: probeSeconds,
                                       deviceNames: deviceNames,
                                       completion: completion)
                            }
                        }
                    }
                }
            }
        }
    }

    /// Blocking wrapper for CLI/test callers (call from a plain thread).
    @discardableResult
    public func setupSync(
        probePasses: Int = 3,
        makeProbeInput: @escaping (Int) -> [Float],
        timeout: TimeInterval = 120
    ) throws -> SetupReport {
        let done = DispatchSemaphore(value: 0)
        var outcome: Result<SetupReport, SetupError>?
        setup(probePasses: probePasses, makeProbeInput: makeProbeInput) { result in
            outcome = result
            done.signal()
        }
        guard done.wait(timeout: .now() + timeout) == .success, let outcome else {
            throw SetupError.timedOut
        }
        return try outcome.get()
    }

    // MARK: Internals (controller queue)

    private func nameIndex() -> [UInt32: String] {
        Dictionary(uniqueKeysWithValues: failover.activePeers.map {
            ($0.peerID, $0.deviceName)
        })
    }

    private func assignPlan(
        completion: @escaping (Result<[NMPShardPlanEntry], NMPFailoverError>) -> Void
    ) {
        failover.assignInitialPlan { [self] result in
            queue.async { completion(result) }
        }
    }

    private func runProbes(
        remaining: Int, pass: Int,
        makeProbeInput: @escaping (Int) -> [Float],
        completion: @escaping (Result<Void, NMPOrchestrationError>) -> Void
    ) {
        guard remaining > 0 else {
            completion(.success(()))
            return
        }
        orchestrator.infer(input: makeProbeInput(pass)) { [self] result in
            queue.async { [self] in
                switch result {
                case .failure(let error):
                    completion(.failure(error))
                case .success(let report):
                    diagnose(String(format: "probe pass %d: %.1f ms across %d stage(s)",
                                    pass + 1, report.totalSeconds * 1000,
                                    report.perShard.count))
                    runProbes(remaining: remaining - 1, pass: pass + 1,
                              makeProbeInput: makeProbeInput, completion: completion)
                }
            }
        }
    }

    private func finish(
        plan: [NMPShardPlanEntry], probed: Bool, passes: Int,
        probeSeconds: TimeInterval, deviceNames: [UInt32: String],
        completion: (Result<SetupReport, SetupError>) -> Void
    ) {
        let measured = orchestrator.measuredSecondsPerLayer

        // Persist what this session learned (merge into the profile).
        var byDevice: [String: Double] = [:]
        for (peerID, rate) in measured {
            if let name = deviceNames[peerID] { byDevice[name] = rate }
        }
        if !byDevice.isEmpty {
            do {
                try store.save(NMPShardingProfile(
                    modelTag: modelTag, secondsPerLayerByDevice: byDevice))
                diagnose("profile saved to \(store.fileURL.path)")
            } catch {
                diagnose("profile save failed (non-fatal): \(error)")
            }
        }

        let balance = NMPShardBalance.evaluate(
            plan: plan, measuredSecondsPerLayer: measured)
        for line in balance.summaryLines(deviceNames: deviceNames) {
            diagnose(line)
        }
        completion(.success(SetupReport(
            probed: probed, probePasses: passes, probeSeconds: probeSeconds,
            plan: plan, balance: balance,
            measuredSecondsPerLayer: measured)))
    }

    private func diagnose(_ message: String) {
        onDiagnostic?(message)
    }
}
