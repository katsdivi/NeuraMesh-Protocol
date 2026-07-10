//
//  AutoConfig.swift
//  NMP — Phase 9
//
//  One-command mesh setup: discover → benchmark → allocate → apply.
//  Everything here delegates to machinery that exists on its own —
//  NMPFailoverOrchestrator (membership + assignment rounds),
//  NMPAdaptiveShardingController (probe passes + balanced re-plan +
//  profile persistence), NMPActivationCodec (wire format) — this file
//  just sequences it and narrates the steps, so
//
//      swift run nmp-dashboard --auto-config
//
//  is the whole setup story. Benchmarking is one-time per (model,
//  device) pair (Part G answer #3): the profile persists in
//  ~/.nmp/neuramesh_sharding.json and later startups skip the probe
//  phase. If probing fails the mesh keeps the naive class-weight plan
//  (Part G answer #4) — auto-config degrades, never blocks.
//

import Foundation

public enum NMPAutoConfig {

    /// The activation wire format an engine should default to under
    /// auto-configuration:
    /// - llama plans move token-state vectors (mostly zero padding) —
    ///   zero-trim is lossless and ~99% smaller;
    /// - reference plans move dense activations — mixed precision halves
    ///   them at ≤ 2^-11 relative rounding on non-critical values
    ///   (Part G answer #2).
    public static func recommendedWireFormat(engineName: String) -> NMPActivationWireFormat {
        engineName.hasPrefix("llama") ? .zeroTrimmed : .mixedPrecision
    }
}

/// Sequences a full automatic setup over an assembled (handshaked but
/// unassigned) mesh.
public final class NMPAutomaticMeshSetup {

    public struct Report: Sendable {
        public let peerCount: Int
        public let adaptive: NMPAdaptiveShardingController.SetupReport
        public let wireFormat: NMPActivationWireFormat
    }

    /// Progress narration ("[auto-config] …" lines); fires on internal
    /// queues — treat as log output.
    public var onDiagnostic: ((String) -> Void)?

    private let failover: NMPFailoverOrchestrator
    private let orchestrator: NMPInferenceOrchestrator
    private let controller: NMPAdaptiveShardingController
    private let engineName: String

    public init(failover: NMPFailoverOrchestrator,
                orchestrator: NMPInferenceOrchestrator,
                modelTag: String,
                engineName: String,
                store: NMPShardingProfileStore = NMPShardingProfileStore()) {
        self.failover = failover
        self.orchestrator = orchestrator
        self.engineName = engineName
        controller = NMPAdaptiveShardingController(
            failover: failover, orchestrator: orchestrator,
            modelTag: modelTag, store: store)
    }

    /// Runs the four setup steps, blocking until the mesh is live and
    /// balanced (call from a plain thread — CLI main is where this runs).
    @discardableResult
    public func runSync(
        probePasses: Int = 3,
        makeProbeInput: @escaping (Int) -> [Float],
        timeout: TimeInterval = 120
    ) throws -> Report {
        controller.onDiagnostic = { [weak self] in self?.diagnose("  " + $0) }

        // 1. Discovery already happened (the mesh is assembled); report it.
        let peers = failover.activePeers
        diagnose("1/4: mesh membership — \(peers.count) device(s)")
        for peer in peers {
            diagnose("  - \(peer.deviceName) (peerID 0x\(String(peer.peerID, radix: 16)), "
                     + "\(peer.computeClass) class)")
        }

        // 2 + 3. Benchmark (or cached profile) and balanced assignment.
        diagnose("2/4: benchmarking / loading cached profile…")
        let adaptive = try controller.setupSync(
            probePasses: probePasses, makeProbeInput: makeProbeInput,
            timeout: timeout)
        diagnose("3/4: shard assignment applied — "
                 + (adaptive.probed
                    ? String(format: "probed %d pass(es) in %.1f s",
                             adaptive.probePasses, adaptive.probeSeconds)
                    : "cached profile, probe skipped"))

        // 4. Wire format.
        let format = NMPAutoConfig.recommendedWireFormat(engineName: engineName)
        orchestrator.activationWireFormat = format
        diagnose("4/4: activation wire format → \(format.rawValue)")
        diagnose("✓ setup complete")

        return Report(peerCount: peers.count, adaptive: adaptive, wireFormat: format)
    }

    private func diagnose(_ message: String) {
        onDiagnostic?(message)
    }
}
