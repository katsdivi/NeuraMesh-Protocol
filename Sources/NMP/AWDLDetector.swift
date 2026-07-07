//
//  AWDLDetector.swift
//  NMP — Phase 3
//
//  Infers AWDL (Apple Wireless Direct Link) contention from two signals and
//  drives traffic suppression while it lasts:
//
//    1. Sender-side loss rate: NACKed sequences vs packets sent over a
//       sliding 100 ms window. Above 5% (with a minimum sample count so a
//       single early loss doesn't trip it) → engage. Note: FEC-recovered
//       losses never generate NACKs, so this signal only sees loss the FEC
//       layer failed to absorb — exactly the loss that matters.
//    2. Receiver-side latency spike: one-way delay samples derived from the
//       header timestamp (sender wall clock). Absolute values are polluted
//       by clock skew, so the detector compares a rolling median against a
//       calm-period baseline and looks for a SHIFT: median exceeding
//       baseline by 2× (and by at least 5 ms, so a near-zero loopback
//       baseline can't trip on noise).
//
//  Suppression clears after 200 ms of calm (loss below 2%, no latency
//  spike). Thresholds are heuristics tuned on test hardware — documented,
//  configurable, and expected to need adjustment on real meshes (Phase 5+).
//
//  Pure state machine over an injected clock: fully deterministic in tests.
//  PeerConnection feeds it events and polls state on its dispatch queue.
//

import Foundation

// MARK: - Configuration

public struct NMPAWDLConfig: Sendable {
    /// Master switch for contention detection + traffic shaping.
    public var enabled = true

    /// Sliding window for the loss-rate signal.
    public var lossWindow: TimeInterval = 0.1
    /// Engage suppression above this NACKed/sent ratio…
    public var engageLossRate = 0.05
    /// …but only once this many packets were sent inside the window.
    public var minSendSamples = 20
    /// Loss rate below which the link counts as calm.
    public var disengageLossRate = 0.02
    /// Continuous calm required before suppression clears.
    public var disengageAfter: TimeInterval = 0.2

    /// Sliding window for the latency-spike signal.
    public var latencyWindow: TimeInterval = 0.05
    /// Spike = rolling median > baseline × factor…
    public var latencySpikeFactor = 2.0
    /// …and at least this far above baseline (guards near-zero baselines
    /// and clock-skewed absolute values).
    public var latencySpikeMinDelta: TimeInterval = 0.005
    /// Latency samples needed before a baseline is locked in.
    public var minLatencySamples = 10

    /// Longest a deferred packet may wait before being sent regardless.
    public var maxDeferDelay: TimeInterval = 0.2
    /// Deferral buffer capacity; beyond this, send() throws.
    public var maxDeferredPackets = 100

    public init() {}
}

// MARK: - Detector

struct NMPAWDLDetector {

    private let config: NMPAWDLConfig
    private var sentTimes: [TimeInterval] = []
    private var lossTimes: [TimeInterval] = []
    private var latencySamples: [(time: TimeInterval, value: Double)] = []
    private var baselineLatency: Double?
    private var calmSince: TimeInterval?

    private(set) var suppressionActive = false

    init(config: NMPAWDLConfig) {
        self.config = config
    }

    // MARK: Event feed

    mutating func recordSent(at now: TimeInterval) {
        sentTimes.append(now)
    }

    /// One NACK packet may report several lost sequences; each is one loss.
    mutating func recordLosses(_ count: Int, at now: TimeInterval) {
        for _ in 0..<count { lossTimes.append(now) }
    }

    /// One-way delay sample (receiver clock − sender header timestamp).
    /// May be negative under clock skew; only shifts matter.
    mutating func recordLatencySample(_ seconds: Double, at now: TimeInterval) {
        latencySamples.append((now, seconds))
    }

    // MARK: State

    var currentLossRate: Double {
        sentTimes.isEmpty ? 0 : Double(lossTimes.count) / Double(sentTimes.count)
    }

    /// Re-evaluates suppression. Returns true if the state flipped.
    @discardableResult
    mutating func updateState(at now: TimeInterval) -> Bool {
        prune(at: now)

        let lossSignal = sentTimes.count >= config.minSendSamples
            && currentLossRate > config.engageLossRate
        let latencySignal = latencySpiked()

        let wasActive = suppressionActive
        if !suppressionActive {
            updateBaseline()
            if lossSignal || latencySignal {
                suppressionActive = true
                calmSince = nil
            }
        } else {
            let calm = !latencySignal
                && (sentTimes.isEmpty || currentLossRate < config.disengageLossRate)
            if calm {
                if let since = calmSince {
                    if now - since >= config.disengageAfter {
                        suppressionActive = false
                        calmSince = nil
                    }
                } else {
                    calmSince = now
                }
            } else {
                calmSince = nil
            }
        }
        return suppressionActive != wasActive
    }

    // MARK: Signals

    private func rollingMedianLatency() -> Double? {
        guard !latencySamples.isEmpty else { return nil }
        let sorted = latencySamples.map(\.value).sorted()
        return sorted[sorted.count / 2]
    }

    private func latencySpiked() -> Bool {
        guard let baseline = baselineLatency,
              let median = rollingMedianLatency() else { return false }
        let requiredDelta = Swift.max(
            config.latencySpikeMinDelta,
            baseline > 0 ? baseline * (config.latencySpikeFactor - 1) : config.latencySpikeMinDelta)
        return median - baseline > requiredDelta
    }

    private mutating func updateBaseline() {
        guard latencySamples.count >= config.minLatencySamples,
              let median = rollingMedianLatency() else { return }
        if let baseline = baselineLatency {
            // Slow EWMA while calm, so the baseline tracks genuine drift
            // without chasing a spike.
            baselineLatency = baseline * 0.9 + median * 0.1
        } else {
            baselineLatency = median
        }
    }

    private mutating func prune(at now: TimeInterval) {
        let lossCutoff = now - config.lossWindow
        sentTimes.removeAll { $0 < lossCutoff }
        lossTimes.removeAll { $0 < lossCutoff }
        let latencyCutoff = now - config.latencyWindow
        latencySamples.removeAll { $0.time < latencyCutoff }
    }
}
