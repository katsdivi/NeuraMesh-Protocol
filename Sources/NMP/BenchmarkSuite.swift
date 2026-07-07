//
//  BenchmarkSuite.swift
//  NMP — Phase 6
//
//  Stress measurement over an NMPMeshTestbed: latency distribution
//  (p50/p95/p99), throughput under steady loss, peer-drop failover cost,
//  dynamic peer join, and AWDL-like burst loss.
//
//  Terminology: one "generation" of T tokens = T sequential pipeline
//  passes, each feeding its output activations back as the next input —
//  the shape of autoregressive decoding. Latencies are per generation;
//  throughput is tokens/sec across the whole scenario.
//
//  Every scenario verifies output BIT-EXACT against the single-device
//  baseline as it measures: a benchmark that silently corrupted tensors
//  would be measuring garbage.
//
//  Blocking style: runs on the caller's thread (CLI main or a test).
//

import Foundation

// MARK: - Latency statistics

public struct NMPLatencyStats: Sendable {
    public let p50: TimeInterval
    public let p95: TimeInterval
    public let p99: TimeInterval
    public let average: TimeInterval
    public let minimum: TimeInterval
    public let maximum: TimeInterval

    /// Nearest-rank percentiles (index = ceil(q·n) - 1, clamped) — exact
    /// for small n, no interpolation surprises.
    public init?(latencies: [TimeInterval]) {
        guard !latencies.isEmpty else { return nil }
        let sorted = latencies.sorted()
        func rank(_ q: Double) -> TimeInterval {
            let index = Int((q * Double(sorted.count)).rounded(.up)) - 1
            return sorted[max(0, min(sorted.count - 1, index))]
        }
        p50 = rank(0.50)
        p95 = rank(0.95)
        p99 = rank(0.99)
        average = latencies.reduce(0, +) / Double(latencies.count)
        minimum = sorted.first!
        maximum = sorted.last!
    }
}

// MARK: - Result

public struct NMPBenchmarkResult: Sendable {
    public let name: String
    /// Steady loss rate configured for the scenario (bursts noted in `notes`).
    public let lossRate: Double
    public let tokensPerGeneration: Int
    /// One entry per generation, in run order.
    public let latencies: [TimeInterval]
    public let stats: NMPLatencyStats
    /// Total tokens generated / total wall-clock seconds.
    public let throughputTokensPerSecond: Double
    public let notes: String

    public var summaryLines: [String] {
        [
            "\(name):",
            String(format: "    p50: %8.1f ms", stats.p50 * 1000),
            String(format: "    p95: %8.1f ms", stats.p95 * 1000),
            String(format: "    p99: %8.1f ms", stats.p99 * 1000),
            String(format: "    avg: %8.1f ms", stats.average * 1000),
            String(format: "    throughput: %.1f tokens/s", throughputTokensPerSecond),
        ] + (notes.isEmpty ? [] : ["    \(notes)"])
    }
}

// MARK: - Suite

public final class NMPBenchmarkSuite {

    public enum BenchmarkError: Error {
        case outputMismatch(scenario: String, generation: Int)
    }

    /// Sink for progress lines (default: stdout). The dashboard CLI
    /// redirects this into the packet log.
    public var log: (String) -> Void = { print($0) }

    public private(set) var results: [NMPBenchmarkResult] = []

    private let testbed: NMPMeshTestbed

    public init(testbed: NMPMeshTestbed) {
        self.testbed = testbed
    }

    // MARK: Core measured loop

    /// Runs `generations` generations of `tokens` pipeline passes each,
    /// verifying every pass bit-exact against the reference engine.
    ///
    /// `stageTimeout` is deliberately short (stages complete in
    /// milliseconds on a healthy link): under heavy loss a stage whose
    /// chunks the NACK layer gave up on costs one timeout + retry, so
    /// give-up events show up as bounded latency spikes instead of
    /// 30-second stalls.
    @discardableResult
    public func benchmark(
        name: String,
        generations: Int = 10,
        tokens: Int = 8,
        lossRate: Double = 0,
        stageTimeout: TimeInterval = 1.0,
        notes: String = "",
        onGeneration: ((Int) throws -> Void)? = nil
    ) throws -> NMPBenchmarkResult {
        testbed.setLossRate(lossRate)
        defer { testbed.setLossRate(0) }

        var latencies: [TimeInterval] = []
        let suiteBegan = DispatchTime.now()

        for generation in 0..<generations {
            try onGeneration?(generation)

            var activations = testbed.makeInput(seed: 0xBEEF &+ UInt64(generation))
            var expected = activations
            let began = DispatchTime.now()
            for _ in 0..<tokens {
                let report = try testbed.inferSync(input: activations,
                                                   stageTimeout: stageTimeout)
                activations = report.output
                expected = try testbed.baselineOutput(for: expected)
                guard report.output.map(\.bitPattern) == expected.map(\.bitPattern) else {
                    throw BenchmarkError.outputMismatch(scenario: name, generation: generation)
                }
            }
            latencies.append(TimeInterval(
                DispatchTime.now().uptimeNanoseconds - began.uptimeNanoseconds) / 1e9)
        }

        let totalSeconds = TimeInterval(
            DispatchTime.now().uptimeNanoseconds - suiteBegan.uptimeNanoseconds) / 1e9
        let result = NMPBenchmarkResult(
            name: name,
            lossRate: lossRate,
            tokensPerGeneration: tokens,
            latencies: latencies,
            stats: NMPLatencyStats(latencies: latencies)!,
            throughputTokensPerSecond: Double(generations * tokens) / totalSeconds,
            notes: notes)
        results.append(result)
        return result
    }

    // MARK: Scenarios

    /// Scenario 1: clean network, growing generation length.
    public func runTokenScaling(tokenCounts: [Int] = [4, 8, 16, 32]) throws {
        log("Scenario 1: no loss, varying token counts")
        for tokens in tokenCounts {
            let result = try benchmark(name: "no loss, \(tokens) tokens", tokens: tokens)
            printResult(result)
        }
    }

    /// Scenario 2: steady loss sweep. 0.15 included deliberately: it is
    /// where this stack's recovery stops being free (see Benchmarks.md).
    public func runLossSweep(rates: [Double] = [0, 0.01, 0.02, 0.05, 0.10, 0.15],
                             tokens: Int = 8) throws {
        log("Scenario 2: loss rate impact (\(tokens) tokens)")
        for rate in rates {
            let result = try benchmark(
                name: String(format: "loss %.0f%%", rate * 100),
                tokens: tokens, lossRate: rate)
            printResult(result)
        }
    }

    /// Scenario 3: a peer dies mid-benchmark; measures the latency spike
    /// and the recovery. Fails if the mesh has no droppable remote peer.
    @discardableResult
    public func runPeerDrop(generations: Int = 15, tokens: Int = 8,
                            dropAfter: Int = 5) throws -> NMPBenchmarkResult {
        log("Scenario 3: peer drop after generation \(dropAfter)")
        var reshardMS = 0.0
        let result = try benchmark(
            name: "peer drop after gen \(dropAfter)",
            generations: generations, tokens: tokens,
            notes: "re-shard pending"
        ) { [self] generation in
            guard generation == dropAfter,
                  let victim = testbed.remotePeers.last?.capabilities.peerID else { return }
            let plan = try testbed.dropPeerSync(victim)
            reshardMS = (testbed.failover.lastReshardSeconds ?? 0) * 1000
            log(String(format: "  [gen %d] peer 0x%x dropped, re-sharded to %d shard(s) in %.1f ms",
                       generation, victim, plan.count, reshardMS))
        }
        // Rebuild with the measured note (struct is immutable by design).
        let annotated = NMPBenchmarkResult(
            name: result.name, lossRate: result.lossRate,
            tokensPerGeneration: result.tokensPerGeneration,
            latencies: result.latencies, stats: result.stats,
            throughputTokensPerSecond: result.throughputTokensPerSecond,
            notes: String(format: "re-shard took %.1f ms", reshardMS))
        results[results.count - 1] = annotated
        printResult(annotated)
        return annotated
    }

    /// Scenario 4: a new peer joins mid-benchmark and the mesh re-shards
    /// across it.
    @discardableResult
    public func runPeerJoin(generations: Int = 15, tokens: Int = 8,
                            joinAfter: Int = 8) throws -> NMPBenchmarkResult {
        log("Scenario 4: peer join after generation \(joinAfter)")
        var reshardMS = 0.0
        let result = try benchmark(
            name: "peer join after gen \(joinAfter)",
            generations: generations, tokens: tokens,
            notes: "re-shard pending"
        ) { [self] generation in
            guard generation == joinAfter else { return }
            let peer = try testbed.joinNewPeer()
            reshardMS = (testbed.failover.lastReshardSeconds ?? 0) * 1000
            log(String(format: "  [gen %d] peer 0x%x joined, re-sharded in %.1f ms",
                       generation, peer.capabilities.peerID, reshardMS))
        }
        let annotated = NMPBenchmarkResult(
            name: result.name, lossRate: result.lossRate,
            tokensPerGeneration: result.tokensPerGeneration,
            latencies: result.latencies, stats: result.stats,
            throughputTokensPerSecond: result.throughputTokensPerSecond,
            notes: String(format: "re-shard took %.1f ms", reshardMS))
        results[results.count - 1] = annotated
        printResult(annotated)
        return annotated
    }

    /// Scenario 5: AWDL-like burst — elevated loss for a short window in
    /// the middle generations, clean elsewhere.
    @discardableResult
    public func runBurstLoss(generations: Int = 10, tokens: Int = 8,
                             burstRate: Double = 0.10,
                             burstDuration: TimeInterval = 0.3,
                             burstGenerations: ClosedRange<Int> = 3...6) throws
        -> NMPBenchmarkResult {
        log(String(format: "Scenario 5: burst loss %.0f%% for %.0f ms",
                   burstRate * 100, burstDuration * 1000))
        let result = try benchmark(
            name: String(format: "burst %.0f%% / %.0f ms", burstRate * 100, burstDuration * 1000),
            generations: generations, tokens: tokens,
            notes: String(format: "burst during generations %d-%d",
                          burstGenerations.lowerBound, burstGenerations.upperBound)
        ) { [self] generation in
            if burstGenerations.contains(generation) {
                testbed.setBurstLoss(rate: burstRate, duration: burstDuration)
            }
        }
        printResult(result)
        return result
    }

    /// All five scenarios back to back. NOTE: peer drop and join mutate
    /// mesh membership; drop runs last-but-one and join restores a peer.
    public func runComprehensive() throws -> [NMPBenchmarkResult] {
        log("\n=== COMPREHENSIVE NeuraMesh BENCHMARK ===\n")
        try runTokenScaling()
        try runLossSweep()
        try runBurstLoss()
        try runPeerDrop()
        try runPeerJoin()
        return results
    }

    // MARK: Reporting

    public func printResult(_ result: NMPBenchmarkResult) {
        for line in result.summaryLines { log("  " + line) }
    }

    /// Two CSVs: a per-scenario summary and the raw per-generation
    /// latencies (for plotting distributions).
    public func exportCSV(to directory: URL) throws {
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)

        var summary = "scenario,loss_rate,tokens_per_generation,generations,"
            + "p50_ms,p95_ms,p99_ms,avg_ms,min_ms,max_ms,throughput_tokens_per_s,notes\n"
        for result in results {
            let s = result.stats
            summary += [
                csvEscape(result.name),
                String(format: "%.4f", result.lossRate),
                String(result.tokensPerGeneration),
                String(result.latencies.count),
                String(format: "%.3f", s.p50 * 1000),
                String(format: "%.3f", s.p95 * 1000),
                String(format: "%.3f", s.p99 * 1000),
                String(format: "%.3f", s.average * 1000),
                String(format: "%.3f", s.minimum * 1000),
                String(format: "%.3f", s.maximum * 1000),
                String(format: "%.2f", result.throughputTokensPerSecond),
                csvEscape(result.notes),
            ].joined(separator: ",") + "\n"
        }
        try summary.write(to: directory.appendingPathComponent("benchmark_summary.csv"),
                          atomically: true, encoding: .utf8)

        var raw = "scenario,generation,latency_ms\n"
        for result in results {
            for (index, latency) in result.latencies.enumerated() {
                raw += "\(csvEscape(result.name)),\(index),"
                    + String(format: "%.3f\n", latency * 1000)
            }
        }
        try raw.write(to: directory.appendingPathComponent("benchmark_latencies.csv"),
                      atomically: true, encoding: .utf8)
    }

    private func csvEscape(_ field: String) -> String {
        if field.contains(",") || field.contains("\"") || field.contains("\n") {
            return "\"" + field.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        }
        return field
    }
}
