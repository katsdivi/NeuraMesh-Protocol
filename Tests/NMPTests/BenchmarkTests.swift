//
//  BenchmarkTests.swift
//  NMPTests — Phase 6
//
//  The benchmark suite's measurement machinery (percentiles, loss
//  injector determinism, burst expiry, CSV export) plus the headline
//  scenarios at reduced iteration counts: throughput decline under loss,
//  peer-drop spike + recovery, burst-loss recovery, and peer-join
//  re-sharding. Numbers assert DIRECTION and BOUNDS, not exact values —
//  absolute latency depends on the host.
//

import XCTest
@testable import NMP

final class BenchmarkTests: XCTestCase {

    // MARK: Percentiles

    func testLatencyStatsNearestRankPercentiles() throws {
        // 1..100 ms: percentiles are exact by construction.
        let latencies = (1...100).map { TimeInterval($0) / 1000 }
        let stats = try XCTUnwrap(NMPLatencyStats(latencies: latencies.shuffled()))
        XCTAssertEqual(stats.p50, 0.050, accuracy: 1e-9)
        XCTAssertEqual(stats.p95, 0.095, accuracy: 1e-9)
        XCTAssertEqual(stats.p99, 0.099, accuracy: 1e-9)
        XCTAssertEqual(stats.average, 0.0505, accuracy: 1e-9)
        XCTAssertEqual(stats.minimum, 0.001, accuracy: 1e-9)
        XCTAssertEqual(stats.maximum, 0.100, accuracy: 1e-9)

        // Small n never indexes out of range (the classic p99 off-by-one).
        let tiny = try XCTUnwrap(NMPLatencyStats(latencies: [0.005]))
        XCTAssertEqual(tiny.p99, 0.005)
        XCTAssertNil(NMPLatencyStats(latencies: []))
    }

    // MARK: Loss injector

    func testLossInjectorDropsAtConfiguredRate() {
        let (raw, remote) = NMPInMemoryTransport.pair()
        _ = remote
        let injector = NMPPacketLossInjector(wrapping: raw, seed: 0x5EED)

        injector.setLossRate(0)
        for _ in 0..<200 { injector.send(Data([1])) }
        XCTAssertEqual(injector.droppedCount, 0, "rate 0 drops nothing")

        injector.reset()
        injector.setLossRate(0.2)
        for _ in 0..<2000 { injector.send(Data([1])) }
        // Deterministic seed: the exact count is stable; assert the band.
        XCTAssertEqual(Double(injector.droppedCount) / 2000, 0.2, accuracy: 0.05,
                       "20% configured, got \(injector.droppedCount)/2000")

        injector.reset()
        injector.blackhole()
        for _ in 0..<50 { injector.send(Data([1])) }
        XCTAssertEqual(injector.droppedCount, 50, "blackhole drops everything")
    }

    func testBurstLossExpiresBackToBaseRate() {
        let (raw, remote) = NMPInMemoryTransport.pair()
        _ = remote
        let injector = NMPPacketLossInjector(wrapping: raw, seed: 0x5EED)

        injector.setBurstLoss(rate: 1.0, duration: 0.15)
        for _ in 0..<20 { injector.send(Data([1])) }
        XCTAssertEqual(injector.droppedCount, 20, "inside the burst window")

        Thread.sleep(forTimeInterval: 0.25)
        let droppedDuringBurst = injector.droppedCount
        for _ in 0..<20 { injector.send(Data([1])) }
        XCTAssertEqual(injector.droppedCount, droppedDuringBurst,
                       "after expiry the base rate (0) applies")
    }

    // MARK: Scenario: throughput under loss

    func testThroughputDeclinesUnderHeavyLoss() throws {
        // 4 KB tensors → 4 chunks per direction per stage. Measured
        // behavior of this stack: at ≤10% loss FEC + flush-expedited NACKs
        // recover essentially for free (no p50 movement — that is the
        // protocol working as designed, verified by the recovery-event
        // counters below). Degradation becomes unambiguous at 15%, where
        // NACK rounds start getting lost themselves: ~25% throughput drop
        // and a 4-5× p95 tail on this hardware. The assertions use half
        // those margins.
        let testbed = try NMPMeshTestbed(layerCount: 8, hiddenSize: 1024,
                                         remotePeerCount: 1)
        _ = try testbed.startSync()

        var sawRecoveries = false
        testbed.onPacketEvent = { _, event in
            switch event {
            case .fecRecovered, .retransmitted: sawRecoveries = true
            default: break
            }
        }

        let suite = NMPBenchmarkSuite(testbed: testbed)
        suite.log = { _ in } // quiet

        _ = try suite.benchmark(name: "warmup", generations: 2, tokens: 4)
        let clean = try suite.benchmark(name: "clean", generations: 12, tokens: 4,
                                        lossRate: 0)
        XCTAssertFalse(sawRecoveries, "clean run must need no recoveries")

        let lossy = try suite.benchmark(name: "15% loss", generations: 12, tokens: 4,
                                        lossRate: 0.15)

        XCTAssertTrue(sawRecoveries,
                      "15% loss must exercise the FEC/NACK recovery path")
        XCTAssertLessThan(lossy.throughputTokensPerSecond,
                          clean.throughputTokensPerSecond * 0.92,
                          "throughput must decline under 15% loss: clean "
                          + "\(clean.throughputTokensPerSecond) vs lossy "
                          + "\(lossy.throughputTokensPerSecond) tokens/s")
        XCTAssertGreaterThan(lossy.stats.p95, clean.stats.p95 * 1.5,
                             "loss must show up in the latency tail: clean p95 "
                             + "\(clean.stats.p95 * 1000) ms vs lossy p95 "
                             + "\(lossy.stats.p95 * 1000) ms")
    }

    // MARK: Scenario: peer drop

    func testPeerDropSpikesThenRecovers() throws {
        let testbed = try NMPMeshTestbed(layerCount: 12, hiddenSize: 512,
                                         remotePeerCount: 2)
        _ = try testbed.startSync()
        let suite = NMPBenchmarkSuite(testbed: testbed)
        suite.log = { _ in }

        let result = try suite.runPeerDrop(generations: 9, tokens: 4, dropAfter: 4)

        // The scenario itself verifies bit-exactness every pass; here we
        // check the recovery shape: the mesh kept producing after the drop
        // and the re-shard met its budget.
        XCTAssertEqual(result.latencies.count, 9, "no generation was lost")
        let reshard = try XCTUnwrap(testbed.failover.lastReshardSeconds)
        XCTAssertLessThan(reshard, 0.5, "re-shard budget")
        XCTAssertEqual(testbed.failover.activePlan.count, 2,
                       "pipeline shrank to the survivors")

        // Post-drop generations settle back near the clean baseline:
        // the last generation must not still carry failover cost.
        let preDrop = Array(result.latencies[0..<4])
        let settled = Array(result.latencies.suffix(3))
        let preDropMedian = preDrop.sorted()[preDrop.count / 2]
        let settledMedian = settled.sorted()[settled.count / 2]
        XCTAssertLessThan(settledMedian, preDropMedian * 5,
                          "post-failover latency must normalize (pre "
                          + "\(preDropMedian * 1000) ms, settled "
                          + "\(settledMedian * 1000) ms)")
    }

    // MARK: Scenario: burst loss (AWDL-like)

    func testBurstLossRecoveryWithinOneSecond() throws {
        let testbed = try NMPMeshTestbed(layerCount: 8, hiddenSize: 1024,
                                         remotePeerCount: 1)
        _ = try testbed.startSync()
        let suite = NMPBenchmarkSuite(testbed: testbed)
        suite.log = { _ in }

        let result = try suite.runBurstLoss(
            generations: 8, tokens: 4,
            burstRate: 0.10, burstDuration: 0.3, burstGenerations: 2...4)

        XCTAssertEqual(result.latencies.count, 8,
                       "every generation completed despite the bursts "
                       + "(bit-exactness checked inside the scenario)")
        // Recovery: the first fully-clean generation after the burst window
        // finishes within 1 s.
        let postBurst = result.latencies[6]
        XCTAssertLessThan(postBurst, 1.0,
                          "latency must recover <1 s after the burst; got "
                          + "\(postBurst * 1000) ms")
    }

    // MARK: Scenario: peer join

    func testPeerJoinReshardsAndKeepsMeasuring() throws {
        let testbed = try NMPMeshTestbed(layerCount: 12, hiddenSize: 512,
                                         remotePeerCount: 1)
        _ = try testbed.startSync()
        XCTAssertEqual(testbed.failover.activePlan.count, 2)
        let suite = NMPBenchmarkSuite(testbed: testbed)
        suite.log = { _ in }

        let result = try suite.runPeerJoin(generations: 8, tokens: 4, joinAfter: 3)

        XCTAssertEqual(result.latencies.count, 8)
        XCTAssertEqual(testbed.failover.activePlan.count, 3,
                       "the joined peer serves a shard")
        let reshard = try XCTUnwrap(testbed.failover.lastReshardSeconds)
        XCTAssertLessThan(reshard, 0.5, "join re-shard budget")
    }

    // MARK: CSV export

    func testCSVExportWritesSummaryAndRawLatencies() throws {
        let testbed = try NMPMeshTestbed(layerCount: 8, hiddenSize: 96,
                                         remotePeerCount: 1)
        _ = try testbed.startSync()
        let suite = NMPBenchmarkSuite(testbed: testbed)
        suite.log = { _ in }
        _ = try suite.benchmark(name: "csv, test", generations: 3, tokens: 2)

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("nmp-bench-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try suite.exportCSV(to: directory)

        let summary = try String(
            contentsOf: directory.appendingPathComponent("benchmark_summary.csv"),
            encoding: .utf8)
        let summaryLines = summary.split(separator: "\n")
        XCTAssertTrue(summaryLines[0].hasPrefix("scenario,loss_rate,"))
        XCTAssertEqual(summaryLines.count, 2, "header + one scenario")
        XCTAssertTrue(summaryLines[1].hasPrefix("\"csv, test\""),
                      "comma in scenario name must be quoted")

        let raw = try String(
            contentsOf: directory.appendingPathComponent("benchmark_latencies.csv"),
            encoding: .utf8)
        XCTAssertEqual(raw.split(separator: "\n").count, 1 + 3,
                       "header + one row per generation")
    }
}
