//
//  AdaptiveShardingTests.swift
//  NMP — Phase 9
//
//  Pins the adaptive layer-balancing loop: balance math, profile
//  persistence (including corrupt-cache tolerance), and the end-to-end
//  controller behavior over a real heterogeneous in-process mesh —
//  probe passes must shift layers away from a deliberately slowed peer,
//  and a second startup must reuse the persisted profile without
//  re-probing.
//

import XCTest
@testable import NMP

// MARK: - Balance math

final class ShardBalanceTests: XCTestCase {

    func testEvaluateUsesMeasurementsAndFindsBottleneck() {
        let plan = [
            NMPShardPlanEntry(peerID: 1, shardIndex: 0, startLayer: 0, endLayer: 6),
            NMPShardPlanEntry(peerID: 2, shardIndex: 1, startLayer: 6, endLayer: 12),
        ]
        let balance = NMPShardBalance.evaluate(
            plan: plan, measuredSecondsPerLayer: [1: 0.001, 2: 0.003])
        // Stage seconds: 6 ms and 18 ms → pipeline 18 ms, mean 12 ms.
        XCTAssertEqual(balance.pipelineSeconds, 0.018, accuracy: 1e-9)
        XCTAssertEqual(balance.balanceEfficiency, 12.0 / 18.0, accuracy: 1e-9)
    }

    func testUnmeasuredPeersBorrowTheMeanRate() {
        let plan = [
            NMPShardPlanEntry(peerID: 1, shardIndex: 0, startLayer: 0, endLayer: 4),
            NMPShardPlanEntry(peerID: 9, shardIndex: 1, startLayer: 4, endLayer: 8),
        ]
        let balance = NMPShardBalance.evaluate(
            plan: plan, measuredSecondsPerLayer: [1: 0.002])
        XCTAssertEqual(balance.stages[1].estimatedSeconds ?? -1, 0.008, accuracy: 1e-9)
        // No measurements at all → no estimates, zero efficiency (unknown).
        let unknown = NMPShardBalance.evaluate(plan: plan, measuredSecondsPerLayer: [:])
        XCTAssertNil(unknown.stages[0].estimatedSeconds)
        XCTAssertEqual(unknown.balanceEfficiency, 0)
    }

    func testPerfectlyBalancedPlanScoresOne() {
        let plan = [
            NMPShardPlanEntry(peerID: 1, shardIndex: 0, startLayer: 0, endLayer: 9),
            NMPShardPlanEntry(peerID: 2, shardIndex: 1, startLayer: 9, endLayer: 12),
        ]
        // 9 layers × 1 ms == 3 layers × 3 ms.
        let balance = NMPShardBalance.evaluate(
            plan: plan, measuredSecondsPerLayer: [1: 0.001, 2: 0.003])
        XCTAssertEqual(balance.balanceEfficiency, 1.0, accuracy: 1e-9)
    }
}

// MARK: - Profile persistence

final class ShardingProfileStoreTests: XCTestCase {

    private func temporaryStore() -> NMPShardingProfileStore {
        NMPShardingProfileStore(fileURL: FileManager.default.temporaryDirectory
            .appendingPathComponent("nmp-tests-\(UUID().uuidString)/sharding.json"))
    }

    func testSaveLoadRoundTripAndMerge() throws {
        let store = temporaryStore()
        defer { store.clear() }
        XCTAssertEqual(store.load(), [])

        try store.save(NMPShardingProfile(
            modelTag: "model-a",
            secondsPerLayerByDevice: ["mac": 0.002, "iphone": 0.0022]))
        try store.save(NMPShardingProfile(
            modelTag: "model-b", secondsPerLayerByDevice: ["mac": 0.01]))
        // Same model, one device re-measured, one added → merged.
        try store.save(NMPShardingProfile(
            modelTag: "model-a",
            secondsPerLayerByDevice: ["iphone": 0.0025, "ipad": 0.003]))

        let profile = try XCTUnwrap(store.profile(forModelTag: "model-a"))
        XCTAssertEqual(profile.secondsPerLayerByDevice,
                       ["mac": 0.002, "iphone": 0.0025, "ipad": 0.003])
        XCTAssertEqual(store.load().count, 2)
    }

    func testCorruptCacheFileLoadsAsEmptyInsteadOfThrowing() throws {
        let store = temporaryStore()
        defer { store.clear() }
        try FileManager.default.createDirectory(
            at: store.fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try Data("not json{{{".utf8).write(to: store.fileURL)
        XCTAssertEqual(store.load(), [])
        // And saving over a corrupt file recovers it.
        try store.save(NMPShardingProfile(
            modelTag: "m", secondsPerLayerByDevice: ["d": 1]))
        XCTAssertEqual(store.load().count, 1)
    }
}

// MARK: - Controller over a heterogeneous mesh

final class AdaptiveShardingControllerTests: XCTestCase {

    private func temporaryStore() -> NMPShardingProfileStore {
        NMPShardingProfileStore(fileURL: FileManager.default.temporaryDirectory
            .appendingPathComponent("nmp-tests-\(UUID().uuidString)/sharding.json"))
    }

    /// One peer is 4× slower than the other. The naive plan splits layers
    /// by class weight (all .high → evenly); after probing, the slow peer
    /// must hold FEWER layers than the fast one, and the profile must be
    /// on disk.
    func testProbePassesRebalanceAHeterogeneousMesh() throws {
        let testbed = try NMPMeshTestbed(
            layerCount: 12, hiddenSize: 256, remotePeerCount: 2,
            simulatedSecondsPerLayer: 0.002,
            simulatedPeerSlowdowns: [1.0, 4.0])
        let store = temporaryStore()
        defer { store.clear() }

        let controller = NMPAdaptiveShardingController(
            failover: testbed.failover, orchestrator: testbed.orchestrator,
            modelTag: testbed.modelTag, store: store)
        let report = try controller.setupSync(
            probePasses: 2, makeProbeInput: { testbed.makeInput(seed: UInt64($0)) })

        XCTAssertTrue(report.probed)
        let fastPeer = testbed.remotePeers[0].capabilities.peerID
        let slowPeer = testbed.remotePeers[1].capabilities.peerID
        let fastSpan = try XCTUnwrap(report.plan.first { $0.peerID == fastPeer }).layerSpan
        let slowSpan = try XCTUnwrap(report.plan.first { $0.peerID == slowPeer }).layerSpan
        XCTAssertLessThan(slowSpan, fastSpan,
                          "the 4× slower peer must receive fewer layers")

        // The rebalanced plan still covers every layer contiguously.
        XCTAssertEqual(report.plan.map(\.layerSpan).reduce(0, +), 12)

        // Measurements reflect the injected slowdown (~4×, generous band
        // for scheduler noise).
        let measured = report.measuredSecondsPerLayer
        let fastRate = try XCTUnwrap(measured[fastPeer])
        let slowRate = try XCTUnwrap(measured[slowPeer])
        XCTAssertGreaterThan(slowRate / fastRate, 2.0)

        // Profile persisted for every device (coordinator included).
        let profile = try XCTUnwrap(store.profile(forModelTag: testbed.modelTag))
        XCTAssertEqual(profile.secondsPerLayerByDevice.count, 3)

        // The balanced mesh still computes the right answer.
        let input = testbed.makeInput(seed: 0xADA)
        let output = try testbed.inferSync(input: input)
        XCTAssertEqual(output.output.map(\.bitPattern),
                       try testbed.baselineOutput(for: input).map(\.bitPattern))
    }

    /// A complete cached profile skips the probe phase entirely (Part G
    /// answer #3: benchmark once, reuse forever) — and still produces a
    /// balanced plan, because the cache seeds the sharder.
    func testSecondStartupUsesCachedProfileWithoutProbing() throws {
        let store = temporaryStore()
        defer { store.clear() }

        // Session 1: probe and persist.
        let first = try NMPMeshTestbed(
            layerCount: 12, hiddenSize: 256, remotePeerCount: 2,
            simulatedSecondsPerLayer: 0.002,
            simulatedPeerSlowdowns: [1.0, 4.0])
        _ = try NMPAdaptiveShardingController(
            failover: first.failover, orchestrator: first.orchestrator,
            modelTag: first.modelTag, store: store
        ).setupSync(probePasses: 2,
                    makeProbeInput: { first.makeInput(seed: UInt64($0)) })

        // Session 2: same device names, fresh mesh — must skip probing.
        let second = try NMPMeshTestbed(
            layerCount: 12, hiddenSize: 256, remotePeerCount: 2,
            simulatedSecondsPerLayer: 0.002,
            simulatedPeerSlowdowns: [1.0, 4.0])
        let report = try NMPAdaptiveShardingController(
            failover: second.failover, orchestrator: second.orchestrator,
            modelTag: second.modelTag, store: store
        ).setupSync(probePasses: 2,
                    makeProbeInput: { second.makeInput(seed: UInt64($0)) })

        XCTAssertFalse(report.probed, "complete profile cache must skip probing")
        let slowPeer = second.remotePeers[1].capabilities.peerID
        let fastPeer = second.remotePeers[0].capabilities.peerID
        XCTAssertLessThan(
            try XCTUnwrap(report.plan.first { $0.peerID == slowPeer }).layerSpan,
            try XCTUnwrap(report.plan.first { $0.peerID == fastPeer }).layerSpan,
            "cached measurements must balance the first plan directly")
    }

    /// Zero probe passes keeps the naive plan (the fallback path).
    func testZeroProbePassesKeepsNaivePlan() throws {
        let testbed = try NMPMeshTestbed(layerCount: 12, hiddenSize: 256,
                                         remotePeerCount: 2)
        let store = temporaryStore()
        defer { store.clear() }
        let report = try NMPAdaptiveShardingController(
            failover: testbed.failover, orchestrator: testbed.orchestrator,
            modelTag: testbed.modelTag, store: store
        ).setupSync(probePasses: 0,
                    makeProbeInput: { testbed.makeInput(seed: UInt64($0)) })
        XCTAssertFalse(report.probed)
        XCTAssertEqual(report.plan.map(\.layerSpan).reduce(0, +), 12)
    }
}

// MARK: - Auto-config facade

final class AutoConfigTests: XCTestCase {

    func testRecommendedWireFormats() {
        XCTAssertEqual(NMPAutoConfig.recommendedWireFormat(engineName: "llamaCpp"),
                       .zeroTrimmed)
        XCTAssertEqual(NMPAutoConfig.recommendedWireFormat(engineName: "reference"),
                       .mixedPrecision)
    }

    func testAutomaticSetupBalancesAndAppliesWireFormat() throws {
        let testbed = try NMPMeshTestbed(
            layerCount: 12, hiddenSize: 256, remotePeerCount: 2,
            simulatedSecondsPerLayer: 0.001,
            simulatedPeerSlowdowns: [1.0, 3.0])
        let store = NMPShardingProfileStore(
            fileURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("nmp-tests-\(UUID().uuidString)/sharding.json"))
        defer { store.clear() }

        let setup = NMPAutomaticMeshSetup(
            failover: testbed.failover, orchestrator: testbed.orchestrator,
            modelTag: testbed.modelTag, engineName: "reference", store: store)
        let report = try setup.runSync(
            probePasses: 2, makeProbeInput: { testbed.makeInput(seed: UInt64($0)) })

        XCTAssertEqual(report.peerCount, 3) // coordinator + 2 remotes
        XCTAssertEqual(report.wireFormat, .mixedPrecision)
        XCTAssertEqual(testbed.orchestrator.activationWireFormat, .mixedPrecision)
        XCTAssertGreaterThan(report.adaptive.balance.balanceEfficiency, 0.5)

        // The auto-configured mesh serves inference (compressed wire).
        let result = try testbed.inferSync(input: testbed.makeInput(seed: 7))
        XCTAssertEqual(result.output.count, 256)
    }
}
