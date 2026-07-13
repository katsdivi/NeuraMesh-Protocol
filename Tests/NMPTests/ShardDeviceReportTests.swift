//
//  ShardDeviceReportTests.swift
//  NMP — Phase 10 surfacing contract (pure Swift, always runs)
//

import XCTest
@testable import NMP

final class ShardDeviceReportTests: XCTestCase {

    private func plan2way() -> [NMPShardPlanEntry] {
        [NMPShardPlanEntry(peerID: 0x2, shardIndex: 0, startLayer: 0, endLayer: 12),
         NMPShardPlanEntry(peerID: 0x3, shardIndex: 1, startLayer: 12, endLayer: 24)]
    }

    func testDeviceRowsCarryRangeAndLoadedMB() {
        let devices = NMPShardReport.devices(
            plan: plan2way(),
            loadedBytesByPeer: [0x2: 219_500_000, 0x3: 265_900_000])
        XCTAssertEqual(devices.count, 2)
        XCTAssertEqual(devices[0].layerRange, "layers 0-11")
        XCTAssertEqual(devices[0].layerCount, 12)
        XCTAssertEqual(devices[0].loadedMB, 209.3, accuracy: 0.2)
        XCTAssertEqual(devices[1].layerRange, "layers 12-23")
        XCTAssertTrue(devices[0].summary.contains("layers 0-11"))
        XCTAssertTrue(devices[0].summary.contains("MB"))
    }

    func testMissingBytesFallsBackToRangeOnly() {
        let devices = NMPShardReport.devices(plan: plan2way(), loadedBytesByPeer: [:])
        XCTAssertEqual(devices[0].loadedBytes, 0)
        XCTAssertEqual(devices[0].summary, "layers 0-11") // no MB when unknown
    }

    func testHonestyInvariantHoldsForRealSplit() {
        // Each shard < whole model → invariant holds.
        let devices = NMPShardReport.devices(
            plan: plan2way(),
            loadedBytesByPeer: [0x2: 219_500_000, 0x3: 265_900_000])
        XCTAssertTrue(NMPShardReport.noPeerHoldsWholeModel(
            devices, fullModelBytes: 490_000_000))
    }

    func testHonestyInvariantFailsIfAPeerHoldsEverything() {
        // A peer that loaded the whole model is NOT a real split.
        let devices = NMPShardReport.devices(
            plan: plan2way(),
            loadedBytesByPeer: [0x2: 490_000_000, 0x3: 265_900_000])
        XCTAssertFalse(NMPShardReport.noPeerHoldsWholeModel(
            devices, fullModelBytes: 490_000_000))
    }

    func testEmptySpanRendersZeroLayers() {
        let standby = NMPShardDeviceInfo(
            peerID: 0x9, shardIndex: 3, startLayer: 24, endLayer: 24, loadedBytes: 0)
        XCTAssertEqual(standby.layerRange, "0 layers")
        XCTAssertEqual(standby.layerCount, 0)
    }
}
