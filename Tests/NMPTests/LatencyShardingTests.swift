//
//  LatencyShardingTests.swift
//  NMPTests — network-aware (latency-optimal) balancing
//
//  planByLatency minimizes measured per-token wall-clock
//  (Σ compute + Σ round trips), not per-stage compute balance. These pin
//  the behaviors that motivated it: a fast-but-far peer is left out for a
//  small model (Mac-only wins), pulled in when its compute saving beats its
//  hop, and forced in when capacity requires it.
//

import XCTest
@testable import NMP

final class LatencyShardingTests: XCTestCase {

    private let coordID: UInt32 = 1
    private let phoneID: UInt32 = 2

    private func caps(_ id: UInt32, _ cls: NMPComputeClass, ramMB: UInt32 = 16384)
        -> NMPCapabilities {
        NMPCapabilities(peerID: id, deviceName: "d\(id)", ramMB: ramMB,
                        computeClass: cls)
    }

    private func layers(_ plan: NMPShardPlan) -> [UInt32: Int] {
        Dictionary(plan.entries.map { ($0.peerID, $0.layerSpan) },
                   uniquingKeysWith: +)
    }

    func testSmallModelExcludesFarPhone() {
        // Coordinator and phone compute at the SAME rate, but the phone costs
        // a 40 ms round trip. For a small model the round trip dwarfs any
        // compute saving, so the phone should get nothing — Mac-only.
        let plan = NMPModelSharder.planByLatency(
            layerCount: 24,
            coordinatorPeerID: coordID,
            peers: [caps(coordID, .high), caps(phoneID, .high)],
            computeSecondsPerLayer: [coordID: 0.002, phoneID: 0.002],
            roundTripSeconds: [phoneID: 0.040])
        XCTAssertEqual(layers(plan)[coordID], 24)
        XCTAssertNil(layers(plan)[phoneID])
        XCTAssertEqual(plan.exclusions.first?.peerID, phoneID)
    }

    func testFasterPhoneWithCheapHopIsUsed() {
        // Phone computes 5× faster and the hop is tiny: offloading clearly
        // wins, so it should take (nearly) everything.
        let plan = NMPModelSharder.planByLatency(
            layerCount: 24,
            coordinatorPeerID: coordID,
            peers: [caps(coordID, .high), caps(phoneID, .high)],
            computeSecondsPerLayer: [coordID: 0.010, phoneID: 0.002],
            roundTripSeconds: [phoneID: 0.001])
        XCTAssertGreaterThan(layers(plan)[phoneID] ?? 0, 12)
        XCTAssertTrue(plan.exclusions.isEmpty)
    }

    func testCapacityForcesDistribution() {
        // A model too big for the coordinator's RAM MUST spill to the phone
        // even though the phone is slower and far — capacity beats latency.
        // 100 layers × 1 GB/layer; coordinator holds ~40, phone ~80.
        let gigPerLayer = 1024 * 1_048_576
        let plan = NMPModelSharder.planByLatency(
            layerCount: 100,
            coordinatorPeerID: coordID,
            peers: [caps(coordID, .high, ramMB: 64 * 1024),   // ~40 layers @0.6
                    caps(phoneID, .medium, ramMB: 128 * 1024)],
            computeSecondsPerLayer: [coordID: 0.002, phoneID: 0.020],
            roundTripSeconds: [phoneID: 0.050],
            layerCapacities: [
                coordID: NMPModelSharder.layerCapacity(
                    ramMB: 64 * 1024, bytesPerLayer: gigPerLayer),
                phoneID: NMPModelSharder.layerCapacity(
                    ramMB: 128 * 1024, bytesPerLayer: gigPerLayer),
            ])
        let split = layers(plan)
        XCTAssertGreaterThan(split[phoneID] ?? 0, 0, "capacity forces the phone in")
        XCTAssertEqual((split[coordID] ?? 0) + (split[phoneID] ?? 0), 100)
        XCTAssertEqual(plan.capacityShortfall, 0)
    }

    func testAllLayersPlacedAndContiguousFromZero() {
        let plan = NMPModelSharder.planByLatency(
            layerCount: 24,
            coordinatorPeerID: coordID,
            peers: [caps(coordID, .high), caps(phoneID, .high)],
            computeSecondsPerLayer: [coordID: 0.010, phoneID: 0.002],
            roundTripSeconds: [phoneID: 0.001])
        let sorted = plan.entries.sorted { $0.startLayer < $1.startLayer }
        XCTAssertEqual(sorted.first?.startLayer, 0)
        XCTAssertEqual(sorted.last?.endLayer, 24)
        for (a, b) in zip(sorted, sorted.dropFirst()) {
            XCTAssertEqual(a.endLayer, b.startLayer, "ranges must be contiguous")
        }
        // When the coordinator holds layers it leads the pipeline (owns the
        // embedding → no inbound hop). Here the phone is so much cheaper it
        // takes everything, so the coordinator legitimately holds none.
        if let coord = plan.entries.first(where: { $0.peerID == coordID }) {
            XCTAssertEqual(coord.shardIndex, 0)
        }
    }

    func testCapacityPlanEqualizesUtilization() {
        // Coordinator has 2× the phone's RAM → it should take ~2× the layers,
        // so both end at roughly the same % full (the "no device fills up"
        // plan). Capacities: 60 vs 30 layers → 24 total splits ~16/8.
        let plan = NMPModelSharder.planByCapacity(
            layerCount: 24,
            coordinatorPeerID: coordID,
            peers: [caps(coordID, .high), caps(phoneID, .high)],
            layerCapacities: [coordID: 60, phoneID: 30])
        let split = layers(plan)
        XCTAssertEqual((split[coordID] ?? 0) + (split[phoneID] ?? 0), 24)
        XCTAssertEqual(split[coordID], 16)
        XCTAssertEqual(split[phoneID], 8)
    }

    func testCapacityPlanEvenSplitWhenUnbounded() {
        let plan = NMPModelSharder.planByCapacity(
            layerCount: 24,
            coordinatorPeerID: coordID,
            peers: [caps(coordID, .high), caps(phoneID, .high)])
        let split = layers(plan)
        XCTAssertEqual(split[coordID], 12)
        XCTAssertEqual(split[phoneID], 12)
    }

    func testColdMeshIsConservativeAboutHops() {
        // No measurements: same class, so equal proxy rates. With no compute
        // advantage, the default hop makes Mac-only the latency winner — the
        // fix for a cold mesh over-loading a fast phone before it's measured.
        let plan = NMPModelSharder.planByLatency(
            layerCount: 24,
            coordinatorPeerID: coordID,
            peers: [caps(coordID, .high), caps(phoneID, .high)])
        XCTAssertEqual(layers(plan)[coordID], 24)
        XCTAssertNil(layers(plan)[phoneID])
    }

    // MARK: BUG-4 — churn artifacts must never read as a free network

    func testZeroedRoundTripIsNotTrustedAsFree() {
        // The post-churn inversion: the phone's measured compute is a bit
        // faster AND its round-trip entry is 0 (a zeroed artifact — a radio
        // hop is never free). Trusting the 0 once moved ALL 24 layers onto
        // the phone ("best for speed" paying a Wi-Fi hop per token). With
        // the honest prior, the 24 ms compute saving loses to the 30 ms
        // default hop → Mac-only.
        let plan = NMPModelSharder.planByLatency(
            layerCount: 24,
            coordinatorPeerID: coordID,
            peers: [caps(coordID, .high), caps(phoneID, .high)],
            computeSecondsPerLayer: [coordID: 0.0025, phoneID: 0.0015],
            roundTripSeconds: [phoneID: 0.0])
        XCTAssertEqual(layers(plan)[coordID], 24)
        XCTAssertNil(layers(plan)[phoneID])
        XCTAssertEqual(plan.exclusions.first?.peerID, phoneID)
    }

    func testUnmeasuredRoundTripUsesConservativePrior() {
        // Fresh per-connection peerID after a rejoin: NO round-trip entry
        // at all. "No measurement yet" must cost the conservative default
        // hop, not zero → coordinator-only for a small model, even though
        // the phone's measured compute is faster.
        let plan = NMPModelSharder.planByLatency(
            layerCount: 24,
            coordinatorPeerID: coordID,
            peers: [caps(coordID, .high), caps(phoneID, .high)],
            computeSecondsPerLayer: [coordID: 0.0025, phoneID: 0.0015])
        XCTAssertEqual(layers(plan)[coordID], 24)
        XCTAssertNil(layers(plan)[phoneID])
    }

    func testHonestMeasuredRoundTripAboveFloorIsStillUsed() {
        // The plausibility floor must not discard real measurements: a
        // genuinely cheap measured 2 ms hop with a 5× faster phone —
        // offloading clearly wins.
        let plan = NMPModelSharder.planByLatency(
            layerCount: 24,
            coordinatorPeerID: coordID,
            peers: [caps(coordID, .high), caps(phoneID, .high)],
            computeSecondsPerLayer: [coordID: 0.010, phoneID: 0.002],
            roundTripSeconds: [phoneID: 0.002])
        XCTAssertGreaterThan(layers(plan)[phoneID] ?? 0, 12)
        XCTAssertTrue(plan.exclusions.isEmpty)
    }

    func testCapacityStillForcesDistributionDespiteZeroedRoundTrip() {
        // The BUG-4 fix must not break the capacity-forced path: a model
        // too big for the coordinator spills to the phone regardless of
        // how its round trip was (mis)measured.
        let gigPerLayer = 1024 * 1_048_576
        let plan = NMPModelSharder.planByLatency(
            layerCount: 100,
            coordinatorPeerID: coordID,
            peers: [caps(coordID, .high, ramMB: 64 * 1024),
                    caps(phoneID, .medium, ramMB: 128 * 1024)],
            computeSecondsPerLayer: [coordID: 0.002, phoneID: 0.020],
            roundTripSeconds: [phoneID: 0.0],   // zeroed churn artifact
            layerCapacities: [
                coordID: NMPModelSharder.layerCapacity(
                    ramMB: 64 * 1024, bytesPerLayer: gigPerLayer),
                phoneID: NMPModelSharder.layerCapacity(
                    ramMB: 128 * 1024, bytesPerLayer: gigPerLayer),
            ])
        let split = layers(plan)
        XCTAssertGreaterThan(split[phoneID] ?? 0, 0, "capacity forces the phone in")
        XCTAssertEqual((split[coordID] ?? 0) + (split[phoneID] ?? 0), 100)
        XCTAssertEqual(plan.capacityShortfall, 0)
    }
}
