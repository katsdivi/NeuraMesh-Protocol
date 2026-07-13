//
//  ShardMessagesTests.swift
//  NMPTests — Phase 5
//
//  Wire codecs for shard orchestration: round trips, malformed-input
//  rejection, tensor chunking/reassembly under reordering and duplication,
//  and the deterministic sharder's apportionment properties.
//

import XCTest
@testable import NMP

// MARK: - Codecs

final class ShardMessagesTests: XCTestCase {

    func testTensorCodecRoundTripIsExact() throws {
        let values: [Float] = [0, 1, -1, .pi, 1e-38, 3.4e38, -0.0,
                               .infinity, -.infinity, .leastNonzeroMagnitude]
        let decoded = try NMPTensorCodec.decode(NMPTensorCodec.encode(values))
        XCTAssertEqual(decoded.count, values.count)
        for (a, b) in zip(decoded, values) {
            XCTAssertEqual(a.bitPattern, b.bitPattern, "bit-exact round trip required")
        }
        // NaN preserves its bit pattern too.
        let nan = try NMPTensorCodec.decode(NMPTensorCodec.encode([Float.nan]))
        XCTAssertEqual(nan[0].bitPattern, Float.nan.bitPattern)
    }

    func testShardAssignRoundTripAndRejection() throws {
        let assign = NMPShardAssign(
            shardIndex: 1, pipelineLength: 3, startLayer: 16, endLayer: 24,
            totalLayers: 32, hiddenSize: 4096, modelTag: "llama-7B-q4_K_M")
        let decoded = try NMPShardAssign.decode(assign.encode())
        XCTAssertEqual(decoded, assign)

        XCTAssertThrowsError(try NMPShardAssign.decode(assign.encode().prefix(5)))
        var badVersion = assign.encode()
        badVersion[0] = 9
        XCTAssertThrowsError(try NMPShardAssign.decode(badVersion)) {
            XCTAssertEqual($0 as? NMPShardCodecError, .unsupportedVersion(9))
        }
    }

    func testMetaAndMetricsAndAckRoundTrips() throws {
        let request = NMPInferRequestMeta(
            requestID: 7, startLayer: 0, endLayer: 16, totalBytes: 16_384, chunkCount: 16)
        XCTAssertEqual(try NMPInferRequestMeta.decode(request.encode()), request)

        let response = NMPInferResponseMeta(
            requestID: 7, status: .ok, computeMicros: 12_500,
            totalBytes: 16_384, chunkCount: 16)
        XCTAssertEqual(try NMPInferResponseMeta.decode(response.encode()), response)

        let metrics = NMPPeerMetrics(
            peerID: 0xaabb_ccdd, inferenceLatencyMicros: 9_000,
            memoryUsageMB: 412, currentLoadPercent: 37)
        XCTAssertEqual(try NMPPeerMetrics.decode(metrics.encode()), metrics)

        let ack = NMPShardAck(shardIndex: 2, status: .ready)
        XCTAssertEqual(try NMPShardAck.decode(ack.encode()), ack)

        // Kind confusion is rejected, not misparsed.
        XCTAssertThrowsError(try NMPInferRequestMeta.decode(response.encode())) {
            guard case NMPShardCodecError.wrongKind = $0 else {
                return XCTFail("expected .wrongKind, got \($0)")
            }
        }
    }

    // MARK: Chunking

    func testChunkSplitCoversExactlyAndRespectsMTU() throws {
        let tensor = Data((0..<10_000).map { UInt8($0 % 251) })
        let chunks = try NMPTensorChunk.split(requestID: 1, tensorBytes: tensor)

        XCTAssertEqual(chunks.count, 10) // ceil(10000/1024)
        XCTAssertTrue(chunks.allSatisfy { $0.payload.count <= NMPTensorChunk.defaultChunkBytes })
        XCTAssertEqual(chunks.reduce(Data()) { $0 + $1.payload }, tensor)
        // Every chunk's full datagram (payload + envelope 7 + header 20 +
        // tag 16) stays under a 1500-byte MTU.
        for chunk in chunks {
            XCTAssertLessThan(chunk.encode().count + 36, 1500)
        }
    }

    func testReassemblyOutOfOrderAndDuplicates() throws {
        let tensor = Data((0..<5_000).map { UInt8($0 % 249) })
        let chunks = try NMPTensorChunk.split(requestID: 9, tensorBytes: tensor)
        let reassembler = NMPTensorReassembler()

        // Meta arrives first; chunks arrive shuffled with a duplicate.
        XCTAssertNil(try reassembler.setExpectation(
            requestID: 9, chunkCount: chunks.count, totalBytes: tensor.count))
        var shuffled = chunks.shuffled()
        shuffled.insert(shuffled[0], at: 1) // duplicate
        var completed: Data?
        for chunk in shuffled {
            if let done = try reassembler.addChunk(chunk) {
                XCTAssertNil(completed, "must complete exactly once")
                completed = done
            }
        }
        XCTAssertEqual(completed, tensor)
    }

    func testReassemblyMetaAfterChunks() throws {
        // Reordering can deliver every chunk before the meta: completion
        // must fire from setExpectation.
        let tensor = Data(repeating: 0x42, count: 2_500)
        let chunks = try NMPTensorChunk.split(requestID: 3, tensorBytes: tensor)
        let reassembler = NMPTensorReassembler()
        for chunk in chunks {
            XCTAssertNil(try reassembler.addChunk(chunk), "incomplete without meta")
        }
        let completed = try reassembler.setExpectation(
            requestID: 3, chunkCount: chunks.count, totalBytes: tensor.count)
        XCTAssertEqual(completed, tensor)
    }

    func testReassemblyEnforcesByteBudget() throws {
        let reassembler = NMPTensorReassembler()
        reassembler.maxTensorBytes = 1_000
        XCTAssertThrowsError(try reassembler.setExpectation(
            requestID: 1, chunkCount: 2, totalBytes: 2_000)) {
            guard case NMPShardCodecError.tensorTooLarge = $0 else {
                return XCTFail("expected .tensorTooLarge, got \($0)")
            }
        }
    }
}

// MARK: - Sharder

final class ModelSharderTests: XCTestCase {

    private func caps(_ peerID: UInt32, _ computeClass: NMPComputeClass) -> NMPCapabilities {
        NMPCapabilities(peerID: peerID, deviceName: "p", ramMB: 8192, computeClass: computeClass)
    }

    /// Structural invariants every plan must satisfy.
    private func assertValid(_ plan: [NMPShardPlanEntry], layerCount: Int) {
        XCTAssertEqual(plan.first?.startLayer, 0)
        XCTAssertEqual(plan.last?.endLayer, layerCount)
        for (i, entry) in plan.enumerated() {
            XCTAssertEqual(entry.shardIndex, i)
            XCTAssertGreaterThan(entry.layerSpan, 0, "no empty shards")
            if i > 0 { XCTAssertEqual(entry.startLayer, plan[i - 1].endLayer, "contiguous") }
        }
    }

    func testPlanIsDeterministicAcrossInputOrder() {
        let peers = [caps(0x01, .high), caps(0x02, .medium), caps(0x03, .low),
                     caps(0x04, .high)]
        let reference = NMPModelSharder.plan(layerCount: 32, peers: peers)
        assertValid(reference, layerCount: 32)
        for _ in 0..<50 {
            XCTAssertEqual(NMPModelSharder.plan(layerCount: 32, peers: peers.shuffled()),
                           reference)
        }
    }

    func testFasterPeersGetMoreLayers() {
        let plan = NMPModelSharder.plan(
            layerCount: 32,
            peers: [caps(0x01, .high), caps(0x02, .medium), caps(0x03, .low)])
        assertValid(plan, layerCount: 32)
        // Weights 4:2:1 over 32 layers → 18/9/5 (largest remainder).
        XCTAssertEqual(plan.map(\.layerSpan), [18, 9, 5])
        XCTAssertEqual(plan.map(\.peerID), [0x01, 0x02, 0x03])
    }

    func testMeasuredLatencyOverridesClassWeight() {
        // Same class, but 0x02 measured 3x faster → gets ~3x the layers.
        let plan = NMPModelSharder.plan(
            layerCount: 32,
            peers: [caps(0x01, .high), caps(0x02, .high)],
            measuredSecondsPerLayer: [0x01: 0.030, 0x02: 0.010])
        assertValid(plan, layerCount: 32)
        XCTAssertEqual(plan.map(\.layerSpan), [8, 24])
    }

    func testSinglePeerTakesEverything() {
        let plan = NMPModelSharder.plan(layerCount: 32, peers: [caps(0x42, .low)])
        XCTAssertEqual(plan, [NMPShardPlanEntry(peerID: 0x42, shardIndex: 0,
                                                startLayer: 0, endLayer: 32)])
    }

    func testMorePeersThanLayersDropsSlowest() {
        let peers = [caps(0x01, .high), caps(0x02, .high), caps(0x03, .medium),
                     caps(0x04, .low)]
        let plan = NMPModelSharder.plan(layerCount: 3, peers: peers)
        assertValid(plan, layerCount: 3)
        XCTAssertEqual(plan.count, 3)
        XCTAssertFalse(plan.contains { $0.peerID == 0x04 }, "slowest sits out")
    }

    func testEveryPeerGetsAtLeastOneLayer() {
        // 4:2:1:1 over 8 layers — the low-class peers must not starve.
        let plan = NMPModelSharder.plan(
            layerCount: 8,
            peers: [caps(0x01, .high), caps(0x02, .medium), caps(0x03, .low), caps(0x04, .low)])
        assertValid(plan, layerCount: 8)
        XCTAssertTrue(plan.allSatisfy { $0.layerSpan >= 1 })
    }

    func testEmptyInputsProduceEmptyPlan() {
        XCTAssertTrue(NMPModelSharder.plan(layerCount: 0, peers: [caps(1, .high)]).isEmpty)
        XCTAssertTrue(NMPModelSharder.plan(layerCount: 32, peers: []).isEmpty)
    }

    // MARK: Capacity-aware planning (Mesh 2.8)

    private func caps(_ peerID: UInt32, _ cc: NMPComputeClass, ramMB: UInt32) -> NMPCapabilities {
        NMPCapabilities(peerID: peerID, deviceName: "p\(peerID)", ramMB: ramMB, computeClass: cc)
    }

    func testUnboundedCapacityMatchesLegacySplit() {
        // No ceilings → identical to the Phase 5 speed-weighted split.
        let peers = [caps(0x01, .high), caps(0x02, .medium), caps(0x03, .low)]
        let detailed = NMPModelSharder.planDetailed(layerCount: 32, peers: peers)
        XCTAssertEqual(detailed.entries.map(\.layerSpan), [18, 9, 5])
        XCTAssertTrue(detailed.exclusions.isEmpty)
        XCTAssertEqual(detailed.capacityShortfall, 0)
    }

    func testCapacityForcesDistributionWhenModelTooBig() {
        // 32 layers, fast Mac can hold only 20, slower phone holds 12 →
        // the phone MUST take 12 even though it's slower. This is the goal.
        let mac = caps(0x01, .high), phone = caps(0x02, .medium)
        let plan = NMPModelSharder.planDetailed(
            layerCount: 32, peers: [mac, phone],
            measuredSecondsPerLayer: [0x01: 0.010, 0x02: 0.040],
            layerCapacities: [0x01: 20, 0x02: 12])
        XCTAssertEqual(plan.capacityShortfall, 0)
        let byPeer = Dictionary(uniqueKeysWithValues:
            plan.entries.map { ($0.peerID, $0.layerSpan) })
        XCTAssertEqual(byPeer[0x01], 20, "Mac capped at its ceiling")
        XCTAssertEqual(byPeer[0x02], 12, "phone forced to hold the overflow")
        XCTAssertTrue(plan.exclusions.isEmpty)
    }

    func testSpeedModePacksFastestAndExcludesUnneededPeer() {
        // Model fits on the Mac alone → Pure Speed puts it all on the Mac,
        // the phone gets 0 with a reason (no capacity need for a hop).
        let mac = caps(0x01, .high), phone = caps(0x02, .medium)
        let plan = NMPModelSharder.planDetailed(
            layerCount: 32, peers: [mac, phone],
            measuredSecondsPerLayer: [0x01: 0.010, 0x02: 0.040],
            layerCapacities: [0x01: 64, 0x02: 64],
            objective: .speed)
        XCTAssertEqual(plan.entries.count, 1)
        XCTAssertEqual(plan.entries.first?.peerID, 0x01)
        XCTAssertEqual(plan.entries.first?.layerSpan, 32)
        XCTAssertEqual(plan.exclusions.map(\.peerID), [0x02])
        XCTAssertTrue(plan.exclusions.first!.reason.contains("0 shards"))
    }

    func testCapacityThenSpeedStillSpreadsWhenModelFits() {
        // Same fits-on-Mac model, DEFAULT objective → the mesh is used, the
        // phone still gets a speed-weighted share (fixes "Mac does it all").
        let mac = caps(0x01, .high), phone = caps(0x02, .medium)
        let plan = NMPModelSharder.planDetailed(
            layerCount: 32, peers: [mac, phone],
            measuredSecondsPerLayer: [0x01: 0.010, 0x02: 0.040],
            layerCapacities: [0x01: 64, 0x02: 64])
        XCTAssertEqual(plan.entries.count, 2, "both devices work in the default mode")
        XCTAssertTrue(plan.exclusions.isEmpty)
    }

    func testZeroCapacityDeviceExcludedAsSpare() {
        let mac = caps(0x01, .high), tiny = caps(0x02, .low)
        let plan = NMPModelSharder.planDetailed(
            layerCount: 8, peers: [mac, tiny],
            layerCapacities: [0x01: 32, 0x02: 0])
        XCTAssertEqual(plan.entries.map(\.peerID), [0x01])
        XCTAssertEqual(plan.exclusions.map(\.peerID), [0x02])
        XCTAssertTrue(plan.exclusions.first!.reason.lowercased().contains("out of memory"))
    }

    func testCapacityShortfallReportedWhenMeshTooSmall() {
        // 32 layers, two devices holding 10 + 8 = 18 → 14 fit nowhere.
        let plan = NMPModelSharder.planDetailed(
            layerCount: 32, peers: [caps(0x01, .high), caps(0x02, .medium)],
            layerCapacities: [0x01: 10, 0x02: 8])
        XCTAssertEqual(plan.capacityShortfall, 14)
        XCTAssertEqual(plan.entries.reduce(0) { $0 + $1.layerSpan }, 18,
                       "assigns everything that fits, best effort")
    }

    func testLayerCapacityFromRAM() {
        // 8 GB, 0.5 GB/layer, 60% headroom → floor(8*0.6/0.5) = 9 layers.
        let cap = NMPModelSharder.layerCapacity(
            ramMB: 8192, bytesPerLayer: 512 * 1_048_576, headroom: 0.6)
        XCTAssertEqual(cap, 9)
        // Unknown footprint → unbounded.
        XCTAssertEqual(NMPModelSharder.layerCapacity(ramMB: 8192, bytesPerLayer: 0), Int.max)
    }
}
