//
//  InferenceIntegrationTests.swift
//  NMPTests — Phase 5
//
//  End-to-end pipelined inference over real NMP connections (MockTransport,
//  full Noise handshake + encryption + FEC + reliability): coordinator +
//  1 and 2 remote shard peers, with the distributed output verified
//  BIT-EXACT against a single-engine baseline. Also: shard assignment
//  rejection, inference under packet loss, timeouts, and metrics flow.
//

import XCTest
@testable import NMP

final class InferenceIntegrationTests: XCTestCase {

    private let layerCount = 8
    private let hiddenSize = 96
    private let modelTag = "ref-test-model"

    /// One coordinator↔peer link: established connections + the peer's
    /// serving engine, all over an in-memory transport pair.
    private struct Link {
        let coordinatorSide: PeerConnection
        let peerSide: PeerConnection
        let shardEngine: NMPPeerShardEngine
        let coordinatorTransport: MockTransport
        let peerTransport: MockTransport
    }

    private func makeEstablishedLink(
        coordinatorID: UInt32, peerID: UInt32, label: String,
        peerEngine: NMPShardComputeEngine
    ) throws -> Link {
        let (tCoord, tPeer) = MockTransport.pair(label: label)
        let qCoord = DispatchQueue(label: "\(label).coord")
        let qPeer = DispatchQueue(label: "\(label).peer")
        let sCoord = NoiseStaticKeyPair()
        let sPeer = NoiseStaticKeyPair()

        let coordinatorSide = try PeerConnection(
            role: .initiator, config: PeerConnectionConfig(localPeerID: coordinatorID),
            transport: tCoord, localStatic: sCoord,
            remoteStaticPublicKey: sPeer.publicKeyData, queue: qCoord)
        let peerSide = try PeerConnection(
            role: .responder, config: PeerConnectionConfig(localPeerID: peerID),
            transport: tPeer, localStatic: sPeer, queue: qPeer)

        let shardEngine = NMPPeerShardEngine(
            connection: peerSide, engine: peerEngine,
            modelTag: modelTag, localPeerID: peerID)
        shardEngine.activate()

        let ready = expectation(description: "\(label) established")
        ready.expectedFulfillmentCount = 2
        coordinatorSide.onEstablished = { _, _ in ready.fulfill() }
        peerSide.onEstablished = { _, _ in ready.fulfill() }
        peerSide.start()
        coordinatorSide.start()
        wait(for: [ready], timeout: 5)

        return Link(coordinatorSide: coordinatorSide, peerSide: peerSide,
                    shardEngine: shardEngine,
                    coordinatorTransport: tCoord, peerTransport: tPeer)
    }

    private func makeEngine() -> NMPReferenceComputeEngine {
        NMPReferenceComputeEngine(layerCount: layerCount, hiddenSize: hiddenSize)
    }

    private func makeInput() -> [Float] {
        var rng = SplitMix64(seed: 0xBEEF)
        return (0..<hiddenSize).map { _ in rng.nextUnitFloat() * 2 - 1 }
    }

    private func caps(_ peerID: UInt32) -> NMPCapabilities {
        NMPCapabilities(peerID: peerID, deviceName: "sim", ramMB: 8192, computeClass: .high)
    }

    private func assignAndWait(_ orchestrator: NMPInferenceOrchestrator,
                               _ plan: [NMPShardPlanEntry],
                               timeout: TimeInterval = 5) -> NMPOrchestrationError? {
        let done = expectation(description: "assigned")
        var failure: NMPOrchestrationError?
        orchestrator.assignShards(plan, timeout: timeout) { result in
            if case .failure(let error) = result { failure = error }
            done.fulfill()
        }
        wait(for: [done], timeout: timeout + 2)
        return failure
    }

    // MARK: 2-peer mesh (Mac + iPhone shape)

    func testTwoPeerPipelineMatchesSingleDeviceBitExact() throws {
        let coordinatorID: UInt32 = 0x0000_0001
        let peerID: UInt32 = 0x0000_0002
        let link = try makeEstablishedLink(
            coordinatorID: coordinatorID, peerID: peerID,
            label: "mesh2", peerEngine: makeEngine())

        let orchestrator = NMPInferenceOrchestrator(
            localPeerID: coordinatorID, engine: makeEngine(),
            modelTag: modelTag, queue: DispatchQueue(label: "orch2"))
        orchestrator.attachPeer(peerID: peerID, connection: link.coordinatorSide)

        let plan = NMPModelSharder.plan(
            layerCount: layerCount, peers: [caps(coordinatorID), caps(peerID)])
        XCTAssertEqual(plan.map(\.peerID), [coordinatorID, peerID])
        XCTAssertEqual(plan.map(\.layerSpan), [4, 4])
        XCTAssertNil(assignAndWait(orchestrator, plan))

        let input = makeInput()
        let done = expectation(description: "inference")
        var report: NMPInferenceReport?
        orchestrator.infer(input: input) { result in
            report = try? result.get()
            done.fulfill()
        }
        wait(for: [done], timeout: 10)

        let baseline = try makeEngine().runLayers(start: 0, end: layerCount, input: input)
        guard let report else { return XCTFail("inference failed") }
        // Success criterion says ±0.01 logits; the deterministic reference
        // engine lets us hold the mesh to BIT-EXACT.
        XCTAssertEqual(report.output.map(\.bitPattern), baseline.map(\.bitPattern),
                       "2-peer mesh output must equal single-device output")

        XCTAssertEqual(report.perShard.count, 2)
        XCTAssertTrue(report.perShard[0].isLocal)
        XCTAssertFalse(report.perShard[1].isLocal)
        XCTAssertGreaterThan(report.networkPayloadBytes, 0)
        XCTAssertGreaterThan(report.totalSeconds, 0)
        print("[NMP] 2-peer inference: total "
              + String(format: "%.2f", report.totalSeconds * 1000) + " ms, network payload "
              + "\(report.networkPayloadBytes) B, network time "
              + String(format: "%.2f", report.networkSeconds * 1000) + " ms")
    }

    // MARK: 3-peer mesh

    func testThreePeerPipelineMatchesSingleDeviceBitExact() throws {
        let coordinatorID: UInt32 = 0x0000_0001
        let peerB: UInt32 = 0x0000_0002
        let peerC: UInt32 = 0x0000_0003
        let linkB = try makeEstablishedLink(
            coordinatorID: coordinatorID, peerID: peerB,
            label: "mesh3.b", peerEngine: makeEngine())
        let linkC = try makeEstablishedLink(
            coordinatorID: coordinatorID, peerID: peerC,
            label: "mesh3.c", peerEngine: makeEngine())

        let orchestrator = NMPInferenceOrchestrator(
            localPeerID: coordinatorID, engine: makeEngine(),
            modelTag: modelTag, queue: DispatchQueue(label: "orch3"))
        orchestrator.attachPeer(peerID: peerB, connection: linkB.coordinatorSide)
        orchestrator.attachPeer(peerID: peerC, connection: linkC.coordinatorSide)

        let plan = NMPModelSharder.plan(
            layerCount: layerCount,
            peers: [caps(coordinatorID), caps(peerB), caps(peerC)])
        XCTAssertNil(assignAndWait(orchestrator, plan))

        let input = makeInput()
        let done = expectation(description: "inference")
        var report: NMPInferenceReport?
        orchestrator.infer(input: input) { result in
            report = try? result.get()
            done.fulfill()
        }
        wait(for: [done], timeout: 10)

        let baseline = try makeEngine().runLayers(start: 0, end: layerCount, input: input)
        XCTAssertEqual(report?.output.map(\.bitPattern), baseline.map(\.bitPattern),
                       "3-peer mesh output must equal single-device output")
        XCTAssertEqual(report?.perShard.count, 3)
    }

    // MARK: Loss resilience

    func testInferenceSurvivesPacketLoss() throws {
        let coordinatorID: UInt32 = 0x0000_0001
        let peerID: UInt32 = 0x0000_0002
        let link = try makeEstablishedLink(
            coordinatorID: coordinatorID, peerID: peerID,
            label: "lossy", peerEngine: makeEngine())

        // Drop every 9th datagram in both directions AFTER establishment —
        // FEC absorbs singles, NACK repairs the rest.
        var sent = 0
        let dropEvery = 9
        let dropper: (Data) -> Bool = { _ in
            sent += 1
            return sent % dropEvery == 0
        }
        link.coordinatorTransport.dropOutbound = dropper
        link.peerTransport.dropOutbound = dropper

        let orchestrator = NMPInferenceOrchestrator(
            localPeerID: coordinatorID, engine: makeEngine(),
            modelTag: modelTag, queue: DispatchQueue(label: "orch.lossy"))
        orchestrator.attachPeer(peerID: peerID, connection: link.coordinatorSide)

        let plan = NMPModelSharder.plan(
            layerCount: layerCount, peers: [caps(coordinatorID), caps(peerID)])
        XCTAssertNil(assignAndWait(orchestrator, plan))

        let input = makeInput()
        let done = expectation(description: "lossy inference")
        var report: NMPInferenceReport?
        orchestrator.infer(input: input, stageTimeout: 15) { result in
            report = try? result.get()
            done.fulfill()
        }
        wait(for: [done], timeout: 20)

        let baseline = try makeEngine().runLayers(start: 0, end: layerCount, input: input)
        XCTAssertEqual(report?.output.map(\.bitPattern), baseline.map(\.bitPattern),
                       "loss must be repaired, never silently corrupted")
    }

    // MARK: Failure paths

    func testAssignmentRejectedOnModelMismatch() throws {
        let coordinatorID: UInt32 = 0x01
        let peerID: UInt32 = 0x02
        let link = try makeEstablishedLink(
            coordinatorID: coordinatorID, peerID: peerID,
            label: "mismatch", peerEngine: makeEngine())
        _ = link

        let orchestrator = NMPInferenceOrchestrator(
            localPeerID: coordinatorID, engine: makeEngine(),
            modelTag: "some-OTHER-model", // peer loaded modelTag
            queue: DispatchQueue(label: "orch.mismatch"))
        orchestrator.attachPeer(peerID: peerID, connection: link.coordinatorSide)

        let plan = NMPModelSharder.plan(
            layerCount: layerCount, peers: [caps(coordinatorID), caps(peerID)])
        let failure = assignAndWait(orchestrator, plan)
        XCTAssertEqual(failure, .assignmentRejected(
            peerID: peerID, status: .rejectedModelMismatch))
    }

    func testInferenceTimesOutWhenPeerSilent() throws {
        let coordinatorID: UInt32 = 0x01
        let peerID: UInt32 = 0x02
        // Peer connection established but NO shard engine activated: the
        // peer never answers inference traffic.
        let (tCoord, tPeer) = MockTransport.pair(label: "silent")
        let sCoord = NoiseStaticKeyPair(); let sPeer = NoiseStaticKeyPair()
        let coordinatorSide = try PeerConnection(
            role: .initiator, config: PeerConnectionConfig(localPeerID: coordinatorID),
            transport: tCoord, localStatic: sCoord,
            remoteStaticPublicKey: sPeer.publicKeyData,
            queue: DispatchQueue(label: "silent.coord"))
        let peerSide = try PeerConnection(
            role: .responder, config: PeerConnectionConfig(localPeerID: peerID),
            transport: tPeer, localStatic: sPeer,
            queue: DispatchQueue(label: "silent.peer"))
        let ready = expectation(description: "established")
        ready.expectedFulfillmentCount = 2
        coordinatorSide.onEstablished = { _, _ in ready.fulfill() }
        peerSide.onEstablished = { _, _ in ready.fulfill() }
        peerSide.start(); coordinatorSide.start()
        wait(for: [ready], timeout: 5)

        let orchestrator = NMPInferenceOrchestrator(
            localPeerID: coordinatorID, engine: makeEngine(),
            modelTag: modelTag, queue: DispatchQueue(label: "orch.silent"))
        orchestrator.attachPeer(peerID: peerID, connection: coordinatorSide)

        let failure = assignAndWait(orchestrator,
                                    NMPModelSharder.plan(layerCount: layerCount,
                                                         peers: [caps(coordinatorID), caps(peerID)]),
                                    timeout: 0.5)
        XCTAssertEqual(failure, .assignmentTimeout(unacked: [peerID]))
    }

    // MARK: Metrics

    func testPeerMetricsFlowBackAfterServing() throws {
        let coordinatorID: UInt32 = 0x01
        let peerID: UInt32 = 0x02
        let link = try makeEstablishedLink(
            coordinatorID: coordinatorID, peerID: peerID,
            label: "metrics", peerEngine: makeEngine())

        let orchestrator = NMPInferenceOrchestrator(
            localPeerID: coordinatorID, engine: makeEngine(),
            modelTag: modelTag, queue: DispatchQueue(label: "orch.metrics"))
        orchestrator.attachPeer(peerID: peerID, connection: link.coordinatorSide)

        let gotMetrics = expectation(description: "metrics received")
        var metrics: NMPPeerMetrics?
        orchestrator.onPeerMetrics = { received in
            if metrics == nil {
                metrics = received
                gotMetrics.fulfill()
            }
        }

        let plan = NMPModelSharder.plan(
            layerCount: layerCount, peers: [caps(coordinatorID), caps(peerID)])
        XCTAssertNil(assignAndWait(orchestrator, plan))
        let done = expectation(description: "inference")
        orchestrator.infer(input: makeInput()) { _ in done.fulfill() }
        wait(for: [done, gotMetrics], timeout: 10)

        XCTAssertEqual(metrics?.peerID, peerID)
        // Peer measured seconds/layer feeds the next re-plan.
        let measured = orchestrator.measuredSecondsPerLayer
        XCTAssertNotNil(measured[peerID])
        XCTAssertNotNil(measured[coordinatorID])
    }
}
