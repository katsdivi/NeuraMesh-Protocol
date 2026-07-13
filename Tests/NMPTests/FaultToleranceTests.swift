//
//  FaultToleranceTests.swift
//  NMPTests — Phase 6
//
//  Peer-loss detection (health monitor with an injected clock — the
//  5-second timeout is exercised without waiting 5 seconds), failover
//  re-sharding over a real in-process mesh, bit-exact inference after
//  failover, the all-peers-dead error path, and the <500 ms re-shard
//  latency budget.
//

import XCTest
@testable import NMP

final class FaultToleranceTests: XCTestCase {

    // MARK: Health monitor (injected clock)

    func testHealthMonitorDetectsSilentPeerAfterTimeout() {
        var now = Date(timeIntervalSince1970: 1000)
        let monitor = NMPPeerHealthMonitor(heartbeatTimeout: 5.0, clock: { now })

        monitor.track(peerID: 0xA)
        monitor.track(peerID: 0xB)
        XCTAssertEqual(monitor.checkHealth(), [], "fresh peers are healthy")

        // 4.9 s of silence: inside the timeout, still healthy.
        now = now.addingTimeInterval(4.9)
        XCTAssertEqual(monitor.checkHealth(), [])

        // 5.1 s: both peers dead.
        now = now.addingTimeInterval(0.2)
        XCTAssertEqual(monitor.checkHealth(), [0xA, 0xB])
    }

    func testHealthMonitorActivityRefreshesDeadline() {
        var now = Date(timeIntervalSince1970: 1000)
        let monitor = NMPPeerHealthMonitor(heartbeatTimeout: 5.0, clock: { now })

        monitor.track(peerID: 0xA)
        monitor.track(peerID: 0xB)

        now = now.addingTimeInterval(4.0)
        monitor.recordActivity(peerID: 0xA) // only A speaks

        now = now.addingTimeInterval(3.0)   // A silent 3 s, B silent 7 s
        XCTAssertEqual(monitor.checkHealth(), [0xB])

        monitor.forget(peerID: 0xB)
        XCTAssertEqual(monitor.checkHealth(), [], "forgotten peers are not reported")
        XCTAssertEqual(monitor.trackedPeers, [0xA])
    }

    func testHealthMonitorIgnoresActivityFromUntrackedPeers() {
        var now = Date(timeIntervalSince1970: 0)
        let monitor = NMPPeerHealthMonitor(heartbeatTimeout: 5.0, clock: { now })
        monitor.recordActivity(peerID: 0xDEAD) // never tracked
        now = now.addingTimeInterval(100)
        XCTAssertEqual(monitor.checkHealth(), [])
        XCTAssertEqual(monitor.trackedPeers, [])
    }

    // MARK: Failover over a real mesh

    func testFailoverReShardsToRemainingPeers() throws {
        let testbed = try NMPMeshTestbed(layerCount: 12, hiddenSize: 96,
                                         remotePeerCount: 2)
        let initialPlan = try testbed.startSync()
        XCTAssertEqual(initialPlan.count, 3, "coordinator + 2 remotes")

        let victim = testbed.remotePeers[1].capabilities.peerID
        let newPlan = try testbed.dropPeerSync(victim)

        XCTAssertEqual(newPlan.count, 2, "survivors: coordinator + 1 remote")
        XCTAssertFalse(newPlan.contains { $0.peerID == victim },
                       "dead peer must not appear in the new plan")
        // The new plan still covers every layer, contiguously.
        XCTAssertEqual(newPlan.first?.startLayer, 0)
        XCTAssertEqual(newPlan.last?.endLayer, 12)
        for (previous, next) in zip(newPlan, newPlan.dropFirst()) {
            XCTAssertEqual(previous.endLayer, next.startLayer, "no gaps or overlaps")
        }
        XCTAssertEqual(testbed.failover.activePlan, newPlan)
    }

    func testStandbyPeersReceiveZeroAssignmentInsteadOfHanging() throws {
        // Pure Speed with the weightless reference engine (unbounded
        // capacity) packs everything onto the single fastest device — the
        // coordinator. The remotes hold 0 layers, but must be TOLD so
        // (a standby assignment), not left waiting on "waiting for
        // coordinator". This is the exact hang the phone hit.
        let testbed = try NMPMeshTestbed(layerCount: 12, hiddenSize: 96,
                                         remotePeerCount: 2)
        _ = try testbed.startSync()

        testbed.failover.shardingObjective = .speed
        let plan = try replanSync(testbed.failover)
        XCTAssertEqual(plan.count, 1, "Pure Speed packs the fastest device")
        XCTAssertEqual(plan.first?.peerID, testbed.coordinatorID)

        // Both remotes must accept a zero-layer standby assignment. It is
        // sent fire-and-forget (never blocks the assignment round), so it
        // lands just after replan completes — poll briefly for it.
        for remote in testbed.remotePeers {
            let deadline = Date().addingTimeInterval(2)
            while Date() < deadline,
                  remote.shardEngine.assignment?.startLayer
                    != remote.shardEngine.assignment?.endLayer {
                Thread.sleep(forTimeInterval: 0.02)
            }
            let assignment = remote.shardEngine.assignment
            XCTAssertNotNil(assignment, "standby peer must be assigned, not left waiting")
            XCTAssertEqual(assignment?.startLayer, assignment?.endLayer,
                           "standby = a zero-layer assignment")
        }
        // The failover reports WHY each is idle, for the UI.
        XCTAssertEqual(testbed.failover.activeExclusions.count, 2)
        XCTAssertTrue(testbed.failover.activeExclusions.allSatisfy {
            $0.reason.contains("0 shards")
        })

        // Flipping back to the default re-engages every device.
        testbed.failover.shardingObjective = .capacityThenSpeed
        let spread = try replanSync(testbed.failover)
        XCTAssertEqual(spread.count, 3, "default spreads across the whole mesh")
        XCTAssertTrue(testbed.failover.activeExclusions.isEmpty)
    }

    private func replanSync(_ failover: NMPFailoverOrchestrator) throws -> [NMPShardPlanEntry] {
        let done = expectation(description: "replan")
        var outcome: Result<[NMPShardPlanEntry], NMPFailoverError>?
        failover.replan { outcome = $0; done.fulfill() }
        wait(for: [done], timeout: 5)
        switch try XCTUnwrap(outcome) {
        case .success(let plan): return plan
        case .failure(let error): throw error
        }
    }

    func testInferenceRemainsBitExactAfterFailover() throws {
        let testbed = try NMPMeshTestbed(layerCount: 12, hiddenSize: 96,
                                         remotePeerCount: 2)
        _ = try testbed.startSync()
        let input = testbed.makeInput()
        let baseline = try testbed.baselineOutput(for: input)

        let before = try testbed.inferSync(input: input)
        XCTAssertEqual(before.output.map(\.bitPattern), baseline.map(\.bitPattern),
                       "pre-drop output must match single-device baseline")

        _ = try testbed.dropPeerSync(testbed.remotePeers[0].capabilities.peerID)

        let after = try testbed.inferSync(input: input)
        XCTAssertEqual(after.output.map(\.bitPattern), baseline.map(\.bitPattern),
                       "post-failover output must be bit-identical — failover "
                       + "must never corrupt the numerics")
        XCTAssertEqual(after.perShard.count, 2)
    }

    func testAllPeersDeadFailsExplicitlyWithoutHanging() throws {
        // A coordinator that contributes no compute of its own: the mesh's
        // only compute peer dying leaves nothing to re-shard onto.
        let testbed = try NMPMeshTestbed(layerCount: 8, hiddenSize: 96,
                                         remotePeerCount: 1)
        let orchestrator = NMPInferenceOrchestrator(
            localPeerID: 0xC0, engine: NMPReferenceComputeEngine(layerCount: 8, hiddenSize: 96),
            modelTag: "testbed-ref-model", queue: DispatchQueue(label: "orch.allDead"))
        let onlyPeer = testbed.remotePeers[0]
        let failover = NMPFailoverOrchestrator(
            orchestrator: orchestrator,
            layerCount: 8,
            localPeerID: 0xC0,
            peers: [onlyPeer.capabilities], // no local compute in the mesh
            queue: DispatchQueue(label: "failover.allDead"))

        let deadCalled = expectation(description: "onAllPeersDead")
        failover.onAllPeersDead = { deadCalled.fulfill() }

        let done = expectation(description: "drop completes")
        var failure: NMPFailoverError?
        failover.handlePeerDrop(onlyPeer.capabilities.peerID) { result in
            if case .failure(let error) = result { failure = error }
            done.fulfill()
        }
        wait(for: [done, deadCalled], timeout: 3)
        XCTAssertEqual(failure, .allPeersDead)
    }

    func testDropOfUnknownPeerIsRejected() throws {
        let testbed = try NMPMeshTestbed(layerCount: 8, hiddenSize: 96,
                                         remotePeerCount: 1)
        _ = try testbed.startSync()

        let done = expectation(description: "drop completes")
        var failure: NMPFailoverError?
        testbed.failover.handlePeerDrop(0xFFFF_FFFF) { result in
            if case .failure(let error) = result { failure = error }
            done.fulfill()
        }
        wait(for: [done], timeout: 3)
        XCTAssertEqual(failure, .unknownPeer(0xFFFF_FFFF))
    }

    func testReshardCompletesUnder500ms() throws {
        let testbed = try NMPMeshTestbed(layerCount: 24, hiddenSize: 512,
                                         remotePeerCount: 3)
        _ = try testbed.startSync()

        _ = try testbed.dropPeerSync(testbed.remotePeers[2].capabilities.peerID)

        let seconds = try XCTUnwrap(testbed.failover.lastReshardSeconds)
        XCTAssertLessThan(seconds, 0.5,
                          "re-shard (plan + SHARD_ASSIGN ack round) must beat 500 ms; "
                          + "took \(seconds * 1000) ms")
    }

    func testPeerJoinTriggersReshardAndStaysBitExact() throws {
        let testbed = try NMPMeshTestbed(layerCount: 12, hiddenSize: 96,
                                         remotePeerCount: 1)
        let initialPlan = try testbed.startSync()
        XCTAssertEqual(initialPlan.count, 2)

        let joined = try testbed.joinNewPeer()
        let plan = testbed.failover.activePlan
        XCTAssertEqual(plan.count, 3, "new peer folded into the pipeline")
        XCTAssertTrue(plan.contains { $0.peerID == joined.capabilities.peerID })

        let input = testbed.makeInput()
        let report = try testbed.inferSync(input: input)
        XCTAssertEqual(report.output.map(\.bitPattern),
                       try testbed.baselineOutput(for: input).map(\.bitPattern),
                       "3-way mesh after join must match single-device output")
        XCTAssertEqual(report.perShard.count, 3)
    }

    // MARK: Liveness wiring

    func testOrchestratorActivityFeedsHealthMonitor() throws {
        let testbed = try NMPMeshTestbed(layerCount: 8, hiddenSize: 96,
                                         remotePeerCount: 1)
        _ = try testbed.startSync()
        let peerID = testbed.remotePeers[0].capabilities.peerID

        let before = try XCTUnwrap(
            testbed.failover.healthMonitor.lastActivityDate(peerID: peerID))
        // Serving an inference sends packets back → heartbeat refreshes.
        Thread.sleep(forTimeInterval: 0.01)
        _ = try testbed.inferSync(input: testbed.makeInput())

        let after = try XCTUnwrap(
            testbed.failover.healthMonitor.lastActivityDate(peerID: peerID))
        XCTAssertGreaterThan(after, before,
                             "inference traffic must refresh the peer's heartbeat")
    }

    func testAutomaticHealthCheckDropsSilentPeer() throws {
        // Scaled-down timings (0.5 s heartbeat, 0.1 s poll) so the test runs
        // in ~1 s; the production ratio (5 s heartbeat, 1 s poll) is the
        // same machinery with bigger constants.
        //
        // Liveness is ACTIVITY-based (the pipeline is the heartbeat), so
        // the mesh must be under load for detection to work: a background
        // loop keeps inferring. The healthy peer's stage completes each
        // pass (refreshing its heartbeat); the silenced peer's stage times
        // out — only IT goes quiet past the deadline.
        let testbed = try NMPMeshTestbed(layerCount: 8, hiddenSize: 96,
                                         remotePeerCount: 2,
                                         heartbeatTimeout: 0.5)
        _ = try testbed.startSync()

        let victim = testbed.remotePeers[1].capabilities.peerID
        let resharded = expectation(description: "auto failover re-sharded")
        testbed.failover.onResharded = { plan, _ in
            if !plan.contains(where: { $0.peerID == victim }) {
                resharded.fulfill()
            }
        }

        let trafficRunning = NSLock()
        var keepDriving = true
        let driver = Thread { [testbed] in
            while true {
                trafficRunning.lock()
                let go = keepDriving
                trafficRunning.unlock()
                guard go else { break }
                // Short stage timeout: passes through the dead peer fail
                // fast instead of stalling the heartbeat traffic.
                _ = try? testbed.inferSync(input: testbed.makeInput(),
                                           stageTimeout: 0.2)
            }
        }
        driver.start()
        defer {
            trafficRunning.lock()
            keepDriving = false
            trafficRunning.unlock()
        }

        let silencedAt = Date()
        testbed.silencePeer(victim)
        testbed.failover.startHealthChecks(interval: 0.1)
        defer { testbed.failover.stopHealthChecks() }

        wait(for: [resharded], timeout: 5)
        let detectionSeconds = Date().timeIntervalSince(silencedAt)
        // Budget: heartbeat timeout + poll interval + re-shard + CI slack.
        // Production equivalent: 5 s heartbeat + 1 s poll < 5.5 s spec
        // budget counted from the peer's actual last packet.
        XCTAssertLessThan(detectionSeconds, 3.0,
                          "detection+failover took \(detectionSeconds) s "
                          + "with a 0.5 s heartbeat timeout")
        XCTAssertFalse(testbed.failover.activePlan.contains { $0.peerID == victim })
    }
}
