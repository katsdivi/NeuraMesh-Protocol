//
//  PeerChurnTests.swift
//  NMPTests — BUG-3 ghost retirement + BUG-7 async auto-balance
//
//  iOS peers mint a NEW peerID per connection, so a backgrounded phone
//  that comes back looks like a brand-new peer while its stale identity
//  keeps a membership slot — and its place in the routed plan. These pin
//  the coordinator-node behaviors that fix that: same-device rejoins
//  retire the ghost (unique live devices only), stage measurements never
//  record a phantom "free" round trip, and a real auto-balance mode change
//  answers immediately while its SHARD_ASSIGN round (which can legitimately
//  vault-stream for ~30 s) continues in the background.
//

import XCTest
@testable import NMP

final class PeerChurnTests: XCTestCase {

    private func caps(_ id: UInt32, name: String = "iPhone18,2",
                      ramMB: UInt32 = 8192,
                      cls: NMPComputeClass = .high) -> NMPCapabilities {
        NMPCapabilities(peerID: id, deviceName: name, ramMB: ramMB,
                        computeClass: cls)
    }

    private func makeNode(tag: String) -> NMPCoordinatorNode {
        NMPCoordinatorNode(
            engine: NMPReferenceComputeEngine(layerCount: 8, hiddenSize: 32),
            modelTag: tag, localPeerID: 0xC0)
    }

    // MARK: Same-physical-device identity (BUG-3b)

    func testSamePhysicalDeviceMatcher() {
        XCTAssertTrue(NMPCoordinatorNode.isSamePhysicalDevice(caps(1), caps(2)),
                      "same hardware marker + RAM + class = same device")
        XCTAssertFalse(NMPCoordinatorNode.isSamePhysicalDevice(
            caps(1), caps(2, name: "iPhone17,1")))
        XCTAssertFalse(NMPCoordinatorNode.isSamePhysicalDevice(
            caps(1), caps(2, ramMB: 6144)))
        XCTAssertFalse(NMPCoordinatorNode.isSamePhysicalDevice(
            caps(1), caps(2, cls: .medium)))
        // An empty hardware marker identifies nothing.
        XCTAssertFalse(NMPCoordinatorNode.isSamePhysicalDevice(
            caps(1, name: ""), caps(2, name: "")))
    }

    func testStalePeerIDsFindsGhostsNotSelf() {
        let ready: [UInt32: NMPCapabilities] = [
            0xA: caps(0xA),                      // ghost (same device)
            0xB: caps(0xB, name: "Mac14,9"),     // different device
            0xC: caps(0xC),                      // second ghost
        ]
        XCTAssertEqual(NMPCoordinatorNode.stalePeerIDs(
            replacing: caps(0xD), currentID: 0xD, among: ready), [0xA, 0xC])
        // A peer's own (re-used) ID is never its own ghost.
        XCTAssertEqual(NMPCoordinatorNode.stalePeerIDs(
            replacing: caps(0xA), currentID: 0xA, among: [0xA: caps(0xA)]), [])
    }

    // MARK: Ghost retirement on rejoin (BUG-3b/c)

    func testSameDeviceRejoinRetiresGhostPeer() {
        let node = makeNode(tag: "ghost-test")

        var lost: [UInt32] = []     // mutated only on the node queue
        let ghostRetired = expectation(description: "ghost retired")
        node.onPeerLost = { id in
            lost.append(id)
            if id == 0x111 { ghostRetired.fulfill() }
        }
        let readyTwice = expectation(description: "both adoptions ready")
        readyTwice.expectedFulfillmentCount = 2
        node.onPeerReady = { _ in readyTwice.fulfill() }

        // The phone joins, backgrounds (its entry lingers), then rejoins
        // with a NEW per-connection peerID — the documented iOS behavior.
        node.adoptPeer(caps(0x111), remoteID: 0x111, connection: nil)
        node.adoptPeer(caps(0x222), remoteID: 0x222, connection: nil)

        wait(for: [ghostRetired, readyTwice], timeout: 3)
        let snapshot = expectation(description: "membership read")
        var members: [UInt32: NMPCapabilities] = [:]
        node.withMembership { members = $0; snapshot.fulfill() }
        wait(for: [snapshot], timeout: 2)
        XCTAssertEqual(Array(members.keys), [0x222],
                       "membership counts unique live devices — no ghost")
        XCTAssertEqual(lost, [0x111],
                       "the stale identity must be reported lost so alive "
                       + "counts and plans drop it")
    }

    func testDistinctDevicesAreNotRetiredAsGhosts() {
        let node = makeNode(tag: "distinct-test")

        let noLoss = expectation(description: "no peer retired")
        noLoss.isInverted = true
        node.onPeerLost = { _ in noLoss.fulfill() }
        let bothReady = expectation(description: "both ready")
        bothReady.expectedFulfillmentCount = 2
        node.onPeerReady = { _ in bothReady.fulfill() }

        node.adoptPeer(caps(0x1), remoteID: 0x1, connection: nil)
        node.adoptPeer(caps(0x2, name: "Mac14,9", ramMB: 32768),
                       remoteID: 0x2, connection: nil)

        wait(for: [bothReady], timeout: 3)
        wait(for: [noLoss], timeout: 0.5)
        let snapshot = expectation(description: "membership read")
        var members: [UInt32: NMPCapabilities] = [:]
        node.withMembership { members = $0; snapshot.fulfill() }
        wait(for: [snapshot], timeout: 2)
        XCTAssertEqual(Set(members.keys), [0x1, 0x2])
        node.onPeerLost = nil
    }

    // MARK: Round-trip measurement hygiene (BUG-4, write side)

    func testLocalStagesRecordNoRoundTrip() throws {
        // A local stage has no network hop. Recording one (as 0) is the
        // zeroed-latency artifact the speed planner once consumed as a
        // genuinely free network. Remote trips, when recorded, are > 0.
        let testbed = try NMPMeshTestbed(layerCount: 8, hiddenSize: 96,
                                         remotePeerCount: 1)
        _ = try testbed.startSync()
        _ = try testbed.inferSync(input: testbed.makeInput())

        let trips = testbed.orchestrator.measuredRoundTripSeconds
        XCTAssertNil(trips[testbed.coordinatorID],
                     "local stages must not create round-trip entries")
        let peerID = testbed.remotePeers[0].capabilities.peerID
        if let trip = trips[peerID] {
            XCTAssertGreaterThan(trip, 0, "a stored round trip is never 0")
        }
        for (_, trip) in trips {
            XCTAssertGreaterThan(trip, 0)
        }
    }

    // MARK: BUG-7 — auto-balance mode change answers immediately

    func testAutoBalanceSetterReturnsBeforeReshardCompletes() throws {
        // A real (in-memory, full-crypto) peer link whose far side NEVER
        // acks SHARD_ASSIGN — the stand-in for a phone vault-streaming its
        // new layers for up to 30 s. The mode change must not block on it.
        let (rawCoordinator, rawPeer) = NMPInMemoryTransport.pair(label: "bug7")
        let coordinatorStatic = NoiseStaticKeyPair()
        let peerStatic = NoiseStaticKeyPair()
        let coordinatorSide = try PeerConnection(
            role: .initiator,
            config: PeerConnectionConfig(localPeerID: 0xC0),
            transport: rawCoordinator,
            localStatic: coordinatorStatic,
            remoteStaticPublicKey: peerStatic.publicKeyData,
            queue: DispatchQueue(label: "bug7.coordinator"))
        let peerSide = try PeerConnection(
            role: .responder,
            config: PeerConnectionConfig(localPeerID: 0x2),
            transport: rawPeer,
            localStatic: peerStatic,
            queue: DispatchQueue(label: "bug7.peer"))
        let established = expectation(description: "handshake")
        established.expectedFulfillmentCount = 2
        coordinatorSide.onEstablished = { _, _ in established.fulfill() }
        peerSide.onEstablished = { _, _ in established.fulfill() }
        peerSide.start()
        coordinatorSide.start()
        wait(for: [established], timeout: 5)
        // No NMPPeerShardEngine on peerSide → SHARD_ASSIGN is never acked.

        let node = makeNode(tag: "bug7-test")
        let adopted = expectation(description: "peer adopted")
        node.onPeerReady = { _ in adopted.fulfill() }
        node.adoptPeer(caps(0x2), remoteID: 0x2, connection: coordinatorSide)
        wait(for: [adopted], timeout: 3)

        let backgroundDone = expectation(description: "background re-shard")
        backgroundDone.isInverted = true
        node.onBackgroundReshard = { _ in backgroundDone.fulfill() }

        // Manual mode hands the silent peer real layers, so its ack round
        // will sit in the 30 s vault-stream window. The setter must still
        // answer immediately with the staged plan (the observed bug: the
        // HTTP response blocked 22–30+ s and clients timed out).
        let answered = expectation(description: "setter answered")
        let began = Date()
        var staged: [NMPShardPlanEntry] = []
        node.setAutoBalance(false) { result in
            if case .success(let entries) = result { staged = entries }
            answered.fulfill()
        }
        wait(for: [answered], timeout: 2)
        XCTAssertLessThan(Date().timeIntervalSince(began), 2.0,
                          "mode change must answer in ms, not the ack window")
        XCTAssertTrue(staged.contains { $0.peerID == 0x2 },
                      "the staged plan really does hand the peer layers "
                      + "(a no-op would not exercise the bug)")
        // …and the re-shard is genuinely still in flight (nothing acked):
        // commit-on-ack keeps the old plan serving meanwhile.
        wait(for: [backgroundDone], timeout: 1.0)
        node.onBackgroundReshard = nil   // the 30 s timeout fires post-test
    }

    func testAutoBalanceBackgroundReshardCommitsWhenAcksArrive() {
        // Coordinator-only mesh: the staged plan needs no remote acks, so
        // the background round must complete and report success — the
        // async path still commits, it just no longer blocks the caller.
        let node = makeNode(tag: "bug7-commit-test")

        var answeredFirst = false   // both closures run on the node queue
        let answered = expectation(description: "setter answered")
        let committed = expectation(description: "background commit")
        node.onBackgroundReshard = { result in
            XCTAssertTrue(answeredFirst,
                          "the setter's completion must fire before the "
                          + "background outcome")
            if case .success = result { committed.fulfill() }
        }
        node.setAutoBalance(true) { result in
            answeredFirst = true
            if case .failure(let error) = result {
                XCTFail("setter failed: \(error)")
            }
            answered.fulfill()
        }
        wait(for: [answered, committed], timeout: 3)
    }
}
