//
//  DiscoveryIntegrationTests.swift
//  NMPTests — Phase 4
//
//  End-to-end mesh self-assembly with NO real network: a MockDiscoveryHub
//  plays the role of mDNS (services registered by publishers are replayed
//  to browsers, updates and removals fan out), so 3-peer assembly,
//  coordinator failover, capability propagation, and staleness expiry are
//  fully deterministic.
//

import XCTest
import Network
@testable import NMP

// MARK: - Mock mDNS

/// The "network": tracks registered services and fans events out to every
/// started source — including the publisher's own, exactly like a real
/// Bonjour browser seeing the local service (managers must self-filter).
private final class MockDiscoveryHub {
    private var services: [UInt32: NMPCapabilities] = [:]
    private var sources: [MockDiscoverySource] = []

    func makeSource() -> MockDiscoverySource {
        let source = MockDiscoverySource(hub: self)
        sources.append(source)
        return source
    }

    func register(_ capabilities: NMPCapabilities) {
        services[capabilities.peerID] = capabilities
        for source in sources where source.isStarted {
            source.onPeerFound?(capabilities, nil)
        }
    }

    func unregister(_ peerID: UInt32) {
        guard services.removeValue(forKey: peerID) != nil else { return }
        for source in sources where source.isStarted {
            source.onPeerLost?(peerID)
        }
    }

    /// Bonjour semantics: a browser that starts late still sees every
    /// service currently registered.
    func replayServices(to source: MockDiscoverySource) {
        for capabilities in services.values {
            source.onPeerFound?(capabilities, nil)
        }
    }
}

private final class MockDiscoverySource: NMPPeerDiscoverySource {
    var onPeerFound: ((NMPCapabilities, NWEndpoint?) -> Void)?
    var onPeerLost: ((UInt32) -> Void)?
    private(set) var isStarted = false
    private unowned let hub: MockDiscoveryHub

    init(hub: MockDiscoveryHub) { self.hub = hub }

    func start() throws {
        isStarted = true
        hub.replayServices(to: self)
    }

    func stop() { isStarted = false }
}

private final class MockPublisher: NMPCapabilityPublisher {
    private unowned let hub: MockDiscoveryHub
    private var capabilities: NMPCapabilities

    init(hub: MockDiscoveryHub, capabilities: NMPCapabilities) {
        self.hub = hub
        self.capabilities = capabilities
    }

    func start() throws { hub.register(capabilities) }

    func update(capabilities: NMPCapabilities) {
        self.capabilities = capabilities
        hub.register(capabilities)
    }

    func stop() { hub.unregister(capabilities.peerID) }
}

// MARK: - Tests

final class DiscoveryIntegrationTests: XCTestCase {

    private struct Node {
        let manager: NMPPeerDiscoveryManager
        let queue: DispatchQueue
        // Event logs, appended on the manager queue, read after flush().
        let log: EventLog
    }

    private final class EventLog {
        var discovered: [UInt32] = []
        var updated: [NMPCapabilities] = []
        var removed: [UInt32] = []
        var coordinatorChanges: [UInt32?] = []
    }

    private func makeNode(
        hub: MockDiscoveryHub,
        capabilities: NMPCapabilities,
        tune: ((inout NMPDiscoveryConfig) -> Void)? = nil
    ) -> Node {
        var config = NMPDiscoveryConfig()
        config.capabilityRefreshInterval = 0 // ticks driven manually in tests
        tune?(&config)
        let queue = DispatchQueue(label: "nmp.test.discovery.\(capabilities.peerID)")
        let manager = NMPPeerDiscoveryManager(
            localCapabilities: capabilities,
            publisher: MockPublisher(hub: hub, capabilities: capabilities),
            source: hub.makeSource(),
            config: config,
            queue: queue)
        let log = EventLog()
        manager.onPeerDiscovered = { log.discovered.append($0.peerID) }
        manager.onPeerUpdated = { log.updated.append($0) }
        manager.onPeerRemoved = { log.removed.append($0) }
        manager.onCoordinatorChanged = { log.coordinatorChanges.append($0) }
        return Node(manager: manager, queue: queue, log: log)
    }

    private func caps(
        _ peerID: UInt32, _ computeClass: NMPComputeClass, load: Double = 0
    ) -> NMPCapabilities {
        NMPCapabilities(
            peerID: peerID, deviceName: "node-\(String(peerID, radix: 16))",
            ramMB: 8192, computeClass: computeClass, currentLoadPercent: load)
    }

    /// Drains every node's queue until cross-queue event hops settle.
    /// Events bounce at most hub → source → queue.async once per action,
    /// but a drain on A can enqueue onto B, so run a few rounds.
    private func flush(_ nodes: [Node]) {
        for _ in 0..<3 {
            for node in nodes { node.queue.sync {} }
        }
    }

    private func start(_ nodes: [Node]) throws {
        for node in nodes {
            try node.queue.sync { try node.manager.start() }
        }
        flush(nodes)
    }

    // MARK: Mesh assembly

    func testMeshSelfAssembles() throws {
        let hub = MockDiscoveryHub()
        // A: high compute, LOWEST peerID → expected coordinator everywhere.
        let a = makeNode(hub: hub, capabilities: caps(0x0000_0001, .high))
        let b = makeNode(hub: hub, capabilities: caps(0xaabb_ccdd, .high))
        let c = makeNode(hub: hub, capabilities: caps(0x9988_7766, .medium))
        let nodes = [a, b, c]
        try start(nodes)

        let expectedOthers: [(Node, Set<UInt32>)] = [
            (a, [0xaabb_ccdd, 0x9988_7766]),
            (b, [0x0000_0001, 0x9988_7766]),
            (c, [0x0000_0001, 0xaabb_ccdd]),
        ]
        for (node, others) in expectedOthers {
            XCTAssertEqual(Set(node.manager.discoveredPeers.keys), others,
                           "every node discovers every OTHER node")
            XCTAssertEqual(Set(node.log.discovered), others)
            XCTAssertEqual(node.manager.currentCoordinator, 0x0000_0001,
                           "all nodes must agree on the coordinator")
        }
        XCTAssertTrue(a.manager.isCoordinator)
        XCTAssertFalse(b.manager.isCoordinator)
        XCTAssertFalse(c.manager.isCoordinator)
    }

    func testOwnServiceIsIgnoredAndSinglePeerCoordinatesItself() throws {
        let hub = MockDiscoveryHub()
        let solo = makeNode(hub: hub, capabilities: caps(0x42, .low))
        try start([solo])

        // The hub echoed the node's own registration back at it (as real
        // Bonjour does); the manager must not list itself as a remote peer.
        XCTAssertTrue(solo.manager.discoveredPeers.isEmpty)
        XCTAssertTrue(solo.log.discovered.isEmpty)
        // Alone on the mesh, even a low-tier device coordinates.
        XCTAssertEqual(solo.manager.currentCoordinator, 0x42)
        XCTAssertTrue(solo.manager.isCoordinator)
        XCTAssertEqual(solo.log.coordinatorChanges, [0x42])
    }

    func testCoordinatorChangeOnPeerDrop() throws {
        let hub = MockDiscoveryHub()
        let a = makeNode(hub: hub, capabilities: caps(0x01, .high))
        let b = makeNode(hub: hub, capabilities: caps(0x02, .high))
        let c = makeNode(hub: hub, capabilities: caps(0x03, .medium))
        let nodes = [a, b, c]
        try start(nodes)
        XCTAssertEqual(b.manager.currentCoordinator, 0x01)

        // Coordinator leaves the mesh (its service deregisters).
        a.queue.sync { a.manager.stop() }
        flush(nodes)

        for node in [b, c] {
            XCTAssertEqual(node.manager.discoveredPeers[0x01]?.capabilities, nil)
            XCTAssertEqual(node.log.removed, [0x01])
            XCTAssertEqual(node.manager.currentCoordinator, 0x02,
                           "survivors re-elect the next peer in the total order")
        }
        XCTAssertTrue(b.manager.isCoordinator)
        // Change log: initial self-election, converge on 0x01, failover to 0x02.
        XCTAssertEqual(b.log.coordinatorChanges.last, 0x02)
    }

    func testCapabilityUpdatePropagates() throws {
        let hub = MockDiscoveryHub()
        let a = makeNode(hub: hub, capabilities: caps(0x01, .high))
        let b = makeNode(hub: hub, capabilities: caps(0x02, .high, load: 10))
        let nodes = [a, b]
        try start(nodes)
        XCTAssertEqual(a.manager.discoveredPeers[0x02]?.capabilities.currentLoadPercent, 10)

        // B's load spikes; it re-advertises (TXT update on real Bonjour).
        b.queue.sync {
            b.manager.updateLocalCapabilities(self.caps(0x02, .high, load: 80))
        }
        flush(nodes)

        XCTAssertEqual(a.manager.discoveredPeers[0x02]?.capabilities.currentLoadPercent, 80,
                       "peers must see the refreshed capability")
        XCTAssertEqual(a.log.updated.map(\.currentLoadPercent), [80])
        XCTAssertEqual(a.log.discovered, [0x02], "an update must not re-fire discovery")
        // Load changes never move the coordinatorship.
        XCTAssertEqual(a.manager.currentCoordinator, 0x01)
    }

    func testPeerRemovalOnTTLExpiry() throws {
        // A source that misses removal events (mDNS goodbye lost): the
        // staleness sweep evicts the peer once its TTL window lapses.
        let hub = MockDiscoveryHub()
        let node = makeNode(hub: hub, capabilities: caps(0x01, .medium)) {
            $0.peerStaleTimeout = 1.0
        }
        var now: TimeInterval = 100
        node.manager.clock = { now }
        try start([node])

        node.queue.sync {} // ensure start settled
        hub.register(caps(0x02, .high)) // ghost peer: registers, then goes silent
        flush([node])
        XCTAssertEqual(node.manager.currentCoordinator, 0x02)

        // Within the TTL window: still a member.
        now = 100.9
        node.queue.sync { node.manager.expireStalePeers(at: now) }
        XCTAssertNotNil(node.manager.discoveredPeers[0x02])

        // Window lapses: evicted, election falls back to the local device.
        now = 101.5
        node.queue.sync { node.manager.expireStalePeers(at: now) }
        XCTAssertNil(node.manager.discoveredPeers[0x02])
        XCTAssertEqual(node.log.removed, [0x02])
        XCTAssertEqual(node.manager.currentCoordinator, 0x01)
        XCTAssertTrue(node.manager.isCoordinator)
    }

    func testElectionDeterministicAcrossJoinOrders() throws {
        // The same 3 devices assembling in all 6 join orders must converge
        // on the same coordinator every time.
        let capabilities = [caps(0x05, .medium), caps(0x0a, .high), caps(0x0f, .high)]
        let permutations: [[Int]] = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        for order in permutations {
            let hub = MockDiscoveryHub()
            let nodes = order.map { makeNode(hub: hub, capabilities: capabilities[$0]) }
            try start(nodes)
            for node in nodes {
                XCTAssertEqual(node.manager.currentCoordinator, 0x0a,
                               "join order \(order) diverged")
            }
        }
    }

    // MARK: PeerConnection integration

    func testCoordinatorFlagReachesPeerConnection() throws {
        let (transport, _) = MockTransport.pair()
        let queue = DispatchQueue(label: "nmp.test.conn")
        let connection = try PeerConnection(
            role: .responder,
            config: PeerConnectionConfig(localPeerID: 0x01),
            transport: transport,
            localStatic: NoiseStaticKeyPair(),
            queue: queue)
        XCTAssertFalse(connection.isCoordinator)
        XCTAssertNil(connection.remoteCapabilities)

        let remote = caps(0x02, .high)
        connection.updateDiscoveryState(isCoordinator: true, remoteCapabilities: remote)
        queue.sync {}
        XCTAssertTrue(connection.isCoordinator)
        XCTAssertEqual(connection.remoteCapabilities, remote)

        // Election flips away; the last-known remote capabilities persist.
        connection.updateDiscoveryState(isCoordinator: false)
        queue.sync {}
        XCTAssertFalse(connection.isCoordinator)
        XCTAssertEqual(connection.remoteCapabilities, remote)
    }
}
