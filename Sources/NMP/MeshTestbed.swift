//
//  MeshTestbed.swift
//  NMP — Phase 6
//
//  A complete in-process mesh: one coordinator (orchestrator + failover +
//  local engine) and N remote shard peers, each connected over an
//  in-memory transport pair wrapped in loss injectors. Every link runs
//  the REAL stack — Noise IK handshake, AES-GCM, sequencing, FEC, NACK —
//  so what the dashboard shows and the benchmarks measure is the actual
//  protocol behavior, minus only the physical radio.
//
//  Shared by: BenchmarkSuite (drives load through it), DashboardServer's
//  CLI (observes it live), and the Phase 6 test suites.
//
//  Blocking style: construction and the `*Sync` helpers block the calling
//  thread on semaphores. Call from a plain thread (CLI main, XCTest) —
//  never from one of the mesh's own queues.
//

import Foundation

public enum NMPMeshTestbedError: Error {
    case handshakeTimeout(peerID: UInt32)
    case inferenceTimeout
    case failoverTimeout
    case orchestration(NMPOrchestrationError)
    case failover(NMPFailoverError)
}

public final class NMPMeshTestbed {

    // MARK: One remote shard peer's plumbing

    public final class RemotePeer {
        public let capabilities: NMPCapabilities
        /// Coordinator's end of the link (attached to the orchestrator).
        public let coordinatorSide: PeerConnection
        /// Peer's end (owned by its shard engine).
        public let peerSide: PeerConnection
        public let shardEngine: NMPPeerShardEngine
        /// Drops datagrams the coordinator sends toward this peer.
        public let coordinatorInjector: NMPPacketLossInjector
        /// Drops datagrams this peer sends toward the coordinator.
        public let peerInjector: NMPPacketLossInjector

        init(capabilities: NMPCapabilities, coordinatorSide: PeerConnection,
             peerSide: PeerConnection, shardEngine: NMPPeerShardEngine,
             coordinatorInjector: NMPPacketLossInjector,
             peerInjector: NMPPacketLossInjector) {
            self.capabilities = capabilities
            self.coordinatorSide = coordinatorSide
            self.peerSide = peerSide
            self.shardEngine = shardEngine
            self.coordinatorInjector = coordinatorInjector
            self.peerInjector = peerInjector
        }
    }

    // MARK: Configuration & state

    public let coordinatorID: UInt32 = 0x0000_0001
    public let layerCount: Int
    public let hiddenSize: Int
    public let modelTag: String

    public let orchestrator: NMPInferenceOrchestrator
    public private(set) var failover: NMPFailoverOrchestrator
    /// Insertion-ordered (peerID ascending — IDs are handed out serially).
    public private(set) var remotePeers: [RemotePeer] = []

    /// Aggregated loss-recovery events from BOTH ends of every link,
    /// labeled with the remote peer the link belongs to. Fires on the
    /// emitting connection's queue.
    public var onPacketEvent: ((UInt32, NMPPacketEvent) -> Void)?

    private let coordinatorEngine: NMPReferenceComputeEngine
    private let orchestratorQueue = DispatchQueue(label: "nmp.testbed.orchestrator")
    private let failoverQueue = DispatchQueue(label: "nmp.testbed.failover")
    private var nextPeerID: UInt32 = 0x0000_0002
    private var linkCounter = 0
    private let simulatedSecondsPerLayer: TimeInterval
    private let simulatedPeerSlowdowns: [Double]

    /// Builds the mesh and completes every handshake before returning.
    /// - Parameters:
    ///   - simulatedSecondsPerLayer: artificial per-layer compute delay on
    ///     every engine, so benchmark pipelines resemble real model stage
    ///     times instead of being pure network measurements.
    ///   - simulatedPeerSlowdowns: Phase 9 — per-remote-peer multipliers on
    ///     `simulatedSecondsPerLayer` (by join order; missing entries = 1),
    ///     making the mesh HETEROGENEOUS so adaptive sharding has real
    ///     speed differences to balance against.
    public init(
        layerCount: Int = 12,
        hiddenSize: Int = 1024,
        remotePeerCount: Int = 2,
        modelTag: String = "testbed-ref-model",
        simulatedSecondsPerLayer: TimeInterval = 0,
        simulatedPeerSlowdowns: [Double] = [],
        handshakeTimeout: TimeInterval = 5,
        heartbeatTimeout: TimeInterval = 5
    ) throws {
        self.layerCount = layerCount
        self.hiddenSize = hiddenSize
        self.modelTag = modelTag
        self.simulatedSecondsPerLayer = simulatedSecondsPerLayer
        self.simulatedPeerSlowdowns = simulatedPeerSlowdowns

        coordinatorEngine = NMPReferenceComputeEngine(
            layerCount: layerCount, hiddenSize: hiddenSize)
        coordinatorEngine.simulatedSecondsPerLayer = simulatedSecondsPerLayer

        orchestrator = NMPInferenceOrchestrator(
            localPeerID: coordinatorID, engine: coordinatorEngine,
            modelTag: modelTag, queue: orchestratorQueue)

        // Failover starts with just the coordinator; connected peers are
        // added below (they flow through the same accounting as later joins).
        failover = NMPFailoverOrchestrator(
            orchestrator: orchestrator,
            layerCount: layerCount,
            localPeerID: coordinatorID,
            peers: [Self.capabilities(peerID: coordinatorID)],
            healthMonitor: NMPPeerHealthMonitor(heartbeatTimeout: heartbeatTimeout),
            queue: failoverQueue)

        for _ in 0..<remotePeerCount {
            let peer = try makeConnectedPeer(timeout: handshakeTimeout)
            failover.registerPeer(peer.capabilities, connection: peer.coordinatorSide)
        }
        // registerPeer is async on the failover queue; callers reading
        // activePeers right after init (the adaptive controller does)
        // must see the full mesh, not however many registrations had
        // drained. This was a real intermittent bug: under load the
        // controller saw a partial mesh, decided the profile cache was
        // incomplete, and probed when it should not have.
        failover.waitForMembership()
    }

    private static func capabilities(peerID: UInt32) -> NMPCapabilities {
        NMPCapabilities(peerID: peerID, deviceName: "testbed-\(String(peerID, radix: 16))",
                        ramMB: 8192, computeClass: .high)
    }

    // MARK: Mesh assembly

    /// Builds a fully handshaked link + serving shard engine. Does NOT
    /// attach it to the orchestrator (init and `joinNewPeer` differ there).
    private func makeConnectedPeer(timeout: TimeInterval) throws -> RemotePeer {
        let peerID = nextPeerID
        nextPeerID += 1
        linkCounter += 1
        let label = "nmp.testbed.link\(linkCounter)"

        let (rawCoordinator, rawPeer) = NMPInMemoryTransport.pair(label: label)
        let coordinatorInjector = NMPPacketLossInjector(
            wrapping: rawCoordinator, seed: 0xC0DE_0000 | UInt64(peerID))
        let peerInjector = NMPPacketLossInjector(
            wrapping: rawPeer, seed: 0xFEED_0000 | UInt64(peerID))

        let coordinatorStatic = NoiseStaticKeyPair()
        let peerStatic = NoiseStaticKeyPair()

        let coordinatorSide = try PeerConnection(
            role: .initiator,
            config: PeerConnectionConfig(localPeerID: coordinatorID),
            transport: coordinatorInjector,
            localStatic: coordinatorStatic,
            remoteStaticPublicKey: peerStatic.publicKeyData,
            queue: DispatchQueue(label: "\(label).coordinator"))
        let peerSide = try PeerConnection(
            role: .responder,
            config: PeerConnectionConfig(localPeerID: peerID),
            transport: peerInjector,
            localStatic: peerStatic,
            queue: DispatchQueue(label: "\(label).peer"))

        let engine = NMPReferenceComputeEngine(
            layerCount: layerCount, hiddenSize: hiddenSize)
        let slowdown = remotePeers.count < simulatedPeerSlowdowns.count
            ? simulatedPeerSlowdowns[remotePeers.count] : 1
        engine.simulatedSecondsPerLayer = simulatedSecondsPerLayer * slowdown
        let shardEngine = NMPPeerShardEngine(
            connection: peerSide, engine: engine,
            modelTag: modelTag, localPeerID: peerID)
        shardEngine.activate()

        coordinatorSide.onPacketEvent = { [weak self] event in
            self?.onPacketEvent?(peerID, event)
        }
        peerSide.onPacketEvent = { [weak self] event in
            self?.onPacketEvent?(peerID, event)
        }

        let established = DispatchSemaphore(value: 0)
        coordinatorSide.onEstablished = { _, _ in established.signal() }
        peerSide.onEstablished = { _, _ in established.signal() }
        peerSide.start()
        coordinatorSide.start()
        for _ in 0..<2 {
            guard established.wait(timeout: .now() + timeout) == .success else {
                throw NMPMeshTestbedError.handshakeTimeout(peerID: peerID)
            }
        }

        let peer = RemotePeer(
            capabilities: Self.capabilities(peerID: peerID),
            coordinatorSide: coordinatorSide, peerSide: peerSide,
            shardEngine: shardEngine,
            coordinatorInjector: coordinatorInjector, peerInjector: peerInjector)
        remotePeers.append(peer)
        return peer
    }

    /// Connects a brand-new peer and re-shards the mesh across it
    /// (Scenario: dynamic peer join). Returns the joined peer.
    @discardableResult
    public func joinNewPeer(handshakeTimeout: TimeInterval = 5) throws -> RemotePeer {
        let peer = try makeConnectedPeer(timeout: handshakeTimeout)
        try joinSync(peer)
        return peer
    }

    private func joinSync(_ peer: RemotePeer, timeout: TimeInterval = 10) throws {
        let done = DispatchSemaphore(value: 0)
        var failure: NMPFailoverError?
        failover.handlePeerJoin(peer.capabilities,
                                connection: peer.coordinatorSide) { result in
            if case .failure(let error) = result { failure = error }
            done.signal()
        }
        guard done.wait(timeout: .now() + timeout + 2) == .success else {
            throw NMPMeshTestbedError.failoverTimeout
        }
        if let failure { throw NMPMeshTestbedError.failover(failure) }
    }

    // MARK: Fault injection

    /// Simulates abrupt peer death: both directions of its link go dark.
    /// The peer process is still "running" — it just can't be heard,
    /// exactly like a device walking out of Wi-Fi range.
    public func silencePeer(_ peerID: UInt32) {
        guard let peer = remotePeers.first(where: { $0.capabilities.peerID == peerID }) else {
            return
        }
        peer.coordinatorInjector.blackhole()
        peer.peerInjector.blackhole()
    }

    /// Runs the failover path for `peerID` and blocks until the mesh has
    /// re-sharded (or failed explicitly).
    public func dropPeerSync(_ peerID: UInt32,
                             timeout: TimeInterval = 10) throws -> [NMPShardPlanEntry] {
        silencePeer(peerID)
        let done = DispatchSemaphore(value: 0)
        var outcome: Result<[NMPShardPlanEntry], NMPFailoverError> = .failure(.allPeersDead)
        failover.handlePeerDrop(peerID, timeout: timeout) { result in
            outcome = result
            done.signal()
        }
        guard done.wait(timeout: .now() + timeout + 2) == .success else {
            throw NMPMeshTestbedError.failoverTimeout
        }
        switch outcome {
        case .success(let plan):
            remotePeers.removeAll { $0.capabilities.peerID == peerID }
            return plan
        case .failure(let error):
            throw NMPMeshTestbedError.failover(error)
        }
    }

    /// Applies a steady loss rate to every link, both directions.
    public func setLossRate(_ rate: Double) {
        for peer in remotePeers {
            peer.coordinatorInjector.setLossRate(rate)
            peer.peerInjector.setLossRate(rate)
        }
    }

    /// AWDL-like burst on every link.
    public func setBurstLoss(rate: Double, duration: TimeInterval) {
        for peer in remotePeers {
            peer.coordinatorInjector.setBurstLoss(rate: rate, duration: duration)
            peer.peerInjector.setBurstLoss(rate: rate, duration: duration)
        }
    }

    // MARK: Driving the mesh

    /// Computes and assigns the initial shard plan; blocks until acked.
    public func startSync(timeout: TimeInterval = 10) throws -> [NMPShardPlanEntry] {
        let done = DispatchSemaphore(value: 0)
        var outcome: Result<[NMPShardPlanEntry], NMPFailoverError> = .failure(.allPeersDead)
        failover.assignInitialPlan(timeout: timeout) { result in
            outcome = result
            done.signal()
        }
        guard done.wait(timeout: .now() + timeout + 2) == .success else {
            throw NMPMeshTestbedError.failoverTimeout
        }
        switch outcome {
        case .success(let plan): return plan
        case .failure(let error): throw NMPMeshTestbedError.failover(error)
        }
    }

    /// One pipeline pass; blocks until the output tensor is back.
    public func inferSync(input: [Float],
                          stageTimeout: TimeInterval = 30) throws -> NMPInferenceReport {
        let done = DispatchSemaphore(value: 0)
        var outcome: Result<NMPInferenceReport, NMPOrchestrationError>?
        orchestrator.infer(input: input, stageTimeout: stageTimeout) { result in
            outcome = result
            done.signal()
        }
        guard done.wait(timeout: .now() + stageTimeout * 2 + 5) == .success,
              let outcome else {
            throw NMPMeshTestbedError.inferenceTimeout
        }
        switch outcome {
        case .success(let report): return report
        case .failure(let error): throw NMPMeshTestbedError.orchestration(error)
        }
    }

    // MARK: Reference values

    /// Deterministic pseudo-random activation vector.
    public func makeInput(seed: UInt64 = 0xBEEF) -> [Float] {
        var rng = SplitMix64(seed: seed)
        return (0..<hiddenSize).map { _ in rng.nextUnitFloat() * 2 - 1 }
    }

    /// Single-device ground truth the mesh output must match bit-exactly.
    public func baselineOutput(for input: [Float]) throws -> [Float] {
        try NMPReferenceComputeEngine(layerCount: layerCount, hiddenSize: hiddenSize)
            .runLayers(start: 0, end: layerCount, input: input)
    }
}
