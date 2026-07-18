//
//  PeerNode.swift
//  NMP — Phase 5
//
//  Turn-key mesh roles. Everything the CLI tools and the iOS app need,
//  so those front-ends stay thin UI/argument shells:
//
//  - NMPPeerNode: a compute peer. Binds a UDP listener, advertises itself
//    over Bonjour (capabilities + UDP port + Noise static key in the TXT
//    record), accepts the coordinator's handshake, and serves its
//    assigned shard. This is what runs on the iPhone.
//
//  - NMPCoordinatorNode: the coordinator. Browses for peers, dials every
//    dialable one (TXT carries port + key — zero manual configuration),
//    and exposes the orchestrator for shard assignment and inference.
//    This is what runs on the Mac.
//
//  PORT TRICK (load-bearing): a peer binds its NMP UDP listener to an
//  ephemeral port P, then binds its Bonjour anchor (TCP) listener to the
//  SAME P. The SRV record therefore advertises P, and a coordinator can
//  dial the browse result's `.service` endpoint over UDP directly —
//  Network.framework resolves the SRV host and connects UDP to P, which
//  is exactly the NMP listener. No address parsing, no TXT-record IP
//  hacks, and it keeps working when the peer's DHCP address changes.
//
//  Security model for the auto-dial path: the responder's static key is
//  taken from its TXT record, i.e. trust-on-first-use against an active
//  LAN attacker; the responder accepts any authenticated initiator. Fine
//  for a benchmark mesh on your own Wi-Fi; production pins keys via
//  `PeerConnectionConfig.authorizedStaticKeys` (see Phase5_Design.md).
//

import Foundation
import Network

// MARK: - Peer node (runs on the iPhone / any compute peer)

public final class NMPPeerNode {

    // MARK: Callbacks (invoked on `queue`)

    /// Human-readable lifecycle updates for UIs and logs.
    public var onStatus: ((String) -> Void)?
    public var onAssigned: ((NMPShardAssign) -> Void)?
    /// requestID, layer range served, pure compute seconds.
    public var onServed: ((UInt32, Range<Int>, TimeInterval) -> Void)?
    public var onDiagnostic: ((String) -> Void)?
    /// The mesh runs a DIFFERENT model than this node loaded (its shard
    /// assignment was rejected). Carries the mesh's model tag — the app
    /// should re-pick its engine to match and restart the node.
    public var onModelMismatch: ((String) -> Void)?

    // MARK: State

    public let peerID: UInt32
    public private(set) var listeningPort: UInt16?
    public private(set) var assignment: NMPShardAssign?
    public private(set) var servedCount = 0

    private let engine: NMPShardComputeEngine
    private let modelTag: String
    private let queue: DispatchQueue
    private let staticKeys = NoiseStaticKeyPair()

    private var listener: UDPListener?
    private var publisher: NMPBonjourPublisher?
    private var connection: PeerConnection?
    private var shardEngine: NMPPeerShardEngine?

    public init(
        engine: NMPShardComputeEngine,
        modelTag: String,
        peerID: UInt32 = UInt32.random(in: 1...UInt32.max),
        queue: DispatchQueue = DispatchQueue(label: "nmp.peer.node")
    ) {
        self.engine = engine
        self.modelTag = modelTag
        self.peerID = peerID
        self.queue = queue
    }

    public func start() throws {
        let listener = try UDPListener(port: NWEndpoint.Port(rawValue: 0)!, queue: queue)
        self.listener = listener

        listener.onStateChange = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                guard self.publisher == nil, let port = listener.port?.rawValue else { return }
                self.listeningPort = port
                self.advertise(on: port)
            case .failed(let error):
                self.onStatus?("UDP listener failed: \(error)")
            default:
                break
            }
        }
        listener.onNewTransport = { [weak self] transport, endpoint in
            self?.accept(transport: transport, from: endpoint)
        }

        onStatus?("peer \(String(format: "%08x", peerID)) starting "
                  + "(\(engine.layerCount) layers × \(engine.hiddenSize), model '\(modelTag)')")
        listener.start()
    }

    public func stop() {
        queue.async { [self] in
            publisher?.stop(); publisher = nil
            connection?.close(); connection = nil
            listener?.cancel(); listener = nil
        }
    }

    private func advertise(on port: UInt16) {
        var capabilities = NMPSystemCapabilityProbe.measure(peerID: peerID)
        capabilities.udpPort = port
        capabilities.noiseStaticPublicKey = staticKeys.publicKeyData
        capabilities.modelFormats = ["gguf"]

        // Bonjour anchor on the SAME port number as the UDP listener — see
        // the header comment; this is what makes the peer dialable from a
        // bare browse result.
        let publisher = NMPBonjourPublisher(capabilities: capabilities, port: port)
        self.publisher = publisher
        do {
            try publisher.start()
            onStatus?("advertising \(NMPBonjour.serviceName(for: peerID)) on UDP port \(port)")
        } catch {
            onStatus?("Bonjour publish failed: \(error)")
        }
    }

    private func accept(transport: UDPTransport, from endpoint: NWEndpoint) {
        // One coordinator at a time; a new inbound flow replaces the old
        // session (covers coordinator restarts without peer restarts).
        if connection != nil {
            onDiagnostic?("replacing existing coordinator session (new flow from \(endpoint))")
            connection?.close()
        }
        do {
            let connection = try PeerConnection(
                role: .responder,
                config: PeerConnectionConfig(localPeerID: peerID),
                transport: transport,
                localStatic: staticKeys,
                queue: queue)
            self.connection = connection

            let shardEngine = NMPPeerShardEngine(
                connection: connection, engine: engine,
                modelTag: modelTag, localPeerID: peerID)
            self.shardEngine = shardEngine
            shardEngine.onAssigned = { [weak self] assign in
                self?.assignment = assign
                self?.onStatus?("assigned shard \(assign.shardIndex): layers "
                                + "\(assign.startLayer)..<\(assign.endLayer) of \(assign.totalLayers)")
                self?.onAssigned?(assign)
            }
            shardEngine.onInferenceServed = { [weak self] requestID, layers, seconds in
                self?.servedCount += 1
                self?.onServed?(requestID, layers, seconds)
            }
            shardEngine.onModelMismatch = { [weak self] meshTag in
                self?.onStatus?("mesh runs '\(meshTag)' but this peer loaded a "
                                + "different model — assignment rejected")
                self?.onModelMismatch?(meshTag)
            }
            shardEngine.onDiagnostic = { [weak self] message in
                self?.onDiagnostic?(message)
            }
            shardEngine.activate()

            connection.onEstablished = { [weak self] _, remoteID in
                self?.onStatus?("coordinator \(String(format: "%08x", remoteID)) connected")
            }
            connection.onFailed = { [weak self] error in
                self?.onStatus?("coordinator session failed: \(error)")
            }
            connection.start()
        } catch {
            onStatus?("failed to accept coordinator connection: \(error)")
        }
    }
}

// MARK: - Coordinator node (runs on the Mac)

public final class NMPCoordinatorNode {

    // MARK: Callbacks (invoked on `queue`)

    public var onStatus: ((String) -> Void)?
    /// A peer finished the NMP handshake and is attachable for sharding.
    public var onPeerReady: ((NMPCapabilities) -> Void)?
    public var onPeerLost: ((UInt32) -> Void)?
    public var onPeerMetrics: ((NMPPeerMetrics) -> Void)?
    /// Outcome of a re-shard that `setAutoBalance` kicked off in the
    /// background AFTER already invoking its completion (BUG-7: the mode
    /// change must answer in milliseconds even when a peer vault-streams
    /// its new layers for ~30 s before acking). On failure the previous
    /// plan keeps serving (the orchestrator commits on ack) — wire this to
    /// surface the correction to the UI.
    public var onBackgroundReshard: ((Result<[NMPShardPlanEntry],
                                             NMPOrchestrationError>) -> Void)?

    // MARK: State

    public let localPeerID: UInt32
    public let orchestrator: NMPInferenceOrchestrator
    public private(set) var readyPeers: [UInt32: NMPCapabilities] = [:]

    /// Auto mode (default): every re-plan splits layers by MEASURED speed +
    /// capacity, so the mesh converges to a balanced pipeline on its own.
    /// Manual mode: the operator's per-peer `manualShares` cap each device's
    /// slice instead. Toggled from the UI; all reads/writes stay on `queue`.
    public private(set) var autoBalance = true
    /// peerID → compute-share cap in (0, 1]; only consulted in manual mode.
    /// A 0 share excludes the peer (Mac-only, no per-token round trip).
    public private(set) var manualShares: [UInt32: Double] = [:]
    /// Model weight bytes per layer, for the auto planner's RAM ceilings.
    /// 0 (default) = unbounded capacity (e.g. the weightless reference engine
    /// or before the coordinator knows the model's footprint).
    public var modelBytesPerLayer: Int = 0

    /// How auto mode splits the model. Previewed side-by-side before applying
    /// (see `candidatePlans`): SPEED minimizes measured per-token latency,
    /// CAPACITY minimizes the peak per-device memory load (so no device fills
    /// up), BALANCED spreads by speed across the whole mesh under RAM ceilings.
    public enum PlanStrategy: String, Sendable, CaseIterable {
        case speed, balanced, capacity
    }
    public private(set) var planStrategy: PlanStrategy = .speed

    private let engine: NMPShardComputeEngine
    private let queue: DispatchQueue
    private let staticKeys = NoiseStaticKeyPair()
    private var discovery: NMPPeerDiscoveryManager?
    private var browser: NMPBonjourBrowser?
    private var connections: [UInt32: PeerConnection] = [:]
    private var connectionQueues: [UInt32: DispatchQueue] = [:]
    private var localCapabilities: NMPCapabilities

    public init(
        engine: NMPShardComputeEngine,
        modelTag: String,
        localPeerID: UInt32 = UInt32.random(in: 1...UInt32.max),
        queue: DispatchQueue = DispatchQueue(label: "nmp.coordinator.node")
    ) {
        self.engine = engine
        self.localPeerID = localPeerID
        self.queue = queue
        self.localCapabilities = NMPSystemCapabilityProbe.measure(peerID: localPeerID)
        self.orchestrator = NMPInferenceOrchestrator(
            localPeerID: localPeerID, engine: engine, modelTag: modelTag, queue: queue)
        orchestrator.onPeerMetrics = { [weak self] metrics in
            self?.onPeerMetrics?(metrics)
        }
        // BUG-3: a peer whose compute path stalls (consecutive stage
        // timeouts) is retired IMMEDIATELY, even while its heartbeat stays
        // chatty — a backgrounded iPhone echoes pings with a frozen engine,
        // and waiting minutes for a transport-level failure blackholes
        // every generation routed through it. The orchestrator's callback
        // fires on `queue` (they share it), so this runs queue-safe.
        orchestrator.onPeerComputeStalled = { [weak self] peerID, count in
            self?.retirePeer(peerID,
                             reason: "compute stalled — \(count) consecutive stage "
                                     + "timeouts (its heartbeat alone doesn't prove "
                                     + "it can serve layers)",
                             redialAfter: 10)
        }
    }

    public func start() throws {
        dispatchPrecondition(condition: .onQueue(queue))
        let browser = NMPBonjourBrowser()
        self.browser = browser
        let discovery = NMPPeerDiscoveryManager(
            localCapabilities: localCapabilities,
            publisher: nil, // browse-only: the coordinator dials, peers advertise
            source: browser,
            queue: queue)
        self.discovery = discovery

        discovery.onPeerDiscovered = { [weak self] capabilities in
            self?.dialIfPossible(capabilities)
        }
        discovery.onPeerRemoved = { [weak self] peerID in
            self?.retirePeer(peerID, reason: "its Bonjour record disappeared",
                             redialAfter: nil)
        }
        try discovery.start()
        onStatus?("browsing for \(NMPBonjour.serviceType) peers …")
    }

    public func stop() {
        queue.async { [self] in
            discovery?.stop()
            for connection in connections.values { connection.close() }
            connections.removeAll()
            readyPeers.removeAll()
        }
    }

    private func dialIfPossible(_ capabilities: NMPCapabilities) {
        guard connections[capabilities.peerID] == nil else { return }
        guard capabilities.udpPort != 0,
              let remoteStatic = capabilities.noiseStaticPublicKey else {
            onStatus?("peer \(String(format: "%08x", capabilities.peerID)) "
                      + "not dialable (no port/key in TXT) — skipping")
            return
        }
        guard let endpoint = discovery?.discoveredPeers[capabilities.peerID]?.endpoint else {
            onStatus?("peer \(String(format: "%08x", capabilities.peerID)) has no endpoint yet")
            return
        }

        onStatus?("dialing \(capabilities.deviceName) "
                  + "(\(String(format: "%08x", capabilities.peerID)), "
                  + "\(capabilities.computeClass.label), port \(capabilities.udpPort)) …")

        let connectionQueue = DispatchQueue(
            label: "nmp.coordinator.link.\(capabilities.peerID)")
        let transport = UDPTransport(endpoint: endpoint, queue: connectionQueue)
        do {
            let connection = try PeerConnection(
                role: .initiator,
                config: PeerConnectionConfig(localPeerID: localPeerID),
                transport: transport,
                localStatic: staticKeys,
                remoteStaticPublicKey: remoteStatic,
                queue: connectionQueue)
            connections[capabilities.peerID] = connection
            connectionQueues[capabilities.peerID] = connectionQueue

            connection.onEstablished = { [weak self] _, remoteID in
                self?.adoptPeer(capabilities, remoteID: remoteID,
                                connection: connection)
            }
            connection.onFailed = { [weak self] error in
                self?.retirePeer(capabilities.peerID,
                                 reason: "connection failed: \(error)",
                                 redialAfter: nil)
            }
            connection.start()
        } catch {
            onStatus?("failed to dial peer: \(error)")
        }
    }

    // MARK: Membership lifecycle (adopt / retire)

    /// Registers a freshly handshaked peer — after retiring any GHOST
    /// entry for the same physical device. iOS peers mint a NEW peerID per
    /// connection (documented behavior), so a backgrounded phone that
    /// comes back arrives as a brand-new peer while its stale identity
    /// keeps a membership slot AND its place in the routed plan —
    /// generations then chase the ghost with 100% timeouts while /health
    /// counts it alive (BUG-3). Safe from any queue (hops to `queue`;
    /// membership is queue-owned, per the membership-locking discipline).
    /// `connection` is nil only in tests (no wire to attach).
    internal func adoptPeer(_ capabilities: NMPCapabilities, remoteID: UInt32,
                            connection: PeerConnection?) {
        queue.async { [self] in
            for ghostID in Self.stalePeerIDs(replacing: capabilities,
                                             currentID: remoteID,
                                             among: readyPeers) {
                retireLocked(ghostID,
                             reason: "same device rejoined as "
                                     + "\(String(format: "%08x", remoteID)) — retiring "
                                     + "the stale peer entry")
            }
            readyPeers[remoteID] = capabilities
            if let connection {
                connections[remoteID] = connection
                orchestrator.attachPeer(peerID: remoteID, connection: connection)
            }
            onStatus?("peer \(String(format: "%08x", remoteID)) established "
                      + "(handshake complete)")
            onPeerReady?(capabilities)
        }
    }

    /// Snapshots membership ON the node queue and hands it to `completion`
    /// (also on the queue). `readyPeers` is queue-owned; unlocked
    /// cross-queue dictionary reads are the exact crash class the
    /// membership-locking rule exists for — tests and tools must read
    /// through here instead.
    internal func withMembership(
        _ completion: @escaping ([UInt32: NMPCapabilities]) -> Void
    ) {
        queue.async { [self] in completion(readyPeers) }
    }

    /// Removes a peer from membership and routing: closes its link,
    /// detaches it from the orchestrator (failing any in-flight stage
    /// toward it), and reports the loss so the front-end drops it from
    /// alive counts and re-shards around it. Safe from any queue.
    /// `redialAfter`: for a compute stall, one delayed re-dial folds a
    /// transiently wedged peer back in IF it still advertises — a
    /// backgrounded phone can't answer the dial, so a true ghost stays out.
    internal func retirePeer(_ peerID: UInt32, reason: String,
                             redialAfter: TimeInterval?) {
        queue.async { [self] in
            retireLocked(peerID, reason: reason)
            guard let cooldown = redialAfter else { return }
            queue.asyncAfter(deadline: .now() + cooldown) { [weak self] in
                guard let self,
                      self.connections[peerID] == nil,
                      let rediscovered = self.discovery?
                          .discoveredPeers[peerID]?.capabilities else { return }
                self.onStatus?("peer \(String(format: "%08x", peerID)) still "
                               + "advertised after retirement — re-dialing once")
                self.dialIfPossible(rediscovered)
            }
        }
    }

    /// MUST run on `queue`. Idempotent — a peer already gone is a no-op,
    /// so the discovery-removal, connection-failure, stall, and ghost
    /// paths can all race safely.
    private func retireLocked(_ peerID: UInt32, reason: String) {
        let hadConnection = connections[peerID] != nil
        let wasReady = readyPeers.removeValue(forKey: peerID) != nil
        guard wasReady || hadConnection else { return }
        onStatus?("retiring peer \(String(format: "%08x", peerID)): \(reason)")
        connections.removeValue(forKey: peerID)?.close()
        connectionQueues.removeValue(forKey: peerID)
        orchestrator.detachPeer(peerID: peerID)
        if wasReady { onPeerLost?(peerID) }
    }

    /// Ready-peer entries that are the SAME physical device as `caps` under
    /// a different peerID — the ghosts a rejoin must retire. The hardware
    /// marker (deviceName is the hardware model string), RAM, and compute
    /// class identify a device across the per-connection peerIDs iOS mints.
    /// Limitation: two distinct devices of the identical model AND RAM
    /// collide; the wrongly retired one is re-dialed on its next
    /// advertisement (trusted benchmark LAN trade-off).
    internal static func stalePeerIDs(
        replacing caps: NMPCapabilities, currentID: UInt32,
        among ready: [UInt32: NMPCapabilities]
    ) -> [UInt32] {
        ready.filter { id, existing in
            id != currentID && isSamePhysicalDevice(existing, caps)
        }.keys.sorted()
    }

    internal static func isSamePhysicalDevice(
        _ a: NMPCapabilities, _ b: NMPCapabilities
    ) -> Bool {
        !a.deviceName.isEmpty
            && a.deviceName == b.deviceName
            && a.ramMB == b.ramMB
            && a.computeClass == b.computeClass
    }

    /// Plans over the local device + all ready peers, then broadcasts.
    public func planAndAssign(
        completion: @escaping (Result<[NMPShardPlanEntry], NMPOrchestrationError>) -> Void
    ) {
        queue.async { [self] in replanLocked(completion: completion) }
    }

    /// Re-plans and re-assigns with the CURRENT mode (auto = measured,
    /// manual = operator shares). Use after a mode/share change.
    public func replan(
        completion: @escaping (Result<[NMPShardPlanEntry], NMPOrchestrationError>) -> Void
    ) {
        queue.async { [self] in replanLocked(completion: completion) }
    }

    /// Switches auto/manual balancing. The mode flips and the completion
    /// fires IMMEDIATELY with the staged plan; the SHARD_ASSIGN round runs
    /// in the background (BUG-7: a peer newly handed layers may
    /// vault-stream them for up to ~30 s before acking, and an HTTP caller
    /// must not hang on that — real toggles used to block 22–30 s). The
    /// orchestrator commits on ack, so the OLD plan keeps serving until
    /// the round lands; the background outcome is reported via
    /// `onBackgroundReshard` (failure = the staged plan was never applied).
    /// Turning auto ON discards the manual caps so the mesh rebalances
    /// purely by speed.
    public func setAutoBalance(
        _ on: Bool,
        completion: @escaping (Result<[NMPShardPlanEntry], NMPOrchestrationError>) -> Void
    ) {
        queue.async { [self] in
            autoBalance = on
            if on { manualShares.removeAll() }
            let staged = stagePlanLocked()
            guard !staged.entries.isEmpty else {
                completion(.failure(.emptyPlan))
                return
            }
            completion(.success(staged.entries))
            assignLocked(staged) { [weak self] result in
                if case .failure(let error) = result {
                    self?.onStatus?("auto-balance re-shard failed in the "
                                    + "background: \(error) — the previous plan "
                                    + "keeps serving")
                }
                self?.onBackgroundReshard?(result)
            }
        }
    }

    /// Sets one peer's compute-share cap and re-plans. Implies manual mode —
    /// an explicit allocation only makes sense with auto off, so this flips
    /// it off (matching the UI: sliders are live only in manual mode).
    public func setManualShare(
        peerID: UInt32, share: Double,
        completion: @escaping (Result<[NMPShardPlanEntry], NMPOrchestrationError>) -> Void
    ) {
        queue.async { [self] in
            autoBalance = false
            manualShares[peerID] = min(1.0, max(0.0, share))
            replanLocked(completion: completion)
        }
    }

    /// Auto-convergence tick: in auto mode, recompute the measured-optimal
    /// plan and re-assign ONLY if it differs materially from the live plan
    /// (so a mesh that just measured a fast/slow device rebalances once, then
    /// stays put instead of churning SHARD_ASSIGN rounds). Never re-shards
    /// mid-generation. `completion` gets the new plan when it rebalanced,
    /// `nil` when nothing changed (no-op — caller need not touch the UI).
    public func autoRebalanceTick(
        inflight: Bool,
        completion: @escaping (Result<[NMPShardPlanEntry], NMPOrchestrationError>?) -> Void
    ) {
        queue.async { [self] in
            guard autoBalance, !inflight, !readyPeers.isEmpty else {
                completion(nil); return
            }
            let plan = autoPlan()
            let newPlan = plan.entries
            guard Self.materiallyDiffers(newPlan, from: orchestrator.plan) else {
                completion(nil); return
            }
            orchestrator.assignShards(
                newPlan, idlePeers: plan.exclusions.map(\.peerID)) { result in
                completion(result.map { newPlan })
            }
        }
    }

    /// MUST run on `queue`. The measured latency-optimal plan for the current
    /// members: minimizes per-token wall-clock (compute + round trips), so a
    /// fast-but-far peer is used only when it genuinely helps or capacity
    /// forces it. Capacity ceilings come from each device's RAM.
    private func autoPlan() -> NMPShardPlan {
        planFor(strategy: planStrategy,
                members: [localCapabilities] + Array(readyPeers.values))
    }

    /// RAM-derived layer ceiling per device (empty = unbounded footprint).
    private func layerCapacityMap(_ members: [NMPCapabilities]) -> [UInt32: Int] {
        guard modelBytesPerLayer > 0 else { return [:] }
        var caps: [UInt32: Int] = [:]
        for m in members {
            caps[m.peerID] = NMPModelSharder.layerCapacity(
                ramMB: m.ramMB, bytesPerLayer: modelBytesPerLayer)
        }
        return caps
    }

    /// The plan a given strategy produces for `members`. MUST run on `queue`.
    private func planFor(strategy: PlanStrategy,
                         members: [NMPCapabilities]) -> NMPShardPlan {
        let caps = layerCapacityMap(members)
        switch strategy {
        case .speed:
            return NMPModelSharder.planByLatency(
                layerCount: engine.layerCount, coordinatorPeerID: localPeerID,
                peers: members,
                computeSecondsPerLayer: orchestrator.measuredSecondsPerLayer,
                roundTripSeconds: orchestrator.measuredRoundTripSeconds,
                layerCapacities: caps)
        case .capacity:
            return NMPModelSharder.planByCapacity(
                layerCount: engine.layerCount, coordinatorPeerID: localPeerID,
                peers: members, layerCapacities: caps)
        case .balanced:
            return NMPModelSharder.planDetailed(
                layerCount: engine.layerCount, peers: members,
                measuredSecondsPerLayer: orchestrator.measuredSecondsPerLayer,
                layerCapacities: caps, objective: .capacityThenSpeed)
        }
    }

    /// All three strategies' plans for the current mesh, for the pre-shard
    /// preview — the operator compares per-device footprints, then applies one
    /// with `setPlanStrategy`. Completion fires on `queue`.
    public func candidatePlans(
        completion: @escaping ([(strategy: PlanStrategy, plan: NMPShardPlan)]) -> Void
    ) {
        queue.async { [self] in
            let members = [localCapabilities] + Array(readyPeers.values)
            completion(PlanStrategy.allCases.map {
                ($0, planFor(strategy: $0, members: members))
            })
        }
    }

    /// Picks the auto strategy and re-shards onto it (auto on, manual caps
    /// cleared) — the "apply this previewed plan" action.
    public func setPlanStrategy(
        _ strategy: PlanStrategy,
        completion: @escaping (Result<[NMPShardPlanEntry], NMPOrchestrationError>) -> Void
    ) {
        queue.async { [self] in
            autoBalance = true
            manualShares.removeAll()
            planStrategy = strategy
            replanLocked(completion: completion)
        }
    }

    /// A plan computed for the current mode but not yet assigned.
    private struct StagedPlan {
        let entries: [NMPShardPlanEntry]
        let idlePeers: [UInt32]
    }

    /// MUST run on `queue`. Builds (but does not assign) a plan for the
    /// current mode. In manual mode a peer explicitly set to 0 is EXCLUDED
    /// (dropped from the plan and parked as idle) — the sharder floors
    /// positive shares at 5%, so "Mac-only" needs a real exclusion, not a
    /// tiny share.
    private func stagePlanLocked() -> StagedPlan {
        let peers = Array(readyPeers.values)
        if autoBalance {
            // Network-aware: minimize measured per-token latency.
            let plan = autoPlan()
            return StagedPlan(entries: plan.entries,
                              idlePeers: plan.exclusions.map(\.peerID))
        }
        // Manual: operator shares; a 0 share fully excludes the peer (the
        // sharder floors positive shares at 5%, so "Mac-only" needs a real
        // drop, not a tiny share).
        let excluded = peers
            .filter { (manualShares[$0.peerID] ?? 1.0) <= 0 }
            .map(\.peerID)
        let members = [localCapabilities]
            + peers.filter { !excluded.contains($0.peerID) }
        let entries = NMPModelSharder.plan(
            layerCount: engine.layerCount,
            peers: members,
            measuredSecondsPerLayer: orchestrator.measuredSecondsPerLayer,
            computeShares: manualShares)
        return StagedPlan(entries: entries, idlePeers: excluded)
    }

    /// MUST run on `queue`. Runs the SHARD_ASSIGN round for a staged plan.
    private func assignLocked(
        _ staged: StagedPlan,
        completion: @escaping (Result<[NMPShardPlanEntry], NMPOrchestrationError>) -> Void
    ) {
        // An operator toggle can hand a peer a range it doesn't hold yet, so
        // it must vault-stream those layers before it can ACK the assignment.
        // Give that far more room than the 10 s default (a phone streaming
        // ~half the model over Wi-Fi) so a re-balance doesn't spuriously time
        // out; a genuinely dead peer still fails, just later.
        orchestrator.assignShards(staged.entries, idlePeers: staged.idlePeers,
                                  timeout: 30) { result in
            completion(result.map { staged.entries })
        }
    }

    /// MUST run on `queue`. Builds and assigns a plan for the current mode.
    private func replanLocked(
        completion: @escaping (Result<[NMPShardPlanEntry], NMPOrchestrationError>) -> Void
    ) {
        assignLocked(stagePlanLocked(), completion: completion)
    }

    /// True when two plans differ enough to be worth a re-shard: any peer's
    /// layer count moves by ≥2, or the membership of the plan changes. A ±1
    /// jitter is ignored so measurement noise doesn't cause churn.
    private static func materiallyDiffers(
        _ a: [NMPShardPlanEntry], from b: [NMPShardPlanEntry]
    ) -> Bool {
        let countsA = Dictionary(a.map { ($0.peerID, $0.layerSpan) },
                                 uniquingKeysWith: +)
        let countsB = Dictionary(b.map { ($0.peerID, $0.layerSpan) },
                                 uniquingKeysWith: +)
        if Set(countsA.keys) != Set(countsB.keys) { return true }
        for (peerID, spanA) in countsA {
            if abs(spanA - (countsB[peerID] ?? 0)) >= 2 { return true }
        }
        return false
    }
}
