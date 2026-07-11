//
//  FaultToleranceOrchestrator.swift
//  NMP — Phase 6
//
//  Failover: when a shard peer dies mid-mesh, re-shard the model across
//  the survivors and rebroadcast SHARD_ASSIGN, so the next inference runs
//  on the reduced mesh instead of timing out forever. When a peer joins,
//  fold it into the plan the same way.
//
//  Layering: this wraps NMPInferenceOrchestrator (which owns connections
//  and the assignment protocol) and NMPModelSharder (which owns the
//  greedy heaviest-layers→fastest-peers split, weighted by measured
//  seconds-per-layer where available). Nothing here touches the wire —
//  a re-shard IS a normal assignment round over the surviving links.
//
//  Detection: `startHealthChecks` polls the health monitor; the liveness
//  signal is packet activity (wire `orchestrator.onPeerActivity` — done in
//  `adopt(orchestrator:)`). Detection therefore needs traffic in flight;
//  a mesh that is idle has nothing to fail over anyway.
//
//  Threading: callback style, one serial queue, same as every other NMP
//  component. Completions fire on `queue`.
//

import Foundation

public enum NMPFailoverError: Error, Equatable, Sendable {
    /// The dropped peer was the last compute-capable member: nothing left
    /// to re-shard onto. Surfaced explicitly — never a hang.
    case allPeersDead
    /// Drop requested for a peer that is not in the active assignment.
    case unknownPeer(UInt32)
    /// The re-shard's assignment round failed (peer rejected or timed out).
    case reshardFailed(NMPOrchestrationError)
    /// A failover is already being processed; retry after it completes.
    case failoverInProgress
}

public final class NMPFailoverOrchestrator {

    // MARK: Callbacks (invoked on `queue`)

    /// A peer was declared dead (before the re-shard starts).
    public var onPeerDropped: ((UInt32) -> Void)?
    /// A new plan is live: (plan, seconds the re-shard took end-to-end,
    /// including the SHARD_ASSIGN ack round trip).
    public var onResharded: (([NMPShardPlanEntry], TimeInterval) -> Void)?
    /// The mesh is out of compute peers.
    public var onAllPeersDead: (() -> Void)?
    public var onDiagnostic: ((String) -> Void)?

    // MARK: State

    /// Capabilities of every mesh member still considered alive. Mutated
    /// only on `queue`; the lock makes cross-queue READS safe — the
    /// Phase 9 adaptive controller, the dashboard CLI, and the Mesh 2.1
    /// device panel all read membership from their own queues, and an
    /// unlocked Array read racing a removeAll/append is a real crash
    /// (seen as index-out-of-range inside Collection.map).
    public var activePeers: [NMPCapabilities] {
        membershipLock.lock()
        defer { membershipLock.unlock() }
        return storedActivePeers
    }
    /// The plan currently assigned across `activePeers`. Same locking
    /// story as `activePeers`.
    public var activePlan: [NMPShardPlanEntry] {
        membershipLock.lock()
        defer { membershipLock.unlock() }
        return storedActivePlan
    }
    private var storedActivePeers: [NMPCapabilities]
    private var storedActivePlan: [NMPShardPlanEntry] = []
    private let membershipLock = NSLock()

    private func mutateMembership(_ body: (inout [NMPCapabilities],
                                           inout [NMPShardPlanEntry]) -> Void) {
        membershipLock.lock()
        defer { membershipLock.unlock() }
        body(&storedActivePeers, &storedActivePlan)
    }
    /// Wall-clock seconds the most recent re-shard took (nil before any).
    public private(set) var lastReshardSeconds: TimeInterval?

    public let healthMonitor: NMPPeerHealthMonitor

    private let orchestrator: NMPInferenceOrchestrator
    private let layerCount: Int
    private let localPeerID: UInt32
    private let queue: DispatchQueue
    private var healthTimer: DispatchSourceTimer?
    private var failoverInProgress = false

    /// - Parameters:
    ///   - peers: initial mesh membership INCLUDING the coordinator itself
    ///     (unless the coordinator contributes no compute).
    ///   - queue: serial queue owning failover state and callbacks. May be
    ///     the orchestrator's queue or a dedicated one.
    public init(
        orchestrator: NMPInferenceOrchestrator,
        layerCount: Int,
        localPeerID: UInt32,
        peers: [NMPCapabilities],
        healthMonitor: NMPPeerHealthMonitor = NMPPeerHealthMonitor(),
        queue: DispatchQueue
    ) {
        self.orchestrator = orchestrator
        self.layerCount = layerCount
        self.localPeerID = localPeerID
        self.storedActivePeers = peers
        self.healthMonitor = healthMonitor
        self.queue = queue

        // Liveness feed: every packet a shard peer sends the orchestrator
        // refreshes that peer's heartbeat. Remote peers only — the local
        // engine cannot "go silent".
        orchestrator.onPeerActivity = { [weak healthMonitor] peerID in
            healthMonitor?.recordActivity(peerID: peerID)
        }
        for peer in peers where peer.peerID != localPeerID {
            healthMonitor.track(peerID: peer.peerID)
        }
    }

    // MARK: Membership

    /// Adds a peer to membership WITHOUT re-sharding — mesh assembly before
    /// `assignInitialPlan`. For joins after the mesh is live, use
    /// `handlePeerJoin` (which re-shards). Idempotent per peerID.
    public func registerPeer(_ capabilities: NMPCapabilities, connection: PeerConnection) {
        queue.async { [self] in
            orchestrator.attachPeer(peerID: capabilities.peerID, connection: connection)
            mutateMembership { peers, _ in
                peers.removeAll { $0.peerID == capabilities.peerID }
                peers.append(capabilities)
            }
            healthMonitor.track(peerID: capabilities.peerID)
        }
    }

    /// Blocks until membership operations enqueued so far (registerPeer)
    /// have been applied. Mesh-assembly helper: `registerPeer` is async
    /// on `queue`, so code that registers peers and then immediately
    /// reads `activePeers` from ANOTHER queue (the Phase 9 adaptive
    /// controller, testbeds) would otherwise race the registration and
    /// see a partial mesh. Never call from `queue` itself.
    public func waitForMembership(timeout: TimeInterval = 5) {
        let settled = DispatchSemaphore(value: 0)
        queue.async { settled.signal() }
        _ = settled.wait(timeout: .now() + timeout)
    }

    // MARK: Initial plan

    /// Computes and assigns the first plan across `activePeers`.
    public func assignInitialPlan(
        timeout: TimeInterval = 10,
        completion: @escaping (Result<[NMPShardPlanEntry], NMPFailoverError>) -> Void
    ) {
        queue.async { [self] in
            reshard(reason: "initial plan", timeout: timeout, completion: completion)
        }
    }

    /// Mesh 2.1: re-plans over the CURRENT membership and re-assigns —
    /// how the Devices panel applies a changed compute share without a
    /// membership event. Same machinery as failover's re-shard (one
    /// normal SHARD_ASSIGN round); safe alongside live traffic exactly
    /// as failover re-shards are.
    public func replan(
        reason: String = "allocation change",
        timeout: TimeInterval = 10,
        completion: @escaping (Result<[NMPShardPlanEntry], NMPFailoverError>) -> Void
    ) {
        queue.async { [self] in
            guard !failoverInProgress else {
                completion(.failure(.failoverInProgress))
                return
            }
            reshard(reason: reason, timeout: timeout, completion: completion)
        }
    }

    // MARK: Failover

    /// Removes `deadPeerID` from the mesh, re-shards the survivors, and
    /// broadcasts the new assignment. Completion fires on `queue` with the
    /// new plan or an explicit error — never hangs.
    public func handlePeerDrop(
        _ deadPeerID: UInt32,
        timeout: TimeInterval = 10,
        completion: @escaping (Result<[NMPShardPlanEntry], NMPFailoverError>) -> Void
    ) {
        queue.async { [self] in
            guard !failoverInProgress else {
                completion(.failure(.failoverInProgress))
                return
            }
            guard activePeers.contains(where: { $0.peerID == deadPeerID }) else {
                completion(.failure(.unknownPeer(deadPeerID)))
                return
            }

            onDiagnostic?("peer \(hex(deadPeerID)) lost, re-sharding across "
                          + "\(activePeers.count - 1) survivor(s)")
            mutateMembership { peers, _ in
                peers.removeAll { $0.peerID == deadPeerID }
            }
            healthMonitor.forget(peerID: deadPeerID)
            // Detach fails anything in flight toward the dead peer and stops
            // routing its stale packets into the orchestrator.
            orchestrator.detachPeer(peerID: deadPeerID)
            onPeerDropped?(deadPeerID)

            guard !activePeers.isEmpty else {
                onAllPeersDead?()
                completion(.failure(.allPeersDead))
                return
            }
            reshard(reason: "peer \(hex(deadPeerID)) dropped",
                    timeout: timeout, completion: completion)
        }
    }

    /// Folds a newly connected peer into the mesh and re-shards. The
    /// connection must already be ESTABLISHED and running a shard engine.
    public func handlePeerJoin(
        _ capabilities: NMPCapabilities,
        connection: PeerConnection,
        timeout: TimeInterval = 10,
        completion: @escaping (Result<[NMPShardPlanEntry], NMPFailoverError>) -> Void
    ) {
        queue.async { [self] in
            guard !failoverInProgress else {
                completion(.failure(.failoverInProgress))
                return
            }
            orchestrator.attachPeer(peerID: capabilities.peerID, connection: connection)
            mutateMembership { peers, _ in
                peers.removeAll { $0.peerID == capabilities.peerID }
                peers.append(capabilities)
            }
            healthMonitor.track(peerID: capabilities.peerID)
            reshard(reason: "peer \(hex(capabilities.peerID)) joined",
                    timeout: timeout, completion: completion)
        }
    }

    // MARK: Health-driven detection

    /// Polls the health monitor every `interval`; each dead peer triggers a
    /// failover automatically. 1 s polling bounds detection latency at
    /// heartbeatTimeout + 1 s (5 s + 1 s < the 5.5 s spec budget).
    public func startHealthChecks(interval: TimeInterval = 1.0) {
        queue.async { [self] in
            healthTimer?.cancel()
            let timer = DispatchSource.makeTimerSource(queue: queue)
            timer.schedule(deadline: .now() + interval, repeating: interval)
            timer.setEventHandler { [weak self] in
                self?.runHealthCheck()
            }
            timer.resume()
            healthTimer = timer
        }
    }

    public func stopHealthChecks() {
        queue.async { [self] in
            healthTimer?.cancel()
            healthTimer = nil
        }
    }

    private func runHealthCheck() {
        guard !failoverInProgress else { return }
        guard let dead = healthMonitor.checkHealth().first else { return }
        handlePeerDrop(dead) { [weak self] result in
            if case .failure(let error) = result, error != .allPeersDead {
                self?.onDiagnostic?("auto-failover for \(hex(dead)) failed: \(error)")
            }
        }
    }

    // MARK: Re-shard core

    /// Plans over `activePeers` (weighted by measured seconds-per-layer so
    /// the heaviest spans land on the fastest peers) and runs an assignment
    /// round. Runs on `queue`.
    private func reshard(
        reason: String,
        timeout: TimeInterval,
        completion: @escaping (Result<[NMPShardPlanEntry], NMPFailoverError>) -> Void
    ) {
        let began = DispatchTime.now()
        let newPlan = NMPModelSharder.plan(
            layerCount: layerCount,
            peers: activePeers,
            measuredSecondsPerLayer: orchestrator.measuredSecondsPerLayer,
            computeShares: orchestrator.computeShares)
        guard !newPlan.isEmpty else {
            onAllPeersDead?()
            completion(.failure(.allPeersDead))
            return
        }

        failoverInProgress = true
        orchestrator.assignShards(newPlan, timeout: timeout) { [weak self] result in
            // Fires on the orchestrator queue; hop home.
            self?.queue.async {
                guard let self else { return }
                self.failoverInProgress = false
                switch result {
                case .failure(let error):
                    self.onDiagnostic?("re-shard (\(reason)) failed: \(error)")
                    completion(.failure(.reshardFailed(error)))
                case .success:
                    let seconds = TimeInterval(
                        DispatchTime.now().uptimeNanoseconds - began.uptimeNanoseconds) / 1e9
                    self.mutateMembership { _, plan in plan = newPlan }
                    self.lastReshardSeconds = seconds
                    self.onDiagnostic?("re-shard (\(reason)): \(newPlan.count) shard(s) in "
                                       + String(format: "%.1f ms", seconds * 1000))
                    self.onResharded?(newPlan, seconds)
                    completion(.success(newPlan))
                }
            }
        }
    }
}

private func hex(_ peerID: UInt32) -> String {
    "0x" + String(peerID, radix: 16)
}
