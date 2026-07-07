//
//  PeerDiscoveryManager.swift
//  NMP — Phase 4
//
//  Ties discovery (Bonjour), capability advertisement, and coordinator
//  election together: publishes the local capability TXT record, maintains
//  the discovered-peer table from browse events, feeds every membership
//  change into the election, and surfaces the results.
//
//  Discovery transport is abstracted behind two small protocols so the
//  orchestration logic is deterministic under test (mock sources drive a
//  3-peer mesh with no real network) while production wires in the Bonjour
//  publisher/browser from Bonjour.swift.
//
//  Threading model mirrors PeerConnection: a caller-supplied serial queue
//  owns all state; source callbacks hop onto it; all callbacks fire on it;
//  mutating public methods must be called on it.
//

import Foundation
import Network

// MARK: - Discovery abstractions

/// Something that advertises the local device's capabilities to the
/// network. Production: `NMPBonjourPublisher`.
public protocol NMPCapabilityPublisher: AnyObject {
    func start() throws
    /// Re-advertise with fresh capabilities (TXT record update).
    func update(capabilities: NMPCapabilities)
    func stop()
}

/// Something that reports peers appearing/updating/disappearing.
/// Production: `NMPBonjourBrowser`. Callbacks may fire on any queue; the
/// manager rendezvouses them onto its own.
public protocol NMPPeerDiscoverySource: AnyObject {
    /// New peer, or capability update for a known peer.
    var onPeerFound: ((NMPCapabilities, NWEndpoint?) -> Void)? { get set }
    /// Peer's service disappeared (explicit removal or mDNS TTL expiry).
    var onPeerLost: ((UInt32) -> Void)? { get set }
    func start() throws
    func stop()
}

// MARK: - Discovered peer

public struct NMPDiscoveredPeer {
    public let capabilities: NMPCapabilities
    /// Where to dial the peer (Phase 5 uses this for shard connections).
    public let endpoint: NWEndpoint?
    /// Monotonic time of the last found/update event (staleness sweep).
    public var lastSeen: TimeInterval
}

// MARK: - Configuration

public struct NMPDiscoveryConfig: Sendable {
    /// How often local capabilities are re-measured and re-advertised.
    public var capabilityRefreshInterval: TimeInterval = 5
    /// Remove a peer not re-seen within this window. 0 disables the sweep
    /// (the default): Bonjour reports removals explicitly and does NOT
    /// re-announce unchanged services, so a wall-clock sweep would evict
    /// live-but-quiet peers. Enable only for sources without reliable
    /// removal events.
    public var peerStaleTimeout: TimeInterval = 0

    public init() {}
}

// MARK: - Manager

public final class NMPPeerDiscoveryManager {

    // MARK: Callbacks (invoked on `queue`)

    /// A peer joined the mesh (first sighting).
    public var onPeerDiscovered: ((NMPCapabilities) -> Void)?
    /// A known peer re-advertised with different capabilities.
    public var onPeerUpdated: ((NMPCapabilities) -> Void)?
    /// A peer left the mesh (service removed or stale).
    public var onPeerRemoved: ((UInt32) -> Void)?
    /// Election outcome changed. nil is impossible while the manager runs
    /// (the local device is always a member) but kept optional for symmetry
    /// with NMPCoordinatorElection.
    public var onCoordinatorChanged: ((UInt32?) -> Void)?

    // MARK: State

    public private(set) var localCapabilities: NMPCapabilities
    /// Remote peers only (never contains the local device).
    public private(set) var discoveredPeers: [UInt32: NMPDiscoveredPeer] = [:]
    public var currentCoordinator: UInt32? { election.currentCoordinator }
    /// True when the election ranks the local device first.
    public var isCoordinator: Bool { currentCoordinator == localCapabilities.peerID }

    private let publisher: NMPCapabilityPublisher?
    private let source: NMPPeerDiscoverySource
    private let election = NMPCoordinatorElection()
    private let config: NMPDiscoveryConfig
    private let queue: DispatchQueue
    /// Fresh capabilities on each refresh tick (e.g. re-measured load).
    /// Called on `queue`. nil = capabilities only change via
    /// `updateLocalCapabilities`.
    private let localCapabilityProvider: (() -> NMPCapabilities)?
    private var refreshTimer: DispatchSourceTimer?
    private var started = false

    /// Injectable clock (monotonic seconds) so staleness tests are
    /// deterministic.
    var clock: () -> TimeInterval = { PeerConnection.monotonicNow() }

    public init(
        localCapabilities: NMPCapabilities,
        publisher: NMPCapabilityPublisher?,
        source: NMPPeerDiscoverySource,
        config: NMPDiscoveryConfig = NMPDiscoveryConfig(),
        queue: DispatchQueue,
        localCapabilityProvider: (() -> NMPCapabilities)? = nil
    ) {
        self.localCapabilities = localCapabilities
        self.publisher = publisher
        self.source = source
        self.config = config
        self.queue = queue
        self.localCapabilityProvider = localCapabilityProvider
    }

    // MARK: Lifecycle

    public func start() throws {
        dispatchPrecondition(condition: .onQueue(queue))
        guard !started else { return }
        started = true

        // The local device is a mesh member from the outset: with zero
        // remote peers, it is its own coordinator.
        election.onCoordinatorChanged = { [weak self] winner in
            self?.onCoordinatorChanged?(winner)
        }
        election.upsert(localCapabilities)

        source.onPeerFound = { [weak self] capabilities, endpoint in
            self?.queue.async { self?.handleFound(capabilities, endpoint: endpoint) }
        }
        source.onPeerLost = { [weak self] peerID in
            self?.queue.async { self?.handleLost(peerID) }
        }
        try source.start()
        try publisher?.start()
        armRefreshTimer()
    }

    public func stop() {
        dispatchPrecondition(condition: .onQueue(queue))
        refreshTimer?.cancel(); refreshTimer = nil
        source.stop()
        publisher?.stop()
        started = false
    }

    // MARK: Capability updates

    /// Replaces the local capability advertisement: re-publishes the TXT
    /// record and re-runs the election (a compute-class change can move
    /// the coordinatorship).
    public func updateLocalCapabilities(_ capabilities: NMPCapabilities) {
        dispatchPrecondition(condition: .onQueue(queue))
        localCapabilities = capabilities
        publisher?.update(capabilities: capabilities)
        election.upsert(capabilities)
    }

    // MARK: Peer events

    private func handleFound(_ capabilities: NMPCapabilities, endpoint: NWEndpoint?) {
        // Bonjour browsers see the local device's own service — ignore it.
        guard capabilities.peerID != localCapabilities.peerID else { return }
        let known = discoveredPeers[capabilities.peerID]
        discoveredPeers[capabilities.peerID] = NMPDiscoveredPeer(
            capabilities: capabilities, endpoint: endpoint, lastSeen: clock())
        election.upsert(capabilities)
        if known == nil {
            onPeerDiscovered?(capabilities)
        } else if known?.capabilities != capabilities {
            onPeerUpdated?(capabilities)
        }
    }

    private func handleLost(_ peerID: UInt32) {
        guard discoveredPeers.removeValue(forKey: peerID) != nil else { return }
        election.remove(peerID: peerID)
        onPeerRemoved?(peerID)
    }

    /// Removes peers not re-seen within `peerStaleTimeout`. No-op when the
    /// sweep is disabled (timeout 0). Called from the refresh timer;
    /// exposed for deterministic tests.
    func expireStalePeers(at now: TimeInterval) {
        guard config.peerStaleTimeout > 0 else { return }
        let stale = discoveredPeers.filter { now - $0.value.lastSeen > config.peerStaleTimeout }
        for peerID in stale.keys.sorted() { handleLost(peerID) }
    }

    // MARK: Refresh timer

    private func armRefreshTimer() {
        guard config.capabilityRefreshInterval > 0 else { return }
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + config.capabilityRefreshInterval,
                       repeating: config.capabilityRefreshInterval)
        timer.setEventHandler { [weak self] in
            guard let self, self.started else { return }
            if let provider = self.localCapabilityProvider {
                let fresh = provider()
                if fresh != self.localCapabilities {
                    self.updateLocalCapabilities(fresh)
                }
            }
            self.expireStalePeers(at: self.clock())
        }
        timer.resume()
        refreshTimer = timer
    }
}
