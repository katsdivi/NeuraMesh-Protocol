//
//  PeerHealthMonitor.swift
//  NMP — Phase 6
//
//  Activity-based peer liveness. NMP has no dedicated keepalive packet:
//  during inference the pipeline itself is the heartbeat (every response
//  chunk, ack, and metrics packet counts as activity), which is exactly
//  the traffic whose absence matters. A tracked peer that has been silent
//  longer than `heartbeatTimeout` is reported dead by `checkHealth()`.
//
//  Compute-path verdict (BUG-3): heartbeat activity alone is NOT proof a
//  peer can serve its shard. A backgrounded iPhone keeps echoing pings and
//  metrics while its compute engine is frozen — every generation then
//  times out against a peer the activity clock swears is alive. So the
//  monitor also carries a compute-stall flag, fed by consecutive stage
//  timeouts (see NMPInferenceOrchestrator.onPeerComputeStalled), and
//  `checkHealth()` reports a stalled peer as dead EVEN IF packets keep
//  arriving: activity must never veto compute-path evidence.
//
//  The clock is injectable so tests exercise the 5-second timeout without
//  waiting 5 seconds. Thread-safe: callers record activity from connection
//  queues while the failover orchestrator polls from its own queue.
//

import Foundation

public final class NMPPeerHealthMonitor {

    /// Silence longer than this marks a peer dead (spec: 5 seconds).
    public let heartbeatTimeout: TimeInterval

    private let now: () -> Date
    private let lock = NSLock()
    private var lastActivity: [UInt32: Date] = [:]
    /// Peers whose compute path is stalled (consecutive stage timeouts).
    /// Reported dead by `checkHealth()` regardless of packet activity.
    private var computeStalled: Set<UInt32> = []

    /// - Parameter clock: injectable time source (tests advance it manually).
    public init(heartbeatTimeout: TimeInterval = 5.0,
                clock: @escaping () -> Date = Date.init) {
        self.heartbeatTimeout = heartbeatTimeout
        self.now = clock
    }

    /// Begins monitoring a peer; the deadline starts from "now" so a peer
    /// is never declared dead before it had a full timeout to speak. Also
    /// clears any stale compute-stall flag (a re-tracked peer starts fresh).
    public func track(peerID: UInt32) {
        lock.lock(); defer { lock.unlock() }
        lastActivity[peerID] = now()
        computeStalled.remove(peerID)
    }

    /// Stops monitoring (peer left the mesh deliberately or was failed over).
    public func forget(peerID: UInt32) {
        lock.lock(); defer { lock.unlock() }
        lastActivity.removeValue(forKey: peerID)
        computeStalled.remove(peerID)
    }

    /// Any authenticated packet from the peer counts as a heartbeat.
    /// No-op for untracked peers — activity alone does not enroll a peer.
    /// Deliberately does NOT clear a compute-stall flag: pings and metrics
    /// prove the radio works, not that the shard engine does.
    public func recordActivity(peerID: UInt32) {
        lock.lock(); defer { lock.unlock() }
        guard lastActivity[peerID] != nil else { return }
        lastActivity[peerID] = now()
    }

    /// The peer's compute path timed out enough consecutive times to call
    /// it stalled (the orchestrator does the counting). From here on
    /// `checkHealth()` reports the peer dead even while its heartbeat
    /// stays fresh. No-op for untracked peers.
    public func recordComputeStall(peerID: UInt32) {
        lock.lock(); defer { lock.unlock() }
        guard lastActivity[peerID] != nil else { return }
        computeStalled.insert(peerID)
    }

    /// A stage completed against the peer after it was flagged — the stall
    /// was transient; restore the activity-only verdict.
    public func recordComputeRecovery(peerID: UInt32) {
        lock.lock(); defer { lock.unlock() }
        computeStalled.remove(peerID)
    }

    /// True if the peer is currently flagged compute-stalled.
    public func isComputeStalled(peerID: UInt32) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return computeStalled.contains(peerID)
    }

    /// Tracked peers considered dead, sorted: silent for longer than
    /// `heartbeatTimeout`, or compute-stalled (which fresh heartbeat
    /// activity does not veto).
    public func checkHealth() -> [UInt32] {
        lock.lock(); defer { lock.unlock() }
        let deadline = now()
        let silent = lastActivity
            .filter { deadline.timeIntervalSince($0.value) > heartbeatTimeout }
            .keys
        return Set(silent).union(computeStalled).sorted()
    }

    public var trackedPeers: [UInt32] {
        lock.lock(); defer { lock.unlock() }
        return lastActivity.keys.sorted()
    }

    public func lastActivityDate(peerID: UInt32) -> Date? {
        lock.lock(); defer { lock.unlock() }
        return lastActivity[peerID]
    }
}
