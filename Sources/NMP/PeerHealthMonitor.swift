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

    /// - Parameter clock: injectable time source (tests advance it manually).
    public init(heartbeatTimeout: TimeInterval = 5.0,
                clock: @escaping () -> Date = Date.init) {
        self.heartbeatTimeout = heartbeatTimeout
        self.now = clock
    }

    /// Begins monitoring a peer; the deadline starts from "now" so a peer
    /// is never declared dead before it had a full timeout to speak.
    public func track(peerID: UInt32) {
        lock.lock(); defer { lock.unlock() }
        lastActivity[peerID] = now()
    }

    /// Stops monitoring (peer left the mesh deliberately or was failed over).
    public func forget(peerID: UInt32) {
        lock.lock(); defer { lock.unlock() }
        lastActivity.removeValue(forKey: peerID)
    }

    /// Any authenticated packet from the peer counts as a heartbeat.
    /// No-op for untracked peers — activity alone does not enroll a peer.
    public func recordActivity(peerID: UInt32) {
        lock.lock(); defer { lock.unlock() }
        guard lastActivity[peerID] != nil else { return }
        lastActivity[peerID] = now()
    }

    /// Tracked peers silent for longer than `heartbeatTimeout`, sorted.
    public func checkHealth() -> [UInt32] {
        lock.lock(); defer { lock.unlock() }
        let deadline = now()
        return lastActivity
            .filter { deadline.timeIntervalSince($0.value) > heartbeatTimeout }
            .keys.sorted()
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
