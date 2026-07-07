//
//  CoordinatorElection.swift
//  NMP — Phase 4
//
//  Deterministic coordinator election over the current peer set.
//
//  Rule (total order, no randomness, no communication round):
//
//    coordinator = peer with the highest computeClass,
//                  ties broken by the LOWEST peerID
//
//  Because every peer ranks the same capability set identically, all peers
//  that share a consistent view of the mesh independently elect the same
//  coordinator — no ballots, no consensus protocol. Divergent views (a peer
//  that hasn't yet discovered everyone) converge as discovery converges;
//  Phase 6 fault tolerance handles the in-flight work a coordinator change
//  can orphan.
//
//  DELIBERATE: `currentLoadPercent` does NOT participate. Load fluctuates
//  every refresh interval; ranking on it would thrash the coordinator every
//  5 seconds. Membership changes are the only re-election trigger.
//

import Foundation

public final class NMPCoordinatorElection {

    /// Fired when a membership change alters the election outcome.
    /// nil = mesh is empty (no coordinator).
    public var onCoordinatorChanged: ((UInt32?) -> Void)?

    public private(set) var peers: [UInt32: NMPCapabilities] = [:]
    public private(set) var currentCoordinator: UInt32?

    public init() {}

    /// Pure election over the current peer set. Repeated calls with the
    /// same peer set always return the same result.
    public func electedCoordinator() -> UInt32? {
        peers.values.min { a, b in
            if a.computeClass != b.computeClass {
                return a.computeClass > b.computeClass // higher class first
            }
            return a.peerID < b.peerID // lower peerID wins ties
        }?.peerID
    }

    /// Adds or updates a peer's capabilities and re-runs the election.
    /// Returns true if the coordinator changed.
    @discardableResult
    public func upsert(_ capabilities: NMPCapabilities) -> Bool {
        peers[capabilities.peerID] = capabilities
        return reelect()
    }

    /// Removes a peer and re-runs the election. Returns true if the
    /// coordinator changed (i.e. the coordinator itself dropped).
    @discardableResult
    public func remove(peerID: UInt32) -> Bool {
        guard peers.removeValue(forKey: peerID) != nil else { return false }
        return reelect()
    }

    private func reelect() -> Bool {
        let winner = electedCoordinator()
        guard winner != currentCoordinator else { return false }
        currentCoordinator = winner
        onCoordinatorChanged?(winner)
        return true
    }
}
