//
//  CoordinatorElectionTests.swift
//  NMPTests — Phase 4
//
//  Deterministic coordinator election: highest compute class wins, ties
//  break to the lowest peerID, membership changes re-elect, load never
//  influences the outcome.
//

import XCTest
@testable import NMP

final class CoordinatorElectionTests: XCTestCase {

    private func caps(
        _ peerID: UInt32,
        _ computeClass: NMPComputeClass,
        load: Double = 0
    ) -> NMPCapabilities {
        NMPCapabilities(
            peerID: peerID, deviceName: "peer-\(peerID)", ramMB: 8192,
            computeClass: computeClass, currentLoadPercent: load)
    }

    func testElectionDeterministic() {
        // Same peer set, 100 elections, insertion order shuffled each time:
        // identical result every run.
        let peers = [
            caps(0x1234_5678, .high),   // expected winner
            caps(0xaabb_ccdd, .high),
            caps(0x9988_7766, .medium),
            caps(0x0000_0001, .low),
        ]
        for round in 0..<100 {
            let election = NMPCoordinatorElection()
            for peer in peers.shuffled() { election.upsert(peer) }
            XCTAssertEqual(election.electedCoordinator(), 0x1234_5678, "round \(round)")
            // Re-running on unchanged membership is idempotent.
            XCTAssertEqual(election.electedCoordinator(), election.electedCoordinator())
        }
    }

    func testElectionTiebreakerByPeerID() {
        let election = NMPCoordinatorElection()
        election.upsert(caps(0xaabb_ccdd, .high))
        election.upsert(caps(0x1234_5678, .high))
        XCTAssertEqual(election.currentCoordinator, 0x1234_5678,
                       "equal compute class must break ties to the LOWER peerID")
    }

    func testHigherComputeClassBeatsLowerPeerID() {
        let election = NMPCoordinatorElection()
        election.upsert(caps(0x0000_0001, .medium))
        election.upsert(caps(0xffff_ffff, .high))
        XCTAssertEqual(election.currentCoordinator, 0xffff_ffff)
    }

    func testElectionUpdatesOnPeerAdd() {
        let election = NMPCoordinatorElection()
        var announced: [UInt32?] = []
        election.onCoordinatorChanged = { announced.append($0) }

        election.upsert(caps(0x99, .medium))
        XCTAssertEqual(election.currentCoordinator, 0x99)

        // Stronger peer joins → coordinatorship moves.
        election.upsert(caps(0xaa, .high))
        XCTAssertEqual(election.currentCoordinator, 0xaa)

        // Weaker peer joins → no change, no callback.
        election.upsert(caps(0x01, .low))
        XCTAssertEqual(election.currentCoordinator, 0xaa)
        XCTAssertEqual(announced, [0x99, 0xaa])
    }

    func testElectionUpdatesOnPeerRemove() {
        let election = NMPCoordinatorElection()
        election.upsert(caps(0x10, .high))
        election.upsert(caps(0x20, .medium))
        election.upsert(caps(0x30, .medium))
        XCTAssertEqual(election.currentCoordinator, 0x10)

        // Coordinator drops → next in the total order takes over.
        XCTAssertTrue(election.remove(peerID: 0x10))
        XCTAssertEqual(election.currentCoordinator, 0x20)

        // Non-coordinator drops → no change.
        XCTAssertFalse(election.remove(peerID: 0x30))
        XCTAssertEqual(election.currentCoordinator, 0x20)

        // Removing an unknown peer is a no-op.
        XCTAssertFalse(election.remove(peerID: 0xdead))
    }

    func testSinglePeerIsAlwaysCoordinator() {
        let election = NMPCoordinatorElection()
        election.upsert(caps(0x42, .low))
        XCTAssertEqual(election.currentCoordinator, 0x42,
                       "a 1-peer mesh coordinates itself, even at the lowest tier")
    }

    func testEmptyMeshNoCoordinator() {
        let election = NMPCoordinatorElection()
        XCTAssertNil(election.electedCoordinator())
        XCTAssertNil(election.currentCoordinator)

        // ...and returning to empty announces the vacancy.
        var announced: [UInt32?] = []
        election.upsert(caps(0x42, .high))
        election.onCoordinatorChanged = { announced.append($0) }
        election.remove(peerID: 0x42)
        XCTAssertNil(election.currentCoordinator)
        XCTAssertEqual(announced, [nil])
    }

    func testLoadDoesNotAffectElection() {
        // Deliberate design point: load thrash must not thrash the
        // coordinatorship (see CoordinatorElection.swift header).
        let election = NMPCoordinatorElection()
        var changes = 0
        election.upsert(caps(0x10, .high, load: 0))
        election.upsert(caps(0x20, .high, load: 90))
        election.onCoordinatorChanged = { _ in changes += 1 }

        XCTAssertEqual(election.currentCoordinator, 0x10)
        election.upsert(caps(0x10, .high, load: 100)) // coordinator now maxed out
        XCTAssertEqual(election.currentCoordinator, 0x10)
        XCTAssertEqual(changes, 0)
    }

    func testCapabilityUpdateCanMoveCoordinatorship() {
        // Compute class DOES participate — a re-measured tier moves the crown.
        let election = NMPCoordinatorElection()
        election.upsert(caps(0x10, .medium))
        election.upsert(caps(0x20, .medium))
        XCTAssertEqual(election.currentCoordinator, 0x10)

        election.upsert(caps(0x20, .high))
        XCTAssertEqual(election.currentCoordinator, 0x20)
    }
}
