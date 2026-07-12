//
//  Mesh23Tests.swift
//  NMP — Mesh 2.3
//
//  Per-device telemetry: connection wire-traffic counters, peer resource
//  reports over the mesh, and whole-machine GPU sampling.
//

import XCTest
@testable import NMP

// MARK: - Resource report wire format

final class PeerResourceReportTests: XCTestCase {

    func testCodecRoundTripsEveryField() throws {
        let report = NMPPeerResourceReport(
            peerID: 0xABCD_1234,
            ramTotalMB: 16_384, ramUsedMB: 11_588,
            processFootprintMB: 73,
            storageTotalMB: 233_779, storageFreeMB: 9_011,
            cpuPercent: 10.5, gpuPercent: 42.3,
            hostname: "MacBook-Air-8.local")
        let decoded = try NMPPeerResourceReport.decode(report.encode())
        XCTAssertEqual(decoded, report)
    }

    func testNilPercentagesSurviveTheWire() throws {
        // First CPU sample has no tick delta; GPU is macOS-only. Both ride
        // as the 0xFFFF sentinel and must come back as nil, not 6553.5%.
        let report = NMPPeerResourceReport(
            peerID: 2, ramTotalMB: 6_144, ramUsedMB: 3_072,
            processFootprintMB: 120, storageTotalMB: 131_072,
            storageFreeMB: 65_536, cpuPercent: nil, gpuPercent: nil,
            hostname: "iPhone.local")
        let decoded = try NMPPeerResourceReport.decode(report.encode())
        XCTAssertNil(decoded.cpuPercent)
        XCTAssertNil(decoded.gpuPercent)
        XCTAssertEqual(decoded, report)
    }

    func testSampleInitializerConvertsUnits() {
        let sample = NMPHostResourceSample(
            hostname: "test.local",
            ramTotalBytes: 8 << 30, ramUsedBytes: 4 << 30,
            processFootprintBytes: 100 << 20,
            storageTotalBytes: 250 << 30, storageFreeBytes: 50 << 30,
            cpuPercent: 25, gpuPercent: nil, sampledAt: Date())
        let report = NMPPeerResourceReport(peerID: 7, sample: sample)
        XCTAssertEqual(report.ramTotalMB, 8 * 1024)
        XCTAssertEqual(report.ramUsedMB, 4 * 1024)
        XCTAssertEqual(report.processFootprintMB, 100)
        XCTAssertEqual(report.storageTotalMB, 250 * 1024)
        XCTAssertEqual(report.cpuPercent, 25)
        XCTAssertNil(report.gpuPercent)
        XCTAssertEqual(report.hostname, "test.local")
    }

    func testDecodeRejectsWrongVersionAndTruncation() {
        var bytes = NMPPeerResourceReport(
            peerID: 1, ramTotalMB: 1, ramUsedMB: 1, processFootprintMB: 1,
            storageTotalMB: 1, storageFreeMB: 1, cpuPercent: nil,
            gpuPercent: nil, hostname: "x").encode()
        bytes[1] = 99 // future version
        XCTAssertThrowsError(try NMPPeerResourceReport.decode(bytes))
        XCTAssertThrowsError(
            try NMPPeerResourceReport.decode(bytes.prefix(10)))
    }
}

// MARK: - Wire traffic counters

final class WireTrafficTests: XCTestCase {

    /// A live mesh moves real datagrams; the counters must see them from
    /// both ends of the link — coordinator sent ≈ peer received (the
    /// in-memory transport is lossless, so byte-exact).
    func testCountersMatchAcrossALosslessLink() throws {
        let testbed = try NMPMeshTestbed(
            layerCount: 6, hiddenSize: 64, remotePeerCount: 1)
        _ = try testbed.startSync()
        _ = try testbed.inferSync(input: testbed.makeInput())

        let peer = try XCTUnwrap(testbed.remotePeers.first)
        let coordinator = peer.coordinatorSide.trafficTotals
        let remote = peer.peerSide.trafficTotals

        XCTAssertGreaterThan(coordinator.sentBytes, 0,
                             "handshake + SHARD_ASSIGN + tensors went out")
        XCTAssertGreaterThan(coordinator.receivedBytes, 0,
                             "acks + response tensors came back")
        XCTAssertEqual(coordinator.sentBytes, remote.receivedBytes,
                       "every byte the coordinator sent arrived at the peer")
        XCTAssertEqual(coordinator.receivedBytes, remote.sentBytes)
    }

    func testCountersGrowWithTraffic() throws {
        let testbed = try NMPMeshTestbed(
            layerCount: 6, hiddenSize: 64, remotePeerCount: 1)
        _ = try testbed.startSync()
        let connection = try XCTUnwrap(testbed.remotePeers.first).coordinatorSide

        _ = try testbed.inferSync(input: testbed.makeInput())
        let after1 = connection.trafficTotals
        _ = try testbed.inferSync(input: testbed.makeInput(seed: 2))
        let after2 = connection.trafficTotals

        XCTAssertGreaterThan(after2.sentBytes, after1.sentBytes)
        XCTAssertGreaterThan(after2.receivedBytes, after1.receivedBytes)
    }
}

// MARK: - Peer resource reports over the mesh

final class PeerResourceReportFlowTests: XCTestCase {

    /// The shard peers must ship their kernel counters to the coordinator
    /// unprompted: once on assignment, then throttled alongside metrics.
    func testPeersReportResourcesOnAssignment() throws {
        let testbed = try NMPMeshTestbed(
            layerCount: 6, hiddenSize: 64, remotePeerCount: 2)

        var reports: [UInt32: NMPPeerResourceReport] = [:]
        let lock = NSLock()
        let gotBoth = XCTestExpectation(description: "both peers reported")
        testbed.orchestrator.onPeerResourceReport = { report in
            lock.lock()
            reports[report.peerID] = report
            let count = reports.count
            lock.unlock()
            if count == 2 { gotBoth.fulfill() }
        }

        _ = try testbed.startSync() // SHARD_ASSIGN → forced report
        wait(for: [gotBoth], timeout: 5)

        lock.lock()
        defer { lock.unlock() }
        for peer in testbed.remotePeers {
            let report = try XCTUnwrap(reports[peer.capabilities.peerID])
            // In-process peers report THIS host — real counters, and the
            // hostname match is what lets the dashboard label them as
            // sharing the coordinator's hardware.
            XCTAssertEqual(report.hostname, NMPLANIdentity.localHostname())
            XCTAssertGreaterThan(report.ramTotalMB, 1024)
            XCTAssertGreaterThan(report.ramUsedMB, 0)
            XCTAssertGreaterThan(report.processFootprintMB, 1)
            XCTAssertGreaterThan(report.storageTotalMB, 1024)
        }
    }

    /// Serving traffic keeps reports flowing (throttled, so drive passes
    /// until a second one lands).
    func testReportsKeepFlowingWhileServing() throws {
        let testbed = try NMPMeshTestbed(
            layerCount: 6, hiddenSize: 64, remotePeerCount: 1)
        let peerID = try XCTUnwrap(testbed.remotePeers.first).capabilities.peerID

        var reportCount = 0
        let lock = NSLock()
        testbed.orchestrator.onPeerResourceReport = { report in
            guard report.peerID == peerID else { return }
            lock.lock(); reportCount += 1; lock.unlock()
        }
        _ = try testbed.startSync()

        // The throttle interval is 2 s; poll passes until the second
        // report arrives (fixed sleeps are flaky under suite load).
        let deadline = Date().addingTimeInterval(8)
        var seen = 0
        var seed: UInt64 = 1
        repeat {
            _ = try testbed.inferSync(input: testbed.makeInput(seed: seed))
            seed += 1
            Thread.sleep(forTimeInterval: 0.2)
            lock.lock(); seen = reportCount; lock.unlock()
        } while seen < 2 && Date() < deadline
        XCTAssertGreaterThanOrEqual(seen, 2,
                                    "no follow-up resource report within 8 s")
    }
}

// MARK: - Keepalive pings (Mesh 2.4)

final class KeepalivePingTests: XCTestCase {

    func testPingCodecRoundTripsAndPongEchoesNonce() throws {
        let ping = NMPPeerPing(nonce: 0xDEAD_BEEF_CAFE_F00D)
        XCTAssertEqual(try NMPPeerPing.decode(ping.encode()), ping)
        let pong = ping.pongPayload()
        XCTAssertEqual(pong.first, NMPMeshMessageKind.pong.rawValue)
        XCTAssertEqual(Data(pong).readBigEndianUInt64(at: 1), ping.nonce)
    }

    /// The whole point of the ping: an idle-but-alive peer's activity
    /// clock advances because it echoed, so the health monitor will not
    /// read a stalled pipeline as that peer's death.
    func testPingEchoAdvancesTheActivityClock() throws {
        let testbed = try NMPMeshTestbed(
            layerCount: 6, hiddenSize: 64, remotePeerCount: 1)
        _ = try testbed.startSync()
        let peerID = try XCTUnwrap(testbed.remotePeers.first).capabilities.peerID
        let monitor = testbed.failover.healthMonitor

        // Let the post-assignment traffic settle, then take a baseline.
        Thread.sleep(forTimeInterval: 0.3)
        let baseline = try XCTUnwrap(monitor.lastActivityDate(peerID: peerID))

        testbed.orchestrator.sendKeepalivePing(to: peerID)
        var advanced: Date?
        for _ in 0..<20 where advanced == nil {
            Thread.sleep(forTimeInterval: 0.1)
            if let now = monitor.lastActivityDate(peerID: peerID), now > baseline {
                advanced = now
            }
        }
        XCTAssertNotNil(advanced, "pong never arrived — activity clock stuck")
    }
}

// MARK: - GPU sampling

final class GPUSamplingTests: XCTestCase {

    func testGPUUtilizationIsNilOrAValidPercent() {
        // The counter is driver-provided; some Macs/CI runners omit it.
        // What must never happen is a value outside 0...100.
        if let percent = NMPResourceMonitor.gpuUtilizationPercent() {
            XCTAssertGreaterThanOrEqual(percent, 0)
            XCTAssertLessThanOrEqual(percent, 100)
        }
    }

    func testSampleJSONCarriesGPUOnlyWhenMeasured() {
        let with = NMPHostResourceSample(
            hostname: "x", ramTotalBytes: 1, ramUsedBytes: 1,
            processFootprintBytes: 1, storageTotalBytes: 1,
            storageFreeBytes: 1, cpuPercent: nil, gpuPercent: 37.25,
            sampledAt: Date())
        XCTAssertEqual(with.asJSONObject["gpu_percent"] as? Double, 37.3)

        let without = NMPHostResourceSample(
            hostname: "x", ramTotalBytes: 1, ramUsedBytes: 1,
            processFootprintBytes: 1, storageTotalBytes: 1,
            storageFreeBytes: 1, cpuPercent: nil, gpuPercent: nil,
            sampledAt: Date())
        XCTAssertNil(without.asJSONObject["gpu_percent"],
                     "an absent counter must be absent, not 0 — 0 is a claim")
    }
}
