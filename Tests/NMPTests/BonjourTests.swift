//
//  BonjourTests.swift
//  NMPTests — Phase 4
//
//  Bonjour naming and TXT-record serialization are tested deterministically
//  (no network). The publish→browse round trip over real mDNS is exercised
//  in a guarded test: on hosts where mDNS is blocked (managed networks,
//  local-network privacy denials, sandboxed CI) it SKIPS rather than fails
//  — see Phase4_Design.md "Bonjour on restricted networks".
//

import XCTest
import Network
@testable import NMP

final class BonjourTests: XCTestCase {

    // MARK: Deterministic (no network)

    func testServiceNameRoundTrip() {
        XCTAssertEqual(NMPBonjour.serviceName(for: 0xa1b2_c3d4), "NeuraMesh-a1b2c3d4")
        // Leading zeros must not be dropped — names key removal events.
        XCTAssertEqual(NMPBonjour.serviceName(for: 0x1), "NeuraMesh-00000001")

        for peerID: UInt32 in [0, 1, 0xa1b2_c3d4, .max] {
            XCTAssertEqual(
                NMPBonjour.peerID(fromServiceName: NMPBonjour.serviceName(for: peerID)),
                peerID)
        }
        XCTAssertNil(NMPBonjour.peerID(fromServiceName: "SomeOtherService-1234"))
        XCTAssertNil(NMPBonjour.peerID(fromServiceName: "NeuraMesh-nothex"))
    }

    func testCapabilityInTXTRecord() {
        // Capabilities → TXT dictionary → NWTXTRecord → dictionary →
        // Capabilities, intact. This is the exact path the publisher and
        // browser use, minus the radio.
        let caps = NMPCapabilities(
            peerID: 0xdead_beef, deviceName: "iPhone 15 Pro", ramMB: 8192,
            computeClass: .high, currentLoadPercent: 15,
            maxInferenceTokensPerSecond: 31.25, modelFormats: ["gguf"])

        let record = NWTXTRecord(caps.txtDictionary())
        let recovered = NMPCapabilities(txtDictionary: record.dictionary)
        XCTAssertEqual(recovered, caps)
    }

    // MARK: Real mDNS (guarded)

    /// Publisher advertises over real mDNS; browser discovers it and
    /// recovers the capabilities from the TXT record. Covers publish,
    /// browse, and TXT integrity in one round trip. Success criterion:
    /// discovery <2 s after publish.
    func testPublisherAdvertisesAndBrowserDiscovers() throws {
        let caps = NMPCapabilities(
            // Random peer ID so concurrent/stale test runs on the same LAN
            // don't collide on service name.
            peerID: UInt32.random(in: 1...UInt32.max),
            deviceName: "unit-test-host", ramMB: 16_384,
            computeClass: .high, currentLoadPercent: 5,
            maxInferenceTokensPerSecond: 12.5, modelFormats: ["gguf", "safetensors"])

        let publisher = NMPBonjourPublisher(capabilities: caps, port: 0)
        let browser = NMPBonjourBrowser()
        defer {
            browser.stop()
            publisher.stop()
        }

        let discovered = expectation(description: "browser sees published service")
        let lock = NSLock()
        var recovered: NMPCapabilities?
        browser.onPeerFound = { found, _ in
            guard found.peerID == caps.peerID else { return } // other mesh members on the LAN
            lock.lock()
            if recovered == nil {
                recovered = found
                discovered.fulfill()
            }
            lock.unlock()
        }

        try publisher.start()
        try browser.start()
        let published = DispatchTime.now()

        // 5 s is generous vs the 2 s target; if mDNS is blocked outright we
        // skip instead of failing (nothing NMP can do about the network).
        guard XCTWaiter.wait(for: [discovered], timeout: 5) == .completed else {
            throw XCTSkip("mDNS publish/browse did not complete — network likely blocks Bonjour")
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - published.uptimeNanoseconds) / 1e9
        print("[NMP] Bonjour discovery latency: \(String(format: "%.3f", elapsed)) s")
        XCTAssertLessThan(elapsed, 2, "discovery must complete <2s after publish")

        lock.lock(); defer { lock.unlock() }
        XCTAssertEqual(recovered, caps, "capabilities must survive the TXT round trip")
    }
}
