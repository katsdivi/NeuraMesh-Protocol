//
//  AWDLDetectorTests.swift
//  NMPTests — Phase 3
//
//  AWDL contention detection (loss-rate + latency-spike signals, hysteresis)
//  and traffic shaping (deferral policy). All detector tests drive the pure
//  state machine with a synthetic clock — fully deterministic.
//

import XCTest
@testable import NMP

// MARK: - Detector

final class AWDLDetectorTests: XCTestCase {

    private func makeDetector(_ tune: ((inout NMPAWDLConfig) -> Void)? = nil) -> NMPAWDLDetector {
        var config = NMPAWDLConfig()
        tune?(&config)
        return NMPAWDLDetector(config: config)
    }

    func testCalmLinkStaysUnsuppressed() {
        var d = makeDetector()
        for i in 0..<50 { d.recordSent(at: Double(i) * 0.001) }
        d.recordLosses(1, at: 0.05) // 2% — below the 5% engage threshold
        d.updateState(at: 0.05)
        XCTAssertFalse(d.suppressionActive)
    }

    func testLossRateAboveThresholdEngages() {
        var d = makeDetector()
        for i in 0..<50 { d.recordSent(at: Double(i) * 0.001) }
        d.recordLosses(5, at: 0.05) // 10%
        XCTAssertTrue(d.updateState(at: 0.05))
        XCTAssertTrue(d.suppressionActive)
    }

    func testMinimumSampleGuardPreventsEarlyTrip() {
        // 3 losses out of 5 sends is 60%, but 5 sends is statistical noise.
        var d = makeDetector()
        for i in 0..<5 { d.recordSent(at: Double(i) * 0.001) }
        d.recordLosses(3, at: 0.005)
        d.updateState(at: 0.005)
        XCTAssertFalse(d.suppressionActive)
    }

    func testDisengagesAfterSustainedCalm() {
        var d = makeDetector()
        for i in 0..<50 { d.recordSent(at: Double(i) * 0.001) }
        d.recordLosses(10, at: 0.05)
        d.updateState(at: 0.05)
        XCTAssertTrue(d.suppressionActive)

        // 0.5s later the loss events have left the 100ms window; the first
        // calm update starts the timer, a later one (>200ms on) clears.
        d.updateState(at: 0.5)
        XCTAssertTrue(d.suppressionActive, "calm must be sustained, not instantaneous")
        d.updateState(at: 0.75)
        XCTAssertFalse(d.suppressionActive)
    }

    func testCalmTimerResetsOnRelapse() {
        var d = makeDetector()
        for i in 0..<50 { d.recordSent(at: Double(i) * 0.001) }
        d.recordLosses(10, at: 0.05)
        d.updateState(at: 0.05)
        XCTAssertTrue(d.suppressionActive)

        d.updateState(at: 0.5) // calm starts
        // Relapse before the 200ms calm period completes.
        for i in 0..<30 { d.recordSent(at: 0.55 + Double(i) * 0.001) }
        d.recordLosses(10, at: 0.58)
        d.updateState(at: 0.6)
        XCTAssertTrue(d.suppressionActive)
        d.updateState(at: 0.65) // only 50ms of renewed calm — not enough
        XCTAssertTrue(d.suppressionActive)
    }

    func testLatencySpikeEngages() {
        var d = makeDetector()
        // Establish a 10ms baseline.
        for i in 0..<10 { d.recordLatencySample(0.010, at: Double(i) * 0.001) }
        d.updateState(at: 0.010)
        XCTAssertFalse(d.suppressionActive)
        // Median jumps to 25ms — 2.5× baseline and >5ms above it.
        for i in 0..<10 { d.recordLatencySample(0.025, at: 0.07 + Double(i) * 0.001) }
        d.updateState(at: 0.08)
        XCTAssertTrue(d.suppressionActive)
    }

    func testClockSkewedSamplesStillDetectShift() {
        // One-way samples with heavy negative skew (receiver clock behind):
        // absolute values are meaningless, the SHIFT is the signal.
        var d = makeDetector()
        for i in 0..<10 { d.recordLatencySample(-0.500, at: Double(i) * 0.001) }
        d.updateState(at: 0.010)
        XCTAssertFalse(d.suppressionActive)
        for i in 0..<10 { d.recordLatencySample(-0.480, at: 0.07 + Double(i) * 0.001) }
        d.updateState(at: 0.08) // +20ms shift > 5ms min delta
        XCTAssertTrue(d.suppressionActive)
    }
}

// MARK: - Traffic shaper

final class TrafficShaperTests: XCTestCase {

    func testEverythingPassesWhenInactive() {
        let shaper = NMPTrafficShaper(capacity: 10)
        XCTAssertFalse(shaper.shouldDefer(packetType: .data, flags: [], priority: .normal))
        XCTAssertFalse(shaper.shouldDefer(packetType: .nack, flags: [], priority: .normal))
    }

    func testSuppressionDefersOnlyNormalData() {
        var shaper = NMPTrafficShaper(capacity: 10)
        shaper.suppressionActive = true
        // Deferred: plain data at normal priority.
        XCTAssertTrue(shaper.shouldDefer(packetType: .data, flags: [], priority: .normal))
        // Always sent: control/recovery, critical data, FLUSH-flagged data.
        XCTAssertFalse(shaper.shouldDefer(packetType: .nack, flags: [], priority: .normal))
        XCTAssertFalse(shaper.shouldDefer(packetType: .fecRecovery, flags: [], priority: .normal))
        XCTAssertFalse(shaper.shouldDefer(packetType: .control, flags: [], priority: .normal))
        XCTAssertFalse(shaper.shouldDefer(packetType: .data, flags: [], priority: .critical))
        XCTAssertFalse(shaper.shouldDefer(packetType: .data, flags: [.flush], priority: .normal))
    }

    func testDrainPreservesSubmitOrder() throws {
        var shaper = NMPTrafficShaper(capacity: 10)
        shaper.suppressionActive = true
        for i in 0..<3 {
            try shaper.deferPacket(packetType: .data, flags: [], payload: Data([UInt8(i)]))
        }
        let drained = shaper.drain()
        XCTAssertEqual(drained.map(\.payload), [Data([0]), Data([1]), Data([2])])
        XCTAssertFalse(shaper.hasDeferred)
    }

    func testBufferCapacityEnforced() throws {
        var shaper = NMPTrafficShaper(capacity: 2)
        try shaper.deferPacket(packetType: .data, flags: [], payload: Data([0]))
        try shaper.deferPacket(packetType: .data, flags: [], payload: Data([1]))
        XCTAssertThrowsError(try shaper.deferPacket(packetType: .data, flags: [],
                                                    payload: Data([2]))) {
            XCTAssertEqual($0 as? NMPTrafficShaperError, .deferralBufferFull)
        }
    }
}
