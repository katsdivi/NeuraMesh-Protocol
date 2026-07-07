//
//  PacketLossInjector.swift
//  NMP — Phase 6
//
//  Loss simulation for the testing dashboard and benchmark suite: an
//  NMPTransport decorator that drops outbound datagrams at a configured
//  rate, plus a temporary "burst" mode that models AWDL contention spikes
//  (elevated loss for a bounded window, then back to the base rate).
//
//  Deterministic: drops come from splitmix64 seeded at init, so a
//  benchmark run with the same seed sees the same loss pattern —
//  latency comparisons across runs measure the protocol, not the dice.
//
//  Also here: NMPInMemoryTransport, a loopback datagram pair (the Sources
//  twin of the test suite's MockTransport) so the dashboard CLI and
//  benchmarks can assemble full in-process meshes with real handshakes,
//  encryption, FEC, and reliability — no network required.
//

import Foundation

// MARK: - Loss injector

public final class NMPPacketLossInjector: NMPTransport {

    public var onReceive: ((Data) -> Void)? {
        get { inner.onReceive }
        set { inner.onReceive = newValue }
    }
    public var onClosed: ((Error?) -> Void)? {
        get { inner.onClosed }
        set { inner.onClosed = newValue }
    }

    /// Fires (off the caller's lock) whenever a datagram is dropped.
    public var onDatagramDropped: ((Data) -> Void)?

    public private(set) var sentCount = 0
    public private(set) var droppedCount = 0

    private let inner: NMPTransport
    private let lock = NSLock()
    private var baseLossRate: Double = 0
    private var burstRate: Double = 0
    private var burstEndsAt: DispatchTime = .now()
    private var rng: SplitMix64
    /// Blackhole mode: drop everything (simulates a peer vanishing).
    private var blackholed = false

    public init(wrapping inner: NMPTransport, seed: UInt64 = 0x1055_CA5E) {
        self.inner = inner
        self.rng = SplitMix64(seed: seed)
    }

    // MARK: Controls (thread-safe; dashboard calls these off-queue)

    /// Steady-state loss probability in [0, 1].
    public func setLossRate(_ rate: Double) {
        lock.lock(); defer { lock.unlock() }
        baseLossRate = rate.clamped(to: 0...1)
    }

    /// Elevated loss for the next `duration` seconds (AWDL-like burst),
    /// after which the base rate applies again.
    public func setBurstLoss(rate: Double, duration: TimeInterval) {
        lock.lock(); defer { lock.unlock() }
        burstRate = rate.clamped(to: 0...1)
        burstEndsAt = .now() + duration
    }

    /// Simulates abrupt peer death: nothing gets through until `reset()`.
    public func blackhole() {
        lock.lock(); defer { lock.unlock() }
        blackholed = true
    }

    public func reset() {
        lock.lock(); defer { lock.unlock() }
        baseLossRate = 0
        burstRate = 0
        burstEndsAt = .now()
        blackholed = false
        sentCount = 0
        droppedCount = 0
    }

    public var currentLossRate: Double {
        lock.lock(); defer { lock.unlock() }
        return effectiveRateLocked()
    }

    // MARK: NMPTransport

    public func start() { inner.start() }
    public func cancel() { inner.cancel() }

    public func send(_ datagram: Data) {
        lock.lock()
        sentCount += 1
        let rate = effectiveRateLocked()
        let drop = rate >= 1 || (rate > 0 && rng.nextUnitDouble() < rate)
        if drop { droppedCount += 1 }
        lock.unlock()

        if drop {
            onDatagramDropped?(datagram)
            return
        }
        inner.send(datagram)
    }

    private func effectiveRateLocked() -> Double {
        if blackholed { return 1 }
        if DispatchTime.now() < burstEndsAt { return max(burstRate, baseLossRate) }
        return baseLossRate
    }
}

extension SplitMix64 {
    /// Uniform in [0, 1) with 53 bits — plenty for loss dice.
    mutating func nextUnitDouble() -> Double {
        Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }
}

// MARK: - In-memory loopback transport

/// A connected pair of in-process datagram endpoints. Delivery hops onto
/// the receiving side's queue, mirroring how a real UDP socket calls back —
/// the full NMP stack (handshake, GCM, FEC, NACK) runs unmodified on top.
public final class NMPInMemoryTransport: NMPTransport {

    public var onReceive: ((Data) -> Void)?
    public var onClosed: ((Error?) -> Void)?

    public weak var peer: NMPInMemoryTransport?
    private let deliveryQueue: DispatchQueue

    public init(deliveryQueue: DispatchQueue) {
        self.deliveryQueue = deliveryQueue
    }

    public static func pair(label: String = "nmp.loopback")
        -> (NMPInMemoryTransport, NMPInMemoryTransport) {
        let a = NMPInMemoryTransport(deliveryQueue: DispatchQueue(label: "\(label).a"))
        let b = NMPInMemoryTransport(deliveryQueue: DispatchQueue(label: "\(label).b"))
        a.peer = b
        b.peer = a
        return (a, b)
    }

    public func start() {}
    public func cancel() {}

    public func send(_ datagram: Data) {
        guard let peer else { return }
        peer.deliveryQueue.async {
            peer.onReceive?(datagram)
        }
    }
}
