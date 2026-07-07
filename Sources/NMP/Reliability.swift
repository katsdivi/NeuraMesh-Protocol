//
//  Reliability.swift
//  NMP — Phase 2
//
//  NACK-only reliability + retransmission window (spec roadmap item 2).
//
//  Sender side — NMPRetransmitBuffer: the last 64 sealed datagrams are kept,
//  keyed by sequence number, and resent VERBATIM when the peer NACKs them.
//  Verbatim resend is the only sound option: the 20-byte header is the GCM
//  AAD, so flipping the RETRANSMIT flag after sealing would invalidate the
//  tag, and re-sealing under the same nonce with different AAD is the classic
//  GCM "forbidden attack" (two tags under one nonce leak the GHASH key).
//  Re-sealing under a NEW sequence would hide the packet's identity from the
//  receiver's gap tracking. Consequence: the RETRANSMIT flag cannot be set on
//  actual retransmissions — flagged for spec revision in Phase2_Design.md.
//
//  Receiver side — NMPLossTracker: every authenticated sequence number is
//  observed; gaps below the highest seen sequence are recorded as missing.
//  A gap is NACKed after a short reorder delay (so packets that were merely
//  reordered don't trigger spurious retransmits), re-NACKed on an interval,
//  and abandoned after a bounded number of attempts or once it ages out of
//  the 64-packet window (at which point the sender can no longer retransmit
//  it either). The tracker is a pure state machine over an injected clock so
//  tests are deterministic; PeerConnection drives it from a dispatch timer.
//
//  NACK wire format (payload of an encrypted type-0x11 packet):
//
//      u16 count ‖ count × u32 missing sequence numbers   (big-endian)
//
//  A FLUSH-flagged inbound packet short-circuits the reorder delay: all
//  currently-missing sequences become NACKable immediately (the sender set
//  FLUSH on the last packet of a burst, so nothing further is in flight that
//  could fill the gaps naturally).
//

import Foundation

// MARK: - Configuration

public struct NMPReliabilityConfig: Sendable {
    /// Packets a sender keeps for retransmission, and the deepest gap a
    /// receiver will try to recover. Matches NMPReplayWindow.windowSize.
    public static let windowSize: UInt32 = NMPReplayWindow.windowSize

    /// How long a gap may sit unfilled before the first NACK (absorbs plain
    /// UDP reordering without spurious retransmits).
    public var reorderDelay: TimeInterval = 0.008
    /// Interval between repeat NACKs for a still-missing sequence.
    public var nackRetryInterval: TimeInterval = 0.025
    /// NACK transmissions per missing sequence before giving up.
    public var maxNackAttempts: Int = 3

    public init() {}
}

public enum NMPReliabilityError: Error, Equatable, Sendable {
    case malformedNack
}

// MARK: - NACK payload codec

public enum NMPNackCodec {
    /// Sequences per NACK packet is bounded by the window, so the payload
    /// (2 + 64×4 = 258 bytes) always fits a single datagram.
    public static func encode(_ sequences: [UInt32]) -> Data {
        var out = Data(capacity: 2 + sequences.count * 4)
        out.appendBigEndian(UInt16(clamping: sequences.count))
        for seq in sequences.prefix(Int(UInt16.max)) {
            out.appendBigEndian(seq)
        }
        return out
    }

    public static func decode(_ payload: Data) throws -> [UInt32] {
        let bytes = Data(payload) // rebase slice offsets
        guard bytes.count >= 2 else { throw NMPReliabilityError.malformedNack }
        let count = Int(bytes.readBigEndianUInt16(at: 0))
        guard bytes.count == 2 + count * 4 else { throw NMPReliabilityError.malformedNack }
        var sequences: [UInt32] = []
        sequences.reserveCapacity(count)
        for i in 0..<count {
            sequences.append(bytes.readBigEndianUInt32(at: 2 + i * 4))
        }
        return sequences
    }
}

// MARK: - Sender: retransmission window

/// Ring buffer of the last `windowSize` sealed datagrams, keyed by sequence.
/// Sequences from NMPSecureSession.seal are strictly increasing, so slot
/// `seq % windowSize` always holds the newest datagram for that residue.
struct NMPRetransmitBuffer {
    private var slots: [(sequence: UInt32, datagram: Data)?]

    init() {
        slots = Array(repeating: nil, count: Int(NMPReliabilityConfig.windowSize))
    }

    mutating func store(sequence: UInt32, datagram: Data) {
        slots[Int(sequence % NMPReliabilityConfig.windowSize)] = (sequence, datagram)
    }

    /// The exact bytes originally sent for `sequence`, if still buffered.
    func datagram(for sequence: UInt32) -> Data? {
        guard let slot = slots[Int(sequence % NMPReliabilityConfig.windowSize)],
              slot.sequence == sequence else { return nil }
        return slot.datagram
    }
}

// MARK: - Receiver: gap tracking

/// Pure state machine for receiver-side loss detection. All methods take an
/// explicit `now` (seconds, any monotonic origin) so behavior is fully
/// deterministic under test.
struct NMPLossTracker {
    struct Gap {
        var attempts: Int = 0
        var dueAt: TimeInterval
    }

    private let config: NMPReliabilityConfig
    private(set) var highestSeen: UInt32?
    private(set) var missing: [UInt32: Gap] = [:]

    init(config: NMPReliabilityConfig = NMPReliabilityConfig()) {
        self.config = config
    }

    /// Feed every successfully authenticated inbound sequence number.
    /// Returns sequences abandoned because they aged out of the retransmit
    /// window (the sender can no longer resend them).
    @discardableResult
    mutating func observe(sequence: UInt32, at now: TimeInterval) -> [UInt32] {
        guard let highest = highestSeen else {
            highestSeen = sequence
            // Sequences start at 0; anything skipped before the first
            // arrival is already a gap.
            recordGaps(0..<sequence, at: now)
            return []
        }
        if sequence > highest {
            recordGaps((highest + 1)..<sequence, at: now)
            highestSeen = sequence
            return expireAgedOut()
        } else {
            missing.removeValue(forKey: sequence) // late arrival filled a gap
            return []
        }
    }

    /// Sequences whose NACK (first or repeat) is due. Increments attempt
    /// counts and reschedules; exhausted sequences move to `gaveUp`.
    mutating func dueNacks(at now: TimeInterval) -> (nack: [UInt32], gaveUp: [UInt32]) {
        var nack: [UInt32] = []
        var gaveUp: [UInt32] = []
        for (seq, gap) in missing where gap.dueAt <= now {
            if gap.attempts >= config.maxNackAttempts {
                gaveUp.append(seq)
                missing.removeValue(forKey: seq)
            } else {
                nack.append(seq)
                missing[seq] = Gap(attempts: gap.attempts + 1,
                                   dueAt: now + config.nackRetryInterval)
            }
        }
        return (nack.sorted(), gaveUp.sorted())
    }

    /// FLUSH: make every outstanding gap immediately due.
    mutating func expediteAll(at now: TimeInterval) {
        for (seq, gap) in missing where gap.dueAt > now {
            missing[seq] = Gap(attempts: gap.attempts, dueAt: now)
        }
    }

    /// Phase 3: FEC reconstructed this sequence — cancel its pending NACK.
    mutating func markRecovered(_ sequence: UInt32) {
        missing.removeValue(forKey: sequence)
    }

    /// Phase 3: contention inferred — give gaps that haven't been NACKed yet
    /// extra grace so FEC groups get time to complete before we ask the
    /// congested link for retransmits.
    mutating func postponeUnattempted(until dueAt: TimeInterval) {
        for (seq, gap) in missing where gap.attempts == 0 && gap.dueAt < dueAt {
            missing[seq] = Gap(attempts: 0, dueAt: dueAt)
        }
    }

    /// Earliest pending deadline, for timer (re)arming. nil = nothing pending.
    var nextDeadline: TimeInterval? {
        missing.values.map(\.dueAt).min()
    }

    private mutating func recordGaps(_ range: Range<UInt32>, at now: TimeInterval) {
        guard !range.isEmpty else { return }
        // A gap deeper than the window is unrecoverable (the sender's
        // retransmit buffer has already dropped it) — track only the tail.
        let start = range.count > Int(NMPReliabilityConfig.windowSize)
            ? range.upperBound - NMPReliabilityConfig.windowSize
            : range.lowerBound
        for seq in start..<range.upperBound {
            missing[seq] = Gap(dueAt: now + config.reorderDelay)
        }
    }

    /// Drop gaps that fell out of the retransmit window: the sender can no
    /// longer resend them, so NACKing is pointless. Returns what was dropped.
    private mutating func expireAgedOut() -> [UInt32] {
        guard let highest = highestSeen,
              highest >= NMPReliabilityConfig.windowSize else { return [] }
        let oldest = highest - NMPReliabilityConfig.windowSize + 1
        let aged = missing.keys.filter { $0 < oldest }
        for seq in aged {
            missing.removeValue(forKey: seq)
        }
        return aged.sorted()
    }
}
