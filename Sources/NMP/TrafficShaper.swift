//
//  TrafficShaper.swift
//  NMP — Phase 3
//
//  Defers non-critical traffic while AWDL suppression is active, per spec:
//  only FLUSH-flagged packets, control/recovery traffic (NACK, FEC parity),
//  and critical-priority data keep flowing; normal DATA payloads buffer
//  locally until suppression clears or `NMPAWDLConfig.maxDeferDelay` forces
//  them out.
//
//  Deferral holds PLAINTEXT payloads, not sealed datagrams: sealing assigns
//  the sequence number, and deferring a sealed packet while later packets
//  ship would punch permanent holes in the receiver's gap tracking. Sealing
//  at actual send time keeps sequence order identical to wire order.
//
//  NACK-triggered retransmits never pass through here — they resend cached
//  ciphertext directly from the retransmit buffer (always allowed).
//

import Foundation

// MARK: - Priority

public enum NMPSendPriority: Sendable {
    /// Never deferred (shard assignments, control plane).
    case critical
    /// Deferred during AWDL suppression (model activations).
    case normal
}

public enum NMPTrafficShaperError: Error, Equatable, Sendable {
    /// The deferral buffer is full; the caller must back off.
    case deferralBufferFull
}

// MARK: - Shaper

struct NMPTrafficShaper {

    struct DeferredPacket {
        let packetType: NMPPacketType
        let flags: NMPFlags
        let payload: Data
    }

    var suppressionActive = false
    private let capacity: Int
    private var buffer: [DeferredPacket] = []

    init(capacity: Int) {
        self.capacity = capacity
    }

    var hasDeferred: Bool { !buffer.isEmpty }
    var deferredCount: Int { buffer.count }

    /// True if this packet must be deferred rather than sent now.
    func shouldDefer(packetType: NMPPacketType, flags: NMPFlags,
                     priority: NMPSendPriority) -> Bool {
        guard suppressionActive else { return false }
        guard priority == .normal else { return false }         // critical always goes
        guard packetType == .data else { return false }         // control/recovery always go
        return !flags.contains(.flush)                          // FLUSH always goes
    }

    mutating func deferPacket(packetType: NMPPacketType, flags: NMPFlags,
                              payload: Data) throws {
        guard buffer.count < capacity else {
            throw NMPTrafficShaperError.deferralBufferFull
        }
        buffer.append(DeferredPacket(packetType: packetType, flags: flags, payload: payload))
    }

    /// Removes and returns everything deferred, in original submit order.
    mutating func drain() -> [DeferredPacket] {
        let out = buffer
        buffer.removeAll(keepingCapacity: true)
        return out
    }
}
