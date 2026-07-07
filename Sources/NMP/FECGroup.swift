//
//  FECGroup.swift
//  NMP — Phase 3
//
//  FEC group state management, both directions.
//
//  Sender (NMPFECGroupBuilder): sealed DATA packets accumulate into a group;
//  when the group closes — groupSize members reached, or a FLUSH-flagged
//  packet ends the burst early — the builder emits the parity packet payload.
//  The caller must set FEC_GROUP_END on the closing packet BEFORE sealing
//  (the header is GCM AAD), which is why `willCloseGroup` previews the
//  decision separately from `add`.
//
//  Receiver (NMPFECGroupReceiver): data packets are delivered to the
//  application IMMEDIATELY on arrival — FEC is recovery-only and adds zero
//  latency to the no-loss path. (The build prompt's sketch buffered whole
//  groups and delivered on completion; rejected — it would tax the 98%+ of
//  groups that lose nothing. See Phase3_Design.md.) The receiver caches
//  recent payloads by sequence; when a parity descriptor's group has exactly
//  one member missing, the member is reconstructed and handed back for
//  injection. Groups missing 2+ members wait (a NACK retransmit may fill
//  them) and expire after `pendingTimeout` — the Phase 2 NACK path is the
//  fallback either way and needs no help from this layer.
//

import Foundation

// MARK: - Configuration

public struct NMPFECConfig: Sendable {
    /// FEC on/off for this connection. When off, no parity packets are sent
    /// and inbound parity packets are ignored.
    public var enabled = true
    /// Data packets per parity packet. 4 = 25% overhead, recovers any single
    /// loss per group. See Phase3_Design.md for the tradeoff discussion.
    public var groupSize = 4
    /// How long an unreconstructable pending group is kept before being
    /// discarded (a NACK retransmit may still complete it within this time).
    public var pendingTimeout: TimeInterval = 0.05

    public init() {}
}

// MARK: - Sender

struct NMPFECGroupBuilder {
    private let groupSize: Int
    private var members: [(sequence: UInt32, payload: Data)] = []

    init(groupSize: Int) {
        self.groupSize = Swift.max(2, Swift.min(groupSize, NMPFECCodec.maxGroupSize))
    }

    /// True if the packet about to be added will close the group — the
    /// caller sets FEC_GROUP_END on it before sealing.
    func willCloseGroup(flush: Bool) -> Bool {
        flush || members.count == groupSize - 1
    }

    /// Registers a sealed data packet. Returns the parity packet payload
    /// when this packet closed the group, nil otherwise.
    mutating func add(sequence: UInt32, payload: Data, closesGroup: Bool) -> Data? {
        members.append((sequence, payload))
        guard closesGroup else { return nil }
        let parity = NMPFECCodec.computeParity(members.map(\.payload))
        let out = NMPFECCodec.encodeParityPayload(
            sequences: members.map(\.sequence),
            payloadLengths: members.map(\.payload.count),
            parity: parity)
        members.removeAll(keepingCapacity: true)
        return out
    }
}

// MARK: - Receiver

struct NMPFECGroupReceiver {

    struct Recovery: Equatable {
        let sequence: UInt32
        let payload: Data
        let groupID: UInt32
    }

    /// Cached payloads reach back 2× the replay window; a parity packet can
    /// only reference sequences its own arrival proves are recent.
    private static let cacheDepth: UInt32 = NMPReplayWindow.windowSize * 2

    private let pendingTimeout: TimeInterval
    private var payloadCache: [UInt32: Data] = [:]
    private var highestSeen: UInt32 = 0
    private var pending: [UInt32: (descriptor: NMPFECCodec.ParityDescriptor,
                                   arrivedAt: TimeInterval)] = [:]

    init(pendingTimeout: TimeInterval) {
        self.pendingTimeout = pendingTimeout
    }

    /// Feed every decrypted DATA payload. Returns any reconstructions this
    /// arrival unlocked (it may complete a previously-pending group).
    mutating func observeData(sequence: UInt32, payload: Data,
                              at now: TimeInterval) -> [Recovery] {
        payloadCache[sequence] = payload
        if sequence > highestSeen { highestSeen = sequence }
        pruneCache()
        return attemptPending(at: now)
    }

    /// Feed every decrypted FEC_RECOVERY payload. Throws on malformed input.
    mutating func observeParity(_ payload: Data,
                                at now: TimeInterval) throws -> [Recovery] {
        let descriptor = try NMPFECCodec.decodeParityPayload(payload)
        pending[descriptor.groupID] = (descriptor, now)
        return attemptPending(at: now)
    }

    /// Group IDs discarded because they stayed unreconstructable past the
    /// timeout (2+ members missing and no retransmit arrived). Diagnostics
    /// only — the NACK path owns recovery from here.
    mutating func expirePending(at now: TimeInterval) -> [UInt32] {
        let expired = pending.filter { now - $0.value.arrivedAt > pendingTimeout }.map(\.key)
        for id in expired { pending.removeValue(forKey: id) }
        return expired.sorted()
    }

    // MARK: Reconstruction

    private mutating func attemptPending(at now: TimeInterval) -> [Recovery] {
        var recoveries: [Recovery] = []
        for (groupID, entry) in pending {
            let descriptor = entry.descriptor
            let missing = descriptor.sequences.enumerated()
                .filter { payloadCache[$0.element] == nil }

            if missing.isEmpty {
                // Every member arrived on its own; parity was insurance.
                pending.removeValue(forKey: groupID)
            } else if missing.count == 1, let miss = missing.first {
                let surviving = descriptor.sequences.filter { payloadCache[$0] != nil }
                    .compactMap { payloadCache[$0] }
                guard let payload = try? NMPFECCodec.reconstruct(
                    parity: descriptor.parity,
                    surviving: surviving,
                    missingLength: descriptor.lengths[miss.offset]) else {
                    pending.removeValue(forKey: groupID)
                    continue
                }
                payloadCache[miss.element] = payload
                recoveries.append(Recovery(sequence: miss.element,
                                           payload: payload,
                                           groupID: groupID))
                pending.removeValue(forKey: groupID)
            }
            // missing.count >= 2: keep pending; a NACK retransmit may still
            // complete the group before expirePending() drops it.
        }
        _ = expirePending(at: now)
        return recoveries.sorted { $0.sequence < $1.sequence }
    }

    private mutating func pruneCache() {
        guard highestSeen >= Self.cacheDepth else { return }
        let oldest = highestSeen - Self.cacheDepth + 1
        for seq in payloadCache.keys where seq < oldest {
            payloadCache.removeValue(forKey: seq)
        }
    }
}
