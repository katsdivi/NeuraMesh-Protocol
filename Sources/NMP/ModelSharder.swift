//
//  ModelSharder.swift
//  NMP — Phase 5, capacity-aware since Mesh 2.8
//
//  Splits a model's transformer layers into contiguous shards across the
//  mesh. Deterministic: the same peer set, measurements, and capacities
//  always produce the same assignment — the coordinator computes it, but
//  any peer could verify it.
//
//  Two things drive the split:
//
//    1. CAPACITY (hard ceiling). A device can only hold as many layers as
//       its RAM allows — `layerCapacities[peerID]`. This is what makes a
//       model too big for any single device runnable at all: layers MUST
//       spill onto other devices, speed be damned. Absent/`.max` capacity
//       means "unbounded" (e.g. the weightless reference engine), and the
//       sharder then behaves exactly as the Phase 5 speed-only splitter —
//       no capacity cost when nothing is binding.
//
//    2. OBJECTIVE (how to split within the ceilings):
//       - .capacityThenSpeed (default): use the fewest devices needed to
//         hold the model, then BALANCE layers across them by speed so no
//         stage idles waiting on another — steady pipelined throughput,
//         the whole mesh pulling its weight.
//       - .speed: minimize single-token latency — PACK the fastest device
//         to its ceiling, spill only the remainder onto the next fastest.
//         A device that isn't needed for capacity and would only add a
//         network hop gets 0 layers (with a reason). This is the "I want
//         raw speed" mode; when the model fits on one device, that device
//         does all of it and everyone else stands by.
//
//  Pipeline order = (compute class desc, peerID asc) — the same total
//  order the coordinator election uses, so shard 0 (which also embeds the
//  input) tends to land on the coordinator itself.
//

import Foundation

// MARK: - Plan entry

public struct NMPShardPlanEntry: Equatable, Sendable {
    public let peerID: UInt32
    /// Position in the pipeline (0-based; activations flow 0 → N-1).
    public let shardIndex: Int
    public let startLayer: Int
    /// Exclusive.
    public let endLayer: Int

    public var layerSpan: Int { endLayer - startLayer }

    public init(peerID: UInt32, shardIndex: Int, startLayer: Int, endLayer: Int) {
        self.peerID = peerID
        self.shardIndex = shardIndex
        self.startLayer = startLayer
        self.endLayer = endLayer
    }

    /// The whole model as one shard on `peerID` — the only legal llama
    /// plan (llama.cpp executes full models; see LlamaEngine.swift).
    public static func fullRange(peerID: UInt32, layerCount: Int) -> NMPShardPlanEntry {
        NMPShardPlanEntry(peerID: peerID, shardIndex: 0,
                          startLayer: 0, endLayer: layerCount)
    }
}

// MARK: - Objective + detailed plan

public enum NMPShardingObjective: String, Sendable, Codable, CaseIterable {
    /// Spread across the fewest devices that can hold the model, balanced
    /// by speed. Uses the mesh; makes oversized models runnable. Default.
    case capacityThenSpeed
    /// Minimize per-token latency: pack the fastest device, spill the rest.
    /// Excludes devices that only add a network hop.
    case speed

    public var label: String {
        switch self {
        case .capacityThenSpeed: return "Capacity + Speed"
        case .speed: return "Pure Speed"
        }
    }
}

/// A device that received 0 layers, and the measured reason why — so the
/// UI can say "0 shards on this device" and mean something specific.
public struct NMPShardExclusion: Equatable, Sendable {
    public let peerID: UInt32
    public let reason: String
    public init(peerID: UInt32, reason: String) {
        self.peerID = peerID
        self.reason = reason
    }
}

/// A full plan: the peers that hold layers, plus the ones that hold none
/// and why, plus any layers that fit nowhere (the mesh is too small for
/// the model — the honest failure the UI must surface).
public struct NMPShardPlan: Equatable, Sendable {
    public let entries: [NMPShardPlanEntry]
    public let exclusions: [NMPShardExclusion]
    public let objective: NMPShardingObjective
    /// Layers that fit on no device (sum of capacities < layerCount).
    /// > 0 means the model cannot run on this mesh as-is.
    public let capacityShortfall: Int

    public init(entries: [NMPShardPlanEntry], exclusions: [NMPShardExclusion],
                objective: NMPShardingObjective, capacityShortfall: Int) {
        self.entries = entries
        self.exclusions = exclusions
        self.objective = objective
        self.capacityShortfall = capacityShortfall
    }
}

// MARK: - Sharder

public enum NMPModelSharder {

    /// Layers a device with `ramMB` can hold, given each layer's weight
    /// footprint. `headroom` reserves memory for the OS, the KV cache, and
    /// live activations (0.6 = use ~60% of RAM for weights). `bytesPerLayer`
    /// <= 0 means the footprint is unknown (e.g. the weightless reference
    /// engine) → unbounded capacity, so nothing is capacity-bound.
    public static func layerCapacity(
        ramMB: UInt32, bytesPerLayer: Int, headroom: Double = 0.6
    ) -> Int {
        guard bytesPerLayer > 0 else { return Int.max }
        let usableBytes = Double(ramMB) * 1_048_576 * max(0.05, min(1, headroom))
        return max(0, Int(usableBytes / Double(bytesPerLayer)))
    }

    // MARK: - Latency-optimal plan (Phase B: network-aware balancing)

    /// Minimizes MEASURED per-token wall-clock latency, not per-stage compute
    /// balance. Autoregressive decode is sequential (token N+1 needs token N),
    /// so stages don't pipeline-overlap and the real cost is the SUM:
    ///
    ///   latency = Σ(layers · computeRate[device]) + Σ(roundTrip[peer used])
    ///
    /// The coordinator is free (local, no round trip). Each peer that holds
    /// ≥1 layer adds its fixed round trip once, so a peer is used only when its
    /// compute saving outweighs that hop — or when capacity forces it (a model
    /// too big for the coordinator alone). This is why a fast-but-far phone
    /// gets no layers for a small model (Mac-only wins) yet a full share for a
    /// model that won't fit on one device.
    ///
    /// - Parameters:
    ///   - coordinatorPeerID: the local, round-trip-free device (always usable).
    ///   - computeSecondsPerLayer / roundTripSeconds: from the orchestrator's
    ///     measurements; missing entries fall back to class rate / a default
    ///     hop so a cold mesh is conservative, not reckless. "No measurement
    ///     yet" is UNKNOWN, never 0: a remote entry below
    ///     `minimumMeasuredRoundTrip` is treated as unmeasured too, because a
    ///     network hop is never free — churn/restarts once left zeroed
    ///     round-trips that this planner consumed as real 0 ms and answered
    ///     "put ALL layers on the phone" (BUG-4).
    public static func planByLatency(
        layerCount: Int,
        coordinatorPeerID: UInt32,
        peers: [NMPCapabilities],
        computeSecondsPerLayer: [UInt32: Double] = [:],
        roundTripSeconds: [UInt32: Double] = [:],
        layerCapacities: [UInt32: Int] = [:],
        defaultRoundTrip: TimeInterval = 0.03,
        minimumMeasuredRoundTrip: TimeInterval = 0.001
    ) -> NMPShardPlan {
        guard layerCount > 0, !peers.isEmpty else {
            return NMPShardPlan(entries: [], exclusions: [],
                                objective: .speed, capacityShortfall: layerCount)
        }
        func rate(_ p: NMPCapabilities) -> Double {
            if let m = computeSecondsPerLayer[p.peerID], m > 0 { return m }
            switch p.computeClass {              // proxy s/layer before measurement
            case .high: return 0.003
            case .medium: return 0.006
            case .low: return 0.012
            }
        }
        func rtt(_ p: NMPCapabilities) -> Double {
            guard p.peerID != coordinatorPeerID else { return 0 }
            // Only a plausible real measurement counts; anything below the
            // floor (or absent) costs the conservative default hop until an
            // honest measurement exists.
            if let measured = roundTripSeconds[p.peerID],
               measured >= minimumMeasuredRoundTrip {
                return measured
            }
            return defaultRoundTrip
        }
        func cap(_ p: NMPCapabilities) -> Int {
            min(layerCount, max(0, layerCapacities[p.peerID] ?? Int.max))
        }

        guard let coordinator = peers.first(where: { $0.peerID == coordinatorPeerID })
                ?? peers.first else {
            return NMPShardPlan(entries: [], exclusions: [],
                                objective: .speed, capacityShortfall: layerCount)
        }
        let others = peers.filter { $0.peerID != coordinator.peerID }

        // Cost of a participation set: place all layers cheapest-rate-first
        // under capacity, charge each peer that ends up holding ≥1 layer its
        // round trip. Returns nil when the set can't hold the whole model.
        func evaluate(_ participants: [NMPCapabilities])
            -> (cost: Double, counts: [UInt32: Int])? {
            guard participants.reduce(0, { $0 + cap($1) }) >= layerCount else { return nil }
            var remaining = layerCount
            var counts: [UInt32: Int] = [:]
            for p in participants.sorted(by: { rate($0) < rate($1) }) {
                guard remaining > 0 else { break }
                let take = min(cap(p), remaining)
                if take > 0 { counts[p.peerID] = take; remaining -= take }
            }
            guard remaining == 0 else { return nil }
            var cost = 0.0
            for p in participants {
                let n = counts[p.peerID] ?? 0
                cost += Double(n) * rate(p) + (n > 0 ? rtt(p) : 0)
            }
            return (cost, counts)
        }

        // Enumerate which peers may hold layers (coordinator always in). Real
        // meshes are small; cap the search and fall back to "all peers" beyond.
        var best: (cost: Double, counts: [UInt32: Int])?
        if others.count <= 12 {
            for mask in 0..<(1 << others.count) {
                var set = [coordinator]
                for (i, p) in others.enumerated() where (mask & (1 << i)) != 0 {
                    set.append(p)
                }
                if let result = evaluate(set),
                   best == nil || result.cost < best!.cost {
                    best = result
                }
            }
        } else {
            best = evaluate(peers)
        }

        guard let winner = best else {
            // Nowhere near enough capacity even using everything.
            let placed = evaluate(peers)?.counts ?? [:]
            return buildPlan(layerCount: layerCount, coordinator: coordinator,
                             others: others, counts: placed,
                             capacityShortfall: layerCount
                                - placed.values.reduce(0, +), rttFor: rtt)
        }
        return buildPlan(layerCount: layerCount, coordinator: coordinator,
                         others: others, counts: winner.counts,
                         capacityShortfall: 0, rttFor: rtt)
    }

    /// Minimizes the PEAK per-device memory load: distributes layers in
    /// proportion to each device's capacity so every device ends at the same
    /// % full. This is the "no device fills up" plan — use it when devices are
    /// storage/RAM-constrained. Falls back to an even split when capacities
    /// are unknown (unbounded).
    public static func planByCapacity(
        layerCount: Int,
        coordinatorPeerID: UInt32,
        peers: [NMPCapabilities],
        layerCapacities: [UInt32: Int] = [:]
    ) -> NMPShardPlan {
        guard layerCount > 0, !peers.isEmpty else {
            return NMPShardPlan(entries: [], exclusions: [],
                                objective: .capacityThenSpeed,
                                capacityShortfall: layerCount)
        }
        let coordinator = peers.first { $0.peerID == coordinatorPeerID } ?? peers[0]
        let others = peers.filter { $0.peerID != coordinator.peerID }
        let ordered = [coordinator] + others

        // Raw capacity drives the WEIGHTING (relative RAM); the placement
        // limit clamps to the layer count. Clamping before weighting would
        // flatten unequal-but-both-ample devices to an even split.
        func rawCap(_ p: NMPCapabilities) -> Int {
            max(0, layerCapacities[p.peerID] ?? Int.max)
        }
        func cap(_ p: NMPCapabilities) -> Int { min(layerCount, rawCap(p)) }
        // Weight = capacity; unbounded (Int.max) → even weighting so a
        // weightless/unknown mesh still splits sensibly.
        let bounded = ordered.allSatisfy { rawCap($0) < Int.max }
        let weights: [Double] = ordered.map {
            bounded ? Double(rawCap($0)) : 1.0
        }
        let totalWeight = max(weights.reduce(0, +), 1)

        // Ideal (fractional) share per device, then largest-remainder rounding
        // to whole layers, clamped to capacity.
        var counts: [UInt32: Int] = [:]
        var remainders: [(peerID: UInt32, frac: Double, capLeft: Int)] = []
        var placed = 0
        for (i, device) in ordered.enumerated() {
            let ideal = Double(layerCount) * weights[i] / totalWeight
            let n = min(Int(ideal.rounded(.down)), cap(device))
            counts[device.peerID] = n
            placed += n
            remainders.append((device.peerID, ideal - Double(n),
                               cap(device) - n))
        }
        // Hand out the leftover layers to the largest remainders that still
        // have capacity.
        var leftover = layerCount - placed
        for r in remainders.sorted(by: { $0.frac > $1.frac }) where leftover > 0 {
            guard r.capLeft > 0 else { continue }
            let give = min(r.capLeft, leftover)
            counts[r.peerID, default: 0] += give
            leftover -= give
        }
        return buildPlan(layerCount: layerCount, coordinator: coordinator,
                         others: others, counts: counts,
                         capacityShortfall: max(0, leftover),
                         rttFor: { _ in 0 })
    }

    /// Lays out decided per-device layer counts as contiguous pipeline
    /// ranges — coordinator first (it owns the embedding, so it needs no
    /// inbound hop), then peers in speed order — and records who got nothing.
    private static func buildPlan(
        layerCount: Int, coordinator: NMPCapabilities, others: [NMPCapabilities],
        counts: [UInt32: Int], capacityShortfall: Int,
        rttFor: (NMPCapabilities) -> Double
    ) -> NMPShardPlan {
        var entries: [NMPShardPlanEntry] = []
        var exclusions: [NMPShardExclusion] = []
        var cursor = 0
        var shardIndex = 0
        let ordered = [coordinator] + others  // coordinator leads the pipeline
        for device in ordered {
            let n = counts[device.peerID] ?? 0
            if n > 0 {
                entries.append(NMPShardPlanEntry(
                    peerID: device.peerID, shardIndex: shardIndex,
                    startLayer: cursor, endLayer: cursor + n))
                cursor += n
                shardIndex += 1
            } else if device.peerID != coordinator.peerID {
                exclusions.append(NMPShardExclusion(
                    peerID: device.peerID,
                    reason: String(format:
                        "excluded: its %.0f ms round trip outweighs the compute "
                        + "it would offload (Mac-only is faster here)",
                        rttFor(device) * 1000)))
            }
        }
        return NMPShardPlan(entries: entries, exclusions: exclusions,
                            objective: .speed, capacityShortfall: capacityShortfall)
    }

    /// Backward-compatible Phase 5 entry point: speed-weighted split with a
    /// 1-layer floor, no capacity ceilings. Equivalent to `planDetailed`
    /// with unbounded capacities and `.capacityThenSpeed`.
    public static func plan(
        layerCount: Int,
        peers: [NMPCapabilities],
        measuredSecondsPerLayer: [UInt32: Double] = [:],
        computeShares: [UInt32: Double] = [:]
    ) -> [NMPShardPlanEntry] {
        planDetailed(
            layerCount: layerCount, peers: peers,
            measuredSecondsPerLayer: measuredSecondsPerLayer,
            computeShares: computeShares).entries
    }

    /// Produces the full plan (entries + exclusions + shortfall).
    ///
    /// - Parameters:
    ///   - layerCount: total transformer layers to distribute.
    ///   - peers: mesh members INCLUDING the coordinator.
    ///   - measuredSecondsPerLayer: per-peer timing probe results; peers
    ///     absent from the map fall back to compute-class weights.
    ///   - computeShares: Mesh 2.1 allocation caps in (0, 1]; a peer's
    ///     speed score is multiplied by its share.
    ///   - layerCapacities: max layers each peer can hold (RAM ceiling).
    ///     Absent = unbounded. See `layerCapacity(ramMB:bytesPerLayer:)`.
    ///   - objective: how to split within the capacity ceilings.
    public static func planDetailed(
        layerCount: Int,
        peers: [NMPCapabilities],
        measuredSecondsPerLayer: [UInt32: Double] = [:],
        computeShares: [UInt32: Double] = [:],
        layerCapacities: [UInt32: Int] = [:],
        objective: NMPShardingObjective = .capacityThenSpeed
    ) -> NMPShardPlan {
        guard layerCount > 0, !peers.isEmpty else {
            return NMPShardPlan(entries: [], exclusions: [],
                                objective: objective, capacityShortfall: layerCount)
        }

        // Deterministic pipeline order (also the tie-break for equal speed).
        let ordered = peers.sorted { a, b in
            if a.computeClass != b.computeClass { return a.computeClass > b.computeClass }
            return a.peerID < b.peerID
        }

        // Per-peer speed score (higher = faster = deserves more layers) and
        // capacity ceiling.
        func score(_ peer: NMPCapabilities) -> Double {
            let share = min(1.0, max(0.05, computeShares[peer.peerID] ?? 1.0))
            if let measured = measuredSecondsPerLayer[peer.peerID], measured > 0 {
                return share / measured
            }
            switch peer.computeClass {
            case .high: return 4 * share
            case .medium: return 2 * share
            case .low: return 1 * share
            }
        }
        func cap(_ peer: NMPCapabilities) -> Int {
            max(0, layerCapacities[peer.peerID] ?? Int.max)
        }
        func rateMs(_ peer: NMPCapabilities) -> String {
            if let m = measuredSecondsPerLayer[peer.peerID], m > 0 {
                return String(format: "%.1f ms/layer", m * 1000)
            }
            return "\(peer.computeClass.label)-class"
        }

        let totalCapacity = ordered.reduce(0) { $0 + min(cap($1), layerCount) }
        let shortfall = max(0, layerCount - totalCapacity)

        // Fastest-first ordering (speed score, pipeline order as tie-break)
        // for both the pack objective and the surplus-drop tie-break.
        let bySpeed = ordered.enumerated().sorted { l, r in
            let sl = score(l.element), sr = score(r.element)
            if sl != sr { return sl > sr }
            return l.offset < r.offset
        }.map(\.element)

        // Participation set differs by objective:
        //   .capacityThenSpeed — use the whole mesh (every device with
        //     capacity > 0), so work is SPREAD; ceilings just clamp the
        //     shares. With unbounded capacity this is the classic Phase 5
        //     speed-weighted split, unchanged.
        //   .speed — the FEWEST fastest devices that can hold the model;
        //     a device that only adds a Wi-Fi hop is left out.
        var participants: [NMPCapabilities]
        var activeCountForReason: Int
        if objective == .speed {
            participants = []
            var held = 0
            for peer in bySpeed where cap(peer) > 0 {
                participants.append(peer)
                held += min(cap(peer), layerCount)
                if held >= layerCount { break }
            }
            if held < layerCount { participants = ordered.filter { cap($0) > 0 } }
            activeCountForReason = participants.count
        } else {
            participants = ordered.filter { cap($0) > 0 }
            // More devices than layers: the slowest sit this plan out.
            if participants.count > layerCount {
                let keep = Set(bySpeed.filter { cap($0) > 0 }
                    .prefix(layerCount).map(\.peerID))
                participants = participants.filter { keep.contains($0.peerID) }
            }
            activeCountForReason = participants.count
        }

        let participantIDs = Set(participants.map(\.peerID))
        var layerByPeer: [UInt32: Int] = [:]

        if objective == .speed {
            // Pack fastest-first up to each ceiling — lowest per-token
            // latency (fast device does as much as it can hold).
            var remaining = min(layerCount, totalCapacity)
            for peer in bySpeed where participantIDs.contains(peer.peerID) {
                let take = min(cap(peer), remaining)
                if take > 0 { layerByPeer[peer.peerID] = take }
                remaining -= take
                if remaining <= 0 { break }
            }
        } else {
            // Balance across the whole mesh by speed, clamped to ceilings.
            layerByPeer = balancedSplit(
                layerCount: min(layerCount, totalCapacity),
                participants: participants,
                score: score, cap: cap)
        }

        // Materialize contiguous ranges in PIPELINE order; collect the
        // 0-layer devices with a specific reason.
        var entries: [NMPShardPlanEntry] = []
        var exclusions: [NMPShardExclusion] = []
        var nextLayer = 0
        var shardIndex = 0
        for peer in ordered {
            let span = layerByPeer[peer.peerID] ?? 0
            if span > 0 {
                entries.append(NMPShardPlanEntry(
                    peerID: peer.peerID, shardIndex: shardIndex,
                    startLayer: nextLayer, endLayer: nextLayer + span))
                nextLayer += span
                shardIndex += 1
            } else {
                exclusions.append(NMPShardExclusion(
                    peerID: peer.peerID,
                    reason: exclusionReason(
                        peer: peer, objective: objective,
                        cap: cap(peer), rate: rateMs(peer),
                        activeDevices: max(1, activeCountForReason))))
            }
        }

        return NMPShardPlan(
            entries: entries, exclusions: exclusions,
            objective: objective, capacityShortfall: shortfall)
    }

    // MARK: Internals

    /// Largest-remainder apportionment weighted by speed, with a floor of 1
    /// per participant and a hard per-peer capacity ceiling. Overflow past
    /// a ceiling is redistributed to participants that still have room.
    private static func balancedSplit(
        layerCount: Int,
        participants: [NMPCapabilities],
        score: (NMPCapabilities) -> Double,
        cap: (NMPCapabilities) -> Int
    ) -> [UInt32: Int] {
        guard layerCount > 0, !participants.isEmpty else { return [:] }
        var spans = [Int](repeating: 0, count: participants.count)
        // Redistribute in rounds: quota by score among peers with room,
        // clamp to ceilings, repeat with the leftover until placed.
        var remaining = layerCount
        var hasRoom = Array(participants.indices)
        while remaining > 0 && !hasRoom.isEmpty {
            let roundTotal = hasRoom.reduce(0.0) { $0 + score(participants[$1]) }
            guard roundTotal > 0 else { break }
            var quotas = [(idx: Int, whole: Int, frac: Double)]()
            var placedThisRound = 0
            for i in hasRoom {
                let ideal = Double(remaining) * score(participants[i]) / roundTotal
                let room = cap(participants[i]) - spans[i]
                let whole = min(room, Int(ideal))
                quotas.append((i, whole, ideal - Double(Int(ideal))))
                placedThisRound += whole
            }
            for q in quotas { spans[q.idx] += q.whole }
            remaining -= placedThisRound
            // Hand out the remainder by largest fraction, respecting room.
            let byFrac = quotas.sorted { $0.frac > $1.frac || ($0.frac == $1.frac && $0.idx < $1.idx) }
            var progressed = placedThisRound > 0
            for q in byFrac where remaining > 0 {
                if spans[q.idx] < cap(participants[q.idx]) {
                    spans[q.idx] += 1; remaining -= 1; progressed = true
                }
            }
            hasRoom = participants.indices.filter { spans[$0] < cap(participants[$0]) }
            if !progressed { break }
        }
        // Floor: every participant computes at least one layer (take from
        // the largest span that can spare it).
        for i in participants.indices where spans[i] == 0 {
            if let donor = participants.indices
                .filter({ spans[$0] > 1 })
                .max(by: { spans[$0] != spans[$1] ? spans[$0] < spans[$1] : $0 > $1 }) {
                spans[donor] -= 1; spans[i] += 1
            }
        }
        var out: [UInt32: Int] = [:]
        for (i, peer) in participants.enumerated() { out[peer.peerID] = spans[i] }
        return out
    }

    private static func exclusionReason(
        peer: NMPCapabilities, objective: NMPShardingObjective,
        cap: Int, rate: String, activeDevices: Int
    ) -> String {
        if cap == 0 {
            return "0 shards: this device can't hold even one layer of the "
                + "model (out of memory) — it stands by as a hot spare."
        }
        switch objective {
        case .speed:
            return "0 shards: Pure Speed mode packs the fastest device(s); "
                + "routing around this one (\(rate)) avoids a Wi-Fi hop and "
                + "is faster than offloading work to it. Switch to "
                + "Capacity + Speed to use it."
        case .capacityThenSpeed:
            return "0 shards: the model fits on \(activeDevices) faster "
                + "device(s), so distributing to this one would add a Wi-Fi "
                + "round trip without a capacity need."
        }
    }
}
