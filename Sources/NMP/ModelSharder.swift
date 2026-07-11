//
//  ModelSharder.swift
//  NMP — Phase 5
//
//  Splits a model's transformer layers into contiguous shards across the
//  mesh, proportional to each peer's speed. Deterministic: the same peer
//  set and measurements always produce the same assignment — the
//  coordinator computes it, but any peer could verify it.
//
//  Speed weighting: measured seconds-per-layer when available (the
//  orchestrator probes peers with a timing run), otherwise the compute
//  class carries a static weight (high 4 : medium 2 : low 1). Goal: all
//  shards finish in roughly equal wall-clock, so the pipeline stalls on
//  no one.
//
//  Pipeline order = (compute class desc, peerID asc) — the same total
//  order the coordinator election uses, so shard 0 (which also embeds
//  the input) tends to land on the coordinator itself.
//

import Foundation

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

public enum NMPModelSharder {

    /// Produces the shard plan.
    ///
    /// - Parameters:
    ///   - layerCount: total transformer layers to distribute.
    ///   - peers: mesh members INCLUDING the coordinator itself.
    ///   - measuredSecondsPerLayer: optional per-peer timing probe results;
    ///     peers absent from the map fall back to class weights.
    ///   - computeShares: Mesh 2.1 allocation caps in (0, 1]; a peer's
    ///     speed score is multiplied by its share, so a device capped at
    ///     50% receives ~half the layers it otherwise would. Applies to
    ///     BOTH measured and class-weight scores (callers passing
    ///     share-scaled measurements should not also pass shares).
    /// - Returns: contiguous, complete, non-overlapping shards in pipeline
    ///   order. Empty iff `peers` is empty or `layerCount` <= 0.
    ///   If there are more peers than layers, the fastest `layerCount`
    ///   peers are used and the rest sit out this plan.
    public static func plan(
        layerCount: Int,
        peers: [NMPCapabilities],
        measuredSecondsPerLayer: [UInt32: Double] = [:],
        computeShares: [UInt32: Double] = [:]
    ) -> [NMPShardPlanEntry] {
        guard layerCount > 0, !peers.isEmpty else { return [] }

        // Deterministic pipeline order; drop surplus peers (slowest first).
        var ordered = peers.sorted { a, b in
            if a.computeClass != b.computeClass { return a.computeClass > b.computeClass }
            return a.peerID < b.peerID
        }
        if ordered.count > layerCount {
            ordered.removeSubrange(layerCount...)
        }

        // Speed score: higher = faster = more layers.
        let scores: [Double] = ordered.map { peer in
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
        let totalScore = scores.reduce(0, +)

        // Largest-remainder apportionment with a floor of 1 layer each.
        var spans = [Int](repeating: 0, count: ordered.count)
        var remainders = [Double](repeating: 0, count: ordered.count)
        for i in ordered.indices {
            let quota = Double(layerCount) * scores[i] / totalScore
            spans[i] = Int(quota)
            remainders[i] = quota - Double(spans[i])
        }
        var assigned = spans.reduce(0, +)

        // Hand out the remainder by largest fraction (ties → pipeline order).
        let byRemainder = ordered.indices.sorted {
            remainders[$0] != remainders[$1] ? remainders[$0] > remainders[$1] : $0 < $1
        }
        var cursor = 0
        while assigned < layerCount {
            spans[byRemainder[cursor % byRemainder.count]] += 1
            assigned += 1
            cursor += 1
        }

        // Floor: every included peer computes at least one layer (take from
        // the largest span, which can spare it — peers ≤ layers here).
        for i in ordered.indices where spans[i] == 0 {
            let donor = spans.indices.max { a, b in
                spans[a] != spans[b] ? spans[a] < spans[b] : a > b
            }!
            spans[donor] -= 1
            spans[i] += 1
        }

        // Materialize contiguous ranges in pipeline order.
        var entries: [NMPShardPlanEntry] = []
        var nextLayer = 0
        for (index, peer) in ordered.enumerated() {
            let span = spans[index]
            entries.append(NMPShardPlanEntry(
                peerID: peer.peerID, shardIndex: index,
                startLayer: nextLayer, endLayer: nextLayer + span))
            nextLayer += span
        }
        assert(nextLayer == layerCount)
        return entries
    }
}
