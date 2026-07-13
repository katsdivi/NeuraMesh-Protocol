//
//  ShardDeviceReport.swift
//  NMP — Phase 10 (TRUE cross-device layer sharding)
//
//  The surfacing contract for a real sharded plan: given the shard plan and
//  each peer's MEASURED loaded-weight bytes, produce one honest row per
//  device — which layer range it holds and how many MB it actually loaded.
//  The dashboard/iOS panels render these directly, so "no single device holds
//  the whole model" is shown as measured fact, not a claim.
//
//  Pure Swift (no ggml, no llama) — unit-tested and usable anywhere.
//

import Foundation

/// One device's slice of a sharded model.
public struct NMPShardDeviceInfo: Equatable, Sendable {
    public let peerID: UInt32
    public let shardIndex: Int
    public let startLayer: Int
    public let endLayer: Int
    /// Weight bytes this peer actually loaded (0 = unknown / not yet loaded).
    public let loadedBytes: Int
    /// True for the coordinator/tokenizer row (holds no compute layers).
    public let isCoordinator: Bool

    public init(peerID: UInt32, shardIndex: Int, startLayer: Int, endLayer: Int,
                loadedBytes: Int, isCoordinator: Bool = false) {
        self.peerID = peerID
        self.shardIndex = shardIndex
        self.startLayer = startLayer
        self.endLayer = endLayer
        self.loadedBytes = loadedBytes
        self.isCoordinator = isCoordinator
    }

    /// Human label, e.g. "layers 0-11" (empty span → "0 layers").
    public var layerRange: String {
        endLayer > startLayer ? "layers \(startLayer)-\(endLayer - 1)" : "0 layers"
    }
    public var layerCount: Int { max(0, endLayer - startLayer) }
    /// Loaded weights in MB, one decimal.
    public var loadedMB: Double {
        (Double(loadedBytes) / 1_048_576 * 10).rounded() / 10
    }
    /// One-line panel string, e.g. "layers 0-11 · 219.5 MB".
    public var summary: String {
        loadedBytes > 0 ? "\(layerRange) · \(String(format: "%.1f", loadedMB)) MB" : layerRange
    }
}

public enum NMPShardReport {

    /// Builds the per-device rows for a plan. `loadedBytesByPeer` carries each
    /// shard peer's measured loaded weights (absent = 0/unknown). The full
    /// model size is passed only to sanity-check the honesty invariant.
    public static func devices(
        plan: [NMPShardPlanEntry],
        loadedBytesByPeer: [UInt32: Int]) -> [NMPShardDeviceInfo] {
        plan.map { entry in
            NMPShardDeviceInfo(
                peerID: entry.peerID,
                shardIndex: entry.shardIndex,
                startLayer: entry.startLayer,
                endLayer: entry.endLayer,
                loadedBytes: loadedBytesByPeer[entry.peerID] ?? 0)
        }
    }

    /// The honesty check the panels assert: with a genuine split (>1 shard
    /// carrying layers) no single peer's loaded weights reach the whole
    /// model. Returns true when the invariant holds (or is unknowable because
    /// no bytes were reported yet).
    public static func noPeerHoldsWholeModel(
        _ devices: [NMPShardDeviceInfo], fullModelBytes: Int) -> Bool {
        let carrying = devices.filter { $0.layerCount > 0 }
        guard carrying.count > 1, fullModelBytes > 0 else { return true }
        return carrying.allSatisfy { $0.loadedBytes == 0 || $0.loadedBytes < fullModelBytes }
    }
}
