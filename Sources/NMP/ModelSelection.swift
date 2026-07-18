//
//  ModelSelection.swift
//  NMP — Phase C (adaptive model tiering)
//
//  The "optimal per scenario" brain: given the devices in the mesh right now
//  (RAM + free storage) and the model files actually available, pick the
//  BEST model that fits — highest quality with speed headroom — and re-decide
//  it every time membership changes.
//
//  It sits ABOVE the sharder and reuses its capacity math
//  (`NMPModelSharder.layerCapacity` / `planDetailed`). Two ceilings gate a
//  candidate:
//    • STORAGE — every hosting device must have disk for the model file
//      (partial-load still reads its layers from the whole GGUF on disk). A
//      device without room can't host that model → it drops out, and if too
//      few remain the mesh degrades to a smaller model.
//    • RAM — the layer split must fit aggregate RAM under per-device ceilings,
//      with headroom reserved for the KV cache + activations (so the winner is
//      never so large it fragments into tiny, round-trip-heavy shards).
//
//  Pure Swift (no ggml, no llama) — fully unit-tested with synthetic meshes
//  and catalogs, and it reads real GGUFs via the Phase 5 parser.
//

import Foundation

// MARK: - Candidate

/// One model the mesh could run, with the footprint the selector needs.
public struct NMPModelCandidate: Equatable, Sendable {
    public let path: String
    public let name: String
    public let architecture: String
    public let layerCount: Int
    public let hiddenSize: Int
    public let fileBytes: Int
    /// Exact quantized weight bytes per transformer block.
    public let bytesPerLayer: Int
    /// Total weights — the primary quality signal.
    public let totalParameters: Int
    /// False for layer slices / split fragments (vault streaming,
    /// gguf_slice.py output): structurally valid GGUFs that carry only a
    /// layer range of a model. Detected from METADATA, not filename — see
    /// `NMPModelCatalog.isCompleteModel(_:)`. Never boot the mesh on one.
    public let isCompleteModel: Bool

    public init(path: String, name: String, architecture: String,
                layerCount: Int, hiddenSize: Int, fileBytes: Int,
                bytesPerLayer: Int, totalParameters: Int,
                isCompleteModel: Bool = true) {
        self.path = path
        self.name = name
        self.architecture = architecture
        self.layerCount = layerCount
        self.hiddenSize = hiddenSize
        self.fileBytes = fileBytes
        self.bytesPerLayer = bytesPerLayer
        self.totalParameters = totalParameters
        self.isCompleteModel = isCompleteModel
    }

    /// File size rounded to whole MB (the unit storage ceilings compare in).
    public var fileMB: Int { fileBytes / 1_048_576 }
    /// On-disk bits per weight — the quantization level (secondary quality).
    public var bitsPerWeight: Double {
        totalParameters > 0 ? Double(fileBytes) * 8 / Double(totalParameters) : 0
    }

    /// Higher quality first: more parameters, then higher precision.
    public static func higherQuality(_ a: NMPModelCandidate, _ b: NMPModelCandidate) -> Bool {
        if a.totalParameters != b.totalParameters {
            return a.totalParameters > b.totalParameters
        }
        return a.bitsPerWeight > b.bitsPerWeight
    }
}

// MARK: - Catalog

/// Discovers the models actually available (so "what can we run" is found, not
/// hardcoded), reading each GGUF's real footprint.
public enum NMPModelCatalog {

    /// Reads one GGUF into a candidate; nil if it can't be parsed or lacks the
    /// dimensions the selector needs.
    public static func candidate(path: String) -> NMPModelCandidate? {
        let expanded = (path as NSString).expandingTildeInPath
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: expanded),
              let fileBytes = (attrs[.size] as? NSNumber)?.intValue, fileBytes > 0,
              let gguf = try? NMPGGUFModel.load(path: expanded),
              let layers = gguf.layerCount, layers > 0,
              let hidden = gguf.hiddenSize,
              let bpl = gguf.bytesPerLayer(fileBytes: fileBytes) else {
            return nil
        }
        return NMPModelCandidate(
            path: expanded,
            name: gguf.modelName ?? (expanded as NSString).lastPathComponent,
            architecture: gguf.architecture ?? "unknown",
            layerCount: layers, hiddenSize: hidden, fileBytes: fileBytes,
            bytesPerLayer: bpl, totalParameters: gguf.totalParameters,
            isCompleteModel: isCompleteModel(gguf)
                && !filenameSuggestsSlice(expanded))
    }

    /// Filename-convention backstop, applied AFTER the metadata checks (a
    /// slice whose name metadata was scrubbed still must not boot a mesh):
    /// gguf_slice.py's vault outputs are "*_partN.gguf"; ad-hoc slices carry
    /// "_sliced_<a>_<b>" in the filename.
    static func filenameSuggestsSlice(_ path: String) -> Bool {
        let file = (((path as NSString).lastPathComponent) as NSString)
            .deletingPathExtension.lowercased()
        return file.range(of: #"_part\d+$"#, options: .regularExpression) != nil
            || file.range(of: #"_sliced_\d+_\d+"#, options: .regularExpression) != nil
    }

    /// METADATA slice detection (a slice "parses OK", so parsing is not
    /// validation). Two slice flavors exist and each leaks through metadata:
    ///  • NMPGGUFSlicer (Swift) keeps `<arch>.block_count` at the FULL N but
    ///    carries only its range's `blk.<L>.*` tensors → block coverage
    ///    falls short of block_count.
    ///  • scripts/gguf_slice.py renumbers blocks AND overrides block_count,
    ///    but stamps "…_sliced_<a>_<b>" into `general.name`.
    static func isCompleteModel(_ gguf: NMPGGUFModel) -> Bool {
        guard let layers = gguf.layerCount, layers > 0 else { return false }
        let presentBlocks = Set(gguf.tensors.compactMap {
            NMPGGUFSlicer.blockIndex($0.name)
        })
        guard presentBlocks.count >= layers else { return false }
        let name = gguf.modelName ?? ""
        return name.range(of: #"_sliced_\d+_\d+$"#,
                          options: .regularExpression) == nil
    }

    /// Scans a directory for `.gguf` models, highest quality first. Skips
    /// unreadable files and incomplete slices/fragments (metadata check —
    /// the same criterion /api/models/select rejects on), so a vault slice
    /// is never listed, recommended, auto-selected, or adaptively chosen.
    public static func scan(directory: String) -> [NMPModelCandidate] {
        let dir = (directory as NSString).expandingTildeInPath
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: dir) else {
            return []
        }
        return names
            .filter { $0.hasSuffix(".gguf") }
            .compactMap { candidate(path: (dir as NSString).appendingPathComponent($0)) }
            .filter(\.isCompleteModel)
            .sorted(by: NMPModelCandidate.higherQuality)
    }
}

// MARK: - Selection

public struct NMPModelSelection: Equatable, Sendable {
    public let model: NMPModelCandidate
    public let plan: [NMPShardPlanEntry]
    /// Devices that host a slice (had storage + RAM for the winner).
    public let eligiblePeers: [UInt32]
    /// Human-readable "why this model" — including what larger tiers were
    /// skipped and the binding reason.
    public let reason: String
}

public enum NMPModelSelector {

    /// Reserve ~40% of RAM for the OS, KV cache, and live activations — the
    /// same headroom the sharder uses, which also keeps the winning model from
    /// fitting so tightly it fragments into tiny, round-trip-heavy shards.
    public static let defaultHeadroom = 0.6

    /// Picks the highest-quality model that fits the current mesh. Returns nil
    /// only when even the smallest candidate can't fit anywhere.
    public static func pick(
        mesh: [NMPCapabilities],
        catalog: [NMPModelCandidate],
        headroom: Double = defaultHeadroom,
        measuredSecondsPerLayer: [UInt32: Double] = [:],
        computeShares: [UInt32: Double] = [:]
    ) -> NMPModelSelection? {
        let ranked = catalog.sorted(by: NMPModelCandidate.higherQuality)
        var skipped: [String] = []

        for model in ranked {
            // Devices that can HOST this model: room for the file on disk AND
            // enough RAM for at least one of its layers.
            var capacities: [UInt32: Int] = [:]
            var eligible: [NMPCapabilities] = []
            var storageBlocked = 0
            for device in mesh {
                let hasDisk = Int(device.storageFreeMB) >= model.fileMB
                let cap = NMPModelSharder.layerCapacity(
                    ramMB: device.ramMB, bytesPerLayer: model.bytesPerLayer,
                    headroom: headroom)
                if !hasDisk { storageBlocked += 1; continue }
                if cap >= 1 { eligible.append(device); capacities[device.peerID] = cap }
            }
            let totalCapacity = capacities.values.reduce(0, +)

            guard totalCapacity >= model.layerCount, !eligible.isEmpty else {
                let why = storageBlocked > 0
                    ? "\(storageBlocked) device(s) lack \(model.fileMB) MB disk"
                    : "RAM fits only \(totalCapacity)/\(model.layerCount) layers"
                skipped.append("\(model.name) [\(why)]")
                continue
            }

            let plan = NMPModelSharder.planDetailed(
                layerCount: model.layerCount, peers: eligible,
                measuredSecondsPerLayer: measuredSecondsPerLayer,
                computeShares: computeShares,
                layerCapacities: capacities, objective: .capacityThenSpeed)

            var reason = "chose \(model.name) — "
                + "\(model.fileMB) MB, \(model.layerCount) layers, "
                + "fits \(eligible.count)/\(mesh.count) device(s) "
                + "(RAM holds \(totalCapacity) layers)"
            if !skipped.isEmpty {
                reason += "; degraded past " + skipped.joined(separator: ", ")
            }
            return NMPModelSelection(
                model: model, plan: plan.entries,
                eligiblePeers: eligible.map(\.peerID), reason: reason)
        }
        return nil
    }
}

// MARK: - Adaptive controller (churn-driven re-selection)

/// Keeps the mesh on the OPTIMAL model as devices come and go. Feed it the
/// current membership on every join/leave; it re-runs the selector and reports
/// what changed, so the coordinator knows whether to just re-shard (same
/// model) or switch models (reload a different GGUF on the peers). Pure
/// decision logic — the caller performs the reload/re-assign (reusing the
/// churn-safe re-prefill from Phase A).
public final class NMPAdaptiveModelController {

    /// What a membership change implies.
    public enum Decision: Equatable, Sendable {
        /// The best model is unchanged and so is its plan — nothing to do.
        case unchanged(NMPModelSelection)
        /// Same model, but the layer split changed (re-assign the plan).
        case reshard(NMPModelSelection)
        /// A different model is now optimal (reload it on the peers, then
        /// re-assign). `from` is the previous model name ("(none)" on the
        /// first selection).
        case switchModel(from: String, to: NMPModelSelection)
        /// Not even the smallest candidate fits the current mesh.
        case noModelFits

        public var selection: NMPModelSelection? {
            switch self {
            case .unchanged(let s), .reshard(let s), .switchModel(_, let s): return s
            case .noModelFits: return nil
            }
        }
    }

    private let catalog: [NMPModelCandidate]
    private let headroom: Double
    private var current: NMPModelSelection?

    public init(catalog: [NMPModelCandidate],
                headroom: Double = NMPModelSelector.defaultHeadroom) {
        self.catalog = catalog
        self.headroom = headroom
    }

    /// The model + plan currently in force (nil before the first fit).
    public var currentSelection: NMPModelSelection? { current }

    /// Re-decides for `mesh`, updates internal state, and returns what changed.
    public func evaluate(
        mesh: [NMPCapabilities],
        measuredSecondsPerLayer: [UInt32: Double] = [:],
        computeShares: [UInt32: Double] = [:]
    ) -> Decision {
        guard let pick = NMPModelSelector.pick(
            mesh: mesh, catalog: catalog, headroom: headroom,
            measuredSecondsPerLayer: measuredSecondsPerLayer,
            computeShares: computeShares) else {
            current = nil
            return .noModelFits
        }
        let previous = current
        current = pick
        guard let previous else {
            return .switchModel(from: "(none)", to: pick)
        }
        if previous.model.path != pick.model.path {
            return .switchModel(from: previous.model.name, to: pick)
        }
        if previous.plan != pick.plan {
            return .reshard(pick)
        }
        return .unchanged(pick)
    }
}
