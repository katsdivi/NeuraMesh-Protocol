//
//  PeerViewModel.swift
//  NeuraMeshPeer — Phase 5 iOS compute peer
//
//  Observable wrapper around NMPPeerNode. The node runs on its own serial
//  queue; every callback hops to the main actor before touching
//  @Published state.
//
//  Engine (auto-selected by `makeEngine()`):
//    • REAL sharded llama — when a `.gguf` model is on-device (downloaded in
//      the app's Models tab — no manual file drop) AND the ggml shard shim
//      framework is embedded and loadable. The phone then partial-loads ONLY
//      its assigned layer range (real ggml graph surgery) and computes real
//      tokens — this is what lets a Mac + iPhone split Qwen-14B.
//    • Reference stand-in (32 × 4096, weightless) — otherwise, so the app
//      always runs and can join/re-shard even without a model or the shim.
//
//  The compute seam (NMPShardComputeEngine) is identical for both, so nothing
//  else in the app or protocol changes (see Docs/Phase5_Design.md and
//  Docs/CrossDevice_Setup_Guide.md → "Real compute on iPhone").
//

import Foundation
import SwiftUI
import Combine
import NMP

@MainActor
final class PeerViewModel: ObservableObject {

    struct ServedRecord: Identifiable {
        let id: UInt32
        let layers: Range<Int>
        let milliseconds: Double
    }

    @Published var statusLines: [String] = []
    @Published var peerIDHex = "—"
    @Published var port: String = "—"
    @Published var shardDescription = "waiting for coordinator"
    /// True once the coordinator assigns ≥1 layer; false while standing by.
    @Published var holdingLayers = false
    @Published var servedCount = 0
    @Published var lastServed: ServedRecord?
    @Published var memoryMB: UInt32 = 0

    private var node: NMPPeerNode?

    /// Set when the REAL sharded engine is active — the on-device model file,
    /// so the UI can say the phone is doing genuine quantized compute.
    @Published var realModelName: String?

    /// The model store, attached by the App so mesh-follow can move the
    /// selection when the mesh's model changes under us.
    var modelStore: ModelManager?

    /// The mesh runs one model everywhere. When it runs something OTHER
    /// than the locally selected file (this peer's assignment was
    /// rejected), the phone follows the mesh instead of sitting
    /// shard-less: this flag makes the next engine pick skip the local
    /// file and stream the mesh's model from the vault. Cleared whenever
    /// the user picks a model (which switches the whole mesh anyway).
    private var followMeshViaVault = false

    /// Re-pick the engine and restart the node — used when the model changes
    /// (downloaded, selected, or deleted in the Models tab) so the new choice
    /// takes effect live, without quitting the app.
    func applyModelChange() {
        followMeshViaVault = false
        restartNode(reason: "restarting with new model…")
    }

    private func restartNode(reason: String) {
        node?.stop()
        node = nil
        realModelName = nil
        holdingLayers = false
        shardDescription = reason
        servedCount = 0
        lastServed = nil
        start()
    }

    /// The coordinator told us the mesh runs `meshTag`. Prefer a local
    /// copy (real on-device compute, no streaming); otherwise stream the
    /// mesh's model from the vault. Either way the phone re-joins on the
    /// MESH's model — the mesh stays on one model everywhere.
    private func followMesh(to meshTag: String) {
        guard NMPLlamaShardRuntime.isAvailable else {
            appendStatus("mesh runs '\(meshTag)' but the compute shim is not "
                         + "embedded — cannot follow; staying as-is")
            return
        }
        let docs = ModelManager.documentsDirectory()
        let files = (try? FileManager.default.contentsOfDirectory(
            at: docs, includingPropertiesForKeys: nil)) ?? []
        let localCopy = files.first {
            $0.pathExtension.lowercased() == "gguf"
                && NMPModelCatalog.candidate(path: $0.path)?.name == meshTag
        }
        if let localCopy {
            modelStore?.select(localCopy.lastPathComponent)
            followMeshViaVault = false
            appendStatus("mesh runs '\(meshTag)' — switching to the local copy")
        } else {
            followMeshViaVault = true
            appendStatus("mesh runs '\(meshTag)' — no local copy; streaming "
                         + "its layers from the vault instead")
        }
        restartNode(reason: "following the mesh onto \(meshTag)…")
    }

    func start() {
        guard node == nil else { return }

        let (engine, tag) = Self.makeEngine(vaultOnly: followMeshViaVault)
        let node = NMPPeerNode(engine: engine, modelTag: tag)
        self.node = node
        peerIDHex = String(format: "%08x", node.peerID)
        if let shard = engine as? NMPLlamaShardComputeEngine {
            realModelName = shard.modelTag
            appendStatus("real sharded engine: \(shard.modelTag) "
                         + "(\(shard.layerCount) layers) — partial-loads only "
                         + "its assigned range")
        } else if engine is NMPVaultShardComputeEngine {
            appendStatus("vault mode — no model stored locally; will stream ONLY "
                         + "the layers the coordinator assigns (disk ≈ RAM)")
        } else {
            appendStatus("reference engine (weightless stand-in) — embed the shard "
                         + "shim for real compute (models stream from the coordinator)")
        }

        node.onStatus = { [weak self] message in
            Task { @MainActor in
                self?.appendStatus(message)
                if let port = self?.node?.listeningPort {
                    self?.port = String(port)
                }
            }
        }
        node.onAssigned = { [weak self] assign in
            Task { @MainActor in
                if assign.startLayer == assign.endLayer {
                    // Mesh 2.8 standby: the coordinator holds this device in
                    // reserve for this plan (the model fits without it, or
                    // Pure Speed routed around it). It's connected and ready
                    // — NOT stuck waiting — and takes layers the moment the
                    // model grows or the mode changes.
                    self?.shardDescription = "0 shards — standing by "
                        + "(joined & ready; not used by the current plan)"
                    self?.holdingLayers = false
                } else {
                    self?.shardDescription = "shard \(assign.shardIndex) of "
                        + "\(assign.pipelineLength): layers \(assign.startLayer)–"
                        + "\(assign.endLayer - 1) of \(assign.totalLayers)"
                    self?.holdingLayers = true
                }
            }
        }
        node.onModelMismatch = { [weak self] meshTag in
            Task { @MainActor in
                self?.followMesh(to: meshTag)
            }
        }
        node.onServed = { [weak self] requestID, layers, seconds in
            Task { @MainActor in
                guard let self else { return }
                self.servedCount += 1
                self.lastServed = ServedRecord(
                    id: requestID, layers: layers, milliseconds: seconds * 1000)
                self.memoryMB = NMPPeerShardEngine.residentMemoryMB()
            }
        }

        do {
            try node.start()
        } catch {
            appendStatus("start failed: \(error)")
        }
    }

    /// Chooses the real sharded engine when a model file is on-device AND the
    /// ggml shard shim can load; otherwise the reference stand-in. Never throws
    /// — any failure falls back so the peer always starts. `vaultOnly` skips
    /// the local file (mesh-follow: the mesh runs a model we don't hold).
    private static func makeEngine(vaultOnly: Bool = false)
        -> (NMPShardComputeEngine, String) {
        // 1. A locally downloaded model (Models tab) → run it directly. Lets the
        //    phone work standalone, and skips streaming when it already has the file.
        if !vaultOnly,
           let modelPath = ModelManager.selectedModelPath(),
           NMPLlamaShardRuntime.isAvailable,
           let shard = try? NMPLlamaShardComputeEngine(modelPath: modelPath) {
            return (shard, shard.modelTag)
        }
        // 2. Shim present but no local model → VAULT mode (Future Plan #3): stream
        //    ONLY the assigned layers from the coordinator, so the phone stores ≈
        //    its layers, not the whole model. Accepts whatever model it serves.
        if NMPLlamaShardRuntime.isAvailable {
            return (NMPVaultShardComputeEngine(),
                    NMPVaultShardComputeEngine.wildcardModelTag)
        }
        // 3. No shim → weightless reference stand-in (mesh still assembles).
        return (NMPReferenceComputeEngine(layerCount: 32, hiddenSize: 4096),
                "nmp-reference-model")
    }

    private func appendStatus(_ message: String) {
        statusLines.append(message)
        if statusLines.count > 8 { statusLines.removeFirst() }
    }
}
