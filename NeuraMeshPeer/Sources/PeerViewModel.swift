//
//  PeerViewModel.swift
//  NeuraMeshPeer — Phase 5 iOS compute peer
//
//  Observable wrapper around NMPPeerNode. The node runs on its own serial
//  queue; every callback hops to the main actor before touching
//  @Published state.
//
//  Engine: the deterministic reference engine, sized like a 7B model
//  (32 layers × 4096 hidden) so activation tensors are real-sized
//  (16 KB/hop). To run REAL quantized inference instead, conform a
//  llama.cpp wrapper to NMPShardComputeEngine and swap it in here —
//  nothing else in the app or the protocol changes (see
//  Docs/Phase5_Design.md, "The compute seam").
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

    func start() {
        guard node == nil else { return }

        let engine = NMPReferenceComputeEngine(layerCount: 32, hiddenSize: 4096)
        let node = NMPPeerNode(engine: engine, modelTag: "nmp-reference-model")
        self.node = node
        peerIDHex = String(format: "%08x", node.peerID)

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

    private func appendStatus(_ message: String) {
        statusLines.append(message)
        if statusLines.count > 8 { statusLines.removeFirst() }
    }
}
