//
//  PeerStatusView.swift
//  NeuraMeshPeer — Phase 5 iOS compute peer
//
//  Status screen: identity, advertised port, assigned shard, live
//  inference metrics. Keep the phone awake while benchmarking — the
//  screen staying on prevents the app from being suspended mid-mesh.
//

import SwiftUI
import Combine

struct PeerStatusView: View {
    @EnvironmentObject var model: PeerViewModel

    var body: some View {
        NavigationStack {
            List {
                Section("This peer") {
                    row("Peer ID", model.peerIDHex)
                    row("UDP port", model.port)
                    row("Service", "NeuraMesh-\(model.peerIDHex)")
                }

                Section("Shard") {
                    HStack(alignment: .top, spacing: 8) {
                        Circle()
                            .fill(model.holdingLayers ? Color.green
                                  : (model.shardDescription.hasPrefix("0 shards")
                                     ? Color.orange : Color.secondary))
                            .frame(width: 8, height: 8)
                            .padding(.top, 5)
                        Text(model.shardDescription)
                            .font(.callout.monospaced())
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                Section("Inference") {
                    row("Requests served", "\(model.servedCount)")
                    if let last = model.lastServed {
                        row("Last request", "#\(last.id)")
                        row("Layers", "\(last.layers.lowerBound)–\(last.layers.upperBound - 1)")
                        row("Compute time", String(format: "%.1f ms", last.milliseconds))
                    }
                    if model.memoryMB > 0 {
                        row("Memory", "\(model.memoryMB) MB")
                    }
                }

                Section("Log") {
                    ForEach(Array(model.statusLines.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("NeuraMesh Peer")
        }
        // Benchmarks die if iOS suspends the app; keep the screen on.
        .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
        .onDisappear { UIApplication.shared.isIdleTimerDisabled = false }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).font(.callout.monospaced()).foregroundStyle(.secondary)
        }
    }
}
