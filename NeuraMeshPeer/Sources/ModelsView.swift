//
//  ModelsView.swift
//  NeuraMeshPeer — pick / download / manage models on the phone
//
//  No manual file drop: download a model straight into the app, pick which
//  one the peer runs, delete to reclaim space. Everything is qwen (the shard
//  shim's supported architecture); the only per-device flag is whether it
//  fits in storage (and a soft RAM caution for the big ones).
//

import SwiftUI
import NMP

struct ModelsView: View {
    @EnvironmentObject var models: ModelManager
    @EnvironmentObject var peer: PeerViewModel
    @EnvironmentObject var mesh: MeshUILocator

    /// Filename mid-switch (spinner on that row), and the outcome note.
    @State private var switching: String?
    @State private var meshNote = ""

    var body: some View {
        NavigationStack {
            List {
                if !models.shimReady {
                    Section {
                        Label {
                            Text("Compute shim not embedded — models download fine, "
                                 + "but the phone runs the reference stand-in until you "
                                 + "add nmpshard.xcframework (Embed & Sign) and rebuild.")
                                .font(.footnote)
                        } icon: {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.orange)
                        }
                    }
                }

                Section {
                    VStack(alignment: .leading, spacing: 6) {
                        Label("Streaming (recommended)", systemImage: "antenna.radiowaves.left.and.right")
                            .font(.callout.weight(.semibold))
                        Text("With no model downloaded, this phone streams ONLY the "
                             + "layers the coordinator assigns and caches them here — "
                             + "so it stores ≈ its layers, not the whole model.")
                            .font(.footnote).foregroundStyle(.secondary)
                        HStack {
                            Text("Shard cache: \(models.vaultCacheMB) MB")
                                .font(.footnote.monospaced())
                            Spacer()
                            if models.vaultCacheMB > 0 {
                                Button("Clear") { models.clearVaultCache() }
                                    .buttonStyle(.borderless).font(.footnote)
                            }
                        }
                    }
                    .padding(.vertical, 2)
                }

                Section {
                    if models.installed.isEmpty {
                        Text("None. Streaming above needs no download.")
                            .foregroundStyle(.secondary)
                    }
                    ForEach(models.installed, id: \.self) { name in
                        installedRow(name)
                    }
                } header: {
                    Text("Downloaded on this phone")
                } footer: {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("The mesh runs ONE model everywhere: picking a model "
                             + "here switches the whole mesh onto it (the Mac needs "
                             + "the same file in ~/models). With no mesh in reach, "
                             + "the pick applies to this phone alone.")
                        if !meshNote.isEmpty {
                            Text(meshNote).foregroundStyle(
                                meshNote.hasPrefix("⚠️") ? .red : .secondary)
                        }
                    }
                }

                Section {
                    ForEach(models.catalog) { model in
                        catalogRow(model)
                    }
                } header: {
                    Text("Download")
                } footer: {
                    Text("Free storage: \(gb(models.freeStorageMB)) · device RAM: "
                         + "\(gb(models.deviceRAMMB)). Each model needs its whole file "
                         + "on disk; the phone loads only its assigned layers into RAM.")
                }
            }
            .navigationTitle("Models")
        }
    }

    // MARK: rows

    @ViewBuilder
    private func installedRow(_ name: String) -> some View {
        let isSelected = models.selected == name
        HStack {
            Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(isSelected ? Color.accentColor : Color.secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text(name).font(.callout)
                if isSelected {
                    Text(peer.realModelName != nil ? "active — running real compute"
                         : "selected")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer()
            if switching == name {
                ProgressView()
            }
            Button(role: .destructive) {
                models.delete(name)
                peer.applyModelChange()
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.borderless)
            .disabled(switching != nil)
        }
        .contentShape(Rectangle())
        .onTapGesture { pick(name, isSelected: isSelected) }
    }

    /// One model for the whole mesh: a mesh in reach is asked to switch
    /// (it relaunches; this phone reconnects already matching). Only when
    /// NO mesh is around does the pick stay local — standalone use. A
    /// refusal (e.g. the Mac lacks the file) changes NOTHING, so the mesh
    /// never splits across models.
    private func pick(_ name: String, isSelected: Bool) {
        guard !isSelected, switching == nil else { return }
        guard mesh.baseURL != nil else {
            models.select(name)
            peer.applyModelChange()
            meshNote = "no mesh in reach — switched this phone only"
            return
        }
        switching = name
        meshNote = "switching the mesh to \(name)…"
        mesh.switchMeshModel(to: name) { failure in
            switching = nil
            if let failure {
                meshNote = "⚠️ mesh kept its model: \(failure)"
            } else {
                models.select(name)
                peer.applyModelChange()
                meshNote = "mesh is switching to \(name) — it relaunches and "
                    + "this phone reconnects in a few seconds"
            }
        }
    }

    @ViewBuilder
    private func catalogRow(_ model: NMPCatalogModel) -> some View {
        let installed = models.isInstalled(model.filename)
        let downloading = models.progress[model.filename]
        let fits = models.fitsStorage(model)
        let heavy = models.mayBeHeavy(model)
        let err = models.errors[model.filename]

        VStack(alignment: .leading, spacing: 6) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(model.displayName).font(.callout.weight(.semibold))
                        if heavy && fits {
                            Text("heavy").font(.caption2.weight(.bold))
                                .padding(.horizontal, 6).padding(.vertical, 1)
                                .background(Color.orange.opacity(0.2))
                                .clipShape(Capsule())
                        }
                    }
                    Text("\(model.quant) · \(gb(model.sizeMB))")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                trailing(model: model, installed: installed,
                         downloading: downloading, fits: fits)
            }
            if let downloading {
                ProgressView(value: downloading)
                Text("\(Int(downloading * 100))%")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            if !fits && !installed {
                Text("Won’t fit — needs \(gb(model.sizeMB)) free, you have "
                     + "\(gb(models.freeStorageMB)).")
                    .font(.caption).foregroundStyle(.red)
            } else if heavy && !installed {
                Text("Large for this phone — the mesh can still give it fewer "
                     + "layers, or pair it with the Mac.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let err {
                Text(err).font(.caption).foregroundStyle(.red)
            }
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private func trailing(model: NMPCatalogModel, installed: Bool,
                          downloading: Double?, fits: Bool) -> some View {
        if installed {
            Label("Installed", systemImage: "checkmark.circle.fill")
                .labelStyle(.iconOnly).foregroundStyle(.green)
        } else if downloading != nil {
            Button("Cancel") { models.cancelDownload(model.filename) }
                .buttonStyle(.borderless).tint(.red)
        } else {
            Button("Get") { models.download(model) }
                .buttonStyle(.borderless)
                .disabled(!fits)
        }
    }

    private func gb(_ mb: Int) -> String {
        mb >= 1024 ? String(format: "%.1f GB", Double(mb) / 1024) : "\(mb) MB"
    }
}
