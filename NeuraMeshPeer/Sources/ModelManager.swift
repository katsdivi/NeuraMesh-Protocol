//
//  ModelManager.swift
//  NeuraMeshPeer — in-app model management
//
//  So the phone NEVER needs a manual file drop: this downloads GGUF models
//  straight into the app's Documents (where the engine looks), tracks progress,
//  lets you pick which one to run, and deletes them to reclaim space. The
//  curated catalog is qwen-only because the shard shim runs qwen2/qwen3 blocks
//  (a llama-arch variant is future work) — every entry here is guaranteed
//  compatible; the only per-device question is whether it fits.
//
//  No async/await (house style): URLSession download tasks + delegate
//  callbacks, hopped to the main queue for @Published state.
//

import Foundation
import SwiftUI
import NMP

/// One downloadable model. Sizes are the real GGUF file sizes (approx).
struct NMPCatalogModel: Identifiable {
    let id: String          // the on-disk filename, also the download key
    let displayName: String
    let params: String      // "0.5B", "14B" …
    let quant: String       // "Q4_K_M", "Q2_K" …
    let sizeMB: Int
    let url: URL

    var filename: String { id }
}

@MainActor
final class ModelManager: NSObject, ObservableObject, URLSessionDownloadDelegate {

    /// Filenames of `.gguf` files currently in Documents.
    @Published private(set) var installed: [String] = []
    /// The filename the peer should run (persisted). nil ⇒ first installed.
    @Published private(set) var selected: String?
    /// Live download progress, 0…1, keyed by filename.
    @Published private(set) var progress: [String: Double] = [:]
    /// Last download error per filename (cleared on retry).
    @Published private(set) var errors: [String: String] = [:]

    /// Whole-device RAM and free storage, for honest "will it fit" flags.
    let deviceRAMMB = Int(ProcessInfo.processInfo.physicalMemory / 1_048_576)
    /// True once the ggml shard shim framework is embedded & loadable — only
    /// then is on-device compute REAL (otherwise the peer runs the reference
    /// stand-in; models still download and are ready for when it's embedded).
    var shimReady: Bool { NMPLlamaShardRuntime.isAvailable }

    /// Curated, all-qwen (shim-compatible) catalog. Small → large.
    let catalog: [NMPCatalogModel] = [
        ModelManager.make("Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf",
             name: "Qwen2.5 0.5B", params: "0.5B", quant: "Q4_K_M", sizeMB: 491),
        ModelManager.make("Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf",
             name: "Qwen2.5 1.5B", params: "1.5B", quant: "Q4_K_M", sizeMB: 1120),
        ModelManager.make("Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf",
             name: "Qwen2.5 3B", params: "3B", quant: "Q4_K_M", sizeMB: 1930),
        ModelManager.make("Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-q4_k_m.gguf",
             name: "Qwen2.5 7B", params: "7B", quant: "Q4_K_M", sizeMB: 4680),
        // 14B: the flagship split-across-devices target. Q2_K for the phone —
        // each device still needs the WHOLE file on disk (partial-load reads
        // its layers from it), so this is the smallest 14B that's practical.
        ModelManager.make("Qwen2.5-14B-Instruct-GGUF", "qwen2.5-14b-instruct-q2_k.gguf",
             name: "Qwen2.5 14B", params: "14B", quant: "Q2_K", sizeMB: 5770,
             repoOwner: "bartowski", exactFile: "Qwen2.5-14B-Instruct-Q2_K.gguf"),
    ]

    private lazy var session = URLSession(
        configuration: .default, delegate: self, delegateQueue: nil)
    private var tasks: [String: URLSessionDownloadTask] = [:]

    override init() {
        super.init()
        selected = UserDefaults.standard.string(forKey: Self.selectedKey)
        rescan()
    }

    // MARK: catalog helpers

    private static let selectedKey = "nmp.selectedModelFilename"

    private static func make(_ repo: String, _ file: String, name: String,
                             params: String, quant: String, sizeMB: Int,
                             repoOwner: String = "Qwen",
                             exactFile: String? = nil) -> NMPCatalogModel {
        let remote = exactFile ?? file
        let url = URL(string:
            "https://huggingface.co/\(repoOwner)/\(repo)/resolve/main/\(remote)?download=true")!
        return NMPCatalogModel(id: file, displayName: name, params: params,
                               quant: quant, sizeMB: sizeMB, url: url)
    }

    // MARK: filesystem

    nonisolated static func documentsDirectory() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// Where NMPVaultShardComputeEngine caches streamed shard slices (must match
    /// its default). Only ≈ this device's layers live here, not the whole model.
    static func vaultCacheDirectory() -> URL {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return base.appendingPathComponent("neuramesh/vault", isDirectory: true)
    }

    /// Total bytes of streamed shard slices currently cached on this device.
    var vaultCacheMB: Int {
        let dir = Self.vaultCacheDirectory()
        let files = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        let bytes = files.reduce(0) {
            $0 + ((try? $1.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return bytes / 1_048_576
    }

    func clearVaultCache() {
        let dir = Self.vaultCacheDirectory()
        let files = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        for file in files { try? FileManager.default.removeItem(at: file) }
        objectWillChange.send()
    }

    func fileURL(for filename: String) -> URL {
        Self.documentsDirectory().appendingPathComponent(filename)
    }

    func isInstalled(_ filename: String) -> Bool { installed.contains(filename) }

    /// Free storage on the Documents volume, in MB (for "won't fit on disk").
    var freeStorageMB: Int {
        let values = try? Self.documentsDirectory().resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        if let bytes = values?.volumeAvailableCapacityForImportantUsage {
            return Int(bytes / 1_048_576)
        }
        return 0
    }

    /// Enough disk for the whole file (with a little headroom).
    func fitsStorage(_ model: NMPCatalogModel) -> Bool {
        freeStorageMB >= model.sizeMB + 300
    }

    /// A rough "this may be heavy in RAM" caution. On a Mac+iPhone 2-way split
    /// the phone loads ~half the layers; flag when that half approaches device
    /// RAM. Not a hard block — the mesh can give the phone fewer layers.
    func mayBeHeavy(_ model: NMPCatalogModel) -> Bool {
        Double(model.sizeMB) * 0.5 > Double(deviceRAMMB) * 0.5
    }

    func rescan() {
        let docs = Self.documentsDirectory()
        let items = (try? FileManager.default.contentsOfDirectory(
            at: docs, includingPropertiesForKeys: nil)) ?? []
        installed = items
            .filter { $0.pathExtension.lowercased() == "gguf" }
            .map { $0.lastPathComponent }
            .sorted()
        // Keep selection valid; default to the first installed model.
        if let sel = selected, !installed.contains(sel) { selected = nil }
        if selected == nil { selected = installed.first }
        persistSelection()
    }

    // MARK: selection

    func select(_ filename: String) {
        guard installed.contains(filename) else { return }
        selected = filename
        persistSelection()
    }

    private func persistSelection() {
        UserDefaults.standard.set(selected, forKey: Self.selectedKey)
    }

    /// The absolute path the engine should load, or nil (⇒ reference stand-in).
    nonisolated static func selectedModelPath() -> String? {
        let docs = documentsDirectory()
        if let name = UserDefaults.standard.string(forKey: selectedKey) {
            let candidate = docs.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate.path
            }
        }
        // Fallback: first .gguf present.
        let items = (try? FileManager.default.contentsOfDirectory(
            at: docs, includingPropertiesForKeys: nil)) ?? []
        return items.first { $0.pathExtension.lowercased() == "gguf" }?.path
    }

    // MARK: download / delete

    func download(_ model: NMPCatalogModel) {
        guard tasks[model.filename] == nil else { return }   // already downloading
        errors[model.filename] = nil
        progress[model.filename] = 0
        let task = session.downloadTask(with: model.url)
        task.taskDescription = model.filename
        tasks[model.filename] = task
        task.resume()
    }

    func cancelDownload(_ filename: String) {
        tasks[filename]?.cancel()
        tasks[filename] = nil
        progress[filename] = nil
    }

    func delete(_ filename: String) {
        try? FileManager.default.removeItem(at: fileURL(for: filename))
        rescan()
    }

    var isDownloading: Bool { !tasks.isEmpty }

    // MARK: URLSessionDownloadDelegate (called off-main → hop to main)

    nonisolated func urlSession(_ session: URLSession,
                                downloadTask: URLSessionDownloadTask,
                                didWriteData bytesWritten: Int64,
                                totalBytesWritten: Int64,
                                totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let name = downloadTask.taskDescription
        let value = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        DispatchQueue.main.async { [weak self] in
            if let name { self?.progress[name] = value }
        }
    }

    nonisolated func urlSession(_ session: URLSession,
                                downloadTask: URLSessionDownloadTask,
                                didFinishDownloadingTo location: URL) {
        // Move synchronously here — `location` is deleted when this returns.
        let name = downloadTask.taskDescription ?? location.lastPathComponent
        let dest = Self.documentsDirectory().appendingPathComponent(name)
        try? FileManager.default.removeItem(at: dest)
        let moved = (try? FileManager.default.moveItem(at: location, to: dest)) != nil
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.tasks[name] = nil
            self.progress[name] = nil
            if !moved { self.errors[name] = "could not save the downloaded file" }
            self.rescan()
            if self.selected == nil { self.select(name) }
        }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask,
                                didCompleteWithError error: Error?) {
        guard let error else { return }   // success handled above
        let name = task.taskDescription
        // Ignore user-initiated cancels.
        let nsError = error as NSError
        let cancelled = nsError.code == NSURLErrorCancelled
        DispatchQueue.main.async { [weak self] in
            guard let self, let name else { return }
            self.tasks[name] = nil
            self.progress[name] = nil
            if !cancelled { self.errors[name] = error.localizedDescription }
        }
    }
}
