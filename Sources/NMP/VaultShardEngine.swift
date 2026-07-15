//
//  VaultShardEngine.swift
//  NMP — Future Plan #3: the weight vault (peer side)
//
//  A compute engine for a peer that holds NO local model. When the coordinator
//  assigns it a layer range, it streams ONLY those layers from the coordinator's
//  vault (NMPVaultServer), caches the slice on disk, opens it with the real ggml
//  shard engine, and computes. So a phone stores ≈ its layers, not the whole
//  model — disk ≈ RAM.
//
//  It slots in behind the existing NMPShardComputeEngine seam: NMPPeerShardEngine
//  calls provision(for:) when a SHARD_ASSIGN arrives (before validating dims),
//  then runLayers(...) exactly as for a locally-loaded model.
//
//  No async/await (house rule): the fetch is a synchronous LAN download on the
//  peer's queue — one-time per range, cached thereafter.
//

import Foundation

/// Implemented by engines that materialize their weights from a vault on
/// assignment. NMPPeerShardEngine calls this before dimension validation.
public protocol NMPVaultProvisioning: AnyObject {
    func provision(for assign: NMPShardAssign) throws
}

public enum NMPVaultEngineError: Error, Sendable {
    case notProvisioned
    case noVaultEndpoint
    case downloadFailed(String)
}

public final class NMPVaultShardComputeEngine:
    NMPShardComputeEngine, NMPGlobalLayerAware, NMPVaultProvisioning {

    /// Pass this as the peer node's modelTag so it accepts the coordinator's
    /// model (whatever it is) — the peer trusts the slices it will stream.
    public static let wildcardModelTag = "*"

    private let cacheDirectory: URL
    private let maxContext: Int
    private let lock = NSLock()

    private var inner: NMPLlamaShardComputeEngine?
    private var currentRange: Range<Int>?
    private var _globalLayerCount = 0

    public init(cacheDirectory: URL? = nil, maxContext: Int = 4096) {
        self.maxContext = maxContext
        if let cacheDirectory {
            self.cacheDirectory = cacheDirectory
        } else {
            let base = FileManager.default.urls(for: .cachesDirectory,
                                                in: .userDomainMask).first
                ?? URL(fileURLWithPath: NSTemporaryDirectory())
            self.cacheDirectory = base.appendingPathComponent("neuramesh/vault",
                                                              isDirectory: true)
        }
        try? FileManager.default.createDirectory(
            at: self.cacheDirectory, withIntermediateDirectories: true)
    }

    // MARK: NMPShardComputeEngine

    public var layerCount: Int {
        lock.lock(); defer { lock.unlock() }; return inner?.layerCount ?? 0
    }
    public var hiddenSize: Int {
        lock.lock(); defer { lock.unlock() }; return inner?.hiddenSize ?? 0
    }

    public func runLayers(start: Int, end: Int, input: [Float]) throws -> [Float] {
        lock.lock(); let engine = inner; lock.unlock()
        guard let engine else { throw NMPVaultEngineError.notProvisioned }
        return try engine.runLayers(start: start, end: end, input: input)
    }

    // MARK: NMPGlobalLayerAware

    public var globalLayerCount: Int {
        get { lock.lock(); defer { lock.unlock() }; return _globalLayerCount }
        set { lock.lock(); defer { lock.unlock() }; _globalLayerCount = newValue }
    }

    /// The on-disk size of the currently-cached slice (the honest proof this
    /// peer stores only its layers). 0 before the first provision.
    public var cachedSliceBytes: Int {
        lock.lock(); let range = currentRange; let tag = inner?.modelTag; lock.unlock()
        guard let range, let tag else { return 0 }
        let path = slicePath(modelTag: tag, start: range.lowerBound, end: range.upperBound)
        return (try? FileManager.default.attributesOfItem(atPath: path.path)[.size] as? Int)
            .flatMap { $0 } ?? 0
    }

    // MARK: NMPVaultProvisioning

    public func provision(for assign: NMPShardAssign) throws {
        let start = Int(assign.startLayer), end = Int(assign.endLayer)
        guard end > start else { return }   // standby (0 layers): nothing to load

        lock.lock()
        if let range = currentRange, range == start..<end, inner != nil {
            lock.unlock(); return   // already holding exactly this shard
        }
        lock.unlock()

        guard !assign.vaultEndpoint.isEmpty else {
            throw NMPVaultEngineError.noVaultEndpoint
        }

        let dest = slicePath(modelTag: assign.modelTag, start: start, end: end)
        if !FileManager.default.fileExists(atPath: dest.path) {
            try download(from: assign.vaultEndpoint, start: start, end: end, to: dest)
        }

        let engine = try NMPLlamaShardComputeEngine(modelPath: dest.path,
                                                    maxContext: maxContext)
        lock.lock()
        inner = engine
        currentRange = start..<end
        _globalLayerCount = Int(assign.totalLayers)
        lock.unlock()
    }

    // MARK: - Helpers

    private func slicePath(modelTag: String, start: Int, end: Int) -> URL {
        let safe = modelTag.isEmpty ? "model"
            : String(modelTag.map { $0.isLetter || $0.isNumber ? $0 : "-" })
        return cacheDirectory.appendingPathComponent("\(safe)_\(start)_\(end).gguf")
    }

    private func download(from endpoint: String, start: Int, end: Int,
                          to dest: URL) throws {
        guard let url = URL(string:
            "http://\(endpoint)/vault?start=\(start)&end=\(end)") else {
            throw NMPVaultEngineError.downloadFailed("bad vault URL for \(endpoint)")
        }
        // Synchronous LAN GET (one-time per range). Data(contentsOf:) writes to
        // a temp file first, so this stays streaming, not fully in RAM twice.
        let data: Data
        do { data = try Data(contentsOf: url) }
        catch { throw NMPVaultEngineError.downloadFailed(
            "fetch \(url) failed: \(error.localizedDescription)") }
        let tmp = dest.appendingPathExtension("partial")
        try data.write(to: tmp)
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.moveItem(at: tmp, to: dest)
    }
}
