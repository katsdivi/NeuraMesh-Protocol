//
//  MemoryPeerController.swift
//  NeuraMeshPeer — Distributed-memory peer (NEW glue, hackathon build)
//
//  Makes the iPhone a MEMORY peer in the NMP memory mesh, alongside the
//  existing compute peer (PeerViewModel). Where the compute peer computes
//  model layers, this one holds erasure-coded SHARDS of other peers'
//  conversational memories and reconstructs on demand — see
//  Docs/Memory_Mesh.md and Docs/Memory_Mesh_iOS.md.
//
//  WHY A SEPARATE BACKEND ON iOS
//  -----------------------------
//  A Mac memory peer backs its store with a localhost `supermemory-server`
//  (a Node/Mach-O HTTP server). iOS can't run that: no Node runtime, and the
//  OS forbids an app from spawning a separate server executable. So on the
//  phone we inject `NMPLocalMemoryStore` — a fully on-device store (file blob
//  store + NaturalLanguage embeddings, NO network) — through the node's
//  backend-agnostic designated init `NMPMemoryMeshNode(config:store:)`. The
//  seal / shard / distribute / reconstruct code and the encrypted NMP
//  transport are byte-for-byte the same as on Mac.
//
//  WHAT THIS FILE IS / ISN'T
//  -------------------------
//  • It is a small `ObservableObject` the app's UI drives (start / remember /
//    recall / refreshStatus), mirroring PeerViewModel's threading contract:
//    the node runs on its own serial queue; every callback hops to the main
//    actor before touching @Published state.
//  • It is NOT wired into the app's TabView or the .pbxproj here — adding this
//    file to the target and dropping in a screen are the manual Xcode steps
//    documented in Docs/Memory_Mesh_iOS.md. This file only needs to compile
//    against the NMP module (it has NOT been built/run on a device this
//    session — that final step is manual).
//
//  IDENTITY / KEY EXCHANGE (static-key pinned, static roster — no Bonjour)
//  ----------------------------------------------------------------------
//  The node auto-generates & persists this device's Noise static keypair in
//  `keyDir` and pins peers by the `peer<id>.pub` files it finds THERE. There
//  is no discovery for this feature: you pass the Macs' LAN IPs in `peers`,
//  and you must hand-copy public keys both ways (this device's `peer<id>.pub`
//  onto each Mac's keyDir, and each Mac's `peer<id>.pub` into THIS device's
//  keyDir). `ownPublicKeyURL` / `peerKeysDirectory` expose those locations so
//  you can move the files via the Files app or the Xcode container. Adding
//  Bonjour later would remove this manual step.
//

import Foundation
import Combine
import NMP

@MainActor
final class MemoryPeerController: ObservableObject {

    /// One remote memory peer (a Mac running `nmp-memory-peer`): its NMP peer
    /// id, LAN IP / hostname, and UDP port. You get these from each Mac's
    /// config.json (peerID / udpPort) and `ipconfig getifaddr en0` (host).
    struct PeerRef: Identifiable {
        let peerID: UInt32
        let host: String
        let udpPort: UInt16
        var id: UInt32 { peerID }
    }

    // MARK: Inputs (bind these from a settings screen before calling start)

    /// This device's NMP peer id. Must be unique across the mesh and stable
    /// across launches (it names this device's key files + roster slot).
    @Published var peerID: UInt32 = 0xA1
    @Published var deviceName: String = "iphone-memory-peer"
    /// UDP port this peer listens on (the Macs dial THIS host:port — put it in
    /// their rosters). Control port is unused on iOS (no HTTP control surface)
    /// but the config loader requires the field, so we carry a placeholder.
    @Published var udpPort: UInt16 = 7810
    @Published var controlPort: UInt16 = 7811
    /// The Macs. K-of-N reconstruction needs K reachable shard-holders.
    @Published var peers: [PeerRef] = []
    /// XOR erasure scheme. 2-of-3 tolerates exactly one peer loss (the codec
    /// is single-parity ⇒ n == k+1; see Docs/Memory_Mesh.md "Limitations").
    let schemeK = 2
    let schemeN = 3

    // MARK: Published UI state

    @Published private(set) var started = false
    @Published private(set) var statusText = "not started"
    @Published private(set) var lastReceiptJSON = ""
    @Published private(set) var lastRecallJSON = ""
    @Published private(set) var log: [String] = []

    // MARK: Node

    private var node: NMPMemoryMeshNode?

    // MARK: On-device storage layout (app sandbox; no iCloud, no network)

    /// Base folder for all memory-peer state, under Application Support.
    private static func baseDirectory() -> URL {
        let root = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask).first
            ?? FileManager.default.urls(for: .documentDirectory,
                                        in: .userDomainMask)[0]
        return root.appendingPathComponent("MemoryMesh", isDirectory: true)
    }

    /// Noise static keys live here: this device's `peer<id>.key`/`.pub`
    /// (auto-created by the node) plus the Macs' `peer<id>.pub` files you copy
    /// in. The node pins EVERY `.pub` it finds here (except its own).
    var keyDirectory: URL {
        Self.baseDirectory().appendingPathComponent("keys", isDirectory: true)
    }

    /// Where the on-device blob store keeps shards + the searchable index.
    var storeDirectory: URL {
        Self.baseDirectory().appendingPathComponent("store", isDirectory: true)
    }

    /// This device's public key file. Copy it onto every Mac's keyDir so the
    /// Macs can pin THIS phone (Noise IK is mutual). Exists after start().
    var ownPublicKeyURL: URL {
        keyDirectory.appendingPathComponent("peer\(peerID).pub")
    }

    /// Drop each Mac's `peer<id>.pub` here (Files app / Xcode container) so the
    /// phone can pin the Macs. Same folder as our own keys — the node loads all
    /// `.pub` files from it. Surfaced so a UI can show/share the path.
    var peerKeysDirectory: URL { keyDirectory }

    /// This device's public key as base64, for display / copy-out. `nil` until
    /// start() has created the keypair.
    func ownPublicKeyBase64() -> String? {
        guard let data = try? Data(contentsOf: ownPublicKeyURL) else { return nil }
        return data.base64EncodedString()
    }

    // MARK: Lifecycle

    /// Builds the on-device store + config, constructs the memory node, wires
    /// its callbacks to @Published log/status, and starts it. Idempotent: a
    /// second call while running is a no-op.
    func start() {
        guard node == nil else { return }
        do {
            try FileManager.default.createDirectory(
                at: keyDirectory, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(
                at: storeDirectory, withIntermediateDirectories: true)

            // On-device backend — replaces supermemory-server. No server, no
            // network: file blob store + NaturalLanguage embeddings.
            let store = NMPLocalMemoryStore(directory: storeDirectory)

            let config = try makeConfig()
            let node = try NMPMemoryMeshNode(config: config, store: store)

            node.onStatus = { [weak self] message in
                Task { @MainActor in self?.appendLog(message) }
            }
            node.onDiagnostic = { [weak self] message in
                Task { @MainActor in self?.appendLog("(diag) \(message)") }
            }

            try node.start()
            self.node = node
            self.started = true
            appendLog("memory peer \(peerID) started — scheme "
                      + "\(schemeK)-of-\(schemeN), \(peers.count) roster peer(s)")
            refreshStatus()
        } catch {
            appendLog("start failed: \(error)")
        }
    }

    func stop() {
        node?.stop()
        node = nil
        started = false
        statusText = "stopped"
    }

    // MARK: Write / read (results hop back to the main actor for the UI)

    /// Seal → shard K-of-N → distribute one shard per roster peer. On success,
    /// `lastReceiptJSON` holds the receipt (no plaintext is stored anywhere).
    func remember(content: String, title: String) {
        guard let node else { appendLog("remember: peer not started"); return }
        node.createMemory(content: content, title: title) { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                switch result {
                case .success(let receipt):
                    self.lastReceiptJSON = Self.pretty(receipt)
                    let title = (receipt["title"] as? String) ?? "memory"
                    self.appendLog("remembered — \(title)")
                case .failure(let error):
                    self.lastReceiptJSON = "error: \(error.label)"
                    self.appendLog("remember failed: \(error.label)")
                }
            }
        }
    }

    /// Semantic recall: search the local index → gather K shards (own + peers
    /// over NMP) → reconstruct → decrypt. Below quorum this fails LOUDLY (the
    /// error names the unreachable peers) rather than returning wrong output.
    func recall(query: String) {
        guard let node else { appendLog("recall: peer not started"); return }
        node.recall(query: query) { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                switch result {
                case .success(let answer):
                    self.lastRecallJSON = Self.pretty(answer)
                    let count = (answer["shardsUsedCount"] as? Int) ?? 0
                    self.appendLog("recalled using \(count) shard(s)")
                case .failure(let error):
                    self.lastRecallJSON = "error: \(error.label)"
                    self.appendLog("recall failed: \(error.label)")
                }
            }
        }
    }

    /// Pulls a fresh node/link/store snapshot into `statusText`.
    func refreshStatus() {
        guard let node else { statusText = "not started"; return }
        node.status { [weak self] snapshot in
            Task { @MainActor in
                self?.statusText = Self.pretty(snapshot)
            }
        }
    }

    // MARK: Config construction (in code, from the inputs above)

    /// Builds `NMPMemoryPeerConfig` from this controller's inputs.
    ///
    /// NOTE ON APPROACH: `NMPMemoryPeerConfig` / `RemotePeer` now have PUBLIC
    /// inits, so this could be a direct struct literal. We instead build the
    /// same JSON the config loader validates, write it to app storage, and load
    /// it — reusing the exact validation the Mac CLI runs, for free. We
    /// deliberately OMIT the `supermemory` block — that signals an on-device
    /// (native store) peer; `withSupermemory(config:)` is Mac-only.
    private func makeConfig() throws -> NMPMemoryPeerConfig {
        let json: [String: Any] = [
            "peerID": Int(peerID),
            "deviceName": deviceName,
            "udpPort": Int(udpPort),
            "controlPort": Int(controlPort),
            "keyDir": keyDirectory.path,
            "scheme": ["k": schemeK, "n": schemeN],
            "peers": peers.map { peer in
                ["peerID": Int(peer.peerID),
                 "host": peer.host,
                 "udpPort": Int(peer.udpPort)] as [String: Any]
            },
            "indexSummaryChars": 160,
            // No "supermemory" block on purpose — native store replaces it.
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let url = Self.baseDirectory()
            .appendingPathComponent("memory-peer-config.json")
        try FileManager.default.createDirectory(
            at: Self.baseDirectory(), withIntermediateDirectories: true)
        try data.write(to: url, options: [.atomic])
        return try NMPMemoryPeerConfig.load(path: url.path)
    }

    // MARK: Helpers

    private func appendLog(_ message: String) {
        log.append(message)
        if log.count > 12 { log.removeFirst() }
    }

    private static func pretty(_ object: Any) -> String {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys]),
              let text = String(data: data, encoding: .utf8) else {
            return "\(object)"
        }
        return text
    }
}
