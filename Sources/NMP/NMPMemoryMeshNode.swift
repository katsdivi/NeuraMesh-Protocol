//
//  NMPMemoryMeshNode.swift
//  NMP — Memory mesh (hackathon build, NEW code)
//
//  One memory peer: a full-mesh NMP participant whose job is DISTRIBUTED
//  CONVERSATIONAL MEMORY, not compute. Each peer runs its own local
//  Supermemory instance (self-hosted, localhost only) and holds only
//  erasure-coded SHARDS of other memories — never a complete readable copy.
//
//  Write path ("remember"):
//      plaintext → LZFSE + AES-256-GCM (NMPMemorySeal)
//                → K-of-N XOR shards (NMPMemoryShardCodec)
//                → one shard per roster peer over the EXISTING encrypted
//                  NMP transport (PeerConnection.sendBurst — no new
//                  networking), each stored as an opaque base64 document in
//                  THAT peer's local Supermemory, tagged "nmp_shards".
//
//  Read path ("what did I say about…"):
//      local Supermemory semantic search over the plaintext INDEX ENTRIES
//      (tag "nmp_memory_index") → pick memory → gather K shards (own +
//      NMP fetches from surviving roster peers) → XOR-reconstruct →
//      AES-GCM open (tamper-evident) → full plaintext. Fewer than K
//      reachable shards is an EXPLICIT quorum failure, never silent output.
//
//  DOCUMENTED TRADEOFF — the searchable index: a fully opaque shard cannot
//  be semantically searched, so every roster peer stores a small plaintext
//  index entry (title + a bounded snippet, `indexSummaryChars`, default 160)
//  plus the memory's AES key in its Supermemory metadata. What this buys and
//  costs, honestly:
//    - buys: real semantic recall on every surviving peer, even after the
//      author dies; tamper-evident reconstruction.
//    - costs: a single peer DOES see the bounded snippet and, holding the
//      key, could decrypt its own 1/K ciphertext fragment. The guarantee is
//      exactly "no single peer holds a COMPLETE readable copy" (full content
//      needs K shards = a quorum), NOT threshold secrecy against colluding
//      peers holding the key. Shamir-style secrecy would need a different
//      code, out of scope for this build.
//
//  Peer identity: long-term Noise static keys persisted per peer in a
//  shared key directory; initiators pin the responder's public key
//  (Noise IK), responders pin the set of known public keys — the existing
//  static-key peer pinning, unchanged.
//
//  House rules: callback + serial-queue style (no async/await), Apple-native
//  only, NMP prefix, big-endian wire fields.
//

import Foundation
import Network

// MARK: - Errors

public enum NMPMemoryMeshError: Error {
    case configError(String)
    case notEnoughPeers(have: Int, needed: Int)
    case quorumUnavailable(have: Int, needed: Int, detail: String)
    case notFound(String)
    case supermemory(String)
    case seal(String)
    case internalError(String)

    public var httpStatus: Int {
        switch self {
        case .notFound: return 404
        case .notEnoughPeers, .quorumUnavailable: return 503
        default: return 500
        }
    }

    public var label: String {
        switch self {
        case .configError(let d): return "config_error: \(d)"
        case .notEnoughPeers(let have, let needed):
            return "not_enough_peers: have \(have) peers, scheme needs \(needed)"
        case .quorumUnavailable(let have, let needed, let detail):
            return "quorum_unavailable: have \(have) of \(needed) required shards — \(detail)"
        case .notFound(let d): return "not_found: \(d)"
        case .supermemory(let d): return "supermemory: \(d)"
        case .seal(let d): return "seal: \(d)"
        case .internalError(let d): return "internal: \(d)"
        }
    }
}

// MARK: - Config

/// One peer's on-disk configuration (written by scripts/setup_memory_mesh.sh
/// or by hand for cross-device runs). The device's Supermemory API key lives
/// HERE (by value or by file reference) — never hardcoded.
public struct NMPMemoryPeerConfig {
    public struct RemotePeer {
        public let peerID: UInt32
        public let host: String
        public let udpPort: UInt16

        public init(peerID: UInt32, host: String, udpPort: UInt16) {
            self.peerID = peerID
            self.host = host
            self.udpPort = udpPort
        }
    }

    public var peerID: UInt32
    public var deviceName: String
    public var udpPort: UInt16
    public var controlPort: UInt16
    public var keyDir: URL
    public var schemeK: Int
    public var schemeN: Int
    public var peers: [RemotePeer]
    /// Supermemory backend (Mac path). Optional: a peer that injects a native
    /// on-device store (iOS) needs no Supermemory server, so a config may omit
    /// the `supermemory` block entirely.
    public var supermemoryBaseURL: URL?
    public var supermemoryAPIKey: String?
    /// Plaintext snippet length for the searchable index entry (tradeoff dial).
    public var indexSummaryChars: Int

    /// In-code construction (iOS builds the config directly; the Mac path uses
    /// `load(path:)`). Omit the supermemory fields for a native-store peer.
    public init(peerID: UInt32, deviceName: String, udpPort: UInt16,
                controlPort: UInt16, keyDir: URL, schemeK: Int, schemeN: Int,
                peers: [RemotePeer], supermemoryBaseURL: URL? = nil,
                supermemoryAPIKey: String? = nil, indexSummaryChars: Int = 160) {
        self.peerID = peerID
        self.deviceName = deviceName
        self.udpPort = udpPort
        self.controlPort = controlPort
        self.keyDir = keyDir
        self.schemeK = schemeK
        self.schemeN = schemeN
        self.peers = peers
        self.supermemoryBaseURL = supermemoryBaseURL
        self.supermemoryAPIKey = supermemoryAPIKey
        self.indexSummaryChars = indexSummaryChars
    }

    public static func load(path: String) throws -> NMPMemoryPeerConfig {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let data: Data
        do { data = try Data(contentsOf: url) }
        catch { throw NMPMemoryMeshError.configError("cannot read \(url.path): \(error)") }
        guard let o = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NMPMemoryMeshError.configError("\(url.path) is not a JSON object")
        }
        func need<T>(_ key: String, _ as: T.Type) throws -> T {
            guard let v = o[key] as? T else {
                throw NMPMemoryMeshError.configError("missing/invalid '\(key)'")
            }
            return v
        }
        let peerID = UInt32(try need("peerID", Int.self))
        let scheme = try need("scheme", [String: Any].self)
        guard let k = scheme["k"] as? Int, let n = scheme["n"] as? Int else {
            throw NMPMemoryMeshError.configError("scheme needs integer k and n")
        }
        var remotes: [RemotePeer] = []
        for row in (o["peers"] as? [[String: Any]]) ?? [] {
            guard let id = row["peerID"] as? Int,
                  let host = row["host"] as? String,
                  let port = row["udpPort"] as? Int else {
                throw NMPMemoryMeshError.configError("peers[] rows need peerID/host/udpPort")
            }
            remotes.append(RemotePeer(peerID: UInt32(id), host: host,
                                      udpPort: UInt16(port)))
        }
        // The `supermemory` block is OPTIONAL — omit it for an on-device
        // (native store) peer. When present it must be well-formed.
        var baseURL: URL?
        var apiKey: String?
        if let sm = o["supermemory"] as? [String: Any] {
            guard let baseRaw = sm["baseURL"] as? String,
                  let parsed = URL(string: baseRaw) else {
                throw NMPMemoryMeshError.configError("supermemory.baseURL missing/invalid")
            }
            baseURL = parsed
            if let inline = sm["apiKey"] as? String {
                apiKey = inline
            } else if let keyFile = sm["apiKeyFile"] as? String {
                let keyURL = URL(fileURLWithPath: (keyFile as NSString).expandingTildeInPath)
                guard let raw = try? String(contentsOf: keyURL, encoding: .utf8) else {
                    throw NMPMemoryMeshError.configError("cannot read apiKeyFile \(keyURL.path)")
                }
                apiKey = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                throw NMPMemoryMeshError.configError("supermemory needs apiKey or apiKeyFile")
            }
        }
        return NMPMemoryPeerConfig(
            peerID: peerID,
            deviceName: (o["deviceName"] as? String) ?? "memory-peer-\(peerID)",
            udpPort: UInt16(try need("udpPort", Int.self)),
            controlPort: UInt16(try need("controlPort", Int.self)),
            keyDir: URL(fileURLWithPath:
                (try need("keyDir", String.self) as NSString).expandingTildeInPath,
                isDirectory: true),
            schemeK: k, schemeN: n,
            peers: remotes,
            supermemoryBaseURL: baseURL,
            supermemoryAPIKey: apiKey,
            indexSummaryChars: (o["indexSummaryChars"] as? Int) ?? 160)
    }
}

// MARK: - Node

public final class NMPMemoryMeshNode {

    // Supermemory container tags: a peer's own readable notes would live in
    // its own tags; these two are OURS and clearly marked as mesh state.
    static let shardTag = "nmp_shards"
    static let indexTag = "nmp_memory_index"

    public var onStatus: ((String) -> Void)?
    public var onDiagnostic: ((String) -> Void)?

    public let config: NMPMemoryPeerConfig
    let scheme: NMPMemoryShardScheme
    /// The device's memory backend — a localhost Supermemory server (Mac) or a
    /// native on-device store (iOS). The node is backend-agnostic.
    let store: NMPMemoryStore

    /// Serial queue owning ALL node state below.
    let queue = DispatchQueue(label: "nmp.memory.node")

    private let staticKeys: NoiseStaticKeyPair
    private var authorizedKeys: Set<Data> = []
    private var listener: UDPListener?
    private var links: [UInt32: MemoryLink] = [:]
    /// Responder links whose handshake hasn't identified the remote yet.
    private var pendingInbound: [MemoryLink] = []
    private var redialTimer: DispatchSourceTimer?
    private var pingTimer: DispatchSourceTimer?
    private var supermemoryHealthy = false

    /// In-RAM caches. Supermemory is the DURABLE store for both shards and
    /// index entries; these caches bridge its asynchronous ingestion (a doc
    /// is searchable seconds after add) and speed up the hot path. A peer
    /// restarted mid-demo re-serves from Supermemory, not RAM.
    private var indexCache: [String: IndexEntry] = [:]
    private var shardCache: [String: NMPMemoryShardRecord] = [:]  // memoryID → own shard

    private var pendingStores: [String: StoreOperation] = [:]
    private var pendingFetches: [String: FetchOperation] = [:]

    // MARK: Inner types

    struct IndexEntry {
        var memoryID: String
        var title: String
        var summary: String
        var key: Data
        var k: Int
        var n: Int
        var roster: [UInt32]
        var createdAt: Double
        var owner: UInt32

        var headerDictionary: [String: Any] {
            [
                "v": 1,
                "memoryID": memoryID,
                "title": title,
                "summary": summary,
                "keyB64": key.base64EncodedString(),
                "k": k, "n": n,
                "roster": roster.map { Int($0) },
                "createdAt": createdAt,
                "owner": Int(owner),
            ]
        }

        init?(headerDictionary h: [String: Any]) {
            guard let memoryID = h["memoryID"] as? String,
                  let keyB64 = h["keyB64"] as? String,
                  let key = Data(base64Encoded: keyB64),
                  let k = h["k"] as? Int, let n = h["n"] as? Int,
                  let roster = h["roster"] as? [Int] else { return nil }
            self.memoryID = memoryID
            self.title = (h["title"] as? String) ?? ""
            self.summary = (h["summary"] as? String) ?? ""
            self.key = key
            self.k = k
            self.n = n
            self.roster = roster.map { UInt32($0) }
            self.createdAt = (h["createdAt"] as? Double) ?? 0
            self.owner = UInt32((h["owner"] as? Int) ?? 0)
        }

        init(memoryID: String, title: String, summary: String, key: Data,
             k: Int, n: Int, roster: [UInt32], createdAt: Double, owner: UInt32) {
            self.memoryID = memoryID
            self.title = title
            self.summary = summary
            self.key = key
            self.k = k
            self.n = n
            self.roster = roster
            self.createdAt = createdAt
            self.owner = owner
        }

        /// String-valued metadata for the Supermemory index document — the
        /// durable copy of everything reconstruction needs.
        var metadata: [String: String] {
            [
                "type": "nmp-index",
                "memoryID": memoryID,
                "title": title,
                "keyB64": key.base64EncodedString(),
                "k": String(k), "n": String(n),
                "roster": roster.map(String.init).joined(separator: ","),
                "createdAt": String(createdAt),
                "owner": String(owner),
            ]
        }

        init?(metadata m: [String: String]) {
            guard m["type"] == "nmp-index",
                  let memoryID = m["memoryID"],
                  let keyB64 = m["keyB64"], let key = Data(base64Encoded: keyB64),
                  let k = m["k"].flatMap(Int.init),
                  let n = m["n"].flatMap(Int.init),
                  let rosterRaw = m["roster"] else { return nil }
            self.memoryID = memoryID
            self.title = m["title"] ?? ""
            self.summary = ""
            self.key = key
            self.k = k
            self.n = n
            self.roster = rosterRaw.split(separator: ",").compactMap { UInt32($0) }
            self.createdAt = m["createdAt"].flatMap(Double.init) ?? 0
            self.owner = m["owner"].flatMap(UInt32.init) ?? 0
        }
    }

    final class MemoryLink {
        let connection: PeerConnection
        let linkQueue: DispatchQueue
        let reassembler = NMPMemoryReassembler()
        let dialed: Bool
        var remotePeerID: UInt32?
        var established = false
        var nextTransferID: UInt32 = 1
        var lastHeardFrom: TimeInterval = ProcessInfo.processInfo.systemUptime
        var missedPings = 0

        init(connection: PeerConnection, linkQueue: DispatchQueue, dialed: Bool) {
            self.connection = connection
            self.linkQueue = linkQueue
            self.dialed = dialed
        }
    }

    struct StoreOperation {
        var awaiting: Set<UInt32>
        var acks: [UInt32: String]   // peerID → "ok" | error text
        var receipt: [String: Any]
        var completion: (Result<[String: Any], NMPMemoryMeshError>) -> Void
        var timer: DispatchSourceTimer
    }

    struct FetchOperation {
        var entry: IndexEntry
        var collected: [Int: NMPMemoryShardRecord] = [:]
        var sources: [Int: String] = [:]
        var outstanding: Set<UInt32> = []
        /// Peers that explicitly failed us (send error or "shard not found").
        var unreachable: [UInt32] = []
        var completions: [(Result<[String: Any], NMPMemoryMeshError>) -> Void] = []
        var timer: DispatchSourceTimer?
        var query: String
        var via: String = "index search"
    }

    // MARK: Init / lifecycle

    /// Designated init — inject ANY memory backend. iOS passes a native
    /// `NMPLocalMemoryStore`; the Mac path uses `withSupermemory(config:)`.
    public init(config: NMPMemoryPeerConfig, store: NMPMemoryStore) throws {
        self.config = config
        self.scheme = try NMPMemoryShardScheme(k: config.schemeK, n: config.schemeN)
        self.store = store
        self.staticKeys = try Self.loadOrCreateKeys(
            keyDir: config.keyDir, peerID: config.peerID)
    }

    /// Mac path: build a node backed by this device's localhost Supermemory
    /// server. HARD LOCAL-ONLY GUARD: NMPSupermemoryConfig.init refuses any
    /// non-localhost base URL, so a config accidentally pointing at the hosted
    /// platform (console/api.supermemory.ai) cannot even start.
    public static func withSupermemory(config: NMPMemoryPeerConfig) throws -> NMPMemoryMeshNode {
        guard let baseURL = config.supermemoryBaseURL, let apiKey = config.supermemoryAPIKey else {
            throw NMPMemoryMeshError.configError(
                "this config has no `supermemory` block — either add one or "
                + "inject a native store via init(config:store:)")
        }
        let smConfig: NMPSupermemoryConfig
        do {
            smConfig = try NMPSupermemoryConfig(baseURL: baseURL, apiKey: apiKey)
        } catch {
            throw NMPMemoryMeshError.configError(
                "supermemory base URL must be localhost (got \(baseURL)) — "
                + "cloud endpoints are banned: \(error)")
        }
        return try NMPMemoryMeshNode(config: config,
                                     store: NMPSupermemoryClient(config: smConfig))
    }

    public func start() throws {
        // Pin every known public key in the key directory (self excluded is
        // harmless — we never dial ourselves).
        authorizedKeys = Self.loadAuthorizedKeys(keyDir: config.keyDir,
                                                 excluding: config.peerID)

        store.health { [weak self] result in
            guard let self else { return }
            self.queue.async {
                switch result {
                case .success:
                    self.supermemoryHealthy = true
                    self.onStatus?("local memory store (\(self.store.kind)) ready at "
                                   + self.store.localityDescription)
                case .failure(let error):
                    self.supermemoryHealthy = false
                    self.onStatus?("WARNING: local memory store (\(self.store.kind)) "
                                   + "not reachable (\(error)) — writes/search will fail")
                }
            }
        }

        guard let port = NWEndpoint.Port(rawValue: config.udpPort) else {
            throw NMPMemoryMeshError.configError("bad udpPort \(config.udpPort)")
        }
        let listener = try UDPListener(port: port, queue: queue)
        listener.onNewTransport = { [weak self] transport, endpoint in
            self?.acceptInbound(transport: transport, endpoint: endpoint)
        }
        listener.onStateChange = { [weak self] state in
            if case .ready = state {
                self?.onStatus?("NMP UDP listener ready on port \(self?.config.udpPort ?? 0)")
            }
            if case .failed(let error) = state {
                self?.onStatus?("NMP UDP listener FAILED: \(error)")
            }
        }
        listener.start()
        self.listener = listener

        armRedialTimer()
        armPingTimer()
        onStatus?("memory peer \(config.peerID) (\(config.deviceName)) up — "
                  + "scheme \(scheme.k)-of-\(scheme.n), UDP \(config.udpPort), "
                  + "control \(config.controlPort)")
    }

    public func stop() {
        queue.async { [self] in
            redialTimer?.cancel(); redialTimer = nil
            pingTimer?.cancel(); pingTimer = nil
            for link in links.values { link.connection.close() }
            for link in pendingInbound { link.connection.close() }
            links.removeAll(); pendingInbound.removeAll()
            listener?.cancel(); listener = nil
        }
    }

    // MARK: Keys (static-key pinning, persisted)

    static func keyPath(_ dir: URL, _ peerID: UInt32, pub: Bool) -> URL {
        dir.appendingPathComponent("peer\(peerID).\(pub ? "pub" : "key")")
    }

    static func loadOrCreateKeys(keyDir: URL, peerID: UInt32) throws -> NoiseStaticKeyPair {
        try? FileManager.default.createDirectory(
            at: keyDir, withIntermediateDirectories: true)
        let privURL = keyPath(keyDir, peerID, pub: false)
        if let raw = try? Data(contentsOf: privURL) {
            let keys = try NoiseStaticKeyPair(rawPrivateKey: raw)
            // Re-publish the pub file in case it was lost.
            try? keys.publicKeyData.write(to: keyPath(keyDir, peerID, pub: true))
            return keys
        }
        let keys = NoiseStaticKeyPair()
        try keys.privateKey.rawRepresentation.write(to: privURL,
                                                    options: [.atomic])
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o600], ofItemAtPath: privURL.path)
        try keys.publicKeyData.write(to: keyPath(keyDir, peerID, pub: true))
        return keys
    }

    static func loadAuthorizedKeys(keyDir: URL, excluding peerID: UInt32) -> Set<Data> {
        let fm = FileManager.default
        var keys: Set<Data> = []
        let files = (try? fm.contentsOfDirectory(atPath: keyDir.path)) ?? []
        for file in files where file.hasSuffix(".pub") {
            if file == "peer\(peerID).pub" { continue }
            if let data = try? Data(contentsOf: keyDir.appendingPathComponent(file)),
               data.count == 32 {
                keys.insert(data)
            }
        }
        return keys
    }

    // MARK: Link management (full mesh: lower peerID dials higher)

    private func armRedialTimer() {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 1, repeating: 3)
        timer.setEventHandler { [weak self] in self?.dialMissingPeers() }
        timer.resume()
        redialTimer = timer
    }

    /// MUST run on `queue`.
    private func dialMissingPeers() {
        for remote in config.peers where remote.peerID > config.peerID {
            if let link = links[remote.peerID], link.established { continue }
            if links[remote.peerID] != nil { continue } // dial in progress
            dial(remote)
        }
    }

    /// MUST run on `queue`.
    private func dial(_ remote: NMPMemoryPeerConfig.RemotePeer) {
        let pubURL = Self.keyPath(config.keyDir, remote.peerID, pub: true)
        guard let remoteStatic = try? Data(contentsOf: pubURL), remoteStatic.count == 32 else {
            onDiagnostic?("peer \(remote.peerID): no public key at \(pubURL.path) yet "
                          + "— will retry (start that peer once, or copy its .pub here)")
            return
        }
        let linkQueue = DispatchQueue(label: "nmp.memory.link.\(remote.peerID)")
        let transport = UDPTransport(
            host: NWEndpoint.Host(remote.host),
            port: NWEndpoint.Port(rawValue: remote.udpPort)!,
            queue: linkQueue)
        var cfg = PeerConnectionConfig(localPeerID: config.peerID)
        cfg.authorizedStaticKeys = [remoteStatic]
        do {
            let connection = try PeerConnection(
                role: .initiator, config: cfg, transport: transport,
                localStatic: staticKeys, remoteStaticPublicKey: remoteStatic,
                queue: linkQueue)
            let link = MemoryLink(connection: connection, linkQueue: linkQueue,
                                  dialed: true)
            link.remotePeerID = remote.peerID
            links[remote.peerID] = link
            wire(link)
            connection.start()
            onDiagnostic?("dialing peer \(remote.peerID) at "
                          + "\(remote.host):\(remote.udpPort)")
        } catch {
            onDiagnostic?("dial to peer \(remote.peerID) failed to build: \(error)")
        }
    }

    /// MUST run on `queue`.
    private func acceptInbound(transport: UDPTransport, endpoint: NWEndpoint) {
        var cfg = PeerConnectionConfig(localPeerID: config.peerID)
        cfg.authorizedStaticKeys = authorizedKeys
        let linkQueue = DispatchQueue(
            label: "nmp.memory.inbound.\(pendingInbound.count)")
        do {
            let connection = try PeerConnection(
                role: .responder, config: cfg, transport: transport,
                localStatic: staticKeys, queue: linkQueue)
            let link = MemoryLink(connection: connection, linkQueue: linkQueue,
                                  dialed: false)
            pendingInbound.append(link)
            wire(link)
            connection.start()
            onDiagnostic?("inbound flow from \(endpoint) — awaiting handshake")
        } catch {
            onDiagnostic?("failed to accept inbound flow: \(error)")
        }
    }

    /// Attaches callbacks. Establishment and packet events hop to `queue`.
    private func wire(_ link: MemoryLink) {
        link.connection.onEstablished = { [weak self, weak link] _, remoteID in
            guard let self, let link else { return }
            self.queue.async { self.adopt(link, remoteID: remoteID) }
        }
        link.connection.onFailed = { [weak self, weak link] error in
            guard let self, let link else { return }
            self.queue.async { self.retire(link, reason: "\(error)") }
        }
        link.connection.onPacket = { [weak self, weak link] packet in
            guard let self, let link else { return }
            // Reassembly happens on the link queue (its owner); completed
            // messages hop to the node queue for dispatch.
            let message: NMPMemoryMessage?
            do { message = try link.reassembler.absorb(packet.payload) }
            catch {
                self.onDiagnostic?("bad memory payload from "
                                   + "\(link.remotePeerID.map(String.init) ?? "?"): \(error)")
                return
            }
            guard let message else { return }
            self.queue.async {
                link.lastHeardFrom = ProcessInfo.processInfo.systemUptime
                link.missedPings = 0
                self.dispatch(message, from: link)
            }
        }
    }

    /// MUST run on `queue`. A fresh handshake for a peer replaces any stale
    /// link (covers kill-and-restart of either side).
    private func adopt(_ link: MemoryLink, remoteID: UInt32) {
        pendingInbound.removeAll { $0 === link }
        if let old = links[remoteID], old !== link {
            onDiagnostic?("replacing stale link to peer \(remoteID)")
            old.connection.close()
        }
        link.remotePeerID = remoteID
        link.established = true
        link.lastHeardFrom = ProcessInfo.processInfo.systemUptime
        links[remoteID] = link
        onStatus?("link to peer \(remoteID) established "
                  + "(\(link.dialed ? "dialed" : "accepted"))")
    }

    /// MUST run on `queue`.
    private func retire(_ link: MemoryLink, reason: String) {
        pendingInbound.removeAll { $0 === link }
        if let id = link.remotePeerID, links[id] === link {
            links.removeValue(forKey: id)
            onStatus?("link to peer \(id) lost: \(reason)")
        }
        link.connection.close()
    }

    // MARK: Liveness pings

    private func armPingTimer() {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 5, repeating: 5)
        timer.setEventHandler { [weak self] in self?.pingTick() }
        timer.resume()
        pingTimer = timer
    }

    /// MUST run on `queue`. A dead UDP peer produces no error — silence is
    /// the only signal. Three missed pings (~15 s) retires the link; the
    /// redial timer then re-establishes it if the peer comes back.
    private func pingTick() {
        let now = ProcessInfo.processInfo.systemUptime
        for link in Array(links.values) where link.established {
            if now - link.lastHeardFrom > 4 {
                link.missedPings += 1
            }
            if link.missedPings >= 3 {
                retire(link, reason: "no traffic/pong for \(link.missedPings) pings")
                continue
            }
            send(message: NMPMemoryMessage(kind: .ping, header: ["t": now]),
                 over: link) { _ in }
        }
    }

    // MARK: Message send

    /// Chunks and ships one message over one link using the existing
    /// send/burst path. Completion (nil = handed to transport OK) fires on
    /// an arbitrary queue.
    func send(message: NMPMemoryMessage, over link: MemoryLink,
              completion: @escaping (Error?) -> Void) {
        let encoded: Data
        do { encoded = try message.encode() }
        catch { completion(error); return }
        let transferID = link.nextTransferID
        link.nextTransferID &+= 1
        let chunkBytes = link.connection.recommendedChunkBytes
        let payloads = NMPMemoryChunker.split(
            message: encoded, transferID: transferID, chunkBytes: chunkBytes)
        link.connection.sendBurstAsync(payloads: payloads, completion: completion)
    }

    /// MUST run on `queue`.
    private func establishedLink(to peerID: UInt32) -> MemoryLink? {
        guard let link = links[peerID], link.established else { return nil }
        return link
    }

    // MARK: Dispatch

    /// MUST run on `queue`.
    private func dispatch(_ message: NMPMemoryMessage, from link: MemoryLink) {
        switch message.kind {
        case .storeShard: handleStoreShard(message, from: link)
        case .storeAck: handleStoreAck(message, from: link)
        case .fetchShard: handleFetchShard(message, from: link)
        case .fetchResult: handleFetchResult(message, from: link)
        case .ping:
            send(message: NMPMemoryMessage(kind: .pong, header: message.header),
                 over: link) { _ in }
        case .pong:
            break // lastHeardFrom already refreshed in the packet hop
        }
    }

    // MARK: - WRITE PATH

    /// Creates a memory: seal → shard → distribute. The full plaintext is
    /// NOT persisted anywhere; each roster peer (this one included) gets one
    /// opaque shard plus the plaintext index entry in its own Supermemory.
    public func createMemory(
        content: String, title: String,
        completion: @escaping (Result<[String: Any], NMPMemoryMeshError>) -> Void
    ) {
        queue.async { [self] in
            let remoteIDs = links.values
                .filter { $0.established }
                .compactMap { $0.remotePeerID }
                .sorted()
            let roster = ([config.peerID] + remoteIDs.prefix(scheme.n - 1)).sorted()
            guard roster.count == scheme.n else {
                completion(.failure(.notEnoughPeers(
                    have: roster.count, needed: scheme.n)))
                return
            }

            let sealed: NMPMemorySeal.Sealed
            do { sealed = try NMPMemorySeal.seal(plaintext: Data(content.utf8)) }
            catch { completion(.failure(.seal("\(error)"))); return }

            let memoryID = UUID().uuidString
                .replacingOccurrences(of: "-", with: "").lowercased()
            let records = NMPMemoryShardCodec.encode(
                memoryID: memoryID, blob: sealed.ciphertext, scheme: scheme)
            let entry = IndexEntry(
                memoryID: memoryID,
                title: title.isEmpty ? Self.deriveTitle(content) : title,
                summary: Self.makeSummary(title: title, content: content,
                                          maxChars: config.indexSummaryChars),
                key: sealed.key, k: scheme.k, n: scheme.n,
                roster: roster,
                createdAt: Date().timeIntervalSince1970,
                owner: config.peerID)

            // Own slot: cache + persist in OUR Supermemory.
            guard let mySlot = roster.firstIndex(of: config.peerID) else {
                completion(.failure(.internalError("self missing from roster")))
                return
            }
            let myRecord = records[mySlot]
            indexCache[memoryID] = entry
            shardCache[memoryID] = myRecord
            persistShardAndIndex(record: myRecord, entry: entry) { [weak self] err in
                if let err { self?.onDiagnostic?("local supermemory persist: \(err)") }
            }

            // Remote slots: one shard each over NMP.
            var awaiting: Set<UInt32> = []
            for (slot, peerID) in roster.enumerated() where peerID != config.peerID {
                guard let link = establishedLink(to: peerID) else { continue }
                var header = entry.headerDictionary
                header["shardIndex"] = slot
                let message = NMPMemoryMessage(
                    kind: .storeShard, header: header, body: records[slot].encode())
                awaiting.insert(peerID)
                send(message: message, over: link) { [weak self] error in
                    guard let error, let self else { return }
                    self.queue.async {
                        self.resolveStoreAck(memoryID: memoryID, peerID: peerID,
                                             verdict: "send failed: \(error)")
                    }
                }
            }

            let receipt: [String: Any] = [
                "memoryID": memoryID,
                "title": entry.title,
                "scheme": "\(scheme.k)-of-\(scheme.n)",
                "roster": roster.map { Int($0) },
                "indexSummary": entry.summary,
                "bytes": [
                    "plaintext": content.utf8.count,
                    "ciphertext": sealed.ciphertext.count,
                    "perShard": myRecord.payload.count,
                ],
                "note": "full plaintext persisted NOWHERE; each peer holds one "
                      + "opaque shard + the bounded plaintext index entry",
            ]

            let timer = DispatchSource.makeTimerSource(queue: queue)
            timer.schedule(deadline: .now() + 6)
            timer.setEventHandler { [weak self] in
                self?.finishStore(memoryID: memoryID)
            }
            pendingStores[memoryID] = StoreOperation(
                awaiting: awaiting, acks: [:], receipt: receipt,
                completion: completion, timer: timer)
            timer.resume()
            if awaiting.isEmpty { finishStore(memoryID: memoryID) }
        }
    }

    /// MUST run on `queue`.
    private func resolveStoreAck(memoryID: String, peerID: UInt32, verdict: String) {
        guard var op = pendingStores[memoryID] else { return }
        op.acks[peerID] = verdict
        op.awaiting.remove(peerID)
        pendingStores[memoryID] = op
        if op.awaiting.isEmpty { finishStore(memoryID: memoryID) }
    }

    /// MUST run on `queue`.
    private func finishStore(memoryID: String) {
        guard let op = pendingStores.removeValue(forKey: memoryID) else { return }
        op.timer.cancel()
        var receipt = op.receipt
        var acks: [String: String] = [:]
        for (peer, verdict) in op.acks { acks[String(peer)] = verdict }
        for peer in op.awaiting { acks[String(peer)] = "NO ACK (timeout)" }
        receipt["shardAcks"] = acks
        let allStored = op.awaiting.isEmpty
            && op.acks.values.allSatisfy { $0 == "ok" }
        receipt["distributed"] = allStored
        op.completion(.success(receipt))
    }

    /// MUST run on `queue`. Remote peer asked us to hold a shard.
    private func handleStoreShard(_ message: NMPMemoryMessage, from link: MemoryLink) {
        guard let entry = IndexEntry(headerDictionary: message.header) else {
            onDiagnostic?("storeShard with unusable header — dropped")
            return
        }
        let record: NMPMemoryShardRecord
        do { record = try NMPMemoryShardRecord.decode(message.body) }
        catch {
            reply(storeAckFor: entry.memoryID, index: -1, verdict: "bad record: \(error)",
                  over: link)
            return
        }
        indexCache[entry.memoryID] = entry
        shardCache[entry.memoryID] = record
        persistShardAndIndex(record: record, entry: entry) { [weak self, weak link] err in
            guard let self, let link else { return }
            self.queue.async {
                self.reply(storeAckFor: entry.memoryID, index: record.shardIndex,
                           verdict: err ?? "ok", over: link)
            }
        }
        onStatus?("holding shard \(record.shardIndex) of memory "
                  + "\(entry.memoryID.prefix(8))… (\(record.payload.count) B, opaque) "
                  + "from peer \(entry.owner)")
    }

    /// MUST run on `queue`.
    private func reply(storeAckFor memoryID: String, index: Int, verdict: String,
                       over link: MemoryLink) {
        let ack = NMPMemoryMessage(kind: .storeAck, header: [
            "memoryID": memoryID, "shardIndex": index, "verdict": verdict,
        ])
        send(message: ack, over: link) { _ in }
    }

    /// MUST run on `queue`.
    private func handleStoreAck(_ message: NMPMemoryMessage, from link: MemoryLink) {
        guard let memoryID = message.header["memoryID"] as? String,
              let peerID = link.remotePeerID else { return }
        let verdict = (message.header["verdict"] as? String) ?? "ok"
        resolveStoreAck(memoryID: memoryID, peerID: peerID, verdict: verdict)
    }

    /// Stores one shard (opaque base64 doc) + the index entry (searchable
    /// doc) in THIS device's local Supermemory. err = nil on full success.
    private func persistShardAndIndex(record: NMPMemoryShardRecord,
                                      entry: IndexEntry,
                                      completion: @escaping (String?) -> Void) {
        let shardCustomID = "nmp-shard-\(record.memoryID)-\(record.shardIndex)"
        let shardMeta: [String: String] = [
            "type": "nmp-shard",
            "memoryID": record.memoryID,
            "shardIndex": String(record.shardIndex),
            "k": String(record.k), "n": String(record.n),
        ]
        store.addDocument(
            content: record.encode().base64EncodedString(),
            customID: shardCustomID,
            containerTag: Self.shardTag,
            metadata: shardMeta
        ) { [weak self] shardResult in
            guard let self else { return }
            let searchable = entry.title.isEmpty
                ? entry.summary : "\(entry.title)\n\(entry.summary)"
            self.store.addDocument(
                content: searchable,
                customID: "nmp-index-\(entry.memoryID)",
                containerTag: Self.indexTag,
                metadata: entry.metadata
            ) { indexResult in
                switch (shardResult, indexResult) {
                case (.success, .success): completion(nil)
                case (.failure(let e), _): completion("shard doc failed: \(e)")
                case (_, .failure(let e)): completion("index doc failed: \(e)")
                }
            }
        }
    }

    // MARK: - READ PATH

    /// Semantic recall: search the local Supermemory index, then gather K
    /// shards and reconstruct. Fails EXPLICITLY when fewer than K shards
    /// are reachable.
    public func recall(
        query: String,
        completion: @escaping (Result<[String: Any], NMPMemoryMeshError>) -> Void
    ) {
        store.search(query: query, containerTag: Self.indexTag,
                           limit: 5) { [weak self] result in
            guard let self else { return }
            self.queue.async {
                var entry: IndexEntry?
                var via = "supermemory semantic search"
                switch result {
                case .success(let hits):
                    for hit in hits {
                        if let e = IndexEntry(metadata: hit.metadata) {
                            entry = self.indexCache[e.memoryID] ?? e
                            break
                        }
                        // Some responses carry customId but thin metadata.
                        if let custom = hit.customID,
                           custom.hasPrefix("nmp-index-") {
                            let id = String(custom.dropFirst("nmp-index-".count))
                            if let cached = self.indexCache[id] {
                                entry = cached
                                break
                            }
                        }
                    }
                case .failure(let error):
                    self.onDiagnostic?("supermemory search failed (\(error)) — "
                                       + "falling back to in-RAM keyword match")
                }
                if entry == nil {
                    // Freshness bridge only: Supermemory ingestion is async,
                    // so a memory written seconds ago may not be searchable
                    // yet. This naive substring match is NOT the search
                    // story — Supermemory is.
                    entry = self.keywordFallback(query: query)
                    if entry != nil { via = "in-RAM keyword fallback (ingestion lag)" }
                }
                guard let entry else {
                    completion(.failure(.notFound(
                        "no indexed memory matches '\(query)'")))
                    return
                }
                self.gatherAndReconstruct(entry: entry, query: query, via: via,
                                          completion: completion)
            }
        }
    }

    /// MUST run on `queue`.
    private func keywordFallback(query: String) -> IndexEntry? {
        let terms = query.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 }
        guard !terms.isEmpty else { return nil }
        var best: (IndexEntry, Int)?
        for entry in indexCache.values {
            let haystack = "\(entry.title) \(entry.summary)".lowercased()
            let score = terms.reduce(0) { $0 + (haystack.contains($1) ? 1 : 0) }
            if score > 0, score > (best?.1 ?? 0) { best = (entry, score) }
        }
        return best?.0
    }

    /// MUST run on `queue`.
    private func gatherAndReconstruct(
        entry: IndexEntry, query: String, via: String,
        completion: @escaping (Result<[String: Any], NMPMemoryMeshError>) -> Void
    ) {
        let memoryID = entry.memoryID
        if var existing = pendingFetches[memoryID] {
            existing.completions.append(completion)
            pendingFetches[memoryID] = existing
            return
        }
        var op = FetchOperation(entry: entry, completions: [completion],
                                query: query, via: via)

        // 1. Own shard, RAM first.
        if let mine = shardCache[memoryID] {
            op.collected[mine.shardIndex] = mine
            op.sources[mine.shardIndex] = "local (peer \(config.peerID))"
        }

        // 2. Ask every reachable roster peer in parallel.
        for peerID in entry.roster where peerID != config.peerID {
            guard let link = establishedLink(to: peerID) else {
                op.unreachable.append(peerID)
                continue
            }
            op.outstanding.insert(peerID)
            let message = NMPMemoryMessage(kind: .fetchShard,
                                           header: ["memoryID": memoryID])
            send(message: message, over: link) { [weak self] error in
                guard let error, let self else { return }
                self.queue.async {
                    self.onDiagnostic?("fetchShard send to \(peerID) failed: \(error)")
                    self.fetchPeerDone(memoryID: memoryID, peerID: peerID,
                                       record: nil)
                }
            }
        }

        // 3. If RAM had no local copy, also try our own Supermemory (covers
        //    a restarted peer whose RAM is empty but whose store survived).
        let needLocalLookup = op.collected.isEmpty
        pendingFetches[memoryID] = op

        if needLocalLookup {
            lookupAnyLocalShard(entry: entry) { [weak self] record in
                guard let self else { return }
                self.queue.async {
                    if let record, var op = self.pendingFetches[memoryID],
                       op.collected[record.shardIndex] == nil {
                        op.collected[record.shardIndex] = record
                        op.sources[record.shardIndex] =
                            "local supermemory (peer \(self.config.peerID))"
                        self.pendingFetches[memoryID] = op
                    }
                    self.maybeFinishFetch(memoryID: memoryID)
                }
            }
        }

        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 5)
        timer.setEventHandler { [weak self] in
            self?.finishFetch(memoryID: memoryID, timedOut: true)
        }
        pendingFetches[memoryID]?.timer = timer
        timer.resume()
        // When a local Supermemory lookup is in flight, let ITS callback
        // decide completion — finishing here on empty `outstanding` would
        // race that lookup and wrongly report a quorum failure.
        if !needLocalLookup {
            maybeFinishFetch(memoryID: memoryID)
        }
    }

    /// Tries this peer's own Supermemory for ANY shard of the memory (we
    /// don't know our slot after a restart — probe all n custom IDs).
    private func lookupAnyLocalShard(entry: IndexEntry,
                                     completion: @escaping (NMPMemoryShardRecord?) -> Void) {
        var slot = 0
        func tryNext() {
            guard slot < entry.n else { completion(nil); return }
            let customID = "nmp-shard-\(entry.memoryID)-\(slot)"
            slot += 1
            store.getDocument(customID: customID) { result in
                if case .success(let doc) = result,
                   let raw = Data(base64Encoded: doc.content
                        .trimmingCharacters(in: .whitespacesAndNewlines)),
                   let record = try? NMPMemoryShardRecord.decode(raw) {
                    completion(record)
                } else {
                    tryNext()
                }
            }
        }
        tryNext()
    }

    /// MUST run on `queue`.
    private func fetchPeerDone(memoryID: String, peerID: UInt32,
                               record: NMPMemoryShardRecord?) {
        guard var op = pendingFetches[memoryID] else { return }
        op.outstanding.remove(peerID)
        if let record {
            if op.collected[record.shardIndex] == nil {
                op.collected[record.shardIndex] = record
                op.sources[record.shardIndex] = "peer \(peerID) over NMP"
            }
        } else {
            op.unreachable.append(peerID)
        }
        pendingFetches[memoryID] = op
        maybeFinishFetch(memoryID: memoryID)
    }

    /// MUST run on `queue`. Finishes as soon as K shards are in hand (fast
    /// path — no waiting on peers we don't need), or once every peer has
    /// answered. The timer is the backstop for peers that never answer.
    private func maybeFinishFetch(memoryID: String) {
        guard let op = pendingFetches[memoryID] else { return }
        if op.collected.count >= op.entry.k || op.outstanding.isEmpty {
            finishFetch(memoryID: memoryID, timedOut: false)
        }
    }

    /// MUST run on `queue`.
    private func finishFetch(memoryID: String, timedOut: Bool) {
        guard let op = pendingFetches.removeValue(forKey: memoryID) else { return }
        op.timer?.cancel()

        // Peers still outstanding at finish time: if we reached quorum they
        // were simply NOT NEEDED (we didn't wait); if we timed out short of
        // quorum they are genuinely unreachable.
        let reachedQuorum = op.collected.count >= op.entry.k
        var unreachable = op.unreachable
        var notNeeded: [UInt32] = []
        if reachedQuorum {
            notNeeded = op.outstanding.sorted()
        } else {
            unreachable.append(contentsOf: op.outstanding)
        }

        let entry = op.entry
        guard reachedQuorum else {
            let detail = "shards in hand: \(op.collected.keys.sorted()); "
                + "unreachable peers: "
                + unreachable.sorted().map(String.init).joined(separator: ", ")
                + (timedOut ? " (fetch timeout)" : "")
            let failure = NMPMemoryMeshError.quorumUnavailable(
                have: op.collected.count, needed: entry.k, detail: detail)
            for done in op.completions { done(.failure(failure)) }
            return
        }

        do {
            let records = Array(op.collected.values)
            let ciphertext = try NMPMemoryShardCodec.reconstruct(records: records)
            let plaintext = try NMPMemorySeal.open(ciphertext: ciphertext,
                                                   key: entry.key)
            let sourceList = op.collected.keys.sorted().map {
                ["shardIndex": $0, "source": op.sources[$0] ?? "?"] as [String: Any]
            }
            let response: [String: Any] = [
                "memoryID": entry.memoryID,
                "title": entry.title,
                "content": String(data: plaintext, encoding: .utf8) ?? "",
                "scheme": "\(entry.k)-of-\(entry.n)",
                "roster": entry.roster.map { Int($0) },
                "shardsUsed": sourceList,
                "shardsUsedCount": op.collected.count,
                "quorumNeeded": entry.k,
                "unreachablePeers": unreachable.sorted().map { Int($0) },
                "peersNotNeeded": notNeeded.map { Int($0) },
                "integrity": "AES-256-GCM authenticated — reconstruction is tamper-evident",
                "matchedVia": op.via,
            ]
            for done in op.completions { done(.success(response)) }
        } catch {
            let failure = NMPMemoryMeshError.internalError(
                "reconstruction failed: \(error)")
            for done in op.completions { done(.failure(failure)) }
        }
    }

    /// MUST run on `queue`. Serve our shard of a memory to a roster peer.
    private func handleFetchShard(_ message: NMPMemoryMessage, from link: MemoryLink) {
        guard let memoryID = message.header["memoryID"] as? String else { return }
        if let record = shardCache[memoryID] {
            let reply = NMPMemoryMessage(
                kind: .fetchResult,
                header: ["memoryID": memoryID, "found": true],
                body: record.encode())
            send(message: reply, over: link) { _ in }
            onStatus?("served shard \(record.shardIndex) of "
                      + "\(memoryID.prefix(8))… to peer "
                      + "\(link.remotePeerID.map(String.init) ?? "?")")
            return
        }
        // RAM miss (e.g. we restarted) — durable copy lives in Supermemory.
        guard let entry = indexCache[memoryID] else {
            self.send(message: NMPMemoryMessage(
                kind: .fetchResult,
                header: ["memoryID": memoryID, "found": false]), over: link) { _ in }
            return
        }
        lookupAnyLocalShard(entry: entry) { [weak self, weak link] record in
            guard let self, let link else { return }
            self.queue.async {
                if let record { self.shardCache[memoryID] = record }
                let reply = NMPMemoryMessage(
                    kind: .fetchResult,
                    header: ["memoryID": memoryID, "found": record != nil],
                    body: record?.encode() ?? Data())
                self.send(message: reply, over: link) { _ in }
            }
        }
    }

    /// MUST run on `queue`.
    private func handleFetchResult(_ message: NMPMemoryMessage, from link: MemoryLink) {
        guard let memoryID = message.header["memoryID"] as? String,
              let peerID = link.remotePeerID else { return }
        let found = (message.header["found"] as? Bool) ?? false
        let record = found ? try? NMPMemoryShardRecord.decode(message.body) : nil
        fetchPeerDone(memoryID: memoryID, peerID: peerID, record: record)
    }

    // MARK: - Introspection (for the control API)

    public func status(completion: @escaping ([String: Any]) -> Void) {
        queue.async { [self] in
            let linkRows: [[String: Any]] = config.peers.map { remote in
                let link = links[remote.peerID]
                return [
                    "peerID": Int(remote.peerID),
                    "endpoint": "\(remote.host):\(remote.udpPort)",
                    "established": link?.established ?? false,
                ]
            }
            store.health { [self] health in
                let healthy: Bool
                if case .success = health { healthy = true } else { healthy = false }
                queue.async { self.supermemoryHealthy = healthy }
                completion([
                    "peerID": Int(config.peerID),
                    "deviceName": config.deviceName,
                    "scheme": "\(scheme.k)-of-\(scheme.n)",
                    "udpPort": Int(config.udpPort),
                    "links": linkRows,
                    "store": [
                        "kind": store.kind,
                        "locality": store.localityDescription,
                        "healthy": healthy,
                        "localOnly": true,
                    ],
                    "memoriesIndexed": indexCache.count,
                    "shardsHeld": shardCache.count,
                ])
            }
        }
    }

    public func listMemories(completion: @escaping ([[String: Any]]) -> Void) {
        queue.async { [self] in
            let rows = indexCache.values
                .sorted { $0.createdAt > $1.createdAt }
                .map { entry -> [String: Any] in
                    [
                        "memoryID": entry.memoryID,
                        "title": entry.title,
                        "indexSummary": entry.summary,
                        "scheme": "\(entry.k)-of-\(entry.n)",
                        "roster": entry.roster.map { Int($0) },
                        "owner": Int(entry.owner),
                        "holdingShard": shardCache[entry.memoryID]?.shardIndex ?? -1,
                        "createdAt": entry.createdAt,
                    ]
                }
            completion(rows)
        }
    }

    // MARK: - Helpers

    static func deriveTitle(_ content: String) -> String {
        let line = content
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\n", with: " ")
        return line.isEmpty ? "Untitled memory" : String(line.prefix(60))
    }

    static func makeSummary(title: String, content: String, maxChars: Int) -> String {
        let collapsed = content
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if collapsed.count <= maxChars { return collapsed }
        return String(collapsed.prefix(maxChars)) + "…"
    }
}
