//
//  PeerNode.swift
//  NMP — Phase 5
//
//  Turn-key mesh roles. Everything the CLI tools and the iOS app need,
//  so those front-ends stay thin UI/argument shells:
//
//  - NMPPeerNode: a compute peer. Binds a UDP listener, advertises itself
//    over Bonjour (capabilities + UDP port + Noise static key in the TXT
//    record), accepts the coordinator's handshake, and serves its
//    assigned shard. This is what runs on the iPhone.
//
//  - NMPCoordinatorNode: the coordinator. Browses for peers, dials every
//    dialable one (TXT carries port + key — zero manual configuration),
//    and exposes the orchestrator for shard assignment and inference.
//    This is what runs on the Mac.
//
//  PORT TRICK (load-bearing): a peer binds its NMP UDP listener to an
//  ephemeral port P, then binds its Bonjour anchor (TCP) listener to the
//  SAME P. The SRV record therefore advertises P, and a coordinator can
//  dial the browse result's `.service` endpoint over UDP directly —
//  Network.framework resolves the SRV host and connects UDP to P, which
//  is exactly the NMP listener. No address parsing, no TXT-record IP
//  hacks, and it keeps working when the peer's DHCP address changes.
//
//  Security model for the auto-dial path: the responder's static key is
//  taken from its TXT record, i.e. trust-on-first-use against an active
//  LAN attacker; the responder accepts any authenticated initiator. Fine
//  for a benchmark mesh on your own Wi-Fi; production pins keys via
//  `PeerConnectionConfig.authorizedStaticKeys` (see Phase5_Design.md).
//

import Foundation
import Network

// MARK: - Peer node (runs on the iPhone / any compute peer)

public final class NMPPeerNode {

    // MARK: Callbacks (invoked on `queue`)

    /// Human-readable lifecycle updates for UIs and logs.
    public var onStatus: ((String) -> Void)?
    public var onAssigned: ((NMPShardAssign) -> Void)?
    /// requestID, layer range served, pure compute seconds.
    public var onServed: ((UInt32, Range<Int>, TimeInterval) -> Void)?
    public var onDiagnostic: ((String) -> Void)?

    // MARK: State

    public let peerID: UInt32
    public private(set) var listeningPort: UInt16?
    public private(set) var assignment: NMPShardAssign?
    public private(set) var servedCount = 0

    private let engine: NMPShardComputeEngine
    private let modelTag: String
    private let queue: DispatchQueue
    private let staticKeys = NoiseStaticKeyPair()

    private var listener: UDPListener?
    private var publisher: NMPBonjourPublisher?
    private var connection: PeerConnection?
    private var shardEngine: NMPPeerShardEngine?

    public init(
        engine: NMPShardComputeEngine,
        modelTag: String,
        peerID: UInt32 = UInt32.random(in: 1...UInt32.max),
        queue: DispatchQueue = DispatchQueue(label: "nmp.peer.node")
    ) {
        self.engine = engine
        self.modelTag = modelTag
        self.peerID = peerID
        self.queue = queue
    }

    public func start() throws {
        let listener = try UDPListener(port: NWEndpoint.Port(rawValue: 0)!, queue: queue)
        self.listener = listener

        listener.onStateChange = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                guard self.publisher == nil, let port = listener.port?.rawValue else { return }
                self.listeningPort = port
                self.advertise(on: port)
            case .failed(let error):
                self.onStatus?("UDP listener failed: \(error)")
            default:
                break
            }
        }
        listener.onNewTransport = { [weak self] transport, endpoint in
            self?.accept(transport: transport, from: endpoint)
        }

        onStatus?("peer \(String(format: "%08x", peerID)) starting "
                  + "(\(engine.layerCount) layers × \(engine.hiddenSize), model '\(modelTag)')")
        listener.start()
    }

    public func stop() {
        queue.async { [self] in
            publisher?.stop(); publisher = nil
            connection?.close(); connection = nil
            listener?.cancel(); listener = nil
        }
    }

    private func advertise(on port: UInt16) {
        var capabilities = NMPSystemCapabilityProbe.measure(peerID: peerID)
        capabilities.udpPort = port
        capabilities.noiseStaticPublicKey = staticKeys.publicKeyData
        capabilities.modelFormats = ["gguf"]

        // Bonjour anchor on the SAME port number as the UDP listener — see
        // the header comment; this is what makes the peer dialable from a
        // bare browse result.
        let publisher = NMPBonjourPublisher(capabilities: capabilities, port: port)
        self.publisher = publisher
        do {
            try publisher.start()
            onStatus?("advertising \(NMPBonjour.serviceName(for: peerID)) on UDP port \(port)")
        } catch {
            onStatus?("Bonjour publish failed: \(error)")
        }
    }

    private func accept(transport: UDPTransport, from endpoint: NWEndpoint) {
        // One coordinator at a time; a new inbound flow replaces the old
        // session (covers coordinator restarts without peer restarts).
        if connection != nil {
            onDiagnostic?("replacing existing coordinator session (new flow from \(endpoint))")
            connection?.close()
        }
        do {
            let connection = try PeerConnection(
                role: .responder,
                config: PeerConnectionConfig(localPeerID: peerID),
                transport: transport,
                localStatic: staticKeys,
                queue: queue)
            self.connection = connection

            let shardEngine = NMPPeerShardEngine(
                connection: connection, engine: engine,
                modelTag: modelTag, localPeerID: peerID)
            self.shardEngine = shardEngine
            shardEngine.onAssigned = { [weak self] assign in
                self?.assignment = assign
                self?.onStatus?("assigned shard \(assign.shardIndex): layers "
                                + "\(assign.startLayer)..<\(assign.endLayer) of \(assign.totalLayers)")
                self?.onAssigned?(assign)
            }
            shardEngine.onInferenceServed = { [weak self] requestID, layers, seconds in
                self?.servedCount += 1
                self?.onServed?(requestID, layers, seconds)
            }
            shardEngine.onDiagnostic = { [weak self] message in
                self?.onDiagnostic?(message)
            }
            shardEngine.activate()

            connection.onEstablished = { [weak self] _, remoteID in
                self?.onStatus?("coordinator \(String(format: "%08x", remoteID)) connected")
            }
            connection.onFailed = { [weak self] error in
                self?.onStatus?("coordinator session failed: \(error)")
            }
            connection.start()
        } catch {
            onStatus?("failed to accept coordinator connection: \(error)")
        }
    }
}

// MARK: - Coordinator node (runs on the Mac)

public final class NMPCoordinatorNode {

    // MARK: Callbacks (invoked on `queue`)

    public var onStatus: ((String) -> Void)?
    /// A peer finished the NMP handshake and is attachable for sharding.
    public var onPeerReady: ((NMPCapabilities) -> Void)?
    public var onPeerLost: ((UInt32) -> Void)?
    public var onPeerMetrics: ((NMPPeerMetrics) -> Void)?

    // MARK: State

    public let localPeerID: UInt32
    public let orchestrator: NMPInferenceOrchestrator
    public private(set) var readyPeers: [UInt32: NMPCapabilities] = [:]

    private let engine: NMPShardComputeEngine
    private let queue: DispatchQueue
    private let staticKeys = NoiseStaticKeyPair()
    private var discovery: NMPPeerDiscoveryManager?
    private var browser: NMPBonjourBrowser?
    private var connections: [UInt32: PeerConnection] = [:]
    private var connectionQueues: [UInt32: DispatchQueue] = [:]
    private var localCapabilities: NMPCapabilities

    public init(
        engine: NMPShardComputeEngine,
        modelTag: String,
        localPeerID: UInt32 = UInt32.random(in: 1...UInt32.max),
        queue: DispatchQueue = DispatchQueue(label: "nmp.coordinator.node")
    ) {
        self.engine = engine
        self.localPeerID = localPeerID
        self.queue = queue
        self.localCapabilities = NMPSystemCapabilityProbe.measure(peerID: localPeerID)
        self.orchestrator = NMPInferenceOrchestrator(
            localPeerID: localPeerID, engine: engine, modelTag: modelTag, queue: queue)
        orchestrator.onPeerMetrics = { [weak self] metrics in
            self?.onPeerMetrics?(metrics)
        }
    }

    public func start() throws {
        dispatchPrecondition(condition: .onQueue(queue))
        let browser = NMPBonjourBrowser()
        self.browser = browser
        let discovery = NMPPeerDiscoveryManager(
            localCapabilities: localCapabilities,
            publisher: nil, // browse-only: the coordinator dials, peers advertise
            source: browser,
            queue: queue)
        self.discovery = discovery

        discovery.onPeerDiscovered = { [weak self] capabilities in
            self?.dialIfPossible(capabilities)
        }
        discovery.onPeerRemoved = { [weak self] peerID in
            guard let self else { return }
            if self.readyPeers.removeValue(forKey: peerID) != nil {
                self.orchestrator.detachPeer(peerID: peerID)
                self.connections.removeValue(forKey: peerID)?.close()
                self.connectionQueues.removeValue(forKey: peerID)
                self.onPeerLost?(peerID)
            }
        }
        try discovery.start()
        onStatus?("browsing for \(NMPBonjour.serviceType) peers …")
    }

    public func stop() {
        queue.async { [self] in
            discovery?.stop()
            for connection in connections.values { connection.close() }
            connections.removeAll()
            readyPeers.removeAll()
        }
    }

    private func dialIfPossible(_ capabilities: NMPCapabilities) {
        guard connections[capabilities.peerID] == nil else { return }
        guard capabilities.udpPort != 0,
              let remoteStatic = capabilities.noiseStaticPublicKey else {
            onStatus?("peer \(String(format: "%08x", capabilities.peerID)) "
                      + "not dialable (no port/key in TXT) — skipping")
            return
        }
        guard let endpoint = discovery?.discoveredPeers[capabilities.peerID]?.endpoint else {
            onStatus?("peer \(String(format: "%08x", capabilities.peerID)) has no endpoint yet")
            return
        }

        onStatus?("dialing \(capabilities.deviceName) "
                  + "(\(String(format: "%08x", capabilities.peerID)), "
                  + "\(capabilities.computeClass.label), port \(capabilities.udpPort)) …")

        let connectionQueue = DispatchQueue(
            label: "nmp.coordinator.link.\(capabilities.peerID)")
        let transport = UDPTransport(endpoint: endpoint, queue: connectionQueue)
        do {
            let connection = try PeerConnection(
                role: .initiator,
                config: PeerConnectionConfig(localPeerID: localPeerID),
                transport: transport,
                localStatic: staticKeys,
                remoteStaticPublicKey: remoteStatic,
                queue: connectionQueue)
            connections[capabilities.peerID] = connection
            connectionQueues[capabilities.peerID] = connectionQueue

            connection.onEstablished = { [weak self] _, remoteID in
                self?.queue.async {
                    guard let self else { return }
                    self.readyPeers[remoteID] = capabilities
                    self.orchestrator.attachPeer(peerID: remoteID, connection: connection)
                    self.onStatus?("peer \(String(format: "%08x", remoteID)) established "
                                   + "(handshake complete)")
                    self.onPeerReady?(capabilities)
                }
            }
            connection.onFailed = { [weak self] error in
                self?.queue.async {
                    guard let self else { return }
                    self.onStatus?("peer \(String(format: "%08x", capabilities.peerID)) "
                                   + "failed: \(error)")
                    self.connections.removeValue(forKey: capabilities.peerID)
                    if self.readyPeers.removeValue(forKey: capabilities.peerID) != nil {
                        self.orchestrator.detachPeer(peerID: capabilities.peerID)
                        self.onPeerLost?(capabilities.peerID)
                    }
                }
            }
            connection.start()
        } catch {
            onStatus?("failed to dial peer: \(error)")
        }
    }

    /// Plans over the local device + all ready peers, then broadcasts.
    public func planAndAssign(
        completion: @escaping (Result<[NMPShardPlanEntry], NMPOrchestrationError>) -> Void
    ) {
        queue.async { [self] in
            let members = [localCapabilities] + readyPeers.values
            let plan = NMPModelSharder.plan(
                layerCount: engine.layerCount,
                peers: members,
                measuredSecondsPerLayer: orchestrator.measuredSecondsPerLayer)
            orchestrator.assignShards(plan) { result in
                completion(result.map { plan })
            }
        }
    }
}
