//
//  LlamaTestbed.swift
//  NMP — Phase 8 + Phase 10 (cross-device sharding)
//
//  Single-shard mesh assembly for real-LLM engines: one coordinator and
//  (optionally) one in-process shard peer over an in-memory link running
//  the REAL stack — Noise IK handshake, AES-GCM, sequencing, FEC, NACK,
//  loss injection. A llama plan has exactly one full-range shard (see
//  LlamaEngine.swift), so this is the llama counterpart of NMPMeshTestbed:
//
//    .local      — the shard computes inline on the coordinator: the
//                  single-device baseline.
//    .remotePeer — the shard lives behind the link: every token pass
//                  crosses the full transport stack both ways, exactly
//                  what a physical peer costs minus the radio.
//    .sharded    — Phase 10: N peers, each loading the full model,
//                  each assigned a LAYER SUB-RANGE. Proves the
//                  cross-device sharding path end-to-end.
//
//  The engine is protocol-typed: tests drive this with the reference
//  engine too (full-range plans are legal for every engine).
//
//  Blocking style: `startSync`/`inferSync` block the calling thread —
//  call from a plain thread (CLI main, XCTest), never a mesh queue.
//

import Foundation

public final class NMPLlamaTestbed {

    public enum Placement: Sendable, Equatable {
        /// Shard computes inline on the coordinator (single device).
        case local
        /// Shard behind an in-memory link with the full protocol stack.
        case remotePeer
        /// Phase 10: N peers, each with a layer sub-range.
        case sharded(shardCount: Int)
    }

    public let coordinatorID: UInt32 = 0x0000_0001
    public let peerID: UInt32 = 0x0000_0002
    public let placement: Placement
    public let modelTag: String
    public let orchestrator: NMPInferenceOrchestrator
    public var hiddenSize: Int { engine.hiddenSize }
    public var layerCount: Int { engine.layerCount }

    /// Loss-recovery events from both ends of the link (remotePeer only).
    public var onPacketEvent: ((NMPPacketEvent) -> Void)?
    /// Fires after the peer serves each request. The first parameter is the peerID of the peer.
    public var onInferenceServed: ((UInt32, Range<Int>, TimeInterval) -> Void)?

    /// Mesh 2.3: coordinator-side wire totals for the link to the shard
    /// peer (nil when .local — there is no link). sent = toward the peer.
    public var wireTraffic: (sentBytes: UInt64, receivedBytes: UInt64)? {
        coordinatorSide?.trafficTotals
    }

    private let engine: NMPShardComputeEngine
    private let orchestratorQueue = DispatchQueue(label: "nmp.llama.testbed")

    // remotePeer plumbing (nil when .local).
    private var coordinatorSide: PeerConnection?
    private var peerSide: PeerConnection?
    private var shardEngine: NMPPeerShardEngine?
    private var coordinatorInjector: NMPPacketLossInjector?
    private var peerInjector: NMPPacketLossInjector?

    // Phase 10: sharded peers (empty for .local / .remotePeer).
    private struct ShardedPeer {
        let peerID: UInt32
        let coordinatorSide: PeerConnection
        let peerSide: PeerConnection
        let shardEngine: NMPPeerShardEngine
        let coordinatorInjector: NMPPacketLossInjector
        let peerInjector: NMPPacketLossInjector
    }
    private var shardedPeers: [ShardedPeer] = []
    /// Engine instances for sharded peers (retained to keep them alive).
    private var shardedEngines: [NMPShardComputeEngine] = []
    /// Factory for creating additional engine instances (sharded mode).
    private let engineFactory: (() throws -> NMPShardComputeEngine)?

    /// Builds the mesh and (for .remotePeer/.sharded) completes handshakes.
    public init(engine: NMPShardComputeEngine, modelTag: String,
                placement: Placement, handshakeTimeout: TimeInterval = 5,
                engineFactory: (() throws -> NMPShardComputeEngine)? = nil) throws {
        self.engine = engine
        self.modelTag = modelTag
        self.placement = placement
        self.engineFactory = engineFactory

        orchestrator = NMPInferenceOrchestrator(
            localPeerID: coordinatorID, engine: engine,
            modelTag: modelTag, queue: orchestratorQueue)

        switch placement {
        case .local:
            break   // nothing to wire up

        case .remotePeer:
            let (rawCoordinator, rawPeer) = NMPInMemoryTransport.pair(label: "nmp.llama.link")
            let coordinatorInjector = NMPPacketLossInjector(
                wrapping: rawCoordinator, seed: 0xC0DE_11A3)
            let peerInjector = NMPPacketLossInjector(
                wrapping: rawPeer, seed: 0xFEED_11A3)
            self.coordinatorInjector = coordinatorInjector
            self.peerInjector = peerInjector

            let coordinatorStatic = NoiseStaticKeyPair()
            let peerStatic = NoiseStaticKeyPair()
            let coordinatorSide = try PeerConnection(
                role: .initiator,
                config: PeerConnectionConfig(localPeerID: coordinatorID),
                transport: coordinatorInjector,
                localStatic: coordinatorStatic,
                remoteStaticPublicKey: peerStatic.publicKeyData,
                queue: DispatchQueue(label: "nmp.llama.link.coordinator"))
            let peerSide = try PeerConnection(
                role: .responder,
                config: PeerConnectionConfig(localPeerID: peerID),
                transport: peerInjector,
                localStatic: peerStatic,
                queue: DispatchQueue(label: "nmp.llama.link.peer"))
            self.coordinatorSide = coordinatorSide
            self.peerSide = peerSide

            let shardEngine = NMPPeerShardEngine(
                connection: peerSide, engine: engine,
                modelTag: modelTag, localPeerID: peerID)
            let currentPeerID = peerID
            shardEngine.onDiagnostic = { print("[LlamaTestbed Peer \(currentPeerID)] \($0)") }
            shardEngine.onInferenceServed = { [weak self] requestID, layers, seconds in
                self?.onInferenceServed?(currentPeerID, layers, seconds)
            }
            shardEngine.activate()
            self.shardEngine = shardEngine

            coordinatorSide.onPacketEvent = { [weak self] event in
                self?.onPacketEvent?(event)
            }
            peerSide.onPacketEvent = { [weak self] event in
                self?.onPacketEvent?(event)
            }

            let established = DispatchSemaphore(value: 0)
            coordinatorSide.onEstablished = { _, _ in established.signal() }
            peerSide.onEstablished = { _, _ in established.signal() }
            peerSide.start()
            coordinatorSide.start()
            for _ in 0..<2 {
                guard established.wait(timeout: .now() + handshakeTimeout) == .success else {
                    throw NMPMeshTestbedError.handshakeTimeout(peerID: peerID)
                }
            }
            orchestrator.attachPeer(peerID: peerID, connection: coordinatorSide)

        case .sharded(let shardCount):
            guard shardCount >= 2 else {
                throw NMPMeshTestbedError.orchestration(
                    NMPOrchestrationError.emptyPlan)
            }
            guard let factory = engineFactory else {
                throw NMPMeshTestbedError.orchestration(
                    NMPOrchestrationError.emptyPlan)
            }
            // Create N in-memory peers, each with its own engine.
            var nextPeerID: UInt32 = 0x0000_0002
            for i in 0..<shardCount {
                let thisPeerID = nextPeerID
                nextPeerID += 1

                let peerEngine: NMPShardComputeEngine
                if i == 0 {
                    // The first shard can reuse the provided engine.
                    peerEngine = engine
                } else {
                    peerEngine = try factory()
                }
                shardedEngines.append(peerEngine)

                let label = "nmp.llama.shard.\(i)"
                let (rawCoord, rawPeer) = NMPInMemoryTransport.pair(label: label)
                let coordInj = NMPPacketLossInjector(
                    wrapping: rawCoord, seed: 0xC0DE_0000 | UInt64(thisPeerID))
                let peerInj = NMPPacketLossInjector(
                    wrapping: rawPeer, seed: 0xFEED_0000 | UInt64(thisPeerID))

                let coordStatic = NoiseStaticKeyPair()
                let peerStatic = NoiseStaticKeyPair()
                let coordConn = try PeerConnection(
                    role: .initiator,
                    config: PeerConnectionConfig(localPeerID: coordinatorID),
                    transport: coordInj,
                    localStatic: coordStatic,
                    remoteStaticPublicKey: peerStatic.publicKeyData,
                    queue: DispatchQueue(label: "\(label).coordinator"))
                let peerConn = try PeerConnection(
                    role: .responder,
                    config: PeerConnectionConfig(localPeerID: thisPeerID),
                    transport: peerInj,
                    localStatic: peerStatic,
                    queue: DispatchQueue(label: "\(label).peer"))

                let se = NMPPeerShardEngine(
                    connection: peerConn, engine: peerEngine,
                    modelTag: modelTag, localPeerID: thisPeerID)
                se.onDiagnostic = { print("[LlamaTestbed Peer \(thisPeerID)] \($0)") }
                se.onInferenceServed = { [weak self] requestID, layers, seconds in
                    self?.onInferenceServed?(thisPeerID, layers, seconds)
                }
                se.activate()

                let established = DispatchSemaphore(value: 0)
                coordConn.onEstablished = { _, _ in established.signal() }
                peerConn.onEstablished = { _, _ in established.signal() }
                peerConn.start()
                coordConn.start()
                for _ in 0..<2 {
                    guard established.wait(timeout: .now() + handshakeTimeout) == .success else {
                        throw NMPMeshTestbedError.handshakeTimeout(peerID: thisPeerID)
                    }
                }

                orchestrator.attachPeer(peerID: thisPeerID, connection: coordConn)
                shardedPeers.append(ShardedPeer(
                    peerID: thisPeerID,
                    coordinatorSide: coordConn, peerSide: peerConn,
                    shardEngine: se,
                    coordinatorInjector: coordInj, peerInjector: peerInj))
            }
        }
    }

    /// The shard plan this mesh runs.
    public var plan: [NMPShardPlanEntry] {
        switch placement {
        case .local:
            return [.fullRange(peerID: coordinatorID, layerCount: engine.layerCount)]
        case .remotePeer:
            return [.fullRange(peerID: peerID, layerCount: engine.layerCount)]
        case .sharded(let shardCount):
            // Split layers evenly across the shard peers.
            let totalLayers = engine.layerCount
            var entries: [NMPShardPlanEntry] = []
            var nextLayer = 0
            for i in 0..<shardCount {
                let base = totalLayers / shardCount
                let extra = i < (totalLayers % shardCount) ? 1 : 0
                let span = base + extra
                entries.append(NMPShardPlanEntry(
                    peerID: shardedPeers[i].peerID,
                    shardIndex: i,
                    startLayer: nextLayer,
                    endLayer: nextLayer + span))
                nextLayer += span
            }
            return entries
        }
    }

    /// Assigns the plan; blocks until the peer acks (or the plan is local).
    @discardableResult
    public func startSync(timeout: TimeInterval = 10) throws -> [NMPShardPlanEntry] {
        let done = DispatchSemaphore(value: 0)
        var failure: NMPOrchestrationError?
        orchestrator.assignShards(plan, timeout: timeout) { result in
            if case .failure(let error) = result { failure = error }
            done.signal()
        }
        guard done.wait(timeout: .now() + timeout + 2) == .success else {
            throw NMPMeshTestbedError.failoverTimeout
        }
        if let failure { throw NMPMeshTestbedError.orchestration(failure) }
        return plan
    }

    /// One pipeline pass; blocks until the output tensor is back.
    public func inferSync(input: [Float],
                          stageTimeout: TimeInterval = 120) throws -> NMPInferenceReport {
        let done = DispatchSemaphore(value: 0)
        var outcome: Result<NMPInferenceReport, NMPOrchestrationError>?
        orchestrator.infer(input: input, stageTimeout: stageTimeout) { result in
            outcome = result
            done.signal()
        }
        guard done.wait(timeout: .now() + stageTimeout * 2 + 5) == .success,
              let outcome else {
            throw NMPMeshTestbedError.inferenceTimeout
        }
        switch outcome {
        case .success(let report): return report
        case .failure(let error): throw NMPMeshTestbedError.orchestration(error)
        }
    }

    /// Steady loss on the link, both directions (remotePeer only).
    public func setLossRate(_ rate: Double) {
        coordinatorInjector?.setLossRate(rate)
        peerInjector?.setLossRate(rate)
    }
}
