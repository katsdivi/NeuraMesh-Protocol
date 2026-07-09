//
//  LlamaTestbed.swift
//  NMP — Phase 8
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
//
//  The engine is protocol-typed: tests drive this with the reference
//  engine too (full-range plans are legal for every engine).
//
//  Blocking style: `startSync`/`inferSync` block the calling thread —
//  call from a plain thread (CLI main, XCTest), never a mesh queue.
//

import Foundation

public final class NMPLlamaTestbed {

    public enum Placement: String, Sendable {
        /// Shard computes inline on the coordinator (single device).
        case local
        /// Shard behind an in-memory link with the full protocol stack.
        case remotePeer
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
    /// Fires after the peer serves each request (remotePeer only).
    public var onInferenceServed: ((UInt32, Range<Int>, TimeInterval) -> Void)?

    private let engine: NMPShardComputeEngine
    private let orchestratorQueue = DispatchQueue(label: "nmp.llama.testbed")

    // remotePeer plumbing (nil when .local).
    private var coordinatorSide: PeerConnection?
    private var peerSide: PeerConnection?
    private var shardEngine: NMPPeerShardEngine?
    private var coordinatorInjector: NMPPacketLossInjector?
    private var peerInjector: NMPPacketLossInjector?

    /// Builds the mesh and (for .remotePeer) completes the handshake.
    public init(engine: NMPShardComputeEngine, modelTag: String,
                placement: Placement, handshakeTimeout: TimeInterval = 5) throws {
        self.engine = engine
        self.modelTag = modelTag
        self.placement = placement

        orchestrator = NMPInferenceOrchestrator(
            localPeerID: coordinatorID, engine: engine,
            modelTag: modelTag, queue: orchestratorQueue)

        guard placement == .remotePeer else { return }

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
        shardEngine.onInferenceServed = { [weak self] requestID, layers, seconds in
            self?.onInferenceServed?(requestID, layers, seconds)
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
    }

    /// The single-shard plan this mesh runs.
    public var plan: [NMPShardPlanEntry] {
        [.fullRange(peerID: placement == .local ? coordinatorID : peerID,
                    layerCount: engine.layerCount)]
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
