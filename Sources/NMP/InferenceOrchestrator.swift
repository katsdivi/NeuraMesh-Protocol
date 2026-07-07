//
//  InferenceOrchestrator.swift
//  NMP — Phase 5
//
//  The coordinator: owns one PeerConnection per remote shard peer plus a
//  local compute engine, broadcasts the shard plan, and walks inference
//  requests down the pipeline.
//
//  Topology: STAR RELAY. Activations flow coordinator → shard i →
//  coordinator → shard i+1 → … The build-prompt diagram sketches direct
//  peer→peer forwarding; star relay is the deliberate Phase 5 choice —
//  it needs no peer↔peer key material or connections (N links instead of
//  N²), keeps every timing measurement in one place, and the extra hop
//  costs one LAN RTT per stage. Peer→peer forwarding is a Phase 6+
//  optimization once fault tolerance exists.
//
//  Threading: callback style throughout (spec tech-stack rule: no
//  async/await). The orchestrator runs on its own serial queue; each
//  PeerConnection keeps its own queue, and results hop back via
//  `queue.async`. Local shards compute inline on the orchestrator queue —
//  fine while inference is strictly sequential (one request in flight).
//

import Foundation

// MARK: - Errors

public enum NMPOrchestrationError: Error, Equatable, Sendable {
    case emptyPlan
    case peerNotConnected(UInt32)
    case assignmentRejected(peerID: UInt32, status: NMPShardAck.Status)
    case assignmentTimeout(unacked: [UInt32])
    case notAssigned
    case inputWidthMismatch(expected: Int, got: Int)
    case inferenceTimeout(peerID: UInt32, requestID: UInt32)
    case peerReportedFailure(peerID: UInt32, status: NMPInferResponseMeta.Status)
    case sendFailed(String)
}

// MARK: - Report

/// Everything measured during one pipelined inference.
public struct NMPInferenceReport: Sendable {
    public struct ShardTiming: Sendable {
        public let peerID: UInt32
        public let shardIndex: Int
        public let layers: Range<Int>
        /// Pure compute time (peer-reported for remote shards).
        public let computeSeconds: TimeInterval
        /// Full stage time as seen by the coordinator (compute + network
        /// + chunking). For local shards this equals computeSeconds.
        public let stageSeconds: TimeInterval
        public let isLocal: Bool
    }

    public let output: [Float]
    public let totalSeconds: TimeInterval
    public let perShard: [ShardTiming]
    /// Application payload bytes moved over the network (both directions),
    /// for the transport-overhead metric (bytes / |output tensor|).
    public let networkPayloadBytes: Int

    /// Aggregate network share of the total wall clock.
    public var networkSeconds: TimeInterval {
        perShard.filter { !$0.isLocal }
            .reduce(0) { $0 + ($1.stageSeconds - $1.computeSeconds) }
    }
}

// MARK: - Orchestrator

public final class NMPInferenceOrchestrator {

    // MARK: Callbacks (invoked on the orchestrator queue)

    /// Latest metrics packet from a shard peer.
    public var onPeerMetrics: ((NMPPeerMetrics) -> Void)?
    public var onDiagnostic: ((String) -> Void)?
    /// Phase 6: fires for every decrypted packet a shard peer sends us —
    /// the liveness signal `NMPPeerHealthMonitor` feeds on.
    public var onPeerActivity: ((UInt32) -> Void)?

    // MARK: State

    public private(set) var plan: [NMPShardPlanEntry] = []
    /// Phase 6: how many times a remote stage that hit `inferenceTimeout`
    /// is re-sent before the whole inference fails. Under sustained heavy
    /// loss the NACK layer can exhaust its attempts and give a chunk up
    /// for dead (by design — see Reliability.swift); a fresh request with
    /// a new requestID recovers at the cost of one extra stage timeout.
    public var stageRetryLimit = 1
    /// Peer-reported seconds-per-layer from completed inferences; feed
    /// back into `NMPModelSharder.plan(measuredSecondsPerLayer:)` to
    /// re-balance the next assignment round.
    public private(set) var measuredSecondsPerLayer: [UInt32: Double] = [:]

    private let localPeerID: UInt32
    private let engine: NMPShardComputeEngine
    private let modelTag: String
    private let queue: DispatchQueue

    private var connections: [UInt32: PeerConnection] = [:]
    private var reassemblers: [UInt32: NMPTensorReassembler] = [:]

    // Assignment round state.
    private var pendingAcks: Set<UInt32> = []
    private var assignCompletion: ((Result<Void, NMPOrchestrationError>) -> Void)?
    private var assignTimer: DispatchSourceTimer?

    // In-flight remote request state (one at a time; pipeline is serial).
    private struct PendingRequest {
        let peerID: UInt32
        var responseMeta: NMPInferResponseMeta?
        let completion: (Result<(tensor: Data, meta: NMPInferResponseMeta),
                                NMPOrchestrationError>) -> Void
        var timer: DispatchSourceTimer?
    }
    private var pendingRequests: [UInt32: PendingRequest] = [:]
    private var nextRequestID: UInt32 = 1

    public init(
        localPeerID: UInt32,
        engine: NMPShardComputeEngine,
        modelTag: String,
        queue: DispatchQueue
    ) {
        self.localPeerID = localPeerID
        self.engine = engine
        self.modelTag = modelTag
        self.queue = queue
    }

    // MARK: Peer wiring

    /// Registers an ESTABLISHED connection to a shard peer and takes over
    /// its packet stream for response/ack/metrics handling.
    public func attachPeer(peerID: UInt32, connection: PeerConnection) {
        queue.async { [self] in
            connections[peerID] = connection
            reassemblers[peerID] = NMPTensorReassembler()
            connection.onPacket = { [weak self] packet in
                // Hop from the connection queue to the orchestrator queue.
                self?.queue.async { self?.handle(packet, from: peerID) }
            }
        }
    }

    public func detachPeer(peerID: UInt32) {
        queue.async { [self] in
            connections.removeValue(forKey: peerID)
            reassemblers.removeValue(forKey: peerID)
            // Fail anything in flight toward that peer.
            for (requestID, pending) in pendingRequests where pending.peerID == peerID {
                completeRequest(requestID,
                                with: .failure(.peerNotConnected(peerID)))
            }
        }
    }

    // MARK: Shard assignment

    /// Broadcasts SHARD_ASSIGN for every remote entry in `newPlan` and
    /// waits for all acks. Local entries need no ack. Completion fires on
    /// the orchestrator queue.
    public func assignShards(
        _ newPlan: [NMPShardPlanEntry],
        timeout: TimeInterval = 10,
        completion: @escaping (Result<Void, NMPOrchestrationError>) -> Void
    ) {
        queue.async { [self] in
            guard !newPlan.isEmpty else {
                completion(.failure(.emptyPlan))
                return
            }
            let remote = newPlan.filter { $0.peerID != localPeerID }
            for entry in remote where connections[entry.peerID] == nil {
                completion(.failure(.peerNotConnected(entry.peerID)))
                return
            }

            plan = newPlan
            pendingAcks = Set(remote.map(\.peerID))
            assignCompletion = completion

            guard !pendingAcks.isEmpty else {
                assignCompletion = nil
                completion(.success(()))
                return
            }

            for entry in remote {
                let assign = NMPShardAssign(
                    shardIndex: UInt16(entry.shardIndex),
                    pipelineLength: UInt16(newPlan.count),
                    startLayer: UInt16(entry.startLayer),
                    endLayer: UInt16(entry.endLayer),
                    totalLayers: UInt16(engine.layerCount),
                    hiddenSize: UInt32(engine.hiddenSize),
                    modelTag: modelTag)
                connections[entry.peerID]?.sendAsync(
                    packetType: .shardAssign,
                    priority: .critical,
                    payload: assign.encode()
                ) { [weak self] result in
                    if case .failure(let error) = result {
                        self?.queue.async {
                            self?.finishAssignment(.failure(.sendFailed(String(describing: error))))
                        }
                    }
                }
            }

            let timer = DispatchSource.makeTimerSource(queue: queue)
            timer.schedule(deadline: .now() + timeout)
            timer.setEventHandler { [weak self] in
                guard let self, !self.pendingAcks.isEmpty else { return }
                self.finishAssignment(.failure(
                    .assignmentTimeout(unacked: self.pendingAcks.sorted())))
            }
            timer.resume()
            assignTimer = timer
        }
    }

    private func finishAssignment(_ result: Result<Void, NMPOrchestrationError>) {
        assignTimer?.cancel(); assignTimer = nil
        pendingAcks.removeAll()
        let completion = assignCompletion
        assignCompletion = nil
        completion?(result)
    }

    // MARK: Inference

    /// Runs one activation vector through the full pipeline. Completion
    /// fires on the orchestrator queue.
    public func infer(
        input: [Float],
        stageTimeout: TimeInterval = 30,
        completion: @escaping (Result<NMPInferenceReport, NMPOrchestrationError>) -> Void
    ) {
        queue.async { [self] in
            guard !plan.isEmpty else {
                completion(.failure(.emptyPlan))
                return
            }
            guard input.count == engine.hiddenSize else {
                completion(.failure(.inputWidthMismatch(
                    expected: engine.hiddenSize, got: input.count)))
                return
            }
            let began = DispatchTime.now()
            runStage(0, activations: input, timings: [], payloadBytes: 0,
                     stageTimeout: stageTimeout) { result in
                switch result {
                case .failure(let error):
                    completion(.failure(error))
                case .success(let (output, timings, bytes)):
                    let total = TimeInterval(
                        DispatchTime.now().uptimeNanoseconds - began.uptimeNanoseconds) / 1e9
                    completion(.success(NMPInferenceReport(
                        output: output, totalSeconds: total,
                        perShard: timings, networkPayloadBytes: bytes)))
                }
            }
        }
    }

    private typealias StageResult =
        Result<([Float], [NMPInferenceReport.ShardTiming], Int), NMPOrchestrationError>

    private func runStage(
        _ index: Int,
        activations: [Float],
        timings: [NMPInferenceReport.ShardTiming],
        payloadBytes: Int,
        stageTimeout: TimeInterval,
        completion: @escaping (StageResult) -> Void
    ) {
        guard index < plan.count else {
            completion(.success((activations, timings, payloadBytes)))
            return
        }
        let entry = plan[index]
        let stageBegan = DispatchTime.now()

        if entry.peerID == localPeerID {
            do {
                let output = try engine.runLayers(
                    start: entry.startLayer, end: entry.endLayer, input: activations)
                let seconds = TimeInterval(
                    DispatchTime.now().uptimeNanoseconds - stageBegan.uptimeNanoseconds) / 1e9
                recordMeasurement(peerID: entry.peerID, seconds: seconds, layers: entry.layerSpan)
                let timing = NMPInferenceReport.ShardTiming(
                    peerID: entry.peerID, shardIndex: entry.shardIndex,
                    layers: entry.startLayer..<entry.endLayer,
                    computeSeconds: seconds, stageSeconds: seconds, isLocal: true)
                runStage(index + 1, activations: output, timings: timings + [timing],
                         payloadBytes: payloadBytes, stageTimeout: stageTimeout,
                         completion: completion)
            } catch {
                completion(.failure(.sendFailed("local compute: \(error)")))
            }
            return
        }

        sendRemoteWithRetry(entry: entry, activations: activations,
                            timeout: stageTimeout,
                            retriesLeft: stageRetryLimit) { [self] result in
            switch result {
            case .failure(let error):
                completion(.failure(error))
            case .success(let (tensorBytes, meta, sentBytes)):
                let output: [Float]
                do {
                    output = try NMPTensorCodec.decode(tensorBytes)
                } catch {
                    completion(.failure(.sendFailed("response decode: \(error)")))
                    return
                }
                let stageSeconds = TimeInterval(
                    DispatchTime.now().uptimeNanoseconds - stageBegan.uptimeNanoseconds) / 1e9
                let computeSeconds = TimeInterval(meta.computeMicros) / 1e6
                recordMeasurement(peerID: entry.peerID, seconds: computeSeconds,
                                  layers: entry.layerSpan)
                let timing = NMPInferenceReport.ShardTiming(
                    peerID: entry.peerID, shardIndex: entry.shardIndex,
                    layers: entry.startLayer..<entry.endLayer,
                    computeSeconds: computeSeconds, stageSeconds: stageSeconds,
                    isLocal: false)
                runStage(index + 1, activations: output, timings: timings + [timing],
                         payloadBytes: payloadBytes + sentBytes + tensorBytes.count,
                         stageTimeout: stageTimeout, completion: completion)
            }
        }
    }

    /// Retries a timed-out stage with a fresh requestID. A late response
    /// to the abandoned request is ignored (its pendingRequests entry is
    /// gone), so a retry can never deliver stale activations.
    private func sendRemoteWithRetry(
        entry: NMPShardPlanEntry,
        activations: [Float],
        timeout: TimeInterval,
        retriesLeft: Int,
        completion: @escaping (Result<(Data, NMPInferResponseMeta, Int),
                                      NMPOrchestrationError>) -> Void
    ) {
        sendRemote(entry: entry, activations: activations, timeout: timeout) { [self] result in
            if case .failure(.inferenceTimeout) = result, retriesLeft > 0 {
                onDiagnostic?("stage \(entry.shardIndex) (peer \(entry.peerID)) timed out; "
                              + "retrying (\(retriesLeft) attempt(s) left)")
                sendRemoteWithRetry(entry: entry, activations: activations,
                                    timeout: timeout, retriesLeft: retriesLeft - 1,
                                    completion: completion)
                return
            }
            completion(result)
        }
    }

    private func sendRemote(
        entry: NMPShardPlanEntry,
        activations: [Float],
        timeout: TimeInterval,
        completion: @escaping (Result<(Data, NMPInferResponseMeta, Int),
                                      NMPOrchestrationError>) -> Void
    ) {
        guard let connection = connections[entry.peerID] else {
            completion(.failure(.peerNotConnected(entry.peerID)))
            return
        }
        let requestID = nextRequestID
        nextRequestID &+= 1

        let tensor = NMPTensorCodec.encode(activations)
        let chunks: [NMPTensorChunk]
        do {
            chunks = try NMPTensorChunk.split(requestID: requestID, tensorBytes: tensor)
        } catch {
            completion(.failure(.sendFailed("chunking: \(error)")))
            return
        }
        let meta = NMPInferRequestMeta(
            requestID: requestID,
            startLayer: UInt16(entry.startLayer),
            endLayer: UInt16(entry.endLayer),
            totalBytes: UInt32(tensor.count),
            chunkCount: UInt16(chunks.count))

        pendingRequests[requestID] = PendingRequest(
            peerID: entry.peerID,
            responseMeta: nil,
            completion: { result in
                completion(result.map { ($0.tensor, $0.meta, tensor.count) })
            },
            timer: nil)

        connection.sendAsync(priority: .critical, payload: meta.encode(), completion: nil)
        for (index, chunk) in chunks.enumerated() {
            connection.sendAsync(
                flags: index == chunks.count - 1 ? [.flush] : [],
                priority: .critical,
                payload: chunk.encode(),
                completion: nil)
        }

        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + timeout)
        timer.setEventHandler { [weak self] in
            guard let self, self.pendingRequests[requestID] != nil else { return }
            self.reassemblers[entry.peerID]?.abandon(requestID: requestID)
            self.completeRequest(requestID, with: .failure(
                .inferenceTimeout(peerID: entry.peerID, requestID: requestID)))
        }
        timer.resume()
        pendingRequests[requestID]?.timer = timer
    }

    private func recordMeasurement(peerID: UInt32, seconds: TimeInterval, layers: Int) {
        guard layers > 0, seconds > 0 else { return }
        measuredSecondsPerLayer[peerID] = seconds / Double(layers)
    }

    // MARK: Inbound from shard peers

    private func handle(_ packet: NMPPacket, from peerID: UInt32) {
        onPeerActivity?(peerID)
        guard packet.header.packetType == .data else { return }
        let payload = packet.payload

        switch NMPMeshMessageKindOf(payload) {
        case .shardAck:
            handleAck(payload, from: peerID)
        case .inferResponseMeta:
            do {
                let meta = try NMPInferResponseMeta.decode(payload)
                guard var pending = pendingRequests[meta.requestID],
                      pending.peerID == peerID else { return }
                guard meta.status == .ok else {
                    completeRequest(meta.requestID, with: .failure(
                        .peerReportedFailure(peerID: peerID, status: meta.status)))
                    return
                }
                pending.responseMeta = meta
                pendingRequests[meta.requestID] = pending
                if let tensor = try reassemblers[peerID]?.setExpectation(
                    requestID: meta.requestID,
                    chunkCount: Int(meta.chunkCount),
                    totalBytes: Int(meta.totalBytes)) {
                    completeRequest(meta.requestID, with: .success((tensor, meta)))
                }
            } catch {
                onDiagnostic?("bad response meta from \(peerID): \(error)")
            }
        case .tensorChunk:
            do {
                let chunk = try NMPTensorChunk.decode(payload)
                guard let pending = pendingRequests[chunk.requestID],
                      pending.peerID == peerID else { return }
                if let tensor = try reassemblers[peerID]?.addChunk(chunk),
                   let meta = pending.responseMeta {
                    completeRequest(chunk.requestID, with: .success((tensor, meta)))
                }
            } catch {
                onDiagnostic?("bad response chunk from \(peerID): \(error)")
            }
        case .metrics:
            if let metrics = try? NMPPeerMetrics.decode(payload) {
                onPeerMetrics?(metrics)
            }
        default:
            onDiagnostic?("orchestrator ignoring mesh message from \(peerID)")
        }
    }

    private func handleAck(_ payload: Data, from peerID: UInt32) {
        guard let ack = try? NMPShardAck.decode(payload) else { return }
        guard pendingAcks.contains(peerID) else { return }
        if ack.status == .ready {
            pendingAcks.remove(peerID)
            if pendingAcks.isEmpty {
                finishAssignment(.success(()))
            }
        } else {
            finishAssignment(.failure(
                .assignmentRejected(peerID: peerID, status: ack.status)))
        }
    }

    private func completeRequest(
        _ requestID: UInt32,
        with result: Result<(tensor: Data, meta: NMPInferResponseMeta),
                            NMPOrchestrationError>
    ) {
        guard let pending = pendingRequests.removeValue(forKey: requestID) else { return }
        pending.timer?.cancel()
        pending.completion(result)
    }
}
