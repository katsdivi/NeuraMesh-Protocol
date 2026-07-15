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

/// One completed pipeline stage (Phase 9: stages are individually
/// drivable so the pipelined batch executor can overlap sequences).
public struct NMPStageResult: Sendable {
    public let output: [Float]
    public let timing: NMPInferenceReport.ShardTiming
    /// Application payload bytes moved for this stage (0 for local).
    public let payloadBytes: Int
}

// MARK: - Orchestrator

public final class NMPInferenceOrchestrator {

    // MARK: Callbacks (invoked on the orchestrator queue)

    /// Latest metrics packet from a shard peer.
    public var onPeerMetrics: ((NMPPeerMetrics) -> Void)?
    /// Mesh 2.3: a shard peer's own host resource sample (RAM/CPU/GPU/
    /// storage measured ON that peer). Real per-device telemetry for
    /// physical peers; in-process peers report the shared host.
    public var onPeerResourceReport: ((NMPPeerResourceReport) -> Void)?
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
    /// re-balance the next assignment round. Lock-protected: read by the
    /// failover orchestrator (its own queue) and the Phase 9 adaptive
    /// controller while stages write from the orchestrator queue.
    public var measuredSecondsPerLayer: [UInt32: Double] {
        measurementsLock.lock()
        defer { measurementsLock.unlock() }
        return measurements
    }

    /// Phase 9: seeds measurements from a persisted profile so a mesh can
    /// start with last session's balance instead of re-benchmarking.
    public func seedMeasurements(_ map: [UInt32: Double]) {
        measurementsLock.lock()
        defer { measurementsLock.unlock() }
        for (peerID, seconds) in map where seconds > 0 {
            measurements[peerID] = seconds
        }
    }

    /// Phase 9: activation wire format for remote stages. Peers mirror
    /// the request's format, and decode is magic-sniffed, so this can
    /// change between inferences. Keep .float32 toward pre-Phase 9 peers.
    public var activationWireFormat: NMPActivationWireFormat {
        get { measurementsLock.lock(); defer { measurementsLock.unlock() }
              return wireFormat }
        set { measurementsLock.lock(); defer { measurementsLock.unlock() }
              wireFormat = newValue }
    }

    // MARK: Mesh 2.1 — compute shares
    //
    // The Devices panel's allocation slider. A share s in (0, 1] means
    // "this device contributes s of its compute to the mesh": the planner
    // treats the device as 1/s slower, so a peer capped at 50% receives
    // ~half the layers on the next (re-)plan. Shares live HERE, separate
    // from measurements, because live traffic overwrites measurements
    // after every stage — a cap stored inside them would wash out on the
    // next pass.

    /// Current per-peer compute shares (peers absent = 1.0).
    public var computeShares: [UInt32: Double] {
        measurementsLock.lock()
        defer { measurementsLock.unlock() }
        return shares
    }

    /// Sets a peer's mesh compute share, clamped to 0.05...1.0. Takes
    /// effect on the next plan (call the failover orchestrator's
    /// `replan` to apply immediately). Scaling happens inside
    /// `NMPModelSharder.plan(computeShares:)` — the single place shares
    /// touch planning math.
    public func setComputeShare(_ share: Double, forPeer peerID: UInt32) {
        measurementsLock.lock()
        defer { measurementsLock.unlock() }
        let clamped = min(1.0, max(0.05, share))
        if clamped >= 1.0 {
            shares.removeValue(forKey: peerID)
        } else {
            shares[peerID] = clamped
        }
    }

    private var measurements: [UInt32: Double] = [:]
    private var shares: [UInt32: Double] = [:]
    private var wireFormat: NMPActivationWireFormat = .float32
    private var vaultEndpointValue = ""
    private let measurementsLock = NSLock()

    /// Future Plan #3: "host:port" of the coordinator's weight-vault HTTP
    /// server, stamped into every SHARD_ASSIGN so peers with no local model can
    /// stream only their layers. Empty ⇒ peers must already hold the model.
    public var vaultEndpoint: String {
        get { measurementsLock.lock(); defer { measurementsLock.unlock() }
              return vaultEndpointValue }
        set { measurementsLock.lock(); defer { measurementsLock.unlock() }
              vaultEndpointValue = newValue }
    }

    private let localPeerID: UInt32
    private let engine: NMPShardComputeEngine
    private let modelTag: String
    private let queue: DispatchQueue
    /// Local shards compute here instead of inline on `queue`, so a local
    /// stage cannot stall response handling for concurrently in-flight
    /// remote stages (Phase 9 pipelining). Serial: the engine seam is not
    /// re-entrant.
    private let localComputeQueue = DispatchQueue(label: "nmp.orchestrator.local-compute")

    private var connections: [UInt32: PeerConnection] = [:]
    private var reassemblers: [UInt32: NMPTensorReassembler] = [:]

    // Assignment round state.
    /// The plan of the in-flight assignment round — committed to `plan`
    /// only when every remote peer acks (see assignShards).
    private var pendingPlan: [NMPShardPlanEntry]?
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

    /// Mesh 2.4: tiny encrypted keepalive the peer echoes back. The echo
    /// counts as packet activity, letting the health monitor distinguish
    /// "idle because another stage is slow" from "dead". No-op for peers
    /// we hold no connection to (the local shard).
    public func sendKeepalivePing(to peerID: UInt32) {
        queue.async { [self] in
            guard let connection = connections[peerID] else { return }
            connection.sendAsync(priority: .critical,
                                 payload: NMPPeerPing().encode())
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
    /// waits for all acks. Local entries need no ack. `idlePeers` are mesh
    /// members that hold 0 layers this plan — they get a fire-and-forget
    /// zero-layer SHARD_ASSIGN (not ack-tracked) so their UI shows
    /// "standing by" instead of hanging on "waiting for coordinator".
    /// Completion fires on the orchestrator queue.
    public func assignShards(
        _ newPlan: [NMPShardPlanEntry],
        idlePeers: [UInt32] = [],
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

            // Notify standby members (fire-and-forget; never blocks the round).
            let vault = vaultEndpoint
            let idle = NMPShardAssign(
                shardIndex: 0, pipelineLength: UInt16(newPlan.count),
                startLayer: 0, endLayer: 0,
                totalLayers: UInt16(engine.layerCount),
                hiddenSize: UInt32(engine.hiddenSize), modelTag: modelTag,
                vaultEndpoint: vault)
            for peerID in idlePeers where peerID != localPeerID {
                connections[peerID]?.sendAsync(
                    packetType: .shardAssign, priority: .critical,
                    payload: idle.encode())
            }

            // Stage the plan; commit only once every remote peer acks.
            // Committing here (the old behavior) left a live divergence
            // when a peer never acked: callers kept the previous plan
            // (planAndAssign reported failure) while THIS routing table
            // already pointed at the unassigned peer — every generation
            // then died with notAssigned until the next successful
            // re-shard. On failure the old plan keeps serving.
            pendingPlan = newPlan
            pendingAcks = Set(remote.map(\.peerID))
            assignCompletion = completion

            guard !pendingAcks.isEmpty else {
                plan = newPlan
                pendingPlan = nil
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
                    modelTag: modelTag, vaultEndpoint: vault)
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
        if case .success = result, let staged = pendingPlan {
            plan = staged
        }
        pendingPlan = nil
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
        computeStageOnQueue(plan[index], activations: activations,
                            stageTimeout: stageTimeout) { [self] result in
            switch result {
            case .failure(let error):
                completion(.failure(error))
            case .success(let stage):
                runStage(index + 1, activations: stage.output,
                         timings: timings + [stage.timing],
                         payloadBytes: payloadBytes + stage.payloadBytes,
                         stageTimeout: stageTimeout, completion: completion)
            }
        }
    }

    /// Runs ONE pipeline stage (local or remote, with the Phase 6 stage
    /// retry). Public in Phase 9 so `NMPPipelinedBatchExecutor` can keep
    /// every stage of the plan busy with a different sequence. Completion
    /// fires on the orchestrator queue.
    public func computeStage(
        _ entry: NMPShardPlanEntry,
        activations: [Float],
        stageTimeout: TimeInterval = 30,
        completion: @escaping (Result<NMPStageResult, NMPOrchestrationError>) -> Void
    ) {
        queue.async { [self] in
            computeStageOnQueue(entry, activations: activations,
                                stageTimeout: stageTimeout, completion: completion)
        }
    }

    /// Must run on `queue`; completion fires on `queue`.
    private func computeStageOnQueue(
        _ entry: NMPShardPlanEntry,
        activations: [Float],
        stageTimeout: TimeInterval,
        completion: @escaping (Result<NMPStageResult, NMPOrchestrationError>) -> Void
    ) {
        let stageBegan = DispatchTime.now()

        if entry.peerID == localPeerID {
            // Off-queue so a long local stage never blocks the handling of
            // concurrently in-flight remote responses.
            localComputeQueue.async { [self] in
                let outcome: Result<[Float], NMPOrchestrationError>
                do {
                    outcome = .success(try engine.runLayers(
                        start: entry.startLayer, end: entry.endLayer, input: activations))
                } catch {
                    outcome = .failure(.sendFailed("local compute: \(error)"))
                }
                let seconds = TimeInterval(
                    DispatchTime.now().uptimeNanoseconds - stageBegan.uptimeNanoseconds) / 1e9
                queue.async { [self] in
                    switch outcome {
                    case .failure(let error):
                        completion(.failure(error))
                    case .success(let output):
                        recordMeasurement(peerID: entry.peerID, seconds: seconds,
                                          layers: entry.layerSpan)
                        completion(.success(NMPStageResult(
                            output: output,
                            timing: NMPInferenceReport.ShardTiming(
                                peerID: entry.peerID, shardIndex: entry.shardIndex,
                                layers: entry.startLayer..<entry.endLayer,
                                computeSeconds: seconds, stageSeconds: seconds,
                                isLocal: true),
                            payloadBytes: 0)))
                    }
                }
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
                    output = try NMPActivationCodec.decode(tensorBytes)
                } catch {
                    completion(.failure(.sendFailed("response decode: \(error)")))
                    return
                }
                let stageSeconds = TimeInterval(
                    DispatchTime.now().uptimeNanoseconds - stageBegan.uptimeNanoseconds) / 1e9
                let computeSeconds = TimeInterval(meta.computeMicros) / 1e6
                recordMeasurement(peerID: entry.peerID, seconds: computeSeconds,
                                  layers: entry.layerSpan)
                completion(.success(NMPStageResult(
                    output: output,
                    timing: NMPInferenceReport.ShardTiming(
                        peerID: entry.peerID, shardIndex: entry.shardIndex,
                        layers: entry.startLayer..<entry.endLayer,
                        computeSeconds: computeSeconds, stageSeconds: stageSeconds,
                        isLocal: false),
                    payloadBytes: sentBytes + tensorBytes.count)))
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

        let tensor = NMPActivationCodec.encode(activations, format: activationWireFormat)
        let chunks: [NMPTensorChunk]
        do {
            // Link-adaptive chunks: MTU-safe 1024 B on radio, the kernel
            // datagram ceiling on wired/loopback (envelope comes off first).
            let chunkBytes = max(1, connection.recommendedChunkBytes
                                    - NMPTensorChunk.envelopeBytes)
            chunks = try NMPTensorChunk.split(requestID: requestID,
                                              tensorBytes: tensor,
                                              chunkBytes: chunkBytes)
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

        // One burst: single hop onto the connection queue, coalesced
        // transport writes, FLUSH on the last chunk.
        connection.sendBurstAsync(
            payloads: [meta.encode()] + chunks.map { $0.encode() },
            priority: .critical)

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
        measurementsLock.lock()
        defer { measurementsLock.unlock() }
        measurements[peerID] = seconds / Double(layers)
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
        case .resourceReport:
            if let report = try? NMPPeerResourceReport.decode(payload) {
                onPeerResourceReport?(report)
            }
        case .pong:
            // Keepalive echo — its arrival already fired onPeerActivity
            // (every authenticated packet does), which is its whole job.
            break
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
