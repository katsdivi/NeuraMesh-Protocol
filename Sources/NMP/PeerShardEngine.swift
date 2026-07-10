//
//  PeerShardEngine.swift
//  NMP — Phase 5
//
//  The compute-peer side of the mesh (what runs on the iPhone): owns an
//  established PeerConnection to the coordinator, accepts SHARD_ASSIGN,
//  serves inference requests over its assigned layer range, and reports
//  metrics after every serve.
//
//  Threading: the engine takes over `connection.onPacket`, so everything
//  here runs on the connection's serial queue. Compute happens inline on
//  that queue — acceptable because a shard peer serves one coordinator
//  and the pipeline is strictly sequential (there is never a second
//  request racing the one being computed).
//
//  Activations are sent at .critical priority: they ARE the workload, and
//  deferring them under AWDL suppression would stall the whole pipeline
//  (suppression exists to protect exactly this traffic from background
//  noise, not from itself).
//

import Foundation

public final class NMPPeerShardEngine {

    // MARK: Callbacks (invoked on the connection queue)

    public var onAssigned: ((NMPShardAssign) -> Void)?
    /// requestID, layers computed, pure compute seconds.
    public var onInferenceServed: ((UInt32, Range<Int>, TimeInterval) -> Void)?
    public var onDiagnostic: ((String) -> Void)?

    // MARK: State

    public private(set) var assignment: NMPShardAssign?

    private let connection: PeerConnection
    private let engine: NMPShardComputeEngine
    private let localPeerID: UInt32
    /// Model identity this peer loaded; assignments for anything else are
    /// rejected (computing with mismatched weights = silent garbage).
    private let modelTag: String
    private let reassembler = NMPTensorReassembler()
    private var requestMetas: [UInt32: NMPInferRequestMeta] = [:]

    public init(
        connection: PeerConnection,
        engine: NMPShardComputeEngine,
        modelTag: String,
        localPeerID: UInt32
    ) {
        self.connection = connection
        self.engine = engine
        self.modelTag = modelTag
        self.localPeerID = localPeerID
    }

    /// Takes over the connection's packet stream. Call once, after the
    /// connection is created (before or after establishment both work —
    /// packets only flow post-handshake).
    public func activate() {
        connection.onPacket = { [weak self] packet in
            self?.handle(packet)
        }
    }

    // MARK: Inbound dispatch

    private func handle(_ packet: NMPPacket) {
        switch packet.header.packetType {
        case .shardAssign:
            handleAssign(packet.payload)
        case .data:
            handleMeshMessage(packet.payload)
        default:
            onDiagnostic?("peer engine ignoring packet type \(packet.header.packetType)")
        }
    }

    private func handleAssign(_ payload: Data) {
        let assign: NMPShardAssign
        do {
            assign = try NMPShardAssign.decode(payload)
        } catch {
            onDiagnostic?("dropped malformed SHARD_ASSIGN: \(error)")
            return
        }

        let status: NMPShardAck.Status
        if assign.modelTag != modelTag {
            status = .rejectedModelMismatch
            onDiagnostic?("SHARD_ASSIGN for model '\(assign.modelTag)' but loaded '\(modelTag)'")
        } else if assign.startLayer >= assign.endLayer
                    || Int(assign.endLayer) > engine.layerCount
                    || Int(assign.hiddenSize) != engine.hiddenSize {
            status = .rejectedBadRange
            onDiagnostic?("SHARD_ASSIGN rejected: layers \(assign.startLayer)..<\(assign.endLayer) "
                          + "of \(engine.layerCount), hidden \(assign.hiddenSize) vs \(engine.hiddenSize)")
        } else {
            status = .ready
            assignment = assign
            onAssigned?(assign)
        }
        send(payload: NMPShardAck(shardIndex: assign.shardIndex, status: status).encode(),
             context: "shard ack")
    }

    private func handleMeshMessage(_ payload: Data) {
        switch NMPMeshMessageKindOf(payload) {
        case .inferRequestMeta:
            do {
                let meta = try NMPInferRequestMeta.decode(payload)
                // The pipeline is serial: a new request means every older
                // incomplete one was abandoned by the coordinator (stage
                // retry under heavy loss). Drop their partial tensors so
                // sustained loss cannot balloon peer memory.
                requestMetas = requestMetas.filter { $0.key >= meta.requestID }
                reassembler.abandonOlder(than: meta.requestID)
                requestMetas[meta.requestID] = meta
                if let tensor = try reassembler.setExpectation(
                    requestID: meta.requestID,
                    chunkCount: Int(meta.chunkCount),
                    totalBytes: Int(meta.totalBytes)) {
                    serve(requestID: meta.requestID, tensorBytes: tensor)
                }
            } catch {
                onDiagnostic?("bad inference request meta: \(error)")
            }
        case .tensorChunk:
            do {
                let chunk = try NMPTensorChunk.decode(payload)
                if let tensor = try reassembler.addChunk(chunk) {
                    serve(requestID: chunk.requestID, tensorBytes: tensor)
                }
            } catch {
                onDiagnostic?("bad tensor chunk: \(error)")
            }
        default:
            onDiagnostic?("peer engine ignoring mesh message kind "
                          + "\(payload.first.map(String.init) ?? "<empty>")")
        }
    }

    // MARK: Serving

    private func serve(requestID: UInt32, tensorBytes: Data) {
        guard let meta = requestMetas.removeValue(forKey: requestID) else {
            onDiagnostic?("tensor completed for unknown request \(requestID)")
            return
        }
        guard let assignment else {
            respondFailure(requestID: requestID, status: .notAssigned)
            return
        }
        // The coordinator must ask for exactly the assigned range — a
        // divergence means the plans disagree, which is a hard fault.
        guard meta.startLayer == assignment.startLayer,
              meta.endLayer == assignment.endLayer else {
            onDiagnostic?("request \(requestID) wants layers "
                          + "\(meta.startLayer)..<\(meta.endLayer), assigned "
                          + "\(assignment.startLayer)..<\(assignment.endLayer)")
            respondFailure(requestID: requestID, status: .badRequest)
            return
        }

        // Phase 9: requests may arrive zero-trimmed or mixed-precision
        // (magic-sniffed); the response mirrors the request's format, so a
        // coordinator only ever receives what it opted into.
        let requestFormat = NMPActivationCodec.formatOf(tensorBytes)
        let input: [Float]
        do {
            input = try NMPActivationCodec.decode(tensorBytes)
        } catch {
            respondFailure(requestID: requestID, status: .badRequest)
            return
        }

        let began = DispatchTime.now()
        let output: [Float]
        do {
            output = try engine.runLayers(start: Int(meta.startLayer),
                                          end: Int(meta.endLayer),
                                          input: input)
        } catch {
            onDiagnostic?("compute failed for request \(requestID): \(error)")
            respondFailure(requestID: requestID, status: .computeFailed)
            return
        }
        let computeSeconds = TimeInterval(
            DispatchTime.now().uptimeNanoseconds - began.uptimeNanoseconds) / 1e9

        do {
            let tensor = NMPActivationCodec.encode(output, format: requestFormat)
            let chunks = try NMPTensorChunk.split(requestID: requestID, tensorBytes: tensor)
            let response = NMPInferResponseMeta(
                requestID: requestID, status: .ok,
                computeMicros: UInt32(clamping: Int(computeSeconds * 1e6)),
                totalBytes: UInt32(tensor.count),
                chunkCount: UInt16(chunks.count))
            send(payload: response.encode(), context: "response meta")
            for (index, chunk) in chunks.enumerated() {
                // FLUSH on the last chunk: nothing follows to fill receiver
                // gaps, so expedite NACKs and close the FEC group.
                send(payload: chunk.encode(),
                     flags: index == chunks.count - 1 ? [.flush] : [],
                     context: "response chunk \(index)")
            }
        } catch {
            onDiagnostic?("failed to send response \(requestID): \(error)")
            return
        }

        onInferenceServed?(requestID,
                           Int(meta.startLayer)..<Int(meta.endLayer),
                           computeSeconds)
        reportMetrics(computeSeconds: computeSeconds)
    }

    private func respondFailure(requestID: UInt32, status: NMPInferResponseMeta.Status) {
        let response = NMPInferResponseMeta(
            requestID: requestID, status: status,
            computeMicros: 0, totalBytes: 0, chunkCount: 0)
        send(payload: response.encode(), flags: [.flush], context: "failure response")
    }

    private func reportMetrics(computeSeconds: TimeInterval) {
        let metrics = NMPPeerMetrics(
            peerID: localPeerID,
            inferenceLatencyMicros: UInt32(clamping: Int(computeSeconds * 1e6)),
            memoryUsageMB: Self.residentMemoryMB(),
            currentLoadPercent: UInt8(clamping: Int(loadSampler.samplePercent().rounded())))
        send(payload: metrics.encode(), context: "metrics")
    }

    private let loadSampler = NMPCPULoadSampler()

    private func send(payload: Data, flags: NMPFlags = [], context: String) {
        do {
            _ = try connection.send(flags: flags, priority: .critical, payload: payload)
        } catch {
            onDiagnostic?("send failed (\(context)): \(error)")
        }
    }

    /// Resident set size in MB (0 where task_info is unavailable).
    public static func residentMemoryMB() -> UInt32 {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard status == KERN_SUCCESS else { return 0 }
        return UInt32(clamping: info.resident_size / (1024 * 1024))
        #else
        return 0
        #endif
    }
}
