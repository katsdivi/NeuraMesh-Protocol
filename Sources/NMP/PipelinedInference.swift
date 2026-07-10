//
//  PipelinedInference.swift
//  NMP — Phase 9
//
//  Pipeline-parallel BATCH execution. Phase 5's orchestrator walks one
//  activation through the shards strictly serially, so during an N-stage
//  pass, N−1 devices idle. This executor keeps every stage busy with a
//  DIFFERENT sequence: while shard 1 computes sequence A, shard 0 already
//  computes sequence B — classic pipeline parallelism, bounded by the
//  slowest stage instead of the sum of stages.
//
//  PIPELINING HONESTY: this overlaps INDEPENDENT sequences (a batch of
//  prompts, or probe/benchmark traffic). It cannot speed up a single
//  autoregressive generation — token t+1's input IS token t's output, so
//  consecutive tokens of one stream can never occupy two stages at once.
//  Anything claiming otherwise is hiding a dependency violation. The
//  single-stream lever is Phase 9's speculative decoding
//  (SpeculativeDecoder.swift); the multi-stream lever is this file.
//
//  Ordering invariant: stage i admits sequences strictly in batch order,
//  one at a time. Because every stage is serial, sequences exit stage i
//  in the order they entered, so per-peer traffic stays serial — the
//  discipline NMPPeerShardEngine's reassembler already assumes.
//
//  Threading: callback style on a private serial queue; stage work is
//  delegated to NMPInferenceOrchestrator.computeStage (which owns
//  connections, retry, and timing).
//

import Foundation

public final class NMPPipelinedBatchExecutor {

    // MARK: Report

    public struct BatchReport: Sendable {
        /// Final activations, one per input, in input order. Each is
        /// bit-identical to what a serial `infer` would have produced.
        public let outputs: [[Float]]
        /// Wall clock for the whole batch.
        public let totalSeconds: TimeInterval
        /// Per-sequence wall clock (entered stage 0 → left last stage).
        public let perSequenceSeconds: [TimeInterval]
        /// Application payload bytes across all sequences and stages.
        public let networkPayloadBytes: Int
        /// What the same batch would cost run serially: the sum of every
        /// stage time actually measured. The overlap win is
        /// serialEstimateSeconds / totalSeconds.
        public let serialEstimateSeconds: TimeInterval

        public var pipelineSpeedup: Double {
            totalSeconds > 0 ? serialEstimateSeconds / totalSeconds : 0
        }
    }

    public enum BatchError: Error {
        case emptyBatch
        case emptyPlan
        case stageFailed(sequence: Int, error: NMPOrchestrationError)
    }

    // MARK: State

    private let orchestrator: NMPInferenceOrchestrator
    private let queue = DispatchQueue(label: "nmp.pipelined.batch")

    public init(orchestrator: NMPInferenceOrchestrator) {
        self.orchestrator = orchestrator
    }

    /// Runs `inputs` through `plan` with stage overlap. Completion fires
    /// on the executor queue. The plan is passed explicitly (take it from
    /// `startSync`/`activePlan`) so this never races plan mutation.
    public func run(
        inputs: [[Float]],
        plan: [NMPShardPlanEntry],
        stageTimeout: TimeInterval = 30,
        completion: @escaping (Result<BatchReport, BatchError>) -> Void
    ) {
        queue.async { [self] in
            guard !inputs.isEmpty else {
                completion(.failure(.emptyBatch))
                return
            }
            guard !plan.isEmpty else {
                completion(.failure(.emptyPlan))
                return
            }
            let run = Run(inputs: inputs, plan: plan,
                          stageTimeout: stageTimeout, completion: completion)
            schedule(run)
        }
    }

    /// Blocking wrapper (call from a plain thread, never a mesh queue).
    public func runSync(
        inputs: [[Float]],
        plan: [NMPShardPlanEntry],
        stageTimeout: TimeInterval = 30,
        timeout: TimeInterval = 300
    ) throws -> BatchReport {
        let done = DispatchSemaphore(value: 0)
        var outcome: Result<BatchReport, BatchError>?
        run(inputs: inputs, plan: plan, stageTimeout: stageTimeout) { result in
            outcome = result
            done.signal()
        }
        guard done.wait(timeout: .now() + timeout) == .success, let outcome else {
            throw NMPMeshTestbedError.inferenceTimeout
        }
        return try outcome.get()
    }

    // MARK: Scheduler (executor queue)

    private final class Run {
        let plan: [NMPShardPlanEntry]
        let stageTimeout: TimeInterval
        let completion: (Result<BatchReport, BatchError>) -> Void
        let began = DispatchTime.now()

        /// Per-sequence: current activations and the next stage to enter.
        var activations: [[Float]]
        var nextStage: [Int]
        var inFlight: [Bool]
        var started: [DispatchTime?]
        var finished: [TimeInterval?]
        /// Per-stage: free to admit the next sequence?
        var stageFree: [Bool]

        var payloadBytes = 0
        var serialSeconds: TimeInterval = 0
        var failed = false
        var completed = 0

        init(inputs: [[Float]], plan: [NMPShardPlanEntry],
             stageTimeout: TimeInterval,
             completion: @escaping (Result<BatchReport, BatchError>) -> Void) {
            self.plan = plan
            self.stageTimeout = stageTimeout
            self.completion = completion
            activations = inputs
            nextStage = [Int](repeating: 0, count: inputs.count)
            inFlight = [Bool](repeating: false, count: inputs.count)
            started = [DispatchTime?](repeating: nil, count: inputs.count)
            finished = [TimeInterval?](repeating: nil, count: inputs.count)
            stageFree = [Bool](repeating: true, count: plan.count)
        }
    }

    private func schedule(_ run: Run) {
        guard !run.failed else { return }
        for stage in run.plan.indices where run.stageFree[stage] {
            // Batch order: the LOWEST sequence waiting for this stage.
            guard let sequence = run.nextStage.indices.first(where: {
                run.nextStage[$0] == stage && !run.inFlight[$0]
            }) else { continue }

            run.stageFree[stage] = false
            run.inFlight[sequence] = true
            if run.started[sequence] == nil {
                run.started[sequence] = DispatchTime.now()
            }

            orchestrator.computeStage(
                run.plan[stage],
                activations: run.activations[sequence],
                stageTimeout: run.stageTimeout
            ) { [weak self] result in
                // Fires on the orchestrator queue; hop home.
                self?.queue.async {
                    self?.stageDone(run, sequence: sequence, stage: stage,
                                    result: result)
                }
            }
        }
    }

    private func stageDone(
        _ run: Run, sequence: Int, stage: Int,
        result: Result<NMPStageResult, NMPOrchestrationError>
    ) {
        guard !run.failed else { return }
        run.stageFree[stage] = true
        run.inFlight[sequence] = false

        switch result {
        case .failure(let error):
            run.failed = true
            run.completion(.failure(.stageFailed(sequence: sequence, error: error)))
        case .success(let stageResult):
            run.activations[sequence] = stageResult.output
            run.nextStage[sequence] += 1
            run.payloadBytes += stageResult.payloadBytes
            run.serialSeconds += stageResult.timing.stageSeconds

            if run.nextStage[sequence] == run.plan.count {
                run.finished[sequence] = TimeInterval(
                    DispatchTime.now().uptimeNanoseconds
                        - run.started[sequence]!.uptimeNanoseconds) / 1e9
                run.completed += 1
                if run.completed == run.activations.count {
                    let total = TimeInterval(
                        DispatchTime.now().uptimeNanoseconds
                            - run.began.uptimeNanoseconds) / 1e9
                    run.completion(.success(BatchReport(
                        outputs: run.activations,
                        totalSeconds: total,
                        perSequenceSeconds: run.finished.map { $0 ?? 0 },
                        networkPayloadBytes: run.payloadBytes,
                        serialEstimateSeconds: run.serialSeconds)))
                    return
                }
            }
            schedule(run)
        }
    }
}
