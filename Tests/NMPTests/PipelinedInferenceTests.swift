//
//  PipelinedInferenceTests.swift
//  NMP — Phase 9
//
//  Pins pipeline-parallel batch execution: outputs must be bit-identical
//  to serial passes (overlap must never change math), sequences must
//  stay ordered, and — the point of the feature — the batch wall clock
//  must beat the serial sum when stages have real compute to overlap.
//

import XCTest
@testable import NMP

final class PipelinedInferenceTests: XCTestCase {

    func testBatchOutputsAreBitExactAgainstBaseline() throws {
        let testbed = try NMPMeshTestbed(layerCount: 12, hiddenSize: 256,
                                         remotePeerCount: 2)
        let plan = try testbed.startSync()
        let executor = NMPPipelinedBatchExecutor(orchestrator: testbed.orchestrator)

        let inputs = (0..<5).map { testbed.makeInput(seed: UInt64(0xBA7C + $0)) }
        let report = try executor.runSync(inputs: inputs, plan: plan)

        XCTAssertEqual(report.outputs.count, inputs.count)
        for (input, output) in zip(inputs, report.outputs) {
            XCTAssertEqual(output.map(\.bitPattern),
                           try testbed.baselineOutput(for: input).map(\.bitPattern),
                           "pipelining must not change any sequence's math")
        }
        XCTAssertEqual(report.perSequenceSeconds.count, inputs.count)
        XCTAssertGreaterThan(report.networkPayloadBytes, 0)
    }

    /// 3 stages × 2 ms/layer of real (simulated) compute: a batch of 6
    /// must finish well under the serial sum. The theoretical ceiling is
    /// ~3× (three stages); we assert a conservative 1.3× so CI scheduler
    /// noise can't flake the test.
    func testBatchOverlapBeatsSerialExecution() throws {
        let testbed = try NMPMeshTestbed(layerCount: 12, hiddenSize: 256,
                                         remotePeerCount: 2,
                                         simulatedSecondsPerLayer: 0.002)
        let plan = try testbed.startSync()
        XCTAssertEqual(plan.count, 3, "coordinator + 2 peers expected")
        let executor = NMPPipelinedBatchExecutor(orchestrator: testbed.orchestrator)

        let inputs = (0..<6).map { testbed.makeInput(seed: UInt64(0x0E1 + $0)) }
        let report = try executor.runSync(inputs: inputs, plan: plan)

        XCTAssertGreaterThan(report.pipelineSpeedup, 1.3,
                             "overlap won only \(report.pipelineSpeedup)× "
                             + "(total \(report.totalSeconds)s vs serial "
                             + "\(report.serialEstimateSeconds)s)")
    }

    func testSingleSequenceBatchMatchesSerialInfer() throws {
        // Degenerate batch: identical result and roughly identical cost to
        // a plain infer (no overlap available — honesty check).
        let testbed = try NMPMeshTestbed(layerCount: 8, hiddenSize: 256,
                                         remotePeerCount: 1)
        let plan = try testbed.startSync()
        let executor = NMPPipelinedBatchExecutor(orchestrator: testbed.orchestrator)

        let input = testbed.makeInput(seed: 0x501)
        let report = try executor.runSync(inputs: [input], plan: plan)
        let serial = try testbed.inferSync(input: input)
        XCTAssertEqual(report.outputs[0].map(\.bitPattern),
                       serial.output.map(\.bitPattern))
    }

    func testEmptyBatchAndEmptyPlanFailExplicitly() throws {
        let testbed = try NMPMeshTestbed(layerCount: 8, hiddenSize: 256,
                                         remotePeerCount: 1)
        let plan = try testbed.startSync()
        let executor = NMPPipelinedBatchExecutor(orchestrator: testbed.orchestrator)

        XCTAssertThrowsError(try executor.runSync(inputs: [], plan: plan)) {
            guard case NMPPipelinedBatchExecutor.BatchError.emptyBatch = $0 else {
                return XCTFail("expected emptyBatch, got \($0)")
            }
        }
        XCTAssertThrowsError(try executor.runSync(
            inputs: [testbed.makeInput()], plan: [])) {
            guard case NMPPipelinedBatchExecutor.BatchError.emptyPlan = $0 else {
                return XCTFail("expected emptyPlan, got \($0)")
            }
        }
    }
}
