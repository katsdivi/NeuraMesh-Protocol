//
//  ComputeEngineTests.swift
//  NMPTests — Phase 5
//
//  The reference engine's contract with the mesh: bit-exact determinism
//  (two engines = two devices) and layer composition (running [0,N) in one
//  shot must equal running it split at any shard boundary — the property
//  that makes distributed output comparable to single-device output).
//

import XCTest
@testable import NMP

final class ComputeEngineTests: XCTestCase {

    private func makeInput(_ count: Int) -> [Float] {
        var rng = SplitMix64(seed: 0xF00D)
        return (0..<count).map { _ in rng.nextUnitFloat() * 2 - 1 }
    }

    func testTwoEnginesAreBitIdentical() throws {
        // Same model parameters on two engine instances = the Mac and the
        // iPhone loading the same model. Outputs must match bit-for-bit.
        let a = NMPReferenceComputeEngine(layerCount: 32, hiddenSize: 128)
        let b = NMPReferenceComputeEngine(layerCount: 32, hiddenSize: 128)
        let input = makeInput(128)
        let outA = try a.runLayers(start: 0, end: 32, input: input)
        let outB = try b.runLayers(start: 0, end: 32, input: input)
        XCTAssertEqual(outA.map(\.bitPattern), outB.map(\.bitPattern))
    }

    func testLayerCompositionMatchesSingleShot() throws {
        // runLayers(0,32) == runLayers over any split — THE sharding
        // correctness property.
        let engine = NMPReferenceComputeEngine(layerCount: 32, hiddenSize: 96)
        let input = makeInput(96)
        let whole = try engine.runLayers(start: 0, end: 32, input: input)

        for boundaries in [[16], [10, 21], [1, 2, 3, 31]] {
            var cursor = 0
            var state = input
            for boundary in boundaries + [32] {
                state = try engine.runLayers(start: cursor, end: boundary, input: state)
                cursor = boundary
            }
            XCTAssertEqual(state.map(\.bitPattern), whole.map(\.bitPattern),
                           "split at \(boundaries) diverged")
        }
    }

    func testOutputsStayBoundedAcrossDeepStacks() throws {
        // The rational squash keeps activations in (-1, 1) — no overflow
        // regardless of depth.
        let engine = NMPReferenceComputeEngine(layerCount: 200, hiddenSize: 64)
        let out = try engine.runLayers(start: 0, end: 200, input: makeInput(64))
        XCTAssertTrue(out.allSatisfy { $0.isFinite && abs($0) < 1 })
    }

    func testDifferentSeedsAndLayersDiffer() throws {
        let input = makeInput(64)
        let a = NMPReferenceComputeEngine(layerCount: 4, hiddenSize: 64, modelSeed: 1)
        let b = NMPReferenceComputeEngine(layerCount: 4, hiddenSize: 64, modelSeed: 2)
        XCTAssertNotEqual(try a.runLayers(start: 0, end: 4, input: input),
                          try b.runLayers(start: 0, end: 4, input: input))
        XCTAssertNotEqual(try a.runLayers(start: 0, end: 1, input: input),
                          try a.runLayers(start: 1, end: 2, input: input))
    }

    func testValidatesRangeAndWidth() {
        let engine = NMPReferenceComputeEngine(layerCount: 8, hiddenSize: 16)
        XCTAssertThrowsError(try engine.runLayers(start: 4, end: 4, input: makeInput(16)))
        XCTAssertThrowsError(try engine.runLayers(start: 0, end: 9, input: makeInput(16)))
        XCTAssertThrowsError(try engine.runLayers(start: -1, end: 2, input: makeInput(16)))
        XCTAssertThrowsError(try engine.runLayers(start: 0, end: 8, input: makeInput(15))) {
            XCTAssertEqual($0 as? NMPComputeError,
                           .invalidInputWidth(expected: 16, got: 15))
        }
    }

    func testLatencyMeasurementIsPositiveAndScalesWithLayers() {
        let engine = NMPReferenceComputeEngine(layerCount: 16, hiddenSize: 256)
        let one = engine.measureLayerLatency(start: 0, end: 1)
        let sixteen = engine.measureLayerLatency(start: 0, end: 16)
        XCTAssertGreaterThan(one, 0)
        XCTAssertGreaterThan(sixteen, one)
    }
}
