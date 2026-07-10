//
//  OptimizedActivationTests.swift
//  NMP — Phase 9
//
//  Pins the optimized activation wire formats: the manual binary16
//  conversion (bit-level), zero-trim losslessness, mixed-precision error
//  bounds and critical-value exactness, magic sniffing / legacy
//  fallback, and the end-to-end property that a mesh negotiating a
//  compressed format still computes correctly while moving fewer bytes.
//

import XCTest
@testable import NMP

// MARK: - binary16 conversion

final class HalfFloatTests: XCTestCase {

    /// Every half-precision bit pattern must survive decode → encode
    /// unchanged (NaNs canonicalize but stay NaN) — this pins BOTH
    /// directions of the conversion at once.
    func testAllHalfPatternsRoundTrip() {
        for pattern in 0...UInt16.max {
            let value = NMPHalfFloat.decode(pattern)
            if value.isNaN {
                XCTAssertTrue(NMPHalfFloat.decode(NMPHalfFloat.encode(value)).isNaN)
                continue
            }
            XCTAssertEqual(NMPHalfFloat.encode(value), pattern,
                           "pattern 0x\(String(pattern, radix: 16)) → \(value) diverged")
        }
    }

    func testExactValues() {
        for value: Float in [0, 1, -1, 0.5, 2, -0.25, 1024, 65504] {
            XCTAssertEqual(NMPHalfFloat.decode(NMPHalfFloat.encode(value)), value,
                           "\(value) is exactly representable and must round-trip")
        }
        // Signed zero keeps its sign bit.
        XCTAssertEqual(NMPHalfFloat.encode(-0.0), 0x8000)
    }

    func testRoundToNearestEven() {
        // 1 + 2^-11 sits exactly between 1.0 and 1 + 2^-10 → even mantissa (1.0).
        XCTAssertEqual(NMPHalfFloat.decode(NMPHalfFloat.encode(1 + powf(2, -11))), 1)
        // 1 + 3·2^-11 sits between 1+2^-10 and 1+2·2^-10 → even (1+2·2^-10).
        XCTAssertEqual(NMPHalfFloat.decode(NMPHalfFloat.encode(1 + 3 * powf(2, -11))),
                       1 + 2 * powf(2, -10))
    }

    func testOverflowAndUnderflow() {
        // Above the halfway point past 65504 → ±inf.
        XCTAssertEqual(NMPHalfFloat.decode(NMPHalfFloat.encode(65520)), .infinity)
        XCTAssertEqual(NMPHalfFloat.decode(NMPHalfFloat.encode(-1e9)), -.infinity)
        XCTAssertEqual(NMPHalfFloat.decode(NMPHalfFloat.encode(.infinity)), .infinity)
        // Smallest subnormal half; below half of it → 0.
        XCTAssertEqual(NMPHalfFloat.encode(powf(2, -24)), 0x0001)
        XCTAssertEqual(NMPHalfFloat.encode(powf(2, -26)), 0x0000)
    }

    func testRelativeErrorBoundOnRandomValues() {
        var rng = SplitMix64(seed: 0xF16)
        for _ in 0..<10_000 {
            let value = rng.nextUnitFloat() * 2 - 1 // the reference engine's range
            let recovered = NMPHalfFloat.decode(NMPHalfFloat.encode(value))
            XCTAssertLessThanOrEqual(
                abs(recovered - value), max(abs(value) * powf(2, -11), powf(2, -24)),
                "|\(value) - \(recovered)| exceeds binary16 rounding")
        }
    }
}

// MARK: - Activation codec

final class ActivationCodecTests: XCTestCase {

    private func randomActivations(_ count: Int, seed: UInt64 = 0xAC71) -> [Float] {
        var rng = SplitMix64(seed: seed)
        return (0..<count).map { _ in rng.nextUnitFloat() * 2 - 1 }
    }

    func testZeroTrimIsLosslessAndSmallForSparseTensors() throws {
        // A llama-style token-state vector: 3 header + 5 payload slots used
        // of 4096 — the shape every Phase 8 llama message has.
        var sparse = [Float](repeating: 0, count: 4096)
        for (index, value) in [5_000_273, 12, 5, 1, 15043, 590, 338, 3838].enumerated() {
            sparse[index] = Float(value)
        }
        let encoded = NMPActivationCodec.encode(sparse, format: .zeroTrimmed)
        let decoded = try NMPActivationCodec.decode(encoded)
        XCTAssertEqual(decoded.map(\.bitPattern), sparse.map(\.bitPattern),
                       "zero-trim must be bit-exact")
        let rawBytes = NMPTensorCodec.encode(sparse).count
        XCTAssertLessThan(Double(encoded.count) / Double(rawBytes), 0.01,
                          "sparse tensor should shrink by ~99% "
                          + "(got \(encoded.count) of \(rawBytes) B)")
    }

    func testZeroTrimEdgeCases() throws {
        // All zeros → header only; no trailing zeros → full size + header.
        let zeros = [Float](repeating: 0, count: 128)
        XCTAssertEqual(try NMPActivationCodec.decode(
            NMPActivationCodec.encode(zeros, format: .zeroTrimmed)), zeros)

        var dense = randomActivations(128)
        dense[127] = 0.5 // ensure no trailing zero
        let encoded = NMPActivationCodec.encode(dense, format: .zeroTrimmed)
        XCTAssertEqual(try NMPActivationCodec.decode(encoded).map(\.bitPattern),
                       dense.map(\.bitPattern))
    }

    func testMixedPrecisionHalvesDenseTensorsWithinErrorBound() throws {
        let dense = randomActivations(4096)
        let encoded = NMPActivationCodec.encode(dense, format: .mixedPrecision)
        let decoded = try NMPActivationCodec.decode(encoded)

        XCTAssertEqual(decoded.count, dense.count)
        for (original, recovered) in zip(dense, decoded) {
            XCTAssertLessThanOrEqual(
                abs(recovered - original),
                max(abs(original) * powf(2, -11), powf(2, -24)))
        }
        let rawBytes = NMPTensorCodec.encode(dense).count
        let ratio = Double(encoded.count) / Double(rawBytes)
        XCTAssertLessThan(ratio, 0.56, "expected ~52% of raw, got \(ratio)")
    }

    func testMixedPrecisionKeepsCriticalValuesExact() throws {
        // Outliers (layer-norm-scale spikes) must survive bit-exactly.
        var values = randomActivations(1000)
        values[7] = 123.456
        values[500] = -987.25
        let decoded = try NMPActivationCodec.decode(
            NMPActivationCodec.encode(values, format: .mixedPrecision))
        XCTAssertEqual(decoded[7], 123.456)
        XCTAssertEqual(decoded[500], -987.25)
    }

    func testMagicSniffingAndLegacyFallback() throws {
        let values = randomActivations(64)
        for format in [NMPActivationWireFormat.float32, .zeroTrimmed, .mixedPrecision] {
            let encoded = NMPActivationCodec.encode(values, format: format)
            XCTAssertEqual(NMPActivationCodec.formatOf(encoded), format)
        }
        // Phase 8 bytes (raw f32) decode unchanged — old coordinators keep
        // working against Phase 9 peers.
        let legacy = NMPTensorCodec.encode(values)
        XCTAssertEqual(try NMPActivationCodec.decode(legacy).map(\.bitPattern),
                       values.map(\.bitPattern))
    }

    func testTruncatedCompressedPayloadsAreRejected() throws {
        let encoded = NMPActivationCodec.encode(
            randomActivations(64), format: .mixedPrecision)
        XCTAssertThrowsError(try NMPActivationCodec.decode(encoded.prefix(10)))
        XCTAssertThrowsError(try NMPActivationCodec.decode(encoded.prefix(40)))
        let trimmed = NMPActivationCodec.encode(
            randomActivations(64), format: .zeroTrimmed)
        XCTAssertThrowsError(try NMPActivationCodec.decode(trimmed.prefix(11)))
    }
}

// MARK: - Over the mesh

final class ActivationWireFormatMeshTests: XCTestCase {

    /// zero-trim over the full protocol stack is bit-exact vs the
    /// single-device baseline (it is lossless), while a compressed format
    /// is mirrored back by the peer.
    func testZeroTrimmedMeshPassIsBitExact() throws {
        let testbed = try NMPMeshTestbed(layerCount: 8, hiddenSize: 512,
                                         remotePeerCount: 2)
        _ = try testbed.startSync()
        testbed.orchestrator.activationWireFormat = .zeroTrimmed

        let input = testbed.makeInput(seed: 0x21F)
        let report = try testbed.inferSync(input: input)
        XCTAssertEqual(report.output.map(\.bitPattern),
                       try testbed.baselineOutput(for: input).map(\.bitPattern))
    }

    /// mixed precision stays within an accumulated rounding envelope and
    /// moves measurably fewer bytes than raw float32.
    func testMixedPrecisionMeshPassIsCloseAndSmaller() throws {
        let testbed = try NMPMeshTestbed(layerCount: 8, hiddenSize: 512,
                                         remotePeerCount: 2)
        _ = try testbed.startSync()
        let input = testbed.makeInput(seed: 0x9F16)

        let rawReport = try testbed.inferSync(input: input)
        testbed.orchestrator.activationWireFormat = .mixedPrecision
        let compressedReport = try testbed.inferSync(input: input)

        let baseline = try testbed.baselineOutput(for: input)
        for (expected, got) in zip(baseline, compressedReport.output) {
            // 2 remote hops × 2 directions of 2^-11 rounding through a
            // squashing nonlinearity — 1e-2 is a generous envelope.
            XCTAssertEqual(got, expected, accuracy: 0.01)
        }
        let ratio = Double(compressedReport.networkPayloadBytes)
            / Double(rawReport.networkPayloadBytes)
        XCTAssertLessThan(ratio, 0.62,
                          "expected ~52% of the raw payload, got \(ratio)")
    }
}
