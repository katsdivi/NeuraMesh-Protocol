//
//  ComputeEngine.swift
//  NMP — Phase 5
//
//  The compute seam of the mesh: everything above this protocol (sharding,
//  orchestration, transport) is engine-agnostic.
//
//  Implementations:
//
//  - NMPReferenceComputeEngine (here): a deterministic per-layer transform
//    that is BIT-IDENTICAL on every Apple platform (only IEEE-754 basic
//    ops — add/mul/div/abs — no transcendentals, whose last-bit rounding
//    may differ between libm builds). This is what makes cross-device
//    numeric validation exact: Mac-only output must equal Mac+iPhone mesh
//    output bit-for-bit, so ANY transport corruption or shard-boundary bug
//    shows up as a hard test failure rather than hiding inside a float
//    tolerance.
//
//  - llama.cpp (not vendored): bind by conforming a wrapper to
//    NMPShardComputeEngine and calling llama_decode over the assigned
//    layer range. The integration point is documented in Phase5_Design.md;
//    everything else in the pipeline is unchanged when the engine is real.
//

import Foundation

// MARK: - Engine protocol

public protocol NMPShardComputeEngine: AnyObject {
    /// Total transformer layers in the model.
    var layerCount: Int { get }
    /// Activation vector width — the tensor size peers exchange.
    var hiddenSize: Int { get }
    /// Runs layers [start, end) over one activation vector.
    /// Precondition: 0 <= start < end <= layerCount, input.count == hiddenSize.
    func runLayers(start: Int, end: Int, input: [Float]) throws -> [Float]
}

/// An engine that runs a layer SUB-RANGE and therefore needs to know the
/// plan's TOTAL layer count to tell whether its shard is the terminal one
/// (the last shard runs output_norm + lm_head). The peer engine sets this
/// from `SHARD_ASSIGN.totalLayers` on every assignment, so a re-shard that
/// changes the plan size is reflected without a concrete-type check.
public protocol NMPGlobalLayerAware: AnyObject {
    var globalLayerCount: Int { get set }
}

public enum NMPComputeError: Error, Equatable, Sendable {
    case invalidLayerRange(start: Int, end: Int, layerCount: Int)
    case invalidInputWidth(expected: Int, got: Int)
    case noShardAssigned
}

extension NMPShardComputeEngine {
    /// Wall-clock seconds per single run of layers [start, end); used by
    /// the sharder to weight peers by measured speed.
    public func measureLayerLatency(start: Int, end: Int, iterations: Int = 3) -> TimeInterval {
        let input = [Float](repeating: 0.5, count: hiddenSize)
        let began = DispatchTime.now()
        for _ in 0..<max(1, iterations) {
            _ = try? runLayers(start: start, end: end, input: input)
        }
        let nanos = DispatchTime.now().uptimeNanoseconds - began.uptimeNanoseconds
        return TimeInterval(nanos) / 1e9 / TimeInterval(max(1, iterations))
    }
}

// MARK: - Reference engine

/// Deterministic stand-in for real transformer layers. Each layer applies
/// a cheap mixing transform whose coefficients derive from splitmix64 of
/// (modelSeed, layerIndex), followed by the rational nonlinearity
/// x/(1+|x|) — bounded output, so activations never blow up across 32+
/// layers, and every operation is exactly reproducible cross-platform.
public final class NMPReferenceComputeEngine: NMPShardComputeEngine {

    public let layerCount: Int
    public let hiddenSize: Int
    private let modelSeed: UInt64
    /// Artificial per-layer delay (seconds) so latency-oriented tests and
    /// demos can emulate slower devices. 0 = compute at full speed.
    public var simulatedSecondsPerLayer: TimeInterval

    public init(layerCount: Int, hiddenSize: Int, modelSeed: UInt64 = 0x6E6D_7035,
                simulatedSecondsPerLayer: TimeInterval = 0) {
        precondition(layerCount > 0 && hiddenSize > 0)
        self.layerCount = layerCount
        self.hiddenSize = hiddenSize
        self.modelSeed = modelSeed
        self.simulatedSecondsPerLayer = simulatedSecondsPerLayer
    }

    /// Builds an engine sized from real GGUF metadata: the mesh moves
    /// activation tensors of the model's true hidden size across the
    /// model's true layer count, with reference math standing in for the
    /// quantized weights.
    public convenience init(gguf: NMPGGUFModel) throws {
        guard let layers = gguf.layerCount else {
            throw NMPGGUFError.missingMetadata("block_count")
        }
        guard let hidden = gguf.hiddenSize else {
            throw NMPGGUFError.missingMetadata("embedding_length")
        }
        self.init(layerCount: layers, hiddenSize: hidden)
    }

    public func runLayers(start: Int, end: Int, input: [Float]) throws -> [Float] {
        guard start >= 0, start < end, end <= layerCount else {
            throw NMPComputeError.invalidLayerRange(start: start, end: end, layerCount: layerCount)
        }
        guard input.count == hiddenSize else {
            throw NMPComputeError.invalidInputWidth(expected: hiddenSize, got: input.count)
        }
        var state = input
        for layer in start..<end {
            state = applyLayer(index: layer, to: state)
            if simulatedSecondsPerLayer > 0 {
                Thread.sleep(forTimeInterval: simulatedSecondsPerLayer)
            }
        }
        return state
    }

    private func applyLayer(index: Int, to input: [Float]) -> [Float] {
        // Per-layer constants from splitmix64 — identical on every device.
        var rng = SplitMix64(seed: modelSeed ^ UInt64(index) &* 0x9E37_79B9_7F4A_7C15)
        let a = rng.nextUnitFloat() * 1.5 + 0.25  // self weight
        let b = rng.nextUnitFloat() - 0.5          // neighbor weight
        let c = rng.nextUnitFloat() - 0.5          // long-range weight
        let bias = rng.nextUnitFloat() * 0.1
        let stride = 1 + Int(rng.next() % UInt64(max(1, input.count - 1)))

        let n = input.count
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let mixed = a * input[i]
                + b * input[(i + 1) % n]
                + c * input[(i + stride) % n]
                + bias
            out[i] = mixed / (1 + abs(mixed)) // rational squash, no libm
        }
        return out
    }
}

// MARK: - splitmix64

/// Reference splitmix64 (public domain, Vigna). Used for per-layer
/// coefficient derivation — NOT cryptographic.
struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 24 bits of mantissa — exact as Float.
    mutating func nextUnitFloat() -> Float {
        Float(next() >> 40) * (1.0 / 16_777_216.0)
    }
}
