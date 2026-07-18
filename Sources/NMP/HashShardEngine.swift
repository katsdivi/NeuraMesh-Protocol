//
//  HashShardEngine.swift
//  NMP — Plugin Architecture task 4: non-LLM SCAFFOLD (do NOT ship as real)
//
//  A minimal `NMPPlugin` (== NMPShardComputeEngine) conformer for a job that
//  is NOT a transformer: a toy "hash-shard" that folds its input vector into
//  a rolling checksum. Its ONLY purpose is to prove the compute seam accepts
//  a non-LLM job — it compiles, registers in NMPPluginRegistry, and is
//  selectable via `--engine hashShard` on nmp-peer.
//
//  The actual job semantics are deliberately NOT designed:
//    - What a "shard" of a checksum job means (there is no layer axis).
//    - How work is partitioned across peers.
//    - What the wire payload should carry (see Docs/Plugin_Architecture.md §
//      "What is still LLM-specific" — the wire protocol currently assumes
//      layer ranges + tensor-shaped activations, which a real non-LLM job
//      would have to either reuse or extend).
//
//  Everything below marked `TODO(plugin)` is intentionally unimplemented.
//  The placeholder math exists only so a peer that selects this engine stays
//  alive instead of crashing; it computes nothing meaningful.
//

import Foundation

public enum NMPHashShardError: Error, Sendable {
    /// The real hash-shard job is not implemented — surfaced if any caller
    /// expects meaningful output rather than the placeholder checksum.
    case notImplemented(String)
}

/// Scaffold engine. Presents a layer/hidden shape so it slots behind the
/// existing (LLM-shaped) seam unchanged, but does no real work.
public final class NMPHashShardComputeEngine: NMPPlugin {

    public let layerCount: Int
    public let hiddenSize: Int

    public init(layerCount: Int = 8, hiddenSize: Int = 256) {
        precondition(layerCount > 0 && hiddenSize > 0)
        self.layerCount = layerCount
        self.hiddenSize = hiddenSize
    }

    public func runLayers(start: Int, end: Int, input: [Float]) throws -> [Float] {
        guard start >= 0, start < end, end <= layerCount else {
            throw NMPComputeError.invalidLayerRange(start: start, end: end, layerCount: layerCount)
        }
        guard input.count == hiddenSize else {
            throw NMPComputeError.invalidInputWidth(expected: hiddenSize, got: input.count)
        }
        // TODO(plugin): design and implement the real hash-shard job. This
        // placeholder folds the input into an FNV-1a checksum stamped in
        // slot 0 so the pipeline has *something* deterministic to carry —
        // it is NOT the intended behavior.
        var hash: UInt64 = 0xCBF2_9CE4_8422_2325
        for value in input {
            hash = (hash ^ UInt64(value.bitPattern)) &* 0x0000_0100_0000_01B3
        }
        var out = input
        out[0] = Float(bitPattern: UInt32(truncatingIfNeeded: hash))
        return out
    }
}

// MARK: - Stub codec

/// Placeholder text codec so `hashShard` can be named alongside the real
/// plugins in the prompt-inference wiring. Not wired into any CLI's
/// generation path — the job isn't text — and every method is a TODO.
public final class NMPHashShardPromptCodec: NMPPromptCodec {

    public let engineName = "hashShard"

    public init() {}

    public func makeInitialInput(prompt: String) throws -> [Float] {
        // TODO(plugin): define how a hash-shard job is fed its input.
        throw NMPHashShardError.notImplemented("makeInitialInput")
    }

    public func extractToken(from output: [Float], position: Int) throws -> NMPGeneratedToken? {
        throw NMPHashShardError.notImplemented("extractToken")
    }

    public func makeNextInput(after output: [Float], token: NMPGeneratedToken,
                              position: Int) throws -> [Float] {
        throw NMPHashShardError.notImplemented("makeNextInput")
    }

    public func render(tokens: [NMPGeneratedToken]) -> String {
        ""  // TODO(plugin): render hash-shard results.
    }
}
