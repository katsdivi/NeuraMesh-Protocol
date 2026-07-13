//
//  LlamaShardWire.swift
//  NMP — Phase 10 (cross-device sharding)
//
//  Inter-shard wire format for cross-device llama plans. When the model
//  is split across multiple peers, the intermediate shards exchange the
//  RESIDUAL HIDDEN STATE (the real activation the ggml graph-surgery shim
//  hands off between block ranges) alongside the sequence tokens so a
//  downstream shard can continue the forward pass.
//
//  The residual is the FULL activation, never truncated: with no KV cache
//  yet, each shard reprocesses the whole sequence, so the hidden state is
//  `n_embd × tokenCount` floats (position 0's n_embd values, then position
//  1's, …). The vector therefore GROWS past the model's hidden width to
//  hold it — encode() sizes to max(width, header + tokens + residual), and
//  decode() reads every residual float back. (`width` is just the minimum /
//  padding floor for compatibility with fixed-width reference tensors.)
//
//  Wire layout (all values are Float32):
//
//    shard request  (coordinator → non-first peer):
//      [0] magic 0x4E5348 ("NSH" — NMP Shard Hidden)
//      [1] basePos        — sequence base position (0 while there is no KV cache)
//      [2] tokenCount     — how many tokens (T) are packaged
//      [3 ... 3 + T - 1]  tokens — cast to Float
//      [3 + T ...]        residual hidden state — n_embd × T floats
//
//    shard response (non-last peer → coordinator):
//      [0] magic 0x4E5352 ("NSR" — NMP Shard Response)
//      [1] nextPos        — basePos + tokenCount
//      [2] tokenCount     — how many tokens (T) are packaged
//      [3 ... 3 + T - 1]  tokens — cast to Float
//      [3 + T ...]        residual hidden state — n_embd × T floats
//
//  Because the residual grows with the sequence, the mesh must stay on a
//  LOSSLESS activation format (.float32 / .zeroTrimmed) for the hand-off to
//  be bit-exact — .mixedPrecision would round the residual to fp16.
//
//  This file is pure Swift (no llama.cpp) — testable everywhere.
//

import Foundation

public enum NMPLlamaShardWireError: Error, Equatable, Sendable {
    case capacityExceeded(needed: Int, width: Int)
    case notAShardRequest
    case notAShardResponse
    case malformed(String)
}

public enum NMPLlamaShardWire {

    public static let shardRequestMagic: Float = 5_132_104   // 0x4E5348 "NSH"
    public static let shardResponseMagic: Float = 5_132_114  // 0x4E5352 "NSR"
    static let headerWidth = 3

    // MARK: - Request (hidden state + tokens → non-first shard)

    public struct ShardRequest: Equatable, Sendable {
        /// KV cache trim position.
        public let basePos: Int
        /// Tokens currently being decoded.
        public let tokens: [Int32]
        /// The hidden state vector.
        public let hiddenState: [Float]

        public init(basePos: Int, tokens: [Int32], hiddenState: [Float]) {
            self.basePos = basePos
            self.tokens = tokens
            self.hiddenState = hiddenState
        }
    }

    // MARK: - Response (hidden state + tokens from non-last shard)

    public struct ShardResponse: Equatable, Sendable {
        /// basePos + tokenCount from the request.
        public let nextPos: Int
        /// Tokens decoded.
        public let tokens: [Int32]
        /// The hidden state vector.
        public let hiddenState: [Float]

        public init(nextPos: Int, tokens: [Int32], hiddenState: [Float]) {
            self.nextPos = nextPos
            self.tokens = tokens
            self.hiddenState = hiddenState
        }
    }

    // MARK: - Encode

    /// The wire vector is sized EXACTLY to header + tokens + residual — no
    /// truncation (the residual must survive intact) and no zero-padding to
    /// `width` (padding would make decode over-read the extra zeros as
    /// residual, since the hidden length is inferred as "everything after
    /// the tokens"). `width` is accepted for call-site symmetry with the
    /// token-state wire but does not bound the residual.
    public static func encode(_ request: ShardRequest, width: Int) throws -> [Float] {
        let needed = headerWidth + request.tokens.count
        let size = needed + request.hiddenState.count
        var vector = [Float](repeating: 0, count: size)
        vector[0] = shardRequestMagic
        vector[1] = Float(request.basePos)
        vector[2] = Float(request.tokens.count)
        for (i, token) in request.tokens.enumerated() {
            vector[headerWidth + i] = Float(token)
        }
        for i in 0..<request.hiddenState.count {
            vector[needed + i] = request.hiddenState[i]
        }
        return vector
    }

    public static func encode(_ response: ShardResponse, width: Int) throws -> [Float] {
        let needed = headerWidth + response.tokens.count
        let size = needed + response.hiddenState.count
        var vector = [Float](repeating: 0, count: size)
        vector[0] = shardResponseMagic
        vector[1] = Float(response.nextPos)
        vector[2] = Float(response.tokens.count)
        for (i, token) in response.tokens.enumerated() {
            vector[headerWidth + i] = Float(token)
        }
        for i in 0..<response.hiddenState.count {
            vector[needed + i] = response.hiddenState[i]
        }
        return vector
    }

    // MARK: - Decode

    public static func isShardRequest(_ vector: [Float]) -> Bool {
        vector.first == shardRequestMagic
    }

    public static func isShardResponse(_ vector: [Float]) -> Bool {
        vector.first == shardResponseMagic
    }

    public static func decodeShardRequest(_ vector: [Float]) throws -> ShardRequest {
        guard vector.first == shardRequestMagic else {
            throw NMPLlamaShardWireError.notAShardRequest
        }
        guard vector.count >= headerWidth else {
            throw NMPLlamaShardWireError.malformed("width \(vector.count) below header")
        }
        let basePos = Int(vector[1])
        let tokenCount = Int(vector[2])
        guard tokenCount >= 0, headerWidth + tokenCount <= vector.count else {
            throw NMPLlamaShardWireError.malformed("invalid tokenCount \(tokenCount)")
        }
        var tokens: [Int32] = []
        for i in 0..<tokenCount {
            tokens.append(Int32(vector[headerWidth + i]))
        }
        let hiddenOffset = headerWidth + tokenCount
        let hiddenState = Array(vector[hiddenOffset...])
        return ShardRequest(basePos: basePos, tokens: tokens, hiddenState: hiddenState)
    }

    public static func decodeShardResponse(_ vector: [Float]) throws -> ShardResponse {
        guard vector.first == shardResponseMagic else {
            throw NMPLlamaShardWireError.notAShardResponse
        }
        guard vector.count >= headerWidth else {
            throw NMPLlamaShardWireError.malformed("width \(vector.count) below header")
        }
        let nextPos = Int(vector[1])
        let tokenCount = Int(vector[2])
        guard tokenCount >= 0, headerWidth + tokenCount <= vector.count else {
            throw NMPLlamaShardWireError.malformed("invalid tokenCount \(tokenCount)")
        }
        var tokens: [Int32] = []
        for i in 0..<tokenCount {
            tokens.append(Int32(vector[headerWidth + i]))
        }
        let hiddenOffset = headerWidth + tokenCount
        let hiddenState = Array(vector[hiddenOffset...])
        return ShardResponse(nextPos: nextPos, tokens: tokens, hiddenState: hiddenState)
    }
}
