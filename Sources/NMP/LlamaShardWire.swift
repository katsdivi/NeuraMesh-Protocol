//
//  LlamaShardWire.swift
//  NMP — Phase 10 (cross-device sharding)
//
//  Inter-shard wire format for cross-device llama plans. When the model
//  is split across multiple peers, the intermediate shards exchange
//  HIDDEN STATES (the real activation vector from the transformer layers)
//  alongside the current sequence tokens to permit downstream shards to
//  run the decode logic.
//
//  Wire layout (all values are Float32):
//
//    shard request  (coordinator → non-first peer):
//      [0] magic 0x4E5348 ("NSH" — NMP Shard Hidden)
//      [1] basePos        — KV cache trim position
//      [2] tokenCount     — how many tokens are packaged
//      [3 ... 3 + tokenCount - 1] tokens — cast to Float
//      [3 + tokenCount ...] hidden state — n_embd floats (truncated to fit)
//
//    shard response (non-last peer → coordinator):
//      [0] magic 0x4E5352 ("NSR" — NMP Shard Response)
//      [1] nextPos        — basePos + tokenCount
//      [2] tokenCount     — how many tokens are packaged
//      [3 ... 3 + tokenCount - 1] tokens — cast to Float
//      [3 + tokenCount ...] hidden state — n_embd floats (truncated to fit)
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

    // MARK: - Capacity

    /// Max hidden state floats a `width`-wide tensor can carry.
    public static func maxHiddenCapacity(tokensCount: Int, width: Int) -> Int {
        max(0, width - headerWidth - tokensCount)
    }

    // MARK: - Encode

    public static func encode(_ request: ShardRequest, width: Int) throws -> [Float] {
        let needed = headerWidth + request.tokens.count
        let totalCount = needed + request.hiddenState.count
        let size = max(width, totalCount)
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
        let totalCount = needed + response.hiddenState.count
        let size = max(width, totalCount)
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
