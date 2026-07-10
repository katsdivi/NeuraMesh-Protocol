//
//  LlamaWire.swift
//  NMP — Phase 8
//
//  Token-state wire convention for real-LLM shards. The mesh transports
//  fixed-width [Float] activation tensors (hiddenSize wide); a llama.cpp
//  shard cannot accept raw mid-layer activations (llama.cpp's public API
//  executes the whole model per decode step, and the KV cache lives inside
//  the peer's llama context), so for llama plans the tensor carries token
//  state instead:
//
//    request  (coordinator → peer):
//      [0] magic 0x4C5051 ("LPQ")   [1] basePos   [2] tokenCount n
//      [3 ..< 3+n] token ids
//    response (peer → coordinator):
//      [0] magic 0x4C5052 ("LPR")   [1] nextPos   [2] candidateCount k
//      [3 ..< 3+2k] (tokenID, logit) pairs, sorted by logit descending
//
//  Every value is a Float32; token ids and positions are < 2^24 so the
//  encoding is exact. The peer trims its KV cache to basePos before
//  decoding, which makes retried/replayed requests idempotent and lets a
//  fresh prompt (basePos 0) reset the context implicitly. The coordinator
//  samples from the returned candidates (greedy in Phase 8), so mesh
//  output is exactly reproducible: local plan and remote plan produce
//  identical token streams from identical weights.
//
//  This file is pure Swift (no llama.cpp) so the format is unit-testable
//  everywhere.
//

import Foundation

public enum NMPLlamaWireError: Error, Equatable, Sendable {
    /// Vector too narrow for the header + payload.
    case capacityExceeded(needed: Int, width: Int)
    case notARequest
    case notAResponse
    case malformed(String)
    case valueNotExact(Int)
}

public enum NMPLlamaWire {

    public static let requestMagic: Float = 5_000_273  // 0x4C5051 "LPQ"
    public static let responseMagic: Float = 5_000_274 // 0x4C5052 "LPR"
    // Phase 9 speculative decoding: same layouts, distinct magics. A
    // verify request asks the peer for the PER-POSITION greedy argmax of
    // every decoded token (not just the last), so the coordinator can
    // check a whole draft in one round trip.
    public static let verifyRequestMagic: Float = 5_000_275  // 0x4C5053 "LPS"
    public static let verifyResponseMagic: Float = 5_000_276 // 0x4C5054 "LPT"
    static let headerWidth = 3

    /// Largest integer the format may carry (exact as Float32).
    public static let maxExactValue = 1 << 24

    public struct Request: Equatable, Sendable {
        /// Sequence position of the first token; the peer trims its KV
        /// cache to this position before decoding.
        public let basePos: Int
        public let tokens: [Int32]
        public init(basePos: Int, tokens: [Int32]) {
            self.basePos = basePos
            self.tokens = tokens
        }
    }

    public struct Response: Equatable, Sendable {
        /// basePos + decoded token count: where the next request continues.
        public let nextPos: Int
        /// (tokenID, logit) sorted by logit descending — real model logits.
        public let candidates: [(id: Int32, logit: Float)]

        public init(nextPos: Int, candidates: [(id: Int32, logit: Float)]) {
            self.nextPos = nextPos
            self.candidates = candidates
        }

        /// Greedy sample: the argmax candidate (deterministic).
        public var top: (id: Int32, logit: Float)? { candidates.first }

        public static func == (lhs: Response, rhs: Response) -> Bool {
            lhs.nextPos == rhs.nextPos
                && lhs.candidates.count == rhs.candidates.count
                && zip(lhs.candidates, rhs.candidates).allSatisfy {
                    $0.id == $1.id && $0.logit == $1.logit
                }
        }
    }

    /// Per-position greedy verdicts for a verify request: verdicts[i] is
    /// the model's argmax AFTER decoding request token i — i.e. what the
    /// model itself would generate next at that point. The coordinator
    /// accepts the longest draft prefix that matches, plus one bonus
    /// token (the verdict after the last accepted draft).
    public struct VerifyResponse: Equatable, Sendable {
        /// basePos + decoded token count (the full batch — the caller
        /// rewinds via basePos when a draft suffix is rejected).
        public let nextPos: Int
        public let verdicts: [(id: Int32, logit: Float)]

        public init(nextPos: Int, verdicts: [(id: Int32, logit: Float)]) {
            self.nextPos = nextPos
            self.verdicts = verdicts
        }

        public static func == (lhs: VerifyResponse, rhs: VerifyResponse) -> Bool {
            lhs.nextPos == rhs.nextPos
                && lhs.verdicts.count == rhs.verdicts.count
                && zip(lhs.verdicts, rhs.verdicts).allSatisfy {
                    $0.id == $1.id && $0.logit == $1.logit
                }
        }
    }

    /// Max prompt tokens a `width`-wide tensor can carry.
    public static func requestCapacity(width: Int) -> Int {
        max(0, width - headerWidth)
    }

    /// Max per-position verdicts a `width`-wide tensor can carry (same
    /// pair layout as responses).
    public static func verifyCapacity(width: Int) -> Int {
        responseCapacity(width: width)
    }

    /// Max (tokenID, logit) candidates a `width`-wide tensor can carry.
    public static func responseCapacity(width: Int) -> Int {
        max(0, (width - headerWidth) / 2)
    }

    // MARK: Encode

    public static func encode(_ request: Request, width: Int) throws -> [Float] {
        guard request.tokens.count <= requestCapacity(width: width) else {
            throw NMPLlamaWireError.capacityExceeded(
                needed: headerWidth + request.tokens.count, width: width)
        }
        try validateExact(request.basePos)
        try validateExact(request.basePos + request.tokens.count)
        var vector = [Float](repeating: 0, count: width)
        vector[0] = requestMagic
        vector[1] = Float(request.basePos)
        vector[2] = Float(request.tokens.count)
        for (offset, token) in request.tokens.enumerated() {
            try validateExact(Int(token))
            vector[headerWidth + offset] = Float(token)
        }
        return vector
    }

    public static func encode(_ response: Response, width: Int) throws -> [Float] {
        guard response.candidates.count <= responseCapacity(width: width) else {
            throw NMPLlamaWireError.capacityExceeded(
                needed: headerWidth + response.candidates.count * 2, width: width)
        }
        try validateExact(response.nextPos)
        var vector = [Float](repeating: 0, count: width)
        vector[0] = responseMagic
        vector[1] = Float(response.nextPos)
        vector[2] = Float(response.candidates.count)
        for (offset, candidate) in response.candidates.enumerated() {
            try validateExact(Int(candidate.id))
            vector[headerWidth + offset * 2] = Float(candidate.id)
            vector[headerWidth + offset * 2 + 1] = candidate.logit
        }
        return vector
    }

    /// Encodes a verify request: identical layout to a request, verify
    /// magic. The peer decodes the tokens at basePos… and returns one
    /// greedy verdict per position.
    public static func encodeVerify(_ request: Request, width: Int) throws -> [Float] {
        var vector = try encode(request, width: width)
        vector[0] = verifyRequestMagic
        return vector
    }

    public static func encode(_ response: VerifyResponse, width: Int) throws -> [Float] {
        guard response.verdicts.count <= verifyCapacity(width: width) else {
            throw NMPLlamaWireError.capacityExceeded(
                needed: headerWidth + response.verdicts.count * 2, width: width)
        }
        try validateExact(response.nextPos)
        var vector = [Float](repeating: 0, count: width)
        vector[0] = verifyResponseMagic
        vector[1] = Float(response.nextPos)
        vector[2] = Float(response.verdicts.count)
        for (offset, verdict) in response.verdicts.enumerated() {
            try validateExact(Int(verdict.id))
            vector[headerWidth + offset * 2] = Float(verdict.id)
            vector[headerWidth + offset * 2 + 1] = verdict.logit
        }
        return vector
    }

    // MARK: Decode

    public static func isRequest(_ vector: [Float]) -> Bool {
        vector.first == requestMagic
    }

    public static func isVerifyRequest(_ vector: [Float]) -> Bool {
        vector.first == verifyRequestMagic
    }

    public static func decodeRequest(_ vector: [Float]) throws -> Request {
        guard vector.first == requestMagic else { throw NMPLlamaWireError.notARequest }
        return try decodeRequestBody(vector)
    }

    public static func decodeVerifyRequest(_ vector: [Float]) throws -> Request {
        guard vector.first == verifyRequestMagic else { throw NMPLlamaWireError.notARequest }
        return try decodeRequestBody(vector)
    }

    private static func decodeRequestBody(_ vector: [Float]) throws -> Request {
        guard vector.count >= headerWidth else {
            throw NMPLlamaWireError.malformed("width \(vector.count) below header")
        }
        let basePos = try exactInt(vector[1], field: "basePos")
        let count = try exactInt(vector[2], field: "tokenCount")
        guard count >= 0, headerWidth + count <= vector.count else {
            throw NMPLlamaWireError.malformed("tokenCount \(count) exceeds width \(vector.count)")
        }
        let tokens = try (0..<count).map { offset -> Int32 in
            Int32(try exactInt(vector[headerWidth + offset], field: "token[\(offset)]"))
        }
        return Request(basePos: basePos, tokens: tokens)
    }

    public static func decodeVerifyResponse(_ vector: [Float]) throws -> VerifyResponse {
        guard vector.first == verifyResponseMagic else { throw NMPLlamaWireError.notAResponse }
        guard vector.count >= headerWidth else {
            throw NMPLlamaWireError.malformed("width \(vector.count) below header")
        }
        let nextPos = try exactInt(vector[1], field: "nextPos")
        let count = try exactInt(vector[2], field: "verdictCount")
        guard count >= 0, headerWidth + count * 2 <= vector.count else {
            throw NMPLlamaWireError.malformed("verdictCount \(count) exceeds width \(vector.count)")
        }
        let verdicts = try (0..<count).map { offset -> (id: Int32, logit: Float) in
            let id = Int32(try exactInt(vector[headerWidth + offset * 2],
                                        field: "verdict[\(offset)].id"))
            return (id, vector[headerWidth + offset * 2 + 1])
        }
        return VerifyResponse(nextPos: nextPos, verdicts: verdicts)
    }

    public static func decodeResponse(_ vector: [Float]) throws -> Response {
        guard vector.first == responseMagic else { throw NMPLlamaWireError.notAResponse }
        guard vector.count >= headerWidth else {
            throw NMPLlamaWireError.malformed("width \(vector.count) below header")
        }
        let nextPos = try exactInt(vector[1], field: "nextPos")
        let count = try exactInt(vector[2], field: "candidateCount")
        guard count >= 0, headerWidth + count * 2 <= vector.count else {
            throw NMPLlamaWireError.malformed("candidateCount \(count) exceeds width \(vector.count)")
        }
        let candidates = try (0..<count).map { offset -> (id: Int32, logit: Float) in
            let id = Int32(try exactInt(vector[headerWidth + offset * 2],
                                        field: "candidate[\(offset)].id"))
            return (id, vector[headerWidth + offset * 2 + 1])
        }
        return Response(nextPos: nextPos, candidates: candidates)
    }

    // MARK: Exactness

    private static func validateExact(_ value: Int) throws {
        guard value >= 0, value < maxExactValue else {
            throw NMPLlamaWireError.valueNotExact(value)
        }
    }

    private static func exactInt(_ value: Float, field: String) throws -> Int {
        guard value >= 0, value < Float(maxExactValue),
              value == value.rounded(.towardZero) else {
            throw NMPLlamaWireError.malformed("\(field) = \(value) is not an exact index")
        }
        return Int(value)
    }
}
