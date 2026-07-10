//
//  SpeculativeDecodingTests.swift
//  NMP — Phase 9
//
//  Three tiers:
//
//  1. Wire format + drafter logic — pure Swift, always runs.
//
//  2. Full service over the REAL protocol stack with a deterministic toy
//     LM (no GGUF needed): a fake engine that speaks the token-state wire
//     and whose "argmax" is a hash of the decoded tape. This pins the
//     accept/reject/rewind logic exactly: whatever the drafter does,
//     speculative output must equal the plain greedy stream.
//
//  3. Real-model identity — needs shim + GGUF (XCTSkips otherwise):
//     speculative llama output must be byte-identical to Phase 8's
//     non-speculative output, and zero-trim must slash the payload.
//

import XCTest
@testable import NMP

// MARK: - Tier 1: verify wire format

final class LlamaVerifyWireTests: XCTestCase {

    func testVerifyRequestRoundTrip() throws {
        let request = NMPLlamaWire.Request(basePos: 9, tokens: [5, 6, 7, 8])
        let vector = try NMPLlamaWire.encodeVerify(request, width: 64)
        XCTAssertTrue(NMPLlamaWire.isVerifyRequest(vector))
        XCTAssertFalse(NMPLlamaWire.isRequest(vector),
                       "verify and plain requests must never be confused")
        XCTAssertEqual(try NMPLlamaWire.decodeVerifyRequest(vector), request)
        XCTAssertThrowsError(try NMPLlamaWire.decodeRequest(vector))
    }

    func testVerifyResponseRoundTripPreservesVerdicts() throws {
        let response = NMPLlamaWire.VerifyResponse(
            nextPos: 13,
            verdicts: [(11, 3.5), (23, -0.25), (42, 9.0)])
        let vector = try NMPLlamaWire.encode(response, width: 64)
        let decoded = try NMPLlamaWire.decodeVerifyResponse(vector)
        XCTAssertEqual(decoded, response)
        XCTAssertThrowsError(try NMPLlamaWire.decodeResponse(vector),
                             "verify responses use their own magic")
    }

    func testVerifyCapacityIsEnforced() {
        let verdicts = (0..<31).map { (Int32($0), Float($0)) }
        XCTAssertThrowsError(try NMPLlamaWire.encode(
            NMPLlamaWire.VerifyResponse(nextPos: 0, verdicts: verdicts), width: 64))
        XCTAssertNoThrow(try NMPLlamaWire.encode(
            NMPLlamaWire.VerifyResponse(nextPos: 0, verdicts: verdicts), width: 65))
    }
}

// MARK: - Tier 1: prompt-lookup drafter

final class PromptLookupDrafterTests: XCTestCase {

    func testDraftsContinuationOfRepeatedNgram() {
        let drafter = NMPPromptLookupDrafter()
        // "1 2 3 4 5 … 1 2 3" — tail [2,3] last occurred followed by 4,5,9.
        let context: [Int32] = [1, 2, 3, 4, 5, 9, 1, 2, 3]
        XCTAssertEqual(drafter.draft(context: context, count: 3), [4, 5, 9])
        XCTAssertEqual(drafter.draft(context: context, count: 2), [4, 5])
    }

    func testPrefersMostRecentMatch() {
        let drafter = NMPPromptLookupDrafter()
        // [7,8] appears twice with different continuations; the LATER one wins.
        let context: [Int32] = [7, 8, 100, 5, 7, 8, 200, 6, 7, 8]
        XCTAssertEqual(drafter.draft(context: context, count: 1), [200])
    }

    func testNoMatchDraftsNothing() {
        let drafter = NMPPromptLookupDrafter()
        XCTAssertEqual(drafter.draft(context: [1, 2, 3, 4, 5], count: 4), [])
        XCTAssertEqual(drafter.draft(context: [], count: 4), [])
        XCTAssertEqual(drafter.draft(context: [1, 2, 3], count: 0), [])
    }

    func testLongerNgramWinsOverShorter() {
        let drafter = NMPPromptLookupDrafter(maxNgram: 3, minNgram: 2)
        // Tail [5,6,7]: 3-gram match continues with 30; a mere 2-gram
        // match [6,7] (later!) continues with 99. The 3-gram must win.
        let context: [Int32] = [5, 6, 7, 30, 1, 6, 7, 99, 2, 5, 6, 7]
        XCTAssertEqual(drafter.draft(context: context, count: 1), [30])
    }
}

// MARK: - Toy LM (deterministic fake for tier 2)

/// A fake "LLM" speaking the token-state wire: its greedy argmax after a
/// decoded tape is a hash of that tape. It keeps a KV-cache-like tape and
/// honors basePos trimming — exactly llama's contract, minus the math.
private final class ToyTokenEngine: NMPShardComputeEngine {
    let layerCount = 4
    let hiddenSize = 64
    private var tape: [Int32] = []

    static let vocabSize: Int32 = 997

    static func next(after tape: [Int32]) -> Int32 {
        var hash: UInt64 = 0x9E37_79B9_7F4A_7C15
        for token in tape {
            hash = (hash ^ UInt64(UInt32(bitPattern: token))) &* 0x1_0000_0001_B3
        }
        return Int32(hash % UInt64(vocabSize))
    }

    /// The plain greedy stream the mesh must reproduce, speculation or not.
    static func greedyStream(prompt: [Int32], count: Int) -> [Int32] {
        var tape = prompt
        var out: [Int32] = []
        for _ in 0..<count {
            let token = next(after: tape)
            out.append(token)
            tape.append(token)
        }
        return out
    }

    func runLayers(start: Int, end: Int, input: [Float]) throws -> [Float] {
        if NMPLlamaWire.isVerifyRequest(input) {
            let request = try NMPLlamaWire.decodeVerifyRequest(input)
            tape = Array(tape.prefix(request.basePos)) + request.tokens
            let verdicts = (0..<request.tokens.count).map { index in
                (id: Self.next(after: Array(tape.prefix(request.basePos + index + 1))),
                 logit: Float(1))
            }
            return try NMPLlamaWire.encode(
                NMPLlamaWire.VerifyResponse(nextPos: tape.count, verdicts: verdicts),
                width: hiddenSize)
        }
        let request = try NMPLlamaWire.decodeRequest(input)
        tape = Array(tape.prefix(request.basePos)) + request.tokens
        return try NMPLlamaWire.encode(
            NMPLlamaWire.Response(nextPos: tape.count,
                                  candidates: [(Self.next(after: tape), 1)]),
            width: hiddenSize)
    }
}

/// Tokenizer for the toy vocabulary: characters → token ids, pieces are
/// "«id»" markers, and one designated id is EOS.
private final class ToyTokenizer: NMPLlamaTokenizing {
    let hiddenSize = 64
    var eosToken: Int32 = -1 // no EOS by default

    func tokenize(_ text: String, addSpecial: Bool) throws -> [Int32] {
        text.unicodeScalars.map { Int32($0.value % UInt32(ToyTokenEngine.vocabSize)) }
    }

    func pieceBytes(for token: Int32) throws -> Data {
        Data("«\(token)»".utf8)
    }

    func isEndOfGeneration(_ token: Int32) -> Bool {
        token == eosToken
    }
}

/// Always proposes the CORRECT continuation (it knows the toy hash) —
/// the maximal-acceptance regime.
private final class OracleDrafter: NMPSpeculativeDrafter {
    let drafterName = "oracle"
    func draft(context: [Int32], count: Int) -> [Int32] {
        ToyTokenEngine.greedyStream(prompt: context, count: count)
    }
}

/// Always proposes WRONG tokens — the pathological regime; output must
/// still be perfect (drafts only cost round trips, never correctness).
private final class AdversarialDrafter: NMPSpeculativeDrafter {
    let drafterName = "adversarial"
    func draft(context: [Int32], count: Int) -> [Int32] {
        let stream = ToyTokenEngine.greedyStream(prompt: context, count: count)
        return stream.map { ($0 &+ 1) % ToyTokenEngine.vocabSize }
    }
}

// MARK: - Tier 2: service over the real stack

final class SpeculativeServiceTests: XCTestCase {

    private func makeMesh() throws -> NMPLlamaTestbed {
        let testbed = try NMPLlamaTestbed(
            engine: ToyTokenEngine(), modelTag: "toy-lm", placement: .remotePeer)
        try testbed.startSync()
        return testbed
    }

    private func expectedStream(prompt: String, count: Int,
                                tokenizer: ToyTokenizer) throws -> [Int32] {
        ToyTokenEngine.greedyStream(
            prompt: try tokenizer.tokenize(prompt, addSpecial: true), count: count)
    }

    func testOracleDraftsCollapseRoundTripsWithIdenticalOutput() throws {
        let testbed = try makeMesh()
        let tokenizer = ToyTokenizer()
        let service = NMPSpeculativeGenerationService(
            orchestrator: testbed.orchestrator, model: tokenizer,
            drafter: OracleDrafter(), depth: 4)

        let result = try service.runSync(prompt: "hello mesh", maxTokens: 20)
        let expected = try expectedStream(prompt: "hello mesh", count: 20,
                                          tokenizer: tokenizer)

        XCTAssertEqual(result.tokenCount, 20)
        XCTAssertEqual(result.text,
                       expected.map { "«\($0)»" }.joined(),
                       "speculative stream must equal the plain greedy stream")
        let stats = try XCTUnwrap(result.speculation)
        XCTAssertEqual(stats.acceptanceRate, 1.0)
        // Perfect drafts at depth 4: every verify trip yields 5 tokens
        // (4 drafts + bonus). 20 tokens ≈ 1 prompt trip + 4 verify trips.
        XCTAssertLessThanOrEqual(stats.meshRoundTrips, 6,
                                 "expected ~5 round trips, got \(stats.meshRoundTrips)")
        XCTAssertGreaterThan(
            stats.tokensPerRoundTrip(tokenCount: result.tokenCount), 3.0)
    }

    func testAdversarialDraftsNeverCorruptOutput() throws {
        let testbed = try makeMesh()
        let tokenizer = ToyTokenizer()
        let service = NMPSpeculativeGenerationService(
            orchestrator: testbed.orchestrator, model: tokenizer,
            drafter: AdversarialDrafter(), depth: 4)

        let result = try service.runSync(prompt: "hello mesh", maxTokens: 12)
        let expected = try expectedStream(prompt: "hello mesh", count: 12,
                                          tokenizer: tokenizer)

        XCTAssertEqual(result.text, expected.map { "«\($0)»" }.joined(),
                       "an always-wrong drafter must cost trips, not correctness")
        let stats = try XCTUnwrap(result.speculation)
        XCTAssertEqual(stats.acceptedDraftTokens, 0)
        // Every rejected round still yields exactly its bonus token — the
        // Phase 8 rate (one token per trip, plus the prompt trip).
        XCTAssertEqual(stats.meshRoundTrips, 12)
    }

    func testEmptyDraftFallbackMatchesPlainStream() throws {
        let testbed = try makeMesh()
        let tokenizer = ToyTokenizer()
        // Prompt-lookup over hash-noise tokens finds no n-gram repeats →
        // every round falls back to a plain single-token step.
        let service = NMPSpeculativeGenerationService(
            orchestrator: testbed.orchestrator, model: tokenizer,
            drafter: NMPPromptLookupDrafter(), depth: 4)

        let result = try service.runSync(prompt: "xyz", maxTokens: 8)
        let expected = try expectedStream(prompt: "xyz", count: 8,
                                          tokenizer: tokenizer)
        XCTAssertEqual(result.text, expected.map { "«\($0)»" }.joined())
        let stats = try XCTUnwrap(result.speculation)
        XCTAssertGreaterThan(stats.fallbackRounds, 0)
    }

    func testEosInsideAcceptedDraftStopsGeneration() throws {
        let testbed = try makeMesh()
        let tokenizer = ToyTokenizer()
        // Make the 3rd greedy token the EOS: generation must stop before it.
        let prompt = "stop here"
        let stream = try expectedStream(prompt: prompt, count: 8, tokenizer: tokenizer)
        tokenizer.eosToken = stream[2]

        let service = NMPSpeculativeGenerationService(
            orchestrator: testbed.orchestrator, model: tokenizer,
            drafter: OracleDrafter(), depth: 4)
        let result = try service.runSync(prompt: prompt, maxTokens: 8)

        XCTAssertEqual(result.tokenCount, 2,
                       "EOS at position 2 must end generation with 2 tokens")
        XCTAssertEqual(result.text, stream.prefix(2).map { "«\($0)»" }.joined())
    }

    func testBusyRejectionWhileGenerationInFlight() throws {
        let testbed = try makeMesh()
        let service = NMPSpeculativeGenerationService(
            orchestrator: testbed.orchestrator, model: ToyTokenizer(),
            drafter: OracleDrafter())

        let first = expectation(description: "first generation")
        service.run(prompt: "hello mesh again", maxTokens: 40) { _ in
            first.fulfill()
        }
        let second = expectation(description: "second rejected")
        service.run(prompt: "another", maxTokens: 4) { result in
            if case .failure(.busy) = result {
                second.fulfill()
            } else {
                XCTFail("expected .busy, got \(result)")
            }
        }
        wait(for: [second, first], timeout: 30)
    }
}

// MARK: - Tier 3: real model (skips without shim + GGUF)

final class SpeculativeLlamaTests: XCTestCase {

    /// The Phase 9 headline: speculative output over the full protocol
    /// stack is byte-identical to Phase 8's plain greedy output, in fewer
    /// round trips on a self-repetitive prompt.
    func testSpeculativeOutputIdenticalToPlainGreedy() throws {
        let model = try LlamaTestSupport.requireFullModel()
        guard model.runtime.supportsSpeculation else {
            throw XCTSkip("shim predates Phase 9 — rerun scripts/setup_llama.sh")
        }
        let engine = NMPLlamaComputeEngine(model: model)
        // Repetition gives the prompt-lookup drafter something to bite on.
        let prompt = "one two three four one two three four one two"
        let tokens = 16

        let plainBed = try NMPLlamaTestbed(engine: engine, modelTag: model.name,
                                           placement: .remotePeer)
        try plainBed.startSync()
        let plain = NMPPromptInferenceService(
            orchestrator: plainBed.orchestrator,
            codec: NMPLlamaPromptCodec(model: model))
        let plainDone = expectation(description: "plain")
        var plainResult: NMPPromptInferenceService.GenerationResult?
        plain.run(prompt: prompt, maxTokens: tokens) { result in
            plainResult = try? result.get()
            plainDone.fulfill()
        }
        wait(for: [plainDone], timeout: 300)

        let specBed = try NMPLlamaTestbed(engine: engine, modelTag: model.name,
                                          placement: .remotePeer)
        try specBed.startSync()
        specBed.orchestrator.activationWireFormat = .zeroTrimmed
        let speculative = NMPSpeculativeGenerationService(
            orchestrator: specBed.orchestrator, model: model,
            drafter: NMPPromptLookupDrafter())
        let specResult = try speculative.runSync(prompt: prompt, maxTokens: tokens)

        let expected = try XCTUnwrap(plainResult)
        XCTAssertEqual(specResult.text, expected.text,
                       "speculation must not change the token stream")
        let stats = try XCTUnwrap(specResult.speculation)
        XCTAssertGreaterThan(stats.meshRoundTrips, 0)
        XCTAssertLessThanOrEqual(stats.meshRoundTrips, expected.tokenCount + 1)
        // Zero-trim moved dramatically fewer bytes than Phase 8's padded
        // vectors (16 KB × 2 per round trip at width 4096).
        XCTAssertLessThan(specResult.networkPayloadBytes,
                          expected.networkPayloadBytes / 4)
    }

    /// Per-position greedy decode agrees with sequential last-position
    /// decodes — the property draft verification rests on.
    func testDecodeGreedyPerPositionMatchesSequentialDecodes() throws {
        let model = try LlamaTestSupport.requireFullModel()
        guard model.runtime.supportsSpeculation else {
            throw XCTSkip("shim predates Phase 9 — rerun scripts/setup_llama.sh")
        }
        let tokens = try model.tokenize("The capital of France is")

        let batched = try model.decodeGreedyPerPosition(tokens: tokens, basePos: 0)
        XCTAssertEqual(batched.count, tokens.count)

        // Sequential ground truth: argmax after each prefix.
        for prefixLength in [tokens.count - 2, tokens.count] {
            let prefix = Array(tokens.prefix(prefixLength))
            let sequential = try model.decodeTopK(tokens: prefix, basePos: 0, k: 1)
            XCTAssertEqual(batched[prefixLength - 1].id, sequential[0].id,
                           "batched argmax at position \(prefixLength - 1) diverged")
        }
    }
}
