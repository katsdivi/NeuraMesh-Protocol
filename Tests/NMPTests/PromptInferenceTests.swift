//
//  PromptInferenceTests.swift
//  NMPTests — Phase 6+
//
//  The prompt → mesh → text pipeline: deterministic embedding, the
//  autoregressive token loop over a real testbed mesh, busy rejection,
//  and the POST /api/inference HTTP path end-to-end via URLSession
//  against a live NMPDashboardServer on loopback.
//

import XCTest
@testable import NMP

final class PromptInferenceServiceTests: XCTestCase {

    private var testbed: NMPMeshTestbed!
    private var service: NMPPromptInferenceService!

    override func setUpWithError() throws {
        // Small, fast mesh: enough layers/peers to exercise real pipelining.
        testbed = try NMPMeshTestbed(layerCount: 8, hiddenSize: 64, remotePeerCount: 2)
        _ = try testbed.startSync()
        service = NMPPromptInferenceService(
            orchestrator: testbed.orchestrator, hiddenSize: testbed.hiddenSize)
    }

    override func tearDown() {
        service = nil
        testbed = nil
    }

    private func generate(prompt: String, maxTokens: Int,
                          timeout: TimeInterval = 20) throws
        -> NMPPromptInferenceService.GenerationResult {
        var outcome: Result<NMPPromptInferenceService.GenerationResult,
                            NMPPromptInferenceService.ServiceError>?
        let done = expectation(description: "generation")
        service.run(prompt: prompt, maxTokens: maxTokens) { result in
            outcome = result
            done.fulfill()
        }
        wait(for: [done], timeout: timeout)
        return try XCTUnwrap(outcome).get()
    }

    func testEmbeddingIsDeterministicAndPromptSensitive() {
        let a = NMPPromptInferenceService.embed(prompt: "hello mesh", hiddenSize: 64)
        let b = NMPPromptInferenceService.embed(prompt: "hello mesh", hiddenSize: 64)
        let c = NMPPromptInferenceService.embed(prompt: "another prompt", hiddenSize: 64)
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
        XCTAssertEqual(a.count, 64)
    }

    func testGeneratesRequestedTokensThroughMesh() throws {
        let result = try generate(prompt: "distributed inference", maxTokens: 4)
        XCTAssertEqual(result.tokenCount, 4)
        XCTAssertEqual(result.perTokenSeconds.count, 4)
        XCTAssertFalse(result.text.isEmpty)
        XCTAssertGreaterThan(result.networkPayloadBytes, 0,
                             "tokens must have crossed the (in-memory) network")
        XCTAssertEqual(result.shardCount, 3, "coordinator + 2 remote peers")
        XCTAssertEqual(result.engine, "reference")
    }

    func testSamePromptSameText() throws {
        let first = try generate(prompt: "determinism check", maxTokens: 3)
        let second = try generate(prompt: "determinism check", maxTokens: 3)
        XCTAssertEqual(first.text, second.text)
    }

    func testBusyRejectsConcurrentRun() throws {
        let firstDone = expectation(description: "first")
        let secondDone = expectation(description: "second")
        var secondOutcome: Result<NMPPromptInferenceService.GenerationResult,
                                  NMPPromptInferenceService.ServiceError>?
        service.run(prompt: "long generation", maxTokens: 8) { _ in
            firstDone.fulfill()
        }
        service.run(prompt: "should be rejected", maxTokens: 1) { result in
            secondOutcome = result
            secondDone.fulfill()
        }
        wait(for: [secondDone, firstDone], timeout: 20)
        guard case .failure(.busy) = secondOutcome else {
            return XCTFail("expected .busy, got \(String(describing: secondOutcome))")
        }
    }

    func testEmptyPromptRejected() {
        let done = expectation(description: "empty")
        service.run(prompt: "   \n", maxTokens: 2) { result in
            guard case .failure(.emptyPrompt) = result else {
                return XCTFail("expected .emptyPrompt, got \(result)")
            }
            done.fulfill()
        }
        wait(for: [done], timeout: 5)
    }

    func testMaxTokensClamped() throws {
        let result = try generate(
            prompt: "clamp", maxTokens: NMPPromptInferenceService.maxTokensPerRequest + 500,
            timeout: 60)
        XCTAssertEqual(result.tokenCount, NMPPromptInferenceService.maxTokensPerRequest)
    }
}

// MARK: - HTTP endpoint

final class InferenceHTTPTests: XCTestCase {

    private var server: NMPDashboardServer!

    override func setUpWithError() throws {
        server = NMPDashboardServer()
        try server.start(port: 0)
    }

    override func tearDown() {
        server.stop()
        server = nil
    }

    private func post(json: String, timeout: TimeInterval = 10) throws
        -> (status: Int, object: [String: Any]) {
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(server.boundPort)/api/inference")!)
        request.httpMethod = "POST"
        request.httpBody = Data(json.utf8)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        var status = 0
        var object: [String: Any] = [:]
        let done = expectation(description: "response")
        URLSession.shared.dataTask(with: request) { data, response, error in
            XCTAssertNil(error)
            status = (response as? HTTPURLResponse)?.statusCode ?? 0
            if let data,
               let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                object = parsed
            }
            done.fulfill()
        }.resume()
        wait(for: [done], timeout: timeout)
        return (status, object)
    }

    func testNoHandlerReturns503() throws {
        let (status, object) = try post(json: #"{"prompt": "hi"}"#)
        XCTAssertEqual(status, 503)
        XCTAssertNotNil(object["error"])
    }

    func testMalformedBodyReturns400() throws {
        server.onInferenceRequest = { _, respond in
            respond(.failure(status: 500, message: "handler must not be reached"))
        }
        let (status, _) = try post(json: "not json at all")
        XCTAssertEqual(status, 400)
    }

    func testMissingPromptReturns400() throws {
        server.onInferenceRequest = { _, respond in
            respond(.failure(status: 500, message: "handler must not be reached"))
        }
        let (status, object) = try post(json: #"{"max_tokens": 4}"#)
        XCTAssertEqual(status, 400)
        XCTAssertNotNil(object["error"])
    }

    func testSuccessRoundTripSerializesMetrics() throws {
        server.onInferenceRequest = { request, respond in
            XCTAssertEqual(request.prompt, "hello mesh")
            XCTAssertEqual(request.maxTokens, 6)
            respond(.success(NMPPromptInferenceService.GenerationResult(
                text: "Mesh moves tokens.", tokenCount: 6, totalSeconds: 0.3,
                networkPayloadBytes: 4096, shardCount: 3,
                perTokenSeconds: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                engine: "reference")))
        }
        let (status, object) = try post(json: #"{"prompt": "hello mesh", "max_tokens": 6}"#)
        XCTAssertEqual(status, 200)
        XCTAssertEqual(object["output"] as? String, "Mesh moves tokens.")
        XCTAssertEqual(object["token_count"] as? Int, 6)
        XCTAssertEqual(object["latency_ms"] as? Double, 300.0)
        XCTAssertEqual(object["tokens_per_sec"] as? Double, 20.0)
        XCTAssertEqual(object["shard_count"] as? Int, 3)
        XCTAssertEqual(object["engine"] as? String, "reference")
    }

    func testHandlerFailureStatusPropagates() throws {
        server.onInferenceRequest = { _, respond in
            respond(.failure(status: 429, message: "busy"))
        }
        let (status, object) = try post(json: #"{"prompt": "hi"}"#)
        XCTAssertEqual(status, 429)
        XCTAssertEqual(object["error"] as? String, "busy")
    }

    func testGETStillServesDashboard() throws {
        let done = expectation(description: "GET /")
        URLSession.shared.dataTask(
            with: URL(string: "http://127.0.0.1:\(server.boundPort)/")!) { data, response, _ in
            XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
            XCTAssertTrue(String(decoding: data ?? Data(), as: UTF8.self).contains("<"))
            done.fulfill()
        }.resume()
        wait(for: [done], timeout: 10)
    }
}
