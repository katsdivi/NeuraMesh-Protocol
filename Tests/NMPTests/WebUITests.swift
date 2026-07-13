//
//  WebUITests.swift
//  NMP — Mesh 2.0
//
//  The multi-device web surface: the protocol comparison model (NMP row
//  measured, TCP/QUIC anchored to it), the new REST routes over real
//  loopback TCP, static SPA serving with its traversal guard, CORS, and
//  the LAN identity / QR helpers the startup banner uses.
//

import XCTest
@testable import NMP

// MARK: - Comparison model

final class ProtocolComparisonModelTests: XCTestCase {

    private let inputs = NMPProtocolComparisonModel.Inputs(
        tokens: 32, payloadBytes: 11_928, roundTrips: 33,
        measuredTotalSeconds: 2.2786)

    func testNMPRowIsTheMeasuredRun() {
        let estimates = NMPProtocolComparisonModel.compare(inputs)
        XCTAssertEqual(estimates.count, 3)
        let nmp = estimates[0]
        XCTAssertEqual(nmp.name, "NMP")
        XCTAssertTrue(nmp.measured)
        XCTAssertEqual(nmp.totalMs, 2278.6, accuracy: 0.01)
        XCTAssertEqual(nmp.tokensPerSec, 32 / 2.2786, accuracy: 0.01)
        XCTAssertTrue(estimates.dropFirst().allSatisfy { !$0.measured },
                      "TCP/QUIC must be labeled modeled")
    }

    func testModeledProtocolsCostMoreOnCleanLAN() {
        let estimates = NMPProtocolComparisonModel.compare(inputs)
        let nmp = estimates[0]
        for modeled in estimates.dropFirst() {
            XCTAssertGreaterThan(modeled.totalMs, nmp.totalMs,
                                 "\(modeled.name) should carry handshake + per-trip overhead")
            XCTAssertLessThan(modeled.tokensPerSec, nmp.tokensPerSec)
        }
        // TCP (2-RTT handshake, heavier per-trip) costs more than QUIC.
        XCTAssertGreaterThan(estimates[1].totalMs, estimates[2].totalMs)
    }

    func testLossWidensTheGapViaRecoveryCosts() {
        var lossy = inputs
        lossy.lossRate = 0.05
        let clean = NMPProtocolComparisonModel.compare(inputs)
        let underLoss = NMPProtocolComparisonModel.compare(lossy)
        let cleanGap = clean[1].totalMs - clean[0].totalMs
        let lossyGap = underLoss[1].totalMs - underLoss[0].totalMs
        XCTAssertGreaterThan(lossyGap, cleanGap,
                             "TCP retransmit vs FEC must diverge under loss")
        // NMP's own total is anchored to the measured run either way.
        XCTAssertEqual(underLoss[0].totalMs, clean[0].totalMs, accuracy: 0.001)
    }

    func testEveryEstimateCarriesAssumptionsAndJSONShape() {
        for estimate in NMPProtocolComparisonModel.compare(inputs) {
            XCTAssertFalse(estimate.assumptions.isEmpty)
            let object = estimate.asJSONObject
            XCTAssertNotNil(object["name"])
            XCTAssertNotNil(object["measured"])
            XCTAssertNotNil(object["total_ms"])
            XCTAssertNotNil(object["assumptions"])
            XCTAssertTrue(JSONSerialization.isValidJSONObject(object))
        }
    }
}

// MARK: - LAN identity + QR

final class LANIdentityTests: XCTestCase {

    func testHostnameEndsWithLocal() {
        let hostname = NMPLANIdentity.localHostname()
        XCTAssertFalse(hostname.isEmpty)
        XCTAssertTrue(hostname.hasSuffix(".local"))
    }

    func testQRCodeRendersMultilineBlocks() throws {
        guard let qr = NMPQRCode.ascii(for: "http://192.168.1.43:3000") else {
            throw XCTSkip("CoreImage unavailable on this platform")
        }
        let lines = qr.split(separator: "\n")
        XCTAssertGreaterThan(lines.count, 8, "QR should span multiple rows")
        XCTAssertTrue(qr.contains("█"), "QR should contain dark modules")
    }

    func testBannerContainsHostnameAndWarning() {
        let banner = NMPWebUIBanner.render(port: 3000, meshSummary: ["Mesh: test"])
        XCTAssertTrue(banner.contains(NMPLANIdentity.localHostname()))
        XCTAssertTrue(banner.contains(":3000"))
        XCTAssertTrue(banner.contains("Mesh: test"))
        XCTAssertTrue(banner.lowercased().contains("no tls"),
                      "the banner must carry the trusted-LAN warning")
    }
}

// MARK: - Web routes over loopback TCP

final class WebUIRouteTests: XCTestCase {

    private var server: NMPDashboardServer!

    override func setUpWithError() throws {
        server = NMPDashboardServer()
        try server.start(port: 0)
    }

    override func tearDown() {
        server.stop()
        server = nil
    }

    private var base: String { "http://127.0.0.1:\(server.boundPort)" }

    private func request(
        _ method: String, _ path: String, body: [String: Any]? = nil
    ) throws -> (status: Int, headers: [AnyHashable: Any], data: Data) {
        var request = URLRequest(url: URL(string: base + path)!)
        request.httpMethod = method
        if let body {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        let done = expectation(description: "\(method) \(path)")
        var outcome: (Int, [AnyHashable: Any], Data)?
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let http = response as? HTTPURLResponse {
                outcome = (http.statusCode, http.allHeaderFields, data ?? Data())
            } else {
                XCTFail("no response: \(String(describing: error))")
            }
            done.fulfill()
        }.resume()
        wait(for: [done], timeout: 10)
        return try XCTUnwrap(outcome)
    }

    private func json(_ data: Data) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    // MARK: /health + /api/devices

    func testHealthReportsMeshInfoAndPeerCounts() throws {
        var info = NMPDashboardServer.MeshInfo()
        info.engine = "llamaCpp"
        info.modelName = "test-model"
        info.shardCount = 2
        info.wireFormat = "zeroTrimmed"
        info.speculationAvailable = true
        server.meshInfo = info
        server.updatePeerState(peerID: 1, name: "mac", latencyMS: 3,
                               loadPercent: 10, assigned: "layers 0-15", alive: true)
        server.updatePeerState(peerID: 2, name: "iphone", latencyMS: 9,
                               loadPercent: 55, assigned: "layers 16-31", alive: false)

        let health = try json(try request("GET", "/health").data)
        XCTAssertEqual(health["status"] as? String, "ok")
        let mesh = try XCTUnwrap(health["mesh"] as? [String: Any])
        XCTAssertEqual(mesh["engine"] as? String, "llamaCpp")
        XCTAssertEqual(mesh["wire_format"] as? String, "zeroTrimmed")
        XCTAssertEqual(mesh["speculation_available"] as? Bool, true)
        XCTAssertEqual(mesh["peers"] as? Int, 2)
        XCTAssertEqual(mesh["peers_alive"] as? Int, 1)
    }

    func testDevicesListsPeerSnapshotsInPeerIDOrder() throws {
        server.updatePeerState(peerID: 7, name: "late", latencyMS: 1,
                               loadPercent: 1, assigned: "—", alive: true)
        server.updatePeerState(peerID: 2, name: "early", latencyMS: 2,
                               loadPercent: 2, assigned: "layers 0-3", alive: true)

        let (status, _, data) = try request("GET", "/api/devices")
        XCTAssertEqual(status, 200)
        let devices = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        XCTAssertEqual(devices.map { $0["name"] as? String }, ["early", "late"])
        XCTAssertEqual(devices[0]["assigned"] as? String, "layers 0-3")
    }

    // MARK: /api/comparison

    func testComparisonReturnsMeasuredPlusModeledRows() throws {
        let (status, _, data) = try request("POST", "/api/comparison", body: [
            "tokens": 32, "payload_bytes": 11928,
            "round_trips": 33, "measured_total_ms": 2278.6,
        ])
        XCTAssertEqual(status, 200)
        let object = try json(data)
        let protocols = try XCTUnwrap(object["protocols"] as? [[String: Any]])
        XCTAssertEqual(protocols.count, 3)
        XCTAssertEqual(protocols[0]["measured"] as? Bool, true)
        XCTAssertEqual(protocols[1]["measured"] as? Bool, false)
        XCTAssertNotNil(object["note"], "the modeled-vs-measured note is required")
    }

    func testComparisonRejectsMissingFields() throws {
        let (status, _, _) = try request("POST", "/api/comparison",
                                         body: ["tokens": 32])
        XCTAssertEqual(status, 400)
    }

    // MARK: /api/benchmark/run

    private static func fakeGeneration(
        latency: TimeInterval) -> NMPPromptInferenceService.GenerationResult {
        .init(text: "ok", tokenCount: 8, totalSeconds: latency,
              networkPayloadBytes: 1000, shardCount: 1,
              perTokenSeconds: [], engine: "test")
    }

    func testBenchmarkWithoutHandlerIs503() throws {
        let (status, _, _) = try request("POST", "/api/benchmark/run",
                                         body: ["prompt": "hi", "runs": 2])
        XCTAssertEqual(status, 503)
    }

    func testBenchmarkAggregatesRunsWithStdDev() throws {
        server.onBenchmarkRequest = { request, respond in
            XCTAssertEqual(request.runs, 3)
            respond(.success([
                Self.fakeGeneration(latency: 0.100),
                Self.fakeGeneration(latency: 0.200),
                Self.fakeGeneration(latency: 0.300),
            ]))
        }
        let (status, _, data) = try request("POST", "/api/benchmark/run", body: [
            "prompt": "hi", "max_tokens": 8, "runs": 3,
        ])
        XCTAssertEqual(status, 200)
        let object = try json(data)
        XCTAssertEqual(object["avg_latency_ms"] as? Double ?? 0, 200, accuracy: 0.5)
        // Population σ of {100, 200, 300} = 81.65.
        XCTAssertEqual(object["stddev_latency_ms"] as? Double ?? 0, 81.65, accuracy: 0.1)
        XCTAssertEqual((object["runs"] as? [[String: Any]])?.count, 3)
    }

    func testBenchmarkRunsAreClampedTo10() throws {
        server.onBenchmarkRequest = { request, respond in
            XCTAssertEqual(request.runs, 10, "runs must be clamped")
            respond(.success([Self.fakeGeneration(latency: 0.1)]))
        }
        _ = try request("POST", "/api/benchmark/run",
                        body: ["prompt": "hi", "runs": 5000])
    }

    // MARK: /api/inference comparison attachment

    /// Mesh 2.5: enable_comparison attaches the MEASURED transport race
    /// (real sockets on the generation's traffic pattern); the modeled
    /// protocol_comparison no longer rides on inference responses.
    func testInferenceAttachesMeasuredRaceWhenAsked() throws {
        server.onInferenceRequest = { request, respond in
            XCTAssertTrue(request.enableComparison)
            respond(.success(.init(
                text: "hello", tokenCount: 4, totalSeconds: 0.5,
                networkPayloadBytes: 2000, shardCount: 1,
                perTokenSeconds: [0.1, 0.1, 0.1, 0.2], engine: "test")))
        }
        let (status, _, data) = try request("POST", "/api/inference", body: [
            "prompt": "hi", "max_tokens": 4, "enable_comparison": true,
        ])
        XCTAssertEqual(status, 200)
        let object = try json(data)
        XCTAssertEqual(object["round_trips"] as? Int, 4)
        XCTAssertNil(object["protocol_comparison"],
                     "modeled numbers must not ride on inference anymore")
        let transportRace = try XCTUnwrap(
            object["transport_race"] as? [String: Any],
            "error: \(object["transport_race_error"] ?? "none")")
        let race = try XCTUnwrap(transportRace["race"] as? [String: Any])
        XCTAssertGreaterThanOrEqual(
            try XCTUnwrap(race["legs"] as? [[String: Any]]).count, 2)
    }

    // MARK: CORS

    func testPreflightAndCORSHeaders() throws {
        let (status, headers, _) = try request("OPTIONS", "/api/inference")
        XCTAssertEqual(status, 204)
        XCTAssertEqual(headers["Access-Control-Allow-Origin"] as? String, "*")

        let health = try request("GET", "/health")
        XCTAssertEqual(health.headers["Access-Control-Allow-Origin"] as? String, "*")
    }

    // MARK: Static SPA serving

    private func makePublicDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("nmp-web-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("assets"),
            withIntermediateDirectories: true)
        try Data("<html>app shell</html>".utf8)
            .write(to: root.appendingPathComponent("index.html"))
        try Data("console.log('app')".utf8)
            .write(to: root.appendingPathComponent("assets/app.js"))
        return root
    }

    func testStaticServingWithSPAFallback() throws {
        let root = try makePublicDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        server.publicDirectory = root

        let index = try request("GET", "/")
        XCTAssertEqual(index.status, 200)
        XCTAssertTrue(String(decoding: index.data, as: UTF8.self).contains("app shell"))

        let asset = try request("GET", "/assets/app.js")
        XCTAssertEqual(asset.status, 200)
        XCTAssertTrue((asset.headers["Content-Type"] as? String ?? "")
            .contains("javascript"))

        // Extension-less SPA route → app shell; missing asset → honest 404.
        let route = try request("GET", "/benchmark")
        XCTAssertEqual(route.status, 200)
        XCTAssertTrue(String(decoding: route.data, as: UTF8.self).contains("app shell"))
        XCTAssertEqual(try request("GET", "/assets/missing.js").status, 404)

        // Unknown API path never falls through to the shell.
        XCTAssertEqual(try request("GET", "/api/nope").status, 404)

        // Legacy dashboard stays reachable.
        let legacy = try request("GET", "/legacy")
        XCTAssertEqual(legacy.status, 200)
    }

    func testTraversalOutsidePublicRootIsRefused() throws {
        let root = try makePublicDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        // A sibling secret the server must never serve.
        let secret = root.deletingLastPathComponent()
            .appendingPathComponent("nmp-secret-\(UUID().uuidString).txt")
        try Data("secret".utf8).write(to: secret)
        defer { try? FileManager.default.removeItem(at: secret) }
        server.publicDirectory = root

        let traversal = try request(
            "GET", "/..%2F\(secret.lastPathComponent)")
        XCTAssertTrue([403, 404].contains(traversal.status),
                      "traversal must be refused, got \(traversal.status)")
        XCTAssertFalse(String(decoding: traversal.data, as: UTF8.self)
            .contains("secret"))
    }

    func testMissingPublicDirectoryFallsBackToLegacyDashboard() throws {
        // publicDirectory unset → embedded page at /.
        let (status, headers, _) = try request("GET", "/")
        XCTAssertEqual(status, 200)
        XCTAssertTrue((headers["Content-Type"] as? String ?? "").contains("text/html"))
    }
}

// MARK: - Chat prompt template (Mesh 2.7)

final class ChatPromptTests: XCTestCase {

    private func msg(_ role: NMPChatMessage.Role, _ content: String) -> NMPChatMessage {
        NMPChatMessage(role: role, content: content)
    }

    func testLlamaTemplateWrapsTurnsInInstructionBlocks() {
        let prompt = NMPChatPrompt.format(
            messages: [msg(.user, "Hi there"),
                       msg(.assistant, "Hello!"),
                       msg(.user, "How are you?")],
            engine: "llamaCpp")
        XCTAssertEqual(prompt,
                       "[INST] Hi there [/INST] Hello! [INST] How are you? [/INST]")
        XCTAssertFalse(prompt.contains("<s>"),
                       "the tokenizer adds BOS — never as literal text")
    }

    func testLlamaTemplateFoldsSystemIntoFirstInstructionOnly() {
        let prompt = NMPChatPrompt.format(
            messages: [msg(.system, "Be brief."),
                       msg(.user, "One"),
                       msg(.assistant, "1"),
                       msg(.user, "Two")],
            engine: "llamaCpp")
        XCTAssertEqual(prompt,
                       "[INST] <<SYS>>\nBe brief.\n<</SYS>>\n\nOne [/INST] 1 "
                       + "[INST] Two [/INST]")
        XCTAssertEqual(prompt.components(separatedBy: "<<SYS>>").count, 2,
                       "system prompt rides the FIRST instruction only")
    }

    func testTranscriptTemplateForReferenceEngine() {
        let prompt = NMPChatPrompt.format(
            messages: [msg(.system, "Talk like a pirate."),
                       msg(.user, "Hi"),
                       msg(.assistant, "Arr"),
                       msg(.user, "Bye")],
            engine: "reference")
        XCTAssertEqual(prompt,
                       "Talk like a pirate.\nUser: Hi\nAssistant: Arr\n"
                       + "User: Bye\nAssistant:")
    }
}

// MARK: - /api/chat route (Mesh 2.7)

final class ChatRouteTests: XCTestCase {

    private var server: NMPDashboardServer!

    override func setUpWithError() throws {
        server = NMPDashboardServer()
        try server.start(port: 0)
    }

    override func tearDown() {
        server.stop()
        server = nil
    }

    private var base: String { "http://127.0.0.1:\(server.boundPort)" }

    private func request(
        _ method: String, _ path: String, body: [String: Any]? = nil
    ) throws -> (status: Int, data: Data) {
        var request = URLRequest(url: URL(string: base + path)!)
        request.httpMethod = method
        if let body {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        let done = expectation(description: "\(method) \(path)")
        var outcome: (Int, Data)?
        URLSession.shared.dataTask(with: request) { data, response, _ in
            if let http = response as? HTTPURLResponse {
                outcome = (http.statusCode, data ?? Data())
            }
            done.fulfill()
        }.resume()
        wait(for: [done], timeout: 10)
        return try XCTUnwrap(outcome)
    }

    func testChatAssemblesEngineTemplateAndReturnsGenerationJSON() throws {
        var info = NMPDashboardServer.MeshInfo()
        info.engine = "llamaCpp"
        server.meshInfo = info

        var seenPrompt: String?
        server.onInferenceRequest = { request, respond in
            seenPrompt = request.prompt
            respond(.success(.init(
                text: "I am well.", tokenCount: 3, totalSeconds: 0.3,
                networkPayloadBytes: 900, shardCount: 1,
                perTokenSeconds: [0.1, 0.1, 0.1], engine: "llamaCpp")))
        }
        let (status, data) = try request("POST", "/api/chat", body: [
            "messages": [
                ["role": "user", "content": "Hi"],
                ["role": "assistant", "content": "Hello!"],
                ["role": "user", "content": "How are you?"],
            ],
            "max_tokens": 8,
        ])
        XCTAssertEqual(status, 200)
        XCTAssertEqual(seenPrompt,
                       "[INST] Hi [/INST] Hello! [INST] How are you? [/INST]")
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["output"] as? String, "I am well.")
        XCTAssertEqual(object["round_trips"] as? Int, 3)
        XCTAssertEqual(object["assembled_prompt_chars"] as? Int,
                       seenPrompt?.count)
    }

    func testChatRejectsEmptyAndNonUserFinalTurns() throws {
        server.onInferenceRequest = { _, _ in XCTFail("must not reach the mesh") }
        let (missing, _) = try request("POST", "/api/chat", body: ["messages": []])
        XCTAssertEqual(missing, 400)
        let (badRole, _) = try request("POST", "/api/chat", body: [
            "messages": [["role": "wizard", "content": "hi"]],
        ])
        XCTAssertEqual(badRole, 400)
        let (assistantLast, _) = try request("POST", "/api/chat", body: [
            "messages": [["role": "user", "content": "hi"],
                         ["role": "assistant", "content": "hello"]],
        ])
        XCTAssertEqual(assistantLast, 400,
                       "the last message must be the user's turn")
    }

    func testChatWithoutPipelineIs503() throws {
        let (status, _) = try request("POST", "/api/chat", body: [
            "messages": [["role": "user", "content": "hi"]],
        ])
        XCTAssertEqual(status, 503)
    }
}

// MARK: - /api/mesh/objective route (Mesh 2.8)

final class ObjectiveRouteTests: XCTestCase {

    private var server: NMPDashboardServer!

    override func setUpWithError() throws {
        server = NMPDashboardServer()
        try server.start(port: 0)
    }
    override func tearDown() { server.stop(); server = nil }

    private func post(_ body: [String: Any]) throws -> (Int, [String: Any]) {
        var request = URLRequest(url: URL(string:
            "http://127.0.0.1:\(server.boundPort)/api/mesh/objective")!)
        request.httpMethod = "POST"
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let done = expectation(description: "post")
        var outcome: (Int, [String: Any])?
        URLSession.shared.dataTask(with: request) { data, response, _ in
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            let object = (data.flatMap {
                try? JSONSerialization.jsonObject(with: $0) } as? [String: Any]) ?? [:]
            outcome = (status, object)
            done.fulfill()
        }.resume()
        wait(for: [done], timeout: 10)
        return try XCTUnwrap(outcome)
    }

    func testObjectiveSwitchReachesHandler() throws {
        var seen: String?
        server.onObjectiveRequest = { objective, respond in
            seen = objective
            respond(.success("switched to \(objective)"))
        }
        let (status, object) = try post(["objective": "speed"])
        XCTAssertEqual(status, 200)
        XCTAssertEqual(seen, "speed")
        XCTAssertEqual(object["objective"] as? String, "speed")
        XCTAssertEqual(object["status"] as? String, "ok")
    }

    func testObjectiveRejectsUnknownValue() throws {
        server.onObjectiveRequest = { _, respond in
            respond(.failure(.init("unknown objective 'turbo'")))
        }
        let (status, object) = try post(["objective": "turbo"])
        XCTAssertEqual(status, 400)
        XCTAssertNotNil(object["error"])
    }

    func testObjectiveMissingFieldIs400() throws {
        server.onObjectiveRequest = { _, _ in XCTFail("must not reach handler") }
        let (status, _) = try post([:])
        XCTAssertEqual(status, 400)
    }

    func testObjectiveWithoutHandlerIs503() throws {
        let (status, _) = try post(["objective": "speed"])
        XCTAssertEqual(status, 503)
    }
}
