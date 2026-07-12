//
//  Mesh21Tests.swift
//  NMP — Mesh 2.1
//
//  Real-time streaming, the measured transport race, live resource
//  metrics, compute-share allocation, and web-client tracking.
//

import XCTest
@testable import NMP

// MARK: - Resource monitor

final class ResourceMonitorTests: XCTestCase {

    func testSampleReportsRealKernelCounters() throws {
        let monitor = NMPResourceMonitor()
        let sample = monitor.sample()

        XCTAssertGreaterThan(sample.ramTotalBytes, 1 << 30,
                             "a Mac has at least 1 GB of RAM")
        XCTAssertGreaterThan(sample.ramUsedBytes, 0)
        XCTAssertLessThanOrEqual(sample.ramUsedBytes, sample.ramTotalBytes)
        XCTAssertGreaterThan(sample.processFootprintBytes, 1 << 20,
                             "this test process occupies more than 1 MB")
        XCTAssertGreaterThan(sample.storageTotalBytes, 1 << 30)
        XCTAssertLessThanOrEqual(sample.storageFreeBytes, sample.storageTotalBytes)
        XCTAssertNil(sample.cpuPercent,
                     "first sample has no tick baseline to diff against")

        // Host CPU tick counters advance at ~100 Hz; poll until a delta
        // shows up (a single fixed sleep is flaky under full-suite load).
        var cpu: Double?
        for _ in 0..<10 where cpu == nil {
            Thread.sleep(forTimeInterval: 0.3)
            cpu = monitor.sample().cpuPercent
        }
        let measured = try XCTUnwrap(cpu, "no CPU tick delta within 3 s")
        XCTAssertGreaterThanOrEqual(measured, 0)
        XCTAssertLessThanOrEqual(measured, 100)
    }

    func testJSONShapeCarriesEveryPanelField() {
        let monitor = NMPResourceMonitor()
        // cpu_percent needs a tick delta; poll rather than trust one sleep.
        var object = monitor.sample().asJSONObject
        for _ in 0..<10 where object["cpu_percent"] == nil {
            Thread.sleep(forTimeInterval: 0.3)
            object = monitor.sample().asJSONObject
        }
        for key in ["hostname", "ram_total_mb", "ram_used_mb",
                    "ram_used_percent", "process_footprint_mb",
                    "storage_total_gb", "storage_free_gb",
                    "storage_used_percent", "cpu_percent"] {
            XCTAssertNotNil(object[key], "missing \(key)")
        }
    }
}

// MARK: - Transport race

final class TransportRaceTests: XCTestCase {

    /// Both legs run real sockets and finish with the plan's exact byte
    /// and trip accounting.
    func testRaceMeasuresBothLegsOverRealSockets() throws {
        let plan = NMPTransportRace.Plan(roundTrips: 6, payloadBytes: 24_000)
        let result = try NMPTransportRace.runSync(plan: plan, timeout: 15)

        // Mesh 2.5 grew the race to four legs; the original two are
        // still the anchors (leg 0 = NMP, leg 1 = plain TCP).
        XCTAssertGreaterThanOrEqual(result.legs.count, 2)
        for leg in result.legs {
            XCTAssertGreaterThan(leg.handshakeMs, 0, "\(leg.name) handshake")
            XCTAssertGreaterThan(leg.transferMs, 0, "\(leg.name) transfer")
            XCTAssertEqual(leg.roundTrips, 6)
            // 24000/6/2 = 2000 B each way per trip.
            XCTAssertEqual(leg.bytesMoved, 2000 * 2 * 6)
            XCTAssertEqual(leg.asJSONObject["measured"] as? Bool, true)
        }
        XCTAssertTrue(result.nmp.transportDescription.contains("AES-256-GCM"))
        XCTAssertTrue(result.legs[1].transportDescription.contains("no TLS"))
        XCTAssertTrue(result.note.contains("measured"))
    }

    /// A single big trip exercises the chunked NMP send path (payload
    /// larger than one 1024-byte chunk).
    func testChunkedTripReassemblesFullByteCount() throws {
        let plan = NMPTransportRace.Plan(roundTrips: 1, payloadBytes: 40_000)
        let result = try NMPTransportRace.runSync(plan: plan, timeout: 15)
        for leg in result.legs {
            XCTAssertEqual(leg.bytesMoved, 20_000 * 2, leg.name)
        }
    }
}

// MARK: - Token streaming

final class TokenStreamingTests: XCTestCase {

    /// onToken fires once per generated token, in order, with the same
    /// pieces the final text is rendered from.
    func testPromptServiceStreamsEveryTokenInOrder() throws {
        let testbed = try NMPMeshTestbed(
            layerCount: 8, hiddenSize: 64, remotePeerCount: 1)
        _ = try testbed.startSync()
        let service = NMPPromptInferenceService(
            orchestrator: testbed.orchestrator, hiddenSize: testbed.hiddenSize)

        var streamed: [(text: String, count: Int, requested: Int)] = []
        service.onToken = { token, count, requested in
            streamed.append((token.text, count, requested))
        }

        let done = expectation(description: "generation")
        var result: NMPPromptInferenceService.GenerationResult?
        service.run(prompt: "stream me", maxTokens: 6) { outcome in
            result = try? outcome.get()
            done.fulfill()
        }
        wait(for: [done], timeout: 30)

        let generation = try XCTUnwrap(result)
        XCTAssertEqual(streamed.count, generation.tokenCount)
        XCTAssertEqual(streamed.map(\.count), Array(1...streamed.count),
                       "counts must be strictly sequential")
        XCTAssertEqual(Set(streamed.map(\.requested)), [6])
        for piece in streamed.map(\.text) {
            XCTAssertTrue(generation.text.lowercased()
                .contains(piece.lowercased()),
                "streamed piece '\(piece)' missing from final text")
        }
    }
}

// MARK: - Compute shares

final class ComputeShareTests: XCTestCase {

    private func peer(_ id: UInt32) -> NMPCapabilities {
        NMPCapabilities(peerID: id, deviceName: "peer-\(id)",
                        ramMB: 8192, computeClass: .high)
    }

    func testHalfShareHalvesAPeersLayers() {
        let peers = [peer(1), peer(2)]
        // Equal measured speeds; peer 2 capped to 50%.
        let measured: [UInt32: Double] = [1: 0.001, 2: 0.001]

        let even = NMPModelSharder.plan(
            layerCount: 24, peers: peers, measuredSecondsPerLayer: measured)
        XCTAssertEqual(even.map(\.layerSpan), [12, 12])

        let capped = NMPModelSharder.plan(
            layerCount: 24, peers: peers, measuredSecondsPerLayer: measured,
            computeShares: [2: 0.5])
        XCTAssertEqual(capped.map(\.layerSpan), [16, 8],
                       "a half-share peer plans as half as fast: 2:1 split")
        XCTAssertEqual(capped.map(\.layerSpan).reduce(0, +), 24,
                       "shares never leave layers unassigned")
    }

    func testSharesScaleClassWeightsForUnmeasuredPeers() {
        let peers = [peer(1), peer(2)]
        let capped = NMPModelSharder.plan(
            layerCount: 24, peers: peers, computeShares: [2: 0.5])
        XCTAssertEqual(capped.map(\.layerSpan), [16, 8])
    }

    func testOrchestratorClampsAndClearsShares() throws {
        let testbed = try NMPMeshTestbed(
            layerCount: 8, hiddenSize: 32, remotePeerCount: 1)
        _ = try testbed.startSync()
        let orchestrator = testbed.orchestrator

        orchestrator.setComputeShare(0.001, forPeer: 7)
        XCTAssertEqual(orchestrator.computeShares[7], 0.05, "floor is 5%")
        orchestrator.setComputeShare(5.0, forPeer: 7)
        XCTAssertNil(orchestrator.computeShares[7],
                     "a full share is the default and is not stored")
    }

    /// The live re-shard: capping a peer's share and re-planning must
    /// visibly shrink its span — the Devices panel's proof of allocation.
    func testReplanAppliesShareToLiveMesh() throws {
        let testbed = try NMPMeshTestbed(
            layerCount: 24, hiddenSize: 32, remotePeerCount: 2)
        _ = try testbed.startSync()
        // Equal seeded speeds so shares are the only variable.
        let peerIDs = testbed.failover.activePeers.map(\.peerID)
        testbed.orchestrator.seedMeasurements(
            Dictionary(uniqueKeysWithValues: peerIDs.map { ($0, 0.001) }))

        let victim = try XCTUnwrap(peerIDs.last)
        testbed.orchestrator.setComputeShare(0.25, forPeer: victim)

        let done = expectation(description: "replan")
        var plan: [NMPShardPlanEntry] = []
        testbed.failover.replan { result in
            plan = (try? result.get()) ?? []
            done.fulfill()
        }
        wait(for: [done], timeout: 15)

        let victimSpan = try XCTUnwrap(
            plan.first { $0.peerID == victim }?.layerSpan)
        let otherSpans = plan.filter { $0.peerID != victim }.map(\.layerSpan)
        XCTAssertFalse(plan.isEmpty)
        XCTAssertLessThan(victimSpan, try XCTUnwrap(otherSpans.min()),
                          "the quarter-share peer must hold the smallest span")
        XCTAssertEqual(plan.map(\.layerSpan).reduce(0, +), 24)
    }
}

// MARK: - Web routes (Mesh 2.1)

final class Mesh21RouteTests: XCTestCase {

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
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let http = response as? HTTPURLResponse {
                outcome = (http.statusCode, data ?? Data())
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

    // MARK: Web-client tracking

    func testHTTPClientsAppearInHealthAndClientList() throws {
        XCTAssertEqual(server.webClientCount, 0)
        _ = try request("GET", "/health")

        let health = try json(try request("GET", "/health").data)
        let mesh = try XCTUnwrap(health["mesh"] as? [String: Any])
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(mesh["web_clients"] as? Int), 1,
            "the requester itself is a live web client")

        let (status, data) = try request("GET", "/api/clients")
        XCTAssertEqual(status, 200)
        let clients = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        XCTAssertGreaterThanOrEqual(clients.count, 1)
        XCTAssertNotNil(clients[0]["address"])
        XCTAssertNotNil(clients[0]["user_agent"])
        XCTAssertEqual(clients[0]["websocket"] as? Bool, false)
    }

    // MARK: Device metrics + allocation endpoints

    func testDeviceMetricsIs503UntilWiredThenServesHandlerPayload() throws {
        XCTAssertEqual(try request("GET", "/api/devices/metrics").status, 503)

        server.onDeviceMetricsRequest = { respond in
            respond(["host": ["hostname": "test-host"],
                     "generation_in_flight": false,
                     "peers": [] as [[String: Any]]])
        }
        let (status, data) = try request("GET", "/api/devices/metrics")
        XCTAssertEqual(status, 200)
        let object = try json(data)
        XCTAssertEqual((object["host"] as? [String: Any])?["hostname"] as? String,
                       "test-host")
    }

    func testAllocateValidatesPathAndBodyThenAppliesShare() throws {
        var received: (peerID: UInt32, share: Double)?
        server.onAllocationRequest = { peerID, share, respond in
            received = (peerID, share)
            respond(.success("peer-\(peerID): L0-11"))
        }

        // Bad peer id in the path.
        XCTAssertEqual(try request(
            "POST", "/api/devices/zzz/allocate",
            body: ["share": 0.5]).status, 400)
        // Share out of range.
        XCTAssertEqual(try request(
            "POST", "/api/devices/a1/allocate",
            body: ["share": 1.5]).status, 400)

        let (status, data) = try request(
            "POST", "/api/devices/a1/allocate", body: ["share": 0.5])
        XCTAssertEqual(status, 200)
        XCTAssertEqual(received?.peerID, 0xA1)
        XCTAssertEqual(received?.share, 0.5)
        let object = try json(data)
        XCTAssertEqual(object["summary"] as? String, "peer-161: L0-11")
    }

    // MARK: Comparison run

    func testComparisonRunIs503UntilWired() throws {
        XCTAssertEqual(try request(
            "POST", "/api/comparison/run",
            body: ["prompt": "x"]).status, 503)
    }

    func testComparisonRunServesMeasuredLegsAndSplicedProjection() throws {
        // A fabricated 1000 ms generation whose NMP transport cost 100 ms
        // vs TCP's 250 ms: the splice must land on exactly 1150 ms.
        let generation = NMPPromptInferenceService.GenerationResult(
            text: "measured output", tokenCount: 10, totalSeconds: 1.0,
            networkPayloadBytes: 12_000, shardCount: 1,
            perTokenSeconds: Array(repeating: 0.1, count: 10),
            engine: "test")
        let nmp = NMPTransportRace.LegResult(
            name: "NMP", transportDescription: "test",
            handshakeMs: 1, transferMs: 99, roundTrips: 10, bytesMoved: 12_000)
        let tcp = NMPTransportRace.LegResult(
            name: "TCP", transportDescription: "test",
            handshakeMs: 5, transferMs: 245, roundTrips: 10, bytesMoved: 12_000)

        server.onComparisonRunRequest = { request, respond in
            XCTAssertEqual(request.prompt, "race me")
            respond(.success(.init(
                generation: generation,
                race: NMPTransportRace.RaceResult(legs: [nmp, tcp]))))
        }

        let (status, data) = try request(
            "POST", "/api/comparison/run",
            body: ["prompt": "race me", "max_tokens": 10])
        XCTAssertEqual(status, 200)
        let object = try json(data)

        let race = try XCTUnwrap(object["race"] as? [String: Any])
        let legs = try XCTUnwrap(race["legs"] as? [[String: Any]])
        XCTAssertEqual(legs.count, 2)
        XCTAssertEqual(legs[0]["measured"] as? Bool, true)
        XCTAssertEqual(legs[1]["measured"] as? Bool, true)

        let projected = try XCTUnwrap(object["projected"] as? [[String: Any]])
        XCTAssertEqual(projected[0]["total_ms"] as? Double, 1000)
        XCTAssertEqual(projected[1]["total_ms"] as? Double, 1150,
                       "splice = 1000 - (1+99) + (5+245)")
        XCTAssertEqual(projected[1]["tokens_per_sec"] as? Double,
                       (10.0 / 1.15 * 100).rounded() / 100)
    }

    // MARK: Streaming broadcasts reach WebSocket clients

    func testGenerationEventsStreamToWebSocketClients() throws {
        let task = URLSession.shared.webSocketTask(
            with: URL(string: "ws://127.0.0.1:\(server.boundPort)/ws")!)
        task.resume()
        defer { task.cancel(with: .goingAway, reason: nil) }

        // Ping completes only after a correct RFC 6455 upgrade.
        let connected = expectation(description: "ws connected")
        task.sendPing { error in
            XCTAssertNil(error)
            connected.fulfill()
        }
        wait(for: [connected], timeout: 5)

        var received: [[String: Any]] = []
        let sawComplete = expectation(description: "generation events")
        func receiveNext() {
            task.receive { result in
                guard case .success(let message) = result,
                      case .string(let text) = message,
                      let object = try? JSONSerialization.jsonObject(
                        with: Data(text.utf8)) as? [String: Any] else { return }
                received.append(object)
                if object["type"] as? String == "generation_complete" {
                    sawComplete.fulfill()
                } else {
                    receiveNext()
                }
            }
        }
        receiveNext()

        server.reportGenerationStarted(prompt: "hi", maxTokens: 2,
                                       speculative: false)
        server.reportGenerationToken(text: "hello", index: 42,
                                     count: 1, requested: 2)
        server.reportGenerationComplete(NMPPromptInferenceService.GenerationResult(
            text: "hello world", tokenCount: 2, totalSeconds: 0.5,
            networkPayloadBytes: 100, shardCount: 1,
            perTokenSeconds: [0.25, 0.25], engine: "test"))
        wait(for: [sawComplete], timeout: 10)

        let types = received.compactMap { $0["type"] as? String }
        XCTAssertTrue(types.contains("generation_started"))
        XCTAssertTrue(types.contains("generation_token"))
        XCTAssertTrue(types.contains("generation_complete"))
        let token = try XCTUnwrap(
            received.first { $0["type"] as? String == "generation_token" })
        XCTAssertEqual(token["text"] as? String, "hello")
        XCTAssertEqual(token["count"] as? Int, 1)
        XCTAssertEqual(token["requested"] as? Int, 2)
        let complete = try XCTUnwrap(
            received.first { $0["type"] as? String == "generation_complete" })
        XCTAssertEqual(complete["output"] as? String, "hello world")
        XCTAssertEqual(complete["tokens_per_sec"] as? Double, 4)
    }
}
