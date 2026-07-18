//
//  ApiContractTests.swift
//  NMPTests — Fable findings regression suite (BUG-2/5/6/10/12/13/14/15/16/17/20)
//
//  Pins the API contracts fixed after the 2026-07-17 test run: honest
//  status classes (429 busy / 409 nothing-to-do / 404 unknown device),
//  the /health readiness flag, explicit clamping echoes, chat-save
//  hygiene, generation-event source labels, and metadata-based slice
//  detection. Servers bind port 0 (ephemeral) so parallel runs never
//  collide; no new listeners outside that.
//

import XCTest
@testable import NMP

final class ApiContractTests: XCTestCase {

    private var server: NMPDashboardServer!

    override func setUpWithError() throws {
        server = NMPDashboardServer()
        try server.start(port: 0)
        XCTAssertGreaterThan(server.boundPort, 0)
    }

    override func tearDown() {
        server.stop()
        server = nil
    }

    // MARK: Plumbing (same style as WebUITests)

    private func request(
        _ method: String, _ path: String, body: [String: Any]? = nil
    ) throws -> (status: Int, object: [String: Any]) {
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(server.boundPort)" + path)!)
        request.httpMethod = method
        if let body {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        let done = expectation(description: "\(method) \(path)")
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

    private static func generation(
        tokens: Int = 4, seconds: TimeInterval = 0.4
    ) -> NMPPromptInferenceService.GenerationResult {
        .init(text: "ok", tokenCount: tokens, totalSeconds: seconds,
              networkPayloadBytes: 1000, shardCount: 1,
              perTokenSeconds: Array(repeating: seconds / Double(max(1, tokens)),
                                     count: tokens),
              engine: "test")
    }

    // MARK: BUG-12 — /health readiness vs liveness

    func testHealthReportsReadinessSeparateFromLiveness() throws {
        // Fresh server: HTTP is up (status ok) but nothing can generate yet.
        let before = try request("GET", "/health")
        XCTAssertEqual(before.status, 200)
        XCTAssertEqual(before.object["status"] as? String, "ok")
        XCTAssertEqual(before.object["ready"] as? Bool, false,
                       "a mesh that cannot generate must not read as ready")

        server.setReady(true)
        let after = try request("GET", "/health")
        XCTAssertEqual(after.object["ready"] as? Bool, true)
        // BUG-18: the counting rule is stated next to the counts.
        let mesh = try XCTUnwrap(after.object["mesh"] as? [String: Any])
        XCTAssertNotNil(mesh["peers_note"])
    }

    // MARK: BUG-15 — max_tokens: reject nonsense, echo real clamps

    func testInferenceRejectsNonPositiveMaxTokens() throws {
        server.onInferenceRequest = { _, _ in XCTFail("must not reach the mesh") }
        for bad in [0, -5] {
            let (status, object) = try request(
                "POST", "/api/inference", body: ["prompt": "hi", "max_tokens": bad])
            XCTAssertEqual(status, 400, "max_tokens \(bad) must be a 400")
            XCTAssertNotNil(object["error"])
        }
    }

    func testInferenceClampsOverCapAndSaysSo() throws {
        var seen: Int?
        server.onInferenceRequest = { request, respond in
            seen = request.maxTokens
            respond(.success(Self.generation()))
        }
        let (status, object) = try request(
            "POST", "/api/inference", body: ["prompt": "hi", "max_tokens": 100_000])
        XCTAssertEqual(status, 200)
        XCTAssertEqual(seen, NMPPromptInferenceService.maxTokensPerRequest)
        XCTAssertEqual(object["max_tokens_effective"] as? Int,
                       NMPPromptInferenceService.maxTokensPerRequest,
                       "a clamp must be visible in the response")
    }

    func testInferenceWithinCapCarriesNoClampMarker() throws {
        server.onInferenceRequest = { _, respond in
            respond(.success(Self.generation()))
        }
        let (_, object) = try request(
            "POST", "/api/inference", body: ["prompt": "hi", "max_tokens": 8])
        XCTAssertNil(object["max_tokens_effective"],
                     "no clamp happened, so no marker")
    }

    func testChatRejectsNonPositiveMaxTokens() throws {
        server.onInferenceRequest = { _, _ in XCTFail("must not reach the mesh") }
        let (status, _) = try request("POST", "/api/chat", body: [
            "messages": [["role": "user", "content": "hi"]],
            "max_tokens": 0,
        ])
        XCTAssertEqual(status, 400)
    }

    // MARK: BUG-15 — benchmark run-count echo

    func testBenchmarkEchoesRequestedAndCompletedRuns() throws {
        server.onBenchmarkRequest = { request, respond in
            XCTAssertEqual(request.runs, 10, "runs clamp to 1...10")
            respond(.success([Self.generation(), Self.generation()]))
        }
        let (status, object) = try request(
            "POST", "/api/benchmark/run", body: ["prompt": "hi", "runs": 100])
        XCTAssertEqual(status, 200)
        XCTAssertEqual(object["runs_requested"] as? Int, 100)
        XCTAssertEqual(object["runs_completed"] as? Int, 2)
        XCTAssertEqual((object["runs"] as? [[String: Any]])?.count, 2)
    }

    // MARK: BUG-5 / BUG-6 — failure status classes propagate

    func testBenchmarkBusyContentionIs429() throws {
        server.onBenchmarkRequest = { _, respond in
            respond(.failure(.init(
                "an inference is already running — retry shortly", status: 429)))
        }
        let (status, object) = try request(
            "POST", "/api/benchmark/run", body: ["prompt": "hi"])
        XCTAssertEqual(status, 429)
        XCTAssertEqual(object["error"] as? String,
                       "an inference is already running — retry shortly")
    }

    func testComparisonRunStatusClassesPropagate() throws {
        // 409: honest nothing-to-race refusal (local placement).
        server.onComparisonRunRequest = { _, respond in
            respond(.failure(.init(
                "this run moved no mesh traffic (local placement?) — nothing to race",
                status: 409)))
        }
        let conflict = try request("POST", "/api/comparison/run", body: ["prompt": "x"])
        XCTAssertEqual(conflict.status, 409)
        XCTAssertNotNil(conflict.object["error"])

        // 429: busy contention.
        server.onComparisonRunRequest = { _, respond in
            respond(.failure(.init(
                "an inference is already running — retry shortly", status: 429)))
        }
        let busy = try request("POST", "/api/comparison/run", body: ["prompt": "x"])
        XCTAssertEqual(busy.status, 429)

        // Default stays a genuine 500.
        server.onComparisonRunRequest = { _, respond in
            respond(.failure(.init("transport race failed: boom")))
        }
        let fault = try request("POST", "/api/comparison/run", body: ["prompt": "x"])
        XCTAssertEqual(fault.status, 500)
    }

    // MARK: BUG-10 / BUG-19 — allocation: 404 unknowns, explicit echo

    func testAllocateUnknownDeviceIs404() throws {
        server.onAllocationRequest = { _, _, respond in
            respond(.failure(.init("unknown device", status: 404)))
        }
        let (status, object) = try request(
            "POST", "/api/devices/ffffffff/allocate", body: ["share": 0.5])
        XCTAssertEqual(status, 404)
        XCTAssertEqual(object["error"] as? String, "unknown device")
    }

    func testAllocateSuccessEchoesShareRequestedAndAssignment() throws {
        server.onAllocationRequest = { _, _, respond in
            respond(.success("manual — 0xa1: L14-23; plan: 0x1: L0-13, 0xa1: L14-23"))
        }
        let (status, object) = try request(
            "POST", "/api/devices/a1/allocate", body: ["share": 0.5])
        XCTAssertEqual(status, 200)
        XCTAssertEqual(object["share_requested"] as? Double, 0.5)
        XCTAssertEqual(object["share"] as? Double, 0.5, "compat key kept")
        XCTAssertTrue((object["summary"] as? String ?? "").contains("L14-23"),
                      "the resulting assignment rides the summary")
    }

    // MARK: BUG-2 (bonus) — model select: no-op re-select, 400 class

    func testModelSelectAlreadyActiveSkipsReconnect() throws {
        server.onModelSelectRequest = { _, reply in
            reply(.success(.init(summary: "already active", reconnecting: false)))
        }
        let (status, object) = try request(
            "POST", "/api/models/select", body: ["path": "~/models/active.gguf"])
        XCTAssertEqual(status, 200)
        XCTAssertEqual(object["summary"] as? String, "already active")
        XCTAssertEqual(object["reconnecting"] as? Bool, false)
    }

    func testModelSelectRejectionStatusPropagates() throws {
        server.onModelSelectRequest = { _, reply in
            reply(.failure(.init(
                "‘slice’ is a vault slice — not a complete model", status: 400)))
        }
        let (status, object) = try request(
            "POST", "/api/models/select", body: ["path": "slice.gguf"])
        XCTAssertEqual(status, 400)
        XCTAssertTrue((object["error"] as? String ?? "").contains("vault slice"))
    }

    // MARK: BUG-13 / BUG-14 — chat save hygiene

    private func makeChatStore() -> (NMPChatStore, URL) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("nmp-chatapi-\(UUID().uuidString)")
        return (NMPChatStore(directory: dir, deviceName: "test-device"), dir)
    }

    func testEmptyChatSaveNeedsAnExplicitTitle() throws {
        let (store, dir) = makeChatStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        server.chatStore = store

        // Bare empty save: input error, no silent "New chat" row (BUG-13).
        let rejected = try request("POST", "/api/chats", body: ["messages": []])
        XCTAssertEqual(rejected.status, 400)
        // Whitespace-only titles do not count as explicit.
        let blankTitle = try request(
            "POST", "/api/chats", body: ["title": "  ", "messages": []])
        XCTAssertEqual(blankTitle.status, 400)

        // The web UI's new-chat button names the row explicitly — allowed.
        let named = try request(
            "POST", "/api/chats", body: ["title": "New chat", "messages": []])
        XCTAssertEqual(named.status, 200)
        XCTAssertEqual(named.object["title"] as? String, "New chat")
        XCTAssertEqual(named.object["message_count"] as? Int, 0)
    }

    func testChatSaveStampsBaseModelDisplayName() throws {
        let (store, dir) = makeChatStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        server.chatStore = store
        var info = NMPDashboardServer.MeshInfo()
        info.engine = "llamaShard"
        // Worst case at save time: the engine tag carries a slice artifact.
        info.modelName = "qwen2.5-0.5b-instruct_sliced_0_12"
        info.modelDisplayName = "Qwen2.5 0.5B Instruct"
        server.meshInfo = info

        // No model in the body → the active model's display name.
        let defaulted = try request("POST", "/api/chats", body: [
            "messages": [["role": "user", "content": "hi"]],
        ])
        XCTAssertEqual(defaulted.object["model"] as? String, "Qwen2.5 0.5B Instruct")

        // A slug or sliced spelling of the SAME model normalizes too.
        for spelling in ["qwen2.5-0.5b-instruct",
                         "qwen2.5-0.5b-instruct_sliced_0_12",
                         "Qwen2.5 0.5B Instruct"] {
            let saved = try request("POST", "/api/chats", body: [
                "model": spelling,
                "messages": [["role": "user", "content": "hi"]],
            ])
            XCTAssertEqual(saved.object["model"] as? String, "Qwen2.5 0.5B Instruct",
                           "spelling '\(spelling)' must store the display name")
        }

        // An explicitly DIFFERENT model name is kept (slice-stripped).
        let other = try request("POST", "/api/chats", body: [
            "model": "Qwen2.5 1.5B Instruct",
            "messages": [["role": "user", "content": "hi"]],
        ])
        XCTAssertEqual(other.object["model"] as? String, "Qwen2.5 1.5B Instruct")
    }

    func testChatModelNameNormalizer() {
        var info = NMPDashboardServer.MeshInfo()
        info.modelName = "qwen2.5-0.5b-instruct"
        info.modelDisplayName = ""
        // No display name known: the (slice-stripped) tag is the best truth.
        XCTAssertEqual(NMPDashboardServer.chatModelName(
            requested: "qwen2.5-0.5b-instruct_sliced_0_12", meshInfo: info),
            "qwen2.5-0.5b-instruct")
        XCTAssertEqual(NMPDashboardServer.chatModelName(
            requested: "", meshInfo: info), "qwen2.5-0.5b-instruct")
    }

    // MARK: BUG-16 — generation events carry their source

    func testGenerationEventsCarrySourceLabels() throws {
        let task = URLSession.shared.webSocketTask(
            with: URL(string: "ws://127.0.0.1:\(server.boundPort)/ws")!)
        task.resume()
        defer { task.cancel(with: .normalClosure, reason: nil) }
        let connected = expectation(description: "ws connected")
        task.sendPing { error in
            XCTAssertNil(error)
            connected.fulfill()
        }
        wait(for: [connected], timeout: 5)

        var events: [[String: Any]] = []
        let sawComplete = expectation(description: "framed benchmark generation")
        func receiveNext() {
            task.receive { result in
                guard case .success(.string(let text)) = result,
                      let object = try? JSONSerialization.jsonObject(
                        with: Data(text.utf8)) as? [String: Any] else { return }
                events.append(object)
                if object["type"] as? String == "generation_complete" {
                    sawComplete.fulfill()
                } else {
                    receiveNext()
                }
            }
        }
        receiveNext()

        // A benchmark-path generation must be framed AND labeled.
        server.reportGenerationStarted(prompt: "hi", maxTokens: 2,
                                       speculative: false, source: "benchmark")
        server.reportGenerationToken(text: "x", index: 0, count: 1,
                                     requested: 2, source: "benchmark")
        server.reportGenerationComplete(Self.generation(), source: "benchmark")
        wait(for: [sawComplete], timeout: 10)

        let types = events.compactMap { $0["type"] as? String }
        XCTAssertTrue(types.contains("generation_started"))
        XCTAssertTrue(types.contains("generation_token"))
        XCTAssertTrue(types.contains("generation_complete"))
        for event in events where (event["type"] as? String)?
            .hasPrefix("generation") == true {
            XCTAssertEqual(event["source"] as? String, "benchmark",
                           "\(event["type"] ?? "?") must carry its source")
        }
    }

    func testPromptServiceOwnsActiveSourceOnlyWhenRunning() {
        // The source label is claimed by the generation that actually got
        // the slot — a busy-rejected request never relabels the stream.
        let testbed = try? NMPMeshTestbed(
            layerCount: 4, hiddenSize: 32, remotePeerCount: 0)
        guard let testbed, (try? testbed.startSync()) != nil else {
            return XCTFail("in-process testbed failed to assemble")
        }
        let service = NMPPromptInferenceService(
            orchestrator: testbed.orchestrator, hiddenSize: 32)
        let first = expectation(description: "first run")
        service.run(prompt: "hello mesh", maxTokens: 2, source: "benchmark") { _ in
            first.fulfill()
        }
        // Immediately contend from another surface: must reject busy
        // WITHOUT stealing the label (or must run after — either way the
        // label always matches the run that owns the slot).
        let second = expectation(description: "second run resolved")
        service.run(prompt: "contender", maxTokens: 1, source: "inference") { result in
            if case .failure(.busy) = result {
                // rejected — the benchmark run keeps its label
            }
            second.fulfill()
        }
        wait(for: [first, second], timeout: 20)
    }

    // MARK: BUG-20 — manual_mode mirrors allocation_supported

    func testMetricsManualModeMirrorsDeprecatedFlag() throws {
        server.onDeviceMetricsRequest = { respond in
            // Same shape the CLI ships (llamaShard path, auto mode ON).
            respond([
                "manual_mode": false,
                "allocation_supported": false,
                "allocation_supported_note":
                    "deprecated name — same value as manual_mode",
                "peers": [] as [[String: Any]],
            ])
        }
        let (status, object) = try request("GET", "/api/devices/metrics")
        XCTAssertEqual(status, 200)
        XCTAssertEqual(object["manual_mode"] as? Bool,
                       object["allocation_supported"] as? Bool)
        XCTAssertNotNil(object["allocation_supported_note"])
    }
}

// MARK: - BUG-2 — metadata slice detection (no server needed)

final class ModelSliceDetectionTests: XCTestCase {

    /// A synthetic qwen2 GGUF. `blocks` are the blk indices present;
    /// `blockCount` is what the metadata claims; `name` is general.name.
    private func makeGGUF(name: String, blockCount: UInt32,
                          blocks: [Int]) -> Data {
        var b = GGUFBuilder.header(
            tensorCount: UInt64(blocks.count + 1), kvCount: 4)
        b.kvString("general.architecture", "qwen2")
        b.kvString("general.name", name)
        b.kvU32("qwen2.block_count", blockCount)
        b.kvU32("qwen2.embedding_length", 64)
        // token_embd + one tensor per present block, 64 bytes apart.
        b.str("token_embd.weight")
        b.u32(1); b.u64(16); b.u32(0); b.u64(0)
        for (i, layer) in blocks.enumerated() {
            b.str("blk.\(layer).attn_q.weight")
            b.u32(1); b.u64(16); b.u32(0); b.u64(UInt64((i + 1) * 64))
        }
        // Pad a data section so every tensor has a nonzero size.
        let dataBytes = (blocks.count + 2) * 64
        b.data.append(Data(count: dataBytes))
        return b.data
    }

    func testCompleteModelPassesBothMetadataChecks() throws {
        let gguf = try NMPGGUFModel.parse(makeGGUF(
            name: "Tiny Qwen", blockCount: 3, blocks: [0, 1, 2]))
        XCTAssertTrue(NMPModelCatalog.isCompleteModel(gguf))
    }

    func testSwiftSlicerStyleSliceIsIncomplete() throws {
        // NMPGGUFSlicer keeps block_count at the FULL N but ships only its
        // range's global-named blocks.
        let gguf = try NMPGGUFModel.parse(makeGGUF(
            name: "Tiny Qwen", blockCount: 6, blocks: [2, 3]))
        XCTAssertFalse(NMPModelCatalog.isCompleteModel(gguf),
                       "block coverage below block_count = a slice")
    }

    func testPythonSlicerStyleSliceIsIncomplete() throws {
        // scripts/gguf_slice.py renumbers blocks and overrides block_count
        // (so coverage looks fine) but stamps the name.
        let gguf = try NMPGGUFModel.parse(makeGGUF(
            name: "qwen2.5-0.5b-instruct_sliced_0_12",
            blockCount: 2, blocks: [0, 1]))
        XCTAssertFalse(NMPModelCatalog.isCompleteModel(gguf),
                       "general.name slice stamp = a slice")
    }

    func testRealVaultSliceOnThisMachineIsFlagged() throws {
        // The exact file BUG-2 was reproduced with (self-contained python
        // slice: block_count 12, blocks renumbered 0-based, general.name
        // stamped "…_sliced_0_12"). Machine-specific, so skip elsewhere.
        let path = ("~/models/qwen2.5-0.5b-instruct-q4_k_m_part1.gguf"
                    as NSString).expandingTildeInPath
        try XCTSkipUnless(FileManager.default.fileExists(atPath: path),
                          "no local vault slice to check against")
        let slice = try XCTUnwrap(NMPModelCatalog.candidate(path: path))
        XCTAssertFalse(slice.isCompleteModel,
                       "the real _part1 slice must be flagged incomplete")
        XCTAssertFalse(NMPModelCatalog.scan(directory: "~/models")
            .contains { $0.path == path },
            "the catalog must omit the slice")
    }

    func testFilenameBackstopCatchesScrubbedSlices() {
        XCTAssertTrue(NMPModelCatalog.filenameSuggestsSlice(
            "/m/qwen2.5-0.5b-instruct-q4_k_m_part1.gguf"))
        XCTAssertTrue(NMPModelCatalog.filenameSuggestsSlice(
            "/m/foo_sliced_0_12.gguf"))
        XCTAssertFalse(NMPModelCatalog.filenameSuggestsSlice(
            "/m/qwen2.5-0.5b-instruct-q4_k_m.gguf"))
        XCTAssertFalse(NMPModelCatalog.filenameSuggestsSlice(
            "/m/llama-2-7b-chat.Q4_K_M.gguf"))
    }

    func testCatalogOmitsSlicesAndCandidateFlagsThem() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("nmp-models-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try makeGGUF(name: "Tiny Qwen", blockCount: 3, blocks: [0, 1, 2])
            .write(to: dir.appendingPathComponent("full.gguf"))
        try makeGGUF(name: "Tiny Qwen", blockCount: 6, blocks: [2, 3])
            .write(to: dir.appendingPathComponent("fragment.gguf"))

        // The scan (backs /api/models, the recommender, the adaptive
        // controller, boot auto-select) never surfaces the slice…
        let scanned = NMPModelCatalog.scan(directory: dir.path)
        XCTAssertEqual(scanned.map(\.name), ["Tiny Qwen"])
        XCTAssertTrue(scanned.allSatisfy(\.isCompleteModel))

        // …and select's criterion is the SAME flag on the direct candidate.
        let fragment = try XCTUnwrap(NMPModelCatalog.candidate(
            path: dir.appendingPathComponent("fragment.gguf").path))
        XCTAssertFalse(fragment.isCompleteModel)
        let full = try XCTUnwrap(NMPModelCatalog.candidate(
            path: dir.appendingPathComponent("full.gguf").path))
        XCTAssertTrue(full.isCompleteModel)
    }
}
