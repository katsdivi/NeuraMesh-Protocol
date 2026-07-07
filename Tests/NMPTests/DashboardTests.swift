//
//  DashboardTests.swift
//  NMPTests — Phase 6
//
//  The dashboard server end-to-end over real loopback TCP: HTTP page
//  serving, RFC 6455 WebSocket handshake + framing (verified against the
//  RFC's own example key), live broadcasts, and inbound control messages.
//  Clients are plain URLSession / URLSessionWebSocketTask — the same
//  stack a browser-adjacent client would use.
//
//  Servers bind port 0 (ephemeral) so parallel test runs never collide.
//

import XCTest
@testable import NMP

final class DashboardTests: XCTestCase {

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

    private var baseURL: URL {
        URL(string: "http://127.0.0.1:\(server.boundPort)")!
    }

    private func openWebSocket() -> URLSessionWebSocketTask {
        let task = URLSession.shared.webSocketTask(
            with: URL(string: "ws://127.0.0.1:\(server.boundPort)/ws")!)
        task.resume()
        return task
    }

    /// Receives messages until `predicate` matches one (guards against
    /// interleaved broadcasts), fulfilling `expectation`.
    private func receiveUntil(_ task: URLSessionWebSocketTask,
                              _ expectation: XCTestExpectation,
                              predicate: @escaping ([String: Any]) -> Bool) {
        task.receive { [self] result in
            guard case .success(.string(let text)) = result,
                  let object = try? JSONSerialization.jsonObject(
                    with: Data(text.utf8)) as? [String: Any] else { return }
            if predicate(object) {
                expectation.fulfill()
            } else {
                receiveUntil(task, expectation, predicate: predicate)
            }
        }
    }

    /// The WebSocket connect handshake completes only when the server's
    /// Sec-WebSocket-Accept is correct, so a successful round trip also
    /// verifies the upgrade path.
    private func awaitConnected(_ task: URLSessionWebSocketTask,
                                timeout: TimeInterval = 5) {
        let ponged = expectation(description: "pong")
        task.sendPing { error in
            XCTAssertNil(error, "WebSocket ping failed: \(String(describing: error))")
            ponged.fulfill()
        }
        wait(for: [ponged], timeout: timeout)
    }

    // MARK: RFC 6455 conformance

    func testAcceptKeyMatchesRFCExample() {
        // RFC 6455 §1.3's worked example.
        XCTAssertEqual(
            NMPWebSocketCodec.acceptKey(forClientKey: "dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")
    }

    func testFrameCodecRoundTripAndMaskedDecode() {
        // Server frame round trip.
        let payload = Data("hello mesh".utf8)
        var buffer = NMPWebSocketCodec.encodeFrame(opcode: .text, payload: payload)
        let decoded = NMPWebSocketCodec.decodeFrame(from: &buffer)
        XCTAssertEqual(decoded, NMPWebSocketFrame(opcode: .text, payload: payload))
        XCTAssertTrue(buffer.isEmpty, "frame fully consumed")

        // Client-style masked frame (RFC 6455 §5.3).
        let mask: [UInt8] = [0x11, 0x22, 0x33, 0x44]
        var masked = Data([0x81, 0x80 | UInt8(payload.count)])
        masked.append(contentsOf: mask)
        for (index, byte) in payload.enumerated() {
            masked.append(byte ^ mask[index % 4])
        }
        let unmasked = NMPWebSocketCodec.decodeFrame(from: &masked)
        XCTAssertEqual(unmasked?.payload, payload)

        // Partial frame: nothing consumed, nil returned.
        var partial = NMPWebSocketCodec.encodeFrame(opcode: .text, payload: payload)
        partial.removeLast(4)
        let before = partial.count
        XCTAssertNil(NMPWebSocketCodec.decodeFrame(from: &partial))
        XCTAssertEqual(partial.count, before)

        // Extended 16-bit length.
        let big = Data(repeating: 0xAB, count: 600)
        var bigBuffer = NMPWebSocketCodec.encodeFrame(opcode: .binary, payload: big)
        XCTAssertEqual(NMPWebSocketCodec.decodeFrame(from: &bigBuffer)?.payload, big)
    }

    // MARK: HTTP

    func testServerServesDashboardHTML() throws {
        let done = expectation(description: "GET /")
        var body: String?
        var status: Int?
        URLSession.shared.dataTask(with: baseURL) { data, response, _ in
            status = (response as? HTTPURLResponse)?.statusCode
            body = data.flatMap { String(data: $0, encoding: .utf8) }
            done.fulfill()
        }.resume()
        wait(for: [done], timeout: 5)

        XCTAssertEqual(status, 200)
        let html = try XCTUnwrap(body)
        XCTAssertTrue(html.contains("NeuraMesh Dashboard"))
        XCTAssertTrue(html.contains("/ws"), "page must connect back to the WS endpoint")
    }

    func testUnknownPathReturns404() {
        let done = expectation(description: "GET /nope")
        var status: Int?
        URLSession.shared.dataTask(
            with: baseURL.appendingPathComponent("nope")) { _, response, _ in
            status = (response as? HTTPURLResponse)?.statusCode
            done.fulfill()
        }.resume()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(status, 404)
    }

    // MARK: WebSocket live updates

    func testWebSocketConnectsAndReceivesPeerUpdate() {
        let task = openWebSocket()
        defer { task.cancel(with: .normalClosure, reason: nil) }
        awaitConnected(task)

        let got = expectation(description: "peer_update received")
        receiveUntil(task, got) { object in
            object["type"] as? String == "peer_update"
                && object["peerID"] as? String == "2"
                && object["latencyMS"] as? Int == 12
                && object["alive"] as? Bool == true
        }
        server.updatePeerState(peerID: 0x2, name: "testbed-2", latencyMS: 12,
                               loadPercent: 34, assigned: "layers 4-7", alive: true)
        wait(for: [got], timeout: 5)
    }

    func testPacketEventsAppearInLogStream() {
        let task = openWebSocket()
        defer { task.cancel(with: .normalClosure, reason: nil) }
        awaitConnected(task)

        let gotFEC = expectation(description: "fec event")
        receiveUntil(task, gotFEC) { object in
            object["type"] as? String == "packet_event"
                && object["event"] as? String == "fec_recovered"
                && object["seq"] as? [Int] == [42]
        }
        server.reportPacketEvent(.fecRecovered(sequence: 42), peerID: 0x3)
        wait(for: [gotFEC], timeout: 5)

        let gotNACK = expectation(description: "nack event")
        receiveUntil(task, gotNACK) { object in
            object["type"] as? String == "packet_event"
                && object["event"] as? String == "nack_sent"
                && object["seq"] as? [Int] == [7, 9]
        }
        server.reportPacketEvent(.nackSent(sequences: [7, 9]), peerID: 0x3)
        wait(for: [gotNACK], timeout: 5)
    }

    func testInferenceProgressBroadcast() {
        let task = openWebSocket()
        defer { task.cancel(with: .normalClosure, reason: nil) }
        awaitConnected(task)

        let got = expectation(description: "progress")
        receiveUntil(task, got) { object in
            object["type"] as? String == "inference_progress"
                && object["progress"] as? Int == 50
                && object["stage"] as? String == "shard 1/2"
        }
        server.updateInferenceProgress(progress: 0.5, stage: "shard 1/2")
        wait(for: [got], timeout: 5)
    }

    // MARK: Inbound control

    func testControlMessagesReachTheMesh() {
        let task = openWebSocket()
        defer { task.cancel(with: .normalClosure, reason: nil) }
        awaitConnected(task)

        var received: [NMPDashboardServer.ControlMessage] = []
        let lossSet = expectation(description: "set_loss_rate")
        let dropInjected = expectation(description: "inject_peer_drop")
        server.onControl = { control in
            received.append(control)
            switch control {
            case .setLossRate: lossSet.fulfill()
            case .injectPeerDrop: dropInjected.fulfill()
            default: break
            }
        }

        task.send(.string(#"{"type":"set_loss_rate","rate":0.05}"#)) { _ in }
        task.send(.string(#"{"type":"inject_peer_drop"}"#)) { _ in }
        wait(for: [lossSet, dropInjected], timeout: 5)

        XCTAssertTrue(received.contains(.setLossRate(0.05)))
        XCTAssertTrue(received.contains(.injectPeerDrop))
    }

    func testLossInjectionViaControlPathActuallyDropsPackets() throws {
        // Full loop: WS control message → onControl → injector →
        // datagrams dropped, the way the dashboard slider drives a mesh.
        let (raw, _) = NMPInMemoryTransport.pair()
        let injector = NMPPacketLossInjector(wrapping: raw)

        let task = openWebSocket()
        defer { task.cancel(with: .normalClosure, reason: nil) }
        awaitConnected(task)

        let applied = expectation(description: "loss applied")
        server.onControl = { control in
            if case .setLossRate(let rate) = control {
                injector.setLossRate(rate)
                applied.fulfill()
            }
        }
        task.send(.string(#"{"type":"set_loss_rate","rate":1.0}"#)) { _ in }
        wait(for: [applied], timeout: 5)

        for _ in 0..<20 { injector.send(Data([0x00])) }
        XCTAssertEqual(injector.droppedCount, 20,
                       "rate 1.0 must black-hole every datagram")
    }

    // MARK: Robustness

    func testRapidBroadcastsDoNotCrashOrWedge() {
        let task = openWebSocket()
        defer { task.cancel(with: .normalClosure, reason: nil) }
        awaitConnected(task)

        // Hammer every broadcast type from multiple threads.
        DispatchQueue.concurrentPerform(iterations: 200) { index in
            server.updatePeerState(peerID: UInt32(index % 4), name: "p",
                                   latencyMS: index, loadPercent: index % 100,
                                   assigned: "layers 0-3", alive: index % 2 == 0)
            server.updateInferenceProgress(progress: Double(index) / 200,
                                           stage: "stage \(index)")
            server.reportPacketEvent(.fecRecovered(sequence: UInt32(index)),
                                     peerID: 0x2)
        }

        // Server still responsive afterwards.
        let got = expectation(description: "still alive")
        receiveUntil(task, got) { object in
            object["type"] as? String == "mesh_event"
                && object["message"] as? String == "survived"
        }
        server.reportMeshEvent("survived")
        wait(for: [got], timeout: 10)
    }
}
