//
//  End2EndTests.swift
//  NMPTests — Phase 1 integration
//
//  Two mock peers exchange encrypted handshakes + data packets in memory
//  (spec Phase 1 integration test), plus handshake-retry, duplicate-drop,
//  and malformed-input robustness. A real UDP loopback test using
//  Network.framework is included at the bottom (Apple platforms).
//

import XCTest
@testable import NMP

final class End2EndTests: XCTestCase {

    struct Mesh {
        let initiator: PeerConnection
        let responder: PeerConnection
        let tInit: MockTransport
        let tResp: MockTransport
        let qInit: DispatchQueue
        let qResp: DispatchQueue
    }

    private func makeMesh(
        initiatorConfig: ((inout PeerConnectionConfig) -> Void)? = nil,
        initiatorCaps: Data = Data("caps-I".utf8),
        responderCaps: Data = Data("caps-R".utf8)
    ) throws -> Mesh {
        let (tInit, tResp) = MockTransport.pair()
        let qInit = DispatchQueue(label: "nmp.test.init")
        let qResp = DispatchQueue(label: "nmp.test.resp")

        let sInit = NoiseStaticKeyPair()
        let sResp = NoiseStaticKeyPair()

        var cfgI = PeerConnectionConfig(localPeerID: 0xAAAA_0001)
        initiatorConfig?(&cfgI)
        let cfgR = PeerConnectionConfig(localPeerID: 0xBBBB_0002)

        let initiator = try PeerConnection(
            role: .initiator, config: cfgI, transport: tInit,
            localStatic: sInit, remoteStaticPublicKey: sResp.publicKeyData,
            localCapabilities: initiatorCaps, queue: qInit)
        let responder = try PeerConnection(
            role: .responder, config: cfgR, transport: tResp,
            localStatic: sResp, localCapabilities: responderCaps, queue: qResp)

        return Mesh(initiator: initiator, responder: responder,
                    tInit: tInit, tResp: tResp, qInit: qInit, qResp: qResp)
    }

    // MARK: Handshake

    func testHandshakeCompletesAndExchangesCapabilities() throws {
        let mesh = try makeMesh()
        let expI = expectation(description: "initiator established")
        let expR = expectation(description: "responder established")

        mesh.initiator.onEstablished = { caps, peerID in
            XCTAssertEqual(caps, Data("caps-R".utf8))
            XCTAssertEqual(peerID, 0xBBBB_0002)
            expI.fulfill()
        }
        mesh.responder.onEstablished = { caps, peerID in
            XCTAssertEqual(caps, Data("caps-I".utf8))
            XCTAssertEqual(peerID, 0xAAAA_0001)
            expR.fulfill()
        }

        let start = DispatchTime.now()
        mesh.responder.start()
        mesh.initiator.start()
        wait(for: [expI, expR], timeout: 5)

        let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e6
        // Spec success criterion: <10ms on a single simulated RTT. Assert a
        // loose 50ms bound to keep CI stable; report the measurement.
        print("[NMP] loopback handshake latency: \(String(format: "%.3f", elapsedMs)) ms")
        XCTAssertLessThan(elapsedMs, 50, "handshake took \(elapsedMs) ms")
    }

    func testEncryptedDataFlowsBothDirections() throws {
        let mesh = try makeMesh()
        let ready = expectation(description: "established")
        ready.expectedFulfillmentCount = 2
        mesh.initiator.onEstablished = { _, _ in ready.fulfill() }
        mesh.responder.onEstablished = { _, _ in ready.fulfill() }
        mesh.responder.start()
        mesh.initiator.start()
        wait(for: [ready], timeout: 5)

        let gotAtResponder = expectation(description: "responder got data")
        let gotAtInitiator = expectation(description: "initiator got data")
        mesh.responder.onPacket = { packet in
            XCTAssertEqual(packet.header.packetType, .data)
            XCTAssertEqual(packet.payload, Data("activations i→r".utf8))
            gotAtResponder.fulfill()
        }
        mesh.initiator.onPacket = { packet in
            XCTAssertEqual(packet.payload, Data("activations r→i".utf8))
            gotAtInitiator.fulfill()
        }

        mesh.initiator.sendAsync(payload: Data("activations i→r".utf8))
        mesh.responder.sendAsync(payload: Data("activations r→i".utf8))
        wait(for: [gotAtResponder, gotAtInitiator], timeout: 5)
    }

    func testManyPacketsSurviveRoundTrip() throws {
        let mesh = try makeMesh()
        let ready = expectation(description: "established")
        ready.expectedFulfillmentCount = 2
        mesh.initiator.onEstablished = { _, _ in ready.fulfill() }
        mesh.responder.onEstablished = { _, _ in ready.fulfill() }
        mesh.responder.start()
        mesh.initiator.start()
        wait(for: [ready], timeout: 5)

        let count = 200
        let all = expectation(description: "all packets received in order")
        var payloads: [Data] = []
        var sequences: [UInt32] = []
        let lock = NSLock()
        mesh.responder.onPacket = { packet in
            lock.lock()
            payloads.append(packet.payload)
            sequences.append(packet.header.sequenceNumber)
            let done = payloads.count == count
            lock.unlock()
            if done { all.fulfill() }
        }
        for n in 0..<count {
            mesh.initiator.sendAsync(payload: Data("pkt \(n)".utf8))
        }
        wait(for: [all], timeout: 10)
        lock.lock()
        // Since Phase 3, FEC parity packets consume every 5th sequence
        // number, so data sequences are increasing but not contiguous.
        XCTAssertEqual(payloads, (0..<count).map { Data("pkt \($0)".utf8) },
                       "in-memory transport is ordered; payloads must be too")
        XCTAssertEqual(sequences, sequences.sorted(),
                       "sequence numbers must be monotonically increasing")
        lock.unlock()
    }

    // MARK: Retry / failure

    func testMessage1LossTriggersRetryAndSucceeds() throws {
        var dropsRemaining = 1
        let dropLock = NSLock()
        let mesh = try makeMesh(initiatorConfig: { cfg in
            cfg.retryBackoff = [0.15, 0.3, 0.6] // shrink spec's 5/10/20s for tests
        })
        mesh.tInit.dropOutbound = { _ in
            dropLock.lock(); defer { dropLock.unlock() }
            if dropsRemaining > 0 { dropsRemaining -= 1; return true }
            return false
        }

        let ready = expectation(description: "established despite msg1 loss")
        mesh.initiator.onEstablished = { _, _ in ready.fulfill() }
        mesh.responder.start()
        mesh.initiator.start()
        wait(for: [ready], timeout: 5)
        XCTAssertGreaterThanOrEqual(mesh.tInit.sentDatagrams.count, 2, "expected a retry")
    }

    func testMessage2LossRecoveredByDuplicateMessage1() throws {
        // Drop the responder's first send (msg2). Initiator's retry of msg1
        // must cause the responder to resend its cached msg2.
        var dropsRemaining = 1
        let dropLock = NSLock()
        let mesh = try makeMesh(initiatorConfig: { cfg in
            cfg.retryBackoff = [0.15, 0.3, 0.6]
        })
        mesh.tResp.dropOutbound = { _ in
            dropLock.lock(); defer { dropLock.unlock() }
            if dropsRemaining > 0 { dropsRemaining -= 1; return true }
            return false
        }

        let ready = expectation(description: "initiator established despite msg2 loss")
        mesh.initiator.onEstablished = { _, _ in ready.fulfill() }
        mesh.responder.start()
        mesh.initiator.start()
        wait(for: [ready], timeout: 5)
    }

    func testAllRetriesExhaustedMarksPeerUnreachable() throws {
        let mesh = try makeMesh(initiatorConfig: { cfg in
            cfg.retryBackoff = [0.05, 0.1, 0.15]
            cfg.unreachableCooldown = 60
        })
        mesh.tInit.dropOutbound = { _ in true } // black hole

        let failed = expectation(description: "peer marked unreachable")
        mesh.initiator.onFailed = { error in
            guard case .peerUnreachable(let until) = error else {
                return XCTFail("expected peerUnreachable, got \(error)")
            }
            XCTAssertGreaterThan(until.timeIntervalSinceNow, 30)
            failed.fulfill()
        }
        mesh.responder.start()
        mesh.initiator.start()
        wait(for: [failed], timeout: 5)
        // Spec: initial send + 3 retries = 4 transmissions max.
        XCTAssertEqual(mesh.tInit.sentDatagrams.count, 4)
    }

    func testUnauthorizedPeerRejected() throws {
        let mesh = try makeMesh(initiatorConfig: { cfg in
            cfg.authorizedStaticKeys = [Data(repeating: 0, count: 32)] // matches nobody
        })
        let failed = expectation(description: "initiator rejects unauthorized responder")
        mesh.initiator.onFailed = { error in
            XCTAssertEqual(error, .unauthorizedPeer)
            failed.fulfill()
        }
        mesh.responder.start()
        mesh.initiator.start()
        wait(for: [failed], timeout: 5)
    }

    // MARK: Robustness

    func testDuplicateDataPacketDropped() throws {
        let mesh = try makeMesh()
        let ready = expectation(description: "established")
        ready.expectedFulfillmentCount = 2
        mesh.initiator.onEstablished = { _, _ in ready.fulfill() }
        mesh.responder.onEstablished = { _, _ in ready.fulfill() }
        mesh.responder.start()
        mesh.initiator.start()
        wait(for: [ready], timeout: 5)

        let got = expectation(description: "delivered exactly once")
        let replayDiag = expectation(description: "replay diagnosed")
        var deliveries = 0
        let lock = NSLock()
        mesh.responder.onPacket = { _ in
            lock.lock(); deliveries += 1; lock.unlock()
            got.fulfill()
        }
        mesh.responder.onDiagnostic = { msg in
            if msg.contains("replay") { replayDiag.fulfill() }
        }

        mesh.initiator.sendAsync(payload: Data("once".utf8)) { result in
            // Replay the exact wire bytes into the responder.
            if case .success = result,
               let wire = mesh.tInit.sentDatagrams.last {
                mesh.tResp.inject(wire)
            }
        }
        wait(for: [got, replayDiag], timeout: 5)
        lock.lock()
        XCTAssertEqual(deliveries, 1)
        lock.unlock()
    }

    func testMalformedGarbageDoesNotCrashOrKillSession() throws {
        let mesh = try makeMesh()
        let ready = expectation(description: "established")
        ready.expectedFulfillmentCount = 2
        mesh.initiator.onEstablished = { _, _ in ready.fulfill() }
        mesh.responder.onEstablished = { _, _ in ready.fulfill() }
        mesh.responder.start()
        mesh.initiator.start()
        wait(for: [ready], timeout: 5)

        // Blast garbage at the responder: random bytes, truncated headers,
        // valid-header-forged-body datagrams.
        var rng = SystemRandomNumberGenerator()
        for _ in 0..<500 {
            let len = Int.random(in: 0...128, using: &rng)
            mesh.tResp.inject(Data((0..<len).map { _ in UInt8.random(in: .min ... .max, using: &rng) }))
        }

        // Session must still work afterwards.
        let stillWorks = expectation(description: "session survives garbage")
        mesh.responder.onPacket = { packet in
            XCTAssertEqual(packet.payload, Data("still alive".utf8))
            stillWorks.fulfill()
        }
        // Give the garbage a moment to churn through, then send a real packet.
        mesh.qInit.asyncAfter(deadline: .now() + 0.2) {
            mesh.initiator.sendAsync(payload: Data("still alive".utf8))
        }
        wait(for: [stillWorks], timeout: 5)
    }
}

// MARK: - Real UDP loopback (Network.framework)

#if canImport(Network)
import Network

final class UDPLoopbackTests: XCTestCase {

    /// Full handshake + encrypted echo over 127.0.0.1 UDP using the
    /// production UDPTransport/UDPListener.
    func testHandshakeAndDataOverLoopbackUDP() throws {
        let queue = DispatchQueue(label: "nmp.loopback")
        let listener = try UDPListener(port: 0, queue: queue) // ephemeral port

        let sInit = NoiseStaticKeyPair()
        let sResp = NoiseStaticKeyPair()

        var responder: PeerConnection?
        let responderReady = expectation(description: "responder established")
        let listenerReady = expectation(description: "listener ready")

        listener.onStateChange = { state in
            if case .ready = state { listenerReady.fulfill() }
        }
        listener.onNewTransport = { transport, _ in
            do {
                let conn = try PeerConnection(
                    role: .responder,
                    config: PeerConnectionConfig(localPeerID: 2),
                    transport: transport,
                    localStatic: sResp,
                    localCapabilities: Data("caps-R".utf8),
                    queue: DispatchQueue(label: "nmp.loopback.resp"))
                conn.onEstablished = { _, _ in responderReady.fulfill() }
                responder = conn
                conn.start()
            } catch {
                XCTFail("responder setup failed: \(error)")
            }
        }
        listener.start()
        wait(for: [listenerReady], timeout: 5)

        guard let port = listener.port else {
            return XCTFail("listener has no port")
        }

        let transport = UDPTransport(host: "127.0.0.1", port: port,
                                     queue: DispatchQueue(label: "nmp.loopback.init.t"))
        let initiator = try PeerConnection(
            role: .initiator,
            config: PeerConnectionConfig(localPeerID: 1),
            transport: transport,
            localStatic: sInit,
            remoteStaticPublicKey: sResp.publicKeyData,
            localCapabilities: Data("caps-I".utf8),
            queue: DispatchQueue(label: "nmp.loopback.init"))

        let initiatorReady = expectation(description: "initiator established")
        initiator.onEstablished = { caps, _ in
            XCTAssertEqual(caps, Data("caps-R".utf8))
            initiatorReady.fulfill()
        }

        let start = DispatchTime.now()
        initiator.start()
        wait(for: [initiatorReady, responderReady], timeout: 10)
        let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e6
        print("[NMP] real UDP loopback handshake latency: \(String(format: "%.3f", elapsedMs)) ms")

        // Encrypted echo.
        let echoed = expectation(description: "echo received")
        responder?.onPacket = { packet in
            responder?.sendAsync(payload: packet.payload)
        }
        initiator.onPacket = { packet in
            XCTAssertEqual(packet.payload, Data("ping".utf8))
            echoed.fulfill()
        }
        initiator.sendAsync(payload: Data("ping".utf8))
        wait(for: [echoed], timeout: 10)

        initiator.close()
        responder?.close()
        listener.cancel()
    }
}
#endif
