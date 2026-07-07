//
//  SymmetricCryptoTests.swift
//  NMPTests — Phase 1
//
//  Session-layer AES-256-GCM: seal/open round-trip, header-as-AAD tamper
//  detection, replay protection, nonce construction.
//

import XCTest
@testable import NMP

final class SymmetricCryptoTests: XCTestCase {

    /// Runs a real Noise IK handshake and wires two sessions A↔B.
    private func makeSessionPair(
        seedA: UInt64 = 0x1111_2222_3333_4444,
        seedB: UInt64 = 0xAAAA_BBBB_CCCC_DDDD
    ) throws -> (a: NMPSecureSession, b: NMPSecureSession) {
        let sA = NoiseStaticKeyPair()
        let sB = NoiseStaticKeyPair()
        let i = try NoiseIKHandshake(role: .initiator, localStatic: sA,
                                     remoteStaticPublicKey: sB.publicKeyData)
        let r = try NoiseIKHandshake(role: .responder, localStatic: sB)
        _ = try r.readMessage1(try i.writeMessage1(payload: Data()))
        _ = try i.readMessage2(try r.writeMessage2(payload: Data()))
        let resI = try i.finalize()
        let resR = try r.finalize()

        let a = NMPSecureSession(keys: NMPSessionKeys(
            handshake: resI, localNonceSeed: seedA, remoteNonceSeed: seedB))
        let b = NMPSecureSession(keys: NMPSessionKeys(
            handshake: resR, localNonceSeed: seedB, remoteNonceSeed: seedA))
        return (a, b)
    }

    // MARK: Round-trip

    func testSealOpenRoundTrip() throws {
        let (a, b) = try makeSessionPair()
        let payload = Data("layer-14 activations".utf8)
        let (wire, seq) = try a.seal(packetType: .data, flags: [.flush],
                                     senderPeerID: 7, payload: payload,
                                     timestampNanos: 42)
        XCTAssertEqual(seq, 0)
        let packet = try b.open(datagram: wire)
        XCTAssertEqual(packet.payload, payload)
        XCTAssertEqual(packet.header.packetType, .data)
        XCTAssertEqual(packet.header.flags, [.flush])
        XCTAssertEqual(packet.header.senderPeerID, 7)
        XCTAssertEqual(packet.header.sequenceNumber, 0)
        XCTAssertEqual(packet.header.timestampNanos, 42)
    }

    func testBidirectionalTraffic() throws {
        let (a, b) = try makeSessionPair()
        for n in 0..<50 {
            let pA = Data("a→b #\(n)".utf8)
            let (wireA, _) = try a.seal(packetType: .data, senderPeerID: 1,
                                        payload: pA, timestampNanos: 0)
            XCTAssertEqual(try b.open(datagram: wireA).payload, pA)

            let pB = Data("b→a #\(n)".utf8)
            let (wireB, _) = try b.seal(packetType: .data, senderPeerID: 2,
                                        payload: pB, timestampNanos: 0)
            XCTAssertEqual(try a.open(datagram: wireB).payload, pB)
        }
    }

    func testSequenceNumbersIncrement() throws {
        let (a, _) = try makeSessionPair()
        for expected in 0..<5 {
            let (_, seq) = try a.seal(packetType: .data, senderPeerID: 1,
                                      payload: Data([1]), timestampNanos: 0)
            XCTAssertEqual(seq, UInt32(expected))
        }
    }

    func testEmptyAndMaxishPayloads() throws {
        let (a, b) = try makeSessionPair()
        for size in [0, 1, 1500, 65_000] {
            let payload = Data(repeating: 0x5A, count: size)
            let (wire, _) = try a.seal(packetType: .data, senderPeerID: 1,
                                       payload: payload, timestampNanos: 0)
            XCTAssertEqual(try b.open(datagram: wire).payload, payload, "size=\(size)")
        }
    }

    // MARK: Tamper detection (AAD = header)

    func testTamperedHeaderRejected() throws {
        let (a, b) = try makeSessionPair()
        let (wire, _) = try a.seal(packetType: .data, senderPeerID: 1,
                                   payload: Data("p".utf8), timestampNanos: 0)
        // Flip a bit in the peer-ID field (header byte 8). Header still decodes,
        // but GCM's AAD check must fail.
        var forged = wire
        forged[8] ^= 0x01
        XCTAssertThrowsError(try b.open(datagram: forged)) {
            XCTAssertEqual($0 as? NMPCryptoError, .authenticationFailed)
        }
    }

    func testTamperedCiphertextRejected() throws {
        let (a, b) = try makeSessionPair()
        let (wire, _) = try a.seal(packetType: .data, senderPeerID: 1,
                                   payload: Data(repeating: 9, count: 32),
                                   timestampNanos: 0)
        var forged = wire
        forged[NMPHeader.byteCount + 3] ^= 0xFF
        XCTAssertThrowsError(try b.open(datagram: forged)) {
            XCTAssertEqual($0 as? NMPCryptoError, .authenticationFailed)
        }
    }

    func testTamperedTagRejected() throws {
        let (a, b) = try makeSessionPair()
        let (wire, _) = try a.seal(packetType: .data, senderPeerID: 1,
                                   payload: Data("p".utf8), timestampNanos: 0)
        var forged = wire
        forged[forged.count - 1] ^= 0x80
        XCTAssertThrowsError(try b.open(datagram: forged)) {
            XCTAssertEqual($0 as? NMPCryptoError, .authenticationFailed)
        }
    }

    func testWrongDirectionKeyRejected() throws {
        // A packet sealed by A must not decrypt on A's own recv path (i.e. a
        // reflected packet), because directions use distinct keys.
        let (a, _) = try makeSessionPair()
        let (wire, _) = try a.seal(packetType: .data, senderPeerID: 1,
                                   payload: Data("p".utf8), timestampNanos: 0)
        XCTAssertThrowsError(try a.open(datagram: wire))
    }

    // MARK: Replay protection

    func testDuplicateRejected() throws {
        let (a, b) = try makeSessionPair()
        let (wire, _) = try a.seal(packetType: .data, senderPeerID: 1,
                                   payload: Data("p".utf8), timestampNanos: 0)
        _ = try b.open(datagram: wire)
        XCTAssertThrowsError(try b.open(datagram: wire)) {
            guard case .replayDetected(let seq, let last)? = $0 as? NMPCryptoError else {
                return XCTFail("expected replayDetected, got \($0)")
            }
            XCTAssertEqual(seq, 0)
            XCTAssertEqual(last, 0)
        }
    }

    func testReorderedWithinWindowAccepted() throws {
        // Phase 2: the sliding window accepts unseen sequences inside the
        // window (loss buffer / retransmits need this); Phase 1 rejected them.
        let (a, b) = try makeSessionPair()
        let (w0, _) = try a.seal(packetType: .data, senderPeerID: 1,
                                 payload: Data([0]), timestampNanos: 0)
        let (w1, _) = try a.seal(packetType: .data, senderPeerID: 1,
                                 payload: Data([1]), timestampNanos: 0)
        _ = try b.open(datagram: w1) // deliver seq 1 first
        XCTAssertEqual(try b.open(datagram: w0).payload, Data([0])) // late seq 0 OK
        // …but only once: the reordered packet is now marked seen.
        XCTAssertThrowsError(try b.open(datagram: w0)) {
            guard case .replayDetected? = $0 as? NMPCryptoError else {
                return XCTFail("expected replayDetected, got \($0)")
            }
        }
    }

    func testSequenceOlderThanWindowRejected() throws {
        let (a, b) = try makeSessionPair()
        var wires: [Data] = []
        for n in 0...Int(NMPReplayWindow.windowSize) { // seq 0...64
            let (w, _) = try a.seal(packetType: .data, senderPeerID: 1,
                                    payload: Data([UInt8(n & 0xFF)]), timestampNanos: 0)
            wires.append(w)
        }
        _ = try b.open(datagram: wires[Int(NMPReplayWindow.windowSize)]) // highest = 64
        // seq 1 is age 63 — still inside the window.
        XCTAssertNoThrow(try b.open(datagram: wires[1]))
        // seq 0 is age 64 — fell off the window, must be rejected.
        XCTAssertThrowsError(try b.open(datagram: wires[0])) {
            guard case .replayDetected? = $0 as? NMPCryptoError else {
                return XCTFail("expected replayDetected, got \($0)")
            }
        }
    }

    func testWindowShiftClearsStaleBits() throws {
        // Jumping far ahead (> window) resets the bitmap; the new highest is
        // marked seen and its duplicate is still rejected.
        let (a, b) = try makeSessionPair()
        var last: Data = Data()
        for n in 0..<200 {
            let (w, _) = try a.seal(packetType: .data, senderPeerID: 1,
                                    payload: Data([UInt8(n & 0xFF)]), timestampNanos: 0)
            last = w
            if n == 0 { _ = try b.open(datagram: w) } // only deliver seq 0…
        }
        _ = try b.open(datagram: last) // …then jump straight to seq 199
        XCTAssertThrowsError(try b.open(datagram: last)) {
            guard case .replayDetected? = $0 as? NMPCryptoError else {
                return XCTFail("expected replayDetected, got \($0)")
            }
        }
    }

    func testReplayStateNotAdvancedOnAuthFailure() throws {
        let (a, b) = try makeSessionPair()
        let (w0, _) = try a.seal(packetType: .data, senderPeerID: 1,
                                 payload: Data([0]), timestampNanos: 0)
        var forged = w0
        forged[forged.count - 1] ^= 1
        XCTAssertThrowsError(try b.open(datagram: forged))
        // The genuine packet must still be accepted afterward.
        XCTAssertNoThrow(try b.open(datagram: w0))
    }

    // MARK: Structure / guards

    func testNonceConstruction() {
        let nonce = NMPSecureSession.nonce(seed: 0x0102030405060708, sequence: 0x0A0B0C0D)
        XCTAssertEqual([UInt8](nonce),
                       [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                        0x0A, 0x0B, 0x0C, 0x0D])
    }

    func testHandshakeTypeCannotBeSealed() throws {
        let (a, _) = try makeSessionPair()
        XCTAssertThrowsError(try a.seal(packetType: .handshakeMsg1, senderPeerID: 1,
                                        payload: Data(), timestampNanos: 0)) {
            XCTAssertEqual($0 as? NMPCryptoError, .notEncryptedPacket)
        }
    }

    func testTruncatedDatagramRejected() throws {
        let (a, b) = try makeSessionPair()
        let (wire, _) = try a.seal(packetType: .data, senderPeerID: 1,
                                   payload: Data(repeating: 1, count: 100),
                                   timestampNanos: 0)
        XCTAssertThrowsError(try b.open(datagram: wire.prefix(wire.count - 20)))
    }
}
