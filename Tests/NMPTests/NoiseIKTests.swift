//
//  NoiseIKTests.swift
//  NMPTests — Phase 1
//
//  Direct tests of the Noise_IK_25519_AESGCM_SHA256 implementation
//  (initiator/responder message flow, key derivation, failure modes).
//
//  Property tests (round-trip, key agreement, tamper rejection) plus, since
//  Phase 2, the published cacophony known-answer vector for this pattern
//  (NoiseIKVectorTests below) — closing the interop gap flagged in
//  Phase1_Design.md.
//

import XCTest
import CryptoKit
@testable import NMP

final class NoiseIKTests: XCTestCase {

    private func runHandshake(
        prologueInitiator: Data = Data(),
        prologueResponder: Data = Data(),
        msg1Payload: Data = Data("init-caps".utf8),
        msg2Payload: Data = Data("resp-caps".utf8)
    ) throws -> (i: NoiseHandshakeResult, r: NoiseHandshakeResult,
                 rxAtResponder: Data, rxAtInitiator: Data,
                 iStatic: NoiseStaticKeyPair, rStatic: NoiseStaticKeyPair) {
        let iStatic = NoiseStaticKeyPair()
        let rStatic = NoiseStaticKeyPair()

        let initiator = try NoiseIKHandshake(
            role: .initiator, localStatic: iStatic,
            remoteStaticPublicKey: rStatic.publicKeyData, prologue: prologueInitiator)
        let responder = try NoiseIKHandshake(
            role: .responder, localStatic: rStatic, prologue: prologueResponder)

        let msg1 = try initiator.writeMessage1(payload: msg1Payload)
        let rxAtResponder = try responder.readMessage1(msg1)
        let msg2 = try responder.writeMessage2(payload: msg2Payload)
        let rxAtInitiator = try initiator.readMessage2(msg2)

        return (try initiator.finalize(), try responder.finalize(),
                rxAtResponder, rxAtInitiator, iStatic, rStatic)
    }

    // MARK: Happy path

    func testHandshakeDerivesMatchingKeys() throws {
        let hs = try runHandshake()
        // Initiator's send key must be responder's recv key and vice versa.
        XCTAssertEqual(hs.i.sendKey, hs.r.recvKey)
        XCTAssertEqual(hs.i.recvKey, hs.r.sendKey)
        XCTAssertNotEqual(hs.i.sendKey, hs.i.recvKey, "directions must use distinct keys")
        XCTAssertEqual(hs.i.sendKey.count, 32)
        XCTAssertEqual(hs.i.recvKey.count, 32)
    }

    func testHandshakeHashMatches() throws {
        let hs = try runHandshake()
        XCTAssertEqual(hs.i.handshakeHash, hs.r.handshakeHash)
        XCTAssertEqual(hs.i.handshakeHash.count, 32)
    }

    func testPayloadsDeliveredAndAuthenticated() throws {
        let hs = try runHandshake(
            msg1Payload: Data("capability-A".utf8),
            msg2Payload: Data("capability-B".utf8))
        XCTAssertEqual(hs.rxAtResponder, Data("capability-A".utf8))
        XCTAssertEqual(hs.rxAtInitiator, Data("capability-B".utf8))
    }

    func testMutualAuthentication() throws {
        let hs = try runHandshake()
        XCTAssertEqual(hs.i.remoteStaticPublicKey, hs.rStatic.publicKeyData)
        XCTAssertEqual(hs.r.remoteStaticPublicKey, hs.iStatic.publicKeyData)
    }

    func testSessionsAreUnique() throws {
        // Two handshakes between the same static keys must derive different
        // session keys (fresh ephemerals).
        let iStatic = NoiseStaticKeyPair()
        let rStatic = NoiseStaticKeyPair()
        var keys: [Data] = []
        for _ in 0..<2 {
            let i = try NoiseIKHandshake(role: .initiator, localStatic: iStatic,
                                         remoteStaticPublicKey: rStatic.publicKeyData)
            let r = try NoiseIKHandshake(role: .responder, localStatic: rStatic)
            _ = try r.readMessage1(try i.writeMessage1(payload: Data()))
            _ = try i.readMessage2(try r.writeMessage2(payload: Data()))
            keys.append(try i.finalize().sendKey)
        }
        XCTAssertNotEqual(keys[0], keys[1])
    }

    // MARK: Failure modes

    func testTamperedMessage1Rejected() throws {
        let iStatic = NoiseStaticKeyPair()
        let rStatic = NoiseStaticKeyPair()
        let i = try NoiseIKHandshake(role: .initiator, localStatic: iStatic,
                                     remoteStaticPublicKey: rStatic.publicKeyData)
        let r = try NoiseIKHandshake(role: .responder, localStatic: rStatic)
        var msg1 = try i.writeMessage1(payload: Data("x".utf8))
        msg1[40] ^= 0xFF // corrupt the encrypted static section
        XCTAssertThrowsError(try r.readMessage1(msg1))
    }

    func testWrongResponderStaticFails() throws {
        // Initiator encrypts to the WRONG responder key → responder cannot
        // decrypt message 1 (identity binding of IK).
        let iStatic = NoiseStaticKeyPair()
        let rStatic = NoiseStaticKeyPair()
        let wrongStatic = NoiseStaticKeyPair()
        let i = try NoiseIKHandshake(role: .initiator, localStatic: iStatic,
                                     remoteStaticPublicKey: wrongStatic.publicKeyData)
        let r = try NoiseIKHandshake(role: .responder, localStatic: rStatic)
        let msg1 = try i.writeMessage1(payload: Data())
        XCTAssertThrowsError(try r.readMessage1(msg1))
    }

    func testPrologueMismatchFails() throws {
        XCTAssertThrowsError(
            try runHandshake(prologueInitiator: Data("a".utf8),
                             prologueResponder: Data("b".utf8)))
    }

    func testTruncatedMessagesRejected() throws {
        let iStatic = NoiseStaticKeyPair()
        let rStatic = NoiseStaticKeyPair()
        let i = try NoiseIKHandshake(role: .initiator, localStatic: iStatic,
                                     remoteStaticPublicKey: rStatic.publicKeyData)
        let r = try NoiseIKHandshake(role: .responder, localStatic: rStatic)
        let msg1 = try i.writeMessage1(payload: Data())
        XCTAssertThrowsError(try r.readMessage1(msg1.prefix(30)))
        XCTAssertThrowsError(try r.readMessage1(Data()))
    }

    func testFinalizeBeforeCompleteThrows() throws {
        let i = try NoiseIKHandshake(role: .initiator, localStatic: NoiseStaticKeyPair(),
                                     remoteStaticPublicKey: NoiseStaticKeyPair().publicKeyData)
        XCTAssertThrowsError(try i.finalize()) {
            XCTAssertEqual($0 as? NoiseError, .handshakeNotComplete)
        }
    }

    func testMessageSizes() throws {
        // msg1 = e(32) + enc_s(48) + enc_payload(len+16)
        let payload = Data(repeating: 1, count: 10)
        let i = try NoiseIKHandshake(role: .initiator, localStatic: NoiseStaticKeyPair(),
                                     remoteStaticPublicKey: NoiseStaticKeyPair().publicKeyData)
        let msg1 = try i.writeMessage1(payload: payload)
        XCTAssertEqual(msg1.count, 32 + 48 + payload.count + 16)
    }
}

// MARK: - Published known-answer vector (Phase 2 carry-in)

/// The official cacophony test vector for `Noise_IK_25519_AESGCM_SHA256`
/// (as shipped in snow's tests/vectors/cacophony.txt). Fixed static and
/// ephemeral keys drive the handshake; every ciphertext, the handshake hash,
/// and both transport keys must match the reference byte-for-byte. This
/// closes the "algorithm verified, Swift translation not" gap flagged in
/// Docs/Phase1_Design.md.
final class NoiseIKVectorTests: XCTestCase {

    // Vector inputs.
    private let prologue        = hexData("4a6f686e2047616c74")            // "John Galt"
    private let initStatic      = hexData("e61ef9919cde45dd5f82166404bd08e38bceb5dfdfded0a34c8df7ed542214d1")
    private let initEphemeral   = hexData("893e28b9dc6ca8d611ab664754b8ceb7bac5117349a4439a6b0569da977c464a")
    private let respStatic      = hexData("4a3acbfdb163dec651dfa3194dece676d437029c62a408b4c5ea9114246e4893")
    private let respEphemeral   = hexData("bbdb4cdbd309f1a1f2e1456967fe288cadd6f712d65dc7b7793d5e63da6b375b")
    private let initRemoteStatic = hexData("31e0303fd6418d2f8c0e78b91f22e8caed0fbe48656dcf4767e4834f701b8f62")

    // Vector expectations.
    private let expectedHandshakeHash = hexData("669c8640d9e42a3cda2f232f78597ceefb01daa6e3df81181ccce6fc6b5026bf")
    private let msg1Payload = hexData("4c756477696720766f6e204d69736573") // "Ludwig von Mises"
    private let msg1Cipher = hexData(
        "ca35def5ae56cec33dc2036731ab14896bc4c75dbb07a61f879f8e3afa4c7944" +
        "4e417bc55c7a8166c993356c1be41ef67818a292426f301556c7f26b21d25ddb" +
        "097153891a9a956cff47b83e63ad8d701c1342c209cff1ca5ecd43402762ac24" +
        "9e3bd3a4c0a145fe07cb5dae28ea13a3")
    private let msg2Payload = hexData("4d757272617920526f746862617264") // "Murray Rothbard"
    private let msg2Cipher = hexData(
        "95ebc60d2b1fa672c1f46a8aa265ef51bfe38e7ccb39ec5be34069f144808843" +
        "af2ccf9972e22afc67aeafcd25162f7f98c363b7762e3e4cb7d272e39f27a5")
    // Transport messages alternate initiator→responder starting with msg 3.
    private let transportVectors: [(payload: Data, cipher: Data)] = [
        (hexData("462e20412e20486179656b"),                     // "F. A. Hayek", k1 n=0
         hexData("66acfc92e3197de166809e6d4d5d003dcc819a84bc3522ca53c9d9")),
        (hexData("4361726c204d656e676572"),                     // "Carl Menger", k2 n=0
         hexData("71f89aa6533a6de70b0826864dd75f60806ee40170c16290189eb3")),
        (hexData("4a65616e2d426170746973746520536179"),         // "Jean-Baptiste Say", k1 n=1
         hexData("4795a3423550c8bf00386bd496a3e2c76c10669d2a75ab8f79b5094c5412a25705")),
        (hexData("457567656e2042f6686d20766f6e2042617765726b"), // "Eugen Böhm von Bawerk", k2 n=1
         hexData("aa0bb39097555c918e40be82abc2b909eb79d9eb87adb07e268fc37323a6cf904fd01fb391")),
    ]

    func testCacophonyIKVector() throws {
        let iStatic = try NoiseStaticKeyPair(rawPrivateKey: initStatic)
        let rStatic = try NoiseStaticKeyPair(rawPrivateKey: respStatic)
        // Sanity: the vector's init_remote_static is the responder's public key.
        XCTAssertEqual(rStatic.publicKeyData, initRemoteStatic)

        let i = try NoiseIKHandshake(role: .initiator, localStatic: iStatic,
                                     remoteStaticPublicKey: initRemoteStatic,
                                     prologue: prologue)
        let r = try NoiseIKHandshake(role: .responder, localStatic: rStatic,
                                     prologue: prologue)
        i.ephemeralOverrideForTesting =
            try Curve25519.KeyAgreement.PrivateKey(rawRepresentation: initEphemeral)
        r.ephemeralOverrideForTesting =
            try Curve25519.KeyAgreement.PrivateKey(rawRepresentation: respEphemeral)

        // Message 1: exact ciphertext, and the responder recovers the payload.
        let msg1 = try i.writeMessage1(payload: msg1Payload)
        XCTAssertEqual(msg1, msg1Cipher)
        XCTAssertEqual(try r.readMessage1(msg1), msg1Payload)

        // Message 2.
        let msg2 = try r.writeMessage2(payload: msg2Payload)
        XCTAssertEqual(msg2, msg2Cipher)
        XCTAssertEqual(try i.readMessage2(msg2), msg2Payload)

        // Handshake hash and transport keys.
        let resI = try i.finalize()
        let resR = try r.finalize()
        XCTAssertEqual(resI.handshakeHash, expectedHandshakeHash)
        XCTAssertEqual(resR.handshakeHash, expectedHandshakeHash)
        XCTAssertEqual(resI.sendKey, resR.recvKey) // k1
        XCTAssertEqual(resI.recvKey, resR.sendKey) // k2

        // Transport messages: Noise CipherState framing (nonce = 4 zero bytes
        // ‖ u64 BE counter, empty AD). NMP's own transport nonces differ by
        // design; this verifies Split() produced the reference keys.
        for (n, vector) in transportVectors.enumerated() {
            let key = n % 2 == 0 ? resI.sendKey : resI.recvKey
            let counter = UInt64(n / 2)
            var nonce = Data(repeating: 0, count: 4)
            nonce.appendBigEndian(counter)
            let sealed = try AES.GCM.seal(
                vector.payload,
                using: SymmetricKey(data: key),
                nonce: AES.GCM.Nonce(data: nonce))
            XCTAssertEqual(sealed.ciphertext + sealed.tag, vector.cipher,
                           "transport message \(n + 2) mismatch")
        }
    }
}

private func hexData(_ hex: String) -> Data {
    precondition(hex.count % 2 == 0)
    var out = Data(capacity: hex.count / 2)
    var index = hex.startIndex
    while index < hex.endIndex {
        let next = hex.index(index, offsetBy: 2)
        out.append(UInt8(hex[index..<next], radix: 16)!)
        index = next
    }
    return out
}
