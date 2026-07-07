//
//  NoiseIK.swift
//  NMP — Phase 1
//
//  Noise_IK_25519_AESGCM_SHA256, implemented directly from the Noise Protocol
//  Framework specification (revision 34), using CryptoKit primitives:
//  Curve25519 key agreement, AES-256-GCM, SHA-256, HMAC-SHA256-based HKDF.
//
//  IK pattern:
//      <- s            (pre-message: initiator knows responder's static key)
//      ...
//      -> e, es, s, ss (message 1)
//      <- e, ee, se    (message 2)
//
//  1-RTT mutual authentication. After message 2, Split() yields two
//  AES-256-GCM transport keys (initiator→responder, responder→initiator).
//
//  Known issues flagged in Docs/Phase1_Design.md — NOT silently fixed here:
//  constant-time properties of CryptoKit ops, nonce exhaustion at 2^32
//  transport packets, and the loose clock-sync assumption.
//

import Foundation
import CryptoKit

// MARK: - Errors

public enum NoiseError: Error, Equatable, Sendable {
    case invalidState(String)
    case decryptFailed              // AEAD authentication failure
    case malformedMessage           // truncated / wrong length
    case invalidPublicKey
    case handshakeNotComplete
    case handshakeAlreadyComplete
}

// MARK: - Static identity

/// A peer's long-term Curve25519 static key pair. Distributed out-of-band
/// or derived from a pairing code (spec §1).
public struct NoiseStaticKeyPair: Sendable {
    public let privateKey: Curve25519.KeyAgreement.PrivateKey
    public var publicKeyData: Data { privateKey.publicKey.rawRepresentation }

    public init() {
        self.privateKey = Curve25519.KeyAgreement.PrivateKey()
    }
    public init(rawPrivateKey: Data) throws {
        do {
            self.privateKey = try Curve25519.KeyAgreement.PrivateKey(rawRepresentation: rawPrivateKey)
        } catch {
            throw NoiseError.invalidPublicKey
        }
    }
}

// MARK: - CipherState (Noise §5.1)

struct NoiseCipherState {
    private(set) var key: SymmetricKey?
    private(set) var nonce: UInt64 = 0

    var hasKey: Bool { key != nil }

    mutating func initializeKey(_ k: Data) {
        precondition(k.count == 32)
        key = SymmetricKey(data: k)
        nonce = 0
    }

    /// Noise AESGCM nonce: 32 bits of zeros followed by 64-bit big-endian n.
    private static func gcmNonce(_ n: UInt64) -> Data {
        var d = Data(repeating: 0, count: 4)
        d.appendBigEndian(n)
        return d
    }

    mutating func encrypt(ad: Data, plaintext: Data) throws -> Data {
        guard let key else { return plaintext } // no key yet: identity (Noise §5.1)
        let nonceData = Self.gcmNonce(nonce)
        let sealed: AES.GCM.SealedBox
        do {
            sealed = try AES.GCM.seal(
                plaintext,
                using: key,
                nonce: AES.GCM.Nonce(data: nonceData),
                authenticating: ad
            )
        } catch {
            throw NoiseError.invalidState("AES-GCM seal failed: \(error)")
        }
        nonce += 1
        return sealed.ciphertext + sealed.tag
    }

    mutating func decrypt(ad: Data, ciphertextAndTag: Data) throws -> Data {
        guard let key else { return ciphertextAndTag }
        guard ciphertextAndTag.count >= 16 else { throw NoiseError.malformedMessage }
        let nonceData = Self.gcmNonce(nonce)
        let ct = ciphertextAndTag.prefix(ciphertextAndTag.count - 16)
        let tag = ciphertextAndTag.suffix(16)
        do {
            let box = try AES.GCM.SealedBox(
                nonce: AES.GCM.Nonce(data: nonceData),
                ciphertext: ct,
                tag: tag
            )
            let plaintext = try AES.GCM.open(box, using: key, authenticating: ad)
            nonce += 1
            return plaintext
        } catch {
            throw NoiseError.decryptFailed
        }
    }
}

// MARK: - HKDF (Noise §4.3)

enum NoiseKDF {
    static func hmacSHA256(key: Data, data: Data) -> Data {
        let mac = HMAC<SHA256>.authenticationCode(for: data, using: SymmetricKey(data: key))
        return Data(mac)
    }

    /// Noise HKDF with two outputs.
    static func hkdf2(chainingKey: Data, inputKeyMaterial: Data) -> (Data, Data) {
        let tempKey = hmacSHA256(key: chainingKey, data: inputKeyMaterial)
        let out1 = hmacSHA256(key: tempKey, data: Data([0x01]))
        let out2 = hmacSHA256(key: tempKey, data: out1 + Data([0x02]))
        return (out1, out2)
    }
}

// MARK: - SymmetricState (Noise §5.2)

struct NoiseSymmetricState {
    private(set) var chainingKey: Data
    private(set) var handshakeHash: Data
    var cipher = NoiseCipherState()

    init(protocolName: String) {
        let nameData = Data(protocolName.utf8)
        if nameData.count <= 32 {
            handshakeHash = nameData + Data(repeating: 0, count: 32 - nameData.count)
        } else {
            handshakeHash = Data(SHA256.hash(data: nameData))
        }
        chainingKey = handshakeHash
    }

    mutating func mixHash(_ data: Data) {
        handshakeHash = Data(SHA256.hash(data: handshakeHash + data))
    }

    mutating func mixKey(_ inputKeyMaterial: Data) {
        let (ck, tempK) = NoiseKDF.hkdf2(chainingKey: chainingKey, inputKeyMaterial: inputKeyMaterial)
        chainingKey = ck
        cipher.initializeKey(tempK)
    }

    mutating func encryptAndHash(_ plaintext: Data) throws -> Data {
        let ciphertext = try cipher.encrypt(ad: handshakeHash, plaintext: plaintext)
        mixHash(ciphertext)
        return ciphertext
    }

    mutating func decryptAndHash(_ ciphertext: Data) throws -> Data {
        let plaintext = try cipher.decrypt(ad: handshakeHash, ciphertextAndTag: ciphertext)
        mixHash(ciphertext)
        return plaintext
    }

    /// Noise §5.2 Split: derive the two transport keys.
    /// First key encrypts initiator→responder, second responder→initiator.
    func split() -> (Data, Data) {
        NoiseKDF.hkdf2(chainingKey: chainingKey, inputKeyMaterial: Data())
    }
}

// MARK: - Handshake result

public struct NoiseHandshakeResult: Sendable {
    /// AES-256-GCM key for packets this peer SENDS.
    public let sendKey: Data
    /// AES-256-GCM key for packets this peer RECEIVES.
    public let recvKey: Data
    /// Channel-binding value (Noise handshake hash).
    public let handshakeHash: Data
    /// Remote peer's authenticated static public key (32 bytes).
    public let remoteStaticPublicKey: Data
}

// MARK: - HandshakeState (Noise §5.3, IK only)

public final class NoiseIKHandshake {

    public enum Role: Sendable { case initiator, responder }

    public static let protocolName = "Noise_IK_25519_AESGCM_SHA256"
    static let dhLen = 32
    static let tagLen = 16

    private let role: Role
    private var symmetric: NoiseSymmetricState
    private let localStatic: NoiseStaticKeyPair
    private var localEphemeral: Curve25519.KeyAgreement.PrivateKey?
    private var remoteStaticPub: Data?     // known upfront for initiator; learned from msg1 for responder
    private var remoteEphemeralPub: Data?
    private var complete = false

    /// Test-only: fixed ephemeral so known-answer vectors (cacophony/snow)
    /// are reproducible. Internal — reachable only via @testable import.
    var ephemeralOverrideForTesting: Curve25519.KeyAgreement.PrivateKey?

    /// - Parameters:
    ///   - role: initiator must supply `remoteStaticPublicKey`; responder must not.
    ///   - prologue: optional context data both sides must agree on
    ///     (mixed into the transcript; mismatch fails the handshake).
    public init(
        role: Role,
        localStatic: NoiseStaticKeyPair,
        remoteStaticPublicKey: Data? = nil,
        prologue: Data = Data()
    ) throws {
        if role == .initiator {
            guard let rs = remoteStaticPublicKey, rs.count == Self.dhLen else {
                throw NoiseError.invalidPublicKey
            }
            self.remoteStaticPub = rs
        } else if remoteStaticPublicKey != nil {
            // Responder learns the initiator's static from message 1.
            throw NoiseError.invalidState("responder must not pre-set remote static")
        }

        self.role = role
        self.localStatic = localStatic
        self.symmetric = NoiseSymmetricState(protocolName: Self.protocolName)
        symmetric.mixHash(prologue)

        // IK pre-message "<- s": mix responder's static public key.
        switch role {
        case .initiator: symmetric.mixHash(remoteStaticPub!)
        case .responder: symmetric.mixHash(localStatic.publicKeyData)
        }
    }

    private func dh(_ priv: Curve25519.KeyAgreement.PrivateKey, _ pubData: Data) throws -> Data {
        let pub: Curve25519.KeyAgreement.PublicKey
        do {
            pub = try Curve25519.KeyAgreement.PublicKey(rawRepresentation: pubData)
        } catch {
            throw NoiseError.invalidPublicKey
        }
        let shared: SharedSecret
        do {
            shared = try priv.sharedSecretFromKeyAgreement(with: pub)
        } catch {
            throw NoiseError.invalidPublicKey
        }
        return shared.withUnsafeBytes { Data($0) }
    }

    // MARK: Message 1: -> e, es, s, ss  (initiator writes)

    public func writeMessage1(payload: Data) throws -> Data {
        guard role == .initiator, localEphemeral == nil, !complete else {
            throw NoiseError.invalidState("writeMessage1: wrong role or state")
        }
        guard let rs = remoteStaticPub else { throw NoiseError.invalidPublicKey }

        var out = Data()

        // e
        let e = ephemeralOverrideForTesting ?? Curve25519.KeyAgreement.PrivateKey()
        localEphemeral = e
        let ePub = e.publicKey.rawRepresentation
        out.append(ePub)
        symmetric.mixHash(ePub)

        // es
        symmetric.mixKey(try dh(e, rs))

        // s (encrypted static identity)
        out.append(try symmetric.encryptAndHash(localStatic.publicKeyData))

        // ss
        symmetric.mixKey(try dh(localStatic.privateKey, rs))

        // payload (capability advertisement)
        out.append(try symmetric.encryptAndHash(payload))
        return out
    }

    public func readMessage1(_ message: Data) throws -> Data {
        guard role == .responder, remoteEphemeralPub == nil, !complete else {
            throw NoiseError.invalidState("readMessage1: wrong role or state")
        }
        // e(32) + enc_s(32+16) + enc_payload(>=16)
        let minLen = Self.dhLen + (Self.dhLen + Self.tagLen) + Self.tagLen
        guard message.count >= minLen else { throw NoiseError.malformedMessage }
        let msg = Data(message) // rebase indices

        var offset = 0

        // e
        let re = msg.subdata(in: offset..<offset + Self.dhLen)
        offset += Self.dhLen
        remoteEphemeralPub = re
        symmetric.mixHash(re)

        // es
        symmetric.mixKey(try dh(localStatic.privateKey, re))

        // s
        let encStatic = msg.subdata(in: offset..<offset + Self.dhLen + Self.tagLen)
        offset += Self.dhLen + Self.tagLen
        let rs = try symmetric.decryptAndHash(encStatic)
        guard rs.count == Self.dhLen else { throw NoiseError.malformedMessage }
        remoteStaticPub = rs

        // ss
        symmetric.mixKey(try dh(localStatic.privateKey, rs))

        // payload
        let encPayload = msg.subdata(in: offset..<msg.count)
        return try symmetric.decryptAndHash(encPayload)
    }

    // MARK: Message 2: <- e, ee, se  (responder writes)

    public func writeMessage2(payload: Data) throws -> Data {
        guard role == .responder, localEphemeral == nil, !complete,
              let re = remoteEphemeralPub, let rs = remoteStaticPub else {
            throw NoiseError.invalidState("writeMessage2: message 1 not yet processed")
        }

        var out = Data()

        // e
        let e = ephemeralOverrideForTesting ?? Curve25519.KeyAgreement.PrivateKey()
        localEphemeral = e
        let ePub = e.publicKey.rawRepresentation
        out.append(ePub)
        symmetric.mixHash(ePub)

        // ee
        symmetric.mixKey(try dh(e, re))

        // se = DH(initiator static, responder ephemeral); responder computes DH(e, rs).
        symmetric.mixKey(try dh(e, rs))

        // payload (capability advertisement)
        out.append(try symmetric.encryptAndHash(payload))
        complete = true
        return out
    }

    public func readMessage2(_ message: Data) throws -> Data {
        guard role == .initiator, !complete, let e = localEphemeral else {
            throw NoiseError.invalidState("readMessage2: message 1 not yet sent")
        }
        let minLen = Self.dhLen + Self.tagLen
        guard message.count >= minLen else { throw NoiseError.malformedMessage }
        let msg = Data(message)

        // e
        let re = msg.subdata(in: 0..<Self.dhLen)
        remoteEphemeralPub = re
        symmetric.mixHash(re)

        // ee
        symmetric.mixKey(try dh(e, re))

        // se (initiator side: DH(s, re))
        symmetric.mixKey(try dh(localStatic.privateKey, re))

        // payload
        let payload = try symmetric.decryptAndHash(msg.subdata(in: Self.dhLen..<msg.count))
        complete = true
        return payload
    }

    // MARK: Finalize

    /// Call after message 2 has been written (responder) or read (initiator).
    public func finalize() throws -> NoiseHandshakeResult {
        guard complete else { throw NoiseError.handshakeNotComplete }
        guard let rs = remoteStaticPub else { throw NoiseError.invalidState("no remote static") }
        let (k1, k2) = symmetric.split() // k1: initiator→responder, k2: responder→initiator
        switch role {
        case .initiator:
            return NoiseHandshakeResult(
                sendKey: k1, recvKey: k2,
                handshakeHash: symmetric.handshakeHash,
                remoteStaticPublicKey: rs)
        case .responder:
            return NoiseHandshakeResult(
                sendKey: k2, recvKey: k1,
                handshakeHash: symmetric.handshakeHash,
                remoteStaticPublicKey: rs)
        }
    }
}
