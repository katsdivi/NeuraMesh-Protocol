//
//  SymmetricCrypto.swift
//  NMP — Phase 1 (replay window widened in Phase 2)
//
//  Post-handshake transport encryption per NMP_Specification.md §3:
//    - AES-256-GCM per session, keys from NoiseIKHandshake.finalize().
//    - Nonce (12 bytes) = nonce_seed (8 bytes) ‖ sequence_number (4 bytes).
//      Each direction uses the seed advertised by the SENDER of that direction.
//    - AAD = the 20-byte NMP header (tamper-evident header).
//    - Replay protection: DTLS/QUIC-style sliding window (highest authenticated
//      sequence + 64-bit bitmap). A sequence is accepted iff it is newer than
//      the highest seen, or within the 64-packet window and not yet seen.
//      Phase 1 shipped strict monotonic acceptance; Phase 2 widened it so the
//      loss buffer can hold reordered/retransmitted datagrams (the carry-in
//      flagged in Docs/Phase1_Design.md). Window state still advances only
//      after successful GCM authentication.
//

import Foundation
import CryptoKit

// MARK: - Errors

public enum NMPCryptoError: Error, Equatable, Sendable {
    case authenticationFailed          // GCM tag or AAD mismatch → drop packet
    case replayDetected(sequence: UInt32, lastSeen: UInt32)
    case notEncryptedPacket            // T=0 packet handed to the session layer
    case truncatedCiphertext
    case payloadLengthMismatch
    /// Sequence space exhausted (2^32 packets sent). Nonces would repeat —
    /// catastrophic for GCM. Caller MUST re-handshake (spec Phase 1 known issue).
    case rekeyRequired
}

// MARK: - Session keys

public struct NMPSessionKeys: Sendable {
    public let sendKey: SymmetricKey
    public let recvKey: SymmetricKey
    /// 8-byte seed this peer generated; prefixes nonces for packets we SEND.
    public let localNonceSeed: UInt64
    /// 8-byte seed the remote peer generated; prefixes nonces for packets we RECEIVE.
    public let remoteNonceSeed: UInt64

    public init(handshake: NoiseHandshakeResult, localNonceSeed: UInt64, remoteNonceSeed: UInt64) {
        self.sendKey = SymmetricKey(data: handshake.sendKey)
        self.recvKey = SymmetricKey(data: handshake.recvKey)
        self.localNonceSeed = localNonceSeed
        self.remoteNonceSeed = remoteNonceSeed
    }
}

// MARK: - Replay window

/// Sliding anti-replay window (RFC 6479 / DTLS style), sized to match the
/// Phase 2 loss buffer: a retransmitted packet is acceptable for exactly as
/// long as its sender can still retransmit it.
///
/// `bitmap` bit i tracks sequence `highest - i` (bit 0 = highest itself).
struct NMPReplayWindow: Sendable {
    /// Must stay ≥ the reliability layer's retransmit window (spec: 64).
    static let windowSize: UInt32 = 64

    private(set) var highest: UInt32?
    private var bitmap: UInt64 = 0

    /// True if `sequence` would be accepted (newer than anything seen, or
    /// inside the window and not yet marked). Does not mutate state.
    func isAcceptable(_ sequence: UInt32) -> Bool {
        guard let highest else { return true }
        if sequence > highest { return true }
        let age = highest - sequence
        guard age < Self.windowSize else { return false } // too old to track
        return bitmap & (1 << UInt64(age)) == 0
    }

    /// Marks `sequence` as seen. Call only after GCM authentication succeeds.
    mutating func record(_ sequence: UInt32) {
        guard let h = highest else {
            highest = sequence
            bitmap = 1
            return
        }
        if sequence > h {
            let shift = UInt64(sequence - h)
            bitmap = shift >= 64 ? 1 : (bitmap << shift) | 1
            highest = sequence
        } else {
            bitmap |= 1 << UInt64(h - sequence)
        }
    }
}

// MARK: - Secure session

/// Seals and opens encrypted NMP packets for one established peer session.
/// Not thread-safe by itself: PeerConnection serializes access on its queue.
public final class NMPSecureSession {

    private let keys: NMPSessionKeys

    /// Sequence number for the next packet we send. Wraps are NOT allowed —
    /// see `NMPCryptoError.rekeyRequired`.
    private(set) var nextSendSequence: UInt32 = 0

    /// Sliding anti-replay window over remote sequence numbers.
    private(set) var replayWindow = NMPReplayWindow()

    /// Highest sequence number accepted from the remote peer, or nil if none yet.
    var lastAcceptedRemoteSequence: UInt32? { replayWindow.highest }

    public init(keys: NMPSessionKeys) {
        self.keys = keys
    }

    // MARK: Nonce

    static func nonce(seed: UInt64, sequence: UInt32) -> Data {
        var d = Data(capacity: 12)
        d.appendBigEndian(seed)
        d.appendBigEndian(sequence)
        return d
    }

    // MARK: Seal (send path)

    /// Builds a complete encrypted NMP datagram:
    /// header (plaintext, used as AAD) + ciphertext + 16-byte GCM tag.
    /// Assigns the sequence number; caller supplies everything else.
    public func seal(
        packetType: NMPPacketType,
        flags: NMPFlags = [],
        senderPeerID: UInt32,
        payload: Data,
        timestampNanos: UInt64
    ) throws -> (datagram: Data, sequenceNumber: UInt32) {
        guard !packetType.isHandshake else { throw NMPCryptoError.notEncryptedPacket }
        guard payload.count <= NMPHeader.maxPayloadLength else {
            throw NMPCryptoError.payloadLengthMismatch
        }
        // Nonce exhaustion guard: refuse to wrap the 32-bit sequence space.
        guard nextSendSequence != UInt32.max else { throw NMPCryptoError.rekeyRequired }

        let sequence = nextSendSequence
        let header = NMPHeader(
            isEncrypted: true,
            flags: flags,
            packetType: packetType,
            payloadLength: UInt16(payload.count),
            sequenceNumber: sequence,
            senderPeerID: senderPeerID,
            timestampNanos: timestampNanos
        )
        let headerBytes = NMPPacketCodec.encodeHeader(header)
        let nonce = Self.nonce(seed: keys.localNonceSeed, sequence: sequence)

        let sealed: AES.GCM.SealedBox
        do {
            sealed = try AES.GCM.seal(
                payload,
                using: keys.sendKey,
                nonce: AES.GCM.Nonce(data: nonce),
                authenticating: headerBytes
            )
        } catch {
            throw NMPCryptoError.authenticationFailed
        }

        nextSendSequence += 1
        return (headerBytes + sealed.ciphertext + sealed.tag, sequence)
    }

    // MARK: Open (receive path)

    /// Validates, replay-checks, and decrypts an encrypted NMP datagram.
    /// On success returns the packet with decrypted payload and advances the
    /// replay window. All failures leave replay state untouched.
    public func open(datagram: Data) throws -> NMPPacket {
        let header = try NMPPacketCodec.decodeHeader(datagram)
        guard header.isEncrypted else { throw NMPCryptoError.notEncryptedPacket }

        // Replay check BEFORE any crypto work (cheap reject). Reordered
        // sequences inside the sliding window are legitimate (Phase 2 loss
        // buffer / retransmits); duplicates and too-old sequences are not.
        guard replayWindow.isAcceptable(header.sequenceNumber) else {
            throw NMPCryptoError.replayDetected(
                sequence: header.sequenceNumber,
                lastSeen: replayWindow.highest ?? header.sequenceNumber)
        }

        let expectedTotal = NMPHeader.byteCount + Int(header.payloadLength) + NMPHeader.gcmTagByteCount
        guard datagram.count >= NMPHeader.byteCount + NMPHeader.gcmTagByteCount else {
            throw NMPCryptoError.truncatedCiphertext
        }
        guard datagram.count == expectedTotal else {
            throw NMPCryptoError.payloadLengthMismatch
        }

        let bytes = Data(datagram) // rebase
        let headerBytes = bytes.prefix(NMPHeader.byteCount)
        let ciphertext = bytes.subdata(
            in: NMPHeader.byteCount..<(NMPHeader.byteCount + Int(header.payloadLength)))
        let tag = bytes.suffix(NMPHeader.gcmTagByteCount)

        let nonce = Self.nonce(seed: keys.remoteNonceSeed, sequence: header.sequenceNumber)
        let plaintext: Data
        do {
            let box = try AES.GCM.SealedBox(
                nonce: AES.GCM.Nonce(data: nonce),
                ciphertext: ciphertext,
                tag: tag
            )
            plaintext = try AES.GCM.open(box, using: keys.recvKey, authenticating: headerBytes)
        } catch {
            throw NMPCryptoError.authenticationFailed
        }

        // Only advance replay state after successful authentication.
        replayWindow.record(header.sequenceNumber)
        return NMPPacket(header: header, payload: plaintext)
    }

    /// Marks a sequence as seen without a datagram (Phase 3): an FEC-
    /// reconstructed packet's content was authenticated transitively — the
    /// parity packet and every surviving group member each passed GCM. If
    /// the original datagram straggles in later (or a NACK retransmit races
    /// the recovery), it is then dropped as a replay instead of delivering
    /// the payload twice.
    func markSequenceSeen(_ sequence: UInt32) {
        replayWindow.record(sequence)
    }
}
