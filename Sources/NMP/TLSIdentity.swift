//
//  TLSIdentity.swift
//  NMP — Mesh 2.5
//
//  An ephemeral, self-signed TLS identity built with nothing but
//  CryptoKit and the Security framework — the missing piece that kept
//  TLS 1.3 and QUIC out of the measured transport race ("QUIC needs a
//  TLS identity, which a zero-dependency LAN tool can't conjure").
//  It can: X.509 is just DER, and CryptoKit signs it.
//
//  What this builds:
//  - a fresh P-256 key pair,
//  - a minimal self-signed X.509 v3 certificate (ecdsa-with-SHA256,
//    CN only, 24 h validity), hand-encoded DER,
//  - a SecIdentity by staging cert + key in the keychain (macOS's only
//    route to SecIdentity), tagged with a UUID label and REMOVED again
//    by `cleanup()` — nothing persists past the race.
//
//  Trust model: the race client does NOT trust this cert as a CA would.
//  It pins the exact DER bytes it just generated (byte-equality in the
//  verify block). Nothing outside this process ever trusts it.
//
//  macOS-only: the race runs on the dashboard Mac; iOS never races.
//

import Foundation
#if os(macOS)
import Security

public final class NMPEphemeralTLSIdentity {

    public enum IdentityError: Error, CustomStringConvertible {
        case keyCreateFailed(String)
        case signingFailed(String)
        case certificateRejected
        case keychainAdd(OSStatus)
        case identityNotFound(OSStatus)

        public var description: String {
            switch self {
            case .keyCreateFailed(let detail):
                return "keychain key generation failed: \(detail)"
            case .signingFailed(let detail):
                return "certificate signing failed: \(detail)"
            case .certificateRejected:
                return "Security framework rejected the generated DER"
            case .keychainAdd(let status):
                return "keychain add failed (OSStatus \(status))"
            case .identityNotFound(let status):
                return "no identity for the staged cert+key (OSStatus \(status))"
            }
        }
    }

    /// The staged identity, usable via `sec_identity_create`.
    public let identity: SecIdentity
    /// Exact certificate bytes — what the race client pins.
    public let certificateDER: Data

    private let certificate: SecCertificate
    private let privateKey: SecKey
    private var cleaned = false
    private let lock = NSLock()

    private let keyTag: Data

    /// Generates key + certificate and stages them in the default
    /// keychain. Call `cleanup()` when done (deinit also sweeps).
    public init(commonName: String = "NMP Transport Race (ephemeral)") throws {
        // SecIdentity only exists as the keychain's join of a cert and
        // its private key, so the key is BORN in the keychain
        // (kSecAttrIsPermanent) instead of imported: the legacy file
        // keychain rejects SecItemAdd of an ephemeral SecKey (-25304),
        // and the data-protection keychain needs an
        // application-identifier entitlement that unsigned CLI and test
        // processes don't have (-34018). UUID tag/label mark the items
        // as ours (and identify strays if a crash ever skips cleanup).
        let label = "NMP race ephemeral \(UUID().uuidString)"
        let tag = Data(label.utf8)
        var createError: Unmanaged<CFError>?
        guard let secKey = SecKeyCreateRandomKey([
            kSecAttrKeyType: kSecAttrKeyTypeECSECPrimeRandom,
            kSecAttrKeySizeInBits: 256,
            kSecUseDataProtectionKeychain: false,
            kSecAttrLabel: label,
            kSecPrivateKeyAttrs: [
                kSecAttrIsPermanent: true,
                kSecAttrApplicationTag: tag,
            ] as CFDictionary,
        ] as CFDictionary, &createError) else {
            let detail = createError.map { String(describing: $0.takeRetainedValue()) }
                ?? "unknown"
            throw IdentityError.keyCreateFailed(detail)
        }
        let removeKey = {
            SecItemDelete([kSecClass: kSecClassKey,
                           kSecAttrApplicationTag: tag,
                           kSecUseDataProtectionKeychain: false] as CFDictionary)
        }

        guard let publicKey = SecKeyCopyPublicKey(secKey),
              let publicX963 = SecKeyCopyExternalRepresentation(
                publicKey, &createError) as Data? else {
            removeKey()
            throw IdentityError.keyCreateFailed("no public key representation")
        }

        let der: Data
        do {
            der = try Self.selfSignedCertificateDER(
                publicKeyX963: publicX963, signWith: secKey,
                commonName: commonName)
        } catch {
            removeKey()
            throw error
        }
        guard let certificate = SecCertificateCreateWithData(nil, der as CFData) else {
            removeKey()
            throw IdentityError.certificateRejected
        }

        let addCert = SecItemAdd([
            kSecClass: kSecClassCertificate,
            kSecValueRef: certificate,
            kSecAttrLabel: label,
            kSecUseDataProtectionKeychain: false,
        ] as CFDictionary, nil)
        guard addCert == errSecSuccess else {
            removeKey()
            throw IdentityError.keychainAdd(addCert)
        }

        // The keychain pairs key and cert by public-key hash; fish the
        // resulting identity out and verify it is OUR cert (another
        // identity could share the label namespace, never the DER).
        var found: CFTypeRef?
        let status = SecItemCopyMatching([
            kSecClass: kSecClassIdentity,
            kSecMatchLimit: kSecMatchLimitAll,
            kSecReturnRef: true,
        ] as CFDictionary, &found)
        var staged: SecIdentity?
        if status == errSecSuccess, let list = found as? [SecIdentity] {
            for candidate in list {
                var candidateCert: SecCertificate?
                guard SecIdentityCopyCertificate(candidate, &candidateCert)
                        == errSecSuccess,
                      let candidateCert else { continue }
                if SecCertificateCopyData(candidateCert) as Data == der {
                    staged = candidate
                    break
                }
            }
        }
        guard let staged else {
            SecItemDelete([kSecClass: kSecClassCertificate,
                           kSecValueRef: certificate] as CFDictionary)
            removeKey()
            throw IdentityError.identityNotFound(status)
        }

        self.identity = staged
        self.certificateDER = der
        self.certificate = certificate
        self.privateKey = secKey
        self.keyTag = tag
    }

    deinit { cleanup() }

    /// Removes the staged cert and key from the keychain. Idempotent.
    public func cleanup() {
        lock.lock()
        defer { lock.unlock() }
        guard !cleaned else { return }
        cleaned = true
        SecItemDelete([kSecClass: kSecClassCertificate,
                       kSecValueRef: certificate] as CFDictionary)
        SecItemDelete([kSecClass: kSecClassKey,
                       kSecAttrApplicationTag: keyTag,
                       kSecUseDataProtectionKeychain: false] as CFDictionary)
    }

    // MARK: X.509 construction

    /// Minimal self-signed certificate:
    ///   Certificate ::= SEQUENCE { tbsCertificate, ecdsa-with-SHA256,
    ///                              BIT STRING signature }
    /// TBS carries v3, a random positive serial, issuer == subject
    /// (single CN RDN), 24 h validity around now, and the P-256 SPKI.
    /// No extensions — the client pins bytes, it never evaluates policy.
    static func selfSignedCertificateDER(publicKeyX963: Data,
                                         signWith key: SecKey,
                                         commonName: String) throws -> Data {
        // Pre-encoded OIDs (contents include their 0x06 tag).
        let oidECPublicKey: [UInt8] = [0x06, 0x07, 0x2A, 0x86, 0x48,
                                       0xCE, 0x3D, 0x02, 0x01]
        let oidPrime256v1: [UInt8] = [0x06, 0x08, 0x2A, 0x86, 0x48,
                                      0xCE, 0x3D, 0x03, 0x01, 0x07]
        let oidECDSAWithSHA256: [UInt8] = [0x06, 0x08, 0x2A, 0x86, 0x48,
                                           0xCE, 0x3D, 0x04, 0x03, 0x02]
        let oidCommonName: [UInt8] = [0x06, 0x03, 0x55, 0x04, 0x03]

        let signatureAlgorithm = derSequence(oidECDSAWithSHA256)

        // Name ::= SEQUENCE { SET { SEQUENCE { OID cn, UTF8String } } }
        let cnBytes = [UInt8](commonName.utf8)
        let name = derSequence(derWrap(0x31, derSequence(
            oidCommonName + derWrap(0x0C, cnBytes))))

        // Validity: UTCTime, backdated 1 h for clock skew, 24 h ahead.
        let now = Date()
        let validity = derSequence(
            derUTCTime(now.addingTimeInterval(-3600))
            + derUTCTime(now.addingTimeInterval(24 * 3600)))

        // SubjectPublicKeyInfo with the uncompressed point (04||X||Y).
        let spki = derSequence(
            derSequence(oidECPublicKey + oidPrime256v1)
            + derBitString([UInt8](publicKeyX963)))

        // Random positive serial (top bit cleared keeps it positive
        // without a leading-zero byte).
        var serial = [UInt8](repeating: 0, count: 8)
        for index in serial.indices {
            serial[index] = UInt8.random(in: 0...255)
        }
        serial[0] &= 0x7F
        if serial[0] == 0 { serial[0] = 1 }

        let tbs = derSequence(
            derWrap(0xA0, [0x02, 0x01, 0x02])       // [0] EXPLICIT v3
            + derWrap(0x02, serial)                  // serialNumber
            + signatureAlgorithm                     // signature
            + name                                   // issuer
            + validity
            + name                                   // subject (= issuer)
            + spki)

        // X9.62 message signing = SHA-256 then DER-encoded ECDSA — the
        // exact signature form X.509 carries.
        var signError: Unmanaged<CFError>?
        guard let signature = SecKeyCreateSignature(
            key, .ecdsaSignatureMessageX962SHA256,
            Data(tbs) as CFData, &signError) as Data? else {
            let detail = signError.map { String(describing: $0.takeRetainedValue()) }
                ?? "unknown"
            throw IdentityError.signingFailed(detail)
        }

        return Data(derSequence(
            tbs + signatureAlgorithm + derBitString([UInt8](signature))))
    }

    // MARK: DER helpers

    static func derLength(_ count: Int) -> [UInt8] {
        if count < 0x80 { return [UInt8(count)] }
        if count <= 0xFF { return [0x81, UInt8(count)] }
        return [0x82, UInt8(count >> 8), UInt8(count & 0xFF)]
    }

    static func derWrap(_ tag: UInt8, _ content: [UInt8]) -> [UInt8] {
        [tag] + derLength(content.count) + content
    }

    static func derSequence(_ content: [UInt8]) -> [UInt8] {
        derWrap(0x30, content)
    }

    static func derBitString(_ content: [UInt8]) -> [UInt8] {
        derWrap(0x03, [0x00] + content) // 0 unused bits
    }

    static func derUTCTime(_ date: Date) -> [UInt8] {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyMMddHHmmss'Z'"
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return derWrap(0x17, [UInt8](formatter.string(from: date).utf8))
    }
}
#endif
