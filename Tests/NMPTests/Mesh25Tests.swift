//
//  Mesh25Tests.swift
//  NMP — Mesh 2.5
//
//  The fully-measured transport race: an ephemeral self-signed TLS
//  identity (hand-rolled X.509 over CryptoKit), and TCP+TLS 1.3 + QUIC
//  legs racing the NMP stack over real loopback sockets — no protocol
//  is modeled anymore.
//

import XCTest
import Network
@testable import NMP

#if os(macOS)

// MARK: - Ephemeral TLS identity

final class TLSIdentityTests: XCTestCase {

    func testStagesAnIdentityAndCleansUpAfterItself() throws {
        let identity = try NMPEphemeralTLSIdentity(commonName: "NMP Test CN")

        // The DER must be a certificate the Security framework accepts,
        // carrying the CN we asked for.
        let cert = try XCTUnwrap(SecCertificateCreateWithData(
            nil, identity.certificateDER as CFData))
        XCTAssertEqual(SecCertificateCopySubjectSummary(cert) as String?,
                       "NMP Test CN")

        // The staged identity must join OUR certificate to a private key.
        var identityCert: SecCertificate?
        XCTAssertEqual(SecIdentityCopyCertificate(identity.identity,
                                                  &identityCert),
                       errSecSuccess)
        XCTAssertEqual(SecCertificateCopyData(try XCTUnwrap(identityCert)) as Data,
                       identity.certificateDER)
        var key: SecKey?
        XCTAssertEqual(SecIdentityCopyPrivateKey(identity.identity, &key),
                       errSecSuccess)
        XCTAssertNotNil(key)

        // It must be usable where the race needs it.
        XCTAssertNotNil(sec_identity_create(identity.identity))

        identity.cleanup()
        identity.cleanup() // idempotent
    }

    func testEachIdentityIsUnique() throws {
        let first = try NMPEphemeralTLSIdentity()
        defer { first.cleanup() }
        let second = try NMPEphemeralTLSIdentity()
        defer { second.cleanup() }
        XCTAssertNotEqual(first.certificateDER, second.certificateDER,
                          "fresh key + random serial every time")
    }
}

// MARK: - The four-leg race

final class TransportRaceAllLegsTests: XCTestCase {

    /// One race, all four legs, every number a wall-clock measurement.
    /// This is the test that retires "QUIC stays modeled".
    func testRaceMeasuresNMPTCPTLSAndQUIC() throws {
        let plan = NMPTransportRace.Plan(roundTrips: 4, payloadBytes: 64 << 10)
        let result = try NMPTransportRace.runSync(plan: plan, timeout: 15)

        XCTAssertEqual(result.legs.map(\.name),
                       ["NMP", "TCP", "TCP+TLS 1.3", "QUIC"],
                       "all four legs must race — note: \(result.note)")
        for leg in result.legs {
            XCTAssertGreaterThan(leg.handshakeMs, 0, "\(leg.name) handshake")
            XCTAssertGreaterThan(leg.transferMs, 0, "\(leg.name) transfer")
            XCTAssertEqual(leg.roundTrips, 4, leg.name)
            XCTAssertEqual(leg.bytesMoved, plan.bytesPerDirection * 2 * 4,
                           "\(leg.name) must move the same bytes as every "
                           + "other leg")
        }
        XCTAssertFalse(result.note.contains("SKIPPED"),
                       "on a dev Mac nothing should be skipped: \(result.note)")
    }

    func testRaceJSONCarriesEveryLegAsMeasured() throws {
        let plan = NMPTransportRace.Plan(roundTrips: 2, payloadBytes: 8 << 10)
        let result = try NMPTransportRace.runSync(plan: plan, timeout: 15)
        let legs = try XCTUnwrap(result.asJSONObject["legs"] as? [[String: Any]])
        XCTAssertEqual(legs.count, result.legs.count)
        for leg in legs {
            XCTAssertEqual(leg["measured"] as? Bool, true,
                           "\(leg["name"] ?? "?") must be flagged measured")
        }
    }
}

// MARK: - Device display names

final class DeviceDisplayNameTests: XCTestCase {

    /// Hardware identifiers are a generation off the marketing names —
    /// "iPhone18,1" IS the iPhone 17 Pro. Known ids get the friendly
    /// name plus the id; unknown ids pass through verbatim (a wrong
    /// name is worse than a code).
    func testKnownIdentifiersGetMarketingNames() {
        XCTAssertEqual(
            NMPSystemCapabilityProbe.displayName(forHardwareIdentifier: "iPhone18,1"),
            "iPhone 17 Pro (iPhone18,1)")
        XCTAssertEqual(
            NMPSystemCapabilityProbe.displayName(forHardwareIdentifier: "iPhone17,1"),
            "iPhone 16 Pro (iPhone17,1)")
    }

    func testUnknownIdentifiersPassThroughUnchanged() {
        XCTAssertEqual(
            NMPSystemCapabilityProbe.displayName(forHardwareIdentifier: "iPhone99,9"),
            "iPhone99,9")
        XCTAssertEqual(
            NMPSystemCapabilityProbe.displayName(forHardwareIdentifier: "Mac16,13"),
            "Mac16,13")
    }
}

// MARK: - Inference attaches the measured race

final class InferenceRaceAttachTests: XCTestCase {

    private var server: NMPDashboardServer!

    override func setUpWithError() throws {
        server = NMPDashboardServer()
        try server.start(port: 0)
    }

    override func tearDown() {
        server.stop()
        server = nil
    }

    /// "Compare protocols" on an inference must come back with the
    /// MEASURED race on the generation's traffic pattern — the modeled
    /// protocol_comparison is gone from this path.
    func testEnableComparisonAttachesMeasuredRaceNotAModel() throws {
        server.onInferenceRequest = { request, respond in
            respond(.success(NMPPromptInferenceService.GenerationResult(
                text: "raced output", tokenCount: 4, totalSeconds: 0.4,
                networkPayloadBytes: 16_000, shardCount: 1,
                perTokenSeconds: Array(repeating: 0.1, count: 4),
                engine: "test")))
        }

        var request = URLRequest(url: URL(
            string: "http://127.0.0.1:\(server.boundPort)/api/inference")!)
        request.httpMethod = "POST"
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "prompt": "race me", "max_tokens": 4, "enable_comparison": true,
        ])
        let done = expectation(description: "inference + race")
        var payload: Data?
        URLSession.shared.dataTask(with: request) { data, response, _ in
            XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
            payload = data
            done.fulfill()
        }.resume()
        wait(for: [done], timeout: 30)

        let object = try XCTUnwrap(JSONSerialization.jsonObject(
            with: try XCTUnwrap(payload)) as? [String: Any])
        XCTAssertNil(object["protocol_comparison"],
                     "the modeled comparison must not ride along anymore")
        let transportRace = try XCTUnwrap(
            object["transport_race"] as? [String: Any],
            "error: \(object["transport_race_error"] ?? "none")")
        let race = try XCTUnwrap(transportRace["race"] as? [String: Any])
        let legs = try XCTUnwrap(race["legs"] as? [[String: Any]])
        XCTAssertGreaterThanOrEqual(legs.count, 2)
        for leg in legs {
            XCTAssertEqual(leg["measured"] as? Bool, true)
        }
        // 4 round trips (one per pass) × the real payload.
        XCTAssertEqual(legs[0]["round_trips"] as? Int, 4)
        let projected = try XCTUnwrap(
            transportRace["projected"] as? [[String: Any]])
        XCTAssertEqual(projected.count, legs.count)
        XCTAssertEqual(projected[0]["basis"] as? String, "measured")
    }
}

// MARK: - Splice projections

final class SpliceProjectionTests: XCTestCase {

    /// The splice is arithmetic on measurements: generation wall clock
    /// with NMP's transport time swapped for each leg's.
    func testProjectionSplicesEveryNonNMPLeg() throws {
        let plan = NMPTransportRace.Plan(roundTrips: 2, payloadBytes: 4 << 10)
        let race = try NMPTransportRace.runSync(plan: plan, timeout: 15)

        let generationMs = 1000.0
        let projected = NMPDashboardServer.projectedJSON(
            race: race, generationMs: generationMs, tokenCount: 10)

        XCTAssertEqual(projected.count, race.legs.count,
                       "the run itself + one splice per other leg")
        XCTAssertEqual(projected[0]["basis"] as? String, "measured")
        for (leg, row) in zip(race.legs.dropFirst(), projected.dropFirst()) {
            let expected = generationMs - race.nmp.totalMs + leg.totalMs
            let total = try XCTUnwrap(row["total_ms"] as? Double)
            XCTAssertEqual(total, (expected * 100).rounded() / 100,
                           accuracy: 0.01, leg.name)
            XCTAssertEqual(row["basis"] as? String, "measured splice")
        }
    }
}

#endif
