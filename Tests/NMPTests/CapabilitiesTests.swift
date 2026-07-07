//
//  CapabilitiesTests.swift
//  NMPTests — Phase 4
//
//  Capability struct: binary wire format round-trips, forward
//  compatibility (trailing bytes ignored), malformed-input rejection,
//  compute class ordering, and the Bonjour TXT dictionary form.
//

import XCTest
@testable import NMP

final class CapabilitiesTests: XCTestCase {

    private let sample = NMPCapabilities(
        peerID: 0xa1b2_c3d4,
        deviceName: "MacBook Pro M3",
        ramMB: 18_432,
        computeClass: .high,
        currentLoadPercent: 15,
        maxInferenceTokensPerSecond: 42.5,
        modelFormats: ["gguf", "safetensors"]
    )

    // MARK: Binary wire format

    func testCapabilitiesEncodeDecode() throws {
        let encoded = try sample.encode()
        let decoded = try NMPCapabilities.decode(encoded)
        XCTAssertEqual(decoded, sample)
        // Byte-exact: re-encoding the decoded struct reproduces the wire bytes.
        XCTAssertEqual(try decoded.encode(), encoded)
    }

    func testEncodeDecodeEmptyOptionalFields() throws {
        let minimal = NMPCapabilities(
            peerID: 1, deviceName: "", ramMB: 0, computeClass: .low)
        let decoded = try NMPCapabilities.decode(try minimal.encode())
        XCTAssertEqual(decoded, minimal)
        XCTAssertTrue(decoded.modelFormats.isEmpty)
    }

    func testDecodeIgnoresTrailingBytes() throws {
        // Forward compatibility: a future revision appends fields after the
        // v1 layout; a v1 decoder must parse the prefix and ignore the rest.
        var encoded = try sample.encode()
        encoded.append(contentsOf: [0xDE, 0xAD, 0xBE, 0xEF, 0x42])
        let decoded = try NMPCapabilities.decode(encoded)
        XCTAssertEqual(decoded, sample)
    }

    func testDecodeRejectsTruncated() throws {
        let encoded = try sample.encode()
        for length in [0, 1, NMPCapabilities.fixedPrefixByteCount - 1, encoded.count - 1] {
            XCTAssertThrowsError(try NMPCapabilities.decode(encoded.prefix(length))) {
                guard case NMPCapabilitiesError.truncated = $0 else {
                    return XCTFail("expected .truncated for length \(length), got \($0)")
                }
            }
        }
    }

    func testDecodeRejectsUnsupportedVersion() throws {
        var encoded = try sample.encode()
        encoded[0] = 99
        XCTAssertThrowsError(try NMPCapabilities.decode(encoded)) {
            XCTAssertEqual($0 as? NMPCapabilitiesError, .unsupportedVersion(99))
        }
    }

    func testDecodeRejectsUnknownComputeClass() throws {
        var encoded = try sample.encode()
        encoded[9] = 7
        XCTAssertThrowsError(try NMPCapabilities.decode(encoded)) {
            XCTAssertEqual($0 as? NMPCapabilitiesError, .unknownComputeClass(7))
        }
    }

    func testEncodeClampsLoadPercent() throws {
        var over = sample
        over.currentLoadPercent = 350
        let decoded = try NMPCapabilities.decode(try over.encode())
        XCTAssertEqual(decoded.currentLoadPercent, 100)

        var negative = sample
        negative.currentLoadPercent = -12
        XCTAssertEqual(
            try NMPCapabilities.decode(try negative.encode()).currentLoadPercent, 0)
    }

    func testEncodeRejectsOverlongDeviceName() {
        var oversized = sample
        oversized.deviceName = String(repeating: "x", count: 256)
        XCTAssertThrowsError(try oversized.encode()) {
            XCTAssertEqual($0 as? NMPCapabilitiesError, .stringTooLong(256))
        }
    }

    func testDecodeFromSliceWithNonZeroStartIndex() throws {
        // Regression guard: decode must rebase Data slices (same class of
        // bug the packet codec defends against).
        var padded = Data([0xFF, 0xFF, 0xFF])
        try padded.append(sample.encode())
        let slice = padded.dropFirst(3)
        XCTAssertEqual(try NMPCapabilities.decode(slice), sample)
    }

    // MARK: v2 reachability fields (Phase 5)

    func testV2ReachabilityFieldsRoundTrip() throws {
        var reachable = sample
        reachable.udpPort = 51_820
        reachable.noiseStaticPublicKey = Data((0..<32).map { UInt8($0) })

        // Binary round trip.
        let decoded = try NMPCapabilities.decode(try reachable.encode())
        XCTAssertEqual(decoded, reachable)

        // TXT round trip.
        let dict = reachable.txtDictionary()
        XCTAssertEqual(dict["port"], "51820")
        XCTAssertEqual(NMPCapabilities(txtDictionary: dict), reachable)
    }

    func testV1BlobDecodesWithDefaultReachability() throws {
        // A Phase 4 (v1) advertisement: same layout, no port/pk tail.
        var v1Blob = try sample.encode()
        v1Blob = v1Blob.dropLast(3) // strip v2 tail: port(2) + pkLen(1)=0
        v1Blob[0] = 1
        let decoded = try NMPCapabilities.decode(v1Blob)
        XCTAssertEqual(decoded.udpPort, 0)
        XCTAssertNil(decoded.noiseStaticPublicKey)
        XCTAssertEqual(decoded.peerID, sample.peerID)
        XCTAssertEqual(decoded.modelFormats, sample.modelFormats)
    }

    func testEncodeAndDecodeRejectBadKeyLength() throws {
        var bad = sample
        bad.noiseStaticPublicKey = Data([1, 2, 3])
        XCTAssertThrowsError(try bad.encode()) {
            XCTAssertEqual($0 as? NMPCapabilitiesError, .invalidPublicKeyLength(3))
        }

        // TXT parse drops a malformed pk instead of rejecting the peer.
        var dict = sample.txtDictionary()
        dict["pk"] = Data([1, 2, 3]).base64EncodedString()
        XCTAssertNil(NMPCapabilities(txtDictionary: dict)?.noiseStaticPublicKey)
    }

    // MARK: Compute class

    func testComputeClassOrdering() {
        XCTAssertLessThan(NMPComputeClass.low, .medium)
        XCTAssertLessThan(NMPComputeClass.medium, .high)
        XCTAssertEqual([NMPComputeClass.high, .low, .medium].sorted(), [.low, .medium, .high])
    }

    func testComputeClassPriority() {
        XCTAssertEqual(NMPComputeClass.low.priority, 0)
        XCTAssertEqual(NMPComputeClass.medium.priority, 1)
        XCTAssertEqual(NMPComputeClass.high.priority, 2)
    }

    func testComputeClassLabelRoundTrip() {
        for tier in NMPComputeClass.allCases {
            XCTAssertEqual(NMPComputeClass(label: tier.label), tier)
        }
        XCTAssertNil(NMPComputeClass(label: "turbo"))
    }

    // MARK: TXT dictionary

    func testTXTDictionaryRoundTrip() {
        let dict = sample.txtDictionary()
        let parsed = NMPCapabilities(txtDictionary: dict)
        XCTAssertEqual(parsed, sample)
    }

    func testTXTParseRequiresIDAndCompute() {
        var dict = sample.txtDictionary()
        dict.removeValue(forKey: "id")
        XCTAssertNil(NMPCapabilities(txtDictionary: dict))

        dict = sample.txtDictionary()
        dict.removeValue(forKey: "compute")
        XCTAssertNil(NMPCapabilities(txtDictionary: dict))

        dict = sample.txtDictionary()
        dict["compute"] = "warp9"
        XCTAssertNil(NMPCapabilities(txtDictionary: dict))
    }

    func testTXTParseDefaultsOptionalKeysAndIgnoresUnknown() {
        let parsed = NMPCapabilities(txtDictionary: [
            "id": "2a", "compute": "medium", "futureKey": "ignored",
        ])
        XCTAssertEqual(parsed?.peerID, 0x2a)
        XCTAssertEqual(parsed?.computeClass, .medium)
        XCTAssertEqual(parsed?.deviceName, "")
        XCTAssertEqual(parsed?.ramMB, 0)
        XCTAssertEqual(parsed?.modelFormats, [])
    }

    // MARK: Local probe

    func testSystemProbeProducesPlausibleCapabilities() {
        let measured = NMPSystemCapabilityProbe.measure(peerID: 0x1234)
        XCTAssertEqual(measured.peerID, 0x1234)
        XCTAssertFalse(measured.deviceName.isEmpty)
        XCTAssertGreaterThan(measured.ramMB, 0)
        XCTAssertTrue((0...100).contains(measured.currentLoadPercent))
        // Whatever the hardware, the result must survive the wire.
        XCTAssertNoThrow(try measured.encode())
    }
}
