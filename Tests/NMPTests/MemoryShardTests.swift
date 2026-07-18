//
//  MemoryShardTests.swift
//  NMPTests — Memory mesh
//

import XCTest
@testable import NMP

final class MemoryShardTests: XCTestCase {

    // MARK: Helpers

    private func randomData(_ count: Int) -> Data {
        var data = Data(count: count)
        data.withUnsafeMutableBytes { buffer in
            guard let base = buffer.baseAddress else { return }
            var i = 0
            while i + 8 <= count {
                base.storeBytes(of: UInt64.random(in: .min ... .max),
                                toByteOffset: i, as: UInt64.self)
                i += 8
            }
            while i < count {
                base.storeBytes(of: UInt8.random(in: .min ... .max),
                                toByteOffset: i, as: UInt8.self)
                i += 1
            }
        }
        return data
    }

    private func scheme(_ k: Int, _ n: Int) throws -> NMPMemoryShardScheme {
        try NMPMemoryShardScheme(k: k, n: n)
    }

    // MARK: Scheme validation

    func testSchemeRejectsInvalidCombinations() {
        let bad: [(k: Int, n: Int)] = [
            (0, 1),    // k < 1
            (-1, 0),   // k < 1
            (2, 4),    // n != k+1 (would need Reed-Solomon)
            (3, 3),    // n != k+1
            (16, 17),  // n > 16
        ]
        for (k, n) in bad {
            XCTAssertThrowsError(try NMPMemoryShardScheme(k: k, n: n),
                                 "k=\(k) n=\(n)") { error in
                XCTAssertEqual(error as? NMPMemoryShardError, .invalidScheme(k: k, n: n))
            }
        }
    }

    func testSchemeAcceptsValidCombinations() throws {
        let s23 = try scheme(2, 3)
        XCTAssertEqual(s23.k, 2)
        XCTAssertEqual(s23.n, 3)
        let s34 = try scheme(3, 4)
        XCTAssertEqual(s34.k, 3)
        XCTAssertEqual(s34.n, 4)
        XCTAssertNoThrow(try NMPMemoryShardScheme(k: 15, n: 16))
    }

    // MARK: Encode / reconstruct with single-shard loss

    func testDropAnySingleShardReconstructs() throws {
        let sizes: (Int) -> [Int] = { k in [1, 1023, k * 512, 256 * 1024] }
        for (k, n) in [(2, 3), (3, 4)] {
            let s = try scheme(k, n)
            for size in sizes(k) {
                let blob = randomData(size)
                let records = NMPMemoryShardCodec.encode(
                    memoryID: "mem-\(k)-\(size)", blob: blob, scheme: s)
                XCTAssertEqual(records.count, n)
                XCTAssertEqual(records.map(\.shardIndex), Array(0...k))

                for dropped in 0..<n {
                    var survivors = records
                    survivors.remove(at: dropped)
                    survivors.shuffle() // order must not matter
                    let rebuilt = try NMPMemoryShardCodec.reconstruct(records: survivors)
                    XCTAssertEqual(rebuilt, blob,
                                   "k=\(k) size=\(size) dropped=\(dropped)")
                }
            }
        }
    }

    func testReconstructFromAllShards() throws {
        let blob = randomData(4096)
        let records = NMPMemoryShardCodec.encode(
            memoryID: "all", blob: blob, scheme: try scheme(3, 4))
        XCTAssertEqual(try NMPMemoryShardCodec.reconstruct(records: records), blob)
    }

    func testEmptyBlobRoundTrips() throws {
        let records = NMPMemoryShardCodec.encode(
            memoryID: "empty", blob: Data(), scheme: try scheme(2, 3))
        XCTAssertEqual(records.count, 3)
        XCTAssertTrue(records.allSatisfy { $0.payload.isEmpty })
        for dropped in 0..<3 {
            var survivors = records
            survivors.remove(at: dropped)
            XCTAssertEqual(try NMPMemoryShardCodec.reconstruct(records: survivors), Data())
        }
    }

    // MARK: Failure modes — explicit, never silent

    func testFewerThanKShardsThrowsInsufficient() throws {
        let records = NMPMemoryShardCodec.encode(
            memoryID: "short", blob: randomData(999), scheme: try scheme(3, 4))
        let two = Array(records.prefix(2)) // k-1 = 2 distinct shards
        XCTAssertThrowsError(try NMPMemoryShardCodec.reconstruct(records: two)) { error in
            XCTAssertEqual(error as? NMPMemoryShardError,
                           .insufficientShards(have: 2, needed: 3))
        }
        XCTAssertThrowsError(try NMPMemoryShardCodec.reconstruct(records: [])) { error in
            XCTAssertEqual(error as? NMPMemoryShardError,
                           .insufficientShards(have: 0, needed: 1))
        }
    }

    func testDuplicateShardIndexThrowsMismatch() throws {
        let records = NMPMemoryShardCodec.encode(
            memoryID: "dup", blob: randomData(100), scheme: try scheme(2, 3))
        // Three records, but two share index 0 — must NOT count as k distinct.
        XCTAssertThrowsError(
            try NMPMemoryShardCodec.reconstruct(records: [records[0], records[0], records[1]])
        ) { error in
            guard case .shardMismatch = error as? NMPMemoryShardError else {
                return XCTFail("expected shardMismatch, got \(error)")
            }
        }
    }

    func testMixedMemoryIDsThrowMismatch() throws {
        let s = try scheme(2, 3)
        let a = NMPMemoryShardCodec.encode(memoryID: "a", blob: randomData(100), scheme: s)
        let b = NMPMemoryShardCodec.encode(memoryID: "b", blob: randomData(100), scheme: s)
        XCTAssertThrowsError(
            try NMPMemoryShardCodec.reconstruct(records: [a[0], b[1], a[2]])
        ) { error in
            guard case .shardMismatch = error as? NMPMemoryShardError else {
                return XCTFail("expected shardMismatch, got \(error)")
            }
        }
    }

    func testMismatchedPayloadLengthThrowsMismatch() throws {
        let records = NMPMemoryShardCodec.encode(
            memoryID: "len", blob: randomData(100), scheme: try scheme(2, 3))
        let corrupt = NMPMemoryShardRecord(
            memoryID: "len", shardIndex: 1, k: 2, n: 3, blobLength: 100,
            payload: records[1].payload + Data([0x00]))
        XCTAssertThrowsError(
            try NMPMemoryShardCodec.reconstruct(records: [records[0], corrupt])
        ) { error in
            guard case .shardMismatch = error as? NMPMemoryShardError else {
                return XCTFail("expected shardMismatch, got \(error)")
            }
        }
    }

    // MARK: Record wire format

    func testRecordEncodeDecodeRoundTrip() throws {
        let records = NMPMemoryShardCodec.encode(
            memoryID: "round-trip ✓", blob: randomData(1023), scheme: try scheme(3, 4))
        for record in records {
            let decoded = try NMPMemoryShardRecord.decode(record.encode())
            XCTAssertEqual(decoded, record)
        }
    }

    func testTruncatedRecordThrows() throws {
        let wire = NMPMemoryShardCodec.encode(
            memoryID: "trunc", blob: randomData(64), scheme: try scheme(2, 3))[0].encode()
        for length in 0..<wire.count {
            XCTAssertThrowsError(
                try NMPMemoryShardRecord.decode(Data(wire.prefix(length))),
                "prefix length \(length)")
        }
    }

    func testWrongVersionThrows() throws {
        var wire = NMPMemoryShardCodec.encode(
            memoryID: "ver", blob: randomData(64), scheme: try scheme(2, 3))[0].encode()
        wire[0] = 9
        XCTAssertThrowsError(try NMPMemoryShardRecord.decode(wire)) { error in
            XCTAssertEqual(error as? NMPMemoryShardError, .unsupportedVersion(9))
        }
    }

    // MARK: Sealing

    func testSealOpenRoundTrip() throws {
        let plaintexts = [
            Data(),                                              // empty
            Data("hello mesh".utf8),
            Data(String(repeating: "conversational memory. ", count: 100_000).utf8), // ~2.3 MB
        ]
        for plaintext in plaintexts {
            let sealed = try NMPMemorySeal.seal(plaintext: plaintext)
            XCTAssertEqual(sealed.key.count, 32)
            let opened = try NMPMemorySeal.open(ciphertext: sealed.ciphertext,
                                                key: sealed.key)
            XCTAssertEqual(opened, plaintext, "size \(plaintext.count)")
        }
    }

    func testOpenWithWrongKeyThrows() throws {
        let sealed = try NMPMemorySeal.seal(plaintext: Data("secret".utf8))
        let wrongKey = try NMPMemorySeal.seal(plaintext: Data("other".utf8)).key
        XCTAssertThrowsError(
            try NMPMemorySeal.open(ciphertext: sealed.ciphertext, key: wrongKey)
        ) { error in
            guard case .openFailed = error as? NMPMemoryShardError else {
                return XCTFail("expected openFailed, got \(error)")
            }
        }
    }

    func testOpenWithFlippedCiphertextByteThrows() throws {
        let sealed = try NMPMemorySeal.seal(plaintext: randomData(500))
        var tampered = sealed.ciphertext
        tampered[tampered.count / 2] ^= 0xFF
        XCTAssertThrowsError(
            try NMPMemorySeal.open(ciphertext: tampered, key: sealed.key)
        ) { error in
            guard case .openFailed = error as? NMPMemoryShardError else {
                return XCTFail("expected openFailed, got \(error)")
            }
        }
    }

    func testCompressionShrinksRepetitiveText() throws {
        // 100 KB of highly repetitive text must seal far smaller than plaintext.
        let plaintext = Data(String(repeating: "the mesh remembers. ", count: 5120).utf8)
        XCTAssertGreaterThanOrEqual(plaintext.count, 100 * 1024)
        let sealed = try NMPMemorySeal.seal(plaintext: plaintext)
        XCTAssertLessThan(sealed.ciphertext.count, plaintext.count / 4,
                          "ciphertext \(sealed.ciphertext.count) B not much smaller than \(plaintext.count) B")
        XCTAssertEqual(try NMPMemorySeal.open(ciphertext: sealed.ciphertext,
                                              key: sealed.key), plaintext)
    }

    // MARK: End-to-end

    func testSealShardDropReconstructOpen() throws {
        let plaintext = Data(String(repeating: "what the peers said at dusk. ",
                                    count: 4000).utf8)
        let sealed = try NMPMemorySeal.seal(plaintext: plaintext)
        let records = NMPMemoryShardCodec.encode(
            memoryID: "e2e", blob: sealed.ciphertext, scheme: try scheme(2, 3))

        for dropped in 0..<3 {
            var survivors = records
            survivors.remove(at: dropped)
            // Wire round-trip each surviving shard, as the mesh would.
            let arrived = try survivors.map {
                try NMPMemoryShardRecord.decode($0.encode())
            }
            let ciphertext = try NMPMemoryShardCodec.reconstruct(records: arrived)
            let opened = try NMPMemorySeal.open(ciphertext: ciphertext, key: sealed.key)
            XCTAssertEqual(opened, plaintext, "dropped=\(dropped)")
        }
    }
}
