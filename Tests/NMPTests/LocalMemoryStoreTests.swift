//
//  LocalMemoryStoreTests.swift
//  NMPTests — on-device memory store
//
//  Round-trips for the fully on-device NMPMemoryStore backend: add/getDocument
//  by customID (content + metadata + status "done"), .notFound for unknown ids,
//  minted ids for nil customIDs, semantic/lexical search ranking, container-tag
//  isolation, on-disk durability across store instances, and exact metadata
//  round-tripping. Search assertions are lexically distinct enough to rank the
//  same way under every embedding backend (sentence / word-avg / lexical), so
//  they are deterministic regardless of which is active on the host.
//

import XCTest
@testable import NMP

final class LocalMemoryStoreTests: XCTestCase {

    private var dir: URL!
    private var store: NMPLocalMemoryStore!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("nmp-memstore-\(UUID().uuidString)")
        store = NMPLocalMemoryStore(directory: dir)
    }

    override func tearDownWithError() throws {
        store = nil
        try? FileManager.default.removeItem(at: dir)
    }

    // MARK: Sync helpers over the async-callback API

    @discardableResult
    private func add(_ store: NMPLocalMemoryStore, content: String,
                     customID: String?, tag: String = "chat",
                     metadata: [String: String] = [:]) -> Result<String, NMPMemoryStoreError> {
        let exp = expectation(description: "add")
        var out: Result<String, NMPMemoryStoreError>!
        store.addDocument(content: content, customID: customID,
                          containerTag: tag, metadata: metadata) {
            out = $0; exp.fulfill()
        }
        wait(for: [exp], timeout: 5)
        return out
    }

    private func get(_ store: NMPLocalMemoryStore,
                     customID: String) -> Result<NMPStoredDocument, NMPMemoryStoreError> {
        let exp = expectation(description: "get")
        var out: Result<NMPStoredDocument, NMPMemoryStoreError>!
        store.getDocument(customID: customID) { out = $0; exp.fulfill() }
        wait(for: [exp], timeout: 5)
        return out
    }

    private func search(_ store: NMPLocalMemoryStore, query: String,
                        tag: String = "chat", limit: Int = 10) -> [NMPStoredHit] {
        let exp = expectation(description: "search")
        var out: [NMPStoredHit] = []
        store.search(query: query, containerTag: tag, limit: limit) {
            if case let .success(hits) = $0 { out = hits }
            exp.fulfill()
        }
        wait(for: [exp], timeout: 5)
        return out
    }

    // MARK: Tests

    func testKindAndLocality() {
        XCTAssertEqual(store.kind, "on-device")
        XCTAssertTrue(store.localityDescription.hasPrefix("on-device"),
                      "unexpected locality: \(store.localityDescription)")
    }

    func testHealthSucceedsOnWritableDirectory() {
        let exp = expectation(description: "health")
        store.health {
            if case .failure(let e) = $0 { XCTFail("health failed: \(e)") }
            exp.fulfill()
        }
        wait(for: [exp], timeout: 5)
    }

    func testAddThenGetRoundTrips() {
        let meta = ["source": "unit-test", "kind": "index"]
        let addResult = add(store, content: "hello memory mesh",
                            customID: "nmp-index-abc", metadata: meta)
        guard case let .success(id) = addResult else {
            return XCTFail("add failed: \(addResult)")
        }
        XCTAssertEqual(id, "nmp-index-abc")

        guard case let .success(doc) = get(store, customID: "nmp-index-abc") else {
            return XCTFail("get failed")
        }
        XCTAssertEqual(doc.content, "hello memory mesh")
        XCTAssertEqual(doc.customID, "nmp-index-abc")
        XCTAssertEqual(doc.documentID, "nmp-index-abc")
        XCTAssertEqual(doc.status, "done")
        XCTAssertEqual(doc.metadata, meta)
    }

    func testGetMissingReturnsNotFound() {
        guard case let .failure(error) = get(store, customID: "does-not-exist") else {
            return XCTFail("expected .notFound")
        }
        guard case .notFound = error else {
            return XCTFail("expected .notFound, got \(error)")
        }
    }

    func testAddWithNilCustomIDMintsFetchableID() {
        let result = add(store, content: "no id given", customID: nil)
        guard case let .success(id) = result else {
            return XCTFail("add failed: \(result)")
        }
        XCTAssertFalse(id.isEmpty, "minted id must be non-empty")
        // A nil-customID doc has no customID key, so it is intentionally not
        // fetchable by customID — the mint only guarantees a returned id.
        guard case .failure(.notFound) = get(store, customID: id) else {
            return XCTFail("nil-customID doc should not be fetchable by customID")
        }
    }

    func testReStoreSameCustomIDOverwrites() {
        add(store, content: "first", customID: "nmp-shard-x-0")
        add(store, content: "second", customID: "nmp-shard-x-0")
        guard case let .success(doc) = get(store, customID: "nmp-shard-x-0") else {
            return XCTFail("get failed")
        }
        XCTAssertEqual(doc.content, "second")
        // Idempotent: still exactly one hit for that content in search.
        let hits = search(store, query: "second")
        XCTAssertEqual(hits.filter { $0.customID == "nmp-shard-x-0" }.count, 1)
    }

    func testSearchRanksClosestFirst() {
        add(store, content: "the wine cellar lock combination is 47-19-33",
            customID: "doc-wine")
        add(store, content: "quarterly revenue grew twelve percent",
            customID: "doc-revenue")
        add(store, content: "the dog needs a vet appointment",
            customID: "doc-dog")

        let hits = search(store, query: "where is the wine cellar code")
        XCTAssertFalse(hits.isEmpty, "expected hits")
        XCTAssertEqual(hits.first?.customID, "doc-wine",
                       "wine doc should rank first; got \(hits.map { $0.customID ?? "?" })")
        XCTAssertGreaterThan(hits.first?.score ?? 0, 0)
        // Scores are cosine in 0...1.
        for hit in hits {
            XCTAssertGreaterThanOrEqual(hit.score, 0)
            XCTAssertLessThanOrEqual(hit.score, 1)
        }
    }

    func testSearchRespectsLimit() {
        add(store, content: "the wine cellar lock combination is 47-19-33",
            customID: "doc-wine")
        add(store, content: "quarterly revenue grew twelve percent",
            customID: "doc-revenue")
        add(store, content: "the dog needs a vet appointment",
            customID: "doc-dog")
        XCTAssertEqual(search(store, query: "wine", limit: 1).count, 1)
    }

    func testEmptyQueryAndEmptyStoreReturnEmpty() {
        XCTAssertTrue(search(store, query: "anything").isEmpty,
                      "empty store should return no hits")
        add(store, content: "some content", customID: "doc-1")
        XCTAssertTrue(search(store, query: "   ").isEmpty,
                      "blank query should return no hits")
    }

    func testContainerTagIsolation() {
        add(store, content: "the wine cellar lock combination is 47-19-33",
            customID: "a-wine", tag: "tagA")
        add(store, content: "the wine cellar lock combination is 47-19-33",
            customID: "b-wine", tag: "tagB")

        let hitsB = search(store, query: "wine cellar", tag: "tagB")
        XCTAssertFalse(hitsB.isEmpty)
        XCTAssertTrue(hitsB.allSatisfy { $0.customID == "b-wine" },
                      "tagB search returned a non-tagB doc: \(hitsB.map { $0.customID ?? "?" })")
        XCTAssertFalse(hitsB.contains { $0.customID == "a-wine" },
                       "a doc in tagA must never surface in a tagB search")
    }

    func testPersistenceAcrossStoreInstances() {
        let meta = ["kind": "shard", "index": "3"]
        add(store, content: "durable base64 payload here",
            customID: "nmp-shard-mid-3", metadata: meta)
        // Drop the reference; a brand-new store on the SAME directory must
        // re-serve the document from disk.
        store = nil
        let reopened = NMPLocalMemoryStore(directory: dir)
        guard case let .success(doc) = get(reopened, customID: "nmp-shard-mid-3") else {
            return XCTFail("reopened store did not find the persisted doc")
        }
        XCTAssertEqual(doc.content, "durable base64 payload here")
        XCTAssertEqual(doc.metadata, meta)
        XCTAssertEqual(doc.status, "done")
    }

    func testMetadataMultipleKeysRoundTrip() {
        let meta = [
            "source": "peer-42",
            "container": "nmp-chat",
            "author": "divyam",
            "sha": "deadbeefcafef00d",
        ]
        add(store, content: "payload", customID: "doc-meta", metadata: meta)
        guard case let .success(doc) = get(store, customID: "doc-meta") else {
            return XCTFail("get failed")
        }
        XCTAssertEqual(doc.metadata, meta)
    }
}
