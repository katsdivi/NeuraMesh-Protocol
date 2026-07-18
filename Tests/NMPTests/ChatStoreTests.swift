//
//  ChatStoreTests.swift
//  NMPTests — persistent chat history
//
//  The local chat store: create/load/list/delete round trips, title
//  derivation, LZFSE compression of inactive conversations (and that a
//  packed conversation still loads and re-activates on a new turn), and a
//  smaller on-disk footprint after packing.
//

import XCTest
@testable import NMP

final class ChatStoreTests: XCTestCase {

    private var dir: URL!
    private var store: NMPChatStore!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("nmp-chat-\(UUID().uuidString)")
        store = NMPChatStore(directory: dir, deviceName: "TestBox")
    }

    override func tearDownWithError() throws {
        store.stopSweeping()
        try? FileManager.default.removeItem(at: dir)
    }

    // Small sync helpers over the async callback API.
    private func save(id: String = "", title: String = "",
                      model: String = "qwen2.5-0.5b",
                      messages: [NMPChatMessage]) -> NMPChatStore.Summary? {
        let exp = expectation(description: "save")
        var out: NMPChatStore.Summary?
        store.save(id: id, title: title, model: model, messages: messages) {
            out = $0; exp.fulfill()
        }
        wait(for: [exp], timeout: 5)
        return out
    }

    private func list() -> [NMPChatStore.Summary] {
        let exp = expectation(description: "list")
        var out: [NMPChatStore.Summary] = []
        store.list { out = $0; exp.fulfill() }
        wait(for: [exp], timeout: 5)
        return out
    }

    private func load(_ id: String) -> NMPChatStore.Conversation? {
        let exp = expectation(description: "load")
        var out: NMPChatStore.Conversation?
        store.load(id: id) { out = $0; exp.fulfill() }
        wait(for: [exp], timeout: 5)
        return out
    }

    private func turns(_ pairs: (NMPChatMessage.Role, String)...) -> [NMPChatMessage] {
        pairs.map { NMPChatMessage(role: $0.0, content: $0.1) }
    }

    func testSaveAssignsIdAndDerivesTitle() throws {
        let summary = try XCTUnwrap(save(messages: turns(
            (.user, "What is a mesh network?"),
            (.assistant, "A network of interconnected nodes."))))
        XCTAssertFalse(summary.id.isEmpty)
        XCTAssertEqual(summary.title, "What is a mesh network?")
        XCTAssertEqual(summary.device, "TestBox")
        XCTAssertEqual(summary.messageCount, 2)
        XCTAssertFalse(summary.compressed)
    }

    func testSaveThenLoadRoundTrips() throws {
        let msgs = turns((.user, "hi"), (.assistant, "hello"))
        let summary = try XCTUnwrap(save(messages: msgs))
        let loaded = try XCTUnwrap(load(summary.id))
        XCTAssertEqual(loaded.messages, msgs)
        XCTAssertEqual(loaded.id, summary.id)
    }

    func testUpdatePreservesIdAndCreatedAt() throws {
        let first = try XCTUnwrap(save(messages: turns((.user, "one"))))
        let created = first.createdAt
        Thread.sleep(forTimeInterval: 0.01)
        let second = try XCTUnwrap(save(
            id: first.id,
            messages: turns((.user, "one"), (.assistant, "1"), (.user, "two"))))
        XCTAssertEqual(second.id, first.id)
        // createdAt is preserved across the update. Compare with tolerance:
        // the index persists times as a 1970-epoch Double, whose round trip
        // loses sub-microsecond precision vs. the original Date.
        XCTAssertEqual(second.createdAt.timeIntervalSince1970,
                       created.timeIntervalSince1970, accuracy: 0.001)
        XCTAssertGreaterThanOrEqual(second.updatedAt, created)
        XCTAssertEqual(second.messageCount, 3)
        XCTAssertEqual(list().count, 1)                    // not a new row
    }

    func testListNewestFirst() throws {
        let a = try XCTUnwrap(save(messages: turns((.user, "alpha"))))
        Thread.sleep(forTimeInterval: 0.01)
        let b = try XCTUnwrap(save(messages: turns((.user, "beta"))))
        let ids = list().map(\.id)
        XCTAssertEqual(ids, [b.id, a.id])
    }

    func testDeleteRemovesConversationAndRow() throws {
        let s = try XCTUnwrap(save(messages: turns((.user, "bye"))))
        let exp = expectation(description: "delete")
        var existed = false
        store.delete(id: s.id) { existed = $0; exp.fulfill() }
        wait(for: [exp], timeout: 5)
        XCTAssertTrue(existed)
        XCTAssertNil(load(s.id))
        XCTAssertTrue(list().isEmpty)
    }

    func testCompressInactiveShrinksDiskAndStillLoads() throws {
        // A chatty conversation so LZFSE has something to chew on.
        var msgs: [NMPChatMessage] = []
        for i in 0..<40 {
            msgs.append(NMPChatMessage(role: .user, content:
                "Tell me fact number \(i) about distributed systems please."))
            msgs.append(NMPChatMessage(role: .assistant, content:
                "Fact \(i): consensus under partition is bounded by CAP; "
                + "quorums trade availability for consistency in the usual way."))
        }
        let s = try XCTUnwrap(save(messages: msgs))
        let activePath = dir.appendingPathComponent("\(s.id).json")
        let packedPath = dir.appendingPathComponent("\(s.id).json.lzfse")

        let activeSize = try FileManager.default
            .attributesOfItem(atPath: activePath.path)[.size] as? Int ?? 0
        XCTAssertGreaterThan(activeSize, 0)

        // Everything is "inactive" with a zero threshold.
        store.inactivityThreshold = 0
        let packed = store.compressInactive(now: Date().addingTimeInterval(10))
        XCTAssertEqual(packed, 1)

        XCTAssertFalse(FileManager.default.fileExists(atPath: activePath.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: packedPath.path))
        let packedSize = try FileManager.default
            .attributesOfItem(atPath: packedPath.path)[.size] as? Int ?? 0
        XCTAssertLessThan(packedSize, activeSize)          // it actually saved space

        // List marks it compressed; load transparently inflates.
        XCTAssertTrue(try XCTUnwrap(list().first).compressed)
        let loaded = try XCTUnwrap(load(s.id))
        XCTAssertEqual(loaded.messages, msgs)
    }

    func testNewTurnReactivatesPackedConversation() throws {
        let s = try XCTUnwrap(save(messages: turns((.user, "hello there mesh"))))
        store.inactivityThreshold = 0
        XCTAssertEqual(store.compressInactive(now: Date().addingTimeInterval(10)), 1)
        XCTAssertTrue(try XCTUnwrap(list().first).compressed)

        // A new turn writes the active JSON and clears the packed copy.
        let updated = try XCTUnwrap(save(
            id: s.id,
            messages: turns((.user, "hello there mesh"), (.assistant, "hi"))))
        XCTAssertFalse(updated.compressed)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("\(s.id).json").path))
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("\(s.id).json.lzfse").path))
    }

    func testStorePersistsAcrossReopen() throws {
        let s = try XCTUnwrap(save(messages: turns((.user, "durable?"))))
        // A fresh store over the same directory sees the prior conversation.
        let reopened = NMPChatStore(directory: dir, deviceName: "TestBox")
        let exp = expectation(description: "reopen-load")
        var loaded: NMPChatStore.Conversation?
        reopened.load(id: s.id) { loaded = $0; exp.fulfill() }
        wait(for: [exp], timeout: 5)
        XCTAssertEqual(loaded?.messages, turns((.user, "durable?")))
    }
}
