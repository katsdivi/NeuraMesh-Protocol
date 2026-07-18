//
//  ChatStore.swift
//  NeuraMeshPeer — local chat history
//
//  This phone owns the chats it authored. Each conversation is one JSON
//  file in Application Support; a small index.json manifest backs the
//  history list so it never has to open (or inflate) every file. A quiet
//  conversation is rewritten from JSON to LZFSE and transparently inflated
//  on load — the same scheme, and the same on-disk record schema
//  (snake_case fields, unix-seconds dates), as the Mac coordinator's
//  NMPChatStore, so a future mesh-shared view can read either over NMP.
//
//  Apple-native only: Foundation's NSData.compressed(using:) is
//  libcompression. An actor serializes all disk + manifest work off the
//  main actor.
//

import Foundation

// MARK: On-disk records (schema shared with the coordinator)

struct StoredMessage: Codable, Equatable {
    let role: String       // "user" | "assistant" | "system"
    let content: String
}

struct StoredConversation: Codable, Equatable {
    let id: String
    var title: String
    var device: String
    var model: String
    let created_at: Double
    var updated_at: Double
    var messages: [StoredMessage]
}

struct ChatRow: Codable, Equatable, Identifiable {
    let id: String
    var title: String
    var device: String
    var model: String
    var created_at: Double
    var updated_at: Double
    var message_count: Int
    var compressed: Bool

    var updatedDate: Date { Date(timeIntervalSince1970: updated_at) }
}

// MARK: Store

actor ChatStore {

    private let directory: URL
    private let deviceName: String
    /// Quiet-for-this-long ⇒ eligible for compression.
    private let inactivityThreshold: TimeInterval

    init(deviceName: String, inactivityThreshold: TimeInterval = 300) {
        self.deviceName = deviceName
        self.inactivityThreshold = inactivityThreshold
        let base = (try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true))
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        directory = base.appendingPathComponent("chats", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
    }

    // Paths
    private func activeURL(_ id: String) -> URL {
        directory.appendingPathComponent("\(id).json")
    }
    private func packedURL(_ id: String) -> URL {
        directory.appendingPathComponent("\(id).json.lzfse")
    }
    private var indexURL: URL { directory.appendingPathComponent("index.json") }

    // MARK: API

    /// Summaries, newest activity first.
    func list() -> [ChatRow] {
        readIndex().values.sorted { $0.updated_at > $1.updated_at }
    }

    /// Full conversation, inflating a packed copy. nil if unknown/unreadable.
    func load(id: String) -> StoredConversation? {
        readConversation(id: id)
    }

    /// Create (blank id) or replace; returns the stored row.
    @discardableResult
    func save(id: String, title: String, model: String,
              messages: [StoredMessage]) -> ChatRow? {
        let now = Date().timeIntervalSince1970
        let cid = id.isEmpty ? Self.newID() : id
        var index = readIndex()
        let created = index[cid]?.created_at ?? now
        let resolvedTitle = Self.resolveTitle(explicit: title, messages: messages)
        let conversation = StoredConversation(
            id: cid, title: resolvedTitle, device: deviceName, model: model,
            created_at: created, updated_at: now, messages: messages)
        guard writeActive(conversation) else { return nil }
        try? FileManager.default.removeItem(at: packedURL(cid))   // re-activate
        let row = ChatRow(
            id: cid, title: resolvedTitle, device: deviceName, model: model,
            created_at: created, updated_at: now,
            message_count: messages.count, compressed: false)
        index[cid] = row
        writeIndex(index)
        return row
    }

    func delete(id: String) {
        try? FileManager.default.removeItem(at: activeURL(id))
        try? FileManager.default.removeItem(at: packedURL(id))
        var index = readIndex()
        index.removeValue(forKey: id)
        writeIndex(index)
    }

    /// Compress conversations quiet longer than the threshold. Returns count.
    @discardableResult
    func compressInactive(now: Date = Date()) -> Int {
        var index = readIndex()
        var packed = 0
        for (id, row) in index where !row.compressed {
            guard now.timeIntervalSince1970 - row.updated_at
                    > inactivityThreshold else { continue }
            if packConversation(id: id) {
                index[id]?.compressed = true
                packed += 1
            }
        }
        if packed > 0 { writeIndex(index) }
        return packed
    }

    // MARK: Disk — conversations

    private func readConversation(id: String) -> StoredConversation? {
        let fm = FileManager.default
        if let data = try? Data(contentsOf: activeURL(id)) {
            return try? JSONDecoder().decode(StoredConversation.self, from: data)
        }
        if fm.fileExists(atPath: packedURL(id).path),
           let packed = try? Data(contentsOf: packedURL(id)),
           let inflated = try? (packed as NSData).decompressed(using: .lzfse) {
            return try? JSONDecoder().decode(
                StoredConversation.self, from: inflated as Data)
        }
        return nil
    }

    private func writeActive(_ c: StoredConversation) -> Bool {
        guard let data = try? JSONEncoder().encode(c) else { return false }
        do {
            try data.write(to: activeURL(c.id), options: .atomic)
            return true
        } catch { return false }
    }

    private func packConversation(id: String) -> Bool {
        guard let data = try? Data(contentsOf: activeURL(id)),
              let packed = try? (data as NSData).compressed(using: .lzfse)
        else { return false }
        do {
            try (packed as Data).write(to: packedURL(id), options: .atomic)
            try FileManager.default.removeItem(at: activeURL(id))
            return true
        } catch {
            try? FileManager.default.removeItem(at: packedURL(id))
            return false
        }
    }

    // MARK: Disk — index manifest

    private func readIndex() -> [String: ChatRow] {
        guard let data = try? Data(contentsOf: indexURL),
              let rows = try? JSONDecoder().decode([ChatRow].self, from: data)
        else { return [:] }
        return Dictionary(rows.map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
    }

    private func writeIndex(_ index: [String: ChatRow]) {
        let rows = index.values.sorted { $0.updated_at > $1.updated_at }
        if let data = try? JSONEncoder().encode(rows) {
            try? data.write(to: indexURL, options: .atomic)
        }
    }

    // MARK: Helpers

    private static func newID() -> String {
        UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    }

    private static func resolveTitle(explicit: String,
                                     messages: [StoredMessage]) -> String {
        let trimmed = explicit.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { return String(trimmed.prefix(80)) }
        if let firstUser = messages.first(where: { $0.role == "user" }) {
            let line = firstUser.content
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: "\n", with: " ")
            if !line.isEmpty { return String(line.prefix(80)) }
        }
        return "New chat"
    }
}
