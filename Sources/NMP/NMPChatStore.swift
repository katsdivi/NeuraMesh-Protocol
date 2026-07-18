//
//  NMPChatStore.swift
//  NeuraMesh — persistent, space-thrifty chat history
//
//  A local-first store of chat conversations for the device that authored
//  them. Each conversation is one JSON file on disk; a small `index.json`
//  manifest holds the summaries the UI lists (so a list never has to open —
//  or inflate — every conversation). A conversation that has gone quiet is
//  rewritten from JSON to LZFSE on disk (chat text compresses ~3–4×); it is
//  transparently inflated on load and re-expanded when a new turn arrives.
//
//  Callback + serial-queue idiom (no async/await), Apple-native only:
//  Foundation's `NSData.compressed(using:)` is `libcompression`, so this
//  adds no dependency. Nothing here touches the mesh — a device stores only
//  its own chats. Cross-device browsing (Phase 2) reads this store over NMP
//  behind a pairing check; the `device` field and stable `id` are carried
//  now so those records stay meaningful across the wire later.
//

import Foundation

public final class NMPChatStore {

    // MARK: Records

    /// A full conversation as persisted and served.
    public struct Conversation: Equatable, Sendable {
        public var id: String
        public var title: String
        /// Human name of the device that owns this conversation (for the
        /// Phase 2 device switcher; today always this device).
        public var device: String
        /// Model the conversation was held with, best-effort (for display).
        public var model: String
        public var createdAt: Date
        public var updatedAt: Date
        public var messages: [NMPChatMessage]

        public init(id: String, title: String, device: String, model: String,
                    createdAt: Date, updatedAt: Date,
                    messages: [NMPChatMessage]) {
            self.id = id
            self.title = title
            self.device = device
            self.model = model
            self.createdAt = createdAt
            self.updatedAt = updatedAt
            self.messages = messages
        }
    }

    /// The lightweight row the sidebar lists — no message bodies.
    public struct Summary: Equatable, Sendable {
        public var id: String
        public var title: String
        public var device: String
        public var model: String
        public var createdAt: Date
        public var updatedAt: Date
        public var messageCount: Int
        /// True once the on-disk copy has been squeezed to LZFSE.
        public var compressed: Bool
    }

    // MARK: State

    private let directory: URL
    private let deviceName: String
    /// One serial queue owns all disk + manifest mutation, so callers never
    /// race the index against a concurrent save/delete/sweep.
    private let queue = DispatchQueue(label: "nmp.chat.store")
    private var sweepTimer: DispatchSourceTimer?

    /// Conversations untouched for longer than this get compressed by the
    /// sweep — the "session isn't active any more" signal.
    public var inactivityThreshold: TimeInterval = 300

    // MARK: Init

    /// `directory` is created if missing. `deviceName` stamps new
    /// conversations. Construction is cheap and synchronous; disk work all
    /// happens on `queue`.
    public init(directory: URL, deviceName: String) {
        self.directory = directory
        self.deviceName = deviceName
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
    }

    // MARK: Paths

    private func activeURL(_ id: String) -> URL {
        directory.appendingPathComponent("\(id).json")
    }
    private func packedURL(_ id: String) -> URL {
        directory.appendingPathComponent("\(id).json.lzfse")
    }
    private var indexURL: URL {
        directory.appendingPathComponent("index.json")
    }

    // MARK: Public API (completions fire on `queue`)

    /// Every conversation's summary, newest activity first.
    public func list(completion: @escaping ([Summary]) -> Void) {
        queue.async { [self] in
            let summaries = readIndex().values
                .sorted { $0.updatedAt > $1.updatedAt }
            completion(summaries)
        }
    }

    /// The full conversation, inflating the packed copy if needed. `nil`
    /// when the id is unknown or the file is unreadable.
    public func load(id: String, completion: @escaping (Conversation?) -> Void) {
        queue.async { [self] in
            completion(readConversation(id: id))
        }
    }

    /// Creates (empty `id`) or replaces a conversation. `title` is derived
    /// from the first user turn when blank. Returns the stored summary so
    /// the caller learns the generated id and canonical title. A save always
    /// lands the body as plain JSON — an active conversation is, by
    /// definition, active — and refreshes the manifest.
    public func save(id: String, title: String, model: String,
                     messages: [NMPChatMessage],
                     completion: @escaping (Summary?) -> Void) {
        queue.async { [self] in
            let now = Date()
            let cid = id.isEmpty ? Self.newID() : id
            var index = readIndex()
            let created = index[cid]?.createdAt ?? now
            let device = index[cid]?.device ?? deviceName
            let resolvedTitle = Self.resolveTitle(explicit: title,
                                                  messages: messages)
            let conversation = Conversation(
                id: cid, title: resolvedTitle, device: device, model: model,
                createdAt: created, updatedAt: now, messages: messages)

            guard writeActive(conversation) else {
                completion(nil)
                return
            }
            // A fresh turn re-activates a previously packed conversation.
            try? FileManager.default.removeItem(at: packedURL(cid))

            let summary = Summary(
                id: cid, title: resolvedTitle, device: device, model: model,
                createdAt: created, updatedAt: now,
                messageCount: messages.count, compressed: false)
            index[cid] = summary
            writeIndex(index)
            completion(summary)
        }
    }

    /// Removes a conversation (either on-disk form) and its manifest row.
    public func delete(id: String, completion: @escaping (Bool) -> Void) {
        queue.async { [self] in
            try? FileManager.default.removeItem(at: activeURL(id))
            try? FileManager.default.removeItem(at: packedURL(id))
            var index = readIndex()
            let existed = index.removeValue(forKey: id) != nil
            writeIndex(index)
            completion(existed)
        }
    }

    /// Compresses every conversation whose last activity is older than
    /// `inactivityThreshold`. Safe to call repeatedly; already-packed
    /// conversations are skipped. Returns how many it packed this pass.
    @discardableResult
    public func compressInactive(now: Date = Date()) -> Int {
        queue.sync { [self] in
            var index = readIndex()
            var packed = 0
            for (id, summary) in index where !summary.compressed {
                guard now.timeIntervalSince(summary.updatedAt)
                        > inactivityThreshold else { continue }
                if packConversation(id: id) {
                    index[id]?.compressed = true
                    packed += 1
                }
            }
            if packed > 0 { writeIndex(index) }
            return packed
        }
    }

    /// Starts a background sweep every `interval` seconds. No-op if already
    /// running. Stops with `stopSweeping()` / deinit.
    public func startSweeping(interval: TimeInterval = 120) {
        queue.async { [self] in
            guard sweepTimer == nil else { return }
            let timer = DispatchSource.makeTimerSource(queue: queue)
            timer.schedule(deadline: .now() + interval, repeating: interval)
            timer.setEventHandler { [weak self] in
                _ = self?.compressInactiveLocked(now: Date())
            }
            sweepTimer = timer
            timer.resume()
        }
    }

    public func stopSweeping() {
        queue.async { [self] in
            sweepTimer?.cancel()
            sweepTimer = nil
        }
    }

    deinit {
        sweepTimer?.cancel()
    }

    // MARK: Disk — conversations

    /// MUST run on `queue`.
    private func readConversation(id: String) -> Conversation? {
        let fm = FileManager.default
        if let data = try? Data(contentsOf: activeURL(id)) {
            return Self.decodeConversation(data)
        }
        if fm.fileExists(atPath: packedURL(id).path),
           let packed = try? Data(contentsOf: packedURL(id)),
           let inflated = try? (packed as NSData).decompressed(using: .lzfse) {
            return Self.decodeConversation(inflated as Data)
        }
        return nil
    }

    /// MUST run on `queue`. Writes the plain-JSON active form.
    private func writeActive(_ conversation: Conversation) -> Bool {
        guard let data = Self.encodeConversation(conversation) else {
            return false
        }
        do {
            try data.write(to: activeURL(conversation.id), options: .atomic)
            return true
        } catch {
            return false
        }
    }

    /// MUST run on `queue`. Rewrites a conversation's active JSON as LZFSE
    /// and drops the plain copy. Returns false (and leaves the active file)
    /// if anything goes wrong, so a conversation is never lost to a failed
    /// squeeze.
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

    /// The sweep timer already holds `queue`; call the compression body
    /// directly instead of re-entering via `queue.sync` (which would
    /// deadlock the serial queue on itself).
    @discardableResult
    private func compressInactiveLocked(now: Date) -> Int {
        var index = readIndex()
        var packed = 0
        for (id, summary) in index where !summary.compressed {
            guard now.timeIntervalSince(summary.updatedAt)
                    > inactivityThreshold else { continue }
            if packConversation(id: id) {
                index[id]?.compressed = true
                packed += 1
            }
        }
        if packed > 0 { writeIndex(index) }
        return packed
    }

    // MARK: Disk — index manifest

    /// MUST run on `queue`.
    private func readIndex() -> [String: Summary] {
        guard let data = try? Data(contentsOf: indexURL),
              let rows = try? JSONSerialization.jsonObject(with: data)
                as? [[String: Any]] else { return [:] }
        var index: [String: Summary] = [:]
        for row in rows {
            guard let id = row["id"] as? String,
                  let created = row["created_at"] as? Double,
                  let updated = row["updated_at"] as? Double else { continue }
            index[id] = Summary(
                id: id,
                title: (row["title"] as? String) ?? "Untitled",
                device: (row["device"] as? String) ?? deviceName,
                model: (row["model"] as? String) ?? "",
                createdAt: Date(timeIntervalSince1970: created),
                updatedAt: Date(timeIntervalSince1970: updated),
                messageCount: (row["message_count"] as? Int) ?? 0,
                compressed: (row["compressed"] as? Bool) ?? false)
        }
        return index
    }

    /// MUST run on `queue`.
    private func writeIndex(_ index: [String: Summary]) {
        let rows: [[String: Any]] = index.values
            .sorted { $0.updatedAt > $1.updatedAt }
            .map { s in
                [
                    "id": s.id,
                    "title": s.title,
                    "device": s.device,
                    "model": s.model,
                    "created_at": s.createdAt.timeIntervalSince1970,
                    "updated_at": s.updatedAt.timeIntervalSince1970,
                    "message_count": s.messageCount,
                    "compressed": s.compressed,
                ]
            }
        if let data = try? JSONSerialization.data(
            withJSONObject: rows, options: [.prettyPrinted]) {
            try? data.write(to: indexURL, options: .atomic)
        }
    }

    // MARK: Codec (conversation ⇄ JSON)

    static func encodeConversation(_ c: Conversation) -> Data? {
        let object: [String: Any] = [
            "id": c.id,
            "title": c.title,
            "device": c.device,
            "model": c.model,
            "created_at": c.createdAt.timeIntervalSince1970,
            "updated_at": c.updatedAt.timeIntervalSince1970,
            "messages": c.messages.map {
                ["role": $0.role.rawValue, "content": $0.content]
            },
        ]
        return try? JSONSerialization.data(withJSONObject: object)
    }

    static func decodeConversation(_ data: Data) -> Conversation? {
        guard let o = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any],
              let id = o["id"] as? String,
              let created = o["created_at"] as? Double,
              let updated = o["updated_at"] as? Double,
              let rawMessages = o["messages"] as? [[String: Any]]
        else { return nil }
        let messages: [NMPChatMessage] = rawMessages.compactMap { raw in
            guard let roleRaw = raw["role"] as? String,
                  let role = NMPChatMessage.Role(rawValue: roleRaw),
                  let content = raw["content"] as? String else { return nil }
            return NMPChatMessage(role: role, content: content)
        }
        return Conversation(
            id: id,
            title: (o["title"] as? String) ?? "Untitled",
            device: (o["device"] as? String) ?? "",
            model: (o["model"] as? String) ?? "",
            createdAt: Date(timeIntervalSince1970: created),
            updatedAt: Date(timeIntervalSince1970: updated),
            messages: messages)
    }

    // MARK: Helpers

    /// URL-safe, sortable-ish id: no filesystem-hostile characters.
    static func newID() -> String {
        UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    }

    /// Uses an explicit title when given; otherwise the first user turn,
    /// trimmed to a sane length; else a placeholder.
    static func resolveTitle(explicit: String,
                             messages: [NMPChatMessage]) -> String {
        let trimmed = explicit.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { return String(trimmed.prefix(80)) }
        if let firstUser = messages.first(where: { $0.role == .user }) {
            let line = firstUser.content
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: "\n", with: " ")
            if !line.isEmpty { return String(line.prefix(80)) }
        }
        return "New chat"
    }
}
