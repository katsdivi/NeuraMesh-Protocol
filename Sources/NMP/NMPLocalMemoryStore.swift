//
//  NMPLocalMemoryStore.swift
//  NMP — Memory mesh
//
//  A fully ON-DEVICE `NMPMemoryStore`: the memory backend for a device that
//  cannot run the Node `supermemory-server` binary (an iPhone, or any host
//  without the sidecar). It provides the same two capabilities the mesh needs
//  from a memory store — opaque blob storage keyed by a deterministic id, and
//  semantic search over stored plaintext — using nothing but Apple frameworks
//  (Foundation + NaturalLanguage). Because it never opens a socket, it is
//  local-only BY CONSTRUCTION: there is no network path to a cloud endpoint,
//  which satisfies the mesh's no-cloud constraint even more strongly than the
//  Supermemory client's localhost guard.
//
//  Persistence: each document is one small JSON file on disk under `directory`
//  (documentID, customID, containerTag, content, metadata, status). A peer that
//  restarts re-loads its files and re-serves its shards. Modeled on the file-
//  store idiom in NMPChatStore.swift — a serial queue owns all disk + in-memory
//  mutation, writes are atomic, JSON goes through JSONSerialization, and every
//  completion fires on that queue (house rule: callbacks + serial queue, no
//  async/await).
//
//  Embeddings — a robust fallback chain so search ranks sensibly on ANY host:
//    1. NLEmbedding.sentenceEmbedding(.english): dense sentence vectors, cosine
//       ranked. localityDescription "on-device (NLEmbedding sentence)".
//    2. else NLEmbedding.wordEmbedding(.english): average the per-word vectors
//       (skip OOV), cosine over the average. "on-device (NLEmbedding word-avg)".
//    3. else a deterministic LEXICAL score: cosine over term-frequency (bag-of-
//       words) vectors of the tokenized text. "on-device (lexical)". This last
//       leg needs no model assets, so search always ranks — even on CI, or a
//       platform without NaturalLanguage.
//  Doc vectors are cached in memory keyed by documentID so repeated searches
//  don't re-embed (the cache entry is dropped when a doc is (re-)stored).
//

import Foundation
#if canImport(NaturalLanguage)
import NaturalLanguage
#endif

public final class NMPLocalMemoryStore: NMPMemoryStore {

    // MARK: Record

    /// One stored document, exactly as persisted and served.
    private struct Record {
        var documentID: String
        var customID: String?
        var containerTag: String
        var content: String
        var metadata: [String: String]
        var status: String
    }

    // MARK: State

    private let directory: URL
    /// One serial queue owns disk + in-memory mutation and fires every
    /// completion, so callers never race the record map or the embedding cache.
    private let queue: DispatchQueue
    private let embedder = Embedder()

    /// Loaded documents, keyed by documentID.
    private var records: [String: Record] = [:]
    /// Cached doc vectors, keyed by documentID; dropped on (re-)store.
    private var embeddingCache: [String: Vector] = [:]

    public let kind = "on-device"
    public var localityDescription: String { embedder.localityDescription }

    // MARK: Init

    /// `directory` is created if missing and its existing document files are
    /// loaded synchronously, so a store built on a directory a previous store
    /// wrote is immediately durable. All later disk work happens on `queue`.
    public init(directory: URL,
                queue: DispatchQueue = DispatchQueue(label: "nmp.local.memstore")) {
        self.directory = directory
        self.queue = queue
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        loadFromDisk()
    }

    // MARK: NMPMemoryStore

    public func health(completion: @escaping (Result<Void, NMPMemoryStoreError>) -> Void) {
        queue.async { [self] in
            let fm = FileManager.default
            var isDir: ObjCBool = false
            if !fm.fileExists(atPath: directory.path, isDirectory: &isDir) {
                try? fm.createDirectory(at: directory,
                                        withIntermediateDirectories: true)
            }
            if fm.isWritableFile(atPath: directory.path) {
                completion(.success(()))
            } else {
                completion(.failure(.unavailable(
                    "store directory not writable: \(directory.path)")))
            }
        }
    }

    public func addDocument(content: String, customID: String?,
                            containerTag: String, metadata: [String: String],
                            completion: @escaping (Result<String, NMPMemoryStoreError>) -> Void) {
        queue.async { [self] in
            // customID (when given) IS the documentID — a re-store under the
            // same customID overwrites idempotently. A nil customID mints a
            // fresh UUID documentID and stays customID-less (not fetchable by
            // customID, by design).
            let documentID = customID ?? Self.mintID()
            let record = Record(documentID: documentID, customID: customID,
                                containerTag: containerTag, content: content,
                                metadata: metadata, status: "done")
            records[documentID] = record
            embeddingCache[documentID] = nil  // recompute on next search
            guard write(record) else {
                completion(.failure(.backend(
                    "failed to write document \(documentID)")))
                return
            }
            completion(.success(documentID))
        }
    }

    public func getDocument(customID: String,
                            completion: @escaping (Result<NMPStoredDocument, NMPMemoryStoreError>) -> Void) {
        queue.async { [self] in
            // Scan all tags — customIDs are globally unique in this system.
            guard let record = records.values.first(where: { $0.customID == customID }) else {
                completion(.failure(.notFound))
                return
            }
            completion(.success(NMPStoredDocument(
                documentID: record.documentID, customID: record.customID,
                content: record.content, metadata: record.metadata,
                status: record.status)))
        }
    }

    public func search(query: String, containerTag: String, limit: Int,
                       completion: @escaping (Result<[NMPStoredHit], NMPMemoryStoreError>) -> Void) {
        queue.async { [self] in
            let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
            guard limit > 0, !trimmed.isEmpty else {
                completion(.success([]))
                return
            }
            let candidates = records.values.filter { $0.containerTag == containerTag }
            guard !candidates.isEmpty else {
                completion(.success([]))
                return
            }
            let queryVector = embedder.embed(query)
            let hits = candidates
                .map { record -> NMPStoredHit in
                    let docVector = vector(for: record)
                    let score = Self.cosine(queryVector, docVector)
                    return NMPStoredHit(
                        documentID: record.documentID, customID: record.customID,
                        content: record.content, score: score,
                        metadata: record.metadata)
                }
                .sorted { $0.score > $1.score }
            completion(.success(Array(hits.prefix(limit))))
        }
    }

    // MARK: Embedding cache (MUST run on `queue`)

    private func vector(for record: Record) -> Vector {
        if let cached = embeddingCache[record.documentID] { return cached }
        let v = embedder.embed(record.content)
        embeddingCache[record.documentID] = v
        return v
    }

    // MARK: Disk (MUST run on `queue`, except loadFromDisk in init)

    private func fileURL(for documentID: String) -> URL {
        // documentID → a filesystem-safe, collision-free name. Percent-encoding
        // is reversible, so distinct ids never share a file.
        let safe = documentID.addingPercentEncoding(
            withAllowedCharacters: .alphanumerics) ?? documentID
        return directory.appendingPathComponent("\(safe).json")
    }

    @discardableResult
    private func write(_ record: Record) -> Bool {
        let object: [String: Any] = [
            "document_id": record.documentID,
            "custom_id": record.customID as Any,  // NSNull when nil
            "container_tag": record.containerTag,
            "content": record.content,
            "metadata": record.metadata,
            "status": record.status,
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: object) else {
            return false
        }
        do {
            try data.write(to: fileURL(for: record.documentID), options: .atomic)
            return true
        } catch {
            return false
        }
    }

    private func loadFromDisk() {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: directory.path) else {
            return
        }
        for name in names where name.hasSuffix(".json") {
            let url = directory.appendingPathComponent(name)
            guard let data = try? Data(contentsOf: url),
                  let record = Self.decode(data) else { continue }
            records[record.documentID] = record
        }
    }

    private static func decode(_ data: Data) -> Record? {
        guard let o = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any],
              let documentID = o["document_id"] as? String,
              let containerTag = o["container_tag"] as? String,
              let content = o["content"] as? String
        else { return nil }
        return Record(
            documentID: documentID,
            customID: o["custom_id"] as? String,  // nil for NSNull / missing
            containerTag: containerTag,
            content: content,
            metadata: (o["metadata"] as? [String: String]) ?? [:],
            status: (o["status"] as? String) ?? "done")
    }

    private static func mintID() -> String {
        UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    }

    // MARK: - Vector

    /// A text's embedding: a dense vector (sentence / word-avg backends) or a
    /// sparse term-frequency map (lexical backend). Query and docs always use
    /// the same backend, so cosine only ever compares like with like.
    private enum Vector {
        case dense([Double])
        case sparse([String: Double])
    }

    /// Cosine similarity clamped to 0...1 (negatives — possible for dense
    /// vectors — floor at 0, per the protocol's score contract).
    private static func cosine(_ a: Vector, _ b: Vector) -> Double {
        let raw: Double
        switch (a, b) {
        case let (.dense(x), .dense(y)):
            raw = denseCosine(x, y)
        case let (.sparse(x), .sparse(y)):
            raw = sparseCosine(x, y)
        default:
            raw = 0  // mismatched representations never co-occur, but be safe
        }
        if raw.isNaN { return 0 }
        return min(1, max(0, raw))
    }

    private static func denseCosine(_ x: [Double], _ y: [Double]) -> Double {
        guard x.count == y.count, !x.isEmpty else { return 0 }
        var dot = 0.0, nx = 0.0, ny = 0.0
        for i in 0..<x.count {
            dot += x[i] * y[i]
            nx += x[i] * x[i]
            ny += y[i] * y[i]
        }
        guard nx > 0, ny > 0 else { return 0 }
        return dot / (nx.squareRoot() * ny.squareRoot())
    }

    private static func sparseCosine(_ x: [String: Double], _ y: [String: Double]) -> Double {
        guard !x.isEmpty, !y.isEmpty else { return 0 }
        var dot = 0.0
        let (small, large) = x.count <= y.count ? (x, y) : (y, x)
        for (k, v) in small { if let w = large[k] { dot += v * w } }
        let nx = x.values.reduce(0) { $0 + $1 * $1 }.squareRoot()
        let ny = y.values.reduce(0) { $0 + $1 * $1 }.squareRoot()
        guard nx > 0, ny > 0 else { return 0 }
        return dot / (nx * ny)
    }

    /// Lowercase, split on non-alphanumerics, drop empties.
    private static func tokenize(_ text: String) -> [String] {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }

    // MARK: - Embedder

    /// Chooses the best available on-device embedding backend ONCE, then embeds
    /// text into a `Vector`. See the file header for the fallback chain.
    private final class Embedder {
        private enum Mode { case sentence, wordAvg, lexical }
        private let mode: Mode
        #if canImport(NaturalLanguage)
        private let sentence: NLEmbedding?
        private let word: NLEmbedding?
        #endif

        init() {
            #if canImport(NaturalLanguage)
            if let s = NLEmbedding.sentenceEmbedding(for: .english) {
                sentence = s; word = nil; mode = .sentence
            } else if let w = NLEmbedding.wordEmbedding(for: .english) {
                sentence = nil; word = w; mode = .wordAvg
            } else {
                sentence = nil; word = nil; mode = .lexical
            }
            #else
            mode = .lexical
            #endif
        }

        var localityDescription: String {
            switch mode {
            case .sentence: return "on-device (NLEmbedding sentence)"
            case .wordAvg:  return "on-device (NLEmbedding word-avg)"
            case .lexical:  return "on-device (lexical)"
            }
        }

        func embed(_ text: String) -> Vector {
            #if canImport(NaturalLanguage)
            switch mode {
            case .sentence:
                if let v = sentence?.vector(for: text), !v.isEmpty {
                    return .dense(v)
                }
                // A specific string may lack a sentence vector; degrade to the
                // deterministic lexical form rather than a zero vector.
                return lexical(text)
            case .wordAvg:
                if let v = wordAverage(text) { return .dense(v) }
                return lexical(text)
            case .lexical:
                return lexical(text)
            }
            #else
            return lexical(text)
            #endif
        }

        #if canImport(NaturalLanguage)
        /// Average the per-word vectors, skipping out-of-vocabulary tokens.
        private func wordAverage(_ text: String) -> [Double]? {
            guard let word = word else { return nil }
            var sum: [Double] = []
            var count = 0
            for token in NMPLocalMemoryStore.tokenize(text) {
                guard let v = word.vector(for: token) else { continue }
                if sum.isEmpty {
                    sum = v
                } else if sum.count == v.count {
                    for i in 0..<sum.count { sum[i] += v[i] }
                }
                count += 1
            }
            guard count > 0, !sum.isEmpty else { return nil }
            return sum.map { $0 / Double(count) }
        }
        #endif

        /// Bag-of-words term-frequency vector.
        private func lexical(_ text: String) -> Vector {
            var tf: [String: Double] = [:]
            for token in NMPLocalMemoryStore.tokenize(text) {
                tf[token, default: 0] += 1
            }
            return .sparse(tf)
        }
    }
}
