//
//  NMPMemoryStore.swift
//  NMP — Memory mesh (hackathon build, NEW code)
//
//  The storage seam of the memory mesh. NMPMemoryMeshNode needs exactly two
//  capabilities from "a device's memory store":
//
//    1. store / fetch an opaque blob by a deterministic key (a shard, or an
//       index entry), and
//    2. semantic search over the plaintext index entries.
//
//  Both have more than one backend:
//    - NMPSupermemoryClient  — HTTP to a self-hosted supermemory-server
//                              (Macs; localhost only).
//    - NMPLocalMemoryStore   — a fully on-device store (iPhone / anywhere the
//                              Node server can't run): a file-backed blob store
//                              plus on-device embeddings for search.
//
//  This protocol is what the node depends on, so a peer can run either backend
//  (or a mesh can MIX them — Macs on Supermemory, a phone on the native store)
//  without the write/read paths knowing the difference. All backends are, by
//  construction, LOCAL to their device: no backend ever contacts a cloud
//  endpoint (the Supermemory client enforces localhost; the native store has
//  no network at all).
//
//  Callback style (house rule): every completion fires on the store's own
//  serial queue.
//

import Foundation

// MARK: - Neutral errors + records

public enum NMPMemoryStoreError: Error, Sendable {
    /// No document with that id (or it is not yet fetchable).
    case notFound
    /// The backend is unreachable / not ready (server down, embedder failed).
    case unavailable(String)
    /// Any other backend-reported failure.
    case backend(String)
}

/// A stored document, backend-neutral. `content` is the verbatim text/blob the
/// caller stored (for a shard this is opaque base64; for an index entry it is
/// the searchable plaintext).
public struct NMPStoredDocument: Sendable {
    public let documentID: String
    public let customID: String?
    public let content: String
    public let metadata: [String: String]
    /// Ingestion/readiness status as the backend reports it (native stores
    /// return "done" immediately; Supermemory reports queued/…/done).
    public let status: String

    public init(documentID: String, customID: String?, content: String,
                metadata: [String: String], status: String) {
        self.documentID = documentID
        self.customID = customID
        self.content = content
        self.metadata = metadata
        self.status = status
    }
}

/// One search hit, backend-neutral.
public struct NMPStoredHit: Sendable {
    public let documentID: String
    public let customID: String?
    public let content: String
    public let score: Double
    public let metadata: [String: String]

    public init(documentID: String, customID: String?, content: String,
                score: Double, metadata: [String: String]) {
        self.documentID = documentID
        self.customID = customID
        self.content = content
        self.score = score
        self.metadata = metadata
    }
}

// MARK: - Store protocol

public protocol NMPMemoryStore: AnyObject {
    /// Short backend id for diagnostics/status, e.g. "supermemory" or
    /// "on-device".
    var kind: String { get }
    /// Human description of WHERE the store lives — a URL for Supermemory,
    /// "on-device (…)" for the native store. Surfaced in /status.
    var localityDescription: String { get }

    /// Reachability / readiness. Fails fast at peer startup if the backend
    /// isn't up.
    func health(completion: @escaping (Result<Void, NMPMemoryStoreError>) -> Void)

    /// Store a document under `customID` (a deterministic key the caller
    /// mints). Returns the backend's document id.
    func addDocument(content: String, customID: String?, containerTag: String,
                     metadata: [String: String],
                     completion: @escaping (Result<String, NMPMemoryStoreError>) -> Void)

    /// Exact lookup by the caller's `customID` (NOT semantic). `.notFound`
    /// when absent.
    func getDocument(customID: String,
                     completion: @escaping (Result<NMPStoredDocument, NMPMemoryStoreError>) -> Void)

    /// Semantic search restricted to one container tag.
    func search(query: String, containerTag: String, limit: Int,
                completion: @escaping (Result<[NMPStoredHit], NMPMemoryStoreError>) -> Void)
}
