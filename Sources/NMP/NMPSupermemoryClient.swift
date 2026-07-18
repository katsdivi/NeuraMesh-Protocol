//
//  NMPSupermemoryClient.swift
//  NMP — Memory mesh (hackathon build, NEW code)
//
//  A minimal HTTP client for ONE device's OWN self-hosted Supermemory
//  instance (the `supermemory-server` local binary, http://localhost:6767).
//  Every peer runs its own instance; this client only ever talks to the
//  loopback one on that device.
//
//  LOCAL-ONLY GUARD (hackathon constraint, enforced in code): the config
//  initializer REFUSES any base URL whose host is not localhost / 127.0.0.1
//  / ::1. The build must never reach the hosted platform
//  (console.supermemory.ai / api.supermemory.ai) — a mistyped config fails
//  fast at construction instead of silently shipping memories to the cloud.
//
//  Endpoints used (verified empirically against supermemory-server; see
//  Docs/Fable_Findings or the memory-mesh README for the probe log):
//    - health:  GET  /v3/documents/processing   (cheapest authenticated 200)
//    - add:     POST /v3/documents               {content, customId,
//                                                  containerTag, metadata}
//                                                  → {id, status}
//    - get:     GET  /v3/documents/{customId}     (the server accepts a
//                                                  customId directly) → full
//                                                  document incl. content
//    - search:  POST /v4/search                   {q, containerTag, limit,
//                                                  searchMode:"documents",
//                                                  threshold} → {results:[…]}
//
//  Ingestion is asynchronous: an added document is `queued` → `embedding` →
//  `indexing` → `done`, and becomes searchable around the `indexing` stage
//  (~1 s warm; the first ever ingestion is slower while the local embedding
//  model warms). Callers that need read-your-writes immediately after add
//  should fetch by customId (exact, available once `done`) rather than rely
//  on search. IMPORTANT: search defaults to `searchMode:"memories"`, which
//  needs an LLM provider and returns nothing on a default local instance —
//  this client always sends `searchMode:"documents"` so chunk search works
//  with local embeddings alone.
//
//  Auth: `Authorization: Bearer <key>`. House rules: no async/await —
//  URLSession completion handlers hop back onto a serial queue; Foundation
//  only; NMP prefix.
//

import Foundation

// MARK: - Errors

public enum NMPSupermemoryError: Error, Sendable {
    /// The configured base URL is not loopback — cloud endpoints are banned.
    case nonLocalBaseURL(String)
    case transport(String)
    case httpStatus(Int, String)
    case malformedResponse(String)
    case notFound
}

// MARK: - Config

/// Configuration for ONE device's own local Supermemory instance.
public struct NMPSupermemoryConfig: Sendable {
    public let baseURL: URL
    public let apiKey: String

    /// Hosts accepted as "this device's own instance".
    static let localHosts: Set<String> = ["localhost", "127.0.0.1", "::1", "[::1]"]

    /// Throws `.nonLocalBaseURL` unless the host is loopback.
    public init(baseURL: URL, apiKey: String) throws {
        let host = (baseURL.host ?? "").lowercased()
        guard Self.localHosts.contains(host) else {
            throw NMPSupermemoryError.nonLocalBaseURL(
                "base URL host '\(host)' is not localhost — cloud Supermemory "
                + "endpoints (console/api.supermemory.ai) are forbidden in this build")
        }
        self.baseURL = baseURL
        self.apiKey = apiKey
    }
}

// MARK: - Client

public final class NMPSupermemoryClient: NMPMemoryStore {

    public let kind = "supermemory"
    public var localityDescription: String { config.baseURL.absoluteString }

    private let config: NMPSupermemoryConfig
    private let queue: DispatchQueue
    private let session: URLSession

    public init(config: NMPSupermemoryConfig,
                queue: DispatchQueue = DispatchQueue(label: "nmp.supermemory.client")) {
        self.config = config
        self.queue = queue
        let sessionConfig = URLSessionConfiguration.ephemeral
        sessionConfig.timeoutIntervalForRequest = 5   // localhost — short
        sessionConfig.waitsForConnectivity = false
        self.session = URLSession(configuration: sessionConfig)
    }

    /// Maps the client's transport-level error onto the neutral store error.
    private static func neutral(_ error: NMPSupermemoryError) -> NMPMemoryStoreError {
        switch error {
        case .notFound: return .notFound
        case .nonLocalBaseURL(let m): return .unavailable(m)
        case .transport(let m): return .unavailable(m)
        case .httpStatus(let code, let body): return .backend("HTTP \(code): \(body)")
        case .malformedResponse(let m): return .backend(m)
        }
    }

    // MARK: NMPMemoryStore (completions fire on `queue`)

    /// Server reachability + API-key validity (used at peer startup).
    public func health(completion: @escaping (Result<Void, NMPMemoryStoreError>) -> Void) {
        var request = makeRequest(path: "/v3/documents/processing", method: "GET")
        request.timeoutInterval = 3
        perform(request) { result in
            completion(result.map { _ in () }.mapError(Self.neutral))
        }
    }

    /// Adds a document. Returns the server-assigned document id.
    public func addDocument(content: String, customID: String?, containerTag: String,
                            metadata: [String: String],
                            completion: @escaping (Result<String, NMPMemoryStoreError>) -> Void) {
        var body: [String: Any] = [
            "content": content,
            "containerTag": containerTag,
        ]
        if let customID { body["customId"] = customID }
        if !metadata.isEmpty { body["metadata"] = metadata }

        guard let request = makeJSONRequest(path: "/v3/documents", method: "POST",
                                            body: body) else {
            completion(.failure(.backend("could not encode add-document body")))
            return
        }
        perform(request) { result in
            switch result {
            case .success(let data):
                guard let object = try? JSONSerialization.jsonObject(with: data)
                        as? [String: Any],
                      let id = object["id"] as? String else {
                    completion(.failure(.backend("add-document: no id in response")))
                    return
                }
                completion(.success(id))
            case .failure(let error):
                completion(.failure(Self.neutral(error)))
            }
        }
    }

    /// Exact lookup by the customID we minted at add time (NOT semantic).
    /// The server accepts a customId in place of its own id. `.notFound`
    /// when absent (or still `queued`, before it is fetchable).
    public func getDocument(customID: String,
                            completion: @escaping (Result<NMPStoredDocument, NMPMemoryStoreError>) -> Void) {
        let escaped = customID.addingPercentEncoding(
            withAllowedCharacters: .urlPathAllowed) ?? customID
        let request = makeRequest(path: "/v3/documents/\(escaped)", method: "GET")
        perform(request) { result in
            switch result {
            case .success(let data):
                guard let o = try? JSONSerialization.jsonObject(with: data)
                        as? [String: Any] else {
                    completion(.failure(.backend("get-document: not JSON")))
                    return
                }
                let doc = NMPStoredDocument(
                    documentID: (o["id"] as? String) ?? customID,
                    customID: (o["customId"] as? String) ?? customID,
                    content: (o["content"] as? String) ?? "",
                    metadata: Self.stringMetadata(o["metadata"]),
                    status: (o["status"] as? String) ?? "unknown")
                completion(.success(doc))
            case .failure(.httpStatus(404, _)):
                completion(.failure(.notFound))
            case .failure(let error):
                completion(.failure(Self.neutral(error)))
            }
        }
    }

    /// Hybrid chunk search restricted to one container tag. Always uses
    /// `searchMode:"documents"` so it works with local embeddings and no LLM.
    public func search(query: String, containerTag: String, limit: Int,
                       completion: @escaping (Result<[NMPStoredHit], NMPMemoryStoreError>) -> Void) {
        let body: [String: Any] = [
            "q": query,
            "containerTag": containerTag,
            "limit": limit,
            "searchMode": "documents",
            "threshold": 0,
        ]
        guard let request = makeJSONRequest(path: "/v4/search", method: "POST",
                                            body: body) else {
            completion(.failure(.backend("could not encode search body")))
            return
        }
        perform(request) { result in
            switch result {
            case .success(let data):
                guard let o = try? JSONSerialization.jsonObject(with: data)
                        as? [String: Any],
                      let results = o["results"] as? [[String: Any]] else {
                    completion(.failure(.backend("search: no results array")))
                    return
                }
                completion(.success(results.map(Self.parseHit)))
            case .failure(let error):
                completion(.failure(Self.neutral(error)))
            }
        }
    }

    // MARK: - Request building

    private func makeRequest(path: String, method: String) -> URLRequest {
        var request = URLRequest(url: config.baseURL.appendingPathComponent(
            path.hasPrefix("/") ? String(path.dropFirst()) : path))
        request.httpMethod = method
        request.setValue("Bearer \(config.apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        return request
    }

    private func makeJSONRequest(path: String, method: String,
                                 body: [String: Any]) -> URLRequest? {
        guard let data = try? JSONSerialization.data(withJSONObject: body) else {
            return nil
        }
        var request = makeRequest(path: path, method: method)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = data
        return request
    }

    private func perform(_ request: URLRequest,
                         completion: @escaping (Result<Data, NMPSupermemoryError>) -> Void) {
        let task = session.dataTask(with: request) { [queue] data, response, error in
            queue.async {
                if let error {
                    completion(.failure(.transport(error.localizedDescription)))
                    return
                }
                guard let http = response as? HTTPURLResponse else {
                    completion(.failure(.transport("no HTTP response")))
                    return
                }
                let data = data ?? Data()
                guard (200..<300).contains(http.statusCode) else {
                    let snippet = String(data: data.prefix(200), encoding: .utf8) ?? ""
                    completion(.failure(.httpStatus(http.statusCode, snippet)))
                    return
                }
                completion(.success(data))
            }
        }
        task.resume()
    }

    // MARK: - Parsing helpers

    /// Coerces a server metadata object (string/number/bool values) to
    /// `[String: String]`, which is all this layer stores.
    static func stringMetadata(_ raw: Any?) -> [String: String] {
        guard let dict = raw as? [String: Any] else { return [:] }
        var out: [String: String] = [:]
        for (key, value) in dict {
            switch value {
            case let s as String: out[key] = s
            case let b as Bool: out[key] = b ? "true" : "false"
            case let n as NSNumber: out[key] = n.stringValue
            default: out[key] = "\(value)"
            }
        }
        return out
    }

    /// One /v4/search result → NMPStoredHit. A chunk hit carries `chunk` text,
    /// `similarity`, chunk `metadata`, and a `documents` array whose first
    /// entry's `id` is the customId when one was set at add time.
    static func parseHit(_ result: [String: Any]) -> NMPStoredHit {
        let documents = result["documents"] as? [[String: Any]] ?? []
        let firstDoc = documents.first
        let customID = firstDoc?["id"] as? String
        let content = (result["chunk"] as? String)
            ?? (result["memory"] as? String)
            ?? (firstDoc?["title"] as? String)
            ?? ""
        let score = (result["similarity"] as? Double)
            ?? (result["score"] as? Double) ?? 0
        // Prefer document-level metadata (carries our nmp-index fields);
        // fall back to chunk-level.
        let metadata = Self.stringMetadata(firstDoc?["metadata"] ?? result["metadata"])
        return NMPStoredHit(
            documentID: (result["id"] as? String) ?? customID ?? "",
            customID: customID,
            content: content,
            score: score,
            metadata: metadata)
    }
}
