//
//  MeshUILocator.swift
//  NeuraMeshPeer — Mesh 2.9
//
//  One shared Bonjour browser for the coordinator's web/API socket
//  (`_neuramesh-ui._tcp`, whose TXT record carries the UI's real host +
//  port). Chat sends generations through it; Models uses it to switch
//  the MESH's model — both tabs share a single live answer to "where is
//  the mesh UI?" instead of running two browsers.
//

import Foundation
import Network
import SwiftUI

@MainActor
final class MeshUILocator: ObservableObject {

    @Published private(set) var baseURL: URL?
    @Published private(set) var statusLine = "Looking for the mesh UI…"

    private var browser: NWBrowser?
    private let browseQueue = DispatchQueue(label: "nmp.peer.ui.browse")

    func start() {
        guard browser == nil else { return }
        let browser = NWBrowser(
            for: .bonjourWithTXTRecord(type: "_neuramesh-ui._tcp", domain: nil),
            using: NWParameters())
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            // First advert wins; the TXT record names the real UI socket.
            guard case .bonjour(let txt)? = results.first?.metadata,
                  let host = txt.dictionary["host"],
                  let port = txt.dictionary["port"],
                  let url = URL(string: "http://\(host):\(port)") else {
                Task { @MainActor [weak self] in
                    self?.baseURL = nil
                    self?.statusLine = results.isEmpty
                        ? "Looking for the mesh UI…"
                        : "Mesh found, but its advert has no host — "
                          + "update the coordinator"
                }
                return
            }
            Task { @MainActor [weak self] in
                self?.baseURL = url
                self?.statusLine = "Mesh: \(host):\(port)"
            }
        }
        browser.start(queue: browseQueue)
        self.browser = browser
    }

    /// POST /api/models/select — asks the coordinator to move the WHOLE
    /// mesh onto `model` (a filename or GGUF model name; the Mac resolves
    /// it against its ~/models). On success the mesh relaunches and every
    /// device follows. Completion runs on the main actor: nil = accepted,
    /// otherwise the server's reason (e.g. the Mac doesn't have the file).
    func switchMeshModel(to model: String,
                         completion: @escaping @MainActor (String?) -> Void) {
        guard let baseURL else {
            Task { @MainActor in completion("no mesh in reach") }
            return
        }
        var request = URLRequest(
            url: baseURL.appendingPathComponent("api/models/select"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(
            withJSONObject: ["path": model])
        request.timeoutInterval = 15
        URLSession.shared.dataTask(with: request) { data, response, error in
            let failure: String?
            if let error {
                failure = error.localizedDescription
            } else if let data,
                      let object = try? JSONSerialization.jsonObject(with: data)
                        as? [String: Any],
                      let message = object["error"] as? String {
                failure = message
            } else if let http = response as? HTTPURLResponse,
                      http.statusCode >= 400 {
                failure = "mesh refused the switch (HTTP \(http.statusCode))"
            } else {
                failure = nil
            }
            Task { @MainActor in completion(failure) }
        }.resume()
    }
}
