//
//  ChatView.swift
//  NeuraMeshPeer — Mesh 2.7
//
//  Chat with the mesh from the phone that powers it. The coordinator's
//  web server owns the generation pipeline (POST /api/chat assembles the
//  engine's template server-side — same one the web UI uses); this view
//  finds it via the `_neuramesh-ui._tcp` Bonjour advert, whose TXT record
//  carries the UI's real host + port (the advert itself sits on a
//  throwaway ephemeral listener).
//
//  The conversation lives in this view; the mesh is stateless — every
//  turn resends the whole transcript.
//

import SwiftUI
import Network

@MainActor
final class MeshChatModel: ObservableObject {

    struct Turn: Identifiable, Equatable {
        enum Role: String { case user, assistant }
        let id = UUID()
        let role: Role
        var text: String
    }

    @Published var turns: [Turn] = []
    @Published var draft = ""
    @Published var generating = false
    @Published var statusLine = "Looking for the mesh UI…"
    @Published var lastStats = ""
    @Published private(set) var baseURL: URL?

    private var browser: NWBrowser?
    private let browseQueue = DispatchQueue(label: "nmp.peer.chat.browse")

    func start() {
        guard browser == nil else { return }
        let browser = NWBrowser(
            for: .bonjourWithTXTRecord(
                type: "_neuramesh-ui._tcp", domain: nil),
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

    func send() {
        let content = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty, !generating, let baseURL else { return }
        turns.append(Turn(role: .user, text: content))
        draft = ""
        generating = true
        lastStats = ""

        // The whole transcript, every turn — the mesh holds no session.
        let messages = turns.map {
            ["role": $0.role == .user ? "user" : "assistant",
             "content": $0.text]
        }
        var request = URLRequest(url: baseURL.appendingPathComponent("api/chat"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: [
            "messages": messages,
            "max_tokens": 64,
        ] as [String: Any])
        request.timeoutInterval = 120 // a 7B model on a phone mesh is not fast

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.generating = false
                if let error {
                    self.turns.append(Turn(role: .assistant,
                                           text: "⚠️ \(error.localizedDescription)"))
                    return
                }
                guard let data,
                      let object = try? JSONSerialization.jsonObject(with: data)
                        as? [String: Any] else {
                    self.turns.append(Turn(role: .assistant,
                                           text: "⚠️ unreadable reply from the mesh"))
                    return
                }
                if let failure = object["error"] as? String {
                    self.turns.append(Turn(role: .assistant, text: "⚠️ \(failure)"))
                    return
                }
                let output = (object["output"] as? String) ?? ""
                self.turns.append(Turn(
                    role: .assistant,
                    text: output.isEmpty ? "(no tokens emitted)" : output))
                if let tps = object["tokens_per_sec"] as? Double,
                   let trips = object["round_trips"] as? Int {
                    self.lastStats = String(
                        format: "%.1f tok/s · %d mesh round trips", tps, trips)
                }
            }
        }.resume()
    }

    func clear() {
        turns.removeAll()
        lastStats = ""
    }
}

struct ChatView: View {
    @StateObject private var model = MeshChatModel()

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Circle()
                    .fill(model.baseURL == nil ? .orange : .green)
                    .frame(width: 8, height: 8)
                Text(model.statusLine)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if !model.turns.isEmpty {
                    Button("Clear") { model.clear() }
                        .font(.caption)
                        .disabled(model.generating)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 8) {
                        if model.turns.isEmpty {
                            Text("Every reply is generated across the mesh, "
                                 + "one token per round trip.")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                                .multilineTextAlignment(.center)
                                .padding(.top, 40)
                        }
                        ForEach(model.turns) { turn in
                            bubble(for: turn)
                        }
                        if model.generating {
                            HStack {
                                ProgressView()
                                Text("generating on the mesh…")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                                Spacer()
                            }
                            .padding(.horizontal)
                            .id("pending")
                        }
                    }
                    .padding(.vertical, 8)
                }
                .onChange(of: model.turns) { _ in
                    if let last = model.turns.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }

            if !model.lastStats.isEmpty {
                Text(model.lastStats)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.bottom, 4)
            }

            HStack(spacing: 8) {
                TextField("Message the mesh…", text: $model.draft,
                          axis: .vertical)
                    .lineLimit(1...4)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { model.send() }
                Button {
                    model.send()
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                }
                .disabled(model.generating || model.baseURL == nil
                          || model.draft.trimmingCharacters(
                                in: .whitespacesAndNewlines).isEmpty)
            }
            .padding()
        }
        .onAppear { model.start() }
    }

    @ViewBuilder
    private func bubble(for turn: MeshChatModel.Turn) -> some View {
        HStack {
            if turn.role == .user { Spacer(minLength: 40) }
            Text(turn.text)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(turn.role == .user
                            ? Color.accentColor
                            : Color(.secondarySystemBackground))
                .foregroundStyle(turn.role == .user ? .white : .primary)
                .clipShape(RoundedRectangle(cornerRadius: 14))
            if turn.role == .assistant { Spacer(minLength: 40) }
        }
        .padding(.horizontal)
        .id(turn.id)
    }
}
