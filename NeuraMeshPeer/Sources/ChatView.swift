//
//  ChatView.swift
//  NeuraMeshPeer — Mesh 2.7
//
//  Chat with the mesh from the phone that powers it. The coordinator's
//  web server owns the generation pipeline (POST /api/chat assembles the
//  engine's template server-side — same one the web UI uses); the shared
//  MeshUILocator finds it via the `_neuramesh-ui._tcp` Bonjour advert.
//
//  The conversation lives in this view; the mesh is stateless — every
//  turn resends the whole transcript.
//

import SwiftUI

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
    @Published var lastStats = ""

    func send(to baseURL: URL?) {
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
    @EnvironmentObject private var mesh: MeshUILocator
    @StateObject private var model = MeshChatModel()
    // Keyboard focus is explicit so it can always be RELEASED: dragging
    // the conversation, tapping outside the composer, the keyboard's Done
    // button, and sending all dismiss it — without this, the keyboard
    // covered the tab bar with no way down (the composer's multiline
    // TextField swallows Return as a newline, so onSubmit never fires).
    @FocusState private var composing: Bool

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Circle()
                    .fill(mesh.baseURL == nil ? .orange : .green)
                    .frame(width: 8, height: 8)
                Text(mesh.statusLine)
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
                .scrollDismissesKeyboard(.immediately)
                .contentShape(Rectangle())
                .onTapGesture { composing = false }
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
                    .focused($composing)
                Button {
                    composing = false
                    model.send(to: mesh.baseURL)
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                }
                .disabled(model.generating || mesh.baseURL == nil
                          || model.draft.trimmingCharacters(
                                in: .whitespacesAndNewlines).isEmpty)
            }
            .padding()
        }
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") { composing = false }
            }
        }
        .onAppear { mesh.start() }
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
