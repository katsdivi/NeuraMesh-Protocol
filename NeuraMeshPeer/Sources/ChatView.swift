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
    // NOTE: the composer's draft lives in the VIEW's @State, not here — an
    // @Published draft re-published the whole model on every keystroke, so
    // SwiftUI re-rendered the entire transcript + history sheet per character
    // (the "very laggy typing"). The view hands the text to `send` instead.
    @Published var generating = false
    @Published var lastStats = ""
    /// Saved conversations on THIS phone (the history sheet's list).
    @Published var conversations: [ChatRow] = []

    /// This phone's local history store. deviceName is read on the main
    /// actor (UIDevice is main-actor) and handed to the store.
    private let store = ChatStore(deviceName: UIDevice.current.name)
    /// id of the conversation being appended to; "" = unsaved new chat.
    private var activeID = ""
    /// Best-effort model label for the record (the mesh serves one model).
    var modelName = ""

    func send(_ text: String, to baseURL: URL?) {
        let content = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty, !generating, let baseURL else { return }
        turns.append(Turn(role: .user, text: content))
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
                self.persistCurrent()   // save this turn to local history
            }
        }.resume()
    }

    // MARK: History

    /// Loads the saved-conversation list for the history sheet.
    func refreshList() {
        Task {
            let rows = await store.list()
            await MainActor.run { self.conversations = rows }
        }
    }

    /// Persists the current transcript locally, learning the id for a
    /// brand-new chat. Best-effort — generation already succeeded.
    private func persistCurrent() {
        let messages = turns.map {
            StoredMessage(role: $0.role.rawValue, content: $0.text)
        }
        guard !messages.isEmpty else { return }
        let id = activeID
        Task {
            let row = await store.save(id: id, title: "", model: modelName,
                                       messages: messages)
            await MainActor.run {
                if let row { self.activeID = row.id }
                self.refreshList()
            }
        }
    }

    /// Starts a fresh, unsaved conversation.
    func newChat() {
        activeID = ""
        turns.removeAll()
        lastStats = ""
    }

    /// Opens a saved conversation into the thread.
    func open(_ row: ChatRow) {
        guard !generating else { return }
        Task {
            guard let c = await store.load(id: row.id) else { return }
            await MainActor.run {
                self.activeID = c.id
                self.turns = c.messages.map {
                    Turn(role: $0.role == "user" ? .user : .assistant,
                         text: $0.content)
                }
                self.lastStats = ""
            }
        }
    }

    /// Deletes a saved conversation; if it's the open one, resets to new.
    func delete(_ row: ChatRow) {
        Task {
            await store.delete(id: row.id)
            await MainActor.run {
                if row.id == self.activeID { self.newChat() }
                self.refreshList()
            }
        }
    }

    /// Packs conversations that have gone quiet (call on background/launch).
    func compressInactive() {
        Task { _ = await store.compressInactive() }
    }

    func clear() {
        turns.removeAll()
        lastStats = ""
        activeID = ""
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
    @State private var showingHistory = false
    // The composer's text lives here, not on the model, so typing only
    // re-renders the composer — not the whole transcript + history sheet.
    @State private var draft = ""
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Button {
                    model.refreshList()
                    showingHistory = true
                } label: {
                    Image(systemName: "clock.arrow.circlepath")
                }
                Circle()
                    .fill(mesh.baseURL == nil ? .orange : .green)
                    .frame(width: 8, height: 8)
                Text(mesh.statusLine)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    model.newChat()
                    draft = ""
                } label: {
                    Image(systemName: "square.and.pencil")
                }
                .disabled(model.generating)
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
                TextField("Message the mesh…", text: $draft,
                          axis: .vertical)
                    .lineLimit(1...4)
                    .textFieldStyle(.roundedBorder)
                    .focused($composing)
                Button {
                    composing = false
                    model.send(draft, to: mesh.baseURL)
                    draft = ""
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                }
                .disabled(model.generating || mesh.baseURL == nil
                          || draft.trimmingCharacters(
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
        .onAppear {
            mesh.start()
            model.refreshList()
        }
        .onChange(of: scenePhase) { phase in
            // Leaving the app is exactly "the session went inactive" — pack
            // quiet conversations so history is thrifty on disk.
            if phase == .background { model.compressInactive() }
        }
        .sheet(isPresented: $showingHistory) { historySheet }
    }

    private var historySheet: some View {
        NavigationView {
            List {
                if model.conversations.isEmpty {
                    Text("No saved chats yet.")
                        .foregroundStyle(.secondary)
                }
                ForEach(model.conversations) { row in
                    Button {
                        model.open(row)
                        showingHistory = false
                    } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(row.title)
                                .lineLimit(1)
                                .foregroundStyle(.primary)
                            HStack(spacing: 6) {
                                Text(row.updatedDate, style: .relative)
                                Text("· \(row.message_count) msg")
                                if row.compressed {
                                    Text("· zip").foregroundStyle(.tint)
                                }
                            }
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        }
                    }
                }
                .onDelete { offsets in
                    for index in offsets {
                        model.delete(model.conversations[index])
                    }
                }
            }
            .navigationTitle("History")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("New") {
                        model.newChat()
                        draft = ""
                        showingHistory = false
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { showingHistory = false }
                }
            }
        }
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
