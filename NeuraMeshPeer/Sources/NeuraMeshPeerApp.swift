//
//  NeuraMeshPeerApp.swift
//  NeuraMeshPeer — Phase 5 iOS compute peer
//
//  The iPhone side of the mesh. All mesh logic lives in the NMP library
//  (NMPPeerNode); this app is a status screen around it, plus (Mesh 2.7)
//  a chat tab that talks to the coordinator's generation pipeline — the
//  phone both POWERS the mesh (Peer tab) and USES it (Chat tab).
//
//  The Xcode project is checked in: open
//  NeuraMeshPeer/NeuraMeshPeer.xcodeproj, pick your signing team, ⌘R.
//  Full walkthrough: Docs/CrossDevice_Setup_Guide.md.
//

import SwiftUI
import Combine

@main
struct NeuraMeshPeerApp: App {
    @StateObject private var model = PeerViewModel()

    var body: some Scene {
        WindowGroup {
            TabView {
                PeerStatusView()
                    .environmentObject(model)
                    .tabItem { Label("Peer", systemImage: "antenna.radiowaves.left.and.right") }
                ChatView()
                    .tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right") }
            }
            .onAppear { model.start() }
        }
    }
}
