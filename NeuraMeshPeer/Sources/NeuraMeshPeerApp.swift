//
//  NeuraMeshPeerApp.swift
//  NeuraMeshPeer — Phase 5 iOS compute peer
//
//  The iPhone side of the mesh. All mesh logic lives in the NMP library
//  (NMPPeerNode); this app is a status screen around it.
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
            PeerStatusView()
                .environmentObject(model)
                .onAppear { model.start() }
        }
    }
}
