//
//  NeuraMeshPeerApp.swift
//  NeuraMeshPeer — Phase 5 iOS compute peer
//
//  The iPhone side of the mesh. All mesh logic lives in the NMP library
//  (NMPPeerNode); this app is a status screen around it.
//
//  Project setup is 5 minutes by hand — follow
//  Docs/CrossDevice_Setup_Guide.md step by step (create an iOS App
//  project, add the NeuraMeshProtocol local package, drop these three
//  files in, add two Info.plist keys, run on your iPhone).
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
