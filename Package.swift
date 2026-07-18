// swift-tools-version:5.8
// NeuraMesh Protocol (NMP)
// Apple-native: Network.framework + CryptoKit. Build with Xcode 14.2+ / Swift 5.8+.
//
// Phase 5 adds two executables for the cross-device mesh harness:
//   swift run nmp-peer         — compute peer (same runtime the iOS app embeds)
//   swift run nmp-coordinator  — coordinator + benchmark driver
// Phase 6 adds the testing dashboard:
//   swift run nmp-dashboard    — simulated mesh + web dashboard on :8080

import PackageDescription

let package = Package(
    name: "NeuraMeshProtocol",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "NMP", targets: ["NMP"]),
        .executable(name: "nmp-peer", targets: ["NMPPeerCLI"]),
        .executable(name: "nmp-coordinator", targets: ["NMPCoordinatorCLI"]),
        .executable(name: "nmp-dashboard", targets: ["NMPDashboardCLI"]),
        .executable(name: "nmp-memory-peer", targets: ["NMPMemoryPeerCLI"]),
    ],
    targets: [
        .target(
            name: "NMP",
            path: "Sources/NMP",
            resources: [
                .copy("Resources/dashboard.html"),
            ]
        ),
        .executableTarget(
            name: "NMPPeerCLI",
            dependencies: ["NMP"],
            path: "Sources/NMPPeerCLI"
        ),
        .executableTarget(
            name: "NMPCoordinatorCLI",
            dependencies: ["NMP"],
            path: "Sources/NMPCoordinatorCLI"
        ),
        .executableTarget(
            name: "NMPDashboardCLI",
            dependencies: ["NMP"],
            path: "Sources/NMPDashboardCLI"
        ),
        .executableTarget(
            name: "NMPMemoryPeerCLI",
            dependencies: ["NMP"],
            path: "Sources/NMPMemoryPeerCLI"
        ),
        .testTarget(
            name: "NMPTests",
            dependencies: ["NMP"],
            path: "Tests/NMPTests"
        ),
    ]
)
