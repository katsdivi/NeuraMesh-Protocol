// swift-tools-version:5.8
// NeuraMesh Protocol (NMP) — Phase 1: Core Transport
// Apple-native: Network.framework + CryptoKit. Build with Xcode 14.2+ / Swift 5.8+.

import PackageDescription

let package = Package(
    name: "NeuraMeshProtocol",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "NMP", targets: ["NMP"]),
    ],
    targets: [
        .target(
            name: "NMP",
            path: "Sources/NMP"
        ),
        .testTarget(
            name: "NMPTests",
            dependencies: ["NMP"],
            path: "Tests/NMPTests"
        ),
    ]
)
