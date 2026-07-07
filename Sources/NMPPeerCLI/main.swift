//
//  main.swift
//  nmp-peer — Phase 5 compute-peer CLI
//
//  Runs the same NMPPeerNode runtime the iOS app embeds: binds a UDP
//  listener, advertises over Bonjour (capabilities + port + static key),
//  accepts the coordinator's Noise IK handshake, serves its assigned
//  shard, and logs every inference it computes.
//
//  Usage:
//    swift run nmp-peer [--layers N] [--hidden N] [--gguf path]
//                       [--tag modelTag] [--slow msPerLayer]
//
//  --gguf sizes the engine from a real model file's metadata (layer count
//  and hidden size are read from the container; compute itself is the
//  deterministic reference engine — see Phase5_Design.md for the
//  llama.cpp binding point).
//  --slow adds artificial per-layer delay to emulate a weaker device when
//  demoing load-balanced sharding on identical hardware.
//

import Foundation
import NMP

// MARK: - Arguments

struct PeerArguments {
    var layers = 32
    var hidden = 4096
    var ggufPath: String?
    var modelTag = "nmp-reference-model"
    var slowMillisPerLayer = 0.0

    static func parse() -> PeerArguments {
        var arguments = PeerArguments()
        var iterator = CommandLine.arguments.dropFirst().makeIterator()
        while let flag = iterator.next() {
            let value = { iterator.next() }
            switch flag {
            case "--layers": arguments.layers = value().flatMap(Int.init) ?? arguments.layers
            case "--hidden": arguments.hidden = value().flatMap(Int.init) ?? arguments.hidden
            case "--gguf": arguments.ggufPath = value()
            case "--tag": arguments.modelTag = value() ?? arguments.modelTag
            case "--slow": arguments.slowMillisPerLayer = value().flatMap(Double.init) ?? 0
            case "--help", "-h":
                print("""
                usage: nmp-peer [--layers N] [--hidden N] [--gguf path] \
                [--tag modelTag] [--slow msPerLayer]
                """)
                exit(0)
            default:
                FileHandle.standardError.write(Data("unknown flag \(flag)\n".utf8))
                exit(2)
            }
        }
        return arguments
    }
}

// MARK: - Engine setup

let arguments = PeerArguments.parse()
let engine: NMPReferenceComputeEngine
var modelTag = arguments.modelTag

if let path = arguments.ggufPath {
    do {
        let gguf = try NMPGGUFModel.load(path: path)
        engine = try NMPReferenceComputeEngine(gguf: gguf)
        if let name = gguf.modelName { modelTag = name }
        print("[peer] loaded GGUF: \(gguf.modelName ?? path) — "
              + "\(engine.layerCount) layers × \(engine.hiddenSize) hidden "
              + "(\(gguf.tensors.count) tensors, \(gguf.architecture ?? "?") arch)")
    } catch {
        FileHandle.standardError.write(Data("failed to load GGUF at \(path): \(error)\n".utf8))
        exit(1)
    }
} else {
    engine = NMPReferenceComputeEngine(layerCount: arguments.layers, hiddenSize: arguments.hidden)
}
engine.simulatedSecondsPerLayer = arguments.slowMillisPerLayer / 1000

// MARK: - Run

let node = NMPPeerNode(engine: engine, modelTag: modelTag)

func stamp() -> String {
    let formatter = DateFormatter()
    formatter.dateFormat = "HH:mm:ss.SSS"
    return formatter.string(from: Date())
}

node.onStatus = { print("[peer \(stamp())] \($0)") }
node.onDiagnostic = { print("[peer \(stamp())] (diag) \($0)") }
node.onServed = { requestID, layers, seconds in
    print("[peer \(stamp())] served request \(requestID): layers "
          + "\(layers.lowerBound)..<\(layers.upperBound) in "
          + String(format: "%.2f", seconds * 1000) + " ms "
          + "(mem \(NMPPeerShardEngine.residentMemoryMB()) MB)")
}

do {
    try node.start()
} catch {
    FileHandle.standardError.write(Data("failed to start peer: \(error)\n".utf8))
    exit(1)
}

print("[peer] running — Ctrl+C to stop")
dispatchMain()
