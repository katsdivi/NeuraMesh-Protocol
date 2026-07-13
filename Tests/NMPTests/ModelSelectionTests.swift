//
//  ModelSelectionTests.swift
//  NMP — Phase C (adaptive model tiering)
//
//  The "optimal per scenario" selector, tested deterministically with
//  synthetic meshes + catalogs, plus a real-GGUF footprint/catalog check.
//

import XCTest
@testable import NMP

final class ModelSelectionTests: XCTestCase {

    // MARK: Synthetic fixtures

    private func model(_ name: String, params: Int, layers: Int, hidden: Int,
                       fileMB: Int, bplMB: Int) -> NMPModelCandidate {
        NMPModelCandidate(
            path: "/models/\(name).gguf", name: name, architecture: "qwen2",
            layerCount: layers, hiddenSize: hidden,
            fileBytes: fileMB * 1_048_576, bytesPerLayer: bplMB * 1_048_576,
            totalParameters: params)
    }

    /// A realistic tier ladder (footprints ≈ Qwen2.5 q4 sizes).
    private var ladder: [NMPModelCandidate] {
        [model("qwen14b", params: 14_000_000_000, layers: 48, hidden: 5120, fileMB: 9000, bplMB: 180),
         model("qwen7b",  params:  7_000_000_000, layers: 32, hidden: 4096, fileMB: 4700, bplMB: 140),
         model("qwen3b",  params:  3_000_000_000, layers: 36, hidden: 2048, fileMB: 2000, bplMB: 55),
         model("qwen05b", params:    500_000_000, layers: 24, hidden: 896,  fileMB: 400,  bplMB: 15)]
    }

    private func device(_ id: UInt32, ramMB: UInt32, storageMB: UInt32) -> NMPCapabilities {
        NMPCapabilities(peerID: id, deviceName: "dev\(id)", ramMB: ramMB,
                        computeClass: .high, storageFreeMB: storageMB)
    }

    private func assertCoversAllLayers(_ selection: NMPModelSelection) {
        let assigned = selection.plan.map(\.layerSpan).reduce(0, +)
        XCTAssertEqual(assigned, selection.model.layerCount,
                       "plan must cover every layer exactly once")
    }

    // MARK: Optimal pick

    func testTwoStrongDevicesGetTheLargestModel() throws {
        let mesh = [device(1, ramMB: 32_768, storageMB: 500_000),
                    device(2, ramMB: 32_768, storageMB: 500_000)]
        let pick = try XCTUnwrap(NMPModelSelector.pick(mesh: mesh, catalog: ladder))
        XCTAssertEqual(pick.model.name, "qwen14b")
        XCTAssertEqual(pick.eligiblePeers.count, 2)
        assertCoversAllLayers(pick)
    }

    func testSingleDeviceDegradesOnRAM() throws {
        // 8 GB device: can store 14B but can't hold its 48 layers → 7B.
        let mesh = [device(1, ramMB: 8_192, storageMB: 60_000)]
        let pick = try XCTUnwrap(NMPModelSelector.pick(mesh: mesh, catalog: ladder))
        XCTAssertEqual(pick.model.name, "qwen7b")
        XCTAssertTrue(pick.reason.contains("degraded past"))
        XCTAssertTrue(pick.reason.contains("qwen14b"))
        assertCoversAllLayers(pick)
    }

    func testStorageShortfallForcesDegrade() throws {
        // Huge RAM, tiny free disk: 14B and 7B files don't fit → 3B.
        let mesh = [device(1, ramMB: 65_536, storageMB: 3_000),
                    device(2, ramMB: 65_536, storageMB: 3_000)]
        let pick = try XCTUnwrap(NMPModelSelector.pick(mesh: mesh, catalog: ladder))
        XCTAssertEqual(pick.model.name, "qwen3b")
        XCTAssertTrue(pick.reason.contains("disk"), "reason must cite the storage ceiling")
        assertCoversAllLayers(pick)
    }

    func testMixedStorageExcludesTheStarvedDevice() throws {
        // One device can't store 14B; the other can host it alone.
        let mesh = [device(1, ramMB: 64_000, storageMB: 500_000), // fits 14B
                    device(2, ramMB: 64_000, storageMB: 2_000)]   // no disk for 14B
        let pick = try XCTUnwrap(NMPModelSelector.pick(mesh: mesh, catalog: ladder))
        XCTAssertEqual(pick.model.name, "qwen14b")
        XCTAssertEqual(pick.eligiblePeers, [1], "only the device with disk hosts it")
    }

    func testUpgradesWhenADeviceJoins() throws {
        let one = [device(1, ramMB: 8_192, storageMB: 60_000)]
        let before = try XCTUnwrap(NMPModelSelector.pick(mesh: one, catalog: ladder))
        XCTAssertEqual(before.model.name, "qwen7b")

        let two = one + [device(2, ramMB: 32_768, storageMB: 500_000)]
        let after = try XCTUnwrap(NMPModelSelector.pick(mesh: two, catalog: ladder))
        XCTAssertEqual(after.model.name, "qwen14b", "a joined device should enable a bigger model")
    }

    func testReturnsNilWhenNothingFits() {
        // 100 MB free disk: even the 400 MB 0.5B file won't fit.
        let mesh = [device(1, ramMB: 1_024, storageMB: 100)]
        XCTAssertNil(NMPModelSelector.pick(mesh: mesh, catalog: ladder))
    }

    func testHigherQuantWinsAtEqualParams() throws {
        let q4 = model("m-q4", params: 1_000_000_000, layers: 20, hidden: 1024, fileMB: 600, bplMB: 28)
        let q2 = model("m-q2", params: 1_000_000_000, layers: 20, hidden: 1024, fileMB: 400, bplMB: 18)
        let mesh = [device(1, ramMB: 32_768, storageMB: 500_000)]
        let pick = try XCTUnwrap(NMPModelSelector.pick(mesh: mesh, catalog: [q2, q4]))
        XCTAssertEqual(pick.model.name, "m-q4", "same params ⇒ higher precision wins")
    }

    // MARK: Real GGUF footprint + catalog

    func testRealGGUFFootprintIsPlausible() throws {
        guard let path = LlamaTestSupport.modelPath,
              let cand = NMPModelCatalog.candidate(path: path) else {
            throw XCTSkip("no GGUF model at NMP_LLAMA_MODEL / ~/models")
        }
        XCTAssertGreaterThan(cand.layerCount, 0)
        XCTAssertGreaterThanOrEqual(cand.hiddenSize, 256)
        XCTAssertGreaterThan(cand.bytesPerLayer, 0)
        XCTAssertGreaterThan(cand.totalParameters, 100_000_000)
        // Block weights are the bulk but not all of the file.
        XCTAssertLessThan(cand.bytesPerLayer * cand.layerCount, cand.fileBytes)
        XCTAssertGreaterThan(cand.bitsPerWeight, 1)   // some quantization
        XCTAssertLessThan(cand.bitsPerWeight, 33)     // not wider than f32
    }

    func testRealCatalogScanSortsByQuality() throws {
        let dir = ("~/models" as NSString).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: dir) else {
            throw XCTSkip("no ~/models directory")
        }
        let catalog = NMPModelCatalog.scan(directory: dir)
        guard catalog.count >= 2 else { throw XCTSkip("need ≥2 models in ~/models") }
        // Sorted highest-quality (most params) first.
        for (a, b) in zip(catalog, catalog.dropFirst()) {
            XCTAssertGreaterThanOrEqual(a.totalParameters, b.totalParameters)
        }
    }
}
