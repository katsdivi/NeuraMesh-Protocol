//
//  TransportRaceBenchmarkTests.swift
//  NMP — Phase B
//
//  The benchmark runs REAL loopback sockets (no gating needed), so this
//  verifies the aggregation contract, not a superiority claim (which is the
//  measured finding, and depends on the network condition).
//

import XCTest
@testable import NMP

final class TransportRaceBenchmarkTests: XCTestCase {

    func testRaceBenchmarkAggregatesLegsAcrossTrials() throws {
        let shape = NMPTransportRaceBenchmark.Shape(
            name: "unit", roundTrips: 4, payloadBytes: 8_192)
        let report = try NMPTransportRaceBenchmark.run(
            trials: 2, condition: "unit", shapes: [shape], timeout: 20)

        XCTAssertEqual(report.trials, 2)
        XCTAssertFalse(report.rows.isEmpty)

        let legs = Set(report.rows.map(\.leg))
        XCTAssertTrue(legs.contains("NMP"), "NMP leg must be measured")
        XCTAssertTrue(legs.contains("TCP"), "plain TCP floor must be measured")

        for row in report.rows {
            XCTAssertEqual(row.shape, "unit")
            XCTAssertEqual(row.condition, "unit")
            XCTAssertEqual(row.trials, 2)
            XCTAssertGreaterThan(row.total.p50, 0, "\(row.leg) total must be measured")
            XCTAssertGreaterThanOrEqual(row.total.p95, row.total.p50)
        }

        // CSV is well-formed with the documented header.
        XCTAssertTrue(report.csv.hasPrefix(
            "shape,condition,leg,transport,trials,round_trips,bytes_moved,"
            + "p50_ms,p95_ms,mean_ms,per_trip_p50_ms"))
    }

    func testDefaultShapesCoverPrefillAndDecode() {
        let names = NMPTransportRaceBenchmark.defaultShapes.map(\.name)
        XCTAssertTrue(names.contains("prefill-burst"))
        XCTAssertTrue(names.contains { $0.hasPrefix("decode") })
    }
}
