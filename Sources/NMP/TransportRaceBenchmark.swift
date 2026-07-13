//
//  TransportRaceBenchmark.swift
//  NMP — Phase B (justify NMP vs existing protocols)
//
//  Turns the one-shot NMPTransportRace into repeatable evidence: run the same
//  real 4-way race (NMP UDP full stack vs TCP vs TCP+TLS 1.3 vs QUIC) over
//  many trials across a few realistic mesh traffic shapes, and aggregate each
//  leg's wall-clock latency into p50/p95/mean. Emits a CSV for
//  Docs/Protocol_Comparison.md.
//
//  Honesty (same as the race itself): every number is measured over loopback
//  sockets on this machine — no radio time in ANY leg. Plain TCP is the
//  unencrypted floor NMP must beat while doing per-packet AES-256-GCM; TLS 1.3
//  is the like-for-like encrypted comparison. Real loss is NOT injected in
//  process: the legs bind in the 20000..<40000 band so `scripts/loss_lab.sh`
//  can shape them at the OS level — run this benchmark under it and pass the
//  intended rate as `condition` so the CSV records the setting alongside the
//  measured cost.
//
//  Traffic shapes are drawn from real KV-cached mesh generations: a decode
//  step ships ~n_embd floats per token (many small round trips), a prefill
//  ships the whole prompt's residual at once (one large round trip).
//

import Foundation

public enum NMPTransportRaceBenchmark {

    /// A named replay pattern (round trips × total application bytes).
    public struct Shape: Sendable {
        public let name: String
        public let plan: NMPTransportRace.Plan
        public init(name: String, roundTrips: Int, payloadBytes: Int) {
            self.name = name
            self.plan = NMPTransportRace.Plan(
                roundTrips: roundTrips, payloadBytes: payloadBytes)
        }
    }

    /// Representative mesh generations (n_embd ≈ 896 F32 ≈ 3.5 KB/activation):
    /// a bursty prefill and two chatty decode lengths.
    public static let defaultShapes: [Shape] = [
        Shape(name: "prefill-burst", roundTrips: 1, payloadBytes: 262_144),
        Shape(name: "decode-32", roundTrips: 32, payloadBytes: 229_376),
        Shape(name: "decode-128", roundTrips: 128, payloadBytes: 917_504),
    ]

    /// One leg's aggregated latency for one shape.
    public struct LegStat: Sendable {
        public let shape: String
        public let condition: String
        public let leg: String
        public let transport: String
        public let trials: Int
        public let roundTrips: Int
        public let bytesMoved: Int
        /// Percentiles of total time (handshake + transfer), in seconds.
        public let total: NMPLatencyStats
        /// Percentiles of per-round-trip transfer time, in seconds.
        public let perTrip: NMPLatencyStats
    }

    public struct Report: Sendable {
        public let condition: String
        public let trials: Int
        public let rows: [LegStat]

        public var csv: String {
            var out = "shape,condition,leg,transport,trials,round_trips,"
                + "bytes_moved,p50_ms,p95_ms,mean_ms,per_trip_p50_ms\n"
            for r in rows {
                out += [
                    r.shape, r.condition, r.leg,
                    "\"\(r.transport)\"", String(r.trials), String(r.roundTrips),
                    String(r.bytesMoved),
                    String(format: "%.3f", r.total.p50 * 1000),
                    String(format: "%.3f", r.total.p95 * 1000),
                    String(format: "%.3f", r.total.average * 1000),
                    String(format: "%.3f", r.perTrip.p50 * 1000),
                ].joined(separator: ",") + "\n"
            }
            return out
        }

        public var summaryLines: [String] {
            var lines = ["transport race — condition=\(condition), trials=\(trials)"]
            let byShape = Dictionary(grouping: rows, by: \.shape)
            for shape in byShape.keys.sorted() {
                lines.append("  \(shape):")
                for r in byShape[shape]!.sorted(by: { $0.total.p50 < $1.total.p50 }) {
                    lines.append(String(
                        format: "    %-10s p50 %7.2f ms   p95 %7.2f ms   mean %7.2f ms",
                        (r.leg as NSString).utf8String!,
                        r.total.p50 * 1000, r.total.p95 * 1000, r.total.average * 1000))
                }
            }
            return lines
        }
    }

    /// Runs `trials` races per shape and aggregates per leg. `condition`
    /// labels the network state (e.g. "clean", "loss2pct" while loss_lab.sh is
    /// active) and is recorded verbatim in the output. Blocking — call from a
    /// plain thread.
    public static func run(
        trials: Int = 20,
        condition: String = "clean",
        shapes: [Shape] = defaultShapes,
        timeout: TimeInterval = 20,
        progress: ((String) -> Void)? = nil
    ) throws -> Report {
        let trialCount = max(1, trials)
        var rows: [LegStat] = []

        for shape in shapes {
            // leg name -> (transport desc, [totalSec], [perTripSec], roundTrips, bytes)
            var totals: [String: [TimeInterval]] = [:]
            var perTrips: [String: [TimeInterval]] = [:]
            var transportOf: [String: String] = [:]
            var tripsOf: [String: Int] = [:]
            var bytesOf: [String: Int] = [:]
            var order: [String] = []

            for trial in 1...trialCount {
                progress?("\(shape.name): trial \(trial)/\(trialCount)")
                let result = try NMPTransportRace.runSync(plan: shape.plan, timeout: timeout)
                for leg in result.legs {
                    if totals[leg.name] == nil { order.append(leg.name) }
                    totals[leg.name, default: []].append(leg.totalMs / 1000)
                    perTrips[leg.name, default: []].append(leg.perTripMs / 1000)
                    transportOf[leg.name] = leg.transportDescription
                    tripsOf[leg.name] = leg.roundTrips
                    bytesOf[leg.name] = leg.bytesMoved
                }
            }

            for leg in order {
                guard let total = NMPLatencyStats(latencies: totals[leg] ?? []),
                      let perTrip = NMPLatencyStats(latencies: perTrips[leg] ?? []) else { continue }
                rows.append(LegStat(
                    shape: shape.name, condition: condition, leg: leg,
                    transport: transportOf[leg] ?? "", trials: trialCount,
                    roundTrips: tripsOf[leg] ?? shape.plan.roundTrips,
                    bytesMoved: bytesOf[leg] ?? shape.plan.payloadBytes,
                    total: total, perTrip: perTrip))
            }
        }
        return Report(condition: condition, trials: trialCount, rows: rows)
    }

    /// Convenience: run and write `transport_race_<condition>.csv` to `dir`.
    @discardableResult
    public static func runAndExport(
        trials: Int = 20, condition: String = "clean",
        to directory: URL, progress: ((String) -> Void)? = nil
    ) throws -> Report {
        let report = try run(trials: trials, condition: condition, progress: progress)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let safe = condition.replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: " ", with: "_")
        let url = directory.appendingPathComponent("transport_race_\(safe).csv")
        try report.csv.write(to: url, atomically: true, encoding: .utf8)
        return report
    }
}
