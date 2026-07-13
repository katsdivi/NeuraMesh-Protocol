//
//  WebUI.swift
//  NMP — Mesh 2.0
//
//  Everything that turns the local testing dashboard into a multi-device
//  web experience: LAN discovery (Bonjour service advert + real local
//  hostname/IPs), the startup banner with a scannable QR code, and the
//  protocol comparison model behind POST /api/comparison.
//
//  HOSTNAME HONESTY: advertising a Bonjour SERVICE named "neuramesh"
//  does NOT create a `neuramesh.local` hostname — mDNS hostnames belong
//  to the machine (its "Local Hostname" in System Settings ▸ Sharing).
//  Browsers resolve hostnames, not service names. So the banner prints
//  the Mac's REAL `<hostname>.local` plus its LAN IPs; the service
//  advert (`_neuramesh-ui._tcp`) additionally makes the UI discoverable
//  to Bonjour browsers. Renaming the machine to "neuramesh" is what
//  makes `http://neuramesh.local:3000` literally true — the setup guide
//  says so instead of pretending.
//
//  COMPARISON HONESTY: NMP's row is REAL — the measured wall clock,
//  payload, and round trips of the generation that just ran. TCP and
//  QUIC rows are MODELED: the same measured run re-priced with each
//  protocol's transport costs (handshake RTTs, per-trip overhead, loss
//  recovery), using this repo's own measured constants where they exist
//  (Noise IK loopback handshake ≈ 1.0 ms; Phase 3 FEC recovery ≈ 0.15 ms
//  vs NACK ≈ 9.4 ms) and documented coarse constants where they don't.
//  Every estimate carries `measured: false` and its assumptions — the UI
//  must render them as a model, not a benchmark.
//

import Foundation
import Network
#if canImport(CoreImage)
import CoreImage
#endif
#if canImport(SystemConfiguration)
import SystemConfiguration
#endif

// MARK: - Protocol comparison model

public enum NMPProtocolComparisonModel {

    public struct Inputs: Sendable {
        public var tokens: Int
        /// Application payload actually moved (both directions), measured.
        public var payloadBytes: Int
        /// Mesh round trips actually spent, measured.
        public var roundTrips: Int
        /// Wall clock of the real NMP generation.
        public var measuredTotalSeconds: TimeInterval
        /// Assumed LAN round-trip time for the modeled protocols.
        public var lanRTTMs: Double
        /// Packet loss rate to model recovery costs under (0 = clean).
        public var lossRate: Double

        public init(tokens: Int, payloadBytes: Int, roundTrips: Int,
                    measuredTotalSeconds: TimeInterval,
                    lanRTTMs: Double = 2.0, lossRate: Double = 0.0) {
            self.tokens = tokens
            self.payloadBytes = payloadBytes
            self.roundTrips = roundTrips
            self.measuredTotalSeconds = measuredTotalSeconds
            self.lanRTTMs = lanRTTMs
            self.lossRate = lossRate
        }
    }

    public struct Estimate: Sendable {
        public let name: String
        /// true = the actual run; false = model output.
        public let measured: Bool
        public let handshakeMs: Double
        /// Transport overhead added per mesh round trip (0 for the anchor).
        public let perTripOverheadMs: Double
        /// Cost of recovering ONE lost packet.
        public let lossRecoveryMs: Double
        public let totalMs: Double
        public let tokensPerSec: Double
        public let assumptions: String

        public var asJSONObject: [String: Any] {
            [
                "name": name,
                "measured": measured,
                "handshake_ms": rounded(handshakeMs),
                "per_trip_overhead_ms": rounded(perTripOverheadMs),
                "loss_recovery_ms": rounded(lossRecoveryMs),
                "total_ms": rounded(totalMs),
                "tokens_per_sec": rounded(tokensPerSec),
                "assumptions": assumptions,
            ]
        }

        private func rounded(_ value: Double) -> Double {
            (value * 100).rounded() / 100
        }
    }

    // Constants measured IN THIS REPO (see test logs / Phase 3 docs).
    /// Noise IK 1-RTT handshake, measured over UDP loopback.
    public static let nmpHandshakeMs = 1.0
    /// Phase 3: XOR-FEC reconstruction, drop → delivery.
    public static let nmpFECRecoveryMs = 0.15

    /// The measured NMP run re-priced per protocol. Element 0 is the real
    /// run; the rest are models anchored to it.
    public static func compare(_ inputs: Inputs) -> [Estimate] {
        let rtt = max(0.1, inputs.lanRTTMs)
        let trips = max(1, inputs.roundTrips)
        let nmpTotalMs = inputs.measuredTotalSeconds * 1000

        // Expected number of lost packets across the generation: data
        // chunks (≤1400 B each after headers) plus per-trip control.
        let packets = Double(max(1, inputs.payloadBytes / 1400 + trips * 2))
        let expectedLosses = inputs.lossRate * packets

        func estimate(name: String, measured: Bool, handshakeMs: Double,
                      perTripMs: Double, recoveryMs: Double,
                      assumptions: String) -> Estimate {
            let total = nmpTotalMs
                + (handshakeMs - Self.nmpHandshakeMs)
                + perTripMs * Double(trips)
                + (recoveryMs - Self.nmpFECRecoveryMs) * expectedLosses
            return Estimate(
                name: name, measured: measured,
                handshakeMs: handshakeMs, perTripOverheadMs: perTripMs,
                lossRecoveryMs: recoveryMs,
                totalMs: total,
                tokensPerSec: total > 0 ? Double(inputs.tokens) / (total / 1000) : 0,
                assumptions: assumptions)
        }

        let nmp = Estimate(
            name: "NMP", measured: true,
            handshakeMs: Self.nmpHandshakeMs, perTripOverheadMs: 0,
            lossRecoveryMs: Self.nmpFECRecoveryMs,
            totalMs: nmpTotalMs,
            tokensPerSec: nmpTotalMs > 0
                ? Double(inputs.tokens) / (nmpTotalMs / 1000) : 0,
            assumptions: "measured run (wall clock, payload, round trips); "
                + "handshake and FEC recovery measured in-repo")

        let tcp = estimate(
            name: "TCP+TLS 1.3", measured: false,
            handshakeMs: 2 * rtt + 1.0,   // SYN RTT + TLS 1.3 RTT + crypto
            perTripMs: 0.25,              // kernel framing, ACK interplay, HOL risk
            recoveryMs: max(1.5 * rtt, 10.0), // fast retransmit; RTO tail worse
            assumptions: "modeled: measured NMP run re-priced — 2×RTT+1 ms "
                + "handshake, +0.25 ms/trip, loss = fast retransmit "
                + "(≥1.5×RTT, floor 10 ms; RTO tails excluded)")

        let quic = estimate(
            name: "QUIC", measured: false,
            handshakeMs: rtt + 1.0,       // 1-RTT + TLS 1.3 crypto
            perTripMs: 0.15,              // userspace stack + per-packet crypto
            recoveryMs: 1.25 * rtt,       // NACK-driven retransmit
            assumptions: "modeled: measured NMP run re-priced — 1×RTT+1 ms "
                + "handshake, +0.15 ms/trip, loss = 1.25×RTT retransmit")

        return [nmp, tcp, quic]
    }
}

// MARK: - LAN identity

/// What this machine is actually reachable as on the local network.
public enum NMPLANIdentity {

    /// The machine's mDNS hostname ("<name>.local"). The Local Hostname
    /// (System Settings ▸ Sharing) is the authoritative source — it is
    /// what Bonjour actually answers to. gethostname() is only the
    /// fallback: on DHCP networks it can be the bare IP ("192.168.1.90"),
    /// which must not get ".local" glued onto it.
    public static func localHostname() -> String {
        // SCDynamicStore is macOS-only: SystemConfiguration exists on iOS
        // but ships without this API, so canImport alone is not enough
        // (found by the first iOS build of the package since Mesh 2.2).
        #if canImport(SystemConfiguration) && os(macOS)
        if let name = SCDynamicStoreCopyLocalHostName(nil) as String?,
           !name.isEmpty {
            return name + ".local"
        }
        #endif
        var buffer = [CChar](repeating: 0, count: 256)
        guard gethostname(&buffer, buffer.count) == 0 else { return "localhost" }
        let name = String(cString: buffer)
        if name.hasSuffix(".local") { return name }
        // A dotted-quad "hostname" is an IP — return it verbatim.
        let isIPv4 = !name.isEmpty && name.allSatisfy { $0.isNumber || $0 == "." }
        return isIPv4 ? name : name + ".local"
    }

    /// Non-loopback IPv4 addresses (Wi-Fi/Ethernet), for the banner and
    /// for phones whose browsers won't resolve .local.
    public static func localIPv4Addresses() -> [String] {
        var addresses: [String] = []
        var interfaces: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&interfaces) == 0, let first = interfaces else { return [] }
        defer { freeifaddrs(interfaces) }

        var cursor: UnsafeMutablePointer<ifaddrs>? = first
        while let entry = cursor {
            defer { cursor = entry.pointee.ifa_next }
            guard let addr = entry.pointee.ifa_addr,
                  addr.pointee.sa_family == UInt8(AF_INET),
                  entry.pointee.ifa_flags & UInt32(IFF_LOOPBACK) == 0,
                  entry.pointee.ifa_flags & UInt32(IFF_UP) != 0 else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(addr, socklen_t(addr.pointee.sa_len),
                           &host, socklen_t(host.count),
                           nil, 0, NI_NUMERICHOST) == 0 {
                let address = String(cString: host)
                if !address.isEmpty { addresses.append(address) }
            }
        }
        return addresses
    }
}

// MARK: - Bonjour advert

/// Advertises the web UI as a Bonjour service so Bonjour browsers (and a
/// future companion app) can find it without typing addresses. This does
/// NOT rename the machine — see the header comment.
public final class NMPWebUIBroadcaster {

    public static let serviceType = "_neuramesh-ui._tcp"

    public var onDiagnostic: ((String) -> Void)?

    private let serviceName: String
    private let port: UInt16
    private var listener: NWListener?
    private let queue = DispatchQueue(label: "nmp.webui.bonjour")

    public init(serviceName: String = "NeuraMesh", port: UInt16) {
        self.serviceName = serviceName
        self.port = port
    }

    /// Registers the advert (its own tiny listener on an ephemeral port —
    /// the TXT record carries the real UI port, so the web server's own
    /// socket is untouched).
    public func start() {
        let listener: NWListener
        do {
            listener = try NWListener(using: .tcp)
        } catch {
            onDiagnostic?("bonjour advert failed: \(error)")
            return
        }
        // "host" (Mesh 2.7) lets the peer app build the UI's URL straight
        // from the browse result — no throwaway resolution connection.
        let txt = NWTXTRecord(["port": String(port), "proto": "http",
                               "host": NMPLANIdentity.localHostname()])
        listener.service = NWListener.Service(
            name: serviceName, type: Self.serviceType, txtRecord: txt)
        listener.newConnectionHandler = { $0.cancel() } // advert only
        listener.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                self?.onDiagnostic?(
                    "bonjour: advertised '\(self?.serviceName ?? "")' "
                    + "(\(Self.serviceType)), UI port \(self?.port ?? 0)")
            case .failed(let error):
                self?.onDiagnostic?("bonjour advert failed: \(error)")
            default:
                break
            }
        }
        listener.start(queue: queue)
        self.listener = listener
    }

    public func stop() {
        listener?.cancel()
        listener = nil
    }
}

// MARK: - QR code

/// ASCII QR rendering via CoreImage's CIQRCodeGenerator (Apple-native —
/// no third-party dependency, and no hand-rolled Reed-Solomon).
public enum NMPQRCode {

    /// Unicode half-block rendering (2 QR rows per text line), or nil
    /// where CoreImage is unavailable.
    public static func ascii(for text: String) -> String? {
        #if canImport(CoreImage)
        guard let filter = CIFilter(name: "CIQRCodeGenerator") else { return nil }
        filter.setValue(Data(text.utf8), forKey: "inputMessage")
        filter.setValue("L", forKey: "inputCorrectionLevel")
        guard let image = filter.outputImage else { return nil }

        let context = CIContext(options: [.useSoftwareRenderer: true])
        guard let cgImage = context.createCGImage(image, from: image.extent) else {
            return nil
        }
        let width = cgImage.width
        let height = cgImage.height
        var pixels = [UInt8](repeating: 0, count: width * height)
        guard let bitmap = CGContext(
            data: &pixels, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
        bitmap.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        func dark(_ x: Int, _ y: Int) -> Bool {
            y < height && x < width && pixels[y * width + x] < 128
        }

        var lines: [String] = []
        var y = 0
        while y < height {
            var line = ""
            for x in 0..<width {
                switch (dark(x, y), dark(x, y + 1)) {
                case (true, true): line += "█"
                case (true, false): line += "▀"
                case (false, true): line += "▄"
                case (false, false): line += " "
                }
            }
            lines.append(line)
            y += 2
        }
        return lines.joined(separator: "\n")
        #else
        return nil
        #endif
    }
}

// MARK: - Startup banner

public enum NMPWebUIBanner {

    /// The multi-device access story, printed once at startup: real
    /// hostname, real IPs, QR code for the first URL.
    public static func render(port: UInt16, meshSummary: [String]) -> String {
        let hostname = NMPLANIdentity.localHostname()
        let ips = NMPLANIdentity.localIPv4Addresses()
        let primaryURL = "http://\(hostname):\(port)"

        var lines = [
            "",
            "════════════════════════════════════════════════════════════",
            "  NeuraMesh Web UI ready — open from any device on this Wi-Fi",
            "════════════════════════════════════════════════════════════",
            "",
            "  Mac / iPhone / iPad browser:",
            "      \(primaryURL)",
        ]
        if !ips.isEmpty {
            lines.append("  If .local doesn't resolve, use the IP:")
            for ip in ips.prefix(3) {
                lines.append("      http://\(ip):\(port)")
            }
        }
        lines.append("")
        for line in meshSummary {
            lines.append("  \(line)")
        }
        lines.append("")
        lines.append("  All devices see the same live mesh state.")
        lines.append("  Local network only — no TLS/auth; don't port-forward this.")

        if let qr = NMPQRCode.ascii(for: ips.first.map { "http://\($0):\(port)" }
                                        ?? primaryURL) {
            lines.append("")
            lines.append("  Scan from your phone:")
            for row in qr.split(separator: "\n") {
                lines.append("    \(row)")
            }
        }
        lines.append("")
        lines.append("  Install as an app (one-time, no Xcode): open the URL on the")
        lines.append("  phone, then Share ▸ Add to Home Screen. Next time, just tap")
        lines.append("  the NeuraMesh icon — it reconnects to this mesh by itself.")
        lines.append("════════════════════════════════════════════════════════════")
        lines.append("")
        return lines.joined(separator: "\n")
    }
}
