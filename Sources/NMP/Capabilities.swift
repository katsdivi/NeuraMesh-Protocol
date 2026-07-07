//
//  Capabilities.swift
//  NMP — Phase 4
//
//  Capability advertisement: what a device brings to the mesh (RAM, compute
//  class, current load, inference throughput, supported model formats).
//
//  Two encodings, one source of truth:
//
//  1. Binary wire format (big-endian, versioned) — carried in the Noise
//     handshake payload and CAPABILITY_ADV (0x13) packets. Decoders IGNORE
//     trailing bytes beyond the fields they know, so future phases can
//     append fields without breaking deployed peers (forward compatible).
//
//  2. Bonjour TXT dictionary — key/value strings advertised via mDNS so
//     peers learn capabilities BEFORE any connection is made. TXT parsing
//     is lenient: unknown keys are ignored, optional keys default.
//
//  Quantization: `currentLoadPercent` is carried as a whole percent (u8)
//  and `maxInferenceTokensPerSecond` as centi-tokens/sec (u32) — encode →
//  decode is byte-exact, decode(encode(x)) == x only up to that precision.
//

import Foundation

// MARK: - Compute class

/// Coarse device performance tier. Raw value doubles as election priority
/// (higher = preferred coordinator), so the ordering is part of the wire
/// contract: low(0) < medium(1) < high(2).
public enum NMPComputeClass: UInt8, CaseIterable, Comparable, Sendable {
    /// iPhone SE, older iPads.
    case low = 0
    /// iPhone 14 class, iPad Air.
    case medium = 1
    /// M-series Macs, iPhone 15 Pro class, latest iPads.
    case high = 2

    /// Election priority — alias for the raw tier, named for readability
    /// at election call sites.
    public var priority: UInt8 { rawValue }

    public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    /// TXT-record string form ("high"/"medium"/"low").
    public var label: String {
        switch self {
        case .low: return "low"
        case .medium: return "medium"
        case .high: return "high"
        }
    }

    public init?(label: String) {
        switch label {
        case "low": self = .low
        case "medium": self = .medium
        case "high": self = .high
        default: return nil
        }
    }
}

// MARK: - Errors

public enum NMPCapabilitiesError: Error, Equatable, Sendable {
    case truncated(expectedAtLeast: Int, got: Int)
    case unsupportedVersion(UInt8)
    case unknownComputeClass(UInt8)
    case invalidUTF8
    case stringTooLong(Int)
    case tooManyModelFormats(Int)
}

// MARK: - Capabilities

public struct NMPCapabilities: Equatable, Sendable {
    /// Binary format version this implementation writes.
    public static let formatVersion: UInt8 = 1
    /// Fixed-length prefix: version(1) + peerID(4) + ramMB(4) + compute(1)
    /// + load(1) + tpsCenti(4) + nameLen(1).
    static let fixedPrefixByteCount = 16

    public var peerID: UInt32
    /// Human-readable model, e.g. "MacBook Pro M3", "iPhone 15 Pro".
    /// Encoded as UTF-8, max 255 bytes.
    public var deviceName: String
    /// Total physical RAM in MB.
    public var ramMB: UInt32
    public var computeClass: NMPComputeClass
    /// System load 0–100. Clamped and rounded to a whole percent on encode.
    public var currentLoadPercent: Double
    /// Measured or estimated inference throughput. Carried with 0.01 tok/s
    /// resolution (u32 centi-tokens/sec, so ≤ ~42.9M tok/s).
    public var maxInferenceTokensPerSecond: Double
    /// Supported model container formats, e.g. ["gguf", "safetensors"].
    /// Max 255 entries, each ≤255 UTF-8 bytes.
    public var modelFormats: [String]

    public init(
        peerID: UInt32,
        deviceName: String,
        ramMB: UInt32,
        computeClass: NMPComputeClass,
        currentLoadPercent: Double = 0,
        maxInferenceTokensPerSecond: Double = 0,
        modelFormats: [String] = []
    ) {
        self.peerID = peerID
        self.deviceName = deviceName
        self.ramMB = ramMB
        self.computeClass = computeClass
        self.currentLoadPercent = currentLoadPercent
        self.maxInferenceTokensPerSecond = maxInferenceTokensPerSecond
        self.modelFormats = modelFormats
    }

    // MARK: Binary wire format

    /// Wire layout (all multi-byte fields big-endian):
    ///
    ///   byte 0       FORMAT_VERSION (u8) = 1
    ///   bytes 1-4    PEER_ID (u32)
    ///   bytes 5-8    RAM_MB (u32)
    ///   byte 9       COMPUTE_CLASS (u8: 0 low, 1 medium, 2 high)
    ///   byte 10      LOAD_PERCENT (u8, 0-100)
    ///   bytes 11-14  TOKENS_PER_SEC (u32, centi-tokens/sec)
    ///   byte 15      NAME_LEN (u8), then NAME (UTF-8)
    ///   next         FMT_COUNT (u8), then FMT_COUNT × (LEN u8 ‖ UTF-8)
    ///   trailing     reserved for future fields — v1 decoders ignore it
    public func encode() throws -> Data {
        let nameBytes = Data(deviceName.utf8)
        guard nameBytes.count <= Int(UInt8.max) else {
            throw NMPCapabilitiesError.stringTooLong(nameBytes.count)
        }
        guard modelFormats.count <= Int(UInt8.max) else {
            throw NMPCapabilitiesError.tooManyModelFormats(modelFormats.count)
        }

        var out = Data(capacity: Self.fixedPrefixByteCount + nameBytes.count + 1)
        out.append(Self.formatVersion)
        out.appendBigEndian(peerID)
        out.appendBigEndian(ramMB)
        out.append(computeClass.rawValue)
        out.append(UInt8(currentLoadPercent.clamped(to: 0...100).rounded()))
        let tpsCenti = (maxInferenceTokensPerSecond * 100).rounded()
            .clamped(to: 0...Double(UInt32.max))
        out.appendBigEndian(UInt32(tpsCenti))
        out.append(UInt8(nameBytes.count))
        out.append(nameBytes)
        out.append(UInt8(modelFormats.count))
        for format in modelFormats {
            let bytes = Data(format.utf8)
            guard bytes.count <= Int(UInt8.max) else {
                throw NMPCapabilitiesError.stringTooLong(bytes.count)
            }
            out.append(UInt8(bytes.count))
            out.append(bytes)
        }
        return out
    }

    public static func decode(_ data: Data) throws -> NMPCapabilities {
        // Rebase so indices start at 0 regardless of slice offsets.
        let bytes = Data(data)
        guard bytes.count >= fixedPrefixByteCount else {
            throw NMPCapabilitiesError.truncated(
                expectedAtLeast: fixedPrefixByteCount, got: bytes.count)
        }
        let version = bytes[0]
        guard version == formatVersion else {
            throw NMPCapabilitiesError.unsupportedVersion(version)
        }
        guard let compute = NMPComputeClass(rawValue: bytes[9]) else {
            throw NMPCapabilitiesError.unknownComputeClass(bytes[9])
        }

        var cursor = fixedPrefixByteCount
        let name = try readString(bytes, length: Int(bytes[15]), cursor: &cursor)

        guard bytes.count >= cursor + 1 else {
            throw NMPCapabilitiesError.truncated(expectedAtLeast: cursor + 1, got: bytes.count)
        }
        let formatCount = Int(bytes[cursor]); cursor += 1
        var formats: [String] = []
        formats.reserveCapacity(formatCount)
        for _ in 0..<formatCount {
            guard bytes.count >= cursor + 1 else {
                throw NMPCapabilitiesError.truncated(expectedAtLeast: cursor + 1, got: bytes.count)
            }
            let length = Int(bytes[cursor]); cursor += 1
            formats.append(try readString(bytes, length: length, cursor: &cursor))
        }
        // Any bytes past `cursor` are fields from a future revision: ignored.

        return NMPCapabilities(
            peerID: bytes.readBigEndianUInt32(at: 1),
            deviceName: name,
            ramMB: bytes.readBigEndianUInt32(at: 5),
            computeClass: compute,
            currentLoadPercent: Double(bytes[10]),
            maxInferenceTokensPerSecond: Double(bytes.readBigEndianUInt32(at: 11)) / 100,
            modelFormats: formats
        )
    }

    private static func readString(_ bytes: Data, length: Int, cursor: inout Int) throws -> String {
        // `cursor` points at the first content byte (the length byte was
        // already consumed by the caller).
        guard bytes.count >= cursor + length else {
            throw NMPCapabilitiesError.truncated(expectedAtLeast: cursor + length, got: bytes.count)
        }
        guard let string = String(data: bytes.subdata(in: cursor..<cursor + length),
                                  encoding: .utf8) else {
            throw NMPCapabilitiesError.invalidUTF8
        }
        cursor += length
        return string
    }

    // MARK: Bonjour TXT dictionary

    /// TXT record key/value form. Keys are short by mDNS convention (TXT
    /// records should stay well under 1300 bytes).
    public func txtDictionary() -> [String: String] {
        [
            "v": String(Self.formatVersion),
            "id": String(peerID, radix: 16),
            "name": String(deviceName.prefix(63)),
            "ram": String(ramMB),
            "compute": computeClass.label,
            "load": String(Int(currentLoadPercent.clamped(to: 0...100).rounded())),
            "tps": String(format: "%.2f", maxInferenceTokensPerSecond),
            "fmt": modelFormats.joined(separator: ","),
        ]
    }

    /// Lenient TXT parse: `id` and `compute` are required (a peer we cannot
    /// key or rank is useless to the election); everything else defaults.
    /// Unknown keys are ignored. Returns nil if required keys are missing
    /// or malformed.
    public init?(txtDictionary dict: [String: String]) {
        guard let idString = dict["id"], let id = UInt32(idString, radix: 16),
              let computeLabel = dict["compute"],
              let compute = NMPComputeClass(label: computeLabel) else {
            return nil
        }
        let formats = (dict["fmt"] ?? "")
            .split(separator: ",")
            .map(String.init)
        self.init(
            peerID: id,
            deviceName: dict["name"] ?? "",
            ramMB: dict["ram"].flatMap(UInt32.init) ?? 0,
            computeClass: compute,
            currentLoadPercent: dict["load"].flatMap(Double.init)?.clamped(to: 0...100) ?? 0,
            maxInferenceTokensPerSecond: dict["tps"].flatMap(Double.init) ?? 0,
            modelFormats: formats
        )
    }
}

// MARK: - Local measurement

/// Measures this device's capabilities: physical RAM from the OS, compute
/// class estimated from platform + RAM, CPU load from Mach host statistics.
///
/// KNOWN LIMITATION (flagged in Phase4_Design.md): CPU load is a proxy for
/// inference capacity — GPU/ANE contention is invisible to it. Phase 5+
/// should refine by measuring actual inference latency.
public enum NMPSystemCapabilityProbe {

    public static func measure(
        peerID: UInt32,
        modelFormats: [String] = ["gguf"],
        loadSampler: NMPCPULoadSampler? = nil
    ) -> NMPCapabilities {
        let ramMB = UInt32(clamping: ProcessInfo.processInfo.physicalMemory / (1024 * 1024))
        return NMPCapabilities(
            peerID: peerID,
            deviceName: deviceModel(),
            ramMB: ramMB,
            computeClass: estimateComputeClass(ramMB: ramMB),
            currentLoadPercent: loadSampler?.samplePercent() ?? 0,
            maxInferenceTokensPerSecond: 0, // measured in Phase 5, estimated 0 until then
            modelFormats: modelFormats
        )
    }

    /// Hardware model string ("Mac14,9", "iPhone16,1"), falling back to the
    /// host name where sysctl is unavailable.
    public static func deviceModel() -> String {
        #if canImport(Darwin)
        for key in ["hw.model", "hw.machine"] {
            var size = 0
            guard sysctlbyname(key, nil, &size, nil, 0) == 0, size > 0 else { continue }
            var buffer = [CChar](repeating: 0, count: size)
            guard sysctlbyname(key, &buffer, &size, nil, 0) == 0 else { continue }
            let model = String(cString: buffer)
            if !model.isEmpty { return model }
        }
        #endif
        return ProcessInfo.processInfo.hostName
    }

    /// RAM-based tier estimate. Apple Silicon Macs and iPhone Pro devices
    /// carry ≥8 GB; the 6 GB band covers iPhone 14-class devices. Coarse by
    /// design — the election only needs a stable ordering, and Phase 5
    /// replaces estimates with measured throughput.
    public static func estimateComputeClass(ramMB: UInt32) -> NMPComputeClass {
        switch ramMB {
        case 8000...: return .high
        case 5500..<8000: return .medium
        default: return .low
        }
    }
}

/// Delta-based CPU load: each `samplePercent()` returns busy/total tick
/// ratio since the previous call (first call: since boot). Poll on the
/// capability-refresh cadence (5 s) for a meaningful window.
public final class NMPCPULoadSampler {
    #if canImport(Darwin)
    private var previousBusy: UInt64 = 0
    private var previousTotal: UInt64 = 0
    #endif

    public init() {}

    public func samplePercent() -> Double {
        #if canImport(Darwin)
        var info = host_cpu_load_info_data_t()
        var size = mach_msg_type_number_t(
            MemoryLayout<host_cpu_load_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(size)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &size)
            }
        }
        guard status == KERN_SUCCESS else { return 0 }
        let busy = UInt64(info.cpu_ticks.0) + UInt64(info.cpu_ticks.1) + UInt64(info.cpu_ticks.3)
        let total = busy + UInt64(info.cpu_ticks.2)
        defer { previousBusy = busy; previousTotal = total }
        let deltaTotal = total &- previousTotal
        guard deltaTotal > 0 else { return 0 }
        return Double(busy &- previousBusy) / Double(deltaTotal) * 100
        #else
        return 0
        #endif
    }
}

// MARK: - Clamping helper

extension Double {
    func clamped(to range: ClosedRange<Double>) -> Double {
        Swift.min(Swift.max(self, range.lowerBound), range.upperBound)
    }
}
