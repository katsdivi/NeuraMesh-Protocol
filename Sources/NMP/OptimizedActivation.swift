//
//  OptimizedActivation.swift
//  NMP — Phase 9
//
//  Smaller activation messages. Phase 5 moves every tensor as raw
//  big-endian Float32 (hiddenSize × 4 bytes — 16 KB per direction for a
//  7B model), chunked into ≤1024-byte packets. Phase 9 adds two wire
//  formats behind one self-describing codec:
//
//  - .zeroTrimmed ("NMPZ"): LOSSLESS. Drops the trailing zero run and
//    re-pads on decode. Built for llama token-state vectors (LlamaWire),
//    which occupy 3 + n slots of a 4096-wide tensor and pad the rest with
//    zeros: a single-token request shrinks 16 KB → ~30 B, a 40-candidate
//    response 16 KB → ~350 B. Bit-exact round trip, so Phase 8's greedy
//    determinism guarantee is untouched.
//
//  - .mixedPrecision ("NMPH"): LOSSY (bounded). Stores every value as
//    IEEE-754 binary16 and keeps the top NMP_CRITICAL_RATIO of values (by
//    magnitude) at full Float32 — layer-norm-scale outliers survive, the
//    bulk pays ≤ 2^-11 relative rounding. ~50% smaller for dense
//    activation tensors (the reference-engine path). NOT for llama plans:
//    fp16 would round large token ids and near-tied logits.
//
//  Compatibility: `decode` sniffs a 4-byte magic and falls back to the
//  Phase 5 raw-Float32 layout, so a Phase 9 peer serves Phase 8
//  coordinators unchanged; NMPPeerShardEngine mirrors the request's
//  format in its response, so a coordinator only ever gets back what it
//  opted into. Raw activations cannot collide with the magics: both
//  decode as Float32 bit patterns > 10^31, far outside anything the
//  reference squash (|x| < 1) or LlamaWire (< 2^24) produces.
//
//  The half-float conversion is implemented manually (no Float16 type):
//  identical results on every architecture the mesh spans, and
//  round-to-nearest-even exactly as the IEEE conversion would.
//

import Foundation

// MARK: - Wire format selection

public enum NMPActivationWireFormat: String, Sendable {
    /// Phase 5 raw big-endian Float32 (the compatibility default).
    case float32
    /// Trailing-zero truncation — lossless, for sparse/token-state tensors.
    case zeroTrimmed
    /// binary16 bulk + Float32 critical values — bounded-loss, for dense
    /// activation tensors.
    case mixedPrecision
}

public enum NMPActivationCodecError: Error, Equatable, Sendable {
    case truncated(expectedAtLeast: Int, got: Int)
    case malformed(String)
}

// MARK: - Codec

public enum NMPActivationCodec {

    static let zeroTrimMagic: UInt32 = 0x4E4D_505A       // "NMPZ"
    static let mixedPrecisionMagic: UInt32 = 0x4E4D_5048 // "NMPH"

    /// Share of values kept at full precision in .mixedPrecision.
    public static let criticalRatio = 0.02

    // MARK: Encode

    public static func encode(_ floats: [Float],
                              format: NMPActivationWireFormat) -> Data {
        switch format {
        case .float32:
            return NMPTensorCodec.encode(floats)
        case .zeroTrimmed:
            return encodeZeroTrimmed(floats)
        case .mixedPrecision:
            return encodeMixedPrecision(floats)
        }
    }

    /// Decodes any of the three formats (magic-sniffed).
    public static func decode(_ data: Data) throws -> [Float] {
        let bytes = Data(data)
        if bytes.count >= 4 {
            switch bytes.readBigEndianUInt32(at: 0) {
            case zeroTrimMagic: return try decodeZeroTrimmed(bytes)
            case mixedPrecisionMagic: return try decodeMixedPrecision(bytes)
            default: break
            }
        }
        return try NMPTensorCodec.decode(bytes)
    }

    /// The format `data` is encoded in (for response mirroring).
    public static func formatOf(_ data: Data) -> NMPActivationWireFormat {
        guard data.count >= 4 else { return .float32 }
        switch Data(data).readBigEndianUInt32(at: 0) {
        case zeroTrimMagic: return .zeroTrimmed
        case mixedPrecisionMagic: return .mixedPrecision
        default: return .float32
        }
    }

    // MARK: Zero-trimmed (lossless)

    /// magic(u32) ‖ totalCount(u32) ‖ significantCount(u32) ‖
    /// significant Float32 bit patterns
    private static func encodeZeroTrimmed(_ floats: [Float]) -> Data {
        var significant = floats.count
        while significant > 0 && floats[significant - 1] == 0 {
            significant -= 1
        }
        var out = Data(capacity: 12 + significant * 4)
        out.appendBigEndian(zeroTrimMagic)
        out.appendBigEndian(UInt32(floats.count))
        out.appendBigEndian(UInt32(significant))
        for i in 0..<significant { out.appendBigEndian(floats[i].bitPattern) }
        return out
    }

    private static func decodeZeroTrimmed(_ bytes: Data) throws -> [Float] {
        guard bytes.count >= 12 else {
            throw NMPActivationCodecError.truncated(expectedAtLeast: 12, got: bytes.count)
        }
        let total = Int(bytes.readBigEndianUInt32(at: 4))
        let significant = Int(bytes.readBigEndianUInt32(at: 8))
        guard significant <= total else {
            throw NMPActivationCodecError.malformed(
                "significant \(significant) exceeds total \(total)")
        }
        guard bytes.count >= 12 + significant * 4 else {
            throw NMPActivationCodecError.truncated(
                expectedAtLeast: 12 + significant * 4, got: bytes.count)
        }
        var floats = [Float](repeating: 0, count: total)
        for i in 0..<significant {
            floats[i] = Float(bitPattern: bytes.readBigEndianUInt32(at: 12 + i * 4))
        }
        return floats
    }

    // MARK: Mixed precision (bounded loss)

    /// magic(u32) ‖ totalCount(u32) ‖ criticalCount(u32) ‖
    /// binary16 bit patterns (u16 × totalCount) ‖
    /// (index u32 ‖ Float32 bit pattern) × criticalCount
    private static func encodeMixedPrecision(_ floats: [Float]) -> Data {
        let criticalCount = floats.isEmpty
            ? 0 : max(1, Int((Double(floats.count) * criticalRatio).rounded(.up)))
        // Top-k by magnitude, index-ascending for a deterministic layout.
        let criticalIndices = floats.indices
            .sorted { abs(floats[$0]) != abs(floats[$1])
                ? abs(floats[$0]) > abs(floats[$1]) : $0 < $1 }
            .prefix(criticalCount)
            .sorted()

        var out = Data(capacity: 12 + floats.count * 2 + criticalCount * 8)
        out.appendBigEndian(mixedPrecisionMagic)
        out.appendBigEndian(UInt32(floats.count))
        out.appendBigEndian(UInt32(criticalIndices.count))
        for value in floats { out.appendBigEndian(NMPHalfFloat.encode(value)) }
        for index in criticalIndices {
            out.appendBigEndian(UInt32(index))
            out.appendBigEndian(floats[index].bitPattern)
        }
        return out
    }

    private static func decodeMixedPrecision(_ bytes: Data) throws -> [Float] {
        guard bytes.count >= 12 else {
            throw NMPActivationCodecError.truncated(expectedAtLeast: 12, got: bytes.count)
        }
        let total = Int(bytes.readBigEndianUInt32(at: 4))
        let criticalCount = Int(bytes.readBigEndianUInt32(at: 8))
        let needed = 12 + total * 2 + criticalCount * 8
        guard bytes.count >= needed else {
            throw NMPActivationCodecError.truncated(expectedAtLeast: needed, got: bytes.count)
        }
        var floats = [Float](repeating: 0, count: total)
        for i in 0..<total {
            floats[i] = NMPHalfFloat.decode(bytes.readBigEndianUInt16(at: 12 + i * 2))
        }
        let criticalBase = 12 + total * 2
        for c in 0..<criticalCount {
            let offset = criticalBase + c * 8
            let index = Int(bytes.readBigEndianUInt32(at: offset))
            guard index < total else {
                throw NMPActivationCodecError.malformed(
                    "critical index \(index) out of range \(total)")
            }
            floats[index] = Float(bitPattern: bytes.readBigEndianUInt32(at: offset + 4))
        }
        return floats
    }
}

// MARK: - binary16

/// Portable IEEE-754 binary16 conversion, round-to-nearest-even. Manual
/// (rather than the Float16 type) so every platform produces identical
/// bytes — the same reason the reference engine avoids libm.
public enum NMPHalfFloat {

    /// Float32 → binary16 bit pattern.
    public static func encode(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let exponent = Int((bits >> 23) & 0xFF)
        let mantissa = bits & 0x7F_FFFF

        if exponent == 0xFF { // Inf / NaN
            return sign | 0x7C00 | (mantissa != 0 ? 0x0200 : 0)
        }

        // Unbiased exponent, rebiased for binary16 (bias 15).
        let halfExponent = exponent - 127 + 15

        if halfExponent >= 0x1F { // overflow → Inf
            return sign | 0x7C00
        }
        if halfExponent <= 0 { // subnormal or zero
            if halfExponent < -10 { return sign } // underflow → ±0
            // Implicit leading 1, then shift into subnormal position.
            let full = mantissa | 0x80_0000
            let shift = UInt32(14 - halfExponent)
            var half = UInt16(full >> shift)
            // Round to nearest, ties to even.
            let remainder = full & ((1 << shift) - 1)
            let halfway = UInt32(1) << (shift - 1)
            if remainder > halfway || (remainder == halfway && half & 1 == 1) {
                half += 1 // may carry into the exponent — that is correct
            }
            return sign | half
        }

        var half = UInt16(halfExponent << 10) | UInt16(mantissa >> 13)
        let remainder = mantissa & 0x1FFF
        if remainder > 0x1000 || (remainder == 0x1000 && half & 1 == 1) {
            half += 1 // mantissa carry rolls into the exponent correctly
        }
        return sign | half
    }

    /// binary16 bit pattern → Float32 (exact — every half is a float).
    public static func decode(_ half: UInt16) -> Float {
        let sign = UInt32(half & 0x8000) << 16
        let exponent = UInt32((half >> 10) & 0x1F)
        let mantissa = UInt32(half & 0x3FF)

        if exponent == 0 {
            if mantissa == 0 { return Float(bitPattern: sign) } // ±0
            // Subnormal: normalize into the float32 exponent range.
            var e: UInt32 = 127 - 15 + 1
            var m = mantissa
            while m & 0x400 == 0 {
                m <<= 1
                e -= 1
            }
            return Float(bitPattern: sign | (e << 23) | ((m & 0x3FF) << 13))
        }
        if exponent == 0x1F { // Inf / NaN
            return Float(bitPattern: sign | 0x7F80_0000 | (mantissa << 13))
        }
        // Rebias 15 → 127. Add BEFORE anything subtracts: exponent is
        // unsigned and can be as low as 1.
        return Float(bitPattern: sign | ((exponent + 112) << 23) | (mantissa << 13))
    }
}
