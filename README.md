# NeuraMesh Protocol (NMP)

Custom UDP-based transport protocol for distributed AI inference across Apple device meshes.
NACK-only reliability, XOR FEC, 1-RTT Noise IK handshake, AES-256-GCM per session.

**Status: Phase 4 complete** — core transport (Phase 1: handshake + encryption + sequencing),
NACK-only reliability with a 64-packet retransmission window and sliding replay window
(Phase 2), XOR FEC over 4-packet groups + AWDL contention suppression (Phase 3:
sub-millisecond loss recovery, ~75× faster than the NACK path), and zero-configuration
mesh assembly (Phase 4: Bonjour/mDNS discovery, capability advertisement via TXT
records, deterministic coordinator election). 143 tests pass. Phases 5–6 (sharding,
fault tolerance) not yet started.

## Requirements

- Xcode 14.2+ / Swift 5.8+
- macOS 13+ or iOS 16+ (Network.framework, CryptoKit)

## Build & Test

```bash
cd NeuraMeshProtocol
swift build
swift test          # unit + loopback integration tests
```

Or open the folder directly in Xcode (`File > Open…`) — SwiftPM packages open natively;
no `.xcodeproj` is required or checked in.

## Modules

| File | Purpose |
|---|---|
| `PacketCodec.swift` | NMP packet header encode/decode (20-byte header, big-endian) |
| `NoiseIK.swift` | `Noise_IK_25519_AESGCM_SHA256` handshake, implemented from the Noise spec (verified against the published cacophony vector) |
| `SymmetricCrypto.swift` | Per-session AES-256-GCM with `nonce_seed ‖ seq` nonces, header AAD, 64-bit sliding replay window |
| `PeerConnection.swift` | Handshake state machine, retry/backoff, encrypted send/recv, NACK servicing |
| `Reliability.swift` | Phase 2: NACK payload codec, 64-packet retransmit ring, receiver loss tracker |
| `FECCodec.swift` | Phase 3: CRC32, word-wise XOR parity, parity packet wire format |
| `FECGroup.swift` | Phase 3: sender group builder + receiver reconstructor |
| `AWDLDetector.swift` | Phase 3: contention inference (loss rate + latency shift, hysteresis) |
| `TrafficShaper.swift` | Phase 3: defers non-critical data during inferred AWDL contention |
| `Capabilities.swift` | Phase 4: capability struct, binary + TXT encodings, local measurement |
| `CoordinatorElection.swift` | Phase 4: deterministic election (highest compute class, ties → lowest peerID) |
| `Bonjour.swift` | Phase 4: mDNS service publishing/browsing with capabilities in TXT records |
| `PeerDiscoveryManager.swift` | Phase 4: discovery + capability refresh + election orchestration |
| `UDPTransport.swift` | Network.framework UDP transport + transport abstraction for tests |

## Success Criteria

Phase 1 (validated 2026-07-07 on Apple Silicon macOS):

- [x] Handshake completes in <10 ms (measured 1.4 ms mock loopback, 2.3 ms real UDP loopback)
- [x] Packet encryption/decryption byte-perfect (round-trip tests)
- [x] Replay protection rejects duplicate packets
- [x] No crashes on malformed packets (fuzz-ish codec tests included)

Phase 2:

- [x] Lost packets recovered via NACK-triggered verbatim retransmit (measured ≈9 ms, target <100 ms)
- [x] Reordered packets inside the 64-packet window accepted; duplicates still rejected
- [x] Unrecoverable losses surfaced via `onUnrecoverableLoss` (Phase 3 FEC input)
- [x] Noise IK implementation matches the published cacophony known-answer vector byte-for-byte

Phase 3:

- [x] Parity computation <100 µs per 4×1400 B group (measured 6–16 µs)
- [x] FEC reconstruction <1 ms (measured ≈0.01 ms; end-to-end drop→delivery ≈0.15 ms)
- [x] ≥80% of losses at 2% loss rate recovered without NACK (measured 100%)
- [x] Recovery latency <50% of Phase 2 NACK path (measured ≈1%: 0.13 ms vs 9.4 ms)
- [x] AWDL suppression defers normal data, passes critical/FLUSH/control, backstop-flushes at 200 ms

Phase 4:

- [x] Bonjour discovery <2 s after service publish (measured 0.94 s over real mDNS)
- [x] Coordinator election deterministic — all peers agree, across all join orders
- [x] Capability encode/decode round trip byte-exact; trailing bytes ignored (extensible)
- [x] No manual peer IP configuration needed (Bonjour publishes + browses `_neuramesh._tcp`)
- [x] 0 regressions: 143 tests pass (108 Phase 1–3 + 35 new)

## Design Docs

- `Docs/NMP_Specification.md` — protocol spec (source of truth)
- `Docs/Phase1_Design.md` — Phase 1 decisions, tradeoffs, and flagged known issues
  (constant-time properties, nonce exhaustion at 2^32, clock-sync assumption)
- `Docs/Phase2_Design.md` — Phase 2 reliability design: verbatim-retransmit rationale
  (header-as-AAD ⇒ RETRANSMIT flag unusable, flagged for spec revision), coupled
  64-packet windows, NACK scheduling, remaining known issues
- `Docs/Phase3_Design.md` — Phase 3 FEC + AWDL design: parity wire format (explicit
  member list — why base+count and bare CRC32 group IDs don't survive interleaving),
  N=4 group-size tradeoff, zero-added-latency grouping, detection heuristics and their
  limits, measured benchmark table
- `Docs/Phase4_Design.md` — Phase 4 discovery design: Bonjour choice and TXT-record
  capability propagation, election algorithm and why load is excluded, capability
  measurement limits, known issues (mDNS on restricted networks, re-shard triggers
  flagged for Phase 5+)
