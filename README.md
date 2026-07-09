# NeuraMesh Protocol (NMP)

Custom UDP-based transport protocol for distributed AI inference across Apple device meshes.
NACK-only reliability, XOR FEC, 1-RTT Noise IK handshake, AES-256-GCM per session.

**Status: Phases 1–6 + 8 complete.** Core transport (Phase 1: handshake + encryption +
sequencing), NACK-only reliability with a 64-packet retransmission window and sliding
replay window (Phase 2), XOR FEC over 4-packet groups + AWDL contention suppression
(Phase 3: sub-millisecond loss recovery, ~75× faster than the NACK path),
zero-configuration mesh assembly (Phase 4: Bonjour/mDNS discovery, capability
advertisement via TXT records, deterministic coordinator election), sharded multi-peer
inference (Phase 5: GGUF parsing, proportional layer sharding, pipelined orchestration,
bit-exact output at 1.02× single-device latency), production hardening (Phase 6:
peer-drop detection + failover with 0.4 ms re-sharding, stage retry under unrecoverable
loss, web testing dashboard, comprehensive benchmark suite), and **real LLM inference
(Phase 8: llama.cpp behind the engine seam — quantized GGUF models, coordinator holds
only the tokenizer, weights on the peer, every token one real mesh round trip; see
`Docs/Phase8_Design.md`)**. **231 tests pass, 0 failures.**

## Requirements

- Xcode 14.2+ / Swift 5.8+
- macOS 13+ or iOS 16+ (Network.framework, CryptoKit)

## Build, Test & Run

```bash
cd NeuraMeshProtocol
swift build
swift test                             # 231 unit + loopback integration tests

swift run nmp-peer                     # compute peer (cross-device mesh)
swift run nmp-coordinator              # coordinator + cross-device benchmark
swift run nmp-dashboard                # testing dashboard on http://localhost:8080
swift run nmp-dashboard --benchmark    # headless benchmark suite → Results/*.csv
```

### Real LLM inference (Phase 8)

```bash
brew install llama.cpp && scripts/setup_llama.sh    # one-time: build the shim

# single device / full-stack-in-one-process (POST /api/inference for text):
swift run nmp-dashboard --engine llamaCpp --model ~/models/model.gguf --placement local
swift run nmp-dashboard --engine llamaCpp --model ~/models/model.gguf   # remote shard

# real two-process / two-device mesh (weights on the peer, tokenizer on the coordinator):
swift run nmp-peer        --engine llamaCpp --model ~/models/model.gguf
swift run nmp-coordinator --engine llamaCpp --model ~/models/model.gguf \
                          --prompt "The capital of France is" --tokens 16
```

A llama plan is one full-range shard — llama.cpp cannot execute layer
sub-ranges, so "distributed" means real remote execution over the real
transport, with greedy sampling making output identical across placements.
Step-by-step guide, measured results, and the design rationale:
`Docs/Phase8_Design.md`.

Or open the folder directly in Xcode (`File > Open…`) — SwiftPM packages open natively;
no `.xcodeproj` is required or checked in. Cross-device setup (Mac coordinator +
iPhone peer): see `Docs/CrossDevice_Setup_Guide.md`.

## Modules

| File | Purpose |
|---|---|
| `PacketCodec.swift` | NMP packet header encode/decode (20-byte header, big-endian) |
| `NoiseIK.swift` | `Noise_IK_25519_AESGCM_SHA256` handshake, implemented from the Noise spec (verified against the published cacophony vector) |
| `SymmetricCrypto.swift` | Per-session AES-256-GCM with `nonce_seed ‖ seq` nonces, header AAD, 64-bit sliding replay window |
| `PeerConnection.swift` | Handshake state machine, retry/backoff, encrypted send/recv, NACK servicing, packet-event stream |
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
| `GGUF.swift` | Phase 5: GGUF v2/v3 container parsing (memory-mapped, hostile-count guards) |
| `ComputeEngine.swift` | Phase 5: engine seam (`NMPShardComputeEngine`) + deterministic reference engine |
| `ModelSharder.swift` | Phase 5: proportional layer apportionment (measured speed or class weights) |
| `ShardMessages.swift` | Phase 5: SHARD_ASSIGN + inference wire formats, tensor chunking/reassembly |
| `InferenceOrchestrator.swift` | Phase 5: coordinator pipeline walker; Phase 6: stage retry, activity feed |
| `PeerShardEngine.swift` | Phase 5: peer-side shard serving + metrics reporting |
| `PeerNode.swift` | Phase 5: turn-key peer/coordinator runtimes (CLIs + iOS app) |
| `PeerHealthMonitor.swift` | Phase 6: activity-based liveness, 5 s heartbeat timeout, injectable clock |
| `FaultToleranceOrchestrator.swift` | Phase 6: failover — drop/join re-sharding, health-check polling |
| `PacketLossInjector.swift` | Phase 6: deterministic loss/burst/blackhole transport decorator + in-memory loopback |
| `MeshTestbed.swift` | Phase 6: full in-process mesh (real crypto/FEC/NACK) for dashboard, benchmarks, tests |
| `BenchmarkSuite.swift` | Phase 6: scenario runner, p50/p95/p99, CSV export |
| `DashboardServer.swift` | Phase 6: HTTP + RFC 6455 WebSocket server on NWListener (zero dependencies) |
| `PromptInferenceService.swift` | Phase 6+: prompt → per-token mesh passes; Phase 8: codec-driven token loop with EOS early stop |
| `PromptCodec.swift` | Phase 8: text seam (`NMPPromptCodec`) — reference pseudo-text vs real-LLM codecs |
| `LlamaWire.swift` | Phase 8: token-state wire format riding inside activation tensors (exact-as-Float32) |
| `LlamaRuntime.swift` | Phase 8: dlopen binding to the C shim + `NMPLlamaModel` handle (weights or vocab-only) |
| `LlamaEngine.swift` | Phase 8: `NMPLlamaComputeEngine` (real forward passes, full-range shards) + llama prompt codec |
| `LlamaTestbed.swift` | Phase 8: single-shard mesh assembly — local baseline or full-stack in-process peer |

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

Phase 5:

- [x] Mesh output vs single device: target ±0.01 — measured **bit-exact** (deterministic reference engine)
- [x] Mesh latency <2× single device (measured 1.02× with emulated 5 ms/layer compute)
- [x] Inference correct under 11% injected loss (FEC + NACK repair, bit-exact)
- [x] Discovery → dial → handshake → shard-assigned, unattended, ~3 s
- [x] 0 regressions: 178 tests pass

Phase 6:

- [x] Peer drop detected within 5.5 s (5 s heartbeat + 1 s poll, counted from last packet)
- [x] Re-sharding completes in <500 ms (measured **0.4 ms**: plan + SHARD_ASSIGN ack round)
- [x] Inference output bit-exact after failover; all-peers-dead fails explicitly (no hang)
- [x] Dashboard on localhost:8080 — live peer state, inference monitor, loss slider, packet log
- [x] Benchmarks: loss ≤2% free; 5%/10%/15% → -49%/-55%/-81% throughput; burst recovery <1 s; peer join re-shard 0.4 ms
- [x] 0 regressions: **207 tests pass** (178 prior + 29 new)

Phase 8 (validated 2026-07-09, Apple M3, Qwen2.5-0.5B Q4_K_M):

- [x] llama.cpp loads quantized GGUF models behind `NMPShardComputeEngine` (shim dlopen'd, package stays dependency-free)
- [x] Real LLM text via POST /api/inference — `engine: "llamaCpp"`, coherent output, EOS-aware early stop
- [x] Remote full-range shard over the real stack: output IDENTICAL to single device (greedy determinism), payload measured
- [x] Two-process mesh over UDP + Bonjour: coordinator vocab-only, weights on the peer — 16.9–18.5 tok/s, per-token p50 ≈58 ms, 3/3 runs identical
- [x] Llama-2-7B-Chat Q4_K_M over the same real mesh: 8.7–12.1 tok/s, per-token p50 ≈68 ms, runs identical, coordinator stays tokenizer-sized
- [x] Protocol overhead honest and small: ≈8–17 ms/token, shrinking as model compute grows
- [x] 0 regressions: **231 tests pass** (12 new: wire format, engine, vocab-only, mesh identity, EOS accounting)

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
- `Docs/Phase5_Design.md` — Phase 5 sharding design: the compute seam, star-relay
  topology rationale, application-layer tensor chunking, measured overhead table
- `Docs/Phase6_Design.md` — Phase 6 fault tolerance: activity-based liveness, failover
  path, stage retry vs NACK give-up, dashboard architecture, what's deliberately out
- `Docs/Phase8_Design.md` — Phase 8 llama.cpp integration: why llama shards are
  full-range (no public sub-range API, KV-cache locality), the dlopen'd C shim, the
  token-state wire format, step-by-step testing guide with measured results
- `Docs/Benchmarks.md` — how to run the benchmark suite and interpret the results
- `Docs/CrossDevice_Setup_Guide.md` — Mac + iPhone mesh walkthrough
