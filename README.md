# NeuraMesh Protocol (NMP)

Custom UDP-based transport protocol for distributed AI inference across Apple device meshes.
NACK-only reliability, XOR FEC, 1-RTT Noise IK handshake, AES-256-GCM per session.

**Status: Phases 1–6 + 8–9 complete.** Core transport (Phase 1: handshake + encryption +
sequencing), NACK-only reliability with a 64-packet retransmission window and sliding
replay window (Phase 2), XOR FEC over 4-packet groups + AWDL contention suppression
(Phase 3: sub-millisecond loss recovery, ~75× faster than the NACK path),
zero-configuration mesh assembly (Phase 4: Bonjour/mDNS discovery, capability
advertisement via TXT records, deterministic coordinator election), sharded multi-peer
inference (Phase 5: GGUF parsing, proportional layer sharding, pipelined orchestration,
bit-exact output at 1.02× single-device latency), production hardening (Phase 6:
peer-drop detection + failover with 0.4 ms re-sharding, stage retry under unrecoverable
loss, web testing dashboard, comprehensive benchmark suite), **real LLM inference
(Phase 8: llama.cpp behind the engine seam — quantized GGUF models, coordinator holds
only the tokenizer, weights on the peer, every token one real mesh round trip; see
`Docs/Phase8_Design.md`)**, and **the fast mesh (Phase 9: benchmark-driven adaptive
layer sharding with persisted device profiles, zero-trim + mixed-precision activation
wire formats, pipeline-parallel batch execution, draft/verify speculative decoding,
one-command auto-configuration; see `Docs/Phase9_Design.md`)**, plus **Mesh 2.0/2.1
(browser UI served by the coordinator itself — same live interface on every device
on the Wi-Fi, real-time token streaming to every open browser, ACTUALLY-measured
NMP-vs-TCP transport race, live device resource panel with compute-share sliders
that re-shard the mesh, web-client tracking, benchmarking center, QR/Bonjour
discovery; see `Docs/Mesh2_WebUI_Guide.md`)**.
**306 tests pass, 0 failures.**

## Requirements

- Xcode 14.2+ / Swift 5.8+
- macOS 13+ or iOS 16+ (Network.framework, CryptoKit)

## Build, Test & Run

```bash
cd NeuraMeshProtocol
swift build
swift test                             # 306 unit + loopback integration tests

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

### Fast mesh (Phase 9): one-command setup, compressed wire, speculation

```bash
swift run nmp-dashboard --auto-config              # reference mesh: probe → balance → persist
swift run nmp-dashboard --engine llamaCpp --model ~/models/model.gguf \
                        --auto-config              # llama: zero-trim wire (lossless, ~99% smaller)

# speculative decoding: drafts locally, verifies a whole draft in ONE round trip
curl -X POST localhost:8080/api/inference \
     -d '{"prompt":"...","max_tokens":32,"enable_speculation":true}'
# or serve everything speculatively, optionally with a small same-vocab draft model:
swift run nmp-dashboard --engine llamaCpp --model ~/models/model.gguf \
                        --auto-config --speculation [--draft-model small.gguf]

# real two-process mesh with the same levers:
swift run nmp-coordinator --engine llamaCpp --model ~/models/model.gguf \
                          --prompt "..." --tokens 32 --speculation --zero-trim
```

Auto-config benchmarks each device with real probe passes, balances layer
spans to the measured speeds, persists the profile (`~/.nmp/`), and picks
the wire format. Speculative output is token-for-token identical to plain
greedy output — drafts only ever save round trips, never change text.
Guide + measured results: `Docs/Phase9_Design.md`.

### Mesh 2.0/2.1: multi-device web UI, live streaming, measured race

```bash
swift run nmp-dashboard --ui           # browser UI on port 3000, all interfaces
swift run nmp-dashboard --ui --engine llamaCpp \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf --auto-config
```

The coordinator itself serves a React UI (prebuilt in `Public/` — npm only
needed to edit `web/`). The startup banner prints your Mac's real
`<hostname>.local:3000`, its LAN IPs, and a QR code; open it from Mac,
iPhone, iPad simultaneously — every tab shows the same live mesh. Views:
mesh dashboard, inference runner (speculation toggle, protocol comparison),
device panel, benchmark center, protocol comparison, chaos slider.
Trusted LAN only (no TLS/auth). Setup guide: `Docs/Mesh2_WebUI_Guide.md`.

Mesh 2.1 adds, on the same stack:

- **Real-time streaming** — every confirmed token is broadcast over the
  existing WebSocket as it is generated; a run submitted from the laptop
  appears token-by-token on the phone (and vice versa), with identical
  final metrics everywhere.
- **A measured transport race** (`POST /api/comparison/run`) — runs a real
  generation, then replays its exact traffic pattern over real loopback
  sockets: the full NMP stack (Noise IK + AES-256-GCM + FEC over UDP) vs
  plain kernel TCP. Both legs are wall-clock measurements; QUIC stays in
  the clearly-labeled model (it needs a TLS identity to race honestly).
- **Device panel** (`GET /api/devices/metrics`) — live kernel counters
  (RAM, storage, CPU, this process's footprint) plus per-peer mesh facts,
  and a **compute-share slider** (`POST /api/devices/:id/allocate`) that
  actually re-shards the live mesh: cap a device at 40% and watch its
  layer span shrink on every open browser. Reference mesh only — a llama
  plan is one full-range shard, and the API says so.
- **Web-client tracking** — `/health` reports `web_clients`; the phone
  shows up the moment it opens the page (`GET /api/clients` lists them).

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
| `AdaptiveSharding.swift` | Phase 9: probe-driven layer balancing, balance reporting, persisted device profiles |
| `OptimizedActivation.swift` | Phase 9: zero-trim (lossless) + mixed-precision (binary16 + critical f32) wire formats |
| `PipelinedInference.swift` | Phase 9: pipeline-parallel batch executor (independent sequences overlap across stages) |
| `SpeculativeDecoder.swift` | Phase 9: draft/verify speculative decoding — prompt-lookup + draft-model drafters |
| `AutoConfig.swift` | Phase 9: one-command setup — membership → benchmark → balance → wire format |
| `WebUI.swift` | Mesh 2.0: protocol comparison model (measured NMP + modeled TCP/QUIC), LAN identity, Bonjour advert, CoreImage QR banner |
| `TransportRace.swift` | Mesh 2.1: MEASURED protocol race — a run's traffic pattern replayed over the real NMP UDP stack vs plain kernel TCP |
| `ResourceMonitor.swift` | Mesh 2.1: live host resource sampling (Mach/BSD kernel counters) for the device panel |
| `web/` → `Public/` | Mesh 2.0/2.1: React UI source → committed build the coordinator serves (`--ui`) |

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

Phase 9 (validated 2026-07-10, Apple M3, Llama-2-7B-Chat Q4_K_M, 32 tokens):

- [x] One-command auto-config: membership → probe passes → balanced shards → wire format, profile persisted to `~/.nmp/` and reused (probe skipped on restart)
- [x] Adaptive sharding measurably rebalances a heterogeneous mesh (4×-slower peer gets a smaller span; bit-exact output preserved) — pinned by test
- [x] Zero-trim wire: 1 048 576 B → **11 928 B** per 32-token generation (−98.9%, lossless), remote throughput 13.4 → 14.0+ tok/s (≈ the 14.3–14.5 local baseline)
- [x] Mixed-precision wire: ~52% of raw for dense activations, top-2% outliers bit-exact, all 65 536 binary16 patterns pinned by test
- [x] Pipelined batch execution: ~2.4× measured overlap on a 3-stage mesh, outputs bit-identical to serial passes
- [x] Speculative decoding: 32 tokens in **8 round trips** (4.0 tok/trip, 100% draft acceptance) on repetitive text; output token-for-token identical to plain greedy in every configuration, adversarial drafter included
- [x] 0 regressions: **272 tests pass** (41 new: half-float bit-level, codecs, balance math, profiles, heterogeneous rebalancing, batch overlap, verify wire, drafters, toy-LM speculative service, real-model speculative identity)

Mesh 2.0 (validated 2026-07-10, Apple M3):

- [x] `--ui`: coordinator serves the React app on all interfaces (default :3000); same live state on every device (3 s polling + WebSocket pushes)
- [x] Startup banner prints the REAL `<hostname>.local` + LAN IPs + scannable CoreImage QR (no fake `neuramesh.local` claims); Bonjour advert `_neuramesh-ui._tcp`
- [x] REST surface: `/health`, `/api/devices`, `/api/benchmark/run` (avg + σ), `/api/comparison`, extended `/api/inference` (`round_trips`, `wire_format`, `enable_comparison`, `enable_speculation`) — CORS'd, SPA-served with traversal guard
- [x] Protocol comparison is honest: NMP row = the measured run; TCP/QUIC = that run re-priced with labeled modeled costs anchored to in-repo measurements (1.0 ms Noise IK handshake, 0.15 ms FEC recovery)
- [x] TinyLlama-1.1B draft model measured on Llama-2-7B: natural-text acceptance 0% (prompt-lookup) → **54–72%**, round trips 32 → **10–12**, payload → ~1.4 KB, output identical; wall clock loses in-process (shared GPU) and is documented as a physical-mesh win
- [x] Zero-dependency rule intact: no Vapor — the NWListener server grew the routes; QR via CoreImage; React toolchain isolated in `web/` with its build committed
- [x] 0 regressions: **291 tests pass** (19 new: comparison model math, routes, CORS, SPA fallback + traversal, benchmark σ, banner + QR)

Mesh 2.1 (validated 2026-07-11, Apple M3):

- [x] Real-time streaming: every confirmed token broadcast over `/ws` as generated (`generation_started/token/complete/failed`); verified end-to-end — a spectator WebSocket client receives the full token stream of a run submitted over HTTP by another client, final text identical
- [x] Measured transport race (`POST /api/comparison/run`): real generation + its traffic pattern replayed over real loopback sockets — full NMP stack (Noise IK + AES-256-GCM + FEC over UDP, chunked like mesh traffic) vs plain kernel TCP; every number wall-clock, zero modeled fields; llama zero-trim run measured at ~0.5 ms total NMP-vs-raw-TCP overhead for full encryption
- [x] Device panel: `GET /api/devices/metrics` serves live kernel counters (host_statistics64 RAM, statfs storage, CPU tick deltas, task_info process footprint) + per-peer plan/speed/share facts; honest about in-process peers sharing the host
- [x] Compute-share allocation actually allocates: `POST /api/devices/:id/allocate {"share":0.4}` re-plans through the normal SHARD_ASSIGN round — verified live: the capped peer's span shrank 6 → 3 layers, every browser sees the new plan + `allocation_update` push
- [x] Web-client tracking: `/health.web_clients` + `GET /api/clients` — a phone opening the page appears immediately (WebSocket) and ages out 15 s after leaving
- [x] Two pre-existing races found and fixed: unlocked `activePeers`/`activePlan` reads crashing under load (now lock-protected), and async `registerPeer` racing the adaptive controller's membership read (now settled via `waitForMembership`)
- [x] 0 regressions: **306 tests pass, verified over 5 consecutive full-suite runs** (15 new: resource monitor kernel counters, transport race byte/trip accounting, token streaming order, share-driven re-planning, new routes, WS generation events)

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
- `Docs/Phase9_Design.md` — Phase 9 fast mesh: adaptive sharding loop, the two wire
  formats and their determinism ledger, why single-stream pipelining is impossible
  (and what batching + speculation buy instead), draft/verify protocol, measured
  results, honest limits of prompt-lookup drafting
- `Docs/Mesh2_WebUI_Guide.md` — Mesh 2.0 multi-device web UI: architecture, zero-friction
  setup guide, hostname honesty (`.local` reality), measured-vs-modeled comparison rules,
  TinyLlama draft-model measurements
- `Docs/Benchmarks.md` — how to run the benchmark suite and interpret the results
- `Docs/CrossDevice_Setup_Guide.md` — Mac + iPhone mesh walkthrough
