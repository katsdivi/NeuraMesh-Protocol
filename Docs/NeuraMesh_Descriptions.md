# NeuraMesh — What We Built & Complete Feature List

*Last updated 2026-07-17. Status markers: ✅ built & working, 🚧 partially
built / in progress, ❌ designed but not built. Every "measured" number
elsewhere in `Docs/` is a real wall-clock number, never modeled — see
`CLAUDE.md`'s "Honest measurement" rule. This document is the answer to
"does NeuraMesh do X?" — for the full narrative read `Project_Overview.md`;
for how to actually run each feature read `Start_Here.md`.*

---

## 1. What NeuraMesh is, in one paragraph

NeuraMesh turns the Apple devices you already own — a Mac, an iPhone, an
iPad — into one inference machine. It's a custom UDP transport protocol
(**NMP**) purpose-built for the pathological workload of distributed LLM
inference (many small encrypted round trips over lossy shared Wi-Fi,
between mutually-known peers, needing instant reconnection), plus a
distributed inference runtime on top: zero-config mesh assembly, real
layer-sharded LLM execution across devices, a live web dashboard/PWA,
and a native iPhone peer app. Zero third-party dependencies — Apple-native
frameworks only (Network.framework, CryptoKit, CoreImage).

**386+ tests, 0 failures**, Apple Silicon, no async/await (callback-style
on serial dispatch queues throughout, per the hard spec rule).

---

## 2. Repo map

| Path | What it is | Status |
|---|---|---|
| `NeuraMeshProtocol/` | **The active project.** Swift package: the NMP transport, the mesh/inference runtime, the CLIs (`nmp-peer`, `nmp-coordinator`, `nmp-dashboard`), the React web UI source (`web/`) and its committed build (`Public/`), and the iOS peer app (`NeuraMeshPeer/`). Its own git repo — always run git commands from inside it. | ✅ actively developed |
| `NeuraMeshProtocol/NeuraMeshPeer/` | Native iOS app (SwiftUI) — the real compute peer for phones/iPads; also a chat client and mesh controller. Checked-in Xcode project, build-verified. | ✅ actively developed |
| `NeuraMeshProtocol/web/` → `Public/` | React/TypeScript source for the mesh dashboard; built output committed so the coordinator serves it with zero npm dependency at runtime. | ✅ actively developed |
| `neuramesh-app/` | An earlier, separate Next.js/PostgreSQL prototype of a web app for the mesh (auth, jobs table, simulated devices/benchmarks). Superseded in practice by the Mesh 2.x dashboard built directly into `NeuraMeshProtocol/web`. | 🚧 stale (last touched 2026-07-08) |
| `NMPzip/` | Old snapshot of an early protocol design. Not active code. | ❌ inactive |
| `ProtocolComparison/` | Standalone Swift benchmark harness comparing NMP/TCP/QUIC/UDP in isolation (separate from the in-repo `TransportRace`/`--benchmark-race`). | 🚧 standalone tool, not part of the live mesh |

Everything below describes `NeuraMeshProtocol` — the thing that's actually alive.

---

## 3. Complete feature checklist

### 3.1 Transport & security (Phase 1)

- ✅ **Custom UDP transport protocol (NMP)** — 20-byte big-endian header, built on `Network.framework`.
- ✅ **1-RTT Noise IK handshake** (`Noise_IK_25519_AESGCM_SHA256`), implemented from spec and verified byte-for-byte against the published Cacophony known-answer vector. Measured 1.2–2.9 ms vs TCP+TLS 1.3's 20–26 ms and QUIC's 8–12 ms.
- ✅ **Per-session AES-256-GCM encryption** on every packet, header as AAD, unique `nonce_seed ‖ seq` nonces.
- ✅ **Replay protection** — 64-bit sliding replay window rejects duplicates/replays.
- ✅ **Static-key pinning** for production use (`PeerConnectionConfig.authorizedStaticKeys`); Bonjour-advertised keys are trust-on-first-use for the benchmark mesh only.

### 3.2 Reliability & loss recovery (Phase 2, 3, Mesh 2.6)

- ✅ **NACK-only reliability** — no ACKs, no sender timers; receivers detect gaps and request retransmission from a 64-packet retransmit ring. Measured recovery ≈9 ms.
- ✅ **XOR FEC (forward error correction)** — one parity packet per 4-packet group; a single loss is rebuilt receiver-side with **no round trip**, measured end-to-end ≈0.15 ms (≈75× faster than the NACK path).
- ✅ **Loss transparent up to 10%** — throughput and latency effectively unaffected; graceful degradation to 20% (−28%); honest breaking point at 25% (NACK rounds themselves start getting lost).
- ✅ **AWDL contention detection** — infers peer-to-peer Wi-Fi contention from loss-corroborated latency shifts (a pure latency spike with zero loss is *not* treated as contention — this was a real bug, now regression-pinned) and defers non-critical traffic; critical/tensor traffic is never deferred.
- ✅ **Link-adaptive chunking** (`NMPLinkKind`) — radio paths get MTU-safe 1350 B chunks + FEC + AWDL shaping; wired/loopback paths get kernel-ceiling ~9 KB datagrams with parity/shaping switched off (there's no airtime to protect).
- ✅ **Burst sending** (`sendBurst`/`sendBurstAsync`) — one queue hop per tensor, coalesced writes via `NWConnection.batch`, FLUSH marks the last chunk. Used by the orchestrator, shard engine, and the transport race.
- ✅ **Deterministic loss/burst/blackhole injection** (`PacketLossInjector`) for testing — a "chaos slider" in the dashboard, an in-memory loopback transport for the full test suite.

### 3.3 Mesh assembly & discovery (Phase 4)

- ✅ **Zero-configuration discovery** — Bonjour/mDNS (`_neuramesh._tcp`), no manual IP entry anywhere. Measured <1 s to discover a peer.
- ✅ **Capability advertisement** in mDNS TXT records (compute class, measured speed, free storage, etc.), extensible binary + TXT encodings.
- ✅ **Deterministic coordinator election** — every peer independently computes the same coordinator (highest compute class, ties broken by lowest peer ID) regardless of join order.
- ✅ **LAN peers join the live web dashboard** (Mesh 2.4) — the reference dashboard browses Bonjour, dials a real iPhone or second Mac over real UDP, and folds it into the web-visible mesh live.

### 3.4 Distributed inference — engines

- ✅ **Deterministic reference engine** — a weightless, bit-exact stand-in used as the correctness oracle everywhere (re-sharding, speculation, failover, streaming): output is placement-invariant, so "distributed == local, byte for byte" is an executable test.
- ✅ **Proportional layer sharding** — layer spans assigned proportional to measured seconds-per-layer (or class weights before measurements exist), scaled further by per-peer compute shares.
- ✅ **Real LLM inference via llama.cpp** (Phase 8, engine `llamaCpp`) — quantized GGUF models, bound via a `dlopen`'d C shim (package itself stays zero-dependency). llama.cpp can't execute layer sub-ranges, so this mode is always **one full-range shard**: tokenizer on the coordinator, full weights on one peer, one real encrypted mesh round trip per token. Verified: Llama-2-7B-Chat Q4_K_M at 8.7–12.1 tok/s over a real two-device UDP mesh, output token-for-token identical to single-device.
- ✅ **Real cross-device layer sharding of a real model** (engine `llamaShard`, M4 core → hardened through "Phase A/B/C") — a hand-built **ggml graph-surgery shim** (`nmp_shard_shim.c`) that constructs the transformer forward graph directly, runs an arbitrary block range `[start, end)`, and loads **only that range's weights**. This is the actual "no single device holds the whole model" story:
  - ✅ Bit-exact vs whole-model llama.cpp output (verified on Qwen2.5-0.5B, 2-way and 3-way splits).
  - ✅ Falsification-tested (zeroing the cross-device residual collapses output — proves the hand-off is load-bearing, not decorative).
  - ✅ Partial memory load proven live (e.g. 209 MB + 254 MB for a 24-layer split — neither peer holds all of it).
  - ✅ Per-shard persistent KV cache (ABI 2) — O(n) decode instead of reprocessing the full sequence each step; wire payload shrank from `n_embd × T` to `n_embd` per token.
  - ✅ Arch-generic from GGUF metadata — runs qwen2 (QKV bias) and qwen3 (QK-norm) via tensor detection; llama-arch models need a NORMAL-RoPE shim variant (not yet built).
  - ✅ Runs over the **real network**, not just in-process: `nmp-coordinator --engine llamaShard` + `nmp-peer --engine llamaShard` split one GGUF across real devices over UDP/Bonjour/Noise IK.
  - 🚧 Scaling to 14B-class models on real devices — the machinery is proven at 0.5B/1.5B; the 14B end-to-end run is gated only on disk space for the file (`scripts/setup_qwen14b.sh`), not on any missing capability.
  - 🚧 Real compute *on the iPhone* via this engine — the app auto-selects it when a `.gguf` + the embedded `nmpshard.xcframework` are present; the on-device Apple-signed run is the one unproven link.

### 3.5 Capacity-aware sharding & churn resilience (Mesh 2.8, Phase A)

- ✅ **Capacity-aware layer planning** — hard per-device layer ceilings derived from RAM and the model's real memory footprint; a model too big for one device is *forced* to spill across the mesh instead of failing.
- ✅ **Two switchable sharding objectives** (live, from the Devices tab or `POST /api/mesh/objective`): *Capacity + Speed* (spread across the mesh, balanced by measured speed — the default) and *Pure Speed* (pack the fastest device to its ceiling, route around peers that would only add a Wi-Fi hop).
- ✅ **Sharding plan preview** (`GET /api/mesh/plans`) — three candidate splits (speed / balanced / capacity) shown side-by-side with per-device layers/footprint/%RAM before you apply one.
- ✅ **Zero-layer standby, explained** — a peer assigned no layers gets an explicit "0 shards · standby" state with the measured reason (out of memory, not needed, or routed around for speed), never a silent hang.
- ✅ **Simulated peers retire automatically** the moment a real LAN device joins, so the mesh becomes genuinely cross-device instead of a real phone competing with fast in-process phantoms.
- ✅ **Churn-safe re-sharding** — join/leave events trigger correct re-splits; KV cache is invalidated and the sequence is re-prefilled from scratch on a re-shard so output stays bit-exact; commit-on-ack (a plan is only committed once every remote peer acknowledges it, so a rejected assignment can't strand live routing on an unassigned peer).
- ✅ **Peer drop detection & failover** (Phase 6) — activity-based liveness, 5 s heartbeat timeout, detected within 5.5 s; re-sharding completes in a measured 0.4 ms; output stays bit-exact after failover; all-peers-dead fails explicitly instead of hanging.

### 3.6 Adaptive model selection (Phase C)

- ✅ **Model catalog & real footprint reading** (`NMPModelCatalog`) — scans a models directory, reads each GGUF's exact per-layer memory footprint, parameter count, and quantization straight from the file (not guessed).
- ✅ **Highest-quality-model-that-fits selector** (`NMPModelSelector`) — checks both a *storage* ceiling (every hosting device needs disk for the file) and a *RAM* ceiling (the layer split must fit aggregate RAM with headroom), returns the model + plan + the reason.
- ✅ **Churn-driven re-selection** (`NMPAdaptiveModelController`) — every membership change re-runs the selector and reports what changed: unchanged / reshard (same model, new split) / switch model (reload a different GGUF) / no model fits. A device leaving degrades the mesh to a smaller model; the same device returning upgrades it back — verified in tests (14B → 7B → 14B across simulated churn).
- ✅ **Auto-select on boot** — `nmp-dashboard --engine llamaShard` with no `--model` picks the best model for the host automatically and shows the choice + reason in the event log.
- ✅ **Mesh-wide model sync** (Mesh 2.9) — `POST /api/models/select` accepts a bare model name (not just a Mac file path), so a phone can request the whole mesh switch models; a rejected switch leaves the mesh untouched, no partial state.

### 3.7 Weight vault streaming (Mesh 2.9 / Future Plan #3)

- ✅ **A peer with no local model can still hold real layers** — the coordinator (which holds the full GGUF) slices the model in-process (`NMPGGUFSlicer`, a zero-dependency Swift GGUF writer) and streams *only* the assigned shard's bytes to the peer over plain HTTP (`NMPVaultServer`, trusted-LAN). The peer caches the slice on disk, opens it with the real shard shim, and computes.
- ✅ **Disk ≈ RAM, not disk ≈ whole model** — a phone with no downloaded model stores roughly its own layers' worth of disk, not the full file. Proven live: a `nmp-peer` with zero local models streamed a 209 MB slice of a 469 MB model and produced correct, deterministic output.
- ✅ **No compute-engine or C-shim changes required** — a slice preserves global block names and `block_count` so it loads in the existing shim unmodified. Weights move once per assigned range (cached across joins), never per token.
- ✅ **iOS support** — the peer app streams the same way and shows the shard cache in its Models tab.

### 3.8 Speed levers (Phase 9)

- ✅ **Zero-trim wire format** — lossless compression of llama token-state tensors; measured **−98.9%** payload for a 32-token generation (1,048,576 B → 11,928 B).
- ✅ **Mixed-precision wire format** — binary16 (fp16) for bulk activations + exact fp32 for the top-2% outliers; ~52% of raw size, every one of the 65,536 binary16 bit patterns pinned by test.
- ✅ **Pipeline-parallel batch execution** — independent sequences overlap across pipeline stages; measured ~2.4× overlap on a 3-stage mesh, outputs bit-identical to serial passes.
- ✅ **Speculative decoding (draft/verify)** — a drafter proposes several tokens, the mesh verifies a whole draft in **one round trip**; output is token-for-token identical to plain greedy decoding in every configuration (never trades correctness for speed).
  - ✅ **Prompt-lookup drafting** (built in, no extra model) — shines on repetitive text: measured 32 tokens in 8 round trips (100% acceptance) on repetitive input.
  - ✅ **Small same-vocab draft-model drafting** (e.g. TinyLlama-1.1B drafting for Llama-2-7B) — measured 54–72% acceptance on natural text, round trips 32 → 10–12, payload down to ~1.4 KB.
- ✅ **One-command auto-configuration** (`--auto-config`) — probes each device with real passes, balances layer spans to the measured speeds, persists the device-speed profile to `~/.nmp/` (skipped on restart if already measured), and picks the wire format automatically.

### 3.9 Web dashboard / control surface (Mesh 2.x)

- ✅ **Coordinator-served React web UI** (`--ui`, port 3000 by default, all interfaces) — the same live app on every device on the Wi-Fi; zero npm needed at runtime (prebuilt into `Public/`).
- ✅ **Real hostname + LAN IP + scannable QR code** printed at startup (no fake `.local` claims).
- ✅ **Installable PWA** (Mesh 2.2) — scan the QR once, "Add to Home Screen," and from then on the app auto-discovers and reconnects to the mesh with zero re-pairing and no Xcode. (A PWA is a control surface only — browsers can't open UDP sockets, so a phone that should *contribute compute* still needs the native peer app.)
- ✅ **Real-time token streaming** (Mesh 2.1) — every confirmed token is broadcast over WebSocket as it's generated; a run submitted from one device appears token-by-token on every other open browser, with identical final metrics everywhere.
- ✅ **Live device panel** — real kernel counters (RAM, storage, CPU, this process's footprint via `host_statistics64`/`statfs`/task info), per-peer plan/speed/share facts, whole-machine GPU% (Mesh 2.3), honest handling of in-process peers (labeled as sharing the host, not drawn as fake separate hardware).
- ✅ **Compute-share allocation slider** — dragging a peer's share re-plans the live mesh through a real SHARD_ASSIGN round; verified live (capping a peer at 40% visibly shrinks its layer span on every open browser). Reference-mesh only — a llama full-range shard has nothing to reallocate, and the API says so rather than pretending.
- ✅ **Per-device telemetry** (Mesh 2.3) — live ↓/↑ network per link (measured at the wire boundary), requests served per peer, peer-reported resources over the mesh itself (`NMPPeerResourceReport`), phone-sized responsive layout.
- ✅ **Web-client tracking** — `/health.web_clients` and `GET /api/clients`; a phone opening the page shows up immediately and ages out after leaving.
- ✅ **Benchmark center** — N sequential generations → average tok/s, latency ± σ, per-run table.
- ✅ **Fully-measured 4-leg transport race** (Mesh 2.1 → 2.5/2.6) — a real generation's exact traffic pattern replayed over real sockets for all four legs: production NMP (Noise IK + AES-256-GCM + FEC over UDP) vs plain kernel TCP vs TCP+TLS 1.3 vs QUIC. Nothing is modeled — if a TLS identity can't be staged, that leg is skipped and says so. Repeatable benchmark harness (`--benchmark-race`) with p50/p95/mean per traffic shape, CSV export (Phase B).
- ✅ **Chaos slider** — injects real loss into the datapath on the reference mesh; watch FEC/NACK recovery events in real time in the event log.
- ✅ **Legacy peer-drop injection** on the pre-Mesh-2.0 dashboard page (`/legacy`) for manually testing failover.

### 3.10 Chat (Mesh 2.7)

- ✅ **Chat with the mesh** — `POST /api/chat` runs the same generation pipeline as `/api/inference`, assembling the prompt server-side from a whole conversation so every client shares one template per engine family (ChatML for qwen, `[INST]` for llama-2). The mesh itself stays stateless; clients resend the transcript each turn.
- ✅ **Persistent, local-first chat history** (added after Mesh 2.7) — each device stores only the chats it authored as one JSON file per conversation + an index manifest; quiet conversations auto-compress to LZFSE and transparently re-inflate on the next turn. CRUD over `/api/chats`.
- ✅ **Two-pane ChatGPT-style web UI** — sidebar of saved conversations + thread, streams replies over the existing token WebSocket, shows per-reply mesh stats (tok/s, round trips, payload).
- ✅ **iPhone chat tab** — the peer app both contributes compute *and* is a full chat client to the mesh; discovers the coordinator via the same Bonjour advert used for the UI.
- 🚧 **Cross-device chat sharing/pairing** (Phase 2 of the chat-history design) — a paired peer *browsing* another device's chat history live over the encrypted transport (never storing a copy locally), gated behind an explicit pairing/PIN. Designed, not built.

### 3.11 Native iPhone/iPad peer app (`NeuraMeshPeer`)

- ✅ **Real NMP compute peer** — the only way a phone can contribute actual compute (browsers can't open UDP sockets).
- ✅ **Checked-in, build-verified Xcode project** — no manual project assembly; ~5 minutes one-time signing setup, then a plain ⌘R.
- ✅ **Peer status tab** — live connection/shard state, "0 shards · standby" when idle, layer range + loaded MB when holding a real shard.
- ✅ **In-app model management** (Mesh 2.9) — download/select/delete GGUF models on-device (`ModelManager`/`ModelsView`), no manual file drop over USB required.
- ✅ **Auto-follow the mesh's model** — if the mesh switches models and the phone doesn't have it, it auto-fetches (local copy or vault streaming) instead of sitting shard-less.
- ✅ **Chat tab** with persistent local history (see §3.10), swipe-to-delete, LZFSE compression on backgrounding.
- ✅ **Headless rebuild/reinstall** via `xcodebuild` + `devicectl` — no Xcode UI required once signing is set up.

### 3.12 Testing & tooling

- ✅ **386+ tests, 0 regressions**, covering: Noise IK known-answer vectors, bit-level codec pins (every binary16 pattern), full-mesh integration under seeded loss, real-model bit-exactness (llama.cpp and the ggml shard shim), adaptive-model-controller churn scenarios, speculative-decoding correctness (including an adversarial drafter), measured performance gates.
- ✅ **Headless benchmark suite** (`--benchmark`) → CSV export with p50/p95/p99.
- ✅ **One-command loopback mesh demo** (`scripts/setup_mesh_test.sh --realistic`).
- ✅ **Real packet-loss lab** (`sudo scripts/loss_lab.sh`) — shapes actual OS-level loss (`dnctl`/`pfctl`) onto race ports for a non-simulated loss measurement.
- ✅ **In-process full-mesh testbeds** (`MeshTestbed`, `LlamaTestbed`) — real crypto/FEC/NACK over an in-memory loopback, what the dashboard/benchmarks/most tests actually run on.

---

## 4. What you can do today (task-oriented)

| You want to… | How |
|---|---|
| See the whole thing running in a browser on your phone and Mac at once | `swift run nmp-dashboard --ui` → scan the QR |
| Chat with a real local LLM distributed across your devices | `swift run nmp-dashboard --ui --engine llamaShard --model ~/models/….gguf --auto-config` |
| Run a model too big for any single device | Same as above with a model whose footprint exceeds one device's RAM — capacity-aware sharding spills it automatically; the mesh even lets a phone with zero local storage stream its slice from the vault |
| Watch the mesh survive a dropped peer | Chaos slider / legacy inject-drop page — re-shard completes in ~0.4 ms |
| Watch encryption cost basically nothing | Compare tab → real 4-leg transport race — NMP beats TLS 1.3 everywhere and ties/beats QUIC at inference-shaped traffic |
| See tokens stream to a second device in real time | Open Run on two browsers/devices, submit from one |
| Make a phone a real compute peer | Install `NeuraMeshPeer` on the phone (one-time signing), launch it, it joins over Bonjour automatically |
| Have the mesh pick the best model for what's connected right now | Omit `--model`; the adaptive selector auto-picks and re-decides on every join/leave |
| Get 4× fewer round trips per answer | `--speculation` (built-in prompt-lookup drafting, or `--draft-model` for a small same-vocab drafter) |
| Shrink the wire traffic ~99% | `--auto-config` turns on the zero-trim wire format automatically for llama engines |

---

## 5. The stack, bottom to top

1. **Transport** — `UDPTransport`, `PacketCodec`, `NoiseIK`, `SymmetricCrypto`.
2. **Reliability** — `Reliability` (NACK), `FECCodec`/`FECGroup` (XOR parity), `AWDLDetector`/`TrafficShaper` (link-aware shaping).
3. **Mesh assembly** — `Bonjour`, `Capabilities`, `CoordinatorElection`, `PeerDiscoveryManager`.
4. **Inference** — `GGUF`, `ModelSharder`, `ShardMessages`, `InferenceOrchestrator`, `PeerShardEngine`, `PeerNode`.
5. **Fault tolerance** — `PeerHealthMonitor`, `FaultToleranceOrchestrator`.
6. **Engines** — the compute seam `NMPShardComputeEngine`: the deterministic reference engine, `LlamaEngine` (llama.cpp, full-range shard only), `LlamaShardEngine` (real ggml graph-surgery sub-range sharding, KV-cached), `VaultShardEngine` (weight-streaming variant).
7. **Adaptive layer** — `AdaptiveSharding`, `NMPModelCatalog`/`NMPModelSelector`/`NMPAdaptiveModelController`, `OptimizedActivation` (zero-trim/mixed-precision), `PipelinedInference`, `SpeculativeDecoder`, `AutoConfig`.
8. **Testbeds** — `MeshTestbed`, `LlamaTestbed`, `PacketLossInjector`.
9. **Dashboard/Web** — `DashboardServer` (hand-rolled HTTP + WebSocket on NWListener), `WebUI`, `ResourceMonitor`, `TransportRace`, `TLSIdentity`, `NMPChatStore`, `NMPGGUFSlicer`/`NMPVaultServer`; served UI in `Public/` (source in `web/`).
10. **Native peer app** — `NeuraMeshPeer` (SwiftUI): `PeerViewModel`, `ChatView`/`ChatStore`, `ModelsView`/`ModelManager`, `PeerStatusView`.

---

## 6. Honest limitations (things that don't work yet, stated plainly)

- **llama.cpp-mode (`engine llamaCpp`) is always one full-range shard** — it cannot execute layer sub-ranges. Real sub-range sharding needs the newer `llamaShard` engine instead.
- **`llamaShard` sub-range engine is proven at 0.5B/1.5B scale, not yet run end-to-end at 14B on real devices** — gated on model download size, not on missing capability.
- **Real on-device compute on iPhone via the ggml shard shim is unproven** — the framework build + physical-device run is the one link not yet validated with real Apple signing.
- **The dashboard/web UI is trusted-LAN only** — no TLS, no auth, never to be port-forwarded, by design.
- **A browser tab is a control surface, not a compute peer** — browsers have no UDP sockets; only the native app contributes compute.
- **Loss recovery breaks down at ~25% sustained loss** — real Wi-Fi rarely sustains that, but it's the measured edge.
- **AWDL heuristics were tuned on current hardware** and need re-validation on other device generations.
- **Cross-device chat *sharing* (browsing another paired device's history) is designed but not built** — today each device only ever sees the chats it authored.
- **Known open bug (as of 2026-07-17, see `Docs/Fable_Findings.md`)**: generation on a Qwen2.5-1.5B model can segfault the coordinator on the first token (100% reproducible on the tree tested); sits on uncommitted inference-path changes and is under triage. The 0.5B model is unaffected. There is currently **no supervisor** — a coordinator crash requires a manual relaunch.
- **Borrowed remote peers (a rented GPU/second machine over the internet)** and **standalone weight-vault-only nodes as a first-class role** are designed (`Future_Plans.md`) but not built.

---

## 7. Roadmap (see `Docs/Future_Plans.md` for full detail)

- Scale real cross-device sharding to 14B-class models end to end on physical hardware.
- Validate the iPhone's real ggml-shim compute path on a signed physical device.
- Borrowed/remote peers — join a mesh over the internet as an offload peer (explicit dial + key pinning, honestly labeled by its higher RTT), never over SSH/TCP.
- Real-Wi-Fi (non-loopback) loss validation of the transport race and the Mac+iPhone mesh.
- Adaptive FEC parity rate (scale to measured loss instead of a fixed rate).
- Connection migration across network changes without a re-handshake.
- KV-cache streaming / prefill sharding for long-context workloads.
- Cross-device chat history browsing behind explicit pairing.

---

## 8. Where to read more

| Doc | For |
|---|---|
| `Start_Here.md` | the operator's manual — every launch mode, UI tour, testing playbook, peer setup, troubleshooting |
| `Project_Overview.md` | the narrative: core problem, architecture, measured numbers, design rationale |
| `Future_Plans.md` | the honest roadmap with status per item |
| `Mesh28_CapacitySharding.md` | capacity-aware sharding design |
| `Mesh2_WebUI_Guide.md` | web UI architecture and endpoint reference |
| `Phase1`–`Phase9_Design.md` | protocol internals per phase (crypto, reliability, FEC, discovery, sharding, failover, llama, speed levers) |
| `NMP_Specification.md` | the wire protocol, source of truth |
| `Benchmarks.md` / `CaseStudy_PacketLoss.md` / `Protocol_Comparison.md` | every measured number, and the loss-resilience/transport-race deep dives |
| `CrossDevice_Setup_Guide.md` | Mac + iPhone click-by-click setup |
| `Fable_Findings.md` | latest full-surface test report and open bugs |
| `CLAUDE.md` | hard rules and architecture for anyone (human or Claude) working on this code |
