# Contributor Guide — Read This Before You Change Anything

You just opened this repo and want to make a change. This document is the
map: what NeuraMesh is, where every piece lives, the seams you plug into,
the rules you must not break, and the exact loop to build/test/change
safely. It complements — does not repeat — two neighbours:

- `Start_Here.md` — the **operator's manual** (which command to run, when).
- `Project_Overview.md` — **what's built and the measured results**.
- `Future_Plans.md` — what's **not** built yet (don't assume it exists).

If you read one thing, read `CLAUDE.md` at the repo root — it is the
non-negotiable ruleset, condensed. This guide is the human-friendly tour.

---

## 1. What NeuraMesh is, in three sentences

NeuraMesh Protocol (NMP) is a **custom UDP transport plus a distributed
inference runtime** that turns several Apple devices on a LAN into one
machine that can run a model too big for any of them alone. It ships model
layers across devices and passes only the small activation vector between
them per token — never the weights. It is Apple-native with **zero
third-party dependencies**, callback-style (no async/await), and every
number it reports is labeled measured-or-modeled.

---

## 2. The stack, bottom to top — and which file is what

NMP is layered. Each layer only talks to the one below it through a narrow
seam, so you can change one without understanding all of them. Bottom to top:

| Layer | What it does | Start reading in |
|---|---|---|
| **1. Transport** | UDP, 20-byte big-endian header, 1-RTT Noise IK handshake, AES-256-GCM sessions | `UDPTransport.swift`, `PacketCodec.swift`, `NoiseIK.swift`, `SymmetricCrypto.swift` |
| **2. Reliability** | NACK-only retransmit + XOR FEC over 4-packet groups; AWDL contention backoff | `Reliability.swift`, `FECCodec.swift`/`FECGroup.swift`, `AWDLDetector.swift`/`TrafficShaper.swift` |
| **3. Mesh assembly** | Bonjour/mDNS discovery, capabilities in TXT, deterministic coordinator election | `Bonjour.swift`, `Capabilities.swift`, `CoordinatorElection.swift`, `PeerDiscoveryManager.swift` |
| **4. Inference** | layer sharding, per-token pipeline walk, bit-exact verification | `ModelSharder.swift`, `ShardMessages.swift`, `InferenceOrchestrator.swift`, `PeerShardEngine.swift`, `PeerNode.swift` |
| **5. Fault tolerance** | drop/join re-sharding, health monitoring | `FaultToleranceOrchestrator.swift`, `PeerHealthMonitor.swift` |
| **6. Engines** | the compute seam — reference (shardable) + llama.cpp (real) | `ComputeEngine.swift`, `LlamaEngine.swift`, `LlamaRuntime.swift`, `GGUF.swift` |
| **7. Testbeds** | full in-process meshes with real crypto/FEC over in-memory loopback | `MeshTestbed.swift`, `LlamaTestbed.swift`, `PacketLossInjector.swift` |
| **8. Dashboard/Web** | hand-rolled HTTP + WebSocket, device metrics, the transport race | `DashboardServer.swift`, `WebUI.swift`, `ResourceMonitor.swift`, `TransportRace.swift` |

Executables live in `Sources/NMP{DashboardCLI,PeerCLI,CoordinatorCLI}/`.
The web UI source is `web/` (React/TS); its **built output is committed to
`Public/`** so the dashboard runs without npm.

---

## 3. The seams — where you actually plug things in

Most changes are one of these five. Find yours, change *only* that seam.

- **New compute backend?** Conform a class to `NMPShardComputeEngine`
  (`ComputeEngine.swift`): `layerCount`, `hiddenSize`,
  `runLayers(start:end:input:)`. Everything above (sharding, transport,
  UI) is engine-agnostic and untouched. The reference engine is the
  example; `LlamaEngine` is the real one.
- **New transport to race against?** The transport race
  (`TransportRace.swift`) already runs NMP-UDP vs TCP vs TCP+TLS 1.3 vs
  QUIC over real loopback sockets on a generation's real traffic pattern.
  Add a leg there; the race is engine-agnostic, so it works with reference
  *or* real-LLM traffic unchanged.
- **New sharding strategy?** `NMPShardingObjective` in `ModelSharder.swift`
  (today: `.capacityThenSpeed`, `.speed`). Add a case, teach
  `planDetailed` how it places layers, surface it in the objective toggle.
- **New wire message?** Packet types live in `ShardMessages.swift`
  (control plane) / `PacketCodec.swift` (transport). **Big-endian** for NMP
  packets; the llama token-state wire (`LlamaWire`) is Float32.
- **New dashboard metric or endpoint?** `DashboardServer.swift` routes;
  `web/src/api.ts` types the client; a component in `web/src/components/`
  renders it. Rebuild `web/` and commit `Public/` too.

---

## 4. The rules you must not break (the short version)

Full list in `CLAUDE.md`. The ones that bite newcomers:

1. **No async/await.** Callback style with serial `DispatchQueue`s. This is
   a spec rule, not a preference.
2. **Zero SwiftPM dependencies.** Apple-native only (Network.framework,
   CryptoKit, CoreImage, SystemConfiguration). llama.cpp is `dlopen`'d
   through a C shim — **never linked**.
3. **`NMP` prefix** on public types. **Big-endian** NMP wire formats.
4. **Honest measurement.** Measured vs modeled is *always* labeled — in
   code, UI, and docs. Never dress a constant or a model as "measured."
5. **Trusted-LAN only.** No TLS/auth on the dashboard; never port-forward
   it. Compute peers are the native app (browsers have no UDP).

---

## 5. The build / test / change loop

```bash
cd ~/neuramesh/NeuraMeshProtocol   # this is its OWN git repo (nested) — commit HERE

swift build                        # the real compile check
swift test                         # full suite (300+ tests, ~20 s)
swift test --filter Mesh21Tests    # one class
swift test --filter ModelSharderTests/testFasterPeersGetMoreLayers  # one method
```

- **`swift build` is the source of truth.** SourceKit "Cannot find type in
  scope" in your editor is usually cross-file indexing lag — build to
  confirm.
- **Run the suite more than once** for anything concurrency-adjacent. A
  single green run has hidden real flakes before.
- **Edited `web/`?** `cd web && npm run build` (runs `tsc --noEmit` + vite
  → `../Public/`), then commit **both** `web/` and `Public/`.
- **Changed behaviour?** Add/extend a test. The reference engine is
  deterministic and bit-exact, so correctness bugs surface as hard test
  failures, not float drift — lean on that.

---

## 6. Testing with the real LLM

By default the dashboard runs the **reference engine** (weightless, so it
can shard across many devices — the only engine that can today; see
`Future_Plans.md` #1). To test with a **real model**:

```bash
# one-time: build the dlopen shim (needs Homebrew llama.cpp)
brew install llama.cpp && scripts/setup_llama.sh   # → Vendor/llama/libnmpllama.dylib

# run the dashboard on a real GGUF (real tokens, real NMP transport):
swift run nmp-dashboard --ui --engine llamaCpp \
  --model ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf --auto-config

# enable the real-LLM test suite:
NMP_LLAMA_MODEL=~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf swift test --filter Llama
```

Or just: `scripts/run_real_llm.sh [model.gguf]`.

**Two honest limits to know before you rely on this:**

- **A real-LLM plan is one full-range shard.** llama.cpp can't run a layer
  sub-range, so the whole model sits on one peer (tokenizer on the
  coordinator, one real NMP round-trip per token). Splitting a model across
  devices so *neither holds all of it* is `Future_Plans.md` #1 — not built.
  Use the reference engine for the multi-device split story.
- **The transport race works with either engine.** `enable_comparison` on
  `/api/inference` (or `POST /api/comparison/run`) replays the real
  generation's traffic over real sockets — NMP-UDP vs TCP vs TLS 1.3 vs
  QUIC — so you can race the *real* model's per-token pattern against every
  other protocol. That path is engine-agnostic and stays that way.

---

## 7. Gotchas that cost real debugging time

These are in `CLAUDE.md` too, but they're the ones you'll hit:

- **WebSocket after upgrade:** frames sharing a TCP segment with the 101
  response break Safari/`URLSessionWebSocketTask`. Post-upgrade sends in
  `DashboardServer` are delayed 250 ms on purpose — keep that pattern.
- **macOS TIME_WAIT:** bind test/race listeners in `20000..<40000`, not the
  ephemeral range, or you hit deterministic `EADDRINUSE`.
- **Membership reads** must go through `membershipLock`/`waitForMembership()`
  — an unlocked cross-queue read of `activePeers` caused a real crash.
- **Hostname:** use `NMPLANIdentity.localHostname()`, not `gethostname()`.
- **llama.cpp:** `vocab_only` loads report 0 layers/hidden (recover via the
  GGUF parser); new shim symbols must be bound as **optional** `dlsym` so
  old dylibs keep working.

---

## 8. "I want to change X" → go here

| I want to… | Start at |
|---|---|
| plug in a different model engine | `ComputeEngine.swift` (the seam) |
| make sharding split differently | `ModelSharder.swift` → `NMPShardingObjective` |
| add a protocol to the race | `TransportRace.swift` |
| add a dashboard metric/endpoint | `DashboardServer.swift` + `web/src/api.ts` + a component |
| change the wire format | `PacketCodec.swift` / `ShardMessages.swift` (big-endian!) |
| touch handshake/crypto | `NoiseIK.swift` / `SymmetricCrypto.swift` (has published test vectors — keep them green) |
| run a real model | §6 above |
| understand a design decision | `Docs/Phase*_Design.md` for that phase |
| know what's NOT built | `Future_Plans.md` |
