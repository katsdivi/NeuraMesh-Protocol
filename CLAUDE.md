# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo location gotcha

This directory is its **own git repo nested inside** `~/neuramesh` (which itself sits under a larger repo at `~`). Always run git commands from inside `NeuraMeshProtocol` — commits belong here, not in the outer repos. Sibling folders `NMPzip/` and `ProtocolComparison/` are an old snapshot and a comparison harness, not active code.

## Commands

```bash
swift build
swift test                                        # full suite (306 tests)
swift test --filter Mesh21Tests                   # one test class
swift test --filter Mesh21Tests/testSpliceMath    # one test method

# Executables
swift run nmp-dashboard                # simulated mesh + dashboard on :8080
swift run nmp-dashboard --ui           # + React UI on :3000, all interfaces (LAN)
swift run nmp-dashboard --benchmark    # headless suite → Results/*.csv
swift run nmp-peer                     # compute peer (cross-device)
swift run nmp-coordinator              # coordinator + cross-device benchmark
scripts/setup_mesh_test.sh --realistic # one-command loopback mesh demo

# Real llama.cpp inference (Phase 8+)
brew install llama.cpp && scripts/setup_llama.sh          # one-time shim build → Vendor/llama/libnmpllama.dylib (gitignored)
NMP_LLAMA_MODEL=~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf swift test   # enables llama tests (XCTSkip without shim+model)
swift run nmp-dashboard --ui --engine llamaCpp --model ~/models/llama-2-7b-chat.Q4_K_M.gguf --auto-config
```

- Models live in `~/models/`: `qwen2.5-0.5b-instruct-q4_k_m.gguf` (fast tests), `llama-2-7b-chat.Q4_K_M.gguf` (validated 4.08 GB), `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (draft model — shares Llama-2's vocab; draft models must share the target's vocab).
- **Web UI**: React/TypeScript source in `web/`, but the **built output is committed to `Public/`** so the dashboard works without npm. After editing `web/`: `cd web && npm run build` (runs `tsc --noEmit` + vite, outputs to `../Public/`), then commit both. PWA icons regenerate via `scripts/make_pwa_icons.py` (stdlib-only).
- Capturing dashboard stdout in scripts requires a pty: `script -q <log> <cmd>` (block-buffered when redirected).
- A single green test run proves little — pre-existing flakes have hidden behind single runs before. For concurrency-adjacent changes, run the suite several times in a row.
- SourceKit "Cannot find type in scope" diagnostics are usually cross-file indexing lag; `swift build` is the real check.

## Hard rules (from the spec — do not violate)

- **No async/await.** Callback style with serial `DispatchQueue`s throughout.
- **Zero SwiftPM dependencies.** Apple-native only: Network.framework, CryptoKit, CoreImage, SystemConfiguration. No Vapor, no third-party anything. llama.cpp is bound via a `dlopen`'d C shim, never linked.
- **`NMP` prefix** on public types.
- **Big-endian wire formats** for NMP packets (GGUF is little-endian per its spec; llama token-state wire is Float32 values).
- **Honest measurement**: measured vs modeled numbers are always labeled as such in code, UI, and docs. Never present a constant or a model as "measured".
- **Security stance**: the dashboard/web UI is trusted-LAN only — no TLS, no auth. Never expose or port-forward it. The Bonjour TXT-advertised session key is trust-on-first-use for the benchmark mesh; production pins keys via `PeerConnectionConfig.authorizedStaticKeys`.

## Architecture

**NMP is a custom UDP transport protocol for distributed AI inference across Apple devices.** The stack, bottom to top:

1. **Transport** (`UDPTransport`, `PacketCodec`, `NoiseIK`, `SymmetricCrypto`): 20-byte big-endian header, 1-RTT Noise IK handshake, per-session AES-256-GCM with replay window.
2. **Reliability** (`Reliability`, `FECCodec`/`FECGroup`, `AWDLDetector`/`TrafficShaper`): NACK-only retransmission (64-packet ring) + XOR FEC over 4-packet groups; AWDL contention inference defers non-critical traffic.
3. **Mesh assembly** (`Bonjour`, `Capabilities`, `CoordinatorElection`, `PeerDiscoveryManager`): zero-config mDNS discovery, capabilities in TXT records, deterministic election.
4. **Inference** (`GGUF`, `ModelSharder`, `ShardMessages`, `InferenceOrchestrator`, `PeerShardEngine`, `PeerNode`): proportional layer sharding (measured seconds/layer, or class weights before measurements exist; both scaled by per-peer compute shares), pipeline walking, bit-exact verification.
5. **Fault tolerance** (`FaultToleranceOrchestrator`, `PeerHealthMonitor`): drop/join re-sharding. Membership reads must go through `membershipLock` / `waitForMembership()` — unlocked cross-queue reads of `activePeers` caused a real crash.
6. **Engines** — the compute seam is `NMPShardComputeEngine` (`ComputeEngine.swift`): a deterministic reference engine (shardable, used by most tests) and llama.cpp (`LlamaRuntime`/`LlamaEngine`, dlopen shim). **llama.cpp cannot execute layer sub-ranges** (KV cache is per-context), so a llama plan is always ONE full-range shard; "distributed" = tokenizer on the coordinator, weights on the peer, one real mesh round trip per token. Greedy sampling ⇒ identical output across placements — used as a correctness oracle everywhere (speculation, streaming, placement).
7. **Testbeds** (`MeshTestbed`, `LlamaTestbed`, `PacketLossInjector`): full in-process meshes with real crypto/FEC/NACK over an in-memory loopback — this is what the dashboard, benchmarks, and most integration tests run on.
8. **Dashboard/Web** (`DashboardServer`, `WebUI`, `ResourceMonitor`, `TransportRace`): hand-rolled HTTP + RFC 6455 WebSocket on NWListener. Serves the committed React build in `Public/` (`--ui`), streams `generation_*` token events over `/ws`, exposes device metrics + compute-share allocation (`POST /api/devices/:id/allocate` → re-shard live), and a **measured** transport race (`POST /api/comparison/run`: replays a generation's traffic over real loopback sockets, full NMP UDP stack vs plain kernel TCP; QUIC stays modeled — racing it honestly needs a TLS identity). The web app is an installable PWA served by the mesh itself; the service worker only registers in secure contexts, and the app must keep working without it (LAN http). A PWA is a control surface only — browsers have no UDP, so a phone contributing compute needs the native peer app (`Docs/CrossDevice_Setup_Guide.md`).

Design rationale and measured results per phase are in `Docs/Phase*_Design.md`; the operator's manual (every mode, how to test each feature, connecting peers, troubleshooting) is `Docs/Start_Here.md`.

## Known gotchas (each cost real debugging time)

- **CFNetwork WebSocket**: server frames sharing a TCP segment with the 101 upgrade response break `URLSessionWebSocketTask`/Safari. Post-upgrade broadcasts in `DashboardServer` are delayed 250 ms (`queue.asyncAfter`) — keep that pattern for any new post-upgrade sends.
- **macOS TIME_WAIT**: TCP listeners bound in the ephemeral range (49152+) hit deterministic `EADDRINUSE` after in-process socket churn. Bind test/race listeners in `20000..<40000` (see `TransportRace.swift`).
- **Hostname**: use `SCDynamicStoreCopyLocalHostName` (via `NMPLANIdentity.localHostname()`), not `gethostname()` — the latter can return a bare DHCP IP that must not get `.local` appended.
- **llama.cpp**: `vocab_only` loads report 0 layers/hidden (recover via the GGUF parser); ggml-metal aborts at exit unless handles are freed (the shim has an atexit sweep); new optional shim symbols must be bound as optional `dlsym` so old dylibs keep working.
- **CPU%/timing tests**: kernel tick deltas need real elapsed time — use poll loops, not fixed sleeps.
