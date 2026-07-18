# NeuraMesh Protocol (NMP)

**Turn the Apple devices you already own into one AI machine.**

NeuraMesh is a custom UDP transport protocol *and* a distributed-inference
runtime that stitches your Mac, iPhone, and iPad into a single private mesh —
so their combined RAM and compute can run models none of them could load
alone, entirely on your LAN, with nothing leaving the network. It is
Apple-native to the metal: **Network.framework + CryptoKit only, zero
third-party SwiftPM dependencies, no async/await** — a hand-built stack from
the 20-byte wire header up to a live web dashboard.

> **A note on the numbers in this README.** This project has one non-negotiable
> rule: *measured and modeled numbers are always labeled as such — never
> present a constant or a model as "measured."* Every performance figure below
> is a wall-clock measurement from the repo's own docs, carried with its label
> (and its conditions — most transport numbers are **loopback**, which isolates
> protocol cost from radio time). Where the docs give no hard number, the
> behavior is described qualitatively rather than invented.

---

## Table of contents

1. [Why this exists](#why-this-exists)
2. [What you can actually do with it](#what-you-can-actually-do-with-it)
3. [The distributed memory mesh (Supermemory)](#the-distributed-memory-mesh-supermemory)
4. [How we beat TCP (and TLS, and QUIC)](#how-we-beat-tcp-and-tls-and-quic)
5. [Architecture](#architecture)
6. [Getting started](#getting-started)
7. [Project layout](#project-layout)
8. [Status & tests](#status--tests)
9. [Documentation index](#documentation-index)

---

## Why this exists

Modern LLMs don't fit on one consumer device — but households and teams own
*several*. A Mac, an iPhone, an iPad sitting idle in the same room have enough
combined memory to serve a model none of them can hold individually. The only
thing standing between them and acting as one machine is **the network**.

Distributed inference is a pathological workload for off-the-shelf transports:

- **Every token is a round trip.** Generating one token means shipping a hidden
  state to a peer, waiting for its layers, and shipping the result back — for
  *every* token. Latency compounds 32× over a 32-token answer, so per-trip cost
  is everything.
- **Wi-Fi loses packets, and TCP answers loss with head-of-line blocking.** One
  dropped segment stalls every byte behind it for a full retransmit round trip
  — the worst possible failure mode for a pipeline that lives and dies by
  per-trip latency.
- **TLS handshakes are heavy, and mesh peers come and go.** A phone joins,
  leaves the room, rejoins. Re-establishing TCP+TLS 1.3 costs ~20–26 ms per
  connection (measured, loopback — see below); a mesh that re-handshakes often
  pays that constantly.
- **The mesh shares the airwave.** Peer-to-peer Wi-Fi (Apple's AWDL) contends
  with the very traffic it carries; a transport that blasts blindly degrades
  its own link.

Nothing off the shelf is shaped for *"many encrypted small-tensor round trips
over lossy shared radio, between mutually-known peers, with instant
reconnection."* So NeuraMesh is the transport that is.

---

## What you can actually do with it

These aren't hypotheticals — each maps to a runnable mode in the repo.

### Run a bigger model than any one device can hold

NeuraMesh apportions an LLM's **layers** across peers proportionally to each
device's *measured* speed (`AdaptiveSharding` probes real forward passes and
persists a per-device profile). The coordinator walks the layer pipeline
peer-to-peer, shipping activation tensors as chunked, encrypted payloads; a
**weight vault** streams a peer only the layers it was assigned, so a device
stores roughly *its slice* of the model, not the whole thing (disk ≈ RAM). The
combined memory of your Mac + iPhone + iPad runs a model that would never load
on one of them — fully local, no cloud, private by construction.

> **Honest scope.** The deterministic *reference* engine shards layers across
> any number of peers and is bit-exact across any split (it's the correctness
> oracle). The real-LLM path uses **llama.cpp**, which cannot execute layer
> sub-ranges (its KV cache is per-context), so a llama mesh today is
> *tokenizer-on-coordinator + weights-on-peer*: genuine remote execution over
> the real transport, one encrypted round trip per token — but not yet N-way
> layer splitting for real weights. True multi-device splitting of real models
> waits on engine support (an MLX-style backend behind the same seam). This is
> documented plainly, never glossed. See `Docs/Phase8_Design.md`.

### A private AI memory that survives losing a device

Your assistant's conversational memories are **erasure-coded across your
devices** — no single device holds a complete readable copy, and losing (or
having someone seize) one device doesn't lose your memory. See the dedicated
section below.

### Offline / air-gapped / privacy-sensitive AI

Everything runs on the LAN. The Supermemory servers are hard-guarded to
localhost; the dashboard is trusted-LAN only. No data leaves your network —
suitable for offline, air-gapped, or privacy-sensitive settings.

### A family / team / household compute pool

Bonjour zero-config discovery means no IPs are typed anywhere: peers publish
`_neuramesh._tcp` with their capabilities in TXT records, discovery takes
under a second, and a deterministic election picks the coordinator. Idle Apple
devices on the same Wi-Fi assemble themselves into a mesh.

---

## The distributed memory mesh (Supermemory)

NeuraMesh includes a **distributed memory layer** built on Supermemory, so an
assistant's long-term memory is private *and* resilient across your devices.
Full details: `Docs/Memory_Mesh.md`; on-device iPhone backend:
`Docs/Memory_Mesh_iOS.md`.

### Each device runs its own local Supermemory — never the cloud

Every device runs its **own local, self-hosted `supermemory-server`, bound to
localhost**. This is enforced in code: `NMPSupermemoryConfig.init` *throws*
`nonLocalBaseURL` unless the host is `localhost` / `127.0.0.1` / `::1`, cloud
endpoints (`console`/`api.supermemory.ai`) cannot be configured, and the setup
script asserts no config `baseURL` contains `supermemory.ai`. Your memories are
never sent to a Supermemory cloud.

### Seal → shard → scatter, one shard per peer

Writing a memory:

1. **Seal** — LZFSE-compress, then AES-256-GCM encrypt under a fresh random
   256-bit key. The result is opaque ciphertext; the plaintext is persisted
   **nowhere**.
2. **Shard** — split the ciphertext with a K-of-N XOR erasure code into N
   shards, any K of which reconstruct the original.
3. **Scatter** — send exactly one shard to each roster peer over the existing
   encrypted, key-pinned NMP transport. Each peer stores its single opaque
   shard (tagged `nmp_shards`).

Reading gathers **K** shards (its own local one plus peers' over NMP),
reconstructs, and decrypts. Below quorum it **fails loudly** — HTTP 503 with an
explicit `quorum_unavailable` error naming the unreachable peers — and never
returns wrong output. GCM's auth tag makes reconstruction tamper-evident.

### The guarantee — stated precisely and honestly

**No single peer holds a COMPLETE readable copy; full content requires a K-of-N
quorum of shards.** To keep memories *semantically searchable* even after the
author's device is gone, each peer also stores a small **plaintext index entry**
(title + a bounded ~160-char snippet + the AES key + the roster). This is an
honest tradeoff, and the docs state it plainly: because that index entry holds
the key, a single peer *could* decrypt its own 1/K fragment and does see the
bounded snippet — so **the quorum protects *completeness*, not secrecy against a
key-holding peer.** It is deliberately **not** sold as threshold secrecy or
zero-knowledge (that would need Shamir sharing / Reed-Solomon, out of scope).

### The phone can join too

The iPhone can't run the Node `supermemory-server` (iOS has no Node runtime and
forbids spawning a server), so it runs an **equivalent on-device native store**
— a file blob store plus Apple `NaturalLanguage` embeddings for semantic search,
with a sentence → word-average → lexical fallback chain and **no network at
all**. A single mesh can *mix* backends: Macs on Supermemory, a phone on the
native store, behind one `NMPMemoryStore` seam.

### Verified live

Measured this session: a mixed 3-peer mesh — two Macs on Supermemory, one peer
on the native on-device store — sharded a memory 2-of-3, then a Supermemory peer
was **killed with `kill -9`**, and a survivor reconstructed the full plaintext
using its own shard plus the native-store peer's shard served over NMP. Kill a
*second* peer (drop below quorum) and recall fails explicitly. Semantic recall
was confirmed end-to-end (measured similarity **0.71** for the target memory vs
0.48 for an unrelated one; warm-instance ingestion searchable **~1.2 s** after
add). The memory codec + native store carry **29 passing unit tests** (17 shard
+ 12 native store).

---

## How we beat TCP (and TLS, and QUIC)

**The claim, honestly scoped:** for the traffic distributed inference actually
produces — many small, latency-sensitive per-token round trips — NMP is the
**fastest *secure* transport**, and its NACK+FEC design beats TCP's
head-of-line blocking under loss. The repo backs this with a **fully measured**
four-leg transport race (`NMPTransportRace`): a real generation's exact traffic
(round trips × payload bytes) replayed over four real transport stacks on
loopback, each doing a genuine handshake and moving the same bytes.

### Handshake latency (measured, loopback)

| NMP (Noise IK) | TCP+TLS 1.3 | QUIC |
|---|---|---|
| **1.2–2.9 ms** | 20–26 ms | 8–12 ms |

*Source: `Docs/Project_Overview.md`.* Noise IK completes an encrypted session in
**one round trip** because both peers already know each other's static keys —
no certificate exchange, no identity-hiding overhead. For a mesh where peers
come and go, this is the difference that matters.

### The transport race — clean loopback, p50 total time (handshake + transfer)

Measured, 20 trials per shape, lower is better (*source:
`Docs/Protocol_Comparison.md`*). Traffic shapes come from real KV-cached mesh
generations (~3.5 KB/activation):

| Shape | NMP | TCP (no crypto) | TCP+TLS 1.3 | QUIC |
|---|---:|---:|---:|---:|
| prefill-burst (one large trip) | **1.82 ms** | 0.53 ms | 17.72 ms | 8.75 ms |
| decode-32 (many small trips) | **4.65 ms** | 2.81 ms | 19.57 ms | 11.31 ms |
| decode-128 (many small trips) | **14.34 ms** | 6.68 ms | 28.75 ms | 19.86 ms |

A second measured run (median of 5, `Docs/Project_Overview.md`) at different
shapes: 32 trips × 64 KB/dir — NMP **18.3 ms**, TCP 4.4, TLS 23.7, QUIC 17.2;
16 trips × 4 KB/dir — NMP **3.1 ms**, TCP 1.5, TLS 19.3, QUIC 9.7.

**How to read this honestly:**

- **Against the encrypted transports — the real comparison — NMP wins
  decisively**, 1.4–10× faster than TLS/QUIC on every shape (e.g. decode-128:
  NMP 14.3 ms vs QUIC 19.9 ms vs TLS 28.8 ms). NMP does per-packet AES-256-GCM
  just like TLS/QUIC, but its 1-RTT Noise IK + lean AEAD datagrams avoid the
  heavy per-connection and per-record costs they pay.
- **Plain TCP is faster because it does *nothing*** — no encryption, no framing.
  It's the *floor*, included to show exactly what NMP's security costs. You'd
  never ship inference over unauthenticated plaintext on a shared LAN; the bar
  NMP clears is "fastest transport that actually protects the traffic."
- **Loopback isolates protocol cost — radio time is absent from every leg.**
  These compare stack overhead, not Wi-Fi. The race refuses to *model* any leg
  it can't actually run (it reports the skip instead).

### Why NMP is designed to pull ahead under loss

Clean loopback doesn't exercise the reason NMP is UDP-based. TCP (and TLS/QUIC
over it) suffer **head-of-line blocking**: one dropped segment stalls every byte
behind it for a retransmit round trip. NMP instead uses **NACK-only
retransmission + XOR FEC** — every 4 DATA packets carry one XOR parity packet,
so a single loss per group is rebuilt receiver-side with **no round trip at
all**, while other packets keep flowing. Plus AWDL-aware traffic shaping backs
off only on radio paths where airtime contention is real.

Measured in isolation (*source: `Docs/Protocol_Comparison.md`*): **FEC
reconstructs a lost activation packet in ~0.17 ms vs ~10 ms for a NACK round
trip** — the recovery cost that head-of-line blocking imposes on TCP is exactly
the currency per-token inference can't spare.

### Loss resilience — measured, in-process mesh with real crypto/FEC/NACK

Injected loss, medians of 3 paired runs (*source: `Docs/Project_Overview.md`,
2026-07-13*):

| Injected loss | Throughput vs clean | p95 latency vs clean |
|---|---|---|
| 2% | 1.00× (free) | 1.00× |
| 5% | 1.00× (free) | 1.19× |
| 10% | ~1× (within run noise) | 1.11× |
| 15% | 0.88× | 1.97× |
| 20% | 0.72× | 1.97× |
| 25% | inference times out — the honest breaking point | — |

Loss up to ~10% is effectively free; the knee at 25% (where NACK rounds
themselves get lost) is documented as the measured edge, not hidden.

### Real-model inference (measured, Apple M3)

Llama-2-7B-Chat Q4_K_M over a real **two-process UDP mesh** (weights on the
peer, tokenizer on the coordinator): **8.7–12.1 tok/s**, per-token p50 **≈68 ms**,
protocol overhead **≈8–17 ms/token and shrinking as model compute grows**,
output **token-for-token identical** to single-device greedy. *Source:
`Docs/Project_Overview.md` / `Docs/Phase8_Design.md`.* Failover re-shards in
**0.4 ms** (measured, loopback) with peer drop detected within **5.5 s**.

---

## Architecture

**NMP is a custom UDP transport protocol for distributed AI inference across
Apple devices.** The stack, bottom to top:

```
┌──────────────────────────────────────────────────────────────────┐
│  Web / control  DashboardServer · WebUI · TransportRace · Resource │  React PWA, live token
│                 (hand-rolled HTTP + RFC 6455 WS on NWListener)     │  streaming, measured race
├──────────────────────────────────────────────────────────────────┤
│  Fault tolerance  PeerHealthMonitor · FaultToleranceOrchestrator   │  drop/join re-sharding (0.4 ms)
├──────────────────────────────────────────────────────────────────┤
│  Inference      GGUF · ModelSharder · InferenceOrchestrator ·      │  proportional layer sharding,
│                 PeerShardEngine · Llama{Runtime,Engine,Wire}       │  pipeline walk, bit-exact verify
├──────────────────────────────────────────────────────────────────┤
│  Mesh assembly  Bonjour · Capabilities · CoordinatorElection       │  zero-config mDNS, deterministic
├──────────────────────────────────────────────────────────────────┤
│  Reliability    Reliability(NACK) · FECCodec/FECGroup · AWDL       │  NACK-only + XOR FEC, link-aware
├──────────────────────────────────────────────────────────────────┤
│  Transport      UDPTransport · PacketCodec · NoiseIK · Symmetric   │  20-byte BE header, 1-RTT Noise IK,
│                 Crypto                                              │  per-session AES-256-GCM + replay
└──────────────────────────────────────────────────────────────────┘
```

1. **Transport** — 20-byte big-endian header, 1-RTT Noise IK handshake
   (`Noise_IK_25519_AESGCM_SHA256`, verified against the published cacophony
   test vector), per-session AES-256-GCM with the header as AAD and a 64-bit
   sliding replay window.
2. **Reliability** — NACK-only retransmission from a 64-packet ring (no ACKs, no
   sender timers: silence means delivered) + XOR FEC over 4-packet groups
   (recovers a single loss per group with no round trip). Link awareness
   (`NMPLinkKind`) packs MTU-safe 1350 B chunks + parity + AWDL shaping on radio
   paths, and ships kernel-ceiling ~9 KB datagrams with zero protective overhead
   on wired/loopback.
3. **Mesh assembly** — Bonjour zero-config discovery (capabilities in TXT
   records), deterministic capability-based coordinator election.
4. **Inference** — GGUF parsing, proportional layer sharding by measured
   seconds-per-layer, pipeline walking, bit-exact verification. Two engines
   behind one seam (`NMPShardComputeEngine`): a deterministic reference engine
   (the correctness oracle) and real llama.cpp via a `dlopen`'d C shim (never
   linked — the package stays dependency-free).
5. **Fault tolerance** — drop/join re-sharding, activity-based liveness.
6. **Web / control** — a hand-rolled HTTP + WebSocket server serves a React PWA
   (`--ui`) with live token streaming, per-device telemetry, compute-share
   sliders that re-shard the live mesh, and the measured transport race.

**Hard rules** (from the spec): no async/await (callback style on serial
dispatch queues); zero SwiftPM dependencies (Apple-native only); big-endian NMP
wire formats; honest measurement (measured vs modeled always labeled);
trusted-LAN security stance (no TLS/auth on the dashboard — never port-forward
it).

---

## Getting started

**Requirements:** macOS 13+ (Apple Silicon), Xcode 14.2+ / Swift 5.8+. All mesh
devices must be on the **same Wi-Fi** (not a guest/hotspot network — those block
mDNS discovery).

### Build, test, run

```bash
cd ~/neuramesh/NeuraMeshProtocol
swift build
swift test                       # full suite (~20 s)

swift run nmp-dashboard          # simulated mesh + dashboard on :8080
swift run nmp-dashboard --ui     # + React UI on :3000, all interfaces (LAN)
swift run nmp-dashboard --benchmark   # headless suite → Results/*.csv
```

`swift run nmp-dashboard --ui` is the 60-second start: a live reference mesh
(real handshakes, encryption, FEC) plus a browser UI on port **3000** and a
startup banner with your Mac's real `http://<hostname>.local:3000`, its LAN IPs,
and a scannable QR code. Open the same URL on Mac, iPhone, and iPad — every tab
shows the same live mesh. On the phone, Safari **Share ▸ Add to Home Screen**
installs it as a PWA served by the mesh itself (no Xcode). *A PWA is a control
surface — a phone that should contribute **compute** needs the native peer app,
since browsers can't open UDP sockets.*

### Real LLM inference (llama.cpp)

```bash
# one-time: build the dlopen'd shim
brew install llama.cpp && scripts/setup_llama.sh     # → Vendor/llama/libnmpllama.dylib

# get a model
mkdir -p ~/models && cd ~/models
curl -LO https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf

# full stack in one process, with one-command auto-config (probe → balance → persist → wire format)
swift run nmp-dashboard --ui --engine llamaCpp \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf --auto-config
```

`--auto-config` benchmarks each device with real probe passes, balances layer
spans to the measured speeds, persists the profile to `~/.nmp/`, and picks the
compressed wire format (llama: **zero-trim**, measured lossless
1,048,576 B → 11,928 B, −98.9%, per 32-token generation). Add `--speculation`
(with an optional `--draft-model` sharing the target's vocabulary) for
draft/verify decoding whose output is token-for-token identical to plain greedy.

### Real two-process / two-device mesh

```bash
# terminal 1 — the compute peer (weights live HERE):
swift run nmp-peer --engine llamaCpp --model ~/models/llama-2-7b-chat.Q4_K_M.gguf

# terminal 2 — the coordinator (tokenizer only in llama mode):
swift run nmp-coordinator --engine llamaCpp --model ~/models/llama-2-7b-chat.Q4_K_M.gguf \
    --prompt "The capital of France is" --tokens 16
```

Bonjour publish/browse → capability exchange → deterministic election → Noise IK
over real UDP → shard assignment → every token one real encrypted round trip.
For **Mac + iPhone** (native compute peer), follow `Docs/CrossDevice_Setup_Guide.md`
(and `Docs/Memory_Mesh_iOS.md` for the on-device memory backend).

### The distributed memory-mesh demo

```bash
# 0. install a local, self-hosted supermemory-server (localhost only)
curl -fsSL https://supermemory.ai/install | bash

# 1. stand up 3 independent local Supermemory instances + per-peer configs
scripts/setup_memory_mesh.sh start

# 2. run the three memory peers (each holds ONE shard)
swift run nmp-memory-peer --config ~/.neuramesh-memdemo/peer1/config.json
swift run nmp-memory-peer --config ~/.neuramesh-memdemo/peer2/config.json
swift run nmp-memory-peer --config ~/.neuramesh-memdemo/peer3/config.json
# (a phone or Mac can instead run the native on-device backend: nmp-memory-peer --local-store)

# 3. drive the kill-a-peer demo: write → prove 1 shard/peer → recall →
#    kill -9 a peer → recall survives from the quorum → (--kill-two shows explicit failure)
scripts/run_memory_demo.sh
```

Full operator's manual — every mode, how to test each feature, connecting peers,
troubleshooting: **`Docs/Start_Here.md`**.

---

## Project layout

| Path | What |
|---|---|
| `Sources/NMP/` | the whole protocol + runtime library (`NMP`) — transport, reliability, FEC, mesh, sharding, engines, memory mesh, dashboard |
| `Sources/NMPPeerCLI/` | `nmp-peer` — a compute peer (the same runtime the iOS app embeds) |
| `Sources/NMPCoordinatorCLI/` | `nmp-coordinator` — coordinator + cross-device benchmark driver |
| `Sources/NMPDashboardCLI/` | `nmp-dashboard` — simulated mesh + web dashboard / UI |
| `Sources/NMPMemoryPeerCLI/` | `nmp-memory-peer` — a distributed-memory peer (Supermemory or native store) |
| `Tests/NMPTests/` | the full XCTest suite (unit + in-process mesh integration under seeded loss) |
| `Docs/` | design docs, benchmarks, setup guides, protocol spec |
| `scripts/` | setup + demo scripts (`setup_llama.sh`, `setup_memory_mesh.sh`, `run_memory_demo.sh`, `loss_lab.sh`, …) |
| `web/` → `Public/` | React/TypeScript UI source → committed build the coordinator serves (`--ui`); npm only needed to edit `web/` |
| `NeuraMeshPeer/` | the iOS peer app (Xcode project, pre-wired and build-verified) |
| `Vendor/llama/` | the `dlopen`'d llama.cpp shim (gitignored; built by `scripts/setup_llama.sh`) |

---

## Status & tests

**Full suite: 474 tests, 0 failures** (measured this session) — including
bit-level codec pins, Noise IK known-answer vectors, full-mesh integration under
seeded loss, memory shard/seal + native-store coverage, and measured performance
gates.

- **Apple-native, zero SwiftPM dependencies** — Network.framework, CryptoKit,
  CoreImage, SystemConfiguration, NaturalLanguage. No Vapor, no third-party
  anything. llama.cpp is reached via a `dlopen`'d C shim, never linked.
- No async/await anywhere — callback style on serial dispatch queues.

```bash
swift test                                          # everything
swift test --filter NoiseIKTests                    # crypto known-answer vectors
swift test --filter "MemoryShardTests|LocalMemoryStoreTests"   # the memory mesh
```

---

## Documentation index

- **`Docs/Start_Here.md`** — the operator's manual: every launch mode, tab-by-tab
  UI guide, a testing playbook for every feature, connecting peers,
  troubleshooting. **Read this first.**
- `Docs/Project_Overview.md` — the whole story: core problem, architecture, how a
  token flows, measured state of the world, honest limitations, roadmap.
- `Docs/Memory_Mesh.md` — the distributed memory mesh (seal/shard/scatter, the
  searchable-index tradeoff, the live kill-a-peer measurement).
- `Docs/Memory_Mesh_iOS.md` — the on-device iPhone memory backend + integration runbook.
- `Docs/Protocol_Comparison.md` — why NMP, not TCP/TLS/QUIC: the measured
  four-leg race, honestly scoped.
- `Docs/Benchmarks.md` — how to run the benchmark suite and interpret the results.
- `Docs/CaseStudy_PacketLoss.md` — loss-resilience deep dive (FEC / NACK / FLUSH /
  link-aware restraint), measured micro + macro numbers.
- `Docs/CrossDevice_Setup_Guide.md` — click-by-click Mac + iPhone mesh setup.
- `Docs/NMP_Specification.md` — the wire protocol, source of truth.
- `Docs/Phase8_Design.md` / `Docs/Phase9_Design.md` — llama.cpp integration and
  the fast-mesh speed levers (adaptive sharding, wire compression, speculation).

---

Made with 🤍 by Divyam Kataria
