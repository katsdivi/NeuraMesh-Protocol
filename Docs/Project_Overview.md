# NeuraMesh: What We Built

*Last updated 2026-07-13. Every number in this document is a wall-clock
measurement on the hardware named next to it — nothing is modeled or
estimated. Where a measurement is loopback (protocol cost isolated from
radio time), it says so.*

## The core problem

Modern LLMs don't fit on one consumer device, but households own several:
a Mac, an iPhone, an iPad. Their combined RAM and compute could serve
models none of them can hold alone — if the devices could act as one
machine.

The obstacle is the network. Distributed inference is a pathological
workload for off-the-shelf transports:

- **Per-token round trips.** Generating one token means shipping a hidden
  state to a peer, waiting for its layers, and shipping the result back —
  for every token. Latency compounds 32× for a 32-token answer.
- **Wi-Fi loses packets.** TCP answers loss with head-of-line blocking:
  one lost segment stalls every byte behind it for a retransmission
  round trip. For a pipeline that lives and dies by per-trip latency,
  that's the worst possible failure mode.
- **TLS handshakes are heavy.** Mesh peers come and go (phone joins,
  leaves the room, rejoins). TCP+TLS 1.3 costs ~20 ms per connection
  establishment (measured, loopback); QUIC ~8–12 ms. A mesh that
  re-handshakes often pays that constantly.
- **The mesh shares the airwave.** Peer-to-peer Wi-Fi (AWDL) contends
  with the traffic it carries; a transport that blasts blindly makes its
  own conditions worse.

Nothing off the shelf is shaped for "many encrypted small-tensor round
trips over lossy shared radio, between mutually-known peers, with instant
reconnection." So we built the transport that is.

## What NeuraMesh is

**NMP (NeuraMesh Protocol)** is a custom UDP transport plus a distributed
inference runtime, Apple-native (Network.framework + CryptoKit, zero
third-party dependencies), that turns Apple devices on a LAN into one
inference machine.

The one-sentence architecture: **a 1-RTT Noise IK handshake and
AES-256-GCM sessions over UDP, loss handled by XOR FEC + NACK-only
retransmission instead of TCP-style stalls, mesh assembly by Bonjour with
deterministic coordinator election, and model layers sharded across peers
proportionally to their measured speed.**

## How it works — the stack

1. **Transport** (`UDPTransport`, `PacketCodec`, `NoiseIK`,
   `SymmetricCrypto`). 20-byte big-endian header; Noise IK
   (`Noise_IK_25519_AESGCM_SHA256`, verified against the published
   cacophony test vector) completes in one round trip because both sides
   already know each other's static keys — measured **1.2–2.9 ms** vs
   TLS 1.3's 20–26 ms and QUIC's 8–12 ms on the same loopback. Every
   packet is AES-256-GCM sealed with the header as AAD and a sliding
   replay window.
2. **Reliability** (`Reliability`, `FECCodec`/`FECGroup`). No ACKs, no
   sender timers: receivers detect sequence gaps and send NACKs; senders
   retransmit verbatim from a 64-packet ring. On top, every 4 DATA
   packets get one XOR parity packet, so a single loss per group is
   rebuilt receiver-side in ~0.15 ms end-to-end — no round trip at all.
   Measured: loss ≤10% costs nothing; details in
   `Docs/CaseStudy_PacketLoss.md`.
3. **Link awareness** (`AWDLDetector`, `TrafficShaper`, `NMPLinkKind`).
   The transport classifies its physical path. On radio it packs 1350 B
   chunks (MTU-fit, never fragmenting), emits FEC parity, and infers AWDL
   contention (loss-corroborated latency shifts) to defer background
   traffic. On wired/loopback it ships kernel-ceiling 9 KB datagrams and
   skips parity and shaping entirely — there is no airtime to protect.
   Tensor traffic is `critical` priority and is never deferred anywhere.
4. **Mesh assembly** (`Bonjour`, `Capabilities`, `CoordinatorElection`).
   Zero configuration: peers publish `_neuramesh._tcp` with capabilities
   in TXT records, discovery takes <1 s, election is deterministic
   (every peer computes the same coordinator).
5. **Inference** (`GGUF`, `ModelSharder`, `InferenceOrchestrator`,
   `PeerShardEngine`). The coordinator walks the layer pipeline
   peer-to-peer, shipping activations as chunked, encrypted tensors.
   Layer shares are proportional to measured seconds-per-layer
   (`AdaptiveSharding` probes real passes and persists device profiles).
   Two engines sit behind one seam: a deterministic reference engine
   (bit-exact across any sharding — the correctness oracle) and
   llama.cpp via a dlopen'd C shim for real models.
6. **Fault tolerance** (`PeerHealthMonitor`, `FaultToleranceOrchestrator`).
   Peer drop detected within 5.5 s, re-sharding in 0.4 ms, output
   bit-exact after failover.
7. **Speed levers** (Phase 9): zero-trim wire format (lossless, −98.9%
   payload for llama token-state tensors), mixed-precision (fp16 bulk +
   top-2% fp32, ~52% of raw for dense activations), pipeline-parallel
   batching (~2.4× overlap), and draft/verify speculative decoding
   (32 tokens in 8 round trips at full acceptance, output token-identical
   to plain greedy). `--auto-config` turns all of it on and picks sane
   values.
8. **Control surface** (Mesh 2.x): the coordinator serves its own React
   web UI (installable PWA) with live token streaming, per-device
   telemetry, compute-share sliders that re-shard the live mesh, a chaos
   slider, and a **fully-measured transport race** — a real generation's
   traffic replayed over the production NMP stack vs kernel TCP vs
   TCP+TLS 1.3 vs QUIC, all four legs wall-clock on real sockets.

### One token, end to end

Prompt arrives at the coordinator → tokenizer produces the input state →
`InferenceOrchestrator` seals it into ≤9 KB (wired) or ≤1350 B (radio)
chunks and bursts them (one queue hop, coalesced writes, FLUSH on the
last chunk) to the peer holding layers 0–15 → that peer's
`PeerShardEngine` reassembles, runs its layers, bursts the result to the
next hop → … → final logits return to the coordinator → token sampled,
broadcast to every open browser over WebSocket, loop. A lost chunk along
the way is rebuilt from parity (radio) or NACK-refilled (~9 ms worst
case) without stalling the packets behind it.

## Measured state of the world

Transport race, median of 5 runs, Apple M-series, loopback (radio time
absent from every leg — this isolates protocol/stack cost):

| Shape | NMP | plain TCP | TCP+TLS 1.3 | QUIC |
|---|---|---|---|---|
| 32 trips × 64 KB/direction | 18.3 ms | 4.4 ms | 23.7 ms | 17.2 ms |
| 16 trips × 4 KB/direction | 3.1 ms | 1.5 ms | 19.3 ms | 9.7 ms |

NMP beats TCP+TLS on both shapes and QUIC decisively at inference-shaped
traffic; at bulk it's within noise of QUIC. Plain TCP is the floor, not a
peer — it carries no encryption, no framing, and stalls under loss.
Handshake alone: NMP 1.2–2.9 ms, TLS 20–26 ms, QUIC 8–12 ms.

Loss resilience (in-process mesh, real crypto/FEC/NACK, injected loss,
12 generations × 4 tokens, medians of 3 paired runs, 2026-07-13):

| Injected loss | Throughput vs clean | p95 latency vs clean |
|---|---|---|
| 2% | 1.00× (free) | 1.00× |
| 5% | 1.00× (free) | 1.19× |
| 10% | ~1× (within run noise) | 1.11× |
| 15% | 0.88× | 1.97× |
| 20% | 0.72× | 1.97× |
| 25% | inference times out — the honest breaking point | — |

Real-model inference (Apple M3): Llama-2-7B-Chat Q4_K_M over a real
two-process UDP mesh at 8.7–12.1 tok/s (per-token p50 ≈68 ms, protocol
overhead ≈8–17 ms/token and shrinking as model compute grows), output
token-for-token identical to single-device greedy. Full tables live in
the phase design docs.

**331 tests, 0 failures**, including bit-level codec pins, Noise
known-answer vectors, full-mesh integration under seeded loss, and
measured performance gates.

## Design decisions that define the project

- **Noise IK over TLS.** Mesh peers are mutually known (pinned static
  keys), so the identity-hiding, certificate-carrying TLS handshake is
  pure overhead. Noise IK gets an encrypted session in 1 RTT.
- **NACK-only reliability.** ACK clocks and sender timers exist to
  serve unknown, congested internets. On a LAN we invert it: silence
  means delivered; receivers speak only when something is missing.
- **FEC before retransmission.** A retransmit costs a round trip — the
  exact currency inference can't spare. One parity packet per four
  buys back single losses for 25% overhead, and only on radio paths
  where loss actually happens.
- **The link decides.** Chunk size, FEC, and contention shaping all key
  off the physical path (`NMPLinkKind`). Radio gets MTU-safe packets,
  parity, and AWDL care; wired/loopback gets 9 KB datagrams and zero
  protective overhead.
- **Honest measurement as a hard rule.** Measured and modeled numbers
  are never mixed. The transport race refuses to model a leg it can't
  run (it reports the skip instead). Benchmarks that got stale when the
  protocol improved were re-measured, not massaged.
- **Zero dependencies, no async/await.** Apple frameworks only,
  callback style on serial dispatch queues throughout, llama.cpp
  reached via dlopen so the package itself stays dependency-free.
- **Determinism as the correctness oracle.** Greedy sampling makes
  output placement-invariant, so "distributed == local, byte for byte"
  is an executable test, used everywhere from re-sharding to
  speculative decoding.

## Honest limitations

- llama.cpp cannot execute layer sub-ranges (KV cache is per-context),
  so a llama mesh is tokenizer-on-coordinator + weights-on-peer — real
  remote execution, but not multi-way layer splitting. The reference
  engine shards arbitrarily; real multi-device model splitting waits on
  engine support.
- The dashboard/web UI is trusted-LAN only: no TLS, no auth, never to be
  port-forwarded.
- A browser tab is a control surface, not a compute peer — browsers
  can't open UDP sockets; contributing compute requires the native peer
  app.
- AWDL heuristics were tuned on this hardware and need re-validation on
  more device generations.
- Loss recovery breaks down at 25% sustained loss (NACK rounds
  themselves get lost). Real Wi-Fi rarely sustains that, but it's the
  measured edge.

## Where this goes

Near term (measurement and credibility):
1. **Real-Wi-Fi loss validation** — the loss lab (`scripts/loss_lab.sh`)
   shapes real loss onto race ports; run the race and the Mac+iPhone
   mesh under it and publish the tables next to the loopback ones.
2. **Cross-device speedup story** — Mac+iPhone benchmark with
   auto-config (mixed-precision wire + measured shares), reported as
   tokens/sec vs the best single device.

Medium term (protocol):
3. **Adaptive FEC** — scale parity rate to measured loss instead of a
   fixed 25%; clean radio links currently pay for protection they don't
   use.
4. **Datagram packing** — coalesce small control/token-state packets
   into shared datagrams on radio (bulk tensors already fill the MTU).
5. **Connection migration** — survive a peer hopping networks (Wi-Fi ↔
   AWDL) without a re-handshake, QUIC-style but with the 1-RTT Noise
   economics.

Longer term (runtime):
6. **True multi-device model splitting for real LLMs** — either an
   engine with sub-range execution + KV-cache transfer, or MLX behind
   the same `NMPShardComputeEngine` seam.
7. **KV-cache streaming and prefill sharding** — today's mesh moves
   per-token states; moving attention caches unlocks long-context
   workloads.
8. **More radios** — Thread/local 5 GHz mesh characterization; the
   link-kind seam is already where that policy would plug in.
