# Future Plans — NeuraMesh Roadmap

Things NeuraMesh is *designed to grow into* but has **not built yet**. Each
entry says what it is, why it matters, how it fits the machinery that
already exists, the honest cost, and its status. Nothing here is claimed as
working — that separation is deliberate (see the "Honest measurement" rule
in `CLAUDE.md`). For what *is* built and measured, read `Project_Overview.md`.

The through-line for all three: **weights are pinned to where they compute;
only the small thing (the activation vector) crosses the wire per token.**
Everything below is a different way to place weights and compute without
breaking that rule.

---

## 1. True cross-device sharding of a *real* model (the big one)

**Status: 🚧 IN PROGRESS — real sharding works end-to-end through the mesh
(bit-exact, KV-cached, partial-load proven) and is exposed in the dashboard
and the compute peer. Remaining: scaling to 14B on real devices. Goal: run
Qwen-14B across a Mac + iPhone (+ any other devices) so no single device holds
the whole model.**

### What works now (2026-07-13, verified)

The real engine is `scripts/llama-shim/nmp_shard_shim.c` — a **ggml
graph-surgery** shim (build: `scripts/setup_shard.sh`). It builds the
transformer forward directly in ggml, so it can run an arbitrary block range
`[start,end)` and **load only those blocks' weights**. It is bound into Swift
(`LlamaShardRuntime.swift` → `NMPLlamaShardComputeEngine`) and runs through
the real mesh. Verified against llama.cpp on Qwen2.5-0.5B:

- **Forward is bit-exact.** Hand-built ggml Qwen2 forward reproduces
  llama.cpp's greedy output exactly (two prompts).
- **The split is real.** 2-way (@6/@12/@18) and 3-way (@8,16) layer splits
  all match the whole-model output; only the residual crosses the "wire". A
  falsification test (zeroing the residual) collapses the output — proving
  downstream shards genuinely depend on the hand-off, which the earlier fake
  did not.
- **Memory is actually reduced.** Each shard partial-loads only its blocks
  (e.g. 219 MB + 266 MB for a 24-layer @12 split — neither holds all).
- **Arch-generic.** Params are read from GGUF metadata (runs 0.5B and 14B);
  qwen2 (QKV bias) and qwen3 (QK-norm) are handled by tensor detection.
- **It runs through the actual mesh, not just a harness.** The Swift engine
  (`NMPLlamaShardComputeEngine`) partial-loads its assigned range on
  SHARD_ASSIGN and routes `runLayers` through the shim; the residual crosses
  the real transport (Noise IK, AES-GCM, FEC, NACK) inside `NMPLlamaShardWire`;
  and a 2-way and a 3-way split each produce text IDENTICAL to the
  single-device baseline, with every peer's loaded weights a strict subset of
  the model (see `LlamaShardTests` / `LlamaMeshIntegrationTests`).
- **Per-shard KV cache (ABI 2), bit-exact.** Each shard keeps a persistent
  per-layer K/V cache, so a decode step processes only the NEW token and
  attends over cached keys/values — O(n) per token instead of reprocessing the
  whole sequence, and the wire hand-off shrinks from `n_embd × T` to `n_embd`
  per token. `n_past` (the request's basePos) is the authoritative cache
  length, which keeps a replayed step idempotent. Still bit-exact vs the
  whole-model run (F32 cache).

### Surfaced in the dashboard + peer (2026-07-13, verified live)

- `nmp-dashboard --engine llamaShard --model M.gguf --placement sharded:N`
  runs the real N-way sharded mesh. The Devices panel shows each peer's
  MEASURED layer range and loaded MB via `NMPShardReport` (e.g. peer 0 "layers
  0-11 · 209.4 MB", peer 1 "layers 12-23 · 253.6 MB" — neither holds the whole
  ~485 MB model), and `POST /api/inference` returns real text
  ("Paris. It is the largest city in Europe…") across 2 shards.
- `nmp-peer --engine llamaShard --model M.gguf` makes a real device a shard
  peer: it partial-loads ONLY the range the coordinator assigns and logs the
  loaded MB. This is the native/iOS compute-peer path.

### Adaptive model tiering (2026-07-14, storage + RAM aware)

The mesh now picks the OPTIMAL model for whatever devices are present, and
re-decides on churn:

- `NMPCapabilities` advertises free storage; `NMPModelCatalog` discovers the
  GGUFs actually on disk and reads each one's real footprint (exact
  `bytesPerLayer`, params, quant) via the Phase 5 GGUF parser.
- `NMPModelSelector` picks the highest-quality model that FITS — every hosting
  device needs disk for the file (storage ceiling → **degrade** when a device
  can't hold it) AND the layer split must fit aggregate RAM with headroom
  (reusing `layerCapacity`, which also guards against speed-killing
  fragmentation).
- `NMPAdaptiveModelController` turns a membership change into a decision:
  unchanged / reshard (same model) / **switch model** (reload a different GGUF)
  / no-fit — so a device leaving degrades the model and a device returning
  upgrades it. The reload reuses the Phase A churn-safe re-prefill.
- `nmp-dashboard --engine llamaShard` (no `--model`) auto-selects the best
  model from `~/models` for the host and shows the choice + reason.

### Remaining (needs the target model + real devices)

- **Live re-selection in the running dashboard**: today it auto-selects at
  startup; wiring the controller to reload models on live join/leave is the
  last integration step, best validated on the real 14B across a Mac + iPhone.
- **The 14B end-to-end run**, measured (the selector/shim are arch-generic for
  qwen2/qwen3; llama-arch models need a NORMAL-RoPE shim variant first).

### The earlier fake (preserved in history)

A prior pass (`gemini_implementation.md`) added `nmp_llama_decode_embd` etc.
that **did not shard**: every peer loaded the full model, ran all layers
twice, and faked correctness with hardcoded per-position RMS constants
(`get_rms_scale`, tuned to one prompt). Superseded by the ggml shim above.

### The problem

The reference engine (`ComputeEngine.swift`) can split 32 layers across N
peers today — that's what the whole capacity-aware sharder, the pipeline
walker, and the dashboard's multi-shard views run on. But it's *weightless*
(deterministic stand-in math), so it proves the transport and orchestration,
not real model quality.

The real engine (`LlamaEngine.swift`, via the llama.cpp shim) produces real
tokens — but **llama.cpp cannot execute a layer sub-range**. Its public API
decodes the *whole* model per step, and the KV cache lives inside one
context. So a llama plan is always **one full-range shard**: tokenizer on
the coordinator, the entire model on a single peer. That peer still needs
enough RAM for all 32 layers. Which means today you cannot run a model that
is too big for *any single device* — exactly the case the project exists for.

### What has to change

Break the "one full-range shard" limit by executing layer sub-ranges
against the ggml compute graph directly, instead of calling
`llama_decode` (which insists on the full model):

- Build the ggml graph for layers `[start, end)` only, feed it the incoming
  activation tensor, run it, return the output activation — the same
  `runLayers(start:end:input:)` contract the reference engine already
  satisfies. The seam (`NMPShardComputeEngine`) does not change; only the
  llama implementation does.
- Own the KV cache **per shard** (each peer keeps the K/V for its own
  layers) instead of one cache per whole-model context. This is the real
  surgery — positions and cache slots have to be threaded per sub-range.
- Load only the assigned layers' tensors on each peer (partial GGUF load),
  so the phone holds *its* layers' weights and nothing more. That is what
  finally makes the 14B fit: Mac holds layers 0–15, phone holds 16–31,
  neither holds all 32.

### Why it fits what exists

Everything above the engine seam is already sub-range-aware: the sharder
emits `[start, end)` per peer, `PeerShardEngine` computes a range, the
pipeline walks ranges, and bit-exact verification compares placements. The
reference engine is the working proof that the rest of the stack handles a
real N-way split. This item is *only* the llama-side execution of a
sub-range — the mesh around it is ready.

### Honest cost

This is ggml-internals work (graph construction, per-shard KV, partial
weight load), not a binding tweak. It is the largest single item on this
list. Until it lands, the honest split story runs on the reference engine,
and the real-LLM story runs single-peer (see `Contributor_Guide.md` →
"Testing with the real LLM").

---

## 2. Borrowed peers — a remote GPU or second machine over the internet

**Status: not built. Small, and mostly free from the capacity sharder.**

### The idea

Renting a cloud GPU (or SSHing into a beefy box) is fast for one reason: the
**weights are already resident there**, so you ship it a tiny prompt and it
ships back tiny tokens — kilobytes over the wire, never the gigabytes of
weights. That is exactly NMP's activation-passing discipline. So a remote
box should be able to **join the mesh as just another peer** — one with a
huge RAM + compute ceiling — and the existing capacity-aware sharder
(`planDetailed`) would hand it a large share automatically. No new sharding
logic; it's a peer with big numbers.

### What has to change

- **Reachability past the LAN.** Discovery today is Bonjour/mDNS
  (LAN-only). A borrowed peer needs an explicit dial (address + static
  key) — the pinning path already exists via
  `PeerConnectionConfig.authorizedStaticKeys`; this is a manual-add UI +
  a WAN-reachable UDP endpoint, not new crypto.
- **Keep NMP as the pipe — do *not* use SSH.** SSH is TCP: connection-
  oriented and head-of-line blocked, the wrong shape for chatty per-token
  round trips. NMP's UDP + Noise + NACK/FEC is built to beat it, and the
  transport race (`TransportRace.swift`) is what proves it. Borrow SSH's
  *concept* (ship work to resident compute), reject its *mechanism*.
- **Honest labeling.** A borrowed peer is **offload, not local mesh**, and
  its RTT jumps from ~1–5 ms (Mac↔iPhone Wi-Fi) to ~10–50 ms (internet).
  The UI must label it as such — same honesty bar as measured-vs-modeled.
  Reuse the Mesh 2.8 per-device badge machinery (`excluded` /
  `exclusion_reason`) to add an `offload` / `internet-RTT` tag.

### Why it fits what exists

The capacity sharder already treats "a peer with more RAM and more speed"
correctly — it just spreads or packs by those numbers. A borrowed peer is
that, with the numbers turned up. The work is reachability + labeling, not
scheduling.

### Honest cost

Low for a second Mac on the same tailnet/VPN; medium for a true cloud GPU
(NAT traversal / a relay for the UDP endpoint). Good for "I need one more
slab of VRAM," not for "keep it fully on my own devices."

---

## 3. Weight-vault role — disaggregate storage from compute

**Status: not built. Depends conceptually on #1.**

### The idea

Three resources, not two: **storage** (flash — the iPhone has plenty),
**RAM** (where weights must be resident to compute — scarce), and **compute**
(FLOPS). The sharder today ceilings on RAM (`layerCapacity(ramMB:...)`),
which is correct. This item lets **storage float free of compute**: a device
with lots of flash but weak compute becomes a **weight vault** — it stores
layers at rest and serves them, at *load time*, into the RAM of whichever
device will compute them.

The one law that bounds it: a layer computes where its weights are
RAM-resident, and moving weights costs bandwidth. So the vault pays that
cost **once per session** (page weights over at startup), **never per
token** — per-token weight streaming (~281 MB/layer for Qwen3-14B over
Wi-Fi) is a ~100× slowdown and is explicitly ruled out by the speed
constraint. This solves the "the GGUF won't fit on the Mac's *disk*" case;
it cannot solve the "won't fit in any device's *RAM*" case.

### What has to change

- A `weight-vault` peer role: holds layers at rest, serves them on request,
  may compute **zero** layers. Slots into the existing 0-shard standby /
  exclusion machinery — a vault shows *"0 shards · serving weights from
  vault"* instead of a bare standby.
- A one-time weight-transfer phase at session start (vault → compute peer's
  RAM), reusing the chunked NMP transport already used for activations.
- Depends on #1's partial weight load to be worth much: without sub-range
  execution, whoever computes still needs the whole model resident, so the
  vault only helps the single-peer-full-model case.

### Honest cost

Medium, and gated on #1. Most valuable when the Mac is *disk*-bound but
*RAM*-fine — page the model off the phone's flash into Mac RAM once, then
run at local speed.

---

## Not on the roadmap (and why)

- **Per-token weight streaming.** Moving weights every token violates the
  hard speed constraint (~100× slower over Wi-Fi). Ruled out on purpose.
- **SSH / TCP as the mesh transport.** Head-of-line blocking is the wrong
  shape for per-token round trips; NMP exists precisely to avoid it.
- **Browser peers doing compute.** Browsers have no UDP. A PWA is a control
  surface only; compute peers are the native app. (See `CLAUDE.md`.)
