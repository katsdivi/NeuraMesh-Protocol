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

**Status: not built. This is the blocker for the headline goal (run
Qwen3-14B across a Mac + iPhone so neither holds the whole model).**

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
