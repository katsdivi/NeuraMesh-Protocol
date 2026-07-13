# Mesh 2.8 — Capacity-Aware Sharding

The goal that drives NeuraMesh: **run a model too big for any one device by
splitting it across the mesh** (e.g. Qwen3-14B across a Mac + iPhone). Two
things stood in the way, and both are fixed here.

## The problems this fixes

1. **iPhone stuck on "waiting for coordinator."** A peer that the plan gave
   0 layers was never told anything — it waited forever. Now every joined
   member receives an explicit assignment, including a **zero-layer standby
   assignment** when it holds nothing. It shows "0 shards — standing by",
   not a hang.

2. **The Mac did all the compute.** The dashboard seeds the mesh with
   in-process *simulated* peers (a demo convenience). A real iPhone joining
   was a 5th member competing with fast Mac-local phantoms, so it got
   crumbs. Now **the simulated peers retire the moment a real LAN device
   joins** — the mesh becomes genuinely cross-device.

3. **The sharder only understood speed.** For an oversized model, speed is
   the wrong objective — the model physically won't fit on the fastest
   device, so layers *must* spill regardless of speed. The sharder is now
   **capacity-aware**.

## The two objectives (switchable live, Devices tab)

Capacity is always a hard ceiling: a device holds at most
`layerCapacity(ramMB, bytesPerLayer)` layers. Within that ceiling:

- **Capacity + Speed (default).** Spread across the whole mesh, balanced by
  measured speed, so every device pulls its weight and oversized models run.
  With no capacity limit (the weightless reference engine) this is exactly
  the classic Phase 5 speed-weighted split — **no speed cost when nothing is
  binding.**

- **Pure Speed.** Minimize per-token latency: pack the fastest device to its
  ceiling, spill only the remainder onto the next fastest. A device that
  would only add a Wi-Fi hop gets **0 shards, with the reason shown** ("Pure
  Speed mode packs the fastest device(s); routing around this one avoids a
  Wi-Fi hop"). This is the "I want raw speed" mode.

Switch it from the Devices tab, or `POST /api/mesh/objective {"objective":
"speed" | "capacityThenSpeed"}`. Either way it re-shards through the normal
SHARD_ASSIGN round.

## UI transparency (0 shards, always explained)

Any device holding 0 layers shows a **"0 shards · standby"** badge and the
specific measured reason — capacity too small ("out of memory — hot spare"),
not needed ("the model fits on N faster devices"), or Pure Speed routing.
A model too big for the whole mesh surfaces a **capacity shortfall** warning
("N layers fit on no device").

## Modeling a large model with the reference engine

The reference engine is weightless, so capacity never binds on its own. To
see the real behavior (and rehearse the Qwen goal), declare the model's
memory footprint:

```bash
# Simulate Qwen3-14B (~9 GB q4): 32 layers, ~281 MB/layer. Capacity now
# binds, so the mesh is forced to distribute across devices.
swift run nmp-dashboard --ui --model-gb 9

# Start with no simulated peers (pure real-device testing):
swift run nmp-dashboard --ui --sim-peers 0
```

`--model-gb` is a *modeled* footprint (labeled as such — the compute and
transport are real). For a real llama mesh, capacity comes from the actual
GGUF file size / layer count, so this is exactly the machinery that runs
Qwen3-14B for real once the weights are present.

## What did NOT change

NMP's transport speed. Capacity-aware planning is pure control-plane logic
that runs once per re-shard (sub-millisecond); the data plane — Noise IK,
AES-256-GCM, FEC, the link-adaptive burst sender — is untouched. When
capacity isn't binding, the plan is byte-identical to before.

## Tests

- `ModelSharderTests` (+6): capacity ceilings, forced distribution, Pure
  Speed packing + exclusion, capacity shortfall, RAM→capacity math, and
  that unbounded capacity reproduces the legacy split exactly.
- `FaultToleranceTests` (+1): a standby peer receives a zero-layer
  assignment instead of hanging, and flipping the objective re-engages it.
- `ObjectiveRouteTests` (+4): the live objective switch endpoint.
