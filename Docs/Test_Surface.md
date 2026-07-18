# NeuraMesh — Test Surface

Complete inventory of endpoints and features for bug/user testing. The mesh
runs a dashboard at **http://localhost:3000** (or `http://macbook-air-8.local:3000`
on the LAN). Engine: `llamaShard`. A coordinator (this Mac) plus an iPhone 17
Pro peer form the mesh. Trusted-LAN, no auth.

## Reaching the mesh

```bash
BASE=http://localhost:3000
curl -s $BASE/health | python3 -m json.tool
```

## HTTP REST API

### GET
| Path | Returns | Notes |
|---|---|---|
| `/health` | `status` (HTTP liveness) + **`ready`** (true only once the real engine+model+plan can generate) + mesh facts | the nav pill gates on `ready`, not `status`; `mesh.peers_note` states the counting rule |
| `/api/devices` | device rows derived from the CURRENT plan + membership (id, name, assigned, alive) | an in-mesh 0-layer device shows an explicit "0 layers — excluded…" assignment, never a stale range; a departed peer's row is removed |
| `/api/devices/metrics` | live host kernel counters + per-peer cards + `auto_balance` + **`manual_mode`** (+ deprecated alias `allocation_supported`) + totals | coordinator card ALWAYS present (excluded-style at 0 layers); each card carries `loaded_mb_basis` (measured vs modeled); `totals.devices_note` states the counting rule |
| `/api/clients` | web UI clients currently viewing | |
| `/api/models` | installed models in `~/models` with compatibility flags (usable/compatible/fits_host/active/recommended) | scan is cached (~25 ms); only qwen2/qwen3 are `usable`; vault slices / split fragments are omitted (metadata check — the same criterion select rejects on) |
| `/api/chats` | saved conversation summaries (this device), newest first | empty array if history disabled |
| `/api/chats/<id>` | one full conversation (messages), inflating if LZFSE-compressed | 404 for unknown id |
| `/api/mesh/plans` | 3 candidate shard plans (speed / balanced / capacity) with per-device layers, footprint MB, % of RAM | `current_strategy` + `plans[]` + `footprint_note` (footprint_mb is modeled weights-only; runtime `loaded_mb` is larger) |

### POST
| Path | Body | Effect |
|---|---|---|
| `/api/inference` | `{prompt, max_tokens, enable_speculation?, enable_comparison?}` | runs a real sharded generation; streams tokens over `/ws`. `max_tokens ≤ 0` → 400; > 128 is clamped AND echoed via `max_tokens_effective` |
| `/api/chat` | `{messages:[{role,content}], max_tokens, enable_speculation?}` | folds transcript into the engine template, runs generation; same `max_tokens` rules as `/api/inference` |
| `/api/chats` | `{id?, title?, model?, messages:[...]}` | create (blank id) or update a saved conversation; returns the stored summary. `messages: []` → 400 UNLESS an explicit non-empty `title` is sent (the web new-chat button does); the stored `model` is always the base model's display name (slugs normalized, `…_sliced_a_b` artifacts stripped) |
| `/api/chats/<id>/delete` | `{}` | delete a saved conversation |
| `/api/benchmark/run` | `{prompt, max_tokens, runs}` | multi-run benchmark; returns avg tok/s, latency, stddev, plus `runs_requested`/`runs_completed` (runs clamp to 1..10). Busy → **429** |
| `/api/comparison` | `{tokens, payload_bytes, round_trips, measured_total_ms, lan_rtt_ms?, loss_rate?}` | MODELED protocol estimates — 4 rows matching the measured race: NMP (measured) + TCP + TCP+TLS 1.3 + QUIC (modeled) |
| `/api/comparison/run` | `{prompt, max_tokens, enable_speculation?}` | MEASURED transport race: NMP vs TCP vs TCP+TLS 1.3 vs QUIC (loopback) + splice projection. Busy → **429**; local placement (no mesh traffic) → **409** with an explanatory error, never fabricated numbers |
| `/api/models/select` | `{path}` (or bare name/filename) | switch the active model — **RE-EXECS the coordinator process** (page shows "reconnecting…"); only qwen2/qwen3 accepted; vault slices / split fragments → 400; selecting the already-active model → 200 `{"summary":"already active","reconnecting":false}` with NO re-exec |
| `/api/mesh/objective` | `{objective}` | set sharding objective (`capacityThenSpeed` / `speed`) — wired in reference AND llamaShard modes (llamaShard maps speed→speed plan, capacityThenSpeed→balanced plan); unknown value → 400 |
| `/api/mesh/autobalance` | `{enabled: bool}` | Auto (measured latency) vs Manual (operator compute-share sliders). Answers in ms with the STAGED plan (`summary` ends "staged; re-shard applying in the background"); the devices/plan state updates when the background SHARD_ASSIGN round commits |
| `/api/mesh/strategy` | `{strategy: "speed"\|"balanced"\|"capacity"}` | apply a previewed plan; re-shards the mesh |
| `/api/devices/<hexid>/allocate` | `{share: 0..1}` | manual compute-share for a peer (flips to manual mode); **share 0 excludes the peer**; unknown device id → **404** `{"error":"unknown device"}`; success echoes `share_requested` and the resulting assignment in `summary`; applying a nonzero share to an idle phone makes it VAULT-STREAM its layers (can take seconds) |

### WebSocket
| Path | Stream |
|---|---|
| `/ws` | live `generation_*` token events + mesh events (re-shards, joins/leaves); the UI's live transcript. EVERY generation — including benchmark/comparison runs — is framed by `generation_started`/`generation_complete` (or `generation_failed`), and every `generation_*` event carries `"source": "inference"\|"chat"\|"benchmark"\|"comparison"` |

## Web UI tabs (React, served at `/`)
- **Mesh** — overview/dashboard
- **Run** — single-prompt inference
- **Chat** — conversation with a saved-history sidebar (new / select / delete; auto-titled; compressed when idle)
- **Models** — installed models; per-card "Restart the mesh onto this model" (needs ≥2 usable models to be actionable)
- **Devices** — host resources, per-peer cards, **Auto/Manual** balance toggle, compute-share sliders (manual only, 0–100%, 0 = exclude), **Sharding plan** preview (Preview → 3 cards with fill bars → Apply)
- **Benchmark** — run benchmark sweeps
- **Compare** — protocol comparison / measured transport race
- **Pressure** — load/pressure view
- **Settings** — mesh controls

## iPhone app (NeuraMeshPeer)
- Joins the mesh as a compute shard peer (foreground-only; iOS suspends it in background)
- **Chat** — with local persisted history (LZFSE-compressed, own store), history sheet (clock=list, pencil=new, swipe=delete)
- **Models** — download other Qwen sizes + switch the whole mesh
- Streams its shard slices from the Mac's vault (`Library/Caches`)

## Features
- Distributed sharded inference across Mac + iPhone (one real UDP mesh round-trip per token)
- **Network-aware balancing**: auto mode minimizes measured per-token latency (Σ compute + round-trips) → Mac-only for small models, distributes for big ones
- **Plan preview**: speed / balanced / capacity, each with per-device RAM footprint
- Manual compute-share allocation + peer exclusion
- Model switching (0.5B, 1.5B Qwen) + model download (phone)
- Persistent, compressed chat history (per device, local-first)
- Fault tolerance: peer drop/join re-shards live; a stalled peer is bounded by a 4 s per-token timeout
- Transport race (NMP vs TCP/TLS/QUIC), benchmark suite
- Installable PWA

## Known behaviors / gotchas (don't file these as bugs)
- Inference is **serialized** — a second concurrent generation returns HTTP 429 "an inference is already running" on `/api/inference`, `/api/chat`, `/api/benchmark/run`, AND `/api/comparison/run` (by design; the pipeline is sequential).
- `/api/models/select` **re-execs** the coordinator (the process restarts) — everything reconnects. During the restart `/health` serves `status:"ok"` (HTTP is up) with `ready:false` until the engine+model+plan are actually live — gate on `ready`.
- Applying `balanced`/`capacity` (or a nonzero manual share) to the idle phone makes it stream layers — the re-shard can take up to ~30 s and shows "re-sharding…".
- For a small model, auto/speed correctly gives the phone **0 layers** (it shows as connected-but-excluded, not vanished).
- The dashboard/UI is trusted-LAN, no auth — expected.
