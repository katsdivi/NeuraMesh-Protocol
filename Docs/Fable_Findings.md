# NeuraMesh — Fable Test Findings (2026-07-17)

Lead-tester run: six parallel test agents + lead, exercising every endpoint
and feature in `Docs/Test_Surface.md` against the live mesh (coordinator on
this Mac + iPhone 17 Pro peer, engine `llamaShard`). Testing began on the
Qwen2.5-1.5B baseline; after BUG-1 was isolated, all functional coverage ran
on `qwen2.5-0.5b-instruct`. The coordinator was booted 6 times during the run
(4 crashes + 2 clean restarts by the lead).

---

## ✅ FIX STATUS — second pass, same day (4 fix agents + lead)

**All 20 bugs below are FIXED and live-verified.** Uncommitted in the working
tree (nothing was committed). Verification: full `swift test` green ×5 (mesh
agent) + ×2 (api agent) + ×3 with `NMP_LLAMA_MODEL`=1.5B on the integrated
tree; mesh redeployed on the **1.5B default** and a live acid battery passed:
1.5B generation 200 @ 11.9 tok/s ("Paris is the capital of France."), slice
select → 400, already-active select → 200 no re-exec, unknown-device allocate
→ 404, objective wired (200/400), modeled comparison 4 rows, local-placement
race → 409, benchmark echoes runs_requested/completed, max_tokens 0 → 400 /
100000 → clamped+`max_tokens_effective`, empty chat → 400 (allowed with
explicit title), metrics: `manual_mode` + always-present coordinator card +
labeled totals/footprints, `/api/devices` shows explicit exclusion rows.

Headline root causes:
- **BUG-1**: the 1.5B GGUF has NO `output.weight` (tied word embeddings);
  the shim's eval graph passed the NULL `ggml_get_tensor` result straight to
  `ggml_mul_mat` (`->ne[0]` = offset 0x10). Fixed: tied-LM-head fallback to
  `token_embd.weight` + open-time tensor validation + NULL-checked graph
  build (segfault class → clean thrown error). Verified bit-exact vs a
  llama.cpp greedy oracle on both models; `NMPGGUFSlicer` got matching
  tied-awareness (tail slice carries `token_embd.weight`).
- **BUG-4**: `recordMeasurement` stored `max(0, stage − selfCompute)` — a
  throttled peer yields RTT 0, consumed as a free hop. Fixed on both sides:
  bogus RTTs never stored; unmeasured RTTs cost a conservative 30 ms prior.
- **BUG-3**: liveness was activity-only (a backgrounded iPhone keeps pinging)
  and the real mesh had no eviction path; per-connection rejoin IDs left
  routed-to ghosts. Fixed: compute-stall evidence (2 consecutive stage
  timeouts) overrides heartbeats → immediate retire + re-shard; same-device
  rejoin retires the stale identity; dashboard drops departed peers.
- **BUG-7**: autobalance now stages the plan and answers immediately; the
  assign round commits in the background (`onBackgroundReshard`, new benign
  `.assignmentSuperseded` outcome).

Residuals — all three closed in a follow-up pass the same day:
1. **iPhone app rebuilt + reinstalled** (fixed shim via
   `setup_shard_ios.sh` xcframework → headless `xcodebuild` →
   `devicectl install/launch`). END-TO-END PROOF on the live mesh: capacity
   strategy vault-streamed the 1.5B TAIL slice (layers 16–27, incl. the tied
   LM head) to the phone, and a distributed generation returned coherent text
   at 7.9 tok/s (`shard_count:2`, 240 KB mesh traffic), mesh healthy after.
2. **Shim buffer leak FIXED**: `nmp_shard` now owns its
   `ggml_backend_buffer_t` handles (weights + KV) and frees them in
   `free_shard_internal`; allocation failures now clean-fail the open instead
   of yielding data-less tensors. Proof: RSS flat (+0 MB) across 12 full
   open/free cycles (previously ~470 MB leaked per cycle).
3. **Slice-test flake — mechanism fixed**: the one-off golden mismatch traced
   to leak-driven memory pressure + silently-ignored buffer-alloc failures
   (both now fixed; the 1.5B llama test class dropped 56 s → 21 s).
   Verified by repeated full-suite runs with the 1.5B gates.
4. Web UI rebuilt (`Public/` regenerated); saved-chat rows created before the
   fix keep their old inconsistent `model` strings (display-only).

The original findings below are preserved as the test record.

---

## 1 · Bug list

### SEV-1

**BUG-1 · Any generation on Qwen2.5-1.5B-Instruct segfaults the coordinator (100% reproducible, 4/4).**
The 1.5B — the ACTIVE boot model, marked `usable:true, recommended:true` by
`/api/models`, and actively suggested by `/api/devices/metrics.adaptive_model`
(`"decision":"switch" … "chose Qwen2.5 1.5B Instruct"`) — kills the whole mesh
on the first token of any generation.

```bash
# with the 1.5B active:
curl -s -m 60 -X POST http://localhost:3000/api/inference \
  -H 'Content-Type: application/json' -d '{"prompt":"Say OK.","max_tokens":8}'
# EXPECTED: 200 + tokens (0.5B: 200 in 0.43–0.56 s, 14–18 tok/s)
# ACTUAL:   empty reply (HTTP 000) ~0.8 s; coordinator SIGSEGV; entire mesh down
```

Identical signature in all four crashes (`.ips` 01:14:25 / 01:24:03 / 01:26:10
+ live-symbolicated 4th at 01:27:58): `EXC_BAD_ACCESS — KERN_INVALID_ADDRESS
at 0x10` (null struct + 16), queue `nmp.orchestrator.local-compute`:

```
0  ggml_mul_mat + 32                      libggml-base.0.15.3.dylib
1  nmp_shard_eval + 2776                  libnmpshard.dylib (Vendor/llama, Jul 14)
2  NMPLlamaShard.evalTokensToTopK         Sources/NMP/LlamaShardRuntime.swift:330
   (inside withLockedEval, :355 — eval lock held)
3  NMPLlamaShardComputeEngine.runLayers   Sources/NMP/LlamaShardEngine.swift:138
   (shard.isFirst && shard.isLast full-range local path)
4  NMPInferenceOrchestrator.computeStageOnQueue  InferenceOrchestrator.swift:483
```

Isolation: controlled single request with zero other traffic reproduces it;
the 0.5B is unaffected under full 6-agent load. Shim (Jul 14) and brew ggml
0.15.3 (Jul 9) predate the last known-good 0.5B use (chat saved Jul 16 21:02).
The crash sits on **uncommitted working-tree changes** to the inference path
(`InferenceOrchestrator.swift`, `PromptInferenceService.swift`,
`ModelSharder.swift`, `PeerNode.swift`, `DashboardServer.swift`,
`NMPDashboardCLI/main.swift`) on top of `80a06e6` (Jul 15); the 1.5B gguf
arrived Jul 16 — no evidence it ever completed a generation on this tree.
Aggravating: **no supervisor** (one segfault = mesh down until manual
relaunch); the crasher is the boot default AND recommended AND
adaptive-chooser-suggested, with no feedback loop from crash history to model
flags. Mitigating: the chat store survived every crash byte-intact.
Triage entry: diff the uncommitted inference-path changes vs `80a06e6`; a
single 1.5B inference crashes in <1 s under a debugger.

### SEV-2

**BUG-2 · `/api/models/select` accepts split GGUF fragments; the mesh restarts onto a partial model and serves garbage as HTTP 200.**
```bash
curl -s -m 15 -X POST http://localhost:3000/api/models/select \
  -H 'Content-Type: application/json' \
  -d '{"path":"/Users/divyamkataria/models/qwen2.5-0.5b-instruct-q4_k_m_part1.gguf"}'
# EXPECTED: 4xx (fragment; not even listed by /api/models)
# ACTUAL:   200 "switching to qwen2.5-0.5b-instruct_sliced_0_12 — the mesh restarts…"
```
The mesh re-exec'd onto a **13-of-24-layer slice** and reported healthy.
While the fragment was live: `/api/models` showed **zero** `active:true`
models (exactly-one-active invariant broken), `/health.model` exposed the
internal slice name, and — cross-correlated to the second — the
inference-agent's generations in that window (01:32:05–01:32:46) returned
**punctuation soup and cross-request token bleed as HTTP 200**, self-curing
once the real 0.5B was restored. Related inconsistency: the `_part1/_part2`
files are silently *omitted* from `/api/models` yet *accepted* by select.

**BUG-3 · A stalled in-plan phone shard blackholes ALL inference ~3 min while liveness says alive; rejoins leave a routed-to ghost peer.**
With the iPhone holding layers 14–23 and the app losing foreground, every
generation on `/api/inference` and `/api/chat` failed for ~3 min with
`500 "mesh orchestration failed: inferenceTimeout(peerID: 2283848824, …)"`
(~25 s each) while `/health` said `peers_alive:2` and `/api/devices` showed
the phone `alive:true` throughout. The phone's rejoin minted a NEW peer ID
but generations kept chasing the stale one (seen independently by two agents;
plus a 0.49 tok/s degraded window). The 4 s/token bound held (bounded, clean
JSON errors, no hangs — that part is correct); the bugs are (a) liveness
never reflects the stall, (b) stale-ID routing, (c) ~3 min of guaranteed-fail
before eviction. Repro: `{"strategy":"balanced"}`, background the phone app,
then any small inference → 500 after ~25 s, repeatably, until eviction.
Residue: `/health` kept `peers_alive:2` (ghost + real) until the next reboot.

**BUG-4 · Auto/speed optimizer inversion after churn: "best for speed" puts ALL layers on the phone.**
Later in the session (post re-shards/rejoins/restarts), auto+speed assigned
all 24 layers to the iPhone and 0 to the Mac — every token pays a Wi-Fi round
trip, contradicting the plan's own "fewest hops" note and the same boot's
earlier `exclusion_reason: "Mac-only is faster for this model"`. `/api/devices`
reported `latency_ms:0` for the phone all session (implausible) — the
optimizer likely consumes unmeasured/zeroed latency after restarts. A fresh
boot decides correctly (Mac-only, phone excluded with reason), confirming
this is degraded-state behavior, not design. This inversion is also what
arms BUG-3 (phone in the hot path by default).

**BUG-5 · Busy collisions return HTTP 500 (not 429) on benchmark/comparison endpoints.**
`/api/inference` correctly 429s when busy (dual-race verified: exactly one
200 + one 429 in 1.9 ms, no stuck flag). But while any generation runs:
`/api/comparison/run` → `500 {"error":"generation failed: busy"}`;
`/api/benchmark/run` → `500 {"error":"run 1 failed: busy"}`. Contention is a
normal state being reported as a server error.

**BUG-6 · `/api/comparison/run` returns 500 for the default local-only placement.**
Under speed + 0.5B (phone at 0 layers — the documented-correct placement):
`500 "this run moved no mesh traffic (local placement?) — nothing to race"`.
Honest refusal (nothing fabricated — good), wrong class: should be 4xx/409.
Net effect: the measured race is unavailable in the mesh's default state.

**BUG-7 · Real autobalance mode changes stall 22–30+ s (client timeouts) when the phone holds layers.**
`POST /api/mesh/autobalance` real change with the phone as an active shard:
`{"enabled":false}` timed out at 30 s (HTTP 000); `{"enabled":true}` took
22.6 s; multiple 000s at -m 10/-m 15. No-op toggles: ~50 ms; same toggles
with phone at 0 layers: <30 ms. Roughly within the ~30 s re-shard envelope
but exceeding it at least once, and the asymmetry is undocumented — UI-scale
clients will time out. `/health` stayed <5 ms throughout (not a deadlock).

**BUG-8 · The coordinator vanishes from `/api/devices/metrics` when it holds 0 layers.**
In the all-on-phone state: `peers[]` contains ONLY the phone;
`totals:{devices:1, devices_alive:1}` for a 2-device mesh. (Mirror image of
the excluded-phone case, which IS rendered correctly with `excluded:true` +
`exclusion_reason`.)

**BUG-9 · `/api/devices` serves stale assignments for 0-layer devices — overlapping layer claims.**
After speed→all-phone: coordinator row still `layers 0-13` (with the phone at
0-23 ⇒ 38/24 layers claimed). After `share:0`: phone row still `layers 0-19`
(44/24). Later: BOTH rows `layers 0-23` (48/24), persisting for minutes.
`/api/devices` never renders an excluded device correctly — it either omits
it (fresh boot) or shows stale layers.

**BUG-10 · `/api/devices/<id>/allocate` accepts unknown device ids.**
`curl -X POST …/api/devices/ffffffff/allocate -d '{"share":0.5}'` →
`200 {"peer":"ffffffff","status":"ok"}`. Expected 404/4xx. (Share *values*
are validated: 1.5 / -0.2 / "abc" / missing → clean 400s.)

**BUG-11 · `/api/mesh/objective` is documented but unwired.**
Valid and invalid bodies alike → `503 "no objective handler is wired to this
server"`. Either wire it or remove it from the surface.

### SEV-3

**BUG-12 · Restart serves false-healthy `/health` (~80 s window):** `status:"ok"`
with `engine:"reference", model:"", peers:0` before the real engine loads —
anything gating on `/health` alone reads a nonfunctional mesh as healthy.

**BUG-13 · `POST /api/chats {"messages":[]}` persists an empty "New chat" row**
(200) instead of 4xx. Possibly intentional UI backing — document or reject.

**BUG-14 · Saved-chat `model` field is inconsistent** — display name
(`"Qwen2.5 1.5B Instruct"`), slug (`"qwen2.5-0.5b-instruct"`), or internal
slice artifact (`"qwen2.5-0.5b-instruct_sliced_0_12"`) depending on mesh
state at save time.

**BUG-15 · Silent parameter clamping:** `/api/inference` `max_tokens:0/-5` →
accepted, clamped to 1; `100000` → clamped to 128 with no indicator.
`/api/benchmark/run` `runs:0/-1` → clamped to 1; `runs:100` → capped at 10,
no indication either way.

**BUG-16 · Benchmark/comparison-path generations stream WS tokens with no
`generation_started`/`generation_complete` framing** (`/api/inference` and
`/api/chat` frame correctly) — WS clients can't delimit those transcripts.

**BUG-17 · Modeled `/api/comparison` has no plain-TCP row** (NMP/TLS/QUIC
only) while the measured race has 4 transports — the model's TLS-vs-TCP
claim can't be audited.

**BUG-18 · Device-count disagreement across endpoints at the same instant:**
`/health peers_alive:2` vs metrics `totals.devices:1` vs `/api/devices` 2
rows (each with its own definition of "device"; none labeled).

**BUG-19 · Allocation share not echoed & footprint semantics mixed:**
requested `share:0.5` reported back only as a layers-fraction (0.708);
the same 24 layers appear as 235.8 MB in plans vs 463.0 MB in devices
(model-only vs +KV/overhead? unlabeled), and one range flip-flopped
39.3↔176.1 MB.

**BUG-20 · `allocation_supported` is a mode indicator, not a capability flag**
(false in auto, true in manual, while `/allocate` works in both) — the name
misleads; rename or document.

---

## 2 · Coverage checklist (vs `Docs/Test_Surface.md`)

Legend: ✓ tested · ✗ blocked/not testable. Unless noted, coverage ran on the 0.5B.

| Surface item | Status | One-line result |
|---|---|---|
| GET /health | ✓ | OK + drives outage detection all session; false-healthy restart window (BUG-12); ghost peer count (BUG-3) |
| GET /api/devices | ✓ | Plan rows OK; stale/overlapping assignments for 0-layer devices (BUG-9) |
| GET /api/devices/metrics | ✓ | Counters sane (RAM math exact); excluded phone rendered correctly; coordinator vanishes at 0 layers (BUG-8); `adaptive_model` recommends the crashing 1.5B (BUG-1) |
| GET /api/clients | ✓ | 200s under storm; counts web UI clients 1–3; raw WS conns not deterministically counted (noted) |
| GET /api/models | ✓ | Flags correct, exactly-one-active (except fragment window), cache 16–25 ms byte-identical; fragments omitted (BUG-2) |
| GET /api/chats, /api/chats/<id> | ✓ | Full lifecycle PASS; unknown/traversal ids clean 404 |
| GET /api/mesh/plans | ✓ | 3 plans (speed 24/0, balanced 23/1, capacity 14/10), sums exact, 9.8 MB/layer, % = footprint/ram, all ≤100 |
| POST /api/inference | ✓ | Happy path + 11 edge cases PASS on 0.5B; on 1.5B = BUG-1; max_tokens clamping (BUG-15) |
| POST /api/chat | ✓ | ChatML template correct; multi-turn correct; degenerate bodies clean 400s |
| POST /api/chats (create/update) | ✓ | id/createdAt preserved, title derivation (~81-char truncation), unicode + 100 KiB byte-identical; empty-messages row (BUG-13) |
| POST /api/chats/<id>/delete | ✓ | `{"deleted":true}` 200 / unknown `{"deleted":false}` 404 |
| POST /api/benchmark/run | ✓ | Math exact (avg/stddev recomputed); runs honored/capped (BUG-15); busy→500 (BUG-5) |
| POST /api/comparison (modeled) | ✓ | Labeling honest, math exact, rtt/loss monotonic; no TCP row (BUG-17) |
| POST /api/comparison/run (measured) | ✓ | 4 transports all measured-labeled, leg sums + splice projections exact; 500 on local placement (BUG-6); busy→500 (BUG-5) |
| POST /api/models/select | ✓ | Bad path/malformed/empty → clean 400; bare-filename resolves; same-model select re-execs (~10–14 s, noted); fragment accepted (BUG-2). **0.5B⇄1.5B round-trip deliberately NOT run** — any ambient generation during a 1.5B window kills the mesh (BUG-1); switch mechanics proven 3× on the 0.5B |
| POST /api/mesh/objective | ✓ | Unwired: 503 for all bodies (BUG-11) |
| POST /api/mesh/autobalance | ✓ | Flag flips + re-shards correctly; bad body 400; 22–30 s stall on real change with phone in plan (BUG-7) |
| POST /api/mesh/strategy | ✓ | All 3 applied + verified; `warp`/""/{} → 400; re-shard keeps API <40 ms |
| POST /api/devices/<id>/allocate | ✓ | share 0.5/1/0 all behave (1 keeps a 4-layer coordinator floor; 0 = visible exclusion); bad shares 400; unknown id accepted (BUG-10) |
| WS /ws | ✓ | Full event schema captured (started/token/progress/complete/mesh_event/client_update); unframed benchmark generations (BUG-16) |
| Distributed sharded inference | ✓ | Real phone-in-path generation at 9.1–9.9 tok/s; local 13–18.5 tok/s |
| Network-aware balancing | ✓ | Correct on fresh boot (Mac-only + reasoned exclusion); inverted after churn (BUG-4) |
| Plan preview (3 strategies) | ✓ | Sensible, math exact |
| Manual shares + exclusion | ✓ | Works incl. share 0 (excluded-but-visible in metrics) |
| Model switching | ✓ partial | Mechanics verified on 0.5B; 1.5B blocked by BUG-1 |
| Model download (phone) | ✗ | Phone-UI only — see manual checks |
| Persistent compressed chat history | ✓ | Crash-safe, LZFSE lossless, compression at ~7–8 min idle |
| Fault tolerance (drop/join re-shard) | ✓ | Peer drop → bounded errors + eventual eviction, but see BUG-3; rejoin works (new peer ID each time — known) |
| 4 s per-token timeout | ✓ | Held everywhere (~25 s bounded failures, no hangs) |
| Transport race + benchmark suite | ✓ | Numbers + labels audited exact |
| Speculation flag | ✓ | `speculation_available:false` respected; flag graceful, never 500 |
| Concurrency (the multi-user test) | ✓ | Dual-race, storms, toggle storms, WS flood, slowloris — all invariants held (details §4) |
| Installable PWA | ✗ | Browser-only — see manual checks |

## 3 · Headline measured numbers (0.5B)

| Metric | Value | Label |
|---|---|---|
| Local-placement benchmark (runs:2) | 18.05 tok/s · 886.59 ms avg/16 tok · stddev 13.95 ms | measured |
| Generation with iPhone in path | 9.1–9.9 tok/s (12 tok) | measured |
| Race legs (ms) | NMP 4.28–4.67 · TCP 1.88–2.35 · TLS 20.25–22.87 · QUIC 10.14–11.97 | measured (loopback) |
| Splice projections (ms/12-tok gen) | TCP 1208.88 · TLS 1229.87 · QUIC 1217.14 | projection (measured splice) |
| Stalled-phone failure | 500 after ~24.7 s/request | measured |
| Stale-peer degraded window | 0.49 tok/s | measured |

**Honest-measurement hard-rule audit: PASS everywhere tested** — every number
correctly labeled measured/modeled/projected; recomputed arithmetic exact;
the one no-data case refused rather than fabricated (status code aside, BUG-6).

## 4 · What held up well

- Serialization: simultaneous inferences → exactly one 200 + one 429 (1.9 ms), busy flag never sticks.
- Re-shard vs generation (0.5B): strategy change 0.2 s into a live generation **queued** (6.06 s) — no teardown race.
- Under storm: 12-way metrics floods during re-shards <0.15 s each; 20-way mixed GETs <0.05 s; 5 held WS conns + slowloris never blocked accepts (`/health` <5 ms).
- Chat store: crash-safe across 4 segfaults (one write 11 s pre-crash), byte-identical LZFSE round-trips at 100 KiB, clean delete semantics, correct ordering.
- Parse-level validation: every malformed/missing-field probe across all endpoints → clean 400/404, zero 500s, zero wedges; 50 KB prompt → 400 `promptTooLong` in 10 ms.
- Recovery: phone peer rejoined unaided after every one of 6 boots; model-switch re-exec recovers in ≤14 s.

## 5 · UI / manual checks (agents can't click — for the human)

1. **Models tab**: with the 1.5B still `recommended:true`, does the UI visually push the mesh-killing model (badge/sort)? Does "Restart the mesh onto this model" appear for it? (BUG-1 blast radius.)
2. **Run/Chat tabs during the busy state**: fire a generation from two browser tabs — is the second user shown a friendly "busy" or a raw error? (Related: BUG-5 500s on Benchmark/Compare tabs.)
3. **Compare tab in default placement**: with the phone excluded (speed), does the measured-race card show a sensible "nothing to race" message or an ugly 500? (BUG-6.)
4. **Devices tab**: in manual mode drag a slider to 0 — does the phone stay visible with an exclusion badge? Does the all-on-phone state show the coordinator card disappearing (BUG-8) and stale layer ranges (BUG-9)?
5. **Devices tab auto/manual toggle**: does the UI survive the 22–30 s stall (BUG-7) with a spinner, or does it error/hang?
6. **Nav health pill during a model switch/restart**: does it show "reconnecting…" then green during the false-healthy window (BUG-12), i.e., green before the mesh can actually generate?
7. **Chat tab**: sidebar new/select/delete; auto-title truncation at ~81 chars renders OK; a `compressed:true` chat opens seamlessly; the `model` field garbage (`…_sliced_0_12`, BUG-14) — is it displayed to users?
8. **WS transcript during a Benchmark run**: does the live transcript misrender benchmark generations (no started/complete framing, BUG-16)?
9. **iPhone app**: background the app mid-generation with the phone in the plan → what does the phone UI + web UI show during the 3-min blackhole (BUG-3)? Does the phone's Models view still offer the 1.5B (mesh-killer) for a mesh-wide switch? Model download flow (other Qwen sizes) works?
10. **PWA**: install from Share ▸ Add to Home Screen; verify it still works over plain LAN http (no service worker) and reconnects by itself.
11. **Pressure & Settings tabs**: not exercisable via API inventory — click through for rendering/actions.
12. **`sample` capture**: if the 1.5B crash needs a live backtrace beyond the `.ips` files, run a 1.5B generation with `sample nmp-dashboard 5` attached (it dies in <1 s, so start sampling first).

## 6 · End-of-run mesh state (verified 01:48, epoch 1784233131)

| Item | State |
|---|---|
| /health | 200 ok — llamaShard, **qwen2.5-0.5b-instruct**, peers 1/1 alive (no ghost) |
| Strategy / balance | `speed` + `auto_balance:true` |
| Placement | Coordinator: layers 0–23 + tokenizer; phone connected-but-excluded with correct reason ("Mac-only is faster for this model") |
| Chats | Exactly the 1 pre-existing protected chat (`8829…3d52`, "hi") — loads intact |
| Inference | Final sanity generation: 200, 8 tokens, 18.53 tok/s; nothing in flight |
| Crash artifacts | 3 `.ips` reports (`~/Library/Logs/DiagnosticReports/nmp-dashboard-2026-07-17-*.ips`) + symbolicated logs in the session scratchpad (`dashboard*.log`) |

⚠️ **Deliberate deviation from the brief**: the mesh is left on the **0.5B**,
not the 1.5B it booted with. The 1.5B is a proven one-request mesh-killer
(BUG-1) with live web clients attached; re-arming it as the active model
would leave the mesh one click from death. Switch back only when BUG-1 is
fixed: `curl -X POST localhost:3000/api/models/select -H 'Content-Type:
application/json' -d '{"path":"/Users/divyamkataria/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"}'`
