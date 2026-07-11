# Mesh 2.0/2.1 — Multi-Device Web UI, Protocol Comparison & Setup Guide

**Goal**: replace the terminal-only experience with a browser UI served
by the coordinator itself — same interface, live, on every device on the
Wi-Fi at once — plus a protocol comparison that is honest about what is
measured and what is modeled.

```
Mac coordinator (one process):
  ├─ NMP mesh (Phases 1–9)
  ├─ HTTP + WebSocket server on 0.0.0.0:<port>   (NWListener, no deps)
  ├─ Bonjour service advert (_neuramesh-ui._tcp)
  └─ React UI (built into Public/, committed)

Phone / laptop / tablet browser:
  http://<your-mac>.local:3000   or   http://192.168.x.x:3000
```

## 1. What Mesh 2.0 is (and is not)

Same honesty discipline as Phases 8–9:

- **No new server framework.** The build spec sketched Vapor; this repo's
  rule is Apple-native, zero SwiftPM dependencies — so the existing
  Phase 6 `NMPDashboardServer` (NWListener, hand-rolled HTTP + RFC 6455)
  grew the new routes, CORS, and static-file serving. The React toolchain
  lives in `web/` as a separate npm project whose BUILD OUTPUT is
  committed to `Public/`, so `swift run nmp-dashboard --ui` needs no npm.

- **`neuramesh.local` is not a thing mDNS can conjure.** Advertising a
  Bonjour *service* named "neuramesh" does not create a *hostname*;
  browsers resolve the machine's own Local Hostname (System Settings ▸
  General ▸ Sharing). The startup banner therefore prints your Mac's REAL
  `<hostname>.local` plus its LAN IPs and a QR code (CoreImage — still
  Apple-native) that phones scan straight into the UI. If you want the
  URL to literally read `http://neuramesh.local:3000`, rename the Mac's
  Local Hostname to "neuramesh" — that's a one-line System Settings
  change, and the only honest way to get it.

- **The protocol comparison labels its rows.** The NMP row is the run
  that just executed — measured wall clock, measured payload, measured
  round trips. The TCP+TLS and QUIC rows are that same run *re-priced*
  with each protocol's transport costs (handshake RTTs, per-trip
  overhead, loss recovery), anchored to constants measured in this repo
  where they exist (Noise IK loopback handshake ≈ 1.0 ms; Phase 3 FEC
  recovery ≈ 0.15 ms/packet) and documented coarse constants where they
  don't. Every estimate carries `measured: false` and its assumptions,
  and the UI renders "modeled" badges. No fake benchmarks.

- **Trusted LAN only.** No TLS, no auth — same scope as the Phase 6
  dashboard, now explicitly on all interfaces. The banner and the UI
  both carry the warning: don't port-forward this.

## 2. Architecture

### Server (Swift, `WebUI.swift` + `DashboardServer.swift`)

| route | what it serves |
|---|---|
| `GET /` + SPA routes | the built React app from `Public/` (index.html fallback; traversal-guarded; `/legacy` keeps the Phase 6 page) |
| `GET /health` | `{status, mesh: {engine, model, shard_count, wire_format, speculation_available, peers, peers_alive}}` |
| `GET /api/devices` | live per-peer snapshots (name, layers, latency, load, liveness) |
| `POST /api/inference` | Phase 8/9 generation + `round_trips`, `wire_format`, optional `protocol_comparison` (`enable_comparison: true`) and `speculation` stats (`enable_speculation: true`) |
| `POST /api/benchmark/run` | N sequential generations → avg/σ latency, per-run table (runs clamped to 10) |
| `POST /api/comparison` | the comparison model over any measured numbers (what-if RTT / loss) |
| `GET /ws` | the Phase 6 live stream (peer updates, mesh events, loss control) |

All API responses carry permissive CORS (trusted-LAN tool; enables the
Vite dev server). `NMPWebUIBroadcaster` advertises `_neuramesh-ui._tcp`
with the UI port in its TXT record; `NMPLANIdentity` reports the real
hostname + IPv4s; `NMPQRCode` renders the QR via CIQRCodeGenerator.

### Web app (`web/` → `Public/`)

React 18 + Vite + TypeScript, no other runtime dependencies (no router —
five views behind a tab bar). `src/styles/tokens.css` is the design-token
sheet (the Figma "NeuraMesh UI Kit" values: color scale, Inter/Menlo
type, 4px spacing grid, elevation); every component styles itself off
those variables only.

- **Mesh** — device grid, per-peer cards, live event log (WebSocket).
- **Run** — prompt → generation with throughput/latency/payload/round
  trips, speculation toggle (enabled only when the mesh reports it),
  protocol comparison attached to the real run.
- **Benchmark** — N-run averages with run-to-run σ.
- **Compare** — what-if explorer over `/api/comparison`: re-price a
  measured run at your chosen RTT and loss rate; defaults are the
  Phase 9 zero-trim measurement.
- **Settings** — mesh facts + the packet-loss chaos slider (drives the
  Phase 6 `set_loss_rate` control over the WebSocket).

Every open tab converges on the same state (3 s polling + WebSocket
pushes) — phone and laptop genuinely watch the same mesh.

## 3. Setup guide (zero friction)

### One-time

```bash
git clone <repo> && cd NeuraMeshProtocol
brew install llama.cpp && scripts/setup_llama.sh   # for real-LLM mode
# a model, e.g.:
mkdir -p ~/models && cd ~/models
curl -LO https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

The web UI ships prebuilt in `Public/` — npm is only needed if you edit
`web/` (`cd web && npm install && npm run build`).

### Every time (one command)

```bash
# reference mesh (no model needed):
swift run nmp-dashboard --ui

# real LLM, auto-configured (zero-trim wire, cached profiles):
swift run nmp-dashboard --ui --engine llamaCpp \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf --auto-config

# + speculative decoding with a same-vocab draft model:
swift run nmp-dashboard --ui --engine llamaCpp \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf --auto-config \
    --draft-model ~/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

`--ui` defaults the port to 3000 (pass a port number to override) and
prints the access banner: your real `<hostname>.local:3000`, the LAN
IPs, and a QR code to scan from a phone.

### From your devices

Open the printed URL in any browser on the same Wi-Fi — Mac, iPhone
Safari, iPad, Android. Multiple devices at once is the point.

| symptom | fix |
|---|---|
| `.local` doesn't resolve (some Androids) | use the printed IP URL |
| nothing loads from another device | same Wi-Fi? macOS firewall prompt allowed? guest/isolated network blocks mDNS + peer traffic |
| UI shows "mesh unreachable" | the coordinator process stopped — restart it |
| page is the old Phase 6 dashboard | `Public/index.html` missing — rebuild `web/` or pull the committed build |
| want the URL to say `neuramesh.local` | rename the Mac's Local Hostname (System Settings ▸ General ▸ Sharing) |

## 4. Measured: TinyLlama draft-model speculation (new)

Phase 9 left draft-model speculation unmeasured. Now measured (Apple M3,
target Llama-2-7B-Chat Q4_K_M, drafter TinyLlama-1.1B-Chat Q4_K_M —
same 32 000-token Llama-2 vocabulary — 32 tokens, in-process remote
placement, zero-trim wire):

| prompt | drafting | round trips | acceptance | payload | throughput | output |
|---|---|---|---|---|---|---|
| natural ("The future of AI is") | none (plain) | 32 | — | 11 928 B | 14.3 tok/s | reference |
| natural | prompt-lookup (Phase 9) | 32 | 0/8 | 11 400 B | 11.7 tok/s | **identical** |
| natural | **TinyLlama** | **12** | 21/39 = **54%** | **1 524 B** | 8.5–9.3 tok/s | **identical** |
| Q&A ("What is the capital of France?") | **TinyLlama** | **10** | 23/32 = **72%** | 1 340 B | 11.4 tok/s | **identical** |

What this says, honestly:

- The draft model **fixes the acceptance problem** prompt-lookup has on
  natural text (0% → 54–72%) and cuts round trips ~3× and payload ~8×,
  with output still token-for-token identical.
- **In-process, wall clock LOSES** (8.5–11.4 vs 14.3 tok/s): the 1.1B
  drafter competes with the 7B verifier for the same M3 GPU, and a trip
  saved costs nothing when both ends share one machine. Speculation
  trades *drafting compute* for *round trips* — the trade only pays
  where round trips are expensive: a real two-device mesh (Phase 8
  measured ≈ 8–17 ms protocol + radio per trip; 20 saved trips ≈
  160–340 ms back per 32 tokens) with drafting on the otherwise-idle
  coordinator. That is exactly the deployment the mesh exists for.
- Rule of thumb the docs now state: **prompt-lookup for repetitive
  text, draft model for natural text on a physical mesh, plain Phase 9
  zero-trim when the coordinator and peer share a GPU.**

## 5. Mesh 2.1 — streaming, the measured race, and the device panel

Mesh 2.0 shipped a UI whose inference view was request/response, whose
TCP/QUIC numbers were a model, and whose "peers" never included the
browsers. Mesh 2.1 fixes all three, still on the zero-dependency stack.

### 5.1 Real-time token streaming (every browser, every run)

Both generation services (`NMPPromptInferenceService`,
`NMPSpeculativeGenerationService`) now expose `onToken`, and the
dashboard broadcasts each confirmed token over the EXISTING `/ws`:

```
{"type":"generation_started","prompt":…,"max_tokens":N,"speculative":bool}
{"type":"generation_token","text":…,"index":…,"count":i,"requested":N}
{"type":"generation_complete","output":…,"tokens_per_sec":…,"round_trips":…}
{"type":"generation_failed","error":…}
```

The Run view renders the stream with a live cursor; a phone watching the
dashboard sees the tokens of a run the laptop submitted, as they are
produced, and both converge on identical final metrics. (This also
covers "tokenizer state sync": there is exactly one generation state —
the coordinator's — and everyone watches it.) No SSE endpoint was added:
the Phase 6 WebSocket already reaches every open tab.

### 5.2 The MEASURED race — `POST /api/comparison/run`

The spec asked for "actual protocol comparison, no modeled numbers" and
sketched code with `handshakeMs: 55.0 // measured` hardcoded. This repo
does it for real (`TransportRace.swift`): run a real generation, then
replay its exact traffic pattern — round trips × payload, chunked into
1024-byte sends like real mesh traffic — over real loopback sockets:

- **NMP leg**: production `UDPListener`/`UDPTransport` + `PeerConnection`
  — real Noise IK handshake, AES-256-GCM on every packet, FEC parity.
- **TCP leg**: `NWConnection` stream — real 3-way handshake, raw bytes,
  no TLS, no framing.

Every number in the response is a wall-clock measurement (`measured:
true` on both legs, no modeled fields). The "projected" rows splice the
measured generation with each leg's measured transport time — arithmetic
on measurements, labeled as such. What it honestly shows (Apple M3):

- Reference mesh, float32 wire (147 KB / 6 trips): plain TCP moves the
  bytes ~44 ms faster — raw kernel TCP does no crypto and no FEC. That
  is the real cost of NMP's security + recovery machinery on a clean
  loopback, stated instead of hidden.
- Llama zero-trim (3 KB / 8 trips): NMP handshake 0.98 ms, transfer
  1.4 ms vs TCP 0.36/0.92 ms — **~0.5 ms total for full encryption**,
  and TLS (which production TCP would need) costs more than that in
  handshake alone.
- QUIC is NOT raced: Network.framework QUIC requires a TLS identity
  (a certificate) a zero-dependency LAN tool can't conjure honestly.
  It stays in the what-if model, labeled modeled, on the Compare tab.

### 5.3 Device panel — live resources + allocation that really allocates

`GET /api/devices/metrics` (2 s polling in the Devices tab) serves:

- **Host**: real kernel counters — RAM used (host_statistics64), storage
  (statfs), CPU% (tick deltas), and this process's physical footprint
  (task_info) — watch the footprint jump when a model loads and the CPU
  bar move during generation. Honesty note included in the payload: all
  in-process mesh peers genuinely share this host.
- **Peers**: assigned layer range, span, measured ms/layer, compute
  share, computing flag, liveness.

`POST /api/devices/<id>/allocate {"share": 0.4}` is the slider. It is
NOT a cosmetic knob: the share feeds `NMPModelSharder.plan` (a device at
40% is planned as 2.5× slower) and triggers a live re-plan through the
normal SHARD_ASSIGN round. Verified live: capping one of four peers at
40% shrank its span 6 → 3 layers; every open browser saw the new plan
(`allocation_update` + `peer_update` pushes) and the event-log line.
That visible re-shard — and the host counters moving with it — is the
verification the panel exists to provide. The llama path reports
`allocation_supported: false` with the reason (a llama plan is one
full-range shard; there is nothing to re-balance).

RAM/storage "allocation" sliders from the spec were deliberately NOT
built: with in-process peers there is nothing real for them to cap, and
a slider that does nothing would be UI theater. They render as measured
bars instead.

### 5.4 Web-client tracking (the "iPhone shows up" fix)

Mesh peers and browsers are different populations, tracked separately:
`/health` now carries `web_clients`, `GET /api/clients` lists them
(address, browser, WebSocket-live vs last-seen), and the Devices tab
shows the list. "Connected" = holding `/ws` open, or any HTTP request
within 15 s (the UI polls every 2–3 s). A phone opening the page bumps
the count immediately; closing it drops the WebSocket instantly.

One CFNetwork gotcha worth recording: a server-initiated WebSocket frame
that shares a TCP segment with the 101 upgrade response makes
URLSessionWebSocketTask/Safari fail the handshake — the `client_update`
broadcast is therefore delayed 250 ms off the upgrade path.

### 5.5 The installable PWA (Mesh 2.2)

The web app is a proper PWA served by the mesh itself: manifest +
generated icons (`scripts/make_pwa_icons.py` → `web/public/`), iOS
Add-to-Home-Screen metadata, a "Looking for your mesh…" screen that
polls until the coordinator answers (and re-finds it after outages,
with a "✓ Connected to <hostname>" toast), and the coordinator's real
Local Hostname in `/health` (via `SCDynamicStoreCopyLocalHostName` —
`gethostname()` can return a bare DHCP IP, which must not get `.local`
appended).

Design decisions, honestly:

- **Installed FROM the mesh, not from a public domain.** An HTTPS-hosted
  PWA cannot fetch `http://` LAN endpoints (mixed-content policy) and
  browsers have no Bonjour API — "the hosted app scans your network" is
  not implementable. Plex-style per-device TLS certificates + DNS would
  fix it at the cost of real infrastructure; installing from
  `http://<mac>.local:3000` gives the same journey with zero.
- **The service worker only registers in secure contexts** (https /
  localhost) — browsers refuse SWs over LAN http, and the app works
  fully without one. `web/public/sw.js` is the app-shell cache for
  anyone who later fronts the mesh with TLS. Consequence over plain
  http: tapping the icon while the Mac is off shows a browser error, not
  the finder screen — only TLS can change that.
- **A PWA is a control surface, not a compute peer** — no UDP in
  browsers. Phone-as-peer stays native (`CrossDevice_Setup_Guide.md`).

### 5.6 Pre-existing races found while stabilizing

Full-suite hammering surfaced two latent bugs (both pre-date Mesh 2.1,
both now fixed and pinned by 5 consecutive clean 306-test runs):

- `NMPFailoverOrchestrator.activePeers`/`activePlan` were mutated on the
  failover queue and read unlocked from other queues (adaptive
  controller, CLI) — a racing read crashed with index-out-of-range
  inside `Collection.map`. Both are lock-protected now.
- `registerPeer` is async on the failover queue, and the Phase 9
  adaptive controller read membership immediately after testbed
  assembly — under load it saw a partial mesh, concluded the profile
  cache was incomplete, and probed when it should not have (the
  long-standing `testSecondStartupUsesCachedProfileWithoutProbing`
  flake). Assembly now settles membership via `waitForMembership()`.

### 5.7 Mesh 2.3 — full per-device cards (throughput, serves, resources)

The Devices tab now renders one full card per mesh device, everything
measured, with the measurement point labeled:

- **Network ↓/↑ per device**: `PeerConnection` counts every datagram it
  puts on / takes off the wire (handshake, NACKs, FEC parity and
  retransmits included — this is what actually crossed the link). The
  metrics handler diffs totals between polls into live bytes/sec, plus
  cumulative MB since startup. The coordinator's own shard says "local —
  no network hop" instead of pretending a loopback rate is a network.
- **Computing per device**: requests actually served (counted by the
  peer-side shard engine, not inferred), last stage compute ms, seconds
  since last active — `computing` is now "served within 1.5 s", a real
  activity signal instead of "is in the plan".
- **Peer-reported resources**: every shard peer ships its own kernel
  counters over the mesh (`NMPPeerResourceReport`, mesh message kind
  0x06 — once on SHARD_ASSIGN, then at most every 2 s alongside
  metrics). A **physical** peer (`swift run nmp-peer` on a second Mac,
  the iPhone app) reports its own RAM/CPU/GPU/storage and gets real bars
  of its own. An **in-process** peer reports this same host — the
  hostname match is how the UI knows to say "shares the Mac's hardware"
  instead of drawing four identical bars and calling them four devices.
- **Host GPU%**: whole-machine utilization from the accelerator driver's
  own counter (IOAccelerator `PerformanceStatistics` → "Device
  Utilization %" — what Activity Monitor's GPU history reads). The
  reference engine computes on CPU, so expect it to move under llama.cpp
  (Metal), not under the testbed mesh. There is no public per-process
  GPU split — the bar says "whole machine" because that is what it is.
- **Mesh totals**: devices alive, layers assigned, live wire throughput
  summed across links, total shard computations served.

Verified live: 4-peer reference mesh under heartbeat load shows
~13 KB/s each way per link, serve counters climbing in lockstep, and
the 40%-share re-shard still visible on every open browser.

### 5.8 Mesh 2.4 — LAN peers join the web dashboard live

The reference dashboard now browses for `_neuramesh._tcp` adverts and
dials every peer it finds — the iPhone app or `swift run nmp-peer` on
another Mac — over real UDP (Noise IK, key from the TXT record, same
trust-on-first-use as `nmp-coordinator`), then joins it into the SAME
failover mesh the web panel shows. No flags:

```bash
swift run nmp-dashboard --ui     # Mac
# open the NeuraMeshPeer app on the phone (same Wi-Fi) → within ~2 s the
# event log shows "found … dialing … joined — re-sharded", every open
# browser sees the new layer spans, and the phone's device card shows
# its OWN reported RAM/CPU/storage, live wire throughput, and serves.
```

To make this zero-config, the dashboard mesh now matches the peer-app
defaults — **32 layers × 4096 hidden, model tag `nmp-reference-model`**
(it was 24 × 1024 `testbed-ref-model`; mismatched tags made peers
reject SHARD_ASSIGN). The phone dropping off Wi-Fi (or the app closing)
re-shards back to the in-process peers; reopening the app rejoins.
Reference mesh only — a llama plan is one full-range shard and the
weights live on one device, so the llama dashboard does not dial peers.

Verified live with a real iPhone: discovered and joined in ~2 s,
assigned a real layer span by the measured-speed sharder, ~39 KB/s of
encrypted UDP each way under heartbeat load, serve counter climbing,
and its own kernel counters (RAM/CPU; GPU is nil — iOS exposes no
public counter) on the device card.

## 6. Web endpoint verification (from scratch)

```bash
swift test --filter "WebUIRouteTests|ProtocolComparisonModelTests|LANIdentityTests"
# 19 tests: routes, CORS, SPA fallback + traversal guard, benchmark
# aggregation (σ pinned), comparison model math (loss widens the gap),
# banner + QR.

swift test --filter "ResourceMonitorTests|TransportRaceTests|TokenStreamingTests|ComputeShareTests|Mesh21RouteTests"
# 15 Mesh 2.1 tests: kernel counters, race byte/trip accounting, token
# stream ordering, share-driven re-planning, routes, WS generation events.

swift run nmp-dashboard --ui   # then, from another device on the Wi-Fi:
curl -s http://<hostname>.local:3000/health
curl -s http://<hostname>.local:3000/api/devices/metrics
curl -s http://<hostname>.local:3000/api/clients
curl -s -X POST http://<hostname>.local:3000/api/inference \
    -d '{"prompt":"The future of AI is","max_tokens":16,"enable_comparison":true}'
curl -s -X POST http://<hostname>.local:3000/api/comparison/run \
    -d '{"prompt":"The future of AI is","max_tokens":8}'
curl -s -X POST http://<hostname>.local:3000/api/devices/2/allocate \
    -d '{"share":0.5}'          # reference mesh: watch the re-shard
```

```bash
swift test --filter "PeerResourceReportTests|WireTrafficTests|PeerResourceReportFlowTests|GPUSamplingTests"
# 10 Mesh 2.3 tests: resource-report codec (incl. nil sentinels), wire
# counters byte-exact across a lossless link, reports flowing on
# assignment + while serving, GPU sampler bounds.
```

Full suite: **316 tests, 0 failures** (291 Mesh 2.0 + 15 Mesh 2.1 + 10
Mesh 2.3), verified over 5 consecutive runs.
