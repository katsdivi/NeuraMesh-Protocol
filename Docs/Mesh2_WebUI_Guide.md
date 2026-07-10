# Mesh 2.0 — Multi-Device Web UI, Protocol Comparison & Setup Guide

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

## 5. Web endpoint verification (from scratch)

```bash
swift test --filter "WebUIRouteTests|ProtocolComparisonModelTests|LANIdentityTests"
# 19 tests: routes, CORS, SPA fallback + traversal guard, benchmark
# aggregation (σ pinned), comparison model math (loss widens the gap),
# banner + QR.

swift run nmp-dashboard --ui   # then, from another device on the Wi-Fi:
curl -s http://<hostname>.local:3000/health
curl -s -X POST http://<hostname>.local:3000/api/inference \
    -d '{"prompt":"The future of AI is","max_tokens":16,"enable_comparison":true}'
```

Full suite: **291 tests, 0 failures** (272 Phase 9 + 19 new).
