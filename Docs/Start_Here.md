# Start Here — NeuraMesh Operator's Manual

Everything you can do with this repo, in the order you'd actually do it:
first-time setup, every way to start the mesh, what each screen/endpoint
is for, how to test every feature, and how to connect real peers.
Deep dives live in the other docs — this page tells you **which command
to run, when, and what you should see**.

- New machine? Do [§1 One-time setup](#1-one-time-setup) once.
- Just want it running? Jump to [§2 The 60-second start](#2-the-60-second-start).
- Wondering "how do I test X?" → [§5 Testing playbook](#5-testing-playbook).
- Adding a phone/second Mac? → [§6 Connecting mesh peers](#6-connecting-mesh-peers).
- Something's broken? → [§8 Troubleshooting](#8-troubleshooting).

---

## 1. One-time setup

**Requirements**: macOS 13+, Xcode 14.2+/Swift 5.8+. All devices that
will join a mesh must be on the **same Wi-Fi** (not a guest/hotspot
network — those block mDNS discovery).

```bash
cd ~/neuramesh/NeuraMeshProtocol

# 1. Build + prove everything works (316 tests, ~20 s):
swift build && swift test

# 2. (Only for real-LLM mode) build the llama.cpp shim:
brew install llama.cpp
scripts/setup_llama.sh          # → Vendor/llama/libnmpllama.dylib

# 3. (Only for real-LLM mode) get a model or three:
mkdir -p ~/models && cd ~/models
# the main 7B model (4.08 GB — verify the size after download!):
curl -LO https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
# small + fast, for quick experiments (~400 MB):
curl -LO https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
# draft model for speculation (same Llama-2 vocab — that part matters):
curl -LO https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

The web UI ships **prebuilt** in `Public/` — you never need npm unless
you edit `web/` (then: `cd web && npm install && npm run build`).

> A truncated GGUF download fails with "tensor … not within file bounds".
> Check the byte size against the model page before blaming the code.

---

## 2. The 60-second start

The one command that gives you everything at once:

```bash
swift run nmp-dashboard --ui
```

You get: a live 4-device reference mesh (real handshakes, encryption,
FEC — simulated compute), the browser UI on port **3000**, and a startup
banner with your Mac's real `http://<hostname>.local:3000`, its LAN IPs,
and a **QR code**. Scan it with your phone. Open the same URL on your
Mac. Both screens now show the same live mesh — that's the whole idea.

For the real LLM instead of the reference engine:

```bash
swift run nmp-dashboard --ui --engine llamaCpp \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf --auto-config
```

Add speculation (draft model must share the target's vocabulary):

```bash
swift run nmp-dashboard --ui --engine llamaCpp \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf --auto-config \
    --draft-model ~/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Stop everything with Ctrl-C. That's it.

### 2.1 Install it as an app on your phone (PWA — no Xcode, ever)

One-time, ~20 seconds:

1. Start the mesh on the Mac (any `--ui` command above).
2. On the phone, scan the QR from the banner (or type the URL).
3. iOS Safari: **Share ▸ Add to Home Screen**. Android Chrome:
   **⋮ ▸ Add to Home screen**.
4. A **NeuraMesh** icon appears on the home screen.

Every time after that: **tap the icon**. The app opens full-screen,
shows "Looking for your mesh…", finds the coordinator it was installed
from, flashes "✓ Connected to <your-mac>", and you're in — live
dashboard, streaming inference, device sliders, everything. If the Mac
isn't running yet, the app just keeps looking and connects the moment
`swift run nmp-dashboard --ui` comes up. No re-pairing, no Xcode, no
cables.

Two honest notes:

- **The PWA is a remote control, not a compute peer.** Browsers cannot
  open UDP sockets, so a phone contributing *compute* to the mesh still
  needs the native peer app (`Docs/CrossDevice_Setup_Guide.md`). For
  watching, running, benchmarking, and managing the mesh — the PWA is
  everything.
- **Why install from the mesh and not from a public website:** browsers
  block pages served over HTTPS (any public domain) from talking to
  `http://` devices on your LAN (mixed-content policy), and no browser
  API can do Bonjour discovery. A hosted PWA that connects to your Mac
  would need per-device TLS certificates + DNS plumbing (what Plex
  does). Installing from `http://<your-mac>.local:3000` sidesteps all of
  it: same origin, zero infrastructure, and "which mesh do I connect
  to?" answers itself.

**Security scope, once and clearly**: the dashboard/UI has no TLS and no
auth by design — it is a trusted-LAN testing tool. Never port-forward it
or run it on a network you don't trust.

---

## 3. The launch matrix — every mode, and when to use it

| You want to… | Run |
|---|---|
| See/demo everything in a browser (phone + Mac) | `swift run nmp-dashboard --ui` |
| Same, with a real LLM | `swift run nmp-dashboard --ui --engine llamaCpp --model … --auto-config` |
| The old terminal-era dashboard only (port 8080) | `swift run nmp-dashboard` |
| Real LLM, single device baseline (no mesh transport) | `swift run nmp-dashboard --engine llamaCpp --model … --placement local` |
| Real LLM behind the full protocol stack (in-process peer) | `swift run nmp-dashboard --engine llamaCpp --model …` (remote is the default) |
| Prove the two-process mesh works on one Mac, one command | `scripts/setup_mesh_test.sh --realistic` |
| A real two-process / two-device mesh (see §6) | `swift run nmp-peer …` then `swift run nmp-coordinator …` |
| Headless benchmark suite → CSVs | `swift run nmp-dashboard --benchmark` (→ `Results/*.csv`) |

Dashboard flags that stack onto any of the above:

| Flag | Effect |
|---|---|
| `--ui` | serve the React app on all interfaces (port defaults to 3000; pass a number to override) |
| `--auto-config` | probe devices → balance layer spans → persist profile (`~/.nmp/`) → pick wire format (llama: zero-trim, −98.9% payload) |
| `--probe-passes N` | probe passes for auto-config (default 3) |
| `--speculation` | serve every request speculatively (otherwise per-request opt-in) |
| `--draft-model path.gguf` | small same-vocab drafter; without it speculation uses prompt-lookup |
| `--placement local\|remote` | llama shard inline vs behind the full protocol stack |
| `--gpu-layers N` | llama.cpp GPU offload (default: all) |

`nmp-peer` / `nmp-coordinator` (the real two-process mesh) share the
same idea: `--engine reference|llamaCpp`, `--model`, plus coordinator-side
`--prompt "..." --tokens N --runs N --speculation --draft-model --zero-trim`
and peer-side `--slow msPerLayer` (handicap a peer to watch balancing).
`--help` on any binary prints the full list.

---

## 4. The web UI, tab by tab

Open the banner URL from as many devices as you like. Everything below
updates live on **all** of them (WebSocket pushes + 2–3 s polling).

- **Mesh** — device grid (per-peer latency/load/layers/liveness), model
  facts, web-client count, and the live event log. Leave it open on the
  phone while you work on the Mac; it's the mesh's heartbeat.
- **Run** — type a prompt, hit Run. Tokens appear **as they are
  generated, on every open browser** — submit from the Mac, watch the
  phone. Toggles: *Speculative decoding* (enabled when the mesh supports
  it), *Compare protocols* (attaches the comparison to the run).
- **Devices** — live host resources (RAM / storage / CPU / this
  process's footprint — real kernel counters) + per-peer cards with the
  **Mesh compute share** slider. Drag it and watch the mesh re-shard
  (see §5.4). Also lists the browsers currently connected.
- **Benchmark** — N sequential generations → average tok/s, latency ± σ,
  per-run table.
- **Compare** — top: the **real transport race** (measured NMP vs plain
  TCP, §5.3). Below: the what-if *model* (re-price a measured run at any
  RTT/loss — clearly labeled modeled).
- **Settings** — mesh facts + the packet-loss chaos slider (reference
  mesh: injects real loss into the datapath; watch FEC recover in the
  event log).

---

## 5. Testing playbook

### 5.0 The full suite

```bash
swift test          # 316 tests, 0 failures expected, ~20 s
```

Llama-backed tests skip automatically unless the shim + a model exist;
run them with a model: `NMP_LLAMA_MODEL=~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf swift test`.
Targeted suites, if you're iterating on one area:

```bash
swift test --filter NoiseIKTests            # crypto known-answer vectors
swift test --filter FECIntegrationTests     # loss recovery
swift test --filter AdaptiveSharding        # probe → balance → persist
swift test --filter SpeculativeDecodingTests
swift test --filter "Mesh21RouteTests|TransportRaceTests|TokenStreamingTests"
```

### 5.1 Test the API surface (works against any running dashboard)

```bash
# health + mesh facts + how many browsers are watching:
curl -s localhost:3000/health | python3 -m json.tool

# per-peer state:
curl -s localhost:3000/api/devices | python3 -m json.tool

# who's looking at the UI:
curl -s localhost:3000/api/clients | python3 -m json.tool

# run an inference (add "enable_speculation": true / "enable_comparison": true):
curl -s -X POST localhost:3000/api/inference \
     -d '{"prompt":"The future of AI is","max_tokens":16}' | python3 -m json.tool

# benchmark (runs clamped to 10):
curl -s -X POST localhost:3000/api/benchmark/run \
     -d '{"prompt":"hello mesh","max_tokens":8,"runs":3}' | python3 -m json.tool
```

### 5.2 Test real-time streaming

Open **Run** on two devices (or two browser windows). Submit from one.
The other shows `generation_started`, then each token with a live
cursor, then the final metrics — identical numbers on both. From the
command line, the same events flow on the WebSocket at `/ws` (types
`generation_started` / `generation_token` / `generation_complete`).

### 5.3 Test the measured protocol race

UI: **Compare → Run the race**. API:

```bash
curl -s -X POST localhost:3000/api/comparison/run \
     -d '{"prompt":"What is a mesh network?","max_tokens":8}' | python3 -m json.tool
```

What you get: the real generation's numbers, two **measured** legs (the
full NMP stack vs plain kernel TCP replaying the run's exact traffic),
and a spliced projection. Reading it honestly: raw TCP does no crypto
and no FEC, so on a clean loopback it can win the transfer — the
interesting number is the *gap* (with the llama zero-trim wire, full
encryption costs ~0.5 ms total). QUIC is not raced (it needs a TLS
certificate); it stays in the labeled model below.

### 5.4 Test that resource allocation is real (phone + Mac)

1. Start the **reference** mesh: `swift run nmp-dashboard --ui`.
2. Open **Devices** on your phone AND your Mac.
3. On the phone, drag a peer's *Mesh compute share* to 40%.
4. Watch — on both screens — that peer's assigned layer range shrink
   (e.g. 6 layers → 3), the other peers grow, and the event log print
   the re-shard. Subsequent per-pass timings shift accordingly.

Or from the shell, using a peer id from `/api/devices`:

```bash
curl -s -X POST localhost:3000/api/devices/2/allocate -d '{"share":0.4}'
# → {"status":"ok","summary":"testbed-1: L0-6, testbed-2: L7-9, …"}
```

The slider is only available on the reference mesh — a llama plan is one
full-range shard (llama.cpp can't split layers), and the API says so
instead of pretending. The host RAM/CPU/footprint bars are live real
kernel counters in both modes — watch the footprint jump when a model
loads.

### 5.5 Test fault tolerance & loss recovery (chaos)

Reference mesh (`swift run nmp-dashboard --ui`):

- **Settings → loss slider** to 10%: the event log fills with
  `fec_recovered` / `nack_sent` packet events; inference keeps working,
  output stays bit-exact.
- **Legacy page** (`localhost:3000/legacy`) has *inject peer drop*: a
  peer goes silent, failover re-shards survivors in ~ms, event log shows
  the new plan.

### 5.6 Test speculation properly

Speculation needs the **llamaCpp engine** (and the Phase 9 shim — rerun
`scripts/setup_llama.sh` if it says so). Start the llama dashboard, then:

```bash
# natural text → the TinyLlama drafter (54–72% acceptance measured):
swift run nmp-dashboard --ui --engine llamaCpp --model ~/models/llama-2-7b-chat.Q4_K_M.gguf \
    --auto-config --draft-model ~/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# repetitive text → even the built-in prompt-lookup drafter shines
# (measured 8 round trips for 32 tokens, 100% acceptance):
curl -s -X POST localhost:3000/api/inference \
     -d '{"prompt":"one two three one two three one two","max_tokens":24,"enable_speculation":true}'
```

Check the response's `speculation` block: round trips, acceptance rate,
tokens/trip. Output is token-for-token identical to plain greedy —
always. Rule of thumb (measured, see `Docs/Mesh2_WebUI_Guide.md` §4):
prompt-lookup for repetitive text, draft model for natural text **on a
physical two-device mesh**, plain zero-trim when coordinator and peer
share one GPU (in-process, the drafter steals GPU time from the target).

---

## 6. Connecting mesh peers

Three tiers, in order. Don't skip tier 1.

### 6.1 One command, one Mac (sanity checkpoint)

```bash
scripts/setup_mesh_test.sh --realistic
```

Builds, starts a background peer, lets the coordinator discover it over
real Bonjour + UDP, runs baseline vs mesh inference, checks bit-exact
output. ~30 s. If macOS asks about *finding devices on the local
network* — **Allow** (that's mDNS).

### 6.2 Two processes, real sockets (still one Mac)

```bash
# terminal 1 — the compute peer:
swift run nmp-peer                          # reference engine
# or with the real model (weights live HERE):
swift run nmp-peer --engine llamaCpp --model ~/models/llama-2-7b-chat.Q4_K_M.gguf

# terminal 2 — the coordinator (tokenizer only in llama mode):
swift run nmp-coordinator --engine llamaCpp --model ~/models/llama-2-7b-chat.Q4_K_M.gguf \
    --prompt "The capital of France is" --tokens 16
# optional levers: --speculation [--draft-model …] --zero-trim --runs N
```

What happens: Bonjour publish/browse (`_neuramesh._tcp`) → capability
exchange → deterministic coordinator election → Noise IK handshake over
real UDP → shard assignment → every token one real encrypted mesh round
trip. `--slow 5` on a reference peer handicaps it so you can watch the
sharder give it fewer layers.

### 6.3 Two real devices (Mac + iPhone/iPad/second Mac)

**First decide what the second device is for.** If you want it to
*watch and control* the mesh — install the PWA (§2.1, 20 seconds, no
Xcode) and you're done. Only continue here if you want the device to
*contribute compute* as a real NMP peer (UDP stack in a native app).

Follow **`Docs/CrossDevice_Setup_Guide.md`** top to bottom — it's a
click-by-click walkthrough (Xcode signing for the phone, the Local
Network permission, what every step should print, ~15 min first time).
The short version: same Wi-Fi, peer app on the phone, coordinator on the
Mac, discovery does the rest — no IPs typed anywhere.

Notes that save time:

- **Same Wi-Fi, non-isolated.** Guest networks and hotspots block mDNS.
  If discovery hangs, that's almost always why.
- **Reference engine** shards layers across any number of peers.
  **Llama** is one full-range shard on one peer by construction — the
  mesh buys you *placement* (weights on the peer device), not layer
  splitting; see `Docs/Phase8_Design.md` for why.
- The web UI (`--ui`) currently rides the dashboard binary — for
  two-process meshes use the coordinator's terminal output and
  `Results/` CSVs.

---

## 7. Where things live

| Path | What |
|---|---|
| `~/.nmp/neuramesh_sharding.json` | persisted per-device speed profiles (auto-config skips probing when complete; delete to force re-probe) |
| `Results/*.csv` | benchmark exports (`--benchmark`, coordinator runs) |
| `Vendor/llama/libnmpllama.dylib` | the llama.cpp shim (gitignored; rebuild with `scripts/setup_llama.sh`) |
| `~/models/*.gguf` | your models (any location works; these docs assume `~/models`) |
| `Public/` | the committed web UI build (`web/` is the source) |
| `Docs/Benchmarks.md` | measured numbers with dates + hardware |

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| Phone can't load `http://<mac>.local:3000` | Same Wi-Fi? Use the IP URL from the banner (some Androids don't resolve `.local`). macOS firewall prompt → Allow. |
| Discovery hangs ("no mesh assembled") | Local Network permission (System Settings → Privacy & Security → Local Network), or the network blocks mDNS (guest/hotspot). |
| `--engine llamaCpp` falls back to reference | Shim missing → `scripts/setup_llama.sh`; or model path wrong. The warning line says which. |
| "tensor … not within file bounds" on model load | Truncated GGUF download — re-download, verify byte size. |
| First 7B load stalls for ~a minute | Cold mmap of 4 GB — it's the OS, not the mesh. Subsequent loads are fast. |
| 429 from `/api/inference` | One generation at a time by design — retry when the current one finishes. |
| Allocation API returns an error on llama | Expected: one full-range shard, nothing to re-balance. Use the reference mesh to see re-sharding. |
| UI shows the old Phase 6 page | `Public/index.html` missing — `cd web && npm install && npm run build`, or pull the committed build. |
| Dashboard stdout invisible when scripted | It's block-buffered when redirected — wrap with `script -q out.log <command>` (pty). |
| Want the URL to literally say `neuramesh.local` | Rename the Mac's Local Hostname (System Settings → General → Sharing). mDNS can't fake it any other way. |

---

## 9. Which doc to read when

| Read | When you want |
|---|---|
| **this page** | to operate the thing |
| `CrossDevice_Setup_Guide.md` | click-by-click Mac + iPhone mesh |
| `Mesh2_WebUI_Guide.md` | web UI architecture, streaming/race/devices details, endpoint reference, measured speculation table |
| `Phase9_Design.md` | adaptive sharding, wire compression, speculation design + measurements |
| `Phase8_Design.md` | how llama.cpp is bound, why llama = one shard, distributed-inference honesty |
| `Phase1–6_Design.md` | protocol internals (crypto, reliability, FEC, discovery, sharding, failover) |
| `NMP_Specification.md` | the wire protocol, source of truth |
| `Benchmarks.md` | every measured number in one place |



## 10. follow
 I've cut the setup to the Apple-mandated minimum: the Xcode project is now checked 
  into the repo (2d86c88), pre-wired and build-verified for iOS. The guide's old 10-step
  project assembly is gone — sources, the NMP package link, and the two Info.plist keys
  everyone forgets are all already in it.

  Your setup, one time (~5 min)

  open NeuraMeshPeer/NeuraMeshPeer.xcodeproj

  1. Click the blue project icon → Signing & Capabilities → set Team to your Apple ID
  (free one works; add it under Xcode → Settings → Accounts if empty).
  2. Plug in your iPhone via USB, unlock it, tap Trust, pick the iPhone in Xcode's
  device menu, press ⌘R.
  3. Two one-time phone prompts: Settings → General → VPN & Device Management → Trust
  your Apple ID (then ⌘R again), and Allow the local-network prompt when the app opens.

  Every time after (~1 min, no cable, no Xcode)

  Open the NeuraMeshPeer app on the phone (same Wi-Fi, screen on), then on the Mac:

  swift run nmp-coordinator --peers 1 --wait 60

