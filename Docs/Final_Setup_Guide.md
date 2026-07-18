# NeuraMesh — Final Setup Guide (from scratch → running)

This is the one guide to get NeuraMesh running from a clean Mac and drive
**everything from the web UI** — inference, chat, testing, benchmarking, the
NMP-vs-TCP/QUIC race, the real sharded mesh, and packet-loss chaos. No terminal
commands after the first launch.

> **Honesty note (read once).** Exactly two things can never become a browser
> button, and it's physics, not laziness: a web page cannot **compile native
> code** on your Mac, and it cannot **sign & install an iOS app** on your
> iPhone. So the *one-time bootstrap* and the *iPhone peer install* use a
> terminal / Xcode. Everything you do **after** launch is in the browser.

---

## 0. Prerequisites (one time)

| Need | Check | Get it |
|------|-------|--------|
| macOS on Apple Silicon | — | — |
| Xcode command-line tools | `xcode-select -p` | `xcode-select --install` |
| Homebrew | `brew --version` | https://brew.sh |

That's it. The launcher installs the rest (`ggml`, `llama.cpp`) itself.

---

## 1. Start it — one command

```bash
cd ~/neuramesh/NeuraMeshProtocol
scripts/start.sh
```

That script is idempotent (safe to re-run) and does the whole bootstrap for you:

1. verifies Xcode tools + Homebrew
2. `brew install ggml llama.cpp` (the shard math + the tokenizer)
3. builds the two native shims **once** → `Vendor/llama/*.dylib`
4. makes sure a qwen model is in `~/models` (downloads the tiny **0.5B, ~490 MB**
   if you have none, so you can start immediately)
5. `swift build`
6. launches the **real sharded mesh + web UI** and opens your browser at
   **http://localhost:3000**

First run takes a few minutes (brew + shim compile + model download). Re-runs
start in seconds because every step is skipped once done.

Leave that terminal open — `Ctrl-C` there stops the mesh. You won't type another
command.

**Variants** (rarely needed):

```bash
scripts/start.sh --model ~/models/qwen2.5-14b-instruct-q4_k_m.gguf   # force a model
scripts/start.sh --engine reference   # simulated mesh — no shims/model, instant
```

---

## 2. Everything lives in the browser now

Open **http://localhost:3000**. The top nav has every control:

| Tab | What it does | Replaces which old terminal command |
|-----|--------------|-------------------------------------|
| **Mesh** | Live topology, engine, model, shard count, per-token stream | `nmp-dashboard` stdout |
| **Run** | Type a prompt → real generation through the sharded mesh, timed | `swift run … --prompt` |
| **Chat** | Multi-turn chat over the mesh | — |
| **Models** | **Pick any model in `~/models`.** Incompatible (non-qwen) and too-big models are flagged with the reason; the best-fitting one is marked *recommended*. Click **Use this model** → the mesh restarts onto it and the page reconnects. | `--model` relaunch |
| **Devices** | Every shard: its layer range and loaded MB (the coordinator's is measured; a remote peer's is computed from its range). The split **auto-balances by each device's measured speed + capacity and re-shards automatically when a device joins or leaves** — no manual step. | manual re-shard |
| **Benchmark** | Run the latency/throughput suite in-browser over the real sharded pipeline, see p50/p95 | `nmp-dashboard --benchmark` |
| **Compare** | **The measured transport race: NMP vs TCP vs TLS 1.3 vs QUIC**, run on demand | `nmp-dashboard --benchmark-race` |
| **Pressure / Settings** | Engine/model/wire-format readout. *Note:* the in-browser **packet-loss slider** injects loss into the *simulated* mesh (`--engine reference`); the sharded coordinator is a **real UDP mesh**, so loss testing there is `sudo scripts/loss_lab.sh` (real NIC loss). | `loss_lab.sh` |

**Sharded inference is already what you're running.** With the default
`--engine llamaShard` the coordinator (this Mac) tokenizes AND holds a shard,
and **the moment a second device joins, the split becomes genuinely
cross-device — no single device holds the whole model** (the Devices tab proves
it: each row shows only its layer range + the MB it holds). With just the Mac
it is one shard (the Mac holds all layers, and the Devices row says so — that is
honest, not a bug); add a peer and it re-shards live. Typing in **Run** or
**Chat** executes the real shard pipeline. Nothing else to launch.

> **Testing checklist, all in the UI:** Run a prompt (Run) → confirm real,
> deterministic output → open **Devices** and read the coordinator's layer range
> + loaded MB → **join a second device** (another Mac's `nmp-peer`, or your
> iPhone — §3) and watch the Mesh log announce "re-sharded" and the Devices tab
> redistribute the layer ranges live, no single device holding the whole model →
> re-run and confirm output is still correct → open **Compare** and race the
> transports → open **Settings** and push packet loss to 5% and confirm
> generation still completes (FEC + NACK recovering).

---

## 3. Add your iPhone (real cross-device sharding)

A browser can't be a UDP compute peer — phones have no UDP-from-JS — so an
iPhone contributes compute through the **native NeuraMeshPeer app**, not Safari.
This is the one part that needs Xcode (code-signing is device-bound).

1. **Build the iOS shard framework** (one time, on the Mac):
   ```bash
   scripts/setup_shard_ios.sh          # → Vendor/ios/nmpshard.xcframework
   open NeuraMeshPeer/NeuraMeshPeer.xcodeproj
   ```
   The second line opens the app project in Xcode.
2. **Embed the framework**: in Xcode, select the `NeuraMeshPeer` target →
   *Frameworks, Libraries, and Embedded Content* → **+** → add
   `Vendor/ios/nmpshard.xcframework` → set it to **Embed & Sign**. Pick your
   signing Team (Signing & Capabilities tab).
3. **Build & Run** (⌘R) to your iPhone. **No model download needed** — the phone
   **streams only its assigned layers** from the Mac (the weight vault) and caches
   them. It stores ≈ its layers, *not* the whole model (disk ≈ RAM), so even a 14B
   split works on a phone with little free space. The Models tab shows the shard
   cache (and a Clear button); the full-model download is there only as an
   optional fallback for standalone use.
4. On the Mac, the mesh discovers the phone over Bonjour and **re-shards to
   include it automatically** — watch the Devices tab: the iPhone appears as a
   shard and the layer ranges redistribute. Degrade (phone leaves) and upgrade
   (phone joins) are handled live; each membership change re-streams only the
   changed ranges.

> Until the framework is embedded, the app still runs and joins — it just uses
> the weightless reference engine and says so on the Models tab. Embed the
> framework and it streams real weights and computes.

> The Mac remains the coordinator (tokenizer + a shard) and serves the web UI;
> the phone is a compute peer. You still drive everything from the Mac's
> browser — the phone just adds capacity.

---

## 4. The 14B model

```bash
scripts/setup_qwen14b.sh          # storage-aware: picks q4_k_m / q3_k_m / q2_k
```

It checks free disk and downloads the largest quant that fits. Then either:

- `scripts/start.sh` — it will **auto-select** 14B if this host can hold a shard
  of it (and fall back to a smaller model if not — the adaptive selector picks
  the best model the current mesh can actually run); or
- `scripts/start.sh --model ~/models/qwen2.5-14b-instruct-q4_k_m.gguf` to force it.

The mesh re-decides the optimal model on every device join/leave. With just the
Mac it may pick a smaller model; add the iPhone and it can upgrade. The **Mesh**
tab shows which model was chosen and why.

---

## 5. What runs where (the honest map)

| Task | Where | Terminal? |
|------|-------|-----------|
| Install toolchain + build shims | `scripts/start.sh` | once (physics: native compile) |
| Launch the mesh + UI | `scripts/start.sh` | one command |
| Inference / chat / testing / benchmark / transport race / packet-loss | **browser** | **no** |
| Sharded inference + live re-sharding | **browser** (Run/Chat/Devices) | **no** |
| Pick / switch which model the mesh runs | **browser** (Models tab) | **no** |
| Download a model to the Mac | **browser** picker lists them; big pull via `setup_qwen14b.sh` (or start.sh auto-grabs a small one) | mostly no |
| Get weights onto the iPhone | **automatic** — the phone streams only its assigned layers from the Mac (weight vault); stores ≈ its layers, not the whole model | **no** |
| Embed the iOS compute framework + first Run | Xcode: `open …xcodeproj`, Embed & Sign, ⌘R | Xcode (physics: code-signing) |

---

## 6. Troubleshooting

- **Browser didn't open** → go to http://localhost:3000 manually. On another LAN
  device use `http://<your-mac-name>.local:3000`.
- **`no model in ~/models fits this host`** → free up disk or
  `scripts/start.sh --model <a smaller .gguf>`. The 0.5B always fits.
- **`start.sh` says a shim step failed** → run it directly to see the full log:
  `scripts/setup_shard.sh` (needs `brew install ggml`).
- **iPhone doesn't appear** → both devices on the same Wi-Fi/LAN, app in
  foreground, a `.gguf` present in its Documents, and the framework Embedded &
  Signed (not just linked).
- **Want raw numbers** → the **Compare** and **Benchmark** tabs export the same
  measured CSVs the headless `--benchmark*` modes write to `Results/`.

---

*Related: `Docs/Start_Here.md` (every mode + flag), `Docs/CrossDevice_Setup_Guide.md`
(deeper device wiring), `Docs/Protocol_Comparison.md` (the transport-race
methodology), `Docs/Future_Plans.md` (roadmap).*

---

## Distributed Memory Mesh (kill-a-peer demo)

A memory layer where each peer runs its **own local Supermemory instance** and
memories are sealed (LZFSE + AES-256-GCM) then erasure-coded into K-of-N shards
scattered across peers. No single peer holds a complete readable copy —
reconstruction needs a quorum. The demo below writes a memory, kills a real peer
process, and shows recall still works from the survivors' shards. Design +
honesty details: `Docs/Memory_Mesh.md`.

This runbook uses **K=2 of N=3** on one Mac — three peers, each a simulated
device with its own Supermemory instance on distinct ports (6767/6768/6769). On
**real separate devices** each Supermemory is just `localhost:6767` and each
peer's config points at its own localhost; the distinct ports here only simulate
three devices on one machine. The control HTTP API is **loopback / trusted-LAN
only** — no TLS, no auth (same stance as the dashboard); never port-forward it.

**Prereqs (one time):** install the local Supermemory binary —
`curl -fsSL https://supermemory.ai/install | bash` (installs
`~/.supermemory/bin/supermemory-server`). Then `swift build`.

**Fast path:** after step 1 boots the instances, `scripts/run_memory_demo.sh
--launch` runs steps 2–7 in one shot (launch peers, write, prove one shard per
peer, recall, `kill -9` a peer, recall again from a survivor); add `--kill-two`
for the below-quorum failure. The manual steps below are the same thing, spelled
out.

1. **Stand up the three instances + configs:**
   ```bash
   scripts/setup_memory_mesh.sh start
   ```
   Boots sm1/sm2/sm3 (data dirs `~/.neuramesh-memdemo/sm{1,2,3}`) and writes
   `~/.neuramesh-memdemo/peer{1,2,3}/config.json` (NMP UDP 9411/9412/9413,
   control HTTP 9401/9402/9403). **First boot warms the in-process embedding
   model — can take a minute or two; the script waits up to 180 s per instance
   with progress dots.** If a foreign `supermemory-server` already holds port
   6767 it refuses to reuse it: `lsof -nP -iTCP:6767 -sTCP:LISTEN` then
   `kill <pid>`.

2. **Start the three peers** (three terminals, or backgrounded):
   ```bash
   swift run nmp-memory-peer --config ~/.neuramesh-memdemo/peer1/config.json
   swift run nmp-memory-peer --config ~/.neuramesh-memdemo/peer2/config.json
   swift run nmp-memory-peer --config ~/.neuramesh-memdemo/peer3/config.json
   ```
   Each prints its NMP UDP port + loopback control API. They form a full mesh
   automatically (lower peerID dials higher; static keys auto-generate into the
   shared keyDir on first start) — start all three and the redial timer resolves
   links within a few seconds.

3. **Write a memory:**
   ```bash
   curl -s -X POST http://127.0.0.1:9401/remember \
        -H 'Content-Type: application/json' \
        -d '{"title":"...","content":"..."}'
   ```
   Returns a receipt with `memoryID`, `shardAcks`, and the "persisted NOWHERE"
   note (each peer holds one opaque shard + the bounded plaintext index entry).

4. **Confirm distribution:**
   ```bash
   curl -s http://127.0.0.1:9401/status
   curl -s http://127.0.0.1:9402/status
   curl -s http://127.0.0.1:9403/status
   ```
   Each shows `shardsHeld:1`.

5. **Baseline recall (all alive):**
   ```bash
   curl -s "http://127.0.0.1:9401/recall?q=<terms>"
   ```
   Returns the full original content + `shardsUsed` (only K=2 shards fetched —
   local + one peer over NMP; the third is "not needed").

6. **Kill a peer for real:**
   ```bash
   kill -9 $(pgrep -f "nmp-memory-peer --config.*peer3")
   ```

7. **Recall again from a survivor (peer1)** — the money shot:
   ```bash
   curl -s "http://127.0.0.1:9401/recall?q=<terms>"
   ```
   **Still returns the full content**, reconstructed from the remaining 2 shards
   (K=2). Peer 3 shows under `unreachablePeers` once its link retires (~15 s), or
   `peersNotNeeded` if quorum was met before the link timed out. Real process
   killed mid-demo, memory survives.

8. **(Optional) Kill a second peer** → drop below quorum:
   ```bash
   kill -9 $(pgrep -f "nmp-memory-peer --config.*peer2")
   curl -s "http://127.0.0.1:9401/recall?q=<terms>"
   ```
   Now recall returns **HTTP 503** with
   `quorum_unavailable: have 1 of 2 required shards` — explicit failure, never
   wrong output.

9. **Teardown:**
   ```bash
   scripts/setup_memory_mesh.sh stop     # stops the Supermemory instances it started
   ```
   Ctrl-C the peers.

> **Warmup note.** Supermemory ingestion is asynchronous. On a warm instance a
> doc is searchable **~1.2 s** after write (measured); on a just-booted **cold**
> instance the embedding model warms slowly and docs can sit at `queued` for
> **40 s+** (measured). Warm the instances before demoing, or rely on the read
> path's in-RAM keyword fallback for just-written memories (it bridges the
> freshness gap until Supermemory has indexed — see `Docs/Memory_Mesh.md`).
