# Cross-Device Setup Guide — Run the Mesh on Your Mac + iPhone

> **Most people don't need this page.** If you want your phone to
> *watch and control* the mesh — live dashboard, streaming inference,
> device sliders — install the PWA instead: start the mesh with `--ui`,
> scan the QR, **Share ▸ Add to Home Screen**. 20 seconds, no Xcode; see
> `Docs/Start_Here.md` §2.1. This guide is only for making the phone a
> **compute peer** (running the native NMP UDP stack), which a browser
> app cannot do.

Follow this top to bottom. Every step says exactly what to click or type,
what you should see, and what to do if you don't see it. Total time:
**~15 minutes** the first run, ~1 minute after that.

There are three levels. Each one is a complete, working checkpoint —
finish Level 1 before touching a phone.

| Level | Hardware | Time |
|---|---|---|
| 1. Loopback mesh | Mac only | 2 min |
| 2. Mac + iPhone mesh | Mac + iPhone + USB cable (first install only) | ~15 min |
| 3. Third device | + iPad or second iPhone/Mac | ~5 min |

**Requirements**: macOS 13+, Xcode 15+ (for Level 2), iPhone/iPad on iOS 16+,
and — critical — **all devices on the same Wi-Fi network**. Hotspots,
"guest" networks, and many office networks block the mDNS discovery
traffic this depends on; a home Wi-Fi network works.

---

## Level 1 — Loopback mesh (Mac only, no phone)

Prove the whole pipeline works on one machine first.

**Step 1.** Open Terminal, go to the repo:

```bash
cd NeuraMeshProtocol
```

**Step 2.** Run the one-command demo:

```bash
scripts/setup_mesh_test.sh --realistic
```

**You should see** (~30 s): a build, then the coordinator discovering the
background peer, a shard plan, and finally:

```
=== Results ===
  baseline (1 device) best: ...
  mesh (2 shards)  best: ...  (1.0x× baseline)
  numerics: mesh output BIT-EXACT vs single device ✓
```

**If it fails** with "timed out: no mesh assembled": macOS may have shown
a *"nmp-coordinator would like to find and connect to devices on your
local network"* prompt — click **Allow** and re-run. (System Settings →
Privacy & Security → Local Network if you missed it.)

✅ Checkpoint: the protocol, discovery, sharding, and inference all work.
Everything past here is just putting a peer on another device.

---

## Level 2 — Mac + iPhone mesh

### Part A: Put the peer app on your iPhone (~10 min, once)

**Step 1 — Create the project.**
1. Open Xcode → **File → New → Project…**
2. Pick **iOS → App** → Next.
3. Product Name: `NeuraMeshPeer`. Interface: **SwiftUI**. Language:
   **Swift**. Uncheck tests. → Next.
4. Save it **anywhere except inside the NeuraMeshProtocol folder**
   (e.g. `~/Projects/NeuraMeshPeer`). Nesting it inside the package
   confuses SwiftPM.

**Step 2 — Add the NMP package.**
1. In Xcode: **File → Add Package Dependencies…**
2. Click **Add Local…** (bottom-left).
3. Select your `NeuraMeshProtocol` folder → **Add Package**.
4. In the "Choose Package Products" sheet, check product **NMP** for
   target NeuraMeshPeer → **Add Package**.

**Step 3 — Drop in the app sources.**
1. In Finder, open `NeuraMeshProtocol/NeuraMeshPeer/Sources/` — it has
   3 files: `NeuraMeshPeerApp.swift`, `PeerViewModel.swift`,
   `PeerStatusView.swift`.
2. Drag all 3 into the Xcode project navigator, onto the `NeuraMeshPeer`
   folder (the one containing `ContentView.swift`).
3. In the dialog: check **Copy items if needed** and the NeuraMeshPeer
   target → Finish.
4. **Delete** Xcode's generated `NeuraMeshPeerApp.swift` duplicate (the
   original from the template — it clashes with ours; keep the one you
   just added) and delete `ContentView.swift`. "Move to Trash" for both.

**Step 4 — The two Info.plist keys (the step everyone forgets).**
Without these, iOS silently blocks Bonjour and nothing ever connects.
1. Click the project (blue icon) → target **NeuraMeshPeer** → **Info** tab.
2. Hover any row, click **+**, and add:
   - Key `NSLocalNetworkUsageDescription` (shown as *"Privacy — Local
     Network Usage Description"*), type String, value:
     `NeuraMesh discovers and connects to nearby devices to run distributed AI inference on your local network.`
   - Key `NSBonjourServices` (*"Bonjour services"*), type Array, with one
     item, String, value: `_neuramesh._tcp`
   (Reference copy: `NeuraMeshPeer/Info-additions.plist`.)

**Step 5 — Signing (free Apple ID works).**
1. Target **NeuraMeshPeer** → **Signing & Capabilities** tab.
2. Check **Automatically manage signing**; Team: pick your Apple ID
   (add one via Xcode → Settings → Accounts if the menu is empty).
3. If "Failed to register bundle identifier": make the Bundle Identifier
   unique, e.g. `com.YOURNAME.NeuraMeshPeer`.

**Step 6 — Run it on the phone.**
1. Plug the iPhone in via USB. Unlock it. Tap **Trust** on the phone.
2. In Xcode's device menu (top bar) pick your iPhone (not a simulator).
3. Press **⌘R**.
4. First run only, the phone blocks the app: on the iPhone go to
   **Settings → General → VPN & Device Management** → tap your Apple ID →
   **Trust**. Run again (⌘R).
5. When the app opens, iOS asks *"NeuraMeshPeer would like to find and
   connect to devices on your local network"* → **Allow**. (This prompt
   is one-shot; if you denied it: iPhone Settings → Privacy & Security →
   Local Network → enable NeuraMeshPeer.)

**You should see** on the phone: a "NeuraMesh Peer" screen with a Peer ID,
a UDP port number, and the log line `advertising NeuraMesh-… on UDP
port …`. The phone is now a compute node waiting for a coordinator.

### Part B: Run the mesh (~1 min, every time)

**Step 1.** iPhone: app open, screen on, same Wi-Fi as the Mac.

**Step 2.** Mac terminal:

```bash
cd NeuraMeshProtocol
swift run nmp-coordinator --peers 1 --wait 60
```

(The app's built-in model is 32 layers × 4096 hidden, the CLI default —
the sizes must match, and they do out of the box.)

**You should see**, in order:
1. Mac: `dialing iPhone… (…, high, port …)` within a few seconds.
2. Mac: `peer … established (handshake complete)` — Noise IK over Wi-Fi.
3. Mac: the shard plan — e.g. layers 0..<16 local, 16..<32 → your iPhone.
4. iPhone: shard row updates to `shard 1 of 2: layers 16–31 of 32`.
5. Both: per-run latencies; iPhone's "Requests served" counts up.
6. Mac: `numerics: mesh output BIT-EXACT vs single device ✓`.

That last line is the whole point: **your iPhone computed half the model's
layers over your Wi-Fi, and the result is bit-for-bit identical to the Mac
computing alone.**

To emulate 7B-scale per-layer compute (makes the mesh/baseline ratio
realistic instead of network-dominated), it's the `--slow` flag on the
CLI; the app itself always computes at full speed.

### Troubleshooting (Level 2)

| Symptom | Fix |
|---|---|
| Mac never prints `dialing …` | Same Wi-Fi? Local Network allowed on BOTH devices? Personal hotspot doesn't count as shared Wi-Fi. Test discovery independently: `dns-sd -B _neuramesh._tcp` on the Mac should list the phone within ~2 s. |
| `dns-sd` shows the peer but no `dialing` | The TXT record is missing port/key — you're running an old app build; rebuild the app (⌘R). |
| `dialing …` but never `established` | A firewall is eating UDP: System Settings → Network → Firewall — allow incoming for the tool, or turn it off for the test. |
| `shard assignment failed: assignmentRejected(...rejectedModelMismatch)` | App and CLI model tags differ — you passed `--tag` or `--gguf` to the CLI but the app uses `nmp-reference-model`. Drop the flag or change `PeerViewModel.swift` to match. |
| `assignmentRejected(...rejectedBadRange)` | Layer/hidden sizes differ — you passed `--layers/--hidden` to the CLI. The app is 32×4096; use the defaults. |
| Runs stall mid-benchmark | iPhone locked or app backgrounded — iOS suspends it. Keep the app foregrounded and the screen on (the app disables auto-lock while visible). |
| Everything worked yesterday, dead today | Network changed (different subnet, AP isolation enabled, VPN on the Mac). Turn off VPNs, confirm both devices have same-subnet IPs. |

---

## Level 3 — Add a third device

**Another iPhone/iPad**: repeat Level 2 Part A on that device (same Xcode
project, just select the other device and ⌘R; redo the Trust +
Local-Network prompts on it).

**Another Mac**: no Xcode needed —

```bash
cd NeuraMeshProtocol
swift run nmp-peer
```

Then on the coordinator Mac:

```bash
swift run nmp-coordinator --peers 2 --wait 60
```

**You should see** a 3-shard plan (layer spans proportional to device
speed — the sharder weighs measured seconds/layer once it has them, class
weights before that), all three devices serving, and the same
`BIT-EXACT ✓` verdict.

---

## FAQ

**Where's the real 7B model in all this?** Phase 5 runs the mesh with a
deterministic reference engine sized from real GGUF metadata (`--gguf
path/to/model.gguf` sizes layers/hidden from the file). Real quantized
execution binds llama.cpp behind the `NMPShardComputeEngine` protocol —
one class to write, nothing in the protocol/transport/app changes. See
`Docs/Phase5_Design.md` § "The compute seam".

**Do I need to configure IP addresses anywhere?** No. Peers advertise
port + session key via Bonjour TXT records; the coordinator dials browse
results directly. That's the Phase 4 + 5 zero-config design working.

**Is the traffic encrypted?** Yes — every packet after the Noise IK
handshake is AES-256-GCM with replay protection, same as Phases 1–3.
The TXT-advertised key is trust-on-first-use for the benchmark mesh;
production pins keys (`PeerConnectionConfig.authorizedStaticKeys`).
