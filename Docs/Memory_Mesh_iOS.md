# Memory Mesh on iOS — Making the iPhone a Memory Peer

Put the iPhone into the [distributed memory mesh](Memory_Mesh.md) as a full
peer: it holds erasure-coded **shards** of other peers' memories and
reconstructs on a K-of-N quorum, over the same encrypted NMP transport the
compute mesh uses.

> **Honesty banner.** The glue code (`NeuraMeshPeer/Sources/MemoryPeerController.swift`)
> is written to compile against the `NMP` module as an iOS target uses it, and
> the shard/seal/wire/transport code it drives is **verified on Mac** this
> session (17 codec tests pass; the kill-a-peer demo runs). It has **NOT** been
> built or run on a physical iPhone in this session. Adding the file to the app
> target, dropping in the screen below, and building to a signed device are
> **manual Xcode steps** — the parts that need a device are called out as
> *not-yet-run-on-device*.

This is a sibling to the compute-peer flow in
[CrossDevice_Setup_Guide.md](CrossDevice_Setup_Guide.md); read that first for
signing, USB install, and the Local Network prompt — all identical here.

---

## Why the phone can't run supermemory-server (and what replaces it)

A **Mac** memory peer backs its store with a localhost `supermemory-server`:
a Node/Mach-O HTTP server it talks to over `http://localhost:<port>`
(`NMPSupermemoryClient`, hard localhost-only guard). The phone can't do that:

- **No Node runtime** on iOS, and no way to ship one.
- **iOS forbids spawning a separate executable** — an app is one process; it
  can't fork a server binary.
- The Supermemory integration is an **HTTP-server architecture** — it assumes
  something is listening on a localhost port. Nothing can listen there on iOS.

So on the phone we swap the backend, not the mesh. `NMPMemoryMeshNode` is
backend-agnostic: its designated init `init(config:store:)` takes any
`NMPMemoryStore`. iOS injects **`NMPLocalMemoryStore`** — a fully on-device
store with **no server and no network**:

- **file blob store** for opaque shards + index entries (app sandbox), and
- **on-device embeddings** (Apple's **NaturalLanguage** framework) for the
  semantic search over the plaintext index entries.

Everything above the store — seal (LZFSE + AES-256-GCM), K-of-N XOR sharding,
distribution over the encrypted NMP transport, quorum reconstruction — is the
**same code** that runs on Mac. The Mac path (`withSupermemory(config:)`) is
simply never called on iOS.

---

## The glue: `MemoryPeerController`

`NeuraMeshPeer/Sources/MemoryPeerController.swift` is a `@MainActor`
`ObservableObject` that mirrors `PeerViewModel`'s threading contract (node on
its own serial queue; every callback hops to the main actor before touching
`@Published` state). It:

- builds `NMPLocalMemoryStore` in **Application Support/MemoryMesh/store**;
- builds `NMPMemoryPeerConfig` **in code** from its inputs (`peerID`,
  `deviceName`, `udpPort`, the `peers` roster, `keyDir`, scheme **k=2, n=3**),
  with the `supermemory` block **omitted** — that omission is what marks it an
  on-device peer;
- constructs `try NMPMemoryMeshNode(config:store:)`, wires `onStatus` /
  `onDiagnostic` into a `@Published log`, and exposes `start()`,
  `remember(content:title:)`, `recall(query:)`, `refreshStatus()`;
- exposes `ownPublicKeyURL` and `peerKeysDirectory` for the manual key exchange
  (below).

**Config-construction note.** `NMPMemoryPeerConfig` and `RemotePeer` now expose
`public init`s, so the config can be built as a plain struct literal in code
(supermemory fields default to nil for a native-store peer). The shipped
controller instead builds the same JSON the Mac CLI's loader validates, writes
it to app storage, and calls `NMPMemoryPeerConfig.load(path:)` — same public
surface, same validation, and it round-trips through the identical parser the
Macs use. Either construction path is fine; the loader path just reuses the
validation for free.

---

## Manual Xcode steps

The whole `NMP` package is **already a dependency of the app** (the project has
a local Swift-package reference to the repo root, product `NMP` linked into the
`NeuraMeshPeer` target — same reference the compute peer uses). You do **not**
add a package; you add one file and a screen.

### 1. Add the controller to the target

`open NeuraMeshPeer/NeuraMeshPeer.xcodeproj` → drag
`NeuraMeshPeer/Sources/MemoryPeerController.swift` into the project navigator
(or, if already visible, select it) → File Inspector → **Target Membership** →
check **NeuraMeshPeer**. `import NMP` resolves against the existing dependency.

### 2. Add a minimal screen

Add a new SwiftUI file (e.g. `MemoryView.swift`), target **NeuraMeshPeer**, and
add a `Memory` tab to the `TabView` in `NeuraMeshPeerApp.swift`
(`.tabItem { Label("Memory", systemImage: "brain") }`). A ~30-line screen:

```swift
import SwiftUI

struct MemoryView: View {
    @StateObject private var mem = MemoryPeerController()
    @State private var note = ""
    @State private var query = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Peer") {
                    Text(mem.started ? "running (peer \(mem.peerID))" : "stopped")
                    if !mem.started { Button("Start") { mem.start() } }
                    Button("Refresh status") { mem.refreshStatus() }
                }
                Section("Remember") {
                    TextField("something to remember", text: $note)
                    Button("Remember") { mem.remember(content: note, title: "")
                                         note = "" }
                        .disabled(!mem.started || note.isEmpty)
                }
                Section("Recall") {
                    TextField("what did I say about…", text: $query)
                    Button("Recall") { mem.recall(query: query) }
                        .disabled(!mem.started || query.isEmpty)
                    if !mem.lastRecallJSON.isEmpty {
                        Text(mem.lastRecallJSON).font(.caption.monospaced())
                    }
                }
                Section("Log") {
                    ForEach(Array(mem.log.enumerated()), id: \.offset) { _, l in
                        Text(l).font(.caption2.monospaced()).foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Memory")
        }
    }
}
```

Before `start()`, set `mem.peerID`, `mem.udpPort`, and `mem.peers` (add a small
settings form, or hard-code them for a demo) — see the roster step below.

### 3. Info.plist

The memory node opens **UDP connections on the LAN**, so it needs:

- **`NSLocalNetworkUsageDescription`** — already present in
  `NeuraMeshPeer/Info.plist` (shared with the compute peer). The one-shot iOS
  "find and connect to devices on your local network" prompt covers this too.

It does **NOT** use Bonjour (static roster, not mDNS), so **`NSBonjourServices`
is not required for THIS feature** — unlike the compute app, which needs it to
be discovered. If you ship *only* the memory peer you could drop the Bonjour
key; since the app already ships both, leave the existing plist as-is.

---

## Key exchange reality (static-key pinned, static roster)

There is **no discovery** for this feature. The node uses a **static roster**
(the `peers` list of `host:port`) and pins peers by **static public key**. That
means two manual chores per device pair:

**(1) Put the Macs' LAN IPs in the phone's roster.** On each Mac:
`ipconfig getifaddr en0` for its LAN IP; its `peerID` and `udpPort` are in that
Mac's `peer<i>/config.json` (from `scripts/setup_memory_mesh.sh`). Set
`mem.peers = [PeerRef(peerID:…, host:"192.168.x.y", udpPort:…), …]` before
`start()`. Also add the **phone** to each Mac's roster (`peers[]` entry with the
phone's `peerID`, LAN IP, and `udpPort` — default `0xA1` / `7810` in the
controller) and restart that Mac peer.

**(2) Copy each device's `peer<id>.pub` to the others' keyDir.** Noise IK is
mutual — each side must hold the other's public key:

- **Phone's keyDir** is `Application Support/MemoryMesh/keys` inside the app
  sandbox. After the first `start()`, the phone's own key is at
  `MemoryPeerController.ownPublicKeyURL`
  (`…/keys/peer<peerID>.pub`). Get it **off** the phone via the **Files app**
  (if you add `UIFileSharingEnabled` / `LSSupportsOpeningDocumentsInPlace`) or
  the **Xcode device container** (Xcode → Devices & Simulators → the app →
  *Download Container* → show package contents → `AppData/Library/Application
  Support/MemoryMesh/keys/`). Copy that `.pub` into **each Mac's** keyDir
  (`$ROOT/keys/` from the setup script).
- Copy **each Mac's** `peer<i>.pub` (from the same `$ROOT/keys/`) **into the
  phone's** keyDir — the same `peerKeysDirectory`. Drop them there via the
  Files app or by pushing them into the Xcode container.

Until both `.pub` files are in place, the phone's dial loop logs
`no public key … yet` and simply retries — copy the file and it connects on the
next 3 s redial, no restart needed.

**This is friction, honestly.** It's the price of no-discovery. Adding Bonjour
advertisement/browse to the memory node (as the compute mesh already has) would
auto-exchange endpoints and let trust-on-first-use replace the manual `.pub`
copy — a clear future improvement, out of scope for this build.

---

## Realistic end-to-end (2 Macs + iPhone, K=2-of-3)

Goal: a memory survives losing **any one** of the three peers.

1. **Two Macs.** `scripts/setup_memory_mesh.sh start` with `INSTANCES="1 2"`
   (two localhost Supermemory instances + configs), or run two Macs each with
   their own `nmp-memory-peer --config …`. Edit each config so the roster also
   lists the **phone** (peerID `0xA1`=161, the phone's LAN IP, udpPort `7810`).
   *(Verified on Mac: the 3-peer localhost demo — write, one-shard-per-peer,
   recall, kill a peer, recall survives, kill the second, explicit
   `quorum_unavailable`.)*
2. **iPhone.** In the Memory tab set `peers` to the two Macs (their LAN IPs +
   peerIDs 1 and 2 + udpPorts), `start()`. Exchange `.pub` files both ways per
   above. Watch the log reach `link to peer 1 established` /
   `…peer 2 established`.
   *(Not-yet-run-on-device: this session did not build/run the app on a
   physical iPhone.)*
3. **Remember** on any device. The author seals + shards K=2-of-N=3; each of the
   three peers (the phone included) stores exactly **one** opaque shard plus the
   bounded plaintext index entry.
4. **Recall** on any device — including after you **kill one peer**
   (`kill -9` a Mac process, or background the phone). Two shards remain ≥ K=2,
   so reconstruction succeeds; the response lists which shards were used and any
   `unreachablePeers`.
5. **Kill a second peer** → only one shard reachable `< K` → recall returns an
   explicit `quorum_unavailable` naming the missing peers. **Never** wrong
   output.

### What's verified vs not

| Part | Status |
|---|---|
| Seal / shard / reconstruct / quorum-failure semantics | **Verified on Mac** (17 codec tests; kill-a-peer demo) |
| Encrypted NMP transport, Noise IK, static-key pinning | **Pre-existing, verified** (compute mesh) |
| `NMPLocalMemoryStore` on-device store + NaturalLanguage embeddings | Built in parallel; **not exercised on a device here** |
| `MemoryPeerController` glue (config, node wiring, threading) | **Compiles against the NMP module**; **not built/run on a device** |
| iPhone joining a live 2-Mac memory mesh, recall surviving a kill | **Not yet run on device** — needs Xcode + a signed iPhone |

The remaining step is exactly what iOS always requires and this session cannot
do: open the project, add the file + screen to the target, sign, and ⌘R to a
physical device.
