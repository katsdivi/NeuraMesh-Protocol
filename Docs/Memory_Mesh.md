# Distributed Memory Mesh

A memory layer for NMP where each peer runs its **own local self-hosted
Supermemory instance**, and conversational memories are erasure-coded and
scattered across peers so **no single peer holds a complete readable copy** —
reconstruction needs a K-of-N quorum of shards.

Everything in this doc is either **measured** (from the live demo, labeled as
such) or a stated **design intention**. Where the current implementation is
narrower than the general idea, that limit is written down, not glossed.

---

## Overview

- Each device runs a local Supermemory server (localhost only — a hard guard,
  see below). That server is the device's private memory store.
- Writing a memory: **seal** (compress + encrypt) → **shard** (K-of-N erasure
  code) → **distribute** one shard to each roster peer over the existing NMP
  transport. The full plaintext is persisted **nowhere**.
- Reading a memory: **search** the index → **gather** K shards (local + peers
  over NMP) → **reconstruct** → **open** (decrypt). Below quorum it fails
  loudly (HTTP 503), never returns wrong output.
- The demo kills a real peer process mid-run and recall still succeeds from the
  survivors' shards. Kill a second peer (drop below quorum) and recall fails
  explicitly.

The guarantee, stated precisely: **no single peer holds a COMPLETE readable
copy — full content requires a K-of-N quorum of shards.** This is *not*
threshold secrecy against a peer that holds the AES key (see
[The searchable-index tradeoff](#the-searchable-index-tradeoff)). Do not read it
as zero-knowledge.

---

## Pre-existing vs newly built

**Newly built this session** (all under `Sources/NMP/` unless noted):

- `NMPMemoryShard.swift` — K-of-N XOR erasure codec over arbitrary byte blobs
  plus sealing (`NMPMemoryShardScheme`, `NMPMemoryShardRecord`,
  `NMPMemoryShardCodec`, `NMPMemorySeal`).
- `NMPMemorySeal` (same file) — LZFSE-compress then AES-256-GCM encrypt under a
  fresh random 256-bit key, before sharding.
- `NMPMemoryStore.swift` — the **storage-backend seam**: a protocol (plus
  neutral `NMPStoredDocument` / `NMPStoredHit` / `NMPMemoryStoreError` types)
  giving the node exactly two capabilities — store/fetch an opaque blob by key,
  and semantic search — so a peer can run *either* backend, and a mesh can MIX
  them (Macs on Supermemory, a phone on the native store). Every backend is
  local to its device by construction.
- `NMPSupermemoryClient.swift` — a backend: callback-style HTTP client for one
  device's own local supermemory-server, with a hard localhost-only guard.
  Conforms to `NMPMemoryStore`.
- `NMPLocalMemoryStore.swift` — a backend: a fully **on-device** store (file
  blob store + Apple `NaturalLanguage` embeddings for search, with a
  sentence → word-average → lexical fallback chain; **no network at all**). This
  is what lets an iPhone be a memory peer without the Node `supermemory-server`.
  Conforms to `NMPMemoryStore`.
- `NMPMemoryWire.swift` — wire framing for memory messages plus app-layer
  chunking/reassembly, carried as application DATA over the existing NMP
  transport. Message kinds start at `0x20`, disjoint from the compute mesh's
  `NMPMeshMessageKind` (`0x01`–`0x08`).
- `NMPMemoryMeshNode.swift` — the peer node (backend-agnostic): write path, read
  path, config loader, shard/index persistence into whichever `NMPMemoryStore`
  it is given. `init(config:store:)` injects any backend; `withSupermemory(config:)`
  is the Mac convenience factory.
- `Sources/NMPMemoryPeerCLI/main.swift` — the `nmp-memory-peer` executable
  (added as an executable target in `Package.swift`). `--local-store [dir]` runs
  a peer on the native on-device backend — the same code path an iPhone runs.
- `NeuraMeshPeer/Sources/MemoryPeerController.swift` + `Docs/Memory_Mesh_iOS.md`
  — the iOS integration: a controller that builds a memory node backed by the
  native store, plus the manual Xcode/Info.plist/key-exchange runbook. Compiles
  against the public NMP API; the on-device build is a manual Xcode step (not run
  on a device this session — stated in that doc).
- `scripts/setup_memory_mesh.sh` — stands up 3 independent local Supermemory
  instances + writes per-peer configs. `start|stop|status`.
- `Tests/NMPTests/MemoryShardTests.swift` (17 tests, codec + seal) and
  `Tests/NMPTests/LocalMemoryStoreTests.swift` (12 tests, native store).
  **Measured: all pass** — full suite 474 tests, 0 failures.

**Pre-existing NMP code, reused unchanged** (do not credit these to this
feature):

- The NMP transport: `UDPTransport`, `PacketCodec`, the 20-byte big-endian
  header.
- Noise IK handshake + per-session AES-256-GCM: `NoiseIK`, `SymmetricCrypto`,
  `PeerConnection`.
- Bonjour discovery and static-key peer pinning
  (`PeerConnectionConfig.authorizedStaticKeys`).
- The packet-level XOR erasure algorithm (`FECCodec`/`FECGroup`). The memory
  codec **reuses only the stateless pure primitives**
  `NMPFECCodec.computeParity` / `NMPFECCodec.reconstruct` (free functions) — it
  does **not** touch or repurpose the live packet-transport FEC instance. Shared
  math, not a shared object.

So the memory mesh rides entirely on the existing encrypted, discovered,
key-pinned NMP mesh. What is new is the seal/shard/distribute/reconstruct layer
and the per-device memory-store integration on top of it.

---

## Storage backends (why the phone can join)

The node depends on the `NMPMemoryStore` protocol, not on Supermemory directly.
Two backends implement it, and a single mesh can mix them:

- **`NMPSupermemoryClient`** (Macs) — HTTP to this device's own localhost
  `supermemory-server`.
- **`NMPLocalMemoryStore`** (iPhone, or any Mac via `--local-store`) — a fully
  on-device store: a file blob store for shards/index entries plus Apple
  `NaturalLanguage` embeddings for semantic search, with a
  sentence → word-average → lexical fallback chain and **no network**. The
  iPhone can't run the Node `supermemory-server` (iOS forbids spawning a server
  executable and has no Node runtime), so it runs this instead. See
  `Docs/Memory_Mesh_iOS.md`.

**Measured (live, this session):** a mixed 3-peer mesh — two Macs on Supermemory
and one peer on `NMPLocalMemoryStore` (`--local-store`, active backend
"on-device (NLEmbedding sentence)") — sharded a memory 2-of-3, then a Supermemory
peer was **killed** and peer 1 reconstructed the full plaintext using its own
shard plus **the native-store peer's shard served over NMP**. That exercises the
exact backend code an iPhone runs, in a live mesh, with a real peer down. The
on-device store's own 12 unit tests pass; the iOS *app* wiring compiles against
the public API but was not built on a physical device this session.

---

## Architecture / data flow

**Sealing.** Plaintext memory → LZFSE compress → AES-256-GCM encrypt under a
fresh random 256-bit key. The result is opaque ciphertext. GCM's authentication
tag makes reconstruction **tamper-evident**: a wrong or corrupted shard makes
`open` fail loudly rather than returning garbage.

**Sharding.** The sealed ciphertext is split by a K-of-N XOR erasure code into N
shards, any K of which reconstruct the original. Each shard is opaque ciphertext
at rest.

**Distribution.** One shard goes to each roster peer via
`PeerConnection.sendBurst` over the existing NMP transport (encrypted, key-pinned
links). Each peer persists its one shard into its own local Supermemory, tagged
`nmp_shards`. Peers also store a small plaintext index entry (see next section),
tagged `nmp_memory_index`.

**Read.** The reading node searches the index (Supermemory semantic search of
the index tag, with a fallback — see
[Ingestion latency & the fallback](#ingestion-latency--the-fallback)), then
gathers K shards: its own local shard plus shards fetched from peers over NMP.
Once K are in hand it reconstructs, then opens (decrypts + decompresses). It
fetches only K — surplus peers are reported as `peersNotNeeded`.

**Failure.** If fewer than K shards are reachable, recall returns HTTP 503 with
an explicit `quorum_unavailable` error naming the missing/unreachable peers. No
silent wrong output.

**Local-only guard.** `NMPSupermemoryConfig.init` **throws** `nonLocalBaseURL`
unless the host is `localhost` / `127.0.0.1` / `::1`. Cloud endpoints
(`console`/`api.supermemory.ai`) cannot be configured. The setup script also
asserts no config `baseURL` contains `supermemory.ai`.

---

## The searchable-index tradeoff

A fully opaque shard is **not** semantically searchable. To keep real recall
working, every roster peer also stores a small **plaintext index entry** in its
own Supermemory (tag `nmp_memory_index`, distinct from the opaque `nmp_shards`
and from any of the peer's own native readable memories). Each index entry holds:

- the memory **title**,
- a **bounded plaintext snippet** (default 160 chars, config `indexSummaryChars`),
- the memory's **AES key**,
- the shard **roster**.

**What this buys:** real semantic recall on every surviving peer — even after
the author's device is gone — plus tamper-evident reconstruction.

**What this costs (stated plainly):** a single peer *does* see the bounded
plaintext snippet, and because that peer also holds the AES key in the index
entry's metadata, a single peer *could* decrypt its own 1/K ciphertext fragment.

So the guarantee is exactly: **no single peer holds a COMPLETE readable copy;
full content requires a K-of-N quorum of shards.** It is **not** threshold
secrecy against a peer that has the key — that would require Shamir secret
sharing or a different code, which is out of scope. This is deliberately *not*
sold as zero-knowledge.

---

## Ingestion latency & the fallback

Supermemory ingestion is **asynchronous**: a document moves through
`queued → embedding → indexing → done`, and becomes searchable at roughly the
`indexing` stage. Search must pass `searchMode: "documents"` — the default
`"memories"` mode needs an LLM provider and returns nothing on a default local
instance, so the client hardcodes `"documents"`.

Measured behavior:

- On a **warm** instance, a document was searchable **~1.2 s** after add.
- On a **freshly-booted cold** instance, the in-process local embedding model
  (`Xenova/bge-base-en-v1.5`) takes a while to warm and the queue drains slowly.
  During the live test, documents on a just-booted instance were still at
  `queued` for **40 s+**, and drain rate varied per instance (one demo instance
  drained and served search while another was still `queued` minutes later).
- Once an index document reached `done`, **`/v4/search` over the mesh's own
  index tag returned the right memory live** — measured **similarity 0.71** for
  the wine-cellar memory against the query "wine cellar barolo" (and 0.48 for an
  unrelated memory), confirming the semantic-recall path end to end, not just the
  fallback.

Because of this, the read path has a documented **fallback**: it first does a
Supermemory semantic search over the index tag; if that returns nothing
(ingestion still catching up), it falls back to an **in-RAM keyword match** over
the index entries the node already holds.

This fallback is a **freshness bridge, not the search story.** Supermemory *is*
the search story once a memory is ingested; the in-RAM match only covers the
window between "just written" and "indexed." Practical consequence for demos:
warm the instances first, or rely on the fallback for just-written memories (see
the runbook's warmup note).

---

## Limitations

- **Single XOR parity ⇒ n == k+1.** The erasure codec uses one XOR parity shard,
  so it tolerates **exactly one** shard loss. True arbitrary-N erasure coding
  would need Reed-Solomon — out of scope. (Documented in the code as well.)
- **Not threshold secrecy.** As above: a peer holding the AES key (via its index
  entry) can decrypt its own fragment. The quorum protects *completeness*, not
  *secrecy against a key-holding peer*.
- **Index snippets are plaintext.** The bounded snippet (default 160 chars) is
  readable on every roster peer by design — that is the price of semantic recall.
- **Trusted-LAN posture.** The `nmp-memory-peer` control HTTP API is
  loopback-only, with no TLS and no auth — same stance as the dashboard. Never
  port-forward it.

---

## Files

| Path | What |
|---|---|
| `Sources/NMP/NMPMemoryShard.swift` | K-of-N XOR erasure codec + `NMPMemorySeal` (LZFSE + AES-256-GCM) |
| `Sources/NMP/NMPMemoryStore.swift` | backend seam: `NMPMemoryStore` protocol + neutral doc/hit/error types |
| `Sources/NMP/NMPSupermemoryClient.swift` | Supermemory backend — localhost-only HTTP client |
| `Sources/NMP/NMPLocalMemoryStore.swift` | native on-device backend — file blobs + `NaturalLanguage` embeddings, no network |
| `Sources/NMP/NMPMemoryWire.swift` | memory message framing + chunking/reassembly (kinds `0x20`+) |
| `Sources/NMP/NMPMemoryMeshNode.swift` | the backend-agnostic peer node: write/read paths, config, persistence |
| `Sources/NMPMemoryPeerCLI/main.swift` | `nmp-memory-peer` executable + loopback control API (`--local-store` for the native backend) |
| `NeuraMeshPeer/Sources/MemoryPeerController.swift` | iOS glue: memory node backed by the native store |
| `Docs/Memory_Mesh_iOS.md` | iOS integration runbook (manual Xcode steps, key exchange) |
| `scripts/setup_memory_mesh.sh` | stands up 3 local Supermemory instances + per-peer configs (`start\|stop\|status`) |
| `scripts/run_memory_demo.sh` | drives the kill-a-peer demo (write → prove 1 shard/peer → recall → `kill -9` a peer → recall survives → optional `--kill-two` shows explicit quorum failure) |
| `Tests/NMPTests/MemoryShardTests.swift` | 17 codec + seal tests (all passing) |
| `Tests/NMPTests/LocalMemoryStoreTests.swift` | 12 native-store tests (all passing) |

Reused unchanged: `UDPTransport`, `PacketCodec`, `NoiseIK`, `SymmetricCrypto`,
`PeerConnection`, Bonjour discovery, `PeerConnectionConfig.authorizedStaticKeys`,
and the `NMPFECCodec.computeParity`/`reconstruct` primitives.
