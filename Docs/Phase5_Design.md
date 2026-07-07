# Phase 5 Design — Shard Orchestration + Multi-Peer Inference

Scope: GGUF container parsing, the compute-engine seam, deterministic
model sharding, SHARD_ASSIGN / inference wire formats with application-
layer tensor chunking, the coordinator orchestrator and peer serving
engine, turn-key peer/coordinator runtimes, the CLI benchmark harness,
and the iOS peer app. No fault tolerance (peer drop mid-inference,
re-sharding on join/leave = Phase 6), no peer↔peer direct forwarding.

## Measured results (Apple Silicon, real UDP + Bonjour, two processes)

| Metric | Target | Measured |
|---|---|---|
| Mesh vs single-device latency (emulated 5 ms/layer compute) | <2× | **1.02×** |
| Numeric correctness (mesh vs single device) | ±0.01 | **bit-exact** |
| Network overhead per token | quantified | **2.0× the activation tensor** (in + out per remote shard) |
| Discovery → dial → handshake → assigned | works unattended | **~3 s** |
| Inference under 11% injected loss | correct output | **bit-exact** (FEC + NACK repair) |
| Regression | all prior tests pass | **178/178 tests, 0 failures** |

With network-dominated toy compute (0.5 ms/shard) the same benchmark
reads 2.08× — the ratio is a compute-to-RTT statement, not a protocol
property; at 7B-scale per-layer cost the network share is noise. Star
relay (below) costs one LAN RTT per remote stage; the LATENCY win of
distribution is bounded — the real wins are memory (each device holds
1/N of the weights) and pipelining multiple tokens (Phase 6+).

## What was built

`GGUF.swift` — real GGUF v2/v3 parsing (header, full KV typology,
tensor directory, alignment), little-endian per the GGUF spec,
memory-mapped so a 5 GB file costs only header pages, hostile-count
guards. `ComputeEngine.swift` — the `NMPShardComputeEngine` protocol and
the deterministic reference engine. `ShardMessages.swift` — SHARD_ASSIGN
payload + the DATA envelope (request/response metas, tensor chunks,
acks, metrics) + chunker/reassembler. `ModelSharder.swift` —
largest-remainder proportional layer apportionment. 
`InferenceOrchestrator.swift` / `PeerShardEngine.swift` — the coordinator
and peer halves of the pipeline. `PeerNode.swift` — `NMPPeerNode` and
`NMPCoordinatorNode`, the turn-key runtimes shared by the CLIs and the
iOS app. `Sources/NMPPeerCLI`, `Sources/NMPCoordinatorCLI` — the
harness. `NeuraMeshPeer/` — the iOS app sources + required plist keys.

## The compute seam

Everything above `NMPShardComputeEngine` (sharding, wire formats,
orchestration, discovery, the apps) is engine-agnostic:

```swift
public protocol NMPShardComputeEngine: AnyObject {
    var layerCount: Int { get }
    var hiddenSize: Int { get }
    func runLayers(start: Int, end: Int, input: [Float]) throws -> [Float]
}
```

Phase 5 ships `NMPReferenceComputeEngine`: per-layer mixing with
coefficients from splitmix64 and the rational squash `x/(1+|x|)`. Two
properties are load-bearing and tested:

1. **Cross-platform bit-exactness** — only IEEE-754 basic ops, no
   transcendentals (libm last-bit rounding varies across builds). Every
   device computing the same layers gets bit-identical floats, so the
   mesh is held to `mesh output == single-device output` EXACTLY. Any
   transport corruption, shard-boundary bug, or codec asymmetry is a
   hard failure, not a drift hidden inside a ±0.01 tolerance.
2. **Composition** — `run(0,N) == run(0,k) ∘ run(k,N)` for every split
   point. This is the algebraic property sharding relies on;
   `testLayerCompositionMatchesSingleShot` pins it.

**Binding llama.cpp** (deliberately not vendored — it would pin a fast-
moving C++ dependency into a protocol repo): conform a wrapper that owns
a `llama_context` and implements `runLayers` over the assigned range;
pass it to `NMPPeerNode`/`NMPCoordinatorNode` instead of the reference
engine. GGUF metadata already flows: `--gguf` sizes engines from the
real file. Note llama.cpp's layer-range execution requires either the
`llama_kv_cache`-aware partial-eval API or a per-shard model slice;
that integration (and ±tolerance numerics, since real GPU kernels are
not bit-reproducible) is the first task of a "Phase 5.5" once a real
model is in play.

## Star relay, not peer chain

The build prompt's diagram forwards activations peer→peer. Phase 5
deliberately relays through the coordinator (coordinator → shard i →
coordinator → shard i+1 …):

- N links instead of N², and no peer↔peer key distribution.
- Every timing/byte measurement observed at one place (the report).
- One extra LAN RTT per stage — measured 1–3 ms against the loopback
  mesh, irrelevant vs real per-layer compute.

Peer→peer forwarding belongs with Phase 6 fault tolerance (it needs
liveness detection between non-coordinator peers anyway).

## Tensor chunking (why not one datagram per tensor)

A 4096-wide f32 activation is 16 KB; UDP would IP-fragment it into ~11
fragments, and ONE lost fragment discards the whole datagram — at 2%
fragment loss that's ~20% tensor loss, and each loss costs a full-tensor
retransmit. Instead tensors are split into ≤1024-byte chunks (+7 B
envelope + 36 B NMP header/tag < 1500 MTU): each chunk is an NMP packet,
individually FEC-grouped and NACK-repairable, so a lost chunk costs one
1 KB recovery. `testInferenceSurvivesPacketLoss` drives 1-in-9 datagram
loss through a full inference and still demands bit-exact output.

Request/response metadata travels as a separate meta message rather than
a header on chunk 0, so metas and chunks survive arbitrary reordering
(the reassembler completes from whichever arrives last — tested).

## Dialing discovered peers (Capabilities v2 + the port trick)

Phase 4 discovery told you a peer exists; Phase 5 must CONNECT. Two
additions:

1. **Capabilities v2** appends `udpPort` and the Noise static public key
   (binary: `port(u16) ‖ pkLen(u8) ‖ pk`; TXT: `port`, `pk` base64).
   v1 blobs still decode (fields default) — the Phase 4 forward-compat
   rule paying out one phase later.
2. **Same-port anchor**: a peer binds its NMP UDP listener on ephemeral
   port P, then its Bonjour TCP anchor listener on the same P. The SRV
   record therefore carries P, so the coordinator dials the browse
   result's `.service` endpoint over UDP and Network.framework's own
   SRV resolution lands on the NMP listener. No IP parsing, no TXT
   address hacks, survives DHCP changes.

Security model, stated plainly: the responder's static key is read from
its TXT record — trust-on-first-use against an active LAN attacker —
and peers accept any authenticated initiator. Right for a benchmark
mesh on your own Wi-Fi; production pins fleet keys via
`authorizedStaticKeys` (both hooks exist since Phase 1).

## Sharder

Largest-remainder apportionment of layers proportional to speed score
(measured seconds/layer when known — the orchestrator harvests it from
every response's `computeMicros` — else class weights 4:2:1), floor of
1 layer per included peer, surplus peers dropped slowest-first, pipeline
order = the election's total order (class desc, peerID asc). Pure
function; determinism, proportionality, contiguity, and coverage are
all pinned by tests.

## Testing approach

35 new tests (178 total). GGUF parsing runs against synthetic containers
built byte-by-byte, including hostile counts. Codecs: round trips +
malformed rejection + reorder/duplicate reassembly. The integration
tests assemble REAL meshes — full Noise handshake, encryption, FEC,
NACK over MockTransport — for 2-peer, 3-peer, lossy, mismatch-rejection,
timeout, and metrics paths, always comparing against the single-engine
baseline bit-for-bit. The CLI harness was validated live as two OS
processes over real UDP + Bonjour (results table above).

## Known issues / carried forward

1. **Real quantized inference is a binding, not a rewrite** — see "The
   compute seam". Until then, tokens/sec figures measure the mesh, not
   Llama.
2. **Sequential pipeline** — one token in flight; shard i idles while
   i+1 computes. Pipelining multiple tokens (and with it the 3-peer
   <1.5× target) is Phase 6 territory.
3. **No failover** — a peer dying mid-inference surfaces as a stage
   timeout error; nothing re-shards or retries elsewhere. Phase 6.
4. **TOFU keys on the auto-dial path** (above).
5. **iOS backgrounding** — iOS suspends the peer app when backgrounded;
   the app holds the screen awake while foregrounded, but long-running
   background compute needs a proper strategy (Phase 6+).
6. **Coordinator queue carries local compute** — fine for sequential
   Phase 5; token pipelining must move local shards off-queue.
