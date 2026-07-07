# Phase 4 Design — Peer Discovery + Capability Advertisement + Coordinator Election

Scope: Bonjour/mDNS service publishing and browsing, the capability
advertisement struct and its two encodings, and deterministic coordinator
election. Orchestration/plumbing only — **no protocol changes**: the Noise IK,
NACK, and FEC layers from Phases 1–3 are untouched. No dynamic re-sharding
(Phase 5), no peer-expiry policy tuning or fault tolerance (Phase 6).

## Measured results (Apple Silicon, debug build)

| Metric | Target | Measured |
|---|---|---|
| Bonjour discovery latency (publish → browse hit, real mDNS) | <2 s | **0.94 s** |
| Coordinator agreement (3 mock peers, all 6 join orders) | deterministic | **identical winner every order** |
| Capability encode/decode | byte-exact round trip | **byte-exact** (re-encode reproduces wire bytes) |
| Regression | all Phase 1–3 tests pass | **143/143 tests, 0 failures** (108 carried + 35 new) |

## What was built

`Capabilities.swift`: the capability struct, its binary wire format, its
Bonjour TXT form, and local measurement (`NMPSystemCapabilityProbe`,
`NMPCPULoadSampler`). `CoordinatorElection.swift`: the deterministic election
over the current peer set. `Bonjour.swift`: `NMPBonjourPublisher` /
`NMPBonjourBrowser` over Network.framework (`NWListener.Service`,
`NWBrowser`). `PeerDiscoveryManager.swift`: ties the three together behind
two small protocols (`NMPCapabilityPublisher`, `NMPPeerDiscoverySource`) so
the orchestration is deterministic under test. `PeerConnection` gains only
`isCoordinator` / `remoteCapabilities` and `updateDiscoveryState(...)`.

## Discovery: Bonjour choice and mechanics

Every device publishes `NeuraMesh-{peerID as %08x}._neuramesh._tcp.local.`
with its capability advertisement in the TXT record, and browses the same
type with `NWBrowser.Descriptor.bonjourWithTXTRecord` — so capabilities
arrive **with** the browse result. Consequences:

- **No connection needed to learn the mesh.** The election runs entirely on
  TXT data; NMP (UDP) handshakes are dialed later, where Phase 5 decides
  they are needed. A 10-device mesh does not need 45 handshakes just to
  pick a coordinator.
- **`_tcp` with a real TCP listener, though NMP data is UDP.**
  Network.framework ties Bonjour registration lifetime to a listener, and
  `_tcp` registration is the standard, most interoperable path. The TCP
  port is a discovery anchor only; inbound TCP connections are refused
  (`newConnectionHandler = { $0.cancel() }`).
- **Capability refresh = TXT re-registration.** Re-assigning
  `NWListener.Service` re-registers with the new TXT record; browsers see a
  `.changed` result with `.metadataChanged`. Default cadence 5 s, and only
  when the capabilities actually changed (an unchanged mesh is mDNS-silent).
- **Removal is event-driven, not polled.** Bonjour announces departures
  (goodbye packets / TTL expiry inside mDNSResponder), which surface as
  `.removed` browse results. The manager's own staleness sweep is therefore
  **disabled by default** (`peerStaleTimeout = 0`): Bonjour does not
  re-announce unchanged services, so a wall-clock sweep would evict live,
  quiet peers. The sweep exists (and is tested) for discovery sources
  without reliable removal events; Phase 6 revisits expiry policy.
- The browser reports the local device's own service, as real mDNS does;
  the manager filters it by peer ID.

## Capability advertisement

One struct, two encodings, documented in `Capabilities.swift`:

1. **Binary (big-endian, versioned)** — for the Noise handshake payload and
   `CAPABILITY_ADV` (0x13) packets. Fixed 16-byte prefix (version, peerID,
   ramMB, computeClass, load, tokens/sec), then length-prefixed strings.
   **Extensibility rule: decoders ignore trailing bytes**, so future
   revisions append fields without breaking deployed peers; the version
   byte is only bumped on incompatible relayouts.
2. **TXT dictionary** — short keys (`v`, `id`, `name`, `ram`, `compute`,
   `load`, `tps`, `fmt`), lenient parse: `id` and `compute` required (a
   peer that cannot be keyed or ranked is useless), everything else
   defaults, unknown keys ignored.

Quantization: load is carried as a whole percent (u8), throughput as
centi-tokens/sec (u32). Encode→decode is byte-exact; decode(encode(x)) == x
up to that precision.

Local measurement: RAM from `ProcessInfo.physicalMemory`; compute class
estimated from platform RAM bands (≥8 GB high, ≥5.5 GB medium, else low —
matches M-series/Pro-phone vs mid-tier vs SE-class); CPU load from Mach
`host_statistics` tick deltas between samples.

## Coordinator election

```
coordinator = peer with highest computeClass, ties → LOWEST peerID
```

- **Deterministic and communication-free.** The rule is a total order over
  any capability set, so every peer with the same view elects the same
  coordinator independently — no ballots, no consensus round, nothing on
  the wire beyond the advertisements themselves. Divergent views converge
  as discovery converges.
- **Load deliberately does not participate.** Load fluctuates every refresh
  interval; ranking on it would thrash the coordinatorship every 5 s.
  Membership (and compute-class) changes are the only re-election triggers.
  Verified by `testLoadDoesNotAffectElection`.
- Empty mesh → no coordinator (`nil`); a 1-peer mesh coordinates itself
  regardless of tier. The local device is a member from `start()`, so a
  running manager always has a coordinator.

## Threading model

`NMPPeerDiscoveryManager` mirrors `PeerConnection`: a caller-supplied serial
queue owns all state, source callbacks hop onto it, all callbacks fire on
it, mutating methods assert `dispatchPrecondition(.onQueue)`. The staleness
clock is injectable, so expiry tests run on synthetic time.

## Testing approach

35 new tests. Election and capability codecs are pure — tested directly,
including all 6 join orders of a 3-peer mesh and a 100-round shuffled-input
determinism check. Mesh assembly, coordinator failover, capability
propagation, and TTL expiry run against a `MockDiscoveryHub` that plays
mDNS (registration replay to late browsers, fan-out of updates/removals,
own-service echo) — fully deterministic, no network. One guarded test
exercises the real publish→browse path over mDNS and **skips** (not fails)
where the network blocks Bonjour.

## Known issues / carried forward

1. **Bonjour on restricted networks.** Managed/corporate networks commonly
   block mDNS; macOS local-network privacy can also deny it. There is no
   manual-IP fallback — flagged for Phase 6+.
2. **Capability measurement accuracy.** `currentLoadPercent` is CPU-tick
   based; GPU/ANE contention is invisible to it, and
   `maxInferenceTokensPerSecond` is 0 until Phase 5 measures real inference
   latency. The election is insulated (it ranks only compute class), but
   Phase 5 shard weighting must not trust load blindly.
3. **Dynamic re-shard triggers.** Capabilities refresh every 5 s; a load
   spike between refreshes is invisible, and nothing yet reacts to a >20%
   load change. Phase 5+ concern.
4. **Coordinator failover orphans in-flight work.** Election re-runs on
   coordinator loss (tested), but Phase 5 in-flight inference recovery is a
   Phase 6 (fault tolerance) concern.
5. **Compute-class estimate is RAM-banded.** An 8 GB Intel Mac ranks
   "high". Fine for ordering today's Apple-Silicon-era meshes; Phase 5's
   measured throughput supersedes it.
