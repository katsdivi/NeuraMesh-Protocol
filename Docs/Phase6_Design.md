# Phase 6 Design — Fault Tolerance + Testing Dashboard + Benchmark Suite

Scope: peer-loss detection and failover (re-shard the survivors,
rebroadcast SHARD_ASSIGN), dynamic re-sharding on peer join, stage-level
retry against unrecoverable transport loss, a web-based testing
dashboard (live peer state, inference monitor, loss injection, packet
log), and a comprehensive benchmark suite (latency distribution,
throughput under loss, failover cost). Final phase.

## Measured results (Apple Silicon, in-process mesh, real crypto/FEC/NACK)

| Metric | Target | Measured |
|---|---|---|
| Peer drop detected (5 s heartbeat + 1 s poll) | <5.5 s | **timeout + ≤1 poll** (scaled test: 0.51 s at a 0.5 s heartbeat) |
| Re-shard (plan + SHARD_ASSIGN ack round) | <500 ms | **0.4 ms** |
| Output after failover | bit-exact | **bit-exact** |
| All peers dead | explicit error, no hang | **`.allPeersDead`** |
| Dashboard | live on :8080 | HTTP + WebSocket, tested end-to-end |
| Loss ≤2% (4-peer mesh, 4 KB activations) | quantified | **no measurable cost** (FEC + expedited NACK) |
| Loss 5% / 10% / 15% throughput vs clean | quantified | **-49% / -55% / -81%** |
| Burst loss (300 ms @ 10%) | recovers <1 s | **one ≤1.1 s spike, then clean** |
| Peer join re-shard | <500 ms | **0.4 ms** |
| Regression | all prior tests pass | **207/207 tests, 0 failures** (178 prior + 29 new) |

## What was built

`PeerHealthMonitor.swift` — activity-based liveness with an injectable
clock. `FaultToleranceOrchestrator.swift` — `NMPFailoverOrchestrator`:
membership accounting, drop/join re-sharding, health-check polling.
`PacketLossInjector.swift` — deterministic loss/burst/blackhole
transport decorator + `NMPInMemoryTransport` loopback pairs.
`MeshTestbed.swift` — a full in-process mesh (real Noise, GCM, FEC,
NACK) shared by the dashboard, benchmarks, and tests.
`BenchmarkSuite.swift` — scenario runner, nearest-rank percentiles, CSV
export. `DashboardServer.swift` — HTTP + RFC 6455 WebSocket server on
NWListener (hand-rolled, zero dependencies, one port).
`Resources/dashboard.html` — the frontend. `Sources/NMPDashboardCLI` —
`swift run nmp-dashboard` (live) and `--benchmark` (headless CSV runs).

Plus three small additive hooks on existing types:
`PeerConnection.onPacketEvent` (structured FEC/NACK/retransmit events —
the dashboard's packet log), `NMPInferenceOrchestrator.onPeerActivity`
(the liveness feed), and `NMPInferenceOrchestrator.stageRetryLimit`
(below).

## Liveness: the pipeline is the heartbeat

NMP has no keepalive packet. During inference, every response chunk,
ack, and metrics packet a peer sends refreshes its heartbeat in
`NMPPeerHealthMonitor`; a tracked peer silent for longer than
`heartbeatTimeout` (5 s) is reported dead by the next poll (1 s
interval → detection within the 5.5 s budget, counted from the peer's
last packet).

The consequence is deliberate: **detection requires traffic in
flight**. An idle mesh has uniformly silent peers and nothing to fail
over — the health check must not fire there, which is why activity for
untracked peers is ignored and why the monitor is polled rather than
event-driven. The tests scale the constants (0.5 s heartbeat, 0.1 s
poll) to keep the suite fast; the machinery is identical.

Two peers dying simultaneously converge sequentially: the first
failover's SHARD_ASSIGN round times out against the second dead peer,
returns `.reshardFailed`, and the next health poll drops the second
peer. Bounded by one assignment timeout per extra dead peer.

## Failover

`handlePeerDrop` removes the peer from membership, detaches its
connection (failing anything in flight toward it), re-plans via
`NMPModelSharder.plan(measuredSecondsPerLayer:)` — the same
largest-remainder apportionment as Phase 5, weighted by measured
per-peer speed, so the heaviest spans land on the fastest survivors —
and runs a normal assignment round. Nothing new on the wire: a
re-shard IS a SHARD_ASSIGN broadcast. The 0.4 ms measured cost is the
plan computation plus one ack round trip on loopback; on Wi-Fi it is
one RTT.

Bit-exactness across failover is structural: shard boundaries move but
the layer math is identical, so the reference engine's output cannot
change — and the tests hold it to bit-identical.

`handlePeerJoin` is the same path with membership growing instead of
shrinking. `registerPeer` adds mesh members during assembly without
per-peer re-shard rounds.

## Stage retry: what happens when NACK gives up

Phase 2's reliability is NACK-only with bounded attempts (3), and under
sustained heavy loss the NACK round itself gets lost often enough that
chunks are abandoned (`onUnrecoverableLoss`): measured ~2% per gap at
15% steady loss. Phase 5's orchestrator would then stall until the
stage timeout and fail the whole inference.

Phase 6 adds `stageRetryLimit` (default 1): a remote stage that hits
`inferenceTimeout` is re-sent once with a fresh requestID. The old
request's completion state is already gone, so a straggling late
response is ignored — a retry can never deliver stale activations. The
peer side drops partial tensors for all older requestIDs when a newer
request arrives (`NMPTensorReassembler.abandonOlder(than:)`), so
sustained loss cannot balloon peer memory. Benchmarks run with a 1 s
stage timeout: a give-up event costs ~1 s instead of killing the run.

## Testing dashboard

`swift run nmp-dashboard` assembles a live in-process mesh (coordinator
+ 3 shard peers, 24 layers, 4 KB activations, 2 ms/layer simulated
compute), drives a continuous generation loop, and serves
http://localhost:8080. One NWListener port carries both the page (GET
/) and the update stream (GET /ws → RFC 6455 upgrade; the accept-key
derivation is tested against the RFC's own example vector).

Outbound JSON: `peer_update` (latency/load/assignment/liveness),
`inference_progress`, `packet_event` (FEC recoveries, NACKs sent,
retransmits served, give-ups — from `onPacketEvent` on both ends of
every link), `mesh_event`, `benchmark_result`, `loss_rate`. Inbound
controls: `set_loss_rate` (slider → injectors on every link),
`inject_peer_drop` (silences a peer, runs failover), `start_benchmark`
(loss sweep, results stream back), `reset_metrics`.

The server is a LOCAL testing tool: plain TCP, no TLS, no auth — do not
expose it beyond the machine running the mesh.

## Loss injection

`NMPPacketLossInjector` decorates any `NMPTransport`: steady loss rate,
AWDL-like bursts (elevated rate for a bounded window), and blackhole
(peer-death simulation). Drop decisions come from splitmix64 seeded per
link, so benchmark runs are reproducible — identical event counts across
trials measure the protocol, not the dice.

## What Phase 6 deliberately leaves out

KV-cache / in-flight state migration (the reference engine is
stateless per pass; a real llama.cpp binding would need the dropped
shard's KV re-computed), peer→peer direct forwarding (still star
relay), coordinator death (the coordinator is the single point of
failure by design — Phase 4's election would re-elect, but resuming
another coordinator's pipeline is out of scope), and keepalive traffic
for idle-mesh detection.

## Files

- `Sources/NMP/PeerHealthMonitor.swift`, `Sources/NMP/FaultToleranceOrchestrator.swift`
- `Sources/NMP/PacketLossInjector.swift`, `Sources/NMP/MeshTestbed.swift`
- `Sources/NMP/BenchmarkSuite.swift`, `Sources/NMP/DashboardServer.swift`
- `Sources/NMP/Resources/dashboard.html`, `Sources/NMPDashboardCLI/main.swift`
- `Tests/NMPTests/FaultToleranceTests.swift` (11), `DashboardTests.swift` (10), `BenchmarkTests.swift` (8)
- `Results/benchmark_summary.csv`, `Results/benchmark_latencies.csv`
- `Docs/Benchmarks.md`
