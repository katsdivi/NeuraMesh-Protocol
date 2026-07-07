# NMP Benchmarks — running and interpreting

## Running

```bash
swift run nmp-dashboard --benchmark            # full suite → Results/*.csv
swift run nmp-dashboard --benchmark /tmp/out   # custom output directory
swift run nmp-dashboard                        # interactive: dashboard on :8080
swift test --filter BenchmarkTests             # CI-sized versions of each scenario
```

The suite runs over `NMPMeshTestbed`: a coordinator + 3 shard peers in
one process, connected by in-memory transports wrapped in
`NMPPacketLossInjector`. Everything above the datagram layer is the
real stack — Noise IK handshakes, AES-256-GCM, sequencing, XOR FEC,
NACK reliability — so loss numbers measure actual protocol recovery,
not a simulation of it. Loss dice are splitmix64-seeded per link:
rerunning a scenario reproduces the same loss pattern.

**Units.** One *generation* = N sequential pipeline passes ("tokens"),
each feeding its output activations back as the next input — the shape
of autoregressive decoding. Latencies are per generation; throughput is
tokens/second across the scenario. Every pass is verified bit-exact
against a single-device baseline while being timed.

**Percentiles** are nearest-rank (index `ceil(q·n)-1`, clamped) — exact
for small n. With 10 generations, p95 and p99 are both the maximum;
treat them as "worst observed", not as smooth tail estimates.

## Reference results

Apple M-series, in-process mesh, 4 shards × 6 layers, 4 KB activation
tensors, 2 ms/layer simulated compute (~48 ms/pass of "model time" per
8-token generation is compute; the rest is protocol). From
`Results/benchmark_summary.csv`:

| Scenario | p50 | p95 | throughput |
|---|---|---|---|
| no loss, 8 tokens | 105.6 ms | 106.9 ms | 75.4 tok/s |
| loss 1% | 107.6 ms | 108.9 ms | 74.2 tok/s |
| loss 2% | 107.3 ms | 107.9 ms | 74.6 tok/s |
| loss 5% | 104.2 ms | 1118.6 ms | 38.5 tok/s |
| loss 10% | 130.0 ms | 1158.8 ms | 33.8 tok/s |
| loss 15% | 144.0 ms | 2268.1 ms | 14.0 tok/s |
| burst 10% / 300 ms | 102.7 ms | 1117.2 ms | 38.5 tok/s |
| peer drop after gen 5 | 94.6 ms | 107.0 ms | 81.9 tok/s (re-shard 0.4 ms) |
| peer join after gen 8 | 106.6 ms | 113.8 ms | 78.5 tok/s (re-shard 0.4 ms) |

## How to read these numbers

**Loss ≤2% is free.** p50 and throughput are indistinguishable from
clean. Single losses inside a 4-packet FEC group are reconstructed from
parity in ~0.01 ms; multi-loss groups are repaired by NACK, and because
tensor bursts end with a FLUSH packet, gap detection is expedited — the
repair is one retransmit round trip, sub-millisecond on loopback. This
is the Phase 2+3 design working exactly as intended.

**The knee is where NACK rounds themselves die.** A NACK or its
retransmit is subject to the same loss; with 3 attempts the chance of
abandoning a chunk is ≈(2p)³ per gap — negligible at 2%, ~0.1% at 5%,
~2% at 15%. An abandoned chunk means the stage stalls until the 1 s
stage timeout, then Phase 6's stage retry re-sends the request
(`stageRetryLimit`, default 1). Those events are the 1.1 s / 2.3 s p95
entries: **the p50 barely moves while p95 jumps two orders of
magnitude** — sustained heavy loss turns into rare, bounded stalls
rather than degraded steady-state latency. The throughput column
aggregates both effects.

**Burst loss looks like its steady-state rate while it lasts.** A
300 ms burst at 10% spans ~3 generations; the affected generations pay
the same tail risk as steady 10% (here: one 1.1 s spike), then recovery
is immediate — the first post-burst generation is back at ~103 ms.
There is no lingering state: FEC groups and the NACK window flush with
the traffic.

**Failover costs a re-plan, not a rebuild.** Re-sharding is 0.4 ms on
loopback (plan + one SHARD_ASSIGN ack round); on real Wi-Fi expect one
RTT. Throughput after a drop went UP in this run (81.9 vs 75.4 tok/s)
— one fewer star-relay hop outweighs the extra layers per surviving
peer when compute is cheap. On real models expect the opposite: the
survivors' compute share grows. Detection (not shown in the table) is
bounded by heartbeat timeout + poll interval: 5 s + 1 s defaults,
counted from the dead peer's last packet — the drop scenario here calls
failover directly to separate re-shard cost from detection wait.

**What "throughput" is NOT.** The reference engine emulates compute at
2 ms/layer; real 7B-scale layers cost more, shrinking the protocol's
share of the wall clock (Phase 5 measured 1.02× mesh overhead with
5 ms/layer). Use the loss columns as *relative* statements about the
transport under stress, not absolute tokens/s of any model.

## CSV schema

`benchmark_summary.csv`: one row per scenario —
`scenario, loss_rate, tokens_per_generation, generations, p50_ms,
p95_ms, p99_ms, avg_ms, min_ms, max_ms, throughput_tokens_per_s, notes`.

`benchmark_latencies.csv`: one row per generation —
`scenario, generation, latency_ms` — for plotting distributions and
locating spikes (e.g. the drop generation, burst windows).

## Comparing runs

Keep the mesh shape (peer count, layer count, hidden size,
simulated compute) fixed between runs; all of them shift absolute
numbers. The injector seeds are fixed per link, so identical
configurations see identical loss patterns — a diff between two
summary CSVs isolates the code change. Expect a few percent of noise
on p50 from scheduler jitter; treat p95 differences under 2× as noise
when give-up events are involved (they are quantized to the stage
timeout).
