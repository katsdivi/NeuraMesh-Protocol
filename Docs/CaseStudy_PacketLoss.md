# Case Study: How NeuraMesh Handles Packet Loss

*All numbers are wall-clock measurements (Apple M-series, 2026-07). The
micro-benchmarks are pinned by tests you can run yourself; the macro
sweep is reproducible with the commands at the end.*

## Why loss is the problem that shapes the protocol

Distributed inference generates one token per mesh round trip. A
32-token answer is 32 sequential trips; whatever loss does to one trip,
it does to the whole generation, compounded.

TCP's loss response is head-of-line blocking: a single lost segment
stalls **every byte behind it** until a retransmission completes — one
round trip minimum, usually more with timer backoff. The stream
guarantees order the application doesn't need (tensor chunks carry their
own indices and reassemble in any order) and pays for it in exactly the
currency inference can't spare: latency.

NMP's position: on a LAN mesh, loss should cost a *packet's* worth of
repair, not a *pipeline stall*. Three layers deliver that, and a fourth
makes sure the protocol never manufactures loss conditions itself.

## Layer 1: XOR FEC — repair without a round trip

Every 4 outbound DATA packets form a group; the group closes with one
parity packet (word-wise XOR of the members, CRC32-verified, explicit
member list so interleaved groups survive reordering). Lose any single
member and the receiver rebuilds it locally:

- Parity computation: **6–16 µs** per group (target was <100 µs).
- Reconstruction: **≈0.01 ms**; end-to-end drop→delivery **≈0.15 ms**.
- At 2% injected loss, **100%** of losses were recovered with zero NACKs
  (`FECIntegrationTests.testSingleLossRecoveredViaFECWithoutNack`,
  seeded-random 1000-packet run alongside it).

That 0.15 ms versus the ~9.4 ms NACK path is the headline: **~75× faster
than retransmission**, and no reverse traffic on a link that's already
struggling.

FEC-recovered losses are also deliberately invisible to the sender — no
NACK is ever sent for them — so the contention detector (layer 4) only
sees loss the FEC layer failed to absorb, which is exactly the loss that
should drive behavior changes.

## Layer 2: NACK-only retransmission — the safety net

When FEC can't help (parity lost too, or 2+ losses in one group),
receivers report gaps explicitly. There are no ACKs and no sender
timers: silence means delivered. Senders keep the last 64 sealed
datagrams and retransmit **verbatim** — the header is GCM AAD, so
mutating it post-seal would break the tag, and re-sealing under the same
nonce is the GCM forbidden attack. Measured recovery: **≈9.4–10.2 ms**.

Gaps are only NACKed after an 8 ms reorder delay (UDP reordering must
not trigger spurious retransmits), re-NACKed every 25 ms, and abandoned
honestly (`onUnrecoverableLoss`) if the sender's window has moved on —
at which point the orchestrator retries the whole stage, so a generation
survives even pathological loss.

## Layer 3: FLUSH — bursts end loudly

A tensor ships as a burst of chunks with FLUSH on the last one. FLUSH
tells the receiver "nothing is coming behind this to reveal gaps" — so
gap detection is expedited instead of waiting for later traffic, and the
FEC group closes immediately. Burst sending (`sendBurst`) tightened this
further: chunks leave back-to-back, so a gap is exposed by the very next
packet, microseconds later.

## Layer 4: don't create loss — link-aware restraint

On peer-to-peer Wi-Fi (AWDL), the mesh's own traffic degrades the medium
it depends on. NMP infers contention from two signals — NACKed/sent
ratio over 100 ms, and a shift in one-way delay medians — and defers
*background* traffic while it lasts (tensor traffic is `critical`
priority and never waits).

Two hard-won rules keep this subsystem honest:

- **A latency shift alone is not contention.** A send burst inflates
  one-way delay through the sender's own queue backlog; over a near-zero
  baseline that reads exactly like a contention spike. Early versions
  engaged suppression at 0.0% loss on loopback and turned 0.8 ms round
  trips into 40+ ms ones. The latency signal now requires at least one
  real loss in the window as corroboration.
- **No radio, no shaping.** The transport classifies its path
  (`NMPLinkKind`); loopback and wired Ethernet run with shaping *and*
  FEC parity disabled — there is no airtime to protect and essentially
  nothing to lose, so protection would be pure overhead. Radio paths get
  MTU-safe 1350 B chunks (an IP fragment lost on Wi-Fi kills its whole
  datagram, so NMP never fragments there), parity, and shaping.

## Measured results

In-process mesh with production crypto/FEC/NACK over an in-memory
transport, deterministic injected loss, 12 generations × 4 tokens per
run, medians of 3 paired (clean vs lossy) runs, 2026-07-13:

| Injected loss | Throughput vs clean | p95 latency vs clean |
|---|---|---|
| 2% | 1.00× — free | 1.00× |
| 5% | 1.00× — free | 1.19× |
| 10% | ~1× (within run-to-run noise) | 1.11× |
| 15% | 0.88× | 1.97× |
| 20% | 0.72× | 1.97× |
| 25% | inference times out | — |

Reading it honestly:

- **Up to 10% loss is absorbed silently.** FEC eats single losses per
  group without a round trip; NACKs mop up the rest inside the same
  trip's shadow. Real Wi-Fi rarely exceeds low single digits.
- **15–20% degrades gracefully** — a ~2× tail and up to −28% throughput,
  but every generation completes and output stays bit-exact (loss
  recovery is exercised under seeded loss in the test suite with
  bit-exactness asserted).
- **25% sustained is the breaking point**: NACK rounds themselves get
  lost repeatedly and stages time out. The protocol reports the failure
  explicitly rather than hanging.
- For contrast, TCP under loss forfeits its transfer-speed advantage to
  head-of-line stalls — this is directly observable in the transport
  race under the loss lab (below).

Earlier in the project the same suite measured −49%/−55%/−81% throughput
at 5%/10%/15% loss (Phase 6 sign-off). The current numbers — free, free,
−12% — are the compound effect of FLUSH-expedited gap detection, burst
sending, and FEC/NACK tuning since then. The improvement was large
enough that the loss benchmark's assertion margins went stale and had to
be re-measured at a higher loss rate.

## Reproduce it

```bash
# Micro: FEC vs NACK recovery latency, recovery rates, group mechanics
swift test --filter FECIntegrationTests
swift test --filter BenchmarkTests/testThroughputDeclinesUnderHeavyLoss

# Macro: the dashboard's chaos slider injects live loss into a running mesh
swift run nmp-dashboard --ui        # Pressure tab / chaos slider

# Real packet loss on real sockets (dnctl/pfctl, needs sudo):
sudo scripts/loss_lab.sh            # shapes the race port band
# then run the transport race from the UI (Race transports) and watch
# NMP's FEC recover while the TCP legs stall.
```
