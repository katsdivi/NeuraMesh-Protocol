# Phase 3 Design — XOR FEC + AWDL Suppression

Scope: XOR forward error correction over 4-packet groups (parity type
`0x12`), receiver-side reconstruction of single losses without a NACK round
trip, AWDL contention inference (loss rate + latency shift), and traffic
shaping that defers non-critical data while the link is contended. No
discovery (Phase 4), no sharding (Phase 5).

## Measured results (Apple Silicon, in-memory transport, debug build)

| Metric | Target | Measured |
|---|---|---|
| Parity computation (4×1400 B group) | <100 µs | **6–16 µs** |
| Reconstruction (one 1400 B packet) | <1 ms | **0.005–0.011 ms** |
| End-to-end loss recovery (drop → delivery) | <1 ms | **0.14–0.16 ms** |
| FEC recovery rate at 2% loss (1000 packets) | ≥80% | **100% (13/13)** |
| Recovery latency vs Phase 2 NACK | <50% | **~1% (0.13 ms vs 9.4–10.2 ms)** |

The recovery-latency comparison is measured by the same test harness with
FEC toggled: identical loss injection, identical mesh, `fec.enabled` is the
only variable (`FECIntegrationTests.testRecoveryLatencyFECVersusNack`).

## What was built

`FECCodec.swift`: CRC32 (IEEE), word-wise (UInt64) XOR over zero-padded
payloads, and the parity packet payload format. `FECGroup.swift`: the
sender-side group builder and receiver-side reconstructor. `AWDLDetector.swift`:
contention inference as a pure state machine over an injected clock.
`TrafficShaper.swift`: the deferral policy. `PeerConnection` wires all four
into the Phase 1-2 send/receive paths.

## Wire format

Parity packet (type `0x12` FEC_RECOVERY, encrypted and sequenced like any
other packet), payload:

```
group_id (u32)            CRC32 over the member sequence numbers
count    (u8, 1...16)     data packets in the group
seqs     (count × u32)    member sequence numbers, send order
lengths  (count × u16)    original payload length of each member
parity   (max(lengths) B) XOR of zero-padded member payloads
```

Two deliberate departures from the build-prompt sketch (`group_id ‖ parity`):

**Explicit member sequence list, not base+count.** NACK and control packets
share the per-direction sequence space, so a group's data sequences are not
guaranteed contiguous — a NACK sealed between two `send()` calls punches a
hole. The prompt's CRC32 group_id alone is also not invertible: a receiver
that missed the FEC_GROUP_END-flagged packet could never learn which
sequences the parity covers. Carrying the list makes the parity packet
self-describing; the CRC32 group_id is kept as an integrity cross-check
(recomputed on decode, mismatch = reject). FEC_GROUP_END remains on the
group-closing packet as a wire-visible marker but is no longer load-bearing.

**Per-member lengths.** Group members may have unequal payload sizes; XOR
runs over zero-padded payloads and reconstruction truncates back to the
missing member's true length.

## Decisions and tradeoffs

**Data packets are never delayed by grouping.** The build prompt's flow
buffers 4 packets, then sends all 5 with 1 ms inter-packet pacing, and the
receiver processes packets on group completion. Both ends of that were
rejected: it adds up to ~5 ms latency to EVERY packet to optimize the <5%
that are lost — backwards for an inference mesh. Here, data packets seal and
transmit the moment the application sends them, the builder accumulates
copies, and the parity follows the 4th packet immediately. Receiver-side,
packets are delivered on arrival; FEC only ever ADDS the reconstructed
missing packet. Cost: no artificial loss decorrelation from pacing. If real
AWDL measurement (Phase 5+) shows correlated group loss, pacing can be
reintroduced as a config knob without wire changes.

**Group size N=4 (25% overhead).** One parity per 4 data packets recovers
any single loss per group. At 2% iid loss, the probability that a lost
packet's group has a second loss (making FEC insufficient) is
1−0.98⁴ ≈ 7.8% of losses — and those fall back to NACK, they are not lost.
N=8 halves overhead to 12.5% but doubles both the second-loss probability
per group and the worst-case reconstruction wait. N=2 (50% overhead) buys
little: NACK fallback already covers the residual at +9 ms. N=4 is the
default; `NMPFECConfig.groupSize` accepts 2...16 and the wire format carries
`count`, so tuning needs no format change.

**Recovered packets enter the replay window.** A reconstructed packet's
content is authenticated transitively — the parity packet and every
surviving member each passed GCM individually. `NMPSecureSession.
markSequenceSeen()` records the recovered sequence, so if the original
datagram straggles in later, or a NACK retransmit races the recovery, it is
dropped as a replay instead of delivering the payload twice. The recovered
packet is delivered under a synthesized header (`timestamp = 0` — the real
header was lost with the packet); tests use that marker to distinguish FEC
deliveries from retransmits.

**FEC cancels the pending NACK; NACK remains the fallback.** On recovery,
`lossTracker.markRecovered()` removes the gap before its 8 ms reorder grace
expires, so no NACK is sent — this is where the 10× latency win comes from.
When FEC cannot help (parity lost, 2+ members lost), the Phase 2 machinery
proceeds untouched; a NACK retransmit that completes a 2-loss group even
unlocks the second member via XOR (`testTwoMissingMembersWaitThenExpire`).
Pending parity groups expire after 50 ms.

**AWDL detection: two signals, hysteresis, honest heuristics.** Engage when
(a) NACKed-sequences / packets-sent exceeds 5% over a 100 ms window with at
least 20 sends (a single early loss must not trip it), or (b) the rolling
median of one-way delay samples shifts above a calm-period baseline by 2×
and at least 5 ms, AND at least one loss sits in the window. Disengage
after 200 ms of sustained calm. The latency samples come from header
timestamps (sender wall clock) — absolute values are polluted by clock
skew, so only the SHIFT is used; the 5 ms floor stops a near-zero loopback
baseline from tripping on noise. The loss-corroboration requirement on (b)
was added after the loopback transport race: a burst of back-to-back sends
inflates one-way delay through the sender's own socket/queue backlog, and
over a near-zero baseline that self-induced queueing read exactly like a
contention spike — suppression engaged at 0.0% loss and deferred every DATA
packet, turning 0.8 ms loopback round trips into 40+ ms ones. A latency
shift with zero loss is queueing, not contention. **These thresholds were
tuned on the in-memory test harness and one machine; they must be
re-validated on real device meshes in Phase 5+.** Note also that FEC hides
recovered losses from the sender (no NACK is ever sent), so the loss-rate
signal only sees loss FEC failed to absorb — which is exactly the loss that
should drive suppression.

**Shaping, FEC, and chunk size are gated on the physical path.**
`NMPTransport` now reports an `NMPLinkKind` (`radio` / `wiredOrLoopback` /
`unknown`) and the kernel's datagram ceiling (`maxDatagramBytes`, clamped to
`net.inet.udp.maxdgram` — `NWConnection.maximumDatagramSize` overreports on
loopback and the kernel EMSGSIZEs anything bigger, silently, because UDP
send errors are advisory); `UDPTransport` classifies the resolved `NWPath`
when the connection becomes ready. AWDL contention is a shared-airtime
phenomenon and FEC parity exists to absorb radio loss, so PeerConnection
applies the detector + shaper + parity only when the path is NOT
`wiredOrLoopback` — loopback and wired Ethernet get zero shaping overhead
and zero parity (the NACK path stays armed as the safety net, and the
receive side still consumes parity from a peer that classified differently).
`PeerConnection.recommendedChunkBytes` follows the same split: MTU-packed
1350 B on radio (fills a 1500 B Wi-Fi MTU with VPN headroom but never
fragments — a lost IP fragment kills the whole datagram), the conservative
1024 B default on unknown paths, and the kernel ceiling minus seal
overhead (9180 B stock) on wired/loopback where fragmentation is loss-free
and per-datagram cost dominates. Tensor senders and the transport race both consult it, and ship
bursts through `sendBurst`/`sendBurstAsync` (one queue hop, transport
writes coalesced via `NWConnection.batch`, FLUSH on the last chunk).

**Traffic shaping defers plaintext, not sealed datagrams.** Sealing assigns
the sequence number; deferring a sealed packet while later packets ship
would punch permanent fake gaps into the receiver's loss tracking. The
shaper therefore holds (type, flags, payload) tuples and seals at actual
send time. During suppression, normal-priority DATA defers; NACKs, parity,
FLUSH-flagged data, and `.critical`-priority data always pass; NACK-driven
retransmits bypass the shaper entirely (cached ciphertext). Deferred data
flushes when suppression clears, or after `maxDeferDelay` (200 ms) as a
backstop — activations cannot wait forever, and the backstop also covers
the case where traffic stops entirely and no event ever drives the state
machine to "clear". Beyond 100 buffered packets `send()` throws
`deferralBufferFull` (explicit backpressure to the application).

**API change.** `send()` now returns `UInt32?` — nil means "deferred by
suppression, will be sent automatically". The only dependent
(`ProtocolComparison`'s NMP adapter) discards the result; verified to still
build.

## Known issues — flagged, not silently fixed

**Parity loss is not special-cased.** If the parity packet is lost, the
group is unprotected and single losses in it fall back to NACK (measured
path, tested). The prompt's "send parity twice" idea (Phase 3b) is not
implemented; at 2% loss the parity-lost case costs one NACK round trip for
~2% of groups — not worth 25%→50% overhead.

**Recovered packets lose their original header.** Flags and timestamp are
gone with the lost datagram; the synthesized header carries `timestamp = 0`
and empty flags. A recovered FLUSH packet therefore does not expedite NACKs
on the receiver. Harmless today (the gap it would have flagged is the one
just recovered), but a header-in-payload echo could fix it if Phase 4
control packets need exact flag recovery.

**Suppression state is event-driven.** The detector only re-evaluates when
packets flow (send/receive/NACK). With zero traffic, suppression stays
engaged until the next event; the defer backstop guarantees buffered data
still leaves within 200 ms. A periodic idle timer would be cleaner; deferred
until a real workload shows it matters.

**Heuristic thresholds are unvalidated on real AWDL.** See above — tuned on
test hardware, marked for Phase 5+ device measurement. The 50 ms pending-
group timeout similarly assumes LAN-scale delivery times.

## Test inventory (Phase 3 additions — 37 new tests)

Unit: `FECCodecTests` (9: parity round-trip at every missing index, unequal
lengths, single-member group, CRC32 known vector `0xCBF43926`, wire
round-trip with non-contiguous sequences, malformed-payload rejection, slice
offsets, plus the two performance gates), `FECGroupBuilderTests` (2: emit on
4th, FLUSH early close), `FECGroupReceiverTests` (5: no-loss discard,
single-missing reconstruction, parity-first reordering, 2-missing wait →
retransmit unlock, stale-group expiry), `AWDLDetectorTests` (8: calm, loss
engage, min-sample guard, sustained-calm disengage, relapse resets timer,
loss-corroborated latency spike engages, uncorroborated spike does NOT,
clock-skew tolerance), `TrafficShaperTests` (4),
`LossTrackerTests` +2 (markRecovered, postponeUnattempted).
Integration (`FECIntegrationTests`, 9): single black-holed packet recovered
via FEC with zero NACK retransmits, parity-loss → NACK fallback, 2-losses →
NACK fallback, one loss per group across 3 interleaved groups, 1000-packet
2% seeded-random loss (recovery rate printed + asserted ≥80%), FEC-vs-NACK
recovery latency comparison (asserted <50%), suppression defers + backstop
flushes, critical priority bypasses suppression, wired/loopback link kind
never engages suppression.

Phase 1-2 adjustments: `testManyPacketsSurviveRoundTrip` now asserts payload
order + monotonic sequences (parity packets legitimately consume every 5th
sequence number); `ReliabilityEnd2EndTests` pins `fec.enabled = false` since
it exercises the NACK path in isolation.
