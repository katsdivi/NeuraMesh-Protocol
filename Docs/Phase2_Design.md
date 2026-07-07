# Phase 2 Design — NACK-Only Reliability + Retransmission Window

Scope: receiver-side loss detection with NACK packets (type `0x11`), a
64-packet sender retransmission window, the sliding replay window that makes
out-of-order delivery legal at the crypto layer, and the Phase 1 carry-in of
published Noise test vectors. No FEC (Phase 3), no discovery (Phase 4).

## What was built

`Reliability.swift` adds three pieces. `NMPNackCodec` defines the NACK wire
payload — `u16 count ‖ count × u32 missing sequence numbers`, big-endian,
strictly validated (declared count must exactly match payload length).
`NMPRetransmitBuffer` is the sender's retransmission window: a 64-slot ring
keyed by `sequence % 64` holding the exact sealed datagram bytes.
`NMPLossTracker` is the receiver's gap tracker: a pure state machine over an
injected clock (fully deterministic under test) that records gaps beneath the
highest authenticated sequence, schedules the first NACK after a reorder
grace delay, re-NACKs on an interval, and abandons a sequence after bounded
attempts or once it ages out of the retransmit window. `PeerConnection`
drives the tracker from a dispatch timer on the connection queue and exposes
`onUnrecoverableLoss` for sequences the layer gave up on — Phase 3's FEC is
the intended consumer.

`SymmetricCrypto.swift`'s replay protection changed from strict monotonic to
`NMPReplayWindow`: RFC 6479 / DTLS-style highest-seen + 64-bit bitmap. A
sequence is accepted iff it is newer than anything seen, or within the
64-packet window and not yet marked. State still advances only after GCM
authentication succeeds, preserving the Phase 1 property that forged packets
cannot poison the window.

## Decisions and tradeoffs

**Retransmissions are byte-identical; the RETRANSMIT flag is unusable.** The
20-byte header is the GCM AAD. Flipping the RETRANSMIT flag on a cached
datagram invalidates its tag; re-sealing the same plaintext under the same
nonce with different AAD is the classic GCM "forbidden attack" (two tags
under one nonce leak the GHASH authentication key); and re-sealing under a
fresh sequence number would hide the packet's identity from the receiver's
gap tracking. Verbatim resend of the cached ciphertext is the only sound
option — same nonce, same plaintext, same AAD is literally the same bytes and
leaks nothing. Consequence: the spec's `RETRANSMIT` flag (bit 1) cannot be
set on actual retransmissions. **Flag for spec revision**: either drop the
flag or move it out of the authenticated header. Receivers distinguish
retransmits implicitly (the sequence fills a tracked gap).

**Window sizes are coupled by design.** Retransmit buffer (sender), loss
tracker depth (receiver), and replay window (crypto) are all 64
(`NMPReplayWindow.windowSize` / `NMPReliabilityConfig.windowSize`). A
retransmitted packet is acceptable to the crypto layer exactly as long as its
sender can still produce it, and gaps are abandoned exactly when retransmit
becomes impossible. Growing one of the three requires growing all three.

**Every sealed packet enters the retransmit window, including NACKs.** The
per-direction sequence space is shared by all encrypted packet types, so the
peer's gap tracker cannot distinguish a lost DATA packet from a lost NACK.
Buffering everything keeps every observable gap fillable and costs at most
64 × MTU per peer. A lost NACK is additionally covered by the re-NACK
interval (verified in tests).

**Gap tracking runs over all authenticated sequences, delivery stays
immediate.** The receiver delivers packets to `onPacket` as they decrypt —
Phase 2 adds recovery, not in-order delivery. Inference tensor traffic is
sequenced at a higher layer (shard headers, Phase 5); imposing head-of-line
blocking here would only add latency. NACK packets are consumed internally
and not delivered to the application.

**Reorder grace before the first NACK.** A gap is NACKed only after
`reorderDelay` (default 8 ms) so plain UDP reordering doesn't trigger
spurious retransmits. A FLUSH-flagged packet (bit 2, set by the sender on the
last packet of a burst) expedites all outstanding gaps immediately — nothing
behind it is in flight to fill them naturally. Defaults: 8 ms reorder delay,
25 ms re-NACK interval, 3 attempts; all tunable via
`PeerConnectionConfig.reliability`.

**Deep gaps are truncated to the window.** If arrival jumps more than 64
sequences, only the newest 63 missing sequences are tracked — the rest are
already outside the sender's buffer and unrecoverable. They surface through
`onUnrecoverableLoss` the same way as exhausted NACK attempts.

## Carry-ins closed from Phase 1

**Noise interop: published vector wired in.** `NoiseIKVectorTests` runs the
official cacophony known-answer vector for `Noise_IK_25519_AESGCM_SHA256`
(fixed statics, fixed ephemerals, "John Galt" prologue): message 1 and 2
ciphertexts, the handshake hash, and all four transport-message ciphertexts
must match byte-for-byte — and do. This closes the "algorithm verified,
Swift translation not" gap. A test-only internal hook
(`ephemeralOverrideForTesting`) injects the fixed ephemerals; it is not
reachable from outside the module.

**Sliding replay window** (see above) — the crypto layer no longer rejects
the reordered/retransmitted packets the loss buffer depends on.

## Known issues — flagged, not silently fixed

**Nonce exhaustion / proactive rekey still open.** `seal` still hard-stops at
sequence `0xFFFFFFFF`. The planned proactive re-handshake at ~2^31 packets
needs the Phase 4 control plane (a rekey control message); deferred there,
as anticipated in Phase1_Design.md.

**Recovery is bounded by the window, not guaranteed.** NACK-only reliability
with a 64-packet window recovers isolated and burst loss but not sustained
loss faster than the NACK round trip, nor anything that ages out. That is by
design — Phase 3's XOR FEC covers the residual; `onUnrecoverableLoss` is its
input signal.

**Timestamps remain unvalidated.** NACK scheduling uses local monotonic time
only (`DispatchTime`); the header's wall-clock timestamp is still never
trusted, consistent with the Phase 1 clock-sync assumption.

## Test inventory (Phase 2 additions)

Unit: `NackCodecTests` (5: round-trips, truncation, count/length mismatch,
slice offsets), `RetransmitBufferTests` (2: retrieval, ring eviction),
`LossTrackerTests` (8: in-order, gap detection + reorder grace, late fill,
retry→give-up, FLUSH expedite, leading gap, deep-gap truncation, age-out),
`NoiseIKVectorTests` (1: cacophony known-answer vector),
SymmetricCryptoTests +2/±1 (reorder-within-window accepted, older-than-window
rejected, far-jump bitmap reset).
Integration (`ReliabilityEnd2EndTests`, mock transport with per-sequence
loss injection): single loss recovered via NACK (recovery time printed,
asserted <100 ms; measures ~9-10 ms), 3-packet burst loss, FLUSH-expedited
recovery with the reorder delay pinned high, permanent black-hole surfacing
via `onUnrecoverableLoss`, and recovery despite the NACK itself being lost.

Measured on Apple Silicon (in-memory transport): NACK recovery ≈ 9-10 ms,
dominated by the 8 ms reorder grace delay, well inside the <50 ms NACK /
<100 ms recovery targets.
