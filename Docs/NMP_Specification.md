# NMP Specification (Reference Summary)

The authoritative specification is the NMP build prompt supplied with this project.
This file summarizes the wire-level facts Phases 1-3 implement, for quick reference
while reading the code. Where the build prompt and this summary disagree, the
build prompt wins.

## Handshake — Noise IK

Pattern: `Noise_IK_25519_AESGCM_SHA256`. Pre-shared static Curve25519 keys,
mutual authentication, 1 RTT.

- Msg 1 (initiator→responder): Noise `e, es, s, ss`; NMP header T=0 type `0x00`;
  packet payload = `nonce_seed(8, BE) ‖ noise_message`; Noise payload = capability advert.
- Msg 2 (responder→initiator): Noise `e, ee, se`; header T=0 type `0x01`; same payload framing.
- After msg 2: `Split()` → k1 (initiator→responder), k2 (responder→initiator). AES-256-GCM.
- Failure recovery: responder times out after 5 s waiting for msg 1. Initiator
  retries msg 1 with backoff 5 s / 10 s / 20 s, max 3 retries, then the peer is
  marked unreachable for 60 s.

## Packet format (20-byte header, big-endian)

```
byte 0      V(1) | T(1) | R(1) | FLAGS(5)
byte 1      PACKET_TYPE
bytes 2-3   PAYLOAD_LENGTH (u16)
bytes 4-7   SEQUENCE_NUMBER (u32, per-peer per-direction, wraps at 2^32 — see design doc)
bytes 8-11  SENDER_PEER_ID (u32)
bytes 12-19 TIMESTAMP (u64, ns since epoch, sender clock)
...         PAYLOAD
trailing    GCM_TAG (16 B, only when T=1)
```

FLAGS: bit0 `FEC_GROUP_END`, bit1 `RETRANSMIT`, bit2 `FLUSH`, bits3-4 reserved (0).

Types: `0x00` HANDSHAKE_MSG1, `0x01` HANDSHAKE_MSG2 (T=0);
`0x10` DATA, `0x11` NACK, `0x12` FEC_RECOVERY, `0x13` CAPABILITY_ADV,
`0x14` SHARD_ASSIGN, `0x15` ACK_RANGE, `0xFF` CONTROL (T=1).

## Encryption (T=1 packets)

- AES-256-GCM, session key from Noise Split().
- Nonce (12 B) = `nonce_seed(8) ‖ sequence_number(4)`; each direction uses the
  seed advertised by that direction's sender during the handshake.
- AAD = the 20-byte header (header tampering ⇒ authentication failure).
- Replay: 64-bit sliding window (highest authenticated sequence + bitmap).
  Accept iff newer than highest seen, or within the window and unseen.
  (Phase 1 shipped strict `sequence_number > last_seen`; widened in Phase 2
  so the loss buffer can hold reordered/retransmitted packets.)

## Reliability (Phase 2) — NACK type `0x11`

- NACK payload: `count (u16) ‖ count × missing sequence_number (u32)`, big-endian.
- Sender keeps the last 64 sealed datagrams; a NACKed sequence is resent
  **verbatim** (header is GCM AAD — see Phase2_Design.md; the RETRANSMIT flag
  cannot be set on retransmissions, flagged for spec revision).
- Receiver NACKs a gap after a reorder grace delay (default 8 ms), re-NACKs
  every 25 ms, gives up after 3 attempts or when the gap ages out of the
  64-packet window. FLUSH-flagged packets expedite all outstanding NACKs.

## FEC (Phase 3) — parity type `0x12`

- One parity packet per group of N=4 data packets (configurable 2...16).
- Parity payload: `group_id (u32, CRC32 of member seqs) ‖ count (u8) ‖
  seqs (count × u32) ‖ lengths (count × u16) ‖ parity (max(lengths) bytes)`.
  Member sequences are explicit — they are NOT contiguous when NACK/control
  packets interleave. Parity = XOR of zero-padded member payloads.
- FEC_GROUP_END flag marks the group-closing data packet (wire marker only;
  the parity packet is self-describing).
- Receiver reconstructs a single missing member (parity ⊕ survivors,
  truncated to the member's length), cancels its pending NACK, and marks the
  sequence in the replay window (a straggler/retransmit then drops as a
  replay). 2+ missing members fall back to the Phase 2 NACK path; pending
  groups expire after 50 ms.

## AWDL suppression (Phase 3)

- Engage: NACKed/sent > 5% over 100 ms (min 20 sends), or one-way delay
  median shifted >2× and >5 ms above the calm baseline. Disengage: 200 ms
  of sustained calm. (Heuristics; re-validate on real meshes, Phase 5+.)
- While engaged: normal-priority DATA defers (up to 200 ms / 100 packets);
  NACK, FEC parity, FLUSH-flagged, and critical-priority packets pass;
  retransmits bypass shaping entirely.

## Discovery (Phase 4)

- Every device publishes Bonjour service `NeuraMesh-{peerID as %08x}` of type
  `_neuramesh._tcp` in domain `local.`, and browses the same type
  (with TXT records). The advertised TCP port is a discovery anchor only;
  NMP data stays UDP, inbound TCP is refused.
- TXT record carries the capability advertisement as key/value strings:
  `v` (format version), `id` (peerID, lowercase hex), `name`, `ram` (MB),
  `compute` (`high`/`medium`/`low`), `load` (whole percent), `tps`
  (tokens/sec, 2 decimals), `fmt` (comma-separated model formats).
  Parsers: `id` + `compute` required, everything else defaults, unknown
  keys ignored.
- Binary capability format (handshake payload / CAPABILITY_ADV `0x13`),
  big-endian: `version(u8=1) ‖ peerID(u32) ‖ ramMB(u32) ‖ compute(u8:
  0 low, 1 medium, 2 high) ‖ load(u8, 0-100) ‖ tps(u32, centi-tok/s) ‖
  nameLen(u8) ‖ name(UTF-8) ‖ fmtCount(u8) ‖ fmtCount × (len(u8) ‖ UTF-8)`.
  Decoders MUST ignore trailing bytes (future fields append).
- Coordinator election, run independently by every peer over its current
  view (local device included): **highest compute class wins; ties break
  to the lowest peerID**. Load does not participate (prevents coordinator
  thrash). Deterministic — no randomness, no election messages.
- Capability refresh: re-measure and re-register the TXT record every 5 s
  (only when changed). Peer removal is event-driven via Bonjour removal
  results; wall-clock staleness sweep is off by default.

## Phase roadmap

1. ✅ Core transport: handshake, encryption, sequencing.
2. ✅ NACK-only reliability + retransmission window.
3. ✅ XOR FEC (N=4 groups) + AWDL suppression.
4. ✅ Bonjour discovery + capability advertisement + coordinator election.
5. Shard orchestration + pipelined multi-peer inference (GGUF).
6. Fault tolerance + packet inspector dashboard + mesh simulator + benchmarks.
