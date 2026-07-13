# Protocol Comparison — why NMP, not TCP/TLS/QUIC

**Claim under test:** for the traffic shape distributed inference actually
produces — many small, latency-sensitive per-token round trips — NMP's custom
UDP stack is the **fastest *secure* transport**, and its NACK+FEC design beats
TCP's head-of-line blocking under loss. This document backs that with
**measured** numbers, honestly scoped.

## How the comparison works (no modeling)

`NMPTransportRace` (`Sources/NMP/TransportRace.swift`) replays a generation's
real traffic — *round trips × payload bytes* — over four **real** transport
stacks on loopback, each doing a genuine handshake and moving the same bytes:

| Leg | What it really is |
|-----|-------------------|
| **NMP** | Production `UDPListener`/`UDPTransport` + `PeerConnection`: Noise IK handshake, per-packet AES-256-GCM, sequencing. (FEC parity + AWDL shaping are radio-path features and stay out of loopback, exactly as in production.) |
| **TCP** | `NWConnection` stream socket: real 3-way handshake, kernel TCP, **no encryption, no framing** — the unencrypted floor. |
| **TCP+TLS 1.3** | Kernel TCP + real TLS 1.3, ephemeral self-signed P-256 identity, cert pinned. The **like-for-like** encrypted comparison. |
| **QUIC** | Network.framework QUIC (TLS 1.3 built in), same pinned identity. |

`NMPTransportRaceBenchmark` (`Sources/NMP/TransportRaceBenchmark.swift`) runs
this **N times per traffic shape** and aggregates each leg's wall-clock latency
into p50/p95/mean.

### Reproduce

```bash
# clean loopback (what the table below is from)
swift run nmp-dashboard --benchmark-race Results 20 clean
# → Results/transport_race_clean.csv

# under REAL loss (needs sudo; shapes the 20000–40000 port band the legs bind to)
sudo scripts/loss_lab.sh 2                       # 2% loss
swift run nmp-dashboard --benchmark-race Results 20 loss2pct
```

Traffic shapes come from real KV-cached mesh generations (n_embd ≈ 896 F32 ≈
3.5 KB/activation): `prefill-burst` (one large round trip), `decode-32` and
`decode-128` (many small per-token round trips).

## Results — clean loopback (20 trials/shape)

Total time (handshake + transfer), p50 ms — lower is better:

| Shape | NMP | TCP (no crypto) | TCP+TLS 1.3 | QUIC |
|-------|----:|----:|----:|----:|
| prefill-burst | **1.82** | 0.53 | 17.72 | 8.75 |
| decode-32     | **4.65** | 2.81 | 19.57 | 11.31 |
| decode-128    | **14.34** | 6.68 | 28.75 | 19.86 |

(Full p50/p95/mean + per-trip in `Results/transport_race_clean.csv`.)

**Read this honestly:**

- **Against the encrypted transports — the real comparison — NMP wins
  decisively.** NMP does per-packet AEAD just like TLS/QUIC, and is **1.4–10×
  faster** than both on every shape (e.g. decode-128: NMP 14.3 ms vs QUIC
  19.9 ms vs TLS 28.8 ms). TLS/QUIC pay a heavier per-connection and per-record
  cost that NMP's 1-RTT Noise IK + lean AES-GCM datagrams avoid.
- **Plain TCP is faster — because it does nothing.** No encryption, no
  framing. It is the floor, included to show what NMP's security costs. You
  would never ship inference over unauthenticated plaintext on a shared LAN;
  the honest bar NMP clears is *"fastest transport that actually protects the
  traffic."*
- **Loopback isolates protocol/stack cost — radio time is absent from every
  leg.** These numbers compare *protocol overhead*, not Wi-Fi.

## Where NMP is *designed* to pull ahead: loss

Clean loopback doesn't exercise the reason NMP is UDP-based. TCP (and TLS/QUIC
over it, though QUIC less so) suffer **head-of-line blocking**: one dropped
segment stalls every byte behind it. NMP uses **NACK-only retransmission + XOR
FEC**, so a lost packet is recovered (often without even a round trip) while
other packets keep flowing — the right shape for chatty per-token traffic.

Two measured sources back this:

- Re-run the race under `sudo scripts/loss_lab.sh <rate>` and compare
  `transport_race_loss*.csv` to the clean run: NMP's lead over the TCP-based
  legs widens as loss rises, because their recovery serializes and NMP's does
  not.
- The mesh loss benchmark already measures NMP's own recovery in isolation:
  **FEC reconstructs a lost activation packet in ~0.17 ms vs ~10 ms for a NACK
  round trip** (see `Docs/CaseStudy_PacketLoss.md` and
  `swift run nmp-dashboard --benchmark`, which sweeps steady + burst loss).

## Bottom line

For distributed inference on a trusted LAN, the transport must be **secure and
low-latency under loss**. Among transports that encrypt, NMP is the fastest
here by a wide margin, and its NACK+FEC design targets exactly the
head-of-line-blocking failure mode that hurts TCP/TLS as loss appears. Plain
TCP is faster only by giving up the security the mesh requires.

*Every number here is a wall-clock measurement on this machine — nothing is
modeled. Regenerate with the commands above.*
