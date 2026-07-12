# Pressure-Testing the Mesh (Mesh 2.5)

How to hammer a live NeuraMesh — including a real iPhone peer built from
Xcode — until something breaks, and how to see *what* broke. Everything
here is measured against real processes; nothing simulates a device.

## The three loss/throttle knobs (know which one you're turning)

| Knob | What it shapes | Real packets dropped? |
|---|---|---|
| Chaos slider (Settings tab) | in-process testbed links only | yes, inside the process — exercises FEC/NACK on real NMP frames, but never touches the Wi-Fi link or LAN peers |
| `scripts/loss_lab.sh` (sudo) | loopback ports 20000–39999 → the transport race legs | yes, in the kernel — this is how you get **measured** loss recovery for NMP vs TCP vs TLS vs QUIC |
| Network Link Conditioner (iPhone) | the phone's entire radio | yes, on the actual Wi-Fi path to your real peer |

The phone knob is the only one that stresses the true cross-device path.
Enable it on the iPhone once Developer Mode is on: **Settings ▸ Developer
▸ Network Link Conditioner** → profiles like *100% Loss*, *Very Bad
Network*, *Edge*. Turn it on mid-generation and watch the dashboard: the
keepalive/pass-retry machinery (Mesh 2.4.1) should re-shard around the
phone and recover instead of failing the generation.

## Hammering the live mesh over HTTP

The dashboard is a plain HTTP API, so a shell loop is a load generator.
Sequential soak (run for minutes, watch for drift in latency or RSS):

```bash
for i in $(seq 1 50); do
  curl -s -X POST localhost:3000/api/inference \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"pressure run '$i'","max_tokens":64}' \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r.get("latency_ms"), r.get("tokens_per_sec"), r.get("error",""))'
done
```

Concurrent burst (the coordinator serializes generations — bursts test
queueing and socket handling, and every request must still answer):

```bash
for i in 1 2 3 4 5; do
  curl -s -X POST localhost:3000/api/inference \
    -d '{"prompt":"burst '$i'","max_tokens":16}' > /tmp/burst_$i.json &
done; wait; grep -l error /tmp/burst_*.json || echo "all bursts clean"
```

Max-payload passes (128 tokens × 4096-float activations ≈ 16 KB/pass
each way on the reference mesh): set `"max_tokens": 128`.

While hammering, background/foreground the iPhone app repeatedly — that
is the harshest real-world event the mesh sees (iOS suspends the peer
mid-stage; expect re-shard + pass retry, not a failed generation).

What counts as a bug: a generation that returns an `error` when at least
one peer stayed healthy; a peer that never rejoins after the app comes
back; dashboard memory growing run over run; a stuck generation with no
`generation_*` events on `/ws`.

## Watching the iPhone side in Xcode

With the app running from Xcode (⌘R):

- **Live logs**: the Xcode console shows joins, shard assignments, and
  per-pass compute. A peer that goes quiet while the Mac shows it alive
  is a keepalive bug — report it.
- **Memory**: Debug navigator (⌘7) ▸ Memory. Steady state should be
  flat between generations; a stair-step per run is a leak. For proof:
  Product ▸ Profile (⌘I) ▸ **Leaks** template, then hammer from the Mac
  and watch for red leak flags during tensor traffic.
- **CPU/energy**: Instruments ▸ **Time Profiler** while a generation
  runs shows where compute goes (should be the layer loops, not codec or
  lock churn). **Energy Log** on-device shows if the radio, not compute,
  dominates.
- **Network**: Instruments ▸ **Network** template shows the encrypted
  UDP flows; per-generation byte counts should match the dashboard's
  wire_in/out figures (they're measured at the same seam).

## Protocol-level fuzzing

The packet codec and mesh-message decoders are fuzzed in the test suite
(malformed headers, truncations, wrong versions, replay) — `swift test
--filter PacketCodec` and the ShardMessages decode tests. The listener
also survives raw garbage from the network: any datagram that fails
authentication is dropped before it reaches protocol state. To fuzz the
live listener from the Mac:

```bash
# 10k random datagrams at the coordinator's UDP port (see startup log)
python3 - <<'EOF'
import os, random, socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for _ in range(10_000):
    s.sendto(os.urandom(random.randint(1, 1400)), ("127.0.0.1", PORT))
EOF
```

The mesh must keep serving inference throughout — AEAD authentication
rejects every forged packet. If a fuzz run changes any behaviour other
than log noise, that's a finding.

## Measured loss recovery (the loss lab)

```bash
sudo scripts/loss_lab.sh start 5 10     # 5% loss + 10 ms delay
curl -s -X POST localhost:3000/api/comparison/run \
  -d '{"prompt":"loss lab","max_tokens":16}' | python3 -m json.tool
sudo scripts/loss_lab.sh stop
```

Compare the same race clean vs shaped: the per-leg `transfer_ms` growth
IS the loss-recovery cost, measured — NMP recovers via FEC parity
without a round trip; TCP pays retransmission timers; QUIC pays its own
loss detection. No constants, no model.
