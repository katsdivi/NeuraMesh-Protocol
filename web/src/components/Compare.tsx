import { useState } from 'react';
import { api, type ProtocolEstimate } from '../api';
import { ProtocolComparison } from './ProtocolComparison';

/**
 * Standalone what-if explorer over POST /api/comparison: take a measured
 * run's numbers (defaults = the Phase 9 zero-trim measurement) and see
 * how the same run prices out over modeled TCP/QUIC at a chosen RTT and
 * loss rate.
 */
export function Compare() {
  const [tokens, setTokens] = useState(32);
  const [payloadKB, setPayloadKB] = useState(11.9);
  const [roundTrips, setRoundTrips] = useState(33);
  const [measuredMs, setMeasuredMs] = useState(2280);
  const [rttMs, setRttMs] = useState(2);
  const [lossPercent, setLossPercent] = useState(0);
  const [result, setResult] = useState<{
    note: string;
    protocols: ProtocolEstimate[];
  } | null>(null);
  const [error, setError] = useState('');

  const run = async () => {
    setError('');
    try {
      setResult(
        await api.comparison({
          tokens,
          payload_bytes: Math.round(payloadKB * 1024),
          round_trips: roundTrips,
          measured_total_ms: measuredMs,
          lan_rtt_ms: rttMs,
          loss_rate: lossPercent / 100,
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div>
      <h1>Protocol What-If</h1>
      <div className="note-box">
        Feed in a measured run (defaults: the Phase 9 Llama-2-7B zero-trim
        measurement) and re-price it over modeled TCP+TLS and QUIC at your
        chosen RTT and loss rate. Run an inference with “Compare protocols”
        checked to use live numbers instead.
      </div>

      <div className="card">
        <div className="form-row">
          <div className="form-group">
            <label>Tokens</label>
            <input
              type="number"
              value={tokens}
              min={1}
              onChange={(event) => setTokens(Number(event.target.value) || 1)}
            />
          </div>
          <div className="form-group">
            <label>Payload (KB)</label>
            <input
              type="number"
              value={payloadKB}
              min={0}
              step={0.1}
              onChange={(event) => setPayloadKB(Number(event.target.value) || 0)}
            />
          </div>
          <div className="form-group">
            <label>Round trips</label>
            <input
              type="number"
              value={roundTrips}
              min={1}
              onChange={(event) => setRoundTrips(Number(event.target.value) || 1)}
            />
          </div>
        </div>
        <div className="form-row">
          <div className="form-group">
            <label>Measured total (ms)</label>
            <input
              type="number"
              value={measuredMs}
              min={1}
              onChange={(event) => setMeasuredMs(Number(event.target.value) || 1)}
            />
          </div>
          <div className="form-group">
            <label>LAN RTT (ms)</label>
            <input
              type="number"
              value={rttMs}
              min={0.1}
              step={0.5}
              onChange={(event) => setRttMs(Number(event.target.value) || 0.1)}
            />
          </div>
          <div className="form-group">
            <label>Loss rate (%)</label>
            <input
              type="number"
              value={lossPercent}
              min={0}
              max={20}
              step={0.5}
              onChange={(event) =>
                setLossPercent(Math.max(0, Math.min(20, Number(event.target.value) || 0)))
              }
            />
          </div>
        </div>
        <button className="primary-button" onClick={run}>
          Compare
        </button>
        {error && <div className="error-box">{error}</div>}
      </div>

      {result && (
        <ProtocolComparison protocols={result.protocols} note={result.note} />
      )}

      <div className="card" style={{ marginTop: 'var(--spacing-lg)' }}>
        <h3>Why NMP is built for the local mesh</h3>
        <ul className="assumptions" style={{ fontSize: 'var(--text-body)' }}>
          <li>
            <strong>Handshake:</strong> Noise IK completes in 1 RTT (~1 ms measured
            on loopback) — TCP+TLS 1.3 needs 2 RTTs before the first byte.
          </li>
          <li>
            <strong>Loss recovery:</strong> XOR FEC reconstructs a lost packet in
            ~0.15 ms (measured), ~75× faster than the NACK path and orders of
            magnitude ahead of TCP retransmission timers.
          </li>
          <li>
            <strong>Payload:</strong> Phase 9 zero-trim moves ~12 KB per 32-token
            generation where the Phase 8 wire moved ~1 MB — fewer packets, less
            FEC/NACK exposure per token.
          </li>
          <li>
            <strong>Scope honesty:</strong> NMP targets sub-50 ms LANs. Across the
            internet, use QUIC — that's what it's optimized for.
          </li>
        </ul>
      </div>
    </div>
  );
}
