import type { ProtocolEstimate, RaceLeg, RaceProjection } from '../api';

/**
 * Mesh 2.5: the fully-measured race table — NMP vs plain TCP vs
 * TCP+TLS 1.3 vs QUIC over real loopback sockets, plus the per-transport
 * whole-generation splice (arithmetic on measurements, labeled as such).
 */
export function RaceResults({
  race,
  projected,
}: {
  race: { note: string; legs: RaceLeg[] };
  projected: RaceProjection[];
}) {
  return (
    <div className="card" style={{ marginTop: 'var(--spacing-lg)' }}>
      <h3>Measured Transport Race</h3>
      <div className="note-box">{race.note}</div>
      <div className="comparison-table">
        <table>
          <thead>
            <tr>
              <th>Leg</th>
              <th>Transport</th>
              <th>Handshake</th>
              <th>Transfer</th>
              <th>Per trip</th>
              <th>Total</th>
            </tr>
          </thead>
          <tbody>
            {race.legs.map((leg) => (
              <tr key={leg.name} className={leg.name === 'NMP' ? 'nmp' : ''}>
                <td>
                  <strong>{leg.name}</strong>{' '}
                  <span className="badge measured">measured</span>
                </td>
                <td style={{ fontSize: 'var(--text-caption)' }}>{leg.transport}</td>
                <td>{leg.handshake_ms.toFixed(2)} ms</td>
                <td>{leg.transfer_ms.toFixed(1)} ms</td>
                <td>{leg.per_trip_ms.toFixed(2)} ms</td>
                <td>
                  <strong>{leg.total_ms.toFixed(1)} ms</strong>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {projected.length > 0 && (
        <>
          <h3 style={{ marginTop: 'var(--spacing-md)' }}>
            Whole generation, per transport
          </h3>
          <div className="grid">
            {projected.map((projection) => (
              <div className="metric-card" key={projection.name}>
                <div className="metric-label">{projection.name}</div>
                <div className="metric-value">
                  {projection.tokens_per_sec.toFixed(1)}
                </div>
                <div className="metric-sub">
                  tok/s · {projection.total_ms.toFixed(0)} ms · {projection.basis}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

/**
 * Renders the measured-vs-modeled comparison. Honesty is load-bearing
 * here: the NMP row is a real run; TCP/QUIC are that run re-priced with
 * modeled transport costs — badges and the assumptions list say so.
 */
export function ProtocolComparison({
  protocols,
  note,
}: {
  protocols: ProtocolEstimate[];
  note?: string;
}) {
  const maxThroughput = Math.max(...protocols.map((p) => p.tokens_per_sec), 1);
  const nmp = protocols.find((p) => p.measured);
  const others = protocols.filter((p) => !p.measured);

  return (
    <div className="card" style={{ marginTop: 'var(--spacing-lg)' }}>
      <h3>Protocol Comparison</h3>
      {note && <div className="note-box">{note}</div>}

      <div className="comparison-table">
        <table>
          <thead>
            <tr>
              <th>Protocol</th>
              <th>Throughput</th>
              <th>Total</th>
              <th>Handshake</th>
              <th>Per-trip overhead</th>
              <th>Loss recovery</th>
            </tr>
          </thead>
          <tbody>
            {protocols.map((proto) => (
              <tr key={proto.name} className={proto.measured ? 'nmp' : ''}>
                <td>
                  <strong>{proto.name}</strong>
                  <span className={`badge ${proto.measured ? 'measured' : 'modeled'}`}>
                    {proto.measured ? 'measured' : 'modeled'}
                  </span>
                </td>
                <td>
                  <div className="throughput-bar">
                    <div
                      className="fill"
                      style={{
                        width: `${(proto.tokens_per_sec / maxThroughput) * 100}%`,
                        opacity: proto.measured ? 1 : 0.45,
                      }}
                    />
                    <span>{proto.tokens_per_sec.toFixed(1)} tok/s</span>
                  </div>
                </td>
                <td>{proto.total_ms.toFixed(0)} ms</td>
                <td>{proto.handshake_ms.toFixed(1)} ms</td>
                <td>
                  {proto.per_trip_overhead_ms > 0
                    ? `+${proto.per_trip_overhead_ms.toFixed(2)} ms`
                    : '—'}
                </td>
                <td>{proto.loss_recovery_ms.toFixed(2)} ms/pkt</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {nmp && others.length > 0 && (
        <div className="note-box">
          Under this model, the measured NMP run would cost{' '}
          {others
            .map(
              (proto) =>
                `${(proto.total_ms - nmp.total_ms).toFixed(1)} ms more over ${proto.name}`,
            )
            .join(' and ')}
          . The gap widens with loss: NMP recovers a lost packet in{' '}
          {nmp.loss_recovery_ms.toFixed(2)} ms via FEC (measured), vs{' '}
          {others.map((p) => `${p.loss_recovery_ms.toFixed(1)} ms (${p.name})`).join(', ')}.
        </div>
      )}

      <ul className="assumptions">
        {protocols.map((proto) => (
          <li key={proto.name}>
            <strong>{proto.name}:</strong> {proto.assumptions}
          </li>
        ))}
      </ul>
    </div>
  );
}
