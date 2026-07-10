import type { ProtocolEstimate } from '../api';

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
