import type { Device, MeshHealth } from '../api';
import type { MeshEvent } from '../hooks/useMesh';

export function Dashboard({
  health,
  devices,
  events,
  reachable,
}: {
  health: MeshHealth | null;
  devices: Device[];
  events: MeshEvent[];
  reachable: boolean;
}) {
  const mesh = health?.mesh;
  return (
    <div>
      <h1>Your Mesh</h1>

      <div className="grid">
        <div className="metric-card">
          <div className="metric-label">Devices</div>
          <div className="metric-value">
            {mesh ? `${mesh.peers_alive}/${mesh.peers}` : '—'}
          </div>
          <div className="metric-sub">alive / total</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Model</div>
          <div className="metric-value" style={{ fontSize: '1.1rem' }}>
            {mesh?.model || '—'}
          </div>
          <div className="metric-sub">{mesh?.engine} engine</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Shards</div>
          <div className="metric-value">{mesh?.shard_count ?? '—'}</div>
          <div className="metric-sub">wire: {mesh?.wire_format ?? '—'}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Status</div>
          <div className={`metric-value ${reachable ? 'online' : 'offline'}`}>
            {reachable ? '✓ Online' : '✗ Offline'}
          </div>
          <div className="metric-sub">
            {mesh?.speculation_available ? 'speculation ready' : ' '}
          </div>
        </div>
      </div>

      <h2>Peers</h2>
      {devices.length === 0 && (
        <div className="card">No peer state reported yet — run an inference.</div>
      )}
      {devices.map((device) => (
        <div key={device.id} className="device-card">
          <div className={`device-dot ${device.alive ? 'alive' : 'dead'}`} />
          <div className="device-name">{device.name}</div>
          <div className="device-stats">
            <span>{device.assigned}</span>
            <span>latency {device.latency_ms} ms</span>
            <span>load {device.load_percent}%</span>
            <span>{device.alive ? 'online' : 'lost'}</span>
          </div>
        </div>
      ))}

      <h2 style={{ marginTop: 'var(--spacing-xl)' }}>Live Events</h2>
      <div className="event-log">
        {events.length === 0 && <div>— quiet —</div>}
        {events.map((event, index) => (
          <div key={index}>
            [{event.time}] {event.message}
          </div>
        ))}
      </div>
    </div>
  );
}
