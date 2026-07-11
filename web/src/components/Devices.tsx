import { useCallback, useEffect, useRef, useState } from 'react';
import { api, type DeviceMetrics, type WebClient } from '../api';

/**
 * Mesh 2.1 device management: live host resources (real kernel counters,
 * 2 s polling) plus per-peer mesh facts and the compute-share slider.
 * Moving a slider POSTs /api/devices/:id/allocate, the coordinator
 * re-shards, and the new layer spans show up here (and on every other
 * open device) — that visible re-shard is the proof the allocation is
 * real, not UI theater.
 */
export function Devices() {
  const [metrics, setMetrics] = useState<DeviceMetrics | null>(null);
  const [clients, setClients] = useState<WebClient[]>([]);
  const [unavailable, setUnavailable] = useState(false);
  const [busyPeer, setBusyPeer] = useState<string | null>(null);
  const [lastAction, setLastAction] = useState('');
  const [error, setError] = useState('');
  // Slider positions the user is currently dragging (peer id → share),
  // so polling doesn't yank the thumb mid-drag.
  const [draft, setDraft] = useState<Record<string, number>>({});
  const draftRef = useRef(draft);
  draftRef.current = draft;

  const refresh = useCallback(async () => {
    try {
      const [nextMetrics, nextClients] = await Promise.all([
        api.deviceMetrics(),
        api.clients(),
      ]);
      setMetrics(nextMetrics);
      setClients(nextClients);
      setUnavailable(false);
    } catch {
      setUnavailable(true);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 2000);
    return () => clearInterval(timer);
  }, [refresh]);

  const commitShare = async (peerId: string, share: number) => {
    setBusyPeer(peerId);
    setError('');
    try {
      const response = await api.allocate(peerId, share);
      setLastAction(`Re-sharded: ${response.summary}`);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyPeer(null);
      setDraft((current) => {
        const next = { ...current };
        delete next[peerId];
        return next;
      });
    }
  };

  const host = metrics?.host;

  return (
    <div>
      <h1>Devices</h1>

      {unavailable && (
        <div className="error-box">
          /api/devices/metrics is unreachable — is the coordinator running?
        </div>
      )}

      {host && (
        <div className="card">
          <h3>
            {host.hostname}
            {metrics?.generation_in_flight && (
              <span className="badge measured" style={{ marginLeft: 8 }}>
                generating
              </span>
            )}
          </h3>
          <ResourceBar
            label="RAM"
            percent={host.ram_used_percent}
            detail={`${(host.ram_used_mb / 1024).toFixed(1)} / ${(
              host.ram_total_mb / 1024
            ).toFixed(0)} GB used`}
          />
          <ResourceBar
            label="This process"
            percent={(host.process_footprint_mb / host.ram_total_mb) * 100}
            detail={`${host.process_footprint_mb} MB footprint — watch it move while a model loads or a generation runs`}
          />
          <ResourceBar
            label="Storage"
            percent={host.storage_used_percent}
            detail={`${host.storage_free_gb.toFixed(0)} GB free of ${host.storage_total_gb.toFixed(0)} GB`}
          />
          <ResourceBar
            label="CPU"
            percent={host.cpu_percent ?? 0}
            detail={
              host.cpu_percent === undefined
                ? 'first sample — utilization needs two readings'
                : `${host.cpu_percent.toFixed(1)}% across all cores`
            }
          />
          <div className="note-box" style={{ marginTop: 'var(--spacing-md)' }}>
            {metrics?.host_note}
          </div>
        </div>
      )}

      <h2>Mesh peers</h2>
      {metrics && !metrics.allocation_supported && (
        <div className="note-box">{metrics.allocation_note}</div>
      )}
      {metrics?.peers.map((peer) => {
        const share = draft[peer.id] ?? peer.compute_share;
        return (
          <div key={peer.id} className="card">
            <div className="device-card" style={{ border: 'none', padding: 0 }}>
              <div className={`device-dot ${peer.alive ? 'alive' : 'dead'}`} />
              <div className="device-name">{peer.name}</div>
              <div className="device-stats">
                <span>{peer.assigned}</span>
                {peer.layer_span !== undefined && (
                  <span>{peer.layer_span} layer(s)</span>
                )}
                {peer.measured_ms_per_layer !== undefined && (
                  <span>{peer.measured_ms_per_layer.toFixed(2)} ms/layer</span>
                )}
                {peer.computing && <span className="badge measured">computing</span>}
              </div>
            </div>
            {metrics.allocation_supported && (
              <div className="allocation-row">
                <label>
                  Mesh compute share: <strong>{Math.round(share * 100)}%</strong>
                </label>
                <input
                  type="range"
                  min={10}
                  max={100}
                  step={5}
                  value={Math.round(share * 100)}
                  disabled={busyPeer !== null}
                  onChange={(event) =>
                    setDraft((current) => ({
                      ...current,
                      [peer.id]: Number(event.target.value) / 100,
                    }))
                  }
                  onPointerUp={() => {
                    const pending = draftRef.current[peer.id];
                    if (pending !== undefined) commitShare(peer.id, pending);
                  }}
                  onKeyUp={() => {
                    const pending = draftRef.current[peer.id];
                    if (pending !== undefined) commitShare(peer.id, pending);
                  }}
                />
                {busyPeer === peer.id && <span>re-sharding…</span>}
              </div>
            )}
          </div>
        );
      })}

      {lastAction && <div className="note-box">{lastAction}</div>}
      {error && <div className="error-box">{error}</div>}

      <h2>Web UI clients</h2>
      <div className="card">
        {clients.length === 0 && <div>none (which is odd — you are one)</div>}
        {clients.map((client, index) => (
          <div key={index} className="device-card" style={{ border: 'none' }}>
            <div className={`device-dot ${client.websocket ? 'alive' : 'dead'}`} />
            <div className="device-name">{client.address}</div>
            <div className="device-stats">
              <span>{shortAgent(client.user_agent)}</span>
              <span>
                {client.websocket
                  ? 'live (WebSocket)'
                  : `seen ${client.seconds_since_seen.toFixed(0)} s ago`}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ResourceBar({
  label,
  percent,
  detail,
}: {
  label: string;
  percent: number;
  detail: string;
}) {
  const clamped = Math.max(0, Math.min(100, percent));
  return (
    <div className="resource-row">
      <div className="resource-head">
        <span className="resource-label">{label}</span>
        <span className="resource-detail">{detail}</span>
      </div>
      <div className="resource-bar">
        <div
          className={`resource-fill ${clamped > 85 ? 'hot' : ''}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}

/** "Mozilla/5.0 (iPhone; …" → the device-ish part a human wants. */
function shortAgent(userAgent: string): string {
  const match = userAgent.match(/\(([^)]+)\)/);
  const inner = match ? match[1].split(';')[0].trim() : userAgent;
  return inner.slice(0, 40) || 'unknown browser';
}
