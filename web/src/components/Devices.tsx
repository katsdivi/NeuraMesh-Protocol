import { useCallback, useEffect, useRef, useState } from 'react';
import {
  api,
  type DeviceMetrics,
  type PeerMetric,
  type WebClient,
} from '../api';

/**
 * Mesh 2.1/2.3 device management: live host resources (real kernel
 * counters, 2 s polling), full per-device cards — resources the device
 * itself reported over the mesh, live wire throughput per link, requests
 * actually served — plus mesh totals and the compute-share slider.
 * Moving a slider POSTs /api/devices/:id/allocate, the coordinator
 * re-shards, and the new layer spans show up here (and on every other
 * open device) — that visible re-shard is the proof the allocation is
 * real, not UI theater.
 *
 * MEASUREMENT HONESTY: in-process peers share this Mac's hardware, and
 * their cards say so instead of inventing per-device RAM/GPU numbers.
 * A physical peer (second Mac via `swift run nmp-peer`, iPhone app)
 * reports its own kernel counters and gets real bars of its own.
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
  const totals = metrics?.totals;

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
            <span className="badge modeled" style={{ marginLeft: 8 }}>
              host
            </span>
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
            label="CPU"
            percent={host.cpu_percent ?? 0}
            detail={
              host.cpu_percent === undefined
                ? 'first sample — utilization needs two readings'
                : `${host.cpu_percent.toFixed(1)}% across all cores`
            }
          />
          {host.gpu_percent !== undefined && (
            <ResourceBar
              label="GPU"
              percent={host.gpu_percent}
              detail={`${host.gpu_percent.toFixed(1)}% — whole machine, from the accelerator driver (moves under Metal workloads like llama.cpp)`}
            />
          )}
          <ResourceBar
            label="Storage"
            percent={host.storage_used_percent}
            detail={`${host.storage_free_gb.toFixed(0)} GB free of ${host.storage_total_gb.toFixed(0)} GB`}
          />
          <div className="note-box" style={{ marginTop: 'var(--spacing-md)' }}>
            {metrics?.host_note}
          </div>
        </div>
      )}

      {totals && (
        <>
          <h2>Mesh totals</h2>
          <div className="grid">
            <div className="metric-card">
              <div className="metric-label">Devices</div>
              <div className="metric-value">
                {totals.devices_alive}/{totals.devices}
              </div>
              <div className="metric-sub">alive / in mesh</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Layers assigned</div>
              <div className="metric-value">{totals.layers_assigned}</div>
              <div className="metric-sub">across all shards</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Mesh traffic</div>
              <div className="metric-value">{fmtBps(totals.net_bytes_per_sec)}</div>
              <div className="metric-sub">live, both directions, measured on the wire</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Requests served</div>
              <div className="metric-value">{totals.requests_served.toLocaleString()}</div>
              <div className="metric-sub">shard computations since startup</div>
            </div>
          </div>
        </>
      )}

      <h2>Mesh peers</h2>
      {metrics && !metrics.allocation_supported && (
        <div className="note-box">{metrics.allocation_note}</div>
      )}
      {metrics?.peers.map((peer) => (
        <PeerCard
          key={peer.id}
          peer={peer}
          allocationSupported={metrics.allocation_supported}
          share={draft[peer.id] ?? peer.compute_share}
          busy={busyPeer !== null}
          resharding={busyPeer === peer.id}
          onDraft={(value) =>
            setDraft((current) => ({ ...current, [peer.id]: value }))
          }
          onCommit={() => {
            const pending = draftRef.current[peer.id];
            if (pending !== undefined) commitShare(peer.id, pending);
          }}
        />
      ))}

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

/** One device's full card: identity, computing, network, resources, share. */
function PeerCard({
  peer,
  allocationSupported,
  share,
  busy,
  resharding,
  onDraft,
  onCommit,
}: {
  peer: PeerMetric;
  allocationSupported: boolean;
  share: number;
  busy: boolean;
  resharding: boolean;
  onDraft: (share: number) => void;
  onCommit: () => void;
}) {
  const resources = peer.resources;
  const remoteHardware = resources && !resources.same_host_as_coordinator;
  const hasNetwork =
    peer.net_in_bytes_per_sec !== undefined ||
    peer.wire_in_mb !== undefined;
  // iOS gethostname() says "localhost" — the peer's advertised name is
  // the useful label there.
  const resourcesLabel =
    resources && resources.hostname.startsWith('localhost')
      ? peer.name
      : resources?.hostname;

  if (!peer.alive) {
    return (
      <div className="card peer-card">
        <div className="peer-head">
          <div className="device-dot dead" />
          <div className="device-name">{peer.name}</div>
          <span className="badge modeled">dropped</span>
        </div>
        <div className="fact-empty">
          Dropped from the mesh (closed, backgrounded, or silent past the
          health timeout) — the mesh re-sharded around it. It rejoins
          automatically when it reappears on the network.
        </div>
      </div>
    );
  }

  return (
    <div className="card peer-card">
      <div className="peer-head">
        <div className={`device-dot ${peer.alive ? 'alive' : 'dead'}`} />
        <div className="device-name">{peer.name}</div>
        {peer.is_coordinator && <span className="badge modeled">coordinator</span>}
        {peer.computing && <span className="badge measured">computing</span>}
      </div>

      <div className="peer-sections">
        <div className="peer-section">
          <div className="peer-section-title">Computing</div>
          <dl className="fact-list">
            <div>
              <dt>Assigned</dt>
              <dd>
                {peer.assigned}
                {peer.layer_span !== undefined && peer.layer_span > 0 && (
                  <span className="fact-sub"> ({peer.layer_span} layers)</span>
                )}
              </dd>
            </div>
            {peer.measured_ms_per_layer !== undefined && (
              <div>
                <dt>Per-layer</dt>
                <dd>{peer.measured_ms_per_layer.toFixed(2)} ms (measured)</dd>
              </div>
            )}
            {peer.last_compute_ms !== undefined && (
              <div>
                <dt>Last stage</dt>
                <dd>{peer.last_compute_ms.toFixed(2)} ms compute</dd>
              </div>
            )}
            {peer.requests_served !== undefined && (
              <div>
                <dt>Served</dt>
                <dd>
                  {peer.requests_served.toLocaleString()} requests
                  {peer.seconds_since_active !== undefined &&
                    peer.seconds_since_active < 60 && (
                      <span className="fact-sub">
                        {' '}
                        · active {peer.seconds_since_active.toFixed(1)} s ago
                      </span>
                    )}
                </dd>
              </div>
            )}
          </dl>
        </div>

        <div className="peer-section">
          <div className="peer-section-title">Network</div>
          {hasNetwork ? (
            <dl className="fact-list">
              <div>
                <dt>↓ to device</dt>
                <dd>
                  {fmtBps(peer.net_in_bytes_per_sec ?? 0)}
                  {peer.wire_in_mb !== undefined && (
                    <span className="fact-sub"> · {peer.wire_in_mb.toFixed(1)} MB total</span>
                  )}
                </dd>
              </div>
              <div>
                <dt>↑ from device</dt>
                <dd>
                  {fmtBps(peer.net_out_bytes_per_sec ?? 0)}
                  {peer.wire_out_mb !== undefined && (
                    <span className="fact-sub"> · {peer.wire_out_mb.toFixed(1)} MB total</span>
                  )}
                </dd>
              </div>
            </dl>
          ) : (
            <div className="fact-empty">{peer.link ?? 'no link data'}</div>
          )}
          {hasNetwork && peer.link && (
            <div className="fact-empty">{peer.link}</div>
          )}
        </div>
      </div>

      {remoteHardware && resources && (
        <div className="peer-resources">
          <div className="peer-section-title">
            Resources on {resourcesLabel}
            <span className="badge measured">measured on device</span>
          </div>
          <ResourceBar
            label="RAM"
            percent={resources.ram_used_percent}
            detail={`${(resources.ram_used_mb / 1024).toFixed(1)} / ${(
              resources.ram_total_mb / 1024
            ).toFixed(0)} GB used`}
          />
          <ResourceBar
            label="Mesh process"
            percent={(resources.process_footprint_mb / resources.ram_total_mb) * 100}
            detail={`${resources.process_footprint_mb} MB footprint`}
          />
          {resources.cpu_percent !== undefined && (
            <ResourceBar
              label="CPU"
              percent={resources.cpu_percent}
              detail={`${resources.cpu_percent.toFixed(1)}% across all cores`}
            />
          )}
          {resources.gpu_percent !== undefined && (
            <ResourceBar
              label="GPU"
              percent={resources.gpu_percent}
              detail={`${resources.gpu_percent.toFixed(1)}% — whole device`}
            />
          )}
          <ResourceBar
            label="Storage"
            percent={resources.storage_used_percent}
            detail={`${resources.storage_free_gb.toFixed(0)} GB free of ${resources.storage_total_gb.toFixed(0)} GB`}
          />
        </div>
      )}
      {resources && resources.same_host_as_coordinator && !peer.is_coordinator && (
        <div className="fact-empty">
          In-process peer — it reported {resources.hostname}, the same
          machine as the coordinator, so its hardware is the host card
          above. Its own numbers here would be duplicates, not proof.
          Connect a physical device (<code>swift run nmp-peer</code> on
          another Mac, or the iPhone app) to see real per-device bars.
        </div>
      )}
      {peer.is_coordinator && (
        <div className="fact-empty">
          This is the host machine — its live hardware is the card at the top.
        </div>
      )}

      {allocationSupported && (
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
            disabled={busy}
            onChange={(event) => onDraft(Number(event.target.value) / 100)}
            onPointerUp={onCommit}
            onKeyUp={onCommit}
          />
          {resharding && <span>re-sharding…</span>}
        </div>
      )}
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

/** 1234 → "1.2 KB/s"; 0 stays honest ("0 B/s", not blank). */
function fmtBps(bytesPerSecond: number): string {
  if (bytesPerSecond >= 1_048_576)
    return `${(bytesPerSecond / 1_048_576).toFixed(1)} MB/s`;
  if (bytesPerSecond >= 1024) return `${(bytesPerSecond / 1024).toFixed(1)} KB/s`;
  return `${bytesPerSecond} B/s`;
}

/** "Mozilla/5.0 (iPhone; …" → the device-ish part a human wants. */
function shortAgent(userAgent: string): string {
  const match = userAgent.match(/\(([^)]+)\)/);
  const inner = match ? match[1].split(';')[0].trim() : userAgent;
  return inner.slice(0, 40) || 'unknown browser';
}
