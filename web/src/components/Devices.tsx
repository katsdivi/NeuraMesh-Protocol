import { useCallback, useEffect, useRef, useState } from 'react';
import {
  api,
  type DeviceMetrics,
  type PeerMetric,
  type ShardPlans,
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

  const [objectiveBusy, setObjectiveBusy] = useState(false);
  const switchObjective = async (objective: string) => {
    setObjectiveBusy(true);
    setError('');
    try {
      const response = await api.setObjective(objective);
      setLastAction(`Re-sharded: ${response.summary}`);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setObjectiveBusy(false);
    }
  };

  // Pre-shard plan preview: the 3 candidate splits + their footprints.
  const [plans, setPlans] = useState<ShardPlans | null>(null);
  const [plansBusy, setPlansBusy] = useState(false);
  const loadPlans = async () => {
    setPlansBusy(true);
    setError('');
    try {
      setPlans(await api.meshPlans());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPlansBusy(false);
    }
  };
  const applyPlan = async (strategy: string) => {
    setPlansBusy(true);
    setError('');
    try {
      const res = await api.applyStrategy(strategy);
      setLastAction(`Applied ${strategy} plan: ${res.summary}`);
      await Promise.all([refresh(), loadPlans()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPlansBusy(false);
    }
  };

  const [autoBusy, setAutoBusy] = useState(false);
  // Which mode the in-flight toggle POST is switching TO — a real change
  // re-shards the mesh and can take seconds (longer while the phone holds
  // layers), so the toggle needs a visible in-progress state.
  const [autoPending, setAutoPending] = useState<boolean | null>(null);
  const switchAutoBalance = async (enabled: boolean) => {
    setAutoBusy(true);
    setAutoPending(enabled);
    setError('');
    try {
      const response = await api.setAutoBalance(enabled);
      setLastAction(`${enabled ? 'Auto' : 'Manual'} balancing: ${response.summary}`);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setAutoBusy(false);
      setAutoPending(null);
    }
  };

  const commitShare = async (peerId: string, share: number) => {
    setBusyPeer(peerId);
    setError('');
    try {
      const response = await api.allocate(peerId, share);
      setLastAction(
        `Re-sharded${
          response.share_requested !== undefined
            ? ` (requested share ${Math.round(response.share_requested * 100)}%)`
            : ''
        }: ${response.summary}`,
      );
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
  // Contract: manual_mode is the mode flag going forward; the deprecated
  // allocation_supported carries the same meaning on older servers.
  const manualMode = metrics
    ? metrics.manual_mode ?? metrics.allocation_supported
    : false;

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
            {totals.net_bytes_per_sec !== undefined && (
              <div className="metric-card">
                <div className="metric-label">Mesh traffic</div>
                <div className="metric-value">{fmtBps(totals.net_bytes_per_sec)}</div>
                <div className="metric-sub">live, both directions, measured on the wire</div>
              </div>
            )}
            {totals.requests_served !== undefined && (
              <div className="metric-card">
                <div className="metric-label">Requests served</div>
                <div className="metric-value">{totals.requests_served.toLocaleString()}</div>
                <div className="metric-sub">shard computations since startup</div>
              </div>
            )}
          </div>
        </>
      )}

      {metrics?.sharding_objectives && metrics.sharding_objectives.length > 0 && (
        <div className="card objective-card">
          <div className="objective-head">
            <div>
              <div className="peer-section-title">Sharding strategy</div>
              <div className="objective-explain">
                How the coordinator splits the model's layers across the mesh.
              </div>
            </div>
            <div className="objective-toggle">
              {metrics.sharding_objectives.map((option) => (
                <button
                  key={option.value}
                  className={
                    option.value === metrics.sharding_objective ? 'active' : ''
                  }
                  disabled={objectiveBusy}
                  onClick={() => switchObjective(option.value)}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
          <div className="objective-explain">
            {metrics.sharding_objective === 'speed' ? (
              <>
                <strong>Pure Speed:</strong> pack the fastest device and route
                around slower ones — lowest latency. A device that would only
                add a Wi-Fi hop gets 0 shards (shown below with the reason).
              </>
            ) : (
              <>
                <strong>Capacity + Speed:</strong> spread across the whole mesh,
                balanced by measured speed, so every device pulls its weight and
                models too big for one device still run. Capacity is always a
                hard ceiling.
              </>
            )}
          </div>
          {objectiveBusy && <div className="objective-explain">re-sharding…</div>}
        </div>
      )}

      {metrics?.capacity_shortfall !== undefined &&
        metrics.capacity_shortfall > 0 && (
          <div className="error-box">{metrics.capacity_note}</div>
        )}

      <h2>Mesh peers</h2>
      {metrics && metrics.auto_balance !== undefined && (
        <div className="card objective-card">
          <div className="objective-head">
            <div>
              <strong>Layer balancing</strong>
              <div className="objective-explain">
                {metrics.auto_balance
                  ? 'Auto — layers split by each device’s measured speed & '
                    + 'capacity; re-shards on join/leave and as measurements '
                    + 'converge.'
                  : 'Manual — set each device’s compute share below. 0% '
                    + 'excludes a device (Mac-only, no per-token round trip).'}
              </div>
            </div>
            <div className="objective-toggle">
              <button
                className={`objective-option ${metrics.auto_balance ? 'active' : ''}`}
                disabled={autoBusy || metrics.auto_balance === true}
                onClick={() => switchAutoBalance(true)}
              >
                {autoPending === true ? 'Auto…' : 'Auto'}
              </button>
              <button
                className={`objective-option ${!metrics.auto_balance ? 'active' : ''}`}
                disabled={autoBusy || metrics.auto_balance === false}
                onClick={() => switchAutoBalance(false)}
              >
                {autoPending === false ? 'Manual…' : 'Manual'}
              </button>
            </div>
          </div>
          {autoBusy && (
            <div className="model-switching" style={{ marginTop: 'var(--spacing-sm)', marginBottom: 0 }}>
              <div className="spinner-dot" />
              <span className="objective-explain" style={{ marginTop: 0 }}>
                Switching to {autoPending ? 'auto' : 'manual'} — re-sharding
                the mesh. This can take a while when a phone holds layers;
                leave this tab open.
              </span>
            </div>
          )}
        </div>
      )}

      {metrics && metrics.auto_balance !== undefined && (
        <div className="card plan-preview">
          <div className="objective-head">
            <div>
              <strong>Sharding plan</strong>
              <div className="objective-explain">
                Compare how each strategy splits the model across your devices —
                and how full each one gets — then apply one.
              </div>
            </div>
            <button
              className="ghost-button"
              disabled={plansBusy}
              onClick={loadPlans}
            >
              {plans ? 'Refresh' : 'Preview plans'}
            </button>
          </div>
          {plans && (
            <div className="plan-grid">
              {plans.plans.map((plan) => {
                const isCurrent = plan.strategy === plans.current_strategy;
                return (
                  <div
                    key={plan.strategy}
                    className={`plan-option${isCurrent ? ' active' : ''}`}
                  >
                    <div className="plan-option-head">
                      <strong>{plan.label}</strong>
                      {isCurrent && <span className="badge">current</span>}
                    </div>
                    <div className="plan-note">{plan.note}</div>
                    <div className="plan-devices">
                      {plan.devices.map((d) => (
                        <div key={d.id} className="plan-device">
                          <div className="plan-device-top">
                            <span className="plan-device-name">
                              {d.is_coordinator ? 'This Mac' : d.name}
                              {d.excluded && (
                                <span className="plan-excl"> · idle</span>
                              )}
                            </span>
                            <span className="plan-device-layers">
                              {d.layers} layer{d.layers === 1 ? '' : 's'}
                              {d.footprint_mb > 0 && ` · ${d.footprint_mb} MB`}
                            </span>
                          </div>
                          <div className="plan-bar">
                            <div
                              className={`plan-bar-fill${
                                d.percent >= 85 ? ' hot' : ''
                              }`}
                              style={{ width: `${Math.min(100, d.percent)}%` }}
                            />
                          </div>
                          <div className="plan-device-pct">
                            {d.ram_mb > 0
                              ? `${d.percent}% of ${(d.ram_mb / 1024).toFixed(0)} GB RAM`
                              : 'footprint unknown'}
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="plan-foot">
                      {plan.fits ? (
                        <span className="plan-peak">
                          peak device {plan.max_device_percent}% full
                        </span>
                      ) : (
                        <span className="plan-peak hot">
                          ✗ won’t fit ({plan.capacity_shortfall} layers over)
                        </span>
                      )}
                      <button
                        className="primary-button"
                        disabled={plansBusy || isCurrent || !plan.fits}
                        onClick={() => applyPlan(plan.strategy)}
                      >
                        {isCurrent ? 'Applied' : 'Apply'}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
      {metrics && !manualMode
        && metrics.auto_balance === undefined && (
        <div className="note-box">{metrics.allocation_note}</div>
      )}
      {metrics?.peers.map((peer) => (
        <PeerCard
          key={peer.id}
          peer={peer}
          allocationSupported={manualMode}
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
        {peer.excluded && <span className="badge standby">0 shards · standby</span>}
      </div>

      {peer.excluded && peer.exclusion_reason && (
        <div className="standby-reason">
          <strong>0 shards on this device.</strong> {peer.exclusion_reason}
        </div>
      )}

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
            min={0}
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
