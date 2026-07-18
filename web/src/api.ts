// Typed client for the nmp-dashboard REST + WebSocket surface.
// Same-origin: the Swift server serves both this app and /api.

export interface MeshHealth {
  status: string;
  /** True only when the real engine + model are live. During a restart
   *  there is a window where status is "ok" but ready is false (the
   *  placeholder reference engine answers). Absent on older servers. */
  ready?: boolean;
  hostname: string;
  mesh: {
    engine: string;
    model: string;
    shard_count: number;
    wire_format: string;
    speculation_available: boolean;
    peers: number;
    peers_alive: number;
    web_clients: number;
  };
}

export interface WebClient {
  address: string;
  user_agent: string;
  websocket: boolean;
  seconds_since_seen: number;
}

export interface HostSample {
  hostname: string;
  ram_total_mb: number;
  ram_used_mb: number;
  ram_used_percent: number;
  process_footprint_mb: number;
  storage_total_gb: number;
  storage_free_gb: number;
  storage_used_percent: number;
  cpu_percent?: number;
  /** Whole-machine GPU utilization from the accelerator driver (macOS). */
  gpu_percent?: number;
}

/** Mesh 2.3: a peer's own kernel counters, reported over the mesh. */
export interface PeerResources {
  hostname: string;
  /** True = in-process peer (or same box): it shares the host card's
   * hardware, so per-device bars would be theater. */
  same_host_as_coordinator: boolean;
  age_seconds: number;
  ram_total_mb: number;
  ram_used_mb: number;
  ram_used_percent: number;
  process_footprint_mb: number;
  storage_total_gb: number;
  storage_free_gb: number;
  storage_used_percent: number;
  cpu_percent?: number;
  gpu_percent?: number;
}

export interface PeerMetric {
  id: string;
  name: string;
  alive: boolean;
  assigned: string;
  layer_span?: number;
  compute_share: number;
  computing: boolean;
  load_percent?: number;
  measured_ms_per_layer?: number;
  // Mesh 2.3 per-device telemetry (all measured)
  is_coordinator?: boolean;
  link?: string;
  /** Mesh 2.8: this device holds 0 layers under the current plan. */
  excluded?: boolean;
  /** Mesh 2.8: specific reason it holds 0 layers (capacity / speed). */
  exclusion_reason?: string;
  requests_served?: number;
  last_compute_ms?: number;
  seconds_since_active?: number;
  /** Bytes/sec into the device (coordinator → peer), live. */
  net_in_bytes_per_sec?: number;
  /** Bytes/sec out of the device (peer → coordinator), live. */
  net_out_bytes_per_sec?: number;
  /** Total MB that crossed this link since the mesh came up. */
  wire_in_mb?: number;
  wire_out_mb?: number;
  resources?: PeerResources;
}

export interface MeshTotals {
  devices: number;
  devices_alive: number;
  layers_assigned: number;
  /** Absent in shard mode (the real-shard coordinator doesn't track these
   * mesh-wide yet) — the UI hides the cards instead of showing zeros. */
  requests_served?: number;
  net_bytes_per_sec?: number;
  generation_in_flight?: boolean;
}

export interface ShardingObjective {
  value: string;
  label: string;
}

/** One device's footprint under a candidate shard plan. */
export interface PlanDevice {
  id: string;
  name: string;
  layers: number;
  footprint_mb: number;
  ram_mb: number;
  /** Footprint as a % of the device's RAM (how full it'd get). */
  percent: number;
  is_coordinator: boolean;
  excluded: boolean;
}

/** A candidate shard plan (speed / balanced / capacity) to preview. */
export interface PlanCandidate {
  strategy: string;
  label: string;
  note: string;
  /** False when the model can't fit even across the whole mesh. */
  fits: boolean;
  capacity_shortfall: number;
  /** The fullest any single device gets under this plan. */
  max_device_percent: number;
  devices: PlanDevice[];
}

export interface ShardPlans {
  current_strategy: string;
  plans: PlanCandidate[];
}

export interface DeviceMetrics {
  host: HostSample;
  host_note: string;
  generation_in_flight: boolean;
  /** Deprecated mode indicator (true = manual sliders drive the split).
   *  Kept for older servers; prefer manual_mode. */
  allocation_supported: boolean;
  /** True = manual mode (operator compute-share sliders drive the split).
   *  Same meaning as the deprecated allocation_supported. */
  manual_mode?: boolean;
  allocation_note: string;
  /** Shard mode: true = layers auto-balance by measured speed + capacity;
   *  false = manual (the operator's compute-share sliders drive the split). */
  auto_balance?: boolean;
  peers: PeerMetric[];
  totals?: MeshTotals;
  // Mesh 2.8: live layer-distribution strategy.
  sharding_objective?: string;
  sharding_objective_label?: string;
  sharding_objectives?: ShardingObjective[];
  capacity_shortfall?: number;
  capacity_note?: string;
}

export interface RaceLeg {
  name: string;
  transport: string;
  measured: boolean;
  handshake_ms: number;
  transfer_ms: number;
  per_trip_ms: number;
  total_ms: number;
  round_trips: number;
  bytes_moved: number;
}

export interface RaceProjection {
  name: string;
  total_ms: number;
  tokens_per_sec: number;
  basis: string;
}

export interface TransportRace {
  race: { note: string; legs: RaceLeg[] };
  projected: RaceProjection[];
}

export interface ComparisonRun {
  note: string;
  generation: {
    output: string;
    token_count: number;
    latency_ms: number;
    tokens_per_sec: number;
    network_payload_bytes: number;
    round_trips: number;
    engine: string;
  };
  race: { note: string; legs: RaceLeg[] };
  projected: RaceProjection[];
}

export interface Device {
  id: string;
  name: string;
  latency_ms: number;
  load_percent: number;
  assigned: string;
  alive: boolean;
}

export interface SpeculationStats {
  drafter: string;
  mesh_round_trips: number;
  drafted_tokens: number;
  accepted_draft_tokens: number;
  fallback_rounds: number;
  acceptance_rate: number;
  tokens_per_round_trip: number;
}

export interface ProtocolEstimate {
  name: string;
  measured: boolean;
  handshake_ms: number;
  per_trip_overhead_ms: number;
  loss_recovery_ms: number;
  total_ms: number;
  tokens_per_sec: number;
  assumptions: string;
}

export interface InferenceResult {
  output: string;
  token_count: number;
  latency_ms: number;
  tokens_per_sec: number;
  network_payload_bytes: number;
  shard_count: number;
  engine: string;
  round_trips: number;
  wire_format: string;
  speculation?: SpeculationStats;
  /** Present when the server clamped the requested max_tokens — the
   *  value the generation actually ran with. */
  max_tokens_effective?: number;
  /** Mesh 2.5: "Compare protocols" runs the MEASURED transport race on
   *  the generation's real traffic pattern — no modeled numbers here. */
  transport_race?: TransportRace;
  transport_race_error?: string;
}

export interface BenchmarkRun {
  run: number;
  tokens_per_sec: number;
  latency_ms: number;
  token_count: number;
  payload_bytes: number;
}

export interface BenchmarkResults {
  prompt: string;
  avg_tokens_per_sec: number;
  avg_latency_ms: number;
  stddev_latency_ms: number;
  /** Echo of the requested run count — differs from runs.length when the
   *  server clamped it. Absent on older servers. */
  runs_requested?: number;
  runs: BenchmarkRun[];
}

/** Mesh 2.7: one chat turn, as POST /api/chat consumes it. */
export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/** A saved conversation's sidebar row (no message bodies). */
export interface ChatSummary {
  id: string;
  title: string;
  /** The device that owns this conversation (its LAN hostname). */
  device: string;
  model: string;
  created_at: number;
  updated_at: number;
  message_count: number;
  /** The on-disk copy has been squeezed to LZFSE (quiet session). */
  compressed: boolean;
}

/** A saved conversation with its full transcript. */
export interface SavedConversation extends ChatSummary {
  messages: ChatMessage[];
}

/** An installed model in ~/models, with the flags the picker needs. */
export interface ModelInfo {
  path: string;
  name: string;
  arch: string;
  size_mb: number;
  params: number;
  layers: number;
  bits_per_weight: number;
  /** The shard shim runs this architecture (qwen2/qwen3 only, today). */
  compatible: boolean;
  /** This host has the RAM to hold it. */
  fits_host: boolean;
  /** compatible AND fits_host — safe to select. */
  usable: boolean;
  /** The highest-quality usable model (the mesh's own pick). */
  recommended: boolean;
  /** The model the mesh is serving right now. */
  active: boolean;
  /** Empty when usable; otherwise why it can't run here. */
  note: string;
}

/** A non-2xx API reply, carrying the HTTP status so callers can treat
 *  expected states (429 busy, 409 nothing-to-race) as flow, not faults. */
export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new ApiError(
      response.status,
      data.error ?? `${response.status} ${response.statusText}`,
    );
  }
  return data as T;
}

export const api = {
  health: (): Promise<MeshHealth> => fetch('/health').then((r) => r.json()),

  devices: (): Promise<Device[]> => fetch('/api/devices').then((r) => r.json()),

  inference: (request: {
    prompt: string;
    max_tokens: number;
    enable_speculation?: boolean;
    enable_comparison?: boolean;
  }): Promise<InferenceResult> => post('/api/inference', request),

  /** Mesh 2.7: chat — the server folds the transcript into the engine's
   *  template and runs the same mesh generation as /api/inference. */
  chat: (request: {
    messages: ChatMessage[];
    max_tokens: number;
    enable_speculation?: boolean;
  }): Promise<InferenceResult> => post('/api/chat', request),

  /** Saved conversations on THIS device, newest first. Empty when history
   *  is disabled server-side. */
  listChats: (): Promise<ChatSummary[]> =>
    fetch('/api/chats').then((r) => (r.ok ? r.json() : [])),

  /** One saved conversation with its full transcript. */
  loadChat: (id: string): Promise<SavedConversation> =>
    fetch(`/api/chats/${id}`).then((r) => {
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    }),

  /** Create (blank id) or replace a conversation; returns the stored row
   *  (with the generated id + derived title for a new chat). */
  saveChat: (request: {
    id?: string;
    title?: string;
    model?: string;
    messages: ChatMessage[];
  }): Promise<ChatSummary> => post('/api/chats', request),

  /** Delete a saved conversation. */
  deleteChat: (id: string): Promise<{ deleted: boolean }> =>
    post(`/api/chats/${id}/delete`, {}),

  benchmark: (request: {
    prompt: string;
    max_tokens: number;
    runs: number;
  }): Promise<BenchmarkResults> => post('/api/benchmark/run', request),

  comparison: (request: {
    tokens: number;
    payload_bytes: number;
    round_trips: number;
    measured_total_ms: number;
    lan_rtt_ms?: number;
    loss_rate?: number;
  }): Promise<{ note: string; protocols: ProtocolEstimate[] }> =>
    post('/api/comparison', request),

  /** Mesh 2.1: real generation + measured NMP-vs-TCP transport race. */
  comparisonRun: (request: {
    prompt: string;
    max_tokens: number;
    enable_speculation?: boolean;
  }): Promise<ComparisonRun> => post('/api/comparison/run', request),

  /** Mesh 2.1: live host + per-peer resource metrics. */
  deviceMetrics: (): Promise<DeviceMetrics> =>
    fetch('/api/devices/metrics').then((r) => {
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    }),

  /** Mesh 2.1: browsers currently looking at the dashboard. */
  clients: (): Promise<WebClient[]> =>
    fetch('/api/clients').then((r) => r.json()),

  /** Mesh 2.1: set a peer's mesh compute share (re-shards the mesh).
   *  New servers echo share_requested + the resulting assignment and
   *  404 unknown device ids. */
  allocate: (
    peerId: string,
    share: number,
  ): Promise<{
    status: string;
    summary: string;
    share_requested?: number;
    assigned?: string;
  }> => post(`/api/devices/${peerId}/allocate`, { share }),

  /** Auto (balance by measured speed + capacity) vs manual (operator
   *  compute-share sliders). Switching re-shards the mesh. */
  setAutoBalance: (
    enabled: boolean,
  ): Promise<{ status: string; auto_balance: boolean; summary: string }> =>
    post('/api/mesh/autobalance', { enabled }),

  /** Preview the candidate shard plans (speed / balanced / capacity) with
   *  per-device footprints, before committing one. */
  meshPlans: (): Promise<ShardPlans> =>
    fetch('/api/mesh/plans').then((r) => {
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    }),

  /** Apply a previewed plan (re-shards the mesh). */
  applyStrategy: (
    strategy: string,
  ): Promise<{ status: string; strategy: string; summary: string }> =>
    post('/api/mesh/strategy', { strategy }),

  /** Mesh 2.8: switch the sharding objective (re-shards the mesh). */
  setObjective: (
    objective: string,
  ): Promise<{ status: string; objective: string; summary: string }> =>
    post('/api/mesh/objective', { objective }),

  /** The installed models with compatibility flags (sharded engine only). */
  models: (): Promise<{ models: ModelInfo[] }> =>
    fetch('/api/models').then((r) => {
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    }),

  /** Switch the active model — the mesh relaunches onto it and the page
   *  reconnects. Rejects (throws) for incompatible / too-big / missing. */
  selectModel: (
    path: string,
  ): Promise<{ status: string; summary: string; reconnecting: boolean }> =>
    post('/api/models/select', { path }),
};

/** Live event stream (the Phase 6 dashboard WebSocket). */
export function openEventSocket(
  onEvent: (event: Record<string, unknown>) => void,
): { send: (message: Record<string, unknown>) => void; close: () => void } {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${scheme}://${location.host}/ws`);
  socket.onmessage = (message) => {
    try {
      onEvent(JSON.parse(message.data));
    } catch {
      /* non-JSON frame — ignore */
    }
  };
  return {
    send: (message) => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(message));
      }
    },
    close: () => socket.close(),
  };
}
