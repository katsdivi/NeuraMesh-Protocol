// Typed client for the nmp-dashboard REST + WebSocket surface.
// Same-origin: the Swift server serves both this app and /api.

export interface MeshHealth {
  status: string;
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
}

export interface DeviceMetrics {
  host: HostSample;
  host_note: string;
  generation_in_flight: boolean;
  allocation_supported: boolean;
  allocation_note: string;
  peers: PeerMetric[];
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
  projected: {
    name: string;
    total_ms: number;
    tokens_per_sec: number;
    basis: string;
  }[];
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
  protocol_comparison?: { note: string; protocols: ProtocolEstimate[] };
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
  runs: BenchmarkRun[];
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error ?? `${response.status} ${response.statusText}`);
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

  /** Mesh 2.1: set a peer's mesh compute share (re-shards the mesh). */
  allocate: (
    peerId: string,
    share: number,
  ): Promise<{ status: string; summary: string }> =>
    post(`/api/devices/${peerId}/allocate`, { share }),
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
