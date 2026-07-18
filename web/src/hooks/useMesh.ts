import { useEffect, useRef, useState } from 'react';
import { api, openEventSocket, type Device, type MeshHealth } from '../api';

export interface MeshEvent {
  time: string;
  message: string;
}

/**
 * Mesh 2.1: the live generation every browser watches together. Folded
 * from generation_* WebSocket events, so a phone sees the tokens of a
 * run a laptop submitted (and vice versa) as they are produced.
 */
export interface LiveGeneration {
  status: 'idle' | 'running' | 'done' | 'failed';
  /** Who kicked this generation off: "inference" | "chat" | "benchmark" |
   *  "comparison". Lets tabs label or filter generations that aren't
   *  theirs instead of interleaving them. Absent on older servers. */
  source?: string;
  prompt?: string;
  speculative?: boolean;
  requested?: number;
  tokens: string[];
  result?: {
    output: string;
    token_count: number;
    latency_ms: number;
    tokens_per_sec: number;
    network_payload_bytes: number;
    round_trips: number;
    engine: string;
    acceptance_rate?: number;
  };
  error?: string;
}

const IDLE: LiveGeneration = { status: 'idle', tokens: [] };

/**
 * Live mesh state: /health + /api/devices polled every 3 s, refreshed
 * instantly by WebSocket peer_update / mesh_event pushes. Every open
 * browser tab (phone, laptop, …) converges on the same state.
 */
export function useMesh() {
  const [health, setHealth] = useState<MeshHealth | null>(null);
  const [devices, setDevices] = useState<Device[]>([]);
  const [events, setEvents] = useState<MeshEvent[]>([]);
  const [reachable, setReachable] = useState(true);
  const [liveGeneration, setLiveGeneration] = useState<LiveGeneration>(IDLE);
  const socketRef = useRef<ReturnType<typeof openEventSocket> | null>(null);

  useEffect(() => {
    let cancelled = false;

    const refresh = async () => {
      try {
        const [nextHealth, nextDevices] = await Promise.all([
          api.health(),
          api.devices(),
        ]);
        if (cancelled) return;
        setHealth(nextHealth);
        setDevices(nextDevices);
        setReachable(true);
      } catch {
        if (!cancelled) setReachable(false);
      }
    };

    refresh();
    const timer = setInterval(refresh, 3000);

    const socket = openEventSocket((event) => {
      switch (event.type) {
        case 'peer_update':
        case 'client_update':
          refresh();
          break;
        case 'mesh_event':
          if (typeof event.message === 'string') {
            setEvents((log) =>
              [
                {
                  time: new Date().toLocaleTimeString(),
                  message: event.message as string,
                },
                ...log,
              ].slice(0, 50),
            );
          }
          break;
        case 'generation_started':
          setLiveGeneration({
            status: 'running',
            source:
              typeof event.source === 'string' ? event.source : undefined,
            prompt: String(event.prompt ?? ''),
            speculative: Boolean(event.speculative),
            requested: Number(event.max_tokens ?? 0),
            tokens: [],
          });
          break;
        case 'generation_token':
          setLiveGeneration((current) => ({
            ...current,
            status: 'running',
            source:
              current.source ??
              (typeof event.source === 'string' ? event.source : undefined),
            requested: Number(event.requested ?? current.requested ?? 0),
            tokens: [...current.tokens, String(event.text ?? '')],
          }));
          break;
        case 'generation_complete':
          setLiveGeneration((current) => ({
            ...current,
            status: 'done',
            result: {
              output: String(event.output ?? ''),
              token_count: Number(event.token_count ?? 0),
              latency_ms: Number(event.latency_ms ?? 0),
              tokens_per_sec: Number(event.tokens_per_sec ?? 0),
              network_payload_bytes: Number(event.network_payload_bytes ?? 0),
              round_trips: Number(event.round_trips ?? 0),
              engine: String(event.engine ?? ''),
              acceptance_rate:
                event.acceptance_rate === undefined
                  ? undefined
                  : Number(event.acceptance_rate),
            },
          }));
          break;
        case 'generation_failed':
          setLiveGeneration((current) => ({
            ...current,
            status: 'failed',
            error: String(event.error ?? 'generation failed'),
          }));
          break;
        default:
          break;
      }
    });
    socketRef.current = socket;

    return () => {
      cancelled = true;
      clearInterval(timer);
      socket.close();
    };
  }, []);

  return {
    health,
    devices,
    events,
    reachable,
    liveGeneration,
    sendControl: (message: Record<string, unknown>) =>
      socketRef.current?.send(message),
  };
}
