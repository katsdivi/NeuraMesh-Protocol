import { useEffect, useRef, useState } from 'react';
import { api, openEventSocket, type Device, type MeshHealth } from '../api';

export interface MeshEvent {
  time: string;
  message: string;
}

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
      if (event.type === 'peer_update') {
        refresh();
      } else if (event.type === 'mesh_event' && typeof event.message === 'string') {
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
    sendControl: (message: Record<string, unknown>) =>
      socketRef.current?.send(message),
  };
}
