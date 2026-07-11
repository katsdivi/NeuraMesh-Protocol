import { useEffect, useRef, useState } from 'react';
import { Navbar, type View } from './components/Navbar';
import { Dashboard } from './components/Dashboard';
import { Inference } from './components/Inference';
import { Devices } from './components/Devices';
import { Benchmarking } from './components/Benchmarking';
import { Compare } from './components/Compare';
import { Settings } from './components/Settings';
import { useMesh } from './hooks/useMesh';

export function App() {
  const [view, setView] = useState<View>('dashboard');
  const { health, devices, events, reachable, liveGeneration, sendControl } =
    useMesh();

  // PWA connection flow: "looking for your mesh" until the first /health
  // answers, a brief "connected" toast when it does (also after any
  // outage), then the app. Reconnects are automatic — the hook never
  // stops polling.
  const connected = reachable && health !== null;
  const [toast, setToast] = useState('');
  const wasConnected = useRef(false);
  useEffect(() => {
    if (connected && !wasConnected.current) {
      wasConnected.current = true;
      setToast(`Connected to ${health!.hostname}`);
      const timer = setTimeout(() => setToast(''), 3500);
      return () => clearTimeout(timer);
    }
    if (!connected && wasConnected.current) {
      wasConnected.current = false; // re-toast on recovery
    }
  }, [connected, health]);

  if (!connected) {
    return (
      <MeshFinder
        lost={health !== null}
        hostname={health?.hostname}
      />
    );
  }

  return (
    <div className="app">
      <Navbar view={view} onNavigate={setView} health={health} reachable={reachable} />
      {toast && <div className="connect-toast">✓ {toast}</div>}
      <main className="page">
        {view === 'dashboard' && (
          <Dashboard
            health={health}
            devices={devices}
            events={events}
            reachable={reachable}
          />
        )}
        {view === 'run' && <Inference health={health} live={liveGeneration} />}
        {view === 'devices' && <Devices />}
        {view === 'benchmark' && <Benchmarking />}
        {view === 'compare' && <Compare />}
        {view === 'settings' && <Settings health={health} sendControl={sendControl} />}
      </main>
    </div>
  );
}

/**
 * Full-screen state while the mesh is unreachable. `lost` = we had it
 * and it went away (coordinator stopped / left the Wi-Fi) vs a fresh
 * open still searching. Either way the app keeps polling and snaps
 * back the moment the coordinator answers — no button to press.
 */
function MeshFinder({ lost, hostname }: { lost: boolean; hostname?: string }) {
  return (
    <div className="mesh-finder">
      <img src="/icon-192.png" alt="" className="finder-icon" />
      <h1>{lost ? 'Mesh connection lost' : 'Looking for your mesh…'}</h1>
      <div className="finder-pulse">
        <span /><span /><span />
      </div>
      <p>
        {lost
          ? `${hostname ?? 'The coordinator'} stopped answering — reconnecting automatically.`
          : 'Connecting to the coordinator this app was installed from.'}
      </p>
      <div className="finder-hint">
        Make sure the mesh is running on your Mac
        (<code>swift run nmp-dashboard --ui</code>) and this device is on
        the same Wi-Fi. This screen goes away by itself.
      </div>
    </div>
  );
}
