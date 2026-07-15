import { Component, useEffect, useRef, useState, type ReactNode } from 'react';
import { Navbar, type View } from './components/Navbar';
import { Dashboard } from './components/Dashboard';
import { Inference } from './components/Inference';
import { Chat } from './components/Chat';
import { Models } from './components/Models';
import { Devices } from './components/Devices';
import { Benchmarking } from './components/Benchmarking';
import { Compare } from './components/Compare';
import { Pressure } from './components/Pressure';
import { Settings } from './components/Settings';
import { useMesh } from './hooks/useMesh';

export function App() {
  const [view, setView] = useState<View>('dashboard');
  const { health, devices, events, reachable, liveGeneration, sendControl } =
    useMesh();

  // PWA connection flow: "looking for your mesh" until the first /health
  // answers, a "connected" toast for 3 s when it does (also after any
  // outage), then the toast shrinks to a green dot in the top-right.
  // Reconnects are automatic — the hook never stops polling.
  //
  // The effect keys on `connected` ONLY: health is a fresh object every
  // 2 s poll, and having it in the deps re-ran the effect each poll —
  // whose cleanup cancelled the hide timer before it could fire, so the
  // toast never went away (the installed-PWA bug).
  const connected = reachable && health !== null;
  const [toast, setToast] = useState('');
  const [dot, setDot] = useState(false);
  const wasConnected = useRef(false);
  const healthRef = useRef(health);
  healthRef.current = health;
  useEffect(() => {
    if (connected && !wasConnected.current) {
      wasConnected.current = true;
      setToast(`Connected to ${healthRef.current?.hostname ?? 'mesh'}`);
      setDot(false);
      const timer = setTimeout(() => {
        setToast('');
        setDot(true);
      }, 3000);
      return () => clearTimeout(timer);
    }
    if (!connected && wasConnected.current) {
      wasConnected.current = false; // re-toast on recovery
      setDot(false);
    }
  }, [connected]);

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
      {!toast && dot && (
        <div
          className="connect-dot"
          title={`Connected to ${health?.hostname ?? 'mesh'}`}
          aria-label="connected"
        />
      )}
      <main className="page">
        <ViewBoundary view={view}>
        {view === 'dashboard' && (
          <Dashboard
            health={health}
            devices={devices}
            events={events}
            reachable={reachable}
          />
        )}
        {view === 'run' && <Inference health={health} live={liveGeneration} />}
        {view === 'chat' && <Chat health={health} live={liveGeneration} />}
        {view === 'models' && <Models />}
        {view === 'devices' && <Devices />}
        {view === 'benchmark' && <Benchmarking />}
        {view === 'compare' && <Compare />}
        {view === 'pressure' && <Pressure />}
        {view === 'settings' && <Settings health={health} sendControl={sendControl} />}
        </ViewBoundary>
      </main>
    </div>
  );
}

/**
 * Without a boundary, one tab throwing during render unmounts the WHOLE
 * app (React's default) — a blank page where even the tabs that work
 * look broken. Contain the blast radius to the current view; switching
 * tabs retries with a fresh subtree.
 */
class ViewBoundary extends Component<
  { view: View; children: ReactNode },
  { failed: string }
> {
  state = { failed: '' };

  static getDerivedStateFromError(error: unknown) {
    return { failed: error instanceof Error ? error.message : String(error) };
  }

  componentDidUpdate(previous: { view: View }) {
    if (previous.view !== this.props.view && this.state.failed) {
      this.setState({ failed: '' });
    }
  }

  render() {
    if (this.state.failed) {
      return (
        <div className="error-box">
          This tab hit a rendering error: {this.state.failed}. The rest of
          the app is unaffected — switch tabs to continue (coming back here
          retries).
        </div>
      );
    }
    return this.props.children;
  }
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
