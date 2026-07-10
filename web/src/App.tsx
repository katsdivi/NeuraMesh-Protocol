import { useState } from 'react';
import { Navbar, type View } from './components/Navbar';
import { Dashboard } from './components/Dashboard';
import { Inference } from './components/Inference';
import { Benchmarking } from './components/Benchmarking';
import { Compare } from './components/Compare';
import { Settings } from './components/Settings';
import { useMesh } from './hooks/useMesh';

export function App() {
  const [view, setView] = useState<View>('dashboard');
  const { health, devices, events, reachable, sendControl } = useMesh();

  return (
    <div className="app">
      <Navbar view={view} onNavigate={setView} health={health} reachable={reachable} />
      <main className="page">
        {view === 'dashboard' && (
          <Dashboard
            health={health}
            devices={devices}
            events={events}
            reachable={reachable}
          />
        )}
        {view === 'run' && <Inference health={health} />}
        {view === 'benchmark' && <Benchmarking />}
        {view === 'compare' && <Compare />}
        {view === 'settings' && <Settings health={health} sendControl={sendControl} />}
      </main>
    </div>
  );
}
