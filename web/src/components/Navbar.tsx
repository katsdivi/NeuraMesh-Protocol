import type { MeshHealth } from '../api';

export type View = 'dashboard' | 'run' | 'benchmark' | 'compare' | 'settings';

const TABS: { id: View; label: string }[] = [
  { id: 'dashboard', label: 'Mesh' },
  { id: 'run', label: 'Run' },
  { id: 'benchmark', label: 'Benchmark' },
  { id: 'compare', label: 'Compare' },
  { id: 'settings', label: 'Settings' },
];

export function Navbar({
  view,
  onNavigate,
  health,
  reachable,
}: {
  view: View;
  onNavigate: (view: View) => void;
  health: MeshHealth | null;
  reachable: boolean;
}) {
  return (
    <header className="navbar">
      <div className="brand">
        Neura<span>Mesh</span>
      </div>
      <nav>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={tab.id === view ? 'active' : ''}
            onClick={() => onNavigate(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
      <div className={`mesh-pill ${reachable ? 'online' : 'offline'}`}>
        {reachable
          ? `${health?.mesh.engine ?? '…'} · ${health?.mesh.peers_alive ?? 0} peer(s)`
          : 'mesh unreachable'}
      </div>
    </header>
  );
}
