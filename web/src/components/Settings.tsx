import { useState } from 'react';
import type { MeshHealth } from '../api';

export function Settings({
  health,
  sendControl,
}: {
  health: MeshHealth | null;
  sendControl: (message: Record<string, unknown>) => void;
}) {
  const [lossPercent, setLossPercent] = useState(0);
  const mesh = health?.mesh;

  return (
    <div>
      <h1>Settings</h1>

      <div className="card">
        <h3>Mesh</h3>
        <div className="settings-row">
          <div>
            <div className="label">Engine</div>
            <div className="hint">set at launch via --engine</div>
          </div>
          <code>{mesh?.engine ?? '—'}</code>
        </div>
        <div className="settings-row">
          <div>
            <div className="label">Model</div>
          </div>
          <code>{mesh?.model || '—'}</code>
        </div>
        <div className="settings-row">
          <div>
            <div className="label">Wire format</div>
            <div className="hint">zeroTrimmed/mixedPrecision via --auto-config</div>
          </div>
          <code>{mesh?.wire_format ?? '—'}</code>
        </div>
        <div className="settings-row">
          <div>
            <div className="label">Speculative decoding</div>
            <div className="hint">needs a Phase 9 shim; drafter via --draft-model</div>
          </div>
          <code>{mesh?.speculation_available ? 'available' : 'unavailable'}</code>
        </div>
      </div>

      <div className="card" style={{ marginTop: 'var(--spacing-lg)' }}>
        <h3>Chaos</h3>
        <div className="settings-row">
          <div>
            <div className="label">Injected packet loss: {lossPercent}%</div>
            <div className="hint">
              exercises FEC + NACK recovery under real inference (in-process
              links only)
            </div>
          </div>
          <input
            type="range"
            min={0}
            max={15}
            step={1}
            value={lossPercent}
            onChange={(event) => {
              const next = Number(event.target.value);
              setLossPercent(next);
              sendControl({ type: 'set_loss_rate', rate: next / 100 });
            }}
          />
        </div>
      </div>

      <div className="note-box" style={{ marginTop: 'var(--spacing-lg)' }}>
        This UI is served by the mesh coordinator on your local network with
        no TLS or authentication — every device on this Wi-Fi shares the same
        live state. Don't port-forward it.
      </div>
    </div>
  );
}
