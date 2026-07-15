import { useCallback, useEffect, useState } from 'react';
import { api, type ModelInfo } from '../api';

/**
 * Model picker. Lists every GGUF in the Mac's ~/models with honest
 * compatibility flags from the server:
 *   • recommended — the mesh's own best-fitting pick (green)
 *   • active      — the model serving right now
 *   • not usable  — incompatible architecture or too big for this host,
 *                   with the reason spelled out (never a dead-end)
 *
 * "Use this model" POSTs /api/models/select; the mesh relaunches onto it and
 * this page reconnects on its own (the global mesh-finder handles the gap).
 * Only offered by the sharded engine (--engine llamaShard); other engines
 * return 503 and we say so.
 */
export function Models() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [unavailable, setUnavailable] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [switching, setSwitching] = useState('');
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    try {
      const { models } = await api.models();
      setModels(models);
      setUnavailable(false);
    } catch {
      setUnavailable(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 4000);
    return () => clearInterval(timer);
  }, [refresh]);

  const use = async (model: ModelInfo) => {
    setBusy(model.path);
    setError('');
    try {
      const res = await api.selectModel(model.path);
      setSwitching(res.summary);
      // The server re-execs; polling will fail briefly, then the app's
      // mesh-finder takes over and snaps back when it's up on the new model.
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(null);
    }
  };

  if (loading) {
    return (
      <div>
        <h1>Models</h1>
        <div className="card"><p className="hint">Reading ~/models…</p></div>
      </div>
    );
  }

  if (unavailable) {
    return (
      <div>
        <h1>Models</h1>
        <div className="card">
          <p>
            Model switching is available in the <strong>sharded engine</strong>.
            Launch with <code>scripts/start.sh</code> (or
            <code> --engine llamaShard</code>) to pick and switch models here.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <h1>Models</h1>
      <p className="hint" style={{ marginTop: 'calc(-1 * var(--spacing-sm))' }}>
        Every GGUF in <code>~/models</code>. Pick any — the mesh restarts onto it
        and this page reconnects. Incompatible or too-big models are flagged, not
        hidden.
      </p>

      {switching && (
        <div className="card model-switching">
          <div className="spinner-dot" />
          <div>
            <strong>Switching model…</strong>
            <div className="hint">{switching}</div>
          </div>
        </div>
      )}
      {error && <div className="error-box" style={{ marginBottom: 'var(--spacing-md)' }}>{error}</div>}

      <div className="model-list">
        {models.map((m) => (
          <div
            key={m.path}
            className={`card model-card${m.active ? ' active' : ''}${m.usable ? '' : ' blocked'}`}
          >
            <div className="model-head">
              <div className="model-title">
                <span className="model-name">{m.name}</span>
                {m.recommended && <span className="badge recommended">recommended</span>}
                {m.active && <span className="badge modeled">active now</span>}
                {!m.compatible && <span className="badge standby">incompatible</span>}
                {m.compatible && !m.fits_host && <span className="badge standby">won’t fit</span>}
              </div>
              <button
                className="primary-button"
                disabled={!m.usable || m.active || busy !== null}
                onClick={() => use(m)}
                title={
                  !m.usable ? m.note : m.active ? 'Already running' : 'Restart the mesh onto this model'
                }
              >
                {m.active ? 'In use' : busy === m.path ? 'Switching…' : 'Use this model'}
              </button>
            </div>

            <div className="model-specs">
              <span>{m.arch}</span>
              <span>{(m.size_mb / 1024).toFixed(1)} GB</span>
              <span>{(m.params / 1e9).toFixed(m.params < 1e9 ? 2 : 1)} B params</span>
              <span>{m.layers} layers</span>
              <span>{m.bits_per_weight.toFixed(1)} bpw</span>
            </div>

            {m.note && <div className="model-note">{m.note}</div>}
          </div>
        ))}
        {models.length === 0 && (
          <div className="card">
            <p className="hint">
              No models in <code>~/models</code>. Download one:
              <code> scripts/setup_qwen14b.sh</code> (14B) — or the launcher grabs
              a small qwen automatically.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
