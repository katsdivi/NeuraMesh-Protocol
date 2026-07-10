import { useState } from 'react';
import { api, type BenchmarkResults } from '../api';

export function Benchmarking() {
  const [prompt, setPrompt] = useState('The future of AI is');
  const [maxTokens, setMaxTokens] = useState(32);
  const [runs, setRuns] = useState(3);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<BenchmarkResults | null>(null);
  const [error, setError] = useState('');

  const start = async () => {
    setRunning(true);
    setError('');
    try {
      setResults(await api.benchmark({ prompt, max_tokens: maxTokens, runs }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div>
      <h1>Benchmark Your Mesh</h1>

      <div className="card">
        <div className="form-group">
          <label>Prompt</label>
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            rows={2}
          />
        </div>
        <div className="form-row">
          <div className="form-group">
            <label>Max tokens</label>
            <input
              type="number"
              value={maxTokens}
              min={1}
              max={128}
              onChange={(event) =>
                setMaxTokens(Math.max(1, Math.min(128, Number(event.target.value) || 1)))
              }
            />
          </div>
          <div className="form-group">
            <label>Runs</label>
            <input
              type="number"
              value={runs}
              min={1}
              max={10}
              onChange={(event) =>
                setRuns(Math.max(1, Math.min(10, Number(event.target.value) || 1)))
              }
            />
          </div>
        </div>
        <button
          className="primary-button"
          disabled={running || !prompt.trim()}
          onClick={start}
        >
          {running ? `Benchmarking (${runs} runs)…` : 'Start Benchmark'}
        </button>
        {error && <div className="error-box">{error}</div>}
      </div>

      {results && (
        <div style={{ marginTop: 'var(--spacing-lg)' }}>
          <div className="grid">
            <div className="metric-card">
              <div className="metric-label">Avg Throughput</div>
              <div className="metric-value">{results.avg_tokens_per_sec.toFixed(1)}</div>
              <div className="metric-sub">tokens / sec</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Avg Latency</div>
              <div className="metric-value">{results.avg_latency_ms.toFixed(0)} ms</div>
              <div className="metric-sub">per generation</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Latency σ</div>
              <div className="metric-value">{results.stddev_latency_ms.toFixed(1)} ms</div>
              <div className="metric-sub">run-to-run spread</div>
            </div>
          </div>

          <div className="card">
            <h3>Runs</h3>
            {results.runs.map((run) => (
              <div key={run.run} className="run-item">
                <span>run {run.run}</span>
                <span>{run.tokens_per_sec.toFixed(1)} tok/s</span>
                <span>{run.latency_ms.toFixed(0)} ms</span>
                <span>{run.token_count} tokens</span>
                <span>{(run.payload_bytes / 1024).toFixed(1)} KB</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
