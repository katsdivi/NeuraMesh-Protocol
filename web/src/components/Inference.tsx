import { useState } from 'react';
import { useInference } from '../hooks/useInference';
import { ProtocolComparison } from './ProtocolComparison';
import type { MeshHealth } from '../api';
import type { LiveGeneration } from '../hooks/useMesh';

export function Inference({
  health,
  live,
}: {
  health: MeshHealth | null;
  live: LiveGeneration;
}) {
  const { submit, result, loading, error } = useInference();
  const [prompt, setPrompt] = useState('The future of AI is');
  const [maxTokens, setMaxTokens] = useState(32);
  const [speculation, setSpeculation] = useState(false);
  const [comparison, setComparison] = useState(true);

  const speculationAvailable = health?.mesh.speculation_available ?? false;
  // Reference-engine tokens are bare words; llama pieces carry their own
  // leading spaces.
  const joiner = health?.mesh.engine === 'reference' ? ' ' : '';
  const streaming = live.status === 'running';
  // A finished run this tab did NOT submit still deserves its numbers.
  const spectatorResult = !result && !loading && live.status === 'done'
    ? live.result : undefined;

  return (
    <div>
      <h1>Run Inference</h1>

      <div className="card">
        <div className="form-group">
          <label>Prompt</label>
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Enter your prompt…"
            rows={3}
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
          <div className="form-group checkbox-group">
            <input
              id="comparison"
              type="checkbox"
              checked={comparison}
              onChange={(event) => setComparison(event.target.checked)}
            />
            <label htmlFor="comparison" style={{ textTransform: 'none', margin: 0 }}>
              Compare protocols
            </label>
          </div>
          <div className="form-group checkbox-group">
            <input
              id="speculation"
              type="checkbox"
              checked={speculation}
              disabled={!speculationAvailable}
              onChange={(event) => setSpeculation(event.target.checked)}
            />
            <label htmlFor="speculation" style={{ textTransform: 'none', margin: 0 }}>
              Speculative decoding
              {!speculationAvailable &&
                (health?.mesh.engine === 'llamaCpp'
                  ? ' (needs the Phase 9 shim — rerun scripts/setup_llama.sh)'
                  : ' (llama only — the reference engine has no drafts to verify)')}
            </label>
          </div>
        </div>

        <button
          className="primary-button"
          disabled={loading || !prompt.trim()}
          onClick={() =>
            submit({
              prompt,
              max_tokens: maxTokens,
              enable_speculation: speculation,
              enable_comparison: comparison,
            })
          }
        >
          {loading ? 'Running…' : 'Run Inference'}
        </button>

        {error && <div className="error-box">{error}</div>}
      </div>

      {streaming && (
        <div>
          <div className="note-box">
            Live — streaming to every open browser
            {live.prompt ? ` · “${live.prompt}”` : ''}
            {live.speculative ? ' · speculative' : ''} ·{' '}
            {live.tokens.length}/{live.requested || '?'} tokens
          </div>
          <div className="output-box streaming">
            {live.tokens.join(joiner) || '…'}
            <span className="cursor">▋</span>
          </div>
        </div>
      )}

      {live.status === 'failed' && !loading && (
        <div className="error-box">{live.error}</div>
      )}

      {spectatorResult && (
        <div>
          <div className="note-box">
            Result of the run just streamed (submitted from another device).
          </div>
          <div className="output-box">
            {spectatorResult.output || '(no tokens emitted)'}
          </div>
          <div className="grid">
            <div className="metric-card">
              <div className="metric-label">Throughput</div>
              <div className="metric-value">
                {spectatorResult.tokens_per_sec.toFixed(1)}
              </div>
              <div className="metric-sub">tokens / sec</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Latency</div>
              <div className="metric-value">
                {spectatorResult.latency_ms.toFixed(0)} ms
              </div>
              <div className="metric-sub">{spectatorResult.token_count} tokens</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Payload</div>
              <div className="metric-value">
                {(spectatorResult.network_payload_bytes / 1024).toFixed(1)} KB
              </div>
              <div className="metric-sub">{spectatorResult.engine}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Round Trips</div>
              <div className="metric-value">{spectatorResult.round_trips}</div>
              <div className="metric-sub">
                {spectatorResult.acceptance_rate !== undefined
                  ? `${(spectatorResult.acceptance_rate * 100).toFixed(0)}% draft acceptance`
                  : 'one per pass'}
              </div>
            </div>
          </div>
        </div>
      )}

      {result && !streaming && (
        <div>
          <div className="output-box">{result.output || '(no tokens emitted)'}</div>

          <div className="grid">
            <div className="metric-card">
              <div className="metric-label">Throughput</div>
              <div className="metric-value">{result.tokens_per_sec.toFixed(1)}</div>
              <div className="metric-sub">tokens / sec</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Latency</div>
              <div className="metric-value">{result.latency_ms.toFixed(0)} ms</div>
              <div className="metric-sub">{result.token_count} tokens</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Payload</div>
              <div className="metric-value">
                {(result.network_payload_bytes / 1024).toFixed(1)} KB
              </div>
              <div className="metric-sub">wire: {result.wire_format}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Round Trips</div>
              <div className="metric-value">{result.round_trips}</div>
              <div className="metric-sub">
                {result.speculation
                  ? `${result.speculation.tokens_per_round_trip.toFixed(1)} tok/trip`
                  : 'one per pass'}
              </div>
            </div>
          </div>

          {result.speculation && (
            <div className="note-box">
              Speculative decoding ({result.speculation.drafter}):{' '}
              {result.speculation.accepted_draft_tokens}/{result.speculation.drafted_tokens}{' '}
              drafts accepted ({(result.speculation.acceptance_rate * 100).toFixed(0)}%),{' '}
              {result.speculation.fallback_rounds} fallback round(s). Output is
              token-for-token identical to plain greedy decoding.
            </div>
          )}

          {result.protocol_comparison && (
            <ProtocolComparison
              protocols={result.protocol_comparison.protocols}
              note={result.protocol_comparison.note}
            />
          )}
        </div>
      )}
    </div>
  );
}
