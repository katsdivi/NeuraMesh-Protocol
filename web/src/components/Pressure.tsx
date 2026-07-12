import { useRef, useState } from 'react';
import { api, type Device } from '../api';

/**
 * Mesh 2.5: the pressure lab — the terminal hammer scripts as buttons.
 * Every phase drives REAL inference through the live mesh (including
 * LAN peers like a phone); nothing here is simulated:
 *
 *  - soak: N sequential generations; latency drift or a mid-soak step
 *    (a peer dropped, mesh re-sharded) shows up in the run table.
 *  - burst: M simultaneous requests. The coordinator serializes
 *    generations and sheds overload with 429s — burst PROVES that
 *    instead of hiding it (429s are the correct outcome, not failures).
 *  - max payload: 128-token generations, the largest activation traffic
 *    a run can produce.
 *
 * Peer health is snapshotted before and after each phase so a device
 * that drops under load is called out explicitly.
 */

interface RunRow {
  phase: string;
  label: string;
  ok: boolean;
  latencyMs?: number;
  tokensPerSec?: number;
  detail?: string;
}

interface PhaseSummary {
  phase: string;
  verdict: string;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

async function peerSnapshot(): Promise<Device[]> {
  try {
    return await api.devices();
  } catch {
    return [];
  }
}

function droppedBetween(before: Device[], after: Device[]): string[] {
  const aliveAfter = new Set(after.filter((d) => d.alive).map((d) => d.id));
  return before
    .filter((d) => d.alive && !aliveAfter.has(d.id))
    .map((d) => d.name);
}

export function Pressure() {
  const [rows, setRows] = useState<RunRow[]>([]);
  const [summaries, setSummaries] = useState<PhaseSummary[]>([]);
  const [running, setRunning] = useState<string | null>(null);
  const [soakRuns, setSoakRuns] = useState(10);
  const [soakTokens, setSoakTokens] = useState(64);
  const [burstSize, setBurstSize] = useState(5);
  const stopRequested = useRef(false);

  const addRow = (row: RunRow) => setRows((current) => [...current, row]);
  const addSummary = (summary: PhaseSummary) =>
    setSummaries((current) => [...current.filter((s) => s.phase !== summary.phase), summary]);

  const begin = (phase: string) => {
    stopRequested.current = false;
    setRunning(phase);
    setRows((current) => current.filter((row) => row.phase !== phase));
  };

  const runSoak = async () => {
    begin('soak');
    const before = await peerSnapshot();
    const latencies: number[] = [];
    let failures = 0;
    for (let i = 0; i < soakRuns && !stopRequested.current; i += 1) {
      try {
        const result = await api.inference({
          prompt: `pressure soak ${i}`,
          max_tokens: soakTokens,
        });
        latencies.push(result.latency_ms);
        addRow({
          phase: 'soak',
          label: `run ${i + 1}/${soakRuns}`,
          ok: true,
          latencyMs: result.latency_ms,
          tokensPerSec: result.tokens_per_sec,
        });
      } catch (error) {
        failures += 1;
        addRow({
          phase: 'soak',
          label: `run ${i + 1}/${soakRuns}`,
          ok: false,
          detail: error instanceof Error ? error.message : String(error),
        });
      }
    }
    const dropped = droppedBetween(before, await peerSnapshot());
    const latencyLine = latencies.length
      ? `latency min/median/max ${Math.min(...latencies).toFixed(0)}/` +
        `${median(latencies).toFixed(0)}/${Math.max(...latencies).toFixed(0)} ms`
      : 'no successful runs';
    addSummary({
      phase: 'soak',
      verdict:
        `${latencies.length}/${soakRuns} ok, ${failures} failed — ${latencyLine}` +
        (dropped.length
          ? `. DROPPED DURING SOAK: ${dropped.join(', ')} (the mesh re-sharded; ` +
            'a latency step in the table marks the moment)'
          : '. No peers dropped.'),
    });
    setRunning(null);
  };

  const runBurst = async () => {
    begin('burst');
    const results = await Promise.all(
      Array.from({ length: burstSize }, (_, i) =>
        api
          .inference({ prompt: `burst ${i}`, max_tokens: 16 })
          .then((result) => ({ ok: true as const, result }))
          .catch((error: unknown) => ({
            ok: false as const,
            message: error instanceof Error ? error.message : String(error),
          })),
      ),
    );
    let accepted = 0;
    let shed = 0;
    results.forEach((outcome, i) => {
      if (outcome.ok) {
        accepted += 1;
        addRow({
          phase: 'burst',
          label: `request ${i + 1}`,
          ok: true,
          latencyMs: outcome.result.latency_ms,
          tokensPerSec: outcome.result.tokens_per_sec,
        });
      } else {
        const is429 = outcome.message.includes('429')
          || outcome.message.toLowerCase().includes('busy')
          || outcome.message.toLowerCase().includes('generation');
        if (is429) shed += 1;
        addRow({
          phase: 'burst',
          label: `request ${i + 1}`,
          ok: is429,
          detail: is429 ? `shed with: ${outcome.message}` : outcome.message,
        });
      }
    });
    addSummary({
      phase: 'burst',
      verdict:
        `${accepted} served, ${shed} shed cleanly of ${burstSize} simultaneous. ` +
        'One generation at a time is the design; shedding the rest with an ' +
        'error beats queueing them into a stall. Anything not served AND ' +
        'not shed is a bug.',
    });
    setRunning(null);
  };

  const runMaxPayload = async () => {
    begin('max');
    const before = await peerSnapshot();
    for (let i = 0; i < 2 && !stopRequested.current; i += 1) {
      try {
        const result = await api.inference({
          prompt: `max payload ${i}`,
          max_tokens: 128,
        });
        addRow({
          phase: 'max',
          label: `128-token run ${i + 1}/2`,
          ok: true,
          latencyMs: result.latency_ms,
          tokensPerSec: result.tokens_per_sec,
          detail: `${(result.network_payload_bytes / 1024).toFixed(0)} KB moved`,
        });
      } catch (error) {
        addRow({
          phase: 'max',
          label: `128-token run ${i + 1}/2`,
          ok: false,
          detail: error instanceof Error ? error.message : String(error),
        });
      }
    }
    const dropped = droppedBetween(before, await peerSnapshot());
    addSummary({
      phase: 'max',
      verdict: dropped.length
        ? `DROPPED UNDER MAX PAYLOAD: ${dropped.join(', ')}`
        : 'Largest per-run traffic the mesh can generate; no peers dropped.',
    });
    setRunning(null);
  };

  return (
    <div>
      <h1>Pressure Lab</h1>
      <div className="note-box">
        Every phase drives real inference through the live mesh — phone
        peers included. Watch the Devices tab (or the phone's Xcode
        console) in parallel; a peer that drops mid-phase is reported in
        the phase verdict. Two knobs live outside the browser: real
        packet loss for the transport race needs{' '}
        <code>sudo scripts/loss_lab.sh</code>, and throttling the phone's
        actual radio is Settings ▸ Developer ▸ Network Link Conditioner
        on the phone (see Docs/Pressure_Test_Guide.md).
      </div>

      <div className="card">
        <div className="form-row">
          <div className="form-group">
            <label>Soak runs</label>
            <input
              type="number"
              value={soakRuns}
              min={1}
              max={100}
              onChange={(event) =>
                setSoakRuns(Math.max(1, Math.min(100, Number(event.target.value) || 1)))
              }
            />
          </div>
          <div className="form-group">
            <label>Tokens per soak run</label>
            <input
              type="number"
              value={soakTokens}
              min={1}
              max={128}
              onChange={(event) =>
                setSoakTokens(Math.max(1, Math.min(128, Number(event.target.value) || 1)))
              }
            />
          </div>
          <div className="form-group">
            <label>Burst size</label>
            <input
              type="number"
              value={burstSize}
              min={2}
              max={12}
              onChange={(event) =>
                setBurstSize(Math.max(2, Math.min(12, Number(event.target.value) || 2)))
              }
            />
          </div>
        </div>
        <div className="form-row">
          <button
            className="primary-button"
            disabled={running !== null}
            onClick={runSoak}
          >
            {running === 'soak' ? 'Soaking…' : 'Run soak'}
          </button>
          <button
            className="primary-button"
            disabled={running !== null}
            onClick={runBurst}
          >
            {running === 'burst' ? 'Bursting…' : 'Run burst'}
          </button>
          <button
            className="primary-button"
            disabled={running !== null}
            onClick={runMaxPayload}
          >
            {running === 'max' ? 'Running…' : 'Run max payload'}
          </button>
          {running && (
            <button
              className="primary-button"
              onClick={() => {
                stopRequested.current = true;
              }}
            >
              Stop after current run
            </button>
          )}
        </div>
      </div>

      {summaries.map((summary) => (
        <div className="note-box" key={summary.phase} style={{ marginTop: 'var(--spacing-md)' }}>
          <strong>{summary.phase}:</strong> {summary.verdict}
        </div>
      ))}

      {rows.length > 0 && (
        <div className="card" style={{ marginTop: 'var(--spacing-md)' }}>
          <div className="comparison-table">
            <table>
              <thead>
                <tr>
                  <th>Phase</th>
                  <th>Run</th>
                  <th>Outcome</th>
                  <th>Latency</th>
                  <th>Throughput</th>
                  <th>Detail</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => (
                  <tr key={index}>
                    <td>{row.phase}</td>
                    <td>{row.label}</td>
                    <td>{row.ok ? '✓' : '✗'}</td>
                    <td>{row.latencyMs !== undefined ? `${row.latencyMs.toFixed(0)} ms` : '—'}</td>
                    <td>
                      {row.tokensPerSec !== undefined
                        ? `${row.tokensPerSec.toFixed(1)} tok/s`
                        : '—'}
                    </td>
                    <td style={{ fontSize: 'var(--text-caption)' }}>{row.detail ?? ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
