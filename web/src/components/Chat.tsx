import { useEffect, useRef, useState } from 'react';
import { api, type ChatMessage, type MeshHealth } from '../api';
import type { LiveGeneration } from '../hooks/useMesh';

/**
 * Mesh 2.7: a chat conversation run via the mesh. The mesh itself is
 * stateless — every turn resends the whole transcript to POST /api/chat,
 * where the engine-appropriate template folds it into one prompt (the
 * same template the iPhone peer app uses). Tokens of the in-flight reply
 * stream in through the shared generation WebSocket events.
 */
export function Chat({
  health,
  live,
}: {
  health: MeshHealth | null;
  live: LiveGeneration;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [maxTokens, setMaxTokens] = useState(64);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [lastStats, setLastStats] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  const engine = health?.mesh.engine;
  // Reference-engine tokens are bare words; llama pieces carry their own
  // leading spaces.
  const joiner = engine === 'reference' ? ' ' : '';
  // Stream into the pending bubble only for a run THIS tab submitted —
  // the generation events are global to every open browser.
  const streamingReply =
    loading && live.status === 'running' ? live.tokens.join(joiner) : '';

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages, streamingReply]);

  const send = async () => {
    const content = draft.trim();
    if (!content || loading) return;
    const transcript: ChatMessage[] = [
      ...messages,
      { role: 'user', content },
    ];
    setMessages(transcript);
    setDraft('');
    setLoading(true);
    setError('');
    try {
      const result = await api.chat({
        messages: transcript,
        max_tokens: maxTokens,
      });
      setMessages([
        ...transcript,
        { role: 'assistant', content: result.output || '(no tokens emitted)' },
      ]);
      setLastStats(
        `${result.tokens_per_sec.toFixed(1)} tok/s · `
        + `${result.round_trips} round trips · `
        + `${(result.network_payload_bytes / 1024).toFixed(1)} KB over the mesh · `
        + `wire: ${result.wire_format}`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-page">
      <h1>Chat</h1>

      {engine === 'reference' && (
        <div className="note-box">
          The reference engine speaks deterministic placeholder vocabulary —
          this chat exercises the full mesh protocol, not English. For real
          conversation, launch with <code>--engine llamaCpp</code> and a chat
          model (e.g. llama-2-7b-chat).
        </div>
      )}

      <div className="chat-window card" ref={scrollRef}>
        {messages.length === 0 && !streamingReply && (
          <div className="chat-empty">
            Say something — every reply is generated across the mesh, one
            token per round trip.
          </div>
        )}
        {messages.map((message, index) => (
          <div key={index} className={`chat-bubble ${message.role}`}>
            {message.content}
          </div>
        ))}
        {loading && (
          <div className="chat-bubble assistant streaming">
            {streamingReply || '…'}
            <span className="cursor">▋</span>
          </div>
        )}
      </div>

      {error && <div className="error-box">{error}</div>}
      {lastStats && !loading && (
        <div className="chat-stats">{lastStats}</div>
      )}

      <div className="chat-composer">
        <textarea
          value={draft}
          rows={2}
          placeholder="Message the mesh…"
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              send();
            }
          }}
        />
        <div className="chat-controls">
          <label>
            Max tokens
            <input
              type="number"
              value={maxTokens}
              min={1}
              max={256}
              onChange={(event) =>
                setMaxTokens(
                  Math.max(1, Math.min(256, Number(event.target.value) || 1)),
                )
              }
            />
          </label>
          <button
            className="primary-button"
            disabled={loading || !draft.trim()}
            onClick={send}
          >
            {loading ? 'Generating…' : 'Send'}
          </button>
          <button
            className="ghost-button"
            disabled={loading || messages.length === 0}
            onClick={() => {
              setMessages([]);
              setLastStats('');
              setError('');
            }}
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}
