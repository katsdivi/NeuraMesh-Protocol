import { useEffect, useRef, useState } from 'react';
import { api, type ChatMessage, type ChatSummary, type MeshHealth } from '../api';
import type { LiveGeneration } from '../hooks/useMesh';

/**
 * Mesh 3.0: a chat tab with saved history. The mesh stays stateless —
 * every turn resends the whole transcript to POST /api/chat — but each
 * exchange is now persisted to POST /api/chats on THIS device (the
 * coordinator), so conversations survive a refresh and list in a sidebar
 * the way ChatGPT & co. do. Quiet conversations get compressed to LZFSE on
 * disk server-side; loading transparently inflates them. Tokens of the
 * in-flight reply still stream in through the shared generation WebSocket.
 */
export function Chat({
  health,
  live,
}: {
  health: MeshHealth | null;
  live: LiveGeneration;
}) {
  const [conversations, setConversations] = useState<ChatSummary[]>([]);
  const [activeId, setActiveId] = useState<string>(''); // '' = unsaved new chat
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [maxTokens, setMaxTokens] = useState(64);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [lastStats, setLastStats] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const engine = health?.mesh.engine;
  // Reference-engine tokens are bare words; llama pieces carry their own
  // leading spaces.
  const joiner = engine === 'reference' ? ' ' : '';
  // Only stream generations that are actually chat turns into the reply
  // bubble — WS events now carry a source ("chat" | "inference" |
  // "benchmark" | "comparison"), so someone else's benchmark run no
  // longer interleaves into this conversation. Old servers send no
  // source; keep the previous behavior there.
  const streamingReply =
    loading &&
    live.status === 'running' &&
    (live.source === undefined || live.source === 'chat')
      ? live.tokens.join(joiner)
      : '';

  const refreshList = () =>
    api.listChats().then(setConversations).catch(() => {});

  // Load the sidebar once on mount.
  useEffect(() => {
    refreshList();
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages, streamingReply]);

  // Auto-grow the composer with its content, ChatGPT-style (CSS caps it).
  useEffect(() => {
    const ta = inputRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${ta.scrollHeight}px`;
  }, [draft]);

  const newChat = () => {
    setActiveId('');
    setMessages([]);
    setLastStats('');
    setError('');
    setDraft('');
  };

  const openChat = async (id: string) => {
    if (loading || id === activeId) return;
    setError('');
    setLastStats('');
    try {
      const c = await api.loadChat(id);
      setActiveId(c.id);
      setMessages(c.messages);
    } catch {
      setError('Could not open that conversation.');
    }
  };

  const removeChat = async (id: string, event: React.MouseEvent) => {
    event.stopPropagation();
    try {
      await api.deleteChat(id);
    } catch {
      /* best-effort */
    }
    if (id === activeId) newChat();
    refreshList();
  };

  const send = async () => {
    const content = draft.trim();
    if (!content || loading) return;
    const transcript: ChatMessage[] = [...messages, { role: 'user', content }];
    setMessages(transcript);
    setDraft('');
    setLoading(true);
    setError('');
    try {
      const result = await api.chat({
        messages: transcript,
        max_tokens: maxTokens,
      });
      const full: ChatMessage[] = [
        ...transcript,
        {
          role: 'assistant',
          content: result.output || '(no tokens emitted)',
        },
      ];
      setMessages(full);
      setLastStats(
        `${result.tokens_per_sec.toFixed(1)} tok/s · `
          + `${result.round_trips} round trips · `
          + `${(result.network_payload_bytes / 1024).toFixed(1)} KB over the mesh · `
          + `wire: ${result.wire_format}`
          + (result.max_tokens_effective !== undefined
            ? ` · max tokens clamped to ${result.max_tokens_effective}`
            : ''),
      );
      // Persist the turn locally, learning the id for a brand-new chat.
      try {
        const saved = await api.saveChat({ id: activeId, messages: full });
        if (!activeId) setActiveId(saved.id);
        refreshList();
      } catch {
        /* generation still succeeded; history is best-effort */
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const modelLabel = health?.mesh.model || engine || 'mesh';
  const examplePrompts = [
    'Explain how the NeuraMesh sharding works.',
    'Write a haiku about distributed inference.',
    'What can you help me with?',
  ];

  return (
    <div className="cg">
      <aside className="cg-sidebar">
        <button className="cg-newchat" onClick={newChat}>
          <span className="cg-plus">+</span> New chat
        </button>
        <div className="cg-history">
          {conversations.length === 0 && (
            <div className="cg-history-empty">No saved chats yet.</div>
          )}
          {conversations.map((c) => (
            <div
              key={c.id}
              className={`cg-history-item ${c.id === activeId ? 'active' : ''}`}
              onClick={() => openChat(c.id)}
              title={c.title}
            >
              <span className="cg-history-glyph">💬</span>
              <div className="cg-history-main">
                <span className="cg-history-title">{c.title}</span>
                <span className="cg-history-meta">
                  {relativeTime(c.updated_at)} · {c.message_count} msg
                  {c.compressed && (
                    <span className="cg-zip" title="compressed on disk">
                      {' '}· zip
                    </span>
                  )}
                </span>
              </div>
              <button
                className="cg-history-del"
                title="Delete conversation"
                onClick={(e) => removeChat(c.id, e)}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
        <div className="cg-sidebar-foot">
          Model: <span className="cg-model">{modelLabel}</span>
        </div>
      </aside>

      <section className="cg-conversation">
        <div className="cg-thread" ref={scrollRef}>
          {messages.length === 0 && !streamingReply ? (
            <div className="cg-welcome">
              <div className="cg-welcome-mark">◆</div>
              <h1 className="cg-welcome-title">NeuraMesh Chat</h1>
              <p className="cg-welcome-sub">
                Every reply is generated across the mesh, one token per round
                trip. Conversations are saved on this device.
              </p>
              {engine === 'reference' && (
                <div className="cg-note">
                  The reference engine speaks deterministic placeholder
                  vocabulary — this exercises the full mesh protocol, not
                  English. For real conversation, launch with{' '}
                  <code>--engine llamaShard</code> and a chat model (e.g.
                  qwen2.5-instruct).
                </div>
              )}
              <div className="cg-examples">
                {examplePrompts.map((p) => (
                  <button
                    key={p}
                    className="cg-example"
                    onClick={() => setDraft(p)}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages.map((message, index) => (
                <ChatRow key={index} role={message.role} model={modelLabel}>
                  {message.content}
                </ChatRow>
              ))}
              {loading && (
                <ChatRow role="assistant" model={modelLabel} streaming>
                  {streamingReply ? (
                    <>
                      {streamingReply}
                      <span className="cg-cursor">▋</span>
                    </>
                  ) : (
                    <span className="cg-typing">
                      <span></span>
                      <span></span>
                      <span></span>
                    </span>
                  )}
                </ChatRow>
              )}
            </>
          )}
        </div>

        <div className="cg-composer-wrap">
          {error && <div className="cg-error">{error}</div>}
          {lastStats && !loading && <div className="cg-stats">{lastStats}</div>}
          <div className="cg-composer">
            <textarea
              ref={inputRef}
              className="cg-input"
              value={draft}
              rows={1}
              placeholder="Message NeuraMesh…"
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  send();
                }
              }}
            />
            <button
              className="cg-send"
              disabled={loading || !draft.trim()}
              title={loading ? 'Generating…' : 'Send'}
              onClick={send}
            >
              {loading ? <span className="cg-spinner" /> : '↑'}
            </button>
          </div>
          <div className="cg-footline">
            <label className="cg-maxtokens">
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
            <span className="cg-hint">
              Enter to send · Shift+Enter for a new line
            </span>
          </div>
        </div>
      </section>
    </div>
  );
}

/** One ChatGPT-style message row: full-bleed band, centered avatar+content. */
function ChatRow({
  role,
  model,
  streaming,
  children,
}: {
  role: ChatMessage['role'];
  model: string;
  streaming?: boolean;
  children: React.ReactNode;
}) {
  const isUser = role === 'user';
  return (
    <div className={`cg-row ${role} ${streaming ? 'streaming' : ''}`}>
      <div className="cg-row-inner">
        <div className={`cg-avatar ${isUser ? 'user' : 'assistant'}`}>
          {isUser ? 'You' : '◆'}
        </div>
        <div className="cg-message">
          <div className="cg-role">{isUser ? 'You' : model}</div>
          <div className="cg-text">{children}</div>
        </div>
      </div>
    </div>
  );
}

/** Compact "3m ago" / "2h ago" / "Apr 3" from a unix-seconds timestamp. */
function relativeTime(unixSeconds: number): string {
  const seconds = Math.max(0, Date.now() / 1000 - unixSeconds);
  if (seconds < 60) return 'just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(unixSeconds * 1000).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
  });
}
