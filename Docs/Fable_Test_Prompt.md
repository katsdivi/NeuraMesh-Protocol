# Fable multi-agent test prompt

Paste the block below into Claude Fable (run from inside `NeuraMeshProtocol`).
It spawns several subagents in parallel to test the whole product fast.

---

You are the lead tester for **NeuraMesh**, a live distributed-AI-inference mesh
running right now: a dashboard at **http://localhost:3000**, a coordinator (this
Mac) plus an iPhone peer. Your job is to exercise EVERY endpoint and feature as
real users would, find bugs, and document everything. **Report bugs — do not fix
them.**

**First**, read `Docs/Test_Surface.md` — the complete inventory of endpoints,
UI tabs, features, and known-by-design behaviors. Confirm the mesh is up:
`curl -s http://localhost:3000/health`.

**You MUST use multiple subagents running in parallel** (spawn them in a single
turn via the Agent/Task tool, run_in_background) — both to go fast and because
several agents hitting one mesh at once IS the multi-user concurrency test.
Spawn these agents concurrently, each owning a domain:

1. **inference-agent** — `/api/inference`, `/api/chat`, and the `/ws` token
   stream. Happy paths + edge cases: empty prompt, `max_tokens` at 1 / 128 /
   over the cap, huge prompt, malformed JSON, missing fields, unicode/emoji.
2. **chat-history-agent** — `/api/chats` list/save/load/delete lifecycle.
   Verify: save returns an id + derived title; update preserves id/createdAt;
   idle conversations compress (a `compressed:true` row still loads intact);
   delete removes it; unknown id → 404; reopen persistence.
3. **sharding-agent** — `/api/mesh/plans`, `/api/mesh/strategy`,
   `/api/mesh/autobalance`, `/api/devices/<id>/allocate`, `/api/devices/metrics`.
   Verify the 3 plans differ sensibly, footprints/% look right and never exceed
   100%, apply each strategy, toggle auto/manual, allocate shares including 0
   (exclusion) and 1, bad strategy/share rejected, excluded phone stays VISIBLE.
   NOTE: applying balanced/capacity or a nonzero share makes the phone stream —
   allow up to ~30 s. Restore **auto + speed** when done.
4. **models-agent** (OWNS model switching — no other agent touches
   `/api/models/select`) — `/api/models` correctness (usable/compatible/active
   flags), then switch 0.5B ⇄ 1.5B once each and confirm the mesh comes back
   (the process RE-EXECS; wait for `/health` to return before continuing).
5. **benchmark-compare-agent** — `/api/benchmark/run`, `/api/comparison`,
   `/api/comparison/run` (the measured NMP-vs-TCP/TLS/QUIC race). Sanity-check
   numbers are present, labeled measured vs modeled, and internally consistent.
6. **concurrency-agent** (the "many users at once" test) — while others run,
   hammer the mesh: fire 10+ simultaneous `/api/devices/metrics` polls DURING a
   `/api/mesh/strategy` re-shard (watch for hangs/deadlocks — every endpoint
   should keep responding in <1 s); fire two `/api/inference` at once (expect
   exactly one 429 "already running", never a hang or a stuck busy flag — verify
   a later request still succeeds); rapid autobalance/strategy toggles. Any
   endpoint that stops responding is a SEV-1.

Rules for every agent:
- For each item: state the request, the EXPECTED result, the ACTUAL result.
  Anything that diverges (wrong output, 500, hang, wrong status code, bad math,
  stale/again-and-again errors) is a bug.
- The mesh is shared + stateful. Coordinate: only models-agent switches models;
  everyone restores auto/speed and stops inference before finishing.
- If the whole dashboard stops responding (even `/health` times out), STOP,
  note it as a deadlock/hang SEV-1 with the last action, and report — the lead
  can capture a `sample <pid>` backtrace.

When agents finish, consolidate into **`Docs/Fable_Findings.md`**:
- A **coverage checklist** — every endpoint + feature from Test_Surface.md
  marked ✓ tested / ✗ blocked, with a one-line result.
- A **bug list**, each: severity (SEV-1 hang/crash/data-loss → SEV-3 cosmetic),
  area, exact repro (curl command), expected vs actual.
- A short **UI-manual-check** list — anything only verifiable by clicking in the
  browser or the phone app (agents can't click), for the human to run.

End by leaving the mesh healthy: auto mode, speed strategy, no inference in
flight, model back on the default.
