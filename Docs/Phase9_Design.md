# Phase 9 — Adaptive Sharding, Wire Optimization & Speculative Decoding

**Goal**: make the mesh *fast* — balance shards by measured device speed,
shrink every activation message, overlap independent work across stages,
and break Phase 8's one-token-per-round-trip ceiling — all behind one
command, and all without giving up a single correctness guarantee.

---

## 1. What Phase 9 is (and is not)

Phase 8 ended with two honest cost models:

- **Reference (multi-shard) meshes**: layers split across peers by static
  class weights; a 10% slower peer stalls the whole pipeline because
  nothing ever *measured* before the first plan.
- **Llama (single-shard) meshes**: every generated token is one full mesh
  round trip carrying a 16 KB tensor each way — of which a handful of
  floats are real content (token-state vectors pad to `hiddenSize`).

Phase 9 attacks all of it, with the same honesty discipline as Phase 8:

| lever | applies to | what it must never do |
|---|---|---|
| adaptive layer sharding | multi-shard (reference) plans | change any output bit |
| zero-trim wire format | llama token-state tensors | lose a single bit (lossless) |
| mixed-precision wire format | dense activation tensors | exceed 2⁻¹¹ relative rounding on non-critical values |
| pipelined batching | batches of INDEPENDENT sequences | reorder or cross-contaminate sequences |
| speculative decoding | llama generation streams | emit any token the target model would not have |

**What Phase 9 is not**: mid-layer llama sharding. llama.cpp still cannot
execute a layer sub-range through its public API (see `Phase8_Design.md`
§1); a llama plan remains one full-range shard. And single-stream
pipelining is a physical impossibility, not an unimplemented feature —
token *t+1*'s input IS token *t*'s output, so consecutive tokens of one
autoregressive stream can never occupy two pipeline stages at once. The
single-stream lever is speculation; the multi-stream lever is batching.

## 2. Architecture

### 2.1 Adaptive layer sharding (`AdaptiveSharding.swift`)

Phase 5's `NMPModelSharder` already splits layers proportionally to
`measuredSecondsPerLayer` — but a fresh mesh had no measurements, so its
first plan ran on class weights until live traffic slowly converged it.
Phase 9 closes the loop explicitly:

```
naive plan → N real probe passes → re-plan on measurements → re-assign
     ↑                                                          |
     └────────── persisted profile seeds the NEXT startup ──────┘
```

- **Probes are real pipeline passes** — the peers time themselves with
  the same `computeMicros` path production inference uses, so a probe
  measures the true serving stack, not a synthetic kernel.
- **Re-assignment is one normal SHARD_ASSIGN round** — the exact
  machinery Phase 6 failover already exercises.
- **Profiles persist** in `~/.nmp/neuramesh_sharding.json`, keyed by
  `(modelTag, deviceName)` (peer IDs are per-session; device names
  survive restarts). A complete cached profile skips the probe phase —
  benchmark once per model+device, reuse forever. A corrupt cache loads
  as empty and never blocks startup; a failed probe keeps the naive plan.
- `NMPShardBalance` reports plan quality: per-stage estimated latency,
  pipeline latency (max stage — throughput is set by the slowest stage),
  and balance efficiency (mean/max; 1.0 = nobody idles).

### 2.2 Activation wire formats (`OptimizedActivation.swift`)

`NMPActivationCodec` adds two self-describing formats next to Phase 5's
raw big-endian Float32; decode sniffs a 4-byte magic and falls back to
raw, and `NMPPeerShardEngine` **mirrors the request's format in its
response** — so a Phase 8 coordinator keeps working against a Phase 9
peer, and a coordinator only ever receives what it opted into.

- **`.zeroTrimmed` ("NMPZ")** — drops the trailing zero run, re-pads on
  decode. *Lossless.* Built for llama token-state vectors, which use
  3 + n slots of a 4096-wide tensor: a one-token request shrinks
  16 384 B → 28 B, a 40-candidate response → 344 B. Fewer bytes also
  means 1 chunk instead of 16 — less FEC/NACK exposure per token.
- **`.mixedPrecision` ("NMPH")** — every value as IEEE binary16 plus the
  top 2% by magnitude kept at full Float32 (outliers survive exactly;
  the bulk pays ≤ 2⁻¹¹ relative rounding). ~52% of raw for dense
  activation tensors. **Not** for llama plans: fp16 would round token
  ids ≥ 2048 and near-tied logits.

The binary16 conversion is hand-written (round-to-nearest-even, all
65 536 half patterns pinned by test) rather than using the `Float16`
type — identical bytes on every architecture, the same reason the
reference engine avoids libm.

Determinism ledger: `.float32` and `.zeroTrimmed` keep every bit-exact
guarantee from Phases 5–8. `.mixedPrecision` trades bounded rounding for
bandwidth and is therefore *opt-in* (auto-config selects it only for
reference plans, where the tolerance is measured in tests).

### 2.3 Pipelined batch execution (`PipelinedInference.swift`)

`NMPInferenceOrchestrator.computeStage` (extracted from the serial
`infer` loop) runs ONE stage — local or remote, with Phase 6 stage retry
— so `NMPPipelinedBatchExecutor` can keep every stage busy with a
*different* sequence: while shard 1 computes sequence A, shard 0 already
computes sequence B. Local shards now compute on a dedicated queue so a
long local stage never blocks concurrent remote completions.

Ordering invariant: stage *i* admits sequences strictly in batch order,
one at a time. Since every stage is serial, sequences exit in entry
order — per-peer traffic stays serial, which is the discipline the
peer-side reassembler (`abandonOlder`) already assumes. Outputs are
bit-identical to serial passes; the win is wall clock, bounded by the
slowest stage instead of the sum of stages (measured ~2.4× on a 3-stage
in-process mesh, ceiling 3×).

### 2.4 Speculative decoding (`SpeculativeDecoder.swift` + shim v9)

Phase 8 spends one full mesh round trip per token. Phase 9 changes the
exchange rate:

```
coordinator drafts:   D₁ … D_d                    (cheap, local)
one round trip:       [T, D₁ … D_d]               (T = last confirmed token)
peer returns:         greedy argmax after EVERY position   (verify wire)
accepted:             longest prefix where Dᵢ == argmax after Dᵢ₋₁
                      + one BONUS token (argmax after the last accepted)
                      = up to d+1 tokens per round trip
```

- **Wire**: `NMPLlamaWire` gains verify magics ("LPS"/"LPT") with the
  same layouts as request/response — a verify response carries one
  (id, logit) verdict per decoded position instead of top-k of the last.
- **Shim**: one new function, `nmp_llama_decode_greedy` — a single
  `llama_decode` batch with logits kept at every position, argmax per
  position. Bound as an *optional* symbol: a pre-Phase 9 dylib keeps
  working (speculation reports unavailable with a rebuild hint).
- **Rejection is free**: the next request's `basePos` trims the peer's
  KV cache back to the last accepted position — the same idempotent
  rewind Phase 8 built for loss-recovery retries. No new protocol state.
- **Determinism**: drafts are accepted only where they EQUAL the target
  model's greedy argmax, and the bonus token IS that argmax — so the
  emitted stream is token-for-token identical to the non-speculative
  stream regardless of drafter quality. A bad drafter costs round
  trips, never correctness (pinned by an adversarial-drafter test).

Drafters (`NMPSpeculativeDrafter` seam, depth 4 by default):

- **`NMPPromptLookupDrafter`** (default) — n-gram continuation lookup
  over the generation's own context; no second model, no extra RAM
  (llama.cpp ships the same idea as its `lookup` example). Shines on
  repetitive/structured text; degrades to exactly Phase 8 elsewhere.
- **`NMPLlamaDraftModelDrafter`** — a small same-vocabulary GGUF on the
  coordinator (e.g. TinyLlama-1.1B drafting for Llama-2-7B) decodes
  drafts greedily; its KV cache rewinds by the same prefix-trim rule.

### 2.5 Auto-configuration (`AutoConfig.swift`)

```bash
swift run nmp-dashboard --auto-config
```

sequences everything: report membership → benchmark (or load the cached
profile) → balanced assignment → recommended wire format
(llama → `.zeroTrimmed`, reference → `.mixedPrecision`). Flags:
`--probe-passes N`, `--speculation` (serve every request speculatively;
without it `{"enable_speculation": true}` opts in per request),
`--draft-model path.gguf`. The real-mesh CLI gets the same levers:
`nmp-coordinator --speculation [--draft-model …] [--zero-trim]`.

## 3. Testing guide — from scratch

### Prerequisites

Same as Phase 8 (`brew install llama.cpp`, a GGUF model), plus one
**required** step — the shim gained a function:

```bash
cd NeuraMeshProtocol
scripts/setup_llama.sh          # rebuild → Vendor/llama/libnmpllama.dylib
```

### Test 0: unit + integration suite

```bash
NMP_LLAMA_MODEL=~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf swift test
```

Expect **272 tests, 0 failures** (41 new in Phase 9: half-float
bit-level, codec round trips and mesh payloads, balance math, profile
persistence, heterogeneous-mesh rebalancing, pipelined batch overlap +
bit-exactness, verify wire, drafters, and the speculative service driven
over the real stack by a deterministic toy LM — plus real-model
speculative identity).

### Test 1: adaptive sharding (reference mesh, one command)

```bash
rm -f ~/.nmp/neuramesh_sharding.json     # force a fresh probe phase
swift run nmp-dashboard --auto-config
```

Expect the staged narration: membership (4 devices), probe passes,
per-device layer spans with estimated stage latency, pipeline latency,
balance efficiency, `activation wire format → mixedPrecision`. Restart
without deleting the profile: `probe phase skipped (profile cache is
complete)` — same balanced plan, instantly.

### Test 2: single device llama baseline

```bash
swift run nmp-dashboard --engine llamaCpp \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf --placement local
curl -s -X POST http://127.0.0.1:8080/api/inference \
    -d '{"prompt":"The future of AI is","max_tokens":32}'
```

Note `tokens_per_sec` — the baseline all mesh numbers compare against.

### Test 3: distributed + wire optimization

```bash
swift run nmp-dashboard --engine llamaCpp \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf --auto-config
# same curl
```

Verify against Test 2 and a plain `--placement remote` run:

- `output` **identical** in all three (zero-trim is lossless; greedy
  determinism holds);
- `network_payload_bytes` collapses ~500× (see §4);
- the latency delta vs Test 2 is the true remaining protocol overhead.

### Test 4: speculative decoding

```bash
# same dashboard as Test 3 (auto-config), per-request opt-in:
curl -s -X POST http://127.0.0.1:8080/api/inference \
    -d '{"prompt":"one two three four one two three four one two","max_tokens":32,"enable_speculation":true}'
```

Verify:

- `output` identical to the same prompt without `enable_speculation`;
- the `speculation` object: `tokens_per_round_trip` > 1 on repetitive
  prompts (prompt-lookup drafting), `acceptance_rate`, `fallback_rounds`;
- on non-repetitive prompts the drafter finds little; round trips never
  exceed Phase 8's one per token, though rejected draft batches waste
  some peer compute (see §4) — use a draft model for natural text.

For higher acceptance on natural text, hand the coordinator a small
same-vocabulary draft model:

```bash
swift run nmp-dashboard --engine llamaCpp \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf \
    --auto-config --speculation --draft-model ~/models/tinyllama-1.1b-chat.Q4_K_M.gguf
```

### Test 5: real two-process mesh (unchanged commands + new flags)

```bash
# Terminal A — owns the weights (Phase 9 build serves verify requests):
swift run nmp-peer --engine llamaCpp --model ~/models/llama-2-7b-chat.Q4_K_M.gguf

# Terminal B — tokenizer only, drafts locally, zero-trimmed wire:
swift run nmp-coordinator --engine llamaCpp \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf \
    --prompt "one two three four one two three four one two" \
    --tokens 32 --runs 3 --speculation --zero-trim
```

Expect per-run speculative accounting and `determinism: N runs IDENTICAL
output ✓`. (`--zero-trim` requires the peer to be a Phase 9 build; leave
it off against an older peer.)

## 4. Measured results (Apple M3, 16 GB, Llama-2-7B-Chat Q4_K_M)

Prompt `"The future of AI is"`, 32 tokens, in-process remote placement
(full protocol stack), best of 2 post-warm-up runs:

| configuration | throughput | payload (32 tokens) | round trips | output |
|---|---|---|---|---|
| local shard (baseline) | 14.3–14.5 tok/s | 0 B | — | reference text |
| remote, Phase 8 wire | 13.3–13.5 tok/s | 1 048 576 B | 33 | **identical** |
| remote, `--auto-config` (zero-trim) | 13.8–14.3 tok/s | **11 928 B** | 33 | **identical** |

Speculation (same auto-configured mesh, `enable_speculation: true`):

| prompt | drafting | throughput | round trips | acceptance | output |
|---|---|---|---|---|---|
| repetitive ("one two three four …") | prompt-lookup | **16.5 tok/s** (plain: 15.2) | **8** (4.0 tok/trip) | 25/25 = 100% | **identical to plain** |
| natural ("The future of AI is") | prompt-lookup | 11.7 tok/s | 32 (1.0 tok/trip) | 0/8 | **identical to plain** |

What the numbers say:

- **Wire**: Phase 8 moved ~1 MB per 32-token generation; zero-trim moves
  ~12 KB — a **98.9% reduction** with bit-identical output. Per-token
  transport drops from 16 chunks each way to 1, and remote throughput
  climbs to within ~2% of the single-device baseline: the in-process
  mesh is now essentially compute-bound.
- **Speculation**: where the drafter bites (repetitive/structured text),
  32 tokens cost **8 round trips instead of 33** at 100% acceptance and
  the payload drops to ~1.1 KB. In-process, one trip is cheap, so wall
  clock only improves ~8%; on a real radio (per-token p50 ≈68 ms in
  Phase 8's two-device runs, ~8–17 ms of it per-trip overhead) cutting
  trips 4× is worth far more. Where the drafter finds nothing, round
  trips never exceed Phase 8's — but *rejected* draft batches do waste
  some peer compute (11.7 vs ~14 tok/s on the natural prompt), which is
  why prompt-lookup speculation is per-request opt-in and natural-text
  workloads should use `--draft-model` instead.
- **Adaptive sharding** pays on heterogeneous multi-shard meshes (the
  reference path today, real mid-layer llama sharding when it lands): a
  4×-slower peer measurably shrinks its span (pinned by test), and the
  auto-config narration reports ~99% balance efficiency on the
  homogeneous in-process mesh.

## 5. Troubleshooting

| symptom | fix |
|---|---|
| `speculationUnsupported` / "--speculation needs a Phase 9 shim" | rerun `scripts/setup_llama.sh` (new shim function) |
| peer fails to decode requests after `--zero-trim` | the peer is a pre-Phase 9 build — rebuild it or drop the flag |
| acceptance rate ~0 on natural prompts with prompt-lookup | expected — use `--draft-model` with a same-vocabulary small GGUF |
| draft model warning "vocab N ≠ target M" | draft and target must share a tokenizer (e.g. TinyLlama for Llama-2, Qwen-0.5B for larger Qwen) |
| auto-config keeps re-probing | device names changed between runs, or the profile file is unwritable — check `~/.nmp/` |
| mesh output no longer bit-exact vs baseline | you opted into `.mixedPrecision`; use `.float32`/`.zeroTrimmed` where bit-exactness is required |

## 6. What Phase 10 could add

- iOS peer deployment (llama.cpp xcframework, RAM guardrails).
- True mid-layer llama sharding (ggml-graph splitting) — adaptive
  sharding and pipelined batching are already engine-agnostic and would
  apply to real llama shards unchanged.
- Tree/multi-candidate speculation (verify several draft branches per
  trip — the verify wire already carries per-position verdicts).
- Peer→peer activation forwarding (skip the coordinator hop).
