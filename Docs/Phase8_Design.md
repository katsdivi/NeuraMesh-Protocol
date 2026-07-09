# Phase 8 — llama.cpp Integration: Real LLM Inference over the Mesh

**Goal**: replace the deterministic reference engine with real Llama-family
models (quantized GGUF via llama.cpp) and validate single-device and
distributed operation — reproducibly, from scratch.

---

## 1. What Phase 8 is (and is not)

Phases 1–6 built an engine-agnostic pipeline above one seam:

```swift
protocol NMPShardComputeEngine {
    var layerCount: Int { get }
    var hiddenSize: Int { get }
    func runLayers(start: Int, end: Int, input: [Float]) throws -> [Float]
}
```

Phase 8 implements `NMPLlamaComputeEngine` behind that seam. One design
fact drives everything else:

> **llama.cpp cannot execute a layer sub-range.** Its public C API runs
> the whole model per decode step (`llama_decode`), and the KV cache —
> which autoregressive generation depends on — lives inside a single
> `llama_context`. There is no "run layers 16–31 over this activation
> vector" entry point. Splitting a llama model mid-layer requires
> ggml-graph-level surgery (or llama.cpp's own RPC backend, which would
> replace NMP's transport — defeating the point of this project).

So a llama shard **owns the model's full layer range**, and a llama plan
has exactly **one shard**. What the mesh still provides — and what Phase 8
tests — is **real remote execution**:

- the **coordinator loads only the tokenizer** (`vocab_only`, a few MB);
- the **peer owns the weights** and performs real forward passes
  (Metal-accelerated via llama.cpp);
- **every generated token is one full NMP round trip** — Noise IK
  handshake, AES-256-GCM, sequencing, FEC, NACK recovery, measured
  latency — exactly the traffic a physical two-device mesh carries.

Because the response carries real logits and the coordinator samples
greedily, output is **deterministic**: a local plan and a remote plan
produce identical token streams from identical weights. That property
replaces Phase 5's bit-exactness check as the cross-placement validation.

Mid-layer llama sharding (true pipeline parallelism) is a Phase 9+
candidate and requires working below llama.cpp's public API.

## 2. Architecture

### Binding: dlopen'd C shim (zero SwiftPM dependencies preserved)

llama.cpp's C structs (`llama_model_params`, `llama_batch`, …) change
layout between releases, so binding them from Swift via `dlsym` would be
version roulette. Instead:

```
scripts/llama-shim/nmp_llama_shim.c   scalar/pointer-only ABI, compiled
                                      against the INSTALLED llama.h
scripts/setup_llama.sh                brew-based build → Vendor/llama/
                                        libnmpllama.dylib
Sources/NMP/LlamaRuntime.swift        dlopen/dlsym + NMPLlamaModel handle
```

- `swift build` / `swift test` never link llama.cpp — machines without it
  keep working (llama tests `XCTSkip`).
- The shim registers an `atexit` sweep that frees still-open handles, so
  ggml-metal's teardown assertions never fire on process exit.
- Shim search order: `$NMP_LLAMA_LIB` → `./Vendor/llama/libnmpllama.dylib`
  → `~/.nmp/libnmpllama.dylib`.

### Token-state wire format (`LlamaWire.swift`)

The mesh moves fixed-width `[Float]` tensors (`hiddenSize` wide). For
llama plans the tensor carries token state instead of raw activations:

```
request:   [magic, basePos, n, token₀ … tokenₙ₋₁]
response:  [magic, nextPos, k, (tokenID, logit) × k]   k ≤ 40, sorted
```

All values are exact as Float32 (< 2²⁴). The peer trims its KV cache to
`basePos` before decoding — retried/replayed requests are idempotent, and
a fresh prompt (basePos 0) implicitly resets the context. The codec is
stateless: positions travel in the vectors, so loss-recovery retries
cannot desynchronize a generation.

### Text seam (`PromptCodec.swift`)

`NMPPromptInferenceService` now drives an `NMPPromptCodec`:

- `NMPReferencePromptCodec` — Phase 6 pseudo-text behavior, bit-for-bit;
- `NMPLlamaPromptCodec` — real tokenizer (works on a vocab-only handle),
  token-state vectors, greedy sampling over real logits, EOS-aware early
  stop.

### Assemblies

- `NMPLlamaTestbed` — coordinator + one in-process shard peer over an
  in-memory link running the full protocol stack (`.remotePeer`), or the
  single-device baseline (`.local`).
- `nmp-peer --engine llamaCpp` / `nmp-coordinator --engine llamaCpp` —
  the real path: UDP + Bonjour, two processes or two machines.

## 3. Testing guide — from scratch

### Prerequisites (one-time)

```bash
# 1. llama.cpp (library + headers, Metal-enabled)
brew install llama.cpp

# 2. The shim
cd NeuraMeshProtocol
scripts/setup_llama.sh          # → Vendor/llama/libnmpllama.dylib

# 3. A model. Small = fast iteration (~470 MB):
mkdir -p ~/models && cd ~/models
curl -LO https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

# Headline model (~4 GB) — needs ≥ 8 GB free RAM on the peer device:
curl -LO https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

### Test 0: unit + integration suite

```bash
NMP_LLAMA_MODEL=~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf swift test
```

Expect: all tests pass. The `Llama*` suites load the real model; without
the shim or model they skip (everything else is unaffected).

### Test 1: single device (baseline)

```bash
swift run nmp-dashboard --engine llamaCpp \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf --placement local
# then, in another terminal:
curl -s -X POST http://127.0.0.1:8080/api/inference \
    -d '{"prompt":"The capital of France is","max_tokens":16}'
```

Verify in the response:
- `engine: "llamaCpp"` — not `"reference"`;
- `output` is coherent model text;
- `network_payload_bytes: 0` (local shard — nothing crossed the stack);
- note `latency_ms` / `tokens_per_sec` as your baseline.

### Test 2: distributed (full stack in one process)

Same command with `--placement remote` (the default): the shard moves
behind an in-memory link running the complete protocol stack. Re-POST the
same prompt and verify:

- `output` is **identical** to Test 1 (greedy determinism);
- `network_payload_bytes > 0` — every token crossed the stack both ways;
- the latency delta vs Test 1 is the true protocol overhead (the
  dashboard's loss slider now exercises FEC/NACK under real inference).

### Test 3: distributed (two processes / two devices, real UDP)

```bash
# Terminal A (or a second Mac on the same LAN) — owns the weights:
swift run nmp-peer --engine llamaCpp \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf

# Terminal B — tokenizer only (loads no weights):
swift run nmp-coordinator --engine llamaCpp \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf \
    --prompt "The capital of France is" --tokens 16 --runs 3
```

The coordinator discovers the peer over Bonjour, dials it (Noise IK over
UDP), assigns the full-range shard, and streams generations. Verify the
final report:

- real text, one line per run;
- `determinism: N runs IDENTICAL output ✓`;
- per-token p50 ≈ peer compute + one LAN RTT + protocol overhead;
- the peer's log shows real serves with resident memory.

Two physical devices: identical commands; only the network changes. The
iPhone peer additionally needs the shim built for iOS (llama.cpp as an
xcframework) — Phase 9 scope; Mac↔Mac works today.

### Measured results (Apple M3, 16 GB, prompt "The capital of France is")

Qwen2.5-0.5B-Instruct Q4_K_M (469 MB):

| configuration | 16 tokens | throughput | payload |
|---|---|---|---|
| Test 1: local shard | 826 ms | 19.4 tok/s | 0 B |
| Test 2: in-process remote (full stack) | 1103 ms | 14.5 tok/s | 114 688 B |
| Test 3: two processes over UDP loopback | 949 ms | 16.9–18.5 tok/s | 114 688 B |

Output identical across all three: *"Paris. It is the largest city in
Europe and the second largest in the world"*.

Llama-2-7B-Chat Q4_K_M (3.8 GB), Test 3 (two processes, real UDP +
Bonjour; coordinator resident set stays tokenizer-sized):

| runs | 16 tokens | throughput | per-token p50 | payload |
|---|---|---|---|---|
| 2/2 identical ✓ | 1323–1843 ms | 8.7–12.1 tok/s | ≈ 68 ms | 524 288 B |

Output: *"Paris. It is located in the northern central part of the
country and is known"* — real Metal-accelerated 7B inference where every
token crossed the encrypted mesh. Per-token protocol cost ≈ 8–17 ms
(measured with the small model, where compute doesn't mask it); against
7B-scale compute that overhead is < 20% here and shrinks as models grow.

Note: give a freshly downloaded multi-GB model one warm-up load (or a
minute of quiet) — the first cold-cache load competes with the file still
being flushed to disk and can look like a hang.

## 4. Troubleshooting

| symptom | fix |
|---|---|
| `libraryNotFound` | run `scripts/setup_llama.sh`; or set `NMP_LLAMA_LIB` |
| `openFailed` | check the `--model` path; file must be a `.gguf` |
| peer rejects assignment (`rejectedModelMismatch`) | coordinator and peer must point at the **same** GGUF (the model tag is its `general.name`) |
| peer rejects assignment (`rejectedBadRange`) | mismatched layer/hidden metadata — both sides must use the same file version |
| no peers discovered | same Wi-Fi/LAN, Local Network permission granted, mDNS not blocked |
| generation slow / peer OOM | use a smaller quant (Q3) or smaller model; watch the peer's `mem` log lines |
| llama.cpp log spam wanted | set `NMP_LLAMA_VERBOSE=1` |

## 5. What Phase 9 adds

- iOS peer deployment (llama.cpp xcframework, RAM guardrails, bundled
  model caching).
- True mid-layer llama sharding (ggml-graph splitting) — restores
  multi-shard pipelining with real weights.
- Protocol benchmarking with real compute (NMP vs TCP/QUIC baselines).
