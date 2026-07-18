# NMP Plugin Architecture

*Formalizes the compute seam that has always existed in NMP. This is a
documentation/organization pass — no transport, crypto, reliability, or
discovery code changed, and no working behavior changed. Every claim below is
checkable by reading the cited file.*

## 1. What a plugin is

The mesh's compute seam is the protocol `NMPShardComputeEngine`
(`Sources/NMP/ComputeEngine.swift:29`). Everything above it — job placement,
capacity ceilings, churn-safe re-sharding, commit-on-ack, encrypted transport
— is written against that protocol, not against any concrete engine. That
protocol *is* the plugin interface.

`Sources/NMP/NMPPlugin.swift` names it as such:

```swift
public typealias NMPPlugin = NMPShardComputeEngine
```

### Why a typealias instead of a rename

A hard rename of `NMPShardComputeEngine` → `NMPPlugin` would ripple through
~10 source files and every test that names the type (the conformers in
`LlamaEngine.swift`, `LlamaShardEngine.swift`, `VaultShardEngine.swift`,
`ComputeEngine.swift`, plus `PeerNode`, `PeerShardEngine`,
`InferenceOrchestrator`, and the three CLIs). Under the "don't change working
behavior" constraint that is pure risk for zero functional gain. The typealias
lets new code and these docs speak "plugin" while the existing type name and
all conformers stay put. **Judgment call, flagged:** the canonical Swift name
remains `NMPShardComputeEngine`; `NMPPlugin` is the documented public alias.

## 2. The interface a plugin must implement

### Required (`NMPShardComputeEngine`, `ComputeEngine.swift:29`)

| Member | Contract |
| --- | --- |
| `var layerCount: Int` | Total units of work in the job (transformer layers today). The coordinator advertises this as the plan size and the capacity ceiling denominator. |
| `var hiddenSize: Int` | Width of the activation vector peers exchange — the tensor size on the wire. Both ends must agree or the assignment is rejected (`PeerShardEngine.swift:123`). |
| `func runLayers(start:end:input:) throws -> [Float]` | Run units `[start, end)` over one input vector and return the next vector. Precondition: `0 <= start < end <= layerCount`, `input.count == hiddenSize`. |

### Error modes the runtime understands (`NMPComputeError`, `ComputeEngine.swift:48`)

- `invalidLayerRange(start:end:layerCount:)`
- `invalidInputWidth(expected:got:)`
- `noShardAssigned`

Any thrown error is caught at `PeerShardEngine.swift:227` and turned into a
`computeFailed` failure response — the plugin cannot crash the peer by
throwing. The runtime never inspects the *kind* of error; it only cares that
`runLayers` either returns a `[Float]` of length `hiddenSize` or throws.

### Optional capability protocols (opt-in via `as?` casts)

The runtime probes for two refinement protocols and degrades gracefully when a
plugin does not adopt them:

- **`NMPGlobalLayerAware`** (`ComputeEngine.swift:44`) — a plugin that runs a
  *sub-range* needs the plan's total layer count to know whether its shard is
  terminal. `PeerShardEngine` sets `globalLayerCount` from
  `SHARD_ASSIGN.totalLayers` on every assignment
  (`PeerShardEngine.swift:130`). Plugins that always run the full range (the
  reference engine, `llamaCpp`) do not adopt it.
- **`NMPVaultProvisioning`** (`VaultShardEngine.swift:23`) — a plugin that
  holds no weights until assigned streams its slice from the coordinator's
  vault. `PeerShardEngine` calls `provision(for:)` before dimension validation
  (`PeerShardEngine.swift:98`).

### The default helper

`measureLayerLatency(start:end:iterations:)` (`ComputeEngine.swift:57`) is a
protocol extension the sharder uses to weight peers by measured speed. A plugin
gets it for free; it just calls `runLayers` on a dummy input.

### The text seam (only for text-generating plugins)

Prompt inference (`NMPPromptInferenceService`) drives a token loop over the
plugin. How text becomes the first vector, how a token is read from an output
vector, and how the next input is built are captured by a *separate* protocol,
`NMPPromptCodec` (`Sources/NMP/PromptCodec.swift:35`). A plugin whose job is
not text (the hashShard stub) has no meaningful codec — this is one of the
LLM-specific seams called out in §5.

## 3. What the mesh guarantees to any plugin

These properties hold for **any** conforming `NMPShardComputeEngine`, because
the machinery is written against the protocol, not the concrete type:

- **Capacity-aware placement.** `NMPModelSharder`
  (`Sources/NMP/ModelSharder.swift`) plans shards under the
  `capacityThenSpeed` objective (`ModelSharder.swift:71`): it fits units to
  peers by declared capacity first, then weights by measured speed. Work that
  fits nowhere is reported as `capacityShortfall` (`ModelSharder.swift:104`),
  surfaced as "N layer(s) fit nowhere"
  (`FaultToleranceOrchestrator.swift:418`).
- **Churn recovery.** `NMPFaultToleranceOrchestrator` re-shards on peer join
  (`FaultToleranceOrchestrator.swift:292`) and drop
  (`FaultToleranceOrchestrator.swift:268`) without dropping the generation.
- **Commit-on-ack.** A new plan is *staged* (`pendingPlan`) and committed to
  the live `plan` only when every remote peer acks the assignment
  (`InferenceOrchestrator.swift:237`, `pendingAcks`). A superseded round is
  discarded, not half-applied.
- **Encrypted, reliable transport.** Activations travel over the full NMP
  stack — Noise IK handshake, per-session AES-256-GCM, NACK retransmission,
  XOR FEC — unchanged. The plugin never touches the wire; it only produces and
  consumes `[Float]`.

None of these guarantees required plugin-specific code: they see `layerCount`,
`hiddenSize`, and `runLayers`, and nothing else.

## 4. The plugin registry (single source of truth)

Before this pass, `--engine` was parsed and dispatched in three places with
duplicated `if kind == "…"` chains: `NMPPeerCLI/main.swift`,
`NMPCoordinatorCLI/main.swift`, `NMPDashboardCLI/main.swift`. Adding a plugin
meant editing all three plus their `--help` strings.

Now there is one catalog: `NMPPluginRegistry` (`NMPPlugin.swift`). Each entry
is an `NMPPluginDescriptor` carrying its `--engine` id, one-line summary,
`isLLM` flag, `requiresModelFile` flag, and — for pure-compute plugins — a
`makeGeneric` factory that builds the engine from an `NMPPluginContext`
(the CLI flags). The three CLIs:

- validate `--engine` against the registry and fail fast on an unknown id
  (`NMPPeerCLI/main.swift`, `NMPCoordinatorCLI/main.swift`,
  `NMPDashboardCLI/main.swift`);
- print `NMPPluginRegistry.helpBlock` in their usage text;
- build **pure-compute** plugins (reference, hashShard) through
  `descriptor.makeGeneric` instead of hand-rolling construction.

### Where the registry does *not* reach (flagged, honest)

The two LLM plugins leave `makeGeneric` **nil** because their construction is
entangled with per-CLI orchestration that is out of scope to unify here:

- `llamaCpp` / `llamaShard` need a **vocab-only tokenizer** built on the
  coordinator/dashboard (`NMPLlamaModel(…, vocabOnly: true)`), which the peer
  does not build.
- `llamaShard` on the dashboard stands up a **weight vault**
  (`NMPVaultServer`) and auto-selects a model for the host.

Those paths still branch in the CLIs. The registry owns the *catalog,
validation, help, and pure-compute construction*; the LLM-specific wiring is
left where it lives and is explicitly marked. **Judgment call, flagged:**
fully abstracting tokenizer + vault construction behind the registry would
touch the coordinator/dashboard orchestration, which risks the "no behavior
change" constraint for little near-term benefit. Deferred.

Net effect for the common case: **adding a pure-compute plugin is a one-file
change** — write the engine, append one descriptor to `NMPPluginRegistry.all`,
and it is selectable on `nmp-peer` with help text and validation for free.

## 5. What is still LLM-specific (not yet generalized)

Being honest about the seam: the *engine protocol* is generic, but several
layers around it still assume "the job is an LLM shard." A genuinely non-LLM
plugin would need real work in these places — none of it done here:

1. **The wire protocol speaks layers and tensors.** `NMPShardAssign`
   (`Sources/NMP/ShardMessages.swift:80`) carries `startLayer`, `endLayer`,
   `totalLayers`, `hiddenSize`, `modelTag`; `NMPInferRequestMeta`
   (`ShardMessages.swift:177`) carries `startLayer`/`endLayer`; payloads are
   float `tensorChunk`s reassembled by `NMPTensorReassembler`. This is
   application-layer framing (the NMP spec / `NMP.md` transport is untouched
   and out of scope), but a non-tensor job would have to either encode itself
   into this float-vector shape or extend the framing. The hashShard stub
   takes the first option: it pretends to have a layer/hidden shape.
2. **`runLayers` is named and typed for transformer layers.** Input and output
   are `[Float]` of width `hiddenSize`. A job whose natural unit is not a
   fixed-width float vector is forced through this signature.
3. **The text seam assumes tokens.** `NMPPromptCodec` and `NMPChatPrompt`
   (`PromptCodec.swift`) are about tokenization, sampling, EOS, and chat
   templates — meaningless for a checksum job. `NMPHashShardPromptCodec` exists
   only as a compiling placeholder; every method throws `notImplemented`.
4. **The dashboard/web UI is LLM-shaped.** Every tab in `web/src/components`
   speaks models, layers, shards, and tokens (Models, Inference, Chat,
   Devices). There is no generic "job" vocabulary; a non-LLM plugin would have
   no honest UI today. The hashShard stub is therefore peer-CLI-only and is
   explicitly rejected on the dashboard (`NMPDashboardCLI/main.swift`).

**Summary:** the compute *seam* is generic and proven (four conforming engines
plus the stub). The *framing, the text loop, and the UI* above it are still
LLM-specific. That boundary is the honest state of the architecture.

## 6. How to add a new plugin (worked example: hashShard)

`hashShard` is the scaffold added by this pass to prove the seam accepts a
non-LLM job. It is a toy checksum job with **TODO logic** — it is not a real
capability. Files:

- `Sources/NMP/HashShardEngine.swift` — `NMPHashShardComputeEngine`
  conforms to `NMPPlugin`. `runLayers` validates its range/width like any
  engine, then folds the input into an FNV-1a checksum (a placeholder so a
  peer that selects it stays alive; every real decision is marked
  `TODO(plugin)`). `NMPHashShardPromptCodec` is a stub whose methods throw
  `notImplemented`.
- `Sources/NMP/NMPPlugin.swift` — one `NMPPluginDescriptor` (`hashShard`)
  appended to `NMPPluginRegistry.all`, with a `makeGeneric` factory.

That is the entire change surface for a pure-compute plugin. It is selectable:

```bash
swift run nmp-peer --engine hashShard
swift run nmp-peer --help            # lists it under "plugins:"
swift run nmp-peer --engine bogus    # fails with the catalog
```

To turn a scaffold like this into a real plugin you would, in rough order:

1. Implement `runLayers` with the real job (replace the checksum TODO).
2. Decide the wire shape — reuse the float-tensor framing (as the stub does)
   or extend `ShardMessages.swift` (§5.1), which is a larger change.
3. If the job is interactive/streaming, implement a real `NMPPromptCodec`;
   otherwise drive it through whatever request path fits.
4. If it needs a UI, add generic-job vocabulary to the dashboard (§5.4).

Steps 1–2 exercise the seam that already works; steps 3–4 are the
LLM-specific layers that would need generalizing.
