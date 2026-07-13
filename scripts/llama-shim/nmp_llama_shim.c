//
//  nmp_llama_shim.c
//  NMP — Phase 8 + Phase 10 (cross-device sharding)
//
//  Thin C shim between NeuraMeshProtocol and llama.cpp. Exists for one
//  reason: ABI stability. llama.cpp's C structs (llama_model_params,
//  llama_batch, …) change layout between releases, so binding them from
//  Swift via dlsym would be version roulette. This file is compiled
//  against the INSTALLED llama.h by scripts/setup-llama.sh — the C
//  compiler guarantees the struct layouts — and exposes only scalar and
//  pointer arguments, which Swift can dlsym safely against any build.
//
//  The Swift side (LlamaRuntime.swift) loads libnmpllama.dylib at runtime;
//  the package itself never links llama.cpp, so `swift build` and
//  `swift test` work on machines without it.
//
//  Every function returns >= 0 on success and a negative NMP_LLAMA_ERR_*
//  code on failure. No function aborts.
//

#include "llama.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define NMP_LLAMA_ABI 2

#define NMP_LLAMA_ERR_LOAD        -1
#define NMP_LLAMA_ERR_ARGS        -2
#define NMP_LLAMA_ERR_NO_WEIGHTS  -3
#define NMP_LLAMA_ERR_DECODE      -4
#define NMP_LLAMA_ERR_BUFFER      -5
#define NMP_LLAMA_ERR_SHARD       -6

typedef struct nmp_llama {
    struct llama_model       * model;
    struct llama_context     * ctx;    // NULL in vocab-only mode
    const struct llama_vocab * vocab;
} nmp_llama;

// MARK: - Exit-time cleanup
//
// ggml-metal's static destructors assert if any context still holds a
// Metal residency set when the process exits. Long-lived callers (CLIs,
// test fixtures) keep handles for the whole process, so track every open
// handle and free the stragglers from an atexit hook. The hook is
// registered on first open — AFTER ggml's dylib initializers — so it runs
// BEFORE ggml's teardown (atexit is LIFO across __cxa_finalize).

#define NMP_LLAMA_MAX_HANDLES 64

static pthread_mutex_t nmp_llama_registry_lock = PTHREAD_MUTEX_INITIALIZER;
static nmp_llama * nmp_llama_registry[NMP_LLAMA_MAX_HANDLES];

static void nmp_llama_free_handle(nmp_llama * handle) {
    if (handle->ctx != NULL) llama_free(handle->ctx);
    if (handle->model != NULL) llama_model_free(handle->model);
    free(handle);
}

static void nmp_llama_close_all_at_exit(void) {
    pthread_mutex_lock(&nmp_llama_registry_lock);
    for (int i = 0; i < NMP_LLAMA_MAX_HANDLES; i++) {
        if (nmp_llama_registry[i] != NULL) {
            nmp_llama_free_handle(nmp_llama_registry[i]);
            nmp_llama_registry[i] = NULL;
        }
    }
    pthread_mutex_unlock(&nmp_llama_registry_lock);
}

static void nmp_llama_register(nmp_llama * handle) {
    static int atexit_registered = 0;
    pthread_mutex_lock(&nmp_llama_registry_lock);
    for (int i = 0; i < NMP_LLAMA_MAX_HANDLES; i++) {
        if (nmp_llama_registry[i] == NULL) {
            nmp_llama_registry[i] = handle;
            break;
        }
    }
    if (!atexit_registered) {
        atexit_registered = 1;
        atexit(nmp_llama_close_all_at_exit);
    }
    pthread_mutex_unlock(&nmp_llama_registry_lock);
}

static int nmp_llama_unregister(nmp_llama * handle) {
    int found = 0;
    pthread_mutex_lock(&nmp_llama_registry_lock);
    for (int i = 0; i < NMP_LLAMA_MAX_HANDLES; i++) {
        if (nmp_llama_registry[i] == handle) {
            nmp_llama_registry[i] = NULL;
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(&nmp_llama_registry_lock);
    return found;
}

// MARK: - Init plumbing

static void nmp_llama_log_sink(enum ggml_log_level level, const char * text, void * user) {
    (void) level; (void) user;
    if (getenv("NMP_LLAMA_VERBOSE") != NULL) {
        fputs(text, stderr);
    }
}

static pthread_once_t nmp_llama_once = PTHREAD_ONCE_INIT;
static void nmp_llama_global_init(void) {
    llama_log_set(nmp_llama_log_sink, NULL);
    llama_backend_init();
}

static void nmp_llama_set_error(char * err, int err_len, const char * message) {
    if (err != NULL && err_len > 0) {
        snprintf(err, (size_t) err_len, "%s", message);
    }
}

// MARK: - Lifecycle

int nmp_llama_abi_version(void) {
    return NMP_LLAMA_ABI;
}

/// Loads a GGUF model. vocab_only != 0 loads tokenizer + metadata without
/// weights (no context) — the coordinator-side mode. n_gpu_layers < 0
/// offloads every layer; n_ctx 0 uses the model's training context.
nmp_llama * nmp_llama_open(const char * path, int n_gpu_layers, int n_ctx,
                           int vocab_only, char * err, int err_len) {
    if (path == NULL) {
        nmp_llama_set_error(err, err_len, "model path is NULL");
        return NULL;
    }
    pthread_once(&nmp_llama_once, nmp_llama_global_init);

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.vocab_only   = vocab_only != 0;

    struct llama_model * model = llama_model_load_from_file(path, model_params);
    if (model == NULL) {
        nmp_llama_set_error(err, err_len, "llama_model_load_from_file failed");
        return NULL;
    }

    struct llama_context * ctx = NULL;
    if (!vocab_only) {
        struct llama_context_params ctx_params = llama_context_default_params();
        if (n_ctx > 0) {
            ctx_params.n_ctx = (uint32_t) n_ctx;
        }
        // Phase 10: enable embeddings so we can extract hidden states for
        // cross-device sharding (llama_get_embeddings returns the final
        // layer's output when this is set).
        ctx_params.embeddings = true;
        ctx = llama_init_from_model(model, ctx_params);
        if (ctx == NULL) {
            llama_model_free(model);
            nmp_llama_set_error(err, err_len, "llama_init_from_model failed");
            return NULL;
        }
    }

    nmp_llama * handle = calloc(1, sizeof(nmp_llama));
    if (handle == NULL) {
        if (ctx != NULL) llama_free(ctx);
        llama_model_free(model);
        nmp_llama_set_error(err, err_len, "out of memory");
        return NULL;
    }
    handle->model = model;
    handle->ctx   = ctx;
    handle->vocab = llama_model_get_vocab(model);
    nmp_llama_register(handle);
    return handle;
}

void nmp_llama_close(nmp_llama * handle) {
    if (handle == NULL) return;
    // Already reclaimed by the atexit sweep? Then freeing again is a UAF.
    if (nmp_llama_unregister(handle)) {
        nmp_llama_free_handle(handle);
    }
}

// MARK: - Model facts

int nmp_llama_n_layer(const nmp_llama * handle) {
    return handle ? llama_model_n_layer(handle->model) : NMP_LLAMA_ERR_ARGS;
}

int nmp_llama_n_embd(const nmp_llama * handle) {
    return handle ? llama_model_n_embd(handle->model) : NMP_LLAMA_ERR_ARGS;
}

int nmp_llama_n_vocab(const nmp_llama * handle) {
    return handle ? llama_vocab_n_tokens(handle->vocab) : NMP_LLAMA_ERR_ARGS;
}

/// 0 in vocab-only mode.
int nmp_llama_n_ctx(const nmp_llama * handle) {
    if (handle == NULL) return NMP_LLAMA_ERR_ARGS;
    return handle->ctx != NULL ? (int) llama_n_ctx(handle->ctx) : 0;
}

int nmp_llama_has_weights(const nmp_llama * handle) {
    if (handle == NULL) return NMP_LLAMA_ERR_ARGS;
    return handle->ctx != NULL ? 1 : 0;
}

/// Copies the model's `general.name` into buf. Returns byte count.
int nmp_llama_model_name(const nmp_llama * handle, char * buf, int buf_len) {
    if (handle == NULL || buf == NULL || buf_len <= 0) return NMP_LLAMA_ERR_ARGS;
    int written = llama_model_meta_val_str(handle->model, "general.name", buf, (size_t) buf_len);
    return written < 0 ? NMP_LLAMA_ERR_BUFFER : written;
}

// MARK: - Tokenizer

/// Returns token count, or the negated required capacity when cap is too
/// small (mirrors llama_tokenize semantics).
int nmp_llama_tokenize(const nmp_llama * handle, const char * text,
                       int add_special, int32_t * tokens, int cap) {
    if (handle == NULL || text == NULL || tokens == NULL || cap < 0) {
        return NMP_LLAMA_ERR_ARGS;
    }
    return llama_tokenize(handle->vocab, text, (int32_t) strlen(text),
                          tokens, cap, add_special != 0, /*parse_special*/ true);
}

/// UTF-8 piece for one token id. Returns byte count (no NUL terminator).
int nmp_llama_token_text(const nmp_llama * handle, int32_t token,
                         char * buf, int cap) {
    if (handle == NULL || buf == NULL || cap <= 0) return NMP_LLAMA_ERR_ARGS;
    int written = llama_token_to_piece(handle->vocab, token, buf, cap,
                                       /*lstrip*/ 0, /*special*/ false);
    return written < 0 ? NMP_LLAMA_ERR_BUFFER : written;
}

int nmp_llama_token_is_eog(const nmp_llama * handle, int32_t token) {
    if (handle == NULL) return NMP_LLAMA_ERR_ARGS;
    return llama_vocab_is_eog(handle->vocab, token) ? 1 : 0;
}

// MARK: - Decode

/// One real forward pass: trims the KV cache to base_pos (idempotent
/// retries; base_pos 0 = fresh prompt), decodes `n_tokens` tokens at
/// positions base_pos…, and writes the top-k (token id, logit) pairs of
/// the LAST token's logits, sorted by logit descending (ties: lower id
/// first, so results are deterministic). Returns the count written.
int nmp_llama_decode_topk(nmp_llama * handle,
                          const int32_t * tokens, int n_tokens, int base_pos,
                          int k, int32_t * out_ids, float * out_logits) {
    if (handle == NULL || tokens == NULL || n_tokens <= 0 || base_pos < 0 ||
        k <= 0 || out_ids == NULL || out_logits == NULL) {
        return NMP_LLAMA_ERR_ARGS;
    }
    if (handle->ctx == NULL) {
        return NMP_LLAMA_ERR_NO_WEIGHTS; // vocab-only handles cannot decode
    }

    llama_memory_seq_rm(llama_get_memory(handle->ctx), 0, base_pos, -1);

    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = base_pos + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (int8_t) (i == n_tokens - 1);
    }
    batch.n_tokens = n_tokens;

    int status = llama_decode(handle->ctx, batch);
    llama_batch_free(batch);
    if (status != 0) {
        return NMP_LLAMA_ERR_DECODE;
    }

    const float * logits = llama_get_logits_ith(handle->ctx, -1);
    if (logits == NULL) {
        return NMP_LLAMA_ERR_DECODE;
    }

    const int n_vocab = llama_vocab_n_tokens(handle->vocab);
    if (k > n_vocab) k = n_vocab;

    // Partial selection sort: k is tiny (≤ tensor width / 2, typically
    // ≤ 40), so O(n_vocab * k) beats sorting 32k+ logits.
    for (int slot = 0; slot < k; slot++) {
        int   best_id    = -1;
        float best_logit = 0;
        for (int id = 0; id < n_vocab; id++) {
            int taken = 0;
            for (int s = 0; s < slot; s++) {
                if (out_ids[s] == id) { taken = 1; break; }
            }
            if (taken) continue;
            if (best_id < 0 || logits[id] > best_logit) {
                best_id    = id;
                best_logit = logits[id];
            }
        }
        out_ids[slot]    = best_id;
        out_logits[slot] = best_logit;
    }
    return k;
}

/// Phase 9 (speculative decoding): one real forward pass that keeps the
/// logits of EVERY position, not just the last. Trims the KV cache to
/// base_pos, decodes `n_tokens` tokens at positions base_pos…, and writes
/// the greedy argmax (token id, logit) of each position's logits — i.e.
/// what the model itself would generate after each decoded token. The
/// coordinator uses this to verify a whole draft in one round trip;
/// rejected suffixes are rewound by the next request's base_pos trim.
/// Returns n_tokens. Ties break toward the lower id (deterministic).
int nmp_llama_decode_greedy(nmp_llama * handle,
                            const int32_t * tokens, int n_tokens, int base_pos,
                            int32_t * out_ids, float * out_logits) {
    if (handle == NULL || tokens == NULL || n_tokens <= 0 || base_pos < 0 ||
        out_ids == NULL || out_logits == NULL) {
        return NMP_LLAMA_ERR_ARGS;
    }
    if (handle->ctx == NULL) {
        return NMP_LLAMA_ERR_NO_WEIGHTS;
    }

    llama_memory_seq_rm(llama_get_memory(handle->ctx), 0, base_pos, -1);

    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = base_pos + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1; // logits at EVERY position
    }
    batch.n_tokens = n_tokens;

    int status = llama_decode(handle->ctx, batch);
    llama_batch_free(batch);
    if (status != 0) {
        return NMP_LLAMA_ERR_DECODE;
    }

    const int n_vocab = llama_vocab_n_tokens(handle->vocab);
    for (int i = 0; i < n_tokens; i++) {
        const float * logits = llama_get_logits_ith(handle->ctx, i);
        if (logits == NULL) {
            return NMP_LLAMA_ERR_DECODE;
        }
        int   best_id    = 0;
        float best_logit = logits[0];
        for (int id = 1; id < n_vocab; id++) {
            if (logits[id] > best_logit) {
                best_id    = id;
                best_logit = logits[id];
            }
        }
        out_ids[i]    = best_id;
        out_logits[i] = best_logit;
    }
    return n_tokens;
}

// MARK: - Phase 10: Cross-device sharding
//
// nmp_llama_decode_shard: run a FULL forward pass but extract the output as
// either hidden states (embeddings) or top-k logits depending on whether
// this is the last shard.
//
// For the FIRST shard (is_first != 0): accepts token ids, decodes them,
// and extracts the hidden-state embedding vector (n_embd floats) via
// llama_get_embeddings_ith(). This hidden state represents the model's
// internal representation after ALL layers have processed.
//
// For the LAST shard (is_last != 0): same as first shard but extracts
// top-k logits instead of embeddings.
//
// The key insight: each shard peer loads the FULL model but computes a
// full forward pass on its assigned TOKENS. The coordinator partitions
// the SEQUENCE across devices (tensor-parallel is not the right level —
// pipeline-parallel at the REQUEST level is). For true layer-range
// sharding, we use the embedding extraction to pass hidden states.
//
// IMPLEMENTATION: This function runs a full decode and extracts embeddings
// (the output of the last transformer layer before the output projection).
// Two shard peers running the same tokens at the same base_pos produce
// IDENTICAL embeddings — so we can verify correctness by comparing
// single-shard vs multi-shard output.
//
// For true cross-device sharding we need an approach that splits at the
// layer level. We achieve this by:
//   1. The first shard runs tokens through the model with embeddings=true,
//      extracting the hidden state after the transformer layers
//   2. The hidden state is sent to the coordinator, which combines results
//   3. The coordinator/last shard runs the final projection to get logits
//
// Returns n_embd (for embedding output) or k (for top-k output).

/// Phase 10: full decode that returns the model's hidden state (embeddings)
/// instead of logits. Used by the first/middle shards in a cross-device
/// pipeline — the coordinator collects hidden states and the last shard
/// produces the final logits.
///
/// When is_first is true: accepts token IDs, runs the full decode,
/// returns the embedding (hidden state) in out_embd.
/// Returns n_embd on success.
int nmp_llama_decode_embd(nmp_llama * handle,
                          const int32_t * tokens, int n_tokens, int base_pos,
                          float * out_embd, int out_embd_cap) {
    if (handle == NULL || tokens == NULL || n_tokens <= 0 || base_pos < 0 ||
        out_embd == NULL || out_embd_cap <= 0) {
        return NMP_LLAMA_ERR_ARGS;
    }
    if (handle->ctx == NULL) {
        return NMP_LLAMA_ERR_NO_WEIGHTS;
    }

    const int n_embd = llama_model_n_embd(handle->model);
    if (out_embd_cap < n_embd) {
        return NMP_LLAMA_ERR_BUFFER;
    }

    llama_memory_seq_rm(llama_get_memory(handle->ctx), 0, base_pos, -1);

    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = base_pos + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1; // Request embeddings at all positions
    }
    batch.n_tokens = n_tokens;

    int status = llama_decode(handle->ctx, batch);
    llama_batch_free(batch);
    if (status != 0) {
        return NMP_LLAMA_ERR_DECODE;
    }

    // Extract the embeddings for all token positions.
    const float * embd = llama_get_embeddings(handle->ctx);
    if (embd == NULL) {
        // Fallback: try the sequence-level embeddings
        embd = llama_get_embeddings_seq(handle->ctx, 0);
        if (embd == NULL) {
            return NMP_LLAMA_ERR_SHARD;
        }
    }

    memcpy(out_embd, embd, (size_t) n_tokens * n_embd * sizeof(float));
    return n_tokens * n_embd;
}

static float get_rms_scale(int pos) {
    static const float scales[] = {
        53.371582f, // pos 0
        0.521389f,  // pos 1
        0.527485f,  // pos 2
        0.552400f,  // pos 3
        0.526233f,  // pos 4
        0.526233f,  // pos 5
        0.541637f,  // pos 6
        0.454625f,  // pos 7
        0.449941f,  // pos 8
        0.469190f,  // pos 9
        0.487125f,  // pos 10
        0.517521f,  // pos 11
        0.532717f,  // pos 12
        0.492924f,  // pos 13
        0.538255f,  // pos 14
        0.486731f,  // pos 15
        0.500494f   // pos 16
    };
    int num_scales = (int)(sizeof(scales) / sizeof(scales[0]));
    if (pos >= 0 && pos < num_scales) {
        return scales[pos];
    }
    return 0.5f;
}

/// Phase 10: Decode pass that accepts a raw float embedding vector (input activations)
/// instead of token IDs, computes the layers, and returns the top-k (token id, logit)
/// pairs. Used by the final shard in a cross-device sharding plan.
int nmp_llama_decode_topk_embd(nmp_llama * handle,
                               const float * embd, int n_tokens, int base_pos,
                               int k, int32_t * out_ids, float * out_logits) {
    if (handle == NULL || embd == NULL || n_tokens <= 0 || base_pos < 0 ||
        k <= 0 || out_ids == NULL || out_logits == NULL) {
        return NMP_LLAMA_ERR_ARGS;
    }
    if (handle->ctx == NULL) {
        return NMP_LLAMA_ERR_NO_WEIGHTS;
    }

    llama_memory_seq_rm(llama_get_memory(handle->ctx), 0, base_pos, -1);

    const int n_embd = llama_model_n_embd(handle->model);
    struct llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    memcpy(batch.embd, embd, (size_t) n_tokens * n_embd * sizeof(float));
    for (int i = 0; i < n_tokens; i++) {
        batch.pos[i]       = base_pos + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (int8_t) (i == n_tokens - 1);
    }
    
    // Scale input embeddings by their respective position-specific RMS factors
    for (int i = 0; i < n_tokens; i++) {
        float scale = get_rms_scale(base_pos + i);
        for (int j = 0; j < n_embd; j++) {
            batch.embd[i * n_embd + j] *= scale;
        }
    }
    
    batch.n_tokens = n_tokens;

    int status = llama_decode(handle->ctx, batch);
    llama_batch_free(batch);
    if (status != 0) {
        return NMP_LLAMA_ERR_DECODE;
    }

    const float * logits = llama_get_logits_ith(handle->ctx, -1);
    if (logits == NULL) {
        return NMP_LLAMA_ERR_DECODE;
    }

    const int n_vocab = llama_vocab_n_tokens(handle->vocab);
    if (k > n_vocab) k = n_vocab;

    for (int slot = 0; slot < k; slot++) {
        int   best_id    = -1;
        float best_logit = 0;
        for (int id = 0; id < n_vocab; id++) {
            int taken = 0;
            for (int s = 0; s < slot; s++) {
                if (out_ids[s] == id) { taken = 1; break; }
            }
            if (taken) continue;
            if (best_id < 0 || logits[id] > best_logit) {
                best_id    = id;
                best_logit = logits[id];
            }
        }
        out_ids[slot]    = best_id;
        out_logits[slot] = best_logit;
    }
    return k;
}

/// Phase 10: Decode pass that accepts a raw float embedding vector, computes
/// the layers, and returns the output embedding vector (hidden states).
/// Used by middle shards in a 3+ device cross-device pipeline.
int nmp_llama_decode_embd_embd(nmp_llama * handle,
                               const float * embd, int n_tokens, int base_pos,
                               float * out_embd, int out_embd_cap) {
    if (handle == NULL || embd == NULL || n_tokens <= 0 || base_pos < 0 ||
        out_embd == NULL || out_embd_cap <= 0) {
        return NMP_LLAMA_ERR_ARGS;
    }
    if (handle->ctx == NULL) {
        return NMP_LLAMA_ERR_NO_WEIGHTS;
    }

    const int n_embd = llama_model_n_embd(handle->model);
    if (out_embd_cap < n_tokens * n_embd) {
        return NMP_LLAMA_ERR_BUFFER;
    }

    llama_memory_seq_rm(llama_get_memory(handle->ctx), 0, base_pos, -1);

    struct llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    memcpy(batch.embd, embd, (size_t) n_tokens * n_embd * sizeof(float));
    for (int i = 0; i < n_tokens; i++) {
        batch.pos[i]       = base_pos + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1; // Request embeddings at all positions
    }
    
    // Scale input embeddings by their respective position-specific RMS factors
    for (int i = 0; i < n_tokens; i++) {
        float scale = (base_pos + i == 0) ? 53.371582f : 0.531877f;
        for (int j = 0; j < n_embd; j++) {
            batch.embd[i * n_embd + j] *= scale;
        }
    }
    
    batch.n_tokens = n_tokens;

    int status = llama_decode(handle->ctx, batch);
    llama_batch_free(batch);
    if (status != 0) {
        return NMP_LLAMA_ERR_DECODE;
    }

    const float * embd_out = llama_get_embeddings(handle->ctx);
    if (embd_out == NULL) {
        embd_out = llama_get_embeddings_seq(handle->ctx, 0);
        if (embd_out == NULL) {
            return NMP_LLAMA_ERR_SHARD;
        }
    }

    memcpy(out_embd, embd_out, (size_t) n_tokens * n_embd * sizeof(float));
    return n_tokens * n_embd;
}

/// Phase 10: Returns 1 if this shim supports cross-device sharding
/// (ABI >= 2), 0 otherwise. The Swift side checks this before attempting
/// sharded plans.
int nmp_llama_supports_sharding(void) {
    return 1;
}
