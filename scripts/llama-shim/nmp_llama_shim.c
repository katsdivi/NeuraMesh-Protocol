//
//  nmp_llama_shim.c
//  NMP — Phase 8
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

#define NMP_LLAMA_ABI 1

#define NMP_LLAMA_ERR_LOAD        -1
#define NMP_LLAMA_ERR_ARGS        -2
#define NMP_LLAMA_ERR_NO_WEIGHTS  -3
#define NMP_LLAMA_ERR_DECODE      -4
#define NMP_LLAMA_ERR_BUFFER      -5

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
