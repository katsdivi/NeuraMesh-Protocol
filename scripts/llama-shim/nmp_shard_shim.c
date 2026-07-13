//
//  nmp_shard_shim.c
//  NMP — Phase 10: TRUE cross-device layer sharding (ggml graph surgery)
//
//  Unlike nmp_llama_shim.c (which drives llama.cpp's high-level API and can
//  only run the WHOLE model per decode), this shim builds the transformer
//  forward pass directly in ggml so it can execute an arbitrary contiguous
//  block range [start,end) — and load ONLY those blocks' weights. That is
//  what lets a model too big for one device run split across the mesh:
//
//    shard A [0,k)   : tokens   -> blocks[0,k)            -> hidden residual
//    shard M [k,m)   : residual -> blocks[k,m)            -> hidden residual
//    shard B [m,N)   : residual -> blocks[m,N) + norm+lm  -> top-k logits
//
//  Only the n_embd hidden residual crosses the wire (kilobytes), never the
//  weights. Greedy output is bit-identical to the whole-model run — verified
//  against llama.cpp on Qwen2.5-0.5B (all split points + a falsification
//  test). Arch params are read from GGUF metadata, so the same code runs
//  0.5B and 14B; qwen2 (QKV bias) and qwen3 (QK-norm, no bias) are both
//  handled by tensor-presence detection.
//
//  KV CACHE (ABI 2): each shard keeps a persistent per-layer K/V cache, so a
//  decode step processes ONLY the new token(s) and attends over the cached
//  keys/values — O(n) per step instead of reprocessing the whole sequence.
//  The residual that crosses the wire likewise shrinks to n_embd per new
//  token. `n_past` is the authoritative cache length: new K/V are written at
//  [n_past, n_past+n_tokens) and attention reads [0, n_past+n_tokens), which
//  makes a replayed step idempotent (same semantics as llama.cpp trimming
//  its KV to basePos). The cache is F32, so the cached path stays bit-exact
//  against the whole-sequence reprocess.
//
//  Compiled against the standalone `ggml` brew formula (ggml.h + libggml),
//  dlopen'd by the Swift runtime exactly like the llama shim.
//
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define NMP_SHARD_ABI 2
#define NMP_SHARD_DEFAULT_CTX 4096

typedef struct {
    int n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff;
    float eps, rope_base;
    int has_qkv_bias, has_qk_norm;
} nmp_arch;

struct nmp_shard {
    nmp_arch a;
    int start, end;
    struct ggml_context *ctx;   // holds ONLY this shard's tensors
    ggml_backend_t be;
    size_t bytes_loaded;
    int n_tensors;

    // Persistent per-layer KV cache (one entry per LOCAL block, index l-start).
    int max_ctx;
    struct ggml_context *kv_ctx;
    struct ggml_tensor **k_cache;   // each [head_dim, n_head_kv, max_ctx] F32
    struct ggml_tensor **v_cache;   // each [head_dim, n_head_kv, max_ctx] F32
};

static int32_t md_i32(struct gguf_context *g, const char *arch, const char *suf, int32_t dflt) {
    char key[160]; snprintf(key, sizeof key, "%s.%s", arch, suf);
    int64_t i = gguf_find_key(g, key);
    return i < 0 ? dflt : (int32_t) gguf_get_val_u32(g, i);
}
static float md_f32(struct gguf_context *g, const char *arch, const char *suf, float dflt) {
    char key[160]; snprintf(key, sizeof key, "%s.%s", arch, suf);
    int64_t i = gguf_find_key(g, key);
    return i < 0 ? dflt : gguf_get_val_f32(g, i);
}
static int block_index(const char *name) {
    return strncmp(name, "blk.", 4) == 0 ? atoi(name + 4) : -1;
}
static int want(const char *name, int start, int end, int N) {
    if (strcmp(name, "token_embd.weight") == 0)  return start == 0;
    if (strcmp(name, "output_norm.weight") == 0) return end == N;
    if (strcmp(name, "output.weight") == 0)      return end == N;
    int L = block_index(name);
    return (L >= start && L < end);
}
static struct ggml_tensor *TT(struct nmp_shard *s, const char *fmt, int l) {
    char nm[160];
    if (l < 0) snprintf(nm, sizeof nm, "%s", fmt); else snprintf(nm, sizeof nm, fmt, l);
    return ggml_get_tensor(s->ctx, nm);
}

// ---- exported ABI ----

int nmp_shard_abi_version(void) { return NMP_SHARD_ABI; }

// Open a shard that owns blocks [start,end); partial-loads only those weights
// (+ token_embd if start==0, + output_norm/output if end==n_layer). max_ctx is
// the KV cache capacity (<=0 -> default); it bounds prompt+generated tokens.
struct nmp_shard *nmp_shard_open(const char *path, int start, int end, int max_ctx) {
    ggml_backend_load_all();
    struct nmp_shard *s = calloc(1, sizeof *s);
    s->start = start; s->end = end;
    s->max_ctx = max_ctx > 0 ? max_ctx : NMP_SHARD_DEFAULT_CTX;
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    s->be = ggml_backend_dev_init(dev, NULL);

    struct ggml_context *meta = NULL;
    struct gguf_init_params ip = { .no_alloc = true, .ctx = &meta };
    struct gguf_context *g = gguf_init_from_file(path, ip);
    if (!g) { free(s); return NULL; }

    const char *arch = gguf_get_val_str(g, gguf_find_key(g, "general.architecture"));
    nmp_arch *a = &s->a;
    a->n_layer   = md_i32(g, arch, "block_count", 0);
    a->n_embd    = md_i32(g, arch, "embedding_length", 0);
    a->n_head    = md_i32(g, arch, "attention.head_count", 0);
    a->n_head_kv = md_i32(g, arch, "attention.head_count_kv", a->n_head);
    a->n_ff      = md_i32(g, arch, "feed_forward_length", 0);
    a->head_dim  = md_i32(g, arch, "attention.key_length", a->n_head ? a->n_embd / a->n_head : 0);
    a->eps       = md_f32(g, arch, "attention.layer_norm_rms_epsilon", 1e-6f);
    a->rope_base = md_f32(g, arch, "rope.freq_base", 1000000.0f);
    if (end < 0 || end > a->n_layer) s->end = end = a->n_layer;

    s->ctx = ggml_init((struct ggml_init_params){ ggml_tensor_overhead() * 2048, NULL, true });
    for (struct ggml_tensor *t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) {
        const char *nm = ggml_get_name(t);
        if (!want(nm, start, end, a->n_layer)) continue;
        struct ggml_tensor *d = ggml_new_tensor(s->ctx, t->type, ggml_n_dims(t), t->ne);
        ggml_set_name(d, nm);
        if (strstr(nm, "attn_q.bias"))  a->has_qkv_bias = 1;
        if (strstr(nm, "attn_q_norm"))  a->has_qk_norm  = 1;
    }
    ggml_backend_alloc_ctx_tensors(s->ctx, s->be);

    FILE *f = fopen(path, "rb");
    size_t doff = gguf_get_data_offset(g);
    for (struct ggml_tensor *d = ggml_get_first_tensor(s->ctx); d; d = ggml_get_next_tensor(s->ctx, d)) {
        int64_t ti = gguf_find_tensor(g, ggml_get_name(d));
        size_t off = doff + gguf_get_tensor_offset(g, ti), nb = ggml_nbytes(d);
        void *b = malloc(nb);
        fseek(f, (long) off, SEEK_SET);
        if (fread(b, 1, nb, f) != nb) { free(b); fclose(f); gguf_free(g); return NULL; }
        ggml_backend_tensor_set(d, b, 0, nb); free(b);
        s->bytes_loaded += nb; s->n_tensors++;
    }
    fclose(f);
    gguf_free(g);

    // Allocate the persistent per-layer KV cache (F32, one pair per local block).
    int n_local = s->end - s->start;
    s->k_cache = calloc(n_local > 0 ? n_local : 1, sizeof *s->k_cache);
    s->v_cache = calloc(n_local > 0 ? n_local : 1, sizeof *s->v_cache);
    s->kv_ctx = ggml_init((struct ggml_init_params){
        ggml_tensor_overhead() * (size_t)(2 * n_local + 8), NULL, true });
    for (int i = 0; i < n_local; i++) {
        s->k_cache[i] = ggml_new_tensor_3d(s->kv_ctx, GGML_TYPE_F32,
                                           a->head_dim, a->n_head_kv, s->max_ctx);
        s->v_cache[i] = ggml_new_tensor_3d(s->kv_ctx, GGML_TYPE_F32,
                                           a->head_dim, a->n_head_kv, s->max_ctx);
    }
    ggml_backend_alloc_ctx_tensors(s->kv_ctx, s->be);
    return s;
}

void nmp_shard_arch(struct nmp_shard *s, int *n_layer, int *n_embd, int *n_head,
                    int *n_head_kv, int *n_ff, int *start, int *end) {
    if (n_layer)   *n_layer = s->a.n_layer;
    if (n_embd)    *n_embd = s->a.n_embd;
    if (n_head)    *n_head = s->a.n_head;
    if (n_head_kv) *n_head_kv = s->a.n_head_kv;
    if (n_ff)      *n_ff = s->a.n_ff;
    if (start)     *start = s->start;
    if (end)       *end = s->end;
}
long nmp_shard_bytes(struct nmp_shard *s) { return (long) s->bytes_loaded; }
int  nmp_shard_max_ctx(struct nmp_shard *s) { return s->max_ctx; }

// Runs blocks [start,end) over n_tokens NEW positions starting at n_past,
// writing each block's new K/V into its persistent cache and attending over
// the whole cached range [0, n_past+n_tokens). `gf` receives the cache-store
// nodes so they execute before the attention reads them.
static struct ggml_tensor *run_blocks(struct nmp_shard *s, struct ggml_context *ctx,
                                      struct ggml_cgraph *gf, struct ggml_tensor *cur,
                                      struct ggml_tensor *pos, struct ggml_tensor *mask,
                                      int n_tokens, int n_past) {
    nmp_arch *a = &s->a;
    int n_kv = n_past + n_tokens;
    for (int l = s->start; l < s->end; l++) {
        int ci = l - s->start;
        struct ggml_tensor *inpL = cur;
        struct ggml_tensor *x = ggml_mul(ctx, ggml_rms_norm(ctx, cur, a->eps), TT(s, "blk.%d.attn_norm.weight", l));
        struct ggml_tensor *q = ggml_mul_mat(ctx, TT(s, "blk.%d.attn_q.weight", l), x);
        struct ggml_tensor *k = ggml_mul_mat(ctx, TT(s, "blk.%d.attn_k.weight", l), x);
        struct ggml_tensor *v = ggml_mul_mat(ctx, TT(s, "blk.%d.attn_v.weight", l), x);
        if (a->has_qkv_bias) {
            q = ggml_add(ctx, q, TT(s, "blk.%d.attn_q.bias", l));
            k = ggml_add(ctx, k, TT(s, "blk.%d.attn_k.bias", l));
            v = ggml_add(ctx, v, TT(s, "blk.%d.attn_v.bias", l));
        }
        q = ggml_reshape_3d(ctx, q, a->head_dim, a->n_head, n_tokens);
        k = ggml_reshape_3d(ctx, k, a->head_dim, a->n_head_kv, n_tokens);
        v = ggml_reshape_3d(ctx, v, a->head_dim, a->n_head_kv, n_tokens);
        if (a->has_qk_norm) {
            q = ggml_mul(ctx, ggml_rms_norm(ctx, q, a->eps), TT(s, "blk.%d.attn_q_norm.weight", l));
            k = ggml_mul(ctx, ggml_rms_norm(ctx, k, a->eps), TT(s, "blk.%d.attn_k_norm.weight", l));
        }
        q = ggml_rope_ext(ctx, q, pos, NULL, a->head_dim, GGML_ROPE_TYPE_NEOX, 32768, a->rope_base, 1, 0, 1, 32, 1);
        k = ggml_rope_ext(ctx, k, pos, NULL, a->head_dim, GGML_ROPE_TYPE_NEOX, 32768, a->rope_base, 1, 0, 1, 32, 1);

        // Store the new K/V into the persistent cache at [n_past, n_kv).
        struct ggml_tensor *kc = s->k_cache[ci], *vc = s->v_cache[ci];
        size_t slot = (size_t) n_past * kc->nb[2];
        struct ggml_tensor *k_dst = ggml_view_3d(ctx, kc, a->head_dim, a->n_head_kv, n_tokens,
                                                 kc->nb[1], kc->nb[2], slot);
        struct ggml_tensor *v_dst = ggml_view_3d(ctx, vc, a->head_dim, a->n_head_kv, n_tokens,
                                                 vc->nb[1], vc->nb[2], slot);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cont(ctx, k), k_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cont(ctx, v), v_dst));

        // Attention over the full cached range [0, n_kv).
        struct ggml_tensor *Kfull = ggml_view_3d(ctx, kc, a->head_dim, a->n_head_kv, n_kv,
                                                 kc->nb[1], kc->nb[2], 0);
        struct ggml_tensor *Vfull = ggml_view_3d(ctx, vc, a->head_dim, a->n_head_kv, n_kv,
                                                 vc->nb[1], vc->nb[2], 0);
        struct ggml_tensor *Q = ggml_permute(ctx, q, 0, 2, 1, 3);
        struct ggml_tensor *K = ggml_permute(ctx, Kfull, 0, 2, 1, 3);
        struct ggml_tensor *kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, K, Q), mask,
                                                   1.0f / sqrtf((float) a->head_dim), 0);
        struct ggml_tensor *vt = ggml_cont(ctx, ggml_permute(ctx, Vfull, 1, 2, 0, 3));
        struct ggml_tensor *kqv = ggml_permute(ctx, ggml_mul_mat(ctx, vt, kq), 0, 2, 1, 3);
        cur = ggml_mul_mat(ctx, TT(s, "blk.%d.attn_output.weight", l), ggml_cont_2d(ctx, kqv, a->n_embd, n_tokens));
        cur = ggml_add(ctx, cur, inpL);
        struct ggml_tensor *inpFF = cur;
        x = ggml_mul(ctx, ggml_rms_norm(ctx, cur, a->eps), TT(s, "blk.%d.ffn_norm.weight", l));
        struct ggml_tensor *gt = ggml_silu(ctx, ggml_mul_mat(ctx, TT(s, "blk.%d.ffn_gate.weight", l), x));
        struct ggml_tensor *ut = ggml_mul_mat(ctx, TT(s, "blk.%d.ffn_up.weight", l), x);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, TT(s, "blk.%d.ffn_down.weight", l), ggml_mul(ctx, gt, ut)), inpFF);
    }
    return cur;
}

// Evaluate this shard over n_tokens NEW positions starting at n_past.
//   first shard (start==0): pass tokens (int32[n_tokens]), in_hidden = NULL.
//   else:                    pass in_hidden (float[n_embd*n_tokens]), tokens = NULL.
//   non-last shard: writes out_hidden[n_embd*n_tokens] (residual to ship on).
//   last shard (end==n_layer): writes out_ids[k]/out_logits[k] (top-k at last).
// Returns 0 on success, negative on error.
int nmp_shard_eval(struct nmp_shard *s, const int32_t *tokens, const float *in_hidden,
                   int n_tokens, int n_past, float *out_hidden,
                   int k, int32_t *out_ids, float *out_logits) {
    nmp_arch *a = &s->a;
    int first = (s->start == 0), last = (s->end == a->n_layer);
    if (first && !tokens) return -1;
    if (!first && !in_hidden) return -2;
    if (n_tokens <= 0 || n_past < 0) return -3;
    if (n_past + n_tokens > s->max_ctx) return -4;

    int n_kv = n_past + n_tokens;
    size_t mem = ggml_tensor_overhead() * 16384 + ggml_graph_overhead();
    struct ggml_context *ctx = ggml_init((struct ggml_init_params){ mem, NULL, true });
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    struct ggml_tensor *pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens); ggml_set_input(pos);
    struct ggml_tensor *mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens); ggml_set_input(mask);
    struct ggml_tensor *tok = NULL, *hin = NULL, *cur;
    if (first) { tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens); ggml_set_input(tok);
                 cur = ggml_get_rows(ctx, TT(s, "token_embd.weight", -1), tok); }
    else       { hin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a->n_embd, n_tokens); ggml_set_input(hin); cur = hin; }
    cur = run_blocks(s, ctx, gf, cur, pos, mask, n_tokens, n_past);
    struct ggml_tensor *outp;
    if (last) {
        cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, a->eps), TT(s, "output_norm.weight", -1));
        struct ggml_tensor *lastcol = ggml_view_1d(ctx, cur, a->n_embd, (size_t)(n_tokens - 1) * cur->nb[1]);
        outp = ggml_mul_mat(ctx, TT(s, "output.weight", -1), lastcol);
    } else outp = cur;

    ggml_build_forward_expand(gf, outp);
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s->be));
    ggml_gallocr_alloc_graph(al, gf);

    if (first) ggml_backend_tensor_set(tok, tokens, 0, n_tokens * sizeof(int32_t));
    else       ggml_backend_tensor_set(hin, in_hidden, 0, (size_t) a->n_embd * n_tokens * sizeof(float));
    int32_t *pb = malloc(n_tokens * sizeof(int32_t));
    for (int i = 0; i < n_tokens; i++) pb[i] = n_past + i;
    ggml_backend_tensor_set(pos, pb, 0, n_tokens * sizeof(int32_t)); free(pb);
    // Causal mask over the cached range: query i (absolute pos n_past+i)
    // attends key j (absolute pos j) iff j <= n_past+i.
    float *m = malloc((size_t) n_kv * n_tokens * sizeof(float));
    for (int qi = 0; qi < n_tokens; qi++)
        for (int kj = 0; kj < n_kv; kj++)
            m[qi * n_kv + kj] = (kj <= n_past + qi) ? 0.0f : -INFINITY;
    ggml_backend_tensor_set(mask, m, 0, (size_t) n_kv * n_tokens * sizeof(float)); free(m);

    ggml_backend_graph_compute(s->be, gf);

    if (last) {
        int nv = (int) outp->ne[0];
        float *lg = malloc(nv * sizeof(float));
        ggml_backend_tensor_get(outp, lg, 0, nv * sizeof(float));
        if (k > nv) k = nv;
        for (int slot = 0; slot < k; slot++) {
            int best = -1; float bl = -INFINITY;
            for (int id = 0; id < nv; id++) {
                int taken = 0;
                for (int p = 0; p < slot; p++) if (out_ids[p] == id) { taken = 1; break; }
                if (!taken && (best < 0 || lg[id] > bl)) { best = id; bl = lg[id]; }
            }
            out_ids[slot] = best; out_logits[slot] = bl;
        }
        free(lg);
    } else {
        ggml_backend_tensor_get(outp, out_hidden, 0, (size_t) a->n_embd * n_tokens * sizeof(float));
    }
    ggml_gallocr_free(al);
    ggml_free(ctx);
    return 0;
}

void nmp_shard_free(struct nmp_shard *s) {
    if (!s) return;
    if (s->kv_ctx) ggml_free(s->kv_ctx);
    free(s->k_cache);
    free(s->v_cache);
    if (s->ctx) ggml_free(s->ctx);
    if (s->be)  ggml_backend_free(s->be);
    free(s);
}

#ifdef NMP_SHARD_MAIN
// Self-test: chain two partial-loaded shards with a KV cache — prefill the
// prompt (n_past 0), then decode one token at a time (n_past grows). Expect
// the llama.cpp greedy stream for "The capital of France is". Build:
//   clang -DNMP_SHARD_MAIN nmp_shard_shim.c -I$(brew --prefix ggml)/include \
//     -L$(brew --prefix ggml)/lib -lggml -lggml-base -lm -o shardtest
int main(int argc, char **argv) {
    const char *path = argv[1];
    int split = argc > 2 ? atoi(argv[2]) : 12;
    struct nmp_shard *A = nmp_shard_open(path, 0, split, 512);
    int N; nmp_shard_arch(A, &N, NULL, NULL, NULL, NULL, NULL, NULL);
    struct nmp_shard *B = nmp_shard_open(path, split, N, 512);
    printf("A[0,%d): %.1f MB   B[%d,%d): %.1f MB   (neither holds the whole model)\n",
           split, nmp_shard_bytes(A) / 1e6, split, N, nmp_shard_bytes(B) / 1e6);
    int32_t seq[64] = {785, 6722, 315, 9625, 374}; int np = 5, ng = 8, ne;
    nmp_shard_arch(A, NULL, &ne, NULL, NULL, NULL, NULL, NULL);
    float *wire = malloc((size_t) 64 * ne * sizeof(float));
    printf("generated:");
    int n_past = 0;
    // First call carries the whole prompt; later calls carry one token.
    int n_tokens = np;
    int32_t next[64]; memcpy(next, seq, np * sizeof(int32_t));
    for (int step = 0; step < ng; step++) {
        nmp_shard_eval(A, next, NULL, n_tokens, n_past, wire, 0, NULL, NULL);
        int32_t id; float lg;
        nmp_shard_eval(B, NULL, wire, n_tokens, n_past, NULL, 1, &id, &lg);
        printf(" %d", id); fflush(stdout);
        n_past += n_tokens;
        next[0] = id; n_tokens = 1;   // decode one token at a time
    }
    printf("\nexpected : 12095 13 1084 374 279 7772 3283 304\n");
    nmp_shard_free(A); nmp_shard_free(B); free(wire);
    return 0;
}
#endif
