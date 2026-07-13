#!/bin/bash
#
# setup_shard.sh — Phase 10: build the ggml graph-surgery shard shim.
#
# Produces Vendor/llama/libnmpshard.dylib (gitignored), dlopen'd by the Swift
# runtime for TRUE cross-device layer sharding. Unlike the llama shim, this
# links the standalone `ggml` brew formula (llama.cpp bakes ggml in statically
# and ships no ggml.h, so it can't be used for graph surgery).
#
# Prerequisite:  brew install ggml
# Usage:         scripts/setup_shard.sh
# Self-test:     scripts/setup_shard.sh --test   (chains 2 shards, checks output)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_ROOT/scripts/llama-shim/nmp_shard_shim.c"
OUT_DIR="$REPO_ROOT/Vendor/llama"
OUT="$OUT_DIR/libnmpshard.dylib"

if ! command -v brew >/dev/null; then
    echo "error: Homebrew not found" >&2; exit 1
fi
if ! GGML_PREFIX="$(brew --prefix ggml 2>/dev/null)" || [ ! -f "$GGML_PREFIX/include/ggml.h" ]; then
    echo "error: the 'ggml' formula is not installed. Run:  brew install ggml" >&2; exit 1
fi

mkdir -p "$OUT_DIR"

# Bake the ggml backend directory so the shim can load ONLY the CPU/BLAS
# backends and SKIP Metal — a 2nd Metal instance (llama.cpp's static ggml has
# one too) conflicts at process teardown. The shard shim computes on CPU, so
# this costs nothing and keeps the test binary's exit clean.
LIBEXEC="$GGML_PREFIX/libexec"

if [ "${1:-}" = "--test" ]; then
    echo "building self-test…"
    clang -DNMP_SHARD_MAIN -DNMP_GGML_LIBEXEC="\"$LIBEXEC\"" -O2 "$SRC" \
        -I"$GGML_PREFIX/include" -L"$GGML_PREFIX/lib" -lggml -lggml-base \
        -Wl,-rpath,"$GGML_PREFIX/lib" -lm -o "$OUT_DIR/shardtest"
    MODEL="${2:-$HOME/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
    "$OUT_DIR/shardtest" "$MODEL" 12 2>/dev/null
    exit 0
fi

echo "building $OUT …"
clang -O2 -dynamiclib -DNMP_GGML_LIBEXEC="\"$LIBEXEC\"" "$SRC" \
    -I"$GGML_PREFIX/include" -L"$GGML_PREFIX/lib" -lggml -lggml-base \
    -Wl,-rpath,"$GGML_PREFIX/lib" -lm -o "$OUT"
echo "ok: $OUT"
echo "exports: $(nm -gU "$OUT" | grep -c nmp_shard) nmp_shard_* symbols"
