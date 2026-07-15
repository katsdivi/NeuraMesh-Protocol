#!/bin/bash
#
# start.sh — ONE command to go from a fresh Mac to a running NeuraMesh with the
# full web UI. After this finishes, EVERYTHING you do day-to-day — running
# inference, chatting, testing, benchmarking, racing NMP vs TCP/QUIC, watching
# the sharded mesh, injecting packet loss — happens in the browser. No more
# terminal commands.
#
# What it does (idempotent — safe to re-run; skips anything already done):
#   1. verifies Xcode command-line tools + Homebrew
#   2. installs the ggml + llama.cpp brew formulae (the tokenizer + shard math)
#   3. builds the two native shims once (Vendor/llama/*.dylib, gitignored)
#   4. makes sure a qwen GGUF exists in ~/models (grabs the tiny 0.5B if not)
#   5. builds the Swift package
#   6. launches the dashboard with the REAL sharded engine + web UI, and opens
#      your browser at http://localhost:3000
#
# Usage:
#   scripts/start.sh                       # auto-select the best-fitting model
#   scripts/start.sh --model ~/models/X.gguf   # force a specific GGUF
#   scripts/start.sh --engine reference    # no shims/model needed (simulated mesh)
#
# The ONLY thing this can't move into the browser is itself (a web page can't
# compile native code or sign an iOS app) and the iPhone peer (see
# Docs/Final_Setup_Guide.md → "Add your iPhone").
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Small qwen model auto-download target (arch the shard shim runs: qwen2).
MODELS_DIR="$HOME/models"
SMALL_MODEL="qwen2.5-0.5b-instruct-q4_k_m.gguf"
SMALL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/${SMALL_MODEL}?download=true"

# Pass-through engine (default: real sharding). Peek at args so we can skip the
# shim/model bootstrap when the user only wants the simulated reference mesh,
# and so we only inject our default --engine when the user didn't pass one.
ENGINE="llamaShard"
USER_SET_ENGINE=0
prev=""
for a in "$@"; do
    [ "$prev" = "--engine" ] && { ENGINE="$a"; USER_SET_ENGINE=1; }
    prev="$a"
done

say() { printf '\033[1;36m[start]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[start] error:\033[0m %s\n' "$*" >&2; exit 1; }

# --- 1. prerequisites -------------------------------------------------------
xcode-select -p >/dev/null 2>&1 || die "Xcode command-line tools missing. Run: xcode-select --install"
command -v swift >/dev/null || die "swift not found (install Xcode or the toolchain)."

if [ "$ENGINE" = "llamaShard" ] || [ "$ENGINE" = "llamaCpp" ]; then
    command -v brew >/dev/null || die "Homebrew not found. Install from https://brew.sh then re-run."

    # --- 2. brew deps -------------------------------------------------------
    brew list ggml      >/dev/null 2>&1 || { say "installing ggml (shard math) …";       brew install ggml; }
    brew list llama.cpp >/dev/null 2>&1 || { say "installing llama.cpp (tokenizer) …";    brew install llama.cpp; }

    # --- 3. native shims (one-time) ----------------------------------------
    if [ ! -f "Vendor/llama/libnmpshard.dylib" ]; then
        say "building the shard shim (one-time, ~30s) …"
        scripts/setup_shard.sh
    fi
    if [ ! -f "Vendor/llama/libnmpllama.dylib" ]; then
        say "building the tokenizer shim (one-time) …"
        scripts/setup_llama.sh
    fi

    # --- 4. a model to serve ------------------------------------------------
    mkdir -p "$MODELS_DIR"
    if ! ls "$MODELS_DIR"/qwen*.gguf >/dev/null 2>&1; then
        say "no qwen model in ~/models — downloading the tiny 0.5B (~490 MB) so you can start now …"
        say "  (for the big one later: scripts/setup_qwen14b.sh)"
        curl -L -C - --fail -o "$MODELS_DIR/$SMALL_MODEL" "$SMALL_URL" \
            || die "download failed — grab any qwen2.5 *.gguf into ~/models and re-run, or pass --model."
    fi
fi

# --- 5. build ---------------------------------------------------------------
say "building the Swift package …"
swift build >/dev/null

# --- 6. launch + open browser ----------------------------------------------
URL="http://localhost:3000"
say "launching the mesh + web UI → $URL"
say "  Ctrl-C here stops the mesh. Everything else lives in the browser."
( sleep 4; command -v open >/dev/null && open "$URL" >/dev/null 2>&1 || true ) &

# --ui forces port 3000 and the LAN-reachable web app. When the user didn't
# name an engine, default to the real sharded mesh (auto-selects the
# best-fitting model from ~/models).
if [ "$USER_SET_ENGINE" -eq 1 ]; then
    exec swift run nmp-dashboard --ui "$@"
else
    exec swift run nmp-dashboard --ui --engine llamaShard "$@"
fi
