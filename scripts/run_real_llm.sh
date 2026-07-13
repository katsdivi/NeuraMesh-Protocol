#!/bin/bash
#
# run_real_llm.sh — start the dashboard on a REAL llama.cpp model.
#
# The dashboard defaults to the weightless reference engine (the only engine
# that can shard layers across many devices today — see Docs/Future_Plans.md).
# This script flips it to a real GGUF: real tokens, a real NMP round-trip per
# token, and the transport race (NMP-UDP vs TCP vs TLS 1.3 vs QUIC) still runs
# on the real generation's traffic pattern.
#
# Honest limit: llama.cpp can't run a layer sub-range, so a real-LLM plan is
# ONE full-range shard (the whole model on one peer). Splitting a model so no
# single device holds all of it is Docs/Future_Plans.md #1 — not built yet.
#
# Prerequisite (one-time):  brew install llama.cpp && scripts/setup_llama.sh
# Usage:                    scripts/run_real_llm.sh [path/to/model.gguf]
#                           (defaults to the small Qwen for a fast start)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_MODEL="$HOME/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL="${1:-$DEFAULT_MODEL}"

if [ ! -f "$REPO_ROOT/Vendor/llama/libnmpllama.dylib" ]; then
    echo "error: llama shim missing. Run:  brew install llama.cpp && scripts/setup_llama.sh" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "error: model not found: $MODEL" >&2
    echo "       pass a GGUF path, or drop one in ~/models/ (see Docs/Start_Here.md §1)" >&2
    exit 1
fi

echo "[run_real_llm] engine=llamaCpp  model=$MODEL"
echo "[run_real_llm] UI on :3000  ·  transport race enabled per inference"
cd "$REPO_ROOT"
exec swift run nmp-dashboard --ui --engine llamaCpp --model "$MODEL" --auto-config
