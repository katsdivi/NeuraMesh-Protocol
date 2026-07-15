#!/bin/bash
#
# run_sharded_mesh.sh — launch a REAL cross-device sharded-llama mesh: a
# coordinator (tokenizer + one shard) plus one or more shard peers, each
# partial-loading ONLY its assigned layer range. Splits ONE GGUF so no single
# device holds the whole model in RAM.
#
# On ONE Mac (loopback smoke test): this starts the coordinator + N local peer
# processes over real UDP — a genuine multi-process split, no second machine
# needed. On real devices: run the coordinator here and start peers on the
# other devices (another Mac: `swift run nmp-peer --engine llamaShard --model
# <same.gguf>`; iPhone: the NeuraMeshPeer app with the model in Documents +
# the shim framework embedded — see scripts/setup_shard_ios.sh).
#
# Usage:  scripts/run_sharded_mesh.sh [model.gguf] [num_local_peers] [prompt]
#         model.gguf       default ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
#         num_local_peers  extra peer PROCESSES to spawn locally (default 1;
#                          0 = expect peers on OTHER devices)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${1:-$HOME/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
LOCAL_PEERS="${2:-1}"
PROMPT="${3:-The capital of France is}"

if [ ! -f "$MODEL" ]; then
    echo "error: model not found: $MODEL" >&2
    echo "  small model: it ships in ~/models, or download 14B: scripts/setup_qwen14b.sh" >&2
    exit 1
fi

# The shim must be built (macOS): Vendor/llama/libnmpshard.dylib.
if [ ! -f "$REPO_ROOT/Vendor/llama/libnmpshard.dylib" ]; then
    echo "[mesh] building the shard shim (one-time) …"
    brew list ggml >/dev/null 2>&1 || brew install ggml
    scripts/setup_shard.sh
fi
brew list llama.cpp >/dev/null 2>&1 || { echo "[mesh] installing llama.cpp for the tokenizer …"; brew install llama.cpp; }
[ -f "$REPO_ROOT/Vendor/llama/libnmpllama.dylib" ] || scripts/setup_llama.sh

echo "[mesh] building executables …"
swift build >/dev/null

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

if [ "$LOCAL_PEERS" -gt 0 ]; then
    echo "[mesh] starting $LOCAL_PEERS local shard peer process(es) over real UDP …"
    for i in $(seq 1 "$LOCAL_PEERS"); do
        swift run nmp-peer --engine llamaShard --model "$MODEL" >/tmp/nmp-shard-peer-$i.log 2>&1 &
        PIDS+=($!)
    done
    sleep 6
fi

echo "[mesh] starting coordinator (this device holds a shard + tokenizes) …"
swift run nmp-coordinator --engine llamaShard --model "$MODEL" \
    --peers "$LOCAL_PEERS" --wait 60 --prompt "$PROMPT" --tokens 16 --runs 2

echo "[mesh] done. Peer logs: /tmp/nmp-shard-peer-*.log"
