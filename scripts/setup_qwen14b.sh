#!/bin/bash
#
# setup_qwen14b.sh — one-command download of a Qwen-14B GGUF for the sharded
# mesh. Qwen2.5-14B is qwen2 arch (QKV bias), which the ggml shard shim runs,
# and the model selector auto-picks it once every hosting device has disk for
# the file.
#
# STORAGE MATH (read this): the shard engine partial-loads only its assigned
# layers into RAM, but each hosting device reads those layers from the WHOLE
# GGUF on disk — so every device that hosts a shard needs free disk for the
# entire file (that is the storage ceiling the selector enforces). Pick a
# quant that fits your SMALLEST device's free disk:
#
#   quant       file size    RAM to hold ALL layers (1 device)   good for
#   q4_k_m      ~8.9 GB      ~11 GB                               Mac + roomy iPad
#   q3_k_m      ~7.1 GB      ~9 GB                                tighter devices
#   q2_k        ~5.8 GB      ~7.5 GB                              iPhone-friendly
#
# Split across N devices, each holds ~1/N of the layers in RAM — but still the
# whole file on disk. (Disaggregating disk from compute is Future_Plans #3.)
#
# Usage:   scripts/setup_qwen14b.sh [q4_k_m|q3_k_m|q2_k]   (default q4_k_m)
#          MODELS_DIR=~/models REPO=bartowski/Qwen2.5-14B-Instruct-GGUF scripts/setup_qwen14b.sh
#
set -euo pipefail

QUANT="${1:-q4_k_m}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
REPO="${REPO:-bartowski/Qwen2.5-14B-Instruct-GGUF}"

# bartowski file naming: Qwen2.5-14B-Instruct-<QUANT_UPPER>.gguf
QUANT_UPPER="$(echo "$QUANT" | tr '[:lower:]' '[:upper:]')"
FILE="Qwen2.5-14B-Instruct-${QUANT_UPPER}.gguf"
URL="https://huggingface.co/${REPO}/resolve/main/${FILE}?download=true"
DEST="$MODELS_DIR/${FILE}"

# Approx sizes (MB) for the disk pre-check.
case "$QUANT" in
    q4_k_m) NEED_MB=9200 ;;
    q3_k_m) NEED_MB=7300 ;;
    q2_k)   NEED_MB=6000 ;;
    *)      NEED_MB=9200 ;;
esac

mkdir -p "$MODELS_DIR"

# Storage-aware pre-check: refuse if this Mac lacks room (matches the mesh's
# own storage ceiling — no point downloading a file no device can host).
FREE_MB=$(df -m "$MODELS_DIR" | awk 'NR==2 {print $4}')
echo "[qwen14b] target: $DEST"
echo "[qwen14b] free disk here: ${FREE_MB} MB, need ~${NEED_MB} MB for $QUANT"
if [ "$FREE_MB" -lt "$NEED_MB" ]; then
    echo "error: not enough free disk for $QUANT (need ~${NEED_MB} MB). Try a smaller quant:" >&2
    echo "       scripts/setup_qwen14b.sh q2_k" >&2
    exit 1
fi

if [ -f "$DEST" ]; then
    echo "[qwen14b] already present ($(du -h "$DEST" | cut -f1)) — skipping download."
else
    echo "[qwen14b] downloading (resumable) …"
    # -C - resumes a partial file; -L follows the HF redirect to the CDN.
    curl -L -C - --fail -o "$DEST" "$URL" || {
        echo "error: download failed. Check the repo/quant, or set REPO=… to another" >&2
        echo "       GGUF publisher. Manual: https://huggingface.co/${REPO}/tree/main" >&2
        exit 1
    }
fi

echo ""
echo "ok: $DEST ($(du -h "$DEST" | cut -f1))"
echo "next — small-model smoke test first, then 14B:"
echo "  scripts/run_sharded_mesh.sh                 # 0.5B across 2 local shards"
echo "  scripts/run_sharded_mesh.sh \"$DEST\"         # 14B once devices have disk"
