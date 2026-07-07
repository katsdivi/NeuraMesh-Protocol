#!/bin/bash
#
# setup_mesh_test.sh — one-command mesh demo on a single Mac.
#
# Builds the package, starts an nmp-peer in the background, runs the
# coordinator benchmark against it over REAL UDP + Bonjour (loopback
# network, full Noise handshake, chunked tensors), then cleans up.
#
# This is the zero-extra-hardware sanity check. For the Mac + iPhone
# mesh, follow Docs/CrossDevice_Setup_Guide.md.
#
# Usage:
#   scripts/setup_mesh_test.sh                # quick demo (16×512, fast)
#   scripts/setup_mesh_test.sh --realistic    # 32 layers, 4096 hidden,
#                                             # 5 ms/layer emulated compute
set -euo pipefail
cd "$(dirname "$0")/.."

LAYERS=16 HIDDEN=512 SLOW=0 TOKENS=8 RUNS=3
if [[ "${1:-}" == "--realistic" ]]; then
    LAYERS=32 HIDDEN=4096 SLOW=5 TOKENS=4 RUNS=3
fi

echo "==> building (swift build)"
swift build

PEER_LOG="$(mktemp -t nmp-peer-log)"
echo "==> starting background peer (log: $PEER_LOG)"
./.build/debug/nmp-peer --layers "$LAYERS" --hidden "$HIDDEN" --slow "$SLOW" \
    > "$PEER_LOG" 2>&1 &
PEER_PID=$!
trap 'kill "$PEER_PID" 2>/dev/null || true' EXIT
sleep 2

echo "==> running coordinator benchmark"
./.build/debug/nmp-coordinator --peers 1 --layers "$LAYERS" --hidden "$HIDDEN" \
    --slow "$SLOW" --runs "$RUNS" --tokens "$TOKENS" --wait 30

echo
echo "==> peer log tail:"
tail -n 8 "$PEER_LOG"
echo
echo "Mesh demo complete. Next: add your iPhone as a peer —"
echo "see Docs/CrossDevice_Setup_Guide.md"
