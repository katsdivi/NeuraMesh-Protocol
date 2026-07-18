#!/usr/bin/env bash
#
# run_memory_demo.sh — drive the distributed-memory "kill-a-peer" demo.
#
# Assumes:
#   - three Supermemory instances are up and three peer configs exist
#     (run scripts/setup_memory_mesh.sh start first), AND
#   - the three peers are running:
#         swift run nmp-memory-peer --config ~/.neuramesh-memdemo/peer1/config.json
#         swift run nmp-memory-peer --config ~/.neuramesh-memdemo/peer2/config.json
#         swift run nmp-memory-peer --config ~/.neuramesh-memdemo/peer3/config.json
#     (or use --launch here to background them from .build/debug/nmp-memory-peer)
#
# This script writes a memory, proves each peer holds only ONE opaque shard,
# recalls it, then KILLS one peer for real and recalls again from a survivor —
# the K-of-2 quorum still reconstructs. With --kill-two it also shows the
# below-quorum path fail EXPLICITLY (never a silent wrong answer).
#
# It changes no on-disk state beyond the memory it writes and (optionally) the
# peer processes it launches/kills. Control APIs are loopback only.

set -uo pipefail
ROOT="$HOME/.neuramesh-memdemo"
BIN="$(cd "$(dirname "$0")/.." && pwd)/.build/debug/nmp-memory-peer"
C1=9401; C2=9402; C3=9403
LAUNCH=0; KILL_TWO=0
for arg in "$@"; do
  case "$arg" in
    --launch)   LAUNCH=1 ;;
    --kill-two) KILL_TWO=1 ;;
    *) echo "usage: $0 [--launch] [--kill-two]"; exit 2 ;;
  esac
done

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
ctl() { curl -s -m 12 "$@"; }   # control-API call

peer_pid() { pgrep -f "nmp-memory-peer --config.*peer$1" | head -1; }

launch_peers() {
  [[ -x "$BIN" ]] || { echo "build first: swift build"; exit 1; }
  mkdir -p "$ROOT/peerlogs"
  for i in 1 2 3; do
    if [[ -z "$(peer_pid "$i")" ]]; then
      nohup "$BIN" --config "$ROOT/peer$i/config.json" \
        > "$ROOT/peerlogs/peer$i.log" 2>&1 &
      echo "  launched peer$i (pid $!)"
    else
      echo "  peer$i already running (pid $(peer_pid "$i"))"
    fi
  done
  echo "  waiting for the mesh to form…"; sleep 10
}

require_mesh() {
  local up; up=$(ctl "http://127.0.0.1:$C1/status" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(sum(1 for l in d['links'] if l['established']))" 2>/dev/null)
  if [[ "${up:-0}" -lt 2 ]]; then
    echo "peer1 does not see 2 established links yet (saw ${up:-0}). Are all three peers running?"
    echo "Tip: rerun with --launch, or start the peers manually, then retry."
    exit 1
  fi
  echo "  peer1 sees $up established NMP links ✓"
}

[[ "$LAUNCH" == 1 ]] && { say "Launching peers"; launch_peers; }

say "1. Mesh check"
require_mesh

say "2. Write a memory on peer1 (seal → shard 2-of-3 → distribute)"
Q="wine cellar combination barolo hendersons"
RESP=$(ctl -X POST "http://127.0.0.1:$C1/remember" -H 'Content-Type: application/json' \
  -d '{"title":"Wine cellar combo","content":"Remember: the wine cellar lock combination is 47-19-33, the spare key is taped under the third shelf, and we promised the Hendersons two bottles of the 2011 Barolo for helping move the piano."}')
echo "$RESP" | python3 -c "import sys,json;d=json.load(sys.stdin);print('  memoryID:',d['memoryID']);print('  distributed:',d['distributed'],'  shardAcks:',d['shardAcks']);print('  bytes:',d['bytes']);print('  note:',d['note'])"

say "3. Each peer holds exactly ONE opaque shard"
for i in 1 2 3; do
  port=$((9400+i))
  ctl "http://127.0.0.1:$port/status" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(f'  peer$i: shardsHeld={d[\"shardsHeld\"]}  store={d[\"store\"][\"kind\"]}@{d[\"store\"][\"locality\"]} localOnly={d[\"store\"][\"localOnly\"]}')"
done

say "4. Baseline recall (all peers alive)"
ctl "http://127.0.0.1:$C1/recall?q=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$Q")" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('  content:',d['content'][:72],'…');print('  shardsUsed:',[s['source'] for s in d['shardsUsed']],' matchedVia:',d['matchedVia'])"

say "5. KILL peer 3 (real process)"
PID3=$(peer_pid 3)
if [[ -n "$PID3" ]]; then echo "  kill -9 $PID3"; kill -9 "$PID3"; else echo "  peer3 not found (already down?)"; fi
echo "  waiting ~15s for peer1 to retire the dead link…"; sleep 16
ctl "http://127.0.0.1:$C1/status" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('  peer1 links:',[(l['peerID'],'up' if l['established'] else 'DOWN') for l in d['links']])"

say "6. Recall from surviving peer 1 — must STILL reconstruct (quorum = 2)"
ctl "http://127.0.0.1:$C1/recall?q=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$Q")" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);
print('  content:',d['content'][:72],'…');
print('  shardsUsed:',[s['source'] for s in d['shardsUsed']]);
print('  unreachablePeers:',d.get('unreachablePeers'),' peersNotNeeded:',d.get('peersNotNeeded'));
print('  integrity:',d['integrity'])"
echo "  ^ peer 3 was killed for real; the memory survived."

if [[ "$KILL_TWO" == 1 ]]; then
  say "7. Kill peer 2 too — only 1 shard reachable, below quorum → explicit failure"
  PID2=$(peer_pid 2); [[ -n "$PID2" ]] && { echo "  kill -9 $PID2"; kill -9 "$PID2"; }
  echo "  waiting ~15s…"; sleep 16
  code=$(curl -s -o /tmp/memdemo_fail.json -w "%{http_code}" -m 12 \
    "http://127.0.0.1:$C1/recall?q=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$Q")")
  echo "  HTTP $code"
  python3 -c "import json;print('  error:',json.load(open('/tmp/memdemo_fail.json')).get('error'))"
  echo "  ^ explicit quorum failure — never a silent wrong answer."
fi

say "Demo complete"
echo "Teardown: kill the remaining peers, then scripts/setup_memory_mesh.sh stop"
