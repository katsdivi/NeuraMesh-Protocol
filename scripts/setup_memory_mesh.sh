#!/usr/bin/env bash
#
# setup_memory_mesh.sh — stand up a 3-peer DISTRIBUTED-MEMORY demo on one Mac.
#
# Each "peer" here simulates a separate device. On real hardware every device
# runs its OWN Supermemory instance at localhost:6767; simulating three on one
# Mac needs three data dirs on three ports. Port plan:
#
#     peer i   Supermemory   NMP UDP   control HTTP
#     -----   -----------   -------   ------------
#       1        6767         9411        9401
#       2        6768         9412        9402
#       3        6769         9413        9403
#
# Everything lives under ~/.neuramesh-memdemo/ :
#     sm1/ sm2/ sm3/    Supermemory data dirs (each holds its OWN api-key)
#     peer1..3/         peer config.json
#     keys/             shared Noise static keys (peer<i>.key / .pub)
#     logs/  pids/      Supermemory instance logs + pidfiles
#
# This script ONLY manages the Supermemory instances + peer configs. It does
# NOT launch the Swift peers — the demo runbook (README "kill-a-peer demo")
# does that with `swift run nmp-memory-peer --config …`. Local-only: every
# config's supermemory.baseURL is localhost; a cloud endpoint is refused both
# here (grep guard) and in NMPSupermemoryConfig.
#
# Usage: setup_memory_mesh.sh {start|stop|status}
#        INSTANCES="2 3" setup_memory_mesh.sh start   # subset (testing)

set -euo pipefail

PREFIX="[memdemo]"
ROOT="$HOME/.neuramesh-memdemo"
BIN="$HOME/.supermemory/bin/supermemory-server"
INSTALL_HINT='curl -fsSL https://supermemory.ai/install | bash'
INSTANCES="${INSTANCES:-1 2 3}"

log() { echo "$PREFIX $*"; }
err() { echo "$PREFIX ERROR: $*" >&2; }

sm_port()      { echo $((6766 + $1)); }   # 1→6767 2→6768 3→6769
udp_port()     { echo $((9410 + $1)); }   # 1→9411 …
control_port() { echo $((9400 + $1)); }   # 1→9401 …

datadir() { echo "$ROOT/sm$1"; }
pidfile() { echo "$ROOT/pids/supermemory-sm$1.pid"; }
logfile() { echo "$ROOT/logs/supermemory-sm$1.log"; }

# Cheapest authenticated readiness probe (verified: no /health endpoint;
# GET /v3/documents/processing returns 200 when the instance is serving).
sm_healthy() {
  local port="$1" key="${2:-}"
  if [[ -n "$key" ]]; then
    curl -fsS -m 3 -o /dev/null \
      -H "Authorization: Bearer $key" \
      "http://localhost:$port/v3/documents/processing" 2>/dev/null
  else
    # Before we know the key, the landing page is an unauthenticated 200.
    curl -fsS -m 3 -o /dev/null "http://localhost:$port/" 2>/dev/null
  fi
}

pid_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

ensure_layout() {
  mkdir -p "$ROOT"/{pids,logs,keys}
  local i
  for i in $INSTANCES; do mkdir -p "$(datadir "$i")" "$ROOT/peer$i"; done
}

# --- start one Supermemory instance -----------------------------------------
start_instance() {
  local i="$1" port dir pf lf
  port="$(sm_port "$i")"; dir="$(datadir "$i")"
  pf="$(pidfile "$i")"; lf="$(logfile "$i")"

  # Already ours and alive?
  if [[ -f "$pf" ]] && pid_alive "$(cat "$pf")" && sm_healthy "$port"; then
    log "sm$i already running (pid $(cat "$pf"), port $port)"
    return 0
  fi

  # A FOREIGN instance is on our port (e.g. a leftover dev server whose data
  # dir is not ours). Reusing someone else's instance/key would be dishonest
  # and would scatter shards into the wrong store — refuse and instruct.
  if sm_healthy "$port" && { [[ ! -f "$pf" ]] || ! pid_alive "$(cat "$pf" 2>/dev/null)"; }; then
    err "port $port is already serving a Supermemory instance this script did"
    err "not start (likely a leftover dev server). Stop it first, e.g.:"
    err "    lsof -nP -iTCP:$port -sTCP:LISTEN      # find the pid"
    err "    kill <pid>                             # or: pkill -f supermemory-server"
    err "then re-run. (We will NOT reuse a foreign instance's data dir/key.)"
    return 1
  fi

  log "starting sm$i on port $port (data dir $dir) …"
  SUPERMEMORY_DATA_DIR="$dir" \
  PORT="$port" \
  SUPERMEMORY_DISABLE_TELEMETRY=1 \
    nohup "$BIN" >"$lf" 2>&1 &
  echo $! > "$pf"

  # First boot warms the local embedding model — be patient (up to 180s).
  local waited=0 timeout=180
  printf '%s waiting for sm%s health' "$PREFIX" "$i"
  while (( waited < timeout )); do
    if sm_healthy "$port"; then echo " — up (${waited}s)"; break; fi
    if ! pid_alive "$(cat "$pf")"; then
      echo; err "sm$i process died on boot — see $lf"; tail -5 "$lf" >&2 || true
      return 1
    fi
    printf '.'; sleep 2; waited=$((waited + 2))
  done
  if ! sm_healthy "$port"; then
    echo; err "sm$i not healthy after ${timeout}s — see $lf"; return 1
  fi

  # Each instance mints its own api-key file — assert it exists (never print).
  local keyf="$dir/api-key"
  local tries=0
  while [[ ! -s "$keyf" ]] && (( tries < 10 )); do sleep 1; tries=$((tries+1)); done
  if [[ ! -s "$keyf" ]]; then
    err "sm$i produced no api-key at $keyf"; return 1
  fi
  log "sm$i healthy; api-key at $keyf (not shown)"
}

# --- write one peer's config.json -------------------------------------------
write_config() {
  local i="$1" cfg="$ROOT/peer$i/config.json"
  local port; port="$(sm_port "$i")"
  local keyf; keyf="$(datadir "$i")/api-key"

  # Roster = the OTHER instances in $INSTANCES (full mesh).
  local peers_json="" j first=1
  for j in $INSTANCES; do
    [[ "$j" == "$i" ]] && continue
    [[ $first -eq 0 ]] && peers_json+=","
    peers_json+="{\"peerID\": $j, \"host\": \"127.0.0.1\", \"udpPort\": $(udp_port "$j")}"
    first=0
  done

  cat > "$cfg" <<JSON
{
  "peerID": $i,
  "deviceName": "demo-peer-$i",
  "udpPort": $(udp_port "$i"),
  "controlPort": $(control_port "$i"),
  "keyDir": "$ROOT/keys",
  "scheme": { "k": 2, "n": 3 },
  "peers": [ $peers_json ],
  "supermemory": {
    "baseURL": "http://localhost:$port",
    "apiKeyFile": "$keyf"
  },
  "indexSummaryChars": 160
}
JSON
  log "wrote $cfg (supermemory http://localhost:$port)"
}

# --- subcommands ------------------------------------------------------------
cmd_start() {
  [[ -x "$BIN" ]] || { err "supermemory-server not found at $BIN"; \
    err "install it first:  $INSTALL_HINT"; exit 1; }
  ensure_layout
  local i
  for i in $INSTANCES; do start_instance "$i"; done
  for i in $INSTANCES; do write_config "$i"; done

  # Belt-and-braces local-only assertion across every generated config.
  if grep -l "supermemory.ai" "$ROOT"/peer*/config.json >/dev/null 2>&1; then
    err "a generated config points at a cloud endpoint (supermemory.ai) — aborting"
    exit 1
  fi
  log "all peer configs use localhost Supermemory base URLs (no cloud) ✓"
  log "memory mesh ready. Launch peers with:"
  for i in $INSTANCES; do
    log "    swift run nmp-memory-peer --config $ROOT/peer$i/config.json"
  done
}

cmd_stop() {
  local i pid
  for i in $INSTANCES; do
    local pf; pf="$(pidfile "$i")"
    [[ -f "$pf" ]] || { log "sm$i: no pidfile (not started by us)"; continue; }
    pid="$(cat "$pf")"
    if pid_alive "$pid"; then
      log "stopping sm$i (pid $pid)"; kill "$pid" 2>/dev/null || true
      local w=0; while pid_alive "$pid" && (( w < 10 )); do sleep 1; w=$((w+1)); done
      pid_alive "$pid" && { log "sm$i still alive, SIGKILL"; kill -9 "$pid" 2>/dev/null || true; }
    else
      log "sm$i already stopped"
    fi
    rm -f "$pf"
  done
  log "stopped. Data dirs preserved under $ROOT (delete manually to reset)."
}

cmd_status() {
  local i
  for i in $INSTANCES; do
    local port pf pid state dir cfg
    port="$(sm_port "$i")"; pf="$(pidfile "$i")"; dir="$(datadir "$i")"
    cfg="$ROOT/peer$i/config.json"
    pid="$( [[ -f "$pf" ]] && cat "$pf" || echo - )"
    if sm_healthy "$port"; then state="HEALTHY"; else state="down"; fi
    echo "$PREFIX sm$i  port=$port  $state  pid=$pid  data=$dir"
    echo "$PREFIX       udp=$(udp_port "$i") control=$(control_port "$i") config=$( [[ -f "$cfg" ]] && echo present || echo MISSING )"
  done
}

case "${1:-}" in
  start)  cmd_start ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  *) echo "usage: $0 {start|stop|status}   (INSTANCES=\"1 2 3\" to subset)"; exit 2 ;;
esac
