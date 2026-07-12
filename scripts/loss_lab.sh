#!/bin/bash
#
# loss_lab.sh — REAL packet loss for the transport race (Mesh 2.5).
#
# The race (Compare tab → "Run the race", or POST /api/comparison/run)
# binds every leg — NMP UDP, TCP, TCP+TLS 1.3, QUIC — on loopback ports
# 20000-39999. This script uses macOS dummynet + pf to drop (and
# optionally delay) packets on exactly that band, so a race run while
# the lab is on yields MEASURED loss-recovery behaviour per protocol:
# NMP's FEC/NACK vs TCP retransmission vs QUIC's loss recovery, as
# wall-clock transfer time. No modeling.
#
# Scope: ONLY loopback traffic to ports 20000-39999. The dashboard
# (:3000/:8080), the live mesh (ephemeral UDP ports), and everything
# else are untouched.
#
# Usage (root required — dummynet and pf are kernel facilities):
#   sudo scripts/loss_lab.sh start <loss-percent> [delay-ms]
#   sudo scripts/loss_lab.sh status
#   sudo scripts/loss_lab.sh stop
#
# Example — 5% loss, 10 ms delay, then race and compare:
#   sudo scripts/loss_lab.sh start 5 10
#   curl -s -X POST localhost:3000/api/comparison/run \
#     -d '{"prompt":"loss lab","max_tokens":16}' | python3 -m json.tool
#   sudo scripts/loss_lab.sh stop
#
set -euo pipefail

ANCHOR="nmp_loss_lab"
PIPE=1
PORTS="20000:39999"

require_root() {
  if [[ $EUID -ne 0 ]]; then
    echo "error: dummynet/pf need root — rerun with sudo" >&2
    exit 1
  fi
}

case "${1:-}" in
  start)
    require_root
    LOSS="${2:?usage: sudo $0 start <loss-percent> [delay-ms]}"
    DELAY="${3:-0}"
    PLR=$(python3 -c "print(min(max(float('$LOSS'),0),95)/100)")

    dnctl pipe $PIPE config plr "$PLR" delay "$DELAY"

    # Graft a dummynet anchor onto the CURRENT ruleset (pfctl -sr keeps
    # whatever is already loaded), then fill the anchor with the
    # port-band rule. 'quick' stops evaluation there; everything else
    # falls through untouched.
    { pfctl -sr 2>/dev/null; echo "dummynet-anchor \"$ANCHOR\""; } | pfctl -q -f -
    printf 'dummynet in quick proto udp from any to any port %s pipe %d\ndummynet in quick proto tcp from any to any port %s pipe %d\n' \
      "$PORTS" $PIPE "$PORTS" $PIPE | pfctl -q -a "$ANCHOR" -f -
    pfctl -q -e 2>/dev/null || true   # already enabled is fine

    echo "loss lab ON: ${LOSS}% loss, ${DELAY} ms delay on loopback ports ${PORTS}"
    echo "run the race now; 'sudo $0 stop' when done"
    ;;

  status)
    require_root
    echo "--- dummynet pipe $PIPE ---"
    dnctl list 2>/dev/null || echo "(no pipes configured)"
    echo "--- pf anchor $ANCHOR ---"
    pfctl -q -a "$ANCHOR" -sr 2>/dev/null || echo "(anchor empty)"
    ;;

  stop)
    require_root
    pfctl -q -a "$ANCHOR" -F rules 2>/dev/null || true
    dnctl -q flush 2>/dev/null || true
    # Restore the stock ruleset (drops our anchor hook). pf stays in
    # whatever enabled/disabled state the system had.
    pfctl -q -f /etc/pf.conf 2>/dev/null || true
    echo "loss lab OFF: shaping removed, stock pf.conf reloaded"
    ;;

  *)
    echo "usage: sudo $0 start <loss-percent> [delay-ms] | status | stop" >&2
    exit 1
    ;;
esac
