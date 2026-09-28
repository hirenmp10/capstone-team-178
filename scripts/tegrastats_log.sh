#!/usr/bin/env bash
# Log tegrastats to logs/tegrastats_<timestamp>.txt, one wall-clock-stamped
# line per sample, with a header saying which mfw services were up.
#
#   scripts/tegrastats_log.sh                 # 1 s samples until Ctrl-C
#   scripts/tegrastats_log.sh 500 120         # 500 ms samples for 120 s
#   scripts/tegrastats_log.sh 1000 60 idle    # label the run (goes in the file name)
#
# The day-1 measurements the plan needs: idle (`build` mode), then each model
# service up (`jetson_mode.sh conversation`), then during the demo.
set -euo pipefail

INTERVAL_MS="${1:-1000}"
DURATION_S="${2:-0}"
LABEL="${3:-}"

case "$INTERVAL_MS" in ''|*[!0-9]*) echo "interval must be whole milliseconds, got '$INTERVAL_MS'" >&2; exit 2 ;; esac
case "$DURATION_S" in ''|*[!0-9]*) echo "duration must be whole seconds (0 = until Ctrl-C), got '$DURATION_S'" >&2; exit 2 ;; esac
case "$LABEL" in *[!A-Za-z0-9_-]*) echo "label may use letters, digits, - and _ only, got '$LABEL'" >&2; exit 2 ;; esac

if ! command -v tegrastats > /dev/null 2>&1; then
    echo "tegrastats not found: run this on the Jetson (JetPack provides /usr/bin/tegrastats)" >&2
    exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO}/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${LOG_DIR}/tegrastats_${STAMP}${LABEL:+_${LABEL}}.txt"

{
    echo "# tegrastats log ${STAMP}  interval=${INTERVAL_MS}ms duration=${DURATION_S}s label=${LABEL:-none}"
    echo "# host: $(uname -n)  kernel: $(uname -r)"
    if [ -r /etc/nv_tegra_release ]; then
        echo "# l4t: $(head -n 1 /etc/nv_tegra_release)"
    fi
    for unit in mfw-robot mfw-llama mfw-llm-shim mfw-speech mfw-detector; do
        echo "# ${unit}: $(systemctl is-active "${unit}.service" 2> /dev/null || true)"
    done
    echo "# MemAvailable: $(awk '/MemAvailable/ { printf "%d", $2 / 1024 }' /proc/meminfo) MB"
} > "$OUT"

echo "logging to ${OUT} (Ctrl-C to stop)"

stamp_lines() {
    while IFS= read -r line; do
        printf '%s %s\n' "$(date +%H:%M:%S)" "$line"
    done
}

if [ "$DURATION_S" -gt 0 ]; then
    # timeout ends tegrastats; its exit status 124 is the expected way out.
    timeout "$DURATION_S" tegrastats --interval "$INTERVAL_MS" | stamp_lines >> "$OUT" || true
else
    trap 'echo; echo "stopped; wrote ${OUT}"' INT
    tegrastats --interval "$INTERVAL_MS" | stamp_lines >> "$OUT" || true
fi

echo "wrote $(grep -vc '^#' "$OUT" || true) sample(s) to ${OUT}"
