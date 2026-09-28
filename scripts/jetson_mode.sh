#!/usr/bin/env bash
# Switch the Jetson Orin Nano between its memory modes (docs/JETSON_MODELS_PLAN.md).
#
#   sudo scripts/jetson_mode.sh install        copy jetson/systemd/* into systemd (once, and after edits)
#   sudo scripts/jetson_mode.sh conversation   robot + Qwen + Canary + Florence, loaded in order
#   sudo scripts/jetson_mode.sh build          stop every mfw service: nothing resident (for builds)
#        scripts/jetson_mode.sh status         what runs, free memory, one tegrastats sample
#
# Models are loaded once per mode, never per command. After any NvMap error 12
# or cudaMalloc out-of-memory in `journalctl -u 'mfw-*'`: reboot, do not retry.
#
# SAFETY: `conversation` starts robot_server. Its first `home` request (sent
# unprompted when run_assistant --hardware connects) attaches every servo AT
# home at full speed. Hand-pose the arm at home right before starting
# run_assistant, and keep the +6 V E-stop in reach.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${REPO}/jetson/systemd"
SYSTEMD_DIR=/etc/systemd/system
ENV_FILE=/etc/mfw/mfw.env
SERVICES=(mfw-robot mfw-llama mfw-llm-shim mfw-speech mfw-detector)
# Model services in load order; the robot server has no model.
MODEL_SERVICES=(mfw-llama mfw-speech mfw-detector)

usage() {
    # The header comment block, without its leading '# '.
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"
    exit 2
}

need_root() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "$1 needs root: sudo $0 $1" >&2
        exit 1
    fi
}

drop_caches() {
    sync
    echo 3 > /proc/sys/vm/drop_caches
    echo 1 > /proc/sys/vm/compact_memory
}

free_mb() {
    awk '/MemAvailable/ { printf "%d", $2 / 1024 }' /proc/meminfo
}

tegra_sample() {
    if command -v tegrastats > /dev/null 2>&1; then
        # tegrastats never exits on its own; take one line.
        timeout 3 tegrastats --interval 1000 2> /dev/null | head -n 1 || true
    else
        echo "(tegrastats not found: not a Jetson?)"
    fi
}

cmd_install() {
    need_root install
    local unit
    for unit in "${UNIT_DIR}"/*.service "${UNIT_DIR}"/*.target; do
        install -m 0644 "$unit" "${SYSTEMD_DIR}/$(basename "$unit")"
        echo "installed ${SYSTEMD_DIR}/$(basename "$unit")"
    done
    chmod 0755 "${UNIT_DIR}/wait_ready.sh"
    if [ ! -f "$ENV_FILE" ]; then
        install -d -m 0755 "$(dirname "$ENV_FILE")"
        install -m 0644 "${UNIT_DIR}/mfw.env.example" "$ENV_FILE"
        echo "created ${ENV_FILE} from mfw.env.example -- EDIT IT (paths, mic index, arm geometry)"
    else
        echo "kept existing ${ENV_FILE}"
    fi
    systemctl daemon-reload
    echo "units reference WorkingDirectory=/home/jetson/mfw and User=jetson; edit them if the checkout lives elsewhere"
    echo "nothing is enabled at boot on purpose; start a mode with: sudo $0 conversation"
}

cmd_conversation() {
    need_root conversation
    [ -f "$ENV_FILE" ] || { echo "missing ${ENV_FILE}: run 'sudo $0 install' first" >&2; exit 1; }
    echo "hand-pose the arm at home now: the servos attach there at full speed when robot_server starts"
    systemctl stop mfw-build.target 2> /dev/null || true
    drop_caches
    echo "MemAvailable before loading: $(free_mb) MB"
    # The target pulls every service in; each unit's After= + ExecStartPost
    # readiness wait serialises the loads llama-server -> speech -> detector.
    local unit failed=0
    systemctl start mfw-conversation.target || failed=1
    for unit in "${SERVICES[@]}"; do
        if ! systemctl is-active --quiet "${unit}.service"; then
            echo "${unit} is not running" >&2
            failed=1
        fi
    done
    cmd_status
    if [ "$failed" -ne 0 ]; then
        echo "conversation mode did not come up; see: journalctl -u 'mfw-*' -b --no-pager | tail -n 80" >&2
        exit 1
    fi
}

cmd_build() {
    need_root build
    systemctl stop mfw-conversation.target "${SERVICES[@]/%/.service}" || true
    systemctl start mfw-build.target 2> /dev/null || true
    drop_caches
    echo "nothing resident; MemAvailable $(free_mb) MB. Build with -j4 (not -j6)."
}

cmd_status() {
    local unit state
    printf '%-26s %s\n' "unit" "state"
    for unit in mfw-conversation.target mfw-build.target "${SERVICES[@]/%/.service}"; do
        state="$(systemctl is-active "$unit" 2> /dev/null || true)"
        printf '%-26s %s\n' "$unit" "${state:-unknown}"
    done
    echo "model load order: ${MODEL_SERVICES[*]}"
    echo "MemAvailable: $(free_mb) MB"
    tegra_sample
}

case "${1:-}" in
    install) cmd_install ;;
    conversation) cmd_conversation ;;
    build) cmd_build ;;
    status) cmd_status ;;
    -h|--help|help|"") usage ;;
    *) echo "unknown mode '$1'" >&2; usage ;;
esac
