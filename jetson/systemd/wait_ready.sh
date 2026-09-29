#!/usr/bin/env bash
# Block until a service is actually ready, so systemd's After= ordering means
# "after the model is loaded", not "after the process was forked".
#
#   wait_ready.sh tcp  HOST PORT TIMEOUT_S     # a TCP port accepts connections
#   wait_ready.sh http URL       TIMEOUT_S     # an HTTP GET answers 2xx
#
# Used as ExecStartPost= in jetson/systemd/mfw-*.service. Exit 0 when ready,
# 1 on timeout (systemd then marks the unit failed and Restart= applies).
set -u

usage() {
    echo "usage: $0 tcp HOST PORT TIMEOUT_S | $0 http URL TIMEOUT_S" >&2
    exit 2
}

kind="${1:-}"
case "$kind" in
    tcp)
        [ "$#" -eq 4 ] || usage
        host="$2"; port="$3"; timeout_s="$4"
        ;;
    http)
        [ "$#" -eq 3 ] || usage
        url="$2"; timeout_s="$3"
        ;;
    *)
        usage
        ;;
esac

case "$timeout_s" in
    ''|*[!0-9]*) echo "TIMEOUT_S must be a whole number of seconds, got '$timeout_s'" >&2; exit 2 ;;
esac

deadline=$(( $(date +%s) + timeout_s ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    if [ "$kind" = "tcp" ]; then
        # bash's /dev/tcp: no netcat needed on a stock JetPack image.
        if (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null; then
            echo "ready: tcp ${host}:${port}"
            exit 0
        fi
    elif command -v curl > /dev/null 2>&1; then
        if curl -fsS -o /dev/null --max-time 2 "$url" 2>/dev/null; then
            echo "ready: ${url}"
            exit 0
        fi
    else
        # No curl on a minimal image: python3 is always there on JetPack.
        if python3 -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)' "$url" 2>/dev/null; then
            echo "ready: ${url}"
            exit 0
        fi
    fi
    sleep 1
done

if [ "$kind" = "tcp" ]; then
    echo "not ready after ${timeout_s} s: tcp ${host}:${port}" >&2
else
    echo "not ready after ${timeout_s} s: ${url}" >&2
fi
exit 1
