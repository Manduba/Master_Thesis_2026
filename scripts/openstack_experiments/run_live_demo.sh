#!/bin/bash

cd ~/fcdm_inference || exit 1

mkdir -p output/openstack_live

LOGGER_LOG="output/openstack_live/http_live_logger.log"
PID_FILE="output/openstack_live/http_live_logger.pid"

BASELINE_SECONDS=300
SIEGE_TIME="55M"
COOLDOWN_SECONDS=600
SIEGE_URL="http://127.0.0.1/"

# Start logger only if not already running
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Logger already running with PID $(cat "$PID_FILE")"
else
    echo "[0] Starting continuous live logger..."
    nohup ~/fcdm_inference/.venv/bin/python log_openstack_http_live_metrics.py > "$LOGGER_LOG" 2>&1 &
    LOGGER_PID=$!
    echo "$LOGGER_PID" > "$PID_FILE"
    echo "Logger PID: $LOGGER_PID"
fi

CYCLE=0

while true; do
    CYCLE=$((CYCLE + 1))
    echo
    echo "=== Live demo cycle $CYCLE started at $(date) ==="

    echo "[1/3] Baseline for ${BASELINE_SECONDS}s..."
    sleep "$BASELINE_SECONDS"

    echo "[2/3] Traffic generation for $SIEGE_TIME ..."
    siege -c 50 -t "$SIEGE_TIME" "$SIEGE_URL" || true

    echo "[3/3] Cooldown for ${COOLDOWN_SECONDS}s..."
    sleep "$COOLDOWN_SECONDS"
done