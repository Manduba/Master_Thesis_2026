#!/bin/bash

cd ~/fcdm_inference || exit 1

PID_FILE="output/openstack_live/http_live_logger.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Stopping logger PID $(cat "$PID_FILE")"
    kill -INT "$(cat "$PID_FILE")"
    rm -f "$PID_FILE"
else
    echo "No running logger found."
fi