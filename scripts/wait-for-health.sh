#!/usr/bin/env bash
set -eo pipefail

API_PORT="${API_PORT:-8080}"
TIMEOUT=600
INTERVAL=5
ELAPSED=0

echo "[health-check] Waiting for inference server on port $API_PORT (timeout: ${TIMEOUT}s)..."

while [ "$ELAPSED" -lt "$TIMEOUT" ]; do
    if curl -sf "http://localhost:${API_PORT}/v1/models" > /dev/null 2>&1; then
        echo ""
        echo "[health-check] Server is healthy after ${ELAPSED}s."
        exit 0
    fi
    printf "."
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo ""
echo "[health-check] Timed out after ${TIMEOUT}s waiting for server on port $API_PORT."
exit 1
