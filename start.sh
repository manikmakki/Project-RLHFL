#!/bin/bash
# RLHFL Sidecar startup — runs FastAPI (admin UI + ES ingester) and trainer worker in the same container.
# uvicorn runs in the background; trainer_worker is the main process so Docker tracks its lifecycle.
set -e

cd /app

echo "[start.sh] Starting RLHFL sidecar..."
echo "[start.sh] Config: ${CONFIG_PATH:-/config/system_config.yaml}"

# Start API server in background
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 &
API_PID=$!
echo "[start.sh] API server PID: $API_PID"

# Wait briefly for the API to be ready before trainer starts (trainer calls localhost:8000 on completion)
sleep 3

# Start trainer worker as foreground process
exec python -m trainer.trainer_worker
