#!/bin/bash
# RLHFL Sidecar startup — runs FastAPI (admin UI + ES ingester) and trainer worker in the same container.
# uvicorn runs in the background; trainer_worker is the main process so Docker tracks its lifecycle.
set -e

# Ensure the dynamic-linker cache is current (safe no-op if already up to date).
ldconfig

# ── Intel xe GPU hang-detection tuning ───────────────────────────────────────
# force_execlist=1  switches from GuC scheduler submission to legacy execlist.
# GuC has a hard 20-second watchdog; execlist has no scheduler-level timeout.
# wedged_mode=0     suppresses automatic GPU context kill on suspected hang,
#                   letting SYCL JIT and large model dispatches run to completion.
# These parameters exist in the xe module on kernel 7.0 (confirmed from live
# listing above).  Writes may silently fail if the kernel marks them read-only
# after module load — the UR_L0_USE_IMMEDIATE_COMMANDLISTS env var is the
# primary fix; these are belt-and-braces.
_XE_PARAMS=/sys/module/xe/parameters
if [ -f "$_XE_PARAMS/force_execlist" ]; then
    echo 1 > "$_XE_PARAMS/force_execlist" 2>/dev/null && \
        echo "[start.sh] xe: force_execlist=1 (legacy execlist, no GuC timeout)" || \
        echo "[start.sh] xe: force_execlist write ignored (read-only after load)"
fi
if [ -f "$_XE_PARAMS/wedged_mode" ]; then
    echo 0 > "$_XE_PARAMS/wedged_mode" 2>/dev/null && \
        echo "[start.sh] xe: wedged_mode=0 (no auto-reset on hang)" || \
        echo "[start.sh] xe: wedged_mode write ignored (read-only after load)"
fi
unset _XE_PARAMS

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
