#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <config.yaml> [gpu_id ...]" >&2
  exit 1
fi

CONFIG="$1"
shift || true

if [[ $# -eq 0 ]]; then
  GPUS=(2 3 4 5 6 7)
else
  GPUS=("$@")
fi

mkdir -p logs
PIDS=()

for GPU in "${GPUS[@]}"; do
  WORKER_ID="$(basename "$CONFIG" .yaml)-gpu${GPU}"
  LOG_PATH="logs/${WORKER_ID}.log"
  echo "[stage7] launching ${WORKER_ID} on cuda:${GPU} -> ${LOG_PATH}"
  dynalloc run \
    --config "$CONFIG" \
    --phase comparison \
    --worker-id "$WORKER_ID" \
    --device "cuda:${GPU}" \
    >"$LOG_PATH" 2>&1 &
  PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    STATUS=1
  fi
done

exit "$STATUS"
