#!/bin/bash
set -euo pipefail

cd /app

: "${TTS_PORT:=7010}"

echo "Starting TTS service on port ${TTS_PORT} (provider=${ASTRA_TTS_PROVIDER:-xtts})"

exec uvicorn tts.main:app \
    --host 0.0.0.0 \
    --port "${TTS_PORT}" \
    --log-level info \
    --no-access-log
