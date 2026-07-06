#!/usr/bin/env bash
set -euo pipefail

: "${HIDREAM_MODEL_REPO:=HiDream-ai/HiDream-O1-Image-Dev}"
: "${HIDREAM_MODEL_PATH:=/models/HiDream-O1-Image-Dev}"
: "${HIDREAM_MODEL_TYPE:=dev}"
: "${HIDREAM_HOST:=0.0.0.0}"
: "${HIDREAM_PORT:=7861}"
: "${HF_HOME:=/models/huggingface}"

export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

echo "[entrypoint] HiDream model repo: $HIDREAM_MODEL_REPO"
echo "[entrypoint] HiDream model path: $HIDREAM_MODEL_PATH"
echo "[entrypoint] HiDream model type: $HIDREAM_MODEL_TYPE"

if [ ! -f "$HIDREAM_MODEL_PATH/config.json" ]; then
  echo "[entrypoint] Model not found. Downloading from Hugging Face..."
  mkdir -p "$HIDREAM_MODEL_PATH"
  huggingface-cli download "$HIDREAM_MODEL_REPO" --local-dir "$HIDREAM_MODEL_PATH"
else
  echo "[entrypoint] Model already present. Skipping download."
fi

exec python3 app.py \
  --model_path "$HIDREAM_MODEL_PATH" \
  --model_type "$HIDREAM_MODEL_TYPE" \
  --host "$HIDREAM_HOST" \
  --port "$HIDREAM_PORT"
