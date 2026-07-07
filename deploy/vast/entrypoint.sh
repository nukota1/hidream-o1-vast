#!/usr/bin/env bash
set -euo pipefail

: "${HIDREAM_MODEL_REPO:=HiDream-ai/HiDream-O1-Image-Dev}"
: "${HIDREAM_MODEL_PATH:=/models/HiDream-O1-Image-Dev}"
: "${HIDREAM_MODEL_TYPE:=dev}"
: "${HIDREAM_WORKFLOW:=o1}"
: "${HIDREAM_HOST:=0.0.0.0}"
: "${HIDREAM_PORT:=7861}"
: "${HF_HOME:=/models/huggingface}"
: "${HIDREAM_I1_REPO_DIR:=/workspace/third_party/HiDream-I1}"
: "${HIDREAM_E11_REPO_DIR:=/workspace/third_party/HiDream-E1}"

export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

echo "[entrypoint] HiDream model repo: $HIDREAM_MODEL_REPO"
echo "[entrypoint] HiDream model path: $HIDREAM_MODEL_PATH"
echo "[entrypoint] HiDream model type: $HIDREAM_MODEL_TYPE"
echo "[entrypoint] HiDream workflow: $HIDREAM_WORKFLOW"

if [ "$HIDREAM_WORKFLOW" = "i1_e11" ]; then
  mkdir -p /workspace/third_party
  if [ ! -d "$HIDREAM_I1_REPO_DIR/.git" ]; then
    echo "[entrypoint] Cloning HiDream-I1 code..."
    git clone --depth 1 https://github.com/HiDream-ai/HiDream-I1.git "$HIDREAM_I1_REPO_DIR"
  fi
  if [ ! -d "$HIDREAM_E11_REPO_DIR/.git" ]; then
    echo "[entrypoint] Cloning HiDream-E1 code..."
    git clone --depth 1 https://github.com/HiDream-ai/HiDream-E1.git "$HIDREAM_E11_REPO_DIR"
  fi
else
  if [ ! -f "$HIDREAM_MODEL_PATH/config.json" ]; then
    echo "[entrypoint] Model not found. Downloading from Hugging Face..."
    mkdir -p "$HIDREAM_MODEL_PATH"
    huggingface-cli download "$HIDREAM_MODEL_REPO" --local-dir "$HIDREAM_MODEL_PATH"
  else
    echo "[entrypoint] Model already present. Skipping download."
  fi
fi

exec python3 app.py \
  --workflow "$HIDREAM_WORKFLOW" \
  --model_path "$HIDREAM_MODEL_PATH" \
  --model_type "$HIDREAM_MODEL_TYPE" \
  --host "$HIDREAM_HOST" \
  --port "$HIDREAM_PORT"
