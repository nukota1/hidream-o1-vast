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
: "${JANKU_MODEL_PATH:=/models/checkpoints/janku-v6.safetensors}"
: "${JANKU_MODEL_MIN_BYTES:=6000000000}"
: "${SDXL_DOWNLOAD_ON_START:=1}"
: "${QWEN_IMAGE_EDIT_MODEL_REPO:=Qwen/Qwen-Image-Edit-2511}"
: "${QWEN_IMAGE_EDIT_DOWNLOAD_ON_START:=1}"
: "${MODEL_DOWNLOAD_RETRIES:=30}"
: "${MODEL_DOWNLOAD_RETRY_DELAY:=10}"
: "${MODEL_DOWNLOAD_CONNECT_TIMEOUT:=60}"
: "${MODEL_DOWNLOAD_LOW_SPEED_TIME:=900}"
: "${MODEL_DOWNLOAD_LOW_SPEED_LIMIT:=1024}"

export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

echo "[entrypoint] HiDream model repo: $HIDREAM_MODEL_REPO"
echo "[entrypoint] HiDream model path: $HIDREAM_MODEL_PATH"
echo "[entrypoint] HiDream model type: $HIDREAM_MODEL_TYPE"
echo "[entrypoint] HiDream workflow: $HIDREAM_WORKFLOW"

file_size_bytes() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo 0
    return 0
  fi
  stat -c%s "$path" 2>/dev/null || wc -c < "$path"
}

valid_download() {
  local path="$1"
  local min_bytes="${2:-0}"
  if [ ! -f "$path" ]; then
    return 1
  fi
  if [ "$min_bytes" -gt 0 ]; then
    local actual_bytes
    actual_bytes="$(file_size_bytes "$path")"
    if [ "$actual_bytes" -lt "$min_bytes" ]; then
      echo "[entrypoint] Incomplete model file detected: $path (${actual_bytes}/${min_bytes} bytes)."
      return 1
    fi
  fi
  return 0
}

curl_download() {
  local url="$1"
  local output="$2"
  shift 2
  curl -fL \
    --retry "$MODEL_DOWNLOAD_RETRIES" \
    --retry-all-errors \
    --retry-delay "$MODEL_DOWNLOAD_RETRY_DELAY" \
    --connect-timeout "$MODEL_DOWNLOAD_CONNECT_TIMEOUT" \
    --speed-time "$MODEL_DOWNLOAD_LOW_SPEED_TIME" \
    --speed-limit "$MODEL_DOWNLOAD_LOW_SPEED_LIMIT" \
    --continue-at - \
    "$@" \
    "$url" \
    -o "$output"
}

download_file() {
  local url="$1"
  local path="$2"
  local min_bytes="${3:-0}"
  local tmp_path="${path}.part"
  if [ -z "$url" ]; then
    return 0
  fi
  if valid_download "$path" "$min_bytes"; then
    echo "[entrypoint] Model already present: $path"
    return 0
  fi
  rm -f "$path"
  echo "[entrypoint] Downloading model to $path"
  mkdir -p "$(dirname "$path")"
  if [ -n "${CIVITAI_TOKEN:-}" ]; then
    if curl_download "$url" "$tmp_path" -H "Authorization: Bearer $CIVITAI_TOKEN"; then
      if ! valid_download "$tmp_path" "$min_bytes"; then
        echo "[entrypoint] Downloaded file is smaller than expected: $tmp_path"
        return 1
      fi
      mv "$tmp_path" "$path"
      echo "[entrypoint] Download complete: $path"
      return 0
    fi
    echo "[entrypoint] Bearer-token download failed. Retrying with Civitai token query..."
    local sep="?"
    case "$url" in
      *\?*) sep="&" ;;
    esac
    if ! curl_download "${url}${sep}token=${CIVITAI_TOKEN}" "$tmp_path"; then
      rm -f "$tmp_path"
      return 1
    fi
  else
    if ! curl_download "$url" "$tmp_path"; then
      rm -f "$tmp_path"
      return 1
    fi
  fi
  if ! valid_download "$tmp_path" "$min_bytes"; then
    echo "[entrypoint] Downloaded file is smaller than expected: $tmp_path"
    rm -f "$tmp_path"
    return 1
  fi
  mv "$tmp_path" "$path"
  echo "[entrypoint] Download complete: $path"
}

prefetch_sdxl_models() {
  set +e
  echo "[entrypoint] Background model prefetch started."
  export JANKU_MODEL_PATH
  download_file "${JANKU_MODEL_URL:-}" "$JANKU_MODEL_PATH" "$JANKU_MODEL_MIN_BYTES"
  local janku_status=$?
  if [ "$janku_status" -ne 0 ]; then
    echo "[entrypoint] JANKU prefetch failed with exit code $janku_status."
  fi
  if [ "$QWEN_IMAGE_EDIT_DOWNLOAD_ON_START" = "1" ]; then
    echo "[entrypoint] Downloading Qwen image edit model cache: $QWEN_IMAGE_EDIT_MODEL_REPO"
    hf download "$QWEN_IMAGE_EDIT_MODEL_REPO"
    local qwen_status=$?
    if [ "$qwen_status" -ne 0 ]; then
      echo "[entrypoint] Qwen image edit prefetch failed with exit code $qwen_status."
    fi
  fi
  echo "[entrypoint] Background model prefetch finished."
}

if [ "$HIDREAM_WORKFLOW" = "sdxl_janku" ]; then
  if [ "$SDXL_DOWNLOAD_ON_START" = "1" ]; then
    echo "[entrypoint] SDXL/JANKU workflow enabled. Model files will be prefetched in the background."
    prefetch_sdxl_models &
  else
    echo "[entrypoint] SDXL/JANKU workflow enabled. Model files will be downloaded on first use."
  fi
elif [ "$HIDREAM_WORKFLOW" = "i1_e11" ]; then
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
    hf download "$HIDREAM_MODEL_REPO" --local-dir "$HIDREAM_MODEL_PATH"
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
