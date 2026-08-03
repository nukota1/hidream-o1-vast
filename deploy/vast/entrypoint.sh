#!/usr/bin/env bash
set -euo pipefail

: "${APP_HOST:=0.0.0.0}"
: "${APP_PORT:=7861}"
: "${HF_HOME:=/models/huggingface}"
: "${U2NET_HOME:=/models/rembg}"
: "${LORA_ROOT:=/models/loras}"
: "${IMAGE_MODEL_FAMILY:=animagine}"
if [ "$IMAGE_MODEL_FAMILY" = "animagine" ]; then
  : "${APP_NAME:=Animagine Image Studio}"
  : "${IMAGE_MODEL_PROFILE:=sdxl-animagine-zero}"
  : "${IMAGE_MODEL_LABEL:=Animagine XL 4.0 Zero}"
else
  : "${APP_NAME:=JANKU Image Studio}"
  : "${IMAGE_MODEL_PROFILE:=sdxl-janku-v777}"
  : "${IMAGE_MODEL_LABEL:=JANKU v7.77}"
fi
: "${JANKU_MODEL_PATH:=/models/checkpoints/JANKUTrainedChenkinNoobai_v777.safetensors}"
: "${JANKU_MODEL_MIN_BYTES:=6900000000}"
: "${JANKU_R2_BUCKET:=ai-model-cache}"
: "${JANKU_R2_KEY:=models/JANKUTrainedChenkinNoobai_v777.safetensors}"
: "${ANIMAGINE_MODEL_REPO:=cagliostrolab/animagine-xl-4.0-zero}"
: "${ANIMAGINE_MODEL_CONFIG:=cagliostrolab/animagine-xl-4.0-zero}"
: "${ANIMAGINE_MODEL_FILE:=animagine-xl-4.0-zero.safetensors}"
: "${ANIMAGINE_MODEL_PATH:=/models/checkpoints/animagine-xl-4.0-zero.safetensors}"
: "${ANIMAGINE_MODEL_MIN_BYTES:=6900000000}"
: "${MODEL_DOWNLOAD_ON_START:=1}"
: "${WAIFU_INPAINT_MODEL:=ShinoharaHare/Waifu-Inpaint-XL}"
: "${WAIFU_INPAINT_DOWNLOAD_ON_START:=0}"
: "${IP_ADAPTER_MODEL:=h94/IP-Adapter}"
: "${IP_ADAPTER_SUBFOLDER:=sdxl_models}"
: "${IP_ADAPTER_WEIGHT_NAME:=ip-adapter-plus_sdxl_vit-h.safetensors}"
: "${IP_ADAPTER_IMAGE_ENCODER_FOLDER:=models/image_encoder}"
: "${IP_ADAPTER_DOWNLOAD_ON_START:=1}"
: "${FLUX_KONTEXT_MODEL:=black-forest-labs/FLUX.1-Kontext-dev}"
: "${HIDREAM_O1_IMAGE_MODEL:=HiDream-ai/HiDream-O1-Image}"
: "${HIDREAM_O1_IMAGE_PATH:=/models/HiDream-O1-Image}"
: "${PROMPT_REFINER_MODEL:=Qwen/Qwen3.5-9B}"
: "${ANIME_SEGMENTATION_MODEL:=isnet-anime}"
: "${ANIME_SEGMENTATION_DOWNLOAD_ON_START:=0}"
: "${PROMPT_REFINER_DOWNLOAD_ON_START:=1}"
: "${RUNTIME_SETUP_ON_START:=1}"
: "${HIDREAM_RUNTIME_SETUP_ON_START:=0}"
: "${PYTORCH_INDEX_URL:=https://download.pytorch.org/whl/cu128}"
: "${PYTORCH_VERSION:=2.7.1}"
: "${TORCHVISION_VERSION:=0.22.1}"
: "${PROMPT_REFINER_LOCAL_FILES_ONLY:=1}"
: "${PROMPT_REFINER_USE_KERNELS:=0}"
: "${HF_HUB_DISABLE_XET:=1}"
: "${HF_HUB_DOWNLOAD_TIMEOUT:=120}"
: "${HF_DOWNLOAD_MAX_ATTEMPTS:=5}"
: "${HF_DOWNLOAD_RETRY_SECONDS:=20}"

export APP_NAME IMAGE_MODEL_FAMILY IMAGE_MODEL_LABEL IMAGE_MODEL_PROFILE
export ANIMAGINE_MODEL_REPO ANIMAGINE_MODEL_CONFIG ANIMAGINE_MODEL_FILE ANIMAGINE_MODEL_PATH ANIMAGINE_MODEL_MIN_BYTES
export HF_HOME U2NET_HOME LORA_ROOT ANIME_SEGMENTATION_MODEL PROMPT_REFINER_LOCAL_FILES_ONLY PROMPT_REFINER_USE_KERNELS
export HF_HUB_DISABLE_XET HF_HUB_DOWNLOAD_TIMEOUT

pip_install_with_retry() {
  local attempt=1
  local max_attempts=5

  until python3 -m pip install --no-cache-dir --retries 10 --timeout 120 "$@"; do
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "[entrypoint] pip install failed after ${max_attempts} attempts."
      return 1
    fi
    echo "[entrypoint] pip install interrupted; retrying in $((attempt * 20)) seconds (${attempt}/${max_attempts})."
    sleep "$((attempt * 20))"
    attempt=$((attempt + 1))
  done
}

hf_download_with_retry() {
  local attempt=1
  local max_attempts="$HF_DOWNLOAD_MAX_ATTEMPTS"
  local retry_seconds="$HF_DOWNLOAD_RETRY_SECONDS"

  case "$max_attempts" in
    ''|*[!0-9]*) max_attempts=5 ;;
  esac
  case "$retry_seconds" in
    ''|*[!0-9]*) retry_seconds=20 ;;
  esac
  if [ "$max_attempts" -lt 1 ]; then
    max_attempts=1
  fi

  until hf download "$@"; do
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "[entrypoint] Hugging Face download failed after ${max_attempts} attempts."
      return 1
    fi
    echo "[entrypoint] Hugging Face download interrupted; retrying in $((attempt * retry_seconds)) seconds (${attempt}/${max_attempts})."
    sleep "$((attempt * retry_seconds))"
    attempt=$((attempt + 1))
  done
}

runtime_dependencies_ready() {
  python3 -c '
import torch, transformers, diffusers, flask, boto3, rembg, peft, safetensors
from transformers import CLIPTextModel
assert torch.cuda.is_available()
assert transformers.__version__ == "5.14.1"
assert diffusers.__version__ == "0.39.0"
' >/dev/null 2>&1
}

install_hidream_runtime() {
  if [ ! -d /opt/hidream-o1-image/.git ]; then
    echo "[entrypoint] Downloading optional HiDream editor source."
    if ! git clone --depth 1 https://github.com/HiDream-ai/HiDream-O1-Image.git /opt/hidream-o1-image; then
      echo "[entrypoint] Optional HiDream source download failed; standard generation will continue."
    fi
  fi
  if [ -d /opt/hidream-o1-image/.git ]; then
    sed -E '/^(torch|torchvision|transformers|flash-attn)/d' /opt/hidream-o1-image/requirements.txt > /tmp/hidream-requirements.txt
    if ! pip_install_with_retry -r /tmp/hidream-requirements.txt; then
      echo "[entrypoint] Optional HiDream dependencies failed; standard generation will continue."
    fi
    sed -i 's/"use_flash_attn": True/"use_flash_attn": False/' /opt/hidream-o1-image/models/pipeline.py
  fi
}

install_runtime_dependencies() {
  if runtime_dependencies_ready; then
    echo "[entrypoint] Python and CUDA dependencies are already installed."
  else
    echo "[entrypoint] Installing PyTorch CUDA ${PYTORCH_VERSION} and application dependencies."
    echo "[entrypoint] This runs only on a fresh disk and can take several minutes."
    pip_install_with_retry --upgrade pip
    pip_install_with_retry \
      --index-url "$PYTORCH_INDEX_URL" \
      "torch==${PYTORCH_VERSION}+cu128" \
      "torchvision==${TORCHVISION_VERSION}+cu128"
    pip_install_with_retry -r /workspace/janku-image-studio/requirements-docker.txt

    runtime_dependencies_ready
  fi

  if [ "$HIDREAM_RUNTIME_SETUP_ON_START" = "1" ]; then
    install_hidream_runtime
  else
    echo "[entrypoint] Optional HiDream runtime setup is disabled."
  fi
}

file_size_bytes() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo 0
    return 0
  fi
  stat -c%s "$path" 2>/dev/null || wc -c < "$path"
}

janku_model_ready() {
  if [ ! -f "$JANKU_MODEL_PATH" ]; then
    return 1
  fi
  local actual_bytes
  actual_bytes="$(file_size_bytes "$JANKU_MODEL_PATH")"
  [ "$actual_bytes" -ge "$JANKU_MODEL_MIN_BYTES" ]
}

download_janku_model() {
  if janku_model_ready; then
    echo "[entrypoint] JANKU model already present: $JANKU_MODEL_PATH"
    return 0
  fi
  if [ -z "${R2_ENDPOINT_URL:-}" ] || [ -z "${R2_ACCESS_KEY_ID:-}" ] || [ -z "${R2_SECRET_ACCESS_KEY:-}" ]; then
    echo "[entrypoint] R2 credentials are required to download JANKU."
    return 1
  fi

  rm -f "$JANKU_MODEL_PATH" "${JANKU_MODEL_PATH}.part"
  python3 /workspace/janku-image-studio/deploy/vast/download_r2_object.py \
    --bucket "$JANKU_R2_BUCKET" \
    --key "$JANKU_R2_KEY" \
    --output "$JANKU_MODEL_PATH" \
    --min-bytes "$JANKU_MODEL_MIN_BYTES"
}

animagine_model_ready() {
  if [ ! -f "$ANIMAGINE_MODEL_PATH" ]; then
    return 1
  fi
  local actual_bytes
  actual_bytes="$(file_size_bytes "$ANIMAGINE_MODEL_PATH")"
  [ "$actual_bytes" -ge "$ANIMAGINE_MODEL_MIN_BYTES" ]
}

download_animagine_model() {
  local failure_marker="${ANIMAGINE_MODEL_PATH}.download_failed"
  rm -f "$failure_marker"
  if animagine_model_ready; then
    echo "[entrypoint] Animagine model already present: $ANIMAGINE_MODEL_PATH"
    return 0
  fi
  echo "[entrypoint] Downloading $IMAGE_MODEL_LABEL from Hugging Face"
  mkdir -p "$(dirname "$ANIMAGINE_MODEL_PATH")"
  if ! hf_download_with_retry "$ANIMAGINE_MODEL_REPO" "$ANIMAGINE_MODEL_FILE" \
    --local-dir "$(dirname "$ANIMAGINE_MODEL_PATH")"; then
    printf '%s\n' \
      "Animagine download failed after retries. Check the Vast.ai instance log and Hugging Face connectivity." \
      > "$failure_marker"
    return 1
  fi
  if ! animagine_model_ready; then
    echo "[entrypoint] Animagine download did not produce a complete model file."
    printf '%s\n' \
      "Animagine download completed without the configured model file. Check ANIMAGINE_MODEL_REPO, ANIMAGINE_MODEL_FILE, and ANIMAGINE_MODEL_PATH." \
      > "$failure_marker"
    return 1
  fi
  rm -f "$failure_marker"
  echo "[entrypoint] Animagine model download complete: $ANIMAGINE_MODEL_PATH"
}

download_image_model() {
  if [ "$IMAGE_MODEL_FAMILY" = "animagine" ]; then
    download_animagine_model
  else
    download_janku_model
  fi
}

prefetch_models() {
  set +e
  echo "[entrypoint] Background model prefetch started."
  download_image_model
  if [ $? -ne 0 ]; then
    echo "[entrypoint] Image-model prefetch failed."
  fi

  if [ "$WAIFU_INPAINT_DOWNLOAD_ON_START" = "1" ]; then
    echo "[entrypoint] Prefetching default editor: $WAIFU_INPAINT_MODEL"
    hf_download_with_retry "$WAIFU_INPAINT_MODEL"
    if [ $? -ne 0 ]; then
      echo "[entrypoint] Waifu-Inpaint-XL prefetch failed. Check HF_TOKEN and model access approval."
    fi
  fi

  if [ "$IP_ADAPTER_DOWNLOAD_ON_START" = "1" ]; then
    echo "[entrypoint] Prefetching character reference adapter: $IP_ADAPTER_MODEL"
    hf_download_with_retry "$IP_ADAPTER_MODEL" \
      "${IP_ADAPTER_SUBFOLDER}/${IP_ADAPTER_WEIGHT_NAME}" \
      "${IP_ADAPTER_IMAGE_ENCODER_FOLDER}/config.json" \
      "${IP_ADAPTER_IMAGE_ENCODER_FOLDER}/model.safetensors"
    if [ $? -ne 0 ]; then
      echo "[entrypoint] Character reference adapter prefetch failed."
    fi
  fi

  if [ "$PROMPT_REFINER_DOWNLOAD_ON_START" = "1" ]; then
    echo "[entrypoint] Prefetching prompt refiner: $PROMPT_REFINER_MODEL"
    hf_download_with_retry "$PROMPT_REFINER_MODEL"
    if [ $? -ne 0 ]; then
      echo "[entrypoint] Prompt refiner prefetch failed."
    fi
  fi

  if [ "$ANIME_SEGMENTATION_DOWNLOAD_ON_START" = "1" ]; then
    echo "[entrypoint] Prefetching anime segmentation model: $ANIME_SEGMENTATION_MODEL"
    python3 -c 'import os; from rembg import new_session; new_session(os.environ["ANIME_SEGMENTATION_MODEL"])'
    if [ $? -ne 0 ]; then
      echo "[entrypoint] Anime segmentation prefetch failed."
    fi
  fi
  echo "[entrypoint] Background model prefetch finished."
}

echo "[entrypoint] App: $APP_NAME"
echo "[entrypoint] Image model: $IMAGE_MODEL_LABEL ($IMAGE_MODEL_FAMILY)"
if [ "$IMAGE_MODEL_FAMILY" = "animagine" ]; then
  echo "[entrypoint] Animagine path: $ANIMAGINE_MODEL_PATH"
else
  echo "[entrypoint] JANKU cache: s3://$JANKU_R2_BUCKET/$JANKU_R2_KEY"
  echo "[entrypoint] JANKU path: $JANKU_MODEL_PATH"
fi

if [ "$RUNTIME_SETUP_ON_START" = "1" ]; then
  install_runtime_dependencies
fi

if [ "$MODEL_DOWNLOAD_ON_START" = "1" ]; then
  prefetch_models &
fi

exec python3 app.py --host "$APP_HOST" --port "$APP_PORT"
