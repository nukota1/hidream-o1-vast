#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-ghcr.io/YOUR_GITHUB_OWNER/hidream-o1-image:latest}"
NAME="${CONTAINER_NAME:-hidream-o1-image}"
MODEL_DIR="${MODEL_DIR:-/workspace/hidream-o1-models}"
ENV_FILE="${ENV_FILE:-./env.vast}"
PORT="${HIDREAM_PORT:-7861}"

mkdir -p "$MODEL_DIR"

docker pull "$IMAGE"

if docker ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
  docker rm -f "$NAME"
fi

docker run -d \
  --name "$NAME" \
  --gpus all \
  --restart unless-stopped \
  --env-file "$ENV_FILE" \
  -p "$PORT:7861" \
  -v "$MODEL_DIR:/models" \
  "$IMAGE"

echo "Started $NAME on port $PORT"
echo "Logs: docker logs -f $NAME"
