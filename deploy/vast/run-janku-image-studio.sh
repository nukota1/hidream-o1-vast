#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate

exec /workspace/janku-image-studio/deploy/vast/entrypoint.sh
