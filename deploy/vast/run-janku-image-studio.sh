#!/bin/bash
set -euo pipefail

. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
source /venv/main/bin/activate

exec /workspace/janku-image-studio/deploy/vast/entrypoint.sh
