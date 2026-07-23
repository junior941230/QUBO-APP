#!/usr/bin/env bash
set -euo pipefail

QUBO_PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
QUBO_HOST_DRIVER_DIR=/lib/x86_64-linux-gnu

# The server image prepends a CUDA 13 forward-compatibility driver. CUDA 13 no
# longer supports the Tesla V100, while the host's 535 driver and this project's
# CUDA 12 packages do. Put the mounted host driver first for this process.
export LD_LIBRARY_PATH="${QUBO_HOST_DRIVER_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${QUBO_PROJECT_DIR}/.venv/bin/python" "${QUBO_PROJECT_DIR}/app.py" "$@"
