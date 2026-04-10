#!/usr/bin/env bash
# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# One-click remote inference for UR3.
#
# Run this on the UR3 robot side. It will:
#   1. SSH to the remote GPU machine and start the inference server
#   2. Set up an SSH port-forwarding tunnel
#   3. Wait for the server to finish loading the model
#   4. Launch the local inference client
#   5. Clean up everything on exit
#
# Usage:
#   bash scripts/run_remote_inference.sh \
#       --host 8.145.57.160 \
#       --ssh-port 1024 \
#       --remote-dir /root/projects/fluxvla \
#       --ckpt-path /root/checkpoints/step-1000.safetensors \
#       --conda-env fluxvla

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Bypass proxy for localhost (health check + inference client)
# ──────────────────────────────────────────────────────────────────────
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

# ──────────────────────────────────────────────────────────────────────
# Resolve project directory
# ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────
HOST=""
SSH_PORT="22"
REMOTE_USER="root"
REMOTE_DIR=""
CKPT_PATH=""
CONDA_ENV="fluxvla"
SERVER_PORT="8080"
SERVER_CONFIG="configs/pi05/pi05_paligemma_ur3_full_finetune.py"
LOCAL_CONFIG="configs/pi05/pi05_paligemma_ur3_remote_inference.py"
HEALTH_TIMEOUT=180
SSH_KEY=""

# PIDs for cleanup
REMOTE_SERVER_PID=""
TUNNEL_PID=""
LOCAL_CLIENT_PID=""

# ──────────────────────────────────────────────────────────────────────
# Usage
# ──────────────────────────────────────────────────────────────────────
usage() {
    cat <<'EOF'
Usage: bash scripts/run_remote_inference.sh [OPTIONS]

Required:
  --host HOST              Remote GPU machine IP / hostname
  --remote-dir DIR         Project directory on the remote machine (absolute)
  --ckpt-path PATH         Checkpoint path on the remote machine (absolute)

Optional:
  --ssh-port PORT          SSH port on remote machine          [default: 22]
  --user USER              SSH username                        [default: root]
  --conda-env ENV          Conda environment on remote machine [default: fluxvla]
  --server-port PORT       Inference server port               [default: 8080]
  --server-config PATH     Server config (relative to remote-dir)
                           [default: configs/pi05/pi05_paligemma_ur3_full_finetune.py]
  --local-config PATH      Local client config (relative to project dir)
                           [default: configs/pi05/pi05_paligemma_ur3_remote_inference.py]
  --health-timeout SECS    Seconds to wait for model loading   [default: 180]
  --ssh-key PATH           Path to SSH identity file
  -h, --help               Show this help message
EOF
    exit "${1:-0}"
}

# ──────────────────────────────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)           HOST="$2";           shift 2 ;;
        --ssh-port)       SSH_PORT="$2";       shift 2 ;;
        --user)           REMOTE_USER="$2";    shift 2 ;;
        --remote-dir)     REMOTE_DIR="$2";     shift 2 ;;
        --ckpt-path)      CKPT_PATH="$2";      shift 2 ;;
        --conda-env)      CONDA_ENV="$2";      shift 2 ;;
        --server-port)    SERVER_PORT="$2";     shift 2 ;;
        --server-config)  SERVER_CONFIG="$2";   shift 2 ;;
        --local-config)   LOCAL_CONFIG="$2";    shift 2 ;;
        --health-timeout) HEALTH_TIMEOUT="$2";  shift 2 ;;
        --ssh-key)        SSH_KEY="$2";         shift 2 ;;
        -h|--help)        usage 0 ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            usage 1
            ;;
    esac
done

# Validate required args
missing=""
[[ -z "$HOST" ]]       && missing="${missing}  --host\n"
[[ -z "$REMOTE_DIR" ]] && missing="${missing}  --remote-dir\n"
[[ -z "$CKPT_PATH" ]]  && missing="${missing}  --ckpt-path\n"
if [[ -n "$missing" ]]; then
    echo -e "ERROR: Missing required arguments:\n${missing}" >&2
    usage 1
fi

# ──────────────────────────────────────────────────────────────────────
# Common SSH options
# ──────────────────────────────────────────────────────────────────────
SSH_OPTS=(
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
    -o LogLevel=ERROR
    -p "$SSH_PORT"
)
[[ -n "$SSH_KEY" ]] && SSH_OPTS+=(-i "$SSH_KEY")

# ──────────────────────────────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────────────────────────────
cleanup() {
    echo "" >&2
    echo "[cleanup] Shutting down..." >&2

    # 1. Kill local client
    if [[ -n "$LOCAL_CLIENT_PID" ]] && kill -0 "$LOCAL_CLIENT_PID" 2>/dev/null; then
        echo "[cleanup] Stopping local client (PID $LOCAL_CLIENT_PID)..." >&2
        kill -TERM "$LOCAL_CLIENT_PID" 2>/dev/null || true
        for _ in 1 2 3; do
            kill -0 "$LOCAL_CLIENT_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$LOCAL_CLIENT_PID" 2>/dev/null || true
    fi

    # 2. Kill SSH tunnel
    if [[ -n "$TUNNEL_PID" ]] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
        echo "[cleanup] Stopping SSH tunnel (PID $TUNNEL_PID)..." >&2
        kill -TERM "$TUNNEL_PID" 2>/dev/null || true
        wait "$TUNNEL_PID" 2>/dev/null || true
    fi

    # 3. Kill remote server
    if [[ -n "$REMOTE_SERVER_PID" ]]; then
        echo "[cleanup] Stopping remote server (remote PID $REMOTE_SERVER_PID)..." >&2
        ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 \
            "${REMOTE_USER}@${HOST}" \
            "kill -TERM ${REMOTE_SERVER_PID} 2>/dev/null; sleep 2; kill -9 ${REMOTE_SERVER_PID} 2>/dev/null; exit 0" \
            2>/dev/null || true
    fi

    echo "[cleanup] Done." >&2
}

trap cleanup EXIT INT TERM

# ──────────────────────────────────────────────────────────────────────
# Step 1: Start remote inference server
# ──────────────────────────────────────────────────────────────────────
echo "[server] Starting inference server on ${REMOTE_USER}@${HOST}:${SSH_PORT}..."
echo "[server] Remote dir: ${REMOTE_DIR}"
echo "[server] Checkpoint: ${CKPT_PATH}"
echo "[server] Conda env:  ${CONDA_ENV}"

REMOTE_CMD=$(cat <<REMOTEOF
# Activate conda
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source "\$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null \
    || true
conda activate ${CONDA_ENV} 2>/dev/null || true

cd "${REMOTE_DIR}" || exit 1

nohup python scripts/inference_server.py \
    --config "${SERVER_CONFIG}" \
    --ckpt-path "${CKPT_PATH}" \
    --host 127.0.0.1 \
    --port ${SERVER_PORT} \
    > /tmp/fluxvla_inference_server.log 2>&1 &

echo \$!
REMOTEOF
)

REMOTE_SERVER_PID=$(ssh "${SSH_OPTS[@]}" -o ConnectTimeout=10 \
    "${REMOTE_USER}@${HOST}" \
    bash -c "'${REMOTE_CMD}'" 2>/dev/null)

if [[ -z "$REMOTE_SERVER_PID" ]] || ! [[ "$REMOTE_SERVER_PID" =~ ^[0-9]+$ ]]; then
    echo "[server] ERROR: Failed to start remote server (got PID: '${REMOTE_SERVER_PID}')" >&2
    exit 1
fi

echo "[server] Remote server started (PID ${REMOTE_SERVER_PID})"
echo "[server] Remote log: /tmp/fluxvla_inference_server.log"

# ──────────────────────────────────────────────────────────────────────
# Step 2: Establish SSH tunnel
# ──────────────────────────────────────────────────────────────────────
echo "[tunnel] Setting up port forwarding localhost:${SERVER_PORT} -> remote:${SERVER_PORT}..."

ssh -N -L "${SERVER_PORT}:localhost:${SERVER_PORT}" \
    "${SSH_OPTS[@]}" \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    "${REMOTE_USER}@${HOST}" &
TUNNEL_PID=$!

sleep 2
if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "[tunnel] ERROR: SSH tunnel failed to start (port ${SERVER_PORT} may be in use)" >&2
    exit 1
fi

echo "[tunnel] SSH tunnel established (PID ${TUNNEL_PID})"

# ──────────────────────────────────────────────────────────────────────
# Step 3: Wait for server health
# ──────────────────────────────────────────────────────────────────────
HEALTH_URL="http://localhost:${SERVER_PORT}/health"
echo "[health] Waiting for server to be ready (timeout: ${HEALTH_TIMEOUT}s)..."

elapsed=0
interval=3
max_interval=10

while [[ $elapsed -lt $HEALTH_TIMEOUT ]]; do
    if response=$(curl -sf --noproxy localhost,127.0.0.1 --max-time 5 "$HEALTH_URL" 2>/dev/null); then
        if echo "$response" | grep -q '"ready"'; then
            echo "[health] Server is ready! (after ${elapsed}s)"
            break
        fi
        echo "[health] Server responded but not ready yet (${elapsed}s elapsed)"
    else
        echo "[health] Server not reachable yet (${elapsed}s elapsed)"
    fi

    # Check tunnel is still alive
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        echo "[health] ERROR: SSH tunnel died unexpectedly" >&2
        exit 1
    fi

    sleep "$interval"
    elapsed=$((elapsed + interval))
    interval=$((interval < max_interval ? interval + 2 : max_interval))
done

if [[ $elapsed -ge $HEALTH_TIMEOUT ]]; then
    echo "[health] ERROR: Timed out after ${HEALTH_TIMEOUT}s" >&2
    echo "[health] Check remote log: ssh ${SSH_OPTS[*]} ${REMOTE_USER}@${HOST} 'cat /tmp/fluxvla_inference_server.log'" >&2
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────
# Step 4: Launch local inference client
# ──────────────────────────────────────────────────────────────────────
echo "[client] Starting local inference client..."
echo "[client] Config: ${LOCAL_CONFIG}"

python "${PROJECT_DIR}/scripts/inference_remote.py" \
    --config "${PROJECT_DIR}/${LOCAL_CONFIG}" \
    --server-url "http://localhost:${SERVER_PORT}"
LOCAL_CLIENT_EXIT=$?

exit ${LOCAL_CLIENT_EXIT}
