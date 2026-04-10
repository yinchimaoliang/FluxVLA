# Remote Inference

## Overview

Remote inference allows you to run model inference on a cloud GPU server while controlling the robot locally.  This is useful when the robot-side device (e.g. UR3 control PC) does not have a GPU powerful enough to run the VLA model.

The architecture is a client-server model connected via SSH tunnel:

```
 UR3 Robot (Client)                       Cloud GPU (Server)
┌─────────────────────┐    SSH Tunnel    ┌──────────────────────┐
│  ROS + Camera Data   │ ──────────────> │  FastAPI Server      │
│  inference_remote.py │ <────────────── │  inference_server.py │
│  Action Execution    │   HTTP / JSON   │  Model on CUDA       │
└─────────────────────┘                  └──────────────────────┘
```

- **Server** (`scripts/inference_server.py`): Loads the model checkpoint onto GPU, exposes `/predict`, `/reset`, and `/health` HTTP endpoints.
- **Client** (`scripts/inference_remote.py`): Collects camera images and joint states via ROS, sends them as base64-encoded JPEG over HTTP, receives denormalized action sequences, and executes them on the robot.

## Quick Start (One-Click Script)

The easiest way to run remote inference is the all-in-one script `scripts/run_remote_inference.sh`.  Run it on the **robot side only** -- it will automatically SSH to the remote machine, start the server, set up the tunnel, and launch the client.

```bash
bash scripts/run_remote_inference.sh \
    --host <REMOTE_IP> \
    --ssh-port <SSH_PORT> \
    --remote-dir <REMOTE_PROJECT_DIR> \
    --ckpt-path <REMOTE_CHECKPOINT_PATH> \
    --conda-env <CONDA_ENV_NAME>
```

**Example:**

```bash
bash scripts/run_remote_inference.sh \
    --host 8.145.57.160 \
    --ssh-port 1024 \
    --remote-dir /root/projects/fluxvla \
    --ckpt-path /root/work_dirs/pi05_ur3/checkpoints/step-1000.safetensors \
    --conda-env fluxvla
```

The script performs these steps automatically:

1. SSHs to the remote machine, activates conda, and starts the inference server in the background.
2. Establishes an SSH port-forwarding tunnel (`localhost:8080 -> remote:8080`).
3. Polls the `/health` endpoint until the model is fully loaded (up to 180 seconds by default).
4. Launches the local inference client which begins the ROS control loop.
5. On exit (Ctrl+C), cleans up: kills the local client, SSH tunnel, and remote server process.

### Full Parameter Reference

| Flag               | Required | Default                                               | Description                                                     |
| ------------------ | -------- | ----------------------------------------------------- | --------------------------------------------------------------- |
| `--host`           | Yes      | --                                                    | Remote GPU machine IP or hostname                               |
| `--ssh-port`       | No       | `22`                                                  | SSH port on the remote machine                                  |
| `--user`           | No       | `root`                                                | SSH username                                                    |
| `--remote-dir`     | Yes      | --                                                    | FluxVLA project directory on the remote machine (absolute path) |
| `--ckpt-path`      | Yes      | --                                                    | Model checkpoint path on the remote machine (absolute path)     |
| `--conda-env`      | No       | `fluxvla`                                             | Conda environment name on the remote machine                    |
| `--server-port`    | No       | `8080`                                                | Port the inference server listens on                            |
| `--server-config`  | No       | `configs/pi05/pi05_paligemma_ur3_full_finetune.py`    | Server-side config (relative to `--remote-dir`)                 |
| `--local-config`   | No       | `configs/pi05/pi05_paligemma_ur3_remote_inference.py` | Client-side config (relative to local project dir)              |
| `--health-timeout` | No       | `180`                                                 | Max seconds to wait for model loading                           |
| `--ssh-key`        | No       | --                                                    | Path to SSH identity file                                       |

## Manual Setup (Without Shell Script)

If you prefer to set up each component separately (e.g. for debugging or custom deployments), follow the three steps below.

### Step 1: Start the Inference Server (on the remote GPU machine)

SSH into the remote machine and run:

```bash
# Activate the environment
conda activate fluxvla

# Start the server
cd /path/to/fluxvla
python scripts/inference_server.py \
    --config configs/pi05/pi05_paligemma_ur3_full_finetune.py \
    --ckpt-path /path/to/checkpoint.safetensors \
    --port 8080
```

The server binds to `127.0.0.1` by default (localhost only), which is a security best practice -- it is only accessible through the SSH tunnel.

You can verify the server is ready by checking:

```bash
curl http://localhost:8080/health
# Expected: {"status":"ready","device":"cuda"}
```

### Step 2: Establish SSH Tunnel (on the robot machine)

On the UR3 control PC, create an SSH tunnel to forward the server port:

```bash
ssh -L 8080:localhost:8080 <user>@<remote-ip> -p <ssh-port> -N &
```

**Example:**

```bash
ssh -L 8080:localhost:8080 root@8.145.57.160 -p 1024 -N &
```

The `-N` flag tells SSH not to execute a remote command (tunnel only). The `&` puts it in the background.

Verify the tunnel works:

```bash
curl http://localhost:8080/health
# Should return the same response as on the remote machine
```

### Step 3: Start the Inference Client (on the robot machine)

```bash
python scripts/inference_remote.py \
    --config configs/pi05/pi05_paligemma_ur3_remote_inference.py
```

You can override the server URL if using a non-default port:

```bash
python scripts/inference_remote.py \
    --config configs/pi05/pi05_paligemma_ur3_remote_inference.py \
    --server-url http://localhost:9090
```

## Configuration Files

| Config                                                | Side   | Contains                                                    |
| ----------------------------------------------------- | ------ | ----------------------------------------------------------- |
| `configs/pi05/pi05_paligemma_ur3_full_finetune.py`    | Server | Model architecture, dataset transforms, normalization stats |
| `configs/pi05/pi05_paligemma_ur3_remote_inference.py` | Client | ROS operator topics, task descriptions, server URL          |

The client config does **not** contain any model definition -- all model-related logic resides on the server.

## Prerequisites

**Remote GPU machine:**

- CUDA-capable GPU
- FluxVLA installed with all dependencies
- Model checkpoint and `dataset_statistics.json` (must be in `../../` relative to the checkpoint)

**Robot-side machine:**

- ROS installed and running (`roscore`)
- FluxVLA installed (at minimum: `requests`, `opencv-python`, `numpy`, `mmengine`)
- Camera and joint state ROS topics publishing
- SSH access to the remote machine
- `curl` (for health check in the one-click script)

## Troubleshooting

**Server fails to start:**
Check the remote log file:

```bash
ssh <user>@<remote-ip> -p <ssh-port> 'cat /tmp/fluxvla_inference_server.log'
```

**Health check times out:**

- Ensure the SSH tunnel is established: `curl http://localhost:8080/health`
- Large models may take longer to load; increase `--health-timeout`
- If behind a proxy, ensure `localhost` is excluded: `export no_proxy=localhost,127.0.0.1`

**Port already in use:**
If port 8080 is occupied, use a different port on both sides:

```bash
bash scripts/run_remote_inference.sh \
    --host 8.145.57.160 \
    --ssh-port 1024 \
    --server-port 9090 \
    --remote-dir /root/projects/fluxvla \
    --ckpt-path /root/checkpoints/step-1000.safetensors
```

**Client cannot connect to ROS:**
Make sure `roscore` is running and the camera / joint state topics are active before starting the client. Check with `rostopic list`.
