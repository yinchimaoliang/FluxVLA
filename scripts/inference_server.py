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
"""FastAPI inference server for remote VLA model inference.

Usage:
    # On the cloud GPU instance:
    python scripts/inference_server.py \
        --config configs/pi05/pi05_paligemma_ur3_full_finetune.py \
        --ckpt-path /path/to/checkpoint.safetensors \
        --port 8080

    # On the robot side, establish SSH tunnel:
    ssh -L 8080:localhost:8080 user@cloud-ip -N
"""

import argparse
import base64
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from mmengine import Config
from pydantic import BaseModel
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    images: Dict[str, str]  # {cam_name: base64-encoded JPEG bytes}
    qpos: List[float]  # 7 floats: 6 joints + 1 gripper
    task_description: str


class PredictResponse(BaseModel):
    actions: List[List[float]]  # (action_chunk, action_dim)


class ResetResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Server-side observation state
# ---------------------------------------------------------------------------


class InferenceState:
    """Maintains the observation window on the server side.

    Mirrors the window logic in ``URInferenceRunner.update_observation_window``
    so that ``PrivateInferenceDataset`` receives the same data structure.
    """

    def __init__(self, camera_names: List[str]):
        self.camera_names = camera_names
        self.observation_window: Optional[deque] = None

    def reset(self):
        self.observation_window = None

    def update(self, qpos: np.ndarray, images: Dict[str, np.ndarray]) -> Dict:
        if self.observation_window is None:
            self.observation_window = deque(maxlen=2)
            dummy_obs = {'qpos': None}
            for name in self.camera_names:
                dummy_obs[name] = None
            self.observation_window.append(dummy_obs)

        observation = {'qpos': qpos}
        for name in self.camera_names:
            observation[name] = images[name]
        self.observation_window.append(observation)
        return self.observation_window[-1]


# ---------------------------------------------------------------------------
# Globals populated by ``setup()``
# ---------------------------------------------------------------------------

app = FastAPI(title='FluxVLA Inference Server')

vla = None
dataset = None
denormalize_action = None
state: Optional[InferenceState] = None
action_chunk: int = 50
mixed_precision_dtype = torch.bfloat16
enable_mixed_precision = True


def setup(cfg,
          ckpt_path: str,
          mp_dtype: str = 'bfloat16',
          mp_enabled: bool = True):
    """Initialise model, dataset transforms, and denormalize transform.

    Mirrors the initialisation logic in ``BaseInferenceRunner.__init__``
    (lines 103-159 of ``base_inference_runner.py``).
    """
    global vla, dataset, denormalize_action, state
    global action_chunk, mixed_precision_dtype, enable_mixed_precision

    from fluxvla.engines import (build_dataset_from_cfg,
                                 build_transform_from_cfg, build_vla_from_cfg)
    from fluxvla.engines.utils.name_map import str_to_dtype
    from fluxvla.engines.utils.torch_utils import set_seed_everywhere

    # ---- Resolve paths (same as BaseInferenceRunner) ----
    data_stat_path = os.path.join(
        Path(ckpt_path).resolve().parent.parent, 'dataset_statistics.json')
    assert os.path.exists(data_stat_path), (
        f'Dataset statistics file not found at {data_stat_path}!')

    inf_cfg = cfg.inference
    inf_cfg.denormalize_action['norm_stats'] = data_stat_path
    inf_cfg.dataset['norm_stats'] = data_stat_path
    inf_cfg.dataset['model_path'] = os.path.dirname(os.path.dirname(ckpt_path))

    # ---- Build components ----
    dataset = build_dataset_from_cfg(inf_cfg.dataset)
    denormalize_action = build_transform_from_cfg(inf_cfg.denormalize_action)

    vla = build_vla_from_cfg(cfg.inference_model)
    if ckpt_path.endswith('.safetensors'):
        state_dict = load_file(ckpt_path, device='cpu')
    else:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    vla.load_state_dict(state_dict, strict=True)

    # ---- Move to GPU ----
    mixed_precision_dtype = str_to_dtype(mp_dtype)
    enable_mixed_precision = mp_enabled
    set_seed_everywhere(7)
    vla.eval()
    if enable_mixed_precision:
        vla.to(device='cuda', dtype=mixed_precision_dtype)
    else:
        vla.cuda()

    action_chunk = inf_cfg.get('action_chunk', 50)

    # ---- Observation state ----
    camera_names = ['cam_high', 'cam_left_wrist']
    state = InferenceState(camera_names)

    logger.info('Server setup complete. Model on CUDA with dtype=%s',
                mixed_precision_dtype)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get('/health')
async def health():
    return {
        'status': 'ready' if vla is not None else 'not_initialized',
        'device': 'cuda',
    }


@app.post('/reset', response_model=ResetResponse)
async def reset():
    if state is not None:
        state.reset()
    return ResetResponse(status='ok')


@app.post('/predict', response_model=PredictResponse)
async def predict(request: PredictRequest):
    if vla is None or dataset is None or state is None:
        raise HTTPException(status_code=503, detail='Server not initialised')

    t0 = time.time()

    # 1. Decode images from base64 JPEG
    images: Dict[str, np.ndarray] = {}
    for cam_name, b64_data in request.images.items():
        jpeg_bytes = base64.b64decode(b64_data)
        img_array = cv2.imdecode(
            np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            raise HTTPException(
                status_code=400,
                detail=f'Failed to decode image for {cam_name}')
        images[cam_name] = img_array

    # 2. Build qpos
    qpos = np.array(request.qpos, dtype=np.float64)

    # 3. Update observation window
    obs = state.update(qpos, images)
    obs['task_description'] = request.task_description

    # 4. Preprocess → model inputs (GPU tensors)
    inputs = dataset(obs)

    # 5. Model inference
    with torch.inference_mode():
        with torch.autocast(
                'cuda',
                dtype=mixed_precision_dtype,
                enabled=enable_mixed_precision):
            raw_action = vla.predict_action(**inputs)

    # 6. Postprocess (denormalize)
    denormalized = denormalize_action(dict(action=raw_action.cpu().numpy()))
    actions = denormalized[:action_chunk]

    t1 = time.time()
    logger.info('Inference took %.3fs, action shape %s', t1 - t0,
                actions.shape)

    return PredictResponse(actions=actions.tolist())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description='FluxVLA remote inference server')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration file.')
    parser.add_argument(
        '--ckpt-path',
        type=str,
        required=True,
        help='Path to the model checkpoint.')
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Bind address (default: 127.0.0.1).')
    parser.add_argument(
        '--port', type=int, default=8080, help='Bind port (default: 8080).')
    parser.add_argument(
        '--mixed-precision-dtype',
        type=str,
        default='bf16',
        help='Mixed precision dtype (default: bfloat16).')
    parser.add_argument(
        '--disable-mixed-precision',
        action='store_true',
        help='Disable mixed precision inference.')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s')

    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.inference.cfg = cfg

    setup(
        cfg,
        args.ckpt_path,
        mp_dtype=args.mixed_precision_dtype,
        mp_enabled=not args.disable_mixed_precision)

    uvicorn.run(
        app, host=args.host, port=args.port, workers=1, log_level='info')
