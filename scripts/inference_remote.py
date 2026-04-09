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
"""Client entry point for remote UR3 inference.

Usage:
    # Ensure SSH tunnel is running:
    #   ssh -L 8080:localhost:8080 user@cloud-gpu -N

    python scripts/inference_remote.py \
        --config configs/pi05/pi05_paligemma_ur3_remote_inference.py

    # Override server URL:
    python scripts/inference_remote.py \
        --config configs/pi05/pi05_paligemma_ur3_remote_inference.py \
        --server-url http://localhost:9090
"""

import argparse

from mmengine import Config

from fluxvla.engines import build_runner_from_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='Remote inference client for UR3 robot')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the client configuration file.')
    parser.add_argument(
        '--server-url',
        type=str,
        default=None,
        help='Override the server URL from config.')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.server_url:
        cfg.inference.server_url = args.server_url

    runner = build_runner_from_cfg(cfg.inference)
    runner.run_setup()
    runner.run()


if __name__ == '__main__':
    main()
