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
"""Client-side config for remote PI0.5 inference on UR3.

This config is used on the robot side only.  It does NOT contain model
architecture or checkpoint information — those live on the cloud GPU
server and are loaded via the server's own config.

Usage:
    # 1. Start SSH tunnel
    ssh -L 8080:localhost:8080 user@cloud-gpu -N

    # 2. Run inference
    python scripts/inference_remote.py \
        --config configs/pi05/pi05_paligemma_ur3_remote_inference.py
"""

inference = dict(
    type='RemoteURInferenceRunner',
    server_url='http://localhost:8080',
    request_timeout=30.0,
    seed=7,
    action_chunk=50,
    publish_rate=30,
    max_publish_step=10000,
    task_descriptions={
        '1': 'grasp the stopper of the dark-colored wide-mouth bottle',
        '2': 'place the bottle stopper upside down on the tabletop',
        '3': 'grasp the body of the dark-colored wide-mouth bottle',
        '4':
        'pour the liquid in the dark-colored wide-mouth bottle into the erlenmeyer flask',  # noqa: E501
        '5': 'put the dark-colored wide-mouth bottle back on the tabletop',
        '6': 'grasp the measuring cylinder',
        '7':
        'pour the liquid in the measuring cylinder into the erlenmeyer flask',
        '8': 'put the measuring cylinder back on the tabletop',
        '9': 'grasp the neck of the erlenmeyer flask',
        '10': 'shake the erlenmeyer flask',
        '11': 'place the erlenmeyer flask back on the tabletop',
    },
    operator=dict(
        type='UROperator',
        img_left_topic='/wrist_camera/color/image_raw',
        img_front_topic='/front_camera/color/image_raw',
        puppet_arm_left_topic='/joint_states',
        puppet_gripper_left_topic='/gripper/position',
        puppet_ee_pose_left_topic='/arm/tcp_pose',
        use_depth_image=False,
    ),
)
