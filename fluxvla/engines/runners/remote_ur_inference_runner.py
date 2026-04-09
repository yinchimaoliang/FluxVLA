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
"""Remote inference runner for UR3 robot.

Extends ``URInferenceRunner`` but delegates model inference to a remote
FastAPI server via HTTP.  All ROS/operator logic remains local on the
robot; only raw observations are sent over the network.

Typical usage (robot side)::

    # 1. SSH tunnel to cloud GPU
    ssh -L 8080:localhost:8080 user@cloud-ip -N

    # 2. Start the remote runner
    python scripts/inference_remote.py \\
        --config configs/pi05/pi05_paligemma_ur3_remote_inference.py
"""

import base64
import time
from collections import deque
from types import SimpleNamespace
from typing import Dict, List, Optional

import cv2
import numpy as np
import requests

from ..utils import build_operator_from_cfg, initialize_overwatch
from ..utils.root import RUNNERS

overwatch = initialize_overwatch(__name__)


@RUNNERS.register_module()
class RemoteURInferenceRunner:
    """UR3 inference runner that delegates model inference to a remote server.

    Keeps all ROS/operator functionality local but sends observations
    to a remote FastAPI server for preprocessing, model inference,
    and postprocessing.  Only requires ``requests``, ``opencv-python``,
    ``numpy`` and ROS on the robot side.

    Args:
        server_url (str): Base URL of the inference server.
        request_timeout (float): HTTP request timeout in seconds.
        seed (int): Random seed (unused locally, kept for config compat).
        action_chunk (int): Number of actions per prediction chunk.
        publish_rate (int): ROS publishing rate in Hz.
        max_publish_step (int): Maximum steps per episode.
        camera_names (List[str]): Camera names matching observation keys.
        operator (Dict): Configuration dict for the ROS operator.
        task_descriptions (Dict): Mapping from task ID to description.
        task_pose_sequences (Dict): Mapping from task ID to pose sequence.
    """

    def __init__(
        self,
        server_url: str = 'http://localhost:8080',
        request_timeout: float = 30.0,
        seed: int = 7,
        action_chunk: int = 50,
        publish_rate: int = 30,
        max_publish_step: int = 10000,
        use_robot_base: bool = False,
        disable_puppet_arm: bool = False,
        camera_names: Optional[List[str]] = None,
        operator: Optional[Dict] = None,
        task_descriptions: Optional[Dict] = None,
        task_pose_sequences: Optional[Dict] = None,
        **kwargs,
    ):
        # Server connection
        self.server_url = server_url.rstrip('/')
        self.request_timeout = request_timeout
        self._session = requests.Session()

        # Action / timing config
        self.seed = seed
        self.action_chunk = action_chunk
        self.publish_rate = publish_rate
        self.max_publish_step = max_publish_step
        self.use_robot_base = use_robot_base
        self.disable_puppet_arm = disable_puppet_arm

        # Observation window (local, for current frame only)
        self.observation_window = None

        # Action context (for cross-chunk continuity, kept for compat)
        self._prev_ctx = None
        self._action_ctx = SimpleNamespace()

        # ---- UR-specific defaults ----
        self.camera_names = camera_names or [
            'cam_high',
            'cam_right_wrist',
            'cam_left_wrist',
        ]

        if operator is None:
            operator = {
                'type': 'UROperator',
                'img_left_topic': '/wrist_camera/color/image_raw',
                'img_front_topic': '/front_camera/color/image_raw',
                'puppet_arm_left_topic': '/joint_states',
                'puppet_gripper_left_topic': '/gripper/position',
                'puppet_ee_pose_left_topic': '/arm/tcp_pose',
                'use_depth_image': False,
            }

        self.task_descriptions = task_descriptions or {}
        self.task_pose_sequences = task_pose_sequences or {}

        # Build ROS operator (only build call on robot side)
        self.ros_operator = build_operator_from_cfg(operator)

        # UR-specific preparation pose (joint angles)
        self.prepare_pose = [
            2.3911736011505127,
            -1.7057769934283655,
            2.1696739196777344,
            -0.5096147696124476,
            1.5789384841918945,
            -15.709390354140687,
        ]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def run_setup(self):
        """Verify the remote server is reachable."""
        try:
            resp = self._session.get(
                f'{self.server_url}/health', timeout=self.request_timeout)
            resp.raise_for_status()
            info = resp.json()
            overwatch.info(f'Remote inference server ready: {info}')
        except requests.RequestException as e:
            raise ConnectionError(
                f'Cannot reach inference server at {self.server_url}: {e}'
            ) from e

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self,
            initial_instruction:
            str = 'place it in the brown paper bag with right arm'):
        """Run the main inference loop (no local GPU required)."""
        import rospy

        overwatch.info('Starting remote inference runner')
        while not rospy.is_shutdown():
            self._run_episode(initial_instruction)

    def _run_episode(self, default_instruction: str):
        """Run a single episode using the remote server."""
        import rospy

        t = 0
        rate = rospy.Rate(self.publish_rate)

        # Reset server observation window for new episode
        self._reset_server()

        while t < self.max_publish_step and not rospy.is_shutdown():
            instructions = self._get_user_task_instruction(default_instruction)
            self._prev_ctx = None
            for instruction in instructions:
                self._action_ctx = SimpleNamespace()
                self._action_ctx.instruction = instruction

                # 1. Collect observation from ROS
                obs = self.update_observation_window()

                # 2. Send to server, get actions back
                actions = self._remote_predict(obs, instruction)

                # 3. Execute locally via ROS
                self._execute_actions(actions, rate)

                self._prev_ctx = self._action_ctx
                t += self.action_chunk
                overwatch.info(f'Published Step {t}')

    # ------------------------------------------------------------------
    # Remote inference
    # ------------------------------------------------------------------

    def _remote_predict(self, obs: Dict, task_description: str) -> np.ndarray:
        """Send observation to the remote server and receive actions.

        Args:
            obs: Observation dict with ``qpos``, ``cam_high``,
                 ``cam_left_wrist`` (BGR uint8 numpy arrays).
            task_description: Task instruction string.

        Returns:
            np.ndarray: Denormalized actions of shape
            ``(action_chunk, 7)``.
        """
        t0 = time.time()

        # JPEG-encode and base64-encode each camera image
        images_b64: Dict[str, str] = {}
        for cam_name in ['cam_high', 'cam_left_wrist']:
            img = obs.get(cam_name)
            if img is None:
                raise RuntimeError(
                    f'Camera image {cam_name!r} is None in observation')
            success, jpeg_buf = cv2.imencode('.jpg', img,
                                             [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise RuntimeError(f'Failed to JPEG-encode {cam_name}')
            images_b64[cam_name] = base64.b64encode(
                jpeg_buf.tobytes()).decode('ascii')

        payload = {
            'images': images_b64,
            'qpos': obs['qpos'].tolist(),
            'task_description': task_description,
        }

        resp = self._session.post(
            f'{self.server_url}/predict',
            json=payload,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()

        actions = np.array(resp.json()['actions'], dtype=np.float64)

        t1 = time.time()
        overwatch.info('Remote predict: %.3fs, payload %d cameras', t1 - t0,
                       len(images_b64))

        return actions

    def _reset_server(self):
        """Reset the server-side observation window."""
        try:
            resp = self._session.post(
                f'{self.server_url}/reset', timeout=self.request_timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            overwatch.warning(f'Failed to reset server state: {e}')

    # ------------------------------------------------------------------
    # ROS observation (inherited logic, no JPEG double-compression)
    # ------------------------------------------------------------------

    def get_ros_observation(self):
        """Get synchronized observation data from ROS topics.

        Returns:
            Tuple of (img_front, img_left, puppet_arm_left,
            puppet_gripper_left).
        """
        import rospy

        rate = rospy.Rate(self.publish_rate)
        print_flag = True
        rate.sleep()

        while not rospy.is_shutdown():
            result = self.ros_operator.get_frame()
            if not result:
                if print_flag:
                    overwatch.info(
                        'Synchronisation failed in get_ros_observation')
                    print_flag = False
                rate.sleep()
                continue

            print_flag = True
            (img_front, img_left, img_front_depth, img_left_depth,
             puppet_arm_left, puppet_ee_pose_left, puppet_gripper_left,
             frame_time_min, frame_time_max) = result

            return (img_front, img_left, puppet_arm_left, puppet_gripper_left)

    def update_observation_window(self) -> Dict:
        """Update observation window with latest ROS data.

        Unlike ``URInferenceRunner``, this version does **not** apply
        ``_apply_jpeg_compression`` because JPEG compression happens
        implicitly when images are encoded for network transmission.
        """
        if self.observation_window is None:
            self.observation_window = deque(maxlen=2)
            dummy_obs = {'qpos': None}
            for camera_name in self.camera_names:
                dummy_obs[camera_name] = None
            self.observation_window.append(dummy_obs)

        img_front, img_left, puppet_arm_left, puppet_gripper_left = (
            self.get_ros_observation())

        # UR joint reordering: [2,1,0,3,4,5]
        qpos = np.concatenate([
            np.array(puppet_arm_left.position)[[2, 1, 0, 3, 4, 5]],
            np.array([puppet_gripper_left.data]),
        ],
                              axis=0)

        observation = {
            'qpos': qpos,
            self.camera_names[0]: img_front,  # cam_high
            self.camera_names[2]: img_left,  # cam_left_wrist
        }

        self.observation_window.append(observation)
        return self.observation_window[-1]

    # ------------------------------------------------------------------
    # Action execution (same as URInferenceRunner)
    # ------------------------------------------------------------------

    def _execute_actions(self, actions: np.ndarray, rate):
        """Execute a sequence of robot actions via ROS.

        Args:
            actions: Array of shape ``(action_chunk, 7)`` with
                denormalized joint + gripper commands.
            rate: ROS rate limiter.
        """
        for action in actions:
            self.ros_operator.servoj(action[:6])
            self.ros_operator.movegrip(action[6])
            rate.sleep()

    # ------------------------------------------------------------------
    # Task management (same as URInferenceRunner)
    # ------------------------------------------------------------------

    def _get_task_description(self, task_id: str) -> str:
        return self.task_descriptions.get(
            task_id, 'place it in the brown paper bag with right arm')

    def _get_user_task_instruction(self, default_instruction: str) -> str:
        task_id = input('Enter task ID (or press Enter for default): ').strip()
        if task_id == '0':
            self._move_to_prepare_pose()
            task_id = input('Enter task ID after reset: ').strip()

        if task_id in self.task_pose_sequences:
            self.execute_task_pose(task_id)
            input('Enter task ID (or press Enter for default): ').strip()

        num_times = int(input('Number of times to repeat the task: '))
        task_description = self._get_task_description(task_id)
        return [task_description] * num_times

    def _move_to_prepare_pose(self):
        """Move robot to the predefined preparation pose."""
        self.ros_operator.movej(self.prepare_pose)
        self.ros_operator.movegrip(0.085)

    def execute_task_pose(self, task_id: str):
        """Execute a preset pose sequence for a task."""
        if task_id in self.task_pose_sequences:
            for joint_angles, gripper_position in (
                    self.task_pose_sequences[task_id]):
                self.ros_operator.movel(joint_angles)
                self.ros_operator.movegrip(gripper_position[0])

    def cleanup(self):
        """Clean up resources."""
        overwatch.info('Cleaning up RemoteURInferenceRunner')
        self._prev_ctx = None
        self._action_ctx = SimpleNamespace()
        if self.observation_window is not None:
            self.observation_window.clear()
        self._session.close()
        overwatch.info('RemoteURInferenceRunner cleanup completed')
