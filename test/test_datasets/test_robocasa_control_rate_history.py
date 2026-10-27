# Copyright 2026 Limx Dynamics
"""Video history follows simulator steps, independently of replanning."""

import numpy as np
import pytest
import torch

from fluxvla.transforms.robocasa_transforms import (
    ProcessRobocasaVideoHistoryInputs, RobocasaEvalDataset)
from fluxvla.transforms.transform_images import PrepareVideo
from fluxvla.transforms.video_action_transforms import ConsistentVideoTransform


def observation(step):
    return {
        'video.ego_view_pad_res256_freq20':
        np.full((8, 8, 3), step, dtype=np.uint8),
        'state.left_arm':
        np.full(7, step, dtype=np.float32),
        'state.right_arm':
        np.zeros(7),
        'state.left_hand':
        np.zeros(6),
        'state.right_hand':
        np.zeros(6),
        'state.waist':
        np.zeros(3),
        'task_description':
        'pick up the bottle',
        'is_new_episode':
        step == 0,
    }


def make_history():
    return ProcessRobocasaVideoHistoryInputs(
        normalize=False,
        resize_size=8,
        frame_stride=3,
        frames_per_block=8,
        max_history_blocks=3)


def test_stride_warmup_and_rolling_window():
    history = make_history()
    expected = {
        0: [0],
        8: [8],
        16: [16],
        24: list(range(0, 25, 3)),
        32: list(range(8, 33, 3)),
        48: list(range(0, 49, 3)),
        72: list(range(0, 73, 3)),
        80: list(range(8, 81, 3)),
    }
    for step in range(81):
        obs = observation(step)
        history.record_observation(obs, step)
        # Environment callers can reuse their image storage.
        obs['video.ego_view_pad_res256_freq20'][...] = 255
        if step in expected:
            result = history(observation(step))
            assert result['history_step_indices'] == expected[step]
            assert result['pixel_values'][:, 0, 0,
                                          0].tolist() == expected[step]
            assert result['states'][0] == step
            assert result['observed_video_history'] is True
            assert max(result['history_step_indices']) == step
    history.record_observation(observation(0), 0)
    assert history(observation(0))['history_step_indices'] == [0]


def test_missing_control_observations_fail_explicitly():
    history = make_history()
    with pytest.raises(ValueError, match='record_observation'):
        history(observation(0))
    history.record_observation(observation(0), 0)
    with pytest.raises(ValueError, match='every environment step'):
        history.record_observation(observation(8), 8)


def test_history_reaches_model_as_cthw(monkeypatch):
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    dataset = RobocasaEvalDataset()

    def tokens(data):
        return dict(
            data,
            lang_tokens=np.array([1, 2]),
            lang_masks=np.array([True, True]))

    dataset.transforms = [
        make_history(),
        ConsistentVideoTransform(
            height=8, width=8, training=False, key='pixel_values'),
        PrepareVideo(num_views=1, frame_window_size=1),
        tokens,
    ]
    for step in range(33):
        obs = observation(step)
        dataset.record_observation(obs, step)
        if step % 8 == 0:
            batch, _ = dataset(obs)
            frames = 1 if step < 24 else 9
            assert batch['images'].shape == (1, 3, frames, 8, 8)
            assert batch['observed_video_history'] is True
            assert batch['reset_history'] is True
            assert batch['states'][0, 0] == step
    assert dataset.last_debug['history_step_indices'] == list(range(8, 33, 3))
