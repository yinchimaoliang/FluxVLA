# Copyright 2026 Limx Dynamics
"""Causal RoboCasa observations retain history and reset between episodes."""

import numpy as np
import pytest
import torch

from fluxvla.transforms.robocasa_transforms import (ProcessRobocasaEvalInputs,
                                                    RobocasaEvalDataset)
from fluxvla.transforms.transform_images import PrepareVideo
from fluxvla.transforms.video_action_transforms import (
    BufferVideoFrames, ConsistentVideoTransform)


@pytest.mark.parametrize('as_tensor', [False, True])
def test_video_history_warmup_rollover_and_reset(as_tensor):
    buffer = BufferVideoFrames(num_frames=4)

    def observe(value, new=False):
        frame = np.full((3, 1, 8, 8), value, dtype=np.float32)
        if as_tensor:
            frame = torch.from_numpy(frame)
        result = buffer(dict(pixel_values=frame, is_new_episode=new))
        frame[...] = -999  # Callers may reuse their observation buffer.
        video = np.asarray(result['pixel_values'])
        assert result['img_masks'].tolist() == [True] * video.shape[1]
        return video[0, :, 0, 0].tolist(), result['reset_history']

    assert observe(0, True) == ([0], True)
    assert observe(1) == ([0, 0, 0, 1], False)
    assert observe(2) == ([0, 0, 1, 2], False)
    assert observe(3) == ([0, 1, 2, 3], False)
    assert observe(4) == ([1, 2, 3, 4], False)
    assert observe(100, True) == ([100], True)
    assert observe(101) == ([100, 100, 100, 101], False)


def test_robocasa_pipeline_passes_history_and_episode_reset(monkeypatch):
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    dataset = RobocasaEvalDataset()

    def tokens(data):
        return dict(
            data,
            lang_tokens=np.array([1, 2]),
            lang_masks=np.array([True, True]))

    dataset.transforms = [
        ProcessRobocasaEvalInputs(resize_size=8, normalize=False),
        ConsistentVideoTransform(
            height=8, width=8, training=False, key='pixel_values'),
        PrepareVideo(num_views=1, frame_window_size=1),
        BufferVideoFrames(num_frames=4),
        tokens,
    ]

    def obs(value, new):
        return {
            'video.ego_view_pad_res256_freq20':
            np.full((8, 8, 3), value, dtype=np.uint8),
            'state.left_arm':
            np.zeros(7),
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
            new,
        }

    first, _ = dataset(obs(0, True))
    continued, _ = dataset(obs(255, False))
    assert first['images'].shape == (1, 3, 1, 8, 8)
    assert continued['images'].shape == (1, 3, 4, 8, 8)
    assert first['reset_history'] is True
    assert continued['reset_history'] is False
    assert continued['img_masks'].shape == (1, 4)
    torch.testing.assert_close(continued['images'][0, 0, :, 0, 0],
                               torch.tensor([-1., -1., -1., 1.]))
    restarted, _ = dataset(obs(255, True))
    assert restarted['reset_history'] is True
    assert restarted['images'].shape == (1, 3, 1, 8, 8)
    assert torch.all(restarted['images'] == 1)
