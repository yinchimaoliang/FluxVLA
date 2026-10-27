# Copyright 2026 Limx Dynamics
"""Observed history must retain the time and conditioning used in training."""

from types import SimpleNamespace

import pytest
import torch

from fluxvla.models.vlas.dreamzero_vla import DreamZeroVLA


@pytest.mark.parametrize('frames', [1, 9, 17, 25])
def test_observed_history_prefills_past_before_next_block(frames):
    video = torch.arange(frames).float().view(1, 1, frames, 1, 1)
    video = video.expand(1, 3, frames, 8, 8)
    calls = []
    condition = torch.zeros(1, 4, 9, 1, 1)

    def encode_image(image, num_frames, height, width):
        torch.testing.assert_close(image, video[:, :, :1].transpose(1, 2))
        assert num_frames == 33
        return torch.zeros(1, 257, 1280), condition, torch.zeros(1, 2, 1, 1, 1)

    def encode_video(images):
        torch.testing.assert_close(images, video)
        return images[:, :2, ::4, :1, :1]

    def sample(**kwargs):
        assert kwargs['current_start_frame'] == 1 + (frames - 1) // 4
        assert kwargs['denoise_frames'] == 2
        assert kwargs['states'].shape == (1, 1, 64)
        assert kwargs['states'][0, 0, 0] == 123
        return torch.zeros(1, 24, 32)

    head = SimpleNamespace(
        num_frame_per_block=2,
        max_state_dim=64,
        num_inference_steps=16,
        reset_inference_state=lambda: None,
        _create_cache_pair=lambda *args: ([], []),
        _single_flowmatching_step=lambda **kwargs: calls.append(kwargs),
        _sample_action_block=sample)
    backbone = SimpleNamespace(
        set_frozen_modules_to_eval_mode=lambda: None,
        encode_image=encode_image,
        encode_video=encode_video)
    vla = SimpleNamespace(
        vla_head=head,
        vlm_backbone=backbone,
        frame_window_size=33,
        _encode_wan_prompts=lambda *args: torch.zeros(1, 512, 4096),
        _prepare_states=lambda state, count: state.unsqueeze(1))
    result = DreamZeroVLA._predict_from_observed_video(
        vla, video, torch.ones(1, 2), torch.ones(1, 2),
        torch.full((1, 58), 123.), None)
    assert result.shape == (1, 24, 32)
    assert [call['start_frame'] for call in calls
            ] == ([0] + list(range(1, 1 + (frames - 1) // 4, 2)))
    assert sum(call['reference_latents'].shape[2]
               for call in calls) == (1 + (frames - 1) // 4)


@pytest.mark.parametrize('frames', [0, 4, 8, 33])
def test_invalid_or_future_consuming_history_is_rejected(frames):
    vla = SimpleNamespace(
        vla_head=SimpleNamespace(num_frame_per_block=2), frame_window_size=33)
    with pytest.raises(ValueError, match='complete past blocks'):
        DreamZeroVLA._predict_from_observed_video(
            vla, torch.zeros(1, 3, frames, 8, 8), None, None, None, None)
