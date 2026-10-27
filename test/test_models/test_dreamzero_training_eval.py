# Copyright 2026 Limx Dynamics
"""Training diagnostics must sample without future labels or RNG drift."""

import math

import pytest
import torch

from fluxvla.models.vlas.dreamzero_vla import (DreamZeroVLA,
                                               _command_error_metrics)


class ProbeVLA(torch.nn.Module):
    compute_training_eval = DreamZeroVLA.compute_training_eval

    def __init__(self):
        super().__init__()
        self.vlm_backbone = torch.nn.Linear(1, 1).eval()
        self.vla_head = torch.nn.Module()
        self.vla_head.official_inference = False
        self.vla_head.num_inference_steps = 12
        self.vla_head.num_action_per_block = 24
        self.vla_head.action_dim = 3
        self.vla_head.inference_kv_cache = object()
        self.vla_head.current_start_frame = 7
        self.fail = False

    def _predict_from_observed_video(self, images, tokens, masks, states,
                                     embodiments):
        assert not self.training
        assert images.shape == (1, 3, 1, 8, 8)
        assert states.shape == (1, 1, 64)
        assert torch.isfinite(images).all() and torch.isfinite(states).all()
        assert tokens.shape == masks.shape == (1, 2, 4)
        assert embodiments.tolist() == [0]
        assert self.vla_head.num_inference_steps == 16
        torch.randn(9)  # The diagnostic must not advance training's RNG.
        self.vla_head.inference_kv_cache = None
        self.vla_head.current_start_frame = 0
        if self.fail:
            raise RuntimeError('sampling failed')
        prediction = torch.full((1, 24, 5), float('nan'))
        prediction[:, :8, :3] = 2
        prediction[:, 8:, :3] = 4
        return prediction


def make_batch(per_dimension):
    images = torch.full((1, 3, 33, 8, 8), float('nan'))
    images[:, :, :1] = 0
    states = torch.full((1, 4, 64), float('nan'))
    states[:, :1] = 0
    actions = torch.full((1, 96, 5), float('nan'))
    actions[:, :24, :3] = 0
    mask = torch.zeros(1, 96, 5) if per_dimension else torch.zeros(1, 96)
    mask[:, :24] = 1
    return dict(
        images=images,
        states=states,
        actions=actions,
        action_masks=mask,
        lang_tokens=torch.ones(1, 2, 4, dtype=torch.long),
        lang_masks=torch.ones(1, 2, 4, dtype=torch.bool),
        embodiment_ids=torch.zeros(1, dtype=torch.long))


@pytest.mark.parametrize('per_dimension', [False, True])
def test_sampling_excludes_future_and_padding_and_preserves_state(
        per_dimension):
    model = ProbeVLA()
    cache = model.vla_head.inference_kv_cache
    random_state = torch.get_rng_state().clone()
    batch = make_batch(per_dimension)
    result = model.compute_training_eval(batch)
    metrics = result['metrics']
    assert metrics['sampled_action_mae_first8'] == pytest.approx(2)
    assert metrics['sampled_action_rmse_first8'] == pytest.approx(2)
    assert metrics['sampled_action_mae_first24'] == pytest.approx(10 / 3)
    assert metrics['sampled_action_rmse_first24'] == pytest.approx(
        math.sqrt(12))
    assert result['video_frames'] is None
    assert model.training and not model.vlm_backbone.training
    assert model.vla_head.inference_kv_cache is cache
    assert model.vla_head.current_start_frame == 7
    assert model.vla_head.num_inference_steps == 12
    assert torch.equal(torch.get_rng_state(), random_state)
    assert torch.isnan(batch['images'][:, :, 1:]).all()


def test_masked_labels_do_not_pollute_metrics():
    model = ProbeVLA()
    batch = make_batch(True)
    batch['action_masks'][:, 4, 1] = 0
    batch['actions'][:, 4, 1] = float('nan')
    assert model.compute_training_eval(
        batch)['metrics']['sampled_action_mae_first8'] == pytest.approx(2)


def test_sampling_failure_restores_state():
    model = ProbeVLA()
    model.fail = True
    cache = model.vla_head.inference_kv_cache
    random_state = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='sampling failed'):
        model.compute_training_eval(make_batch(False))
    assert model.training and not model.vlm_backbone.training
    assert model.vla_head.inference_kv_cache is cache
    assert model.vla_head.current_start_frame == 7
    assert model.vla_head.num_inference_steps == 12
    assert torch.equal(torch.get_rng_state(), random_state)


def test_official_schedule_rejects_wrong_step_count():
    model = ProbeVLA()
    model.vla_head.official_inference = True
    with pytest.raises(ValueError, match='requires 16'):
        model.compute_training_eval(make_batch(False), num_inference_steps=10)


def test_fully_masked_actions_are_not_reported_as_zero():
    batch = make_batch(False)
    batch['action_masks'].zero_()
    with pytest.raises(ValueError, match='fully masked'):
        ProbeVLA().compute_training_eval(batch)


@pytest.mark.parametrize('per_dimension', [False, True])
def test_command_metrics_clip_and_ignore_constant_coordinates(per_dimension):
    model = ProbeVLA()
    model.training_eval_action_groups = dict(arms=[0], hands=[1, 2])
    batch = make_batch(per_dimension)
    # Third coordinate is fixed at 3; its normalized output cannot move it.
    batch['stats'] = [dict(action=dict(min=[-1, -3, 3], max=[1, 3, 3]))]
    original_actions = batch['actions'].clone()
    result = model.compute_training_eval(batch)['metrics']
    assert result['sampled_action_mae_first8'] == pytest.approx(2)
    for count in (8, 24):
        assert result[f'sampled_arms_command_mae_first{count}'] == 1
        assert result[f'sampled_hands_command_mae_first{count}'] == 1.5
        assert result[f'sampled_hands_command_rmse_first{count}'] == (
            pytest.approx(math.sqrt(4.5)))
        assert result[f'sampled_hands_active_normalized_mae_first{count}'] == 1
    torch.testing.assert_close(
        batch['actions'], original_actions, equal_nan=True)


def test_command_metrics_use_each_samples_stats_and_exclude_masked_nan():
    prediction = torch.tensor([[[3., -2., 99.]], [[.5, .25, -99.]]])
    target = torch.tensor([[[0., float('nan'), 0.]], [[.25, -.25, 0.]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[0, 0, 1] = False
    stats = [
        dict(action=dict(min=[-1, -3, 2], max=[1, 3, 2])),
        dict(action=dict(min=[-4, -2, 7], max=[4, 2, 7]))
    ]
    metrics = _command_error_metrics(prediction, target, valid, stats,
                                     dict(arms=[0], hands=[1, 2]), [1])
    # Arm commands differ by 1 in each sample despite different min/max.
    assert metrics['sampled_arms_command_mae_first1'] == 1
    # One active hand command differs by 1, plus two fixed coordinates.
    assert metrics['sampled_hands_command_mae_first1'] == pytest.approx(1 / 3)
    assert metrics['sampled_hands_active_normalized_mae_first1'] == .5
    assert torch.isnan(target[0, 0, 1])
    assert prediction[0, 0, 2] == 99


@pytest.mark.parametrize('stats', [
    None, [], [{}], [dict(action=dict(min=[0], max=[1]))],
    [dict(action=dict(min=[0, 0, 0], max=[1, -1, 0]))]
])
def test_command_metrics_reject_missing_or_invalid_statistics(stats):
    model = ProbeVLA()
    model.training_eval_action_groups = dict(arms=[0], hands=[1, 2])
    batch = make_batch(True)
    batch['stats'] = stats
    with pytest.raises(ValueError, match='Command metrics require'):
        model.compute_training_eval(batch)


def test_command_metrics_do_not_report_empty_group_as_zero():
    values = torch.zeros(1, 24, 3)
    valid = torch.ones_like(values, dtype=torch.bool)
    valid[..., 1:] = False
    stats = [dict(action=dict(min=[-1, -3, 3], max=[1, 3, 3]))]
    with pytest.raises(ValueError, match='no valid nonconstant'):
        _command_error_metrics(values, values, valid, stats,
                               dict(hands=[1, 2]), [8, 24])
