# Copyright 2026 Limx Dynamics

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from mmengine import Config

from fluxvla.engines.runners.fsdp_train_runner import FSDPTrainRunner
from fluxvla.models.backbones.vlms.cosmos25 import Cosmos25Backbone
from fluxvla.models.backbones.vlms.outputs import VLMBackboneOutput
from fluxvla.models.vlas.dit4dit_vla import DiT4DiTVLA

CONFIG = (
    Path(__file__).resolve().parents[2] / 'configs/dit4dit' /
    'dit4dit_robodojo_full_finetune_horizon32_exec24.py')


class _RecordingBackbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([0.5]))
        self.dropout = nn.Dropout(0.2)
        self.last_images = None

    def forward(self, images, lang_tokens, lang_masks, **kwargs):
        self.last_images = images.clone()
        features = images.mean((1, 2, 3, 4))[:, None, None] * self.scale
        return VLMBackboneOutput(last_hidden_state=features.expand(-1, 3, 4))


class _ActionHead(nn.Module):
    action_dim = 16
    ori_action_dim = 14
    action_horizon = 32
    state_dim = 32
    num_inference_timesteps = 4

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([0.5]))
        self.failure = None
        self.last_prediction = None
        self.last_steps = None

    def forward(self, input_features, actions, **kwargs):
        return (self.scale * input_features.mean() - 1).square()

    def predict_action(self, input_features, states, **kwargs):
        self.last_steps = self.num_inference_timesteps
        prediction = torch.randn(
            input_features.shape[0],
            self.action_horizon,
            self.ori_action_dim,
            device=input_features.device) * self.scale
        if self.failure == 'raise':
            raise RuntimeError('sampler failed')
        if self.failure == 'nan':
            prediction.fill_(float('nan'))
        self.last_prediction = prediction.clone()
        return prediction


def _model():
    model = DiT4DiTVLA.__new__(DiT4DiTVLA)
    nn.Module.__init__(model)
    model.vlm_backbone = _RecordingBackbone()
    model.vla_head = _ActionHead()
    model.repeated_diffusion_steps = 1
    model.auxiliary_loss_weights = {}
    return model


def _batch():
    images = torch.ones(2, 3, 9, 16, 16)
    images[:, :, 1:] = 99  # Must never enter the diagnostic's conditioning.
    targets = torch.linspace(-1, 1, 2 * 32 * 16).reshape(2, 32, 16)
    masks = torch.ones_like(targets, dtype=torch.bool)
    masks[..., 14:] = False
    masks[1, 20:] = False
    targets[~masks] = float('nan')  # Invalid targets must not affect metrics.
    return dict(
        images=images,
        states=torch.zeros(2, 1, 32),
        actions=targets,
        action_masks=masks,
        lang_tokens=torch.ones(2, 8, dtype=torch.long),
        lang_masks=torch.ones(2, 8, dtype=torch.bool),
    )


def test_training_eval_uses_current_frame_and_masks_real_action_dimensions():
    model, batch = _model(), _batch()
    # Exercise exactly the entry point used by the FSDP training runner.
    runner = FSDPTrainRunner.__new__(FSDPTrainRunner)
    runner.vla = model
    result = runner.run_training_eval(batch, num_inference_steps=8, seed=42)

    torch.testing.assert_close(model.vlm_backbone.last_images,
                               batch['images'][:, :, :1])
    assert model.vla_head.last_steps == 8
    assert result['video_frames'] is None
    prediction = model.vla_head.last_prediction
    error = prediction - batch['actions'][..., :14]
    mask = batch['action_masks'][..., :14]
    expected = {
        'sampled_action_l1_norm': error.abs()[mask].mean(),
        'sampled_action_mse_norm': error.square()[mask].mean(),
        'sampled_first_action_l1_norm': error[:, 0].abs()[mask[:, 0]].mean(),
    }
    assert result['metrics'].keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(result['metrics'][key], value)

    # Change future observations and labels without changing the prediction.
    batch['images'][:, :, 1:] = -99
    batch['actions'].fill_(0.5)
    runner.run_training_eval(batch, num_inference_steps=8, seed=42)
    torch.testing.assert_close(prediction, model.vla_head.last_prediction)


@pytest.mark.parametrize('failure', [None, 'raise', 'nan'])
def test_training_eval_restores_rng_module_modes_and_sampler(failure):
    model, batch = _model(), _batch()
    model.train()
    model.vlm_backbone.dropout.eval()
    modes = [m.training for m in model.modules()]
    rng = torch.random.get_rng_state().clone()
    keys = set(model.state_dict())
    model.vla_head.failure = failure
    if failure is None:
        first = model.compute_training_eval(batch, num_inference_steps=8)
        second = model.compute_training_eval(batch, num_inference_steps=8)
        for key in first['metrics']:
            torch.testing.assert_close(first['metrics'][key],
                                       second['metrics'][key])
    else:
        with pytest.raises((RuntimeError, ValueError), match='sampl'):
            model.compute_training_eval(batch, num_inference_steps=8)
    assert model.vla_head.num_inference_timesteps == 4
    assert modes == [m.training for m in model.modules()]
    assert keys == set(model.state_dict())
    torch.testing.assert_close(rng, torch.random.get_rng_state())


@pytest.mark.parametrize('steps', [0, -1, True, 1.5])
def test_training_eval_rejects_invalid_sampler_steps(steps):
    with pytest.raises(ValueError, match='positive integer'):
        _model().compute_training_eval(_batch(), num_inference_steps=steps)


@pytest.mark.parametrize('invalid', ['empty', 'first', 'nonfinite'])
def test_training_eval_rejects_invalid_targets(invalid):
    batch = _batch()
    if invalid == 'empty':
        batch['action_masks'].zero_()
    elif invalid == 'first':
        batch['action_masks'][:, 0] = False
    else:
        batch['actions'][:, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='training_eval'):
        _model().compute_training_eval(batch)


def test_training_eval_forward_requires_explicit_batch_and_mode():
    model = _model()
    with pytest.raises(ValueError, match='training_eval_batch'):
        model(forward_mode='training_eval')
    with pytest.raises(ValueError, match='forward_mode'):
        model(forward_mode='invalid')


class _HookTransformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Linear(4, 4)])
        nn.init.constant_(self.transformer_blocks[0].weight, 0.1)
        nn.init.zeros_(self.transformer_blocks[0].bias)

    def forward(self, hidden_states, **kwargs):
        tokens = hidden_states.new_ones(hidden_states.shape[0], 3, 4)
        return (self.transformer_blocks[0](tokens), )


@pytest.mark.parametrize('detached', [True, False])
def test_cosmos_action_only_gradient_is_enabled_by_robodojo_config(detached):
    cfg = Config.fromfile(CONFIG)
    backbone = Cosmos25Backbone.__new__(Cosmos25Backbone)
    nn.Module.__init__(backbone)
    backbone.transformer = _HookTransformer()
    backbone.trainable = True
    backbone.detach_hidden_states = (
        True if detached else cfg.model.vlm_backbone.detach_hidden_states)
    backbone.frozen_submodules = ()
    backbone.extract_layer = 0
    backbone.split_future_frames = True
    backbone.future_loss_type = None  # Isolate ACTION loss, not video loss.
    backbone.num_frames_out = 9
    backbone.num_inference_steps = 1
    backbone.conditional_frame_timestep = 0.0001
    backbone._cached_hidden = []
    backbone._capture_hidden_enabled = True
    backbone._register_hidden_hook()
    backbone._encode_lang_tokens = Mock(return_value=torch.zeros(2, 8, 4))
    backbone._prepare_latents = Mock(
        return_value=(torch.zeros(2, 1, 3, 1, 1), torch.zeros(2, 1, 3, 1, 1),
                      torch.zeros(2, 1, 3, 1, 1), torch.zeros(2, 1, 3, 1, 1)))
    backbone.scheduler = SimpleNamespace(
        set_timesteps=Mock(),
        timesteps=torch.tensor([1.0]),
        sigmas=torch.tensor([1.0, 0.0]))
    model = _model()
    model.vlm_backbone = backbone
    model.train()
    output = model(**_batch())
    output['loss'].backward()
    grad = backbone.transformer.transformer_blocks[0].weight.grad
    assert model.vla_head.scale.grad is not None
    if detached:
        assert grad is None
    else:
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
