# Copyright 2026 Limx Dynamics
"""Weights-only continuation must restore adapters and task projectors."""

import copy

import pytest
import torch
from mmengine import ConfigDict
from peft import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import PretrainedConfig

from fluxvla.engines.runners.ddp_train_runner import DDPTrainRunner


class _Policy(nn.Module):

    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.encoder = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, inputs):
        return self.head(self.encoder(inputs).sin())


def _runner(base, adapter_path=None, alpha=2):
    runner = object.__new__(DDPTrainRunner)
    runner.vla = copy.deepcopy(base)
    runner.cfg = ConfigDict(
        model=dict(
            use_lora=True,
            lora_rank=2,
            lora_alpha=alpha,
            lora_dropout=0.0,
            lora_target_modules=['encoder'],
            modules_to_save=['head']))
    runner.keep_lora_trainable_params_fp32 = True
    runner.enable_mixed_precision_training = True
    runner.mixed_precision_dtype = torch.bfloat16
    runner.keep_params_fp32 = False
    runner.lora_adapter_path = adapter_path
    runner._apply_lora()
    return runner


def _saved_adapter(tmp_path):
    torch.manual_seed(71)
    base = _Policy()
    trained = _runner(base)
    with torch.no_grad():
        for param in trained.vla.parameters():
            if param.requires_grad:
                param.add_(torch.randn_like(param) * 0.1)
    trained.vla.save_pretrained(tmp_path)
    return base, trained


def test_warm_start_restores_predictions_and_trainable_parameters(tmp_path):
    base, trained = _saved_adapter(tmp_path)
    restored = _runner(base, str(tmp_path))
    actual = get_peft_model_state_dict(restored.vla)
    expected = get_peft_model_state_dict(trained.vla)
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    frozen = restored.vla.get_base_model().encoder.base_layer.weight
    torch.testing.assert_close(
        frozen, base.encoder.weight.bfloat16(), rtol=0, atol=0)
    inputs = torch.randn(3, 4)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        before = trained.vla(inputs)
        after = restored.vla(inputs)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    after.square().mean().backward()
    for param in restored.vla.parameters():
        if param.requires_grad:
            assert param.dtype == torch.float32
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
    assert frozen.grad is None


def test_warm_start_rejects_different_adapter_scaling(tmp_path):
    base, _ = _saved_adapter(tmp_path)
    with pytest.raises(ValueError, match='lora_alpha differs'):
        _runner(base, str(tmp_path), alpha=4)


def test_warm_start_rejects_missing_task_projector(tmp_path):
    base, _ = _saved_adapter(tmp_path)
    path = tmp_path / 'adapter_model.safetensors'
    weights = load_file(path)
    del weights['base_model.model.head.bias']
    save_file(weights, path)
    with pytest.raises(ValueError, match='adapter keys differ'):
        _runner(base, str(tmp_path))


def test_merged_weights_cannot_silently_resume_peft(tmp_path):
    base, trained = _saved_adapter(tmp_path)
    restored = _runner(base)
    with pytest.raises(ValueError, match='merged export cannot restore'):
        restored._load_model_state(base.state_dict())
    restored._load_model_state(trained.vla.state_dict())
    for key, value in trained.vla.state_dict().items():
        torch.testing.assert_close(
            restored.vla.state_dict()[key], value, rtol=0, atol=0)
