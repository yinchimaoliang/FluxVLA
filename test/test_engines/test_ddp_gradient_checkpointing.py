# Copyright 2026 Limx Dynamics
"""DDP checkpoint setup must support models without an LLM layer class."""

import copy
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn
from mmengine import ConfigDict
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import \
    CheckpointWrapper
from transformers import PretrainedConfig

from fluxvla.engines.runners import ddp_train_runner
from fluxvla.engines.runners.ddp_train_runner import DDPTrainRunner


class _BackboneBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, inputs):
        return self.proj(inputs).sin()


class _ExpertBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, inputs):
        return self.proj(inputs).cos()


class _TinyVLA(nn.Module):

    def __init__(self, expert_layer_cls):
        super().__init__()
        self.config = PretrainedConfig()
        self.backbone = _BackboneBlock()
        self.llm_expert = nn.Module()
        self.llm_expert.transformer_layer_cls = expert_layer_cls
        self.llm_expert.block = _ExpertBlock()
        self.output = nn.Linear(4, 1)

    def freeze_backbones(self):
        self.requires_grad_(False)

    def from_pretrained(self):
        pass

    def forward(self, inputs):
        return self.output(self.llm_expert.block(self.backbone(inputs)))


@pytest.mark.parametrize(('backbone_cls', 'expert_cls', 'enabled'), [
    (None, None, True),
    (_BackboneBlock, None, True),
    (None, _ExpertBlock, True),
    (_BackboneBlock, _ExpertBlock, True),
    (None, None, False),
])
def test_ddp_lora_checkpoint_setup_and_backward(backbone_cls, expert_cls,
                                                enabled):
    runner = object.__new__(DDPTrainRunner)
    runner.vla = _TinyVLA(expert_cls)
    runner.cfg = ConfigDict(
        model=dict(
            use_lora=True,
            lora_rank=2,
            lora_dropout=0.0,
            lora_target_modules=['proj'],
            modules_to_save=['output'],
        ))
    runner.device_id = 0
    runner.lora_before_device_move = True
    runner.keep_lora_trainable_params_fp32 = False
    runner.enable_gradient_checkpointing = enabled
    runner.llm_transformer_layer_cls = backbone_cls
    runner.static_graph = True
    reference = []

    # Keep setup on CPU, but run real PEFT injection and checkpoint wrapping.
    def keep_on_cpu(device):
        reference.append(copy.deepcopy(runner.vla))

    logger = SimpleNamespace(is_rank_zero=lambda: False)
    with mock.patch.object(torch.cuda, 'empty_cache'), mock.patch.object(
            torch.cuda, 'set_device'), mock.patch.object(
                runner, '_move_vla_to_device', side_effect=keep_on_cpu
            ), mock.patch.object(
                runner, '_setup_optimizer_and_scheduler'), mock.patch.object(
                    ddp_train_runner, 'overwatch', logger), mock.patch.object(
                        ddp_train_runner,
                        'DDP',
                        side_effect=lambda model, **kwargs: model) as wrap_ddp:
        runner.run_setup(n_train_examples=2)

    wrap_ddp.assert_called_once()
    model = runner.vla.get_base_model()
    assert isinstance(model.backbone,
                      CheckpointWrapper) is (enabled
                                             and backbone_cls is not None)
    assert isinstance(model.llm_expert.block,
                      CheckpointWrapper) is (enabled
                                             and expert_cls is not None)

    inputs = torch.randn(2, 4)
    actual = runner.vla(inputs)
    expected = reference[0](inputs)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    trainable_grads = []
    for parameter, reference_parameter in zip(runner.vla.parameters(),
                                              reference[0].parameters()):
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad,
                                       reference_parameter.grad)
            trainable_grads.append(parameter.grad)
    assert any(gradient.abs().sum() > 0 for gradient in trainable_grads)
