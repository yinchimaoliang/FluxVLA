# Copyright 2026 Limx Dynamics
"""Variable DreamZero windows must accumulate as separate micro-batches."""

import copy
from collections import deque
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
from mmengine import Config
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from fluxvla.collators.dict_collator import DictCollator
from fluxvla.engines.runners.ddp_train_runner import DDPTrainRunner

ROOT = Path(__file__).resolve().parents[2]


class _WindowLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 1)

    def forward(self, actions, states):
        return {'loss': self.proj(actions).square().mean()}


@pytest.fixture
def process_group(tmp_path):
    if dist.is_initialized():
        assert dist.get_world_size() == 1
        yield
        return
    dist.init_process_group(
        'gloo',
        init_method=f'file://{tmp_path}/rendezvous',
        rank=0,
        world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('block_counts', [(1, 4), (3, 4)])
def test_dreamzero_variable_windows_accumulate_correctly(
        process_group, block_counts):
    cfg = Config.fromfile(
        ROOT / 'configs/dreamzero' /
        'dreamzero_robocasa_full_data_joint_delta_lora_finetune.py')
    cfg.merge_from_dict({
        'train_dataloader.per_device_batch_size': 1,
        'runner.grad_accumulation_steps': 2,
    })
    collator = DictCollator(
        keys=cfg.runner.collator['keys'],
        meta_keys=cfg.runner.collator.meta_keys)
    samples = [
        dict(
            states=np.zeros((blocks, 64), dtype=np.float32),
            actions=np.random.randn(24 * blocks, 32).astype(np.float32))
        for blocks in block_counts
    ]
    with pytest.raises(ValueError, match='same shape'):
        collator(samples)
    loader = DataLoader(
        samples,
        batch_size=cfg.train_dataloader.per_device_batch_size,
        collate_fn=collator)

    model = _WindowLoss()
    reference = copy.deepcopy(model)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    runner = object.__new__(DDPTrainRunner)
    runner.vla = DDP(
        model,
        find_unused_parameters=True,
        static_graph=cfg.runner.get('static_graph', True))
    runner.optimizer = torch.optim.SGD(runner.vla.parameters(), lr=0.01)
    runner.lr_scheduler = mock.Mock()
    runner.metric = mock.Mock(global_step=1)
    runner.device_id = torch.device('cpu')
    runner.enable_mixed_precision_training = False
    runner.keep_params_fp32 = True
    runner.mixed_precision_dtype = torch.bfloat16
    runner.grad_accumulation_steps = cfg.runner.grad_accumulation_steps
    runner.max_grad_norm = None
    runner.recent_losses = deque(maxlen=runner.grad_accumulation_steps)
    runner.wandb_mode = 'disabled'

    # Exercise first-iteration no_sync and later optimizer steps, using the
    # actual runner's loss scaling, backward, optimizer and zero_grad paths.
    initial_weight = model.proj.weight.detach().clone()
    for _ in range(2):
        reference_loss = sum(reference(**batch)['loss']
                             for batch in loader) / 2
        reference_loss.backward()
        reference_optimizer.step()
        reference_optimizer.zero_grad()
        with mock.patch.object(
                runner.vla, 'no_sync', wraps=runner.vla.no_sync) as no_sync:
            loss, _ = runner._run_accumulated_training_step(
                loader, iter(loader), sampler=None)
        no_sync.assert_called_once()
        assert np.isfinite(loss)
        for actual, expected in zip(model.parameters(),
                                    reference.parameters()):
            torch.testing.assert_close(actual, expected)
    assert not torch.equal(model.proj.weight, initial_weight)
