# Copyright 2026 Limx Dynamics
"""Keep the continuation comparison paired and preserve the GR1 contract."""

from pathlib import Path

import pytest
from mmengine import Config

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / 'configs/dreamzero'


def _load(variant):
    return Config.fromfile(
        CONFIGS / f'dreamzero_robocasa_joint_delta_lora_continue_{variant}.py')


def test_uniform_and_control_differ_only_in_action_weighting(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setenv('DREAMZERO_CONTINUE_ADAPTER', '/saved/100k-adapter')
    control = _load('control')
    uniform = _load('uniform')
    assert control.model.vla_head.action_timestep_weighting == 'scheduler'
    assert uniform.model.vla_head.action_timestep_weighting == 'uniform'
    uniform.model.vla_head.action_timestep_weighting = 'scheduler'
    assert uniform.to_dict() == control.to_dict()
    assert control.runner.lora_adapter_path == '/saved/100k-adapter'
    assert control.runner.resume_from is None
    assert control.model.pretrained_name_or_path.endswith('/DreamZero-AgiBot')
    assert control.model.use_lora
    assert control.runner.keep_lora_trainable_params_fp32
    assert control.runner.static_graph is False


@pytest.mark.parametrize('world_size', [1, 2, 16, 32])
def test_continuation_preserves_global_batch_and_full_multitask_data(
        monkeypatch, world_size):
    monkeypatch.setenv('WORLD_SIZE', str(world_size))
    cfg = _load('uniform')
    assert (cfg.train_dataloader.per_device_batch_size * world_size *
            cfg.runner.grad_accumulation_steps) == 64
    assert cfg.runner.max_steps == 2000
    assert cfg.runner.optimizer.lr == 1e-6
    assert cfg.runner.optimizer.type == 'AdamW'
    dataset = cfg.train_dataloader.dataset.datasets
    assert len(dataset.data_root_path) == 24
    assert dataset.max_chunk_size == 4
    assert dataset.get('train_episode_fraction', 1.0) == 1.0
    assert dataset.action_window_size == 4 * 24
    assert cfg.model.vla_head.dynamics_loss_weight == 1
    assert cfg.model.vla_head.action_loss_weight == 1
    assert cfg.eval.eval_chunk_size == 8
    assert cfg.eval.num_trials_per_task == 50
    assert 'stats' in cfg.runner.collator.meta_keys
    normalize = next(t for t in dataset.transforms
                     if t.type == 'NormalizeStatesAndActions')
    assert normalize.norm_type == 'min_max' and normalize.clip_norm
    groups = cfg.model.training_eval_action_groups
    assert groups.arms == list(range(14))
    assert groups.hands == list(range(14, 26))
    assert groups.waist == list(range(26, 29))


def test_continuation_rejects_incompatible_gpu_count(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '3')
    with pytest.raises(ValueError, match='global batch 64'):
        _load('uniform')
