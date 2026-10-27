# Copyright 2026 Limx Dynamics
"""Uniform weighting must retain the original fresh fine-tuning recipe."""

from pathlib import Path

from mmengine import Config

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / 'configs/dreamzero'


def test_uniform_recipe_starts_fresh_and_only_changes_action_weighting():
    original = Config.fromfile(
        CONFIGS / 'dreamzero_robocasa_full_data_joint_delta_lora_finetune.py')
    uniform = Config.fromfile(
        CONFIGS /
        'dreamzero_robocasa_full_data_joint_delta_lora_uniform_finetune.py')
    assert uniform.model.vla_head.action_timestep_weighting == 'uniform'
    del uniform.model.vla_head.action_timestep_weighting
    assert uniform.to_dict() == original.to_dict()
    assert uniform.model.pretrained_name_or_path.endswith('/DreamZero-AgiBot')
    assert uniform.model.use_lora and uniform.model.init_lora_weights is True
    assert 'lora_adapter_path' not in uniform.runner
    assert uniform.runner.get('resume_from') is None
    assert uniform.runner.max_steps == 100000
    assert uniform.runner.optimizer.lr == 1e-5
    dataset = uniform.train_dataloader.dataset.datasets
    assert len(dataset.data_root_path) == 24
    assert dataset.max_chunk_size == 4
    assert dataset.action_window_size == 96
    assert uniform.eval.eval_chunk_size == 8
