# Copyright 2026 Limx Dynamics

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from mmengine import Config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT / 'configs/dit4dit' /
    'dit4dit_robodojo_full_finetune_horizon32_exec24.py')


def _transform(transforms, name):
    return next(t for t in transforms if t.type == name)


def test_training_budget_uses_robodojo_updates_and_same_effective_batch():
    cfg = Config.fromfile(CONFIG)

    assert cfg.train_dataloader.per_device_batch_size == 2
    assert cfg.train_dataloader.per_device_num_workers == 4
    assert cfg.runner.max_epochs is None
    assert cfg.runner.max_steps == 120000
    assert cfg.runner.grad_accumulation_steps == 4
    assert cfg.runner.metric.grad_accumulation_steps == 4
    assert (cfg.train_dataloader.per_device_batch_size *
            cfg.runner.grad_accumulation_steps == 8)
    assert cfg.runner.enable_gradient_checkpointing is True
    assert cfg.runner.mixed_precision_dtype == 'bf16'
    assert cfg.runner.sharding_strategy == 'full-shard'
    assert cfg.runner.lr_scheduler.warmup_steps == 2000
    assert cfg.runner.lr_scheduler.min_lr == 5e-7
    assert cfg.runner.save_epoch_interval == 1
    assert cfg.runner.save_iter_interval == 2000
    assert cfg.runner.evaluator.type == 'training-eval'
    assert cfg.runner.evaluator.eval_every == 1000
    assert cfg.runner.evaluator.num_inference_steps == 4
    assert cfg.runner.evaluator.save_video is False
    assert cfg.runner.optimizer.lr == 3e-5
    assert cfg.runner.optimizer.paramwise_learning_rate == {
        'vla_head': 1e-4,
        'vlm_backbone.transformer': 1e-5,
    }


def test_action_video_and_inference_contracts_are_consistent():
    cfg = Config.fromfile(CONFIG)
    head = cfg.model.vla_head
    dataset = cfg.train_dataloader.dataset.datasets

    assert cfg.model.type == 'DiT4DiTVLA'
    assert cfg.model.freeze_vlm_backbone is False
    assert cfg.model.vlm_backbone.detach_hidden_states is False
    assert cfg.model.repeated_diffusion_steps == 4
    assert cfg.model.vlm_backbone.frozen_submodules == ['text_encoder', 'vae']
    assert head.action_dim == 16
    assert head.ori_action_dim == 14
    assert head.state_dim == 32
    assert head.action_horizon == dataset.action_window_size == 32
    assert head.future_action_window_size == head.action_horizon - 1
    assert head.num_inference_timesteps == 4
    assert dataset.frame_window_size == cfg.model.vlm_backbone.num_frames_out
    assert dataset.frame_window_size == 9
    assert dataset.frame_sample_stride == 4
    assert ((dataset.frame_window_size - 1) *
            dataset.frame_sample_stride == head.action_horizon)
    assert cfg.eval.eval_chunk_size == cfg.themis.evaluation.execute_horizon
    assert cfg.eval.eval_chunk_size == 24
    assert cfg.eval.eval_chunk_size <= head.action_horizon

    inference = cfg.inference_model
    assert inference.init_empty_weights is True
    assert inference.vlm_backbone.load_pretrained_weights is False
    assert 'load_pretrained_weights' not in cfg.model.vlm_backbone
    assert inference.vla_head == head
    assert 'pretrained_name_or_path' not in cfg.model


def test_robodojo_data_statistics_and_prompts():
    cfg = Config.fromfile(CONFIG)
    wrapper = cfg.train_dataloader.dataset
    dataset = wrapper.datasets
    assert wrapper.type == 'DistributedRepeatingDataset'
    assert 'sampling_weights' not in wrapper
    assert 'sampling_unit' not in wrapper
    assert wrapper.auto_compute_statistics.profile == 'absolute'
    assert dataset.data_root_path == ['./datasets/RoboDojo_lerobot_v21_video']
    assert dataset.use_delta is False
    assert dataset.supervise_terminal_padding is True
    assert wrapper.statistic_name == 'robodojo_arx_x5'
    assert dataset.statistic_name == wrapper.statistic_name
    assert cfg.eval.dataset.unnorm_key == wrapper.statistic_name
    assert cfg.eval.denormalize_action.statistic_name == wrapper.statistic_name
    assert cfg.eval.denormalize_action.action_dim == 14
    assert cfg.eval.denormalize_action.norm_type == 'min_max'
    assert cfg.eval.denormalize_action.clip_normalized_action is True

    for transforms in (dataset.transforms, cfg.eval.dataset.transforms):
        prompt = _transform(transforms, 'ProcessCosmos25Prompt')
        assert prompt.tokenizer == cfg.runner.tokenizer
        assert prompt.tokenizer.model_path == (
            cfg.model.vlm_backbone.base_model + '/tokenizer')
        assert prompt.remove_input_key is True
        video = _transform(transforms, 'PrepareVideo')
        assert video.num_views == 3
        assert video.tile_direction == 'top_bottom_pair'
        assert video.top_view == 0
        assert video.bottom_views == (1, 2)
        targets = _transform(transforms, 'PrepareStateActionTargets')
        assert targets.action_horizon == 32
        assert targets.valid_action_dim == 14
        normalization = _transform(transforms, 'NormalizeStatesAndActions')
        assert normalization.state_norm_type == 'none'
        assert normalization.output_dtype == 'float32'
        assert normalization.normalization_epsilon == 1e-6


def test_robodojo_evaluation_protocol_preserves_existing_task_settings():
    cfg = Config.fromfile(CONFIG)
    reference = Config.fromfile(
        ROOT / 'configs/dit4dit/dit4dit_robodojo_full_finetune.py')
    assert cfg.eval.dataset.type == 'RoboDojoEvalDataset'
    assert 'type' not in cfg.eval  # Themis, not RobotwinEvalRunner.
    assert cfg.eval.report_kind == cfg.eval.task_suite_name == 'robodojo'
    expected_themis = reference.themis.deepcopy()
    expected_themis.evaluation.execute_horizon = 24
    assert cfg.themis == expected_themis
    evaluation = cfg.themis.evaluation
    assert cfg.eval.num_trials_per_task == evaluation.episodes_per_task
    assert (cfg.eval.num_trials_per_task_overrides ==
            cfg.themis.evaluation.episodes_per_task_overrides)


@pytest.mark.parametrize('valid_steps', [5, 32])
def test_synthetic_train_and_eval_batches_match(valid_steps):
    from fluxvla.collators.dict_collator import DictCollator
    from fluxvla.engines import (build_dataset_from_cfg,
                                 build_transform_from_cfg)

    class FakeTokenizer:
        pad_token_id = 0
        model_max_length = 512

        def apply_chat_template(self, conversations, **kwargs):
            return [11, 12] + [0] * 510

    cfg = Config.fromfile(CONFIG)
    rng = np.random.default_rng(42)
    views = [rng.integers(0, 256, (8, 8, 3), dtype=np.uint8) for _ in range(3)]
    states = np.linspace(-1, 1, 14, dtype=np.float32)
    actions = np.linspace(-1, 1, 32 * 14, dtype=np.float32).reshape(32, 14)
    stats = {'action': {'min': [-2.0] * 14, 'max': [2.0] * 14}}
    sample = dict(
        images=[view.transpose(2, 0, 1) for view in views for _ in range(9)],
        img_masks=np.ones(27, dtype=bool),
        frame_masks=np.ones(9, dtype=bool),
        states=states.copy(),
        actions=actions.copy(),
        action_masks=np.arange(32) < valid_steps,
        timestamp=np.asarray(0.0, dtype=np.float32),
        info={},
        stats=stats,
        task_description='stack bowls',
    )
    observation = dict(zip(cfg.themis.transport.image_keys, views))
    observation.update(states=states.copy(), task_description='stack bowls')
    with patch(
            'fluxvla.engines.build_tokenizer_from_cfg',
            return_value=FakeTokenizer()):
        # Start after video decoding; no external data/checkpoints are needed.
        for transform_cfg in cfg.train_dataloader.dataset.datasets.transforms[
                1:]:
            sample = build_transform_from_cfg(transform_cfg)(sample)
        eval_dataset = build_dataset_from_cfg(cfg.eval.dataset)

    with patch.object(torch.Tensor, 'cuda', lambda self: self):
        eval_batch, replay = eval_dataset(observation)
    collator_cfg = dict(cfg.runner.collator)
    collator_cfg.pop('type')
    train_batch = DictCollator(**collator_cfg)([sample])

    assert replay is None
    assert train_batch['images'].shape == (1, 3, 9, 336, 224)
    assert eval_batch['images'].shape == (1, 3, 1, 336, 224)
    torch.testing.assert_close(
        train_batch['images'][:, :, :1],
        eval_batch['images'],
        atol=1e-6,
        rtol=1e-5)
    assert eval_batch['images'].min() >= -1
    assert eval_batch['images'].max() <= 1
    assert train_batch['states'].shape == eval_batch['states'].shape == (1, 1,
                                                                         32)
    torch.testing.assert_close(train_batch['states'].bfloat16(),
                               eval_batch['states'])
    torch.testing.assert_close(train_batch['lang_tokens'],
                               eval_batch['lang_tokens'])
    assert train_batch['actions'].shape == (1, 32, 16)
    assert train_batch['action_masks'].shape == (1, 32, 16)
    assert train_batch['action_masks'][..., :14].sum() == valid_steps * 14
    assert not train_batch['action_masks'][..., 14:].any()
    assert not train_batch['actions'][..., 14:].any()

    denormalize = build_transform_from_cfg(
        dict(
            cfg.eval.denormalize_action, norm_stats={'robodojo_arx_x5':
                                                     stats}))
    result = denormalize({'action': sample['actions'][:, :14]})
    assert result.shape == (32, 14)
    np.testing.assert_allclose(result, actions, atol=1e-6)


def test_robodojo_clips_sampled_joint_and_gripper_commands():
    from fluxvla.engines import build_transform_from_cfg

    cfg = Config.fromfile(CONFIG)
    low = np.full(14, -2.0)
    high = np.full(14, 2.0)
    low[[6, 13]], high[[6, 13]] = 0.0, 1.0
    stats = {'action': {'min': low.tolist(), 'max': high.tolist()}}
    denormalize = build_transform_from_cfg(
        dict(
            cfg.eval.denormalize_action, norm_stats={'robodojo_arx_x5':
                                                     stats}))
    predicted = np.stack([np.full(14, -5.0), np.full(14, 5.0)])
    result = denormalize({'action': predicted})
    np.testing.assert_allclose(result[0], low, atol=1e-6)
    np.testing.assert_allclose(result[1], high, atol=1e-6)
