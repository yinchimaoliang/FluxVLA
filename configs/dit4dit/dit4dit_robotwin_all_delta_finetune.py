# Copyright 2026 Limx Dynamics
"""RoboTwin DiT4DiT with FP32 diffusion and state-relative joint targets.

Start a new work directory: absolute-action checkpoints/statistics from the
base recipe are incompatible. The distinct statistics key makes accidental
evaluation with an old checkpoint fail instead of emitting wrong commands.
This addresses numerical and action-representation issues; rollout parity
with PI0.5 still requires training and clean/random evaluation.
"""

_base_ = ['./dit4dit_robotwin_all_full_finetune.py']

_DELTA_MASK = [True] * 6 + [False] + [True] * 6 + [False]
_STATISTIC_NAME = 'robotwin_all_delta'

# Preserve targets before the wrapper sees them. The frozen Cosmos text
# encoder/VAE stay BF16; the trainable transformer loads FP32 in both paths.
model = dict(
    vla_head=dict(diffusion_dtype='fp32'),
    vlm_backbone=dict(transformer_param_dtype='fp32'))
inference_model = dict(
    vla_head=dict(diffusion_dtype='fp32'),
    vlm_backbone=dict(transformer_param_dtype='fp32'))
# The old 32-rank run used 4 * 32 * 4 = 512 examples per update despite
# its "bs128" directory name. Use 4 * 32 * 1 = 128 on the same allocation.
runner = dict(
    keep_params_fp32=True,
    reduce_in_full_precision=True,
    grad_accumulation_steps=1,
    metric=dict(grad_accumulation_steps=1))

# Delta subtraction must see the raw state, before normalization/padding.
# Keep grippers absolute and normalize at the native 14D width. Raw qpos
# retains the angle branch that sin/cos discards for multi-turn joints.
_state_action_transforms = [
    dict(type='RelativeActions', mask=_DELTA_MASK),
    dict(
        type='NormalizeStatesAndActions',
        action_dim=16,
        state_dim=32,
        state_key='proprio',
        action_key='action',
        norm_type='quantile',
        clip_norm=False,
        output_dtype='float32'),
    dict(
        type='PrepareStateActionTargets',
        state_history_length=1,
        action_horizon=16,
        valid_action_dim=14),
]

train_dataloader = _base_.train_dataloader
train_dataloader.per_device_batch_size = 4
train_dataloader.dataset.statistic_name = _STATISTIC_NAME
train_dataloader.dataset.datasets.statistic_name = _STATISTIC_NAME
train_dataloader.dataset.auto_compute_statistics = dict(
    profile='absolute', delta_mask=_DELTA_MASK)
train_dataloader.dataset.datasets.transforms = [
    transform for transform in train_dataloader.dataset.datasets.transforms
    if transform.type not in ('SinCosKeys', 'NormalizeStatesAndActions',
                              'PrepareStateActionTargets')
] + _state_action_transforms

eval = _base_.eval
eval.unnorm_key = _STATISTIC_NAME
eval.dataset.statistic_name = _STATISTIC_NAME
eval.dataset.transforms = [
    transform for transform in eval.dataset.transforms if transform.type not in
    ('SinCosKeys', 'NormalizeStatesAndActions', 'PrepareStateActionTargets')
] + _state_action_transforms
eval.denormalize_action = dict(
    type='DenormalizeDeltaAction',
    action_dim=14,
    norm_type='quantile',
    statistic_name=_STATISTIC_NAME,
    delta_action_mask=_DELTA_MASK)
