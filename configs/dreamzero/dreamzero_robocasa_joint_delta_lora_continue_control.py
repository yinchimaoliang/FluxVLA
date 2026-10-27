# Copyright 2026 Limx Dynamics
"""Original-objective control for the multi-task continuation experiment.

Start from the original AgiBot base plus the saved 100k adapters. The optimizer
and the 2,000-update schedule start fresh. Do not pass --resume-from or replace
the base with a merged RoboCasa checkpoint. Use a new output directory.

All 24 tasks, augmentation and 1..4-block windows are inherited. Under torchrun
WORLD_SIZE is the total number of GPU processes, so accumulation gives global
batch 64 with two samples per GPU. See docs/dreamzero_robocasa_continuation.md.
"""

import os

_base_ = './dreamzero_robocasa_full_data_joint_delta_lora_finetune.py'

_WORLD_SIZE = int(os.environ.get('WORLD_SIZE', '1'))
if _WORLD_SIZE <= 0 or 64 % (2 * _WORLD_SIZE):
    raise ValueError('DreamZero continuation requires a GPU count that '
                     'divides 32 for global batch 64 at batch 2 per GPU.')

_ADAPTER_PATH = os.environ.get(
    'DREAMZERO_CONTINUE_ADAPTER', './work_dirs/'
    'dreamzero_robocasa_full_data_joint_delta_lora_finetune_'
    '0285ca940_bs64_datafix')

model = dict(
    training_eval_action_groups=dict(
        arms=list(range(14)),
        hands=list(range(14, 26)),
        waist=list(range(26, 29))),
    vla_head=dict(
        dynamics_loss_weight=1.0,
        action_loss_weight=1.0,
        action_timestep_weighting='scheduler'))

train_dataloader = dict(per_device_batch_size=2)

runner = dict(
    lora_adapter_path=_ADAPTER_PATH,
    resume_from=None,
    max_steps=2000,
    grad_accumulation_steps=64 // (2 * _WORLD_SIZE),
    optimizer=dict(lr=1e-6),
    # 100 warmup updates, then cosine decay over this continuation only.
    lr_scheduler=dict(warmup_ratio=0.05),
    save_iter_interval=500,
    max_keep_ckpts=4,
    # Sampled training diagnostics, not held-out validation or task success.
    evaluator=dict(eval_every=250))
