# Copyright 2026 Limx Dynamics
"""Two-A800 Pi05 RoboCasa control for the LLaVA mixed-training experiment.

30,000 robot updates, global batch 16 (2 GPUs x 1 sample x 8 accumulation).
Use the mixed runner with a VLA-only schedule so initialization, trainable
parameters, sharding and loss normalization match the mixed experiment.
No language dataset is loaded by this control. --eval-after-train evaluates
the final policy on the same 24 x 50 RoboCasa episodes as the mixed arm.
"""

_base_ = './pi05_paligemma_robocasa_full_data_full_finetune.py'

model = dict(
    freeze_llm_expert=False,
    pretrained_skip_prefixes=['llm_expert.embed_tokens.'],
    strict_mapping=True,
    preserve_fp32_residuals=True,
    vision_backbone=dict(preserve_fp32_residuals=True))

# Inherit all 24 task roots, transforms and fixed action/state statistics.
train_dataloader = dict(per_device_batch_size=1, per_device_num_workers=0)

runner = dict(
    train_task='mixed',
    mixed_training=dict(
        task_schedule=('vla', ),
        sync_each_microbatch=True,
        loss_weights=dict(vla=1.0, vlm=1.0)),
    max_epochs=None,
    max_steps=30000,
    grad_accumulation_steps=8,
    fsdp_wrap_policy='model',
    sharding_strategy='full-shard',
    enable_gradient_checkpointing=True,
    ema_decay=None,
    save_iter_interval=3000,
    max_keep_ckpts=3,
    optimizer=dict(lr=2e-5),
    lr_scheduler=dict(warmup_steps=900, decay_steps=30000, min_lr=2e-6),
    metric=dict(
        active_trackers=('jsonl', ),
        window_size=1,
        run_dir='work_dirs/pi05_robocasa_vla_control'))

# Inherit the robot evaluator, not the TextVQA evaluator of VLM recipes.
eval = dict(
    output_dir='work_dirs/pi05_robocasa_vla_control',
    deterministic_env=True,
    deterministic_action_sampling=True,
    save_video=False)
