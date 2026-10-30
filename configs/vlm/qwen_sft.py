# Copyright 2026 Limx Dynamics
"""Native Qwen3-VL / Qwen3.5 SFT over LLaVA JSON or JSONL annotations.

Set FLUXVLA_VLM_PATH and FLUXVLA_VLM_TRAIN before launching. Set
FLUXVLA_VLM_PROCESSOR=qwen3_5 for Qwen3.5 and FLUXVLA_VLM_VAL for validation.
The processor and model must refer to the same checkpoint/token vocabulary.
"""

import os

model_path = os.environ.get('FLUXVLA_VLM_PATH',
                            './checkpoints/Qwen3-VL-2B-Instruct')
processor_type = os.environ.get('FLUXVLA_VLM_PROCESSOR', 'qwen3_vl')
train_annotations = os.environ.get('FLUXVLA_VLM_TRAIN',
                                   './datasets/vlm/train.jsonl')
val_annotations = os.environ.get('FLUXVLA_VLM_VAL')
media_root = os.environ.get('FLUXVLA_VLM_MEDIA', './datasets/vlm')

model = dict(
    type='VLMWrapper',
    vlm_backbone=dict(
        type='CausalVLMBackbone',
        model_path=model_path,
        attn_implementation='sdpa',
        torch_dtype='fp32',
        freeze_vision=False),
    freeze_vlm_backbone=False)

transforms = [
    dict(type='BuildVLMMessages'),
    dict(
        type='ProcessVLMSFTInputs',
        processor_path=model_path,
        processor_type=processor_type,
        image_max_pixels=262144,
        video_max_pixels=131072,
        video_max_frames=16,
        truncation_max_length=4096),
]

train_dataloader = dict(
    per_device_batch_size=1,
    per_device_num_workers=2,
    dataset=dict(
        type='LLaVAFormatDataset',
        anno_json_path=train_annotations,
        data_root_path=media_root,
        transforms=transforms,
        seed=42))

runner = dict(
    type='FSDPTrainRunner',
    train_task='vlm',
    max_epochs=1,
    sharding_strategy='full-shard',
    fsdp_wrap_policy='model',
    max_grad_norm=1.0,
    grad_accumulation_steps=8,
    sampler='distributed',
    # The collator masks padding by lengths. ID 0 is only filler and is
    # never supervised; it need not be the tokenizer's EOS/PAD ID.
    collator=dict(type='VLMSFTCollator', pad_token_id=0),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', ),
        run_dir='work_dirs/vlm_sft',
        window_size=1),
    optimizer=dict(type='AdamW', lr=2e-5, weight_decay=0.01),
    lr_scheduler=dict(type='linear-warmup+cosine-decay', warmup_ratio=0.03),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True,
    save_iter_interval=500,
    save_epoch_interval=1,
    seed=42)

if val_annotations:
    eval_dataset = dict(
        type='LLaVAFormatDataset',
        anno_json_path=val_annotations,
        data_root_path=media_root,
        transforms=transforms,
        max_retry_times=0,
        seed=42)
    runner['evaluator'] = dict(
        type='VLMEvaluator', eval_every=100, batch_size=1, max_batches=100)
