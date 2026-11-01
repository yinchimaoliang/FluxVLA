# Copyright 2026 Limx Dynamics
"""Pi05 mixed RoboTwin action training and OSS LLaVA-665K language SFT.

Defaults target two 80-GB GPUs: three VLA updates followed by one VLM update.
Each update accumulates eight local batches of one sample (global batch 16).
--eval-after-train evaluates TextVQA; use the RoboTwin recipe for robot eval.
"""

_base_ = './pi05_paligemma_robotwin_all_data_full_finetune.py'

model = dict(
    freeze_llm_expert=False,
    pretrained_skip_prefixes=['llm_expert.embed_tokens.'],
    strict_mapping=True,
    preserve_fp32_residuals=True,
    vision_backbone=dict(preserve_fp32_residuals=True))

inference_model = dict(
    preserve_fp32_residuals=True,
    vision_backbone=dict(preserve_fp32_residuals=True))

# Reuse the complete robot pipeline, overriding only runtime data locations.
train_dataloader = _base_.train_dataloader
train_dataloader.per_device_batch_size = 1
train_dataloader.per_device_num_workers = 0
train_dataloader.dataset.datasets[0].data_root_path = [
    'datasets/robotwin_clean_lerobotv2.1',
    'datasets/robotwin_randomized_lerobotv2.1',
]

vlm_dataloader = dict(
    per_device_batch_size=1,
    per_device_num_workers=0,
    grad_accumulation_steps=8,
    collator=dict(type='Pi05VLMSFTCollator', pad_token_id=0),
    dataset=dict(
        type='LLaVAFormatDataset',
        anno_json_path='./datasets/vlm/llava665k/train.jsonl',
        data_root_path='/mnt/data/oss/limx_mani_data/raw_data/llava665k',
        index_cache_dir='./datasets/vlm/llava665k/.index',
        seed=42,
        transforms=[
            dict(type='BuildVLMMessages'),
            dict(
                type='ProcessPi05VLMSFTInputs',
                image_size=224,
                max_length=1024,
                tokenizer_path='./checkpoints/pi05_base'),
        ]))

runner = dict(
    train_task='mixed',
    mixed_training=dict(
        task_schedule=('vla', 'vla', 'vla', 'vlm'),
        sync_each_microbatch=True,
        loss_weights=dict(vla=1.0, vlm=1.0)),
    max_epochs=None,
    max_steps=1000,
    grad_accumulation_steps=8,
    # Mixed Pi05 shards MLP calls common to both execution paths.
    fsdp_wrap_policy='model',
    sharding_strategy='full-shard',
    enable_gradient_checkpointing=True,
    ema_decay=None,
    save_iter_interval=250,
    max_keep_ckpts=2,
    optimizer=dict(lr=2e-5),
    metric=dict(
        active_trackers=('jsonl', ),
        window_size=1,
        run_dir='work_dirs/pi05_robotwin_llava665k_mixed'))

eval = dict(
    _delete_=True,
    type='TextVQAEvalRunner',
    checkpoint_dir='work_dirs/pi05_robotwin_llava665k_mixed/checkpoints',
    annotation_path='./datasets/vlm/textvqa/TextVQA_0.5.1_val.json',
    image_root=('/mnt/data/oss/limx_mani_data/raw_data/'
                'llava665k/textvqa/train_images'),
    preprocessing=dict(
        tokenizer_path='./checkpoints/pi05_base',
        image_size=224,
        max_length=1024),
    compare_base=True,
    max_samples=None,
    max_new_tokens=32,
    keep_params_fp32=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    run_dir='work_dirs/pi05_robotwin_llava665k_mixed_textvqa')
