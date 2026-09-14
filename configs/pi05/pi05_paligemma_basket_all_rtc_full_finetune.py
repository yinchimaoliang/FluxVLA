# Copyright 2026 Limx Dynamics
"""Reproduce the 2026-09-09 basket PI0.5 RTC training recipe locally.

8 GPUs x batch 8 x accumulation 2 = 128; 8 epochs = 94,104 updates.
With fewer GPUs, accumulation is increased to preserve global batch 128,
but rank-local RNG/sample ordering is not bit-identical to the 8-GPU run.

Prepare the 64-D checkpoint with tools/expand_pi05_action_checkpoint.py,
or set PI05_CHECKPOINT to the ORIGINAL expanded checkpoint for closer parity.
The original expansion implementation/weights were not included in the logs;
the local expansion is deterministic but is not claimed to be identical.
Use scripts/train.py so automatic action-window statistics are computed.
Do not replace those statistics with episode-averaged quantiles.
"""

import os

DATA_ROOT = os.environ.get(
    'BASKET_DATA_ROOT',
    '/mnt/data/oss/raw_data/oli_basket_full_task_20260521_20260617')
DATA_DATES = ('0521', '0522', '0525', '0526', '0527', '0601', '0602', '0603',
              '0609', '0610', '0611', '0615', '0616', '0617')
DATA_ROOTS = [
    os.path.join(
        DATA_ROOT,
        f'{date}_basket_full_task_prompt_delta_base_filtered_lerobotv2.1')
    for date in DATA_DATES
]
PI05_CHECKPOINT = os.environ.get(
    'PI05_CHECKPOINT',
    './checkpoints/pi05_base_action64_basket/model.safetensors')
PI05_TOKENIZER = os.environ.get('PI05_TOKENIZER', './checkpoints/pi05_base')

# Use torchrun's world size rather than assuming this host has eight GPUs.
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', '8'))
PER_DEVICE_BATCH_SIZE = 8
GLOBAL_BATCH_SIZE = 128
if WORLD_SIZE <= 0 or GLOBAL_BATCH_SIZE % (WORLD_SIZE * PER_DEVICE_BATCH_SIZE):
    raise ValueError('WORLD_SIZE must divide 16 to preserve global batch 128.')
GRAD_ACCUMULATION_STEPS = GLOBAL_BATCH_SIZE // (
    WORLD_SIZE * PER_DEVICE_BATCH_SIZE)
TOTAL_FRAMES = 1_505_539
TARGET_EPOCHS = 8
STEPS_PER_EPOCH = (TOTAL_FRAMES + GLOBAL_BATCH_SIZE - 1) // GLOBAL_BATCH_SIZE
MAX_STEPS = TARGET_EPOCHS * STEPS_PER_EPOCH

model = dict(
    type='PI05FlowMatching',
    llm_backbone=dict(
        type='ConditionGemmaModel',
        adarms_cond_dim=None,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=2048,
        initializer_range=0.02,
        intermediate_size=16384,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        use_cache=True,
        vocab_size=257152),
    vision_backbone=dict(
        type='SigLIPViTBackbone',
        vision_backbone_id='siglip_224',
        openpi_stem_fp32=True,
        vision_config=dict(
            attention_dropout=0.0,
            hidden_act='gelu_pytorch_tanh',
            hidden_size=1152,
            image_size=224,
            intermediate_size=4304,
            layer_norm_eps=1e-06,
            model_type='siglip_vision_model',
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=27,
            patch_size=14,
            projection_dim=2048,
            projector_hidden_act='gelu_fast',
            torch_dtype='float32',
            vision_use_head=False)),
    projector=dict(type='LinearProjector', in_dim=1152, out_dim=2048),
    proj_width=1024,
    n_action_steps=32,
    action_in_proj=dict(type='LinearProjector', in_dim=64, out_dim=1024),
    action_out_proj=dict(type='LinearProjector', in_dim=1024, out_dim=64),
    time_mlp_in=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_mlp_out=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_sampler='beta',
    time_beta_alpha=1.5,
    time_beta_beta=1.0,
    openpi_fp32_flow=True,
    max_action_dim=64,
    ori_action_dim=42,
    loss_action_dim=42,
    zero_padded_action_dims=True,
    trim_action_prediction=True,
    llm_expert=dict(
        type='ConditionGemmaModel',
        attention_bias=False,
        adarms_cond_dim=1024,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        pad_token_id=0,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        transformers_version='4.48.1',
        use_adarms=True,
        use_cache=True,
        vocab_size=257152),
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    pretrained_name_or_path=PI05_CHECKPOINT,
    # Fail fast on a 32-D checkpoint instead of silently skipping projections.
    strict_mapping=True,
    name_mapping={
        'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model',
        'time_mlp_in.projector': 'time_mlp_in',
        'time_mlp_out.projector': 'time_mlp_out',
        'action_in_proj.projector': 'action_in_proj',
        'action_out_proj.projector': 'action_out_proj',
        'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
        'llm_expert.embed_tokens':
        'paligemma_with_expert.gemma_expert.lm_head',
    },
    params_to_change_dtype=[
        'llm_expert.llm.model.layers',
        'vlm_backbone.vlm.model.language_model.layers',
        'vlm_backbone.vlm.model.vision_tower',
        'vlm_backbone.vlm.model.multi_modal_projector',
    ],
    rtc_training_config=dict(
        enabled=True, max_delay=5, distribution='uniform', temperature=1.0))

# No inherited ALOHA inference/evaluation settings: this is a training recipe.
inference_model = model.copy()

train_dataloader = dict(
    per_device_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        shuffle=True,
        seed=42,
        reshuffle_each_epoch=True,
        auto_compute_statistics=dict(profile='absolute'),
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'action', 'timestamp'],
        datasets=dict(
            type='ParquetDataset',
            data_root_path=DATA_ROOTS,
            action_key='action',
            action_window_size=32,
            window_start_idx=0,
            supervise_terminal_padding=True,
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    parquet_keys=[
                        'observation.state', 'timestamp', 'actions', 'info',
                        'stats', 'action_masks'
                    ],
                    video_keys=[
                        'observation.images.head',
                        'observation.images.left_wrist'
                    ],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    }),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=None,
                    state_dim=None,
                    state_key='proprio',
                    action_key='action',
                    norm_type='quantile',
                    discrete_state_dims=[31, 32],
                    discrete_action_dims=[40, 41],
                    discrete_norm_type='min_max',
                    output_dtype='float32'),
                dict(type='PreparePromptWithState'),
                dict(
                    type='ProcessPrompts',
                    max_len=200,
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=PI05_TOKENIZER)),
                dict(type='PadStatesAndActions', model_action_dim=64),
                dict(
                    type='ResizeImagesWithPad',
                    height=224,
                    width=224,
                    backend='pil'),
                dict(type='SimpleNormalizeImages'),
                dict(type='OpenPIImageAugment', base_camera_indices=(0, )),
            ])))

runner = dict(
    type='FSDPTrainRunner',
    max_steps=MAX_STEPS,
    max_epochs=None,
    grad_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    # Local runner equivalent of source save_steps=[11763, 23526, ..., 94104].
    save_iter_interval=STEPS_PER_EPOCH,
    max_keep_ckpts=TARGET_EPOCHS,
    ema_decay=0.99,
    seed=42,
    optimizer=dict(
        type='AdamW',
        lr=2.5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        weight_decay_all_params=True,
        foreach=False,
        fused=True),
    max_grad_norm=1.0,
    sharding_strategy='global-shard-grad-op',
    fsdp_wrap_policy='execution-block',
    reduce_in_full_precision=True,
    sampler=None,
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        schedule_style='openpi',
        warmup_steps=1000,
        decay_steps=MAX_STEPS,
        min_lr=0.0),
    tokenizer=dict(type='PretrainedTokenizer', model_path=PI05_TOKENIZER),
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'timestamp', 'images', 'img_masks', 'lang_tokens',
            'lang_masks', 'actions', 'action_masks'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=100),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True,
    change_key_name=False)
