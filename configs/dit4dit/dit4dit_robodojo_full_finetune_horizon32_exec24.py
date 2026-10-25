# Copyright 2026 Limx Dynamics
"""DiT4DiT full fine-tuning on all RoboDojo ARX-X5 data.

Adapted from dit4dit_robotwin_all_data_full_finetune_horizon32_exec24.py.
Preserve its Cosmos initialization, optimizer, three-epoch training budget,
32-step action horizon, and 24-step execution horizon. Absolute 14D joint
commands are padded to 16; interleaved sin/cos states are padded to 32.
Train on 9 frames at stride 4, spanning 32 control intervals. Three cameras
are tiled into a 336x224 video with matching [-1, 1] train/eval inputs.
Use one RoboDojo dataset, not RoboTwin's clean/randomized 1:1 mixture.
Evaluation uses the existing RoboDojo Themis task/episode protocol.
"""

# Update these paths for the training/evaluation machine.
_DATA_ROOTS = ['./datasets/RoboDojo_lerobot_v21_video']
_STATISTIC_NAME = 'robodojo_arx_x5'
_TOKENIZER_PATH = './checkpoints/Cosmos-Predict2.5-2B/tokenizer'
_ROBODOJO_ROOT = '/root/projects/RoboDojo'
_EXECUTE_HORIZON = 24

_ROBODOJO_GENERALIZATION_BASE_TASKS = (
    'stack_bowls',
    'push_T',
    'pack_objects_into_box',
    'fold_clothes',
    'hang_mugs',
    'sweep_blocks',
    'pour_liquid_into_cup',
    'make_toast',
    'arrange_largest_number',
    'sort_nesting_dolls_by_size',
    'store_laptop_and_headphones',
    'stack_blocks',
)
_ROBODOJO_EPISODE_OVERRIDES = {
    task_name: 25
    for base_task in _ROBODOJO_GENERALIZATION_BASE_TASKS
    for task_name in (base_task, f'{base_task}_random')
}

model = dict(
    type='DiT4DiTVLA',
    freeze_vlm_backbone=False,
    repeated_diffusion_steps=4,
    vla_head=dict(
        type='DiT4DiTActionHead',
        action_dim=16,
        action_horizon=32,
        action_model_type='DiT-B',
        add_pos_embed=True,
        diffusion_model_cfg=dict(
            cross_attention_dim=2048,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            num_layers=16,
            output_dim=2560,
            positional_embeddings=None,
        ),
        future_action_window_size=31,
        hidden_size=2560,
        max_seq_len=1024,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_inference_timesteps=4,
        num_timestep_buckets=1000,
        ori_action_dim=14,
        state_dim=32,
    ),
    vlm_backbone=dict(
        type='Cosmos25Backbone',
        base_model='./checkpoints/Cosmos-Predict2.5-2B',
        conditional_frame_timestep=0.0001,
        detach_hidden_states=True,
        extract_layer=17,
        fixed_seed=None,
        flow_matching_high_sigma_min=None,
        flow_matching_high_sigma_ratio=None,
        flow_matching_time_distribution='uniform',
        frozen_submodules=['text_encoder', 'vae'],
        fsdp_min_num_params=0,
        future_loss_type='flow_matching',
        local_files_only=True,
        num_frames_out=9,
        num_inference_steps=1,
        revision='diffusers/base/post-trained',
        split_future_frames=True,
        torch_dtype='bf16',
        trainable=True,
    ),
)

inference_model = dict(
    model,
    init_empty_weights=True,
    vlm_backbone=dict(model['vlm_backbone'], load_pretrained_weights=False))

train_dataloader = dict(
    dataset=dict(
        type='DistributedRepeatingDataset',
        auto_compute_statistics=dict(profile='absolute', ),
        datasets=dict(
            type='ParquetDataset',
            action_key='action',
            action_window_size=32,
            data_root_path=_DATA_ROOTS,
            frame_sample_stride=4,
            frame_window_size=9,
            statistic_name=_STATISTIC_NAME,
            supervise_terminal_padding=True,
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    name_mappings={
                        'actions': ['actions'],
                        'observation.state': ['states'],
                    },
                    parquet_keys=[
                        'observation.state',
                        'timestamp',
                        'actions',
                        'info',
                        'stats',
                        'action_masks',
                    ],
                    video_backend='pyav',
                    video_keys=[
                        'observation.images.cam_high',
                        'observation.images.cam_left_wrist',
                        'observation.images.cam_right_wrist',
                    ],
                ),
                dict(
                    type='ProcessCosmos25Prompt',
                    input_key='task_description',
                    remove_input_key=True,
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_max_length=512,
                        model_path=_TOKENIZER_PATH,
                    ),
                ),
                dict(
                    type='ResizeImages',
                    backend='torch',
                    height=224,
                    output_layout='flattened_chw',
                    scale_divisor=255.0,
                    width=224,
                ),
                dict(
                    type='NormalizeImages',
                    means=[0.5, 0.5, 0.5],
                    stds=[0.5, 0.5, 0.5],
                ),
                dict(
                    type='PrepareVideo',
                    bottom_height_ratio=0.5,
                    bottom_views=(1, 2),
                    combine_view_masks=True,
                    frame_window_size=9,
                    num_views=3,
                    tile_direction='top_bottom_pair',
                    top_view=0,
                ),
                dict(
                    type='SinCosKeys',
                    backend='torch',
                    expand_axis=0,
                    interleave=True,
                    keys=['states'],
                    target_dims=dict(states=32, ),
                ),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=16,
                    action_key='action',
                    norm_type='min_max',
                    normalization_epsilon=1e-06,
                    output_dtype='float32',
                    state_dim=None,
                    state_key='proprio',
                    state_norm_type='none',
                ),
                # Keep temporal padding masks while masking the two padded
                # action dimensions expected by the 16D ActionDiT head.
                dict(
                    type='PrepareStateActionTargets',
                    state_history_length=1,
                    action_horizon=32,
                    valid_action_dim=14,
                ),
            ],
            use_delta=False,
            window_start_idx=0,
        ),
        name_mappings={
            'action': ['action'],
            'observation.state': ['proprio'],
        },
        reshuffle_each_epoch=True,
        seed=42,
        shuffle=True,
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_STATISTIC_NAME,
    ),
    per_device_batch_size=8,
    per_device_num_workers=4,
)

runner = dict(
    type='FSDPTrainRunner',
    change_key_name=False,
    collator=dict(
        type='DictCollator',
        keys=[
            'states',
            'timestamp',
            'images',
            'img_masks',
            'actions',
            'action_masks',
            'frame_masks',
            'lang_tokens',
            'lang_masks',
        ],
        meta_keys=['info', 'stats'],
    ),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    grad_accumulation_steps=1,
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        min_lr=5e-07,
        warmup_ratio=0.05,
    ),
    max_epochs=3,
    max_grad_norm=1.0,
    max_keep_ckpts=3,
    max_steps=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        grad_accumulation_steps=1,
        run_dir='work_dirs',
        window_size=100,
    ),
    mixed_precision_dtype='bf16',
    optimizer=dict(
        type='AdamW',
        betas=(0.9, 0.95),
        eps=1e-08,
        lr=3e-05,
        paramwise_learning_rate={
            'vla_head': 0.0001,
            'vlm_backbone.transformer': 1e-05,
        },
        weight_decay=1e-08,
        weight_decay_all_params=True,
    ),
    pre_fsdp_param_dtype='fp32',
    reduce_in_full_precision=False,
    sampler=None,
    save_epoch_interval=1,
    save_iter_interval=10000000,
    seed=42,
    sharding_strategy='global-shard-grad-op',
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_max_length=512,
        model_path=_TOKENIZER_PATH,
    ),
)

eval = dict(
    report_kind='robodojo',
    dataset=dict(
        type='RoboDojoEvalDataset',
        transforms=[
            dict(
                type='ProcessEvalInputs',
                img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
            ),
            dict(
                type='ProcessCosmos25Prompt',
                input_key='task_description',
                remove_input_key=True,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_max_length=512,
                    model_path=_TOKENIZER_PATH,
                ),
            ),
            dict(
                type='ResizeImages',
                backend='torch',
                height=224,
                key='pixel_values',
                output_layout='flattened_chw',
                width=224,
            ),
            # RoboDojoEvalDataset requires tensors. Normalize raw [0, 255]
            # pixels here (not again in ResizeImages) to match training.
            dict(
                type='SimpleNormalizeImages',
                key='pixel_values',
                output_type='torch',
            ),
            dict(
                type='PrepareVideo',
                bottom_height_ratio=0.5,
                bottom_views=(1, 2),
                combine_view_masks=True,
                frame_window_size=1,
                num_views=3,
                tile_direction='top_bottom_pair',
                top_view=0,
            ),
            dict(
                type='SinCosKeys',
                backend='torch',
                expand_axis=0,
                interleave=True,
                keys=['states'],
                target_dims=dict(states=32, ),
            ),
            dict(
                type='NormalizeStatesAndActions',
                action_dim=16,
                action_key='action',
                norm_type='min_max',
                normalization_epsilon=1e-06,
                output_dtype='float32',
                state_dim=None,
                state_key='proprio',
                statistics_key='norm_stats',
                state_norm_type='none',
            ),
            dict(
                type='PrepareStateActionTargets',
                state_history_length=1,
                action_horizon=32,
                valid_action_dim=14,
            ),
        ],
        unnorm_key=_STATISTIC_NAME,
    ),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        action_dim=14,
        norm_type='min_max',
        statistic_name=_STATISTIC_NAME,
    ),
    eval_chunk_size=_EXECUTE_HORIZON,
    mixed_precision_dtype='bf16',
    model_family='dit4dit',
    num_trials_per_task=50,
    num_trials_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
    task_suite_name='robodojo',
    unnorm_key=_STATISTIC_NAME,
)

themis = dict(
    transport=dict(
        image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        state_keys=['states'],
        unnorm_key=_STATISTIC_NAME,
    ),
    evaluation=dict(
        environment=dict(
            type='RoboDojoEnvironment',
            task_name='all',
            env_cfg_type='arx_x5',
            robodojo_root=_ROBODOJO_ROOT,
            device_id=1,
            action_mode='joint',
            headless=True,
            save_videos=False,
        ),
        policy_client=dict(type='FluxVLAPolicyClient'),
        reporter=dict(type='FluxVLAEvaluationReporter'),
        evaluator=dict(type='SuccessRateEvaluator'),
        base_seed=0,
        episodes_per_task=50,
        episodes_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
        max_episode_steps=2000,
        execute_horizon=_EXECUTE_HORIZON,
        stop_on_success=True,
    ),
    server=dict(
        runner=dict(type='FluxVLAServingRunner'),
        evaluation_reporting=dict(enabled=True),
        denormalize_context=dict(task_suite_name=_STATISTIC_NAME),
    ),
)

seed = 42
