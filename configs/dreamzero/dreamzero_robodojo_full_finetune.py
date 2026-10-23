# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DreamZero full fine-tuning on RoboDojo ARX-X5 (all tasks) with Themis eval.
The RoboDojo dataset and simulator must be installed separately.

Keep the LIBERO baseline's Wan 14B architecture, frozen T5/CLIP/VAE encoders,
checkpoint mapping, mean/std normalization, and no-cache inference. Adapt to
absolute 14D joint actions padded to 32 and 14D states padded to 64.

Three 128x128 views (head, left wrist, right wrist) are tiled horizontally:
128x384 pixels -> 16x48 VAE latents -> 8x24 DiT patches (frame_seqlen=192).
Nine video frames at stride two cover one 16-action block: three latent frames
are one conditioning frame plus two predicted frames. Execute ten actions per
replan from a single current observation; no persistent KV history is assumed.
This is an initial fine-tuning config, not a reproduced benchmark result.
"""

_ckpt_root = './checkpoints'
_tokenizer = _ckpt_root + '/Wan2.1-I2V-14B-480P/google/umt5-xxl'

_frame_window_size = 9
_frame_sample_stride = 2
_action_horizon = 16
_execute_horizon = 10
_image_size = 128
_num_views = 3
_statistic_name = 'robodojo_arx_x5'

_DATA_ROOTS = ['./datasets/RoboDojo_lerobot_v21_video']

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
    type='DreamZeroVLA',
    num_views=_num_views,
    frame_window_size=_frame_window_size,
    pretrained_name_or_path=  # noqa: E251
    _ckpt_root + '/DreamZero-AgiBot',
    use_cache=False,
    vlm_backbone=dict(
        type='Wan21Backbone',
        text_encoder_path=None,
        image_encoder_path=None,
        vae_path=None,
        tiled=False,
    ),
    vla_head=dict(
        type='DreamZeroHead',
        action_dim=14,
        max_action_dim=32,
        action_horizon=_action_horizon,
        max_state_dim=64,
        num_frames=_frame_window_size,
        num_frame_per_block=2,
        num_action_per_block=_action_horizon,
        num_state_per_block=1,
        frame_seqlen=192,
        hidden_size=1024,
        input_embedding_dim=1536,
        dit_dim=5120,
        dit_ffn_dim=13824,
        dit_num_heads=40,
        dit_num_layers=40,
        dit_freq_dim=256,
        dit_in_dim=36,
        dit_out_dim=16,
        max_num_embodiments=32,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_inference_steps=16,
        use_gradient_checkpointing=True,
        cfg_scale=1.0,
        max_chunk_size=-1,
    ),
    name_mapping={
        'vla_head.model': 'action_head.model',
        'vlm_backbone.text_encoder': 'action_head.text_encoder',
        'vlm_backbone.image_encoder': 'action_head.image_encoder',
        'vlm_backbone.vae': 'action_head.vae',
    },
)

train_dataloader = dict(
    per_device_batch_size=2,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        auto_compute_statistics=dict(profile='absolute'),
        shuffle=True,
        reshuffle_each_epoch=True,
        seed=42,
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_statistic_name,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=_DATA_ROOTS,
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    parquet_keys=[
                        'observation.state',
                        'timestamp',
                        'actions',
                        'info',
                        'stats',
                        'action_masks',
                    ],
                    video_keys=[
                        'observation.images.cam_high',
                        'observation.images.cam_left_wrist',
                        'observation.images.cam_right_wrist',
                    ],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    },
                    embodiment_id=0,
                    video_backend='pyav',
                ),
                dict(type='ParquetPrompter', use_conversation=False),
                dict(
                    type='ProcessPrompts',
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=_tokenizer,
                    ),
                    max_len=512,
                ),
                dict(
                    type='ResizeImages',
                    height=_image_size,
                    width=_image_size,
                ),
                dict(type='SimpleNormalizeImages', output_type='torch'),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=32,
                    state_dim=64,
                    state_key='proprio',
                    action_key='action',
                    norm_type='mean_std',
                    output_dtype='float32',
                ),
                dict(
                    type='PrepareVideo',
                    num_views=_num_views,
                    frame_window_size=_frame_window_size,
                    tile_direction='horizontal',
                    combine_view_masks=True,
                ),
            ],
            action_window_size=_action_horizon,
            action_key='action',
            use_delta=False,
            statistic_name=_statistic_name,
            window_start_idx=0,
            frame_window_size=_frame_window_size,
            frame_sample_stride=_frame_sample_stride,
        ),
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=12,
    optimizer=dict(lr=1e-5, type='AdamW', weight_decay=1e-5),
    max_grad_norm=1.0,
    save_epoch_interval=1,
    max_keep_ckpts=3,
    tokenizer=dict(type='PretrainedTokenizer', model_path=_tokenizer),
    collator=dict(
        type='DictCollator',
        keys=[
            'states',
            'images',
            'img_masks',
            'actions',
            'action_masks',
            'embodiment_ids',
            'frame_masks',
            'lang_tokens',
            'lang_masks',
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats', 'timestamp'],
    ),
    sampler=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=1,
    ),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        warmup_ratio=0.05,
    ),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='full-shard',
    change_key_name=False,
)

# RoboDojo evaluation is launched through Themis, not scripts/eval.py.
eval = dict(
    report_kind='robodojo',
    task_suite_name='robodojo',
    model_family='dreamzero',
    num_trials_per_task=50,
    num_trials_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
    dataset=dict(
        type='RoboDojoEvalDataset',
        unnorm_key=_statistic_name,
        transforms=[
            dict(
                type='ProcessEvalInputs',
                img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
                embodiment_id=0,
            ),
            dict(type='ParquetPrompter', use_conversation=False),
            dict(
                type='ProcessPrompts',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=_tokenizer,
                ),
                max_len=512,
            ),
            dict(
                type='ResizeImages',
                key='pixel_values',
                height=_image_size,
                width=_image_size,
            ),
            dict(
                type='SimpleNormalizeImages',
                key='pixel_values',
                output_type='torch',
            ),
            dict(
                type='NormalizeStatesAndActions',
                action_dim=32,
                state_dim=64,
                state_key='proprio',
                action_key='action',
                norm_type='mean_std',
                output_dtype='float32',
                statistics_key='norm_stats',
            ),
            dict(
                type='PrepareVideo',
                num_views=_num_views,
                frame_window_size=1,
                tile_direction='horizontal',
                combine_view_masks=True,
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='mean_std',
        action_dim=14,
        statistic_name=_statistic_name,
    ),
)

themis = dict(
    transport=dict(
        image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        state_keys=['states'],
        unnorm_key=_statistic_name,
    ),
    evaluation=dict(
        environment=dict(
            type='RoboDojoEnvironment',
            task_name='all',
            env_cfg_type='arx_x5',
            robodojo_root='/root/projects/RoboDojo',
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
        execute_horizon=_execute_horizon,
        stop_on_success=True,
    ),
    server=dict(
        runner=dict(type='FluxVLAServingRunner'),
        evaluation_reporting=dict(enabled=True),
        denormalize_context=dict(task_suite_name=_statistic_name),
    ),
)

seed = 42
