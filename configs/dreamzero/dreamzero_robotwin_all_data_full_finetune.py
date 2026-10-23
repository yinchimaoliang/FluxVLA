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
"""DreamZero full fine-tuning on all 50 RoboTwin tasks, clean/randomized 1:1.
Evaluate both clean and random scenes with the local RoboTwin runner.

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
_statistic_name = 'robotwin_all'

_DATA_ROOTS = [
    './datasets/robotwin_clean_lerobotv2.1',
    './datasets/robotwin_randomized_lerobotv2.1',
]

_TASK_LIST = [
    'adjust_bottle',
    'beat_block_hammer',
    'blocks_ranking_rgb',
    'blocks_ranking_size',
    'click_alarmclock',
    'click_bell',
    'dump_bin_bigbin',
    'grab_roller',
    'handover_block',
    'handover_mic',
    'hanging_mug',
    'lift_pot',
    'move_can_pot',
    'move_pillbottle_pad',
    'move_playingcard_away',
    'move_stapler_pad',
    'open_laptop',
    'open_microwave',
    'pick_diverse_bottles',
    'pick_dual_bottles',
    'place_a2b_left',
    'place_a2b_right',
    'place_bread_basket',
    'place_bread_skillet',
    'place_burger_fries',
    'place_can_basket',
    'place_cans_plasticbox',
    'place_container_plate',
    'place_dual_shoes',
    'place_empty_cup',
    'place_fan',
    'place_mouse_pad',
    'place_object_basket',
    'place_object_scale',
    'place_object_stand',
    'place_phone_stand',
    'place_shoe',
    'press_stapler',
    'put_bottles_dustbin',
    'put_object_cabinet',
    'rotate_qrcode',
    'scan_object',
    'shake_bottle',
    'shake_bottle_horizontally',
    'stack_blocks_three',
    'stack_blocks_two',
    'stack_bowls_three',
    'stack_bowls_two',
    'stamp_seal',
    'turn_switch',
]

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
        type='DistributedBalancedRepeatingDataset',
        auto_compute_statistics=dict(profile='absolute'),
        sampling_weights=[1.0, 1.0],
        sampling_unit='episode',
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

eval = dict(
    type='RobotwinEvalRunner',
    task_suite_name=['clean', 'random'],
    task_list=_TASK_LIST,
    model_family='dreamzero',
    eval_chunk_size=_execute_horizon,
    instruction_type='unseen',
    num_trials_per_task=100,
    seed=7,
    unnorm_key=_statistic_name,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    # Autocast alone leaves the 23B-parameter model in FP32 (>80 GiB).
    enable_model_dtype_cast=True,
    dataset=dict(
        type='PrivateInferenceDataset',
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        embodiment_id=0,
        statistic_name=_statistic_name,
        inject_model_path=False,
        transforms=[
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
                key='images',
                height=_image_size,
                width=_image_size,
            ),
            dict(
                type='SimpleNormalizeImages',
                key='images',
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
                statistics_key='stats',
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
