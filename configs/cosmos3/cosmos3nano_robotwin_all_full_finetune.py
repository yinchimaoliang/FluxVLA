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
#
"""Cosmos3-Nano on all 50 RoboTwin tasks, clean/randomized mixed 1:1.

Recorded absolute 14D joint actions are padded to the native 64D width.
Predict 16 actions; execute 10 before replanning. WAM training pairs these
with 17 video frames at the dataset's 15 FPS. Only complete windows are used
because the Cosmos3 loss does not consume dataset temporal-padding masks.
Two balanced epochs are an initial budget, not a measured optimum.
Evaluate random scenes with eval.task_suite_name=random.
"""

from copy import deepcopy

_ckpt_root = './checkpoints'
_cosmos3_nano_ckpt = _ckpt_root + '/Cosmos3-Nano'
_cosmos3_nano_transformer = _cosmos3_nano_ckpt + '/transformer'
_cosmos3_nano_vision_encoder = _cosmos3_nano_ckpt + '/vision_encoder'
_cosmos3_nano_tokenizer = dict(
    type='PretrainedTokenizer',
    model_path=_cosmos3_nano_ckpt + '/text_tokenizer',
    model_max_length=4096,
    padding_side='right',
    trust_remote_code=True,
)
_action_dim = 14
_max_action_dim = 64
_max_state_dim = 64
_action_horizon = 16
_frame_window_size = _action_horizon + 1
# Match the existing Nano policy recipe: condition on images, not state tokens.
_prepend_state_to_action = False
# Dedicated RoboTwin domain; distinct from LIBERO (5) and RoboCasa (24).
_embodiment_id = 25
_conditioning_fps = 15.0
_image_height = 256
_image_width = 256
_video_height = 384
_video_width = 256
_cfg_dropout_rate = 0.1
_base_lr = 5e-5
_action_lr = _base_lr * 5.0
_per_device_batch_size = 8
_grad_accumulation_steps = 4
_statistic_name = 'robotwin_all'
_data_roots = [
    'datasets/robotwin_clean_lerobotv2.1',
    'datasets/robotwin_randomized_lerobotv2.1',
]
_vision_vae = dict(
    type='Cosmos3Wan22VAE',
    pretrained_name_or_path=(_ckpt_root + '/Wan2.2-TI2V-5B/Wan2.2_VAE.pth'),
    encode_exact_durations=[_frame_window_size],
)
_action_prompt_metadata = dict(
    append_viewpoint=False,
    frame_window_size=_frame_window_size,
    conditioning_fps=_conditioning_fps,
    video_height=_video_height,
    video_width=_video_width,
)
_normalize = dict(
    type='NormalizeStatesAndActions',
    action_dim=_max_action_dim,
    state_dim=_max_state_dim,
    state_key='proprio',
    action_key='action',
    norm_type='quantile',
    output_dtype='float32',
)
# Identical camera resize, value range and layout during training and eval.
_image_transforms = [
    dict(type='ResizeImages', height=_image_height, width=_image_width),
    dict(type='SimpleNormalizeImages'),
]
_video_layout = dict(
    type='PrepareVideo',
    num_views=3,
    tile_direction='top_bottom_pair',
    top_view=0,
    bottom_views=(1, 2),
    bottom_height_ratio=0.5,
)
_task_list = [
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

_cosmos3_nano_special_tokens = dict(
    eos_token_id=151645,
    start_of_generation=151652,
    end_of_generation=151653,
)

_cosmos3_nano_vlm_config = dict(
    model_type='qwen3_vl',
    vocab_size=151936,
    tie_word_embeddings=False,
    image_token_id=151655,
    video_token_id=151656,
    vision_start_token_id=151652,
    vision_end_token_id=151653,
    text_config=dict(
        model_type='qwen3_vl_text',
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act='silu',
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        rope_theta=5000000,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=151643,
        eos_token_id=151645,
        pad_token_id=0,
        tie_word_embeddings=False,
        rope_scaling=dict(
            rope_type='default',
            mrope_interleaved=True,
            mrope_section=[24, 20, 20],
        ),
        layer_types=['full_attention'] * 36,
    ),
    vision_config=dict(
        model_type='qwen3_vl',
        hidden_size=1152,
        hidden_act='gelu_pytorch_tanh',
        intermediate_size=4304,
        depth=27,
        num_heads=16,
        in_channels=3,
        initializer_range=0.02,
        out_hidden_size=4096,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[8, 16, 24],
    ),
)

_cosmos3_nano_name_mapping = {
    # Official Cosmos3 checkpoint keeps this parameter name unchanged.
    'action_modality_embed': 'action_modality_embed',
    'vlm_backbone.model.language_model.embed_tokens.weight':
    'embed_tokens.weight',
    'vlm_backbone.lm_head.weight': 'lm_head.weight',
    'vlm_backbone.model.language_model.norm.weight': 'norm.weight',
    'vlm_backbone.model.language_model.norm_moe_gen.weight':
    'norm_moe_gen.weight',
    'vlm_backbone.model.language_model.layers.': 'layers.',
    'vision_in_proj.projector.': 'proj_in.',
    'vision_out_proj.projector.': 'proj_out.',
    'time_embedder.mlp.0.': 'time_embedder.linear_1.',
    'time_embedder.mlp.2.': 'time_embedder.linear_2.',
    'action_in_proj.': 'action_proj_in.',
    'action_out_proj.': 'action_proj_out.',
    '.self_attn.q_proj.': '.self_attn.to_q.',
    '.self_attn.k_proj.': '.self_attn.to_k.',
    '.self_attn.v_proj.': '.self_attn.to_v.',
    '.self_attn.o_proj.': '.self_attn.to_out.',
    '.self_attn.q_proj_moe_gen.': '.self_attn.add_q_proj.',
    '.self_attn.k_proj_moe_gen.': '.self_attn.add_k_proj.',
    '.self_attn.v_proj_moe_gen.': '.self_attn.add_v_proj.',
    '.self_attn.o_proj_moe_gen.': '.self_attn.to_add_out.',
    '.self_attn.q_norm.': '.self_attn.norm_q.',
    '.self_attn.k_norm.': '.self_attn.norm_k.',
    '.self_attn.q_norm_moe_gen.': '.self_attn.norm_added_q.',
    '.self_attn.k_norm_moe_gen.': '.self_attn.norm_added_k.',
    'vlm_backbone.model.visual.patch_embed.': 'patch_embed.',
    'vlm_backbone.model.visual.blocks.': 'blocks.',
    'vlm_backbone.model.visual.pos_embed.': 'pos_embed.',
    'vlm_backbone.model.visual.merger.': 'merger.',
    'vlm_backbone.model.visual.deepstack_merger_list.':
    'deepstack_merger_list.',
}

model = dict(
    type='Cosmos3FlowMatching',
    # Packed FlashAttention also requires BF16 embeddings during plain eval;
    # autocast alone leaves the embedding/rotary path in FP32.
    torch_dtype='bfloat16',
    vlm_backbone=dict(
        type='Cosmos3MoTBackbone',
        vlm_config=_cosmos3_nano_vlm_config,
        include_visual=False,
        vision_encoder_path=_cosmos3_nano_vision_encoder,
        skip_init_weights=True,
    ),
    vision_latent_dim=48,
    latent_patch_size=2,
    max_action_dim=64,
    num_embodiment_domains=32,
    vision_in_proj=dict(
        type='LinearProjector',
        in_dim=48 * 2 * 2,
        out_dim=4096,
    ),
    vision_out_proj=dict(
        type='LinearProjector',
        in_dim=4096,
        out_dim=48 * 2 * 2,
    ),
    action_in_proj=dict(
        type='DomainAwareLinear',
        input_size=64,
        output_size=4096,
        num_domains=32,
    ),
    action_out_proj=dict(
        type='DomainAwareLinear',
        input_size=4096,
        output_size=64,
        num_domains=32,
    ),
    rectified_flow_training_config=dict(
        shift={
            '256': 3,
            '480': 5,
            '720': 10,
        },
        use_dynamic_shift=False,
        train_time_image_distribution='logitnormal',
        train_time_video_distribution='waver',
        train_time_action_distribution='logitnormal',
        train_time_weight='uniform',
        vision_loss_weight=10.0,  # Official LIBERO recipe: loss_scale=10.0
        independent_action_schedule=False,
        shift_action=None,
        use_high_sigma_strategy=False,
        high_sigma_ratio=0.05,
        high_sigma_timesteps_min=995,
        high_sigma_timesteps_max=1000,
        use_high_sigma_strategy_action=False,
        use_discrete_rf=False,
        normalize_loss_by_active=False,
        action_loss_weight=10.0,
    ),
    rectified_flow_inference_config=dict(
        num_train_timesteps=1000,
        scheduler_type='unipc',
        num_steps=30,
        shift=10.0,
        use_dynamic_shifting=False,
        use_karras_sigmas=False,
    ),
    timestep_scale=0.001,
    packed_attention_backend='flash2',
    position_embedding_type='unified_3d_mrope',
    unified_3d_mrope_reset_spatial_ids=True,
    unified_3d_mrope_temporal_modality_margin=15000,
    enable_fps_modulation=True,
    base_fps=24.0,
    special_tokens=_cosmos3_nano_special_tokens,
    pretrained_name_or_path=_cosmos3_nano_transformer,
    name_mapping=_cosmos3_nano_name_mapping,
    vision_vae=_vision_vae,
    ori_action_dim=_action_dim,
    action_horizon=_action_horizon,
    freeze_vlm_backbone=False,
    freeze_non_moe_vlm_backbone=True,
    enable_vision_loss=True,
)

inference_model = deepcopy(model)
# Eval builds the VAE without the external Wan2.2 file; the finetuned
# checkpoint already carries the frozen VAE weights.
inference_model['vision_vae']['pretrained_name_or_path'] = None

_transforms = [
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
        video_backend='pyav',
        name_mappings={
            'observation.state': ['states'],
            'actions': ['actions'],
        },
        embodiment_id=_embodiment_id,
    ),
    dict(
        type='ProcessCosmos3Prompt',
        tokenizer=_cosmos3_nano_tokenizer,
        max_len=512,
        cfg_dropout_rate=_cfg_dropout_rate,
        format_prompt_as_json=True,
        action_metadata=_action_prompt_metadata,
    ),
    *deepcopy(_image_transforms),
    deepcopy(_normalize),
    dict(
        type='BuildCosmos3Sequence',
        raw_action_dim=_action_dim,
        mode='wam',
        frame_window_size=_frame_window_size,
        prepend_state_to_action=_prepend_state_to_action,
        conditioning_fps=_conditioning_fps,
    ),
    dict(_video_layout, frame_window_size=_frame_window_size),
]

train_dataloader = dict(
    per_device_batch_size=_per_device_batch_size,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedBalancedRepeatingDataset',
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
        auto_compute_statistics=dict(profile='absolute'),
        # Keep roots separate: rejecting an incomplete video window must
        # resample inside that source, preserving the clean/random weights.
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=data_root,
                transforms=_transforms,
                action_window_size=_action_horizon,
                action_key='action',
                use_delta=False,
                statistic_name=_statistic_name,
                window_start_idx=0,
                frame_window_size=_frame_window_size,
                frame_sample_stride=1,
                require_full_window=True,
            ) for data_root in _data_roots
        ],
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=2,
    max_steps=None,
    save_epoch_interval=1,
    save_iter_interval=10000000,
    max_keep_ckpts=3,
    optimizer=dict(
        type='AdamW',
        lr=_base_lr,
        weight_decay=0.05,
        betas=(0.9, 0.99),
        eps=1e-8,
        fused=True,
        exclude_1d_from_weight_decay=False,
        paramwise_learning_rate={
            'action_in_proj.': _action_lr,
            'action_out_proj.': _action_lr,
            'action_modality_embed': _action_lr,
        },
    ),
    max_grad_norm=1.0,
    tokenizer=_cosmos3_nano_tokenizer,
    collator=dict(
        type='DictCollator',
        keys=[
            'images',
            'states',
            'actions',
            'action_masks',
            'img_masks',
            'frame_masks',
            'embodiment_ids',
            'raw_action_dim',
            'conditioning_fps',
            'action_fps',
        ],
        meta_keys=[
            'text_token_ids',
            'sequence_plan',
            'task_description',
            'stats',
            'info',
            'timestamp',
            'viewpoint',
        ],
    ),
    sampler=None,
    grad_accumulation_steps=_grad_accumulation_steps,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=_grad_accumulation_steps,
        window_size=100,
    ),
    lr_scheduler=dict(
        # The cosine policy resolves warmup against the epoch-derived steps.
        type='linear-warmup+cosine-decay',
        warmup_ratio=0.05,
    ),
    seed=42,
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='full-shard',
    change_key_name=False,
)

eval = dict(
    type='RobotwinEvalRunner',
    model_family='cosmos3',
    task_list=_task_list,
    task_suite_name='clean',
    instruction_type='unseen',
    eval_chunk_size=10,
    num_trials_per_task=100,
    seed=7,
    mixed_precision_dtype='bf16',
    unnorm_key=_statistic_name,
    dataset=dict(
        type='PrivateInferenceDataset',
        inject_model_path=False,
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        embodiment_id=_embodiment_id,
        extra_tensor_keys=['conditioning_fps', 'prepend_state_to_action'],
        transforms=[
            dict(
                type='SetCosmos3ActionMetadata',
                conditioning_fps=_conditioning_fps,
                prepend_state_to_action=_prepend_state_to_action,
            ),
            deepcopy(_normalize),
            dict(
                type='ProcessCosmos3Prompt',
                tokenizer=_cosmos3_nano_tokenizer,
                max_len=512,
                cfg_dropout_rate=0.0,
                format_prompt_as_json=True,
                action_metadata=_action_prompt_metadata,
                output_key='lang_tokens',
                output_attention_mask_key='lang_masks',
            ),
            *deepcopy(_image_transforms),
            dict(_video_layout, frame_window_size=1),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='quantile',
        action_dim=_action_dim,
        statistic_name=_statistic_name,
    ),
)
