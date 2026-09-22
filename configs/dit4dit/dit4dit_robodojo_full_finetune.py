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
"""DiT4DiT RoboDojo full-data baseline with FluxThemis evaluation.

Adapt the RoboCasa joint-position recipe, not a released RoboDojo policy:
Cosmos base initialization, 16 absolute joint-action steps, and nine video
frames at stride two. ARX-X5 has 14 joint/gripper values (not EEF poses).
Actions are padded to 16; interleaved sin/cos states are padded from 28 to 32.
All three cameras are tiled horizontally in high/left-wrist/right-wrist order.

Unlike source evaluation's cv2 resize, online observations use the same
float RGB / 255 + torch resize as training. Cosmos owns VAE normalization.
The shorter initial budget, full sharding and Cosmos gradient checkpointing
are intentional resource-saving changes, not a reproduced source score.
"""

import os

_cosmos_base_model = './checkpoints/Cosmos-Predict2.5-2B'
_cosmos_tokenizer = dict(
    type='PretrainedTokenizer',
    model_path=_cosmos_base_model + '/tokenizer',
    model_max_length=512,
)
_ROBODOJO_DATA_ROOT = os.environ.get('ROBODOJO_DATA_ROOT',
                                     './datasets/RoboDojo_lerobot_v21_video')
_statistic_name = 'robodojo_arx_x5'
_camera_keys = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
_action_dim = 16
_ori_action_dim = 14
_state_dim = 32
_action_horizon = 16
_frame_window_size = 9
_frame_sample_stride = 2
_image_size = 224
_grad_accumulation_steps = 4

# 30 ordinary tasks x 50 + 12 generalization pairs x (25 + 25) = 2,100.
_ROBODOJO_GENERALIZATION_BASE_TASKS = ('stack_bowls', 'push_T',
                                       'pack_objects_into_box', 'fold_clothes',
                                       'hang_mugs', 'sweep_blocks',
                                       'pour_liquid_into_cup', 'make_toast',
                                       'arrange_largest_number',
                                       'sort_nesting_dolls_by_size',
                                       'store_laptop_and_headphones',
                                       'stack_blocks')
_ROBODOJO_EPISODE_OVERRIDES = {
    task_name: 25
    for base_task in _ROBODOJO_GENERALIZATION_BASE_TASKS
    for task_name in (base_task, f'{base_task}_random')
}

seed = 42

model = dict(
    type='DiT4DiTVLA',
    # No LIBERO/GR1 policy checkpoint: its action/state projections differ.
    repeated_diffusion_steps=4,
    freeze_vlm_backbone=False,
    vlm_backbone=dict(
        type='Cosmos25Backbone',
        base_model=_cosmos_base_model,
        revision='diffusers/base/post-trained',
        torch_dtype='bf16',
        local_files_only=True,
        load_pretrained_weights=True,
        extract_layer=17,
        trainable=True,
        frozen_submodules=['text_encoder', 'vae'],
        split_future_frames=True,
        num_frames_out=_frame_window_size,
        fixed_seed=None,
        num_inference_steps=1,
        conditional_frame_timestep=0.0001,
        future_loss_type='flow_matching',
        detach_hidden_states=True,
        flow_matching_time_distribution='uniform',
        flow_matching_high_sigma_ratio=None,
        flow_matching_high_sigma_min=None,
        fsdp_min_num_params=0,
    ),
    vla_head=dict(
        type='DiT4DiTActionHead',
        action_model_type='DiT-B',
        hidden_size=2560,
        add_pos_embed=True,
        max_seq_len=1024,
        action_dim=_action_dim,
        ori_action_dim=_ori_action_dim,
        state_dim=_state_dim,
        future_action_window_size=_action_horizon - 1,
        action_horizon=_action_horizon,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        num_inference_timesteps=4,
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
    ),
)

# Evaluation restores the complete FluxVLA checkpoint, including the frozen
# modules; do not load base weights again or remap source policy names.
inference_model = dict(
    model,
    init_empty_weights=True,
    vlm_backbone=dict(model['vlm_backbone'], load_pretrained_weights=False),
)

_resize = dict(
    type='ResizeImages',
    height=_image_size,
    width=_image_size,
    backend='torch',
    scale_divisor=255.0,
    output_layout='flattened_chw',
)
_state_encoding = dict(
    type='SinCosKeys',
    keys=['states'],
    target_dims={'states': _state_dim},
    interleave=True,
    expand_axis=0,
    backend='torch',
)
_prompt = dict(
    type='ProcessCosmos25Prompt',
    tokenizer=_cosmos_tokenizer,
    input_key='task_description',
    remove_input_key=True,
)
_video = dict(
    type='PrepareVideo',
    num_views=len(_camera_keys),
    tile_direction='horizontal',
    combine_view_masks=True,
)

train_dataloader = dict(
    # On 8 GPUs: 2 x 8 x 4 = 64 real samples/optimizer step. The four
    # independent action diffusion draws do not multiply the dataset batch.
    per_device_batch_size=2,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        shuffle=True,
        reshuffle_each_epoch=True,
        seed=seed,
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_statistic_name,
        # Merge all episode metadata min/max bounds at runtime and save them
        # beside the checkpoint. Never reuse N1.7's relative-action stats.
        datasets=dict(
            type='ParquetDataset',
            data_root_path=[_ROBODOJO_DATA_ROOT],
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
                        f'observation.images.{key}' for key in _camera_keys
                    ],
                    video_backend='torchcodec',
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    },
                ),
                _prompt,
                _resize,
                dict(_video, frame_window_size=_frame_window_size),
                _state_encoding,
                dict(
                    type='NormalizeStatesAndActions',
                    state_key='proprio',
                    action_key='action',
                    action_dim=_action_dim,
                    state_dim=_state_dim,
                    normalize_states=False,
                    state_norm_type='none',
                    action_norm_type='min_max',
                    norm_type='min_max',
                    # Normalize both grippers too, without LIBERO's sign
                    # inversion, thresholding or seven-dimensional mask.
                    clip_norm=False,
                    normalization_epsilon=0.0,
                    zero_constant_min_max_dims=True,
                    preserve_input_dtype=True,
                    valid_action_dim=_ori_action_dim,
                    mark_all_action_steps_valid=True,
                    output_dtype='float16',
                ),
            ],
            action_window_size=_action_horizon,
            action_key='action',
            use_delta=False,
            statistic_name=_statistic_name,
            window_start_idx=0,
            supervise_terminal_padding=True,
            frame_window_size=_frame_window_size,
            frame_sample_stride=_frame_sample_stride,
            action_dtype=None,
        ),
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=None,
    # About 2.1 passes over 1,859,602 frames at global batch 64. Evaluate
    # intermediate checkpoints before deciding whether to extend the budget.
    max_steps=60000,
    grad_accumulation_steps=_grad_accumulation_steps,
    # Use the existing runner's no_sync accumulation. Frozen Cosmos text/VAE
    # modules remain BF16 replicas; trainable parameters are fully sharded.
    # Unlike FastWAM's much larger head, this recipe retains accumulation to
    # preserve its global batch of 64 on 8 GPUs.
    sharding_strategy='full-shard',
    pre_fsdp_param_dtype='fp32',
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    reduce_in_full_precision=False,
    optimizer=dict(
        type='AdamW',
        lr=3e-5,
        weight_decay=1e-8,
        weight_decay_all_params=True,
        eps=1e-8,
        betas=(0.9, 0.95),
        fused=True,
        paramwise_learning_rate={
            'vlm_backbone.transformer': 1e-5,
            'vla_head': 1e-4,
        },
    ),
    max_grad_norm=1.0,
    save_iter_interval=10000,
    max_keep_ckpts=3,
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
    tokenizer=_cosmos_tokenizer,
    sampler=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=_grad_accumulation_steps,
        window_size=1,
    ),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        warmup_steps=2000,
        min_lr=5e-7,
    ),
    change_key_name=False,
)

eval = dict(
    report_kind='robodojo',
    task_suite_name='robodojo',
    model_family='dit4dit',
    num_trials_per_task=50,
    num_trials_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
    dataset=dict(
        type='RoboDojoEvalDataset',
        unnorm_key=_statistic_name,
        state_dtype='fp32',
        transforms=[
            dict(type='ProcessEvalInputs', img_keys=_camera_keys),
            _prompt,
            dict(_resize, key='pixel_values'),
            dict(_video, frame_window_size=1),
            _state_encoding,
        ],
    ),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='min_max',
        action_dim=_ori_action_dim,
        statistic_name=_statistic_name,
        clip_normalized_action=True,
    ),
)

themis = dict(
    transport=dict(
        host='127.0.0.1',
        port=5555,
        timeout_s=30.0,
        image_keys=_camera_keys,
        state_keys=['states'],
        unnorm_key=_statistic_name,
        image_encoding='rgb8',
        report_service_name='/fluxvla/report_evaluation',
    ),
    runner=dict(
        type='EvalRunner',
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
        model_client=dict(type='FluxVLAZMQModelClient'),
        evaluator=dict(type='SuccessRateEvaluator'),
        seed=0,
        episodes_per_task=50,
        episodes_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
        max_episode_steps=2000,
        # Match RoboCasa's 12 executed actions out of each 16-step chunk.
        execute_horizon=12,
        stop_on_success=True,
        parallel_workers=1,
        simulator_gpu_ids=None,
        work_dir='work_dirs/dit4dit_robodojo_eval',
    ),
    ros_server=dict(
        dataset_section='eval',
        evaluation_reporting=dict(
            result_output_dir='work_dirs/dit4dit_robodojo_eval'),
        device='cuda:0',
        workers=dict(
            startup_timeout_s=900.0,
            request_timeout_s=120.0,
            lease_timeout_s=900.0,
        ),
        mixed_precision_dtype='bf16',
        enable_mixed_precision=True,
        model_outputs_environment_actions=False,
        forward_seed=False,
        denormalize_context=dict(task_suite_name=_statistic_name),
    ),
)
