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
"""Full-data DreamZero fine-tuning on the RoboCasa GR1 tabletop tasks.

The dataset layout, task list, and closed-loop evaluation setup follow
``configs/pi05/pi05_paligemma_robocasa_full_data_full_finetune.py``. The
DreamZero-specific video, tokenizer, and model settings follow the existing
LIBERO recipe. RoboCasa vector ordering, state encoding, min-max action
normalization, augmentation, and classifier-free guidance follow the released
DreamZero data contract.

Example for two 8-GPU nodes sharing MASTER_ADDR and MASTER_PORT:
    torchrun --nnodes=2 --nproc_per_node=8 \
        --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} scripts/train.py \
        --config \
        configs/dreamzero/dreamzero_robocasa_full_data_full_finetune.py \
        --work-dir \
        work_dirs/dreamzero_robocasa_full_data_full_finetune
"""

_CKPT_ROOT = './checkpoints'
_TOKENIZER = _CKPT_ROOT + '/Wan2.1-I2V-14B-480P/google/umt5-xxl'

# The checkpoint supports four causal chunks (33 raw frames). Each training
# sample uses one complete chunk: nine frames at offsets [0, 6, ..., 48]
# paired with 48 consecutive actions.
_MODEL_NUM_FRAMES = 33
_TRAIN_FRAME_WINDOW_SIZE = 9
_FRAME_SAMPLE_STRIDE = 6
_ACTION_HORIZON = 48
_NUM_VIEWS = 1
_IMAGE_SIZE = 256
_FRAME_SEQUENCE_LENGTH = (_IMAGE_SIZE // 16)**2
_PROMPT_TEMPLATE = 'A single view video shows that a human {task}'
_NEGATIVE_PROMPT = (
    'Vibrant colors, overexposed, static, blurry details, text, subtitles, '
    'style, artwork, painting, image, still, grayscale, dull, worst quality, '
    'low quality, JPEG artifacts, ugly, mutilated, extra fingers, bad hands, '
    'bad face, deformed, disfigured, mutated limbs, fused fingers, stagnant '
    'image, cluttered background, three legs, many people in the background, '
    'walking backwards.')

model = dict(
    type='DreamZeroVLA',
    num_views=_NUM_VIEWS,
    frame_window_size=_MODEL_NUM_FRAMES,
    pretrained_name_or_path=  # noqa: E251
    _CKPT_ROOT + '/DreamZero-AgiBot',
    # Keep DreamZero's causal inference implementation enabled. RoboCasa
    # supplies one current frame per request, which resets the cache exactly
    # as in the released simulator evaluation path.
    use_cache=True,
    # Upstream cached inference pre-fills the KV cache with the first latent
    # produced by the image-conditioning encoder, not a separately encoded
    # one-frame video. Keep this opt-in so existing LIBERO configs are
    # unchanged.
    use_image_condition_for_cache_prefill=True,
    vlm_backbone=dict(
        type='Wan21Backbone',
        text_encoder_path=None,
        image_encoder_path=None,
        vae_path=None,
        tiled=False,
    ),
    vla_head=dict(
        type='DreamZeroHead',
        # RoboCasa GR1 uses 29 joint-position controls. DreamZero pads them to
        # its checkpoint-compatible internal action width of 32.
        action_dim=29,
        max_action_dim=32,
        action_horizon=_ACTION_HORIZON,
        max_state_dim=64,
        num_frames=_MODEL_NUM_FRAMES,
        num_frame_per_block=2,
        num_action_per_block=_ACTION_HORIZON,
        num_state_per_block=1,
        # One 256x256 view -> 32x32 VAE latent -> 16x16 DiT patch grid.
        # This matches the released DreamZero ``gr1_unified`` transform.
        frame_seqlen=_FRAME_SEQUENCE_LENGTH,
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
        # The released DreamZero inference implementation uses 16 denoising
        # steps, irrespective of the stale value in its checkpoint config.
        num_inference_steps=16,
        # Match released cached inference: video and action noise are sampled
        # from separate generators initialized with this fixed seed.
        inference_seed=1140,
        # The released implementation keeps all 16 scheduler updates but runs
        # the DiT on this fixed eight-step subset, reusing the latest velocity
        # prediction on the remaining updates.
        num_dit_compute_steps=8,
        use_gradient_checkpointing=True,
        cfg_scale=5.0,
        validate_action_range=True,
        # Match the local-attention window in the released checkpoint. A
        # training sample below still contains one complete action block.
        max_chunk_size=4,
    ),
    name_mapping={
        'vla_head.model': 'action_head.model',
        'vlm_backbone.text_encoder': 'action_head.text_encoder',
        'vlm_backbone.image_encoder': 'action_head.image_encoder',
        'vlm_backbone.vae': 'action_head.vae',
    },
)

_ROBOCASA_STATISTIC_NAME = 'robocasa_gr1_24tasks_30ep'
_ROBOCASA_DATA_ROOT = './datasets/robocasa_lerobot_V2.1'
_ROBOCASA_TASK_PREFIX = 'gr1_unified'
_ROBOCASA_ENV_SUFFIX = '_GR1ArmsAndWaistFourierHands_Env'

_ROBOCASA_TASKS = [
    'PnPBottleToCabinetClose',
    'PnPCanToDrawerClose',
    'PnPCupToDrawerClose',
    'PnPMilkToMicrowaveClose',
    'PnPPotatoToMicrowaveClose',
    'PnPWineToCabinetClose',
    'PosttrainPnPNovelFromCuttingboardToBasketSplitA',
    'PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA',
    'PosttrainPnPNovelFromCuttingboardToPanSplitA',
    'PosttrainPnPNovelFromCuttingboardToPotSplitA',
    'PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA',
    'PosttrainPnPNovelFromPlacematToBasketSplitA',
    'PosttrainPnPNovelFromPlacematToBowlSplitA',
    'PosttrainPnPNovelFromPlacematToPlateSplitA',
    'PosttrainPnPNovelFromPlacematToTieredshelfSplitA',
    'PosttrainPnPNovelFromPlateToBowlSplitA',
    'PosttrainPnPNovelFromPlateToCardboardboxSplitA',
    'PosttrainPnPNovelFromPlateToPanSplitA',
    'PosttrainPnPNovelFromPlateToPlateSplitA',
    'PosttrainPnPNovelFromTrayToCardboardboxSplitA',
    'PosttrainPnPNovelFromTrayToPlateSplitA',
    'PosttrainPnPNovelFromTrayToPotSplitA',
    'PosttrainPnPNovelFromTrayToTieredbasketSplitA',
    'PosttrainPnPNovelFromTrayToTieredshelfSplitA',
]


def _robocasa_data_path(task_name):
    return f'{_ROBOCASA_DATA_ROOT}/{task_name}'


def _robocasa_task_env(task_name):
    return f'{_ROBOCASA_TASK_PREFIX}/{task_name}{_ROBOCASA_ENV_SUFFIX}'


train_dataloader = dict(
    # Global batch is 2 x world size (128 on 64 GPUs).
    per_device_batch_size=2,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_ROBOCASA_STATISTIC_NAME,
        # Aggregate statistics from the same full 24-task dataset used below.
        # The first-30-episode statistics previously used here do not match
        # this training distribution.
        reshuffle_each_epoch=True,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=[
                _robocasa_data_path(task_name) for task_name in _ROBOCASA_TASKS
            ],
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
                    video_keys=['observation.images.ego_view'],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    },
                    embodiment_id=0,
                ),
                dict(
                    type='RobocasaGR1N15Bridge',
                    # DreamZero's released ``gr1_unified`` transform applies
                    # group-wise sin/cos encoding to the GR1 joint state.
                    apply_state_sincos=True,
                ),
                dict(
                    type='ParquetPrompter',
                    use_conversation=False,
                    lowercase_task_description=True,
                    prompt_template=_PROMPT_TEMPLATE,
                ),
                dict(
                    type='ProcessPrompts',
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=_TOKENIZER,
                    ),
                    max_len=512,
                ),
                dict(
                    type='RandomCropImages',
                    scale=0.95,
                    consistent=True,
                ),
                dict(
                    type='ResizeImages',
                    height=_IMAGE_SIZE,
                    width=_IMAGE_SIZE,
                ),
                dict(
                    type='ColorJitterImages',
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.08,
                    consistent=True,
                ),
                dict(type='SimpleNormalizeImages'),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=32,
                    state_dim=64,
                    state_key='proprio',
                    action_key='action',
                    norm_type='min_max',
                    clip_norm=True,
                    # Match DreamZero: constant min/max action dimensions are
                    # represented by zero rather than the lower endpoint.
                    zero_constant_min_max_dims=True,
                    # Sin/cos state features are already bounded and are not
                    # normalized again in the released DreamZero transform.
                    normalize_states=False,
                ),
                dict(
                    type='PrepareVideo',
                    num_views=_NUM_VIEWS,
                    frame_window_size=_TRAIN_FRAME_WINDOW_SIZE,
                ),
            ],
            # Match one complete block from the released DreamZero sampler.
            action_window_size=_ACTION_HORIZON,
            action_key='action',
            # DreamZero's released ``gr1_unified`` contract uses absolute
            # joint targets. Its relative-action key list is specific to the
            # AgiBot schema and does not match RoboCasa action keys.
            use_delta=False,
            statistic_name=_ROBOCASA_STATISTIC_NAME,
            window_start_idx=0,
            frame_window_size=_TRAIN_FRAME_WINDOW_SIZE,
            frame_sample_stride=_FRAME_SAMPLE_STRIDE,
            require_full_window=True,
        ),
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=None,
    max_steps=100000,
    grad_accumulation_steps=1,
    optimizer=dict(lr=1e-5, type='AdamW', weight_decay=1e-5),
    max_grad_norm=1.0,
    save_epoch_interval=1,
    save_iter_interval=5000,
    max_keep_ckpts=8,
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
        grad_accumulation_steps=1,
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
    type='RobocasaEvalRunner',
    benchmark='robocasa',
    task_suite_name='robocasa',
    model_family='dreamzero',
    task_list=[_robocasa_task_env(task_name) for task_name in _ROBOCASA_TASKS],
    total_tasks=24,
    # The released simulator client executes eight actions before replanning.
    eval_chunk_size=8,
    max_episode_steps=720,
    # Match the reported RoboCasa protocol: 24 tasks x 50 trials = 1200
    # episodes in total.
    num_trials_per_task=50,
    seed=7,
    unnorm_key=_ROBOCASA_STATISTIC_NAME,
    action_order='n15',
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='RobocasaEvalDataset',
        unnorm_key=_ROBOCASA_STATISTIC_NAME,
        # The released RoboCasa contract provides only the current frame.
        img_buffer_len=1,
        transforms=[
            dict(
                type='ProcessRobocasaEvalInputs',
                # The converted ``observation.images.ego_view`` videos use the
                # co-training crop produced by ``process_img_cotrain``.
                img_key='video.ego_view_bg_crop_pad_res256_freq20',
                resize_size=_IMAGE_SIZE,
                center_crop_scale=0.95,
                normalize=True,
                value_range='tanh',
                embodiment_id=0,
            ),
            dict(
                type='RobocasaGR1N15Bridge',
                apply_state_sincos=True,
            ),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=64,
                state_key='proprio',
                action_key='action',
                norm_type='min_max',
                clip_norm=True,
                normalize_states=False,
            ),
            dict(
                type='ParquetPrompter',
                use_conversation=False,
                lowercase_task_description=True,
                prompt_template=_PROMPT_TEMPLATE,
            ),
            dict(
                type='ProcessPrompts',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=_TOKENIZER,
                ),
                max_len=512,
                negative_prompt=_NEGATIVE_PROMPT,
            ),
            dict(
                type='PrepareVideo',
                num_views=_NUM_VIEWS,
                frame_window_size=1,
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeRobocasaAction',
        norm_type='min_max',
        action_dim=29,
        # Flow matching is unconstrained, while the training targets are in
        # [-1, 1]. Avoid mapping early-checkpoint outliers to unsafe joints.
        clip_actions=True,
        stats_order='fluxvla',
    ),
)

themis = dict(
    transport=dict(
        service_name='/fluxvla/predict_action',
        report_service_name='/fluxvla/report_evaluation',
        timeout_s=30.0,
        image_keys=['video.ego_view_bg_crop_pad_res256_freq20'],
        state_keys=[
            'state.left_arm',
            'state.left_hand',
            'state.right_arm',
            'state.right_hand',
            'state.waist',
        ],
        unnorm_key=_ROBOCASA_STATISTIC_NAME,
        image_encoding='rgb8',
    ),
    runner=dict(
        type='EvalRunner',
        environment=dict(
            type='RoboCasaEnvironment',
            task_list=eval['task_list'],
            action_order=eval['action_order'],
            deterministic_env=True,
            prompt_key='annotation.human.coarse_action',
            render_key='video.ego_view_pad_res256_freq20',
        ),
        model_client=dict(type='FluxVLAROSModelClient'),
        evaluator=dict(type='SuccessRateEvaluator'),
        seed=eval['seed'],
        episodes_per_task=eval['num_trials_per_task'],
        max_episode_steps=eval['max_episode_steps'],
        execute_horizon=eval['eval_chunk_size'],
        stop_on_success=True,
        parallel_workers=1,
        simulator_gpu_ids=None,
        work_dir='work_dirs/fluxthemis',
    ),
    ros_server=dict(
        ros_version=1,
        dataset_section='eval',
        evaluation_reporting=dict(
            result_output_dir='work_dirs/fluxthemis',
            report_kind='robocasa',
        ),
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
        denormalize_context={},
        denormalize_per_action=True,
    ),
)
