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
"""Full-data Qwen3-VL-0.6B + GR00T fine-tuning on RoboCasa GR1.

The model and image/tokenizer pipeline follow
``gr00t_qwen3vl_0.6b_libero_10_full_finetune.py``. The RoboCasa dataset,
task list, statistics, GR1-N1.5 bridge, and evaluation setup follow the
existing GR00T/PI0.5 RoboCasa configs.

Example for two 8-GPU nodes:
    torchrun --nnodes=2 --nproc_per_node=8 \
        --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} scripts/train.py \
        --config \
        configs/gr00t/gr00t_qwen3vl_0.6b_robocasa_full_data_full_finetune.py \
        --work-dir \
        work_dirs/gr00t_qwen3vl_0.6b_robocasa_full_data_full_finetune
"""

_QWEN3VL_VLA_ROOT = './checkpoints/gr00t_qwen3vl_0.6b_libero'
_QWEN3VL_VLA_CKPT = (
    _QWEN3VL_VLA_ROOT +
    '/checkpoints/step-104160-epoch-24-loss=0.0358.safetensors')
_QWEN3VL_TOKENIZER = _QWEN3VL_VLA_ROOT + '/tokenizer/'

_QWEN3VL_VLM_CONFIG = dict(
    architectures=['Qwen3VLAForConditionalGeneration'],
    dtype='bfloat16',
    eos_token_id=151645,
    image_token_id=151655,
    model_type='qwen3_vl',
    pad_token_id=151643,
    pos_skipping_range=4096,
    text_config=dict(
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=151643,
        dtype='bfloat16',
        eos_token_id=151645,
        head_dim=128,
        hidden_act='silu',
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=3072,
        max_position_embeddings=262144,
        model_type='qwen3_vl_text',
        num_attention_heads=16,
        num_hidden_layers=28,
        num_key_value_heads=8,
        pad_token_id=None,
        rms_norm_eps=1e-06,
        rope_parameters=dict(
            mrope_interleaved=True,
            mrope_section=[24, 20, 20],
            rope_theta=5000000,
            rope_type='default'),
        tie_word_embeddings=True,
        use_cache=True,
        vocab_size=151936),
    tie_word_embeddings=True,
    use_another_LLM_path='',
    use_pos_skipping=False,
    vision_config=dict(
        deepstack_visual_indexes=[5, 11, 17],
        depth=24,
        dtype='bfloat16',
        hidden_act='gelu_pytorch_tanh',
        hidden_size=1024,
        in_channels=3,
        initializer_range=0.02,
        intermediate_size=4096,
        model_type='qwen3_vl',
        num_heads=16,
        num_position_embeddings=2304,
        out_hidden_size=1024,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2))

model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=_QWEN3VL_VLA_CKPT,
    name_mapping=None,
    strict_mapping=False,
    vlm_backbone=dict(
        type='Qwen3VL',
        vlm_backbone_id='qwen3_0.6b_vl_pt',
        vlm_path=None,
        vlm_config=_QWEN3VL_VLM_CONFIG,
        use_projection=True,
        projection_output_dim=2048,
        projection_type='linear',
        attn_implementation='sdpa'),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        backbone_embedding_dim=2048,
        vl_self_attention_cfg=dict(
            attention_head_dim=64,
            num_attention_heads=32,
            num_layers=4,
            dropout=0.2,
            final_dropout=True,
            positional_embeddings=None),
        diffusion_model_cfg=dict(
            attention_head_dim=48,
            num_attention_heads=32,
            cross_attention_dim=2048,
            num_layers=16,
            output_dim=1024,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            positional_embeddings=None),
        num_inference_timesteps=4,
        num_steps=10,
        action_dim=32,
        ori_action_dim=29),
    freeze_vlm_backbone=False,
    freeze_projector=False)

# Evaluation uses the same Qwen3-VL and flow-matching implementation.
inference_model = model.copy()

_ROBOCASA_STATISTIC_NAME = 'robocasa_gr1_24tasks_30ep'
_ROBOCASA_DATA_ROOT = './datasets/robocasa_lerobot_V2.1'
_OFFICIAL_GR1_STATS_PATH = ('./datasets/robocasa_gr1_24tasks_first30ep/'
                            'official_groot_gr1_dataset_statistics.json')
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
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_ROBOCASA_STATISTIC_NAME,
        dataset_statistics_path=_OFFICIAL_GR1_STATS_PATH,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=[
                _robocasa_data_path(task_name) for task_name in _ROBOCASA_TASKS
            ],
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    embodiment_id=24,
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
                    }),
                # Preserve the GR00T-N1.5 category-24 state/action convention:
                # sin/cos state features and N1.5 joint ordering.
                dict(type='RobocasaGR1N15Bridge'),
                dict(type='ParquetPrompter'),
                dict(
                    type='ProcessPrompts',
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=_QWEN3VL_TOKENIZER,
                    )),
                dict(type='RandomCropImages', scale=0.95),
                dict(type='ResizeImages', height=224, width=224),
                dict(
                    type='ColorJitterImages',
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.08),
                dict(
                    type='QWen2VLImageTransform',
                    min_pixels=56 * 56,
                    max_pixels=28 * 28 * 1280,
                    patch_size=16,
                    temporal_patch_size=2,
                    merge_size=2,
                    image_mean=[0.48145466, 0.4578275, 0.40821073],
                    image_std=[0.26862954, 0.26130258, 0.27577711]),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=32,
                    state_dim=64,
                    state_key='proprio',
                    action_key='action',
                    norm_type='min_max',
                    normalize_states=False),
            ],
            action_window_size=10,
            action_key='action',
            use_delta=False,
            statistic_name=_ROBOCASA_STATISTIC_NAME,
            window_start_idx=0,
        )))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=None,
    max_steps=100000,
    grad_accumulation_steps=1,
    optimizer=dict(lr=1.5e-5, type='AdamW', weight_decay=0.0),
    max_grad_norm=1.0,
    save_iter_interval=5000,
    save_epoch_interval=1,
    max_keep_ckpts=8,
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=_QWEN3VL_TOKENIZER,
    ),
    collator=dict(
        type='DictCollator',
        keys=[
            'states',
            'observation.eepose',
            'timestamp',
            'images',
            'img_masks',
            'lang_tokens',
            'lang_masks',
            'actions',
            'action_masks',
            'embodiment_ids',
            'image_grid_thw',
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        warmup_ratio=0.03,
    ),
    sharding_strategy='full-shard',
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)

eval = dict(
    type='RobocasaEvalRunner',
    benchmark='robocasa',
    task_suite_name='robocasa',
    model_family='groot',
    task_list=[_robocasa_task_env(task_name) for task_name in _ROBOCASA_TASKS],
    total_tasks=24,
    eval_chunk_size=10,
    max_episode_steps=720,
    num_trials_per_task=20,
    seed=7,
    unnorm_key=_ROBOCASA_STATISTIC_NAME,
    action_order='n15',
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='RobocasaEvalDataset',
        unnorm_key=_ROBOCASA_STATISTIC_NAME,
        transforms=[
            dict(
                type='ProcessRobocasaEvalInputs',
                img_key='video.ego_view_bg_crop_pad_res256_freq20',
                resize_size=224,
                center_crop_scale=0.95,
                # Return CHW pixels in [0, 1]. Qwen then only normalizes them.
                normalize=True,
                value_range='unit',
                embodiment_id=24),
            dict(type='RobocasaGR1N15Bridge'),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=64,
                state_key='proprio',
                action_key='action',
                norm_type='min_max',
                normalize_states=False),
            dict(
                type='QWen2VLImageTransform',
                do_rescale=False,
                min_pixels=56 * 56,
                max_pixels=28 * 28 * 1280,
                patch_size=16,
                temporal_patch_size=2,
                merge_size=2,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
                img_key='pixel_values',
                to_tensor=True),
            dict(
                type='LiberoPromptFromInputs',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=_QWEN3VL_TOKENIZER,
                )),
        ]),
    denormalize_action=dict(
        type='DenormalizeRobocasaAction',
        norm_type='min_max',
        action_dim=29,
        clip_actions=False,
        stats_order='fluxvla'),
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
