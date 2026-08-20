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
"""OpenVLA LoRA fine-tuning on the 24-task RoboCasa GR1 dataset."""

seed = 7

model = dict(
    type='OpenVLA',
    vision_backbone=dict(
        type='DinoSigLIPViTBackbone',
        vision_backbone_id='dinosiglip-vit-so-224px',
        dino_config=dict(
            model_id='dino',
            file=  # noqa: E251
            './checkpoints/vit_large_patch14_reg4_dinov2.lvd142m/model.safetensors'  # noqa: E501
        ),
        siglip_config=dict(
            model_id='siglip_224',
            file=  # noqa: E251
            './checkpoints/ViT-SO400M-14-SigLIP/open_clip_model.safetensors'  # noqa: E501
        )),
    llm_backbone=dict(
        type='LLaMa2LLMBackbone',
        llm_backbone_id='llama2-7b-pure_causal',
        llm_family='llama',
        llm_path='./checkpoints/Llama-2-7b-hf',
        llm_max_length=2048,
        hf_token=None,
        inference_mode=False,
        pad_token_id=32000),
    projector=dict(
        type='FusedMLPProjector', fused_vision_dim=2176, llm_dim=4096),
    tokenizer=dict(
        type='ActionTokenizer',
        model_path='./checkpoints/openvla-7b',
        bins=256,
        min_action=-1,
        max_action=1),
    pretrained_name_or_path='./checkpoints/openvla-7b',
    vla_head=dict(type='OpenVLAHead', norm_stats=None, vocab_size=32000),
    freeze_vision_backbone=False,
    freeze_llm_backbone=False,
    freeze_projector=False,
    use_lora=True,
    lora_rank=32,
    lora_alpha=16,
    lora_dropout=0.0,
    lora_target_modules='all-linear',
    name_mapping={
        'llm_backbone.llm': 'language_model',
        'vision_backbone.siglip_featurizer':
        'vision_backbone.fused_featurizer',
        'vision_backbone.dino_featurizer': 'vision_backbone.featurizer',
        'ls1.gamma': 'ls1.scale_factor',
        'ls2.gamma': 'ls2.scale_factor',
        'projector.projector.0': 'projector.fc1',
        'projector.projector.2': 'projector.fc2',
        'projector.projector.4': 'projector.fc3'
    })

_ROBOCASA_STATISTIC_NAME = 'robocasa_gr1_24tasks_30ep'
_ROBOCASA_DATA_ROOT = './datasets/robocasa_lerobot_V2.1'
_OFFICIAL_GR1_STATS_PATH = ('./datasets/robocasa_gr1_24tasks_first30ep/'
                            'official_groot_gr1_dataset_statistics.json')
_ROBOCASA_TASK_PREFIX = 'gr1_unified'
_ROBOCASA_ENV_SUFFIX = '_GR1ArmsAndWaistFourierHands_Env'

_ROBOCASA_TASK_NAMES = [
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
    per_device_batch_size=16,
    per_device_num_workers=8,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_ROBOCASA_STATISTIC_NAME,
        dataset_statistics_path=_OFFICIAL_GR1_STATS_PATH,
        reshuffle_each_epoch=True,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=[
                _robocasa_data_path(task_name)
                for task_name in _ROBOCASA_TASK_NAMES
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
                    # Duplicate the ego view for DINO and SigLIP.
                    video_keys=[
                        'observation.images.ego_view',
                        'observation.images.ego_view',
                    ],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    },
                    dataset_name=_ROBOCASA_STATISTIC_NAME),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=29,
                    state_dim=29,
                    state_key='proprio',
                    action_key='action',
                    norm_type='quantile',
                    state_norm_type='none',
                    action_norm_type='quantile',
                    normalize_states=False,
                    clip_norm=True,
                    normalization_epsilon=1e-8),
                dict(
                    type='ParquetPrompter',
                    lowercase_task_description=True,
                    action_tokenizer=dict(
                        type='ActionTokenizer',
                        model_path='./checkpoints/openvla-7b',
                        bins=256,
                        min_action=-1,
                        max_action=1)),
                dict(
                    type='ProcessPrompts',
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path='./checkpoints/openvla-7b'),
                    max_len=None,
                    with_labels=True),
                dict(
                    type='ResizeImagesLanczos',
                    height=256,
                    width=256,
                    backend='tensorflow'),
                dict(
                    type='ResizeImagesLanczos',
                    height=224,
                    width=224,
                    backend='tensorflow',
                    jpeg_roundtrip=True),
                dict(
                    type='AugImage',
                    rotation_range=0.0,
                    crop_scale=(0.9, 0.9),
                    crop_ratio=(1.0, 1.0),
                    prob=1.0,
                    brightness_delta=0.2,
                    contrast_range=(0.8, 1.2),
                    saturation_range=(0.8, 1.2),
                    hue_delta=0.05,
                    share_across_dinosiglip=True,
                    backend='tensorflow'),
                dict(
                    type='NormalizeImages',
                    means=[[123.515625, 116.04492188, 103.59375],
                           [128, 128, 128]],
                    stds=[[58.27148438, 57.02636719, 57.27539062],
                          [128, 128, 128]]),
            ],
            action_window_size=1,
            action_key='action',
            use_delta=False,
            statistic_name=_ROBOCASA_STATISTIC_NAME,
            window_start_idx=0,
            train_episode_fraction=1.0,
            repeat_to_full_length=True)))

runner = dict(
    type='DDPTrainRunner',
    max_epochs=None,
    max_steps=100000,
    optimizer=dict(lr=5e-4, type='AdamW', weight_decay=None),
    max_grad_norm=None,
    save_iter_interval=5000,
    max_keep_ckpts=8,
    sampler=None,
    collator=dict(
        type='PaddedCollatorForActionPrediction',
        model_max_length=2048,
        pad_token_id=32000,
        padding_side='right',
        pixel_values_dtype='fp16',
        ignore_idx=-100),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler=dict(type='constant'),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    # Official OpenVLA order keeps newly-created LoRA weights in fp32.
    lora_before_device_move=False,
    static_graph=False)

eval = dict(
    type='RobocasaEvalRunner',
    benchmark='robocasa',
    task_suite_name='robocasa',
    model_family='openvla',
    task_list=[
        _robocasa_task_env(task_name) for task_name in _ROBOCASA_TASK_NAMES
    ],
    total_tasks=24,
    eval_chunk_size=1,
    max_episode_steps=720,
    num_trials_per_task=20,
    seed=7,
    unnorm_key=_ROBOCASA_STATISTIC_NAME,
    norm_stats_path=_OFFICIAL_GR1_STATS_PATH,
    action_order='fluxvla',
    rollout_video_key='video.ego_view_bg_crop_pad_res256_freq20',
    dataset=dict(
        type='RobocasaEvalDataset',
        unnorm_key=_ROBOCASA_STATISTIC_NAME,
        transforms=[
            dict(
                type='ProcessRobocasaOpenVLAEvalInputs',
                img_key='video.ego_view_bg_crop_pad_res256_freq20',
                resize_size=224,
                center_crop=True,
                crop_scale=0.9,
                jpeg_roundtrip=True),
            dict(
                type='TransformImage',
                image_resize_strategy='resize-naive',
                input_sizes=[[3, 224, 224], [3, 224, 224]],
                means=[[123.515625, 116.04492188, 103.59375], [128, 128, 128]],
                stds=[[58.27148438, 57.02636719, 57.27539062], [128, 128,
                                                                128]]),
            dict(
                type='LiberoPromptFromInputs',
                prompt_suffix=' ',
                max_len=None,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path='./checkpoints/openvla-7b')),
        ]),
    denormalize_action=dict(
        type='DenormalizeRobocasaAction',
        norm_type='quantile',
        action_dim=29,
        clip_actions=True,
        stats_order='native'))
