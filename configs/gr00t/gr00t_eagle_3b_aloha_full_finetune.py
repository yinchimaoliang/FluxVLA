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

# Dataset contract: agilex_aloha_unified@4.0.0 / dataset 2.0.0.
# Parquet stores only unified_107d robot vectors plus per-dimension masks;
# qpose/eepose are losslessly decoded at runtime to avoid duplicate columns.
# GR00T-N1.5 retains its pretrained 64D state / 32D action envelopes and uses
# packed ALOHA qpose[14] for proprioception and robot commands.
#
# This converted example has no issued target or calibrated ego2cam. Its
# target.unified_107d_mask is false, so every unavailable placeholder target
# is masked out of the loss. A new dataset version with real targets is
# required before effective action fine-tuning.

ALOHA_QPOSE_INDICES = [0, 1, 2, 3, 4, 5, 28, 7, 8, 9, 10, 11, 12, 29]
ALOHA_EEPOSE_INDICES = list(range(14, 28))

model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleBackbone',
        vlm_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model',
        vlm_config=dict(max_input_seq_len=900)),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_inference_timesteps=4,
        num_steps=32,
        action_dim=32,
        ori_action_dim=14),
    freeze_vlm_backbone=False,
    name_mapping={
        'vlm_backbone.vlm': 'backbone.eagle_model',
        'vla_head': 'action_head'
    },
    freeze_projector=False)

inference_model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleInferenceBackbone',
        vlm_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model',
        vlm_config=dict(max_input_seq_len=900)),
    vla_head=dict(
        type='FlowMatchingInferenceHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_steps=32,
        num_inference_timesteps=4,
        ori_action_dim=14,
        action_dim=32,
        max_input_seq_len=900,
        diffusion_model_cfg=dict(
            attention_head_dim=48,
            cross_attention_dim=2048,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            num_attention_heads=32,
            num_layers=16,
            output_dim=1024,
            positional_embeddings=None)))

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.unified_107d': ['proprio'],
            'target.unified_107d': ['action']
        },
        statistic_keys=['observation.unified_107d', 'target.unified_107d'],
        statistic_indices={
            'observation.unified_107d': ALOHA_QPOSE_INDICES,
            'target.unified_107d': ALOHA_QPOSE_INDICES,
        },
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=  # noqa: E251
                [
                    './datasets/RealRobot_AgileX_aloha_lerobot/example_canonical_107d_v3_1',  # noqa: E501
                ],
                expected_dataset_version='2.0.0',
                expected_schema_id='agilex_aloha_unified',
                expected_schema_version='4.0.0',
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        embodiment_id=0,
                        parquet_keys=[
                            'observation.unified_107d',
                            'observation.unified_107d_mask',
                            'observation.ego2cam', 'observation.ego2cam_valid',
                            'timestamp', 'actions', 'info', 'stats',
                            'action_masks'
                        ],
                        video_keys=[
                            'observation.images.cam_high',
                            'observation.images.cam_left_wrist',
                            'observation.images.cam_right_wrist'
                        ]),
                    dict(
                        type='DecodeAlohaUnified107D',
                        qpose_indices=ALOHA_QPOSE_INDICES,
                        eepose_indices=ALOHA_EEPOSE_INDICES),
                    dict(type='ParquetPrompter'),
                    dict(
                        type='ProcessPromptsWithImage',
                        max_len=900,
                        num_images=3,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=  # noqa: E251
                            'fluxvla/models/third_party_models/eagle2_hg_model',  # noqa: E501
                            # special_tokens={'pad_token': '<PAD>'}
                        )),
                    dict(type='ResizeImages', height=224, width=224),
                    dict(
                        type='NormalizeImages',
                        means=[[123.515625, 116.04492188, 103.59375],
                               [123.515625, 116.04492188, 103.59375],
                               [123.515625, 116.04492188, 103.59375]],
                        stds=[[58.27148438, 57.02636719, 57.27539062],
                              [58.27148438, 57.02636719, 57.27539062],
                              [58.27148438, 57.02636719, 57.27539062]],
                    ),
                    dict(
                        type='NormalizeStatesAndActions',
                        state_dim=64,
                        action_dim=32,
                        state_key='proprio',
                        action_key='action',
                        norm_type='mean_std')
                ],
                action_key='target.unified_107d',
                action_mask_key='target.unified_107d_mask',
                action_indices=ALOHA_QPOSE_INDICES,
                window_start_idx=0,
                action_window_size=32)
        ]))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=6,
    optimizer=dict(lr=2e-5, type='AdamW', weight_decay=0.0),
    max_grad_norm=1.0,
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model',
        # special_tokens={'pad_token': '<PAD>'}
    ),
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'observation.eepose', 'observation.ego2cam',
            'observation.ego2cam_valid', 'timestamp', 'images', 'img_masks',
            'lang_tokens', 'lang_masks', 'actions', 'action_masks',
            'embodiment_ids'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
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
    change_key_name=False)

inference = dict(
    type='AlohaInferenceRunner',
    seed=7,
    task_descriptions={
        '1': 'pick up the green bowl with right arm',
        '2': 'place it on the green bowl with left arm',
        '3': 'pick up the green bowl with left arm',
        '4': 'pick up the blue bowl with right arm',
        '5': 'place it on the red plate with left arm',
        '6': 'pick up the golden chocolate ball with right arm',
        '7': 'pick up the tiger toy with right arm',
        '8': 'pick up the robot dog toy with right arm',
        '9': 'pick up the shuttlecock with right arm',
        '10': 'pick up the yellow bowl with right arm',
        '11': 'pick up the giraffe toy with right arm',
        '12': 'place it in the paper bag with right arm',
        '13': 'place it on the blue bowl with left arm',
        '14': 'hold open the brown paper bag with left arm',
        '15': 'pick up the blue bowl with left arm',
    },
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='PrivateInferenceDataset',
        embodiment_id=0,
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        transforms=[
            dict(
                type='ProcessPromptsWithImage',
                max_len=900,
                num_images=3,
                tokenizer=dict(type='PretrainedTokenizer'
                               # special_tokens={'pad_token': '<PAD>'}
                               )),
            dict(type='ResizeImages', height=224, width=224),
            dict(
                type='NormalizeImages',
                means=[[123.515625, 116.04492188, 103.59375],
                       [123.515625, 116.04492188, 103.59375],
                       [123.515625, 116.04492188, 103.59375]],
                stds=[[58.27148438, 57.02636719, 57.27539062],
                      [58.27148438, 57.02636719, 57.27539062],
                      [58.27148438, 57.02636719, 57.27539062]],
            ),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=64,
                state_key='proprio',
                action_key='action',
                norm_type='mean_std')
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction', norm_type='mean_std', action_dim=14),
    operator=dict(
        type='AlohaOperator',
        img_front_topic='/camera_f/color/image_raw',
        img_left_topic='/camera_l/color/image_raw',
        img_right_topic='/camera_r/color/image_raw',
        img_front_depth_topic='/camera_f/depth/image_raw',
        img_left_depth_topic='/camera_l/depth/image_raw',
        img_right_depth_topic='/camera_r/depth/image_raw',
        puppet_arm_left_cmd_topic='/master/joint_left',
        puppet_arm_right_cmd_topic='/master/joint_right',
        puppet_arm_left_topic='/puppet/joint_left',
        puppet_arm_right_topic='/puppet/joint_right',
        robot_base_topic='/odom_raw',
        robot_base_cmd_topic='/cmd_vel',
    ))
