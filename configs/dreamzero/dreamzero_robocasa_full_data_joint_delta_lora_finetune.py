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
"""Aligned DreamZero adaptation on the RoboCasa GR1 tabletop tasks.

This is the recommended replacement for the legacy absolute-action/full-DiT
recipe.  It follows DreamZero's released new-embodiment adaptation contract:

* 24 actions paired with video offsets ``[0, 3, ..., 24]``;
* arm and waist joint deltas, with Fourier-hand commands kept absolute;
* rank-4 LoRA on the DiT plus trainable state/action encoders and decoder.

The old ``dreamzero_robocasa_full_data_full_finetune.py`` remains available to
reproduce and evaluate checkpoints that were already trained with 48-step
absolute actions.  Those checkpoints are not compatible with this data
contract and must not be evaluated with this config.

Example for two 8-GPU nodes sharing MASTER_ADDR and MASTER_PORT:
    torchrun --nnodes=2 --nproc_per_node=8 \
        --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} scripts/train.py \
        --config \
        configs/dreamzero/\
dreamzero_robocasa_full_data_joint_delta_lora_finetune.py \
        --work-dir \
        work_dirs/dreamzero_robocasa_full_data_joint_delta_lora_finetune
"""

_CKPT_ROOT = './checkpoints'
_TOKENIZER = _CKPT_ROOT + '/Wan2.1-I2V-14B-480P/google/umt5-xxl'

# The checkpoint supports four causal chunks (33 raw frames). New-embodiment
# adaptation uses nine observations at offsets [0, 3, ..., 24], paired with
# 24 consecutive actions, as in the released AgiBot/YAM training scripts.
_MODEL_NUM_FRAMES = 33
_TRAIN_FRAME_WINDOW_SIZE = 9
_FRAME_SAMPLE_STRIDE = 3
_ACTION_HORIZON = 24
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
    # Official DreamZero adaptation freezes the pretrained DiT weights and
    # learns rank-4 adapters, while keeping the embodiment-specific
    # state/action encoders and action decoder trainable.
    use_lora=True,
    lora_rank=4,
    lora_alpha=4,
    lora_dropout=0.0,
    lora_target_modules=(
        r'^vla_head\.model\.blocks\.\d+\.(?:self_attn|cross_attn)\.'
        r'(?:q|k|v|o)$|^vla_head\.model\.blocks\.\d+\.ffn\.(?:0|2)$'),
    modules_to_save=[
        'state_encoder',
        'action_encoder',
        'action_decoder',
    ],
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

_ROBOCASA_STATISTIC_NAME = 'robocasa_gr1_24tasks_joint_delta'
_ROBOCASA_DATA_ROOT = './datasets/robocasa_lerobot_V2.1'
_ROBOCASA_TASK_PREFIX = 'gr1_unified'
_ROBOCASA_ENV_SUFFIX = '_GR1ArmsAndWaistFourierHands_Env'

# Raw parquet order: left arm, left hand, right arm, right hand, waist.
_ROBOCASA_JOINT_DELTA_MASK = ([True] * 7 + [False] * 6 + [True] * 7 +
                              [False] * 6 + [True] * 3)
# DreamZero model/evaluation order after RobocasaGR1N15Bridge: left arm,
# right arm, left hand, right hand, waist.
_ROBOCASA_N15_JOINT_DELTA_MASK = ([True] * 7 + [True] * 7 + [False] * 6 +
                                  [False] * 6 + [True] * 3)

# Exact statistics for the 24-step mixed-relative action contract,
# computed from all 24 RoboCasa tasks / 24,000 episodes. Keeping them
# inline avoids a long rank-0 preprocessing pass and NCCL barrier timeout.
_ROBOCASA_DATASET_STATISTICS = {
    'robocasa_gr1_24tasks_joint_delta': {
        'proprio': {
            'mean': [
                -0.17102599661893628, 0.23514659219974143,
                -0.11291017724516655, -1.4712459937182183, 0.17786245443243578,
                0.10149730971890933, -0.006413776953445108, 0.1501827995845328,
                0.1494419544656495, 0.13239416445355168, 0.14923437345912333,
                0.03464704927460434, 0.6337165986729069, -0.32031619897959673,
                -0.3216767278879253, 0.08499577598004783, -1.4728465043313443,
                0.337759733973541, 0.0690973699961625, 0.16150938097923964,
                0.48730800244509526, 0.4577669563030523, 0.42192603277520035,
                0.4522622699974382, 0.07554572365544217, 1.6688002354105917,
                0.0035976467848950486, 0.004405950191815653,
                -8.76750048624244e-05
            ],
            'std': [
                0.37524638882030775, 0.17814931412288376, 0.2653912135335837,
                0.46622207697619644, 0.2902881388466011, 0.2859473255400859,
                0.3154311391865683, 0.41833180143882503, 0.40040645672050007,
                0.35368029790605027, 0.4026344337142858, 0.13890309096880296,
                0.8191262033029768, 0.5024563669478592, 0.2778612776919975,
                0.37864273716884367, 0.6704838466386019, 0.5231829479691116,
                0.3766345142460483, 0.5706149895448006, 0.5937571234710008,
                0.5513828952053532, 0.506991319063774, 0.5441043791035441,
                0.17027079980800192, 0.21279093629856186, 0.06621792097846523,
                0.01964003934576096, 0.007552978323748122
            ],
            'min': [
                -1.6789460182189941, -0.026101894676685333,
                -1.3480229377746582, -2.5160419940948486, -1.9940674304962158,
                -1.3795876502990723, -1.1958755254745483, -1.4389894008636475,
                -1.8303323984146118, -2.4635109901428223, -1.7167329788208008,
                -2.218892812728882, -1.526924967765808, -2.0664756298065186,
                -2.1021976470947266, -2.296651601791382, -2.5318210124969482,
                -3.0013694763183594, -1.4908946752548218, -1.2908861637115479,
                -1.4716511964797974, -2.0171985626220703, -2.412123203277588,
                -1.189025640487671, -0.8325809836387634, -0.21484142541885376,
                -0.5222951769828796, -0.42820972204208374, -0.39791223406791687
            ],
            'max': [
                1.3502349853515625, 1.2633577585220337, 1.2589013576507568,
                0.001734813442453742, 2.521491289138794, 1.526998519897461,
                1.496475338935852, 2.0179455280303955, 2.009377956390381,
                2.6196515560150146, 1.8978251218795776, 3.2151029109954834,
                2.7924649715423584, 1.5148204565048218, 0.003278259886428714,
                1.7851011753082275, 0.0016116079641506076, 3.0015335083007812,
                1.4080945253372192, 1.4516682624816895, 2.7859506607055664,
                2.1664254665374756, 3.0131356716156006, 2.69866681098938,
                1.4733597040176392, 2.079848289489746, 0.937696099281311,
                0.3457968235015869, 0.47687003016471863
            ],
            'q01': [
                -1.414753302335739, -0.0005171521747251973,
                -0.9782302141189575, -2.477926731109619, -0.34331061780452726,
                -0.6772882187366486, -0.9085569721460343, -0.2537090674042702,
                -0.01579869568347931, -0.010405048383399845,
                -0.002593582069966942, -0.14740002006292344,
                -0.0005192354379687458, -1.4483043837547303,
                -1.0833211290836333, -0.8002108770608902, -2.507189002037048,
                -0.7147443491220474, -0.9463434845209122, -1.0013096010684968,
                -0.004114496670663357, -0.004300017701461911,
                -0.0054274908918887374, -0.004352558837272227,
                -0.13891243800520897, 0.5844185560941696, -0.2750973534584045,
                -0.031067517586052418, -0.022482833340764046
            ],
            'q99': [
                0.7154520624876013, 0.7829802078008643, 0.4349772733449915,
                -0.170762614309788, 1.0356361699104308, 0.8310365939140318,
                0.7348084545135478, 1.500377825498581, 1.4995973110198975,
                1.2963906359672546, 1.5020229816436768, 0.6279667210578896,
                1.846041305065155, 0.9360333341360092, -0.00022184939269209443,
                0.8936860918998715, -0.08954395778477237, 1.5831496250629415,
                0.8285187083482737, 1.232260091304779, 1.497841477394104,
                1.4997276926040648, 1.5635861182212825, 1.518557515144348,
                0.6836496728658665, 1.8118253779411315, 0.2004491922259326,
                0.08782679289579387, 0.022298754360526682
            ],
            'count':
            6020058
        },
        'action': {
            'mean': [
                -0.009214382580729107, -0.008712207598760227,
                0.0028056072967646613, 0.013662939325045078,
                8.582446510216132e-05, -0.009820096883617136,
                -0.001631907141514234, -0.21731578963861437,
                -0.21731578963861437, -0.21731578963861437,
                -0.21731578963861437, -0.43463157927722873, 1.1291277003122182,
                -0.036858917356646946, -0.006458900745127183,
                -0.02505556679557638, 0.044850384472803984,
                0.026545887866965705, 0.005072288479375176,
                0.01952781587349067, -0.4750786377853427, -0.4750786377853427,
                -0.4750786377853427, -0.4750786377853427, -0.9501572755706854,
                3.0, 0.00048485272605788333, 0.002588117780747494,
                1.2864913478592177e-05
            ],
            'std': [
                0.12428913729168067, 0.061947062792788575, 0.0944412748544423,
                0.17272552121597168, 0.09978729422790482, 0.10598432729495219,
                0.11319969007553182, 0.8942145260357015, 0.8942145260357015,
                0.8942145260357015, 0.8942145260357015, 1.788429052071403,
                1.4534282718792289, 0.21658750934038998, 0.14868284373253013,
                0.18060408439749426, 0.29878908422589107, 0.22515418628709333,
                0.21927060992832306, 0.2826972122626373, 1.4227790711888844,
                1.4227790711888844, 1.4227790711888844, 1.4227790711888844,
                2.8455581423777687, 0.0, 0.03647094811892615,
                0.0132199972265246, 0.006882453346600453
            ],
            'min': [
                -1.1532576084136963, -0.7710707187652588, -1.3112866878509521,
                -1.5591628551483154, -1.6683788299560547, -1.3634177446365356,
                -1.1948705911636353, -1.5, -1.5, -1.5, -1.5, -3.0, 0.0,
                -1.482505440711975, -1.5787591934204102, -1.6719448566436768,
                -1.9195635318756104, -1.7429898977279663, -1.7818541526794434,
                -2.0731394290924072, -1.5, -1.5, -1.5, -1.5, -3.0, 3.0,
                -0.4199202060699463, -0.26482921838760376, -0.2804606854915619
            ],
            'max': [
                1.206160545349121, 1.0281989574432373, 1.18000328540802,
                1.896183729171753, 1.784941554069519, 1.3858094215393066,
                1.2729451656341553, 1.5, 1.5, 1.5, 1.5, 3.0, 3.0,
                1.6917004585266113, 1.1616510152816772, 1.5212913751602173,
                1.9471081495285034, 1.8994455337524414, 1.6273449659347534,
                2.135958671569824, 1.5, 1.5, 1.5, 1.5, 3.0, 3.0,
                0.4214596152305603, 0.28732696175575256, 0.3515920639038086
            ],
            'q01': [
                -0.4285609748959541, -0.15905308991670608, -0.3703790333867073,
                -0.5776357108354568, -0.27032649517059326, -0.3067216655611992,
                -0.4100598746538162, -1.5, -1.5, -1.5, -1.5, -3.0, 0.0,
                -0.689931880235672, -0.4814644780755043, -0.5751662904024124,
                -0.7938064658641816, -0.5646731853485107, -0.6040554696321487,
                -0.6687421852350235, -1.5, -1.5, -1.5, -1.5, -3.0, 3.0,
                -0.1108991462737322, -0.03692713920027018,
                -0.023431802336126566
            ],
            'q99': [
                0.41350418865680716, 0.2414393600821496, 0.2361499720811846,
                0.620225277543069, 0.3593881157040597, 0.41745162278413783,
                0.37283100038766914, 1.5, 1.5, 1.5, 1.5, 3.0, 3.0,
                0.5794540750980381, 0.38423311710357666, 0.4631571823358538,
                0.9261428171396258, 0.7127108842134486, 0.6551499634981166,
                0.9581202375888829, 1.5, 1.5, 1.5, 1.5, 3.0, 3.0,
                0.11290151685476335, 0.042237360365688814, 0.020694109182804843
            ],
            'count':
            137857392
        }
    }
}

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
    # Match the released LoRA recipe: one sample per GPU and no accumulation.
    # The effective global batch therefore equals the distributed world size.
    per_device_batch_size=1,
    # Keep decoding in the rank process.  Each DreamZero rank temporarily
    # occupies about 90 GiB of host RAM while loading the 23B checkpoint;
    # forking four persistent workers per rank from a debugpy launch can make
    # PyAV abort natively (SIGABRT) before Python can emit a traceback.
    per_device_num_workers=0,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_ROBOCASA_STATISTIC_NAME,
        # Exact full-data statistics for the transformed h24 targets.
        dataset_statistics=_ROBOCASA_DATASET_STATISTICS,
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
                    # TorchCodec 0.7 is installed with torch 2.8 in the
                    # FluxVLA environment.  It returns the same frames for
                    # this dataset while avoiding torchvision's deprecated
                    # PyAV VideoReader path seen immediately before SIGABRT.
                    video_backend='torchcodec',
                ),
                dict(
                    type='RelativeActions',
                    mask=_ROBOCASA_JOINT_DELTA_MASK,
                    state_key='states',
                    action_key='actions',
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
            # Match one complete new-embodiment block from DreamZero.
            action_window_size=_ACTION_HORIZON,
            action_key='action',
            # RelativeActions above performs state-relative conversion using
            # the current observation, not the legacy action-to-action delta.
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
    type='DDPTrainRunner',
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
        type='DenormalizeRobocasaDeltaAction',
        norm_type='min_max',
        action_dim=29,
        delta_action_mask=_ROBOCASA_N15_JOINT_DELTA_MASK,
        state_order='fluxvla',
        action_order='n15',
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
