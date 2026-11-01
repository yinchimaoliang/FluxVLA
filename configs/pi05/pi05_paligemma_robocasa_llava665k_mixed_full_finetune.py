# Copyright 2026 Limx Dynamics
"""Pi05 RoboCasa + OSS LLaVA-665K, paired with the VLA-only control.

40,000 total updates = 30,000 VLA + 10,000 VLM at a 3:1 task ratio.
Both tasks use global batch 16 on two 80-GB GPUs. This matches the control's
robot-data exposure, while adding language updates and their compute cost.
--eval-after-train evaluates RoboCasa robot success, not language loss.
"""

_base_ = './pi05_paligemma_robocasa_vla_control_full_finetune.py'

vlm_dataloader = dict(
    per_device_batch_size=1,
    per_device_num_workers=0,
    grad_accumulation_steps=8,
    collator=dict(type='Pi05VLMSFTCollator', pad_token_id=0),
    dataset=dict(
        type='LLaVAFormatDataset',
        anno_json_path='./datasets/vlm/llava665k/train.jsonl',
        data_root_path='/mnt/data/oss/limx_mani_data/raw_data/llava665k',
        index_cache_dir='./datasets/vlm/llava665k/.index',
        seed=_base_.runner.seed,
        transforms=[
            dict(type='BuildVLMMessages'),
            dict(
                type='ProcessPi05VLMSFTInputs',
                image_size=224,
                max_length=1024,
                tokenizer_path=_base_.runner.tokenizer.model_path),
        ]))

runner = dict(
    mixed_training=dict(task_schedule=('vla', 'vla', 'vla', 'vlm')),
    max_steps=40000,
    save_iter_interval=4000,
    # Scale LR phases with the control's corresponding robot exposure.
    lr_scheduler=dict(warmup_steps=1200, decay_steps=40000),
    metric=dict(run_dir='work_dirs/pi05_robocasa_llava665k_mixed'))

eval = dict(output_dir='work_dirs/pi05_robocasa_llava665k_mixed')
