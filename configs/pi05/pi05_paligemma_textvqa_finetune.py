# Copyright 2026 Limx Dynamics
"""Task-specific no-OCR QA adaptation, NOT another generic LLaVA run.

Prepare TRAIN only with tools/prepare_textvqa_sft.py. Pass the completed
FluxVLA VLM checkpoint via model.pretrained_name_or_path; its native keys
must not use the OpenPI name mapping. Hyperparameters below are a starting
recipe, not a claim of a reproduced task-specialized benchmark result.
"""

_base_ = './pi05_paligemma_llava665k_full_finetune.py'

model = dict(
    pretrained_name_or_path='./work_dirs/pi05_vlm_sft/checkpoints/'
    'latest-checkpoint.safetensors',
    name_mapping=dict(_delete_=True),
    pretrained_skip_prefixes=[])

train_dataloader = dict(
    dataset=dict(
        anno_json_path='./datasets/textvqa/textvqa_train_sft.jsonl',
        data_root_path='./datasets/llava665k/textvqa/train_images'))

# Keep the existing learning rate and architecture; don't tune to val.
runner = dict(
    max_steps=None,
    max_epochs=1,
    metric=dict(run_dir='work_dirs/pi05_textvqa_sft'))

eval = dict(
    checkpoint_dir='work_dirs/pi05_textvqa_sft/checkpoints',
    run_dir='work_dirs/pi05_textvqa_sft_eval',
    ocr_question_path=None)
