# Copyright 2026 Limx Dynamics
"""Start a new no-OCR TextVQA run from the pretrained Pi05 base.

This is NOT random initialization or continuation of a LLaVA SFT run.
Inherit the base checkpoint's OpenPI name mapping and unused expert-embedding
skip rule. Optimizer/scheduler start fresh; no previous run is required.
Prepare official TRAIN only with tools/prepare_textvqa_sft.py.
"""

_base_ = './pi05_paligemma_llava665k_full_finetune.py'

model = dict(
    pretrained_name_or_path='./checkpoints/pi05_base/model.safetensors')

train_dataloader = dict(
    dataset=dict(
        anno_json_path='./datasets/textvqa/textvqa_train_sft.jsonl',
        data_root_path='./datasets/llava665k/textvqa/train_images'))

# Keep the existing starting recipe; do not tune it against validation.
# Global batch = world_size * per_device_batch_size * accumulation_steps.
runner = dict(
    resume_from=None,
    max_steps=None,
    max_epochs=1,
    metric=dict(run_dir='work_dirs/pi05_textvqa_from_base'))

eval = dict(
    checkpoint_dir='work_dirs/pi05_textvqa_from_base/checkpoints',
    run_dir='work_dirs/pi05_textvqa_from_base_eval',
    ocr_question_path=None)
