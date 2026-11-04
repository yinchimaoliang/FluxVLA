# Copyright 2026 Limx Dynamics
"""RoboCasa action training + LLaVA/EgoDex image-text auxiliary training.

Prepare with tools/prepare_pi05_robocasa_egodex.py before launching. Keep the
existing VLA-only and LLaVA-only mixed recipes unchanged for comparisons.
EgoDex is the train/basic_pick_place reset subset, not human-pose supervision.

40K optimizer updates = 30K RoboCasa + 10K VLM. On two GPUs, both tasks use
global batch 16. Within VLM, LLaVA and EgoDex have equal effective row counts;
this is a sample ratio, not an equal token/gradient or optimizer-step ratio.
The auxiliary mixture is an ablation hypothesis, not a promised score gain.
"""

_base_ = './pi05_paligemma_robocasa_llava665k_mixed_full_finetune.py'

vlm_dataloader = dict(
    # Retain the control's memory-conservative microbatch/accumulation.
    per_device_batch_size=1,
    grad_accumulation_steps=8,
    # Two persistent workers/rank prefetch image decoding and tokenization.
    # This is a starting point, not a measured throughput optimum.
    per_device_num_workers=2,
    dataset=dict(
        anno_json_path=[
            dict(
                path='./datasets/vlm/robocasa_egodex/llava_train.jsonl',
                weight=1.0),
            dict(
                path='./datasets/vlm/robocasa_egodex/egodex_train.jsonl',
                # Weights are REPEAT FACTORS, not probabilities. Recompute
                # this from manifest.json if the prepared sources change.
                weight=665118 / 8192),
        ],
        data_root_path=[
            './datasets/llava665k',
            './datasets/vlm/robocasa_egodex',
        ],
        index_cache_dir='./datasets/vlm/robocasa_egodex/.index',
        # Preparation records exclusions. Do not silently replace errors
        # with another source during training and change the mixture.
        max_retry_times=0))

runner = dict(
    metric=dict(run_dir='work_dirs/pi05_robocasa_llava665k_egodex_mixed'))

# Inherit the 24-task, 50-trial robot evaluator and joint-delta statistics.
eval = dict(output_dir='work_dirs/pi05_robocasa_llava665k_egodex_mixed')
