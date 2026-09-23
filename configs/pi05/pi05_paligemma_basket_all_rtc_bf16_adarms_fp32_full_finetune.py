# Copyright 2026 Limx Dynamics
"""Historical basket recipe protecting AdaRMS and residuals in FP32.

For additional attention and early-expert protection, use the
bf16_expert6_fp32 recipe. This config retains its historical precision.
"""

_base_ = ['./pi05_paligemma_basket_all_rtc_full_finetune.py']

model = dict(
    enable_mixed_precision_training=True,
    preserve_fp32_residuals=True,
    vision_backbone=dict(preserve_fp32_residuals=True),
    llm_expert=dict(adarms_fp32=True))
inference_model = model.copy()
# Inference runners read this section, independently of the training runner.
# Keep FP32 weights for the flow projections and protected AdaRMS operations.
inference = dict(
    enable_mixed_precision=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True)
runner = dict(
    max_keep_ckpts=2,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True)
