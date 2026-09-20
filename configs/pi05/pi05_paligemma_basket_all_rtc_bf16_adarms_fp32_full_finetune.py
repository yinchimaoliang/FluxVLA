# Copyright 2026 Limx Dynamics
"""Opt-in selective FP32 protection for native 42-D basket BF16 training.

Preserve FP32 residual streams and the expert's 37 AdaRMS condition
projections. Transformer attention and FFN matrix operations retain BF16
autocast. The original basket and full-FP32 recipes keep their old defaults.
Use a fresh work directory for this numerical correction.

The inference section supplies precision defaults for a robot deployment
config inheriting this recipe. The deployment config must also provide its
runner type, dataset, denormalizer and operator.
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
