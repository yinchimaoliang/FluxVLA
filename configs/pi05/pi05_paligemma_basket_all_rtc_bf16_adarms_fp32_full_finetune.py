# Copyright 2026 Limx Dynamics
"""Historical AdaRMS-only recipe; long-run basket loss degradation observed.

For the additional attention/early-expert protection from ada0381668, use:
pi05_paligemma_basket_all_rtc_bf16_expert6_fp32_full_finetune.py.
Checking out that commit does not change the precision in this recipe.

Preserve FP32 residual streams and the expert's 37 AdaRMS condition
projections. Transformer attention and FFN matrix operations retain BF16
autocast. The original basket and full-FP32 recipes keep their old defaults.
This file is retained to reproduce historical experiments.

The inference section supplies precision defaults for a robot deployment
config inheriting this recipe. The deployment config must also provide its
runner type, dataset, denormalizer and operator.
"""

_base_ = ['./pi05_paligemma_basket_all_rtc_full_finetune.py']

training_recipe_warning = (
    'This historical PI05 basket recipe only protects AdaRMS and residuals; '
    'long-run loss degradation was observed. The attention/early-expert '
    'protection from ada0381668 is NOT enabled by this config by default. '
    'Use configs/pi05/'
    'pi05_paligemma_basket_all_rtc_bf16_expert6_fp32_full_finetune.py '
    'in a fresh work directory to enable that additional protection. '
    'Its six-epoch convergence has not yet been verified.')

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
