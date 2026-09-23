# Copyright 2026 Limx Dynamics
"""Protect the early action expert and attention backward from BF16 errors.

The first six expert blocks use FP32 projections and FFNs. Both Gemma
branches use FP32 math attention; casting SDPA inputs alone does not select
that backend. Vision, language projections/FFNs, and the remaining expert
projections/FFNs retain BF16 autocast. FP32 master parameters are required.

This is a separate recipe so historical experiments retain their defaults.
Start from the base checkpoint in a fresh work directory. Short numerical
and execution checks do not establish long-run convergence.
"""

_base_ = ['./pi05_paligemma_basket_all_rtc_bf16_adarms_fp32_full_finetune.py']

model = dict(
    llm_backbone=dict(attention_math_fp32=True),
    llm_expert=dict(attention_math_fp32=True, fp32_layers=(0, 1, 2, 3, 4, 5)))
inference_model = model.copy()
