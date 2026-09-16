# Copyright 2026 Limx Dynamics
"""Full-precision control for the basket PI0.5 loss instability investigation.

Keep the batch, optimizer, RTC, data and checkpoint initialization unchanged
from the corrected recipe. Disable autocast and the explicit Gemma input cast
together. This matches LeRobot's default FP32 compute choice; it does not claim
parity with a private LeRobot run whose resolved config is unavailable.
Use a fresh work directory so the comparison preserves the original run.
"""

_base_ = ['./pi05_paligemma_basket_all_rtc_full_finetune.py']

model = dict(enable_mixed_precision_training=False)
inference_model = dict(enable_mixed_precision_training=False)
runner = dict(enable_mixed_precision_training=False)
