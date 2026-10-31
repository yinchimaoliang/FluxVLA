# Copyright 2026 Limx Dynamics
"""Compare a Pi05 VLM checkpoint with its base on official TextVQA val.

python scripts/eval.py \
  --config configs/pi05/pi05_paligemma_textvqa_eval.py --ckpt-path latest
Use --ckpt-path base to evaluate only the pretrained baseline.
Explicit --ckpt-path /path/to/checkpoint.safetensors also works.
"""

_base_ = './pi05_paligemma_llava665k_full_finetune.py'
