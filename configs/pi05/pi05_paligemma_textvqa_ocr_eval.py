# Copyright 2026 Limx Dynamics
"""Explicit OCR-assisted TextVQA; never replaces the default no-OCR score.

Use the LLaVA-1.5 eval.zip question file, not annotations containing answers.
Pi05 keeps its own tokenizer, image preprocessing, and trained causal mask.
See docs/pi05_textvqa.md for setup and the LLaVA-only training recipe.
"""

_base_ = './pi05_paligemma_textvqa_eval.py'

eval = dict(
    ocr_question_path='./datasets/textvqa/llava_textvqa_val_v051_ocr.jsonl',
    run_dir='work_dirs/pi05_textvqa_ocr_eval')
