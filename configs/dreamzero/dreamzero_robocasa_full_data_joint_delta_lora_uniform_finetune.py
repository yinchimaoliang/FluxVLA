# Copyright 2026 Limx Dynamics
"""Start a new RoboCasa fine-tune from the original AgiBot pretrained base.

Only action timestep loss weighting changes from the inherited recipe.
LoRA is initialized anew; no previous RoboCasa adapter or optimizer is loaded.
See docs/dreamzero_robocasa_training.md for launch commands and batch sizing.
"""

_base_ = './dreamzero_robocasa_full_data_joint_delta_lora_finetune.py'

model = dict(vla_head=dict(action_timestep_weighting='uniform'))
