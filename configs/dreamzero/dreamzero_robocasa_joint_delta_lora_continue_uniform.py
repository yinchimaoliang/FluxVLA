# Copyright 2026 Limx Dynamics
"""Multi-task continuation with unit action timestep loss weights.

The paired control has identical initialization, data, video loss, optimizer,
schedule and effective batch size. Only action timestep weighting changes.
This remains an experiment; the single-task pilot did not establish a higher
RoboCasa success rate. See docs/dreamzero_robocasa_continuation.md.
"""

_base_ = './dreamzero_robocasa_joint_delta_lora_continue_control.py'

model = dict(vla_head=dict(action_timestep_weighting='uniform'))
