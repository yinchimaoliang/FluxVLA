# Copyright 2026 Limx Dynamics
"""FP32 basket control with state-relative targets for the 31 joint positions.

Keep the nine base pose coordinates and two hand switches unchanged. The
33-D state and 42-D action do not share those tail coordinates. Compute new
statistics over relative action windows; absolute-action statistics and
checkpoints from an absolute-action run must not be reused for this control.
At deployment, denormalize first, then add current state[:31] to actions[:31].
"""

_base_ = ['./pi05_paligemma_basket_all_rtc_fp32_full_finetune.py']

train_dataloader = dict(
    dataset=dict(
        auto_compute_statistics=dict(
            profile='absolute', delta_mask=[True] * 31),
        datasets=dict(
            transforms=(
                _base_.train_dataloader.dataset.datasets.transforms[:1] +
                [dict(type='RelativeActions', mask=[True] * 31)] +
                _base_.train_dataloader.dataset.datasets.transforms[1:]))))
