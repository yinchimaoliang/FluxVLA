# Copyright 2026 Limx Dynamics
"""Compatibility name for Pi05 configurations predating VLMSFTCollator."""

from fluxvla.engines import COLLATORS
from .vlm_sft_collator import VLMSFTCollator


@COLLATORS.register_module()
class Pi05VLMSFTCollator(VLMSFTCollator):
    """Keep legacy imports/configs working with the shared views layout."""

    def __init__(self, pad_token_id=0, ignore_index=-100):
        super().__init__(pad_token_id, ignore_index, image_layout='views')
