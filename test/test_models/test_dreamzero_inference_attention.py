# Copyright 2026 Limx Dynamics
"""Cached denoising must preserve the attention mask used in training."""

import pytest
import torch
import torch.nn.functional as F

from fluxvla.models.third_party_models.dreamzero.modules import \
    wan2_1_submodule
from fluxvla.models.third_party_models.dreamzero.modules.wan_video_dit_action_casual_chunk import \
    CausalWanModel  # noqa: E501


class _MathAttention(torch.nn.Module):
    """Use FP32 math attention to separate mask parity from fused rounding."""

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2),
            v.transpose(1, 2)).transpose(1, 2)


@pytest.mark.parametrize('batch_size,num_blocks', [(1, 1), (2, 3)])
def test_cached_denoising_matches_teacher_forcing(batch_size, num_blocks,
                                                  monkeypatch):
    monkeypatch.setattr(wan2_1_submodule, 'flash_attention',
                        lambda q, k, v, **kwargs: _MathAttention()(q, k, v))
    torch.manual_seed(77)
    model = CausalWanModel(
        model_type='i2v',
        in_dim=6,
        out_dim=2,
        dim=32,
        ffn_dim=64,
        freq_dim=16,
        text_dim=8,
        text_len=5,
        num_heads=2,
        num_layers=3,
        frame_seqlen=4,
        num_frame_per_block=2,
        num_action_per_block=3,
        num_state_per_block=1,
        action_dim=3,
        max_state_dim=4,
        hidden_size=8,
        max_chunk_size=4).eval()
    for block in model.blocks:
        block.self_attn.attn = _MathAttention()
        block.cross_attn.attn = _MathAttention()
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.02)
    frames = 1 + 2 * num_blocks
    clean = torch.randn(batch_size, 2, frames, 4, 4)
    noisy = torch.randn_like(clean)
    actions = torch.randn(batch_size, 3 * num_blocks, 3)
    states = torch.randn(batch_size, num_blocks, 4)
    context = torch.randn(batch_size, 5, 8)
    clip = torch.randn(batch_size, 257, 1280)
    condition = torch.randn(batch_size, 4, frames, 4, 4)
    ids = torch.zeros(batch_size, dtype=torch.long)
    times = torch.full((batch_size, frames), 500.)
    action_times = torch.full((batch_size, 3 * num_blocks), 500.)
    caches = [torch.zeros(2, batch_size, 0, 2, 16) for _ in model.blocks]
    with torch.no_grad():
        train_video, train_actions = model(
            noisy,
            timestep=times,
            timestep_action=action_times,
            context=context,
            seq_len=frames * 4,
            clean_x=clean,
            action=actions,
            state=states,
            embodiment_id=ids,
            y=condition,
            clip_feature=clip)
        _, _, caches = model(
            clean[:, :, :1],
            timestep=torch.zeros(batch_size, 1),
            context=context,
            seq_len=4,
            kv_cache=caches,
            crossattn_cache=None,
            current_start_frame=0,
            y=condition[:, :, :1],
            clip_feature=clip)
        for i in range(num_blocks):
            start = 1 + 2 * i
            video_slice = slice(start, start + 2)
            action_slice = slice(3 * i, 3 * (i + 1))
            video, action, _ = model(
                noisy[:, :, video_slice],
                timestep=times[:, video_slice],
                timestep_action=action_times[:, action_slice],
                context=context,
                seq_len=8,
                kv_cache=caches,
                crossattn_cache=None,
                current_start_frame=start,
                action=actions[:, action_slice],
                state=states[:, i:i + 1],
                embodiment_id=ids,
                y=condition[:, :, video_slice],
                clip_feature=clip)
            torch.testing.assert_close(
                action, train_actions[:, action_slice], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(
                video, train_video[:, :, video_slice], atol=1e-5, rtol=1e-5)
            # Speculative noisy tokens are discarded; commit real video only.
            _, _, caches = model(
                clean[:, :, video_slice],
                timestep=torch.zeros(batch_size, 2),
                context=context,
                seq_len=8,
                kv_cache=caches,
                crossattn_cache=None,
                current_start_frame=start,
                y=condition[:, :, video_slice],
                clip_feature=clip)
