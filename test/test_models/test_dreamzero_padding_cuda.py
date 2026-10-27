# Copyright 2026 Limx Dynamics
"""Real causal Wan attention must isolate complete padded future blocks."""

import pytest
import torch

from fluxvla.models.heads.dreamzero_head import DreamZeroHead
from fluxvla.models.third_party_models.dreamzero.modules.wan_video_vae import \
    VideoVAE_

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required')


def test_causal_vae_preserves_latents_when_future_frames_are_padded():
    vae = VideoVAE_(dim=4, z_dim=16, num_res_blocks=1).cuda().eval()
    video = torch.randn(1, 3, 9, 16, 16, device='cuda')
    padded = torch.cat((video, video[:, :, -1:].expand(-1, -1, 24, -1, -1)),
                       dim=2)
    scale = [torch.zeros(16, device='cuda'), torch.ones(16, device='cuda')]
    with torch.no_grad():
        expected = vae.encode(video, scale)
        actual = vae.encode(padded, scale)[:, :, :expected.shape[2]]
    torch.testing.assert_close(actual, expected)


def test_padded_dit_batch_matches_single_sample_outputs_losses_and_gradients(
        monkeypatch):
    head = DreamZeroHead(
        action_dim=7,
        max_action_dim=32,
        action_horizon=4,
        max_state_dim=64,
        num_frames=33,
        num_frame_per_block=2,
        num_action_per_block=4,
        num_state_per_block=1,
        frame_seqlen=16,
        hidden_size=32,
        input_embedding_dim=1536,
        dit_dim=64,
        dit_ffn_dim=128,
        dit_num_heads=4,
        dit_num_layers=2,
        dit_freq_dim=64,
        dit_in_dim=36,
        dit_out_dim=16,
        max_num_embodiments=1,
        skip_pretrained_loading=True,
        wan_model_path=None,
        use_gradient_checkpointing=True).cuda().train()
    # The default video output projection is zero-initialized. Make video
    # predictions depend on attention as well, so this checks both modalities.
    torch.nn.init.normal_(head.model.head.head.weight, std=0.02)
    monkeypatch.setattr(torch, 'randn_like',
                        lambda value: torch.full_like(value, 0.75))
    monkeypatch.setattr(
        torch, 'randint',
        lambda lo, hi, shape: torch.full(shape, 500, dtype=torch.long))
    batch = dict(
        prompt_embs=torch.randn(2, 512, 4096, device='cuda'),
        latents=torch.randn(2, 16, 9, 8, 8, device='cuda'),
        clip_feas=torch.randn(2, 1, 1280, device='cuda'),
        ys=torch.randn(2, 20, 9, 8, 8, device='cuda'),
        states=torch.randn(2, 4, 64, device='cuda'),
        actions=torch.randn(2, 16, 32, device='cuda'),
        action_masks=torch.ones(2, 16, 32, dtype=torch.bool, device='cuda'),
        embodiment_ids=torch.zeros(2, dtype=torch.long, device='cuda'))
    batch['action_masks'][:, :, 7:] = False
    predictions = []
    hook = head.model.register_forward_hook(
        lambda module, args, output: predictions.append(
            tuple(x.detach() for x in output)))
    singles = []
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for index, blocks in enumerate((1, 4)):
                single = {
                    key: value[index:index + 1]
                    for key, value in batch.items()
                }
                for key in ('latents', 'ys'):
                    single[key] = single[key][:, :, :1 + 2 * blocks]
                single['states'] = single['states'][:, :blocks]
                for key in ('actions', 'action_masks'):
                    single[key] = single[key][:, :4 * blocks]
                singles.append(head(**single))
            # These are entirely future blocks for sample 0. Valid outputs
            # must remain equal even when their values are very different.
            batch['latents'][0, :, 3:] = 10
            batch['ys'][0, :, 3:] = 10
            batch['states'][0, 1:] = 10
            batch['actions'][0, 4:] = 10
            padded = head(
                **batch, num_valid_blocks=torch.tensor([1, 4], device='cuda'))
    finally:
        hook.remove()
    for index, blocks in enumerate((1, 4)):
        torch.testing.assert_close(
            predictions[2][0][index:index + 1, :, :1 + 2 * blocks],
            predictions[index][0],
            rtol=0.02,
            atol=0.01)
        torch.testing.assert_close(
            predictions[2][1][index:index + 1, :4 * blocks],
            predictions[index][1],
            rtol=0.02,
            atol=0.01)
    for key in ('loss', 'dynamics_loss', 'action_loss'):
        torch.testing.assert_close(
            padded[key], (singles[0][key] + singles[1][key]) / 2,
            rtol=0.01,
            atol=0.002)
    parameters = list(head.parameters())
    expected_grads = torch.autograd.grad(
        (singles[0]['loss'] + singles[1]['loss']) / 2,
        parameters,
        allow_unused=True)
    actual_grads = torch.autograd.grad(
        padded['loss'], parameters, allow_unused=True)
    expected = torch.cat(
        [g.flatten() for g in expected_grads if g is not None])
    actual = torch.cat([g.flatten() for g in actual_grads if g is not None])
    assert torch.isfinite(actual).all()
    assert actual.norm() > 0
    assert (actual - expected).norm() / expected.norm() < 0.02
