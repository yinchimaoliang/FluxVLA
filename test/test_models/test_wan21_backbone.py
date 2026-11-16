# Copyright 2026 Limx Dynamics
"""Conditioning follows the encoded latent shape and precision."""

import pytest
import torch

from fluxvla.models.backbones.vlms.wan21_backbone import Wan21Backbone
from fluxvla.models.backbones.vlms.wan_backbone import WanBaseBackbone


class DummyVAE(torch.nn.Module):

    def __init__(self, spatial_stride=8):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.spatial_stride = spatial_stride
        self.input_video = None

    def encode(self, video):
        self.input_video = video.clone()
        batch, _, frames, height, width = video.shape
        shape = (batch, 16, 1 + (frames - 1) // 4,
                 height // self.spatial_stride, width // self.spatial_stride)
        return torch.arange(
            torch.tensor(shape).prod().item(),
            device=video.device).reshape(shape).to(video.dtype) / 256


class DummyImageEncoder(torch.nn.Module):

    def encode_image(self, image):
        return image.mean(dim=(-2, -1))


def make_backbone(spatial_stride=8):
    # Exercise the real method without constructing pretrained encoders.
    backbone = Wan21Backbone.__new__(Wan21Backbone)
    WanBaseBackbone.__init__(backbone)
    backbone.vae = DummyVAE(spatial_stride)
    backbone.image_encoder = DummyImageEncoder()
    return backbone


@pytest.mark.parametrize('num_frames,spatial_stride', [(1, 8), (9, 8), (33, 8),
                                                       (9, 16)])
def test_image_condition_matches_vae_shape_and_dtype(num_frames,
                                                     spatial_stride):
    backbone = make_backbone(spatial_stride)
    image = torch.randn(2, 1, 3, 32, 48, dtype=torch.bfloat16)
    clip, condition, first_latent = backbone.encode_image(
        image, num_frames, 32, 48)
    assert condition.dtype == first_latent.dtype == torch.bfloat16
    assert condition.shape == (2, 20, 1 + (num_frames - 1) // 4,
                               32 // spatial_stride, 48 // spatial_stride)
    assert torch.equal(clip, image.mean(dim=(-2, -1)))
    assert torch.all(condition[:, :4, :1] == 1)
    assert torch.all(condition[:, :4, 1:] == 0)
    assert torch.equal(first_latent, condition[:, 4:, :1])
    assert torch.equal(backbone.vae.input_video[:, :, :1],
                       image.transpose(1, 2))
    assert torch.all(backbone.vae.input_video[:, :, 1:] == 0)


def test_prompt_padding_does_not_truncate_other_samples():

    class TextEncoder(torch.nn.Module):

        def forward(self, tokens, masks):
            return tokens.unsqueeze(-1).expand(-1, -1, 3).float()

    backbone = WanBaseBackbone()
    backbone.text_encoder = TextEncoder()
    tokens = torch.arange(1, 13).reshape(2, 6)
    masks = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0]])
    result = backbone.encode_prompt_embeddings(tokens, masks, torch.bfloat16)
    assert torch.all(result[0, 2:] == 0)
    assert torch.all(result[1, 5:] == 0)
    assert torch.equal(result[1, :5, 0], tokens[1, :5].to(torch.bfloat16))
