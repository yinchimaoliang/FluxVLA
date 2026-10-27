# Copyright 2026 Limx Dynamics

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from mmengine import Config

from fluxvla.engines import build_collator_from_cfg

ROOT = Path(__file__).resolve().parents[2]


def _sample(blocks):
    frames = 1 + 8 * blocks
    return dict(
        states=np.ones((blocks, 64), dtype=np.float32),
        actions=np.ones((24 * blocks, 32), dtype=np.float32),
        action_masks=np.ones(24 * blocks, dtype=bool),
        images=np.arange(3 * frames * 4 * 4,
                         dtype=np.float32).reshape(3, frames, 4, 4),
        img_masks=np.ones(frames, dtype=bool),
        frame_masks=np.ones(frames, dtype=np.float32),
        lang_tokens=np.ones(512, dtype=np.int64),
        lang_masks=np.ones(512, dtype=np.int64),
        embodiment_ids=np.array(0, dtype=np.int64),
        task_description='move the cup',
    )


@pytest.fixture
def collator():
    cfg = Config.fromfile(
        ROOT / 'configs/dreamzero' /
        'dreamzero_robocasa_full_data_joint_delta_lora_finetune.py')
    return build_collator_from_cfg(cfg.runner.collator)


@pytest.mark.parametrize('counts', [(1, ), (1, 4), (3, 4), (2, 2),
                                    (1, 2, 3, 4)])
@pytest.mark.parametrize('use_tensors', [False, True])
def test_collate_preserves_samples_and_masks_padding(collator, counts,
                                                     use_tensors):
    samples = [_sample(blocks) for blocks in counts]
    if use_tensors:
        for sample in samples:
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    sample[key] = torch.from_numpy(value)
    original = copy.deepcopy(samples)
    batch = collator(samples)
    assert batch['images'].shape == (len(counts), 3, 1 + 8 * max(counts), 4, 4)
    assert batch['actions'].shape == (len(counts), 24 * max(counts), 32)
    assert batch['states'].shape == (len(counts), max(counts), 64)
    assert batch['num_valid_blocks'].tolist() == list(counts)
    assert batch['num_valid_blocks'].dtype == torch.long
    assert batch['task_description'] == ['move the cup'] * len(counts)
    for i, blocks in enumerate(counts):
        for key, before in original[i].items():
            if isinstance(before, str):
                continue
            value = torch.as_tensor(before)
            torch.testing.assert_close(torch.as_tensor(samples[i][key]), value)
            if key == 'images':
                torch.testing.assert_close(batch[key][i, :, :value.shape[1]],
                                           value)
                padding = batch[key][i, :, value.shape[1]:]
                torch.testing.assert_close(padding,
                                           value[:, -1:].expand_as(padding))
            elif key in ('states', 'actions', 'action_masks', 'img_masks',
                         'frame_masks'):
                torch.testing.assert_close(batch[key][i, :value.shape[0]],
                                           value)
                assert not batch[key][i, value.shape[0]:].any()
            else:
                torch.testing.assert_close(batch[key][i], value)


def test_per_dimension_action_masks_keep_their_semantics(collator):
    samples = [_sample(1), _sample(4)]
    for sample in samples:
        mask = np.ones_like(sample['actions'], dtype=bool)
        mask[:, 29:] = False
        sample['action_masks'] = mask
    batch = collator(samples)
    assert batch['action_masks'].shape == (2, 96, 32)
    assert batch['action_masks'][0, :24, :29].all()
    assert not batch['action_masks'][0, 24:].any()
    assert not batch['action_masks'][:, :, 29:].any()


@pytest.mark.parametrize('key',
                         ['states', 'actions', 'action_masks', 'frame_masks'])
def test_misaligned_modalities_fail_before_model_forward(collator, key):
    sample = _sample(2)
    sample[key] = sample[key][:-1]
    with pytest.raises(ValueError, match=key):
        collator([sample])
