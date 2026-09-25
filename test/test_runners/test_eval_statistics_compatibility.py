# Copyright 2026 Limx Dynamics

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mmengine import Config

import fluxvla.engines as engines
from fluxvla.datasets.parquet_dataset import PrivateInferenceDataset
from fluxvla.engines.runners import libero_eval_runner
from fluxvla.engines.runners.libero_eval_runner import LiberoEvalRunner


@pytest.fixture
def make_runner(monkeypatch):
    model = SimpleNamespace(eval=lambda: model)
    monkeypatch.setattr(engines, 'build_vla_from_cfg', lambda _: model)
    monkeypatch.setattr(engines, 'build_dataset_from_cfg', lambda cfg: cfg)
    monkeypatch.setattr(engines, 'build_transform_from_cfg', lambda cfg: cfg)
    monkeypatch.setattr(
        libero_eval_runner, 'overwatch',
        SimpleNamespace(local_rank=lambda: 0, distributed_state=None))

    def build(**kwargs):
        return LiberoEvalRunner(
            cfg=Config(dict(model={})),
            seed=7,
            ckpt_path=None,
            model_family='test',
            task_suite_name='libero_10',
            dataset={},
            denormalize_action={},
            **kwargs)

    return build


@pytest.mark.parametrize('names', [('norm_stats_path', ),
                                   ('dataset_stats_path', ),
                                   ('norm_stats_path', 'dataset_stats_path')])
def test_libero_statistics_path_aliases(make_runner, tmp_path, names):
    path = tmp_path / 'statistics.json'
    stats = {'libero_10_no_noops': {'action': {'mean': [1.0]}}}
    path.write_text(json.dumps(stats))
    runner = make_runner(**{name: str(path) for name in names})
    assert runner.dataset['norm_stats'] == str(path)
    assert runner.denormalize_action['norm_stats'] == str(path)
    assert runner.vla.norm_stats == stats


def test_libero_conflicting_statistics_paths_fail(make_runner):
    with pytest.raises(ValueError, match='same statistics file'):
        make_runner(norm_stats_path='one.json', dataset_stats_path='two.json')


def test_libero_statistics_remain_optional_for_checkpoint_free_models(
        make_runner):
    runner = make_runner(requires_dataset_stats=False)
    assert runner.dataset['norm_stats'] is None
    with pytest.raises(AssertionError, match='required'):
        make_runner()


def test_libero_legacy_positional_arguments_keep_their_meaning():
    # The first optional positional value after the statistics path was the
    # execution chunk size; new main-only options must not shift it.
    bound = inspect.signature(LiberoEvalRunner).bind({}, 7, None, 'pi05',
                                                     'libero_10', {}, {}, None,
                                                     'stats.json', 10)
    assert bound.arguments['norm_stats_path'] == 'stats.json'
    assert bound.arguments['eval_chunk_size'] == 10


@pytest.mark.parametrize('tensor_outputs', [False, True])
@pytest.mark.parametrize('inject_model_path', [False, True])
def test_private_inference_preserves_statistics_and_raw_state(
        monkeypatch, tensor_outputs, inject_model_path):
    configs = []
    observed_stats = []

    def build_transform(cfg):
        configs.append(cfg)

        def transform(inputs):
            observed_stats.append(inputs['stats'])
            result = dict(
                images=np.stack(inputs['images']),
                states=inputs['states'] * 0.5,
                lang_tokens=np.array([2, 3]),
                lang_masks=np.array([True, True]))
            if tensor_outputs:
                result = {
                    key: torch.from_numpy(value)
                    for key, value in result.items()
                }
            return result

        return transform

    monkeypatch.setattr(engines, 'build_transform_from_cfg', build_transform)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    statistics = {'custom': {'marker': 17}, 'private': {'marker': 99}}
    # Preserve the pre-rebase fourth positional argument (camera keys).
    dataset = PrivateInferenceDataset(
        statistics, [{}],
        'checkpoint', ['camera'],
        statistic_name='custom',
        inject_model_path=inject_model_path)
    raw_state = np.arange(14, dtype=np.float32)
    original_state = raw_state.copy()
    batch = dataset(
        dict(qpos=raw_state, camera=np.zeros((4, 5, 3), dtype=np.uint8)))
    assert observed_stats == [statistics['custom']]
    assert ('model_path' in configs[0]) == inject_model_path
    assert batch['images'].shape == (1, 1, 3, 4, 5)
    torch.testing.assert_close(batch['states'],
                               torch.from_numpy(original_state[None] * 0.5))
    raw_state[:] = -1
    batch['states'].zero_()
    np.testing.assert_array_equal(dataset.last_raw_state, original_state)
