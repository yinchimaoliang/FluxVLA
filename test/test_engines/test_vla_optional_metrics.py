# Copyright 2026 Limx Dynamics
"""Missing policy measurements must not appear as successful zero errors."""

import pytest
import torch

from fluxvla.engines.metrics.vla_metric import VLAMetric


@pytest.fixture
def metric(monkeypatch):
    tracker = VLAMetric((), 'test', '', {})
    published = {}
    monkeypatch.setattr(tracker, 'log',
                        lambda step, metrics: published.update(metrics))
    return tracker, published


def test_flow_matching_does_not_report_unmeasured_token_accuracy(metric):
    tracker, published = metric
    tracker.commit(loss=torch.tensor(0.3), action_loss=torch.tensor(0.1))
    tracker.push()

    assert published['VLA Train/Loss'] == pytest.approx(0.3)
    assert published['VLA Train/Action Loss'] == pytest.approx(0.1)
    assert 'VLA Train/L1 Loss' not in published
    assert 'VLA Train/Action Token Accuracy' not in published


def test_measured_zero_accuracy_and_mean_l1_are_preserved(metric):
    tracker, published = metric
    tracker.commit(
        loss=torch.tensor(0.3),
        action_accuracy=torch.tensor(0.),
        l1_loss=torch.tensor(0.2))
    tracker.push()

    assert published['VLA Train/Action Token Accuracy'] == 0.0
    assert published['VLA Train/L1 Loss'] == pytest.approx(0.2)


def test_dataset_can_supply_l1_without_token_accuracy(metric):
    tracker, published = metric
    tracker.commit(loss=torch.tensor(0.3))
    tracker.commit_for_dataset('robocasa', l1_loss=torch.tensor(0.4))
    tracker.push()

    assert published['robocasa/L1 Loss'] == pytest.approx(0.4)
    assert 'robocasa/Action Token Accuracy' not in published
