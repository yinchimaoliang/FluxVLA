# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch

from fluxvla.engines.utils.torch_utils import configure_deterministic_training


def test_configure_deterministic_training(monkeypatch):
    calls = []
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    monkeypatch.setattr(torch, 'use_deterministic_algorithms', calls.append)
    monkeypatch.setattr(torch.backends.cudnn, 'deterministic', False)
    monkeypatch.setattr(torch.backends.cudnn, 'benchmark', True)

    configure_deterministic_training(enabled=True)

    assert calls == [True]
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
