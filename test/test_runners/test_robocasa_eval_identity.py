# Copyright 2026 Limx Dynamics
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from fluxvla.engines.runners.robocasa_eval_runner import RobocasaEvalRunner


@pytest.mark.parametrize('startup_seed', ['7', '23', None])
def test_hash_validation_checks_interpreter_startup(startup_seed):
    env = os.environ.copy()
    env.pop('PYTHONHASHSEED', None)
    if startup_seed is not None:
        env['PYTHONHASHSEED'] = startup_seed
    code = '''
import json
from fluxvla.engines.runners.robocasa_eval_runner import RobocasaEvalRunner
from fluxvla.engines.utils.torch_utils import set_seed_everywhere
set_seed_everywhere(7)
try:
    info = RobocasaEvalRunner._python_hash_info()
    print(json.dumps(dict(valid=True, info=info)))
except RuntimeError as error:
    print(json.dumps(dict(valid=False, error=str(error))))
'''
    result = subprocess.run([sys.executable, '-c', code],
                            env=env,
                            text=True,
                            capture_output=True,
                            check=True)
    actual = json.loads(result.stdout.splitlines()[-1])
    assert actual['valid'] == (startup_seed is not None)
    if startup_seed is not None:
        assert actual['info']['seed'] == startup_seed
    else:
        assert 'before starting Python' in actual['error']


@pytest.mark.parametrize('supplied_seed, expected', [(None, '7'),
                                                     ('23', '23')])
def test_eval_launcher_sets_hash_seed_for_child(tmp_path, supplied_seed,
                                                expected):
    executable = tmp_path / 'torchrun'
    executable.write_text('#!/bin/sh\nprintf "%s" "$PYTHONHASHSEED" '
                          '> "$SEED_CAPTURE"\n')
    executable.chmod(0o755)
    env = os.environ.copy()
    env.pop('PYTHONHASHSEED', None)
    if supplied_seed is not None:
        env['PYTHONHASHSEED'] = supplied_seed
    capture = tmp_path / 'seed.txt'
    env.update(PATH=f'{tmp_path}:{env["PATH"]}', SEED_CAPTURE=str(capture))
    script = Path(__file__).resolve().parents[2] / 'scripts/eval.sh'
    subprocess.run(['bash', str(script), 'unused-config.py'],
                   env=env,
                   check=True)
    assert capture.read_text() == expected


def test_pinned_checkpoint_survives_latest_link_change(tmp_path):
    earlier = tmp_path / 'step-10000.safetensors'
    later = tmp_path / 'step-15000.safetensors'
    earlier.write_text('earlier weights')
    later.write_text('later weights')
    latest = tmp_path / 'latest-checkpoint.safetensors'
    latest.symlink_to(earlier)
    selected = RobocasaEvalRunner._pin_checkpoint(str(latest))
    latest.unlink()
    latest.symlink_to(later)
    assert Path(selected).read_text() == 'earlier weights'
    assert RobocasaEvalRunner._build_ckpt_tag(selected) == 'step-10000'


def _identity_worker(rank, rendezvous, root, tamper_hash):
    dist.init_process_group(
        'gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        # Simulate two nodes observing different latest-link targets.
        selected = RobocasaEvalRunner._pin_checkpoint(
            str(Path(root) / f'step-{rank}.pt'))
        run_id = RobocasaEvalRunner._broadcast_from_rank_zero(f'run-{rank}')
        if tamper_hash and rank == 1:
            os.environ['PYTHONHASHSEED'] = '23'
        hash_valid = True
        try:
            RobocasaEvalRunner._python_hash_info()
        except RuntimeError:
            hash_valid = False
        Path(root, f'rank{rank}.json').write_text(
            json.dumps(
                dict(
                    checkpoint=selected, run_id=run_id,
                    hash_valid=hash_valid)))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('tamper_hash', [False, True])
def test_ranks_share_checkpoint_and_run_id(tmp_path, tamper_hash):
    for rank in range(2):
        (tmp_path / f'step-{rank}.pt').write_text(str(rank))
    # Spawn from a script: Python's stdlib also has a package named test,
    # which makes pytest's imported test module ambiguous to multiprocessing.
    subprocess.run(
        [sys.executable,
         str(Path(__file__)),
         str(tmp_path),
         str(tamper_hash)],
        env={
            **os.environ, 'PYTHONHASHSEED': '7'
        },
        check=True)
    outputs = [
        json.loads((tmp_path / f'rank{rank}.json').read_text())
        for rank in range(2)
    ]
    assert outputs[0] == outputs[1]
    assert outputs[0] == dict(
        checkpoint=str(tmp_path / 'step-0.pt'),
        run_id='run-0',
        hash_valid=not tamper_hash)


def test_initial_observation_fingerprint_detects_scene_changes():
    obs = {
        'video.ego_view_bg_crop_pad_res256_freq20':
        np.zeros((4, 4, 3), dtype=np.uint8),
        'state.left_arm':
        np.zeros(7, dtype=np.float32)
    }
    baseline = RobocasaEvalRunner._observation_fingerprint(obs)
    reordered = dict(reversed(list(obs.items())))
    assert RobocasaEvalRunner._observation_fingerprint(reordered) == baseline
    obs['state.left_arm'][0] = 1
    changed = RobocasaEvalRunner._observation_fingerprint(obs)
    assert changed['state_sha256'] != baseline['state_sha256']
    assert changed['image_sha256'] == baseline['image_sha256']
    obs['video.ego_view_bg_crop_pad_res256_freq20'][0, 0, 0] = 1
    assert RobocasaEvalRunner._observation_fingerprint(
        obs)['image_sha256'] != baseline['image_sha256']


if __name__ == '__main__':
    mp.spawn(
        _identity_worker,
        args=(f'file://{sys.argv[1]}/rendezvous', sys.argv[1],
              sys.argv[2] == 'True'),
        nprocs=2,
        join=True)
