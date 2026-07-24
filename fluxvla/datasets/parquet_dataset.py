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
import json
import os
import shlex
from collections import deque
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import concatenate_datasets, load_dataset
from fluxvla.engines import DATASETS, build_transform_from_cfg


@DATASETS.register_module()
class ParquetDataset(Dataset):
    VERSION_FILE = 'meta/fluxvla_dataset_version.json'
    HF_REPO_ID = 'limxdynamics/FluxVLAData'
    HF_REVISION = 'main'

    def __init__(self,
                 data_root_path: Union[str, List[str]],
                 transforms: List[Dict],
                 action_window_size: int = 9,
                 action_key: str = 'observation.state',
                 action_valid_key: Optional[str] = None,
                 action_mask_key: Optional[str] = None,
                 action_indices: Optional[List[int]] = None,
                 use_delta: bool = False,
                 statistic_name: str = 'private',
                 window_start_idx: int = 1,
                 frame_window_size: int = 1,
                 frame_sample_stride: int = 1,
                 train_episode_fraction: float = 1.0,
                 repeat_to_full_length: bool = False,
                 expose_index: bool = False,
                 expected_dataset_version: Optional[str] = None,
                 expected_schema_id: Optional[str] = None,
                 expected_schema_version: Optional[str] = None,
                 expose_subtask_metadata: bool = False,
                 enforce_action_subtask_consistency: bool = False) -> None:
        """Initialize the Parquet dataset.

        Args:
            data_root_path (Union[str, List[str]]): Path(s) to the root
                directory(ies). The metadata will be loaded from
                `data_root_path/meta` and data from `data_root_path/data`.
                If a list is provided, multiple datasets will be loaded and
                concatenated.
            transforms (List[Dict]): List of transformation configurations.
            batch_transform (Union[dict, ConfigDict, Config]):
                Configuration for the batch transformation.
            episodes (list[int]): List of episode indices to include
                in the dataset.
            local_files_only (bool): Whether to use local files only.
            action_horizon (int): The number of time steps for the
                action sequence.
            video_backend (str, optional): Backend for
                video processing.
                Defaults to None.
            action_key (str): Key for the action data.
            action_valid_key (str, optional): Per-frame boolean key declaring
                whether the action value is genuine supervision. Invalid
                actions remain in the returned tensor as placeholders but
                receive a zero entry in ``action_masks``. Defaults to None,
                which treats every temporally valid action as supervised.
            action_mask_key (str, optional): Per-frame vector key declaring
                which action dimensions are genuine supervision. When set,
                ``action_masks`` has shape ``[horizon, action_dim]`` and is
                combined with temporal and optional scalar validity. This is
                useful for unified cross-embodiment action vectors. Defaults
                to None.
            action_indices (list[int], optional): Ordered dimensions selected
                from both ``action_key`` and ``action_mask_key`` before action
                windows are returned. Defaults to None (all dimensions).
            use_delta (bool): Whether to use delta actions.
                Defaults to False.
            statistic_name (str): Name for the statistics collection.
                Defaults to 'private'.
            window_start_idx (int): Start index for the action window.
                Defaults to 1.
            frame_window_size (int): Number of video frames to expose via
                ``frame_timestamps``. Defaults to 1 (single current frame).
            frame_sample_stride (int): Stride (in dataset rows) between the
                sampled video frames. Defaults to 1 (consecutive frames).
                Increase this when the sampled frames should span a longer
                temporal window:
                ``(frame_window_size - 1) * frame_sample_stride`` rows.
            train_episode_fraction (float): Fraction of episodes to sample
                from each data root, preserving original episode order.
                Defaults to 1.0.
            repeat_to_full_length (bool): If True, repeat the selected
                episode subset so `__len__` remains the full dataset length.
                This keeps epoch length based on full statistics while
                sampling only the selected train episode fraction.
            expose_index (bool): Whether to add the concatenated dataset index
                to each raw sample before transforms. This is useful for
                offline sample-weight transforms such as SARM RA-BC.
                Defaults to False.
            expected_dataset_version (str, optional): Expected FluxVLA dataset
                content version. If omitted, no version check is performed so
                existing local datasets remain usable.
            expected_schema_id (str, optional): Expected data-rule schema id
                from ``meta/info.json``.
            expected_schema_version (str, optional): Expected data-rule schema
                version from ``meta/info.json``.
            expose_subtask_metadata (bool): Whether to expose the current
                subtask definition to transforms. The temporary metadata key
                is intended for ``LoadSubtask`` and is not a Parquet column.
                Defaults to False.
            enforce_action_subtask_consistency (bool): Stop and invalidate the
                remainder of an action window when its frame-level
                ``subtask_index`` changes. Defaults to False for compatibility
                with legacy configs.
        """
        super().__init__()
        if not 0 < train_episode_fraction <= 1:
            raise ValueError('train_episode_fraction must be in (0, 1].')
        self.action_window_size = action_window_size
        if isinstance(data_root_path, str):
            data_root_path = [data_root_path]
        self.data_root_path = data_root_path

        meta_root = [os.path.join(path, 'meta') for path in data_root_path]
        data_root = [os.path.join(path, 'data') for path in data_root_path]

        # Merge multiple meta_root
        all_stats = []
        all_tasks = []
        all_subtasks = []
        all_episodes = []
        task_definitions_by_dataset = []
        subtask_definitions_by_dataset = []
        episode_metadata_by_dataset = []
        separate_subtasks_by_dataset = []
        info_list = []

        for dataset_root, root in zip(data_root_path, meta_root):
            info_path = os.path.join(root, 'info.json')
            assert os.path.exists(info_path), \
                f'Metadata file not found at {info_path}'
            with open(os.path.join(root, 'info.json'), 'rb') as f:
                dataset_info = json.load(f)
            info_list.append(dataset_info)
            self._verify_schema_contract(
                dataset_root=dataset_root,
                dataset_info=dataset_info,
                expected_schema_id=expected_schema_id,
                expected_schema_version=expected_schema_version)
            if expected_dataset_version is not None:
                self._verify_dataset_version(
                    dataset_root=dataset_root,
                    expected_dataset_version=expected_dataset_version)

            stats_path = os.path.join(root, 'episodes_stats.jsonl')
            assert os.path.exists(stats_path), \
                f'Statistics file not found at {stats_path}'
            with open(
                    os.path.join(root, 'episodes_stats.jsonl'),
                    'r',
                    encoding='utf-8') as f:
                all_stats.extend([json.loads(line) for line in f])

            tasks_path = os.path.join(root, 'tasks.jsonl')
            assert os.path.exists(tasks_path), \
                f'Tasks file not found at {tasks_path}'
            with open(tasks_path, 'r', encoding='utf-8') as f:
                task_records = [json.loads(line) for line in f]
            all_tasks.append(task_records)
            task_definitions = {
                int(record.get('task_index', task_index)): record
                for task_index, record in enumerate(task_records)
            }
            if len(task_definitions) != len(task_records):
                raise ValueError(f'Duplicate task_index in {tasks_path}.')
            task_definitions_by_dataset.append(task_definitions)

            subtasks_path = os.path.join(root, 'subtasks.jsonl')
            declares_separate_subtasks = ('subtask_index'
                                          in dataset_info.get('features', {}))
            has_subtasks_file = os.path.exists(subtasks_path)
            if declares_separate_subtasks != has_subtasks_file:
                raise ValueError(
                    f'Inconsistent subtask schema in {dataset_root!r}: '
                    f'info.features declares subtask_index='
                    f'{declares_separate_subtasks}, but '
                    f'meta/subtasks.jsonl exists={has_subtasks_file}.')
            has_separate_subtasks = declares_separate_subtasks
            if has_separate_subtasks:
                with open(subtasks_path, 'r', encoding='utf-8') as f:
                    subtask_records = [json.loads(line) for line in f]
                subtask_definitions = {
                    int(record.get('subtask_index', subtask_index)): record
                    for subtask_index, record in enumerate(subtask_records)
                }
                if len(subtask_definitions) != len(subtask_records):
                    raise ValueError(
                        f'Duplicate subtask_index in {subtasks_path}.')
            else:
                # Legacy datasets used tasks.jsonl and per-frame task_index
                # for what is now the subtask namespace.
                subtask_records = task_records
                subtask_definitions = {
                    int(record.get('task_index', subtask_index)): record
                    for subtask_index, record in enumerate(subtask_records)
                }
            all_subtasks.append(subtask_records)
            subtask_definitions_by_dataset.append(subtask_definitions)
            separate_subtasks_by_dataset.append(has_separate_subtasks)

            episodes_path = os.path.join(root, 'episodes.jsonl')
            assert os.path.exists(episodes_path), \
                f'Episodes file not found at {episodes_path}'
            with open(episodes_path, 'r', encoding='utf-8') as f:
                episode_records = [json.loads(line) for line in f]
            all_episodes.extend(episode_records)
            episode_metadata = {
                int(record.get('episode_index', episode_index)): record
                for episode_index, record in enumerate(episode_records)
            }
            if len(episode_metadata) != len(episode_records):
                raise ValueError(
                    f'Duplicate episode_index in {episodes_path}.')
            episode_metadata_by_dataset.append(episode_metadata)

        self.info = info_list
        self.stats = all_stats
        self.tasks = all_tasks
        self.subtasks = all_subtasks
        self.episodes = all_episodes
        self.task_definitions_by_dataset = task_definitions_by_dataset
        self.subtask_definitions_by_dataset = \
            subtask_definitions_by_dataset
        self.episode_metadata_by_dataset = episode_metadata_by_dataset
        self.separate_subtasks_by_dataset = separate_subtasks_by_dataset
        self.expose_subtask_metadata = expose_subtask_metadata
        self.enforce_action_subtask_consistency = \
            enforce_action_subtask_consistency
        # Summarize all data_root
        datasets = []
        dataset_sizes = []  # Record the size of each dataset
        for dataset_idx, root in enumerate(data_root):
            hf_dataset = load_dataset('parquet', data_dir=root, split='train')
            row_index_key = ('subtask_index'
                             if separate_subtasks_by_dataset[dataset_idx] else
                             'task_index')
            if row_index_key not in hf_dataset.column_names:
                raise ValueError(
                    f'Parquet data in {root!r} has no {row_index_key!r} '
                    'declared by its metadata schema.')
            if (separate_subtasks_by_dataset[dataset_idx]
                    and 'task_index' in hf_dataset.column_names):
                raise ValueError(
                    f'Parquet data in {root!r} redundantly stores task_index; '
                    'episode-level task_index belongs in episodes.jsonl.')
            dataset_sizes.append(len(hf_dataset))
            datasets.append(hf_dataset)
        hf_dataset = concatenate_datasets(datasets)
        # Compute cumulative sizes for fast index lookup
        self.dataset_cumulative_sizes = np.cumsum([0] + dataset_sizes)
        self.dataset = hf_dataset
        self.full_length = len(self.dataset)
        self.sample_indices = self._build_sample_indices(
            train_episode_fraction)
        self.effective_length = (
            self.full_length
            if repeat_to_full_length else len(self.sample_indices))
        self.transforms = list()
        self.action_key = action_key
        self.action_valid_key = action_valid_key
        self.action_mask_key = action_mask_key
        self.action_indices = (None if action_indices is None else np.asarray(
            action_indices, dtype=np.int64))
        if self.action_indices is not None:
            if (self.action_indices.ndim != 1 or self.action_indices.size == 0
                    or np.any(self.action_indices < 0) or np.unique(
                        self.action_indices).size != self.action_indices.size):
                raise ValueError(
                    'action_indices must be a non-empty list of unique, '
                    'non-negative integers.')
        self.use_delta = use_delta
        self.statistic_name = statistic_name
        self.window_start_idx = window_start_idx
        self.frame_window_size = frame_window_size
        self.frame_sample_stride = frame_sample_stride
        self.expose_index = expose_index
        for transform in transforms:
            self.transforms.append(build_transform_from_cfg(transform))

    @staticmethod
    def _read_dataset_version(path: str) -> Optional[str]:
        with open(path, 'r', encoding='utf-8') as f:
            raw_version = f.read().strip()
        if not raw_version:
            return None

        try:
            version_data = json.loads(raw_version)
        except json.JSONDecodeError:
            return raw_version

        if isinstance(version_data, dict):
            version = version_data.get('fluxvla_dataset_version',
                                       version_data.get('version'))
            return str(version) if version is not None else None
        if isinstance(version_data, str):
            return version_data
        return str(version_data)

    @classmethod
    def _dataset_refresh_command(cls, dataset_root: str) -> str:
        normalized_root = dataset_root.rstrip(os.sep)
        local_dir = os.path.dirname(normalized_root) or '.'
        remote_dir = os.path.basename(normalized_root)
        return (f'rm -rf {shlex.quote(dataset_root)}\n'
                f'huggingface-cli download {shlex.quote(cls.HF_REPO_ID)} \\\n'
                '  --repo-type dataset \\\n'
                f'  --revision {shlex.quote(cls.HF_REVISION)} \\\n'
                f'  --include {shlex.quote(remote_dir + "/*")} \\\n'
                f'  --local-dir {shlex.quote(local_dir)}')

    @staticmethod
    def _verify_schema_contract(
            dataset_root: str, dataset_info: Dict[str, Any],
            expected_schema_id: Optional[str],
            expected_schema_version: Optional[str]) -> None:
        actual_id = dataset_info.get('schema_id')
        actual_version = dataset_info.get('schema_version')
        if expected_schema_id is not None and actual_id != expected_schema_id:
            raise RuntimeError(
                f'Dataset schema id mismatch for {dataset_root}. Expected '
                f'{expected_schema_id}, but found {actual_id or "missing"} in '
                'meta/info.json.')
        if (expected_schema_version is not None
                and actual_version != expected_schema_version):
            raise RuntimeError(
                f'Dataset schema version mismatch for {dataset_root}. '
                f'Expected {expected_schema_version}, but found '
                f'{actual_version or "missing"} in meta/info.json.')

    def _verify_dataset_version(self, dataset_root: str,
                                expected_dataset_version: str) -> None:
        version_path = os.path.join(dataset_root, self.VERSION_FILE)
        refresh_command = self._dataset_refresh_command(dataset_root)

        if not os.path.exists(version_path):
            raise RuntimeError(
                f'Dataset version file not found at {version_path}. '
                f'Expected FluxVLA dataset version '
                f'{expected_dataset_version}.\n\n'
                f'Please refresh the dataset with:\n\n{refresh_command}')

        dataset_version = self._read_dataset_version(version_path)
        if dataset_version != expected_dataset_version:
            raise RuntimeError(
                f'Dataset version mismatch for {dataset_root}. '
                f'Expected FluxVLA dataset version '
                f'{expected_dataset_version}, but found '
                f'{dataset_version or "missing"} in {version_path}.\n\n'
                f'Please refresh the dataset with:\n\n{refresh_command}')

    def _build_sample_indices(self, episode_fraction: float) -> np.ndarray:
        if episode_fraction == 1.0:
            return np.arange(self.full_length, dtype=np.int64)

        episode_indices = list(self.dataset['episode_index'])
        sample_indices = []
        for start, end in zip(self.dataset_cumulative_sizes[:-1],
                              self.dataset_cumulative_sizes[1:]):
            start, end = int(start), int(end)
            local_episode_indices = episode_indices[start:end]
            ordered_episodes = list(dict.fromkeys(local_episode_indices))
            keep_count = int(len(ordered_episodes) * episode_fraction)
            keep_count = max(1, min(keep_count, len(ordered_episodes)))
            keep_episodes = set(ordered_episodes[:keep_count])
            sample_indices.extend(
                start + offset
                for offset, episode in enumerate(local_episode_indices)
                if episode in keep_episodes)

        if not sample_indices:
            raise ValueError('No samples left after applying episode split.')
        return np.asarray(sample_indices, dtype=np.int64)

    def _resolve_index(self, index: int) -> int:
        sample_index = index % len(self.sample_indices)
        return int(self.sample_indices[sample_index])

    def _rand_another(self):
        """Randomly select another index from the dataset."""
        return int(self.sample_indices[np.random.randint(
            0, len(self.sample_indices))])

    def _get_dataset_index(self, index: int) -> int:
        """Get which dataset in data_root list the index belongs to.

        Args:
            index (int): The index in the concatenated dataset.

        Returns:
            int: The index of the dataset in data_root list (0-based).
        """
        if self.dataset_cumulative_sizes is None:
            return 0
        # Use binary search to find the index of the dataset in data_root list
        dataset_idx = np.searchsorted(
            self.dataset_cumulative_sizes, index, side='right') - 1
        return dataset_idx

    def _get_episode_metadata(self, dataset_idx: int,
                              index: int) -> Dict[str, Any]:
        episode_index = int(self.dataset[index]['episode_index'])
        try:
            return self.episode_metadata_by_dataset[dataset_idx][episode_index]
        except KeyError as exc:
            raise KeyError(
                f'No episode metadata for episode_index={episode_index} in '
                f'dataset {self.data_root_path[dataset_idx]!r}.') from exc

    def _get_task_index(self, dataset_idx: int, index: int) -> int:
        if not self.separate_subtasks_by_dataset[dataset_idx]:
            return int(self.dataset[index]['task_index'])
        episode_metadata = self._get_episode_metadata(dataset_idx, index)
        if 'task_index' in episode_metadata:
            return int(episode_metadata['task_index'])
        raise KeyError(f'Episode metadata has no task_index in dataset '
                       f'{self.data_root_path[dataset_idx]!r}.')

    def _get_task_name(self, dataset_idx: int, index: int) -> str:
        task_index = self._get_task_index(dataset_idx, index)
        task_definition = self.task_definitions_by_dataset[dataset_idx].get(
            task_index)
        if task_definition is None:
            if not self.separate_subtasks_by_dataset[dataset_idx]:
                return 'empty'
            raise KeyError(
                f'No task definition for task_index={task_index} in dataset '
                f'{self.data_root_path[dataset_idx]!r}.')
        return task_definition.get('task', 'empty')

    def _get_subtask_index(self, dataset_idx: int, index: int) -> int:
        key = ('subtask_index'
               if self.separate_subtasks_by_dataset[dataset_idx] else
               'task_index')
        try:
            return int(self.dataset[index][key])
        except KeyError as exc:
            raise KeyError(f'Parquet row has no {key!r} in dataset '
                           f'{self.data_root_path[dataset_idx]!r}.') from exc

    def _get_subtask_definition(self, dataset_idx: int,
                                index: int) -> Dict[str, Any]:
        subtask_index = self._get_subtask_index(dataset_idx, index)
        definition = self.subtask_definitions_by_dataset[dataset_idx].get(
            subtask_index)
        if definition is None:
            if not self.separate_subtasks_by_dataset[dataset_idx]:
                return {'task_index': subtask_index, 'task': 'empty'}
            raise KeyError(
                f'No subtask definition for subtask_index={subtask_index} in '
                f'dataset {self.data_root_path[dataset_idx]!r}.')
        return definition

    def _get_subtask_name(self, dataset_idx: int, index: int) -> str:
        definition = self._get_subtask_definition(dataset_idx, index)
        text_key = ('subtask' if self.separate_subtasks_by_dataset[dataset_idx]
                    else 'task')
        return definition.get(text_key, 'empty')

    def _invalid_start_index(self, index: int, dataset_idx: int,
                             data: Dict[str, Any]) -> bool:
        if self._get_subtask_name(dataset_idx, index) in ('empty', 'static'):
            return True

        first_action_index = index + self.window_start_idx
        if first_action_index == index:
            return False
        if not self._same_episode_and_dataset(first_action_index, dataset_idx,
                                              data):
            return True
        if (self.enforce_action_subtask_consistency
                and self._get_subtask_index(dataset_idx, first_action_index) !=
                self._get_subtask_index(dataset_idx, index)):
            return True
        return self._get_subtask_name(dataset_idx,
                                      first_action_index) in ('empty',
                                                              'static')

    def _same_episode_and_dataset(self, index: int, dataset_idx: int,
                                  data: Dict[str, Any]) -> bool:
        return (0 <= index < len(self.dataset) and data['episode_index']
                == self.dataset[index]['episode_index']
                and self._get_dataset_index(index) == dataset_idx)

    def __getitem__(self, index, dataset_statistics):
        index = self._resolve_index(index)
        data = self.dataset[index]
        # Determine which dataset the data belongs to
        dataset_idx = self._get_dataset_index(index)
        retry_count = 0
        while self._invalid_start_index(index, dataset_idx, data):
            retry_count += 1
            if retry_count >= min(100, len(self.sample_indices)):
                for candidate in self.sample_indices:
                    candidate = int(candidate)
                    candidate_data = self.dataset[candidate]
                    candidate_dataset_idx = self._get_dataset_index(candidate)
                    if not self._invalid_start_index(
                            candidate, candidate_dataset_idx, candidate_data):
                        index = candidate
                        data = candidate_data
                        dataset_idx = candidate_dataset_idx
                        break
                else:
                    raise RuntimeError(
                        'No actionable non-empty/non-static sample exists in '
                        'the selected Parquet dataset episodes.')
                break
            index = self._rand_another()
            data = self.dataset[index]
            # Recalculate dataset_idx
            dataset_idx = self._get_dataset_index(index)
        actions = list()
        action_masks = list()

        def select_action_dimensions(value, key):
            values = np.asarray(value)
            if self.action_indices is None:
                return values
            if values.ndim != 1 or self.action_indices.max(
            ) >= values.shape[0]:
                raise ValueError(
                    f'Cannot select action_indices from {key!r} with shape '
                    f'{values.shape}.')
            return values[self.action_indices]

        action_template_full = np.asarray(data[self.action_key])
        if self.action_mask_key is not None:
            mask_template_full = np.asarray(
                data[self.action_mask_key], dtype=np.float32)
            if mask_template_full.shape != action_template_full.shape:
                raise ValueError(
                    f'Action mask {self.action_mask_key!r} has shape '
                    f'{mask_template_full.shape}, expected '
                    f'{action_template_full.shape} '
                    f'to match {self.action_key!r}.')
            mask_template = select_action_dimensions(mask_template_full,
                                                     self.action_mask_key)
            invalid_action_mask = np.zeros_like(
                mask_template, dtype=np.float32)
        else:
            invalid_action_mask = 0
        current_subtask_index = self._get_subtask_index(dataset_idx, index)

        def pad_action_window():
            if actions:
                padding_action = actions[-1]
            else:
                padding_action = select_action_dimensions(
                    data[self.action_key], self.action_key)
            while len(actions) < self.action_window_size:
                actions.append(padding_action)
                action_masks.append(invalid_action_mask)

        window_idx = self.window_start_idx
        while len(actions) < self.action_window_size:
            action_index = index + window_idx
            valid_window_index = self._same_episode_and_dataset(
                action_index, dataset_idx, data)
            action_subtask_index = (
                self._get_subtask_index(dataset_idx, action_index)
                if valid_window_index else None)
            action_task = (
                self._get_subtask_name(dataset_idx, action_index)
                if valid_window_index else None)
            if (valid_window_index and self.enforce_action_subtask_consistency
                    and action_subtask_index != current_subtask_index):
                pad_action_window()
                break
            if valid_window_index and action_task not in ('empty', 'static'):
                current_action = select_action_dimensions(
                    self.dataset[action_index][self.action_key],
                    self.action_key)
                delta_previous_valid = True
                if self.use_delta:
                    previous_index = action_index - 1
                    delta_previous_valid = self._same_episode_and_dataset(
                        previous_index, dataset_idx, data)
                    if (delta_previous_valid
                            and self.enforce_action_subtask_consistency):
                        delta_previous_valid = (
                            self._get_subtask_index(
                                dataset_idx,
                                previous_index) == action_subtask_index)
                    if delta_previous_valid:
                        previous_action = select_action_dimensions(
                            self.dataset[previous_index][self.action_key],
                            self.action_key)
                        actions.append(current_action - previous_action)
                    else:
                        actions.append(np.zeros_like(current_action))
                else:
                    actions.append(current_action)
                scalar_valid = (
                    self.action_valid_key is None
                    or bool(self.dataset[action_index][self.action_valid_key]))
                scalar_valid = scalar_valid and delta_previous_valid
                if self.action_mask_key is None:
                    action_masks.append(int(scalar_valid))
                else:
                    dimension_mask = np.asarray(
                        self.dataset[action_index][self.action_mask_key],
                        dtype=np.float32)
                    dimension_mask = select_action_dimensions(
                        dimension_mask, self.action_mask_key)
                    if self.use_delta and delta_previous_valid:
                        previous_mask = np.asarray(
                            self.dataset[previous_index][self.action_mask_key],
                            dtype=np.float32)
                        previous_mask = select_action_dimensions(
                            previous_mask, self.action_mask_key)
                        dimension_mask = dimension_mask * previous_mask
                    if not scalar_valid:
                        dimension_mask = np.zeros_like(dimension_mask)
                    action_masks.append(dimension_mask)
            elif action_task == 'empty':
                pad_action_window()
                break
            elif action_task == 'static':
                window_idx += 1
                continue
            else:
                pad_action_window()
                break
            window_idx += 1
        # Collect forward-looking frame timestamps for video models
        if self.frame_window_size > 1:
            frame_timestamps = [data['timestamp']]
            frame_masks = [1]
            for fi in range(1, self.frame_window_size):
                future_idx = index + fi * self.frame_sample_stride
                if (future_idx < len(self.dataset)
                        and self.dataset[future_idx]['episode_index']
                        == data['episode_index'] and
                        self._get_dataset_index(future_idx) == dataset_idx):
                    frame_timestamps.append(
                        self.dataset[future_idx]['timestamp'])
                    frame_masks.append(1)
                else:
                    frame_timestamps.append(frame_timestamps[-1])
                    frame_masks.append(0)
            data['frame_timestamps'] = frame_timestamps
            data['frame_masks'] = np.array(frame_masks, dtype=np.float32)

        data['info'] = self.info[dataset_idx]
        data['stats'] = dataset_statistics[self.statistic_name]
        data['actions'] = np.array(actions, dtype=np.float32)
        data['action_masks'] = np.array(action_masks, dtype=np.float32)
        if self.expose_index:
            data['index'] = np.array(index, dtype=np.int64)
        data['task_index'] = self._get_task_index(dataset_idx, index)
        data['subtask_index'] = self._get_subtask_index(dataset_idx, index)
        data['task_description'] = self._get_task_name(dataset_idx, index)
        data['subtask_description'] = self._get_subtask_name(
            dataset_idx, index)
        data['data_root'] = self.data_root_path[dataset_idx]
        if self.expose_subtask_metadata:
            data['_subtask_definition'] = self._get_subtask_definition(
                dataset_idx, index)
        for transform in self.transforms:
            data = transform(data)

        return data

    def __len__(self):
        return self.effective_length

        # Additional initialization can be added here if needed.


@DATASETS.register_module()
class LiberoParquetEvalDataset:
    """Evaluation dataset pipeline for Libero using Parquet-style transforms.

    This composes Libero eval processing via a list of transforms similar to
    `ParquetDataset`.

    Args:
        norm_stats (str | Dict): Normalization stats dict or path to JSON.
        task_suite_name (str): Name of Libero task suite (for stats keying).
        tokenizer (Dict): Tokenizer config for `build_tokenizer_from_cfg`.
        transforms (List[Dict]): List of transform configs applied in order.
        num_padding_imgs (int): Number of zero image slots appended per step.
        img_buffer_len (int): Number of recent image frames kept for eval.
    """

    def __init__(self,
                 norm_stats: Any,
                 task_suite_name: str,
                 transforms: List[Dict],
                 norm_stats_key: str,
                 num_padding_imgs: int = 0,
                 img_buffer_len: int = 1) -> None:

        # Build image/token transforms (parquet-style sequential list)
        self.transforms = [build_transform_from_cfg(t) for t in transforms]
        self.task_suite_name = task_suite_name
        self.norm_stats_key = norm_stats_key
        self.num_padding_imgs = num_padding_imgs
        assert img_buffer_len >= 1, 'img_buffer_len must be >= 1'
        self.img_buffer_len = img_buffer_len
        self.img_buffer = None
        self.img_mask_buffer = None
        self.img_buffer_updates = 0
        if isinstance(norm_stats, str):
            with open(norm_stats, 'r', encoding='utf-8') as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = norm_stats

    def _reset_img_buffer(self) -> None:
        self.img_buffer = None
        self.img_mask_buffer = None
        self.img_buffer_updates = 0

    def _split_image_frames(self, pixel_values: torch.Tensor) -> List:
        if pixel_values.ndim == 4:
            if pixel_values.shape[0] == 3:
                return [
                    pixel_values[:, i:i + 1].detach().clone()
                    for i in range(pixel_values.shape[1])
                ]
            return [pixel_values.detach().clone()]
        if pixel_values.ndim == 3:
            return [pixel_values.detach().clone()]
        raise ValueError(f'Unsupported image shape: {pixel_values.shape}')

    def _update_img_buffer(self, pixel_values: torch.Tensor,
                           img_masks: List[bool]):
        if self.img_buffer is None:
            self.img_buffer = deque(maxlen=self.img_buffer_len)
            self.img_mask_buffer = deque(maxlen=self.img_buffer_len)

        frames = self._split_image_frames(pixel_values)
        for frame in frames:
            self.img_buffer.append(frame)
            self.img_mask_buffer.append(list(img_masks))

        buffered_frames = list(self.img_buffer)
        buffered_masks = list(self.img_mask_buffer)
        # Match DreamZero causal eval behavior: the first request in a new
        # episode uses a single frame to warm the cache. Later requests pad
        # short histories by repeating the earliest available frame.
        is_first_buffer_update = self.img_buffer_updates == 0
        self.img_buffer_updates += 1
        if (not is_first_buffer_update
                and len(buffered_frames) < self.img_buffer_len):
            pad_len = self.img_buffer_len - len(buffered_frames)
            buffered_frames = [buffered_frames[0]] * pad_len + buffered_frames
            buffered_masks = [buffered_masks[0]] * pad_len + buffered_masks

        if buffered_frames[0].ndim == 4 and buffered_frames[0].shape[0] == 3:
            pixel_values = torch.cat(buffered_frames, dim=1)
        else:
            pixel_values = torch.cat(buffered_frames, dim=0)

        img_masks = [
            mask for frame_masks in buffered_masks for mask in frame_masks
        ]
        return pixel_values, img_masks

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Compose transforms chain (parquet-style) starting from raw inputs
        data: Dict[str, Any] = dict(inputs)
        is_new_episode = bool(data.get('is_new_episode', False))
        if is_new_episode:
            self._reset_img_buffer()
        if self.norm_stats is not None:
            norm_stats = self.norm_stats[self.norm_stats_key]
        else:
            norm_stats = None
        data['norm_stats'] = norm_stats
        for t in self.transforms:
            data = t(data)
        replay_img = data.get('replay_img', None)

        assert 'lang_tokens' in data and 'lang_masks' in data, \
            'Prompt transform must provide lang_tokens and lang_masks'
        tokens = torch.tensor(data['lang_tokens'])
        token_mask = data['lang_masks'].tolist() if hasattr(
            data['lang_masks'], 'tolist') else list(data['lang_masks'])

        # Proprio
        img_masks = data.get('img_masks', None)
        pixel_values = data['pixel_values']
        if img_masks is None:
            # Fallback: all True masks based on pixel_values shape
            num_imgs = pixel_values.shape[0] // 3
            img_masks = [True] * num_imgs
        else:
            img_masks = list(img_masks)
        # Add padding images with zero values and False masks
        if self.num_padding_imgs > 0:
            padding_img = pixel_values.new_zeros(3, pixel_values.shape[-2],
                                                 pixel_values.shape[-1])
            padding_imgs = padding_img.repeat(self.num_padding_imgs, 1, 1)
            pixel_values = torch.cat([pixel_values, padding_imgs], dim=0)
            img_masks.extend([False] * self.num_padding_imgs)
        if self.img_buffer_len > 1:
            pixel_values, img_masks = self._update_img_buffer(
                pixel_values, img_masks)
        batch: Dict[str, Any] = dict(
            images=pixel_values.cuda().unsqueeze(0),
            img_masks=torch.tensor([img_masks]).cuda(),
            lang_tokens=tokens.unsqueeze(0).cuda(),
            lang_masks=torch.tensor(token_mask).unsqueeze(0).cuda(),
        )

        if 'states' in data:
            batch['states'] = torch.from_numpy(
                data['states']).bfloat16().cuda().unsqueeze(0)
        if 'embodiment_ids' in data:
            batch['embodiment_ids'] = torch.from_numpy(
                data['embodiment_ids']).int().cuda().unsqueeze(0)

        if data.get('image_grid_thw', None) is not None:
            batch['image_grid_thw'] = data['image_grid_thw'].unsqueeze(0)

        batch['reset_history'] = is_new_episode

        return batch, replay_img


@DATASETS.register_module()
class PrivateInferenceDataset:
    """Dataset for evaluating Libero with a VLA processor.
    This dataset processes images and prompts for evaluation purposes.
    It resizes images, applies center cropping if specified, and builds
    prompts for the VLA model.

    Args:
        norm_stats (str or Dict): Normalization statistics, which can be a
            JSON string or a dictionary containing 'mean', 'std', 'q01',
            and 'q99' for each feature.
            If a string, it should be a JSON representation of the
            normalization statistics.
        task_suite_name (str): Name of the task suite for evaluation.
        img_keys (List[str]): List of keys to extract images from the input.
            Defaults to ['agentview_image']. Note that the first key
            is used as replay image.
        processor (ConfigDict): Configuration for the VLA processor.
        center_crop (bool): Whether to apply center cropping to images.
            Defaults to False.
        resize_size (int): Size to resize images to. Defaults to 224.
        max_length (int): Maximum length of the input tokens.
            Defaults to 180.
        use_quantiles (bool): Whether to use quantiles for normalization.
            Defaults to True.
    """

    def __init__(self,
                 norm_stats: str,
                 transforms: List[Dict],
                 model_path: str,
                 img_keys: List[str] = ['agentview_image'],
                 center_crop: bool = False,
                 resize_size: int = 224,
                 max_len: int = 180,
                 use_quantiles=True,
                 embodiment_id: int = None) -> None:
        from fluxvla.engines import build_transform_from_cfg
        self.transforms = list()
        for transform in transforms:
            transform['model_path'] = model_path
            self.transforms.append(build_transform_from_cfg(transform))
        if isinstance(norm_stats, str):
            with open(norm_stats, 'r', encoding='utf-8') as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = norm_stats
        self.img_keys = img_keys
        self.center_crop = center_crop
        self.resize_size = resize_size
        self.max_len = max_len
        self.use_quantiles = use_quantiles
        self.embodiment_id = embodiment_id

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the observation for evaluation."""
        imgs = list()
        for img_key in self.img_keys:
            if img_key not in data:
                raise KeyError(
                    'Image key {!r} not found in inputs!'.format(img_key))
            imgs.append(data[img_key].transpose(2, 0, 1))  # HWC to CHW
        inputs = dict(
            images=imgs,
            task_description=data.get('task_description',
                                      'No task description provided'),
            stats=self.norm_stats['private'],
            states=data['qpos'])
        for transform in self.transforms:
            inputs = transform(inputs)

        batch = dict(
            images=torch.from_numpy(
                inputs['images']).unsqueeze(0).cuda(),  # noqa: E501
            img_masks=torch.tensor([[True for _ in range(len(self.img_keys))]
                                    ]).cuda(),  # noqa: E501
            lang_tokens=torch.from_numpy(
                inputs['lang_tokens']).unsqueeze(0).cuda(),
            lang_masks=torch.from_numpy(
                inputs['lang_masks']).unsqueeze(0).cuda(),
            states=torch.from_numpy(
                inputs['states']).float().cuda().unsqueeze(0))
        if self.embodiment_id is not None:
            batch['embodiment_ids'] = torch.from_numpy(
                np.array(self.embodiment_id)).int().cuda().unsqueeze(0)
        return batch

    def _normalize(self, normalized_states: np.ndarray, stats: Dict):
        assert 'min' in stats and stats['min'] is not None
        assert 'max' in stats and stats['max'] is not None
        state_high = np.array(stats['max'])
        state_low = np.array(stats['min'])
        mask = np.array(stats['mask'])
        states = np.where(
            mask,
            np.clip(
                2 * (normalized_states - state_low) /
                (state_high - state_low + 1e-8) - 1, -1, 1), normalized_states)
        return states

    def _normalize_quantile(self, normalized_states: np.ndarray, stats: Dict):
        assert 'q01' in stats and stats['q01'] is not None
        assert 'q99' in stats and stats['q99'] is not None  # noqa: E501
        state_high = np.array(stats['q99'])
        state_low = np.array(stats['q01'])
        if 'mask' in stats:
            mask = np.array(stats['mask'])
        else:
            mask = np.ones_like(state_high, dtype=bool)
        states = np.where(
            mask,
            np.clip(
                2 * (normalized_states - state_low) /
                (state_high - state_low + 1e-8) - 1, -1, 1), normalized_states)
        return states
