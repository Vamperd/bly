from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import ACTION_DIM, PHYSICAL_STATE_DIM, PHYSICAL_STATE_FIELDS
from .schema import load_schema, raw_action_to_relative, resolve_parameter
from .util import file_sha256, load_json


@dataclass(frozen=True)
class WindowRef:
    episode_index: int
    fixed_start: int | None


def read_episode_index(dataset_run: Path) -> list[dict[str, Any]]:
    path = dataset_run / "manifests" / "episodes.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class StateActionWindowDataset(Dataset[dict[str, Any]]):
    """Lazy HDF5 windows. File handles are created only inside each worker process."""

    def __init__(
        self,
        dataset_run: Path,
        split: str,
        window_transitions: int = 128,
        validation_stride: int = 64,
        max_episodes: int | None = None,
        random_crop: bool | None = None,
    ) -> None:
        super().__init__()
        self.dataset_run = dataset_run.expanduser().resolve()
        marker = self.dataset_run / "markers" / "cvae_dataset.ok"
        if not marker.is_file():
            raise FileNotFoundError(f"dataset marker is missing: {marker}")
        self.manifest = load_json(self.dataset_run / "manifests" / "dataset_manifest.json")
        if self.manifest.get("schema_version") != "sonic_state_action_cvae_dataset_v1":
            raise ValueError("unsupported CVAE dataset manifest")
        episodes_path = self.dataset_run / "manifests" / "episodes.jsonl"
        expected_hash = self.manifest.get("episodes_index_sha256")
        if expected_hash and file_sha256(episodes_path) != expected_hash:
            raise ValueError("episodes index hash no longer matches dataset manifest")
        self.episodes = [
            item for item in read_episode_index(self.dataset_run) if item["split"] == split
        ]
        if max_episodes is not None:
            self.episodes = self.episodes[:max_episodes]
        if not self.episodes:
            raise ValueError(f"split {split!r} contains no indexed episodes")
        self.split = split
        self.window = int(window_transitions)
        self.stride = int(validation_stride)
        self.random_crop = split == "train" if random_crop is None else bool(random_crop)
        with np.load(self.dataset_run / "data" / "normalization.npz") as normalization:
            self.state_mean = normalization["physical_state_mean"].astype(np.float32)
            self.state_std = normalization["physical_state_std"].astype(np.float32)
            self.previous_mean = normalization["previous_action_mean"].astype(np.float32)
            self.previous_std = normalization["previous_action_std"].astype(np.float32)
            self.action_mean = normalization["action_mean"].astype(np.float32)
            self.action_std = normalization["action_std"].astype(np.float32)
        if self.state_mean.shape != (PHYSICAL_STATE_DIM,):
            raise ValueError("invalid physical state normalization")
        self.refs: list[WindowRef] = []
        for episode_index, item in enumerate(self.episodes):
            steps = int(item["steps"])
            if self.random_crop:
                repeats = max(1, math.ceil(steps / self.window))
                self.refs.extend(WindowRef(episode_index, None) for _ in range(repeats))
            else:
                max_start = max(0, steps - self.window)
                starts = list(range(0, max_start + 1, self.stride))
                if not starts or starts[-1] != max_start:
                    starts.append(max_start)
                self.refs.extend(WindowRef(episode_index, start) for start in starts)
        self._hdf_handles: dict[str, h5py.File] = {}
        self._schema_cache: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.refs)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_hdf_handles"] = {}
        return state

    def close(self) -> None:
        for handle in self._hdf_handles.values():
            handle.close()
        self._hdf_handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _handle(self, path: str) -> h5py.File:
        handle = self._hdf_handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._hdf_handles[path] = handle
        return handle

    def _schema(self, path: str) -> dict[str, Any]:
        schema = self._schema_cache.get(path)
        if schema is None:
            schema = load_schema(Path(path))
            self._schema_cache[path] = schema
        return schema

    @staticmethod
    def _physical_slice(group: h5py.Group, start: int, stop: int) -> np.ndarray:
        arrays = []
        for name, width in PHYSICAL_STATE_FIELDS:
            values = np.asarray(group[name][start:stop], dtype=np.float32)
            if values.shape != (stop - start, width):
                raise ValueError(f"{group.name}/{name}: unexpected slice {values.shape}")
            arrays.append(values)
        return np.concatenate(arrays, axis=-1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[index]
        metadata = self.episodes[ref.episode_index]
        steps = int(metadata["steps"])
        if ref.fixed_start is None:
            max_start = max(0, steps - self.window)
            start = int(np.random.randint(0, max_start + 1)) if max_start else 0
        else:
            start = ref.fixed_start
        end = min(start + self.window, steps)
        count = end - start
        episode = self._handle(metadata["hdf5_path"])[f"data/{metadata['episode']}"]
        schema = self._schema(metadata["schema_path"])
        env_id = int(metadata["env_id"])

        current_state = self._physical_slice(episode["state_t"], start, end)
        terminal_state = self._physical_slice(episode["state_tp1"], end - 1, end)
        physical = np.concatenate((current_state, terminal_state), axis=0)
        previous_raw = np.asarray(
            episode["state_t/previous_action"][start:end], dtype=np.float32
        )
        terminal_previous_raw = np.asarray(
            episode["state_tp1/previous_action"][end - 1 : end], dtype=np.float32
        )
        previous_raw = np.concatenate((previous_raw, terminal_previous_raw), axis=0)
        previous, scale = raw_action_to_relative(previous_raw, schema, env_id)
        raw_actions = np.asarray(episode["actions"][start:end], dtype=np.float32)
        actions, action_scale = raw_action_to_relative(raw_actions, schema, env_id)
        if not np.array_equal(scale, action_scale):
            raise ValueError(f"{episode.name}: action scale changed within a window")

        state_output = np.zeros((self.window + 1, PHYSICAL_STATE_DIM), dtype=np.float32)
        previous_output = np.zeros((self.window + 1, ACTION_DIM), dtype=np.float32)
        action_output = np.zeros((self.window, ACTION_DIM), dtype=np.float32)
        state_output[: count + 1] = (physical - self.state_mean) / self.state_std
        previous_output[: count + 1] = (previous - self.previous_mean) / self.previous_std
        action_output[:count] = (actions - self.action_mean) / self.action_std
        valid_state = np.zeros(self.window + 1, dtype=bool)
        valid_action = np.zeros(self.window, dtype=bool)
        valid_state[: count + 1] = True
        valid_action[:count] = True
        progress = np.zeros(self.window + 1, dtype=np.float32)
        progress[: count + 1] = np.minimum(
            (start + np.arange(count + 1, dtype=np.float32)) / max(steps, 1), 1.0
        )

        return {
            "physical_state": torch.from_numpy(state_output),
            "previous_action": torch.from_numpy(previous_output),
            "action": torch.from_numpy(action_output),
            "action_scale": torch.from_numpy(action_scale.copy()),
            "valid_state": torch.from_numpy(valid_state),
            "valid_action": torch.from_numpy(valid_action),
            "progress": torch.from_numpy(progress),
            "motion_key": metadata["motion_key"],
            "package": metadata["package"],
            "status": metadata["status"],
            "variant_id": int(metadata["variant_id"]),
            "window_start": start,
            "episode_ref": f"{metadata['source_run']}::{metadata['episode']}",
        }

    def denormalize_state(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.state_mean, device=value.device, dtype=value.dtype)
        std = torch.as_tensor(self.state_std, device=value.device, dtype=value.dtype)
        return value * std + mean

    def denormalize_previous(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.previous_mean, device=value.device, dtype=value.dtype)
        std = torch.as_tensor(self.previous_std, device=value.device, dtype=value.dtype)
        return value * std + mean

    def denormalize_action(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.action_mean, device=value.device, dtype=value.dtype)
        std = torch.as_tensor(self.action_std, device=value.device, dtype=value.dtype)
        return value * std + mean


def worker_seed(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
