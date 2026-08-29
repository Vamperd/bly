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
from .physics_schema import (
    AUXILIARY_TRANSITION_DIM,
    dynamics_context_vector,
    PHYSICS_STATE_DIM,
    REFERENCE_DIM,
    REFERENCE_FRAMES,
    load_physics_schema,
    read_physics_states,
    read_reference_future,
    read_auxiliary_transitions,
    resolve_parameter as resolve_physics_parameter,
    robot_information_vector,
    structured_robot_information,
)
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
        action_energy_crop_probability: float = 0.0,
        action_energy_top_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        self.dataset_run = dataset_run.expanduser().resolve()
        marker = self.dataset_run / "markers" / "cvae_dataset.ok"
        if not marker.is_file():
            raise FileNotFoundError(f"dataset marker is missing: {marker}")
        self.manifest = load_json(self.dataset_run / "manifests" / "dataset_manifest.json")
        manifest_version = self.manifest.get("schema_version")
        if manifest_version not in {
            "sonic_state_action_cvae_dataset_v1",
            "sonic_physics_state_action_cvae_dataset_v3",
            "sonic_physics_state_action_cvae_dataset_v4",
            "sonic_physics_state_action_cvae_dataset_v5",
        }:
            raise ValueError("unsupported CVAE dataset manifest")
        self.physics_v3 = manifest_version in {
            "sonic_physics_state_action_cvae_dataset_v3",
            "sonic_physics_state_action_cvae_dataset_v4",
            "sonic_physics_state_action_cvae_dataset_v5",
        }
        self.physics_v4 = manifest_version in {
            "sonic_physics_state_action_cvae_dataset_v4",
            "sonic_physics_state_action_cvae_dataset_v5",
        }
        self.physics_v5 = manifest_version == "sonic_physics_state_action_cvae_dataset_v5"
        self.reference_available = self.physics_v5
        self.state_dim = PHYSICS_STATE_DIM if self.physics_v3 else PHYSICAL_STATE_DIM
        self.include_previous_action = not self.physics_v3
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
        self.action_energy_crop_probability = float(action_energy_crop_probability)
        self.action_energy_top_fraction = float(action_energy_top_fraction)
        if not 0.0 <= self.action_energy_crop_probability <= 1.0:
            raise ValueError("action energy crop probability must be in [0, 1]")
        if not 0.0 < self.action_energy_top_fraction <= 1.0:
            raise ValueError("action energy top fraction must be in (0, 1]")
        if self.action_energy_crop_probability and not self.physics_v4:
            raise ValueError("Action-energy crop sampling requires a Physics v4 dataset")
        with np.load(self.dataset_run / "data" / "normalization.npz") as normalization:
            self.state_mean = normalization["physical_state_mean"].astype(np.float32)
            self.state_std = normalization["physical_state_std"].astype(np.float32)
            if self.physics_v3:
                self.previous_mean = np.empty((0,), dtype=np.float32)
                self.previous_std = np.empty((0,), dtype=np.float32)
                self.robot_mean = normalization["robot_info_mean"].astype(np.float32)
                self.robot_std = normalization["robot_info_std"].astype(np.float32)
                if self.physics_v4:
                    self.joint_robot_mean = normalization["joint_robot_info_mean"].astype(np.float32)
                    self.joint_robot_std = normalization["joint_robot_info_std"].astype(np.float32)
                    self.global_robot_mean = normalization["global_robot_info_mean"].astype(np.float32)
                    self.global_robot_std = normalization["global_robot_info_std"].astype(np.float32)
                else:
                    self.joint_robot_mean = np.empty((0,), dtype=np.float32)
                    self.joint_robot_std = np.empty((0,), dtype=np.float32)
                    self.global_robot_mean = np.empty((0,), dtype=np.float32)
                    self.global_robot_std = np.empty((0,), dtype=np.float32)
                self.dynamics_mean = normalization["dynamics_context_mean"].astype(np.float32)
                self.dynamics_std = normalization["dynamics_context_std"].astype(np.float32)
                self.auxiliary_mean = normalization["auxiliary_transition_mean"].astype(np.float32)
                self.auxiliary_std = normalization["auxiliary_transition_std"].astype(np.float32)
                self.reference_mean = normalization.get(
                    "reference_future_mean", np.zeros(REFERENCE_DIM, dtype=np.float32)
                ).astype(np.float32)
                self.reference_std = normalization.get(
                    "reference_future_std", np.ones(REFERENCE_DIM, dtype=np.float32)
                ).astype(np.float32)
            else:
                self.previous_mean = normalization["previous_action_mean"].astype(np.float32)
                self.previous_std = normalization["previous_action_std"].astype(np.float32)
                self.robot_mean = np.empty((0,), dtype=np.float32)
                self.robot_std = np.empty((0,), dtype=np.float32)
                self.joint_robot_mean = np.empty((0,), dtype=np.float32)
                self.joint_robot_std = np.empty((0,), dtype=np.float32)
                self.global_robot_mean = np.empty((0,), dtype=np.float32)
                self.global_robot_std = np.empty((0,), dtype=np.float32)
                self.dynamics_mean = np.empty((0,), dtype=np.float32)
                self.dynamics_std = np.empty((0,), dtype=np.float32)
                self.auxiliary_mean = np.empty((0,), dtype=np.float32)
                self.auxiliary_std = np.empty((0,), dtype=np.float32)
                self.reference_mean = np.zeros(REFERENCE_DIM, dtype=np.float32)
                self.reference_std = np.ones(REFERENCE_DIM, dtype=np.float32)
            self.action_mean = normalization["action_mean"].astype(np.float32)
            self.action_std = normalization["action_std"].astype(np.float32)
        if self.state_mean.shape != (self.state_dim,):
            raise ValueError("invalid physical state normalization")
        self.robot_info_dim = int(self.robot_mean.size)
        self.joint_robot_info_dim = int(self.joint_robot_mean.size)
        self.global_robot_info_dim = int(self.global_robot_mean.size)
        self.dynamics_context_dim = int(self.dynamics_mean.size)
        self.auxiliary_dim = int(self.auxiliary_mean.size)
        if self.reference_mean.shape != (REFERENCE_DIM,) or self.reference_std.shape != (
            REFERENCE_DIM,
        ):
            raise ValueError("invalid reference normalization")
        vocabulary = self.manifest.get("representations", {}).get(
            "joint_actuator_type", {}
        ).get("vocabulary", ["unknown"])
        self.actuator_type_to_id = {
            str(name): index for index, name in enumerate(vocabulary)
        }
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
        self._energy_start_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.refs)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_hdf_handles"] = {}
        state["_energy_start_cache"] = {}
        return state

    def close(self) -> None:
        for handle in self._hdf_handles.values():
            handle.close()
        self._hdf_handles.clear()
        self._energy_start_cache.clear()

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
            schema = load_physics_schema(Path(path)) if self.physics_v3 else load_schema(Path(path))
            self._schema_cache[path] = schema
        return schema

    @staticmethod
    def high_energy_window_starts(
        actions: np.ndarray,
        window_transitions: int,
        top_fraction: float,
    ) -> np.ndarray:
        """Return valid starts in the highest-energy fraction of one episode."""
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != ACTION_DIM:
            raise ValueError("Action energy sampling requires [T, 29] Actions")
        window = int(window_transitions)
        if not 0.0 < float(top_fraction) <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        if window <= 0 or values.shape[0] <= window:
            return np.zeros(1, dtype=np.int64)
        width = window - 1
        if width == 0:
            scores = np.zeros(values.shape[0], dtype=np.float32)
        else:
            derivative = np.square(np.diff(values, axis=0)).sum(axis=-1)
            scores = np.convolve(
                derivative,
                np.ones(width, dtype=np.float32),
                mode="valid",
            ) / width
        expected = values.shape[0] - window + 1
        if scores.shape != (expected,):
            raise RuntimeError("Action energy window score length is inconsistent")
        count = max(1, int(math.ceil(scores.size * float(top_fraction))))
        return np.sort(np.argpartition(scores, -count)[-count:]).astype(np.int64)

    def _physics_window_start(
        self,
        episode: h5py.Group,
        steps: int,
        fixed_start: int | None,
    ) -> int:
        if fixed_start is not None:
            return int(fixed_start)
        max_start = max(0, steps - self.window)
        if max_start == 0:
            return 0
        if (
            self.action_energy_crop_probability > 0.0
            and np.random.random() < self.action_energy_crop_probability
        ):
            cache_key = f"{episode.file.filename}:{episode.name}"
            starts = self._energy_start_cache.get(cache_key)
            if starts is None:
                actions = np.asarray(
                    episode["actions/action_target_canonical"], dtype=np.float32
                )
                starts = self.high_energy_window_starts(
                    actions, self.window, self.action_energy_top_fraction
                )
                self._energy_start_cache[cache_key] = starts
            return int(starts[int(np.random.randint(0, len(starts)))])
        return int(np.random.randint(0, max_start + 1))

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
        if self.physics_v3:
            return self._physics_item(index)
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
            "action_before_window": torch.zeros(ACTION_DIM, dtype=torch.float32),
            "action_scale": torch.from_numpy(action_scale.copy()),
            "robot_information": torch.empty(0, dtype=torch.float32),
            "joint_robot_information": torch.empty((29, 0), dtype=torch.float32),
            "joint_actuator_type": torch.zeros(29, dtype=torch.long),
            "global_robot_information": torch.empty(0, dtype=torch.float32),
            "dynamics_context": torch.empty(0, dtype=torch.float32),
            "auxiliary_transition": torch.empty((self.window, 0), dtype=torch.float32),
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

    def _physics_item(self, index: int) -> dict[str, Any]:
        ref = self.refs[index]
        metadata = self.episodes[ref.episode_index]
        steps = int(metadata["steps"])
        stream = self._handle(metadata["hdf5_path"])
        episode = stream[f"data/{metadata['episode']}"]
        start = self._physics_window_start(episode, steps, ref.fixed_start)
        end = min(start + self.window, steps)
        count = end - start
        schema = self._schema(metadata["schema_path"])
        env_id = int(metadata["env_id"])
        physical = read_physics_states(episode["states"], start, end + 1)
        actions = np.asarray(
            episode["actions/action_target_canonical"][start:end], dtype=np.float32
        )
        context = stream[f"contexts/{metadata['context_id']}"]
        robot_info = robot_information_vector(schema, context, env_id)
        if self.physics_v4:
            joint_robot, actuator_type, global_robot = structured_robot_information(
                schema, context, env_id, self.actuator_type_to_id
            )
        else:
            joint_robot = np.empty((29, 0), dtype=np.float32)
            actuator_type = np.zeros(29, dtype=np.int64)
            global_robot = np.empty((0,), dtype=np.float32)
        dynamics_context = dynamics_context_vector(context)
        auxiliary = read_auxiliary_transitions(episode["diagnostics"], start, end)
        action_scale = resolve_physics_parameter(schema["action_scale"], env_id)
        if start > 0:
            action_before = np.asarray(
                episode["actions/action_target_canonical"][start - 1], dtype=np.float32
            )
        elif self.physics_v4:
            action_before = np.asarray(
                episode["actions/initial_processed_target_canonical"], dtype=np.float32
            )
        else:
            action_before = np.zeros(ACTION_DIM, dtype=np.float32)
        if action_before.shape != (ACTION_DIM,) or not np.isfinite(action_before).all():
            raise ValueError(f"{episode.name}: invalid action_before_window")

        state_output = np.zeros((self.window + 1, self.state_dim), dtype=np.float32)
        action_output = np.zeros((self.window, ACTION_DIM), dtype=np.float32)
        auxiliary_output = np.zeros(
            (self.window, AUXILIARY_TRANSITION_DIM), dtype=np.float32
        )
        reference_output = np.zeros(
            (self.window, REFERENCE_FRAMES, REFERENCE_DIM), dtype=np.float32
        )
        reference_offsets = np.zeros(REFERENCE_FRAMES, dtype=np.float32)
        state_output[: count + 1] = (physical - self.state_mean) / self.state_std
        action_output[:count] = (actions - self.action_mean) / self.action_std
        auxiliary_output[:count] = (
            auxiliary - self.auxiliary_mean
        ) / self.auxiliary_std
        if self.reference_available:
            reference, reference_offsets = read_reference_future(episode, start, end)
            reference_output[:count] = (
                reference - self.reference_mean[None, None]
            ) / self.reference_std[None, None]
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
            "previous_action": torch.empty((self.window + 1, 0), dtype=torch.float32),
            "action": torch.from_numpy(action_output),
            "action_before_window": torch.from_numpy(
                ((action_before - self.action_mean) / self.action_std).astype(np.float32)
            ),
            "action_scale": torch.from_numpy(action_scale.copy()),
            "robot_information": torch.from_numpy(
                ((robot_info - self.robot_mean) / self.robot_std).astype(np.float32)
            ),
            "joint_robot_information": torch.from_numpy(
                ((joint_robot - self.joint_robot_mean) / self.joint_robot_std).astype(np.float32)
                if self.physics_v4 else joint_robot
            ),
            "joint_actuator_type": torch.from_numpy(actuator_type),
            "global_robot_information": torch.from_numpy(
                ((global_robot - self.global_robot_mean) / self.global_robot_std).astype(np.float32)
                if self.physics_v4 else global_robot
            ),
            "dynamics_context": torch.from_numpy(
                ((dynamics_context - self.dynamics_mean) / self.dynamics_std).astype(np.float32)
            ),
            "auxiliary_transition": torch.from_numpy(auxiliary_output),
            "reference_future": torch.from_numpy(reference_output),
            "reference_time_offsets": torch.from_numpy(reference_offsets.copy()),
            "reference_available": torch.tensor(self.reference_available, dtype=torch.bool),
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
