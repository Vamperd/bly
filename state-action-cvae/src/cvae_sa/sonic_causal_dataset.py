"""Single-step SONIC-compatible causal samples from Physics v5 datasets.

The legacy CVAE uses 65-State/64-Action windows.  This module deliberately
keeps that code untouched and exposes the new causal contract:

``State[t-9:t+1] + Action[t-10:t] + Reference[t,10,64] -> Action[t]``.
"""

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

from .constants import ACTION_DIM
from .physics_schema import (
    PHYSICS_STATE_DIM,
    REFERENCE_DIM,
    REFERENCE_FRAMES,
    load_physics_schema,
    read_physics_states,
    read_reference_future,
    resolve_parameter,
)
from .util import file_sha256, load_json


HISTORY_STEPS = 10
STATE_DIM = PHYSICS_STATE_DIM
REFERENCE_DIMENSION = REFERENCE_DIM


@dataclass(frozen=True)
class CausalSampleRef:
    episode_index: int
    target_step: int


def _read_episode_index(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class SonicCausalDataset(Dataset[dict[str, Any]]):
    """Lazy Physics-v5 transition dataset with SONIC history semantics."""

    format_version = "sonic_sequence_dataset_v1"

    def __init__(
        self,
        dataset_run: str | Path,
        split: str = "train",
        *,
        max_samples: int | None = None,
    ) -> None:
        self.dataset_run = Path(dataset_run).expanduser().resolve()
        marker = self.dataset_run / "markers" / "cvae_dataset.ok"
        if not marker.is_file():
            raise FileNotFoundError(f"dataset marker is missing: {marker}")
        self.manifest = load_json(self.dataset_run / "manifests" / "dataset_manifest.json")
        if self.manifest.get("schema_version") != "sonic_physics_state_action_cvae_dataset_v5":
            raise ValueError("SonicCausalDataset requires Physics v5")
        episodes_path = self.dataset_run / "manifests" / "episodes.jsonl"
        expected_hash = self.manifest.get("episodes_index_sha256")
        if expected_hash and file_sha256(episodes_path) != expected_hash:
            raise ValueError("episodes index hash no longer matches dataset manifest")
        self.episodes = [
            item for item in _read_episode_index(episodes_path)
            if item.get("split") == split
        ]
        if not self.episodes:
            raise ValueError(f"split {split!r} contains no Physics v5 episodes")
        self.split = str(split)
        with np.load(self.dataset_run / "data" / "normalization.npz") as norm:
            self.state_mean = np.asarray(norm["physical_state_mean"], dtype=np.float32)
            self.state_std = np.asarray(norm["physical_state_std"], dtype=np.float32)
            self.action_mean = np.asarray(norm["action_mean"], dtype=np.float32)
            self.action_std = np.asarray(norm["action_std"], dtype=np.float32)
            self.reference_mean = np.asarray(
                norm["reference_future_mean"], dtype=np.float32
            )
            self.reference_std = np.asarray(
                norm["reference_future_std"], dtype=np.float32
            )
        for name, value, shape in (
            ("state_mean", self.state_mean, (STATE_DIM,)),
            ("state_std", self.state_std, (STATE_DIM,)),
            ("action_mean", self.action_mean, (ACTION_DIM,)),
            ("action_std", self.action_std, (ACTION_DIM,)),
            ("reference_mean", self.reference_mean, (REFERENCE_DIMENSION,)),
            ("reference_std", self.reference_std, (REFERENCE_DIMENSION,)),
        ):
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, found {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN/Inf")
        self.refs: list[CausalSampleRef] = []
        for episode_index, row in enumerate(self.episodes):
            steps = int(row["steps"])
            self.refs.extend(CausalSampleRef(episode_index, step) for step in range(steps))
        if max_samples is not None:
            self.refs = self.refs[: int(max_samples)]
        if not self.refs:
            raise ValueError("Physics v5 dataset contains no causal samples")
        self._handles: dict[str, h5py.File] = {}
        self._schema_cache: dict[str, dict[str, Any]] = {}
        self._episode_cache: dict[int, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.refs)

    @property
    def motion_keys(self) -> list[str]:
        """Motion keys that actually contribute samples after ``max_samples``."""

        return sorted(
            {str(self.episodes[ref.episode_index]["motion_key"]) for ref in self.refs}
        )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = {}
        state["_episode_cache"] = {}
        return state

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _handle(self, path: str) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def _schema(self, path: str) -> dict[str, Any]:
        schema = self._schema_cache.get(path)
        if schema is None:
            schema = load_physics_schema(Path(path))
            self._schema_cache[path] = schema
        return schema

    @staticmethod
    def _history_indices(target_step: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
        state_indices = np.arange(target_step - HISTORY_STEPS + 1, target_step + 1)
        action_indices = np.arange(target_step - HISTORY_STEPS, target_step)
        del steps
        return state_indices, action_indices

    def _episode_arrays(self, episode_index: int) -> dict[str, np.ndarray]:
        cached = self._episode_cache.get(episode_index)
        if cached is not None:
            return cached
        metadata = self.episodes[episode_index]
        steps = int(metadata["steps"])
        stream = self._handle(str(metadata["hdf5_path"]))
        episode = stream[f"data/{metadata['episode']}"]
        schema = self._schema(str(metadata["schema_path"]))
        env_id = int(metadata["env_id"])
        physical = read_physics_states(episode["states"], 0, steps + 1)
        reference, offsets = read_reference_future(episode, 0, steps)
        actions = np.asarray(
            episode["actions/action_target_canonical"], dtype=np.float32
        )
        if actions.shape != (steps, ACTION_DIM):
            raise ValueError(f"{episode.name}: invalid Action shape {actions.shape}")
        if reference.shape != (steps, REFERENCE_FRAMES, REFERENCE_DIMENSION):
            raise ValueError(f"{episode.name}: invalid reference shape {reference.shape}")
        # SONIC's action history is initialized with the reset action.  The
        # v5 recorder stores the initial processed target for the first
        # transition, which is the only stable offline seed available here.
        action_before = np.asarray(
            episode["actions/initial_processed_target_canonical"], dtype=np.float32
        )
        if action_before.shape != (ACTION_DIM,):
            raise ValueError(f"{episode.name}: invalid initial Action seed")
        cached = {
            "physical": physical,
            "reference": reference,
            "offsets": offsets,
            "actions": actions,
            "action_before": action_before,
            "nominal": resolve_parameter(
                schema["nominal_default_joint_pos"], env_id
            ).reshape(ACTION_DIM),
            "scale": resolve_parameter(schema["action_scale"], env_id).reshape(ACTION_DIM),
            "offset": np.asarray(
                stream[f"contexts/{metadata['context_id']}/action_offset"],
                dtype=np.float32,
            ).reshape(ACTION_DIM),
        }
        self._episode_cache[episode_index] = cached
        return cached

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[int(index)]
        metadata = self.episodes[ref.episode_index]
        steps = int(metadata["steps"])
        arrays = self._episode_arrays(ref.episode_index)
        state_indices_raw, action_indices = self._history_indices(ref.target_step, steps)
        state_valid = state_indices_raw >= 0
        state_indices = np.clip(state_indices_raw, 0, max(steps - 1, 0))
        physical = arrays["physical"]
        reference = arrays["reference"]
        offsets = arrays["offsets"]
        actions = arrays["actions"]
        action_before = arrays["action_before"]

        state_history = physical[state_indices]
        action_history = np.zeros((HISTORY_STEPS, ACTION_DIM), dtype=np.float32)
        action_valid = action_indices >= 0
        for position, source in enumerate(action_indices.tolist()):
            action_history[position] = action_before if source < 0 else actions[source]
        target_action = actions[ref.target_step]
        target_reference = reference[ref.target_step]

        state_history = (state_history - self.state_mean[None]) / self.state_std[None]
        action_history = (action_history - self.action_mean[None]) / self.action_std[None]
        target_action = (target_action - self.action_mean) / self.action_std
        target_reference = (
            target_reference - self.reference_mean[None]
        ) / self.reference_std[None]
        mapping = {
            "nominal": arrays["nominal"],
            "scale": arrays["scale"],
            "offset": arrays["offset"],
        }
        return {
            "state_history": torch.from_numpy(state_history.astype(np.float32)),
            "action_history": torch.from_numpy(action_history.astype(np.float32)),
            "reference_future": torch.from_numpy(target_reference.astype(np.float32)),
            "target_action": torch.from_numpy(target_action.astype(np.float32)),
            "state_valid": torch.from_numpy(state_valid.astype(bool)),
            "action_valid": torch.from_numpy(action_valid.astype(bool)),
            "history_validity": torch.from_numpy(
                np.stack((state_valid, action_valid), axis=-1).astype(bool)
            ),
            "reference_time_offsets": torch.from_numpy(offsets.astype(np.float32)),
            "motion_key": str(metadata["motion_key"]),
            "variant_id": int(metadata["variant_id"]),
            "episode_ref": f"{metadata['source_run']}::{metadata['episode']}",
            "target_step": int(ref.target_step),
            "action_mapping": mapping,
        }


def causal_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate tensor fields while preserving audit metadata."""

    if not batch:
        raise ValueError("cannot collate an empty causal batch")
    tensor_keys = {
        key for key, value in batch[0].items() if isinstance(value, torch.Tensor)
    }
    result = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    for key in batch[0]:
        if key in tensor_keys:
            continue
        result[key] = [item[key] for item in batch]
    return result

