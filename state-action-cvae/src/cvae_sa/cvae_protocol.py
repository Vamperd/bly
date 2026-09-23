"""Versioned experiment identities and synchronous, commit-aware sampling.

This module deliberately has no HDF5 dependency and does not change the model.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from .util import atomic_replace

PROTOCOL = "65-token-experiment-v2"
CHECKPOINT = "sonic_65_token_hierarchical_standard_cvae_checkpoint_v2"
LEGACY_CHECKPOINT = "sonic_65_token_hierarchical_standard_cvae_checkpoint_v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def scalar(batch: dict, key: str, i: int, default: Any = None) -> Any:
    value = batch.get(key)
    if value is None:
        return default
    value = value[i]
    return value.item() if isinstance(value, torch.Tensor) and value.numel() == 1 else value


def sample_identity(sample: dict) -> dict:
    required = ("episode_ref", "motion_key", "variant_id", "window_start", "valid_state", "valid_action")
    if any(key not in sample for key in required):
        raise ValueError("stable window identity requires source/episode/motion/variant/start/valid lengths")
    return {
        "episode_ref": str(sample["episode_ref"]), "motion_key": str(sample["motion_key"]),
        "variant_id": int(sample["variant_id"]), "window_start": int(sample["window_start"]),
        "valid_states": int(sample["valid_state"].sum()), "valid_actions": int(sample["valid_action"].sum()),
    }


class Fixtures(Dataset):
    def __init__(self, dataset, indices: list[int], *, expand: bool):
        self.dataset, self.indices, self.expand = dataset, indices, expand

    def __len__(self):
        return len(self.indices) * (8 if self.expand else 1)

    def __getitem__(self, index):
        window, slot = divmod(int(index), 8) if self.expand else (int(index), None)
        sample = dict(self.dataset[self.indices[window]])
        sample["stable_window_id"] = digest(sample_identity(sample))
        sample["window_index"] = self.indices[window]
        sample["fixture_index"] = int(index)
        if slot is not None:
            sample["mask_slot"] = slot
        return sample

    def manifest(self):
        rows = []
        for i in range(len(self.indices)):
            item = self[i * (8 if self.expand else 1)]
            rows.append({**sample_identity(item), "stable_window_id": item["stable_window_id"],
                         "window_index": item["window_index"]})
        if len({row["stable_window_id"] for row in rows}) != len(rows):
            raise ValueError("duplicate selected window identities")
        return rows


class RecoverableSampler:
    """No prefetch: checkpoint state denotes the next unconsumed sample exactly."""
    def __init__(self, size: int, batch_size: int, seed: int):
        if size <= 0 or batch_size <= 0:
            raise ValueError("positive sampler size and batch size required")
        self.size, self.batch_size = size, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator).tolist()
        self.cursor, self.epoch, self.exposures = 0, 0, 0

    def next(self, dataset):
        if self.cursor == self.size:
            self.order = torch.randperm(self.size, generator=self.generator).tolist()
            self.cursor = 0
            self.epoch += 1
        ids = self.order[self.cursor:self.cursor + self.batch_size]
        batch = default_collate([dataset[index] for index in ids])
        batch["sample_ordinal"] = torch.arange(self.exposures, self.exposures + len(ids))
        self.cursor += len(ids)
        self.exposures += len(ids)
        return batch

    def state_dict(self):
        return {"size": self.size, "batch_size": self.batch_size, "order": self.order,
                "cursor": self.cursor, "epoch": self.epoch, "exposures": self.exposures,
                "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        if state["size"] != self.size or state["batch_size"] != self.batch_size:
            raise ValueError("sampler contract mismatch")
        if sorted(state["order"]) != list(range(self.size)) or not 0 <= state["cursor"] <= self.size:
            raise ValueError("invalid sampler permutation/cursor")
        self.order = list(state["order"])
        self.cursor, self.epoch, self.exposures = state["cursor"], state["epoch"], state["exposures"]
        self.generator.set_state(state["generator"].cpu())


def capture_rng():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python": random.getstate(), "numpy": np.random.get_state()}


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


@contextmanager
def isolated_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


def durable_save(path: Path, checkpoint: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(temp, path)
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@contextmanager
def run_lock(run: Path):
    """OS-owned lock, released even after a crash; no stale-lock deletion needed."""
    with (run / "training.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"another trainer owns this run: {run}") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def lr_factor(update_index: int, maximum: int, warmup: int, floor: float, schedule: str):
    """index 0 is the first update; last actual update reaches the requested floor."""
    if maximum < 1 or not 0 <= warmup < maximum or not 0 < floor <= 1:
        raise ValueError("require 0 <= warmup < max_steps and 0 < min_lr_ratio <= 1")
    if warmup and update_index < warmup:
        return (update_index + 1) / warmup
    if schedule == "constant":
        return 1.0
    fraction = min(1., max(0., (update_index - warmup) / max(maximum - 1 - warmup, 1)))
    if maximum == 1 or update_index >= maximum - 1:
        fraction = 1.
    return floor + (1 - floor) * ((1 - fraction) if schedule == "linear" else .5 * (1 + math.cos(math.pi * fraction)))


def quality_warnings(rows: list[dict]) -> list[str]:
    warnings = []
    if len(rows) >= 5:
        first, last = rows[-5]["selection_score"], rows[-1]["selection_score"]
        if (first - last) / max(abs(first), 1e-12) < .01:
            warnings.append("primary_improvement_under_1_percent_over_5_evaluations")
    if len(rows) >= 4:
        historical_best = min(row["selection_score"] for row in rows[:-3])
        if all(row["selection_score"] > 1.2 * max(historical_best, 1e-12) for row in rows[-3:]):
            warnings.append("primary_regression_over_20_percent_for_3_evaluations")
    if len(rows) >= 2 and all("masked_mse" in row for row in rows[-2:]):
        a, b = rows[-2:]
        if b["masked_mse"] > a["masked_mse"] and b["total_loss"] < a["total_loss"]:
            warnings.append("masked_worsens_while_full_improves")
    return warnings
