"""Deterministic motion-key selection for targeted DAgger experiments."""

from __future__ import annotations

from pathlib import Path
import random


def select_motion_keys(motion_file: str | Path, count: int, seed: int) -> list[str]:
    """Return ``count`` motion keys sampled deterministically from a motion directory."""

    path = Path(motion_file).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"motion_file must be a directory for automatic selection: {path}")
    if int(count) <= 0:
        raise ValueError("motion count must be positive")
    keys = sorted(
        {item.stem for item in path.rglob("*.pkl") if item.name != "metadata.pkl"}
    )
    if len(keys) < int(count):
        raise ValueError(
            f"motion_file contains {len(keys)} motions but {int(count)} were requested: {path}"
        )
    return random.Random(int(seed)).sample(keys, int(count))
