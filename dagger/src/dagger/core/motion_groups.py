"""Deterministic motion grouping and pilot selection for CVAE DAgger."""

from __future__ import annotations

from typing import Iterable
import random


def normalize_motion_keys(values: Iterable[object]) -> list[str]:
    """Return a sorted, de-duplicated list of motion keys."""

    return sorted({str(value) for value in values})


def build_motion_groups(values: Iterable[object], group_size: int = 8) -> list[list[str]]:
    """Split sorted motion keys into deterministic contiguous groups."""

    ordered = normalize_motion_keys(values)
    if int(group_size) <= 0:
        raise ValueError("group_size must be positive")
    if not ordered:
        raise ValueError("at least one motion key is required")
    groups = [
        ordered[index : index + int(group_size)]
        for index in range(0, len(ordered), int(group_size))
    ]
    if any(not group for group in groups):
        raise ValueError("motion groups must not be empty")
    return groups


def select_pilot_keys(
    values: Iterable[object],
    *,
    count: int,
    seed: int,
    preferred: Iterable[object] = (),
) -> list[str]:
    """Select pilot keys, keeping preferred keys that exist and filling by seed."""

    ordered = normalize_motion_keys(values)
    if int(count) <= 0:
        raise ValueError("pilot count must be positive")
    if len(ordered) < int(count):
        raise ValueError(
            f"requested {int(count)} pilot keys but only {len(ordered)} are available"
        )
    chosen: list[str] = []
    for value in preferred:
        key = str(value)
        if key in ordered and key not in chosen:
            chosen.append(key)
        if len(chosen) == int(count):
            return chosen
    remaining = [key for key in ordered if key not in chosen]
    needed = int(count) - len(chosen)
    chosen.extend(random.Random(int(seed)).sample(remaining, needed))
    return chosen[: int(count)]
