"""Memory-mapped, uniformly indexed training views of committed DAgger rounds."""
from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import torch


class SonicCausalSamplePool:
    """Keep the full aggregate on disk; stage only sampled minibatches onto CUDA."""

    KEYS = (
        "condition_state", "condition_action", "state_visible", "action_visible",
        "state_valid", "action_valid", "target_state", "target_mask_action",
        "action_before", "action_before_valid", "teacher_action",
    )

    def __init__(self, cache_root: Path):
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.columns: list[dict[str, torch.Tensor]] = []
        self.ends: list[int] = []

    def add_round(self, path: Path, samples=None) -> None:
        cache = self.cache_root / path.name
        if not cache.is_file() or cache.stat().st_mtime_ns < path.stat().st_mtime_ns:
            if samples is None:
                samples = torch.load(path, map_location="cpu", weights_only=False)["samples"]
            if not samples:
                raise ValueError(f"empty DAgger round: {path}")
            columns = {key: torch.stack([row[key] for row in samples]) for key in self.KEYS}
            temporary = cache.with_suffix(".tmp")
            torch.save(columns, temporary)
            temporary.replace(cache)
        columns = torch.load(cache, map_location="cpu", mmap=True, weights_only=True)
        count = len(columns["teacher_action"])
        if not count or any(len(value) != count for value in columns.values()):
            raise ValueError(f"invalid DAgger training cache: {cache}")
        self.columns.append(columns)
        self.ends.append(len(self) + count)

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard = bisect_right(self.ends, index)
        offset = index - (self.ends[shard - 1] if shard else 0)
        return {key: value[offset] for key, value in self.columns[shard].items()}

    def gather(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Preserve the sampled order while gathering whole columns per shard."""
        indices = indices.to(device="cpu", dtype=torch.long)
        result = {
            key: torch.empty((indices.numel(), *value.shape[1:]), dtype=value.dtype)
            for key, value in self.columns[0].items()
        }
        start = 0
        for columns, end in zip(self.columns, self.ends):
            positions = ((indices >= start) & (indices < end)).nonzero().flatten()
            if positions.numel():
                local_indices = indices.index_select(0, positions) - start
                for key, values in columns.items():
                    result[key].index_copy_(0, positions, values.index_select(0, local_indices))
            start = end
        return result
