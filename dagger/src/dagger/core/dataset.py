"""Aggregated DAgger dataset with round-preserving storage."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from .contracts import DaggerRound


class AggregatedDataset:
    def __init__(self) -> None:
        self.rounds: list[DaggerRound] = []

    def append(self, round_data: DaggerRound) -> None:
        self.rounds.append(round_data)

    @property
    def count(self) -> int:
        return sum(item.count for item in self.rounds)

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.rounds:
            raise RuntimeError("cannot read an empty DAgger dataset")
        observations = torch.cat([item.student_obs for item in self.rounds], dim=0)
        actions = torch.cat([item.teacher_action for item in self.rounds], dim=0)
        return observations, actions

    def motion_ids(self) -> torch.Tensor:
        if not self.rounds:
            raise RuntimeError("cannot read an empty DAgger dataset")
        return torch.cat([item.motion_id for item in self.rounds], dim=0)

    def save_round(self, round_index: int, run_dir: str | Path) -> Path:
        path = Path(run_dir) / "data" / "dagger" / f"round_{round_index:02d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.rounds[-1].__dict__, path)
        return path

    @classmethod
    def load_rounds(cls, paths: Iterable[str | Path]) -> "AggregatedDataset":
        result = cls()
        for path in paths:
            payload = torch.load(Path(path), map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise ValueError(f"DAgger round shard is not a tensor mapping: {path}")
            result.append(DaggerRound(**payload))
        return result

