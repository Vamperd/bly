"""Persistent CVAE DAgger shards and round-balanced sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


_REQUIRED = (
    "physical_state",
    "action",
    "valid_state",
    "valid_action",
    "state_mask",
    "action_mask",
    "loss_state_mask",
    "loss_action_mask",
    "teacher_action",
    "student_action",
)


@dataclass
class CvaeDaggerRound:
    """CPU tensors collected for one DAgger round."""

    tensors: dict[str, torch.Tensor]

    def __post_init__(self) -> None:
        missing = [key for key in _REQUIRED if key not in self.tensors]
        if missing:
            raise ValueError(f"CVAE DAgger round is missing fields: {missing}")
        count = int(self.tensors["physical_state"].shape[0])
        expected = {
            "physical_state": (count, 65, 70),
            "action": (count, 64, 29),
            "valid_state": (count, 65),
            "valid_action": (count, 64),
            "state_mask": (count, 65, 70),
            "action_mask": (count, 64, 29),
            "loss_state_mask": (count, 65, 70),
            "loss_action_mask": (count, 64, 29),
            "teacher_action": (count, 29),
            "student_action": (count, 29),
        }
        for key, shape in expected.items():
            value = self.tensors[key]
            if tuple(value.shape) != shape:
                raise ValueError(f"{key} must have shape {shape}, found {tuple(value.shape)}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"non-finite CVAE DAgger values in {key}")
        for key in ("valid_state", "valid_action", "state_mask", "action_mask", "loss_state_mask", "loss_action_mask"):
            self.tensors[key] = self.tensors[key].bool()
        for key, value in list(self.tensors.items()):
            self.tensors[key] = value.detach().cpu()

    @property
    def count(self) -> int:
        return int(self.tensors["physical_state"].shape[0])


class CvaeDaggerDataset:
    """Round-preserving collection used by the external DAgger trainer."""

    def __init__(self) -> None:
        self.rounds: list[CvaeDaggerRound] = []

    def append(self, round_data: CvaeDaggerRound) -> None:
        self.rounds.append(round_data)

    @property
    def count(self) -> int:
        return sum(round_data.count for round_data in self.rounds)

    def save_round(self, round_index: int, run_dir: str | Path) -> Path:
        path = Path(run_dir) / "data" / "dagger" / f"round_{round_index:02d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": "dagger_cvae_round_v1",
                "round_index": int(round_index),
                "count": self.rounds[-1].count,
                "tensors": self.rounds[-1].tensors,
            },
            path,
        )
        return path

    def concatenated(self) -> dict[str, torch.Tensor]:
        if not self.rounds:
            raise RuntimeError("CVAE DAgger dataset is empty")
        keys = self.rounds[0].tensors.keys()
        return {key: torch.cat([item.tensors[key] for item in self.rounds], dim=0) for key in keys}

    def sample(self, batch_size: int, *, generator: torch.Generator) -> dict[str, torch.Tensor]:
        """Sample equal round mass, then random rows within each selected round."""

        if not self.rounds:
            raise RuntimeError("CVAE DAgger dataset is empty")
        selected_rounds = torch.randint(
            len(self.rounds), (int(batch_size),), generator=generator, device="cpu"
        )
        rows: dict[str, list[torch.Tensor]] = {key: [] for key in self.rounds[0].tensors}
        for round_index in selected_rounds.tolist():
            round_data = self.rounds[round_index]
            row = torch.randint(round_data.count, (1,), generator=generator).item()
            for key, value in round_data.tensors.items():
                rows[key].append(value[row])
        return {key: torch.stack(values) for key, values in rows.items()}

    @classmethod
    def load_rounds(cls, paths: list[str | Path]) -> "CvaeDaggerDataset":
        result = cls()
        for path in paths:
            payload: Any = torch.load(Path(path), map_location="cpu", weights_only=False)
            if payload.get("format_version") != "dagger_cvae_round_v1":
                raise ValueError(f"unsupported DAgger shard: {path}")
            result.append(CvaeDaggerRound(dict(payload["tensors"])))
        return result

