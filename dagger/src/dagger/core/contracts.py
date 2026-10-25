"""Stable adapter contracts shared by SONIC, MLP and CVAE students."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


class TeacherAdapter(Protocol):
    def reset(self, num_envs: int) -> None: ...

    def act(self, observation: Any, done: torch.Tensor) -> torch.Tensor: ...


class StudentAdapter(Protocol):
    def reset(self, num_envs: int) -> None: ...

    def observe(self, observation: Any) -> torch.Tensor: ...

    def act(self, student_input: torch.Tensor) -> torch.Tensor: ...

    def train_on_dataset(self, dataset: Any) -> dict[str, float]: ...

    def save(self, path: str) -> None: ...


class CvaeSequenceStudentAdapter(Protocol):
    """Protocol used by the sequence-aware CVAE DAgger loop."""

    def reset(self, num_envs: int, current_state: torch.Tensor) -> None: ...

    def observe(self) -> Any: ...

    def after_step(self, executed_action_canonical: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> None: ...

    def train_on_dataset(self, dataset: Any) -> dict[str, float]: ...

    def save(self, path: str) -> None: ...


@dataclass
class DaggerRound:
    """CPU-resident samples collected during one rollout round."""

    student_obs: torch.Tensor
    teacher_action: torch.Tensor
    student_action: torch.Tensor
    executed_action: torch.Tensor
    done: torch.Tensor
    episode_id: torch.Tensor
    timestep: torch.Tensor
    motion_id: torch.Tensor
    mixture_teacher: torch.Tensor

    def __post_init__(self) -> None:
        count = self.student_obs.shape[0]
        tensors = (
            self.teacher_action,
            self.student_action,
            self.executed_action,
            self.done,
            self.episode_id,
            self.timestep,
            self.motion_id,
            self.mixture_teacher,
        )
        if any(t.shape[0] != count for t in tensors):
            raise ValueError("all DAgger round tensors must have the same sample count")
        if self.student_obs.ndim != 2:
            raise ValueError("student_obs must have shape [N,D]")
        for name, tensor in (
            ("student_obs", self.student_obs),
            ("teacher_action", self.teacher_action),
            ("student_action", self.student_action),
            ("executed_action", self.executed_action),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"non-finite values found in {name}")

    @property
    def count(self) -> int:
        return int(self.student_obs.shape[0])

