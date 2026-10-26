"""Causal 65-token windows used by the CVAE DAgger student.

The CVAE is trained on complete 65-state/64-action windows, while rollout only
knows the history and the current state.  This module keeps those two views
explicit: the complete window is a training target, and ``build_condition``
returns the masked view used before an environment step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


STATE_STEPS = 65
ACTION_STEPS = 64
STATE_DIM = 70
ACTION_DIM = 29
ANCHOR = 32


@dataclass
class CvaeWindow:
    batch: dict[str, torch.Tensor]
    state_mask: torch.Tensor
    action_mask: torch.Tensor
    loss_state_mask: torch.Tensor
    loss_action_mask: torch.Tensor
    history_steps: torch.Tensor | None = None


def _zeros(device: torch.device, batch: int, dtype: torch.dtype = torch.float32):
    return (
        torch.zeros(batch, STATE_STEPS, STATE_DIM, device=device, dtype=dtype),
        torch.zeros(batch, ACTION_STEPS, ACTION_DIM, device=device, dtype=dtype),
        torch.zeros(batch, STATE_STEPS, device=device, dtype=torch.bool),
        torch.zeros(batch, ACTION_STEPS, device=device, dtype=torch.bool),
    )


def causal_masks(
    batch_size: int,
    *,
    device: torch.device,
    valid_state: torch.Tensor,
    valid_action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the rollout condition and the two independent loss masks.

    ``state_mask``/``action_mask`` describe values hidden from the condition
    encoder.  The loss masks are deliberately separate: future Actions are
    hidden at rollout but only the anchor Action has a SONIC teacher label.
    """

    state_mask = torch.zeros(batch_size, STATE_STEPS, STATE_DIM, device=device, dtype=torch.bool)
    action_mask = torch.zeros(batch_size, ACTION_STEPS, ACTION_DIM, device=device, dtype=torch.bool)
    state_mask[:, ANCHOR + 1 :] = True
    action_mask[:, ANCHOR:] = True
    state_mask &= valid_state[..., None]
    action_mask &= valid_action[..., None]

    loss_state_mask = torch.zeros_like(state_mask)
    loss_state_mask[:, ANCHOR + 1 :] = True
    loss_state_mask &= valid_state[..., None]
    loss_action_mask = torch.zeros_like(action_mask)
    loss_action_mask[:, ANCHOR] = True
    loss_action_mask &= valid_action[:, ANCHOR, None, None]
    return state_mask, action_mask, loss_state_mask, loss_action_mask


class CvaeWindowBuffer:
    """Per-environment history with an explicit current-state anchor."""

    def __init__(self, *, device: torch.device, num_envs: int) -> None:
        self.device = device
        self.num_envs = int(num_envs)
        self._states: list[list[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self._actions: list[list[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self._current: list[torch.Tensor | None] = [None] * self.num_envs

    def history_steps(self) -> torch.Tensor:
        """Return how many real history States each environment currently has."""

        return torch.tensor(
            [len(states) for states in self._states],
            device=self.device,
            dtype=torch.long,
        )

    def reset(self, current_state: torch.Tensor) -> None:
        current_state = current_state.to(self.device, dtype=torch.float32)
        if tuple(current_state.shape) != (self.num_envs, STATE_DIM):
            raise ValueError(f"current_state must be [{self.num_envs},70]")
        self._states = [[] for _ in range(self.num_envs)]
        self._actions = [[] for _ in range(self.num_envs)]
        self._current = [current_state[i].detach().clone() for i in range(self.num_envs)]

    def advance(
        self,
        executed_action: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        executed_action = executed_action.to(self.device, dtype=torch.float32)
        next_state = next_state.to(self.device, dtype=torch.float32)
        done = done.to(self.device, dtype=torch.bool)
        if tuple(executed_action.shape) != (self.num_envs, ACTION_DIM):
            raise ValueError(f"executed_action must be [{self.num_envs},29]")
        if tuple(next_state.shape) != (self.num_envs, STATE_DIM):
            raise ValueError(f"next_state must be [{self.num_envs},70]")
        for index in range(self.num_envs):
            if bool(done[index]):
                self._states[index] = []
                self._actions[index] = []
                self._current[index] = next_state[index].detach().clone()
                continue
            current = self._current[index]
            if current is None:
                raise RuntimeError("window buffer was not reset")
            self._states[index].append(current.detach().clone())
            self._actions[index].append(executed_action[index].detach().clone())
            self._states[index] = self._states[index][-ANCHOR:]
            self._actions[index] = self._actions[index][-ANCHOR:]
            self._current[index] = next_state[index].detach().clone()

    def build_condition(self, *, state_mean: torch.Tensor, state_std: torch.Tensor,
                        action_mean: torch.Tensor, action_std: torch.Tensor) -> CvaeWindow:
        if any(value is None for value in self._current):
            raise RuntimeError("window buffer was not reset")
        state, action, valid_state, valid_action = _zeros(self.device, self.num_envs)
        for env_index in range(self.num_envs):
            states = self._states[env_index]
            actions = self._actions[env_index]
            start = ANCHOR - len(states)
            if states:
                state[env_index, start:ANCHOR] = torch.stack(states)
                valid_state[env_index, start:ANCHOR] = True
            if actions:
                action[env_index, start:ANCHOR] = torch.stack(actions)
                valid_action[env_index, start:ANCHOR] = True
            state[env_index, ANCHOR] = self._current[env_index]  # type: ignore[index]
            valid_state[env_index, ANCHOR] = True

        state = (state - state_mean) / state_std.clamp_min(1.0e-6)
        action = (action - action_mean) / action_std.clamp_min(1.0e-6)
        valid_state[:, ANCHOR + 1 :] = True
        valid_action[:, ANCHOR:] = True
        sm, am, lsm, lam = causal_masks(
            self.num_envs, device=self.device, valid_state=valid_state, valid_action=valid_action
        )
        return CvaeWindow(
            batch={
                "physical_state": state,
                "action": action,
                "action_before_window": action[:, 0],
                "valid_state": valid_state,
                "valid_action": valid_action,
            },
            state_mask=sm,
            action_mask=am,
            loss_state_mask=lsm,
            loss_action_mask=lam,
            history_steps=self.history_steps(),
        )


def build_training_window(
    states: torch.Tensor,
    actions: torch.Tensor,
    teacher_action: torch.Tensor,
    student_action: torch.Tensor,
    *,
    step: int,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    state_valid: torch.Tensor | None = None,
    action_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build one complete target window around a local episode step.

    The input condition is causal, but the returned target retains the actual
    future simulator States.  Invalid padding is never included in a loss.
    """

    if states.ndim != 2 or states.shape[-1] != STATE_DIM:
        raise ValueError("states must have shape [T+1,70]")
    if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
        raise ValueError("actions must have shape [T,29]")
    if not 0 <= int(step) < actions.shape[0]:
        raise ValueError("step must identify an executed Action")
    if state_valid is None:
        state_valid = torch.ones(states.shape[0], device=states.device, dtype=torch.bool)
    if action_valid is None:
        action_valid = torch.ones(actions.shape[0], device=actions.device, dtype=torch.bool)
    if tuple(state_valid.shape) != (states.shape[0],) or tuple(action_valid.shape) != (actions.shape[0],):
        raise ValueError("state_valid/action_valid must match the sequence lengths")
    teacher_action = teacher_action.reshape(ACTION_DIM)
    student_action = student_action.reshape(ACTION_DIM)
    start = int(step) - ANCHOR
    state, action, valid_state, valid_action = _zeros(states.device, 1)
    for local in range(STATE_STEPS):
        source = start + local
        if 0 <= source < states.shape[0] and bool(state_valid[source]):
            state[0, local] = states[source]
            valid_state[0, local] = True
    for local in range(ACTION_STEPS):
        source = start + local
        if 0 <= source < actions.shape[0] and bool(action_valid[source]):
            action[0, local] = actions[source]
            valid_action[0, local] = True
    state_mean = state_mean.to(states.device, dtype=states.dtype)
    state_std = state_std.to(states.device, dtype=states.dtype)
    action_mean = action_mean.to(actions.device, dtype=actions.dtype)
    action_std = action_std.to(actions.device, dtype=actions.dtype)
    state_norm = (state - state_mean) / state_std.clamp_min(1.0e-6)
    action_norm = (action - action_mean) / action_std.clamp_min(1.0e-6)
    sm, am, lsm, lam = causal_masks(
        1, device=states.device, valid_state=valid_state, valid_action=valid_action
    )
    return {
        "physical_state": state_norm[0],
        "action": action_norm[0],
        "valid_state": valid_state[0],
        "valid_action": valid_action[0],
        "state_mask": sm[0],
        "action_mask": am[0],
        "loss_state_mask": lsm[0],
        "loss_action_mask": lam[0],
        "teacher_action": (teacher_action - action_mean) / action_std.clamp_min(1.0e-6),
        "student_action": (student_action - action_mean) / action_std.clamp_min(1.0e-6),
        "teacher_action_canonical": teacher_action,
        "student_action_canonical": student_action,
    }

