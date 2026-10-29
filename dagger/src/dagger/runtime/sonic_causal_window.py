"""Runtime 10-step history buffer for the SONIC causal CVAE."""

from __future__ import annotations

from dataclasses import dataclass

import torch


STATE_STEPS = 10
STATE_DIM = 70
ACTION_DIM = 29
REFERENCE_FRAMES = 10
REFERENCE_DIM = 64


@dataclass
class SonicCausalWindow:
    batch: dict[str, torch.Tensor]
    state_valid: torch.Tensor
    action_valid: torch.Tensor
    reference_time_offsets: torch.Tensor


class SonicCausalWindowBuffer:
    def __init__(self, *, device: torch.device, num_envs: int) -> None:
        self.device = device
        self.num_envs = int(num_envs)
        self._states: list[list[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self._actions: list[list[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self._state_valid: list[list[bool]] = [[] for _ in range(self.num_envs)]
        self._action_valid: list[list[bool]] = [[] for _ in range(self.num_envs)]
        self._initial_actions: list[torch.Tensor | None] = [None] * self.num_envs
        self._current: list[torch.Tensor | None] = [None] * self.num_envs

    def reset(
        self,
        current_state: torch.Tensor,
        initial_action: torch.Tensor | None = None,
    ) -> None:
        current_state = current_state.to(self.device, dtype=torch.float32)
        if tuple(current_state.shape) != (self.num_envs, STATE_DIM):
            raise ValueError(f"current_state must be [{self.num_envs},70]")
        if initial_action is None:
            initial_action = torch.zeros(
                self.num_envs, ACTION_DIM, device=self.device, dtype=torch.float32
            )
        initial_action = initial_action.to(self.device, dtype=torch.float32)
        if tuple(initial_action.shape) != (self.num_envs, ACTION_DIM):
            raise ValueError(f"initial_action must be [{self.num_envs},29]")
        self._states = [[current_state[index].detach().clone()] * STATE_STEPS for index in range(self.num_envs)]
        self._actions = [[initial_action[index].detach().clone()] * STATE_STEPS for index in range(self.num_envs)]
        self._state_valid = [[False] * (STATE_STEPS - 1) + [True] for _ in range(self.num_envs)]
        self._action_valid = [[False] * STATE_STEPS for _ in range(self.num_envs)]
        self._initial_actions = [initial_action[index].detach().clone() for index in range(self.num_envs)]
        self._current = [current_state[index].detach().clone() for index in range(self.num_envs)]

    def history_steps(self) -> torch.Tensor:
        return torch.tensor(
            [min(len(value), STATE_STEPS) for value in self._states],
            device=self.device,
            dtype=torch.long,
        )

    def advance(
        self,
        executed_action: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        *,
        clear_on_done: bool = True,
    ) -> None:
        executed_action = executed_action.to(self.device, dtype=torch.float32)
        next_state = next_state.to(self.device, dtype=torch.float32)
        done = done.to(self.device, dtype=torch.bool)
        if tuple(executed_action.shape) != (self.num_envs, ACTION_DIM):
            raise ValueError(f"executed_action must be [{self.num_envs},29]")
        if tuple(next_state.shape) != (self.num_envs, STATE_DIM):
            raise ValueError(f"next_state must be [{self.num_envs},70]")
        for index in range(self.num_envs):
            if bool(done[index]) and clear_on_done:
                self._states[index] = [next_state[index].detach().clone()] * STATE_STEPS
                initial = self._initial_actions[index]
                if initial is None:
                    initial = executed_action[index].detach().clone()
                self._actions[index] = [initial.detach().clone()] * STATE_STEPS
                self._state_valid[index] = [False] * (STATE_STEPS - 1) + [True]
                self._action_valid[index] = [False] * STATE_STEPS
                self._current[index] = next_state[index].detach().clone()
                continue
            self._states[index].append(next_state[index].detach().clone())
            self._actions[index].append(executed_action[index].detach().clone())
            self._state_valid[index].append(True)
            self._action_valid[index].append(True)
            self._states[index] = self._states[index][-STATE_STEPS:]
            self._actions[index] = self._actions[index][-STATE_STEPS:]
            self._state_valid[index] = self._state_valid[index][-STATE_STEPS:]
            self._action_valid[index] = self._action_valid[index][-STATE_STEPS:]
            self._current[index] = next_state[index].detach().clone()

    def build_condition(
        self,
        *,
        reference_future: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        reference_mean: torch.Tensor,
        reference_std: torch.Tensor,
        reference_time_offsets: torch.Tensor | None = None,
    ) -> SonicCausalWindow:
        reference_future = reference_future.to(self.device, dtype=torch.float32)
        if tuple(reference_future.shape) != (self.num_envs, REFERENCE_FRAMES, REFERENCE_DIM):
            raise ValueError(f"reference_future must be [{self.num_envs},10,64]")
        states = torch.stack(
            [torch.stack(values[-STATE_STEPS:]) for values in self._states], dim=0
        )
        actions = torch.stack(
            [torch.stack(values[-STATE_STEPS:]) for values in self._actions], dim=0
        )
        state_mean = state_mean.to(self.device, dtype=torch.float32)
        state_std = state_std.to(self.device, dtype=torch.float32)
        action_mean = action_mean.to(self.device, dtype=torch.float32)
        action_std = action_std.to(self.device, dtype=torch.float32)
        reference_mean = reference_mean.to(self.device, dtype=torch.float32)
        reference_std = reference_std.to(self.device, dtype=torch.float32)
        state_valid = torch.as_tensor(
            self._state_valid, device=self.device, dtype=torch.bool
        )
        action_valid = torch.as_tensor(
            self._action_valid, device=self.device, dtype=torch.bool
        )
        if reference_time_offsets is None:
            reference_time_offsets = torch.arange(
                REFERENCE_FRAMES, device=self.device, dtype=torch.float32
            ) * 0.1
        reference_time_offsets = reference_time_offsets.to(self.device, dtype=torch.float32)
        if tuple(reference_time_offsets.shape) == (REFERENCE_FRAMES,):
            reference_time_offsets = reference_time_offsets[None].expand(
                self.num_envs, -1
            )
        return SonicCausalWindow(
            batch={
                "state_history": (states - state_mean[None, None])
                / state_std.clamp_min(1.0e-6)[None, None],
                "action_history": (actions - action_mean[None, None])
                / action_std.clamp_min(1.0e-6)[None, None],
                "reference_future": (reference_future - reference_mean[None, None])
                / reference_std.clamp_min(1.0e-6)[None, None],
                "target_action": torch.zeros(
                    self.num_envs, ACTION_DIM, device=self.device, dtype=torch.float32
                ),
                "state_valid": state_valid,
                "action_valid": action_valid,
                "history_validity": torch.stack((state_valid, action_valid), dim=-1),
                "reference_time_offsets": reference_time_offsets,
            },
            state_valid=state_valid,
            action_valid=action_valid,
            reference_time_offsets=reference_time_offsets,
        )

