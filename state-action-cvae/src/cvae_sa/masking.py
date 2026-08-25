from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .constants import COMPLETION_NAMES, TASK_NAMES


@dataclass
class MaskBatch:
    state_input: torch.Tensor
    previous_input: torch.Tensor
    action_input: torch.Tensor
    state_loss: torch.Tensor
    previous_loss: torch.Tensor
    action_loss: torch.Tensor
    task_id: int
    task_name: str
    completion_name: str | None
    causal: bool

    def to(self, device: torch.device) -> "MaskBatch":
        return MaskBatch(
            self.state_input.to(device),
            self.previous_input.to(device),
            self.action_input.to(device),
            self.state_loss.to(device),
            self.previous_loss.to(device),
            self.action_loss.to(device),
            self.task_id,
            self.task_name,
            self.completion_name,
            self.causal,
        )


class MaskGenerator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.task_probabilities = torch.tensor(
            config["task_probabilities"], dtype=torch.float32
        )
        self.completion_probabilities = torch.tensor(
            config["completion_probabilities"], dtype=torch.float32
        )
        self.element_fraction = tuple(float(x) for x in config["element_fraction"])
        self.step_count = tuple(int(x) for x in config["step_count"])
        self.feature_fraction = tuple(float(x) for x in config["feature_fraction"])
        if self.task_probabilities.shape != (3,) or not torch.isclose(
            self.task_probabilities.sum(), torch.tensor(1.0), atol=1e-5
        ):
            raise ValueError("task probabilities must contain three values summing to one")
        if self.completion_probabilities.shape != (3,) or not torch.isclose(
            self.completion_probabilities.sum(), torch.tensor(1.0), atol=1e-5
        ):
            raise ValueError("completion probabilities must contain three values summing to one")

    @staticmethod
    def _fraction(low: float, high: float, device: torch.device) -> float:
        return float(torch.empty((), device=device).uniform_(low, high).item())

    @staticmethod
    def _empty(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros_like(batch["physical_state"], dtype=torch.bool),
            torch.zeros_like(batch["previous_action"], dtype=torch.bool),
            torch.zeros_like(batch["action"], dtype=torch.bool),
        )

    def generate(
        self,
        batch: dict[str, torch.Tensor],
        force_task: str | None = None,
        force_completion: str | None = None,
    ) -> MaskBatch:
        device = batch["physical_state"].device
        valid_state = batch["valid_state"].bool()
        valid_action = batch["valid_action"].bool()
        if force_task is None:
            task_id = int(torch.multinomial(self.task_probabilities.to(device), 1).item())
            task_name = TASK_NAMES[task_id]
        else:
            task_name = force_task
            task_id = TASK_NAMES.index(task_name)
        state_input, previous_input, action_input = self._empty(batch)
        state_loss, previous_loss, action_loss = self._empty(batch)
        completion_name: str | None = None

        if task_name == "forward":
            state_input[:, 1:] = valid_state[:, 1:, None]
            state_loss.copy_(state_input)
        elif task_name == "inverse":
            action_input.copy_(valid_action[:, :, None])
            action_loss.copy_(action_input)
        else:
            if force_completion is None:
                completion_id = int(
                    torch.multinomial(self.completion_probabilities.to(device), 1).item()
                )
                completion_name = COMPLETION_NAMES[completion_id]
            else:
                completion_name = force_completion
            if completion_name == "element":
                self._element_masks(
                    state_input, previous_input, action_input, valid_state, valid_action
                )
            elif completion_name == "step":
                self._step_masks(
                    state_input, previous_input, action_input, valid_state, valid_action
                )
            elif completion_name == "feature":
                self._feature_masks(
                    state_input, previous_input, action_input, valid_state, valid_action
                )
            else:
                raise ValueError(f"unsupported completion mask {completion_name!r}")
            state_loss.copy_(state_input)
            previous_loss.copy_(previous_input)
            action_loss.copy_(action_input)

        # action_t and state_{t+1}.previous_action encode the same control command.
        # Hide the duplicate from the input, but never add a second target loss for it.
        if previous_input.shape[-1]:
            previous_input[:, 1:] |= action_input
            previous_loss[:, 1:] &= ~action_loss
        state_input &= valid_state[:, :, None]
        previous_input &= valid_state[:, :, None]
        action_input &= valid_action[:, :, None]
        state_loss &= valid_state[:, :, None]
        previous_loss &= valid_state[:, :, None]
        action_loss &= valid_action[:, :, None]
        return MaskBatch(
            state_input,
            previous_input,
            action_input,
            state_loss,
            previous_loss,
            action_loss,
            task_id,
            task_name,
            completion_name,
            task_name == "forward",
        )

    def _element_masks(
        self,
        state: torch.Tensor,
        previous: torch.Tensor,
        action: torch.Tensor,
        valid_state: torch.Tensor,
        valid_action: torch.Tensor,
    ) -> None:
        device = state.device
        low, high = self.element_fraction
        for target, valid in (
            (state, valid_state),
            (previous, valid_state),
            (action, valid_action),
        ):
            fraction = self._fraction(low, high, device)
            target.copy_((torch.rand(target.shape, device=device) < fraction) & valid[:, :, None])

    def _step_masks(
        self,
        state: torch.Tensor,
        previous: torch.Tensor,
        action: torch.Tensor,
        valid_state: torch.Tensor,
        valid_action: torch.Tensor,
    ) -> None:
        device = state.device
        batch_size = state.shape[0]
        low, high = self.step_count
        for batch_index in range(batch_size):
            mask_state = bool(torch.randint(0, 2, (), device=device).item())
            length = int(torch.randint(low, high + 1, (), device=device).item())
            valid = valid_state[batch_index] if mask_state else valid_action[batch_index]
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            length = min(length, int(valid_indices.numel()))
            contiguous = bool(torch.randint(0, 2, (), device=device).item())
            if contiguous:
                offset = int(
                    torch.randint(0, int(valid_indices.numel()) - length + 1, (), device=device).item()
                )
                indices = valid_indices[offset : offset + length]
            else:
                permutation = torch.randperm(valid_indices.numel(), device=device)[:length]
                indices = valid_indices[permutation]
            if mask_state:
                state[batch_index, indices] = True
                previous[batch_index, indices] = True
            else:
                action[batch_index, indices] = True

    def _feature_masks(
        self,
        state: torch.Tensor,
        previous: torch.Tensor,
        action: torch.Tensor,
        valid_state: torch.Tensor,
        valid_action: torch.Tensor,
    ) -> None:
        device = state.device
        low, high = self.feature_fraction
        for target, valid in (
            (state, valid_state),
            (previous, valid_state),
            (action, valid_action),
        ):
            width = target.shape[-1]
            if width == 0:
                continue
            count = max(1, round(width * self._fraction(low, high, device)))
            dimensions = torch.randperm(width, device=device)[:count]
            target[:, :, dimensions] = valid[:, :, None]
        # Occasionally preserve the physical meaning of base angular velocity or gravity.
        if state.shape[-1] == 70:
            semantic_group = (
                (61, 64)
                if bool(torch.randint(0, 2, (), device=device).item())
                else (64, 67)
            )
        else:
            semantic_group = (
                (58, 61)
                if bool(torch.randint(0, 2, (), device=device).item())
                else (61, 64)
            )
        state[:, :, semantic_group[0] : semantic_group[1]] |= valid_state[:, :, None]


def masked_inputs(
    batch: dict[str, torch.Tensor], masks: MaskBatch, full: bool = False
) -> dict[str, torch.Tensor]:
    if full:
        state = batch["physical_state"]
        previous = batch["previous_action"]
        action = batch["action"]
    else:
        state = batch["physical_state"].masked_fill(masks.state_input, 0.0)
        previous = batch["previous_action"].masked_fill(masks.previous_input, 0.0)
        action = batch["action"].masked_fill(masks.action_input, 0.0)
    return {
        **batch,
        "physical_state": state,
        "previous_action": previous,
        "action": action,
    }
