from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .constants import ACTION_DIM, COMPLETION_NAMES, TASK_NAMES


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
    forward_transition: torch.Tensor | None = None
    inverse_transition: torch.Tensor | None = None
    history_action_transition: torch.Tensor | None = None
    rollout_start: torch.Tensor | None = None
    rollout_horizon: torch.Tensor | None = None

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
            None if self.forward_transition is None else self.forward_transition.to(device),
            None if self.inverse_transition is None else self.inverse_transition.to(device),
            None if self.history_action_transition is None else self.history_action_transition.to(device),
            None if self.rollout_start is None else self.rollout_start.to(device),
            None if self.rollout_horizon is None else self.rollout_horizon.to(device),
        )


class MaskGenerator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.strategy = str(config.get("strategy", "legacy"))
        self.optimizer_step = 0
        self.task_probabilities = torch.tensor(
            config["task_probabilities"], dtype=torch.float32
        )
        self.completion_probabilities = torch.tensor(
            config["completion_probabilities"], dtype=torch.float32
        )
        self.element_fraction = tuple(float(x) for x in config["element_fraction"])
        self.step_count = tuple(int(x) for x in config["step_count"])
        self.feature_fraction = tuple(float(x) for x in config["feature_fraction"])
        self.relation_probabilities = torch.tensor(
            config.get("relation_probabilities", (0.40, 0.35, 0.25)),
            dtype=torch.float32,
        )
        self.forward_subprobabilities = torch.tensor(
            config.get("forward_subprobabilities", (0.25, 0.50, 0.25)),
            dtype=torch.float32,
        )
        self.calibration_steps = tuple(int(x) for x in config.get("calibration_steps", (16, 32)))
        self.physics_step_count = tuple(int(x) for x in config.get("physics_step_count", (1, 32)))
        self.state_step_count = tuple(
            int(x) for x in config.get("state_step_count", self.physics_step_count)
        )
        self.history_action_step_count = tuple(
            int(x) for x in config.get("history_action_step_count", self.physics_step_count)
        )
        self.action_inference_probabilities = torch.tensor(
            config.get("action_inference_probabilities", (4.0 / 7.0, 3.0 / 7.0)),
            dtype=torch.float32,
        )
        self.arbitrary_target_probabilities = torch.tensor(
            config.get("arbitrary_target_probabilities", (0.25, 0.25, 0.50)),
            dtype=torch.float32,
        )
        self.action_step_curriculum = tuple(
            {
                "until_step": int(item["until_step"]),
                "max_length": int(item["max_length"]),
            }
            for item in config.get("action_step_curriculum", ())
        )
        self.action_length_bucket_probabilities = torch.tensor(
            config.get("action_length_bucket_probabilities", (0.20, 0.30, 0.25, 0.25)),
            dtype=torch.float32,
        )
        self.inverse_full_probability = float(
            config.get("inverse_full_128_probability_after_25000", 0.0)
        )
        self.inverse_full_start_step = int(
            config.get("inverse_full_128_start_step", 25_000)
        )
        self.physics_element_fraction = tuple(
            float(x) for x in config.get("physics_element_fraction", (0.10, 0.50))
        )
        self.physics_feature_fraction = tuple(
            float(x) for x in config.get("physics_feature_fraction", (0.10, 0.50))
        )
        self.overlay_fraction = float(config.get("structured_overlay_max_fraction", 0.10))
        self.rollout_start_step = int(config.get("rollout_start_step", 20_000))
        if self.task_probabilities.shape != (3,) or not torch.isclose(
            self.task_probabilities.sum(), torch.tensor(1.0), atol=1e-5
        ):
            raise ValueError("task probabilities must contain three values summing to one")
        if self.completion_probabilities.shape != (3,) or not torch.isclose(
            self.completion_probabilities.sum(), torch.tensor(1.0), atol=1e-5
        ):
            raise ValueError("completion probabilities must contain three values summing to one")
        if self.relation_probabilities.shape != (3,) or not torch.isclose(
            self.relation_probabilities.sum(), torch.tensor(1.0), atol=1e-5
        ):
            raise ValueError("relation probabilities must contain three values summing to one")
        if self.forward_subprobabilities.shape != (3,) or not torch.isclose(
            self.forward_subprobabilities.sum(), torch.tensor(1.0), atol=1e-5
        ):
            raise ValueError("forward subprobabilities must contain three values summing to one")
        for name, probabilities, size in (
            ("action inference", self.action_inference_probabilities, 2),
            ("arbitrary target", self.arbitrary_target_probabilities, 3),
            ("Action length bucket", self.action_length_bucket_probabilities, 4),
        ):
            if probabilities.shape != (size,) or not torch.isclose(
                probabilities.sum(), torch.tensor(1.0), atol=1e-5
            ):
                raise ValueError(f"{name} probabilities must contain {size} values summing to one")
        for name, bounds in (
            ("state_step_count", self.state_step_count),
            ("history_action_step_count", self.history_action_step_count),
        ):
            if len(bounds) != 2 or bounds[0] < 1 or bounds[1] < bounds[0]:
                raise ValueError(f"{name} must be an increasing positive [low, high] pair")
        previous_until = 0
        previous_max = 0
        for stage in self.action_step_curriculum:
            if stage["until_step"] <= previous_until or stage["max_length"] < previous_max:
                raise ValueError("Action curriculum stages must increase in step and max_length")
            previous_until = stage["until_step"]
            previous_max = stage["max_length"]
        if not 0.0 <= self.inverse_full_probability <= 1.0:
            raise ValueError("inverse full-128 probability must be in [0, 1]")

    def set_step(self, optimizer_step: int) -> None:
        self.optimizer_step = int(optimizer_step)

    def curriculum_stage(self) -> dict[str, int] | None:
        if not self.action_step_curriculum:
            return None
        for index, stage in enumerate(self.action_step_curriculum):
            if self.optimizer_step <= stage["until_step"]:
                return {"index": index, **stage}
        return {"index": len(self.action_step_curriculum) - 1, **self.action_step_curriculum[-1]}

    def action_max_length(self) -> int:
        stage = self.curriculum_stage()
        return int(stage["max_length"]) if stage else int(self.physics_step_count[1])

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
        force_length: int | None = None,
        force_target: str | None = None,
        force_granularity: str | None = None,
    ) -> MaskBatch:
        if self.strategy == "physics_bidirectional_v1":
            return self._generate_physics(
                batch,
                force_task,
                force_completion,
                force_length,
                force_target,
                force_granularity,
            )
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
            torch.zeros_like(batch["valid_action"], dtype=torch.bool),
            torch.zeros_like(batch["valid_action"], dtype=torch.bool),
            torch.zeros_like(batch["valid_action"], dtype=torch.bool),
            torch.zeros(batch["physical_state"].shape[0], dtype=torch.long, device=device),
            torch.ones(batch["physical_state"].shape[0], dtype=torch.long, device=device),
        )

    @staticmethod
    def _random_span(valid_count: int, low: int, high: int, device: torch.device) -> tuple[int, int]:
        if valid_count <= 0:
            return 0, 0
        upper = min(max(int(high), 1), valid_count)
        lower = min(max(int(low), 1), upper)
        length = int(torch.randint(lower, upper + 1, (), device=device).item())
        start = int(torch.randint(0, valid_count - length + 1, (), device=device).item())
        return start, length

    def _overlay_random_elements(
        self,
        state_input: torch.Tensor,
        state_loss: torch.Tensor,
        action_input: torch.Tensor,
        action_loss: torch.Tensor,
        valid_state: torch.Tensor,
        valid_action: torch.Tensor,
    ) -> None:
        if self.overlay_fraction <= 0:
            return
        device = state_input.device
        state_fraction = float(torch.rand((), device=device).item()) * self.overlay_fraction
        action_fraction = float(torch.rand((), device=device).item()) * self.overlay_fraction
        state_extra = (
            torch.rand(state_input.shape, device=device) < state_fraction
        ) & valid_state[:, :, None]
        action_extra = (
            torch.rand(action_input.shape, device=device) < action_fraction
        ) & valid_action[:, :, None]
        state_input |= state_extra
        state_loss |= state_extra
        action_input |= action_extra
        action_loss |= action_extra

    def _action_span(
        self,
        valid_count: int,
        device: torch.device,
        force_length: int | None = None,
        history: bool = False,
        allow_full_inverse: bool = False,
    ) -> tuple[int, int]:
        if valid_count <= 0:
            return 0, 0
        if force_length is not None:
            length = min(max(int(force_length), 1), valid_count)
            start = int(torch.randint(0, valid_count - length + 1, (), device=device).item())
            return start, length
        if history:
            return self._random_span(
                valid_count,
                self.history_action_step_count[0],
                min(self.history_action_step_count[1], valid_count),
                device,
            )
        configured_maximum = self.action_max_length()
        if not self.action_step_curriculum and self.optimizer_step < self.rollout_start_step:
            configured_maximum = min(configured_maximum, 8)
        maximum = min(configured_maximum, valid_count)
        if (
            allow_full_inverse
            and valid_count >= 128
            and maximum >= 128
            and self.optimizer_step >= self.inverse_full_start_step
            and bool(torch.rand((), device=device) < self.inverse_full_probability)
        ):
            return self._random_span(valid_count, 128, 128, device)
        bucket_bounds = ((1, 8), (9, 32), (33, 64), (65, 128))
        available = [
            index for index, (low, _high) in enumerate(bucket_bounds) if low <= maximum
        ]
        if not available:
            return self._random_span(valid_count, 1, maximum, device)
        weights = self.action_length_bucket_probabilities.to(device)[available]
        bucket = available[int(torch.multinomial(weights / weights.sum(), 1).item())]
        low, high = bucket_bounds[bucket]
        return self._random_span(valid_count, low, min(high, maximum), device)

    def _generate_physics(
        self,
        batch: dict[str, torch.Tensor],
        force_task: str | None,
        force_completion: str | None,
        force_length: int | None,
        force_target: str | None,
        force_granularity: str | None,
    ) -> MaskBatch:
        del force_completion
        device = batch["physical_state"].device
        valid_state = batch["valid_state"].bool()
        valid_action = batch["valid_action"].bool()
        state_input, previous_input, action_input = self._empty(batch)
        state_loss, previous_loss, action_loss = self._empty(batch)
        forward_transition = torch.zeros_like(valid_action)
        inverse_transition = torch.zeros_like(valid_action)
        history_transition = torch.zeros_like(valid_action)
        batch_size = valid_state.shape[0]
        rollout_start = torch.zeros(batch_size, dtype=torch.long, device=device)
        rollout_horizon = torch.zeros(batch_size, dtype=torch.long, device=device)

        if force_task is None:
            group = int(torch.multinomial(self.relation_probabilities.to(device), 1).item())
            task_name = ("forward", "action_inference", "arbitrary")[group]
        else:
            aliases = {"inverse": "inverse", "completion": "arbitrary"}
            task_name = aliases.get(force_task, force_task)
            group = 0 if task_name.startswith("forward") else 1 if task_name in {
                "action_inference", "inverse", "history_action"
            } else 2

        causal = group == 0
        completion_name: str | None = None
        if group == 0:
            if task_name == "forward" or task_name == "action_inference":
                subtype = int(torch.multinomial(self.forward_subprobabilities.to(device), 1).item())
                if force_task is None and self.optimizer_step < self.rollout_start_step:
                    subtype = 0
                task_name = ("forward_one", "forward_rollout", "forward_cold")[subtype]
            else:
                subtype = {"forward_one": 0, "forward_rollout": 1, "forward_cold": 2}.get(
                    task_name, 0
                )
            for index in range(batch_size):
                count = int(valid_action[index].sum().item())
                if count <= 0:
                    continue
                if subtype == 0 or count == 1:
                    start = int(torch.randint(0, count, (), device=device).item())
                    horizon = 1
                elif subtype == 1:
                    low = min(self.calibration_steps[0], max(1, count - 1))
                    high = min(self.calibration_steps[1], max(1, count - 1))
                    prefix = int(torch.randint(low, high + 1, (), device=device).item())
                    max_horizon = min(self.state_step_count[1], count - prefix)
                    if max_horizon <= 0:
                        prefix, max_horizon = count - 1, 1
                    start = prefix
                    if force_task == "forward_rollout":
                        horizon = min(8, max_horizon)
                    else:
                        horizon = int(
                            torch.randint(1, max_horizon + 1, (), device=device).item()
                        )
                else:
                    start, horizon = 0, count
                state_input[index, start + 1 : start + horizon + 1] = True
                state_loss[index, start + 1 : start + horizon + 1] = True
                forward_transition[index, start : start + horizon] = True
                rollout_start[index] = start
                if self.optimizer_step >= self.rollout_start_step and subtype in {1, 2}:
                    if force_task == "forward_rollout":
                        rollout_horizon[index] = min(8, horizon)
                    else:
                        choices = [value for value in (2, 4, 8) if value <= horizon]
                        if choices:
                            choice = int(
                                torch.randint(0, len(choices), (), device=device).item()
                            )
                            rollout_horizon[index] = choices[choice]
            self._overlay_random_elements(
                state_input, state_loss, action_input, action_loss, valid_state, valid_action
            )
            # S_t and A_t are the declared inputs of every supervised forward
            # transition. The overlay may hide unrelated context, but never
            # either boundary input of the dynamics head.
            for index in range(batch_size):
                targets = torch.nonzero(forward_transition[index], as_tuple=False).flatten()
                if targets.numel():
                    state_input[index, targets] = False
                    state_loss[index, targets] = False
            # The controls whose effects are supervised must remain observable.
            action_input &= ~forward_transition[:, :, None]
            action_loss &= ~forward_transition[:, :, None]
        elif group == 1:
            if task_name == "action_inference":
                subtype = int(
                    torch.multinomial(self.action_inference_probabilities.to(device), 1).item()
                )
                task_name = ("inverse", "history_action")[subtype]
            causal = task_name == "history_action"
            for index in range(batch_size):
                count = int(valid_action[index].sum().item())
                start, length = self._action_span(
                    count,
                    device,
                    force_length=force_length,
                    history=task_name == "history_action",
                    allow_full_inverse=task_name == "inverse",
                )
                if length == 0:
                    continue
                action_input[index, start : start + length] = True
                action_loss[index, start : start + length] = True
                if task_name == "history_action":
                    history_transition[index, start : start + length] = True
                    state_input[index, start + 1 : count + 1] = True
                else:
                    inverse_transition[index, start : start + length] = True
            self._overlay_random_elements(
                state_input, state_loss, action_input, action_loss, valid_state, valid_action
            )
            # Inverse dynamics requires both transition boundary States.
            if task_name == "inverse":
                for index in range(batch_size):
                    targets = torch.nonzero(inverse_transition[index], as_tuple=False).flatten()
                    if targets.numel():
                        state_input[index, targets] = False
                        state_input[index, targets + 1] = False
                        state_loss[index, targets] = False
                        state_loss[index, targets + 1] = False
        else:
            task_name = "arbitrary"
            completion_name = force_granularity or "mixed"
            for index in range(batch_size):
                if force_target is None:
                    target_choice = int(
                        torch.multinomial(
                            self.arbitrary_target_probabilities.to(device), 1
                        ).item()
                    )
                else:
                    target_choice = {"state": 0, "action": 1, "both": 2}[force_target]
                mask_state = target_choice != 1
                mask_action = target_choice != 0
                granularity = (
                    int(torch.randint(0, 4, (), device=device).item())
                    if force_granularity is None
                    else {"element": 0, "step": 1, "feature": 2, "semantic": 3}[
                        force_granularity
                    ]
                )
                if granularity == 0:
                    if mask_state:
                        fraction = self._fraction(*self.physics_element_fraction, device)
                        selected = (torch.rand_like(state_input[index], dtype=torch.float32) < fraction)
                        state_input[index] |= selected & valid_state[index, :, None]
                    if mask_action:
                        fraction = self._fraction(*self.physics_element_fraction, device)
                        selected = (torch.rand_like(action_input[index], dtype=torch.float32) < fraction)
                        action_input[index] |= selected & valid_action[index, :, None]
                elif granularity == 1:
                    count = int(valid_action[index].sum().item())
                    if mask_state:
                        state_maximum = self.state_step_count[1]
                        if (
                            not self.action_step_curriculum
                            and self.optimizer_step < self.rollout_start_step
                        ):
                            state_maximum = min(state_maximum, 8)
                        state_start, state_length = self._random_span(
                            count,
                            self.state_step_count[0],
                            min(state_maximum, count),
                            device,
                        )
                        if force_length is not None:
                            state_length = min(max(int(force_length), 1), count)
                            state_start = int(
                                torch.randint(
                                    0, count - state_length + 1, (), device=device
                                ).item()
                            )
                        state_input[
                            index, state_start : state_start + state_length + 1
                        ] = True
                    if mask_action:
                        action_start, action_length = self._action_span(
                            count, device, force_length=force_length
                        )
                        action_input[
                            index, action_start : action_start + action_length
                        ] = True
                elif granularity == 2:
                    if mask_state:
                        width = state_input.shape[-1]
                        count = max(1, round(width * self._fraction(*self.physics_feature_fraction, device)))
                        dims = torch.randperm(width, device=device)[:count]
                        state_input[index, :, dims] = valid_state[index, :, None]
                    if mask_action:
                        count = max(1, round(ACTION_DIM * self._fraction(*self.physics_feature_fraction, device)))
                        dims = torch.randperm(ACTION_DIM, device=device)[:count]
                        action_input[index, :, dims] = valid_action[index, :, None]
                else:
                    joint_groups = ((0, 6), (6, 12), (12, 15), (15, 22), (22, 29))
                    semantic = int(torch.randint(0, 9, (), device=device).item())
                    if semantic < len(joint_groups):
                        low, high = joint_groups[semantic]
                        if mask_state:
                            state_input[index, :, low:high] = valid_state[index, :, None]
                            state_input[index, :, 29 + low : 29 + high] = valid_state[index, :, None]
                        if mask_action:
                            action_input[index, :, low:high] = valid_action[index, :, None]
                    elif mask_state:
                        low, high = ((58, 61), (61, 64), (64, 67), (68, 70))[semantic - 5]
                        state_input[index, :, low:high] = valid_state[index, :, None]
            state_loss.copy_(state_input)
            action_loss.copy_(action_input)

        state_input &= valid_state[:, :, None]
        action_input &= valid_action[:, :, None]
        state_loss |= state_input
        action_loss |= action_input
        state_loss &= valid_state[:, :, None]
        action_loss &= valid_action[:, :, None]
        if task_name == "history_action":
            # Future States are hidden solely to enforce causality. They are
            # not reconstruction targets for the history-Action objective.
            for index in range(batch_size):
                targets = torch.nonzero(history_transition[index], as_tuple=False).flatten()
                if targets.numel():
                    state_loss[index, int(targets.min().item()) + 1 :] = False
        return MaskBatch(
            state_input,
            previous_input,
            action_input,
            state_loss,
            previous_loss,
            action_loss,
            0 if group == 0 else 1 if group == 1 else 2,
            task_name,
            completion_name,
            causal,
            forward_transition,
            inverse_transition,
            history_transition,
            rollout_start,
            rollout_horizon,
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
