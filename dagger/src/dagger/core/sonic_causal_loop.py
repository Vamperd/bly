"""Single-step SONIC-like DAgger rollout loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dagger.runtime.sonic_causal_window import SonicCausalWindowBuffer


class SonicCausalDaggerLoop:
    def __init__(
        self,
        *,
        env: Any,
        teacher: Any,
        student: Any,
        action_contract: Any,
        num_envs: int,
        rollout_steps: int,
        run_dir: str | Path,
        seed: int,
    ) -> None:
        self.env = env
        self.teacher = teacher
        self.student = student
        self.action_contract = action_contract
        self.num_envs = int(num_envs)
        self.rollout_steps = int(rollout_steps)
        self.run_dir = Path(run_dir)
        self.seed = int(seed)
        self.samples: list[dict[str, torch.Tensor]] = []

    @staticmethod
    def _teacher_observation(observation: dict[str, Any]) -> dict[str, Any]:
        """Keep the student-only reference adapter fields out of SONIC."""

        return {
            key: value
            for key, value in observation.items()
            if not str(key).startswith("student_")
        }

    def _motion_ids(self) -> torch.Tensor:
        motion = self.env.command_manager.get_term("motion")
        value = getattr(motion, "motion_ids", None)
        if value is None:
            return torch.full(
                (self.num_envs,), -1, device=self.env.device, dtype=torch.long
            )
        return value.detach().clone().to(device=self.env.device, dtype=torch.long)

    def _motion_key(self, motion_id: int) -> str:
        keys = getattr(self.env, "motion_keys", None) or []
        return str(keys[motion_id]) if 0 <= motion_id < len(keys) else f"motion_{motion_id}"

    def _initial_canonical_action(self) -> torch.Tensor | None:
        actual_env = getattr(getattr(self.env, "env", self.env), "env", None)
        manager = getattr(actual_env, "action_manager", None)
        previous = getattr(manager, "prev_action", None)
        if not isinstance(previous, torch.Tensor):
            return None
        if tuple(previous.shape) != (self.num_envs, 29):
            return None
        return self.action_contract.raw_to_canonical(previous.detach())

    def collect_round(self, round_index: int, beta: float) -> list[dict[str, torch.Tensor]]:
        observation = self.env.reset()
        current_state = self.env.physics_state()
        self.teacher.reset(self.num_envs)
        self.student.reset(
            self.num_envs,
            current_state,
            initial_action=self._initial_canonical_action(),
        )
        generator = torch.Generator(device="cpu").manual_seed(
            self.seed + int(round_index) * 1009
        )
        done = torch.zeros(self.num_envs, device=self.env.device, dtype=torch.bool)
        collected: list[dict[str, torch.Tensor]] = []
        for step_index in range(self.rollout_steps):
            reference_future = observation.get("student_reference_future")
            if reference_future is None:
                raise RuntimeError("SONIC environment did not expose student_reference_future")
            rollout = self.student.observe(reference_future)
            teacher_raw = self.action_contract.clip_raw(
                self.teacher.act(self._teacher_observation(observation), done).detach()
            )
            teacher_canonical = self.action_contract.raw_to_canonical(teacher_raw)
            teacher_mask = (
                torch.rand(self.num_envs, generator=generator) < float(beta)
            ).to(self.env.device)
            executed_raw = torch.where(
                teacher_mask[:, None], teacher_raw, rollout.action_raw
            )
            next_observation, _reward, next_done, _info = self.env.step(
                {"actions": executed_raw}
            )
            next_done = next_done.to(self.env.device).bool()
            next_state = self.env.physics_state()
            teacher_normalized = (
                teacher_canonical - self.student.action_mean
            ) / self.student.action_std.clamp_min(1.0e-6)
            student_normalized = (
                rollout.action_canonical - self.student.action_mean
            ) / self.student.action_std.clamp_min(1.0e-6)
            executed_canonical = self.action_contract.raw_to_canonical(executed_raw)
            motion_ids = self._motion_ids()
            for index in range(self.num_envs):
                sample = {
                    key: value[index].detach().cpu().clone()
                    for key, value in rollout.batch.items()
                    if isinstance(value, torch.Tensor)
                }
                sample["target_action"] = teacher_normalized[index].detach().cpu()
                sample["teacher_action"] = teacher_normalized[index].detach().cpu()
                sample["student_action"] = student_normalized[index].detach().cpu()
                sample["teacher_action_canonical"] = teacher_canonical[index].detach().cpu()
                sample["student_action_canonical"] = rollout.action_canonical[index].detach().cpu()
                sample["student_requested_action_canonical"] = rollout.requested_action_canonical[index].detach().cpu()
                sample["executed_action_canonical"] = executed_canonical[index].detach().cpu()
                sample["teacher_action_raw"] = teacher_raw[index].detach().cpu()
                sample["student_action_raw"] = rollout.action_raw[index].detach().cpu()
                sample["executed_action_raw"] = executed_raw[index].detach().cpu()
                sample["teacher_executed"] = teacher_mask[index].detach().cpu()
                sample["done"] = next_done[index].detach().cpu()
                sample["motion_id"] = motion_ids[index].detach().cpu()
                sample["target_step"] = torch.tensor(step_index, dtype=torch.long)
                collected.append(sample)
            self.student.after_step(executed_canonical, next_state, next_done)
            observation = next_observation
            done = next_done
            if (step_index + 1) % 100 == 0 or step_index + 1 == self.rollout_steps:
                print(
                    f"[sonic-causal] collect round={round_index} "
                    f"step={step_index + 1}/{self.rollout_steps}",
                    flush=True,
                )
        self.samples.extend(collected)
        return collected

    @torch.no_grad()
    def evaluate_student_only(self, steps: int) -> dict[str, Any]:
        observation = self.env.reset()
        current_state = self.env.physics_state()
        self.student.set_latent_mode("fixed_standard_normal")
        self.student.reset(
            self.num_envs,
            current_state,
            initial_action=self._initial_canonical_action(),
        )
        self.teacher.reset(self.num_envs)
        done = torch.zeros(self.num_envs, device=self.env.device, dtype=torch.bool)
        canonical_sse = 0.0
        raw_sse = 0.0
        canonical_joint_sse = torch.zeros(29, device=self.env.device)
        raw_joint_sse = torch.zeros(29, device=self.env.device)
        clip_sse = 0.0
        clip_count = 0
        count = 0
        done_count = 0
        saturation_count = 0
        action_count = 0
        episode_steps = torch.zeros(self.num_envs, device=self.env.device)
        per_motion: dict[str, dict[str, float]] = {}
        for _ in range(int(steps)):
            reference_future = observation.get("student_reference_future")
            if reference_future is None:
                raise RuntimeError("SONIC environment did not expose student_reference_future")
            rollout = self.student.observe(reference_future)
            teacher_raw = self.action_contract.clip_raw(
                self.teacher.act(self._teacher_observation(observation), done).detach()
            )
            teacher_canonical = self.action_contract.raw_to_canonical(teacher_raw)
            canonical_error = rollout.action_canonical.float() - teacher_canonical.float()
            raw_error = rollout.action_raw.float() - teacher_raw.float()
            canonical_sse += float(canonical_error.square().sum())
            raw_sse += float(raw_error.square().sum())
            canonical_joint_sse += canonical_error.square().sum(dim=0)
            raw_joint_sse += raw_error.square().sum(dim=0)
            clip_delta = rollout.requested_action_canonical - rollout.action_canonical
            clip_sse += float(clip_delta.square().sum())
            clip_count += int(clip_delta.numel())
            count += int(canonical_error.numel())
            motion_ids = self._motion_ids()
            for index in range(self.num_envs):
                key = self._motion_key(int(motion_ids[index]))
                row = per_motion.setdefault(key, {"sse": 0.0, "steps": 0.0, "terminations": 0.0})
                row["sse"] += float(canonical_error[index].square().mean())
                row["steps"] += 1.0
            next_observation, _reward, next_done, _info = self.env.step(
                {"actions": rollout.action_raw}
            )
            next_done = next_done.to(self.env.device).bool()
            episode_steps += 1
            done_count += int(next_done.sum())
            for index in next_done.nonzero(as_tuple=False).flatten().tolist():
                key = self._motion_key(int(motion_ids[index]))
                per_motion[key]["terminations"] += 1.0
                episode_steps[index] = 0
            low = self.action_contract.raw_low
            high = self.action_contract.raw_high
            if torch.isfinite(low).all() and torch.isfinite(high).all():
                saturation_count += int(
                    ((rollout.action_raw <= low + 1e-6) | (rollout.action_raw >= high - 1e-6)).sum()
                )
                action_count += int(rollout.action_raw.numel())
            self.student.after_step(
                self.action_contract.raw_to_canonical(rollout.action_raw),
                self.env.physics_state(),
                next_done,
            )
            observation = next_observation
            done = next_done
        survival_ratio = float((episode_steps >= float(steps)).float().mean())
        return {
            "steps": float(steps),
            "num_envs": float(self.num_envs),
            "teacher_action_rmse": (canonical_sse / max(1, count)) ** 0.5,
            "teacher_action_rmse_raw": (raw_sse / max(1, count)) ** 0.5,
            "termination_rate": done_count / max(1, int(steps) * self.num_envs),
            "survival_ratio": survival_ratio,
            "action_saturation_ratio": saturation_count / max(1, action_count),
            "clip_error_rmse_canonical": (clip_sse / max(1, clip_count)) ** 0.5,
            "per_joint_rmse_canonical": torch.sqrt(
                canonical_joint_sse / max(1, int(steps) * self.num_envs)
            ).detach().cpu().tolist(),
            "per_joint_rmse_raw": torch.sqrt(
                raw_joint_sse / max(1, int(steps) * self.num_envs)
            ).detach().cpu().tolist(),
            "teacher_action_rmse_by_motion": {
                key: {
                    "teacher_action_rmse": (row["sse"] / max(1.0, row["steps"])) ** 0.5,
                    "steps": row["steps"],
                    "terminations": row["terminations"],
                    "termination_rate": row["terminations"] / max(1.0, row["steps"]),
                }
                for key, row in per_motion.items()
            },
        }

