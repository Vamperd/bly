"""Model-independent DAgger rollout and aggregation loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .contracts import DaggerRound, StudentAdapter, TeacherAdapter
from .dataset import AggregatedDataset


class DaggerLoop:
    def __init__(
        self,
        *,
        env: Any,
        teacher: TeacherAdapter,
        student: StudentAdapter,
        num_envs: int,
        rollout_steps: int,
        action_clip_value: float | None,
        run_dir: str | Path,
        seed: int,
        action_low: torch.Tensor | None = None,
        action_high: torch.Tensor | None = None,
    ) -> None:
        self.env = env
        self.teacher = teacher
        self.student = student
        self.num_envs = int(num_envs)
        self.rollout_steps = int(rollout_steps)
        self.action_clip_value = action_clip_value
        self.run_dir = Path(run_dir)
        self.seed = int(seed)
        if (action_low is None) != (action_high is None):
            raise ValueError("action_low and action_high must be provided together")
        device = getattr(self.env, "device", torch.device("cpu"))
        self.action_low = None if action_low is None else action_low.to(device, dtype=torch.float32)
        self.action_high = None if action_high is None else action_high.to(device, dtype=torch.float32)
        self.dataset = AggregatedDataset()

    def collect_round(self, round_index: int, beta: float) -> DaggerRound:
        observation = self.env.reset()
        self.teacher.reset(self.num_envs)
        self.student.reset(self.num_envs)
        teacher_mask = torch.rand(self.num_envs, device=self.env.device) < float(beta)
        done = torch.zeros(self.num_envs, device=self.env.device, dtype=torch.bool)
        episode_id = torch.arange(self.num_envs, device=self.env.device, dtype=torch.long)
        next_episode_id = self.num_envs
        timestep = torch.zeros(self.num_envs, device=self.env.device, dtype=torch.long)
        buffers: dict[str, list[torch.Tensor]] = {key: [] for key in (
            "student_obs",
            "teacher_action",
            "student_action",
            "executed_action",
            "done",
            "episode_id",
            "timestep",
            "motion_id",
            "mixture_teacher",
        )}

        for _ in range(self.rollout_steps):
            if done.any():
                done_ids = done.nonzero(as_tuple=False).flatten()
                teacher_mask[done_ids] = torch.rand(
                    len(done_ids), device=self.env.device
                ) < float(beta)
                episode_id[done_ids] = torch.arange(
                    next_episode_id,
                    next_episode_id + len(done_ids),
                    device=self.env.device,
                )
                next_episode_id += len(done_ids)
                timestep[done_ids] = 0

            student_obs = self.student.observe(observation)
            teacher_action = self.teacher.act(observation, done)
            student_action = self.student.act(student_obs)
            if teacher_action.shape != student_action.shape:
                raise ValueError(
                    "teacher/student action shape mismatch: "
                    f"teacher={tuple(teacher_action.shape)}, "
                    f"student={tuple(student_action.shape)}"
                )
            executed_action = torch.where(
                teacher_mask[:, None],
                self._clip(teacher_action),
                self._clip(student_action),
            )
            if not torch.isfinite(executed_action).all():
                raise FloatingPointError("non-finite executed DAgger action")
            motion_id = self._motion_ids()
            next_observation, _reward, next_done, _info = self.env.step(
                {"actions": executed_action}
            )
            next_done = next_done.to(device=self.env.device).bool()

            buffers["student_obs"].append(student_obs.detach().cpu())
            buffers["teacher_action"].append(teacher_action.detach().cpu())
            buffers["student_action"].append(student_action.detach().cpu())
            buffers["executed_action"].append(executed_action.detach().cpu())
            buffers["done"].append(next_done.cpu())
            buffers["episode_id"].append(episode_id.cpu())
            buffers["timestep"].append(timestep.cpu())
            buffers["motion_id"].append(motion_id.cpu())
            buffers["mixture_teacher"].append(teacher_mask.cpu())
            timestep += 1
            done = next_done
            observation = next_observation

        result = DaggerRound(
            student_obs=torch.cat(buffers["student_obs"]),
            teacher_action=torch.cat(buffers["teacher_action"]),
            student_action=torch.cat(buffers["student_action"]),
            executed_action=torch.cat(buffers["executed_action"]),
            done=torch.cat(buffers["done"]),
            episode_id=torch.cat(buffers["episode_id"]),
            timestep=torch.cat(buffers["timestep"]),
            motion_id=torch.cat(buffers["motion_id"]),
            mixture_teacher=torch.cat(buffers["mixture_teacher"]),
        )
        self.dataset.append(result)
        self.dataset.save_round(round_index, self.run_dir)
        return result

    def train_student(self) -> dict[str, float]:
        metrics = self.student.train_on_dataset(self.dataset)
        metrics["dataset_samples"] = float(self.dataset.count)
        return metrics

    def evaluate_student_only(self, steps: int) -> dict[str, Any]:
        observation = self.env.reset()
        self.student.reset(self.num_envs)
        self.teacher.reset(self.num_envs)
        done = torch.zeros(self.num_envs, device=self.env.device, dtype=torch.bool)
        episode_steps = torch.zeros(self.num_envs, device=self.env.device, dtype=torch.float32)
        episode_lengths: list[float] = []
        done_count = 0
        reward_sum = 0.0
        saturation_count = 0
        action_count = 0
        sse = 0.0
        element_count = 0
        per_motion: dict[str, dict[str, float]] = {}
        for _ in range(int(steps)):
            motion_ids = self._motion_ids()
            student_obs = self.student.observe(observation)
            student_action = self.student.act(student_obs)
            teacher_action = self.teacher.act(observation, done)
            if teacher_action.shape != student_action.shape:
                raise ValueError(
                    "teacher/student action shape mismatch during evaluation: "
                    f"teacher={tuple(teacher_action.shape)}, "
                    f"student={tuple(student_action.shape)}"
                )
            error = student_action.float() - teacher_action.float()
            squared = error.square()
            sse += float(squared.sum().item())
            element_count += int(squared.numel())
            env_mse = squared.mean(dim=-1)
            for index in range(self.num_envs):
                key = self._motion_key(int(motion_ids[index].item()))
                record = per_motion.setdefault(
                    key, {"sse": 0.0, "elements": 0.0, "steps": 0.0, "terminations": 0.0}
                )
                record["sse"] += float(env_mse[index].item())
                record["elements"] += 1.0
                record["steps"] += 1.0
            action = self._clip(student_action)
            if not torch.isfinite(action).all():
                raise FloatingPointError("non-finite evaluation action")
            next_observation, reward, next_done, _info = self.env.step({"actions": action})
            next_done = next_done.to(device=self.env.device).bool()
            episode_steps += 1.0
            done_indices = next_done.nonzero(as_tuple=False).flatten()
            for index in done_indices.tolist():
                episode_lengths.append(float(episode_steps[index].item()))
                episode_steps[index] = 0.0
                key = self._motion_key(int(motion_ids[index].item()))
                per_motion[key]["terminations"] += 1.0
            done_count += int(next_done.sum().item())
            reward_sum += float(reward.detach().float().mean().cpu())
            if self.action_low is not None and self.action_high is not None:
                saturation_count += int(
                    (
                        (action <= self.action_low + 1.0e-6)
                        | (action >= self.action_high - 1.0e-6)
                    ).sum().item()
                )
                action_count += int(action.numel())
            elif self.action_clip_value is not None and self.action_clip_value > 0:
                saturation_count += int((action.abs() >= self.action_clip_value).sum().item())
                action_count += int(action.numel())
            observation = next_observation
            done = next_done
        by_motion = {}
        for key, record in per_motion.items():
            denominator = max(record["elements"], 1.0)
            by_motion[key] = {
                "teacher_action_rmse": (record["sse"] / denominator) ** 0.5,
                "steps": record["steps"],
                "terminations": record["terminations"],
                "termination_rate": record["terminations"] / max(record["steps"], 1.0),
            }
        return {
            "steps": float(steps),
            "num_envs": float(self.num_envs),
            "done_count": float(done_count),
            "mean_reward": reward_sum / max(1, int(steps)),
            "teacher_action_rmse": (sse / element_count) ** 0.5 if element_count else None,
            "teacher_action_rmse_by_motion": by_motion,
            "termination_rate": done_count / max(1, int(steps) * self.num_envs),
            "mean_episode_length": (
                sum(episode_lengths) / len(episode_lengths)
                if episode_lengths
                else float(steps)
            ),
            "action_saturation_ratio": saturation_count / max(1, action_count),
        }

    def _clip(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_low is not None and self.action_high is not None:
            low = self.action_low
            high = self.action_high
            if low.shape[-1] != action.shape[-1]:
                low = low[..., : action.shape[-1]]
                high = high[..., : action.shape[-1]]
            return torch.maximum(torch.minimum(action, high), low)
        if self.action_clip_value is None or self.action_clip_value <= 0:
            return action
        return torch.clamp(action, -self.action_clip_value, self.action_clip_value)

    def _motion_key(self, motion_id: int) -> str:
        keys = getattr(self.env, "motion_keys", None) or []
        if 0 <= motion_id < len(keys):
            return str(keys[motion_id])
        return f"motion_{motion_id}"

    def _motion_ids(self) -> torch.Tensor:
        motion = self.env.command_manager.get_term("motion")
        value = getattr(motion, "motion_ids", None)
        if value is None:
            return torch.full(
                (self.num_envs,), -1, device=self.env.device, dtype=torch.long
            )
        return value.detach().clone().to(device=self.env.device, dtype=torch.long)

