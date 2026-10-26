"""Isaac-in-the-loop DAgger collection for the 65-token CVAE student."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import torch

from dagger.core.cvae_dataset import CvaeDaggerDataset, CvaeDaggerRound
from dagger.runtime.cvae_window import build_training_window


class CvaeDaggerLoop:
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
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        dataset: CvaeDaggerDataset | None = None,
        motion_group: list[str] | tuple[str, ...] | None = None,
        mixture_mode: str = "per_episode",
    ) -> None:
        if mixture_mode not in {"per_step", "per_episode"}:
            raise ValueError("mixture_mode must be per_step or per_episode")
        self.env = env
        self.teacher = teacher
        self.student = student
        self.action_contract = action_contract
        self.num_envs = int(num_envs)
        self.rollout_steps = int(rollout_steps)
        self.run_dir = Path(run_dir)
        self.seed = int(seed)
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.dataset = dataset if dataset is not None else CvaeDaggerDataset()
        self.motion_group = list(motion_group or [])
        self.mixture_mode = str(mixture_mode)

    def _motion_ids(self) -> torch.Tensor:
        motion = self.env.command_manager.get_term("motion")
        value = getattr(motion, "motion_ids", None)
        if value is None:
            return torch.full(
                (self.num_envs,), -1, device=self.env.device, dtype=torch.long
            )
        return value.detach().clone().to(device=self.env.device, dtype=torch.long)

    def _motion_key(self, motion_id: int) -> str:
        keys = getattr(self.env, "motion_keys", None) or self.motion_group
        index = int(motion_id)
        if 0 <= index < len(keys):
            return str(keys[index])
        return f"motion_{index}"

    @staticmethod
    def _new_record(state: torch.Tensor, episode_id: int = 0) -> dict[str, list[torch.Tensor]]:
        return {
            "states": [state.detach().cpu().clone()],
            "state_valid": [torch.tensor(True, dtype=torch.bool)],
            "executed_canonical": [],
            "teacher_canonical": [],
            "student_canonical": [],
            "teacher_raw": [],
            "student_raw": [],
            "executed_raw": [],
            "done": [],
            "motion_id": [],
            "mixture_teacher": [],
            "tracking_error_bucket": [],
            "disturbance_seed": [],
            "episode_id": [],
            "predicted_state": [],
            "predicted_action": [],
            "latent_epsilon_hash": [],
            "student_input_hash": [],
            "history_steps": [],
        }

    def _make_samples(self, record: dict[str, list[torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        states = torch.stack(record["states"])
        if not record["executed_canonical"]:
            return []
        actions = torch.stack(record["executed_canonical"])
        teachers = torch.stack(record["teacher_canonical"])
        students = torch.stack(record["student_canonical"])
        samples: list[dict[str, torch.Tensor]] = []
        for step in range(actions.shape[0]):
            sample = build_training_window(
                states,
                actions,
                teachers[step],
                students[step],
                step=step,
                state_mean=self.state_mean,
                state_std=self.state_std,
                action_mean=self.action_mean,
                action_std=self.action_std,
                state_valid=torch.stack(record["state_valid"]),
            )
            for source, target in (
                ("teacher_raw", "teacher_action_raw"),
                ("student_raw", "student_action_raw"),
                ("executed_raw", "executed_action_raw"),
            ):
                sample[target] = record[source][step].detach().clone()
            sample["done"] = record["done"][step].detach().clone()
            sample["motion_id"] = record["motion_id"][step].detach().clone()
            sample["mixture_teacher"] = record["mixture_teacher"][step].detach().clone()
            sample["tracking_error_bucket"] = record["tracking_error_bucket"][step].detach().clone()
            sample["disturbance_seed"] = record["disturbance_seed"][step].detach().clone()
            sample["episode_id"] = record["episode_id"][step].detach().clone()
            sample["executed_action_canonical"] = record["executed_canonical"][step].detach().clone()
            sample["predicted_state"] = record["predicted_state"][step].detach().clone()
            sample["predicted_action"] = record["predicted_action"][step].detach().clone()
            sample["latent_epsilon_hash"] = record["latent_epsilon_hash"][step].detach().clone()
            sample["student_input_hash"] = record["student_input_hash"][step].detach().clone()
            sample["history_steps"] = record["history_steps"][step].detach().clone()
            state_ratio = (sample["physical_state"].abs() > 6.0).float().mean()
            action_ratio = (sample["action"].abs() > 6.0).float().mean()
            sample["normalization_out_of_range"] = (state_ratio + action_ratio) * 0.5
            sample["timestep"] = torch.tensor(step, dtype=torch.long)
            samples.append(sample)
        return samples

    @staticmethod
    def _stack_samples(samples: list[dict[str, torch.Tensor]]) -> CvaeDaggerRound:
        if not samples:
            raise RuntimeError("DAgger rollout produced no valid CVAE samples")
        keys = samples[0].keys()
        tensors = {key: torch.stack([sample[key] for sample in samples]) for key in keys}
        return CvaeDaggerRound(tensors)

    def collect_round(self, round_index: int, beta: float) -> CvaeDaggerRound:
        observation = self.env.reset()
        current_state = self.env.physics_state()
        if hasattr(self.student, "end_evaluation"):
            self.student.end_evaluation()
        self.teacher.reset(self.num_envs)
        self.student.reset(self.num_envs, current_state)
        random = torch.Generator(device="cpu")
        random.manual_seed(self.seed + int(round_index) * 1009)
        episode_numbers = list(range(self.num_envs))
        episode_records = [
            self._new_record(current_state[index], episode_numbers[index])
            for index in range(self.num_envs)
        ]
        all_samples: list[dict[str, torch.Tensor]] = []
        done = torch.zeros(self.num_envs, device=self.env.device, dtype=torch.bool)
        teacher_mask = (
            torch.rand(self.num_envs, generator=random) < float(beta)
        ).to(self.env.device)

        for _ in range(self.rollout_steps):
            student_rollout = self.student.observe()
            teacher_raw = self.action_contract.clip_raw(
                self.teacher.act(observation, done).detach()
            )
            teacher_canonical = self.action_contract.raw_to_canonical(teacher_raw)
            student_raw = student_rollout.action_raw.detach()
            student_canonical = student_rollout.action_canonical.detach()
            if self.mixture_mode == "per_step":
                teacher_mask = (
                    torch.rand(self.num_envs, generator=random) < float(beta)
                ).to(self.env.device)
            executed_raw = torch.where(teacher_mask[:, None], teacher_raw, student_raw)
            executed_canonical = self.action_contract.raw_to_canonical(executed_raw)
            if teacher_raw.shape != student_raw.shape or tuple(teacher_raw.shape[-1:]) != (29,):
                raise ValueError(
                    f"teacher/student Action shape mismatch: teacher={tuple(teacher_raw.shape)}, "
                    f"student={tuple(student_raw.shape)}"
                )
            next_observation, _reward, next_done, _info = self.env.step(
                {"actions": executed_raw}
            )
            next_done = next_done.to(device=self.env.device).bool()
            next_state = self.env.physics_state()
            motion_ids = self._motion_ids()
            history_steps = (
                student_rollout.window.history_steps
                if student_rollout.window.history_steps is not None
                else torch.zeros(self.num_envs, dtype=torch.long, device=self.env.device)
            )

            for index in range(self.num_envs):
                record = episode_records[index]
                record["executed_canonical"].append(executed_canonical[index].cpu())
                record["teacher_canonical"].append(teacher_canonical[index].cpu())
                record["student_canonical"].append(student_canonical[index].cpu())
                record["teacher_raw"].append(teacher_raw[index].cpu())
                record["student_raw"].append(student_raw[index].cpu())
                record["executed_raw"].append(executed_raw[index].cpu())
                record["done"].append(next_done[index].cpu())
                record["motion_id"].append(motion_ids[index].cpu())
                record["mixture_teacher"].append(teacher_mask[index].cpu())
                record["history_steps"].append(history_steps[index].cpu())
                action_error = torch.linalg.vector_norm(
                    teacher_canonical[index] - student_canonical[index]
                )
                bucket = 0 if float(action_error) < 0.1 else (1 if float(action_error) < 0.5 else 2)
                record["tracking_error_bucket"].append(torch.tensor(bucket, dtype=torch.long))
                record["disturbance_seed"].append(torch.tensor(self.seed, dtype=torch.long))
                record["episode_id"].append(torch.tensor(episode_numbers[index], dtype=torch.long))
                record["predicted_state"].append(student_rollout.predicted_state[index].cpu())
                record["predicted_action"].append(student_rollout.predicted_action[index].cpu())
                latent_bytes = (
                    student_rollout.epsilon_global[index].detach().cpu().numpy().tobytes()
                    + student_rollout.epsilon_local[index].detach().cpu().numpy().tobytes()
                )
                visible_bytes = (
                    student_rollout.window.batch["physical_state"][index].detach().cpu().numpy().tobytes()
                    + student_rollout.window.batch["action"][index].detach().cpu().numpy().tobytes()
                )
                record["latent_epsilon_hash"].append(
                    torch.tensor(int(hashlib.sha1(latent_bytes).hexdigest()[:15], 16), dtype=torch.long)
                )
                record["student_input_hash"].append(
                    torch.tensor(int(hashlib.sha1(visible_bytes).hexdigest()[:15], 16), dtype=torch.long)
                )
                record["states"].append(next_state[index].cpu())
                record["state_valid"].append((~next_done[index]).cpu())
                if bool(next_done[index]):
                    all_samples.extend(self._make_samples(record))
                    episode_numbers[index] += self.num_envs
                    episode_records[index] = self._new_record(
                        next_state[index].cpu(), episode_numbers[index]
                    )

            if self.mixture_mode == "per_episode" and bool(next_done.any()):
                resampled = (
                    torch.rand(self.num_envs, generator=random) < float(beta)
                ).to(self.env.device)
                teacher_mask = torch.where(next_done, resampled, teacher_mask)

            self.student.after_step(executed_canonical, next_state, next_done)
            observation = next_observation
            done = next_done

        for record in episode_records:
            all_samples.extend(self._make_samples(record))
        round_data = self._stack_samples(all_samples)
        round_data.metadata = {
            "round_index": int(round_index),
            "beta": float(beta),
            "motion_keys": list(getattr(self.env, "motion_keys", None) or self.motion_group),
            "motion_group": list(self.motion_group),
            "sample_count": int(round_data.count),
        }
        self.dataset.append(round_data)
        self.dataset.save_round(round_index, self.run_dir)
        return round_data

    @torch.no_grad()
    def evaluate_student_only(
        self,
        steps: int,
        *,
        latent_mode: str = "fixed_standard_normal",
        round_index: int | None = None,
    ) -> dict[str, Any]:
        observation = self.env.reset()
        current_state = self.env.physics_state()
        self.student.set_latent_mode(latent_mode)
        if hasattr(self.student, "begin_evaluation"):
            self.student.begin_evaluation()
        self.student.reset(self.num_envs, current_state)
        self.teacher.reset(self.num_envs)
        done = torch.zeros(self.num_envs, device=self.env.device, dtype=torch.bool)
        episode_steps = torch.zeros(self.num_envs, device=self.env.device, dtype=torch.float32)
        episode_lengths: list[float] = []
        done_count = 0
        reward_sum = 0.0
        saturation_count = 0
        action_count = 0
        element_count = 0
        canonical_sse = 0.0
        raw_sse = 0.0
        normalized_sse = 0.0
        student_abs_max = 0.0
        teacher_abs_max = 0.0
        executed_abs_max = 0.0
        clip_error_max = 0.0
        per_motion: dict[str, dict[str, float]] = {}
        for step_index in range(int(steps)):
            motion_ids = self._motion_ids()
            rollout = self.student.observe()
            student_raw = rollout.action_raw.detach()
            student_canonical = rollout.action_canonical.detach()
            teacher_raw = self.action_contract.clip_raw(
                self.teacher.act(observation, done).detach()
            )
            teacher_canonical = self.action_contract.raw_to_canonical(teacher_raw)
            student_normalized = (
                (student_canonical - self.action_mean) / self.action_std.clamp_min(1.0e-6)
            )
            teacher_normalized = (
                (teacher_canonical - self.action_mean) / self.action_std.clamp_min(1.0e-6)
            )
            error_canonical = student_canonical.float() - teacher_canonical.float()
            error_raw = student_raw.float() - teacher_raw.float()
            error_normalized = student_normalized.float() - teacher_normalized.float()
            canonical_sse += float(error_canonical.square().sum().item())
            raw_sse += float(error_raw.square().sum().item())
            normalized_sse += float(error_normalized.square().sum().item())
            element_count += int(error_canonical.numel())
            student_abs_max = max(student_abs_max, float(student_raw.abs().max().item()))
            teacher_abs_max = max(teacher_abs_max, float(teacher_raw.abs().max().item()))
            if rollout.action_clip_error is not None:
                clip_error_max = max(
                    clip_error_max, float(rollout.action_clip_error.max().item())
                )
            env_canonical_mse = error_canonical.square().mean(dim=-1)
            env_raw_mse = error_raw.square().mean(dim=-1)
            env_normalized_mse = error_normalized.square().mean(dim=-1)
            for index in range(self.num_envs):
                key = self._motion_key(int(motion_ids[index].item()))
                record = per_motion.setdefault(
                    key,
                    {
                        "canonical_sse": 0.0,
                        "raw_sse": 0.0,
                        "normalized_sse": 0.0,
                        "elements": 0.0,
                        "steps": 0.0,
                        "terminations": 0.0,
                    },
                )
                record["canonical_sse"] += float(env_canonical_mse[index].item())
                record["raw_sse"] += float(env_raw_mse[index].item())
                record["normalized_sse"] += float(env_normalized_mse[index].item())
                record["elements"] += 1.0
                record["steps"] += 1.0
            next_observation, reward, next_done, _info = self.env.step(
                {"actions": student_raw}
            )
            next_done = next_done.to(device=self.env.device).bool()
            episode_steps += 1.0
            for index in next_done.nonzero(as_tuple=False).flatten().tolist():
                episode_lengths.append(float(episode_steps[index].item()))
                episode_steps[index] = 0.0
                key = self._motion_key(int(motion_ids[index].item()))
                per_motion[key]["terminations"] += 1.0
            done_count += int(next_done.sum().item())
            reward_sum += float(reward.detach().float().mean().cpu())
            if self.action_contract is not None:
                low = self.action_contract.raw_low
                high = self.action_contract.raw_high
                finite = bool(torch.isfinite(low).all() and torch.isfinite(high).all())
                if finite:
                    saturation_count += int(
                        ((student_raw <= low + 1.0e-6) | (student_raw >= high - 1.0e-6)).sum().item()
                    )
                    action_count += int(student_raw.numel())
                elif getattr(self.student, "action_clip_value", None):
                    value = float(self.student.action_clip_value)
                    saturation_count += int((student_raw.abs() >= value).sum().item())
                    action_count += int(student_raw.numel())
            executed_abs_max = max(executed_abs_max, float(student_raw.abs().max().item()))
            next_state = self.env.physics_state()
            self.student.after_step(
                self.action_contract.raw_to_canonical(student_raw),
                next_state,
                next_done,
            )
            observation = next_observation
            done = next_done
        self.student.set_latent_mode("fixed_standard_normal")
        if hasattr(self.student, "end_evaluation"):
            self.student.end_evaluation()
        by_motion = {}
        for key, record in per_motion.items():
            denominator = max(record["elements"], 1.0)
            by_motion[key] = {
                "teacher_action_rmse": (record["canonical_sse"] / denominator) ** 0.5,
                "teacher_action_rmse_raw": (record["raw_sse"] / denominator) ** 0.5,
                "teacher_action_rmse_normalized": (
                    record["normalized_sse"] / denominator
                )
                ** 0.5,
                "steps": record["steps"],
                "terminations": record["terminations"],
                "termination_rate": record["terminations"] / max(record["steps"], 1.0),
            }
        return {
            "steps": float(steps),
            "num_envs": float(self.num_envs),
            "latent_mode": latent_mode,
            "done_count": float(done_count),
            "mean_reward": reward_sum / max(1, int(steps)),
            "teacher_action_rmse": (canonical_sse / max(1, element_count)) ** 0.5,
            "teacher_action_rmse_raw": (raw_sse / max(1, element_count)) ** 0.5,
            "teacher_action_rmse_normalized": (
                normalized_sse / max(1, element_count)
            )
            ** 0.5,
            "teacher_action_rmse_by_motion": by_motion,
            "termination_rate": done_count / max(1, int(steps) * self.num_envs),
            "mean_episode_length": (
                sum(episode_lengths) / len(episode_lengths)
                if episode_lengths
                else float(steps)
            ),
            "max_episode_length": max(
                episode_lengths
                + [float(episode_steps[index].item()) for index in range(self.num_envs)]
            )
            if (episode_lengths or self.num_envs)
            else float(steps),
            "survived_full_horizon": int(
                (episode_steps >= float(steps)).sum().item()
            ),
            "survival_ratio": float(
                (episode_steps >= float(steps)).sum().item()
            )
            / max(1, self.num_envs),
            "action_saturation_ratio": saturation_count / max(1, action_count),
            "student_action_abs_max": student_abs_max,
            "teacher_action_abs_max": teacher_abs_max,
            "executed_action_abs_max": executed_abs_max,
            "action_clip_error_max": clip_error_max,
            "round_index": round_index,
        }
