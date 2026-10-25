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
    ) -> None:
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
        self.dataset = CvaeDaggerDataset()

    def _motion_ids(self) -> torch.Tensor:
        motion = self.env.command_manager.get_term("motion")
        value = getattr(motion, "motion_ids", None)
        if value is None:
            return torch.full((self.num_envs,), -1, device=self.env.device, dtype=torch.long)
        return value.detach().clone().to(device=self.env.device, dtype=torch.long)

    @staticmethod
    def _new_record(state: torch.Tensor, episode_id: int = 0) -> dict[str, list[torch.Tensor]]:
        return {
            "states": [state.detach().clone()],
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
        self.teacher.reset(self.num_envs)
        self.student.reset(self.num_envs, current_state)
        random = torch.Generator(device="cpu")
        random.manual_seed(self.seed + int(round_index) * 1009)
        mixture_teacher = (torch.rand(self.num_envs, generator=random) < float(beta)).to(self.env.device)
        episode_numbers = list(range(self.num_envs))
        episode_records = [self._new_record(current_state[index], episode_numbers[index]) for index in range(self.num_envs)]
        all_samples: list[dict[str, torch.Tensor]] = []
        done = torch.zeros(self.num_envs, device=self.env.device, dtype=torch.bool)

        for _ in range(self.rollout_steps):
            student_rollout = self.student.observe()
            teacher_raw = self.action_contract.clip_raw(self.teacher.act(observation, done).detach())
            teacher_canonical = self.action_contract.raw_to_canonical(teacher_raw)
            student_raw = student_rollout.action_raw.detach()
            student_canonical = student_rollout.action_canonical.detach()
            executed_raw = torch.where(mixture_teacher[:, None], teacher_raw, student_raw)
            executed_canonical = self.action_contract.raw_to_canonical(executed_raw)
            if teacher_raw.shape != student_raw.shape or tuple(teacher_raw.shape[-1:]) != (29,):
                raise ValueError(
                    f"teacher/student Action shape mismatch: teacher={tuple(teacher_raw.shape)}, "
                    f"student={tuple(student_raw.shape)}"
                )
            next_observation, _reward, next_done, _info = self.env.step({"actions": executed_raw})
            next_done = next_done.to(device=self.env.device).bool()
            next_state = self.env.physics_state()
            motion_ids = self._motion_ids()

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
                record["mixture_teacher"].append(mixture_teacher[index].cpu())
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
                    episode_records[index] = self._new_record(next_state[index].cpu(), episode_numbers[index])
                    mixture_teacher[index] = torch.rand((), generator=random).item() < float(beta)

            self.student.after_step(executed_canonical, next_state, next_done)
            observation = next_observation
            done = next_done

        for record in episode_records:
            all_samples.extend(self._make_samples(record))
        round_data = self._stack_samples(all_samples)
        self.dataset.append(round_data)
        self.dataset.save_round(round_index, self.run_dir)
        return round_data

    @torch.no_grad()
    def evaluate_student_only(self, steps: int, *, latent_mode: str = "fixed_standard_normal") -> dict[str, Any]:
        observation = self.env.reset()
        current_state = self.env.physics_state()
        self.student.set_latent_mode(latent_mode)
        self.student.reset(self.num_envs, current_state)
        done = torch.zeros(self.num_envs, device=self.env.device, dtype=torch.bool)
        reward_sum = 0.0
        done_count = 0
        action_saturation = 0
        action_count = 0
        for _ in range(int(steps)):
            rollout = self.student.observe()
            action = rollout.action_raw
            next_observation, reward, next_done, _info = self.env.step({"actions": action})
            next_done = next_done.to(device=self.env.device).bool()
            next_state = self.env.physics_state()
            self.student.after_step(self.action_contract.raw_to_canonical(action), next_state, next_done)
            done_count += int(next_done.sum())
            reward_sum += float(reward.detach().float().mean().cpu())
            action_saturation += int((action.abs() >= 1.0).sum())
            action_count += int(action.numel())
            observation = next_observation
            done = next_done
        self.student.set_latent_mode("fixed_standard_normal")
        return {
            "steps": float(steps),
            "latent_mode": latent_mode,
            "done_count": float(done_count),
            "mean_reward": reward_sum / max(1, int(steps)),
            "action_saturation_ratio": action_saturation / max(1, action_count),
        }

