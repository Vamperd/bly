"""DAgger adapter for the trained 11/10 MLP CVAE using recorded Physics-v5 S[t+1]."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import default_collate

from cvae_sa.dataset import StateActionWindowDataset
from cvae_sa.mlp_cvae11 import MLPConditionalVAE11
from cvae_sa.physics_schema import read_physics_states
from dagger.runtime.sonic_causal_window import SonicCausalWindowBuffer
from .sonic_causal_student import SonicCausalRollout

CHECKPOINT_FORMAT = "sonic_11_frame_mlp_cvae_dagger_checkpoint_v1"
SOURCE_FORMATS = {"sonic_11_frame_mlp_cvae_checkpoint_v1", CHECKPOINT_FORMAT}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MlpCvae11StudentAdapter:
    def __init__(self, *, checkpoint: str | Path, device: torch.device, action_contract: Any | None,
                 dataset_run: str | Path, seed: int = 20261112, learning_rate: float = 1.0e-5,
                 weight_decay: float = 1.0e-4, kl_beta: float = 0.002) -> None:
        self.device = device
        self.action_contract = action_contract
        self.seed = int(seed)
        self.kl_beta = float(kl_beta)
        self.latent_mode = "sampled_prior"
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("format_version") not in SOURCE_FORMATS:
            raise ValueError("checkpoint is not an 11/10 MLP CVAE checkpoint")
        config = dict(payload.get("config") or {})
        model_config = dict(config.get("model") or payload.get("model_config") or {})
        self.model = MLPConditionalVAE11(model_config).to(device)
        self.model.load_state_dict(payload.get("model") or payload["model_state_dict"], strict=True)
        self.model.eval()
        self.model_config = model_config
        self.model_signature = self.model.architecture_signature()
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = _sha256(checkpoint_path)
        selection = payload.get("selection") or {}
        self.train_motion_keys = [str(v) for v in selection.get("selected_motion_keys", payload.get("train_motion_keys", []))]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
        self.action_mean, self.action_std, self.state_mean, self.state_std = self._load_normalization(dataset_run)
        self.expected_states = self._load_expected_states(dataset_run, self.train_motion_keys)
        self._buffer: SonicCausalWindowBuffer | None = None
        self._motion_keys: list[str] = []
        self._motion_ids = torch.empty(0, dtype=torch.long, device=device)
        self._target_steps = torch.zeros(0, dtype=torch.long, device=device)
        self._episode_counter = 0

    def _load_normalization(self, dataset_run: str | Path):
        dataset = StateActionWindowDataset(dataset_run, "train", window_transitions=10, validation_stride=1, random_crop=False)
        try:
            return tuple(torch.as_tensor(v, device=self.device).float() for v in (dataset.action_mean, dataset.action_std, dataset.state_mean, dataset.state_std))
        finally:
            dataset.close()

    def _load_expected_states(self, dataset_run: str | Path, keys: list[str]) -> dict[str, torch.Tensor]:
        base = StateActionWindowDataset(dataset_run, "train", window_transitions=10, validation_stride=1, random_crop=False)
        result: dict[str, torch.Tensor] = {}
        try:
            wanted = set(keys)
            for row in base.episodes:
                key = str(row["motion_key"])
                if key not in wanted or key in result or str(row.get("status", "")) != "completed":
                    continue
                stream = base._handle(row["hdf5_path"])
                episode = stream[f"data/{row['episode']}"]
                physical = read_physics_states(episode["states"], 0, int(row["steps"]) + 1).astype(np.float32)
                normalized = (physical - base.state_mean[None]) / np.maximum(base.state_std[None], 1.0e-6)
                result[key] = torch.from_numpy(normalized).to(self.device)
            missing = sorted(wanted - set(result))
            if missing:
                raise ValueError(f"recorded Physics-v5 next-State trajectories missing motions: {missing[:5]}")
            return result
        finally:
            base.close()

    def reset(self, num_envs: int, current_state: torch.Tensor, initial_action: torch.Tensor | None = None) -> None:
        self._buffer = SonicCausalWindowBuffer(device=self.device, num_envs=int(num_envs))
        self._buffer.reset(current_state, initial_action=initial_action)
        self._target_steps = torch.zeros(int(num_envs), dtype=torch.long, device=self.device)
        self._motion_ids = torch.full((int(num_envs),), -1, dtype=torch.long, device=self.device)
        self._episode_counter += 1

    def set_motion_context(self, motion_ids: torch.Tensor, motion_keys: list[str]) -> None:
        self._motion_ids = motion_ids.to(self.device, dtype=torch.long).clone()
        self._motion_keys = [str(v) for v in motion_keys]

    def set_latent_mode(self, mode: str) -> None:
        if mode not in {"sampled_prior", "deterministic_prior"}:
            raise ValueError(f"unsupported MLP CVAE latent mode: {mode}")
        self.latent_mode = mode

    def _expected_next(self) -> torch.Tensor:
        values = []
        for index, motion_id in enumerate(self._motion_ids.tolist()):
            if motion_id < 0 or motion_id >= len(self._motion_keys):
                raise RuntimeError("MLP CVAE DAgger has no valid motion context")
            key = self._motion_keys[motion_id]
            states = self.expected_states[key]
            step = min(int(self._target_steps[index].item()) + 1, states.shape[0] - 1)
            values.append(states[step])
        return torch.stack(values, dim=0)

    @torch.no_grad()
    def observe(self, _reference_future: torch.Tensor) -> SonicCausalRollout:
        if self._buffer is None:
            raise RuntimeError("MLP CVAE student has not been reset")
        window = self._buffer.build_condition(
            reference_future=torch.zeros(self._buffer.num_envs, 10, 64, device=self.device),
            state_mean=self.state_mean, state_std=self.state_std,
            action_mean=self.action_mean, action_std=self.action_std,
            reference_mean=torch.zeros(64, device=self.device), reference_std=torch.ones(64, device=self.device),
        )
        history_state = window.batch["state_history"]
        history_action = window.batch["action_history"]
        state_valid = window.state_valid
        action_valid = window.action_valid
        next_state = self._expected_next()[:, None, :]
        condition_state = torch.cat((history_state, next_state), dim=1)
        state_valid_11 = torch.cat((state_valid, torch.ones(state_valid.shape[0], 1, dtype=torch.bool, device=self.device)), dim=1)
        action_valid_10 = action_valid.clone()
        action_valid_10[:, -1] = True
        action_visible = action_valid.clone()
        action_visible[:, -1] = False
        condition_action = history_action.clone()
        condition_action[:, -1] = 0.0
        target_state = torch.zeros(state_valid_11.shape, dtype=torch.bool, device=self.device)
        target_action = torch.zeros(action_valid_10.shape, dtype=torch.bool, device=self.device)
        target_action[:, -1] = True
        action_before_valid = action_valid[:, 0]
        action_before = torch.where(action_before_valid[:, None], history_action[:, 0], torch.zeros_like(history_action[:, 0]))
        with torch.inference_mode():
            output = self.model.prior_forward(
                condition_state=condition_state, condition_action=condition_action,
                state_visible=state_valid_11, action_visible=action_visible,
                state_valid=state_valid_11, action_valid=action_valid_10,
                target_state=target_state, target_action=target_action,
                action_before=action_before, action_before_valid=action_before_valid,
                sample=self.latent_mode == "sampled_prior",
            )
        normalized = output["p_action"][:, -1]
        canonical = normalized * self.action_std + self.action_mean
        raw, achieved = (canonical, canonical) if self.action_contract is None else self.action_contract.canonical_to_raw(canonical)
        batch = {
            "condition_state": condition_state, "condition_action": condition_action,
            "state_visible": state_valid_11, "action_visible": action_visible,
            "state_valid": state_valid_11, "action_valid": action_valid_10,
            "target_state": target_state, "target_mask_action": target_action,
            "action_before": action_before, "action_before_valid": action_before_valid,
        }
        return SonicCausalRollout(achieved, raw, canonical, torch.zeros_like(normalized), batch)

    def after_step(self, executed_action_canonical: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> None:
        if self._buffer is None:
            raise RuntimeError("MLP CVAE student has not been reset")
        self._buffer.advance(executed_action_canonical, next_state, done)
        self._target_steps += 1
        self._target_steps[done.bool()] = 0

    def train_on_samples(self, dynamic_samples: list[dict[str, Any]], *, source_dataset: Any | None, steps: int, batch_size: int, source_mix: float = 0.0, generator: torch.Generator | None = None) -> dict[str, float]:
        del source_dataset, source_mix
        if not dynamic_samples:
            raise RuntimeError("MLP CVAE DAgger has no dynamic samples")
        generator = generator or torch.Generator(device="cpu").manual_seed(self.seed)
        self.model.train()
        losses = []
        for _ in range(int(steps)):
            indices = torch.randint(len(dynamic_samples), (min(int(batch_size), len(dynamic_samples)),), generator=generator).tolist()
            rows = [dynamic_samples[i] for i in indices]
            batch = {key: torch.stack([row[key] for row in rows]).to(self.device) for key in rows[0] if isinstance(rows[0][key], torch.Tensor)}
            teacher = torch.stack([row["teacher_action"] for row in rows]).to(self.device)
            posterior_action = batch["condition_action"].clone()
            posterior_action[:, -1] = teacher
            output = self.model(
                condition_state=batch["condition_state"], condition_action=batch["condition_action"],
                state_visible=batch["state_visible"], action_visible=batch["action_visible"],
                state_valid=batch["state_valid"], action_valid=batch["action_valid"],
                target_state=batch["target_state"], target_action=batch["target_mask_action"],
                action_before=batch["action_before"], action_before_valid=batch["action_before_valid"],
                posterior_state=batch["condition_state"], posterior_action=posterior_action, sample=True,
            )
            target = teacher[:, None, :]
            loss = 0.5 * ((output["p_action"][:, -1] - teacher).float().square().mean() + (output["q_action"][:, -1] - teacher).float().square().mean())
            loss = loss + self.kl_beta * (output["q_mean"].float().square() + torch.exp(output["q_logvar"].float()) - 1.0 - output["q_logvar"].float()).mean() * 0.5
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite MLP CVAE DAgger loss")
            self.optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True); self.optimizer.step(); losses.append(float(loss.detach()))
        self.model.eval()
        return {"train_loss": float(np.mean(losses)) if losses else 0.0, "source_steps": 0.0, "dagger_steps": float(steps), "dataset_samples": float(len(dynamic_samples))}

    def save(self, path: str | Path, *, round_index: int, metadata: dict[str, Any] | None = None) -> None:
        payload = {
            "format_version": CHECKPOINT_FORMAT, "architecture_version": self.model.architecture_signature()["architecture_version"],
            "model_signature": self.model.architecture_signature(), "model": self.model.state_dict(), "config": {"model": self.model_config},
            "optimizer": self.optimizer.state_dict(), "selection": {"selected_motion_keys": self.train_motion_keys},
            "normalization": {"state_mean": self.state_mean.cpu(), "state_std": self.state_std.cpu(), "action_mean": self.action_mean.cpu(), "action_std": self.action_std.cpu()},
            "source_checkpoint": str(self.checkpoint_path), "source_checkpoint_sha256": self.checkpoint_sha256, "round_index": int(round_index),
        }
        if metadata: payload.update(metadata)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, path)
