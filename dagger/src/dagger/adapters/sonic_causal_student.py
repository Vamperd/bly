"""SONIC-like single-step causal CVAE student for DAgger."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate

from cvae_sa.sonic_causal_model import SonicCausalCVAE
from cvae_sa.sonic_causal_objective import compose_dagger_loss
from dagger.runtime.sonic_causal_window import SonicCausalWindowBuffer


CHECKPOINT_FORMAT = "sonic_causal_dagger_student_checkpoint_v1"
C_CHECKPOINT_FORMAT = "sonic_causal_cvae_checkpoint_v1"


@dataclass
class SonicCausalRollout:
    action_canonical: torch.Tensor
    action_raw: torch.Tensor
    requested_action_canonical: torch.Tensor
    epsilon: torch.Tensor
    batch: dict[str, torch.Tensor]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SonicCausalStudentAdapter:
    def __init__(
        self,
        *,
        checkpoint: str | Path,
        device: torch.device,
        action_contract: Any | None,
        normalization_run: str | Path | None = None,
        seed: int = 20261103,
        learning_rate: float = 1.0e-5,
        weight_decay: float = 0.0,
        action_clip_value: float | None = 20.0,
        kl_beta: float = 0.002,
    ) -> None:
        self.device = device
        self.action_contract = action_contract
        self.seed = int(seed)
        self.kl_beta = float(kl_beta)
        self.action_clip_value = action_clip_value
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("format_version") not in {
            C_CHECKPOINT_FORMAT,
            CHECKPOINT_FORMAT,
        }:
            raise ValueError("checkpoint is not a SONIC causal CVAE checkpoint")
        model_config = dict(payload.get("model_config") or {})
        self.model = SonicCausalCVAE(model_config).to(device)
        expected = self.model.architecture_signature(
            self.model.config,
            sum(parameter.numel() for parameter in self.model.parameters()),
        )
        if payload.get("model_signature") != expected:
            raise ValueError("SONIC causal checkpoint architecture signature mismatch")
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.eval()
        self.train_motion_keys = [str(key) for key in payload.get("train_motion_keys") or []]
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = _sha256(checkpoint_path)
        self.model_signature = expected
        self.model_config = self.model.config
        self.normalization_sha256 = None
        normalization = payload.get("normalization")
        if normalization is None:
            if normalization_run is None:
                raise ValueError("causal checkpoint has no normalization and no run was supplied")
            norm_path = Path(normalization_run).expanduser().resolve() / "data" / "normalization.npz"
            with np.load(norm_path) as values:
                normalization = {
                    name: values[name].astype(np.float32)
                    for name in (
                        "physical_state_mean",
                        "physical_state_std",
                        "action_mean",
                        "action_std",
                        "reference_future_mean",
                        "reference_future_std",
                    )
                }
            self.normalization_sha256 = _sha256(norm_path)
        self.state_mean = torch.as_tensor(normalization["state_mean"], device=device).float()
        self.state_std = torch.as_tensor(normalization["state_std"], device=device).float()
        self.action_mean = torch.as_tensor(normalization["action_mean"], device=device).float()
        self.action_std = torch.as_tensor(normalization["action_std"], device=device).float()
        self.reference_mean = torch.as_tensor(normalization["reference_mean"], device=device).float()
        self.reference_std = torch.as_tensor(normalization["reference_std"], device=device).float()
        for value, shape in (
            (self.state_mean, (70,)),
            (self.state_std, (70,)),
            (self.action_mean, (29,)),
            (self.action_std, (29,)),
            (self.reference_mean, (64,)),
            (self.reference_std, (64,)),
        ):
            if tuple(value.shape) != shape:
                raise ValueError(f"causal normalization has shape {value.shape}, expected {shape}")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        self.learning_rate = float(learning_rate)
        self._buffer: SonicCausalWindowBuffer | None = None
        self._epsilon: torch.Tensor | None = None
        self._episode_counter = 0
        self.latent_mode = "fixed_standard_normal"

    def reset(
        self,
        num_envs: int,
        current_state: torch.Tensor,
        initial_action: torch.Tensor | None = None,
    ) -> None:
        self._buffer = SonicCausalWindowBuffer(device=self.device, num_envs=int(num_envs))
        self._buffer.reset(current_state, initial_action=initial_action)
        self._episode_counter += 1
        generator = torch.Generator(device="cpu").manual_seed(
            self.seed + self._episode_counter * 100003
        )
        self._epsilon = torch.randn(
            int(num_envs), self.model.global_latent_dim, generator=generator
        ).to(self.device)

    def set_latent_mode(self, mode: str) -> None:
        if mode not in {"fixed_standard_normal", "zero", "posterior_mean"}:
            raise ValueError(f"unsupported causal latent mode: {mode}")
        self.latent_mode = mode

    def observe(self, reference_future: torch.Tensor) -> SonicCausalRollout:
        if self._buffer is None or self._epsilon is None:
            raise RuntimeError("causal student has not been reset")
        window = self._buffer.build_condition(
            reference_future=reference_future,
            state_mean=self.state_mean,
            state_std=self.state_std,
            action_mean=self.action_mean,
            action_std=self.action_std,
            reference_mean=self.reference_mean,
            reference_std=self.reference_std,
        )
        batch = window.batch
        if self.latent_mode == "posterior_mean":
            output = self.model(batch, stage="A")
        else:
            epsilon = (
                torch.zeros_like(self._epsilon)
                if self.latent_mode == "zero"
                else self._epsilon
            )
            output = self.model.infer_from_condition(batch, epsilon=epsilon)
        canonical = output.action * self.action_std + self.action_mean
        if self.action_contract is None:
            raw = canonical
            achieved = canonical
        else:
            raw, achieved = self.action_contract.canonical_to_raw(canonical)
        if not torch.isfinite(raw).all() or not torch.isfinite(achieved).all():
            raise FloatingPointError("causal student produced non-finite Action")
        return SonicCausalRollout(
            action_canonical=achieved,
            action_raw=raw,
            requested_action_canonical=canonical,
            epsilon=self._epsilon.detach().clone(),
            batch=batch,
        )

    def after_step(
        self,
        executed_action_canonical: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        if self._buffer is None:
            raise RuntimeError("causal student has not been reset")
        self._buffer.advance(executed_action_canonical, next_state, done)
        if bool(done.any()):
            self._episode_counter += 1
            generator = torch.Generator(device="cpu").manual_seed(
                self.seed + self._episode_counter * 100003
            )
            self._epsilon[done] = torch.randn(
                int(done.sum()), self.model.global_latent_dim, generator=generator
            ).to(self.device)

    def _device_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def train_on_samples(
        self,
        dynamic_samples: list[dict[str, Any]],
        *,
        source_dataset: Any | None,
        steps: int,
        batch_size: int,
        source_mix: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> dict[str, float]:
        if not dynamic_samples and source_dataset is None:
            raise RuntimeError("causal DAgger has no training samples")
        if generator is None:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.model.train()
        losses: list[float] = []
        source_count = 0
        dagger_count = 0
        for _ in range(int(steps)):
            rows = []
            use_source = bool(source_dataset is not None and torch.rand((), generator=generator).item() < source_mix)
            if use_source:
                indices = torch.randint(len(source_dataset), (batch_size,), generator=generator).tolist()
                rows = [source_dataset[index] for index in indices]
                source_count += 1
            else:
                indices = torch.randint(len(dynamic_samples), (batch_size,), generator=generator).tolist()
                rows = [dynamic_samples[index] for index in indices]
                dagger_count += 1
            batch = self._device_batch(default_collate(rows))
            posterior = self.model(batch, stage="C")
            standard = self.model.infer_from_condition(
                batch, epsilon=torch.randn_like(posterior.global_latent)
            )
            if posterior.posterior is None:
                raise RuntimeError("causal DAgger posterior is missing")
            from cvae_sa.sonic_causal_objective import compose_dagger_loss

            teacher = batch.get("teacher_action", batch["target_action"])
            loss, _terms = compose_dagger_loss(
                posterior_action=posterior.action,
                standard_normal_action=standard.action,
                teacher_action=teacher,
                posterior=posterior.posterior,
                kl_beta=self.kl_beta,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite causal DAgger loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
            self.optimizer.step()
            losses.append(float(loss.detach()))
        self.model.eval()
        return {
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "source_steps": float(source_count),
            "dagger_steps": float(dagger_count),
            "dataset_samples": float(len(dynamic_samples)),
        }

    def save(self, path: str | Path, *, round_index: int, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": CHECKPOINT_FORMAT,
            "architecture_version": self.model.ARCHITECTURE_VERSION,
            "model_config": self.model.config,
            "model_signature": self.model_signature,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "normalization": {
                "state_mean": self.state_mean.cpu(),
                "state_std": self.state_std.cpu(),
                "action_mean": self.action_mean.cpu(),
                "action_std": self.action_std.cpu(),
                "reference_mean": self.reference_mean.cpu(),
                "reference_std": self.reference_std.cpu(),
            },
            "source_checkpoint": str(self.checkpoint_path),
            "source_checkpoint_sha256": self.checkpoint_sha256,
            "train_motion_keys": list(self.train_motion_keys),
            "round_index": int(round_index),
            "latent_mode": self.latent_mode,
            "kl_beta": self.kl_beta,
        }
        if metadata:
            payload.update(metadata)
        temporary = path.with_name(f"{path.name}.tmp.{__import__('os').getpid()}")
        torch.save(payload, temporary)
        temporary.replace(path)

