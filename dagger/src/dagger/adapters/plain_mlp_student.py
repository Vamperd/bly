"""Plain MLP student adapter."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


class PlainMlpStudent:
    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        device: torch.device,
        hidden_dims: tuple[int, ...] = (2048, 2048, 1024, 1024, 512, 512),
        learning_rate: float = 1.0e-4,
        weight_decay: float = 1.0e-4,
        train_steps: int = 2000,
        batch_size: int = 512,
        seed: int = 0,
        input_terms: tuple[str, ...] | None = None,
        freeze_normalization: bool = False,
        input_mean: torch.Tensor | None = None,
        input_std: torch.Tensor | None = None,
        motion_keys: list[str] | None = None,
        layer_norm: bool = False,
        dropout: float = 0.0,
        lr_schedule: str = "constant",
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.01,
    ) -> None:
        self.device = device
        self.input_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.input_terms = tuple(input_terms) if input_terms is not None else ()
        self.motion_keys = list(motion_keys) if motion_keys is not None else []
        self.freeze_normalization = bool(freeze_normalization)
        self.layer_norm = bool(layer_norm)
        self.dropout = float(dropout)
        self.lr_schedule = str(lr_schedule)
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("lr_schedule must be constant or cosine")
        layers: list[nn.Module] = []
        previous = self.input_dim
        for width in self.hidden_dims:
            layers.append(nn.Linear(previous, width))
            if self.layer_norm:
                layers.append(nn.LayerNorm(width))
            layers.append(nn.SiLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            previous = width
        layers.append(nn.Linear(previous, self.action_dim))
        self.model = nn.Sequential(*layers).to(device)
        self.peak_learning_rate = float(learning_rate)
        self.minimum_learning_rate = self.peak_learning_rate * self.min_lr_ratio
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.train_steps = int(train_steps)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._mean = (
            torch.zeros(self.input_dim, device=device)
            if input_mean is None
            else input_mean.to(device=device, dtype=torch.float32).clone()
        )
        self._std = (
            torch.ones(self.input_dim, device=device)
            if input_std is None
            else input_std.to(device=device, dtype=torch.float32).clone()
        )
        if tuple(self._mean.shape) != (self.input_dim,) or tuple(self._std.shape) != (self.input_dim,):
            raise ValueError("input_mean/input_std must have shape [input_dim]")
        self._normalization_fitted = input_mean is not None and input_std is not None

    def reset(self, num_envs: int) -> None:  # noqa: ARG002
        self.model.eval()

    def observe(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        actor_obs = observation["actor_obs"]
        selected = observation.get("student_reference_features")
        if selected is None:
            raise KeyError("runtime did not provide selected G1 reference features")
        value = torch.cat([actor_obs, selected], dim=-1)
        if value.shape[-1] != self.input_dim:
            raise ValueError(
                f"student input dimension mismatch: expected {self.input_dim}, "
                f"found {value.shape[-1]}"
            )
        if not torch.isfinite(value).all():
            raise FloatingPointError("non-finite student observation")
        return value

    def act(self, student_input: torch.Tensor) -> torch.Tensor:
        value = student_input.to(self.device, dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise FloatingPointError("non-finite student input")
        value = (value - self._mean) / self._std.clamp_min(1.0e-6)
        self.model.eval()
        with torch.no_grad():
            output = self.model(value)
        if not torch.isfinite(output).all():
            raise FloatingPointError("non-finite student action")
        return output

    def train_on_dataset(
        self,
        dataset,
        *,
        label: str = "",
        motion_weights: dict[str, float] | None = None,
        global_step_offset: int = 0,
        total_training_steps: int = 0,
    ) -> dict[str, float]:
        observations, actions = dataset.tensors()
        observations = observations.float()
        actions = actions.float()
        if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
            raise FloatingPointError("non-finite DAgger observations or actions")
        if not self.freeze_normalization or not self._normalization_fitted:
            self._mean = observations.mean(dim=0).to(self.device)
            self._std = observations.std(dim=0).clamp_min(1.0e-4).to(self.device)
            self._normalization_fitted = True
        normalized = (observations - self._mean.cpu()) / self._std.cpu().clamp_min(1.0e-6)
        motion_ids = dataset.motion_ids()
        if motion_weights and self.motion_keys:
            weights = torch.ones(motion_ids.shape[0], dtype=torch.float32)
            for index, motion_id in enumerate(motion_ids.tolist()):
                if 0 <= int(motion_id) < len(self.motion_keys):
                    weights[index] = float(
                        motion_weights.get(self.motion_keys[int(motion_id)], 1.0)
                    )
            weights = weights.clamp_min(1.0e-6)
        else:
            weights = None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + len(dataset.rounds))
        losses: list[float] = []
        learning_rates: list[float] = []
        self.model.train()
        for step_index in range(self.train_steps):
            global_step = int(global_step_offset) + step_index
            total_steps = max(int(total_training_steps), int(global_step_offset) + self.train_steps)
            if self.lr_schedule == "cosine":
                if self.warmup_steps > 0 and global_step < self.warmup_steps:
                    learning_rate = self.peak_learning_rate * float(global_step + 1) / self.warmup_steps
                else:
                    progress = min(
                        1.0,
                        max(
                            0.0,
                            (global_step - self.warmup_steps)
                            / max(1, total_steps - self.warmup_steps),
                        ),
                    )
                    learning_rate = self.minimum_learning_rate + 0.5 * (
                        self.peak_learning_rate - self.minimum_learning_rate
                    ) * (1.0 + math.cos(math.pi * progress))
            else:
                learning_rate = self.peak_learning_rate
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            batch_count = min(self.batch_size, normalized.shape[0])
            if weights is not None:
                indices = torch.multinomial(
                    weights, batch_count, replacement=True, generator=generator
                )
            else:
                indices = torch.randint(
                    normalized.shape[0], (batch_count,), generator=generator
                )
            prediction = self.model(normalized[indices].to(self.device))
            target = actions[indices].to(self.device)
            loss = torch.mean(torch.square(prediction - target))
            if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                raise FloatingPointError("non-finite student prediction or loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0, error_if_nonfinite=True
            )
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            learning_rates.append(learning_rate)
            if self.train_steps >= 100 and (step_index + 1) % 100 == 0:
                prefix = f"{label} " if label else ""
                print(
                    f"[dagger] train {prefix}step={step_index + 1}/{self.train_steps} "
                    f"loss={losses[-1]:.6g} lr={learning_rate:.3g} "
                    f"grad={float(gradient_norm):.3g}",
                    flush=True,
                )
        for parameter in self.model.parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError("non-finite student parameter after training")
        self.model.eval()
        predictions: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, normalized.shape[0], max(self.batch_size, 256)):
                chunk = normalized[start : start + max(self.batch_size, 256)].to(self.device)
                predictions.append(self.model(chunk).cpu())
        prediction = torch.cat(predictions, dim=0)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("non-finite student evaluation prediction")
        rmse = torch.sqrt(torch.mean(torch.square(prediction - actions))).item()
        return {
            "train_loss": losses[-1] if losses else float("nan"),
            "teacher_action_rmse": float(rmse),
            "mean_learning_rate": float(sum(learning_rates) / max(1, len(learning_rates))),
            "final_learning_rate": float(learning_rates[-1]) if learning_rates else 0.0,
            "motion_weighting": bool(weights is not None),
        }

    def save(self, path: str, *, extra_metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": "dagger_plain_mlp_student_checkpoint_v1",
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "input_mean": self._mean.cpu(),
            "input_std": self._std.cpu(),
            "input_dim": self.input_dim,
            "input_terms": list(self.input_terms),
            "hidden_dims": list(self.hidden_dims),
            "layer_norm": self.layer_norm,
            "dropout": self.dropout,
            "lr_schedule": self.lr_schedule,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "action_dim": self.action_dim,
            "normalization_mode": "frozen" if self.freeze_normalization else "recomputed_per_round",
            "motion_keys": list(self.motion_keys),
        }
        if extra_metadata:
            payload.update(extra_metadata)
        torch.save(payload, path)

    def load(self, path: str | Path) -> dict:
        payload = torch.load(Path(path).expanduser(), map_location=self.device, weights_only=False)
        if payload.get("format_version") != "dagger_plain_mlp_student_checkpoint_v1":
            raise ValueError("unsupported plain MLP student checkpoint")
        if int(payload.get("input_dim", -1)) != self.input_dim:
            raise ValueError("checkpoint input_dim does not match student")
        if int(payload.get("action_dim", -1)) != self.action_dim:
            raise ValueError("checkpoint action_dim does not match student")
        if tuple(int(value) for value in payload.get("hidden_dims", ())) != self.hidden_dims:
            raise ValueError("checkpoint hidden_dims do not match student")
        if "layer_norm" in payload and bool(payload["layer_norm"]) != self.layer_norm:
            raise ValueError("checkpoint layer_norm does not match student")
        if "dropout" in payload and abs(float(payload["dropout"]) - self.dropout) > 1.0e-9:
            raise ValueError("checkpoint dropout does not match student")
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        if isinstance(payload.get("optimizer_state_dict"), dict):
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self._mean = payload["input_mean"].to(self.device, dtype=torch.float32)
        self._std = payload["input_std"].to(self.device, dtype=torch.float32)
        self.input_terms = tuple(payload.get("input_terms", ()))
        self.motion_keys = list(payload.get("motion_keys", []))
        self._normalization_fitted = True
        self.model.eval()
        return payload
