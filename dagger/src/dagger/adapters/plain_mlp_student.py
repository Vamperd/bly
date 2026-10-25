"""Plain MLP student adapter."""

from __future__ import annotations

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
    ) -> None:
        self.device = device
        self.input_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.input_terms = tuple(input_terms) if input_terms is not None else ()
        self.motion_keys = list(motion_keys) if motion_keys is not None else []
        self.freeze_normalization = bool(freeze_normalization)
        layers: list[nn.Module] = []
        previous = self.input_dim
        for width in self.hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.SiLU()])
            previous = width
        layers.append(nn.Linear(previous, self.action_dim))
        self.model = nn.Sequential(*layers).to(device)
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
        return value

    def act(self, student_input: torch.Tensor) -> torch.Tensor:
        value = student_input.to(self.device, dtype=torch.float32)
        value = (value - self._mean) / self._std.clamp_min(1.0e-6)
        self.model.eval()
        with torch.no_grad():
            return self.model(value)

    def train_on_dataset(self, dataset) -> dict[str, float]:
        observations, actions = dataset.tensors()
        observations = observations.float()
        actions = actions.float()
        if not self.freeze_normalization or not self._normalization_fitted:
            self._mean = observations.mean(dim=0).to(self.device)
            self._std = observations.std(dim=0).clamp_min(1.0e-4).to(self.device)
            self._normalization_fitted = True
        normalized = (observations - self._mean.cpu()) / self._std.cpu().clamp_min(1.0e-6)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + len(dataset.rounds))
        losses: list[float] = []
        self.model.train()
        for _ in range(self.train_steps):
            indices = torch.randint(
                normalized.shape[0],
                (min(self.batch_size, normalized.shape[0]),),
                generator=generator,
            )
            prediction = self.model(normalized[indices].to(self.device))
            target = actions[indices].to(self.device)
            loss = torch.mean(torch.square(prediction - target))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        self.model.eval()
        predictions: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, normalized.shape[0], max(self.batch_size, 256)):
                chunk = normalized[start : start + max(self.batch_size, 256)].to(self.device)
                predictions.append(self.model(chunk).cpu())
        prediction = torch.cat(predictions, dim=0)
        rmse = torch.sqrt(torch.mean(torch.square(prediction - actions))).item()
        return {
            "train_loss": losses[-1] if losses else float("nan"),
            "teacher_action_rmse": float(rmse),
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
