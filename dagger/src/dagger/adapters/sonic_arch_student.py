"""Student with the exact SONIC release actor (G1 encoder -> FSQ -> g1_dyn), trained by DAgger only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

FORMAT_VERSION = "dagger_sonic_arch_student_checkpoint_v1"


def build_sonic_actor(release_config: Any, env: Any, device: torch.device, seed: int) -> nn.Module:
    from gear_sonic.trl.utils.common import custom_instantiate, materialize_lazy_params

    torch.manual_seed(int(seed))
    model = custom_instantiate(
        release_config.algo.config.actor,
        env_config=env.config,
        algo_config=release_config.algo.config,
        module_dim_dict=getattr(release_config.algo.config, "module_dim", {}),
        _resolve=False,
    ).to(device)
    materialize_lazy_params(model, env.env)
    return model


def tokenizer_layout(
    tokenizer_names: list[str],
    tokenizer_dims: dict[str, Any],
    reference_terms: tuple[str, ...],
    encoder_names: list[str],
) -> dict[str, Any]:
    offset = 0
    slices: dict[str, tuple[int, int]] = {}
    for name in tokenizer_names:
        width = 1
        for value in tuple(tokenizer_dims[name]):
            width *= int(value)
        slices[name] = (offset, offset + width)
        offset += width
    missing = [name for name in reference_terms if name not in slices]
    if missing:
        raise KeyError(f"tokenizer is missing reference terms: {missing}")
    encoder_index = slices.get("encoder_index")
    if encoder_index is not None:
        if encoder_index[1] - encoder_index[0] != len(encoder_names):
            raise ValueError("encoder_index width does not match encoder_sample_probs")
        if "g1" not in encoder_names:
            raise ValueError("encoder_sample_probs has no g1 encoder")
    return {
        "total_dim": offset,
        "reference_slices": [list(slices[name]) for name in reference_terms],
        "encoder_index_slice": None if encoder_index is None else list(encoder_index),
        "g1_column": encoder_names.index("g1") if "g1" in encoder_names else None,
    }


class SonicArchStudent:
    def __init__(
        self,
        *,
        model: nn.Module,
        layout: dict[str, Any],
        actor_obs_dim: int,
        action_dim: int,
        device: torch.device,
        init_mode: str = "random",
        learning_rate: float = 1.0e-4,
        weight_decay: float = 1.0e-4,
        train_steps: int = 2000,
        batch_size: int = 512,
        grad_clip: float = 1.0,
        seed: int = 0,
        input_terms: tuple[str, ...] = (),
        motion_keys: list[str] | None = None,
    ) -> None:
        if init_mode not in {"random", "teacher"}:
            raise ValueError("init_mode must be random or teacher")
        self.model = model.to(device)
        self.layout = dict(layout)
        self.device = device
        self.actor_obs_dim = int(actor_obs_dim)
        self.reference_dim = sum(end - start for start, end in self.layout["reference_slices"])
        self.input_dim = self.actor_obs_dim + self.reference_dim
        self.action_dim = int(action_dim)
        self.init_mode = init_mode
        self.train_steps = int(train_steps)
        self.batch_size = int(batch_size)
        self.grad_clip = float(grad_clip)
        self.seed = int(seed)
        self.peak_learning_rate = float(learning_rate)
        self.input_terms = tuple(input_terms)
        self.motion_keys = list(motion_keys) if motion_keys is not None else []
        backbone = getattr(self.model, "actor_module", None)
        if backbone is not None and hasattr(backbone, "_active_decoders") and "g1_dyn" in getattr(backbone, "decoders", {}):
            # g1_kin only feeds the g1_recon aux loss, which this student does not use.
            backbone._active_decoders = {"g1_dyn"}
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.peak_learning_rate,
            weight_decay=float(weight_decay),
        )

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def reset(self, num_envs: int) -> None:  # noqa: ARG002
        self.model.eval()

    def observe(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        selected = observation.get("student_reference_features")
        if selected is None:
            raise KeyError("runtime did not provide selected G1 reference features")
        value = torch.cat([observation["actor_obs"], selected], dim=-1)
        if value.shape[-1] != self.input_dim:
            raise ValueError(
                f"student input dimension mismatch: expected {self.input_dim}, found {value.shape[-1]}"
            )
        if not torch.isfinite(value).all():
            raise FloatingPointError("non-finite student observation")
        return value

    def build_tokenizer(self, reference: torch.Tensor) -> torch.Tensor:
        tokenizer = reference.new_zeros(reference.shape[0], int(self.layout["total_dim"]))
        offset = 0
        for start, end in self.layout["reference_slices"]:
            width = end - start
            tokenizer[:, start:end] = reference[:, offset : offset + width]
            offset += width
        encoder_slice = self.layout.get("encoder_index_slice")
        if encoder_slice is not None:
            tokenizer[:, encoder_slice[0] + int(self.layout["g1_column"])] = 1.0
        return tokenizer

    def _forward(self, student_input: torch.Tensor, tokenizer: torch.Tensor | None = None) -> torch.Tensor:
        actor_obs = student_input[:, : self.actor_obs_dim]
        if tokenizer is None:
            tokenizer = self.build_tokenizer(student_input[:, self.actor_obs_dim :])
        output = self.model({"actor_obs": actor_obs[:, None], "tokenizer": tokenizer[:, None]})
        return output[:, -1]

    def act(self, student_input: torch.Tensor) -> torch.Tensor:
        value = student_input.to(self.device, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            output = self._forward(value)
        if not torch.isfinite(output).all():
            raise FloatingPointError("non-finite student action")
        return output

    def act_full_tokenizer(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self._forward(
                self.observe(observation).float(), observation["tokenizer"].float()
            )

    def train_on_dataset(
        self,
        dataset,
        *,
        label: str = "",
        motion_weights: dict[str, float] | None = None,
        global_step_offset: int = 0,  # noqa: ARG002
        total_training_steps: int = 0,  # noqa: ARG002
    ) -> dict[str, float]:
        observations, actions = dataset.tensors()
        observations = observations.float()
        actions = actions.float()
        if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
            raise FloatingPointError("non-finite DAgger observations or actions")
        weights = None
        if motion_weights and self.motion_keys:
            weights = torch.ones(observations.shape[0], dtype=torch.float32)
            for index, motion_id in enumerate(dataset.motion_ids().tolist()):
                if 0 <= int(motion_id) < len(self.motion_keys):
                    weights[index] = float(motion_weights.get(self.motion_keys[int(motion_id)], 1.0))
            weights = weights.clamp_min(1.0e-6)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + len(dataset.rounds))
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        losses: list[float] = []
        self.model.train()
        for step_index in range(self.train_steps):
            batch_count = min(self.batch_size, observations.shape[0])
            if weights is not None:
                indices = torch.multinomial(weights, batch_count, replacement=True, generator=generator)
            else:
                indices = torch.randint(observations.shape[0], (batch_count,), generator=generator)
            prediction = self._forward(observations[indices].to(self.device))
            target = actions[indices].to(self.device)
            loss = torch.mean(torch.square(prediction - target))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite student loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(parameters, self.grad_clip, error_if_nonfinite=True)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if self.train_steps >= 100 and (step_index + 1) % 100 == 0:
                prefix = f"{label} " if label else ""
                print(
                    f"[dagger] train {prefix}step={step_index + 1}/{self.train_steps} "
                    f"loss={losses[-1]:.6g} lr={self.peak_learning_rate:.3g} "
                    f"grad={float(gradient_norm):.3g}",
                    flush=True,
                )
        for parameter in parameters:
            if not torch.isfinite(parameter).all():
                raise FloatingPointError("non-finite student parameter after training")
        self.model.eval()
        chunk_size = max(self.batch_size, 256)
        predictions: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, observations.shape[0], chunk_size):
                chunk = observations[start : start + chunk_size].to(self.device)
                predictions.append(self._forward(chunk).cpu())
        prediction = torch.cat(predictions, dim=0)
        rmse = torch.sqrt(torch.mean(torch.square(prediction - actions))).item()
        return {
            "train_loss": losses[-1] if losses else float("nan"),
            "teacher_action_rmse": float(rmse),
            "mean_learning_rate": self.peak_learning_rate,
            "final_learning_rate": self.peak_learning_rate,
            "motion_weighting": bool(weights is not None),
        }

    def save(self, path: str, *, extra_metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": FORMAT_VERSION,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "layout": self.layout,
            "actor_obs_dim": self.actor_obs_dim,
            "input_dim": self.input_dim,
            "input_terms": list(self.input_terms),
            "action_dim": self.action_dim,
            "init_mode": self.init_mode,
            "normalization_mode": "none",
            "motion_keys": list(self.motion_keys),
        }
        if extra_metadata:
            payload.update(extra_metadata)
        torch.save(payload, path)

    def load(self, path: str | Path) -> dict:
        payload = torch.load(Path(path).expanduser(), map_location=self.device, weights_only=False)
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported SONIC-architecture student checkpoint")
        if int(payload.get("input_dim", -1)) != self.input_dim:
            raise ValueError("checkpoint input_dim does not match student")
        if int(payload.get("action_dim", -1)) != self.action_dim:
            raise ValueError("checkpoint action_dim does not match student")
        if payload.get("layout") != self.layout:
            raise ValueError("checkpoint tokenizer layout does not match student")
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        if isinstance(payload.get("optimizer_state_dict"), dict):
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.init_mode = str(payload.get("init_mode", self.init_mode))
        self.motion_keys = list(payload.get("motion_keys", []))
        self.model.eval()
        return payload
