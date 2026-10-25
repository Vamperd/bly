"""65-token CVAE student used by the external SONIC DAgger runner."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from dagger.core.cvae_dataset import CvaeDaggerDataset
from dagger.runtime.cvae_window import ANCHOR, CvaeWindow, CvaeWindowBuffer


@dataclass
class CvaeRollout:
    window: CvaeWindow
    action_normalized: torch.Tensor
    action_canonical: torch.Tensor
    action_raw: torch.Tensor
    predicted_state: torch.Tensor
    predicted_action: torch.Tensor
    epsilon_global: torch.Tensor
    epsilon_local: torch.Tensor


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CvaeStudentAdapter:
    """Wrap the existing CVAE without copying or modifying its source code."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        model_config: str | Path,
        normalization_run: str | Path,
        action_contract: Any,
        device: torch.device,
        seed: int = 20261024,
        learning_rate: float = 1.0e-5,
        weight_decay: float = 0.0,
        train_steps: int = 2000,
        batch_size: int = 32,
        source_mix: float = 0.5,
        mask_seed: int = 20261024,
        kl_beta: float = 1.0e-3,
        kl_warmup: int = 500,
    ) -> None:
        self.device = device
        self.action_contract = action_contract
        self.seed = int(seed)
        self.train_steps = int(train_steps)
        self.batch_size = int(batch_size)
        self.source_mix = float(source_mix)
        self.mask_seed = int(mask_seed)
        self.kl_beta = float(kl_beta)
        self.kl_warmup = max(1, int(kl_warmup))
        self._effective_kl_beta = self.kl_beta
        if not 0.0 <= self.source_mix <= 1.0:
            raise ValueError("source_mix must be in [0,1]")

        self.normalization_run = Path(normalization_run).expanduser().resolve()
        norm_path = self.normalization_run / "data" / "normalization.npz"
        if not norm_path.is_file():
            raise FileNotFoundError(f"CVAE normalization file is missing: {norm_path}")
        self.normalization_sha256 = _sha256(norm_path)
        with np.load(norm_path, allow_pickle=False) as norm:
            self.state_mean = torch.from_numpy(np.asarray(norm["physical_state_mean"], dtype=np.float32)).to(device)
            self.state_std = torch.from_numpy(np.asarray(norm["physical_state_std"], dtype=np.float32)).to(device)
            self.action_mean = torch.from_numpy(np.asarray(norm["action_mean"], dtype=np.float32)).to(device)
            self.action_std = torch.from_numpy(np.asarray(norm["action_std"], dtype=np.float32)).to(device)
        if tuple(self.state_mean.shape) != (70,) or tuple(self.state_std.shape) != (70,):
            raise ValueError("CVAE state normalization must have shape [70]")
        if tuple(self.action_mean.shape) != (29,) or tuple(self.action_std.shape) != (29,):
            raise ValueError("CVAE action normalization must have shape [29]")
        all_stats = torch.cat((self.state_mean, self.state_std, self.action_mean, self.action_std))
        if not torch.isfinite(all_stats).all():
            raise ValueError("CVAE normalization contains non-finite values")
        if bool((self.state_std <= 0).any()) or bool((self.action_std <= 0).any()):
            raise ValueError("CVAE normalization contains non-positive standard deviations")

        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"CVAE student checkpoint is missing: {self.checkpoint_path}")
        self.checkpoint_sha256 = _sha256(self.checkpoint_path)
        self.model, self.model_signature, self.checkpoint_stage = self._load_model(
            self.checkpoint_path, Path(model_config).expanduser().resolve(), device, self.normalization_sha256
        )
        self.model.eval()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        self.learning_rate = float(learning_rate)
        self._source_dataset = None
        self._buffer: CvaeWindowBuffer | None = None
        self._epsilon_global: torch.Tensor | None = None
        self._epsilon_local: torch.Tensor | None = None
        self._episode_counter = 0
        self.latent_mode = "fixed_standard_normal"

    @staticmethod
    def _load_model(checkpoint_path: Path, config_path: Path, device: torch.device, normalization_sha256: str):
        import sys

        cvae_root = config_path.parents[1]
        source_root = cvae_root / "src"
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from cvae_sa.models import build_model
        from cvae_sa.posterior_hierarchical_standard_cvae import load_checkpoint

        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        model_config = dict(config_payload.get("model", config_payload))
        if model_config.get("kind") != "physics_hierarchical_standard_cvae_transformer":
            raise ValueError("CVAE DAgger requires the 65-token hierarchical standard CVAE")
        model = build_model(model_config).to(device)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        signature = payload.get("model_signature", {})
        if signature.get("architecture_version") != "65-token-hierarchical-standard-cvae-v1":
            raise ValueError("student checkpoint has an incompatible CVAE architecture")
        expected_parameters = sum(parameter.numel() for parameter in model.parameters())
        if signature.get("parameter_count") is not None and int(signature["parameter_count"]) != expected_parameters:
            raise ValueError("student checkpoint parameter count differs from model config")
        stage = str(payload.get("stage", payload.get("training_stage", "C"))).upper()
        if stage not in {"C", "KL", "DAGGER"}:
            raise ValueError(f"student checkpoint must be a Stage-C checkpoint, found {stage}")
        checkpoint_identity = payload.get("dataset_identity") or {}
        checkpoint_norm = payload.get("normalization_sha256") or checkpoint_identity.get("normalization_sha256")
        if checkpoint_norm is not None and str(checkpoint_norm) != str(normalization_sha256):
            raise ValueError("student checkpoint normalization hash does not match CVAE dataset")
        if isinstance(payload.get("model_state_dict"), dict):
            model.load_state_dict(payload["model_state_dict"], strict=True)
        else:
            load_checkpoint(model, checkpoint_path, strict=True)
        model.eval()
        return model, signature, stage

    def reset(self, num_envs: int, current_state: torch.Tensor) -> None:
        self._buffer = CvaeWindowBuffer(device=self.device, num_envs=int(num_envs))
        self._buffer.reset(current_state)
        self._episode_counter += 1
        self._new_epsilon(torch.ones(int(num_envs), device=self.device, dtype=torch.bool))

    def _new_epsilon(self, mask: torch.Tensor) -> None:
        if self._buffer is None:
            raise RuntimeError("student has not been reset")
        count = self._buffer.num_envs
        if self._epsilon_global is None:
            self._epsilon_global = torch.zeros(count, self.model.global_latent_dim, device=self.device)
            self._epsilon_local = torch.zeros(count, 16, self.model.local_latent_dim, device=self.device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self._episode_counter * 100003)
        global_values = torch.randn(count, self.model.global_latent_dim, generator=generator)
        local_values = torch.randn(count, 16, self.model.local_latent_dim, generator=generator)
        self._epsilon_global[mask] = global_values.to(self.device)[mask]
        self._epsilon_local[mask] = local_values.to(self.device)[mask]

    def after_step(self, executed_action_canonical: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> None:
        if self._buffer is None:
            raise RuntimeError("student has not been reset")
        self._buffer.advance(executed_action_canonical, next_state, done)
        if bool(done.any()):
            self._episode_counter += 1
            self._new_epsilon(done)

    def set_latent_mode(self, mode: str) -> None:
        mode = str(mode)
        if mode not in {"fixed_standard_normal", "zero", "posterior_mean"}:
            raise ValueError(f"unsupported CVAE rollout latent mode: {mode}")
        self.latent_mode = mode

    @torch.no_grad()
    def observe(self) -> CvaeRollout:
        if self._buffer is None or self._epsilon_global is None or self._epsilon_local is None:
            raise RuntimeError("student has not been reset")
        window = self._buffer.build_condition(
            state_mean=self.state_mean,
            state_std=self.state_std,
            action_mean=self.action_mean,
            action_std=self.action_std,
        )
        if self.latent_mode == "posterior_mean":
            output = self.model(window.batch, stage="A")
        else:
            epsilon = (
                (torch.zeros_like(self._epsilon_global), torch.zeros_like(self._epsilon_local))
                if self.latent_mode == "zero"
                else (self._epsilon_global, self._epsilon_local)
            )
            output = self.model.infer_from_condition(
                window.batch, window.state_mask, window.action_mask, epsilon=epsilon
            )
        action_normalized = output.action[:, ANCHOR].float()
        action_canonical = action_normalized * self.action_std + self.action_mean
        action_raw, achieved = self.action_contract.canonical_to_raw(action_canonical)
        if not torch.isfinite(action_normalized).all() or not torch.isfinite(action_raw).all():
            raise FloatingPointError("CVAE student produced non-finite Action")
        if bool((achieved - action_canonical).abs().amax(dim=-1).gt(1.0e-4).any()):
            raise ValueError("CVAE canonical/raw action round-trip exceeds 1e-4")
        return CvaeRollout(
            window=window,
            action_normalized=action_normalized,
            action_canonical=action_canonical,
            action_raw=action_raw,
            predicted_state=output.physical_state.detach().clone(),
            predicted_action=output.action.detach().clone(),
            epsilon_global=self._epsilon_global.detach().clone(),
            epsilon_local=self._epsilon_local.detach().clone(),
        )

    def act(self, student_input: CvaeRollout) -> torch.Tensor:
        return student_input.action_raw

    def _load_source_dataset(self):
        if self._source_dataset is not None:
            return self._source_dataset
        if self.source_mix <= 0:
            return None
        from cvae_sa.dataset import StateActionWindowDataset

        self._source_dataset = StateActionWindowDataset(
            self.normalization_run,
            "train",
            window_transitions=64,
            validation_stride=64,
            random_crop=True,
        )
        return self._source_dataset

    def _source_batch(self, generator: torch.Generator) -> dict[str, torch.Tensor]:
        dataset = self._load_source_dataset()
        if dataset is None or len(dataset) == 0:
            raise RuntimeError("source CVAE dataset is unavailable")
        indices = torch.randint(len(dataset), (self.batch_size,), generator=generator).tolist()
        return default_collate([dataset[index] for index in indices])

    def _source_loss(self, batch: dict[str, torch.Tensor], step: int) -> tuple[torch.Tensor, dict[str, float]]:
        from cvae_sa.posterior_hierarchical_standard_cvae import hierarchical_kl, weighted_reconstruction_loss
        from cvae_sa.posterior_t64_protocol import make_physical_masks

        sm, am, _ = make_physical_masks(batch, self.mask_seed, dynamic_step=step)
        batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        output = self.model(batch, sm.to(self.device), am.to(self.device), stage="C")
        full_sm = batch["valid_state"][..., None].expand_as(batch["physical_state"])
        full_am = batch["valid_action"][..., None].expand_as(batch["action"])
        reconstruction = weighted_reconstruction_loss(output, batch, full_sm, full_am, masked_weight=0.0, full_weight=1.0)
        kl = hierarchical_kl(output.posterior)
        loss = reconstruction["total"].float() + self._effective_kl_beta * kl["total"].float()
        return loss, {"source_reconstruction": float(reconstruction["total"].detach()), "kl": float(kl["total"].detach())}

    def _dagger_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        from cvae_sa.posterior_hierarchical_standard_cvae import hierarchical_kl

        batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        output = self.model(batch, batch["state_mask"], batch["action_mask"], stage="C")
        state_error = (output.physical_state[..., :68].float() - batch["physical_state"][..., :68].float()).square()
        state_values = state_error.masked_select(batch["loss_state_mask"][..., :68])
        future_state = state_values.mean() if state_values.numel() else output.action.sum() * 0.0
        contact_error = F.binary_cross_entropy_with_logits(
            output.state_contact_logits.float(), batch["physical_state"][..., 68:70].float(), reduction="none"
        ).masked_select(batch["loss_state_mask"][..., 68:70])
        future_contact = contact_error.mean() if contact_error.numel() else future_state * 0.0
        action_error = (output.action[:, ANCHOR] - batch["teacher_action"]).square().mean()
        kl = hierarchical_kl(output.posterior)["total"]
        loss = future_state + 0.1 * future_contact + action_error + self._effective_kl_beta * kl
        return loss, {
            "future_state": float(future_state.detach()),
            "future_contact": float(future_contact.detach()),
            "teacher_action": float(action_error.detach()),
            "kl": float(kl.detach()),
        }

    def train_on_dataset(self, dataset: CvaeDaggerDataset) -> dict[str, float]:
        if not dataset.rounds:
            raise RuntimeError("cannot train CVAE student on an empty DAgger dataset")
        if self.source_mix > 0:
            self._load_source_dataset()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + len(dataset.rounds) * 1009)
        self.model.train()
        losses: list[float] = []
        aggregates: dict[str, list[float]] = {}
        for step in range(self.train_steps):
            self._effective_kl_beta = self.kl_beta * min(1.0, float(step + 1) / self.kl_warmup)
            use_source = self.source_mix > 0 and torch.rand((), generator=generator).item() < self.source_mix
            if use_source:
                batch = self._source_batch(generator)
                loss, metrics = self._source_loss(batch, step)
            else:
                batch = dataset.sample(self.batch_size, generator=generator)
                loss, metrics = self._dagger_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite CVAE DAgger loss at step {step + 1}")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            before = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
            self.optimizer.step()
            losses.append(float(loss.detach()))
            for key, value in metrics.items():
                aggregates.setdefault(key, []).append(value)
            aggregates.setdefault("gradient_norm", []).append(float(before))
        self.model.eval()
        return {
            "train_loss": float(np.mean(losses[-min(100, len(losses)):])),
            "dataset_samples": float(dataset.count),
            **{f"mean_{key}": float(np.mean(values)) for key, values in aggregates.items()},
        }

    def save(self, path: str | Path, *, round_index: int | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": "dagger_cvae_student_checkpoint_v1",
                "stage": "DAGGER",
                "architecture_version": "65-token-hierarchical-standard-cvae-v1",
                "model_signature": self.model_signature,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "normalization_run": str(self.normalization_run),
                "normalization_sha256": self.normalization_sha256,
                "state_mean": self.state_mean.cpu(),
                "state_std": self.state_std.cpu(),
                "action_mean": self.action_mean.cpu(),
                "action_std": self.action_std.cpu(),
                "round_index": round_index,
            },
            path,
        )

