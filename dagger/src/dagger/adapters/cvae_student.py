"""65-token CVAE student used by the external SONIC DAgger runner.

The adapter intentionally imports the CVAE project lazily.  The DAgger
workspace owns the training loop, while ``state-action-cvae`` remains the
read-only source of the model, normalization and long-horizon dataset helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate

from dagger.core.cvae_dataset import CvaeDaggerDataset
from dagger.core.cvae_objective import (
    compose_dagger_loss,
    masked_future_contact_bce,
    masked_future_state_mse,
    teacher_action_mse,
)
from dagger.runtime.cvae_window import (
    ANCHOR,
    CvaeWindow,
    CvaeWindowBuffer,
    causal_masks,
)


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
    action_clip_error: torch.Tensor | None = None


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
        cvae_root: str | Path | None = None,
        seed: int = 20261024,
        learning_rate: float = 1.0e-5,
        weight_decay: float = 0.0,
        train_steps: int = 2000,
        batch_size: int = 32,
        source_mix: float = 0.5,
        source_mix_mode: str = "random",
        mask_seed: int = 20261024,
        kl_beta: float = 1.0e-3,
        kl_warmup: int = 500,
        source_dataset_run: str | Path | None = None,
        source_dataset_kind: str = "base",
        source_mask_mode: str = "dynamic",
        source_causal_fraction: float = 0.0,
        teacher_mode: str = "posterior_sample",
        source_identity: dict[str, Any] | None = None,
        state_weight: float = 1.0,
        contact_weight: float = 0.1,
        teacher_sn_weight: float = 0.5,
        teacher_pm_weight: float = 0.5,
        action_clip_value: float | None = None,
        lr_schedule: str = "constant",
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.01,
        load_optimizer: bool = False,
        raise_on_clip_error: bool = True,
    ) -> None:
        self.device = device
        self.action_contract = action_contract
        self.seed = int(seed)
        self.train_steps = int(train_steps)
        self.batch_size = int(batch_size)
        self.source_mix = float(source_mix)
        self.source_mix_mode = str(source_mix_mode)
        self.mask_seed = int(mask_seed)
        self.kl_beta = float(kl_beta)
        self.kl_warmup = max(1, int(kl_warmup))
        self._effective_kl_beta = self.kl_beta
        self.source_dataset_kind = str(source_dataset_kind)
        self.source_mask_mode = str(source_mask_mode)
        self.source_causal_fraction = float(source_causal_fraction)
        self.source_identity = dict(source_identity or {})
        self.source_mix_exact = bool(self.source_identity.get("source_mix_exact", False))
        self.source_hash_mismatch: str | None = None
        self.state_weight = float(state_weight)
        self.contact_weight = float(contact_weight)
        self.teacher_sn_weight = float(teacher_sn_weight)
        self.teacher_pm_weight = float(teacher_pm_weight)
        self.teacher_mode = str(teacher_mode)
        self.raise_on_clip_error = bool(raise_on_clip_error)
        self.action_clip_value = None if action_clip_value is None else float(action_clip_value)
        self.lr_schedule = str(lr_schedule)
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        self.load_optimizer = bool(load_optimizer)
        self.global_step = 0
        if self.source_mix_mode not in {"random", "alternate"}:
            raise ValueError("source_mix_mode must be random or alternate")
        if self.source_dataset_kind not in {"long_horizon", "base"}:
            raise ValueError("source_dataset_kind must be long_horizon or base")
        if self.source_mask_mode not in {"fixed", "dynamic"}:
            raise ValueError("source_mask_mode must be fixed or dynamic")
        if self.teacher_mode not in {"dual", "posterior_sample"}:
            raise ValueError("teacher_mode must be dual or posterior_sample")
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("lr_schedule must be constant or cosine")
        if not 0.0 <= self.source_mix <= 1.0:
            raise ValueError("source_mix must be in [0,1]")
        if not 0.0 <= self.source_causal_fraction <= 1.0:
            raise ValueError("source_causal_fraction must be in [0,1]")

        self.normalization_run = Path(normalization_run).expanduser().resolve()
        self.source_dataset_run = Path(
            source_dataset_run if source_dataset_run is not None else normalization_run
        ).expanduser().resolve()
        self.cvae_root = (
            Path(cvae_root).expanduser().resolve() if cvae_root is not None else None
        )

        norm_path = self.normalization_run / "data" / "normalization.npz"
        if not norm_path.is_file():
            raise FileNotFoundError(f"CVAE normalization file is missing: {norm_path}")
        self.normalization_sha256 = _sha256(norm_path)
        with np.load(norm_path, allow_pickle=False) as norm:
            self.state_mean = torch.from_numpy(
                np.asarray(norm["physical_state_mean"], dtype=np.float32)
            ).to(device)
            self.state_std = torch.from_numpy(
                np.asarray(norm["physical_state_std"], dtype=np.float32)
            ).to(device)
            self.action_mean = torch.from_numpy(
                np.asarray(norm["action_mean"], dtype=np.float32)
            ).to(device)
            self.action_std = torch.from_numpy(
                np.asarray(norm["action_std"], dtype=np.float32)
            ).to(device)
        if tuple(self.state_mean.shape) != (70,) or tuple(self.state_std.shape) != (70,):
            raise ValueError("CVAE state normalization must have shape [70]")
        if tuple(self.action_mean.shape) != (29,) or tuple(self.action_std.shape) != (29,):
            raise ValueError("CVAE action normalization must have shape [29]")
        all_stats = torch.cat(
            (self.state_mean, self.state_std, self.action_mean, self.action_std)
        )
        if not torch.isfinite(all_stats).all():
            raise ValueError("CVAE normalization contains non-finite values")
        if bool((self.state_std <= 0).any()) or bool((self.action_std <= 0).any()):
            raise ValueError("CVAE normalization contains non-positive standard deviations")

        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"CVAE student checkpoint is missing: {self.checkpoint_path}")
        self.checkpoint_sha256 = _sha256(self.checkpoint_path)
        (
            self.model,
            self.model_signature,
            self.checkpoint_stage,
            self.model_config,
            checkpoint_identity,
            checkpoint_contract,
        ) = self._load_model(
            self.checkpoint_path,
            Path(model_config).expanduser().resolve(),
            device,
            self.normalization_sha256,
            self.cvae_root,
        )
        self.source_identity = {
            **dict(checkpoint_identity),
            **self.source_identity,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_stage": self.checkpoint_stage,
            "training_contract": checkpoint_contract,
            "model_config": self.model_config,
        }
        self.source_mix_exact = bool(self.source_identity.get("source_mix_exact", False))
        self.model.eval()
        self.peak_learning_rate = float(learning_rate)
        self.minimum_learning_rate = self.peak_learning_rate * self.min_lr_ratio
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.learning_rate = float(learning_rate)
        self._source_dataset = None
        self._buffer: CvaeWindowBuffer | None = None
        self._epsilon_global: torch.Tensor | None = None
        self._epsilon_local: torch.Tensor | None = None
        self._episode_counter = 0
        self._evaluation_mode = False
        self.latent_mode = "fixed_standard_normal"
        self.last_action_clip_error: float | None = None
        if self.action_contract is not None and hasattr(
            self.action_contract, "set_raw_action_clip_value"
        ):
            self.action_contract.set_raw_action_clip_value(self.action_clip_value)

    @staticmethod
    def _load_model(
        checkpoint_path: Path,
        config_path: Path,
        device: torch.device,
        normalization_sha256: str,
        cvae_root: Path | None,
    ):
        import sys

        root = cvae_root
        if root is None:
            root = config_path.parents[1]
        source_root = root / "src"
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from cvae_sa.models import build_model

        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        external_config = dict(config_payload.get("model", config_payload))
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        signature = payload.get("model_signature", {})
        if signature.get("architecture_version") != "65-token-hierarchical-standard-cvae-v1":
            raise ValueError("student checkpoint has an incompatible CVAE architecture")
        checkpoint_config = payload.get("model_config")
        if checkpoint_config is not None:
            checkpoint_config = dict(checkpoint_config)
            if checkpoint_config != external_config:
                differing = sorted(
                    key
                    for key in set(checkpoint_config) | set(external_config)
                    if checkpoint_config.get(key) != external_config.get(key)
                )
                raise ValueError(
                    "student model_config differs from checkpoint: "
                    f"{differing}; use the checkpoint's model_config"
                )
            model_config = checkpoint_config
        else:
            model_config = external_config
        if model_config.get("kind") != "physics_hierarchical_standard_cvae_transformer":
            raise ValueError("CVAE DAgger requires the 65-token hierarchical standard CVAE")
        model = build_model(model_config).to(device)
        expected_parameters = sum(parameter.numel() for parameter in model.parameters())
        if signature.get("parameter_count") is not None and int(
            signature["parameter_count"]
        ) != expected_parameters:
            raise ValueError("student checkpoint parameter count differs from model config")
        stage = str(payload.get("stage", payload.get("training_stage", "C"))).upper()
        if stage not in {"C", "KL", "DAGGER"}:
            raise ValueError(f"student checkpoint must be a Stage-C checkpoint, found {stage}")
        checkpoint_identity = payload.get("dataset_identity") or {}
        checkpoint_norm = payload.get("normalization_sha256") or checkpoint_identity.get(
            "normalization_sha256"
        )
        if checkpoint_norm is not None and str(checkpoint_norm) != str(normalization_sha256):
            raise ValueError("student checkpoint normalization hash does not match CVAE dataset")
        state = payload.get("model_state_dict")
        if not isinstance(state, dict):
            state = payload.get("model")
        if not isinstance(state, dict):
            raise ValueError("student checkpoint does not contain model weights")
        model.load_state_dict(state, strict=True)
        model.eval()
        return (
            model,
            signature,
            stage,
            model_config,
            dict(payload.get("dataset_identity") or {}),
            dict(payload.get("training_contract") or {}),
        )

    def reset(self, num_envs: int, current_state: torch.Tensor) -> None:
        self._buffer = CvaeWindowBuffer(device=self.device, num_envs=int(num_envs))
        self._buffer.reset(current_state)
        self._episode_counter += 1
        if (
            self._epsilon_global is None
            or int(self._epsilon_global.shape[0]) != int(num_envs)
        ):
            self._epsilon_global = torch.zeros(
                int(num_envs), self.model.global_latent_dim, device=self.device
            )
            self._epsilon_local = torch.zeros(
                int(num_envs), 16, self.model.local_latent_dim, device=self.device
            )
        self._new_epsilon(
            torch.ones(int(num_envs), device=self.device, dtype=torch.bool)
        )
        if self.action_contract is not None and hasattr(
            self.action_contract, "set_raw_action_clip_value"
        ):
            self.action_contract.set_raw_action_clip_value(self.action_clip_value)

    def _new_epsilon(self, mask: torch.Tensor) -> None:
        if self._buffer is None or self._epsilon_global is None or self._epsilon_local is None:
            raise RuntimeError("student has not been reset")
        count = self._buffer.num_envs
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self._episode_counter * 100003)
        global_values = torch.randn(
            count, self.model.global_latent_dim, generator=generator
        )
        local_values = torch.randn(
            count, 16, self.model.local_latent_dim, generator=generator
        )
        self._epsilon_global[mask] = global_values.to(self.device)[mask]
        self._epsilon_local[mask] = local_values.to(self.device)[mask]

    def after_step(
        self,
        executed_action_canonical: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        if self._buffer is None:
            raise RuntimeError("student has not been reset")
        self._buffer.advance(executed_action_canonical, next_state, done)
        if bool(done.any()) and not self._evaluation_mode:
            self._episode_counter += 1
            self._new_epsilon(done)

    def set_latent_mode(self, mode: str) -> None:
        mode = str(mode)
        if mode not in {"fixed_standard_normal", "zero", "posterior_mean"}:
            raise ValueError(f"unsupported CVAE rollout latent mode: {mode}")
        self.latent_mode = mode

    def begin_evaluation(self) -> None:
        """Restart the fixed standard-normal epsilon sequence for comparable evals."""

        self._evaluation_mode = True
        self._episode_counter = 1_000_000

    def end_evaluation(self) -> None:
        self._evaluation_mode = False

    def set_action_clip_value(self, value: float | None) -> None:
        self.action_clip_value = None if value is None else float(value)
        if self.action_contract is not None and hasattr(
            self.action_contract, "set_raw_action_clip_value"
        ):
            self.action_contract.set_raw_action_clip_value(self.action_clip_value)

    @torch.no_grad()
    def observe(self) -> CvaeRollout:
        if self._buffer is None or self._epsilon_global is None or self._epsilon_local is None:
            raise RuntimeError("student has not been reset")
        self.model.eval()
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
        clip_error = (achieved - action_canonical).abs().amax(dim=-1)
        if self.raise_on_clip_error and bool((clip_error > 1.0e-4).any()):
            raise ValueError("CVAE canonical/raw action round-trip exceeds 1e-4")
        action_canonical = achieved
        if not torch.isfinite(action_normalized).all() or not torch.isfinite(action_raw).all():
            raise FloatingPointError("CVAE student produced non-finite Action")
        self.last_action_clip_error = float(clip_error.max().detach().cpu())
        return CvaeRollout(
            window=window,
            action_normalized=action_normalized,
            action_canonical=action_canonical,
            action_raw=action_raw,
            predicted_state=output.physical_state.detach().clone(),
            predicted_action=output.action.detach().clone(),
            epsilon_global=self._epsilon_global.detach().clone(),
            epsilon_local=self._epsilon_local.detach().clone(),
            action_clip_error=clip_error.detach().clone(),
        )

    def act(self, student_input: CvaeRollout) -> torch.Tensor:
        return student_input.action_raw

    def _load_source_dataset(self):
        if self._source_dataset is not None:
            return self._source_dataset
        if self.source_mix <= 0:
            return None
        source_norm = self.source_dataset_run / "data" / "normalization.npz"
        if source_norm.is_file() and _sha256(source_norm) != self.normalization_sha256:
            raise ValueError(
                "source dataset normalization hash does not match the checkpoint "
                f"normalization: {source_norm}"
            )
        if self.source_dataset_kind == "long_horizon":
            from cvae_sa.long_horizon_dataset import LongHorizonWindowDataset

            expected_windows = self.source_identity.get("long_horizon_windows_sha256")
            expected_groups = self.source_identity.get("long_horizon_groups_sha256")
            for expected, relative, name in (
                (
                    expected_windows,
                    "manifests/long_horizon_windows.jsonl",
                    "windows",
                ),
                (
                    expected_groups,
                    "manifests/long_horizon_groups.jsonl",
                    "groups",
                ),
            ):
                path = self.source_dataset_run / relative
                if not expected or not path.is_file():
                    continue
                actual = _sha256(path)
                if actual != str(expected):
                    if self.source_mix_exact:
                        raise ValueError(
                            f"source long-horizon {name} hash does not match the "
                            f"requested exact identity: {actual} != {expected}"
                        )
                    self.source_hash_mismatch = f"{name}: {actual} != {expected}"
                    self.source_mix_exact = False
            self._source_dataset = LongHorizonWindowDataset(
                self.source_dataset_run, split="train", include_tail=True
            )
        else:
            from cvae_sa.dataset import StateActionWindowDataset

            self._source_dataset = StateActionWindowDataset(
                self.normalization_run,
                "train",
                window_transitions=64,
                validation_stride=64,
                random_crop=True,
            )
        self.source_identity["source_mix_exact"] = bool(self.source_mix_exact)
        self.source_identity["source_hash_mismatch"] = self.source_hash_mismatch
        return self._source_dataset

    def _source_batch(self, generator: torch.Generator) -> dict[str, torch.Tensor]:
        dataset = self._load_source_dataset()
        if dataset is None or len(dataset) == 0:
            raise RuntimeError("source CVAE dataset is unavailable")
        indices = torch.randint(
            len(dataset), (self.batch_size,), generator=generator
        ).tolist()
        return default_collate([dataset[index] for index in indices])

    def _source_loss(
        self, batch: dict[str, torch.Tensor], generator: torch.Generator
    ) -> tuple[torch.Tensor, dict[str, float]]:
        from cvae_sa.posterior_hierarchical_standard_cvae import (
            hierarchical_kl,
            weighted_reconstruction_loss,
        )
        from cvae_sa.posterior_t64_protocol import make_physical_masks

        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        valid_state = batch["valid_state"].bool()
        valid_action = batch["valid_action"].bool()
        use_causal = (
            self.source_causal_fraction > 0.0
            and float(torch.rand((), generator=generator).item())
            < self.source_causal_fraction
        )
        if use_causal and not bool(
            valid_state[:, ANCHOR + 1 :].any() or valid_action[:, ANCHOR].any()
        ):
            use_causal = False
        if use_causal:
            sm, am, lsm, lam = causal_masks(
                int(valid_state.shape[0]),
                device=self.device,
                valid_state=valid_state,
                valid_action=valid_action,
            )
            masked_weight, full_weight = 1.0, 0.0
        else:
            dynamic_step = self.global_step if self.source_mask_mode == "dynamic" else None
            sm, am, _ = make_physical_masks(
                batch, self.mask_seed, dynamic_step=dynamic_step
            )
            lsm = valid_state[..., None].expand_as(batch["physical_state"])
            lam = valid_action[..., None].expand_as(batch["action"])
            masked_weight, full_weight = 0.0, 1.0
        output = self.model(batch, sm.to(self.device), am.to(self.device), stage="C")
        reconstruction = weighted_reconstruction_loss(
            output,
            batch,
            lsm,
            lam,
            masked_weight=masked_weight,
            full_weight=full_weight,
        )
        kl = hierarchical_kl(output.posterior)["total"]
        loss = reconstruction["total"].float() + self._effective_kl_beta * kl.float()
        return loss, {
            "source_reconstruction": float(reconstruction["total"].detach()),
            "source_state": float(reconstruction["state"].detach()),
            "source_action": float(reconstruction["action"].detach()),
            "source_contact": float(reconstruction["contact"].detach()),
            "source_kl": float(kl.detach()),
            "source_causal": float(use_causal),
        }

    def _dagger_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        from cvae_sa.posterior_hierarchical_standard_cvae import hierarchical_kl

        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        if self.teacher_mode == "posterior_sample":
            output = self.model(
                batch,
                batch["state_mask"],
                batch["action_mask"],
                stage="C",
            )
            future_state = masked_future_state_mse(
                output.physical_state,
                batch["physical_state"],
                batch["loss_state_mask"],
            )
            future_contact = masked_future_contact_bce(
                output.state_contact_logits,
                batch["physical_state"],
                batch["loss_state_mask"],
            )
            teacher = teacher_action_mse(output.action, batch["teacher_action"], ANCHOR)
            kl = hierarchical_kl(output.posterior)["total"]
            loss = compose_dagger_loss(
                future_state=future_state,
                future_contact=future_contact,
                teacher_standard_normal=teacher,
                teacher_posterior_mean=teacher,
                kl=kl,
                state_weight=self.state_weight,
                contact_weight=self.contact_weight,
                standard_normal_weight=0.5,
                posterior_mean_weight=0.5,
                kl_beta=self._effective_kl_beta,
            )
            return loss, {
                "future_state": float(future_state.detach()),
                "future_contact": float(future_contact.detach()),
                "teacher_standard_normal": 0.0,
                "teacher_posterior_mean": float(teacher.detach()),
                "kl": float(kl.detach()),
            }
        posterior = self.model.encode_posterior_distribution(batch)
        condition = self.model.encode_condition(
            batch, batch["state_mask"], batch["action_mask"]
        )
        posterior_mean = self.model.decode_from_fused_latents(
            batch,
            self.model.fuse_latents(posterior, condition),
            condition,
        )
        batch_size = int(batch["physical_state"].shape[0])
        epsilon_global = torch.randn(
            batch_size,
            self.model.global_latent_dim,
            device=self.device,
            dtype=torch.float32,
        )
        epsilon_local = torch.randn(
            batch_size,
            16,
            self.model.local_latent_dim,
            device=self.device,
            dtype=torch.float32,
        )
        standard_normal = self.model.decode_from_fused_latents(
            batch,
            self.model.fuse_latents((epsilon_global, epsilon_local), condition),
            condition,
        )
        future_state = masked_future_state_mse(
            posterior_mean.physical_state,
            batch["physical_state"],
            batch["loss_state_mask"],
        )
        future_contact = masked_future_contact_bce(
            posterior_mean.state_contact_logits,
            batch["physical_state"],
            batch["loss_state_mask"],
        )
        teacher_sn = teacher_action_mse(
            standard_normal.action, batch["teacher_action"], ANCHOR
        )
        teacher_pm = teacher_action_mse(
            posterior_mean.action, batch["teacher_action"], ANCHOR
        )
        kl = hierarchical_kl(posterior)["total"]
        loss = compose_dagger_loss(
            future_state=future_state,
            future_contact=future_contact,
            teacher_standard_normal=teacher_sn,
            teacher_posterior_mean=teacher_pm,
            kl=kl,
            state_weight=self.state_weight,
            contact_weight=self.contact_weight,
            standard_normal_weight=self.teacher_sn_weight,
            posterior_mean_weight=self.teacher_pm_weight,
            kl_beta=self._effective_kl_beta,
        )
        return loss, {
            "future_state": float(future_state.detach()),
            "future_contact": float(future_contact.detach()),
            "teacher_standard_normal": float(teacher_sn.detach()),
            "teacher_posterior_mean": float(teacher_pm.detach()),
            "kl": float(kl.detach()),
        }

    def _learning_rate(self, global_step: int, total_steps: int) -> float:
        if self.lr_schedule == "constant":
            return self.peak_learning_rate
        if self.warmup_steps > 0 and global_step < self.warmup_steps:
            return self.peak_learning_rate * float(global_step + 1) / self.warmup_steps
        progress = min(
            1.0,
            max(
                0.0,
                (global_step - self.warmup_steps)
                / max(1, total_steps - self.warmup_steps),
            ),
        )
        return self.minimum_learning_rate + 0.5 * (
            self.peak_learning_rate - self.minimum_learning_rate
        ) * (1.0 + math.cos(math.pi * progress))

    def train_on_dataset(
        self,
        dataset: CvaeDaggerDataset,
        *,
        round_index: int | None = None,
        global_step_offset: int = 0,
        total_training_steps: int = 0,
        motion_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if not dataset.rounds:
            raise RuntimeError("cannot train CVAE student on an empty DAgger dataset")
        if self.source_mix > 0:
            self._load_source_dataset()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self.seed + len(dataset.rounds) * 1009 + int(global_step_offset)
        )
        total_steps = max(
            int(total_training_steps), int(global_step_offset) + self.train_steps
        )
        self.model.train()
        losses: list[float] = []
        learning_rates: list[float] = []
        aggregates: dict[str, list[float]] = {}
        source_steps = 0
        dagger_steps = 0
        for step in range(self.train_steps):
            global_step = int(global_step_offset) + step
            learning_rate = self._learning_rate(global_step, total_steps)
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            self.global_step = global_step
            self._effective_kl_beta = self.kl_beta * min(
                1.0, float(global_step + 1) / self.kl_warmup
            )
            if self.source_mix_mode == "alternate":
                use_source = bool(self.source_mix > 0 and step % 2 == 0)
            else:
                use_source = bool(
                    self.source_mix > 0
                    and float(torch.rand((), generator=generator).item()) < self.source_mix
                )
            if use_source:
                batch = self._source_batch(generator)
                loss, metrics = self._source_loss(batch, generator)
                source_steps += 1
            else:
                batch = dataset.sample(
                    self.batch_size,
                    generator=generator,
                    motion_weights=motion_weights,
                )
                loss, metrics = self._dagger_loss(batch)
                dagger_steps += 1
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite CVAE DAgger loss at step {step + 1}")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            before = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0, error_if_nonfinite=True
            )
            self.optimizer.step()
            losses.append(float(loss.detach()))
            learning_rates.append(learning_rate)
            for key, value in metrics.items():
                aggregates.setdefault(key, []).append(value)
            aggregates.setdefault("gradient_norm", []).append(float(before))
            if self.train_steps >= 100 and (step + 1) % 100 == 0:
                print(
                    f"[dagger] cvae train round={round_index} "
                    f"step={step + 1}/{self.train_steps} "
                    f"loss={losses[-1]:.6g} lr={learning_rate:.3g} "
                    f"source={int(use_source)} grad={float(before):.3g}",
                    flush=True,
                )
        self.model.eval()
        self.global_step = int(global_step_offset) + self.train_steps
        return {
            "train_loss": float(np.mean(losses[-min(100, len(losses)):])),
            "dataset_samples": float(dataset.count),
            "source_steps": float(source_steps),
            "dagger_steps": float(dagger_steps),
            "mean_learning_rate": float(np.mean(learning_rates)) if learning_rates else 0.0,
            "final_learning_rate": float(learning_rates[-1]) if learning_rates else 0.0,
            "motion_weighting": bool(motion_weights),
            **{f"mean_{key}": float(np.mean(values)) for key, values in aggregates.items()},
        }

    def save(
        self,
        path: str | Path,
        *,
        round_index: int | None = None,
        beta: float | None = None,
        motion_keys: list[str] | None = None,
        eval_metrics: dict[str, Any] | None = None,
        sample_count: int | None = None,
        motion_coverage: dict[str, int] | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.source_identity["source_mix_exact"] = bool(self.source_mix_exact)
        self.source_identity["source_hash_mismatch"] = self.source_hash_mismatch
        payload = {
            "format_version": "dagger_cvae_student_checkpoint_v1",
            "stage": "DAGGER",
            "architecture_version": self.model_signature.get("architecture_version")
            or "65-token-hierarchical-standard-cvae-v1",
            "model_signature": self.model_signature,
            "model_config": self.model_config,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "normalization_run": str(self.normalization_run),
            "normalization_sha256": self.normalization_sha256,
            "state_mean": self.state_mean.cpu(),
            "state_std": self.state_std.cpu(),
            "action_mean": self.action_mean.cpu(),
            "action_std": self.action_std.cpu(),
            "round_index": round_index,
            "beta": beta,
            "motion_keys": list(motion_keys or []),
            "eval_metrics": eval_metrics,
            "sample_count": sample_count,
            "motion_coverage": motion_coverage,
            "global_step": int(self.global_step),
            "source_identity": self.source_identity,
            "source_dataset_run": str(self.source_dataset_run),
            "source_dataset_kind": self.source_dataset_kind,
            "source_mask_mode": self.source_mask_mode,
            "source_mix_exact": bool(self.source_mix_exact),
            "source_hash_mismatch": self.source_hash_mismatch,
            "kl_beta": float(self.kl_beta),
            "kl_warmup": int(self.kl_warmup),
            "action_clip_value": self.action_clip_value,
            "teacher_mode": self.teacher_mode,
            "raise_on_clip_error": bool(self.raise_on_clip_error),
        }
        torch.save(payload, path)

    def load_round_checkpoint(
        self, path: str | Path, *, load_optimizer: bool = True
    ) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format_version") != "dagger_cvae_student_checkpoint_v1":
            raise ValueError(f"unsupported CVAE DAgger checkpoint: {path}")
        expected = self.model_signature.get("architecture_version")
        if payload.get("architecture_version") != expected:
            raise ValueError("CVAE DAgger checkpoint architecture mismatch")
        if str(payload.get("normalization_sha256")) != str(self.normalization_sha256):
            raise ValueError("CVAE DAgger checkpoint normalization mismatch")
        state = payload.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("CVAE DAgger checkpoint has no model weights")
        self.model.load_state_dict(state, strict=True)
        if load_optimizer and isinstance(payload.get("optimizer_state_dict"), dict):
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.global_step = int(payload.get("global_step", 0))
        self.source_identity = dict(payload.get("source_identity") or self.source_identity)
        self.source_mix_exact = bool(
            payload.get("source_mix_exact", self.source_mix_exact)
        )
        self.source_hash_mismatch = payload.get("source_hash_mismatch")
        self.checkpoint_stage = str(payload.get("stage", "DAGGER"))
        self.model.eval()
        return payload
