"""11-State/10-Action MLP CVAE with causal and block completion masks."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .mlp_action11 import ACTION_DIM, ACTION_STEPS, STATE_DIM, STATE_STEPS, Action11WindowDataset
from .util import atomic_torch_save, atomic_write_json, atomic_write_text, load_config, seed_everything

ARCHITECTURE_VERSION = "11-frame-mlp-cvae-v1"
CHECKPOINT_FORMAT = "sonic_11_frame_mlp_cvae_checkpoint_v1"
SUMMARY_FORMAT = "sonic_11_frame_mlp_cvae_summary_v1"
LATENT_DIM = 384


@dataclass
class MaskBatch:
    state_visible: torch.Tensor
    action_visible: torch.Tensor
    state_valid: torch.Tensor
    action_valid: torch.Tensor
    target_state: torch.Tensor
    target_action: torch.Tensor
    family: tuple[str, ...]


def _mlp(in_dim: int, widths: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = int(in_dim)
    for width in widths:
        layers.extend((nn.Linear(current, int(width)), nn.LayerNorm(int(width)), nn.GELU()))
        current = int(width)
    layers.append(nn.Linear(current, int(out_dim)))
    return nn.Sequential(*layers)


class MLPConditionalVAE11(nn.Module):
    """Pure MLP posterior/prior CVAE for the fixed 11/10 contract."""
    ARCHITECTURE_VERSION = ARCHITECTURE_VERSION

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.latent_dim = int(config.get("latent_dim", LATENT_DIM))
        self.encoder_widths = [int(v) for v in config.get("encoder_widths", [2816, 1792, 896, 512])]
        self.decoder_widths = [int(v) for v in config.get("decoder_widths", [2048, 1536, 768])]
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.state_steps = STATE_STEPS
        self.action_steps = ACTION_STEPS
        self.condition_dim = (
            STATE_STEPS * STATE_DIM + STATE_STEPS
            + ACTION_STEPS * ACTION_DIM + ACTION_STEPS
            + STATE_STEPS + ACTION_STEPS
            + STATE_STEPS + ACTION_STEPS
            + ACTION_DIM + 1
        )
        self.output_dim = STATE_STEPS * STATE_DIM + ACTION_STEPS * ACTION_DIM
        self.posterior_encoder = _mlp(self.condition_dim, self.encoder_widths, self.encoder_widths[-1])
        self.prior_encoder = _mlp(self.condition_dim, self.encoder_widths, self.encoder_widths[-1])
        self.posterior_mean = nn.Linear(self.encoder_widths[-1], self.latent_dim)
        self.posterior_logvar = nn.Linear(self.encoder_widths[-1], self.latent_dim)
        self.prior_mean = nn.Linear(self.encoder_widths[-1], self.latent_dim)
        self.prior_logvar = nn.Linear(self.encoder_widths[-1], self.latent_dim)
        self.decoder = _mlp(self.condition_dim + self.latent_dim, self.decoder_widths, self.output_dim)
        nn.init.zeros_(self.posterior_logvar.weight)
        nn.init.constant_(self.posterior_logvar.bias, -4.0)
        nn.init.zeros_(self.prior_logvar.weight)
        nn.init.constant_(self.prior_logvar.bias, -4.0)

    def architecture_signature(self) -> dict[str, Any]:
        return {
            "architecture_version": ARCHITECTURE_VERSION,
            "state_shape": [STATE_STEPS, STATE_DIM],
            "action_shape": [ACTION_STEPS, ACTION_DIM],
            "latent_dim": self.latent_dim,
            "encoder_widths": self.encoder_widths,
            "decoder_widths": self.decoder_widths,
            "condition_dim": self.condition_dim,
            "output_dim": self.output_dim,
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "latent_contract": "posterior_q_and_prior_p_global_gaussian_sampled_at_decode",
        }

    def _condition_vector(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        state_visible: torch.Tensor,
        action_visible: torch.Tensor,
        state_valid: torch.Tensor,
        action_valid: torch.Tensor,
        target_state: torch.Tensor,
        target_action: torch.Tensor,
        action_before: torch.Tensor,
        action_before_valid: torch.Tensor,
        *,
        use_visibility: bool = True,
    ) -> torch.Tensor:
        visible_state = state_visible & state_valid
        visible_action = action_visible & action_valid
        state_value_mask = visible_state if use_visibility else state_valid
        action_value_mask = visible_action if use_visibility else action_valid
        state_mask = visible_state.to(state.dtype)
        action_mask = visible_action.to(action.dtype)
        state_value = state * state_value_mask.to(state.dtype).unsqueeze(-1)
        action_value = action * action_value_mask.to(action.dtype).unsqueeze(-1)
        before_valid = action_before_valid.to(action.dtype).unsqueeze(-1)
        before_value = action_before * before_valid
        return torch.cat(
            (
                state_value.flatten(1), state_mask.flatten(1),
                action_value.flatten(1), action_mask.flatten(1),
                state_valid.to(state.dtype), action_valid.to(action.dtype),
                target_state.to(state.dtype), target_action.to(action.dtype),
                before_value, before_valid,
            ),
            dim=1,
        )

    @staticmethod
    def sample_gaussian(mean: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mean
        epsilon = torch.randn_like(mean)
        return mean + torch.exp(0.5 * logvar.clamp(-8.0, 4.0)) * epsilon

    def prior_forward(
        self,
        condition_state: torch.Tensor,
        condition_action: torch.Tensor,
        state_visible: torch.Tensor,
        action_visible: torch.Tensor,
        state_valid: torch.Tensor,
        action_valid: torch.Tensor,
        target_state: torch.Tensor,
        target_action: torch.Tensor,
        action_before: torch.Tensor,
        action_before_valid: torch.Tensor,
        *,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run only p(z|condition) and the deployment decoder path."""
        condition = self._condition_vector(
            condition_state, condition_action, state_visible, action_visible,
            state_valid, action_valid, target_state, target_action,
            action_before, action_before_valid,
        )
        prior_hidden = self.prior_encoder(condition)
        p_mean = self.prior_mean(prior_hidden)
        p_logvar = self.prior_logvar(prior_hidden).clamp(-8.0, 4.0)
        z_p = self.sample_gaussian(p_mean, p_logvar, sample)
        p_output = self.decoder(torch.cat((condition, z_p), dim=1))
        return {
            "p_output": p_output[..., : STATE_STEPS * STATE_DIM].reshape(-1, STATE_STEPS, STATE_DIM),
            "p_action": p_output[..., STATE_STEPS * STATE_DIM :].reshape(-1, ACTION_STEPS, ACTION_DIM),
            "p_mean": p_mean,
            "p_logvar": p_logvar,
        }

    def forward(
        self,
        condition_state: torch.Tensor,
        condition_action: torch.Tensor,
        state_visible: torch.Tensor,
        action_visible: torch.Tensor,
        state_valid: torch.Tensor,
        action_valid: torch.Tensor,
        target_state: torch.Tensor,
        target_action: torch.Tensor,
        action_before: torch.Tensor,
        action_before_valid: torch.Tensor,
        posterior_state: torch.Tensor,
        posterior_action: torch.Tensor,
        *,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        condition = self._condition_vector(
            condition_state, condition_action, state_visible, action_visible,
            state_valid, action_valid, target_state, target_action,
            action_before, action_before_valid,
        )
        posterior_condition = self._condition_vector(
            posterior_state, posterior_action, state_visible, action_visible,
            state_valid, action_valid, target_state, target_action,
            action_before, action_before_valid, use_visibility=False,
        )
        posterior_hidden = self.posterior_encoder(posterior_condition)
        prior_hidden = self.prior_encoder(condition)
        q_mean = self.posterior_mean(posterior_hidden)
        q_logvar = self.posterior_logvar(posterior_hidden).clamp(-8.0, 4.0)
        p_mean = self.prior_mean(prior_hidden)
        p_logvar = self.prior_logvar(prior_hidden).clamp(-8.0, 4.0)
        z_q = self.sample_gaussian(q_mean, q_logvar, sample)
        z_p = self.sample_gaussian(p_mean, p_logvar, sample)
        q_output = self.decoder(torch.cat((condition, z_q), dim=1))
        p_output = self.decoder(torch.cat((condition, z_p), dim=1))
        return {
            "q_output": q_output[..., : STATE_STEPS * STATE_DIM].reshape(-1, STATE_STEPS, STATE_DIM),
            "q_action": q_output[..., STATE_STEPS * STATE_DIM :].reshape(-1, ACTION_STEPS, ACTION_DIM),
            "p_output": p_output[..., : STATE_STEPS * STATE_DIM].reshape(-1, STATE_STEPS, STATE_DIM),
            "p_action": p_output[..., STATE_STEPS * STATE_DIM :].reshape(-1, ACTION_STEPS, ACTION_DIM),
            "q_mean": q_mean,
            "q_logvar": q_logvar,
            "p_mean": p_mean,
            "p_logvar": p_logvar,
        }


def _right_align_causal(
    state: torch.Tensor,
    action: torch.Tensor,
    before: torch.Tensor,
    before_valid: torch.Tensor,
    target_time: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a render-like target-A9 sample from a teacher-forced window."""
    batch = state.shape[0]
    device = state.device
    out_state = torch.zeros_like(state)
    out_action = torch.zeros_like(action)
    state_valid = torch.zeros(batch, STATE_STEPS, dtype=torch.bool, device=device)
    action_valid = torch.zeros(batch, ACTION_STEPS, dtype=torch.bool, device=device)
    target_state = torch.zeros(batch, STATE_STEPS, dtype=torch.bool, device=device)
    target_action = torch.zeros(batch, ACTION_STEPS, dtype=torch.bool, device=device)
    out_before = before.clone()
    out_before_valid = before_valid.clone()
    target_action[:, -1] = True
    action_valid[:, -1] = True
    start = max(0, int(target_time) - 9)
    for frame in range(start, int(target_time) + 2):
        slot = frame - int(target_time) + 9
        out_state[:, slot] = state[:, frame]
        state_valid[:, slot] = True
    for frame in range(start, int(target_time)):
        slot = frame - int(target_time) + 9
        out_action[:, slot] = action[:, frame]
        action_valid[:, slot] = True
    return out_state, out_action, state_valid, action_valid, target_state, target_action, out_before, out_before_valid


class CausalBlockMaskSampler:
    """40% causal + 20% each State-only, Action-only and Both time-block tasks."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def sample(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        state = batch["physical_state"]
        action = batch["action"]
        before = batch["action_before_window"].clone()
        before_valid = batch["action_before_valid"].clone()
        truth_state = state.clone()
        truth_action = action.clone()
        batch_size = int(action.shape[0])
        state_visible = torch.ones(batch_size, STATE_STEPS, dtype=torch.bool)
        action_visible = torch.ones(batch_size, ACTION_STEPS, dtype=torch.bool)
        state_valid = torch.ones(batch_size, STATE_STEPS, dtype=torch.bool)
        action_valid = torch.ones(batch_size, ACTION_STEPS, dtype=torch.bool)
        target_state = torch.zeros(batch_size, STATE_STEPS, dtype=torch.bool)
        target_action = torch.zeros(batch_size, ACTION_STEPS, dtype=torch.bool)
        condition_state = state.clone()
        condition_action = action.clone()
        posterior_state = state.clone()
        posterior_action = action.clone()
        causal_indices = []
        families: list[str] = []
        for row in range(batch_size):
            choice = float(self.rng.random())
            if choice < 0.40:
                target_time = int(self.rng.integers(0, ACTION_STEPS))
                causal_indices.append((row, target_time))
                families.append("causal")
                continue
            if choice < 0.60:
                length = int(self.rng.integers(1, 3))
                start = int(self.rng.integers(1, STATE_STEPS - length))
                target_state[row, start : start + length] = True
                state_visible[row, start : start + length] = False
                families.append("state_only")
                continue
            if choice < 0.80:
                length = int(self.rng.integers(1, 3))
                start = int(self.rng.integers(0, ACTION_STEPS - length + 1))
                target_action[row, start : start + length] = True
                action_visible[row, start : start + length] = False
                families.append("action_only")
                continue
            length = int(self.rng.integers(1, 3))
            start = int(self.rng.integers(1, ACTION_STEPS - length))
            target_state[row, start : start + length] = True
            state_visible[row, start : start + length] = False
            target_action[row, start : start + length] = True
            action_visible[row, start : start + length] = False
            families.append("both")
        for row, target_time in causal_indices:
            row_state, row_action, row_state_valid, row_action_valid, row_target_state, row_target_action, row_before, row_before_valid = _right_align_causal(
                state[row : row + 1], action[row : row + 1], before[row : row + 1], before_valid[row : row + 1], target_time
            )
            causal_truth_action = row_action.clone()
            causal_truth_action[0, -1] = action[row, target_time]
            truth_state[row] = row_state[0]
            truth_action[row] = causal_truth_action[0]
            condition_state[row] = row_state[0]
            condition_action[row] = row_action[0]
            posterior_state[row] = row_state[0]
            posterior_action[row] = causal_truth_action[0]
            state_valid[row] = row_state_valid[0]
            action_valid[row] = row_action_valid[0]
            state_visible[row] = row_state_valid[0]
            action_visible[row] = row_action_valid[0]
            action_visible[row, -1] = False
            target_state[row] = row_target_state[0]
            target_action[row] = row_target_action[0]
            before[row] = row_before[0]
            before_valid[row] = row_before_valid[0]
            condition_action[row, -1] = 0.0
        posterior_state[target_state] = truth_state[target_state]
        posterior_action[target_action] = truth_action[target_action]
        return {
            "condition_state": condition_state,
            "condition_action": condition_action,
            "posterior_state": posterior_state,
            "posterior_action": posterior_action,
            "state_visible": state_visible,
            "action_visible": action_visible,
            "state_valid": state_valid,
            "action_valid": action_valid,
            "target_state": target_state,
            "target_action": target_action,
            "truth_state": truth_state,
            "truth_action": truth_action,
            "action_before": before,
            "action_before_valid": before_valid,
            "family": tuple(families),
        }


def _autocast(device: torch.device, precision: str):
    if str(precision).upper() == "FP32":
        return nullcontext()
    if str(precision).upper() == "BF16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise RuntimeError("BF16 MLP CVAE requires CUDA")


def _masked_mse(prediction: torch.Tensor, truth: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    selected = target.to(prediction.dtype)[..., None]
    return ((prediction - truth).float().square() * selected).sum() / selected.sum().clamp_min(1.0)


def _kl_gaussian(q_mean: torch.Tensor, q_logvar: torch.Tensor, p_mean: torch.Tensor, p_logvar: torch.Tensor) -> torch.Tensor:
    value = p_logvar - q_logvar + (
        torch.exp(q_logvar) + (q_mean - p_mean).square()
    ) / torch.exp(p_logvar).clamp_min(1.0e-8) - 1.0
    return 0.5 * value.sum(dim=-1).mean()


MODEL_INPUT_KEYS = (
    "condition_state", "condition_action", "posterior_state", "posterior_action",
    "state_visible", "action_visible", "state_valid", "action_valid",
    "target_state", "target_action", "action_before", "action_before_valid",
)


def _prepare_batch(
    batch: dict[str, Any],
    masks: dict[str, Any],
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> dict[str, torch.Tensor]:
    del batch
    return {
        key: value.to(device, non_blocking=non_blocking)
        for key, value in masks.items()
        if key != "family" and isinstance(value, torch.Tensor)
    }


def _checkpoint_payload(model: MLPConditionalVAE11, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, step: int, config: dict[str, Any], selection: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT,
        "architecture_version": ARCHITECTURE_VERSION,
        "optimizer_step": int(step),
        "model_signature": model.architecture_signature(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "selection": selection,
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _evaluate(
    model: MLPConditionalVAE11,
    loader: DataLoader,
    device: torch.device,
    sampler: CausalBlockMaskSampler,
    max_batches: int,
    precision: str,
    seed: int,
    *,
    non_blocking: bool = False,
) -> dict[str, Any]:
    model.eval()
    sampler = CausalBlockMaskSampler(seed)
    total_prior_action = 0.0
    total_prior_state = 0.0
    total_q_action = 0.0
    total_q_state = 0.0
    count_action = 0
    count_state = 0
    causal_action = 0.0
    causal_count = 0
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= int(max_batches):
                break
            raw_masks = sampler.sample(batch)
            masks = _prepare_batch(batch, raw_masks, device, non_blocking=non_blocking)
            model_kwargs = {key: masks[key] for key in MODEL_INPUT_KEYS}
            with _autocast(device, precision):
                output = model(**model_kwargs, sample=True)
            state_target = masks["target_state"].to(output["p_output"].dtype)[..., None]
            action_target = masks["target_action"].to(output["p_action"].dtype)[..., None]
            prior_state_error = (output["p_output"].float() - masks["truth_state"].float()).square() * state_target
            prior_action_error = (output["p_action"].float() - masks["truth_action"].float()).square() * action_target
            q_state_error = (output["q_output"].float() - masks["truth_state"].float()).square() * state_target
            q_action_error = (output["q_action"].float() - masks["truth_action"].float()).square() * action_target
            total_prior_state += float(prior_state_error.sum().cpu())
            total_prior_action += float(prior_action_error.sum().cpu())
            total_q_state += float(q_state_error.sum().cpu())
            total_q_action += float(q_action_error.sum().cpu())
            count_state += int(state_target.sum().cpu()) * STATE_DIM
            count_action += int(action_target.sum().cpu()) * ACTION_DIM
            causal = torch.tensor([family == "causal" for family in raw_masks["family"]], device=device)
            if bool(causal.any()) and int(action_target[causal].sum().cpu()):
                causal_action += float(prior_action_error[causal].sum().cpu())
                causal_count += int(action_target[causal].sum().cpu()) * ACTION_DIM
    return {
        "prior_state_rmse": math.sqrt(total_prior_state / max(count_state, 1)),
        "prior_action_rmse": math.sqrt(total_prior_action / max(count_action, 1)),
        "posterior_state_rmse": math.sqrt(total_q_state / max(count_state, 1)),
        "posterior_action_rmse": math.sqrt(total_q_action / max(count_action, 1)),
        "prior_causal_action_rmse": math.sqrt(causal_action / max(causal_count, 1)),
        "masked_state_count": count_state,
        "masked_action_count": count_action,
    }


def run_training(dataset_run: Path, output_run: Path, config: dict[str, Any], *, max_steps_override: int | None = None, smoke: bool = False) -> dict[str, Any]:
    output = output_run.expanduser().resolve()
    for folder in ("data", "manifests", "markers", "plots", "videos", "logs", "checkpoints"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 20261111))
    seed_everything(seed)
    data_cfg = config.get("data", {})
    train = Action11WindowDataset(
        dataset_run, motion_count=int(data_cfg.get("motion_count", 32)),
        selection_seed=int(data_cfg.get("selection_seed", seed)), training=True,
        samples_per_epoch=int(data_cfg.get("train_samples_per_epoch", 16384)),
        start_probability=float(data_cfg.get("episode_start_probability", 0.10)),
        cache_in_memory=True,
    )
    monitor = Action11WindowDataset(
        dataset_run, motion_count=int(data_cfg.get("motion_count", 32)),
        selection_seed=int(data_cfg.get("selection_seed", seed)),
        requested_motion_keys=train.selected_motion_keys, training=False,
        validation_stride=int(data_cfg.get("monitor_stride", 10)), cache_in_memory=True,
    )
    selection = train.selection_manifest()
    atomic_write_json(output / "manifests/motion_selection.json", selection)
    model = MLPConditionalVAE11(config.get("model", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training_cfg = config.get("training", {})
    precision = str(training_cfg.get("precision", "BF16")).upper()
    micro_batch = int(training_cfg.get("micro_batch", 256))
    if device.type == "cuda":
        torch.set_float32_matmul_precision(str(training_cfg.get("matmul_precision", "high")))
    max_steps = int(max_steps_override if max_steps_override is not None else training_cfg.get("max_steps", 300000))
    if smoke:
        max_steps = min(max_steps, 2)
    validation_interval = 1 if smoke else int(training_cfg.get("monitor_interval", 5000))
    checkpoint_interval = 1 if smoke else int(training_cfg.get("checkpoint_interval", 50000))
    eval_batches = int(training_cfg.get("monitor_batches", 32))
    wall_seconds = float(training_cfg.get("max_wall_hours", 10.0)) * 3600.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg.get("learning_rate", 2.0e-4)), weight_decay=float(training_cfg.get("weight_decay", 1.0e-4)))
    warmup = int(training_cfg.get("warmup_steps", 2000))
    min_ratio = float(training_cfg.get("min_lr_ratio", 0.05))
    workers = int(data_cfg.get("num_workers", 2))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    persistent_workers = bool(data_cfg.get("persistent_workers", workers > 0))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2))
    non_blocking = bool(data_cfg.get("non_blocking_transfer", pin_memory and device.type == "cuda"))
    loader_kwargs = {
        "batch_size": micro_batch,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if workers > 0 else False,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train, **loader_kwargs)
    monitor_kwargs = dict(loader_kwargs)
    monitor_kwargs["batch_size"] = int(training_cfg.get("monitor_batch", micro_batch))
    monitor_loader = DataLoader(monitor, **monitor_kwargs)
    iterator = iter(train_loader)
    sampler = CausalBlockMaskSampler(seed + 101)
    best_score = float("inf")
    best_metrics: dict[str, Any] | None = None
    last_metrics: dict[str, Any] | None = None
    started = time.monotonic()
    interrupted = False
    stop_reason = "max_steps"
    def save(name: str, metrics: dict[str, Any], step: int) -> None:
        atomic_torch_save(output / "checkpoints" / name, _checkpoint_payload(model, optimizer, scheduler, step, config, selection, metrics))
    def handle_signal(_signum: int, _frame: Any) -> None:
        nonlocal interrupted, stop_reason
        interrupted = True
        stop_reason = "signal"
    previous_handlers = {signal.SIGTERM: signal.signal(signal.SIGTERM, handle_signal), signal.SIGINT: signal.signal(signal.SIGINT, handle_signal)}
    try:
        for step in range(1, max_steps + 1):
            if interrupted or time.monotonic() - started >= wall_seconds:
                stop_reason = "wall_clock" if not interrupted else stop_reason
                break
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            masks = sampler.sample(batch)
            values = _prepare_batch(batch, masks, device, non_blocking=non_blocking)
            model_kwargs = {key: values[key] for key in MODEL_INPUT_KEYS}
            with _autocast(device, precision):
                output_values = model(**model_kwargs, sample=True)
                q_state = _masked_mse(output_values["q_output"], values["truth_state"], values["target_state"])
                q_action = _masked_mse(output_values["q_action"], values["truth_action"], values["target_action"])
                p_state = _masked_mse(output_values["p_output"], values["truth_state"], values["target_state"])
                p_action = _masked_mse(output_values["p_action"], values["truth_action"], values["target_action"])
                kl_beta = min(float(training_cfg.get("kl_beta", 0.002)), float(training_cfg.get("kl_beta", 0.002)) * step / max(int(training_cfg.get("kl_warmup_steps", 10000)), 1))
                kl = _kl_gaussian(output_values["q_mean"], output_values["q_logvar"], output_values["p_mean"], output_values["p_logvar"])
                loss = 0.25 * (q_state + q_action + p_state + p_action) + kl_beta * kl
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite MLP CVAE loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg.get("gradient_clip", 1.0)))
            optimizer.step()
            scheduler.step()
            if step == 1 or step % validation_interval == 0 or step == max_steps:
                metrics = _evaluate(
                    model,
                    monitor_loader,
                    device,
                    sampler,
                    eval_batches,
                    precision,
                    seed + step,
                    non_blocking=non_blocking,
                )
                model.train()
                metrics["kl_beta"] = kl_beta
                metrics["kl"] = float(kl.detach().cpu())
                last_metrics = metrics
                score = float(metrics["prior_causal_action_rmse"] + metrics["prior_action_rmse"] + metrics["prior_state_rmse"])
                if score < best_score:
                    best_score = score
                    best_metrics = dict(metrics)
                    save("best.pt", metrics, step)
            if step % checkpoint_interval == 0 or step == max_steps:
                save("last.pt", last_metrics or {"pending": True}, step)
            if interrupted:
                break
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        train.base.close()
        monitor.base.close()
    completed_steps = int(step if "step" in locals() else 0)
    summary = {
        "summary_format": SUMMARY_FORMAT,
        "execution_pass": True,
        "quality_pass": None,
        "smoke": bool(smoke),
        "completed_optimizer_steps": completed_steps,
        "requested_max_steps": max_steps,
        "stop_reason": stop_reason,
        "wall_clock_limit_hours": wall_seconds / 3600.0,
        "parameter_count": model.architecture_signature()["parameter_count"],
        "model_contract": model.architecture_signature(),
        "motion_selection": selection,
        "performance": {
            "micro_batch": micro_batch,
            "num_workers": workers,
            "pin_memory": pin_memory,
            "non_blocking_transfer": non_blocking,
            "prefetch_factor": prefetch_factor,
            "monitor_interval": validation_interval,
            "checkpoint_interval": checkpoint_interval,
            "monitor_batches": eval_batches,
        },
        "best_score": None if best_metrics is None else best_score,
        "best_metrics": best_metrics,
        "last_metrics": last_metrics,
        "kl_beta_final": float(training_cfg.get("kl_beta", 0.002)),
        "inference_contract": "sample z_p from prior p(z|visible masked sequence), decode state/action heads",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output / "manifests/mlp_cvae11_summary.json", summary)
    atomic_write_text(output / "markers" / ("cvae_mlp_cvae11_smoke.ok" if smoke else "cvae_mlp_cvae11_execution.ok"), "PASS\n")
    if interrupted:
        atomic_write_text(output / "markers/cvae.interrupted", "SAFE_INTERRUPTION\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    summary = run_training(args.dataset_run, args.output_run, load_config(args.config), max_steps_override=args.max_steps, smoke=args.smoke)
    print(json.dumps({"output_run": str(args.output_run), "completed_optimizer_steps": summary["completed_optimizer_steps"], "parameter_count": summary["parameter_count"], "best_metrics": summary["best_metrics"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
