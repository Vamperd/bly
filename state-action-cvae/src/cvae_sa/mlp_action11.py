"""Deterministic MLP completion of 11-State/10-Action Physics windows.

The train contract is deliberately separate from the 65-token Transformer:
11 normalized States (S0..S10) and 10 normalized Actions (A0..A9) are flattened
with explicit visibility/validity masks, encoded to one deterministic latent,
and decoded to all ten Action rows.  Loss is applied only to masked Action
features.  A10 is never represented because S11 is not part of this contract.

Rendering uses the same model in a right-aligned one-action-at-a-time query:
the target Action is always slot 9, the current real State is slot 9, and the
expected next State is slot 10.  Missing prefix values are zero with valid=0.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import signal
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .dataset import StateActionWindowDataset
from .util import atomic_torch_save, atomic_write_json, atomic_write_text, load_config, seed_everything

STATE_STEPS = 11
ACTION_STEPS = 10
STATE_DIM = 70
ACTION_DIM = 29
ARCHITECTURE_VERSION = "11-frame-mlp-action-completion-v1"
CHECKPOINT_FORMAT = "sonic_11_frame_mlp_action_completion_checkpoint_v1"
SUMMARY_FORMAT = "sonic_11_frame_mlp_action_completion_summary_v1"


@dataclass(frozen=True)
class MaskBatch:
    state_visible: torch.Tensor
    action_visible: torch.Tensor
    state_valid: torch.Tensor
    action_valid: torch.Tensor
    target_action: torch.Tensor
    family: tuple[str, ...]


def _stable_key_order(keys: Iterable[str], seed: int) -> list[str]:
    def score(key: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"{int(seed)}:{key}".encode("utf-8")).hexdigest()
        return digest, key

    return sorted((str(key) for key in keys), key=score)


def select_motion_keys(
    episodes: list[dict[str, Any]], motion_count: int, seed: int,
    requested: list[str] | None = None,
) -> list[str]:
    available = sorted({str(row["motion_key"]) for row in episodes})
    if requested is not None:
        selected = [str(value) for value in requested]
        if len(selected) != int(motion_count) or len(set(selected)) != len(selected):
            raise ValueError("requested motion_keys must contain exactly motion_count unique keys")
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(f"requested motion keys are absent from the train split: {missing[:3]}")
        return sorted(selected)
    if len(available) < int(motion_count):
        raise ValueError(f"train split has only {len(available)} motions; need {motion_count}")
    return sorted(_stable_key_order(available, seed)[: int(motion_count)])


class Action11WindowDataset(Dataset[dict[str, Any]]):
    """Physics-v5 train-only windows with deterministic 32-motion selection."""

    def __init__(
        self,
        dataset_run: Path,
        *,
        motion_count: int = 32,
        selection_seed: int = 20261110,
        requested_motion_keys: list[str] | None = None,
        training: bool,
        samples_per_epoch: int = 8192,
        validation_stride: int = 10,
        start_probability: float = 0.10,
    ) -> None:
        self.training = bool(training)
        self.base = StateActionWindowDataset(
            dataset_run.expanduser().resolve(),
            "train",
            window_transitions=ACTION_STEPS,
            validation_stride=1,
            random_crop=False,
        )
        if not all(int(row["steps"]) >= ACTION_STEPS for row in self.base.episodes):
            raise ValueError("all selected episodes must contain at least 10 transitions")
        self.selected_motion_keys = select_motion_keys(
            self.base.episodes, motion_count, selection_seed, requested_motion_keys
        )
        selected = set(self.selected_motion_keys)
        self.episode_indices = [
            index for index, row in enumerate(self.base.episodes)
            if str(row["motion_key"]) in selected
        ]
        if not self.episode_indices:
            raise ValueError("motion selection produced no episodes")
        self.selection_seed = int(selection_seed)
        self.start_probability = float(start_probability)
        if not 0.0 <= self.start_probability <= 1.0:
            raise ValueError("start_probability must be in [0,1]")
        self.validation_stride = max(1, int(validation_stride))
        self._fixed_refs: list[tuple[int, int]] = []
        for episode_index in self.episode_indices:
            steps = int(self.base.episodes[episode_index]["steps"])
            if not self.training:
                starts = list(range(0, steps - ACTION_STEPS + 1, self.validation_stride))
                if not starts or starts[-1] != steps - ACTION_STEPS:
                    starts.append(steps - ACTION_STEPS)
                self._fixed_refs.extend((episode_index, start) for start in starts)
        self.samples_per_epoch = int(samples_per_epoch)
        if self.training and self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")

    def __len__(self) -> int:
        return self.samples_per_epoch if self.training else len(self._fixed_refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.training:
            rng = np.random.default_rng(self.selection_seed + 1000003 * int(index))
            episode_index = int(self.episode_indices[int(rng.integers(len(self.episode_indices)))])
            steps = int(self.base.episodes[episode_index]["steps"])
            max_start = steps - ACTION_STEPS
            start = 0 if float(rng.random()) < self.start_probability else int(rng.integers(max_start + 1))
        else:
            episode_index, start = self._fixed_refs[int(index)]
        item = dict(self.base.get_window(episode_index, start))
        if int(start) == 0:
            item["action_before_window"] = torch.zeros(ACTION_DIM, dtype=torch.float32)
            item["action_before_valid"] = torch.tensor(False, dtype=torch.bool)
        else:
            item["action_before_valid"] = torch.tensor(True, dtype=torch.bool)
        item["sample_start"] = int(start)
        return item

    @property
    def state_mean(self) -> np.ndarray:
        return self.base.state_mean

    @property
    def state_std(self) -> np.ndarray:
        return self.base.state_std

    @property
    def action_mean(self) -> np.ndarray:
        return self.base.action_mean

    @property
    def action_std(self) -> np.ndarray:
        return self.base.action_std

    def selection_manifest(self) -> dict[str, Any]:
        rows = [self.base.episodes[index] for index in self.episode_indices]
        variants = {key: sorted({int(row["variant_id"]) for row in rows if str(row["motion_key"]) == key})
                    for key in self.selected_motion_keys}
        return {
            "format_version": "sonic_11_frame_mlp_motion_selection_v1",
            "selection_seed": self.selection_seed,
            "motion_count": len(self.selected_motion_keys),
            "selected_motion_keys": self.selected_motion_keys,
            "episode_count": len(rows),
            "variants_by_motion": variants,
            "train_only": True,
            "window_contract": {"state_steps": STATE_STEPS, "action_steps": ACTION_STEPS},
        }


class CompletionMaskSampler:
    """Reasonable train-only completion masks.

    70% masks remove complete temporal State/Action rows. 30% remove feature
    subsets. A target Action always has both endpoint State rows visible unless
    the task is explicitly a context-only gap away from the target. Future
    Actions after an interior target are invalid rather than silently treated
    as zero-valued observations.
    """

    def __init__(self, seed: int, *, step_fraction: float = 0.70, last_action_probability: float = 0.55) -> None:
        if not 0.0 < step_fraction < 1.0:
            raise ValueError("step_fraction must be between 0 and 1")
        self.seed = int(seed)
        self.step_fraction = float(step_fraction)
        self.last_action_probability = float(last_action_probability)
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)


    def _target_time(self) -> int:
        return ACTION_STEPS - 1 if float(self.rng.random()) < self.last_action_probability else int(self.rng.integers(ACTION_STEPS))

    def sample(self, batch_size: int) -> MaskBatch:
        count = int(batch_size)
        if count <= 0:
            raise ValueError("batch_size must be positive")
        state_visible = np.ones((count, STATE_STEPS, STATE_DIM), dtype=bool)
        action_visible = np.ones((count, ACTION_STEPS, ACTION_DIM), dtype=bool)
        state_valid = np.ones((count, STATE_STEPS), dtype=bool)
        action_valid = np.ones((count, ACTION_STEPS), dtype=bool)
        target_action = np.zeros((count, ACTION_STEPS, ACTION_DIM), dtype=bool)
        families: list[str] = []

        for row in range(count):
            if float(self.rng.random()) < self.step_fraction:
                target = self._target_time()
                length = 1 if target == ACTION_STEPS - 1 else int(self.rng.integers(1, 4))
                start = max(0, min(target, target - length + 1))
                stop = min(ACTION_STEPS - 1, start + length - 1)
                target_action[row, start:stop + 1, :] = True
                action_visible[row, start:stop + 1, :] = False
                if stop < ACTION_STEPS - 1:
                    action_visible[row, stop + 1:, :] = False
                    action_valid[row, stop + 1:] = False
                # Only add a complete State gap if it does not remove either
                # endpoint needed by a masked transition.
                if float(self.rng.random()) < 0.30:
                    candidates = [value for value in range(STATE_STEPS - 1)
                                  if value not in set(range(start, stop + 2))]
                    if candidates:
                        gap_start = int(candidates[int(self.rng.integers(len(candidates)))])
                        gap_stop = min(STATE_STEPS - 1, gap_start + int(self.rng.integers(1, 3)))
                        if not any(gap_start <= endpoint <= gap_stop for endpoint in (start, stop + 1)):
                            state_visible[row, gap_start:gap_stop + 1, :] = False
                            state_valid[row, gap_start:gap_stop + 1] = False
                            families.append("step_state_action")
                        else:
                            families.append("step_action")
                    else:
                        families.append("step_action")
                else:
                    families.append("step_action")
            else:
                target = self._target_time()
                feature_count = max(1, int(self.rng.integers(4, 16)))
                features = self.rng.choice(ACTION_DIM, size=feature_count, replace=False)
                target_action[row, target, features] = True
                action_visible[row, target, features] = False
                if target < ACTION_STEPS - 1:
                    action_visible[row, target + 1:, :] = False
                    action_valid[row, target + 1:] = False
                # Hide a feature subset in one or two non-endpoint State rows.
                protected = {target, target + 1}
                eligible = [value for value in range(STATE_STEPS) if value not in protected]
                if eligible:
                    state_time = int(eligible[int(self.rng.integers(len(eligible)))])
                    state_feature_count = max(1, int(self.rng.integers(4, 20)))
                    state_features = self.rng.choice(STATE_DIM, size=state_feature_count, replace=False)
                    state_visible[row, state_time, state_features] = False
                # Also hide a small feature subset from one visible Action row,
                # without making it a loss target.
                prior = list(range(target))
                if prior:
                    action_time = int(prior[int(self.rng.integers(len(prior)))])
                    context_count = max(1, int(self.rng.integers(3, 12)))
                    context_features = self.rng.choice(ACTION_DIM, size=context_count, replace=False)
                    action_visible[row, action_time, context_features] = False
                families.append("feature_state_action")

        if not all(bool(value) for value in target_action.any(axis=(1, 2))):
            raise RuntimeError("mask sampler produced an empty Action target")
        return MaskBatch(
            state_visible=torch.from_numpy(state_visible),
            action_visible=torch.from_numpy(action_visible),
            state_valid=torch.from_numpy(state_valid),
            action_valid=torch.from_numpy(action_valid),
            target_action=torch.from_numpy(target_action),
            family=tuple(families),
        )

    @staticmethod
    def terminal(batch_size: int) -> MaskBatch:
        count = int(batch_size)
        state_visible = torch.ones(count, STATE_STEPS, STATE_DIM, dtype=torch.bool)
        action_visible = torch.ones(count, ACTION_STEPS, ACTION_DIM, dtype=torch.bool)
        state_valid = torch.ones(count, STATE_STEPS, dtype=torch.bool)
        action_valid = torch.ones(count, ACTION_STEPS, dtype=torch.bool)
        target = torch.zeros(count, ACTION_STEPS, ACTION_DIM, dtype=torch.bool)
        target[:, -1, :] = True
        action_visible[:, -1, :] = False
        return MaskBatch(state_visible, action_visible, state_valid, action_valid, target, ("terminal_a9",) * count)


def _mlp(in_dim: int, widths: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = int(in_dim)
    for width in widths:
        layers.extend((nn.Linear(current, int(width)), nn.LayerNorm(int(width)), nn.GELU()))
        current = int(width)
    layers.append(nn.Linear(current, int(out_dim)))
    return nn.Sequential(*layers)


class MLPActionCompletion11(nn.Module):
    """Flattened deterministic encoder/decoder for the 11/10 contract."""

    ARCHITECTURE_VERSION = ARCHITECTURE_VERSION

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        state_dim = int(config.get("state_dim", STATE_DIM))
        action_dim = int(config.get("action_dim", ACTION_DIM))
        state_steps = int(config.get("state_steps", STATE_STEPS))
        action_steps = int(config.get("action_steps", ACTION_STEPS))
        if (state_dim, action_dim, state_steps, action_steps) != (STATE_DIM, ACTION_DIM, STATE_STEPS, ACTION_STEPS):
            raise ValueError("MLP Action11 requires state_steps=11, action_steps=10, state_dim=70, action_dim=29")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_steps = state_steps
        self.action_steps = action_steps
        self.latent_dim = int(config.get("latent_dim", 256))
        self.encoder_widths = [int(value) for value in config.get("encoder_widths", [2048, 1024, 512])]
        self.decoder_widths = [int(value) for value in config.get("decoder_widths", [1024, 2048, 512])]
        self.encoder_input_dim = (
            state_steps * state_dim * 2
            + action_steps * action_dim * 2
            + state_steps + action_steps
            + action_steps * action_dim
            + action_dim + 1
        )
        self.decoder_query_dim = (
            action_steps * action_dim * 2
            + state_steps * state_dim
            + state_steps + action_steps
        )
        self.encoder = _mlp(self.encoder_input_dim, self.encoder_widths[:-1], self.encoder_widths[-1])
        self.to_latent = nn.Linear(self.encoder_widths[-1], self.latent_dim)
        self.decoder = _mlp(self.latent_dim + self.decoder_query_dim, self.decoder_widths, action_steps * action_dim)

    def architecture_signature(self) -> dict[str, Any]:
        return {
            "architecture_version": self.ARCHITECTURE_VERSION,
            "state_shape": [self.state_steps, self.state_dim],
            "action_shape": [self.action_steps, self.action_dim],
            "latent_dim": self.latent_dim,
            "encoder_widths": self.encoder_widths,
            "decoder_widths": self.decoder_widths,
            "encoder_input_dim": self.encoder_input_dim,
            "decoder_query_dim": self.decoder_query_dim,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "target_contract": "masked_action_features_only; A10_absent",
        }

    def forward(
        self,
        physical_state: torch.Tensor,
        action: torch.Tensor,
        state_visible: torch.Tensor,
        action_visible: torch.Tensor,
        state_valid: torch.Tensor,
        action_valid: torch.Tensor,
        target_action: torch.Tensor,
        action_before_window: torch.Tensor,
        action_before_valid: torch.Tensor,
    ) -> torch.Tensor:
        expected = {
            "physical_state": (self.state_steps, self.state_dim),
            "action": (self.action_steps, self.action_dim),
            "state_visible": (self.state_steps, self.state_dim),
            "action_visible": (self.action_steps, self.action_dim),
            "state_valid": (self.state_steps,),
            "action_valid": (self.action_steps,),
            "target_action": (self.action_steps, self.action_dim),
            "action_before_window": (self.action_dim,),
        }
        values = {
            "physical_state": physical_state,
            "action": action,
            "state_visible": state_visible,
            "action_visible": action_visible,
            "state_valid": state_valid,
            "action_valid": action_valid,
            "target_action": target_action,
            "action_before_window": action_before_window,
        }
        for name, shape in expected.items():
            if tuple(values[name].shape[1:]) != shape:
                raise ValueError(f"{name} must have shape [B,{','.join(map(str, shape))}]")
        if tuple(action_before_valid.shape[1:]) != ():
            raise ValueError("action_before_valid must have shape [B]")
        state_mask = (state_visible & state_valid[..., None]).to(physical_state.dtype)
        action_mask = (action_visible & action_valid[..., None]).to(action.dtype)
        state_value = physical_state * state_mask
        action_value = action * action_mask
        before_valid = action_before_valid.to(action_before_window.dtype).unsqueeze(-1)
        before_value = action_before_window * before_valid
        encoder_input = torch.cat(
            (
                state_value.flatten(1), state_mask.flatten(1),
                action_value.flatten(1), action_mask.flatten(1),
                state_valid.to(physical_state.dtype), action_valid.to(action.dtype),
                target_action.to(action.dtype).flatten(1),
                before_value, before_valid,
            ),
            dim=1,
        )
        latent = self.to_latent(self.encoder(encoder_input))
        query = torch.cat(
            (
                target_action.to(action.dtype).flatten(1),
                action_visible.to(action.dtype).flatten(1),
                state_visible.to(physical_state.dtype).flatten(1),
                state_valid.to(physical_state.dtype), action_valid.to(action.dtype),
            ),
            dim=1,
        )
        output = self.decoder(torch.cat((latent, query), dim=1))
        return output.reshape(-1, self.action_steps, self.action_dim)
def build_recursive_query(
    *,
    expected_states: np.ndarray,
    actual_states: list[np.ndarray],
    executed_actions: list[np.ndarray],
    target_step: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Build the render-only right-aligned query for one Action.

    The target is always the final Action slot (relative A9).  For target
    ``t`` the state slots represent S[t-9]..S[t+1], clipped at episode start;
    the last two valid slots are the current real State and expected next
    State.  Prefix slots have value zero and valid=0.  This helper is not used
    by the training sampler.
    """
    expected = np.asarray(expected_states, dtype=np.float32)
    if expected.ndim != 2 or expected.shape[1] != STATE_DIM:
        raise ValueError("expected_states must have shape [N,70]")
    target_step = int(target_step)
    if target_step < 0 or target_step + 1 >= expected.shape[0]:
        raise ValueError("target_step must have an expected next State")
    state_value = np.zeros((STATE_STEPS, STATE_DIM), dtype=np.float32)
    state_visible = np.zeros_like(state_value, dtype=bool)
    state_valid = np.zeros(STATE_STEPS, dtype=bool)
    state_start = max(0, target_step - 9)
    actual = [np.asarray(value, dtype=np.float32) for value in actual_states]
    for frame in range(state_start, target_step + 1):
        slot = frame - state_start
        value = actual[frame] if frame < len(actual) else expected[frame]
        if value.shape != (STATE_DIM,) or not np.isfinite(value).all():
            raise ValueError("actual/expected State contains invalid values")
        state_value[slot] = (value - state_mean) / np.maximum(state_std, 1.0e-6)
        state_visible[slot] = True
        state_valid[slot] = True
    next_slot = STATE_STEPS - 1
    state_value[next_slot] = (expected[target_step + 1] - state_mean) / np.maximum(state_std, 1.0e-6)
    state_visible[next_slot] = True
    state_valid[next_slot] = True

    action_value = np.zeros((ACTION_STEPS, ACTION_DIM), dtype=np.float32)
    action_visible = np.zeros_like(action_value, dtype=bool)
    action_valid = np.zeros(ACTION_STEPS, dtype=bool)
    action_start = max(0, target_step - 9)
    generated = [np.asarray(value, dtype=np.float32) for value in executed_actions]
    for action_index in range(action_start, target_step):
        if action_index >= len(generated):
            continue
        slot = action_index - action_start
        value = generated[action_index]
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ValueError("executed Action contains invalid values")
        action_value[slot] = (value - action_mean) / np.maximum(action_std, 1.0e-6)
        action_visible[slot] = True
        action_valid[slot] = True
    target_slot = ACTION_STEPS - 1
    action_valid[target_slot] = True
    target_action = np.zeros((ACTION_STEPS, ACTION_DIM), dtype=bool)
    target_action[target_slot] = True
    before_valid = target_step >= ACTION_STEPS
    before = np.zeros(ACTION_DIM, dtype=np.float32)
    if before_valid:
        before = (generated[target_step - ACTION_STEPS] - action_mean) / np.maximum(action_std, 1.0e-6)
    return {
        "state": torch.from_numpy(state_value).unsqueeze(0),
        "action": torch.from_numpy(action_value).unsqueeze(0),
        "state_visible": torch.from_numpy(state_visible).unsqueeze(0),
        "action_visible": torch.from_numpy(action_visible).unsqueeze(0),
        "state_valid": torch.from_numpy(state_valid).unsqueeze(0),
        "action_valid": torch.from_numpy(action_valid).unsqueeze(0),
        "target_action": torch.from_numpy(target_action).unsqueeze(0),
        "action_before": torch.from_numpy(before).unsqueeze(0),
        "action_before_valid": torch.tensor([before_valid], dtype=torch.bool),
    }


def predict_recursive_action(
    model: MLPActionCompletion11,
    query: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Return the normalized Action at the fixed render target slot A9."""
    model_inputs = {key: value.to(device) for key, value in query.items()}
    with torch.inference_mode():
        output = model(
            model_inputs["state"], model_inputs["action"],
            model_inputs["state_visible"], model_inputs["action_visible"],
            model_inputs["state_valid"], model_inputs["action_valid"],
            model_inputs["target_action"], model_inputs["action_before"],
            model_inputs["action_before_valid"],
        )
    return output[:, -1]






def _apply_mask(batch: dict[str, Any], masks: MaskBatch, device: torch.device) -> dict[str, torch.Tensor]:
    state = batch["physical_state"].to(device)
    action = batch["action"].to(device)
    state_visible = masks.state_visible.to(device)
    action_visible = masks.action_visible.to(device)
    state_valid = masks.state_valid.to(device)
    action_valid = masks.action_valid.to(device)
    target = masks.target_action.to(device)
    return {
        "state": state * (state_visible & state_valid[..., None]).to(state.dtype),
        "action": action * (action_visible & action_valid[..., None]).to(action.dtype),
        "state_visible": state_visible,
        "action_visible": action_visible,
        "state_valid": state_valid,
        "action_valid": action_valid,
        "target": target,
        "action_before": batch["action_before_window"].to(device),
        "action_before_valid": batch["action_before_valid"].to(device),
        "truth_action": action,
    }


def _forward(model: MLPActionCompletion11, batch: dict[str, Any], masks: MaskBatch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    value = _apply_mask(batch, masks, device)
    prediction = model(
        value["state"], value["action"], value["state_visible"], value["action_visible"],
        value["state_valid"], value["action_valid"], value["target"],
        value["action_before"], value["action_before_valid"],
    )
    return prediction, value["truth_action"]


def masked_mse(prediction: torch.Tensor, truth: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    selected = target.to(prediction.dtype)
    count = selected.sum().clamp_min(1.0)
    return ((prediction - truth).square() * selected).sum() / count


def _autocast(device: torch.device, precision: str):
    if str(precision).upper() == "FP32":
        return nullcontext()
    if str(precision).upper() == "BF16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise RuntimeError("BF16 MLP training requires CUDA")


def _evaluate(
    model: MLPActionCompletion11,
    loader: DataLoader,
    device: torch.device,
    sampler: CompletionMaskSampler,
    max_batches: int,
    precision: str,
) -> dict[str, Any]:
    model.eval()
    sampler.reset()
    total_sse = 0.0
    total_count = 0
    terminal_sse = 0.0
    terminal_count = 0
    family_values: dict[str, list[float]] = {}
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= int(max_batches):
                break
            masks = sampler.sample(len(batch["action"]))
            with _autocast(device, precision):
                prediction, truth = _forward(model, batch, masks, device)
            target = masks.target_action.to(device)
            error = prediction.float() - truth.float().to(device)
            selected = target.to(error.dtype)
            total_sse += float((error.square() * selected).sum().cpu())
            total_count += int(selected.sum().cpu())
            terminal = CompletionMaskSampler.terminal(len(batch["action"]))
            with _autocast(device, precision):
                terminal_prediction, terminal_truth = _forward(model, batch, terminal, device)
            terminal_error = terminal_prediction[:, -1] - terminal_truth[:, -1]
            terminal_sse += float(terminal_error.float().square().sum().cpu())
            terminal_count += int(terminal_error.numel())
            for family in set(masks.family):
                family_mask = torch.tensor([item == family for item in masks.family], device=device)
                family_selected = selected * family_mask[:, None, None].to(selected.dtype)
                count = int(family_selected.sum().cpu())
                if count:
                    family_values.setdefault(family, []).append(float((error.square() * family_selected).sum().cpu()) / count)
    return {
        "mixed_masked_action_rmse": math.sqrt(total_sse / max(total_count, 1)),
        "terminal_a9_action_rmse": math.sqrt(terminal_sse / max(terminal_count, 1)),
        "masked_action_count": total_count,
        "terminal_action_count": terminal_count,
        "family_rmse": {key: math.sqrt(sum(values) / len(values)) for key, values in family_values.items()},
    }


def _checkpoint_payload(
    model: MLPActionCompletion11,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config: dict[str, Any],
    selection: dict[str, Any],
    normalization: dict[str, np.ndarray],
    metrics: dict[str, Any],
) -> dict[str, Any]:
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
        "normalization": {key: value.astype(np.float32) for key, value in normalization.items()},
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_metrics(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def run_training(
    dataset_run: Path,
    output_run: Path,
    config: dict[str, Any],
    *,
    max_steps_override: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    output = output_run.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output run must be new and empty: {output}")
    for folder in ("data", "manifests", "markers", "plots", "videos", "logs", "checkpoints"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 20261110))
    seed_everything(seed)
    data_cfg = config.get("data", {})
    train = Action11WindowDataset(
        dataset_run,
        motion_count=int(data_cfg.get("motion_count", 32)),
        selection_seed=int(data_cfg.get("selection_seed", seed)),
        requested_motion_keys=data_cfg.get("motion_keys"),
        training=True,
        samples_per_epoch=int(data_cfg.get("train_samples_per_epoch", 8192)),
        start_probability=float(data_cfg.get("episode_start_probability", 0.10)),
    )
    monitor = Action11WindowDataset(
        dataset_run,
        motion_count=int(data_cfg.get("motion_count", 32)),
        selection_seed=int(data_cfg.get("selection_seed", seed)),
        requested_motion_keys=train.selected_motion_keys,
        training=False,
        validation_stride=int(data_cfg.get("monitor_stride", 10)),
    )
    selection = train.selection_manifest()
    atomic_write_json(output / "manifests/motion_selection.json", selection)
    atomic_write_json(output / "manifests/data_contract.json", {
        "format_version": "sonic_11_frame_mlp_data_contract_v1",
        "dataset_run": str(dataset_run.expanduser().resolve()),
        "train_only": True,
        "state_shape": [STATE_STEPS, STATE_DIM],
        "action_shape": [ACTION_STEPS, ACTION_DIM],
        "unrepresented_action": "A10 requires S11 and is intentionally absent",
        "primary_task": "known S0..S10 and A0..A8 -> masked A9",
        "recursive_render_input": "right_aligned current real State + expected next State; unknown prefix valid=0",
    })
    model = MLPActionCompletion11(config["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training_cfg = config.get("training", {})
    precision = str(training_cfg.get("precision", "BF16")).upper()
    micro_batch = int(training_cfg.get("micro_batch", 256))
    max_steps = int(max_steps_override if max_steps_override is not None else training_cfg.get("max_steps", 250000))
    if smoke:
        max_steps = min(max_steps, 2)
    validation_interval = 1 if smoke else int(training_cfg.get("monitor_interval", 1000))
    checkpoint_interval = 1 if smoke else int(training_cfg.get("checkpoint_interval", 5000))
    eval_batches = int(training_cfg.get("monitor_batches", 64))
    wall_seconds = float(training_cfg.get("max_wall_hours", 12.0)) * 3600.0
    learning_rate = float(training_cfg.get("learning_rate", 2.0e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1.0e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    warmup = int(training_cfg.get("warmup_steps", 1000))
    min_ratio = float(training_cfg.get("min_lr_ratio", 0.05))

    def schedule(step: int) -> float:
        if step < warmup:
            return max(1.0e-8, (step + 1) / max(warmup, 1))
        progress = min(1.0, (step - warmup) / max(max_steps - warmup, 1))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    train_loader = DataLoader(
        train, batch_size=micro_batch, shuffle=False, num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
    )
    monitor_loader = DataLoader(
        monitor, batch_size=int(training_cfg.get("monitor_batch", micro_batch)), shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)), pin_memory=bool(data_cfg.get("pin_memory", False)),
    )
    train_sampler = CompletionMaskSampler(seed + 101)
    monitor_sampler = CompletionMaskSampler(seed + 202)
    iterator = iter(train_loader)
    best_score = float("inf")
    best_metrics: dict[str, Any] | None = None
    last_metrics: dict[str, Any] | None = None
    started = time.monotonic()
    stop_reason = "max_steps"
    interrupted = False

    def save(name: str, metrics: dict[str, Any], step: int) -> None:
        payload = _checkpoint_payload(
            model, optimizer, scheduler, step, config, selection,
            {"state_mean": train.state_mean, "state_std": train.state_std,
             "action_mean": train.action_mean, "action_std": train.action_std}, metrics,
        )
        atomic_torch_save(output / "checkpoints" / name, payload)

    def handle_signal(_signum: int, _frame: Any) -> None:
        nonlocal interrupted, stop_reason
        interrupted = True
        stop_reason = "signal"

    previous_handlers = {signal.SIGTERM: signal.signal(signal.SIGTERM, handle_signal),
                         signal.SIGINT: signal.signal(signal.SIGINT, handle_signal)}
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
            masks = train_sampler.sample(len(batch["action"]))
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, precision):
                prediction, truth = _forward(model, batch, masks, device)
                loss = masked_mse(prediction, truth, masks.target_action.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite MLP Action11 loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg.get("gradient_clip", 1.0)))
            optimizer.step()
            scheduler.step()
            train_row = {
                "phase": "train", "optimizer_step": step, "loss": float(loss.detach().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.monotonic() - started,
            }
            _write_metrics(output / "logs/metrics.jsonl", train_row)
            if step == 1 or step % validation_interval == 0 or step == max_steps:
                eval_metrics = _evaluate(model, monitor_loader, device, monitor_sampler, eval_batches, precision)
                eval_row = {"phase": "monitor", "optimizer_step": step, **eval_metrics,
                            "elapsed_seconds": time.monotonic() - started}
                _write_metrics(output / "logs/metrics.jsonl", eval_row)
                last_metrics = eval_metrics
                score = float(eval_metrics["terminal_a9_action_rmse"] + eval_metrics["mixed_masked_action_rmse"])
                if score < best_score:
                    best_score = score
                    best_metrics = dict(eval_metrics)
                    save("best.pt", eval_metrics, step)
            if step % checkpoint_interval == 0 or step == max_steps:
                save("last.pt", last_metrics or {"pending": True}, step)
            if interrupted:
                stop_reason = "signal"
                break
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        train.base.close()
        monitor.base.close()

    completed_steps = int(step if "step" in locals() else 0)
    if completed_steps >= max_steps and stop_reason == "max_steps":
        stop_reason = "max_steps"
    summary = {
        "summary_format": SUMMARY_FORMAT,
        "execution_pass": True,
        "quality_pass": None,
        "smoke": bool(smoke),
        "completed_optimizer_steps": completed_steps,
        "requested_max_steps": max_steps,
        "stop_reason": stop_reason,
        "wall_clock_limit_hours": float(wall_seconds / 3600.0),
        "parameter_count": model.architecture_signature()["parameter_count"],
        "model_contract": model.architecture_signature(),
        "train_only": True,
        "motion_selection": selection,
        "best_score": None if best_metrics is None else best_score,
        "best_metrics": best_metrics,
        "last_metrics": last_metrics,
        "checkpoint_policy": "best.pt and last.pt only",
        "a10_policy": "not represented; S11 is unavailable",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output / "manifests/mlp_action11_summary.json", summary)
    atomic_write_text(output / "markers" / ("cvae_mlp_action11_smoke.ok" if smoke else "cvae_mlp_action11_execution.ok"), "PASS\n")
    if interrupted:
        atomic_write_text(output / "markers/cvae.interrupted", "SAFE_INTERRUPTION\n")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs/mlp_action11.json")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    summary = run_training(
        args.dataset_run, args.output_run, config,
        max_steps_override=args.max_steps, smoke=args.smoke,
    )
    print(json.dumps({
        "output_run": str(args.output_run.expanduser().resolve()),
        "completed_optimizer_steps": summary["completed_optimizer_steps"],
        "parameter_count": summary["parameter_count"],
        "best_metrics": summary["best_metrics"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
