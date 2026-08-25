from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import StateActionWindowDataset, worker_seed
from .masking import MaskGenerator
from .models import build_model
from .trainer import _device_batch
from .util import atomic_write_json, atomic_write_text, file_sha256, seed_everything


class Metric:
    def __init__(self) -> None:
        self.absolute_sum = 0.0
        self.square_sum = 0.0
        self.count = 0

    def add(self, error: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        values = error if mask is None else error.masked_select(mask)
        if values.numel() == 0:
            return
        values = values.detach().float()
        self.absolute_sum += float(values.abs().sum().cpu())
        self.square_sum += float(torch.square(values).sum().cpu())
        self.count += values.numel()

    def result(self) -> dict[str, float | int]:
        if not self.count:
            return {"mae": math.nan, "rmse": math.nan, "count": 0}
        return {
            "mae": self.absolute_sum / self.count,
            "rmse": math.sqrt(self.square_sum / self.count),
            "count": self.count,
        }


def _autocast(device: torch.device, amp: str):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(
        device_type="cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16
    )


def _normalization_tensors(
    dataset: StateActionWindowDataset, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "state_mean": torch.as_tensor(dataset.state_mean, device=device),
        "state_std": torch.as_tensor(dataset.state_std, device=device),
        "previous_mean": torch.as_tensor(dataset.previous_mean, device=device),
        "previous_std": torch.as_tensor(dataset.previous_std, device=device),
        "action_mean": torch.as_tensor(dataset.action_mean, device=device),
        "action_std": torch.as_tensor(dataset.action_std, device=device),
        "auxiliary_mean": torch.as_tensor(dataset.auxiliary_mean, device=device),
        "auxiliary_std": torch.as_tensor(dataset.auxiliary_std, device=device),
    }


def _denormalize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return value.float() * std + mean


def _gravity_angle(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = torch.nn.functional.normalize(prediction.float(), dim=-1)
    target = torch.nn.functional.normalize(target.float(), dim=-1)
    cosine = (prediction * target).sum(-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def _shuffled_action_batch(
    batch: dict[str, Any], normalization: dict[str, torch.Tensor]
) -> dict[str, Any]:
    shuffled = dict(batch)
    batch_size = batch["action"].shape[0]
    order = torch.flip(torch.arange(batch_size, device=batch["action"].device), dims=(0,))
    time_shift = max(1, batch["action"].shape[1] // 2)
    shuffled_action = torch.roll(batch["action"][order], shifts=time_shift, dims=1)
    action_relative = (
        shuffled_action * normalization["action_std"] + normalization["action_mean"]
    )
    shuffled_previous = batch["previous_action"].clone()
    if shuffled_previous.shape[-1]:
        shuffled_previous[:, 1:] = (
            action_relative - normalization["previous_mean"]
        ) / normalization["previous_std"]
    shuffled["action"] = shuffled_action
    shuffled["previous_action"] = shuffled_previous
    return shuffled


@torch.no_grad()
def _evaluate_physics_bidirectional(
    model: torch.nn.Module,
    loader: DataLoader,
    masker: MaskGenerator,
    dataset: StateActionWindowDataset,
    device: torch.device,
    amp: str,
    output_run: Path,
    checkpoint_path: Path,
    dataset_run: Path,
    max_batches: int | None,
    threshold: float,
) -> dict[str, Any]:
    metrics = {
        name: Metric() for name in (
            "forward_one_normalized", "rollout_normalized", "inverse_action_normalized",
            "history_action_normalized", "arbitrary_state_normalized",
            "arbitrary_action_normalized", "forward_joint_position_rad",
            "forward_joint_velocity_rad_s", "inverse_action_rad", "history_action_rad",
        )
    }
    forward_baseline = Metric()
    forward_shuffled = Metric()
    inverse_baseline = Metric()
    inverse_shuffled = Metric()
    history_future_max_abs = 0.0
    action_mean = torch.as_tensor(dataset.action_mean, device=device)
    action_std = torch.as_tensor(dataset.action_std, device=device)
    state_mean = torch.as_tensor(dataset.state_mean, device=device)
    state_std = torch.as_tensor(dataset.state_std, device=device)
    masker.set_step(max(masker.rollout_start_step, 20_000))
    batches = 0
    for batch_index, cpu_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _device_batch(cpu_batch, device)
        batches += 1

        forward_masks = masker.generate(batch, force_task="forward_one")
        with _autocast(device, amp):
            forward = model(batch, forward_masks, sample_from_prior=True, deterministic=True)
        target_delta = batch["physical_state"][:, 1:] - batch["physical_state"][:, :-1]
        forward_mask = forward_masks.forward_transition[:, :, None].expand_as(target_delta)
        metrics["forward_one_normalized"].add(forward.forward_delta - target_delta, forward_mask)
        forward_baseline.add(forward.forward_delta - target_delta, forward_mask)
        predicted_next = (batch["physical_state"][:, :-1] + forward.forward_delta) * state_std + state_mean
        target_next = batch["physical_state"][:, 1:] * state_std + state_mean
        metrics["forward_joint_position_rad"].add(
            predicted_next[..., :29] - target_next[..., :29],
            forward_masks.forward_transition[:, :, None],
        )
        metrics["forward_joint_velocity_rad_s"].add(
            predicted_next[..., 29:58] - target_next[..., 29:58],
            forward_masks.forward_transition[:, :, None],
        )
        shuffled = _shuffled_action_batch(batch, {
            "action_std": action_std, "action_mean": action_mean,
            "previous_mean": torch.empty(0, device=device),
            "previous_std": torch.empty(0, device=device),
        })
        with _autocast(device, amp):
            shuffled_output = model(
                shuffled, forward_masks, sample_from_prior=True, deterministic=True
            )
        forward_shuffled.add(shuffled_output.forward_delta - target_delta, forward_mask)

        rollout_masks = masker.generate(batch, force_task="forward_rollout")
        with _autocast(device, amp):
            rollout = model(batch, rollout_masks, sample_from_prior=True, deterministic=True)
        for index in range(batch["physical_state"].shape[0]):
            horizon = int(rollout_masks.rollout_horizon[index].item())
            start = int(rollout_masks.rollout_start[index].item())
            available = min(horizon, rollout.rollout_state.shape[1], batch["physical_state"].shape[1] - start - 1)
            if available > 0:
                metrics["rollout_normalized"].add(
                    rollout.rollout_state[index, :available] -
                    batch["physical_state"][index, start + 1 : start + available + 1]
                )

        inverse_masks = masker.generate(batch, force_task="inverse")
        with _autocast(device, amp):
            inverse = model(batch, inverse_masks, sample_from_prior=True, deterministic=True)
        inverse_mask = inverse_masks.inverse_transition[:, :, None].expand_as(inverse.inverse_action)
        metrics["inverse_action_normalized"].add(
            inverse.inverse_action - batch["action"], inverse_mask
        )
        inverse_baseline.add(inverse.inverse_action - batch["action"], inverse_mask)
        metrics["inverse_action_rad"].add(
            (inverse.inverse_action - batch["action"]) * action_std, inverse_mask
        )
        negative_state = batch["physical_state"].clone()
        negative_state[:, 1:] = torch.roll(negative_state[:, 1:], shifts=1, dims=0)
        inverse_negative_batch = dict(batch)
        inverse_negative_batch["physical_state"] = negative_state
        with _autocast(device, amp):
            inverse_negative = model(
                inverse_negative_batch, inverse_masks,
                sample_from_prior=True, deterministic=True,
            )
        inverse_shuffled.add(
            inverse_negative.inverse_action - batch["action"], inverse_mask
        )

        history_masks = masker.generate(batch, force_task="history_action")
        with _autocast(device, amp):
            history = model(batch, history_masks, sample_from_prior=True, deterministic=True)
        history_mask = history_masks.history_action_transition[:, :, None].expand_as(
            history.history_action
        )
        metrics["history_action_normalized"].add(
            history.history_action - batch["action"], history_mask
        )
        metrics["history_action_rad"].add(
            (history.history_action - batch["action"]) * action_std, history_mask
        )
        changed_batch = dict(batch)
        changed_state = batch["physical_state"].clone()
        changed_state = torch.where(
            history_masks.state_input,
            changed_state + torch.randn_like(changed_state) * 10.0,
            changed_state,
        )
        changed_batch["physical_state"] = changed_state
        with _autocast(device, amp):
            changed_history = model(
                changed_batch, history_masks, sample_from_prior=True, deterministic=True
            )
        if bool(history_mask.any()):
            history_future_max_abs = max(
                history_future_max_abs,
                float((changed_history.history_action - history.history_action).abs().masked_select(history_mask).max().cpu()),
            )

        arbitrary_masks = masker.generate(batch, force_task="arbitrary")
        with _autocast(device, amp):
            arbitrary = model(batch, arbitrary_masks, sample_from_prior=True, deterministic=True)
        metrics["arbitrary_state_normalized"].add(
            arbitrary.physical_state - batch["physical_state"], arbitrary_masks.state_loss
        )
        metrics["arbitrary_action_normalized"].add(
            arbitrary.action - batch["action"], arbitrary_masks.action_loss
        )

    forward_base = float(forward_baseline.result()["rmse"])
    forward_bad = float(forward_shuffled.result()["rmse"])
    inverse_base = float(inverse_baseline.result()["rmse"])
    inverse_bad = float(inverse_shuffled.result()["rmse"])
    forward_degradation = forward_bad / forward_base - 1.0 if forward_base > 0 else math.inf
    inverse_degradation = inverse_bad / inverse_base - 1.0 if inverse_base > 0 else math.inf
    history_invariant = history_future_max_abs <= 1.0e-6
    summary = {
        "passed": bool(
            forward_degradation >= threshold
            and inverse_degradation >= threshold
            and history_invariant
        ),
        "checkpoint": str(checkpoint_path.expanduser().resolve()),
        "dataset_run": str(dataset_run),
        "split": "test",
        "batches": batches,
        "prior_mode": "prior_mean",
        "metrics": {name: metric.result() for name, metric in metrics.items()},
        "negative_controls": {
            "shuffled_action_forward": {
                "baseline_rmse": forward_base,
                "shuffled_rmse": forward_bad,
                "relative_degradation": forward_degradation,
                "required": threshold,
                "passed": bool(forward_degradation >= threshold),
            },
            "shuffled_next_state_inverse": {
                "baseline_rmse": inverse_base,
                "shuffled_rmse": inverse_bad,
                "relative_degradation": inverse_degradation,
                "required": threshold,
                "passed": bool(inverse_degradation >= threshold),
            },
            "history_future_invariance": {
                "max_abs": history_future_max_abs,
                "tolerance": 1.0e-6,
                "passed": history_invariant,
            },
        },
    }
    atomic_write_json(output_run / "manifests/evaluation.json", summary)
    if summary["passed"]:
        atomic_write_text(output_run / "markers/cvae_eval.ok", "PASS\n")
    return summary


@torch.no_grad()
def evaluate(
    dataset_run: Path,
    checkpoint_path: Path,
    output_run: Path,
    max_batches: int | None = None,
) -> dict[str, Any]:
    output_run = output_run.expanduser().resolve()
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "videos"):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") not in {
        "sonic_state_action_cvae_checkpoint_v1",
        "sonic_state_action_cvae_checkpoint_v2",
    }:
        raise ValueError("unsupported checkpoint format")
    config = checkpoint["config"]
    seed_everything(int(config["seed"]))
    dataset_run = dataset_run.expanduser().resolve()
    dataset_hash = file_sha256(dataset_run / "manifests" / "dataset_manifest.json")
    if dataset_hash != checkpoint["dataset_manifest_sha256"]:
        raise ValueError("checkpoint was trained against a different dataset manifest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = StateActionWindowDataset(
        dataset_run,
        "test",
        int(config["data"]["window_transitions"]),
        int(config["data"]["validation_stride"]),
        random_crop=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["micro_batch"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["data"]["num_workers"]) > 0,
        worker_init_fn=worker_seed,
    )
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    masker = MaskGenerator(config["masking"])
    if checkpoint.get("format_version") == "sonic_state_action_cvae_checkpoint_v2":
        summary = _evaluate_physics_bidirectional(
            model, loader, masker, dataset, device, str(config["training"]["amp"]),
            output_run, checkpoint_path, dataset_run, max_batches,
            float(config["evaluation"]["negative_control_min_degradation"]),
        )
        dataset.close()
        return summary
    norm = _normalization_tensors(dataset, device)
    task_specs = (
        ("forward", None),
        ("inverse", None),
        ("completion", "element"),
        ("completion", "step"),
        ("completion", "feature"),
    )
    masked_metrics = {
        f"{task}:{completion or 'none'}": {
            "physical_state": Metric(),
            "previous_action": Metric(),
            "action": Metric(),
        }
        for task, completion in task_specs
    }
    physical_metrics = {
        "joint_position_rad": Metric(),
        "joint_velocity_rad_s": Metric(),
        "action_relative_rad": Metric(),
        "gravity_angular_degrees": Metric(),
        "forward_one_step_joint_position_rad": Metric(),
        "forward_one_step_joint_velocity_rad_s": Metric(),
        "applied_joint_torque_mean_nm": Metric(),
        "foot_contact_impulse_ns": Metric(),
    }
    horizons = {horizon: Metric() for horizon in (1, 8, 32, 128)}
    per_package: dict[str, Metric] = defaultdict(Metric)
    baseline_forward = Metric()
    shuffled_forward = Metric()
    posterior_means: list[np.ndarray] = []
    kl_per_dimension: list[np.ndarray] = []
    batches_seen = 0
    for batch_index, cpu_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _device_batch(cpu_batch, device)
        batches_seen += 1
        for task, completion in task_specs:
            masks = masker.generate(batch, force_task=task, force_completion=completion)
            with _autocast(device, str(config["training"]["amp"])):
                output = model(batch, masks, sample_from_prior=True, deterministic=True)
            key = f"{task}:{completion or 'none'}"
            for group_name, prediction, target, mask in (
                (
                    "physical_state",
                    output.physical_state,
                    batch["physical_state"],
                    masks.state_loss,
                ),
                (
                    "previous_action",
                    output.previous_action,
                    batch["previous_action"],
                    masks.previous_loss,
                ),
                ("action", output.action, batch["action"], masks.action_loss),
            ):
                masked_metrics[key][group_name].add(prediction - target, mask)
            state_prediction = _denormalize(
                output.physical_state, norm["state_mean"], norm["state_std"]
            )
            state_target = _denormalize(
                batch["physical_state"], norm["state_mean"], norm["state_std"]
            )
            physical_metrics["joint_position_rad"].add(
                state_prediction[..., :29] - state_target[..., :29],
                masks.state_loss[..., :29],
            )
            physical_metrics["joint_velocity_rad_s"].add(
                state_prediction[..., 29:58] - state_target[..., 29:58],
                masks.state_loss[..., 29:58],
            )
            action_prediction = _denormalize(
                output.action, norm["action_mean"], norm["action_std"]
            )
            action_target = _denormalize(
                batch["action"], norm["action_mean"], norm["action_std"]
            )
            physical_metrics["action_relative_rad"].add(
                action_prediction - action_target, masks.action_loss
            )
            gravity_slice = slice(64, 67) if dataset.state_dim == 70 else slice(61, 64)
            gravity_mask = masks.state_loss[..., gravity_slice].any(-1)
            physical_metrics["gravity_angular_degrees"].add(
                _gravity_angle(
                    state_prediction[..., gravity_slice], state_target[..., gravity_slice]
                ),
                gravity_mask,
            )
            if task != "forward":
                continue
            if output.auxiliary_transition.shape[-1]:
                auxiliary_prediction = _denormalize(
                    output.auxiliary_transition,
                    norm["auxiliary_mean"],
                    norm["auxiliary_std"],
                )
                auxiliary_target = _denormalize(
                    batch["auxiliary_transition"],
                    norm["auxiliary_mean"],
                    norm["auxiliary_std"],
                )
                physical_metrics["applied_joint_torque_mean_nm"].add(
                    auxiliary_prediction[..., :29] - auxiliary_target[..., :29],
                    batch["valid_action"][:, :, None],
                )
                physical_metrics["foot_contact_impulse_ns"].add(
                    auxiliary_prediction[..., 29:] - auxiliary_target[..., 29:],
                    batch["valid_action"][:, :, None],
                )
            posterior_means.append(output.posterior_mean.float().cpu().numpy())
            q_var = torch.exp(output.posterior_logvar)
            p_var = torch.exp(output.prior_logvar)
            kl = 0.5 * (
                output.prior_logvar
                - output.posterior_logvar
                + (q_var + torch.square(output.posterior_mean - output.prior_mean)) / p_var
                - 1.0
            )
            kl_per_dimension.append(kl.float().cpu().numpy())
            predicted_next_normalized = batch["physical_state"][:, :-1] + output.forward_delta
            predicted_next = _denormalize(
                predicted_next_normalized, norm["state_mean"], norm["state_std"]
            )
            target_next = state_target[:, 1:]
            valid = batch["valid_action"]
            physical_metrics["forward_one_step_joint_position_rad"].add(
                predicted_next[..., :29] - target_next[..., :29], valid[:, :, None]
            )
            physical_metrics["forward_one_step_joint_velocity_rad_s"].add(
                predicted_next[..., 29:58] - target_next[..., 29:58], valid[:, :, None]
            )
            normalized_error = output.forward_delta - (
                batch["physical_state"][:, 1:] - batch["physical_state"][:, :-1]
            )
            baseline_forward.add(normalized_error, valid[:, :, None])
            for horizon, metric in horizons.items():
                if horizon >= output.physical_state.shape[1]:
                    continue
                horizon_valid = batch["valid_state"][:, horizon]
                metric.add(
                    state_prediction[:, horizon, :29] - state_target[:, horizon, :29],
                    horizon_valid[:, None],
                )
            absolute_joint_error = (predicted_next[..., :29] - target_next[..., :29]).abs()
            valid_joint_count = valid.sum(dim=1).clamp_min(1) * 29
            sample_error = (
                absolute_joint_error * valid[:, :, None]
            ).sum(dim=(1, 2)) / valid_joint_count
            for package_index, package in enumerate(cpu_batch["package"]):
                per_package[str(package)].add(sample_error[package_index : package_index + 1])

            negative_batch = _shuffled_action_batch(batch, norm)
            negative_masks = masker.generate(negative_batch, force_task="forward")
            with _autocast(device, str(config["training"]["amp"])):
                negative_output = model(
                    negative_batch, negative_masks, sample_from_prior=True, deterministic=True
                )
            negative_target_delta = (
                batch["physical_state"][:, 1:] - batch["physical_state"][:, :-1]
            )
            shuffled_forward.add(
                negative_output.forward_delta - negative_target_delta, valid[:, :, None]
            )

    baseline_rmse = float(baseline_forward.result()["rmse"])
    shuffled_rmse = float(shuffled_forward.result()["rmse"])
    degradation = shuffled_rmse / baseline_rmse - 1.0 if baseline_rmse > 0 else math.inf
    threshold = float(config["evaluation"]["negative_control_min_degradation"])
    means = np.concatenate(posterior_means, axis=0) if posterior_means else np.empty((0, 0))
    kl_values = np.concatenate(kl_per_dimension, axis=0) if kl_per_dimension else np.empty((0, 0))
    active = (
        (means.var(axis=0) > 1e-2) & (kl_values.mean(axis=0) > 1e-2)
        if means.size
        else np.zeros(0, dtype=bool)
    )
    masked_summary: dict[str, Any] = {}
    for task_name, groups in masked_metrics.items():
        group_results = {name: metric.result() for name, metric in groups.items()}
        finite_group_rmse = [
            float(result["rmse"])
            for result in group_results.values()
            if math.isfinite(float(result["rmse"]))
        ]
        masked_summary[task_name] = {
            "groups": group_results,
            "macro_normalized_rmse": float(np.mean(finite_group_rmse))
            if finite_group_rmse
            else math.nan,
        }
    summary = {
        "passed": bool(degradation >= threshold),
        "checkpoint": str(checkpoint_path.expanduser().resolve()),
        "dataset_run": str(dataset_run),
        "split": "test",
        "batches": batches_seen,
        "masked_normalized": masked_summary,
        "physical_units": {
            name: metric.result() for name, metric in physical_metrics.items()
        },
        "multi_step_joint_position_rad": {
            str(horizon): metric.result() for horizon, metric in horizons.items()
        },
        "per_package_forward_joint_position": {
            name: metric.result() for name, metric in sorted(per_package.items())
        },
        "latent": {
            "dimension": int(means.shape[1]) if means.ndim == 2 else 0,
            "active_dimensions": int(active.sum()),
            "active_mask": active.tolist(),
        },
        "negative_control": {
            "baseline_forward_normalized_rmse": baseline_rmse,
            "shuffled_action_forward_normalized_rmse": shuffled_rmse,
            "relative_degradation": degradation,
            "required_degradation": threshold,
            "passed": bool(degradation >= threshold),
        },
    }
    atomic_write_json(output_run / "manifests" / "evaluation.json", summary)
    if summary["passed"]:
        atomic_write_text(output_run / "markers" / "cvae_eval.ok", "PASS\n")
    dataset.close()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a selected CVAE once on test")
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = evaluate(
        args.dataset_run, args.checkpoint, args.output_run, args.max_batches
    )
    print(f"CVAE evaluation: {'PASS' if summary['passed'] else 'FAIL'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
