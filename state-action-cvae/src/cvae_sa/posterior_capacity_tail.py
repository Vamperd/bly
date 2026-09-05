from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .models import build_model, parameter_count
from .physics_schema import PHYSICS_STATE_FIELDS, load_physics_schema
from .posterior_capacity import (
    DeterministicWindowSubset,
    FIXED_MASK_NAMES,
    MaskBankDataset,
    _device_batch,
    make_fixture_masks,
    selected_window_identities,
    validate_motion_prefix,
)
from .util import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_bytes,
    file_sha256,
    load_json,
    seed_everything,
)


FORMAT_VERSION = "sonic_posterior_capacity_tail_diagnostic_v1"
MARKER_NAME = "cvae_posterior_capacity_tail_diagnostic.ok"
PROGRESSION_THRESHOLD = 1e-2
EXPECTED_MOTIONS = 4
EXPECTED_WINDOW = 128
EXPECTED_PARAMETERS = 25_453_411
PARTIAL_MASK_NAMES = FIXED_MASK_NAMES[3:]
QUANTILES = (0.50, 0.90, 0.95, 0.99)


def _number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"tail diagnostic encountered a non-finite value: {value!r}")
    return result


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    array = np.asarray([_number(value) for value in values], dtype=np.float64)
    quantiles = np.quantile(array, QUANTILES)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "max": float(array.max()),
    }


def state_feature_labels(joint_names: list[str]) -> list[str]:
    if len(joint_names) != 29 or len(set(joint_names)) != 29:
        raise ValueError("F4A requires 29 unique joint names")
    labels: list[str] = []
    axes = {
        "base_lin_vel_robot": ("x", "y", "z"),
        "base_ang_vel_robot": ("x", "y", "z"),
        "gravity_robot": ("x", "y", "z"),
        "foot_contact": ("left", "right"),
    }
    for field, width in PHYSICS_STATE_FIELDS:
        if field in {"joint_pos_canonical", "joint_vel"}:
            labels.extend(f"{field}[{name}]" for name in joint_names)
        elif field in axes:
            names = axes[field]
            if len(names) != width:
                raise RuntimeError(f"unexpected width for {field}")
            labels.extend(f"{field}[{name}]" for name in names)
        elif width == 1:
            labels.append(field)
        else:
            labels.extend(f"{field}[{index}]" for index in range(width))
    if len(labels) != 70:
        raise RuntimeError(f"F4A constructed {len(labels)} State labels instead of 70")
    return labels


def action_feature_labels(joint_names: list[str]) -> list[str]:
    if len(joint_names) != 29:
        raise ValueError("F4A requires 29 Action joint names")
    return [f"action_target_canonical[{name}]" for name in joint_names]


def validate_f4a_checkpoint(
    checkpoint: dict[str, Any], dataset_hash: str,
) -> dict[str, Any]:
    if checkpoint.get("format_version") != "sonic_posterior_capacity_checkpoint_v1":
        raise ValueError("F4A requires a posterior-capacity v1 checkpoint")
    if checkpoint.get("dataset_manifest_sha256") != dataset_hash:
        raise ValueError("F4A checkpoint dataset manifest hash mismatch")
    config = checkpoint.get("config", {})
    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})
    requirements = {
        "motion_count": int(data.get("motion_count", -1)) == EXPECTED_MOTIONS,
        "window_transitions": int(data.get("window_transitions", -1)) == EXPECTED_WINDOW,
        "all_windows": data.get("max_windows") is None,
        "model_kind": model.get("kind") == "physics_posterior_transformer",
        "fixed_masks": training.get("mask_phase") == "fixed",
        "progression_gate": training.get("acceptance_gate") == "progression",
        "kl_disabled": float(training.get("kl_beta", math.nan)) == 0.0,
        "expected_seed": int(config.get("seed", -1)) == 20260830,
        "expected_parameters": int(checkpoint.get("parameter_count", -1)) == EXPECTED_PARAMETERS,
    }
    failed = [name for name, passed in requirements.items() if not passed]
    if failed:
        raise ValueError(f"F4A checkpoint contract mismatch: {failed}")
    return {
        "step": int(checkpoint.get("step", -1)),
        "parameter_count": int(checkpoint["parameter_count"]),
        "seed": int(config["seed"]),
        "motion_count": int(data["motion_count"]),
        "window_transitions": int(data["window_transitions"]),
        "mask_phase": str(training["mask_phase"]),
        "acceptance_gate": str(training["acceptance_gate"]),
        "kl_beta": float(training["kl_beta"]),
    }


def _identity(batch: dict[str, Any], index: int, mask_name: str) -> dict[str, Any]:
    def integer(name: str, default: int = -1) -> int:
        value = batch.get(name)
        if value is None:
            return default
        item = value[index]
        return int(item.detach().cpu()) if isinstance(item, torch.Tensor) else int(item)

    result = {
        "motion_key": str(batch["motion_key"][index]),
        "variant_id": integer("variant_id"),
        "episode_ref": str(batch.get("episode_ref", [""])[index]),
        "window_start": integer("window_start"),
        "source_window_index": integer("source_window_index", integer("window_index")),
        "mask_slot": integer("mask_slot"),
        "mask_name": mask_name,
    }
    result["fixture_id"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def _rmse(squared_error: float, count: int) -> float | None:
    return math.sqrt(squared_error / count) if count else None


def _maximum_location(
    *,
    domain: str,
    error: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
    labels: list[str],
) -> dict[str, Any] | None:
    if not bool(mask.any()):
        return None
    masked = error.masked_fill(~mask, -1.0)
    flat_index = int(masked.reshape(-1).argmax())
    time_index, feature_index = np.unravel_index(flat_index, tuple(masked.shape))
    target_value = _number(target[time_index, feature_index])
    prediction_value = _number(prediction[time_index, feature_index])
    scale = _number(std[feature_index])
    center = _number(mean[feature_index])
    absolute = _number(error[time_index, feature_index])
    return {
        "domain": domain,
        "time_index": int(time_index),
        "feature_index": int(feature_index),
        "feature_name": labels[feature_index],
        "target_normalized": target_value,
        "prediction_normalized": prediction_value,
        "absolute_error_normalized": absolute,
        "normalization_mean": center,
        "normalization_scale": scale,
        "target_physical": target_value * scale + center,
        "prediction_physical": prediction_value * scale + center,
        "absolute_error_physical": absolute * scale,
    }


def _new_feature_accumulator(
    domain: str,
    labels: list[str],
    means: np.ndarray,
    scales: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "domain": domain,
            "feature_index": index,
            "feature_name": label,
            "normalization_mean": _number(means[index]),
            "normalization_scale": _number(scales[index]),
            "chunks": [],
            "squared_error": 0.0,
            "count": 0,
            "exceed_count": 0,
            "worst": None,
        }
        for index, label in enumerate(labels)
    ]


def _accumulate_features(
    accumulators: list[dict[str, Any]],
    error: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    identity: dict[str, Any],
    means: np.ndarray,
    scales: np.ndarray,
) -> None:
    for feature_index, accumulator in enumerate(accumulators):
        feature_mask = mask[:, feature_index]
        if not bool(feature_mask.any()):
            continue
        values = error[:, feature_index].masked_select(feature_mask).numpy().astype(
            np.float64, copy=True
        )
        accumulator["chunks"].append(values)
        accumulator["squared_error"] += float(np.square(values).sum())
        accumulator["count"] += int(values.size)
        accumulator["exceed_count"] += int((values > PROGRESSION_THRESHOLD).sum())
        local_index = int(values.argmax())
        absolute = float(values[local_index])
        current = accumulator["worst"]
        if current is not None and absolute <= float(current["absolute_error_normalized"]):
            continue
        times = torch.nonzero(feature_mask, as_tuple=False).flatten()
        time_index = int(times[local_index])
        normalized_target = _number(target[time_index, feature_index])
        normalized_prediction = _number(prediction[time_index, feature_index])
        scale = _number(scales[feature_index])
        center = _number(means[feature_index])
        accumulator["worst"] = {
            **identity,
            "time_index": time_index,
            "target_normalized": normalized_target,
            "prediction_normalized": normalized_prediction,
            "absolute_error_normalized": absolute,
            "target_physical": normalized_target * scale + center,
            "prediction_physical": normalized_prediction * scale + center,
            "absolute_error_physical": absolute * scale,
        }


def _finish_features(accumulators: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for accumulator in accumulators:
        count = int(accumulator["count"])
        chunks = accumulator.pop("chunks")
        values = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
        absolute_distribution = distribution(values.tolist())
        scale = float(accumulator["normalization_scale"])
        physical_distribution = {
            key: (None if value is None else float(value) * scale)
            for key, value in absolute_distribution.items()
            if key != "count"
        }
        result.append({
            key: value
            for key, value in accumulator.items()
            if key != "squared_error"
        } | {
            "global_normalized_rmse": _rmse(float(accumulator["squared_error"]), count),
            "absolute_error_normalized": absolute_distribution,
            "absolute_error_physical": {
                "count": absolute_distribution["count"], **physical_distribution,
            },
            "threshold": PROGRESSION_THRESHOLD,
            "threshold_exceed_count": int(accumulator["exceed_count"]),
            "threshold_exceed_fraction": (
                float(accumulator["exceed_count"]) / count if count else 0.0
            ),
            "threshold_pass_fraction": (
                1.0 - float(accumulator["exceed_count"]) / count if count else 1.0
            ),
        })
    return result


def summarize_masks(fixture_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in fixture_records:
        grouped[str(record["mask_name"])].append(record)
    if set(grouped) != set(FIXED_MASK_NAMES):
        raise ValueError("F4A did not observe every fixed Mask type")
    summaries: dict[str, dict[str, Any]] = {}
    for name in FIXED_MASK_NAMES:
        records = grouped[name]
        state_squared = sum(float(record["state_squared_error"]) for record in records)
        action_squared = sum(float(record["action_squared_error"]) for record in records)
        state_count = sum(int(record["state_element_count"]) for record in records)
        action_count = sum(int(record["action_element_count"]) for record in records)
        continuous_count = state_count + action_count
        exceed_count = sum(int(record["threshold_exceed_count"]) for record in records)
        contact_correct = sum(int(record["contact_correct"]) for record in records)
        contact_count = sum(int(record["contact_count"]) for record in records)
        max_pass = sum(float(record["max_abs"]) <= PROGRESSION_THRESHOLD for record in records)
        summaries[name] = {
            "fixture_count": len(records),
            "state_rmse": distribution([
                float(record["state_rmse"])
                for record in records if record["state_rmse"] is not None
            ]),
            "action_rmse": distribution([
                float(record["action_rmse"])
                for record in records if record["action_rmse"] is not None
            ]),
            "combined_rmse": distribution([
                float(record["combined_rmse"]) for record in records
            ]),
            "max_abs": distribution([float(record["max_abs"]) for record in records]),
            "global_state_rmse": _rmse(state_squared, state_count),
            "global_action_rmse": _rmse(action_squared, action_count),
            "global_combined_rmse": _rmse(state_squared + action_squared, continuous_count),
            "continuous_element_count": continuous_count,
            "threshold_exceed_count": exceed_count,
            "threshold_exceed_fraction": exceed_count / continuous_count,
            "threshold_pass_fraction": 1.0 - exceed_count / continuous_count,
            "max_abs_pass_fixture_count": int(max_pass),
            "max_abs_pass_fixture_fraction": max_pass / len(records),
            "contact_correct": contact_correct,
            "contact_count": contact_count,
            "contact_accuracy": contact_correct / contact_count if contact_count else 1.0,
        }
    return summaries


def summarize_windows(fixture_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity_fields = (
        "motion_key", "variant_id", "episode_ref", "window_start", "source_window_index",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in fixture_records:
        grouped[tuple(record[field] for field in identity_fields)].append(record)
    summaries: list[dict[str, Any]] = []
    for key, records in grouped.items():
        state_squared = sum(float(record["state_squared_error"]) for record in records)
        action_squared = sum(float(record["action_squared_error"]) for record in records)
        state_count = sum(int(record["state_element_count"]) for record in records)
        action_count = sum(int(record["action_element_count"]) for record in records)
        continuous_count = state_count + action_count
        exceed_count = sum(int(record["threshold_exceed_count"]) for record in records)
        contact_correct = sum(int(record["contact_correct"]) for record in records)
        contact_count = sum(int(record["contact_count"]) for record in records)
        worst = max(records, key=lambda item: float(item["max_abs"]))
        identity = dict(zip(identity_fields, key, strict=True))
        summaries.append({
            **identity,
            "window_id": hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
            "fixture_count": len(records),
            "mask_names": [str(record["mask_name"]) for record in records],
            "global_state_rmse": _rmse(state_squared, state_count),
            "global_action_rmse": _rmse(action_squared, action_count),
            "global_combined_rmse": _rmse(
                state_squared + action_squared, continuous_count
            ),
            "fixture_combined_rmse": distribution([
                float(record["combined_rmse"]) for record in records
            ]),
            "fixture_max_abs": distribution([
                float(record["max_abs"]) for record in records
            ]),
            "continuous_element_count": continuous_count,
            "threshold_exceed_count": exceed_count,
            "threshold_exceed_fraction": exceed_count / continuous_count,
            "contact_correct": contact_correct,
            "contact_count": contact_count,
            "contact_accuracy": contact_correct / contact_count if contact_count else 1.0,
            "worst_fixture": {
                "fixture_id": worst["fixture_id"],
                "mask_name": worst["mask_name"],
                "combined_rmse": worst["combined_rmse"],
                "max_abs": worst["max_abs"],
                "maximum_error": worst["maximum_error"],
            },
        })
    return sorted(
        summaries,
        key=lambda item: (
            int(item["source_window_index"]), int(item["variant_id"]),
            int(item["window_start"]), str(item["motion_key"]),
        ),
    )


def classify_tail_diagnostic(
    mask_summaries: dict[str, dict[str, Any]],
    *,
    element_exceed_fraction: float,
    fixture_exceed_fraction: float,
) -> dict[str, Any]:
    partial_values: list[tuple[str, str, dict[str, Any]]] = []
    for mask_name in PARTIAL_MASK_NAMES:
        summary = mask_summaries[mask_name]
        for metric in ("state_rmse", "action_rmse"):
            values = summary[metric]
            if int(values["count"]):
                partial_values.append((mask_name, metric, values))
    partial_p95_pass = bool(partial_values) and all(
        float(values["p95"]) <= PROGRESSION_THRESHOLD
        for _, _, values in partial_values
    )
    broad_failures = [
        {"mask_name": mask_name, "metric": metric, "p50": values["p50"], "p90": values["p90"]}
        for mask_name, metric, values in partial_values
        if float(values["p50"]) > PROGRESSION_THRESHOLD
        or float(values["p90"]) > PROGRESSION_THRESHOLD
    ]
    concentrated_tail = (
        float(element_exceed_fraction) <= 0.01
        or float(fixture_exceed_fraction) <= 0.05
    )
    partial_p95 = [
        float(mask_summaries[name]["combined_rmse"]["p95"])
        for name in PARTIAL_MASK_NAMES
    ]
    partial_macro_p95 = float(np.mean(partial_p95))
    full_both_p95 = float(mask_summaries["full_both"]["combined_rmse"]["p95"])
    full_both_ratio = full_both_p95 / max(partial_macro_p95, 1e-12)
    global_latent_suspected = full_both_ratio > 3.0
    if broad_failures:
        classification = "broad_reconstruction_failure"
        next_step = "audit optimization and representation capacity before changing the gate"
    elif partial_p95_pass and concentrated_tail:
        classification = "tail_objective_mismatch"
        next_step = "design one per-window-balanced and tail-penalized reconstruction comparison"
    else:
        classification = "mixed_tail_and_reconstruction_failure"
        next_step = "inspect the reported worst windows/features before selecting one training change"
    return {
        "classification": classification,
        "progression_threshold": PROGRESSION_THRESHOLD,
        "partial_p95_pass": partial_p95_pass,
        "tail_concentrated": concentrated_tail,
        "continuous_element_exceed_fraction": float(element_exceed_fraction),
        "fixture_max_abs_exceed_fraction": float(fixture_exceed_fraction),
        "broad_failures": broad_failures,
        "full_both_p95_combined_rmse": full_both_p95,
        "partial_macro_p95_combined_rmse": partial_macro_p95,
        "full_both_to_partial_p95_ratio": full_both_ratio,
        "global_latent_bottleneck_suspected": global_latent_suspected,
        "next_step": next_step,
    }


@torch.no_grad()
def evaluate_tail(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    seed: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    joint_names: list[str],
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    state_labels = state_feature_labels(joint_names)
    action_labels = action_feature_labels(joint_names)
    state_features = _new_feature_accumulator(
        "state", state_labels[:68], state_mean[:68], state_std[:68]
    )
    action_features = _new_feature_accumulator(
        "action", action_labels, action_mean, action_std
    )
    fixture_records: list[dict[str, Any]] = []
    contact_errors: list[dict[str, Any]] = []
    model.eval()
    processed = 0
    for cpu_batch in loader:
        state_mask, action_mask, names = make_fixture_masks(cpu_batch, seed, held_out=False)
        batch = _device_batch(cpu_batch, device)
        output = model(batch, state_mask.to(device), action_mask.to(device))
        state_prediction = output.physical_state[..., :68].detach().cpu()
        action_prediction = output.action.detach().cpu()
        contact_logits = output.state_contact_logits.detach().cpu()
        state_target = cpu_batch["physical_state"][..., :68].float()
        action_target = cpu_batch["action"].float()
        state_error = torch.abs(state_prediction - state_target)
        action_error = torch.abs(action_prediction - action_target)
        for index, mask_name in enumerate(names):
            identity = _identity(cpu_batch, index, mask_name)
            state_target_mask = state_mask[index, :, :68]
            action_target_mask = action_mask[index]
            state_values = state_error[index].masked_select(state_target_mask)
            action_values = action_error[index].masked_select(action_target_mask)
            state_squared = float(torch.square(state_values).sum())
            action_squared = float(torch.square(action_values).sum())
            state_count = int(state_values.numel())
            action_count = int(action_values.numel())
            count = state_count + action_count
            if not count:
                raise ValueError("F4A fixture contains no continuous target")
            state_maximum = _maximum_location(
                domain="state", error=state_error[index], mask=state_target_mask,
                target=state_target[index], prediction=state_prediction[index],
                mean=state_mean[:68], std=state_std[:68], labels=state_labels[:68],
            )
            action_maximum = _maximum_location(
                domain="action", error=action_error[index], mask=action_target_mask,
                target=action_target[index], prediction=action_prediction[index],
                mean=action_mean, std=action_std, labels=action_labels,
            )
            maximum_candidates = [
                item for item in (state_maximum, action_maximum) if item is not None
            ]
            maximum = max(
                maximum_candidates,
                key=lambda item: float(item["absolute_error_normalized"]),
            )
            contact_mask = state_mask[index, :, 68:70]
            predicted_contact = contact_logits[index] >= 0.0
            target_contact = cpu_batch["physical_state"][index, :, 68:70] >= 0.5
            contact_correct = int(
                (predicted_contact == target_contact).masked_select(contact_mask).sum()
            )
            contact_count = int(contact_mask.sum())
            wrong = contact_mask & (predicted_contact != target_contact)
            for time_index, local_feature in torch.nonzero(wrong, as_tuple=False).tolist():
                contact_errors.append({
                    **identity,
                    "time_index": int(time_index),
                    "feature_index": 68 + int(local_feature),
                    "feature_name": state_labels[68 + int(local_feature)],
                    "target": int(target_contact[time_index, local_feature]),
                    "prediction": int(predicted_contact[time_index, local_feature]),
                    "logit": _number(contact_logits[index, time_index, local_feature]),
                })
            exceed_count = int((state_values > PROGRESSION_THRESHOLD).sum()) + int(
                (action_values > PROGRESSION_THRESHOLD).sum()
            )
            record = {
                **identity,
                "state_rmse": _rmse(state_squared, state_count),
                "action_rmse": _rmse(action_squared, action_count),
                "combined_rmse": _rmse(state_squared + action_squared, count),
                "max_abs": float(maximum["absolute_error_normalized"]),
                "maximum_error": maximum,
                "state_squared_error": state_squared,
                "state_element_count": state_count,
                "action_squared_error": action_squared,
                "action_element_count": action_count,
                "continuous_element_count": count,
                "threshold": PROGRESSION_THRESHOLD,
                "threshold_exceed_count": exceed_count,
                "threshold_exceed_fraction": exceed_count / count,
                "contact_correct": contact_correct,
                "contact_count": contact_count,
                "contact_accuracy": contact_correct / contact_count if contact_count else 1.0,
            }
            fixture_records.append(record)
            _accumulate_features(
                state_features, state_error[index], state_target_mask,
                state_target[index], state_prediction[index], identity,
                state_mean[:68], state_std[:68],
            )
            _accumulate_features(
                action_features, action_error[index], action_target_mask,
                action_target[index], action_prediction[index], identity,
                action_mean, action_std,
            )
        processed += len(names)
        if progress is not None:
            progress(processed)
    if not fixture_records:
        raise ValueError("F4A evaluated no fixtures")
    mask_summaries = summarize_masks(fixture_records)
    feature_summaries = _finish_features(state_features) + _finish_features(action_features)
    window_summaries = summarize_windows(fixture_records)
    state_squared = sum(float(record["state_squared_error"]) for record in fixture_records)
    action_squared = sum(float(record["action_squared_error"]) for record in fixture_records)
    state_count = sum(int(record["state_element_count"]) for record in fixture_records)
    action_count = sum(int(record["action_element_count"]) for record in fixture_records)
    continuous_count = state_count + action_count
    exceed_count = sum(int(record["threshold_exceed_count"]) for record in fixture_records)
    fixture_exceed_count = sum(
        float(record["max_abs"]) > PROGRESSION_THRESHOLD for record in fixture_records
    )
    contact_correct = sum(int(record["contact_correct"]) for record in fixture_records)
    contact_count = sum(int(record["contact_count"]) for record in fixture_records)
    global_summary = {
        "fixture_count": len(fixture_records),
        "state_element_count": state_count,
        "action_element_count": action_count,
        "continuous_element_count": continuous_count,
        "global_state_rmse": _rmse(state_squared, state_count),
        "global_action_rmse": _rmse(action_squared, action_count),
        "global_combined_rmse": _rmse(state_squared + action_squared, continuous_count),
        "worst_state_fixture_rmse": max(
            float(record["state_rmse"] or 0.0) for record in fixture_records
        ),
        "worst_action_fixture_rmse": max(
            float(record["action_rmse"] or 0.0) for record in fixture_records
        ),
        "worst_max_abs": max(float(record["max_abs"]) for record in fixture_records),
        "threshold": PROGRESSION_THRESHOLD,
        "threshold_exceed_count": exceed_count,
        "threshold_exceed_fraction": exceed_count / continuous_count,
        "threshold_pass_fraction": 1.0 - exceed_count / continuous_count,
        "max_abs_exceed_fixture_count": int(fixture_exceed_count),
        "max_abs_exceed_fixture_fraction": fixture_exceed_count / len(fixture_records),
        "max_abs_pass_fixture_fraction": 1.0 - fixture_exceed_count / len(fixture_records),
        "contact_correct": contact_correct,
        "contact_count": contact_count,
        "contact_error_count": contact_count - contact_correct,
        "contact_accuracy": contact_correct / contact_count if contact_count else 1.0,
    }
    assessment = classify_tail_diagnostic(
        mask_summaries,
        element_exceed_fraction=global_summary["threshold_exceed_fraction"],
        fixture_exceed_fraction=global_summary["max_abs_exceed_fixture_fraction"],
    )
    return {
        "global": global_summary,
        "masks": mask_summaries,
        "features": feature_summaries,
        "windows": window_summaries,
        "fixtures": fixture_records,
        "contact_errors": contact_errors,
        "top_worst_fixtures": sorted(
            fixture_records, key=lambda item: float(item["max_abs"]), reverse=True
        )[:20],
        "top_worst_windows": sorted(
            window_summaries,
            key=lambda item: (
                float(item["fixture_max_abs"]["max"] or 0.0),
                float(item["global_combined_rmse"] or 0.0),
            ),
            reverse=True,
        )[:20],
        "top_worst_features": sorted(
            feature_summaries,
            key=lambda item: float(item["absolute_error_normalized"]["max"] or 0.0),
            reverse=True,
        )[:20],
        "tail_assessment": assessment,
    }


def _matches(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-5, abs_tol=1e-7)


def validate_source_reproduction(
    observed: dict[str, Any], source_summary: dict[str, Any], checkpoint: dict[str, Any],
) -> dict[str, Any]:
    expected_metrics = source_summary.get("best_metrics") or {}
    expected = {
        "optimizer_step": int(expected_metrics.get("optimizer_step", -1)),
        "fixture_count": int(source_summary.get("mask_fixture_count", -1)),
        "worst_state_fixture_rmse": float(expected_metrics.get("worst_state_rmse", math.inf)),
        "worst_action_fixture_rmse": float(expected_metrics.get("worst_action_rmse", math.inf)),
        "worst_max_abs": float(expected_metrics.get("continuous_max_abs", math.inf)),
        "contact_accuracy": float(expected_metrics.get("contact_accuracy", -1.0)),
        "global_state_rmse": math.sqrt(float(
            expected_metrics.get("reconstruction_loss", {}).get("state", math.inf)
        )),
        "global_action_rmse": math.sqrt(float(
            expected_metrics.get("reconstruction_loss", {}).get("action", math.inf)
        )),
    }
    actual = {
        "optimizer_step": int(checkpoint.get("step", -1)),
        "fixture_count": int(observed["fixture_count"]),
        "worst_state_fixture_rmse": float(observed["worst_state_fixture_rmse"]),
        "worst_action_fixture_rmse": float(observed["worst_action_fixture_rmse"]),
        "worst_max_abs": float(observed["worst_max_abs"]),
        "contact_accuracy": float(observed["contact_accuracy"]),
        "global_state_rmse": float(observed["global_state_rmse"]),
        "global_action_rmse": float(observed["global_action_rmse"]),
    }
    checks = {
        key: (
            actual[key] == expected[key]
            if key in {"optimizer_step", "fixture_count"}
            else _matches(actual[key], expected[key])
        )
        for key in expected
    }
    if not all(checks.values()):
        raise ValueError(
            f"F4A did not reproduce the source checkpoint evaluation: {checks}"
        )
    return {"passed": True, "expected": expected, "observed": actual, "checks": checks}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )


def _svg_bar_chart(result: dict[str, Any]) -> str:
    masks = result["masks"]
    width, height = 1560, 980
    left, right, top, bottom = 100.0, 1520.0, 110.0, 760.0
    labels = list(FIXED_MASK_NAMES)
    series = (
        ("combined RMSE p50", "combined_rmse", "p50", "#2563eb"),
        ("combined RMSE p95", "combined_rmse", "p95", "#d97706"),
        ("max abs", "max_abs", "max", "#dc2626"),
    )
    values = [
        max(float(masks[name][metric][stat] or 1e-12), 1e-12)
        for name in labels for _, metric, stat, _ in series
    ] + [PROGRESSION_THRESHOLD]
    low = math.floor(math.log10(min(values))) - 1
    high = math.ceil(math.log10(max(values))) + 1
    chart_height = bottom - top
    map_y = lambda value: bottom - (
        math.log10(max(float(value), 1e-12)) - low
    ) / (high - low) * chart_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f3f4f6"/>',
        '<text x="28" y="38" font-size="24" font-weight="700" fill="#111827">F4A posterior tail diagnostic</text>',
        '<text x="28" y="66" font-size="13" fill="#4b5563">Same 80 training windows and 10 fixed Masks; read-only checkpoint evaluation</text>',
        f'<rect x="20" y="82" width="1520" height="790" rx="10" fill="#fff" stroke="#d1d5db"/>',
    ]
    for exponent in range(low, high + 1):
        y = map_y(10.0**exponent)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#4b5563">10^{exponent}</text>')
    threshold_y = map_y(PROGRESSION_THRESHOLD)
    parts.append(f'<line x1="{left}" y1="{threshold_y:.1f}" x2="{right}" y2="{threshold_y:.1f}" stroke="#7c3aed" stroke-width="2" stroke-dasharray="7 5"/>')
    parts.append(f'<text x="{right - 4}" y="{threshold_y - 7:.1f}" text-anchor="end" font-size="11" fill="#7c3aed">progression threshold 1e-2</text>')
    group_width = (right - left) / len(labels)
    bar_width = group_width * 0.20
    for group, name in enumerate(labels):
        center = left + (group + 0.5) * group_width
        for index, (_, metric, stat, color) in enumerate(series):
            value = max(float(masks[name][metric][stat] or 1e-12), 1e-12)
            x = center + (index - 1) * bar_width
            y = map_y(value)
            parts.append(f'<rect x="{x - bar_width * 0.42:.1f}" y="{y:.1f}" width="{bar_width * 0.84:.1f}" height="{bottom - y:.1f}" fill="{color}" opacity="0.86"/>')
        escaped = html.escape(name)
        parts.append(f'<text x="{center:.1f}" y="{bottom + 20:.1f}" text-anchor="end" font-size="10" fill="#374151" transform="rotate(-35 {center:.1f} {bottom + 20:.1f})">{escaped}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#374151"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#374151"/>')
    parts.append(f'<text x="28" y="{(top + bottom) / 2:.1f}" text-anchor="middle" font-size="12" fill="#374151" transform="rotate(-90 28 {(top + bottom) / 2:.1f})">Value (log10 scale)</text>')
    for index, (label, _, _, color) in enumerate(series):
        x = 110 + index * 310
        parts.append(f'<rect x="{x}" y="825" width="16" height="11" fill="{color}"/>')
        parts.append(f'<text x="{x + 23}" y="836" font-size="12" fill="#374151">{html.escape(label)}</text>')
    assessment = result["tail_assessment"]
    global_summary = result["global"]
    caption = (
        f"assessment={assessment['classification']}; element exceed fraction="
        f"{global_summary['threshold_exceed_fraction']:.6g}; fixture max exceed fraction="
        f"{global_summary['max_abs_exceed_fixture_fraction']:.6g}"
    )
    parts.append(f'<text x="28" y="915" font-size="13" fill="#111827">{html.escape(caption)}</text>')
    parts.append('<text x="28" y="943" font-size="12" fill="#4b5563">p50/p95 are across fixtures. max abs is the worst normalized continuous element; contact is reported separately in the manifest.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_tail_artifacts(result: dict[str, Any], output_run: Path) -> dict[str, str]:
    output_run = output_run.expanduser().resolve()
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "plots", "videos"):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    fixture_path = output_run / "data/posterior_tail_fixture_metrics.jsonl"
    window_path = output_run / "data/posterior_tail_window_metrics.jsonl"
    feature_path = output_run / "data/posterior_tail_feature_metrics.jsonl"
    contact_path = output_run / "data/posterior_tail_contact_errors.jsonl"
    plot_path = output_run / "plots/posterior_tail_diagnostic.svg"
    manifest_path = output_run / "manifests/posterior_tail_diagnostic.json"
    _write_jsonl(fixture_path, result.pop("fixtures"))
    _write_jsonl(window_path, result.pop("windows"))
    _write_jsonl(feature_path, result["features"])
    _write_jsonl(contact_path, result.pop("contact_errors"))
    result["artifacts"] = {
        "fixture_metrics": str(fixture_path),
        "window_metrics": str(window_path),
        "feature_metrics": str(feature_path),
        "contact_errors": str(contact_path),
        "plot": str(plot_path),
        "manifest": str(manifest_path),
    }
    atomic_write_text(plot_path, _svg_bar_chart(result))
    atomic_write_json(manifest_path, result)
    atomic_write_text(output_run / "markers" / MARKER_NAME, "PASS execution-only\n")
    return result["artifacts"]


def run_tail_diagnostic(
    dataset_run: Path,
    checkpoint_path: Path,
    output_run: Path,
    *,
    batch_size: int = 4,
    num_workers: int = 4,
) -> dict[str, Any]:
    from .dataset import StateActionWindowDataset

    dataset_run = dataset_run.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    if not (dataset_run / "markers/cvae_overfit_subset.ok").is_file():
        raise FileNotFoundError("F4A requires the dedicated overfit subset marker")
    if checkpoint_path.name != "best_progression.pt":
        raise ValueError("F4A must read the F4D best_progression.pt checkpoint")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    source_run = checkpoint_path.parent.parent
    source_summary_path = source_run / "manifests/posterior_capacity_summary.json"
    if not source_summary_path.is_file():
        raise FileNotFoundError("F4A source posterior-capacity summary is missing")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset_hash = file_sha256(dataset_run / "manifests/dataset_manifest.json")
    contract = validate_f4a_checkpoint(checkpoint, dataset_hash)
    source_summary = load_json(source_summary_path)
    if source_summary.get("passed") is not False:
        raise ValueError("F4A source must be the failed F4D run")
    if not bool(source_summary.get("fixed_fixture_identity_match")):
        raise ValueError("F4A source fixed fixtures were not identical")
    config = copy.deepcopy(checkpoint["config"])
    seed = int(config["seed"])
    seed_everything(seed)
    base = StateActionWindowDataset(
        dataset_run,
        "train",
        EXPECTED_WINDOW,
        EXPECTED_WINDOW,
        max_episodes=EXPECTED_MOTIONS * 8,
        random_crop=False,
    )
    try:
        selected_motions = validate_motion_prefix(base, EXPECTED_MOTIONS)
        selected_base = DeterministicWindowSubset(base, None)
        selected_windows = selected_window_identities(base, selected_base.indices)
        if int(source_summary.get("training_mask_seed", -1)) != seed:
            raise ValueError("F4A Mask seed differs from the F4D source summary")
        if len(selected_base) != int(source_summary.get("window_count", -1)):
            raise ValueError("F4A window count differs from the F4D source summary")
        if selected_motions != list(source_summary.get("selected_motion_keys", [])):
            raise ValueError("F4A selected motions differ from the F4D source summary")
        if selected_windows != list(source_summary.get("selected_windows", [])):
            raise ValueError("F4A window identities differ from the F4D source summary")
        fixture_data = MaskBankDataset(selected_base, len(FIXED_MASK_NAMES))
        if len(fixture_data) != int(source_summary.get("mask_fixture_count", -1)):
            raise ValueError("F4A fixture count differs from the F4D source summary")
        schema_paths = sorted({str(row["schema_path"]) for row in base.episodes})
        joint_names: list[str] | None = None
        for schema_path in schema_paths:
            current = list(load_physics_schema(Path(schema_path))["joint_names"])
            if joint_names is None:
                joint_names = current
            elif current != joint_names:
                raise ValueError("F4A source schemas disagree on joint order")
        if joint_names is None:
            raise ValueError("F4A found no source schema")
        config["model"]["state_dim"] = base.state_dim
        model = build_model(config["model"])
        if parameter_count(model) != EXPECTED_PARAMETERS:
            raise ValueError("F4A reconstructed model parameter count mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        workers = int(num_workers)
        loader = DataLoader(
            fixture_data,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=workers,
            drop_last=False,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )

        def report(processed: int) -> None:
            if processed % 100 == 0 or processed == len(fixture_data):
                print(f"F4A evaluated {processed}/{len(fixture_data)} fixtures", flush=True)

        evaluated = evaluate_tail(
            model,
            loader,
            device,
            seed=seed,
            state_mean=base.state_mean,
            state_std=base.state_std,
            action_mean=base.action_mean,
            action_std=base.action_std,
            joint_names=joint_names,
            progress=report,
        )
        reproduction = validate_source_reproduction(
            evaluated["global"], source_summary, checkpoint
        )
        result = {
            "format_version": FORMAT_VERSION,
            "scope": (
                "read-only tail diagnosis of the same F4D training windows and fixed Masks; "
                "no generalization, conditional-prior, or physical-inference claim"
            ),
            "execution_pass": True,
            "quality_pass": False,
            "quality_pass_meaning": "F4A is diagnostic-only; its marker never denotes model quality",
            "dataset_run": str(dataset_run),
            "dataset_manifest_sha256": dataset_hash,
            "checkpoint": {
                **contract,
                "path": str(checkpoint_path),
                "sha256": file_sha256(checkpoint_path),
                "source_run": str(source_run),
                "source_summary_sha256": file_sha256(source_summary_path),
            },
            "fixture_contract": {
                "training_mask_seed": seed,
                "fixed_fixture_identity_match": True,
                "motion_count": EXPECTED_MOTIONS,
                "selected_motion_keys": selected_motions,
                "window_transitions": EXPECTED_WINDOW,
                "window_count": len(selected_base),
                "selected_windows": selected_windows,
                "mask_names": list(FIXED_MASK_NAMES),
                "fixture_count": len(fixture_data),
            },
            "source_reproduction": reproduction,
            **evaluated,
        }
        write_tail_artifacts(result, output_run)
        return result
    finally:
        base.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the read-only F4A tail diagnosis on the failed four-motion checkpoint"
    )
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("F4A batch size must be positive and workers non-negative")
    result = run_tail_diagnostic(
        args.dataset_run,
        args.checkpoint,
        args.output_run,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print("Posterior capacity F4A tail diagnostic: PASS (execution only)")
    print(json.dumps({
        "output_run": str(args.output_run.expanduser().resolve()),
        "global": result["global"],
        "tail_assessment": result["tail_assessment"],
        "artifacts": result["artifacts"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
