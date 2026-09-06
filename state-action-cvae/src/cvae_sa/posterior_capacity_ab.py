from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, default_collate

from .models import PosteriorCapacityOutput, build_model, parameter_count
from .physics_schema import load_physics_schema
from .posterior_capacity import (
    DeterministicWindowSubset,
    FIXED_MASK_NAMES,
    MaskBankDataset,
    ReconstructionLoss,
    _device_batch,
    evaluate_exact,
    make_fixture_masks,
    reconstruction_loss,
    selected_window_identities,
    validate_motion_prefix,
)
from .posterior_capacity_tail import (
    EXPECTED_MOTIONS,
    EXPECTED_PARAMETERS,
    EXPECTED_WINDOW,
    action_feature_labels,
    evaluate_tail,
    state_feature_labels,
    validate_f4a_checkpoint,
    validate_output_isolation,
    validate_source_reproduction,
)
from .util import (
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    canonical_json_bytes,
    file_sha256,
    load_json,
    seed_everything,
)


FORMAT_VERSION = "sonic_posterior_ab_summary_v1"
CHECKPOINT_FORMAT = "sonic_posterior_ab_checkpoint_v1"
COMPARISON_FORMAT = "sonic_posterior_ab_comparison_v1"
SMOKE_MARKER = "cvae_posterior_ab_smoke.ok"
EXECUTION_MARKER = "cvae_posterior_ab_execution.ok"
COMPARISON_MARKER = "cvae_posterior_ab_comparison.ok"
PROGRESSION_MARKER = "cvae_posterior_capacity_progression.ok"
FIXTURE_SEED = 20260830
FORMAL_STEPS = 10_000
EVALUATION_STEPS = (8_000, 9_000, 10_000)
ARMS = ("A", "B")
EXPECTED_WINDOWS = 80
EXPECTED_FIXTURES = EXPECTED_WINDOWS * len(FIXED_MASK_NAMES)
EXPECTED_DATASET_RUN_NAME = "cvae_overfit_subset_20260828_234506"
EXPECTED_SOURCE_RUN_NAME = (
    "cvae_posterior_capacity_fixed_m4_t128_25m_s100000_gprogression_20260904_190425"
)
EXPECTED_F4A_RUN_NAME = "cvae_posterior_capacity_tail_diagnostic_f4a_20260905_200807"


@dataclass(frozen=True)
class ABObjective:
    optimization_total: torch.Tensor
    optimization_state: torch.Tensor
    optimization_action: torch.Tensor
    optimization_contact: torch.Tensor
    raw: ReconstructionLoss


def _finite_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"non-finite {name}")


def tail_mixed_domain_loss(
    squared_error: torch.Tensor,
    mask: torch.Tensor,
    *,
    tail_fraction: float = 0.2,
    tail_mix: float = 0.5,
) -> torch.Tensor | None:
    if squared_error.shape != mask.shape or squared_error.ndim < 2:
        raise ValueError("tail loss error and mask must have matching batched shapes")
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    if not 0.0 <= float(tail_mix) <= 1.0:
        raise ValueError("tail_mix must be in [0, 1]")
    weighted: list[torch.Tensor] = []
    counts: list[int] = []
    for index in range(squared_error.shape[0]):
        values = squared_error[index].masked_select(mask[index])
        count = int(values.numel())
        if not count:
            continue
        top_count = max(1, int(math.ceil(float(tail_fraction) * count)))
        mean = values.mean()
        tail = torch.topk(values, top_count, sorted=False).values.mean()
        weighted.append(count * ((1.0 - float(tail_mix)) * mean + float(tail_mix) * tail))
        counts.append(count)
    if not weighted:
        return None
    return torch.stack(weighted).sum() / sum(counts)


def ab_reconstruction_objective(
    output: PosteriorCapacityOutput,
    batch: dict[str, torch.Tensor],
    state_mask: torch.Tensor,
    action_mask: torch.Tensor,
    arm: str,
    *,
    tail_fraction: float = 0.2,
    tail_mix: float = 0.5,
) -> ABObjective:
    arm = str(arm).upper()
    if arm not in ARMS:
        raise ValueError("F4B-v2 supports only arm A or B; arm C is not triggered")
    raw = reconstruction_loss(output, batch, state_mask, action_mask)
    if arm == "A":
        return ABObjective(raw.total, raw.state, raw.action, raw.contact, raw)
    state_squared = torch.square(
        output.physical_state[..., :68] - batch["physical_state"][..., :68]
    )
    action_squared = torch.square(output.action - batch["action"])
    state = tail_mixed_domain_loss(
        state_squared, state_mask[..., :68],
        tail_fraction=tail_fraction, tail_mix=tail_mix,
    )
    action = tail_mixed_domain_loss(
        action_squared, action_mask,
        tail_fraction=tail_fraction, tail_mix=tail_mix,
    )
    contact_present = bool(state_mask[..., 68:70].any())
    terms = [value for value in (state, action) if value is not None]
    if contact_present:
        terms.append(raw.contact)
    if not terms:
        raise ValueError("posterior AB reconstruction batch contains no targets")
    zero_state = raw.state * 0.0
    return ABObjective(
        torch.stack(terms).mean(),
        state if state is not None else zero_state,
        action if action is not None else raw.action * 0.0,
        raw.contact,
        raw,
    )


def window_identity(item: dict[str, Any]) -> dict[str, Any]:
    def integer(name: str, default: int = -1) -> int:
        value = item.get(name, default)
        if isinstance(value, torch.Tensor):
            return int(value.detach().cpu())
        return int(value)

    return {
        "motion_key": str(item["motion_key"]),
        "variant_id": integer("variant_id"),
        "episode_ref": str(item.get("episode_ref", "")),
        "window_start": integer("window_start"),
        "source_window_index": integer("source_window_index", integer("window_index")),
    }


def batch_sample_identities(batch: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index in range(len(batch["motion_key"])):
        item = {
            key: (
                value[index].detach().cpu()
                if isinstance(value, torch.Tensor)
                else value[index]
            )
            for key, value in batch.items()
            if key in {
                "motion_key", "variant_id", "episode_ref", "window_start",
                "source_window_index", "window_index", "mask_slot",
            }
        }
        identity = window_identity(item)
        identity["mask_slot"] = int(item["mask_slot"])
        identity["mask_name"] = FIXED_MASK_NAMES[identity["mask_slot"]]
        result.append(identity)
    return result


def identity_sha256(identities: Iterable[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(identities))).hexdigest()


def fixture_bitmap_sha256(
    loader: DataLoader[dict[str, Any]], fixture_seed: int = FIXTURE_SEED
) -> str:
    digest = hashlib.sha256()
    seen = 0
    for batch in loader:
        state_mask, action_mask, names = make_fixture_masks(batch, fixture_seed)
        identities = batch_sample_identities(batch)
        for index, (identity, name) in enumerate(zip(identities, names, strict=True)):
            if identity["mask_name"] != name:
                raise ValueError("fixture identity and generated Mask name disagree")
            digest.update(canonical_json_bytes(identity))
            digest.update(np.packbits(state_mask[index].numpy().reshape(-1)).tobytes())
            digest.update(np.packbits(action_mask[index].numpy().reshape(-1)).tobytes())
            seen += 1
    if seen != EXPECTED_FIXTURES:
        raise ValueError(f"expected 800 F4B fixtures, found {seen}")
    return digest.hexdigest()


def donor_index_maps(identities: list[dict[str, Any]]) -> dict[str, list[int]]:
    if len(identities) < 2:
        raise ValueError("latent donor diagnostics require at least two windows")
    order = sorted(range(len(identities)), key=lambda index: tuple(
        identities[index][key]
        for key in ("motion_key", "variant_id", "episode_ref", "window_start", "source_window_index")
    ))
    cross_window = [0] * len(identities)
    for position, target in enumerate(order):
        cross_window[target] = order[(position + 1) % len(order)]
    groups: dict[str, list[int]] = {}
    for index in order:
        groups.setdefault(str(identities[index]["motion_key"]), []).append(index)
    motion_names = sorted(groups)
    if len(motion_names) < 2:
        raise ValueError("cross-motion diagnostic requires multiple motions")
    cross_motion = [0] * len(identities)
    for group_index, name in enumerate(motion_names):
        donor_group = groups[motion_names[(group_index + 1) % len(motion_names)]]
        for rank, target in enumerate(groups[name]):
            cross_motion[target] = donor_group[rank % len(donor_group)]
    for index, donor in enumerate(cross_window):
        if donor == index:
            raise ValueError("cross-window donor resolved to the target window")
    for index, donor in enumerate(cross_motion):
        if identities[index]["motion_key"] == identities[donor]["motion_key"]:
            raise ValueError("cross-motion donor has the target motion")
    return {"cross_window": cross_window, "cross_motion": cross_motion}


def _matches(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-5, abs_tol=1e-7)


def validate_step0(
    exact: dict[str, Any], tail: dict[str, Any], source_summary: dict[str, Any], checkpoint: dict[str, Any]
) -> dict[str, Any]:
    expected = source_summary["best_metrics"]
    checks = {
        "optimizer_step": int(checkpoint.get("step", -1)) == int(expected["optimizer_step"]),
        "fixture_count": int(tail["global"]["fixture_count"]) == int(source_summary["mask_fixture_count"]),
        "worst_state_rmse": _matches(exact["worst_state_rmse"], expected["worst_state_rmse"]),
        "worst_action_rmse": _matches(exact["worst_action_rmse"], expected["worst_action_rmse"]),
        "continuous_max_abs": _matches(exact["continuous_max_abs"], expected["continuous_max_abs"]),
        "contact_accuracy": _matches(exact["contact_accuracy"], expected["contact_accuracy"]),
        "global_state_rmse": _matches(
            tail["global"]["global_state_rmse"], math.sqrt(expected["reconstruction_loss"]["state"])
        ),
        "global_action_rmse": _matches(
            tail["global"]["global_action_rmse"], math.sqrt(expected["reconstruction_loss"]["action"])
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"F4B step0 did not reproduce F4D: {checks}")
    return {"passed": True, "checks": checks}


def _infinite(loader: DataLoader[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, text)


def _full_both_masks(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    state = batch["valid_state"].bool()[..., None].expand_as(batch["physical_state"]).clone()
    action = batch["valid_action"].bool()[..., None].expand_as(batch["action"]).clone()
    return state, action


@torch.no_grad()
def evaluate_full_objective(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    arm: str,
    tail_fraction: float,
    tail_mix: float,
) -> dict[str, Any]:
    """Aggregate the actual optimized objective over all fixed fixtures."""
    model.eval()
    raw_sums = {"state": 0.0, "action": 0.0, "contact": 0.0}
    raw_counts = {"state": 0, "action": 0, "contact": 0}
    optimized_sums = {"state": 0.0, "action": 0.0}
    optimized_counts = {"state": 0, "action": 0}
    for cpu_batch in loader:
        state_mask, action_mask, _ = make_fixture_masks(cpu_batch, FIXTURE_SEED)
        batch = _device_batch(cpu_batch, device)
        state_mask = state_mask.to(device)
        action_mask = action_mask.to(device)
        output = model(batch, state_mask, action_mask)
        state_squared = torch.square(
            output.physical_state[..., :68] - batch["physical_state"][..., :68]
        )
        action_squared = torch.square(output.action - batch["action"])
        contact_values = F.binary_cross_entropy_with_logits(
            output.state_contact_logits,
            batch["physical_state"][..., 68:70],
            reduction="none",
        ).masked_select(state_mask[..., 68:70])
        for name, values in (
            ("state", state_squared.masked_select(state_mask[..., :68])),
            ("action", action_squared.masked_select(action_mask)),
            ("contact", contact_values),
        ):
            raw_sums[name] += float(values.sum().cpu())
            raw_counts[name] += int(values.numel())
        for index in range(state_squared.shape[0]):
            for name, squared, mask in (
                ("state", state_squared[index], state_mask[index, ..., :68]),
                ("action", action_squared[index], action_mask[index]),
            ):
                values = squared.masked_select(mask)
                count = int(values.numel())
                if not count:
                    continue
                if arm == "A":
                    mixed = values.mean()
                else:
                    top_count = max(1, int(math.ceil(tail_fraction * count)))
                    mixed = (1.0 - tail_mix) * values.mean() + tail_mix * torch.topk(
                        values, top_count, sorted=False
                    ).values.mean()
                optimized_sums[name] += count * float(mixed.cpu())
                optimized_counts[name] += count
    raw_components = {
        name: raw_sums[name] / raw_counts[name]
        for name in raw_sums
        if raw_counts[name]
    }
    optimized_components = {
        name: optimized_sums[name] / optimized_counts[name]
        for name in optimized_sums
        if optimized_counts[name]
    }
    if raw_counts["contact"]:
        optimized_components["contact"] = raw_components["contact"]
    if not raw_components or not optimized_components:
        raise ValueError("full F4B objective evaluation contains no targets")
    return {
        "raw_reconstruction": {
            **raw_components,
            "total": sum(raw_components.values()) / len(raw_components),
            "counts": raw_counts,
            "aggregation": "global masked-element means; equal mean of present components",
        },
        "optimization_objective": {
            **optimized_components,
            "total": sum(optimized_components.values()) / len(optimized_components),
            "counts": {**optimized_counts, "contact": raw_counts["contact"]},
            "aggregation": (
                "element-count-weighted per-fixture/per-domain tail mixture; "
                "equal mean of present State, Action, and contact components"
                if arm == "B"
                else "identical to raw reconstruction"
            ),
        },
    }


@torch.no_grad()
def evaluate_latent_donors(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    identities: list[dict[str, Any]] = []
    latents: list[torch.Tensor] = []
    for cpu_batch in loader:
        state_mask, action_mask = _full_both_masks(cpu_batch)
        batch = _device_batch(cpu_batch, device)
        output = model(batch, state_mask.to(device), action_mask.to(device))
        latents.append(output.posterior_mean.detach().cpu())
        for index in range(len(cpu_batch["motion_key"])):
            item = {
                key: value[index] if not isinstance(value, torch.Tensor) else value[index].cpu()
                for key, value in cpu_batch.items()
                if key in {
                    "motion_key", "variant_id", "episode_ref", "window_start",
                    "source_window_index", "window_index",
                }
            }
            identities.append(window_identity(item))
    latent = torch.cat(latents, dim=0)
    maps = donor_index_maps(identities)
    accumulators = {
        name: {
            "state_squared": 0.0, "state_count": 0,
            "action_squared": 0.0, "action_count": 0,
            "legacy_squared": 0.0, "legacy_count": 0,
        }
        for name in ("correct", "cross_window", "cross_motion")
    }
    pairs: dict[str, list[dict[str, Any]]] = {"cross_window": [], "cross_motion": []}
    offset = 0
    for cpu_batch in loader:
        size = len(cpu_batch["motion_key"])
        state_mask, action_mask = _full_both_masks(cpu_batch)
        batch = _device_batch(cpu_batch, device)
        state_mask_device = state_mask.to(device)
        action_mask_device = action_mask.to(device)
        selections = {
            "correct": list(range(offset, offset + size)),
            "cross_window": [maps["cross_window"][index] for index in range(offset, offset + size)],
            "cross_motion": [maps["cross_motion"][index] for index in range(offset, offset + size)],
        }
        for name, donor_indices in selections.items():
            output = model(
                batch, state_mask_device, action_mask_device,
                latent_override=latent[donor_indices].to(device),
            )
            state_error = torch.square(
                output.physical_state[..., :68] - batch["physical_state"][..., :68]
            ).masked_select(state_mask_device[..., :68])
            action_error = torch.square(output.action - batch["action"]).masked_select(
                action_mask_device
            )
            legacy_state = torch.square(
                output.physical_state - batch["physical_state"]
            ).masked_select(state_mask_device)
            accumulator = accumulators[name]
            accumulator["state_squared"] += float(state_error.sum().cpu())
            accumulator["state_count"] += int(state_error.numel())
            accumulator["action_squared"] += float(action_error.sum().cpu())
            accumulator["action_count"] += int(action_error.numel())
            accumulator["legacy_squared"] += float(legacy_state.sum().cpu()) + float(
                action_error.sum().cpu()
            )
            accumulator["legacy_count"] += int(legacy_state.numel() + action_error.numel())
            if name != "correct":
                for local, donor in enumerate(donor_indices):
                    pairs[name].append({
                        "target": identities[offset + local],
                        "donor": identities[donor],
                    })
        offset += size
    if offset != len(identities):
        raise RuntimeError("latent donor evaluation lost window alignment")
    metrics: dict[str, dict[str, Any]] = {}
    for name, accumulator in accumulators.items():
        continuous_squared = accumulator["state_squared"] + accumulator["action_squared"]
        continuous_count = accumulator["state_count"] + accumulator["action_count"]
        metrics[name] = {
            "state_rmse": math.sqrt(accumulator["state_squared"] / accumulator["state_count"]),
            "action_rmse": math.sqrt(accumulator["action_squared"] / accumulator["action_count"]),
            "continuous_combined_rmse_excluding_contact": math.sqrt(
                continuous_squared / continuous_count
            ),
            "legacy_combined_rmse_including_contact": math.sqrt(
                accumulator["legacy_squared"] / accumulator["legacy_count"]
            ),
            "continuous_count": continuous_count,
            "legacy_count": accumulator["legacy_count"],
        }
    correct_continuous = max(metrics["correct"]["continuous_combined_rmse_excluding_contact"], 1e-12)
    correct_legacy = max(metrics["correct"]["legacy_combined_rmse_including_contact"], 1e-12)
    for name in ("cross_window", "cross_motion"):
        metrics[name]["continuous_ratio_to_correct"] = (
            metrics[name]["continuous_combined_rmse_excluding_contact"] / correct_continuous
        )
        metrics[name]["legacy_ratio_to_correct"] = (
            metrics[name]["legacy_combined_rmse_including_contact"] / correct_legacy
        )
    return {
        "window_count": len(identities),
        "identity_order_sha256": identity_sha256(identities),
        "aggregation": {
            "continuous": "State[:68]+Action; excludes contact",
            "legacy": "State[:70]+Action; includes contact probabilities",
        },
        "metrics": metrics,
        "pairs": pairs,
    }


def _identity_key(identity: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(identity[key] for key in (
        "motion_key", "variant_id", "episode_ref", "window_start", "source_window_index"
    ))


@torch.no_grad()
def select_fixed_curve_cases(
    model: torch.nn.Module,
    selected_base: DeterministicWindowSubset,
    top_windows: list[dict[str, Any]],
    device: torch.device,
    joint_names: list[str],
) -> list[dict[str, Any]]:
    model.eval()
    by_identity: dict[tuple[Any, ...], int] = {}
    for index in range(len(selected_base)):
        metadata = selected_base[index]
        by_identity[_identity_key(window_identity(metadata))] = index
    state_labels = state_feature_labels(joint_names)
    cases: list[dict[str, Any]] = []
    for rank, source in enumerate(top_windows[:5]):
        key = _identity_key(source)
        if key not in by_identity:
            raise ValueError("F4A top window is absent from F4B fixed windows")
        item = selected_base[by_identity[key]]
        fixtures = []
        for slot in range(len(FIXED_MASK_NAMES)):
            fixture = dict(item)
            fixture["window_index"] = by_identity[key]
            fixture["mask_slot"] = slot
            fixtures.append(fixture)
        cpu_batch = default_collate(fixtures)
        state_mask, action_mask, names = make_fixture_masks(cpu_batch, FIXTURE_SEED)
        batch = _device_batch(cpu_batch, device)
        output = model(batch, state_mask.to(device), action_mask.to(device))
        errors = torch.abs(
            output.physical_state[..., 29:58].detach().cpu()
            - cpu_batch["physical_state"][..., 29:58]
        )
        velocity_mask = state_mask[..., 29:58]
        candidates: list[tuple[float, int, int, int]] = []
        for slot in range(len(FIXED_MASK_NAMES)):
            for feature in range(29):
                times = torch.nonzero(velocity_mask[slot, :, feature], as_tuple=False).flatten()
                for time_index in times.tolist():
                    candidates.append((
                        float(errors[slot, time_index, feature]), slot, feature, int(time_index)
                    ))
        if not candidates:
            raise ValueError("fixed curve window has no masked joint velocity")
        maximum = max(candidates, key=lambda item: (item[0], -item[1], -item[2], -item[3]))
        _, slot, local_feature, time_index = maximum
        case_identity = window_identity(item)
        case = {
            "rank": rank,
            "window": case_identity,
            "mask_slot": slot,
            "mask_name": names[slot],
            "state_feature_index": 29 + local_feature,
            "feature_name": state_labels[29 + local_feature],
            "step0_max_time_index": time_index,
            "case_id": hashlib.sha256(canonical_json_bytes({
                "window": case_identity,
                "mask_slot": slot,
                "feature": 29 + local_feature,
            })).hexdigest(),
        }
        cases.append(case)
    if len(cases) != 5:
        raise ValueError("F4B requires five frozen curve cases")
    return cases


@torch.no_grad()
def evaluate_fixed_curves(
    model: torch.nn.Module,
    selected_base: DeterministicWindowSubset,
    cases: list[dict[str, Any]],
    device: torch.device,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    control_dt: float,
) -> list[dict[str, Any]]:
    model.eval()
    index_by_key: dict[tuple[Any, ...], int] = {}
    for index in range(len(selected_base)):
        index_by_key[_identity_key(window_identity(selected_base[index]))] = index
    rows: list[dict[str, Any]] = []
    for case in cases:
        base_index = index_by_key[_identity_key(case["window"])]
        item = dict(selected_base[base_index])
        item["window_index"] = base_index
        item["mask_slot"] = int(case["mask_slot"])
        cpu_batch = default_collate([item])
        state_mask, action_mask, names = make_fixture_masks(cpu_batch, FIXTURE_SEED)
        if names[0] != case["mask_name"]:
            raise ValueError("fixed curve Mask identity changed")
        batch = _device_batch(cpu_batch, device)
        output = model(batch, state_mask.to(device), action_mask.to(device))
        feature = int(case["state_feature_index"])
        target = cpu_batch["physical_state"][0, :, feature].float()
        prediction = output.physical_state[0, :, feature].detach().cpu()
        mask = state_mask[0, :, feature]
        scale = float(state_std[feature])
        center = float(state_mean[feature])
        rows.append({
            **case,
            "normalization_mean": center,
            "normalization_scale": scale,
            "target_normalized": target.tolist(),
            "prediction_normalized": prediction.tolist(),
            "absolute_error_normalized": torch.abs(prediction - target).tolist(),
            "masked_absolute_error_normalized": [
                float(value) if bool(selected) else None
                for value, selected in zip(torch.abs(prediction - target), mask, strict=True)
            ],
            "target_physical": (target * scale + center).tolist(),
            "prediction_physical": (prediction * scale + center).tolist(),
            "absolute_error_physical": (torch.abs(prediction - target) * scale).tolist(),
            "masked_absolute_error_physical": [
                float(value * scale) if bool(selected) else None
                for value, selected in zip(torch.abs(prediction - target), mask, strict=True)
            ],
            "control_dt_seconds": float(control_dt),
            "time_seconds": (torch.arange(len(target)) * float(control_dt)).tolist(),
            "masked": mask.tolist(),
            "valid": cpu_batch["valid_state"][0].bool().tolist(),
        })
    return rows


def _svg_series(
    title: str,
    series: list[tuple[str, list[tuple[float, float]], str]],
    *,
    y_label: str,
    log_y: bool,
    horizontal_lines: list[tuple[str, float, str]] | None = None,
    width: int = 1200,
    height: int = 680,
) -> str:
    left, right, top, bottom = 100.0, 30.0, 70.0, 85.0
    plot_width, plot_height = width - left - right, height - top - bottom
    horizontal_lines = horizontal_lines or []
    all_points = [point for _, points, _ in series for point in points]
    all_points.extend((0.0, value) for _, value, _ in horizontal_lines)
    if not all_points:
        all_points = [(0.0, 1.0)]
    xs = [point[0] for point in all_points]
    ys = [max(point[1], 1e-12) if log_y else point[1] for point in all_points]
    if log_y:
        ys = [math.log10(value) for value in ys]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        transformed = math.log10(max(value, 1e-12)) if log_y else value
        return top + (y_max - transformed) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="38" font-size="24" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>',
        f'<text x="{left + plot_width / 2}" y="{height - 25}" text-anchor="middle">Optimizer step</text>',
        f'<text x="24" y="{top + plot_height / 2}" transform="rotate(-90 24 {top + plot_height / 2})" text-anchor="middle">{html.escape(y_label)}</text>',
    ]
    for tick in range(6):
        x = x_min + (x_max - x_min) * tick / 5
        pixel = sx(x)
        parts.append(f'<text x="{pixel:.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-size="12">{x:.0f}</text>')
    for tick in range(6):
        transformed = y_min + (y_max - y_min) * tick / 5
        pixel = top + (y_max - transformed) / (y_max - y_min) * plot_height
        label = f"10^{transformed:.1f}" if log_y else f"{transformed:.4g}"
        parts.append(f'<line x1="{left}" y1="{pixel:.1f}" x2="{left + plot_width}" y2="{pixel:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{pixel + 4:.1f}" text-anchor="end" font-size="12">{label}</text>')
    legend_x = left
    for name, points, color in series:
        if points:
            polyline = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
            parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>')
            for x, y in points:
                parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="{color}"/>')
        parts.append(f'<rect x="{legend_x}" y="50" width="14" height="3" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="55" font-size="12">{html.escape(name)}</text>')
        legend_x += 180
    for name, value, color in horizontal_lines:
        pixel = sy(value)
        parts.append(f'<line x1="{left}" y1="{pixel:.1f}" x2="{left + plot_width}" y2="{pixel:.1f}" stroke="{color}" stroke-dasharray="7 5"/>')
        parts.append(f'<text x="{left + plot_width - 5}" y="{pixel - 5:.1f}" text-anchor="end" font-size="11" fill="{color}">{html.escape(name)}</text>')
    parts.append('<text x="100" y="655" font-size="11" fill="#4b5563">Log plots clip non-positive values to 1e-12. Evaluation is on the same fixed training windows and Masks.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_training_plots(output_run: Path, records: list[dict[str, Any]]) -> dict[str, str]:
    train = [record for record in records if record["phase"] == "train"]
    evaluations = [record for record in records if record["phase"] == "evaluation"]
    if len(train) > 2000:
        indices = np.linspace(0, len(train) - 1, num=2000, dtype=np.int64)
        train_plot = [train[int(index)] for index in indices]
    else:
        train_plot = train
    training_path = output_run / "plots/posterior_ab_training.svg"
    gate_path = output_run / "plots/posterior_ab_gates.svg"

    def ema_points(field: str, alpha: float = 0.05) -> list[tuple[float, float]]:
        result: list[tuple[float, float]] = []
        running: float | None = None
        for row in train:
            value = float(row[field]["total"])
            running = value if running is None else alpha * value + (1.0 - alpha) * running
            result.append((float(row["optimizer_step"]), running))
        return result

    training_series = [
        (
            "Train raw reconstruction",
            [(row["optimizer_step"], row["raw_reconstruction"]["total"]) for row in train_plot],
            "#94a3b8",
        ),
        (
            "Train raw objective",
            [(row["optimizer_step"], row["optimization_objective"]["total"]) for row in train_plot],
            "#c4b5fd",
        ),
        (
            "Reconstruction EMA α=0.05",
            ema_points("raw_reconstruction"),
            "#2563eb",
        ),
        (
            "Objective EMA α=0.05",
            ema_points("optimization_objective"),
            "#7c3aed",
        ),
        (
            "Full fixed-fixture reconstruction",
            [(row["optimizer_step"], row["exact"]["reconstruction_loss"]["total"]) for row in evaluations],
            "#dc2626",
        ),
        (
            "Full fixed-fixture objective",
            [(row["optimizer_step"], row["full_objective"]["optimization_objective"]["total"]) for row in evaluations],
            "#ea580c",
        ),
    ]
    gate_series = [
        (
            "Worst State RMSE",
            [(row["optimizer_step"], row["exact"]["worst_state_rmse"]) for row in evaluations],
            "#2563eb",
        ),
        (
            "Worst Action RMSE",
            [(row["optimizer_step"], row["exact"]["worst_action_rmse"]) for row in evaluations],
            "#059669",
        ),
        (
            "Worst max abs",
            [(row["optimizer_step"], row["exact"]["continuous_max_abs"]) for row in evaluations],
            "#dc2626",
        ),
        (
            "Element exceed fraction",
            [(row["optimizer_step"], row["tail_global"]["threshold_exceed_fraction"]) for row in evaluations],
            "#7c3aed",
        ),
    ]
    atomic_write_text(
        training_path,
        _svg_series(
            "F4B-v2 raw reconstruction and optimization objective",
            training_series,
            y_label="Value (log10 scale)",
            log_y=True,
        ),
    )
    atomic_write_text(
        gate_path,
        _svg_series(
            "F4B-v2 fixed-fixture quality metrics",
            gate_series,
            y_label="Value (log10 scale)",
            log_y=True,
            horizontal_lines=[
                ("progression RMSE/max = 1e-2", 1e-2, "#b91c1c"),
                ("exact max = 1e-3", 1e-3, "#d97706"),
                ("exact RMSE = 1e-4", 1e-4, "#059669"),
            ],
        ),
    )
    return {"training": str(training_path), "gates": str(gate_path)}


def render_fixed_curve_plot(path: Path, rows: list[dict[str, Any]], step: int) -> None:
    width, height = 1200, 250 * len(rows) + 60
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="40" y="34" font-size="23" font-weight="700">Frozen joint-velocity cases at optimizer step {step}</text>',
    ]
    for row_index, row in enumerate(rows):
        y0 = 55 + row_index * 250
        left, plot_width, plot_height = 80.0, 1080.0, 170.0
        target = row["target_physical"]
        prediction = row["prediction_physical"]
        valid = row["valid"]
        error = row["absolute_error_physical"]
        seconds = row["time_seconds"]
        values = [
            value
            for seq in (target, prediction, error)
            for value, is_valid in zip(seq, valid)
            if is_valid
        ]
        low, high = min(values), max(values)
        if high <= low:
            high = low + 1.0

        def sx(index: int) -> float:
            return left + seconds[index] / max(seconds[-1], 1e-12) * plot_width

        def sy(value: float) -> float:
            return y0 + 35 + (high - value) / (high - low) * plot_height

        parts.append(f'<text x="{left}" y="{y0 + 18}" font-size="13">{html.escape(row["feature_name"])} | {html.escape(row["mask_name"])} | {html.escape(row["window"]["motion_key"])}</text>')
        for index, masked in enumerate(row["masked"]):
            if masked:
                parts.append(f'<rect x="{sx(index):.1f}" y="{y0 + 35}" width="{max(plot_width / len(target), 1):.1f}" height="{plot_height}" fill="#fee2e2" opacity="0.45"/>')
        for values_row, color in ((target, "#111827"), (prediction, "#2563eb")):
            points = " ".join(
                f"{sx(index):.1f},{sy(value):.1f}"
                for index, (value, is_valid) in enumerate(zip(values_row, valid)) if is_valid
            )
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        for index, (value, masked, is_valid) in enumerate(zip(error, row["masked"], valid, strict=True)):
            if masked and is_valid:
                parts.append(f'<circle cx="{sx(index):.1f}" cy="{sy(value):.1f}" r="2" fill="#d97706"/>')
        for tick in range(5):
            index = round((len(seconds) - 1) * tick / 4)
            parts.append(f'<text x="{sx(index):.1f}" y="{y0 + 220}" text-anchor="middle" font-size="10">{seconds[index]:.2f}s</text>')
        parts.append(f'<text x="{left}" y="{y0 + 240}" font-size="11">Black: target; blue: prediction; orange points: masked absolute error only; red background: masked target; physical unit rad/s</text>')
    parts.append("</svg>")
    atomic_write_text(path, "\n".join(parts) + "\n")


def _load_joint_names(base: Any) -> list[str]:
    names: list[str] | None = None
    for schema_path in sorted({str(row["schema_path"]) for row in base.episodes}):
        current = list(load_physics_schema(Path(schema_path))["joint_names"])
        if names is None:
            names = current
        elif names != current:
            raise ValueError("F4B source schemas disagree on joint order")
    if names is None:
        raise ValueError("F4B found no source schema")
    return names


def _load_control_dt(base: Any) -> float:
    values = {
        float(load_physics_schema(Path(row["schema_path"]))["simulation"]["control_dt"])
        for row in base.episodes
    }
    if len(values) != 1:
        raise ValueError("F4B source schemas disagree on control_dt")
    return values.pop()


def validate_ab_contract(
    *,
    dataset_run: Path,
    checkpoint_path: Path,
    f4a_run: Path,
    output_run: Path,
    config: dict[str, Any],
    arm: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    arm = arm.upper()
    if arm not in ARMS:
        raise ValueError("F4B-v2 arm must be A or B; F4C is not enabled")
    source_run = validate_output_isolation(dataset_run, checkpoint_path, output_run)
    resolved_f4a = f4a_run.expanduser().resolve()
    resolved_output = output_run.expanduser().resolve()
    expected_names = {
        "dataset": (dataset_run.name, EXPECTED_DATASET_RUN_NAME),
        "source": (source_run.name, EXPECTED_SOURCE_RUN_NAME),
        "F4A": (resolved_f4a.name, EXPECTED_F4A_RUN_NAME),
    }
    wrong_names = [
        name for name, (observed, expected) in expected_names.items() if observed != expected
    ]
    if wrong_names:
        raise ValueError(f"F4B-v2 fixed source identity mismatch: {wrong_names}")
    if resolved_output == resolved_f4a or resolved_output.is_relative_to(resolved_f4a):
        raise ValueError("F4B output run must be isolated from F4A")
    if not (dataset_run / "markers/cvae_overfit_subset.ok").is_file():
        raise FileNotFoundError("F4B requires the dedicated overfit subset marker")
    if checkpoint_path.name != "best_progression.pt" or not checkpoint_path.is_file():
        raise FileNotFoundError("F4B requires the F4D best_progression.pt")
    f4a_marker = resolved_f4a / "markers/cvae_posterior_capacity_tail_diagnostic.ok"
    if not f4a_marker.is_file():
        raise FileNotFoundError("F4B requires the formal F4A execution marker")
    source_summary_path = source_run / "manifests/posterior_capacity_summary.json"
    f4a_manifest_path = resolved_f4a / "manifests/posterior_tail_diagnostic.json"
    source_summary = load_json(source_summary_path)
    f4a_manifest = load_json(f4a_manifest_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset_hash = file_sha256(dataset_run / "manifests/dataset_manifest.json")
    validate_f4a_checkpoint(checkpoint, dataset_hash)
    if source_summary.get("passed") is not False:
        raise ValueError("F4B source must be the failed F4D run")
    if not bool(f4a_manifest.get("execution_pass")):
        raise ValueError("F4A manifest did not complete")
    f4a_checkpoint = f4a_manifest.get("checkpoint", {})
    if f4a_checkpoint.get("sha256") != file_sha256(checkpoint_path):
        raise ValueError("F4A and F4B checkpoint hashes differ")
    expected_config = {
        "format_version": "sonic_posterior_ab_config_v1",
        "fixture_seed": FIXTURE_SEED,
        "data": {
            "motion_count": EXPECTED_MOTIONS,
            "window_transitions": EXPECTED_WINDOW,
            "max_windows": None,
        },
        "training": {
            "micro_batch": 4,
            "gradient_accumulation": 16,
            "max_optimizer_steps": FORMAL_STEPS,
            "validation_interval": 1000,
            "learning_rate": 3e-5,
            "warmup_steps": 250,
            "minimum_learning_rate": 1e-6,
            "gradient_clip": 1.0,
            "posterior_path": "mean",
            "kl_beta": 0.0,
            "free_bits": 0.0,
            "weight_decay": 0.0,
            "tail_fraction": 0.2,
            "tail_mix": 0.5,
        },
    }
    failed: list[str] = []
    if config.get("format_version") != expected_config["format_version"]:
        failed.append("format_version")
    if int(config.get("fixture_seed", -1)) != FIXTURE_SEED:
        failed.append("fixture_seed")
    for name, value in expected_config["data"].items():
        if config.get("data", {}).get(name) != value:
            failed.append(f"data.{name}")
    for name, value in expected_config["training"].items():
        if config.get("training", {}).get(name) != value:
            failed.append(f"training.{name}")
    if failed:
        raise ValueError(f"F4B config contract mismatch: {failed}")
    source_model = checkpoint["config"]["model"]
    target_model = config["model"]
    for key in (
        "kind", "d_model", "encoder_layers", "decoder_layers", "heads",
        "ffn_dim", "latent_dim", "dropout",
    ):
        if target_model.get(key) != source_model.get(key):
            raise ValueError(f"F4B model differs from F4D at {key}")
    return checkpoint, source_summary, f4a_manifest


def _evaluation_paths(output_run: Path, step: int) -> dict[str, Path]:
    stem = f"step_{step:05d}"
    return {
        "fixture": output_run / f"data/{stem}_fixture_metrics.jsonl",
        "window": output_run / f"data/{stem}_window_metrics.jsonl",
        "feature": output_run / f"data/{stem}_feature_metrics.jsonl",
        "contact": output_run / f"data/{stem}_contact_errors.jsonl",
        "donor": output_run / f"data/{stem}_latent_donors.json",
        "curves": output_run / f"data/{stem}_fixed_velocity_curves.json",
        "curve_plot": output_run / f"plots/{stem}_fixed_velocity_curves.svg",
    }


def run_full_evaluation(
    *,
    model: torch.nn.Module,
    validation_loader: DataLoader[dict[str, Any]],
    base_loader: DataLoader[dict[str, Any]],
    selected_base: DeterministicWindowSubset,
    device: torch.device,
    config: dict[str, Any],
    joint_names: list[str],
    fixed_cases: list[dict[str, Any]],
    output_run: Path,
    step: int,
) -> dict[str, Any]:
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training = config["training"]
    exact = evaluate_exact(
        model, validation_loader, device, FIXTURE_SEED, False,
        {key: float(value) for key, value in training["thresholds"].items()},
        {key: float(value) for key, value in training["progression_thresholds"].items()},
        "progression",
    )
    full_objective = evaluate_full_objective(
        model,
        validation_loader,
        device,
        arm=str(config["arm"]),
        tail_fraction=float(training["tail_fraction"]),
        tail_mix=float(training["tail_mix"]),
    )
    for name in ("state", "action", "contact", "total"):
        if not _matches(
            full_objective["raw_reconstruction"][name],
            exact["reconstruction_loss"][name],
        ):
            raise ValueError(f"full objective raw {name} disagrees with exact evaluator")
    tail = evaluate_tail(
        model, validation_loader, device,
        seed=FIXTURE_SEED,
        state_mean=selected_base.base.state_mean,
        state_std=selected_base.base.state_std,
        action_mean=selected_base.base.action_mean,
        action_std=selected_base.base.action_std,
        joint_names=joint_names,
    )
    donors = evaluate_latent_donors(model, base_loader, device)
    curves = evaluate_fixed_curves(
        model, selected_base, fixed_cases, device,
        selected_base.base.state_mean, selected_base.base.state_std,
        float(config["control_dt"]),
    )
    paths = _evaluation_paths(output_run, step)
    _write_jsonl(paths["fixture"], tail.pop("fixtures"))
    _write_jsonl(paths["window"], tail.pop("windows"))
    _write_jsonl(paths["feature"], tail["features"])
    _write_jsonl(paths["contact"], tail.pop("contact_errors"))
    atomic_write_json(paths["donor"], donors)
    atomic_write_json(paths["curves"], {"optimizer_step": step, "cases": curves})
    render_fixed_curve_plot(paths["curve_plot"], curves, step)
    feature_p95 = [
        {
            "domain": row["domain"],
            "feature_index": row["feature_index"],
            "feature_name": row["feature_name"],
            "normalized_p95": row["absolute_error_normalized"]["p95"],
            "physical_p95": row["absolute_error_physical"]["p95"],
            "threshold_exceed_fraction": row["threshold_exceed_fraction"],
        }
        for row in tail["features"]
    ]
    return {
        "phase": "evaluation",
        "optimizer_step": int(step),
        "evaluation_scope": "full fixed-fixture evaluation on the same F4D training windows and Masks",
        "exact": exact,
        "full_objective": full_objective,
        "tail_global": tail["global"],
        "tail_masks": tail["masks"],
        "feature_p95": feature_p95,
        "top_worst_windows": tail["top_worst_windows"],
        "top_worst_features": tail["top_worst_features"],
        "latent_donors": {
            "window_count": donors["window_count"],
            "identity_order_sha256": donors["identity_order_sha256"],
            "aggregation": donors["aggregation"],
            "metrics": donors["metrics"],
            "mapping_artifact": str(paths["donor"]),
        },
        "fixed_curve_artifact": str(paths["curves"]),
        "fixed_curve_plot": str(paths["curve_plot"]),
        "detail_artifacts": {key: str(value) for key, value in paths.items()},
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "evaluation_seconds": time.monotonic() - started,
    }


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    resolved_config: dict[str, Any],
    step: int,
    best_score: float,
    dataset_hash: str,
    source_checkpoint_hash: str,
    fixture_hash: str,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT,
        "optimizer_step": int(step),
        "best_progression_score": float(best_score),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "resolved_config": resolved_config,
        "dataset_manifest_sha256": dataset_hash,
        "source_checkpoint_sha256": source_checkpoint_hash,
        "fixture_bitmap_sha256": fixture_hash,
        "parameter_count": parameter_count(model),
    }


def validate_saved_checkpoint(
    path: Path,
    *,
    expected_step: int,
    dataset_hash: str,
    source_checkpoint_hash: str,
    fixture_hash: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "format_version": checkpoint.get("format_version") == CHECKPOINT_FORMAT,
        "optimizer_step": int(checkpoint.get("optimizer_step", -1)) == int(expected_step),
        "dataset_manifest_sha256": checkpoint.get("dataset_manifest_sha256") == dataset_hash,
        "source_checkpoint_sha256": (
            checkpoint.get("source_checkpoint_sha256") == source_checkpoint_hash
        ),
        "fixture_bitmap_sha256": checkpoint.get("fixture_bitmap_sha256") == fixture_hash,
        "parameter_count": int(checkpoint.get("parameter_count", -1)) == EXPECTED_PARAMETERS,
        "model_state_present": bool(checkpoint.get("model")),
        "optimizer_state_present": bool(checkpoint.get("optimizer")),
        "scheduler_state_present": bool(checkpoint.get("scheduler")),
    }
    if not all(checks.values()):
        raise ValueError(f"F4B saved-checkpoint readback failed: {checks}")
    return {"passed": True, "checks": checks, "sha256": file_sha256(path)}


def _learning_rate_multiplier(
    step: int, *, warmup_steps: int, max_steps: int, minimum_ratio: float
) -> float:
    if step < warmup_steps:
        return max((step + 1) / max(warmup_steps, 1), 1e-8)
    progress = min(
        max((step - warmup_steps) / max(max_steps - warmup_steps, 1), 0.0),
        1.0,
    )
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def _append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _progression_last_three_pass(evaluations: list[dict[str, Any]]) -> bool:
    by_step = {int(row["optimizer_step"]): row for row in evaluations}
    return all(
        step in by_step and bool(by_step[step]["exact"]["progression_gate"]["passed"])
        for step in EVALUATION_STEPS
    )


def run_ab_experiment(
    *,
    dataset_run: Path,
    source_checkpoint: Path,
    f4a_run: Path,
    output_run: Path,
    config: dict[str, Any],
    arm: str,
    optimizer_seed: int,
    smoke: bool = False,
) -> dict[str, Any]:
    from .dataset import StateActionWindowDataset

    dataset_run = dataset_run.expanduser().resolve()
    source_checkpoint = source_checkpoint.expanduser().resolve()
    f4a_run = f4a_run.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    arm = arm.upper()
    checkpoint, source_summary, f4a_manifest = validate_ab_contract(
        dataset_run=dataset_run,
        checkpoint_path=source_checkpoint,
        f4a_run=f4a_run,
        output_run=output_run,
        config=config,
        arm=arm,
    )
    data_config = config["data"]
    training = config["training"]
    dataset_hash = file_sha256(dataset_run / "manifests/dataset_manifest.json")
    checkpoint_hash = file_sha256(source_checkpoint)
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
        fixture_data = MaskBankDataset(selected_base, len(FIXED_MASK_NAMES))
        if len(selected_base) != EXPECTED_WINDOWS or len(fixture_data) != EXPECTED_FIXTURES:
            raise ValueError(
                f"F4B requires 80 windows/800 fixtures; found "
                f"{len(selected_base)}/{len(fixture_data)}"
            )
        if selected_motions != list(source_summary.get("selected_motion_keys", [])):
            raise ValueError("F4B selected motions differ from F4D")
        if selected_windows != list(source_summary.get("selected_windows", [])):
            raise ValueError("F4B selected windows differ from F4D")
        f4a_fixture = f4a_manifest.get("fixture_contract", {})
        if selected_windows != list(f4a_fixture.get("selected_windows", [])):
            raise ValueError("F4B selected windows differ from F4A")
        if int(f4a_fixture.get("fixture_count", -1)) != EXPECTED_FIXTURES:
            raise ValueError("F4A fixture count is not 800")
        if int(f4a_fixture.get("training_mask_seed", -1)) != FIXTURE_SEED:
            raise ValueError("F4A fixture seed differs from F4B")

        resolved_config = copy.deepcopy(config)
        resolved_config["model"]["state_dim"] = base.state_dim
        resolved_config["arm"] = arm
        resolved_config["optimizer_seed"] = int(optimizer_seed)
        resolved_config["smoke"] = bool(smoke)
        model = build_model(resolved_config["model"])
        if parameter_count(model) != EXPECTED_PARAMETERS:
            raise ValueError("F4B reconstructed model parameter count mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        workers = 0 if smoke else int(data_config.get("num_workers", 4))
        validation_loader = DataLoader(
            fixture_data,
            batch_size=int(training["micro_batch"]),
            shuffle=False,
            num_workers=workers,
            generator=torch.Generator().manual_seed(FIXTURE_SEED + 1),
            drop_last=False,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )
        base_loader = DataLoader(
            selected_base,
            batch_size=int(training["micro_batch"]),
            shuffle=False,
            num_workers=workers,
            generator=torch.Generator().manual_seed(FIXTURE_SEED + 2),
            drop_last=False,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )
        bitmap_hash = fixture_bitmap_sha256(validation_loader, FIXTURE_SEED)
        identity_contract_path = output_run / "manifests/posterior_ab_identity_contract.json"
        atomic_write_json(identity_contract_path, {
            "fixture_seed": FIXTURE_SEED,
            "selected_motion_keys": selected_motions,
            "window_transitions": EXPECTED_WINDOW,
            "selected_windows": selected_windows,
            "mask_names": list(FIXED_MASK_NAMES),
            "fixture_bitmap_sha256": bitmap_hash,
        })
        joint_names = _load_joint_names(base)
        resolved_config["control_dt"] = _load_control_dt(base)
        fixed_cases = select_fixed_curve_cases(
            model,
            selected_base,
            list(f4a_manifest.get("top_worst_windows", [])),
            device,
            joint_names,
        )
        fixed_case_hash = identity_sha256(fixed_cases)
        atomic_write_json(
            output_run / "manifests/fixed_velocity_cases.json",
            {"sha256": fixed_case_hash, "cases": fixed_cases},
        )

        metrics_path = output_run / "logs/metrics.jsonl"
        records: list[dict[str, Any]] = []
        step0 = run_full_evaluation(
            model=model,
            validation_loader=validation_loader,
            base_loader=base_loader,
            selected_base=selected_base,
            device=device,
            config=resolved_config,
            joint_names=joint_names,
            fixed_cases=fixed_cases,
            output_run=output_run,
            step=0,
        )
        step0_reproduction = validate_step0(
            step0["exact"], {"global": step0["tail_global"]}, source_summary, checkpoint
        )
        legacy_reproduction = validate_source_reproduction(
            step0["tail_global"], source_summary, checkpoint
        )
        step0["source_reproduction"] = step0_reproduction
        step0["training_context"] = {
            "source_checkpoint_step": int(checkpoint["step"]),
            "learning_rate": None,
            "gradient_norm_before_clip": None,
        }
        records.append(step0)
        _append_record(metrics_path, step0)

        # Reset every optimizer-visible RNG after all step-0 diagnostics.  The
        # training order itself uses a dedicated Generator, so evaluation can
        # never perturb the A/B sample stream.
        seed_everything(int(optimizer_seed))
        train_generator = torch.Generator().manual_seed(int(optimizer_seed))
        train_loader = DataLoader(
            fixture_data,
            batch_size=int(training["micro_batch"]),
            shuffle=True,
            num_workers=workers,
            generator=train_generator,
            drop_last=False,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        max_steps = 2 if smoke else FORMAL_STEPS
        warmup_steps = int(training["warmup_steps"])
        minimum_ratio = float(training["minimum_learning_rate"]) / float(
            training["learning_rate"]
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: _learning_rate_multiplier(
                step,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
                minimum_ratio=minimum_ratio,
            ),
        )
        accumulation = int(training["gradient_accumulation"])
        evaluation_interval = 2 if smoke else int(training["validation_interval"])
        stream = _infinite(train_loader)
        best_score = float(step0["exact"]["progression_gate"]["score"])
        best_evaluation = step0
        evaluations = [step0]
        atomic_torch_save(
            output_run / "checkpoints/best_progression.pt",
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                resolved_config=resolved_config,
                step=0,
                best_score=best_score,
                dataset_hash=dataset_hash,
                source_checkpoint_hash=checkpoint_hash,
                fixture_hash=bitmap_hash,
            ),
        )
        started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for optimizer_step in range(1, max_steps + 1):
            step_started = time.monotonic()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            sums = {
                "raw_total": 0.0,
                "raw_state": 0.0,
                "raw_action": 0.0,
                "raw_contact": 0.0,
                "opt_total": 0.0,
                "opt_state": 0.0,
                "opt_action": 0.0,
                "opt_contact": 0.0,
            }
            sampled: list[dict[str, Any]] = []
            for _ in range(accumulation):
                cpu_batch = next(stream)
                sampled.extend(batch_sample_identities(cpu_batch))
                state_mask, action_mask, _ = make_fixture_masks(cpu_batch, FIXTURE_SEED)
                batch = _device_batch(cpu_batch, device)
                state_mask = state_mask.to(device)
                action_mask = action_mask.to(device)
                output = model(batch, state_mask, action_mask)
                objective = ab_reconstruction_objective(
                    output,
                    batch,
                    state_mask,
                    action_mask,
                    arm,
                    tail_fraction=float(training["tail_fraction"]),
                    tail_mix=float(training["tail_mix"]),
                )
                _finite_tensor("optimization objective", objective.optimization_total)
                (objective.optimization_total / accumulation).backward()
                values = {
                    "raw_total": objective.raw.total,
                    "raw_state": objective.raw.state,
                    "raw_action": objective.raw.action,
                    "raw_contact": objective.raw.contact,
                    "opt_total": objective.optimization_total,
                    "opt_state": objective.optimization_state,
                    "opt_action": objective.optimization_action,
                    "opt_contact": objective.optimization_contact,
                }
                for name, value in values.items():
                    sums[name] += float(value.detach().cpu()) / accumulation
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            _finite_tensor("gradient norm", gradient_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            train_record = {
                "phase": "train",
                "optimizer_step": optimizer_step,
                "arm": arm,
                "sample_identity_sha256": identity_sha256(sampled),
                "sample_count": len(sampled),
                "raw_reconstruction": {
                    "total": sums["raw_total"],
                    "state": sums["raw_state"],
                    "action": sums["raw_action"],
                    "contact": sums["raw_contact"],
                },
                "optimization_objective": {
                    "total": sums["opt_total"],
                    "state": sums["opt_state"],
                    "action": sums["opt_action"],
                    "contact": sums["opt_contact"],
                },
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "gradient_clip_threshold": float(training["gradient_clip"]),
                "gradient_was_clipped": bool(
                    float(gradient_norm.detach().cpu()) > float(training["gradient_clip"])
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "cuda_peak_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                ),
                "step_seconds": time.monotonic() - step_started,
            }
            records.append(train_record)
            _append_record(metrics_path, train_record)
            if optimizer_step % evaluation_interval != 0 and optimizer_step != max_steps:
                continue
            evaluation = run_full_evaluation(
                model=model,
                validation_loader=validation_loader,
                base_loader=base_loader,
                selected_base=selected_base,
                device=device,
                config=resolved_config,
                joint_names=joint_names,
                fixed_cases=fixed_cases,
                output_run=output_run,
                step=optimizer_step,
            )
            evaluation["training_context"] = {
                "learning_rate": train_record["learning_rate"],
                "gradient_norm_before_clip": train_record["gradient_norm_before_clip"],
                "gradient_clip_threshold": train_record["gradient_clip_threshold"],
                "gradient_was_clipped": train_record["gradient_was_clipped"],
                "preceding_train_step_seconds": train_record["step_seconds"],
            }
            evaluations.append(evaluation)
            records.append(evaluation)
            _append_record(metrics_path, evaluation)
            score = float(evaluation["exact"]["progression_gate"]["score"])
            if score < best_score:
                best_score = score
                best_evaluation = evaluation
                atomic_torch_save(
                    output_run / "checkpoints/best_progression.pt",
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        resolved_config=resolved_config,
                        step=optimizer_step,
                        best_score=best_score,
                        dataset_hash=dataset_hash,
                        source_checkpoint_hash=checkpoint_hash,
                        fixture_hash=bitmap_hash,
                    ),
                )
            atomic_torch_save(
                output_run / "checkpoints/last.pt",
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    resolved_config=resolved_config,
                    step=optimizer_step,
                    best_score=best_score,
                    dataset_hash=dataset_hash,
                    source_checkpoint_hash=checkpoint_hash,
                    fixture_hash=bitmap_hash,
                ),
            )
            render_training_plots(output_run, records)
            model.train()

        quality_pass = False if smoke else _progression_last_three_pass(evaluations)
        checkpoint_readback = validate_saved_checkpoint(
            output_run / "checkpoints/last.pt",
            expected_step=max_steps,
            dataset_hash=dataset_hash,
            source_checkpoint_hash=checkpoint_hash,
            fixture_hash=bitmap_hash,
        )
        plots = render_training_plots(output_run, records)
        training_hashes = {
            str(row["optimizer_step"]): row["sample_identity_sha256"]
            for row in records
            if row["phase"] == "train"
        }
        train_records = [row for row in records if row["phase"] == "train"]
        summary = {
            "format_version": FORMAT_VERSION,
            "scope": (
                "seen four-motion fixed-Mask posterior-mean memorization only; "
                "no new-Mask, conditional-prior, unseen-motion, or physical-inference claim"
            ),
            "execution_pass": True,
            "quality_pass": quality_pass,
            "smoke": bool(smoke),
            "arm": arm,
            "fixture_seed": FIXTURE_SEED,
            "optimizer_seed": int(optimizer_seed),
            "dataset_run": str(dataset_run),
            "dataset_manifest_sha256": dataset_hash,
            "source": {
                "checkpoint": str(source_checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "run": str(source_checkpoint.parent.parent),
                "f4a_run": str(f4a_run),
                "f4a_manifest_sha256": file_sha256(
                    f4a_run / "manifests/posterior_tail_diagnostic.json"
                ),
                "step0_reproduction": step0_reproduction,
                "legacy_source_reproduction": legacy_reproduction,
            },
            "data_contract": {
                "motion_count": EXPECTED_MOTIONS,
                "selected_motion_keys": selected_motions,
                "window_transitions": EXPECTED_WINDOW,
                "window_count": len(selected_base),
                "fixture_count": len(fixture_data),
                "fixture_bitmap_sha256": bitmap_hash,
                "selected_windows_sha256": identity_sha256(selected_windows),
                "fixed_velocity_cases_sha256": fixed_case_hash,
                "identity_contract_sha256": file_sha256(identity_contract_path),
            },
            "model_contract": {
                "parameter_count": parameter_count(model),
                "posterior_mean": True,
                "kl_beta": 0.0,
                "dropout": float(resolved_config["model"]["dropout"]),
                "weight_decay": 0.0,
                "f4c_layer_gates_enabled": False,
            },
            "training_contract": {
                "micro_batch": int(training["micro_batch"]),
                "gradient_accumulation": accumulation,
                "effective_batch": int(training["micro_batch"]) * accumulation,
                "completed_optimizer_steps": max_steps,
                "formal_optimizer_steps": FORMAL_STEPS,
                "validation_interval": evaluation_interval,
                "learning_rate": float(training["learning_rate"]),
                "minimum_learning_rate": float(training["minimum_learning_rate"]),
                "warmup_steps": warmup_steps,
                "precision": "FP32",
                "autocast": False,
                "cuda_matmul_allow_tf32": (
                    bool(torch.backends.cuda.matmul.allow_tf32)
                    if torch.cuda.is_available()
                    else None
                ),
                "cudnn_allow_tf32": (
                    bool(torch.backends.cudnn.allow_tf32)
                    if torch.cuda.is_available()
                    else None
                ),
                "gradient_clip_fraction": sum(
                    bool(row["gradient_was_clipped"]) for row in train_records
                ) / len(train_records),
                "mean_train_step_seconds": statistics.fmean(
                    float(row["step_seconds"]) for row in train_records
                ),
                "training_identity_sha256_by_step": training_hashes,
            },
            "evaluations": evaluations,
            "decision_steps": list(EVALUATION_STEPS),
            "best_progression_score": best_score,
            "best_optimizer_step": int(best_evaluation["optimizer_step"]),
            "best_evaluation": best_evaluation,
            "checkpoint_readback": checkpoint_readback,
            "artifacts": {
                "summary": str(output_run / "manifests/posterior_ab_summary.json"),
                "metrics": str(metrics_path),
                "fixed_cases": str(output_run / "manifests/fixed_velocity_cases.json"),
                "identity_contract": str(identity_contract_path),
                "plots": plots,
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_write_json(output_run / "manifests/posterior_ab_summary.json", summary)
        marker_name = SMOKE_MARKER if smoke else EXECUTION_MARKER
        atomic_write_text(
            output_run / f"markers/{marker_name}",
            f"PASS execution_complete=true arm={arm} optimizer_seed={optimizer_seed}\n",
        )
        if quality_pass:
            atomic_write_text(
                output_run / f"markers/{PROGRESSION_MARKER}",
                "PASS last_three_progression_evaluations=true\n",
            )
        elif not smoke:
            atomic_write_text(
                output_run / "markers/cvae.failed",
                "QUALITY_FAIL execution_complete=true progression_last_three=false\n",
            )
        return summary
    finally:
        base.close()


def _evaluation_by_step(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["optimizer_step"]): row
        for row in summary.get("evaluations", [])
    }


def _residual_ratio(candidate: float, baseline: float) -> float:
    candidate = float(candidate)
    baseline = float(baseline)
    if baseline > 0.0:
        return candidate / baseline
    return 1.0 if candidate <= 0.0 else math.inf


def validate_paired_runs(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    if not bool(baseline.get("execution_pass")) or not bool(candidate.get("execution_pass")):
        raise ValueError("comparison requires execution-complete training runs")
    if bool(baseline.get("smoke")) or bool(candidate.get("smoke")):
        raise ValueError("comparison refuses smoke runs")
    if baseline.get("arm") != "A":
        raise ValueError("comparison baseline must be arm A")
    if candidate.get("arm") not in {"B", "C"}:
        raise ValueError("comparison candidate must be arm B or C")
    equal_fields = (
        "fixture_seed",
        "optimizer_seed",
        "dataset_manifest_sha256",
    )
    checks = {name: baseline.get(name) == candidate.get(name) for name in equal_fields}
    for name in (
        "fixture_bitmap_sha256",
        "selected_windows_sha256",
        "fixed_velocity_cases_sha256",
        "identity_contract_sha256",
        "window_count",
        "fixture_count",
    ):
        checks[f"data_contract.{name}"] = (
            baseline.get("data_contract", {}).get(name)
            == candidate.get("data_contract", {}).get(name)
        )
    checks["source.checkpoint_sha256"] = (
        baseline.get("source", {}).get("checkpoint_sha256")
        == candidate.get("source", {}).get("checkpoint_sha256")
    )
    checks["source.f4a_manifest_sha256"] = (
        baseline.get("source", {}).get("f4a_manifest_sha256")
        == candidate.get("source", {}).get("f4a_manifest_sha256")
    )
    if candidate.get("arm") == "B":
        checks["model.parameter_count"] = (
            baseline.get("model_contract", {}).get("parameter_count")
            == candidate.get("model_contract", {}).get("parameter_count")
            == EXPECTED_PARAMETERS
        )
    baseline_hashes = baseline.get("training_contract", {}).get(
        "training_identity_sha256_by_step", {}
    )
    candidate_hashes = candidate.get("training_contract", {}).get(
        "training_identity_sha256_by_step", {}
    )
    checks["training_identity_sha256_by_step"] = baseline_hashes == candidate_hashes
    baseline_steps = _evaluation_by_step(baseline)
    candidate_steps = _evaluation_by_step(candidate)
    checks["decision_steps_present"] = all(
        step in baseline_steps and step in candidate_steps for step in EVALUATION_STEPS
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"A/B paired-run contract mismatch: {failed}")
    return {"passed": True, "checks": checks}


def compare_candidate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    paired = validate_paired_runs(baseline, candidate)
    baseline_steps = _evaluation_by_step(baseline)
    candidate_steps = _evaluation_by_step(candidate)
    source_evaluation = baseline_steps.get(0)
    if source_evaluation is None:
        raise ValueError("baseline is missing the F4D step-0 evaluation")
    source_exact = source_evaluation["exact"]
    per_step: list[dict[str, Any]] = []
    for step in EVALUATION_STEPS:
        base_row = baseline_steps[step]
        candidate_row = candidate_steps[step]
        base_exact = base_row["exact"]
        candidate_exact = candidate_row["exact"]
        base_tail = base_row["tail_global"]
        candidate_tail = candidate_row["tail_global"]
        required_values = [
            base_tail[key]
            for key in (
                "threshold_exceed_fraction", "global_state_rmse", "global_action_rmse"
            )
        ] + [
            candidate_tail[key]
            for key in (
                "threshold_exceed_fraction", "global_state_rmse", "global_action_rmse"
            )
        ] + [
            row[key]
            for row in (base_exact, candidate_exact)
            for key in (
                "continuous_max_abs", "worst_state_rmse", "worst_action_rmse",
                "contact_accuracy",
            )
        ] + [
            base_exact["latent_dependence"]["zero_ratio"],
            candidate_exact["latent_dependence"]["zero_ratio"],
        ]
        if not all(math.isfinite(float(value)) for value in required_values):
            raise ValueError(f"comparison found a non-finite metric at step {step}")
        ratios = {
            "threshold_exceed_fraction": _residual_ratio(
                candidate_tail["threshold_exceed_fraction"],
                base_tail["threshold_exceed_fraction"],
            ),
            "continuous_max_abs": _residual_ratio(
                candidate_exact["continuous_max_abs"],
                base_exact["continuous_max_abs"],
            ),
        }
        protection_values = {}
        for name, key in (
            ("global_state_rmse", "global_state_rmse"),
            ("global_action_rmse", "global_action_rmse"),
        ):
            observed = float(candidate_tail[key])
            protection_values[name] = {
                "observed": observed,
                "baseline_limit": 1.1 * float(base_tail[key]),
                "source_limit": 1.1 * float(source_evaluation["tail_global"][key]),
                "passed": bool(
                    observed <= 1.1 * float(base_tail[key])
                    and observed <= 1.1 * float(source_evaluation["tail_global"][key])
                ),
            }
        for name, key in (
            ("worst_state_rmse", "worst_state_rmse"),
            ("worst_action_rmse", "worst_action_rmse"),
        ):
            observed = float(candidate_exact[key])
            protection_values[name] = {
                "observed": observed,
                "baseline_limit": 1.1 * float(base_exact[key]),
                "source_limit": 1.1 * float(source_exact[key]),
                "passed": bool(
                    observed <= 1.1 * float(base_exact[key])
                    and observed <= 1.1 * float(source_exact[key])
                ),
            }
        contact_pass = math.isclose(
            float(candidate_exact["contact_accuracy"]), 1.0, rel_tol=0.0, abs_tol=0.0
        )
        zero_ratio = float(candidate_exact["latent_dependence"]["zero_ratio"])
        guard_pass = bool(
            contact_pass
            and zero_ratio >= 10.0
            and all(value["passed"] for value in protection_values.values())
        )
        per_step.append({
            "optimizer_step": step,
            "baseline": {
                "threshold_exceed_fraction": float(base_tail["threshold_exceed_fraction"]),
                "continuous_max_abs": float(base_exact["continuous_max_abs"]),
                "global_state_rmse": float(base_tail["global_state_rmse"]),
                "global_action_rmse": float(base_tail["global_action_rmse"]),
                "worst_state_rmse": float(base_exact["worst_state_rmse"]),
                "worst_action_rmse": float(base_exact["worst_action_rmse"]),
                "contact_accuracy": float(base_exact["contact_accuracy"]),
                "zero_latent_ratio": float(base_exact["latent_dependence"]["zero_ratio"]),
                "progression_pass": bool(base_exact["progression_gate"]["passed"]),
            },
            "candidate": {
                "threshold_exceed_fraction": float(candidate_tail["threshold_exceed_fraction"]),
                "continuous_max_abs": float(candidate_exact["continuous_max_abs"]),
                "global_state_rmse": float(candidate_tail["global_state_rmse"]),
                "global_action_rmse": float(candidate_tail["global_action_rmse"]),
                "worst_state_rmse": float(candidate_exact["worst_state_rmse"]),
                "worst_action_rmse": float(candidate_exact["worst_action_rmse"]),
                "contact_accuracy": float(candidate_exact["contact_accuracy"]),
                "zero_latent_ratio": zero_ratio,
                "progression_pass": bool(candidate_exact["progression_gate"]["passed"]),
            },
            "residual_ratios": ratios,
            "candidate_progression_pass": bool(
                candidate_exact["progression_gate"]["passed"]
            ),
            "contact_accuracy": float(candidate_exact["contact_accuracy"]),
            "zero_latent_ratio": zero_ratio,
            "protection": protection_values,
            "guard_pass": guard_pass,
        })
    baseline_exceed = [
        float(baseline_steps[step]["tail_global"]["threshold_exceed_fraction"])
        for step in EVALUATION_STEPS
    ]
    candidate_exceed = [
        float(candidate_steps[step]["tail_global"]["threshold_exceed_fraction"])
        for step in EVALUATION_STEPS
    ]
    baseline_maximum = [
        float(baseline_steps[step]["exact"]["continuous_max_abs"])
        for step in EVALUATION_STEPS
    ]
    candidate_maximum = [
        float(candidate_steps[step]["exact"]["continuous_max_abs"])
        for step in EVALUATION_STEPS
    ]
    residuals = {
        "R_p": _residual_ratio(
            statistics.median(candidate_exceed), statistics.median(baseline_exceed)
        ),
        "R_a": _residual_ratio(
            statistics.median(candidate_maximum), statistics.median(baseline_maximum)
        ),
        "medians": {
            "baseline_threshold_exceed_fraction": statistics.median(baseline_exceed),
            "candidate_threshold_exceed_fraction": statistics.median(candidate_exceed),
            "baseline_continuous_max_abs": statistics.median(baseline_maximum),
            "candidate_continuous_max_abs": statistics.median(candidate_maximum),
        },
        "aggregation": "ratio of three-step medians at optimizer steps 8000, 9000, and 10000",
    }
    residuals["improvements"] = {
        "threshold_exceed_fraction": 1.0 - residuals["R_p"],
        "continuous_max_abs": 1.0 - residuals["R_a"],
    }
    guards_pass = all(row["guard_pass"] for row in per_step)
    progression_pass = all(row["candidate_progression_pass"] for row in per_step)
    return {
        "candidate_arm": candidate["arm"],
        "pairing": paired,
        "per_step": per_step,
        "residuals": residuals,
        "guards_pass": guards_pass,
        "progression_pass_last_three": progression_pass,
        "strong_improvement": bool(
            guards_pass and residuals["R_p"] <= 0.5 and residuals["R_a"] <= 0.5
        ),
        "worth_replicating": bool(
            guards_pass and residuals["R_p"] <= 0.8 and residuals["R_a"] <= 0.8
        ),
    }


def initial_decision(
    baseline: dict[str, Any],
    arm_b: dict[str, Any],
    arm_c: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline_pass = _progression_last_three_pass(list(baseline.get("evaluations", [])))
    comparison_b = compare_candidate(baseline, arm_b)
    if baseline_pass:
        return {
            "decision": "REPLICATE_A_ONLY",
            "reason": "arm A passed progression at steps 8000, 9000, and 10000",
            "implement_f4c": False,
            "next_optimizer_seed": 20260831,
            "replication_arms": ["A"],
            "candidate_comparisons": {"B": comparison_b},
        }
    if comparison_b["guards_pass"] and (
        comparison_b["progression_pass_last_three"] or comparison_b["strong_improvement"]
    ):
        return {
            "decision": "REPLICATE_A_AND_B",
            "reason": "arm B passed progression or reduced both protected residuals by at least 50%",
            "implement_f4c": False,
            "next_optimizer_seed": 20260831,
            "replication_arms": ["A", "B"],
            "candidate_comparisons": {"B": comparison_b},
        }
    if arm_c is None:
        return {
            "decision": "IMPLEMENT_F4C",
            "reason": "arm B did not satisfy the guarded progression/50%-improvement rule",
            "implement_f4c": True,
            "next_optimizer_seed": None,
            "replication_arms": [],
            "candidate_comparisons": {"B": comparison_b},
        }
    comparison_c = compare_candidate(baseline, arm_c)
    eligible = [
        comparison
        for comparison in (comparison_b, comparison_c)
        if comparison["worth_replicating"]
    ]
    if not eligible:
        return {
            "decision": "STOP_LOSS_LATENT_SEED_SEARCH",
            "reason": "neither B nor C achieved guarded 20% improvement",
            "implement_f4c": False,
            "next_optimizer_seed": None,
            "replication_arms": [],
            "candidate_comparisons": {"B": comparison_b, "C": comparison_c},
        }
    winner = min(
        eligible,
        key=lambda comparison: (
            max(comparison["residuals"]["R_p"], comparison["residuals"]["R_a"]),
            0 if comparison["candidate_arm"] == "B" else 1,
        ),
    )
    return {
        "decision": f"REPLICATE_A_AND_{winner['candidate_arm']}",
        "reason": "selected the guarded candidate with the smallest worst residual ratio",
        "implement_f4c": False,
        "next_optimizer_seed": 20260831,
        "replication_arms": ["A", winner["candidate_arm"]],
        "winner": winner["candidate_arm"],
        "candidate_comparisons": {"B": comparison_b, "C": comparison_c},
    }


def replication_decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any] | None,
    initial_comparison: dict[str, Any],
) -> dict[str, Any]:
    initial_decision_name = str(initial_comparison.get("decision", {}).get("decision", ""))
    expected_arms = list(
        initial_comparison.get("decision", {}).get("replication_arms", [])
    )
    if expected_arms not in (["A"], ["A", "B"], ["A", "C"]):
        raise ValueError("initial comparison did not authorize a replication path")
    if int(baseline.get("optimizer_seed", -1)) != 20260831:
        raise ValueError("replication comparison requires optimizer seed 20260831")
    baseline_pass = _progression_last_three_pass(list(baseline.get("evaluations", [])))
    if expected_arms == ["A"]:
        if candidate is not None:
            raise ValueError("A-only replication must not supply a candidate run")
        return {
            "decision": (
                "FOUR_MOTION_PASS_SELECT_A"
                if baseline_pass
                else "STOP_AT_FOUR_MOTIONS_A_NOT_REPRODUCED"
            ),
            "reason": (
                "the second-seed A run passed all three progression evaluations"
                if baseline_pass
                else "the second-seed A run did not reproduce the first-seed quality pass"
            ),
            "initial_decision": initial_decision_name,
            "selected_arm": "A" if baseline_pass else None,
            "four_motion_quality_pass": baseline_pass,
            "strong_improvement_reproduced": False,
            "next_stage": "NEW_32_MOTION_FIXED_RUN" if baseline_pass else "STOP_AND_UPDATE_PLAN",
            "candidate_comparisons": {},
        }
    if candidate is None or candidate.get("arm") != expected_arms[1]:
        raise ValueError("replication candidate does not match the initial comparison")
    current = compare_candidate(baseline, candidate)
    initial_candidate = initial_comparison.get("decision", {}).get(
        "candidate_comparisons", {}
    ).get(expected_arms[1])
    if not isinstance(initial_candidate, dict):
        raise ValueError("initial comparison lacks the selected candidate evidence")
    strong_twice = bool(
        initial_candidate.get("strong_improvement") and current["strong_improvement"]
    )
    if baseline_pass:
        decision = "FOUR_MOTION_PASS_SELECT_A"
        selected = "A"
        reason = "second-seed A passed progression; candidate mechanism is not required"
    elif current["worth_replicating"] and current["progression_pass_last_three"]:
        decision = f"FOUR_MOTION_PASS_SELECT_{candidate['arm']}"
        selected = candidate["arm"]
        reason = "the selected intervention retained at least 20% paired improvement and passed progression"
    elif current["worth_replicating"]:
        decision = "STOP_AT_FOUR_MOTIONS_RELATIVE_GAIN_ONLY"
        selected = None
        reason = "the intervention replicated a protected relative gain but four-motion quality still failed"
    else:
        decision = "STOP_AT_FOUR_MOTIONS_UNSTABLE_INTERVENTION"
        selected = None
        reason = "the intervention did not reproduce a protected 20% gain"
    four_motion_pass = selected is not None
    return {
        "decision": decision,
        "reason": reason,
        "initial_decision": initial_decision_name,
        "selected_arm": selected,
        "four_motion_quality_pass": four_motion_pass,
        "strong_improvement_reproduced": strong_twice,
        "next_stage": "NEW_32_MOTION_FIXED_RUN" if four_motion_pass else "STOP_AND_UPDATE_PLAN",
        "candidate_comparisons": {candidate["arm"]: current},
    }


def _load_training_summary(run: Path) -> tuple[Path, dict[str, Any]]:
    resolved = run.expanduser().resolve()
    path = resolved / "manifests/posterior_ab_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"posterior A/B summary is missing: {path}")
    summary = load_json(path)
    if summary.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported posterior A/B summary: {path}")
    return resolved, summary


def _comparison_svg(result: dict[str, Any]) -> str:
    comparisons = result["decision"]["candidate_comparisons"]
    labels = list(comparisons)
    values = [
        (
            float(comparisons[label]["residuals"]["R_p"]),
            float(comparisons[label]["residuals"]["R_a"]),
        )
        for label in labels
    ]
    width, height = 900, 560
    left, top, plot_width, plot_height = 100.0, 80.0, 740.0, 350.0
    finite = [value for pair in values for value in pair if math.isfinite(value)]
    high = max([1.0, *finite])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="50" y="38" font-size="23" font-weight="700">F4B-v2 paired residual comparison</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>',
        '<text x="24" y="255" transform="rotate(-90 24 255)" text-anchor="middle">Ratio of medians across steps 8k/9k/10k</text>',
    ]
    for threshold, color, label in ((0.5, "#059669", "50% trigger"), (0.8, "#d97706", "20% improvement"), (1.0, "#dc2626", "A baseline")):
        y = top + plot_height * (1.0 - min(threshold / high, 1.0))
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="{color}" stroke-dasharray="7 5"/>')
        parts.append(f'<text x="{left + plot_width - 5}" y="{y - 5:.1f}" text-anchor="end" font-size="11" fill="{color}">{label}</text>')
    group_width = plot_width / max(len(labels), 1)
    for group, (label, pair) in enumerate(zip(labels, values, strict=True)):
        center = left + group_width * (group + 0.5)
        for offset, (name, value, color) in enumerate((
            ("R_p", pair[0], "#2563eb"),
            ("R_a", pair[1], "#7c3aed"),
        )):
            shown = high if not math.isfinite(value) else min(value, high)
            bar_height = plot_height * shown / high
            x = center - 45 + offset * 50
            y = top + plot_height - bar_height
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="38" height="{bar_height:.1f}" fill="{color}"/>')
            value_label = "inf" if not math.isfinite(value) else f"{value:.3f}"
            parts.append(f'<text x="{x + 19:.1f}" y="{max(y - 6, 66):.1f}" text-anchor="middle" font-size="12">{name}={value_label}</text>')
        parts.append(f'<text x="{center:.1f}" y="{top + plot_height + 28}" text-anchor="middle" font-size="14">Arm {html.escape(label)}</text>')
    parts.append(f'<text x="50" y="500" font-size="14">Decision: {html.escape(result["decision"]["decision"])}</text>')
    parts.append('<text x="50" y="525" font-size="11" fill="#4b5563">Ratios are paired by optimizer step; smaller is better. Guards are evaluated separately.</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def run_ab_comparison(
    *,
    output_run: Path,
    run_a: Path,
    run_b: Path | None = None,
    run_c: Path | None = None,
    initial_comparison_run: Path | None = None,
) -> dict[str, Any]:
    output_run = output_run.expanduser().resolve()
    resolved_a, summary_a = _load_training_summary(run_a)
    resolved_b: Path | None = None
    summary_b: dict[str, Any] | None = None
    if run_b is not None:
        resolved_b, summary_b = _load_training_summary(run_b)
    resolved_c: Path | None = None
    summary_c: dict[str, Any] | None = None
    if run_c is not None:
        resolved_c, summary_c = _load_training_summary(run_c)
    for protected in (resolved_a, resolved_b, resolved_c):
        if protected is not None and (
            output_run == protected or output_run.is_relative_to(protected)
        ):
            raise ValueError("comparison output must be isolated from every input run")
    comparison_phase = "initial"
    initial_input: dict[str, Any] | None = None
    if initial_comparison_run is None:
        if summary_b is None:
            raise ValueError("initial comparison requires arm B")
        if int(summary_a.get("optimizer_seed", -1)) != 20260830:
            raise ValueError("initial comparison requires optimizer seed 20260830")
        decision = initial_decision(summary_a, summary_b, summary_c)
    else:
        comparison_phase = "replication"
        resolved_initial = initial_comparison_run.expanduser().resolve()
        if output_run == resolved_initial or output_run.is_relative_to(resolved_initial):
            raise ValueError("comparison output must be isolated from the initial comparison")
        initial_path = resolved_initial / "manifests/posterior_ab_comparison.json"
        if not (resolved_initial / f"markers/{COMPARISON_MARKER}").is_file():
            raise FileNotFoundError("replication requires the initial comparison marker")
        initial_manifest = load_json(initial_path)
        if initial_manifest.get("format_version") != COMPARISON_FORMAT:
            raise ValueError("unsupported initial comparison manifest")
        expected_arms = list(initial_manifest.get("decision", {}).get("replication_arms", []))
        if expected_arms == ["A"]:
            if summary_b is not None or summary_c is not None:
                raise ValueError("initial decision authorized only arm A replication")
            candidate = None
        elif expected_arms == ["A", "B"]:
            if summary_b is None or summary_c is not None:
                raise ValueError("initial decision requires A/B replication")
            candidate = summary_b
        elif expected_arms == ["A", "C"]:
            if summary_c is None or summary_b is not None:
                raise ValueError("initial decision requires A/C replication")
            candidate = summary_c
        else:
            raise ValueError("initial comparison did not authorize replication")
        decision = replication_decision(summary_a, candidate, initial_manifest)
        initial_input = {
            "run": str(resolved_initial),
            "manifest_sha256": file_sha256(initial_path),
        }
    inputs = {
        "A": {
            "run": str(resolved_a),
            "summary_sha256": file_sha256(
                resolved_a / "manifests/posterior_ab_summary.json"
            ),
        },
    }
    if resolved_b is not None:
        inputs["B"] = {
            "run": str(resolved_b),
            "summary_sha256": file_sha256(
                resolved_b / "manifests/posterior_ab_summary.json"
            ),
        }
    if resolved_c is not None:
        inputs["C"] = {
            "run": str(resolved_c),
            "summary_sha256": file_sha256(
                resolved_c / "manifests/posterior_ab_summary.json"
            ),
        }
    result = {
        "format_version": COMPARISON_FORMAT,
        "scope": (
            "paired F4B-v2 comparison on the same four-motion fixed fixtures; "
            "no generalization or conditional-prior claim"
        ),
        "execution_pass": True,
        "comparison_phase": comparison_phase,
        "decision_steps": list(EVALUATION_STEPS),
        "residual_definition": (
            "R_p and R_a are candidate/A ratios of the three-step medians at "
            "optimizer steps 8000, 9000, and 10000"
        ),
        "inputs": inputs,
        "initial_comparison": initial_input,
        "decision": decision,
        "artifacts": {
            "manifest": str(output_run / "manifests/posterior_ab_comparison.json"),
            "plot": str(output_run / "plots/posterior_ab_comparison.svg"),
        },
    }
    atomic_write_json(
        output_run / "manifests/posterior_ab_comparison.json", _json_safe(result)
    )
    atomic_write_text(
        output_run / "plots/posterior_ab_comparison.svg", _comparison_svg(result)
    )
    atomic_write_text(
        output_run / f"markers/{COMPARISON_MARKER}",
        f"PASS decision={decision['decision']}\n",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or compare F4B-v2 posterior A/B arms")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--dataset-run", type=Path, required=True)
    train.add_argument("--source-checkpoint", type=Path, required=True)
    train.add_argument("--f4a-run", type=Path, required=True)
    train.add_argument("--output-run", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--arm", required=True)
    train.add_argument("--optimizer-seed", type=int, required=True)
    train.add_argument("--smoke", action="store_true")
    compare = subparsers.add_parser("compare")
    compare.add_argument("--output-run", type=Path, required=True)
    compare.add_argument("--run-a", type=Path, required=True)
    compare.add_argument("--run-b", type=Path)
    compare.add_argument("--run-c", type=Path)
    compare.add_argument("--initial-comparison-run", type=Path)
    return parser.parse_args()


def _write_failure_manifest(output_run: Path, error: BaseException) -> None:
    output_run = output_run.expanduser().resolve()
    metrics_path = output_run / "logs/metrics.jsonl"
    last_record: dict[str, Any] | None = None
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last_record = json.loads(line)
    atomic_write_json(output_run / "manifests/posterior_ab_failure.json", {
        "format_version": "sonic_posterior_ab_failure_v1",
        "execution_pass": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "last_record": last_record,
        "metrics_path": str(metrics_path),
    })


def main() -> int:
    args = parse_args()
    try:
        if args.command == "train":
            result = run_ab_experiment(
                dataset_run=args.dataset_run,
                source_checkpoint=args.source_checkpoint,
                f4a_run=args.f4a_run,
                output_run=args.output_run,
                config=load_json(args.config),
                arm=args.arm,
                optimizer_seed=args.optimizer_seed,
                smoke=args.smoke,
            )
            print("Posterior F4B-v2: PASS (execution complete)")
            print(json.dumps({
                "output_run": str(args.output_run.expanduser().resolve()),
                "arm": result["arm"],
                "smoke": result["smoke"],
                "quality_pass": result["quality_pass"],
                "completed_optimizer_steps": result["training_contract"]["completed_optimizer_steps"],
                "best_progression_score": result["best_progression_score"],
            }, ensure_ascii=False, indent=2))
            return 0
        result = run_ab_comparison(
            output_run=args.output_run,
            run_a=args.run_a,
            run_b=args.run_b,
            run_c=args.run_c,
            initial_comparison_run=args.initial_comparison_run,
        )
        print("Posterior F4B-v2 comparison: PASS (execution complete)")
        print(json.dumps({
            "output_run": str(args.output_run.expanduser().resolve()),
            "decision": result["decision"]["decision"],
            "reason": result["decision"]["reason"],
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        _write_failure_manifest(args.output_run, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
