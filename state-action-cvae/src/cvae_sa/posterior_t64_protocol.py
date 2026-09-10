from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch.nn import functional as F

from .posterior_capacity import _device_batch, _stable_seed
from .util import atomic_write_json, atomic_write_text, canonical_json_bytes


PHYSICAL_MASK_NAMES = (
    "state_gap_4",
    "state_gap_16",
    "state_rollout",
    "action_gap_4",
    "action_gap_16",
    "full_action",
    "joint_gap_2",
    "joint_gap_8",
)
DIAGNOSTIC_MASK_NAMES = ("full_state", "full_both")


def identity(batch: dict[str, Any], index: int) -> tuple[Any, ...]:
    window_index = batch.get("window_index")
    if window_index is None:
        window_index = torch.arange(len(batch["motion_key"]))
    return (
        str(batch["motion_key"][index]),
        int(batch["variant_id"][index]),
        int(batch["window_start"][index]),
        int(window_index[index]),
    )


def _choose_start(
    valid_transitions: int,
    length: int,
    generator: torch.Generator,
) -> tuple[int, int]:
    length = max(1, min(int(length), int(valid_transitions)))
    start = int(torch.randint(0, valid_transitions - length + 1, (), generator=generator))
    return start, length


def _random_length(
    low: int, high: int, valid_transitions: int, generator: torch.Generator
) -> int:
    upper = max(1, min(int(high), int(valid_transitions)))
    lower = max(1, min(int(low), upper))
    return int(torch.randint(lower, upper + 1, (), generator=generator))


def make_physical_masks(
    batch: dict[str, Any],
    seed: int,
    *,
    held_out: bool = False,
    dynamic_step: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Construct fixed or independently seeded physically structured Masks."""
    valid_state = batch["valid_state"].bool()
    valid_action = batch["valid_action"].bool()
    state_mask = torch.zeros_like(batch["physical_state"], dtype=torch.bool)
    action_mask = torch.zeros_like(batch["action"], dtype=torch.bool)
    slots = batch.get("mask_slot")
    if slots is None:
        slots = torch.tensor(
            [
                _stable_seed(seed, "dynamic-slot", dynamic_step, *identity(batch, index))
                % len(PHYSICAL_MASK_NAMES)
                for index in range(len(batch["motion_key"]))
            ],
            dtype=torch.long,
        )
    names: list[str] = []
    for index, raw_slot in enumerate(slots.tolist()):
        slot = int(raw_slot) % len(PHYSICAL_MASK_NAMES)
        name = PHYSICAL_MASK_NAMES[slot]
        names.append(name)
        valid_transitions = int(valid_action[index].sum())
        if valid_transitions <= 0:
            raise ValueError("T64 physical Mask window has no valid transitions")
        namespace = "dynamic" if dynamic_step is not None else ("heldout" if held_out else "fixed")
        generator = torch.Generator().manual_seed(
            _stable_seed(seed, namespace, dynamic_step, *identity(batch, index), int(raw_slot))
        )
        if slot in {0, 1}:
            length = (4, 16)[slot]
            if held_out or dynamic_step is not None:
                length = _random_length((2, 9)[slot], (8, 24)[slot], valid_transitions, generator)
            start, length = _choose_start(valid_transitions, length, generator)
            # S_start and S_(start+length) are the two visible physical boundaries.
            state_mask[index, start + 1 : start + length] = True
        elif slot == 2:
            start = 0
            if held_out or dynamic_step is not None:
                start = int(torch.randint(0, max(valid_transitions // 4, 1), (), generator=generator))
            state_mask[index, start + 1 : valid_transitions + 1] = True
        elif slot in {3, 4}:
            length = (4, 16)[slot - 3]
            if held_out or dynamic_step is not None:
                length = _random_length((2, 9)[slot - 3], (8, 24)[slot - 3], valid_transitions, generator)
            start, length = _choose_start(valid_transitions, length, generator)
            action_mask[index, start : start + length] = True
        elif slot == 5:
            action_mask[index, :valid_transitions] = True
        else:
            length = (2, 8)[slot - 6]
            if held_out or dynamic_step is not None:
                length = _random_length((1, 5)[slot - 6], (4, 12)[slot - 6], valid_transitions, generator)
            start, length = _choose_start(valid_transitions, length, generator)
            action_mask[index, start : start + length] = True
            state_mask[index, start + 1 : start + length] = True
        state_mask[index] &= valid_state[index, :, None]
        action_mask[index] &= valid_action[index, :, None]
        if not bool(state_mask[index].any()) and not bool(action_mask[index].any()):
            # A one-transition joint gap has no internal State but still masks Action.
            action_mask[index, 0] = True
    return state_mask, action_mask, names


def make_autoencode_masks(
    batch: dict[str, Any], seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    del seed
    state_mask = batch["valid_state"].bool()[..., None].expand_as(
        batch["physical_state"]
    ).clone()
    action_mask = batch["valid_action"].bool()[..., None].expand_as(batch["action"]).clone()
    return state_mask, action_mask, ["full_both"] * len(batch["motion_key"])


def make_diagnostic_masks(
    batch: dict[str, Any], name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if name not in DIAGNOSTIC_MASK_NAMES:
        raise ValueError(f"unknown T64 diagnostic Mask {name!r}")
    state = batch["valid_state"].bool()[..., None].expand_as(batch["physical_state"]).clone()
    action = torch.zeros_like(batch["action"], dtype=torch.bool)
    if name == "full_both":
        action = batch["valid_action"].bool()[..., None].expand_as(batch["action"]).clone()
    return state, action


def window_identity_rows(dataset: Any, indices: Iterable[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window_index, source_index in enumerate(indices):
        ref = dataset.refs[int(source_index)]
        episode = dataset.episodes[ref.episode_index]
        if ref.fixed_start is None:
            raise ValueError("T64 capacity experiments require fixed window starts")
        rows.append({
            "window_index": window_index,
            "source_window_index": int(source_index),
            "motion_key": str(episode["motion_key"]),
            "variant_id": int(episode["variant_id"]),
            "episode": str(episode["episode"]),
            "episode_ref": f"{episode['source_run']}::{episode['episode']}",
            "window_start": int(ref.fixed_start),
        })
    return rows


def rows_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def mask_bank_sha256(
    loader: Iterable[dict[str, Any]],
    maker: Callable[[dict[str, Any]], tuple[torch.Tensor, torch.Tensor, list[str]]],
) -> str:
    digest = hashlib.sha256()
    for batch in loader:
        state_mask, action_mask, names = maker(batch)
        digest.update(state_mask.numpy().tobytes(order="C"))
        digest.update(action_mask.numpy().tobytes(order="C"))
        digest.update(canonical_json_bytes(names))
    return digest.hexdigest()


def reconstruction_loss(
    output: Any,
    batch: dict[str, torch.Tensor],
    state_mask: torch.Tensor,
    action_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    terms: list[torch.Tensor] = []
    continuous_mask = state_mask[..., :68]
    if bool(continuous_mask.any()):
        state = torch.square(
            output.physical_state[..., :68] - batch["physical_state"][..., :68]
        ).masked_select(continuous_mask).mean()
        terms.append(state)
    else:
        state = output.physical_state.sum() * 0.0
    if bool(action_mask.any()):
        action = torch.square(output.action - batch["action"]).masked_select(action_mask).mean()
        terms.append(action)
    else:
        action = output.action.sum() * 0.0
    contact_mask = state_mask[..., 68:70]
    if bool(contact_mask.any()):
        contact = F.binary_cross_entropy_with_logits(
            output.state_contact_logits,
            batch["physical_state"][..., 68:70],
            reduction="none",
        ).masked_select(contact_mask).mean()
        terms.append(contact)
    else:
        contact = output.state_contact_logits.sum() * 0.0
    if not terms:
        raise ValueError("T64 reconstruction batch contains no targets")
    return {"total": torch.stack(terms).mean(), "state": state, "action": action, "contact": contact}


def _gate(
    metrics: dict[str, Any], thresholds: dict[str, float], kind: str, latent: bool
) -> dict[str, Any]:
    if kind == "fit":
        ratios = {
            "global_state_rmse": metrics["global_state_rmse"] / thresholds["global_state_rmse"],
            "global_action_rmse": metrics["global_action_rmse"] / thresholds["global_action_rmse"],
            "worst_mask_state_rmse": metrics["worst_mask_state_rmse"] / thresholds["worst_mask_state_rmse"],
            "worst_mask_action_rmse": metrics["worst_mask_action_rmse"] / thresholds["worst_mask_action_rmse"],
            "continuous_p99_abs": metrics["continuous_p99_abs"] / thresholds["continuous_p99_abs"],
            "contact_accuracy": 1.0 if metrics["contact_accuracy"] == 1.0 else 2.0 - metrics["contact_accuracy"],
        }
    else:
        ratios = {
            "worst_state_rmse": metrics["worst_state_rmse"] / thresholds["worst_state_rmse"],
            "worst_action_rmse": metrics["worst_action_rmse"] / thresholds["worst_action_rmse"],
            "continuous_max_abs": metrics["continuous_max_abs"] / thresholds["continuous_max_abs"],
            "contact_accuracy": 1.0 if metrics["contact_accuracy"] == 1.0 else 2.0 - metrics["contact_accuracy"],
        }
    if latent:
        dependence = metrics["latent_dependence"]["main_ratios"]
        for name in ("zero", "cross_window", "cross_motion"):
            ratios[f"{name}_latent_dependence"] = thresholds["latent_ratio"] / max(
                float(dependence[name]), 1e-12
            )
    score = max(ratios.values())
    return {
        "passed": bool(math.isfinite(score) and score <= 1.0),
        "score": score,
        "thresholds": thresholds,
        "threshold_ratios": ratios,
    }


def _donor_maps(identities: list[dict[str, Any]]) -> dict[str, list[int]]:
    count = len(identities)
    if count < 2:
        return {"cross_window": [0] * count, "cross_motion": [0] * count}
    cross_window = [(index + 1) % count for index in range(count)]
    by_motion: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(identities):
        by_motion[row["motion_key"]].append(index)
    motions = sorted(by_motion)
    if len(motions) < 2:
        cross_motion = cross_window
    else:
        next_motion = {motion: motions[(index + 1) % len(motions)] for index, motion in enumerate(motions)}
        cross_motion = []
        for index, row in enumerate(identities):
            donors = by_motion[next_motion[row["motion_key"]]]
            cross_motion.append(donors[index % len(donors)])
    return {"cross_window": cross_window, "cross_motion": cross_motion}


@torch.no_grad()
def evaluate_diagnostic_masks(
    model: torch.nn.Module,
    base_loader: Iterable[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Report full-State/full-both reconstruction without using either in B/R gates."""
    model.eval()
    result: dict[str, Any] = {}
    for mask_name in DIAGNOSTIC_MASK_NAMES:
        sums = {"state": 0.0, "action": 0.0}
        counts = {"state": 0, "action": 0}
        maximum = 0.0
        contact_correct = contact_count = 0
        for cpu_batch in base_loader:
            state_mask, action_mask = make_diagnostic_masks(cpu_batch, mask_name)
            batch = _device_batch(cpu_batch, device)
            state_mask = state_mask.to(device)
            action_mask = action_mask.to(device)
            output = model(batch, state_mask, action_mask)
            state_error = output.physical_state[..., :68] - batch["physical_state"][..., :68]
            action_error = output.action - batch["action"]
            state_values = state_error.masked_select(state_mask[..., :68])
            action_values = action_error.masked_select(action_mask)
            for name, values in (("state", state_values), ("action", action_values)):
                sums[name] += float(torch.square(values).sum().cpu())
                counts[name] += int(values.numel())
                if values.numel():
                    maximum = max(maximum, float(values.abs().max().cpu()))
            contact_mask = state_mask[..., 68:70]
            predicted = output.state_contact_logits.sigmoid() >= 0.5
            target = batch["physical_state"][..., 68:70] >= 0.5
            contact_correct += int((predicted == target).masked_select(contact_mask).sum().cpu())
            contact_count += int(contact_mask.sum().cpu())
        result[mask_name] = {
            "global_state_rmse": math.sqrt(sums["state"] / counts["state"]) if counts["state"] else 0.0,
            "global_action_rmse": math.sqrt(sums["action"] / counts["action"]) if counts["action"] else 0.0,
            "continuous_max_abs": maximum,
            "contact_accuracy": contact_correct / contact_count if contact_count else 1.0,
            "counts": counts,
            "gate_role": "posterior diagnostic only; excluded from physical fit gate",
        }
    return result


@torch.no_grad()
def evaluate_latent_dependence(
    model: torch.nn.Module,
    base_loader: Iterable[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    identities: list[dict[str, Any]] = []
    globals_: list[torch.Tensor] = []
    locals_: list[torch.Tensor] = []
    for cpu_batch in base_loader:
        state_mask, action_mask = make_diagnostic_masks(cpu_batch, "full_both")
        batch = _device_batch(cpu_batch, device)
        global_latent, local_latents = model.encode_posterior(
            batch, state_mask.to(device), action_mask.to(device)
        )
        globals_.append(global_latent.cpu())
        locals_.append(local_latents.cpu())
        for index in range(len(cpu_batch["motion_key"])):
            identities.append({
                "motion_key": str(cpu_batch["motion_key"][index]),
                "variant_id": int(cpu_batch["variant_id"][index]),
                "window_start": int(cpu_batch["window_start"][index]),
                "window_index": int(cpu_batch["window_index"][index]),
            })
    global_all = torch.cat(globals_)
    local_all = torch.cat(locals_)
    maps = _donor_maps(identities)
    names = (
        "correct", "zero", "cross_window", "cross_motion",
        "cross_window_global", "cross_window_local",
        "cross_motion_global", "cross_motion_local",
    )
    sums = {name: [0.0, 0] for name in names}
    offset = 0
    for cpu_batch in base_loader:
        size = len(cpu_batch["motion_key"])
        state_mask, action_mask = make_diagnostic_masks(cpu_batch, "full_both")
        batch = _device_batch(cpu_batch, device)
        state_mask = state_mask.to(device)
        action_mask = action_mask.to(device)
        own = list(range(offset, offset + size))
        cw = [maps["cross_window"][index] for index in own]
        cm = [maps["cross_motion"][index] for index in own]
        selections = {
            "correct": (global_all[own], local_all[own]),
            "zero": (torch.zeros_like(global_all[own]), torch.zeros_like(local_all[own])),
            "cross_window": (global_all[cw], local_all[cw]),
            "cross_motion": (global_all[cm], local_all[cm]),
            "cross_window_global": (global_all[cw], local_all[own]),
            "cross_window_local": (global_all[own], local_all[cw]),
            "cross_motion_global": (global_all[cm], local_all[own]),
            "cross_motion_local": (global_all[own], local_all[cm]),
        }
        for name, (global_latent, local_latents) in selections.items():
            output = model.decode_from_hierarchical_latent(
                batch,
                state_mask,
                action_mask,
                global_latent.to(device),
                local_latents.to(device),
            )
            state_error = torch.square(
                output.physical_state[..., :68] - batch["physical_state"][..., :68]
            ).masked_select(state_mask[..., :68])
            action_error = torch.square(output.action - batch["action"]).masked_select(action_mask)
            sums[name][0] += float(state_error.sum().cpu()) + float(action_error.sum().cpu())
            sums[name][1] += int(state_error.numel() + action_error.numel())
        offset += size
    rmse = {name: math.sqrt(total / count) for name, (total, count) in sums.items()}
    correct = max(rmse["correct"], 1e-12)
    ratios = {name: value / correct for name, value in rmse.items() if name != "correct"}
    return {
        "window_count": len(identities),
        "identity_sha256": rows_sha256(identities),
        "rmse_excluding_contact": rmse,
        "ratios_to_correct": ratios,
        "main_ratios": {name: ratios[name] for name in ("zero", "cross_window", "cross_motion")},
        "donor_maps": maps,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    validation_loader: Iterable[dict[str, Any]],
    base_loader: Iterable[dict[str, Any]],
    device: torch.device,
    mask_maker: Callable[[dict[str, Any]], tuple[torch.Tensor, torch.Tensor, list[str]]],
    *,
    fit_thresholds: dict[str, float],
    strict_thresholds: dict[str, float],
    exact_thresholds: dict[str, float],
    state_std: torch.Tensor,
    action_std: torch.Tensor,
    latent_diagnostics: bool,
    report_full_sequence: bool = False,
) -> dict[str, Any]:
    model.eval()
    totals = {"state": 0.0, "action": 0.0, "contact": 0.0}
    counts = {"state": 0, "action": 0, "contact": 0}
    cases: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"windows": 0, "worst_state_rmse": 0.0, "worst_action_rmse": 0.0,
                 "continuous_max_abs": 0.0, "contact_correct": 0, "contact_count": 0}
    )
    all_abs: list[torch.Tensor] = []
    feature_sse = torch.zeros(97, dtype=torch.float64)
    feature_physical_sse = torch.zeros(97, dtype=torch.float64)
    feature_count = torch.zeros(97, dtype=torch.long)
    feature_max = torch.zeros(97, dtype=torch.float64)
    feature_physical_max = torch.zeros(97, dtype=torch.float64)
    full_totals = {"state": 0.0, "action": 0.0, "contact": 0.0}
    full_counts = {"state": 0, "action": 0, "contact": 0}
    full_abs: list[torch.Tensor] = []
    full_contact_correct = 0
    for cpu_batch in validation_loader:
        state_mask, action_mask, names = mask_maker(cpu_batch)
        batch = _device_batch(cpu_batch, device)
        state_mask = state_mask.to(device)
        action_mask = action_mask.to(device)
        output = model(batch, state_mask, action_mask)
        state_error = output.physical_state[..., :68] - batch["physical_state"][..., :68]
        action_error = output.action - batch["action"]
        state_values = torch.square(state_error).masked_select(state_mask[..., :68])
        action_values = torch.square(action_error).masked_select(action_mask)
        contact_values = F.binary_cross_entropy_with_logits(
            output.state_contact_logits,
            batch["physical_state"][..., 68:70], reduction="none"
        ).masked_select(state_mask[..., 68:70])
        for key, values in (("state", state_values), ("action", action_values), ("contact", contact_values)):
            totals[key] += float(values.sum().cpu())
            counts[key] += int(values.numel())
        continuous_abs = torch.cat((
            state_error.abs().masked_select(state_mask[..., :68]),
            action_error.abs().masked_select(action_mask),
        ))
        if continuous_abs.numel():
            all_abs.append(continuous_abs.cpu())
        continuous_mask = state_mask[..., :68]
        state_feature_sse = (
            torch.square(state_error) * continuous_mask.to(state_error.dtype)
        ).sum(dim=(0, 1)).double().cpu()
        action_feature_sse = (
            torch.square(action_error) * action_mask.to(action_error.dtype)
        ).sum(dim=(0, 1)).double().cpu()
        batch_feature_sse = torch.cat((state_feature_sse, action_feature_sse))
        batch_feature_count = torch.cat((
            continuous_mask.sum(dim=(0, 1)).cpu(),
            action_mask.sum(dim=(0, 1)).cpu(),
        ))
        batch_feature_max = torch.cat((
            state_error.abs().masked_fill(~continuous_mask, 0.0).amax(dim=(0, 1)).double().cpu(),
            action_error.abs().masked_fill(~action_mask, 0.0).amax(dim=(0, 1)).double().cpu(),
        ))
        scales = torch.cat((state_std[:68], action_std)).double()
        feature_sse += batch_feature_sse
        feature_physical_sse += batch_feature_sse * torch.square(scales)
        feature_count += batch_feature_count
        feature_max = torch.maximum(feature_max, batch_feature_max)
        feature_physical_max = torch.maximum(feature_physical_max, batch_feature_max * scales)
        predictions = output.state_contact_logits.sigmoid() >= 0.5
        targets = batch["physical_state"][..., 68:70] >= 0.5
        if report_full_sequence:
            full_state_mask = batch["valid_state"].bool()[..., None].expand_as(
                batch["physical_state"]
            )
            full_action_mask = batch["valid_action"].bool()[..., None].expand_as(
                batch["action"]
            )
            full_values = {
                "state": torch.square(state_error).masked_select(
                    full_state_mask[..., :68]
                ),
                "action": torch.square(action_error).masked_select(
                    full_action_mask
                ),
                "contact": F.binary_cross_entropy_with_logits(
                    output.state_contact_logits,
                    batch["physical_state"][..., 68:70],
                    reduction="none",
                ).masked_select(full_state_mask[..., 68:70]),
            }
            for key, values in full_values.items():
                full_totals[key] += float(values.sum().cpu())
                full_counts[key] += int(values.numel())
            full_abs.extend((
                state_error.abs().masked_select(full_state_mask[..., :68]).cpu(),
                action_error.abs().masked_select(full_action_mask).cpu(),
            ))
            full_contact_correct += int(
                (predictions == targets)
                .masked_select(full_state_mask[..., 68:70])
                .sum()
                .cpu()
            )
        for index, name in enumerate(names):
            case = cases[name]
            case["windows"] += 1
            sm = state_mask[index, ..., :68]
            am = action_mask[index]
            if bool(sm.any()):
                rmse = float(torch.sqrt(torch.square(state_error[index]).masked_select(sm).mean()).cpu())
                case["worst_state_rmse"] = max(case["worst_state_rmse"], rmse)
            if bool(am.any()):
                rmse = float(torch.sqrt(torch.square(action_error[index]).masked_select(am).mean()).cpu())
                case["worst_action_rmse"] = max(case["worst_action_rmse"], rmse)
            sample_abs = torch.cat((state_error[index].abs().masked_select(sm), action_error[index].abs().masked_select(am)))
            if sample_abs.numel():
                case["continuous_max_abs"] = max(case["continuous_max_abs"], float(sample_abs.max().cpu()))
            cm = state_mask[index, ..., 68:70]
            case["contact_correct"] += int((predictions[index] == targets[index]).masked_select(cm).sum().cpu())
            case["contact_count"] += int(cm.sum().cpu())
    if counts["state"] <= 0 or counts["action"] <= 0:
        raise ValueError("T64 full evaluation requires both State and Action targets")
    abs_values = torch.cat(all_abs)
    for case in cases.values():
        case["contact_accuracy"] = (
            case["contact_correct"] / case["contact_count"] if case["contact_count"] else 1.0
        )
    state_rmse = math.sqrt(totals["state"] / counts["state"])
    action_rmse = math.sqrt(totals["action"] / counts["action"])
    worst_state = max(float(case["worst_state_rmse"]) for case in cases.values())
    worst_action = max(float(case["worst_action_rmse"]) for case in cases.values())
    contact_correct = sum(int(case["contact_correct"]) for case in cases.values())
    contact_count = sum(int(case["contact_count"]) for case in cases.values())
    feature_rows = []
    for index in range(97):
        count = int(feature_count[index])
        feature_rows.append({
            "index": index,
            "domain": "state" if index < 68 else "action",
            "feature_index": index if index < 68 else index - 68,
            "count": count,
            "normalized_rmse": math.sqrt(float(feature_sse[index]) / count) if count else 0.0,
            "physical_rmse": math.sqrt(float(feature_physical_sse[index]) / count) if count else 0.0,
            "normalized_max_abs": float(feature_max[index]),
            "physical_max_abs": float(feature_physical_max[index]),
        })
    metrics: dict[str, Any] = {
        "global_state_rmse": state_rmse,
        "global_action_rmse": action_rmse,
        "worst_mask_state_rmse": worst_state,
        "worst_mask_action_rmse": worst_action,
        "worst_state_rmse": worst_state,
        "worst_action_rmse": worst_action,
        "continuous_p99_abs": float(torch.quantile(abs_values.float(), 0.99)),
        "continuous_max_abs": float(abs_values.max()),
        "contact_accuracy": contact_correct / contact_count if contact_count else 1.0,
        "reconstruction_loss": {
            "state": totals["state"] / counts["state"],
            "action": totals["action"] / counts["action"],
            "contact": totals["contact"] / counts["contact"] if counts["contact"] else 0.0,
            "total": sum(
                totals[key] / counts[key] for key in totals if counts[key]
            ) / sum(1 for key in totals if counts[key]),
            "counts": counts,
            "aggregation": "global masked-element means; equal mean of present components",
        },
        "cases": dict(cases),
        "feature_errors": feature_rows,
    }
    if report_full_sequence:
        full_absolute = torch.cat(full_abs)
        full_means = {
            key: full_totals[key] / full_counts[key] for key in full_totals
        }
        metrics["full_sequence_reconstruction"] = {
            "global_state_rmse": math.sqrt(full_means["state"]),
            "global_action_rmse": math.sqrt(full_means["action"]),
            "continuous_p99_abs": float(
                torch.quantile(full_absolute.float(), 0.99)
            ),
            "continuous_max_abs": float(full_absolute.max()),
            "contact_accuracy": full_contact_correct / full_counts["contact"],
            "reconstruction_loss": {
                **full_means,
                "total": sum(full_means.values()) / len(full_means),
                "counts": full_counts,
                "aggregation": "all valid output elements for every evaluated Mask query",
            },
            "gate_role": "reported alongside masked-target gates; does not replace them",
        }
    if latent_diagnostics:
        metrics["latent_dependence"] = evaluate_latent_dependence(model, base_loader, device)
    metrics["fit_gate"] = _gate(metrics, fit_thresholds, "fit", latent_diagnostics)
    metrics["strict_memory_gate"] = _gate(metrics, strict_thresholds, "strict", latent_diagnostics)
    metrics["legacy_exact_gate"] = _gate(metrics, exact_thresholds, "exact", latent_diagnostics)
    return metrics


def _svg(
    path: Path,
    title: str,
    subtitle: str,
    series: list[tuple[str, list[tuple[float, float]], str]],
    *,
    x_label: str = "Optimizer step",
) -> None:
    all_points = [point for _, points, _ in series for point in points]
    if all_points:
        xmin = min(point[0] for point in all_points)
        xmax = max(point[0] for point in all_points)
        logs = [math.log10(max(point[1], 1e-12)) for point in all_points]
        ymin = math.floor(min(logs))
        ymax = math.ceil(max(logs))
    else:
        xmin, xmax, ymin, ymax = 0.0, 1.0, -12, 0
    if xmin == xmax:
        xmax += 1.0
    if ymin == ymax:
        ymax += 1

    def map_point(point: tuple[float, float]) -> tuple[float, float]:
        x, value = point
        log_value = math.log10(max(value, 1e-12))
        return (
            60 + (x - xmin) / (xmax - xmin) * 910,
            55 + (ymax - log_value) / (ymax - ymin) * 310,
        )

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="420" viewBox="0 0 1000 420">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="24" font-size="18">{title}</text>',
        f'<text x="20" y="44" font-size="11" fill="#555">{subtitle}</text>',
        f'<text x="455" y="410" font-size="12">{x_label}</text>',
        '<text x="15" y="250" font-size="12" transform="rotate(-90 15 250)">Value (log10 scale)</text>',
        '<line x1="60" y1="365" x2="970" y2="365" stroke="#333"/>',
        '<line x1="60" y1="55" x2="60" y2="365" stroke="#333"/>',
    ]
    tick_count = min(9, max(2, ymax - ymin + 1))
    tick_powers = sorted(set(
        round(ymin + index * (ymax - ymin) / (tick_count - 1))
        for index in range(tick_count)
    ))
    for power in tick_powers:
        y = 55 + (ymax - power) / (ymax - ymin) * 310
        lines.append(f'<line x1="60" y1="{y:.2f}" x2="970" y2="{y:.2f}" stroke="#eeeeee"/>')
        lines.append(f'<text x="24" y="{y + 4:.2f}" font-size="9">10^{power}</text>')
    for index in range(5):
        value = xmin + index * (xmax - xmin) / 4
        x = 60 + index * 910 / 4
        lines.append(f'<text x="{x - 10:.2f}" y="380" font-size="9">{value:.0f}</text>')
    for index, (name, points, color) in enumerate(series):
        mapped = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(map_point, points))
        if mapped:
            dash = ' stroke-dasharray="6 4"' if "threshold" in name.lower() else ""
            lines.append(f'<polyline points="{mapped}" fill="none" stroke="{color}" stroke-width="1.5"{dash}/>')
        lines.append(f'<text x="{680 + (index % 2) * 150}" y="{25 + (index // 2) * 15}" font-size="10" fill="{color}">{name}</text>')
    lines.append('<text x="70" y="385" font-size="10">Zeros clipped to 10^-12; full JSONL is not downsampled.</text>')
    lines.append('</svg>')
    atomic_write_text(path, "\n".join(lines) + "\n")


def render_plots(output_run: Path, records: list[dict[str, Any]], best: dict[str, Any] | None) -> dict[str, str]:
    train = [row for row in records if row.get("phase") == "train"]
    evaluations = [row for row in records if row.get("phase") == "evaluation"]
    if len(train) > 2000:
        selected_indices = {
            round(index * (len(train) - 1) / 1999) for index in range(2000)
        }
        train_plot = [row for index, row in enumerate(train) if index in selected_indices]
    else:
        train_plot = train
    ema_points: list[tuple[float, float]] = []
    running: float | None = None
    for row in train:
        value = float(row["reconstruction"]["total"])
        running = value if running is None else 0.05 * value + 0.95 * running
        ema_points.append((row["optimizer_step"], running))
    plots = output_run / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    paths = {
        "training_curves": plots / "training_curves.svg",
        "gate_curves": plots / "gate_curves.svg",
        "mask_breakdown": plots / "mask_breakdown.svg",
        "feature_error": plots / "feature_error.svg",
        "latent_dependence": plots / "latent_dependence.svg",
    }
    _svg(paths["training_curves"], "T64 training and evaluation loss", "Train batches and complete evaluation banks", [
        ("Train batch raw", [(r["optimizer_step"], r["reconstruction"]["total"]) for r in train_plot], "#9ecae1"),
        ("Train EMA alpha=0.05", ema_points, "#3182bd"),
        ("Full evaluation total", [(r["optimizer_step"], r["metrics"]["reconstruction_loss"]["total"]) for r in evaluations], "#08519c"),
    ])
    _svg(paths["gate_curves"], "T64 fit gates", "Fit controls advancement; strict_memory and legacy_exact are diagnostic", [
        ("Fit score", [(r["optimizer_step"], r["metrics"]["fit_gate"]["score"]) for r in evaluations], "#238b45"),
        ("Strict score", [(r["optimizer_step"], r["metrics"]["strict_memory_gate"]["score"]) for r in evaluations], "#f16913"),
        ("Legacy exact score", [(r["optimizer_step"], r["metrics"]["legacy_exact_gate"]["score"]) for r in evaluations], "#cb181d"),
        ("PASS threshold", [(r["optimizer_step"], 1.0) for r in evaluations], "#555555"),
    ])
    mask_points: list[tuple[float, float]] = []
    feature_points: list[tuple[float, float]] = []
    latent_points: list[tuple[float, float]] = []
    if best:
        mask_points = [(index, max(value["worst_state_rmse"], value["worst_action_rmse"])) for index, value in enumerate(best["cases"].values())]
        feature_points = [(row["index"], row["normalized_rmse"]) for row in best["feature_errors"]]
        if "latent_dependence" in best:
            latent_points = [(index, value) for index, value in enumerate(best["latent_dependence"]["ratios_to_correct"].values())]
    _svg(paths["mask_breakdown"], "Latest best Mask breakdown", "Worst-window normalized RMSE for each active Mask family", [("Mask RMSE", mask_points, "#6a51a3")], x_label="Mask family index")
    physical_points = [] if not best else [(row["index"], row["physical_rmse"]) for row in best["feature_errors"]]
    _svg(paths["feature_error"], "97 continuous feature errors", "68 State and 29 Action normalized and physical RMSE values", [
        ("Normalized RMSE", feature_points, "#2171b5"),
        ("Physical RMSE", physical_points, "#41ab5d"),
    ], x_label="Continuous feature index")
    _svg(paths["latent_dependence"], "Hierarchical latent dependence", "Ratios relative to the correct whole latent", [("Donor ratio", latent_points, "#d94801")], x_label="Latent diagnostic index")
    return {key: str(value) for key, value in paths.items()}


def write_evaluation_artifacts(output_run: Path, step: int, metrics: dict[str, Any]) -> dict[str, str]:
    feature_path = output_run / f"data/step_{step:06d}_feature_errors.json"
    latent_path = output_run / f"data/step_{step:06d}_latent_dependence.json"
    atomic_write_json(feature_path, metrics["feature_errors"])
    artifacts = {"feature_errors": str(feature_path)}
    if "latent_dependence" in metrics:
        atomic_write_json(latent_path, metrics["latent_dependence"])
        artifacts["latent_dependence"] = str(latent_path)
    return artifacts


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
