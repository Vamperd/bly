from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .action_masks import semantic_joint_groups
from .masking import MaskGenerator
from .overfit_visualization import write_sensitivity_svg
from .util import atomic_write_json


STATE_GROUPS = {
    "state/joint_position": slice(0, 29),
    "state/joint_velocity": slice(29, 58),
    "state/base_linear_velocity": slice(58, 61),
    "state/base_angular_velocity": slice(61, 64),
    "state/gravity": slice(64, 67),
    "state/base_height": slice(67, 68),
    "state/foot_contact": slice(68, 70),
}


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _occluded(
    batch: dict[str, Any], kind: str, selector: slice | list[int] | None
) -> dict[str, Any]:
    result = dict(batch)
    if kind == "state":
        result["physical_state"] = batch["physical_state"].clone()
        result["physical_state"][..., selector] = 0.0
    elif kind == "action":
        result["action"] = batch["action"].clone()
        result["action"][..., selector] = 0.0
    elif kind == "action_before":
        result["action_before_window"] = torch.zeros_like(batch["action_before_window"])
    else:
        raise ValueError(kind)
    return result


@torch.no_grad()
def evaluate_input_sensitivity(
    model: torch.nn.Module,
    loader: DataLoader,
    masker: MaskGenerator,
    device: torch.device,
    joint_names: list[str],
    output_run: Path,
    max_batches: int = 16,
) -> dict[str, Any]:
    semantic = semantic_joint_groups(joint_names)
    joint_index = {name: index for index, name in enumerate(joint_names)}
    groups: dict[str, tuple[str, slice | list[int] | None]] = {
        name: ("state", selector) for name, selector in STATE_GROUPS.items()
    }
    groups.update({
        f"action/{name}": ("action", [joint_index[joint] for joint in names])
        for name, names in semantic.items()
    })
    groups["action_before_window"] = ("action_before", None)
    values: dict[str, list[float]] = {"baseline": []}
    values.update({name: [] for name in groups})
    model.eval()
    masker.set_step(max(masker.optimizer_step, 40_000))
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = _device_batch(cpu_batch, device)
        masks = masker.generate(batch, force_task="forward_one")
        target = batch["physical_state"][:, 1:] - batch["physical_state"][:, :-1]
        mask = masks.forward_transition[:, :, None].expand_as(target)
        baseline = model(batch, masks, sample_from_prior=True, deterministic=True)
        baseline_error = torch.square(baseline.forward_delta.float() - target.float()).masked_select(mask).mean()
        values["baseline"].append(float(torch.sqrt(baseline_error).cpu()))
        for name, (kind, selector) in groups.items():
            altered = _occluded(batch, kind, selector)
            output = model(altered, masks, sample_from_prior=True, deterministic=True)
            error = torch.square(output.forward_delta.float() - target.float()).masked_select(mask).mean()
            values[name].append(float(torch.sqrt(error).cpu()))
    baseline_rmse = float(np.mean(values["baseline"])) if values["baseline"] else math.inf
    ratios = {
        name: float(np.mean(items)) / max(baseline_rmse, 1e-12)
        for name, items in values.items() if name != "baseline" and items
    }
    result = {
        "interpretation": (
            "Training-set forward sensitivity only; this does not prove that a field is "
            "unnecessary for full-data deployment."
        ),
        "baseline_forward_one_normalized_rmse": baseline_rmse,
        "occluded_forward_rmse_ratio": ratios,
        "max_batches": max_batches,
    }
    atomic_write_json(output_run / "data/input_sensitivity.json", result)
    write_sensitivity_svg(ratios, output_run / "videos/input_sensitivity.svg")
    model.train()
    return result
