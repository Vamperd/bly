"""Small, dependency-free diagnostics for DAgger manifests."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import torch


def same_motion_memorization_pass(metrics: list[dict[str, Any]]) -> bool:
    """Judge same-motion DAgger memorization without claiming generalization."""

    if len(metrics) < 2:
        return False
    first = metrics[0].get("student_only", {})
    last = metrics[-1].get("student_only", {})
    baseline = first.get("teacher_action_rmse")
    final = last.get("teacher_action_rmse")
    if baseline is None or final is None or float(baseline) <= 0.0:
        return False
    if float(final) > 0.5 * float(baseline):
        return False
    first_motion = first.get("teacher_action_rmse_by_motion", {})
    last_motion = last.get("teacher_action_rmse_by_motion", {})
    improved = 0
    for key, value in first_motion.items():
        final_value = last_motion.get(key)
        if final_value is None:
            continue
        baseline_value = float(value.get("teacher_action_rmse", 0.0))
        final_value = float(final_value.get("teacher_action_rmse", 0.0))
        if baseline_value > 0.0 and final_value <= 0.7 * baseline_value:
            improved += 1
    if improved < 3:
        return False
    if float(last.get("termination_rate", 1.0)) > float(first.get("termination_rate", 1.0)):
        return False
    if float(last.get("action_saturation_ratio", 1.0)) >= 0.05:
        return False
    return True


def estimate_teacher_identifiability(dataset: Any, *, decimals: int = 2) -> dict[str, Any]:
    """Estimate whether teacher actions are identifiable without reference.

    The probe deliberately uses only the current State and executed Action
    history.  Exact repeated bins are rare for continuous data; an absent bin
    is therefore reported as ``null`` instead of being treated as a pass.
    """

    groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    for round_data in dataset.rounds:
        tensors = round_data.tensors
        visible_state = tensors["physical_state"][:, 32].float()
        visible_action = tensors["action"][:, :32].float().reshape(tensors["action"].shape[0], -1)
        key_values = torch.cat((visible_state, visible_action), dim=-1)
        key_values = torch.round(key_values * (10**decimals)).to(torch.int64)
        for row, teacher in zip(key_values, tensors["teacher_action"]):
            digest = hashlib.sha1(row.numpy().tobytes()).hexdigest()
            groups[digest].append(teacher.float())
    repeated = [values for values in groups.values() if len(values) >= 2]
    if not repeated:
        return {
            "available": False,
            "group_count": len(groups),
            "repeated_group_count": 0,
            "teacher_not_identifiable": None,
            "note": "no repeated visible-state bins; collect more motions or use a coarser probe",
        }
    variances = [float(torch.stack(values).var(dim=0, unbiased=False).mean()) for values in repeated]
    mean_variance = sum(variances) / len(variances)
    return {
        "available": True,
        "group_count": len(groups),
        "repeated_group_count": len(repeated),
        "mean_teacher_action_variance": mean_variance,
        "max_teacher_action_variance": max(variances),
        "teacher_not_identifiable": bool(mean_variance > 0.05),
        "threshold": 0.05,
    }

