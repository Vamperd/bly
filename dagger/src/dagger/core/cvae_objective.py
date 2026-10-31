"""Pure tensor objectives shared by the CVAE DAgger adapter and tests."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F


def teacher_action_mse(
    predicted_action: torch.Tensor,
    teacher_action: torch.Tensor,
    anchor: int,
) -> torch.Tensor:
    """MSE between the predicted anchor Action and the teacher label."""

    return (predicted_action[:, int(anchor)].float() - teacher_action.float()).square().mean()


def teacher_chunk_mse(
    predicted_action: torch.Tensor,
    teacher_chunk: torch.Tensor,
    weight: torch.Tensor,
    anchor: int,
) -> torch.Tensor:
    """gamma^i-weighted MSE over window Actions anchor..anchor+K-1."""

    steps = int(teacher_chunk.shape[1])
    predicted = predicted_action[:, int(anchor) : int(anchor) + steps].float()
    error = (predicted - teacher_chunk.float()).square().mean(dim=-1)
    weight = weight.float()
    return (error * weight).sum() / weight.sum().clamp_min(1.0e-8)


def masked_future_state_mse(
    predicted_state: torch.Tensor,
    target_state: torch.Tensor,
    loss_state_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE over masked continuous State features (the first 68 dimensions)."""

    error = (
        predicted_state[..., :68].float() - target_state[..., :68].float()
    ).square()
    values = error.masked_select(loss_state_mask[..., :68].bool())
    if values.numel():
        return values.mean()
    return predicted_state.float().sum() * 0.0


def masked_future_contact_bce(
    contact_logits: torch.Tensor,
    target_state: torch.Tensor,
    loss_state_mask: torch.Tensor,
) -> torch.Tensor:
    """Contact BCE over masked State contact targets (dimensions 68:70)."""

    values = F.binary_cross_entropy_with_logits(
        contact_logits.float(),
        target_state[..., 68:70].float(),
        reduction="none",
    ).masked_select(loss_state_mask[..., 68:70].bool())
    if values.numel():
        return values.mean()
    return contact_logits.float().sum() * 0.0


def compose_dagger_loss(
    *,
    future_state: torch.Tensor,
    future_contact: torch.Tensor,
    teacher_standard_normal: torch.Tensor,
    teacher_posterior_mean: torch.Tensor,
    kl: torch.Tensor,
    state_weight: float = 1.0,
    contact_weight: float = 0.1,
    standard_normal_weight: float = 0.5,
    posterior_mean_weight: float = 0.5,
    kl_beta: float = 0.0,
) -> torch.Tensor:
    """Compose the DAgger loss without changing any CVAE-internal reduction."""

    return (
        float(state_weight) * future_state.float()
        + float(contact_weight) * future_contact.float()
        + float(standard_normal_weight) * teacher_standard_normal.float()
        + float(posterior_mean_weight) * teacher_posterior_mean.float()
        + float(kl_beta) * kl.float()
    )


def cvae_early_stop_decision(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    min_teacher_improvement_ratio: float = 0.98,
    termination_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Decide whether a mid-run teacher/termination gate should stop training."""

    baseline_rmse = float(baseline.get("teacher_action_rmse", math.inf))
    current_rmse = float(current.get("teacher_action_rmse", math.inf))
    ratio = (
        current_rmse / baseline_rmse
        if baseline_rmse > 0.0 and math.isfinite(baseline_rmse)
        else math.inf
    )
    baseline_termination = float(baseline.get("termination_rate", 0.0))
    current_termination = float(current.get("termination_rate", math.inf))
    teacher_ok = bool(ratio <= float(min_teacher_improvement_ratio))
    termination_ok = bool(
        current_termination <= baseline_termination + float(termination_tolerance)
    )
    return {
        "pass": bool(teacher_ok and termination_ok),
        "teacher_ok": teacher_ok,
        "termination_ok": termination_ok,
        "baseline_teacher_action_rmse": baseline_rmse,
        "current_teacher_action_rmse": current_rmse,
        "teacher_ratio": ratio,
        "min_teacher_improvement_ratio": float(min_teacher_improvement_ratio),
        "baseline_termination_rate": baseline_termination,
        "current_termination_rate": current_termination,
        "termination_tolerance": float(termination_tolerance),
    }
