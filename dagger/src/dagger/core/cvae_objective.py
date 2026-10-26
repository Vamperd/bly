"""Pure tensor objectives shared by the CVAE DAgger adapter and tests."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def teacher_action_mse(
    predicted_action: torch.Tensor,
    teacher_action: torch.Tensor,
    anchor: int,
) -> torch.Tensor:
    """MSE between the predicted anchor Action and the teacher label."""

    return (predicted_action[:, int(anchor)].float() - teacher_action.float()).square().mean()


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
