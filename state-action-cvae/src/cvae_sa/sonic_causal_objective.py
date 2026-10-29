"""Objectives for the single-step SONIC causal CVAE."""

from __future__ import annotations

import torch

from .sonic_causal_model import SonicGaussianLatent


def action_mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted/target Action shapes differ: {predicted.shape} vs {target.shape}"
        )
    return (predicted.float() - target.float()).square().mean()


def gaussian_kl(posterior: SonicGaussianLatent) -> torch.Tensor:
    mean = posterior.mean.float()
    logvar = posterior.logvar.float()
    return 0.5 * (torch.exp(logvar) + mean.square() - 1.0 - logvar).mean()


def compose_causal_c_loss(
    *,
    posterior_action: torch.Tensor,
    standard_normal_action: torch.Tensor,
    target_action: torch.Tensor,
    posterior: SonicGaussianLatent,
    posterior_weight: float = 1.0,
    standard_normal_weight: float = 0.5,
    kl_beta: float = 0.002,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    posterior_loss = action_mse(posterior_action, target_action)
    standard_normal_loss = action_mse(standard_normal_action, target_action)
    kl = gaussian_kl(posterior)
    total = (
        float(posterior_weight) * posterior_loss
        + float(standard_normal_weight) * standard_normal_loss
        + float(kl_beta) * kl
    )
    return total, {
        "posterior_action": posterior_loss,
        "standard_normal_action": standard_normal_loss,
        "kl": kl,
    }


def compose_dagger_loss(
    *,
    posterior_action: torch.Tensor,
    standard_normal_action: torch.Tensor,
    teacher_action: torch.Tensor,
    posterior: SonicGaussianLatent,
    posterior_weight: float = 0.5,
    standard_normal_weight: float = 0.5,
    kl_beta: float = 0.002,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    posterior_loss = action_mse(posterior_action, teacher_action)
    standard_normal_loss = action_mse(standard_normal_action, teacher_action)
    kl = gaussian_kl(posterior)
    total = (
        float(posterior_weight) * posterior_loss
        + float(standard_normal_weight) * standard_normal_loss
        + float(kl_beta) * kl
    )
    return total, {
        "posterior_action": posterior_loss,
        "standard_normal_action": standard_normal_loss,
        "kl": kl,
    }

