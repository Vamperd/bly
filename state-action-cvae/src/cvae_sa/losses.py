from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .masking import MaskBatch
from .models import ModelOutput


@dataclass
class LossOutput:
    total: torch.Tensor
    masked: torch.Tensor
    forward: torch.Tensor
    inverse: torch.Tensor
    kl: torch.Tensor
    gravity: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu())
            for name in ("total", "masked", "forward", "inverse", "kl", "gravity")
        }


def _masked_huber(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    values = F.huber_loss(prediction, target, reduction="none", delta=1.0)
    return values.masked_select(mask).mean()


def _balanced_masked_loss(
    output: ModelOutput, batch: dict[str, torch.Tensor], masks: MaskBatch
) -> torch.Tensor:
    terms = []
    for prediction, target, mask in (
        (output.physical_state, batch["physical_state"], masks.state_loss),
        (output.previous_action, batch["previous_action"], masks.previous_loss),
        (output.action, batch["action"], masks.action_loss),
    ):
        if bool(mask.any()):
            terms.append(_masked_huber(prediction, target, mask))
    if not terms:
        return output.physical_state.sum() * 0.0
    return torch.stack(terms).mean()


def conditional_kl(
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
    free_bits: float,
) -> torch.Tensor:
    variance_ratio = torch.exp(posterior_logvar - prior_logvar)
    mean_term = torch.square(posterior_mean - prior_mean) * torch.exp(-prior_logvar)
    kl_per_dimension = 0.5 * (
        prior_logvar - posterior_logvar + variance_ratio + mean_term - 1.0
    )
    return kl_per_dimension.clamp_min(float(free_bits)).sum(dim=-1).mean()


def compute_loss(
    output: ModelOutput,
    batch: dict[str, torch.Tensor],
    masks: MaskBatch,
    config: dict[str, Any],
    kl_beta: float,
) -> LossOutput:
    masked = _balanced_masked_loss(output, batch, masks)
    if masks.task_name == "forward":
        target_delta = batch["physical_state"][:, 1:] - batch["physical_state"][:, :-1]
        transition_mask = batch["valid_action"][:, :, None].expand_as(target_delta)
        forward = _masked_huber(output.forward_delta, target_delta, transition_mask)
    else:
        forward = output.forward_delta.sum() * 0.0
    if masks.task_name == "inverse":
        inverse = _masked_huber(output.action, batch["action"], masks.action_loss)
    else:
        inverse = output.action.sum() * 0.0
    kl = conditional_kl(
        output.posterior_mean,
        output.posterior_logvar,
        output.prior_mean,
        output.prior_logvar,
        float(config["free_bits"]),
    )
    gravity_steps = masks.state_loss[..., 61:64].any(dim=-1)
    if bool(gravity_steps.any()):
        gravity_norm = torch.linalg.vector_norm(output.physical_state[..., 61:64], dim=-1)
        gravity = torch.square(gravity_norm - 1.0).masked_select(gravity_steps).mean()
    else:
        gravity = output.physical_state.sum() * 0.0
    total = (
        masked
        + float(config["forward_weight"]) * forward
        + float(config["inverse_weight"]) * inverse
        + float(kl_beta) * kl
        + float(config["gravity_weight"]) * gravity
    )
    return LossOutput(total, masked, forward, inverse, kl, gravity)

