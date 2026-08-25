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
    auxiliary: torch.Tensor
    history_action: torch.Tensor
    rollout: torch.Tensor
    cycle: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu())
            for name in (
                "total", "masked", "forward", "inverse", "kl", "gravity", "auxiliary",
                "history_action", "rollout", "cycle",
            )
        }


def _masked_huber(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    values = F.huber_loss(prediction, target, reduction="none", delta=1.0)
    return values.masked_select(mask).mean()


PHYSICS_CONTINUOUS_FIELDS = (
    slice(0, 29),
    slice(29, 58),
    slice(58, 61),
    slice(61, 64),
    slice(64, 67),
    slice(67, 68),
)


def _balanced_physics_state_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    step_mask: torch.Tensor,
    contact_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Equal-weight the seven semantic State groups for any physics objective."""
    terms = []
    for field in PHYSICS_CONTINUOUS_FIELDS:
        mask = step_mask[..., None].expand(*step_mask.shape, field.stop - field.start)
        if bool(mask.any()):
            terms.append(_masked_huber(prediction[..., field], target[..., field], mask))
    contact_mask = step_mask[..., None].expand(*step_mask.shape, 2)
    if bool(contact_mask.any()):
        if contact_logits is None:
            contact_loss = F.binary_cross_entropy(
                prediction[..., 68:70].clamp(1e-6, 1.0 - 1e-6),
                target[..., 68:70],
                reduction="none",
            )
        else:
            contact_loss = F.binary_cross_entropy_with_logits(
                contact_logits, target[..., 68:70], reduction="none"
            )
        terms.append(contact_loss.masked_select(contact_mask).mean())
    return torch.stack(terms).mean() if terms else prediction.sum() * 0.0


def _balanced_masked_loss(
    output: ModelOutput, batch: dict[str, torch.Tensor], masks: MaskBatch
) -> torch.Tensor:
    if output.state_contact_logits is not None and batch["physical_state"].shape[-1] == 70:
        state_terms = []
        for field in PHYSICS_CONTINUOUS_FIELDS:
            mask = masks.state_loss[..., field]
            if bool(mask.any()):
                state_terms.append(
                    _masked_huber(
                        output.physical_state[..., field],
                        batch["physical_state"][..., field],
                        mask,
                    )
                )
        contact_mask = masks.state_loss[..., 68:70]
        if bool(contact_mask.any()):
            contact = F.binary_cross_entropy_with_logits(
                output.state_contact_logits,
                batch["physical_state"][..., 68:70],
                reduction="none",
            ).masked_select(contact_mask).mean()
            state_terms.append(contact)
        terms = [torch.stack(state_terms).mean()] if state_terms else []
        if bool(masks.action_loss.any()):
            terms.append(_masked_huber(output.action, batch["action"], masks.action_loss))
        if not terms:
            return output.physical_state.sum() * 0.0
        return torch.stack(terms).mean()
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
    if output.forward_contact_logits is not None and bool(masks.forward_transition.any()):
        predicted_next = batch["physical_state"][:, :-1] + output.forward_delta
        forward = _balanced_physics_state_loss(
            predicted_next,
            batch["physical_state"][:, 1:],
            masks.forward_transition,
            output.forward_contact_logits,
        )
    elif masks.task_name == "forward":
        target_delta = batch["physical_state"][:, 1:] - batch["physical_state"][:, :-1]
        transition_mask = batch["valid_action"][:, :, None].expand_as(target_delta)
        forward = _masked_huber(output.forward_delta, target_delta, transition_mask)
    else:
        forward = output.forward_delta.sum() * 0.0
    if output.inverse_action is not None and bool(masks.inverse_transition.any()):
        inverse_mask = masks.inverse_transition[:, :, None].expand_as(output.inverse_action)
        inverse = _masked_huber(output.inverse_action, batch["action"], inverse_mask)
    elif masks.task_name == "inverse":
        inverse = _masked_huber(output.action, batch["action"], masks.action_loss)
    else:
        inverse = output.action.sum() * 0.0
    if output.history_action is not None and bool(masks.history_action_transition.any()):
        history_mask = masks.history_action_transition[:, :, None].expand_as(
            output.history_action
        )
        history_action = _masked_huber(
            output.history_action, batch["action"], history_mask
        )
    else:
        history_action = output.action.sum() * 0.0
    kl = conditional_kl(
        output.posterior_mean,
        output.posterior_logvar,
        output.prior_mean,
        output.prior_logvar,
        float(config["free_bits"]),
    )
    gravity_slice = slice(64, 67) if batch["physical_state"].shape[-1] == 70 else slice(61, 64)
    gravity_steps = masks.state_loss[..., gravity_slice].any(dim=-1)
    if bool(gravity_steps.any()):
        gravity_norm = torch.linalg.vector_norm(
            output.physical_state[..., gravity_slice], dim=-1
        )
        gravity = torch.square(gravity_norm - 1.0).masked_select(gravity_steps).mean()
    else:
        gravity = output.physical_state.sum() * 0.0
    if output.auxiliary_transition.shape[-1] and bool(masks.forward_transition.any()):
        auxiliary_mask = masks.forward_transition[:, :, None].expand_as(
            output.auxiliary_transition
        )
        auxiliary = _masked_huber(
            output.auxiliary_transition,
            batch["auxiliary_transition"],
            auxiliary_mask,
        )
    else:
        auxiliary = output.forward_delta.sum() * 0.0
    rollout = output.forward_delta.sum() * 0.0
    if output.rollout_state is not None and bool((masks.rollout_horizon > 0).any()):
        rollout_terms = []
        for index in range(output.rollout_state.shape[0]):
            horizon = int(masks.rollout_horizon[index].item())
            if horizon <= 0:
                continue
            start = int(masks.rollout_start[index].item())
            available = min(
                horizon,
                output.rollout_state.shape[1],
                batch["physical_state"].shape[1] - start - 1,
            )
            if available <= 0:
                continue
            prediction = output.rollout_state[index, :available]
            target = batch["physical_state"][index, start + 1 : start + available + 1]
            rollout_terms.append(
                _balanced_physics_state_loss(
                    prediction,
                    target,
                    torch.ones(available, dtype=torch.bool, device=prediction.device),
                )
            )
        if rollout_terms:
            rollout = torch.stack(rollout_terms).mean()
    cycle = output.forward_delta.sum() * 0.0
    cycle_terms = []
    if output.cycle_state is not None and bool(masks.inverse_transition.any()):
        cycle_terms.append(
            _balanced_physics_state_loss(
                output.cycle_state,
                batch["physical_state"][:, 1:],
                masks.inverse_transition,
            )
        )
    if output.cycle_action is not None and bool(masks.forward_transition.any()):
        mask = masks.forward_transition[:, :, None].expand_as(output.cycle_action)
        cycle_terms.append(_masked_huber(output.cycle_action, batch["action"], mask))
    if cycle_terms:
        cycle = torch.stack(cycle_terms).mean()
    total = (
        masked
        + float(config["forward_weight"]) * forward
        + float(config["inverse_weight"]) * inverse
        + float(config.get("history_action_weight", 1.0)) * history_action
        + float(config.get("rollout_weight", 1.0)) * rollout
        + float(config.get("cycle_weight", 0.1)) * cycle
        + float(kl_beta) * kl
        + float(config["gravity_weight"]) * gravity
        + float(config.get("auxiliary_weight", 0.1)) * auxiliary
    )
    return LossOutput(
        total, masked, forward, inverse, kl, gravity, auxiliary,
        history_action, rollout, cycle,
    )
