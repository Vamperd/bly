from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, default_collate

from .models import (
    HierarchicalGaussianLatent,
    HierarchicalStandardCVAETransformer,
    build_model,
)
from .posterior_capacity import (
    DeterministicWindowSubset,
    MaskBankDataset,
    _stable_seed,
    validate_motion_prefix,
)
from .posterior_complete_token_protocol import (
    PHYSICAL_RANDOM_MASK_NAMES,
    PHYSICAL_RANDOM_TRAINING_MIXTURE,
    make_dynamic_physical_random_masks,
    make_heldout_physical_random_masks,
    validate_physically_inferable_token_masks,
)
from .posterior_direct_output import assert_output_isolated
from .posterior_hierarchical_conditional_prior import (
    _lr_multiplier,
    _tensor_mapping_sha256,
    validate_h50_a_source,
)
from .posterior_t64_protocol import (
    PHYSICAL_MASK_NAMES,
    _svg,
    append_jsonl,
    evaluate,
    make_autoencode_masks,
    make_physical_masks,
    mask_bank_sha256,
    reconstruction_loss,
    rows_sha256,
    window_identity_rows,
    write_evaluation_artifacts,
)
from .util import (
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    canonical_json_bytes,
    file_sha256,
    load_config,
    load_json,
    seed_everything,
)


CHECKPOINT_FORMAT = "sonic_h50_standard_cvae_checkpoint_v1"
SUMMARY_FORMAT = "sonic_h50_standard_cvae_summary_v1"
LATENT_SCALE_FORMAT = "sonic_h50_standard_cvae_initial_latent_scale_v1"
STAGES = ("fixed", "random", "kl")


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _infinite(loader: DataLoader[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def _base_signature(model_config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "d_model", "posterior_encoder_layers", "condition_encoder_layers",
        "decoder_layers", "heads", "ffn_dim", "global_latent_dim",
        "local_latent_dim", "local_chunks", "chunk_transitions",
        "max_state_steps", "state_dim",
    )
    result = {key: model_config.get(key) for key in keys}
    result.update({
        "kind": "physics_hierarchical_posterior_transformer",
        "profile": "H50",
    })
    return result


def _model_signature(model_config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "d_model", "posterior_encoder_layers", "condition_encoder_layers",
        "conditional_prior_encoder_layers", "decoder_layers", "heads",
        "ffn_dim", "global_latent_dim", "local_latent_dim", "local_chunks",
        "chunk_transitions", "max_state_steps", "state_dim",
    )
    result = {key: model_config.get(key) for key in keys}
    result.update({
        "kind": "physics_hierarchical_standard_cvae_transformer",
        "profile": "H50-SCVAE",
    })
    return result


def initialize_from_h50_a(
    model: HierarchicalStandardCVAETransformer,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    expected_missing = {
        name
        for name in model.state_dict()
        if model.is_conditional_prior_parameter(name)
        or name.startswith("posterior_global_logvar_head.")
        or name.startswith("posterior_local_logvar_head.")
    }
    checks: dict[str, bool] = {
        "unexpected_keys_empty": not incompatible.unexpected_keys,
        "only_new_prior_and_logvar_keys_missing": (
            set(incompatible.missing_keys) == expected_missing
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"H50-A to standard CVAE migration failed: {checks}")
    model.initialize_conditional_prior_from_posterior()
    model.initialize_logvar_heads()
    with torch.no_grad():
        checks.update({
            "prior_state_values_copied": torch.equal(
                model.conditional_prior_state_input.weight[:, : model.state_dim],
                model.state_input.weight[:, : model.state_dim],
            ),
            "prior_action_values_copied": torch.equal(
                model.conditional_prior_action_input.weight[:, :29],
                model.action_input.weight[:, :29],
            ),
            "prior_state_mask_columns_zero": bool(
                torch.count_nonzero(
                    model.conditional_prior_state_input.weight[:, model.state_dim :]
                ) == 0
            ),
            "prior_action_mask_columns_zero": bool(
                torch.count_nonzero(
                    model.conditional_prior_action_input.weight[:, 29:]
                ) == 0
            ),
            "all_logvar_weights_zero": all(
                bool(torch.count_nonzero(head.weight) == 0)
                for head in (
                    model.posterior_global_logvar_head,
                    model.posterior_local_logvar_head,
                    model.conditional_prior_global_logvar_head,
                    model.conditional_prior_local_logvar_head,
                )
            ),
            "all_logvar_biases_minus_four": all(
                bool(torch.all(head.bias == -4.0))
                for head in (
                    model.posterior_global_logvar_head,
                    model.posterior_local_logvar_head,
                    model.conditional_prior_global_logvar_head,
                    model.conditional_prior_local_logvar_head,
                )
            ),
        })
    if not all(checks.values()):
        raise RuntimeError(f"standard CVAE initialization failed: {checks}")
    return {
        "strategy": (
            "H50-A model-only initialization; copy posterior to conditional prior, "
            "zero prior Mask columns, initialize logvar weights=0 and biases=-4"
        ),
        "checks": checks,
        "initialized_state_sha256": _tensor_mapping_sha256(model.state_dict().items()),
        "optimizer_scheduler_rng_restored": False,
    }


def _full_masks(
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        batch["valid_state"].bool()[..., None].expand_as(batch["physical_state"]),
        batch["valid_action"].bool()[..., None].expand_as(batch["action"]),
    )


def weighted_reconstruction_loss(
    output: Any,
    batch: dict[str, torch.Tensor],
    state_mask: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    masked_weight: float = 0.75,
    full_weight: float = 0.25,
) -> dict[str, Any]:
    if not math.isclose(masked_weight + full_weight, 1.0):
        raise ValueError("masked/full reconstruction weights must sum to one")
    full_state, full_action = _full_masks(batch)
    masked = reconstruction_loss(output, batch, state_mask, action_mask)
    full = reconstruction_loss(output, batch, full_state, full_action)
    result: dict[str, Any] = {"masked": masked, "full": full}
    for name in ("total", "state", "action", "contact"):
        result[name] = masked_weight * masked[name] + full_weight * full[name]
    return result


def mean_alignment_loss(
    posterior: HierarchicalGaussianLatent,
    prior: HierarchicalGaussianLatent,
    global_std: torch.Tensor,
    local_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    global_loss = torch.square(
        (prior.global_mean - posterior.global_mean.detach()) / global_std
    ).mean()
    local_loss = torch.square(
        (prior.local_mean - posterior.local_mean.detach()) / local_std
    ).mean()
    return {
        "total": 0.5 * (global_loss + local_loss),
        "global": global_loss,
        "local": local_loss,
    }


def hierarchical_kl(
    posterior: HierarchicalGaussianLatent,
    prior: HierarchicalGaussianLatent,
) -> dict[str, torch.Tensor]:
    def one(mean_q: torch.Tensor, logvar_q: torch.Tensor,
            mean_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            logvar_p - logvar_q
            + (torch.exp(logvar_q) + torch.square(mean_q - mean_p))
            / torch.exp(logvar_p)
            - 1.0
        ).mean()

    global_kl = one(
        posterior.global_mean, posterior.global_logvar,
        prior.global_mean, prior.global_logvar,
    )
    local_kl = one(
        posterior.local_mean, posterior.local_logvar,
        prior.local_mean, prior.local_logvar,
    )
    return {
        "total": 0.5 * (global_kl + local_kl),
        "global": global_kl,
        "local": local_kl,
    }


def alignment_weight(step: int, schedule: list[dict[str, Any]]) -> float:
    for row in schedule:
        start, end = int(row["start"]), int(row["end"])
        if start <= step <= end:
            fraction = (step - start) / max(end - start, 1)
            return float(row["start_value"]) + fraction * (
                float(row["end_value"]) - float(row["start_value"])
            )
    if step > int(schedule[-1]["end"]):
        return float(schedule[-1]["end_value"])
    raise ValueError("alignment schedule does not cover the optimizer step")


@torch.no_grad()
def build_initial_latent_scale(
    model: HierarchicalStandardCVAETransformer,
    base_loader: Iterable[dict[str, Any]],
    device: torch.device,
    output_run: Path,
    *,
    source_hash: str,
    dataset_hash: str,
    window_hash: str,
) -> dict[str, Any]:
    """Cache only the H50-A scale; no H50-A latent remains a target."""
    model.eval()
    globals_: list[torch.Tensor] = []
    locals_: list[torch.Tensor] = []
    for cpu_batch in base_loader:
        batch = _device_batch(cpu_batch, device)
        state_mask = torch.zeros_like(batch["physical_state"], dtype=torch.bool)
        action_mask = torch.zeros_like(batch["action"], dtype=torch.bool)
        global_latent, local_latents = model.encode_posterior(
            batch, state_mask, action_mask
        )
        globals_.append(global_latent.cpu())
        locals_.append(local_latents.cpu())
    global_all = torch.cat(globals_)
    local_all = torch.cat(locals_)
    tensors = {
        "format_version": LATENT_SCALE_FORMAT,
        "source_checkpoint_sha256": source_hash,
        "dataset_manifest_sha256": dataset_hash,
        "selected_windows_sha256": window_hash,
        "global_std": global_all.std(dim=0, unbiased=False).clamp_min(1e-3),
        "local_std": local_all.std(dim=0, unbiased=False).clamp_min(1e-3),
    }
    path = output_run / "data/initial_latent_scale.pt"
    atomic_torch_save(path, tensors)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.equal(tensors["global_std"], reloaded["global_std"]):
        raise RuntimeError("global latent scale failed exact readback")
    if not torch.equal(tensors["local_std"], reloaded["local_std"]):
        raise RuntimeError("local latent scale failed exact readback")
    manifest = {
        "format_version": LATENT_SCALE_FORMAT,
        "path": str(path),
        "sha256": file_sha256(path),
        "source_checkpoint_sha256": source_hash,
        "dataset_manifest_sha256": dataset_hash,
        "selected_windows_sha256": window_hash,
        "window_count": len(global_all),
        "global_shape": list(tensors["global_std"].shape),
        "local_shape": list(tensors["local_std"].shape),
        "standard_deviation_floor": 1e-3,
        "role": "fixed unit scaling only; not a frozen teacher target",
    }
    atomic_write_json(output_run / "manifests/initial_latent_scale.json", manifest)
    return {**tensors, "manifest": manifest}


class _PathView(torch.nn.Module):
    def __init__(
        self,
        model: HierarchicalStandardCVAETransformer,
        latent_source: str,
        *,
        sample_index: int | None = None,
        sample_seed: int = 20260841,
    ) -> None:
        super().__init__()
        self.model = model
        self.latent_source = latent_source
        self.sample_index = sample_index
        self.sample_seed = sample_seed

    def _epsilon(
        self,
        batch: dict[str, Any],
        distribution: HierarchicalGaussianLatent,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.sample_index is None:
            return None
        global_rows: list[torch.Tensor] = []
        local_rows: list[torch.Tensor] = []
        for index in range(len(batch["motion_key"])):
            mask_slot = batch.get("mask_slot")
            slot = int(mask_slot[index]) if mask_slot is not None else -1
            seed = _stable_seed(
                self.sample_seed,
                "standard-cvae-shared-epsilon",
                str(batch["motion_key"][index]),
                int(batch["variant_id"][index]),
                int(batch["window_start"][index]),
                int(batch["window_index"][index]),
                slot,
                int(self.sample_index),
            )
            generator = torch.Generator().manual_seed(seed)
            global_rows.append(torch.randn(
                distribution.global_mean.shape[1:], generator=generator
            ))
            local_rows.append(torch.randn(
                distribution.local_mean.shape[1:], generator=generator
            ))
        dtype, device = distribution.global_mean.dtype, distribution.global_mean.device
        return (
            torch.stack(global_rows).to(device=device, dtype=dtype),
            torch.stack(local_rows).to(device=device, dtype=dtype),
        )

    def forward(
        self,
        batch: dict[str, Any],
        state_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Any:
        if self.latent_source.startswith("posterior"):
            distribution = self.model.encode_posterior_distribution(
                batch, state_mask, action_mask
            )
        else:
            distribution = self.model.encode_conditional_prior_distribution(
                batch, state_mask, action_mask
            )
        if self.latent_source.endswith("sample"):
            global_latent, local_latents = self.model.reparameterize(
                distribution, self._epsilon(batch, distribution)
            )
        else:
            global_latent, local_latents = (
                distribution.global_mean, distribution.local_mean
            )
        return self.model.decode_from_conditioned_latents(
            batch, state_mask, action_mask, global_latent, local_latents
        )


def _alignment_gate(
    metrics: dict[str, float], thresholds: dict[str, float]
) -> dict[str, Any]:
    ratios = {
        "global_standardized_rmse": (
            metrics["global_standardized_rmse"]
            / thresholds["global_standardized_rmse"]
        ),
        "local_standardized_rmse": (
            metrics["local_standardized_rmse"]
            / thresholds["local_standardized_rmse"]
        ),
        "global_cosine": thresholds["global_cosine"]
        / max(metrics["global_cosine"], 1e-12),
        "local_cosine": thresholds["local_cosine"]
        / max(metrics["local_cosine"], 1e-12),
    }
    score = max(ratios.values())
    return {
        "passed": bool(math.isfinite(score) and score <= 1.0),
        "score": score,
        "thresholds": thresholds,
        "threshold_ratios": ratios,
    }


@torch.no_grad()
def evaluate_mean_alignment(
    model: HierarchicalStandardCVAETransformer,
    loader: Iterable[dict[str, Any]],
    maker: Callable[[dict[str, Any]], tuple[torch.Tensor, torch.Tensor, list[str]]],
    scale: dict[str, Any],
    device: torch.device,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    model.eval()
    global_sse = local_sse = 0.0
    global_count = local_count = 0
    global_cosines: list[torch.Tensor] = []
    local_cosines: list[torch.Tensor] = []
    global_std = scale["global_std"].to(device)
    local_std = scale["local_std"].to(device)
    for cpu_batch in loader:
        state_mask, action_mask, _ = maker(cpu_batch)
        batch = _device_batch(cpu_batch, device)
        state_mask = state_mask.to(device)
        action_mask = action_mask.to(device)
        posterior = model.encode_posterior_distribution(
            batch, state_mask, action_mask
        )
        prior = model.encode_conditional_prior_distribution(
            batch, state_mask, action_mask
        )
        global_error = (prior.global_mean - posterior.global_mean) / global_std
        local_error = (prior.local_mean - posterior.local_mean) / local_std
        global_sse += float(torch.square(global_error).sum().cpu())
        local_sse += float(torch.square(local_error).sum().cpu())
        global_count += global_error.numel()
        local_count += local_error.numel()
        global_cosines.append(F.cosine_similarity(
            prior.global_mean, posterior.global_mean, dim=-1
        ).cpu())
        local_cosines.append(F.cosine_similarity(
            prior.local_mean, posterior.local_mean, dim=-1
        ).mean(dim=-1).cpu())
    result = {
        "global_standardized_rmse": math.sqrt(global_sse / global_count),
        "local_standardized_rmse": math.sqrt(local_sse / local_count),
        "global_cosine": float(torch.cat(global_cosines).mean()),
        "local_cosine": float(torch.cat(local_cosines).mean()),
        "sample_count": global_count // int(scale["global_std"].numel()),
    }
    result["gate"] = _alignment_gate(result, thresholds)
    return result


@torch.no_grad()
def evaluate_distribution_statistics(
    model: HierarchicalStandardCVAETransformer,
    loader: Iterable[dict[str, Any]],
    maker: Callable[[dict[str, Any]], tuple[torch.Tensor, torch.Tensor, list[str]]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_samples = 0
    global_kl_sum: torch.Tensor | None = None
    local_kl_sum: torch.Tensor | None = None
    standard_deviations = {
        "posterior_global": 0.0, "posterior_local": 0.0,
        "prior_global": 0.0, "prior_local": 0.0,
    }
    for cpu_batch in loader:
        state_mask, action_mask, _ = maker(cpu_batch)
        batch = _device_batch(cpu_batch, device)
        state_mask = state_mask.to(device)
        action_mask = action_mask.to(device)
        posterior = model.encode_posterior_distribution(batch, state_mask, action_mask)
        prior = model.encode_conditional_prior_distribution(batch, state_mask, action_mask)
        global_element_kl = 0.5 * (
            prior.global_logvar - posterior.global_logvar
            + (
                torch.exp(posterior.global_logvar)
                + torch.square(posterior.global_mean - prior.global_mean)
            ) / torch.exp(prior.global_logvar)
            - 1.0
        )
        local_element_kl = 0.5 * (
            prior.local_logvar - posterior.local_logvar
            + (
                torch.exp(posterior.local_logvar)
                + torch.square(posterior.local_mean - prior.local_mean)
            ) / torch.exp(prior.local_logvar)
            - 1.0
        )
        global_batch_sum = global_element_kl.sum(dim=0).cpu()
        local_batch_sum = local_element_kl.sum(dim=(0, 1)).cpu()
        global_kl_sum = global_batch_sum if global_kl_sum is None else global_kl_sum + global_batch_sum
        local_kl_sum = local_batch_sum if local_kl_sum is None else local_kl_sum + local_batch_sum
        size = posterior.global_mean.shape[0]
        total_samples += size
        standard_deviations["posterior_global"] += float(
            torch.exp(0.5 * posterior.global_logvar).mean().cpu()
        ) * size
        standard_deviations["posterior_local"] += float(
            torch.exp(0.5 * posterior.local_logvar).mean().cpu()
        ) * size
        standard_deviations["prior_global"] += float(
            torch.exp(0.5 * prior.global_logvar).mean().cpu()
        ) * size
        standard_deviations["prior_local"] += float(
            torch.exp(0.5 * prior.local_logvar).mean().cpu()
        ) * size
    assert global_kl_sum is not None and local_kl_sum is not None
    global_by_dimension = global_kl_sum / total_samples
    local_by_dimension = local_kl_sum / (total_samples * model.local_chunks)
    global_mean = float(global_by_dimension.mean())
    local_mean = float(local_by_dimension.mean())
    result = {
        "raw_kl": {
            "global": global_mean,
            "local": local_mean,
            "total": 0.5 * (global_mean + local_mean),
        },
        "mean_standard_deviation": {
            name: value / total_samples for name, value in standard_deviations.items()
        },
        "active_latent_fraction_at_kl_0_01": {
            "global": float((global_by_dimension > 0.01).float().mean()),
            "local": float((local_by_dimension > 0.01).float().mean()),
        },
        "sample_count": total_samples,
    }
    result["finite"] = all(
        math.isfinite(float(value))
        for group in (
            result["raw_kl"], result["mean_standard_deviation"],
            result["active_latent_fraction_at_kl_0_01"],
        )
        for value in group.values()
    )
    return result


def _slice_batch(batch: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    integer_indices = [int(value) for value in indices.tolist()]
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value[indices]
        elif isinstance(value, (list, tuple)):
            result[key] = [value[index] for index in integer_indices]
        else:
            result[key] = value
    return result


def _continuous_sse(
    output: Any,
    batch: dict[str, torch.Tensor],
    state_mask: torch.Tensor,
    action_mask: torch.Tensor,
) -> tuple[float, int]:
    state = torch.square(
        output.physical_state[..., :68] - batch["physical_state"][..., :68]
    ).masked_select(state_mask[..., :68])
    action = torch.square(output.action - batch["action"]).masked_select(action_mask)
    return float(state.sum().cpu() + action.sum().cpu()), state.numel() + action.numel()


@torch.no_grad()
def evaluate_high_mask_latent_usage(
    model: HierarchicalStandardCVAETransformer,
    loader: Iterable[dict[str, Any]],
    maker: Callable[[dict[str, Any]], tuple[torch.Tensor, torch.Tensor, list[str]]],
    device: torch.device,
    minimum_ratio: float,
) -> dict[str, Any]:
    """Replace p(z|c) on high-information-loss physical queries.

    This is a diagnostic, not the old all-Mask tenfold requirement.  It blocks
    KL only when every high-Mask family is insensitive to both zero and donor
    latents.
    """
    high_names = {"state_rollout", "full_action", "joint_gap_8", "physical_triple_0",
                  "physical_triple_1", "physical_triple_2", "physical_triple_3"}
    records: list[dict[str, Any]] = []
    model.eval()
    for cpu_batch in loader:
        state_mask, action_mask, names = maker(cpu_batch)
        selected = torch.tensor(
            [index for index, name in enumerate(names) if name in high_names],
            dtype=torch.long,
        )
        if not selected.numel():
            continue
        cpu_selected = _slice_batch(cpu_batch, selected)
        state_selected = state_mask[selected]
        action_selected = action_mask[selected]
        batch = _device_batch(cpu_selected, device)
        state_device = state_selected.to(device)
        action_device = action_selected.to(device)
        prior = model.encode_conditional_prior_distribution(
            batch, state_device, action_device
        )
        records.append({
            "batch": cpu_selected,
            "state_mask": state_selected,
            "action_mask": action_selected,
            "names": [names[int(index)] for index in selected.tolist()],
            "global": prior.global_mean.cpu(),
            "local": prior.local_mean.cpu(),
        })
    if not records:
        raise ValueError("latent-usage diagnostic found no high-Mask samples")
    global_all = torch.cat([row["global"] for row in records])
    local_all = torch.cat([row["local"] for row in records])
    identities: list[dict[str, Any]] = []
    names_all: list[str] = []
    flat_batches: list[dict[str, Any]] = []
    for row in records:
        cpu_batch = row["batch"]
        for index, name in enumerate(row["names"]):
            identities.append({
                "window_index": int(cpu_batch["window_index"][index]),
                "motion_key": str(cpu_batch["motion_key"][index]),
                "mask": name,
            })
            names_all.append(name)
            flat_batches.append(_slice_batch(cpu_batch, torch.tensor([index])))
    donor_window: list[int] = []
    donor_motion: list[int] = []
    for index, identity in enumerate(identities):
        window_candidate = next((
            (index + offset) % len(identities)
            for offset in range(1, len(identities) + 1)
            if identities[(index + offset) % len(identities)]["window_index"]
            != identity["window_index"]
        ), index)
        motion_candidate = next((
            (index + offset) % len(identities)
            for offset in range(1, len(identities) + 1)
            if identities[(index + offset) % len(identities)]["motion_key"]
            != identity["motion_key"]
        ), window_candidate)
        donor_window.append(window_candidate)
        donor_motion.append(motion_candidate)
    sums: dict[str, dict[str, list[float | int]]] = {}
    offset = 0
    for row in records:
        size = len(row["names"])
        own = list(range(offset, offset + size))
        choices = {
            "correct": (global_all[own], local_all[own]),
            "zero": (
                torch.zeros_like(global_all[own]), torch.zeros_like(local_all[own])
            ),
            "cross_window": (
                global_all[[donor_window[index] for index in own]],
                local_all[[donor_window[index] for index in own]],
            ),
            "cross_motion": (
                global_all[[donor_motion[index] for index in own]],
                local_all[[donor_motion[index] for index in own]],
            ),
        }
        batch = _device_batch(row["batch"], device)
        state_mask = row["state_mask"].to(device)
        action_mask = row["action_mask"].to(device)
        for choice, (global_latent, local_latents) in choices.items():
            output = model.decode_from_conditioned_latents(
                batch, state_mask, action_mask,
                global_latent.to(device), local_latents.to(device),
            )
            for sample_index, mask_name in enumerate(row["names"]):
                total, count = _continuous_sse(
                    output=type("Sample", (), {
                        "physical_state": output.physical_state[sample_index:sample_index + 1],
                        "action": output.action[sample_index:sample_index + 1],
                    })(),
                    batch={
                        "physical_state": batch["physical_state"][sample_index:sample_index + 1],
                        "action": batch["action"][sample_index:sample_index + 1],
                    },
                    state_mask=state_mask[sample_index:sample_index + 1],
                    action_mask=action_mask[sample_index:sample_index + 1],
                )
                sums.setdefault(mask_name, {}).setdefault(choice, [0.0, 0])
                sums[mask_name][choice][0] += total
                sums[mask_name][choice][1] += count
        donor_items = []
        for donor_index in (donor_window[index] for index in own):
            item: dict[str, Any] = {}
            for key, value in flat_batches[donor_index].items():
                if isinstance(value, torch.Tensor):
                    item[key] = value[0]
                elif isinstance(value, list):
                    item[key] = value[0]
                else:
                    item[key] = value
            donor_items.append(item)
        donor_batch = _device_batch(default_collate(donor_items), device)
        condition_permuted = model.decode_from_conditioned_latents(
            donor_batch, state_mask, action_mask,
            global_all[own].to(device), local_all[own].to(device),
        )
        for sample_index, mask_name in enumerate(row["names"]):
            total, count = _continuous_sse(
                output=type("Sample", (), {
                    "physical_state": condition_permuted.physical_state[sample_index:sample_index + 1],
                    "action": condition_permuted.action[sample_index:sample_index + 1],
                })(),
                batch={
                    "physical_state": batch["physical_state"][sample_index:sample_index + 1],
                    "action": batch["action"][sample_index:sample_index + 1],
                },
                state_mask=state_mask[sample_index:sample_index + 1],
                action_mask=action_mask[sample_index:sample_index + 1],
            )
            sums.setdefault(mask_name, {}).setdefault("condition_permutation", [0.0, 0])
            sums[mask_name]["condition_permutation"][0] += total
            sums[mask_name]["condition_permutation"][1] += count
        offset += size
    cases: dict[str, Any] = {}
    for name, choices in sums.items():
        rmse = {
            choice: math.sqrt(float(total) / int(count))
            for choice, (total, count) in choices.items()
        }
        correct = max(rmse["correct"], 1e-12)
        cases[name] = {
            "rmse": rmse,
            "ratios": {
                choice: value / correct
                for choice, value in rmse.items() if choice != "correct"
            },
        }
    ignored = bool(cases) and all(
        max(
            case["ratios"][name]
            for name in ("zero", "cross_window", "cross_motion")
        ) <= minimum_ratio
        for case in cases.values()
    )
    return {
        "cases": cases,
        "minimum_required_ratio": minimum_ratio,
        "all_high_masks_latent_ignored": ignored,
        "passed_for_kl_entry": not ignored,
        "identity_sha256": hashlib.sha256(canonical_json_bytes(identities)).hexdigest(),
    }


def _metric_summary(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "score": float(metrics["fit_gate"]["score"]),
        "state": float(metrics["global_state_rmse"]),
        "action": float(metrics["global_action_rmse"]),
        "worst_state": float(metrics["worst_mask_state_rmse"]),
        "worst_action": float(metrics["worst_mask_action_rmse"]),
        "p99": float(metrics["continuous_p99_abs"]),
        "contact": float(metrics["contact_accuracy"]),
    }


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@torch.no_grad()
def evaluate_sample_paths(
    model: HierarchicalStandardCVAETransformer,
    loader: Iterable[dict[str, Any]],
    base_loader: Iterable[dict[str, Any]],
    maker: Callable[[dict[str, Any]], tuple[torch.Tensor, torch.Tensor, list[str]]],
    device: torch.device,
    *,
    count: int,
    sample_seed: int,
    evaluate_kwargs: dict[str, Any],
    posterior_mean: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    paths: dict[str, list[dict[str, Any]]] = {
        "posterior_sample": [], "conditional_prior_sample": []
    }
    for sample_index in range(count):
        for name, source in (
            ("posterior_sample", "posterior_sample"),
            ("conditional_prior_sample", "prior_sample"),
        ):
            metrics = evaluate(
                _PathView(
                    model, source, sample_index=sample_index,
                    sample_seed=sample_seed,
                ),
                loader, base_loader, device, maker,
                latent_diagnostics=False,
                report_full_sequence=True,
                **evaluate_kwargs,
            )
            paths[name].append(_metric_summary(metrics))
    aggregates: dict[str, Any] = {}
    for name, rows in paths.items():
        fields = rows[0].keys()
        aggregates[name] = {
            field: {
                "mean": sum(row[field] for row in rows) / len(rows),
                "std": float(torch.tensor([row[field] for row in rows]).std(unbiased=False)),
                "p50": _quantile([row[field] for row in rows], 0.50),
                "p95": _quantile([row[field] for row in rows], 0.95),
                "worst": max(row[field] for row in rows),
                "best": min(row[field] for row in rows),
            }
            for field in fields
        }
        aggregates[name]["samples"] = rows
    posterior_mean_error = math.sqrt(
        0.5 * (
            posterior_mean["global_state_rmse"] ** 2
            + posterior_mean["global_action_rmse"] ** 2
        )
    )
    posterior_sample_error = math.sqrt(0.5 * (
        aggregates["posterior_sample"]["state"]["mean"] ** 2
        + aggregates["posterior_sample"]["action"]["mean"] ** 2
    ))
    prior_sample_error = math.sqrt(0.5 * (
        aggregates["conditional_prior_sample"]["state"]["mean"] ** 2
        + aggregates["conditional_prior_sample"]["action"]["mean"] ** 2
    ))
    prior_rows = paths["conditional_prior_sample"]
    thresholds = evaluate_kwargs["fit_thresholds"]
    averaged_prior_pass = bool(
        aggregates["conditional_prior_sample"]["state"]["mean"] <= thresholds["global_state_rmse"]
        and aggregates["conditional_prior_sample"]["action"]["mean"] <= thresholds["global_action_rmse"]
        and aggregates["conditional_prior_sample"]["worst_state"]["mean"] <= thresholds["worst_mask_state_rmse"]
        and aggregates["conditional_prior_sample"]["worst_action"]["mean"] <= thresholds["worst_mask_action_rmse"]
        and aggregates["conditional_prior_sample"]["p99"]["mean"] <= thresholds["continuous_p99_abs"]
    )
    passed = bool(
        averaged_prior_pass
        and aggregates["conditional_prior_sample"]["score"]["p95"]
        <= float(contract["sample_fit_p95"])
        and all(row["contact"] == 1.0 for rows in paths.values() for row in rows)
        and posterior_sample_error / max(posterior_mean_error, 1e-12)
        <= float(contract["posterior_sample_degradation"])
        and prior_sample_error / max(posterior_sample_error, 1e-12)
        <= float(contract["prior_sample_degradation"])
    )
    return {
        "sample_count": count,
        "shared_epsilon": True,
        "sample_seed": sample_seed,
        "paths": aggregates,
        "posterior_sample_to_mean_error_ratio": (
            posterior_sample_error / max(posterior_mean_error, 1e-12)
        ),
        "prior_sample_to_posterior_sample_error_ratio": (
            prior_sample_error / max(posterior_sample_error, 1e-12)
        ),
        "averaged_prior_fit_passed": averaged_prior_pass,
        "passed": passed,
    }


@torch.no_grad()
def evaluate_prior_output_diversity(
    model: HierarchicalStandardCVAETransformer,
    loader: Iterable[dict[str, Any]],
    maker: Callable[[dict[str, Any]], tuple[torch.Tensor, torch.Tensor, list[str]]],
    device: torch.device,
    *,
    count: int,
    sample_seed: int,
) -> dict[str, Any]:
    model.eval()
    total = squared = 0.0
    elements = 0
    maximum = 0.0
    contact_disagreement = contact_elements = 0
    for cpu_batch in loader:
        state_mask, action_mask, _ = maker(cpu_batch)
        batch = _device_batch(cpu_batch, device)
        state_mask = state_mask.to(device)
        action_mask = action_mask.to(device)
        prior = model.encode_conditional_prior_distribution(
            batch, state_mask, action_mask
        )
        states: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        contacts: list[torch.Tensor] = []
        for sample_index in range(count):
            view = _PathView(
                model, "prior_sample", sample_index=sample_index,
                sample_seed=sample_seed,
            )
            global_latent, local_latents = model.reparameterize(
                prior, view._epsilon(batch, prior)
            )
            output = model.decode_from_conditioned_latents(
                batch, state_mask, action_mask, global_latent, local_latents
            )
            states.append(output.physical_state[..., :68])
            actions.append(output.action)
            contacts.append(output.state_contact_logits.sigmoid() >= 0.5)
        state_std = torch.stack(states).std(dim=0, unbiased=False).masked_select(
            state_mask[..., :68]
        )
        action_std = torch.stack(actions).std(dim=0, unbiased=False).masked_select(
            action_mask
        )
        values = torch.cat((state_std, action_std))
        total += float(values.sum().cpu())
        squared += float(torch.square(values).sum().cpu())
        elements += values.numel()
        if values.numel():
            maximum = max(maximum, float(values.max().cpu()))
        contact_stack = torch.stack(contacts)
        contact_mask = state_mask[..., 68:70]
        disagreement = contact_stack.any(dim=0) != contact_stack.all(dim=0)
        contact_disagreement += int(disagreement.masked_select(contact_mask).sum().cpu())
        contact_elements += int(contact_mask.sum().cpu())
    return {
        "sample_count": count,
        "masked_continuous_mean_standard_deviation": total / max(elements, 1),
        "masked_continuous_rms_standard_deviation": math.sqrt(squared / max(elements, 1)),
        "masked_continuous_max_standard_deviation": maximum,
        "contact_class_disagreement_fraction": (
            contact_disagreement / contact_elements if contact_elements else 0.0
        ),
        "continuous_element_count": elements,
        "finite": all(math.isfinite(value) for value in (total, squared, maximum)),
    }


def _parameter_groups(
    model: HierarchicalStandardCVAETransformer,
    stage: str,
    contract: dict[str, Any],
    maximum: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, dict[str, Any]]:
    phase_counts = model.set_standard_training_phase("kl" if stage == "kl" else "mean")
    decoder_prefixes = (
        "decoder.", "decoder_query_base", "decoder_type_embedding.",
        "global_memory_projection.", "local_memory_projection.",
        "film_global_projection.", "film_local_projection.", "film_projection.",
        "state_continuous_output.", "state_contact_output.", "action_output.",
    )
    groups: dict[str, list[torch.nn.Parameter]] = {
        "encoder_condition_mean_heads": [], "decoder": [], "logvar_heads": []
    }
    names_by_group: dict[str, list[str]] = {key: [] for key in groups}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if model.is_logvar_parameter(name):
            group = "logvar_heads"
        elif any(name.startswith(prefix) for prefix in decoder_prefixes):
            group = "decoder"
        else:
            group = "encoder_condition_mean_heads"
        groups[group].append(parameter)
        names_by_group[group].append(name)
    if stage != "kl" and groups["logvar_heads"]:
        raise RuntimeError("logvar heads must remain frozen before KL")
    encoder_lr = float(contract["encoder_learning_rate"])
    decoder_lr = float(contract["decoder_learning_rate"])
    specifications = [
        ("encoder_condition_mean_heads", encoder_lr),
        ("decoder", decoder_lr),
    ]
    if stage == "kl":
        specifications.append(("logvar_heads", float(contract["logvar_learning_rate"])))
    optimizer = torch.optim.AdamW(
        [
            {"params": groups[name], "lr": lr, "name": name}
            for name, lr in specifications
        ],
        weight_decay=0.0,
    )
    warmup = int(contract["warmup_steps"])
    minimum = float(contract["minimum_learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [
            lambda step, peak=lr: _lr_multiplier(
                step, warmup, maximum, minimum / peak
            )
            for _, lr in specifications
        ],
    )
    covered = sum(parameter.numel() for values in groups.values() for parameter in values)
    if covered != phase_counts["trainable"]:
        raise RuntimeError("standard CVAE optimizer does not cover all trainable parameters")
    return optimizer, scheduler, {
        "phase_counts": phase_counts,
        "groups": {
            name: {
                "parameter_count": sum(parameter.numel() for parameter in groups[name]),
                "name_sha256": hashlib.sha256(
                    canonical_json_bytes(sorted(names_by_group[name]))
                ).hexdigest(),
            }
            for name in groups
        },
    }


def _checkpoint(
    model: HierarchicalStandardCVAETransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: dict[str, Any],
    *,
    stage: str,
    optimizer_step: int,
    dataset_hash: str,
    window_hash: str,
    fixed_hash: str,
    heldout_hash: str,
    source_hash: str,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT,
        "stage": stage,
        "optimizer_step": optimizer_step,
        "dataset_manifest_sha256": dataset_hash,
        "selected_windows_sha256": window_hash,
        "fixed_fixture_sha256": fixed_hash,
        "heldout_fixture_sha256": heldout_hash,
        "source_checkpoint_sha256": source_hash,
        "model_signature": _model_signature(config["model"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }


def validate_checkpoint(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "format": checkpoint.get("format_version") == CHECKPOINT_FORMAT,
        "stage": checkpoint.get("stage") == expected["stage"],
        "dataset": checkpoint.get("dataset_manifest_sha256") == expected["dataset"],
        "windows": checkpoint.get("selected_windows_sha256") == expected["windows"],
        "fixed": checkpoint.get("fixed_fixture_sha256") == expected["fixed"],
        "heldout": checkpoint.get("heldout_fixture_sha256") == expected["heldout"],
        "source": checkpoint.get("source_checkpoint_sha256") == expected["source"],
        "parameters": int(checkpoint.get("parameter_count", -1)) == expected["parameters"],
        "model": isinstance(checkpoint.get("model"), dict),
        "optimizer": isinstance(checkpoint.get("optimizer"), dict),
        "scheduler": isinstance(checkpoint.get("scheduler"), dict),
    }
    return {"passed": all(checks.values()), "checks": checks, "sha256": file_sha256(path)}


def load_stage_initialization(
    model: HierarchicalStandardCVAETransformer,
    init_run: Path,
    *,
    required_stage: str,
    dataset_hash: str,
    window_hash: str,
    fixed_hash: str,
    heldout_hash: str,
    source_hash: str,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    init_run = init_run.expanduser().resolve()
    summary_path = init_run / "manifests/standard_cvae_summary.json"
    checkpoint_path = init_run / "checkpoints/best_mean_fit.pt"
    required_marker = (
        "cvae_posterior_standard_cvae_fixed_mean_fit.ok"
        if required_stage == "fixed"
        else "cvae_posterior_standard_cvae_random_physical_mean_fit.ok"
    )
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise ValueError("standard CVAE stage initialization artifacts are missing")
    summary = load_json(summary_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checks = {
        "stage": summary.get("stage") == required_stage,
        "quality": bool(summary.get("quality_pass")),
        "marker": (init_run / "markers" / required_marker).is_file(),
        "checkpoint_format": checkpoint.get("format_version") == CHECKPOINT_FORMAT,
        "checkpoint_stage": checkpoint.get("stage") == required_stage,
        "dataset": checkpoint.get("dataset_manifest_sha256") == dataset_hash,
        "windows": checkpoint.get("selected_windows_sha256") == window_hash,
        "fixed": checkpoint.get("fixed_fixture_sha256") == fixed_hash,
        "heldout": checkpoint.get("heldout_fixture_sha256") == heldout_hash,
        "source": checkpoint.get("source_checkpoint_sha256") == source_hash,
        "parameters": int(checkpoint.get("parameter_count", -1))
        == sum(parameter.numel() for parameter in model.parameters()),
        "signature": checkpoint.get("model_signature") == _model_signature(model_config),
    }
    if not all(checks.values()):
        raise ValueError(f"standard CVAE stage initialization failed: {checks}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return {
        "run": str(init_run),
        "summary": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "model_only": True,
        "optimizer_scheduler_rng_restored": False,
        "checks": checks,
    }


def render_plots(output_run: Path, records: list[dict[str, Any]]) -> dict[str, str]:
    training = [row for row in records if row["phase"] == "train"]
    evaluations = [row for row in records if row["phase"] == "evaluation"]
    training_path = output_run / "plots/training_curves.svg"
    _svg(
        training_path,
        "H50-SCVAE training",
        "Optimizer objective, reconstruction, alignment and KL; Value (log10 scale)",
        [
            ("Objective", [(row["optimizer_step"], row["losses"]["objective"]) for row in training], "#1f77b4"),
            ("Prior reconstruction", [(row["optimizer_step"], row["losses"].get("prior_reconstruction", 0.0)) for row in training], "#ff7f0e"),
            ("Posterior reconstruction", [(row["optimizer_step"], row["losses"].get("posterior_reconstruction", 0.0)) for row in training], "#2ca02c"),
            ("Mean alignment", [(row["optimizer_step"], row["losses"].get("alignment", 0.0)) for row in training], "#9467bd"),
            ("KL", [(row["optimizer_step"], row["losses"].get("kl", 0.0)) for row in training], "#d62728"),
        ],
    )
    alignment_path = output_run / "plots/q_p_alignment.svg"
    _svg(
        alignment_path,
        "q/p mean alignment",
        "Standardized RMSE on the active complete-token Mask bank; Value (log10 scale)",
        [
            ("Global", [(row["optimizer_step"], row["metrics"]["alignment"]["global_standardized_rmse"]) for row in evaluations], "#1f77b4"),
            ("Local", [(row["optimizer_step"], row["metrics"]["alignment"]["local_standardized_rmse"]) for row in evaluations], "#ff7f0e"),
        ],
    )
    mask_path = output_run / "plots/fixed_and_random_mask_breakdown.svg"
    latest_series: list[tuple[str, list[tuple[float, float]], str]] = []
    if evaluations:
        latest = evaluations[-1]["metrics"]
        colors = {"prior_mean": "#1f77b4", "posterior_mean": "#ff7f0e"}
        for path_name in ("prior_mean", "posterior_mean"):
            metrics = latest.get(path_name)
            if metrics:
                latest_series.append((
                    path_name,
                    [(index, max(case["worst_state_rmse"], case["worst_action_rmse"]))
                     for index, case in enumerate(metrics["cases"].values())],
                    colors[path_name],
                ))
    _svg(
        mask_path,
        "Latest Mask breakdown",
        "Worst-window State/Action RMSE by Mask slot; Value (log10 scale)",
        latest_series,
        x_label="Mask slot",
    )
    latent_path = output_run / "plots/latent_three_path_comparison.svg"
    _svg(
        latent_path,
        "Latent path comparison",
        "Mean-path fit scores; KL sample detail is stored in the manifest",
        [
            ("Prior mean", [(row["optimizer_step"], row["metrics"]["prior_mean"]["fit_gate"]["score"]) for row in evaluations], "#1f77b4"),
            ("Posterior mean", [(row["optimizer_step"], row["metrics"]["posterior_mean"]["fit_gate"]["score"]) for row in evaluations], "#ff7f0e"),
        ],
    )
    return {
        "training_curves": str(training_path),
        "fixed_and_random_mask_breakdown": str(mask_path),
        "q_p_alignment": str(alignment_path),
        "latent_three_path_comparison": str(latent_path),
    }


def run_experiment(
    dataset_run: Path,
    output_run: Path,
    source_run: Path,
    config: dict[str, Any],
    *,
    stage: str,
    init_run: Path | None,
    smoke: bool,
    kl_beta_override: float | None = None,
) -> dict[str, Any]:
    from .dataset import StateActionWindowDataset

    if stage not in STAGES:
        raise ValueError(f"standard CVAE stage must be one of {STAGES}")
    if smoke and stage != "fixed":
        raise ValueError("standard CVAE smoke exercises the fresh fixed-stage path")
    if stage in {"random", "kl"} and init_run is None:
        raise ValueError(f"standard CVAE {stage} requires an explicit prior stage run")
    if kl_beta_override is not None:
        if stage != "kl":
            raise ValueError("KL beta override is only valid for the KL stage")
        if float(kl_beta_override) not in {1e-4, 1e-3, 1e-2}:
            raise ValueError("KL beta override must be one of 1e-4, 1e-3, or 1e-2")
    dataset_run = dataset_run.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    source_run = source_run.expanduser().resolve()
    protected = [dataset_run, source_run]
    if init_run is not None:
        protected.append(init_run.expanduser().resolve())
    assert_output_isolated(output_run, protected)
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "plots"):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    if not (dataset_run / "markers/cvae_overfit_subset.ok").is_file():
        raise FileNotFoundError("H50-SCVAE requires the dedicated overfit subset")

    data = config["data"]
    training = config["training"]
    model_config = config["model"]
    if (int(data["motion_count"]), int(data["window_transitions"]), int(data["stride"])) != (32, 64, 64):
        raise ValueError("H50-SCVAE requires 32 motions, T64, stride64")
    if training["random"]["training_mixture"] != PHYSICAL_RANDOM_TRAINING_MIXTURE:
        raise ValueError("physical random training mixture differs from the locked protocol")
    if float(training["weight_decay"]) != 0.0:
        raise ValueError("H50-SCVAE requires weight decay zero")

    base = StateActionWindowDataset(
        dataset_run, "train", 64, 64, max_episodes=256, random_crop=False
    )
    motions = validate_motion_prefix(base, 32)
    if len(base.episodes) != 256 or len(motions) != 32:
        raise ValueError("H50-SCVAE requires exactly 32 motions x 8 variants")
    source_windows = window_identity_rows(base, range(len(base)))
    source_window_hash = rows_sha256(source_windows)
    selected = DeterministicWindowSubset(base, 2 if smoke else data.get("max_windows"))
    windows = window_identity_rows(base, selected.indices)
    window_hash = rows_sha256(windows)
    dataset_hash = file_sha256(dataset_run / "manifests/dataset_manifest.json")
    model_config["state_dim"] = base.state_dim
    checkpoint, source = validate_h50_a_source(
        dataset_run,
        source_run,
        dataset_hash=dataset_hash,
        window_hash=source_window_hash,
        model_config=model_config,
    )
    if checkpoint.get("model_signature") != _base_signature(model_config):
        raise ValueError("H50-A source signature differs from the standard CVAE base")

    with torch.device("meta"):
        meta_model = build_model(model_config)
    if not isinstance(meta_model, HierarchicalStandardCVAETransformer):
        raise TypeError("H50-SCVAE config built the wrong model")
    actual_parameters = sum(parameter.numel() for parameter in meta_model.parameters())
    logvar_parameters = sum(
        parameter.numel() for name, parameter in meta_model.named_parameters()
        if meta_model.is_logvar_parameter(name)
    )
    del meta_model
    if actual_parameters != int(model_config["total_parameter_count"]):
        raise ValueError("H50-SCVAE parameter count differs from the reference config")
    if logvar_parameters != int(model_config["logvar_parameter_count"]):
        raise ValueError("H50-SCVAE logvar parameter count differs from the reference config")

    seed_everything(int(config["initialization_seed"]))
    model = build_model(model_config)
    assert isinstance(model, HierarchicalStandardCVAETransformer)
    initialization = initialize_from_h50_a(model, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    workers = 0 if smoke else int(data["num_workers"])
    micro_batch = int(training["micro_batch"])

    base_data: Dataset[dict[str, Any]] = MaskBankDataset(selected, 1)
    base_loader = DataLoader(
        base_data, batch_size=micro_batch, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    fixed_data = MaskBankDataset(selected, len(PHYSICAL_MASK_NAMES))
    fixed_loader = DataLoader(
        fixed_data, batch_size=micro_batch, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    heldout_data = MaskBankDataset(selected, len(PHYSICAL_RANDOM_MASK_NAMES))
    heldout_loader = DataLoader(
        heldout_data, batch_size=micro_batch, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    fixed_seed = 20260830
    heldout_seed = int(config["heldout_mask_seed"])
    training_mask_seed = int(config["training_mask_seed"])
    fixed_maker = lambda value: make_physical_masks(value, fixed_seed)
    heldout_maker = lambda value: make_heldout_physical_random_masks(value, heldout_seed)
    fixed_hash = mask_bank_sha256(fixed_loader, fixed_maker)
    heldout_hash = mask_bank_sha256(heldout_loader, heldout_maker)
    physical_manifest = {
        "format_version": "sonic_h50_standard_cvae_physical_random_mask_bank_v1",
        "training_seed": training_mask_seed,
        "heldout_seed": heldout_seed,
        "training_mixture": PHYSICAL_RANDOM_TRAINING_MIXTURE,
        "fixed_mask_names": list(PHYSICAL_MASK_NAMES),
        "heldout_mask_names": list(PHYSICAL_RANDOM_MASK_NAMES),
        "fixed_fixture_count": len(fixed_data),
        "heldout_fixture_count": len(heldout_data),
        "fixed_fixture_sha256": fixed_hash,
        "heldout_fixture_sha256": heldout_hash,
        "full_state": False,
        "full_both": False,
        "granularity": "complete State/Action token",
    }
    atomic_write_json(
        output_run / "manifests/physical_random_mask_bank.json", physical_manifest
    )
    scale = build_initial_latent_scale(
        model, base_loader, device, output_run,
        source_hash=source["checkpoint_sha256"],
        dataset_hash=dataset_hash,
        window_hash=window_hash,
    )

    stage_initialization: dict[str, Any] | None = None
    if stage in {"random", "kl"}:
        assert init_run is not None
        stage_initialization = load_stage_initialization(
            model,
            init_run,
            required_stage="fixed" if stage == "random" else "random",
            dataset_hash=dataset_hash,
            window_hash=window_hash,
            fixed_hash=fixed_hash,
            heldout_hash=heldout_hash,
            source_hash=source["checkpoint_sha256"],
            model_config=model_config,
        )

    stage_contract = dict(training[stage])
    if stage == "kl":
        if float(stage_contract["free_bits"]) != 0.0:
            raise ValueError("H50-SCVAE K1 requires free bits zero")
        if kl_beta_override is not None:
            stage_contract["beta"] = float(kl_beta_override)
    maximum = 2 if smoke else int(stage_contract["max_optimizer_steps"])
    optimizer, scheduler, optimizer_contract = _parameter_groups(
        model, stage, stage_contract, maximum
    )
    fit_thresholds = {key: float(value) for key, value in training["fit_thresholds"].items()}
    alignment_thresholds = {
        key: float(value) for key, value in training["alignment_thresholds"].items()
    }
    evaluate_kwargs = {
        "fit_thresholds": fit_thresholds,
        "strict_thresholds": {
            "worst_state_rmse": 0.01, "worst_action_rmse": 0.01,
            "continuous_max_abs": 0.01, "contact_accuracy": 1.0,
            "latent_ratio": 10.0,
        },
        "exact_thresholds": {
            "worst_state_rmse": 1e-4, "worst_action_rmse": 1e-4,
            "continuous_max_abs": 1e-3, "contact_accuracy": 1.0,
            "latent_ratio": 10.0,
        },
        "state_std": torch.from_numpy(base.state_std),
        "action_std": torch.from_numpy(base.action_std),
    }
    metrics_path = output_run / "logs/metrics.jsonl"
    records: list[dict[str, Any]] = []
    last_sample_comparison: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score = math.inf
    best_rank = (math.inf, math.inf, math.inf)
    best_step = -1
    pass_streak = 0

    def run_evaluation(step: int, *, include_heldout: bool, include_samples: bool) -> dict[str, Any]:
        nonlocal last_sample_comparison
        started = time.perf_counter()
        active_loader = heldout_loader if include_heldout else fixed_loader
        active_maker = heldout_maker if include_heldout else fixed_maker
        prior_mean = evaluate(
            _PathView(model, "prior_mean"), active_loader, base_loader, device,
            active_maker, latent_diagnostics=False, report_full_sequence=True,
            **evaluate_kwargs,
        )
        posterior_mean = evaluate(
            _PathView(model, "posterior_mean"), active_loader, base_loader, device,
            active_maker, latent_diagnostics=False, report_full_sequence=True,
            **evaluate_kwargs,
        )
        alignment = evaluate_mean_alignment(
            model, active_loader, active_maker, scale, device, alignment_thresholds
        )
        distribution_statistics = evaluate_distribution_statistics(
            model, active_loader, active_maker, device
        )
        fixed_retention = None
        if include_heldout:
            fixed_retention = {
                "prior_mean": evaluate(
                    _PathView(model, "prior_mean"), fixed_loader, base_loader, device,
                    fixed_maker, latent_diagnostics=False, report_full_sequence=True,
                    **evaluate_kwargs,
                ),
                "posterior_mean": evaluate(
                    _PathView(model, "posterior_mean"), fixed_loader, base_loader, device,
                    fixed_maker, latent_diagnostics=False, report_full_sequence=True,
                    **evaluate_kwargs,
                ),
            }
        latent_usage = evaluate_high_mask_latent_usage(
            model, active_loader, active_maker, device,
            float(training["latent_usage"]["minimum_high_mask_ratio"]),
        )
        full_both_maker = lambda value: make_autoencode_masks(value)
        full_both_diagnostic = {
            "prior_mean": evaluate(
                _PathView(model, "prior_mean"), base_loader, base_loader, device,
                full_both_maker, latent_diagnostics=False,
                report_full_sequence=True, **evaluate_kwargs,
            ),
            "posterior_mean": evaluate(
                _PathView(model, "posterior_mean"), base_loader, base_loader, device,
                full_both_maker, latent_diagnostics=False,
                report_full_sequence=True, **evaluate_kwargs,
            ),
            "gate_role": "information-free diagnostic only; excluded from PASS",
        }
        full_contacts_pass = bool(
            prior_mean["full_sequence_reconstruction"]["contact_accuracy"] == 1.0
            and posterior_mean["full_sequence_reconstruction"]["contact_accuracy"] == 1.0
        )
        if fixed_retention is not None:
            full_contacts_pass = bool(
                full_contacts_pass
                and fixed_retention["prior_mean"]["full_sequence_reconstruction"]["contact_accuracy"] == 1.0
                and fixed_retention["posterior_mean"]["full_sequence_reconstruction"]["contact_accuracy"] == 1.0
            )
        mean_pass = bool(
            prior_mean["fit_gate"]["passed"]
            and posterior_mean["fit_gate"]["passed"]
            and alignment["gate"]["passed"]
            and distribution_statistics["finite"]
            and latent_usage["passed_for_kl_entry"]
            and full_contacts_pass
            and (
                fixed_retention is None
                or (
                    fixed_retention["prior_mean"]["fit_gate"]["passed"]
                    and fixed_retention["posterior_mean"]["fit_gate"]["passed"]
                )
            )
        )
        joint_score = max(
            float(prior_mean["fit_gate"]["score"]),
            float(posterior_mean["fit_gate"]["score"]),
            float(alignment["gate"]["score"]),
            1.0 if latent_usage["passed_for_kl_entry"] else 2.0,
            *(
                [] if fixed_retention is None else [
                    float(fixed_retention["prior_mean"]["fit_gate"]["score"]),
                    float(fixed_retention["posterior_mean"]["fit_gate"]["score"]),
                ]
            ),
        )
        if include_samples:
            last_sample_comparison = evaluate_sample_paths(
                model, active_loader, base_loader, active_maker, device,
                count=int(training["kl"]["sample_count"]),
                sample_seed=heldout_seed,
                evaluate_kwargs=evaluate_kwargs,
                posterior_mean=posterior_mean,
                contract=training["kl"],
            )
            last_sample_comparison["conditional_prior_output_diversity"] = (
                evaluate_prior_output_diversity(
                    model, active_loader, active_maker, device,
                    count=int(training["kl"]["sample_count"]),
                    sample_seed=heldout_seed,
                )
            )
            last_sample_comparison["passed"] = bool(
                last_sample_comparison["passed"]
                and last_sample_comparison[
                    "conditional_prior_output_diversity"
                ]["finite"]
            )
            atomic_write_json(
                output_run / "manifests/kl_three_path_comparison.json",
                {
                    "format_version": "sonic_h50_standard_cvae_three_path_v1",
                    "optimizer_step": step,
                    "posterior_mean": _metric_summary(posterior_mean),
                    "sample_comparison": last_sample_comparison,
                    "decoder_latent_isolation": (
                        "each path is a separate call with exactly one latent; "
                        "posterior and prior latents are never fused"
                    ),
                    "deployment_path": "condition -> p(z|c) -> z_p -> shared D(z_p,c)",
                },
            )
        sample_pass = bool(
            stage != "kl"
            or (last_sample_comparison is not None and last_sample_comparison["passed"])
        )
        result = {
            "optimizer_step": step,
            "evaluation_bank": "heldout_physical_random" if include_heldout else "fixed_physical",
            "prior_mean": prior_mean,
            "posterior_mean": posterior_mean,
            "alignment": alignment,
            "distribution_statistics": distribution_statistics,
            "fixed_retention": fixed_retention,
            "latent_usage": latent_usage,
            "full_both_diagnostic": full_both_diagnostic,
            "full_sequence_contacts_pass": full_contacts_pass,
            "mean_pass": mean_pass,
            "sample_comparison": last_sample_comparison if include_samples else None,
            "joint_gate": {
                "passed": bool(mean_pass and sample_pass),
                "score": max(joint_score, 1.0 if sample_pass else 2.0),
            },
            "evaluation_seconds": time.perf_counter() - started,
        }
        result["artifacts"] = write_evaluation_artifacts(
            output_run, step, prior_mean
        )
        row = {
            "phase": "evaluation", "stage": stage,
            "optimizer_step": step, "metrics": result,
        }
        records.append(row)
        append_jsonl(metrics_path, row)
        render_plots(output_run, records)
        return result

    include_heldout_at_zero = stage in {"random", "kl"}
    initial = run_evaluation(
        0,
        include_heldout=include_heldout_at_zero,
        include_samples=stage == "kl",
    )
    generator = torch.Generator().manual_seed(int(config["training_seed"]) + STAGES.index(stage))
    training_data: Dataset[dict[str, Any]] = (
        fixed_data if stage == "fixed" else base_data
    )
    train_loader = DataLoader(
        training_data, batch_size=micro_batch, shuffle=True, num_workers=workers,
        generator=generator, drop_last=not smoke, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    stream = _infinite(train_loader)
    accumulation = int(training["gradient_accumulation"])
    optimizer.zero_grad(set_to_none=True)
    completed_step = 0
    last_evaluation = initial
    for step in range(1, maximum + 1):
        completed_step = step
        started = time.perf_counter()
        aggregate = {
            "objective": 0.0, "prior_reconstruction": 0.0,
            "posterior_reconstruction": 0.0, "alignment": 0.0,
            "alignment_global": 0.0, "alignment_local": 0.0,
            "kl": 0.0, "kl_global": 0.0, "kl_local": 0.0,
        }
        sample_digest = hashlib.sha256()
        for sample_slot in range(accumulation):
            cpu_batch = next(stream)
            if stage == "fixed":
                state_mask, action_mask, names = fixed_maker(cpu_batch)
            else:
                state_mask, action_mask, names = make_dynamic_physical_random_masks(
                    cpu_batch, training_mask_seed, step, sample_slot
                )
                validate_physically_inferable_token_masks(
                    cpu_batch, state_mask, action_mask
                )
            sample_digest.update(state_mask.numpy().tobytes(order="C"))
            sample_digest.update(action_mask.numpy().tobytes(order="C"))
            sample_digest.update(canonical_json_bytes(names))
            batch = _device_batch(cpu_batch, device)
            state_mask = state_mask.to(device)
            action_mask = action_mask.to(device)
            posterior = model.encode_posterior_distribution(
                batch, state_mask, action_mask
            )
            prior = model.encode_conditional_prior_distribution(
                batch, state_mask, action_mask
            )
            if stage == "kl":
                global_latent, local_latents = model.reparameterize(posterior)
                posterior_output = model.decode_from_conditioned_latents(
                    batch, state_mask, action_mask, global_latent, local_latents
                )
                posterior_reconstruction = weighted_reconstruction_loss(
                    posterior_output, batch, state_mask, action_mask,
                    masked_weight=float(training["masked_reconstruction_weight"]),
                    full_weight=float(training["full_reconstruction_weight"]),
                )
                kl = hierarchical_kl(posterior, prior)
                beta = min(step / max(int(stage_contract["beta_warmup_steps"]), 1), 1.0) * float(stage_contract["beta"])
                objective = posterior_reconstruction["total"] + beta * kl["total"]
                prior_reconstruction = posterior_reconstruction
                alignment = {key: kl["total"] * 0.0 for key in ("total", "global", "local")}
            else:
                prior_output = model.decode_from_conditioned_latents(
                    batch, state_mask, action_mask,
                    prior.global_mean, prior.local_mean,
                )
                posterior_output = model.decode_from_conditioned_latents(
                    batch, state_mask, action_mask,
                    posterior.global_mean, posterior.local_mean,
                )
                prior_reconstruction = weighted_reconstruction_loss(
                    prior_output, batch, state_mask, action_mask,
                    masked_weight=float(training["masked_reconstruction_weight"]),
                    full_weight=float(training["full_reconstruction_weight"]),
                )
                posterior_reconstruction = weighted_reconstruction_loss(
                    posterior_output, batch, state_mask, action_mask,
                    masked_weight=float(training["masked_reconstruction_weight"]),
                    full_weight=float(training["full_reconstruction_weight"]),
                )
                alignment = mean_alignment_loss(
                    posterior, prior,
                    scale["global_std"].to(device), scale["local_std"].to(device),
                )
                lambda_z = (
                    alignment_weight(step, stage_contract["alignment_schedule"])
                    if stage == "fixed" else float(stage_contract["alignment_weight"])
                )
                objective = (
                    prior_reconstruction["total"]
                    + float(training["posterior_reconstruction_weight"])
                    * posterior_reconstruction["total"]
                    + lambda_z * alignment["total"]
                )
                kl = {key: objective * 0.0 for key in ("total", "global", "local")}
                beta = 0.0
            (objective / accumulation).backward()
            values = {
                "objective": objective,
                "prior_reconstruction": prior_reconstruction["total"],
                "posterior_reconstruction": posterior_reconstruction["total"],
                "alignment": alignment["total"],
                "alignment_global": alignment["global"],
                "alignment_local": alignment["local"],
                "kl": kl["total"], "kl_global": kl["global"], "kl_local": kl["local"],
            }
            for name, value in values.items():
                aggregate[name] += float(value.detach().cpu()) / accumulation
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        gradient = torch.nn.utils.clip_grad_norm_(
            trainable, float(training["gradient_clip"])
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        row = {
            "phase": "train", "stage": stage, "optimizer_step": step,
            "losses": aggregate,
            "alignment_weight": 0.0 if stage == "kl" else lambda_z,
            "kl_beta": beta,
            "learning_rates": {
                group["name"]: float(group["lr"]) for group in optimizer.param_groups
            },
            "gradient_norm_before_clip": float(gradient),
            "gradient_was_clipped": float(gradient) > float(training["gradient_clip"]),
            "sample_mask_sha256": sample_digest.hexdigest(),
            "step_seconds": time.perf_counter() - started,
            "cuda_max_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
            ),
        }
        records.append(row)
        append_jsonl(metrics_path, row)

        if smoke:
            should_evaluate = step == maximum
            include_heldout = False
            include_samples = False
        elif stage == "fixed":
            should_evaluate = step % int(stage_contract["validation_interval"]) == 0 or step == maximum
            include_heldout = False
            include_samples = False
        elif stage == "random":
            should_evaluate = step % int(stage_contract["fixed_validation_interval"]) == 0 or step == maximum
            include_heldout = step % int(stage_contract["heldout_validation_interval"]) == 0 or step == maximum
            include_samples = False
        else:
            should_evaluate = bool(
                step % int(stage_contract["mean_validation_interval"]) == 0
                or step % int(stage_contract["sample_validation_interval"]) == 0
                or step == maximum
            )
            include_heldout = True
            include_samples = step % int(stage_contract["sample_validation_interval"]) == 0 or step == maximum
        if not should_evaluate:
            continue
        last_evaluation = run_evaluation(
            step, include_heldout=include_heldout, include_samples=include_samples
        )
        eligible_evaluation = bool(
            stage == "fixed"
            or (stage == "random" and include_heldout)
            or (stage == "kl" and include_samples)
        )
        eligible_pass = bool(
            last_evaluation["joint_gate"]["passed"]
            and eligible_evaluation
        )
        if eligible_evaluation:
            pass_streak = pass_streak + 1 if eligible_pass else 0
        score = float(last_evaluation["joint_gate"]["score"])
        secondary = sum(
            float(last_evaluation[path][name])
            for path in ("prior_mean", "posterior_mean")
            for name in (
                "global_state_rmse", "global_action_rmse",
                "worst_mask_state_rmse", "worst_mask_action_rmse",
                "continuous_p99_abs",
            )
        )
        sample_secondary = (
            float(last_sample_comparison["paths"]["conditional_prior_sample"]["score"]["p95"])
            if stage == "kl" and include_samples and last_sample_comparison is not None
            else 0.0
        )
        rank = (score, sample_secondary, secondary)
        if eligible_evaluation and rank < best_rank:
            best_rank = rank
            best_score, best_metrics, best_step = score, last_evaluation, step
            atomic_torch_save(
                output_run / "checkpoints" / (
                    "best_kl_fit.pt" if stage == "kl" else "best_mean_fit.pt"
                ),
                _checkpoint(
                    model, optimizer, scheduler, config, stage=stage,
                    optimizer_step=step, dataset_hash=dataset_hash,
                    window_hash=window_hash, fixed_hash=fixed_hash,
                    heldout_hash=heldout_hash, source_hash=source["checkpoint_sha256"],
                ),
            )
        atomic_torch_save(
            output_run / "checkpoints/last.pt",
            _checkpoint(
                model, optimizer, scheduler, config, stage=stage,
                optimizer_step=step, dataset_hash=dataset_hash,
                window_hash=window_hash, fixed_hash=fixed_hash,
                heldout_hash=heldout_hash, source_hash=source["checkpoint_sha256"],
            ),
        )
        if not smoke and pass_streak >= int(training["required_pass_streak"]):
            break
        model.train()

    if best_metrics is None:
        best_metrics, best_score, best_step = initial, float(initial["joint_gate"]["score"]), 0
        best_rank = (best_score, math.inf, math.inf)
    evaluations = [row["metrics"] for row in records if row["phase"] == "evaluation"]
    quality_pass = bool(
        not smoke and pass_streak >= int(training["required_pass_streak"])
    )
    plots = render_plots(output_run, records)
    last_path = output_run / "checkpoints/last.pt"
    checkpoint_readback = validate_checkpoint(
        last_path,
        {
            "stage": stage, "dataset": dataset_hash, "windows": window_hash,
            "fixed": fixed_hash, "heldout": heldout_hash,
            "source": source["checkpoint_sha256"], "parameters": actual_parameters,
        },
    )
    if not checkpoint_readback["passed"]:
        raise RuntimeError("H50-SCVAE checkpoint readback failed")
    if smoke:
        next_step = "REVIEW_SMOKE_ARTIFACTS_THEN_RUN_FRESH_FIXED"
    elif stage == "fixed":
        next_step = "RUN_STANDARD_CVAE_RANDOM_PHYSICAL" if quality_pass else "STOP_FIXED_MEAN_QUALITY_FAILED"
    elif stage == "random":
        next_step = "RUN_STANDARD_CVAE_KL" if quality_pass else "STOP_RANDOM_PHYSICAL_MEAN_QUALITY_FAILED"
    else:
        next_step = "ACCEPT_PRIOR_ONLY_DEPLOYMENT_PATH" if quality_pass else "ASSESS_SINGLE_ALLOWED_KL_BETA_RERUN"
    summary = {
        "format_version": SUMMARY_FORMAT,
        "experiment": "H50-SCVAE standard conditional VAE",
        "stage": stage,
        "execution_pass": True,
        "smoke": smoke,
        "quality_pass": quality_pass,
        "completed_optimizer_steps": completed_step,
        "best_optimizer_step": best_step,
        "best_joint_score": best_score,
        "best_rank": list(best_rank),
        "dataset_run": str(dataset_run),
        "dataset_manifest_sha256": dataset_hash,
        "motion_count": 32,
        "episode_count": len(base.episodes),
        "selected_motion_keys": motions,
        "window_transitions": 64,
        "stride": 64,
        "window_count": len(selected),
        "selected_windows_sha256": window_hash,
        "physical_mask_bank": physical_manifest,
        "model_contract": {
            **model_config,
            "actual_parameter_count": actual_parameters,
            "actual_logvar_parameter_count": logvar_parameters,
            "posterior_prior_latent_fusion": False,
            "shared_decoder_parameters": True,
            "separate_decoder_calls_during_training": True,
            "decoder_receives_real_masked_condition": True,
            "deployment_calls_posterior": False,
            "deployment_path": "condition -> p(z|c) -> z_p -> shared D(z_p,c)",
        },
        "source": source,
        "initialization": initialization,
        "stage_initialization": stage_initialization,
        "initial_latent_scale": scale["manifest"],
        "optimizer_contract": optimizer_contract,
        "effective_kl_beta": float(stage_contract["beta"]) if stage == "kl" else 0.0,
        "kl_beta_override": kl_beta_override,
        "initial_evaluation": initial,
        "best_evaluation": best_metrics,
        "last_evaluation": last_evaluation,
        "last_three_evaluations": evaluations[-3:],
        "kl_three_path_comparison": last_sample_comparison,
        "checkpoint_readback": checkpoint_readback,
        "plots": plots,
        "unique_next_step": next_step,
        "scope": (
            "seen 32-motion T64 complete-token physical Mask completion; "
            "no unseen-motion or exhaustive-Mask claim"
        ),
    }
    atomic_write_json(output_run / "manifests/standard_cvae_summary.json", summary)
    if smoke:
        atomic_write_text(output_run / "markers/cvae_posterior_standard_cvae_smoke.ok", "PASS\n")
    else:
        atomic_write_text(output_run / "markers/cvae_posterior_standard_cvae_execution.ok", "PASS\n")
        marker = {
            "fixed": "cvae_posterior_standard_cvae_fixed_mean_fit.ok",
            "random": "cvae_posterior_standard_cvae_random_physical_mean_fit.ok",
            "kl": "cvae_posterior_standard_cvae_kl_fit.ok",
        }[stage]
        if stage == "kl" and last_sample_comparison is not None:
            atomic_write_text(output_run / "markers/cvae_posterior_standard_cvae_kl_comparison.ok", "PASS\n")
        if quality_pass:
            atomic_write_text(output_run / "markers" / marker, "PASS\n")
        else:
            atomic_write_text(
                output_run / "markers/cvae.failed",
                f"QUALITY_FAIL execution_complete=true stage={stage}\n",
            )
    base.close()
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run H50 standard conditional VAE")
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=root / "configs/posterior_hierarchical_standard_cvae_h50.json",
    )
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--init-run", type=Path)
    parser.add_argument("--kl-beta", type=float)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_experiment(
        args.dataset_run,
        args.output_run,
        args.source_run,
        load_config(args.config),
        stage=args.stage,
        init_run=args.init_run,
        smoke=args.smoke,
        kl_beta_override=args.kl_beta,
    )
    print("Posterior H50-SCVAE: PASS (execution complete)")
    print(json.dumps({
        "output_run": str(args.output_run.expanduser().resolve()),
        "stage": args.stage,
        "smoke": args.smoke,
        "quality_pass": summary["quality_pass"],
        "completed_optimizer_steps": summary["completed_optimizer_steps"],
        "best_joint_score": summary["best_joint_score"],
        "unique_next_step": summary["unique_next_step"],
    }, indent=2))
    print(args.output_run.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
