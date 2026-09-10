from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .models import (
    HierarchicalConditionalPriorTransformer,
    build_model,
)
from .posterior_capacity import DeterministicWindowSubset, MaskBankDataset, validate_motion_prefix
from .posterior_complete_token_protocol import (
    RANDOM_TOKEN_MASK_NAMES,
    TRAINING_MIXTURE,
    make_dynamic_random_token_masks,
    make_heldout_random_token_masks,
    validate_complete_token_masks,
)
from .posterior_direct_output import assert_output_isolated
from .posterior_t64_protocol import (
    PHYSICAL_MASK_NAMES,
    _donor_maps,
    _svg,
    append_jsonl,
    deterministic_quantile,
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


CHECKPOINT_FORMAT = "sonic_h50_cpd_checkpoint_v1"
SUMMARY_FORMAT = "sonic_h50_cpd_summary_v1"
TEACHER_CACHE_FORMAT = "sonic_h50_cpd_teacher_cache_v1"
MODES = ("train", "adapt")


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _infinite(loader: DataLoader[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def _tensor_mapping_sha256(mapping: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(mapping, key=lambda row: row[0]):
        tensor = value.detach().cpu().contiguous()
        digest.update(canonical_json_bytes({
            "name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)
        }))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def base_state_sha256(model: HierarchicalConditionalPriorTransformer) -> str:
    return _tensor_mapping_sha256(
        (name, value)
        for name, value in model.state_dict().items()
        if not model.is_conditional_prior_parameter(name)
    )


def prior_state_sha256(model: HierarchicalConditionalPriorTransformer) -> str:
    return _tensor_mapping_sha256(
        (name, value)
        for name, value in model.state_dict().items()
        if model.is_conditional_prior_parameter(name)
    )


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
    result = _base_signature(model_config)
    result.update({
        "kind": "physics_hierarchical_conditional_prior_transformer",
        "profile": "H50-CPD",
        "conditional_prior_encoder_layers": model_config.get(
            "conditional_prior_encoder_layers"
        ),
    })
    return result


def validate_h50_a_source(
    dataset_run: Path,
    source_run: Path,
    *,
    dataset_hash: str,
    window_hash: str,
    model_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_run = source_run.expanduser().resolve()
    summary_path = source_run / "manifests/posterior_hierarchical_t64_summary.json"
    checkpoint_path = source_run / "checkpoints/last.pt"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise ValueError("H50-CPD requires the registered H50-A continuation last.pt")
    summary = load_json(summary_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_step = int(model_config["source_optimizer_step"])
    checks = {
        "source_run_name": source_run.name == str(model_config["source_run_name"]),
        "profile": summary.get("profile") == "H50",
        "stage": summary.get("stage") == "autoencode",
        "formal": not bool(summary.get("smoke", True)),
        "execution_pass": bool(summary.get("execution_pass")),
        "quality_pass": bool(summary.get("quality_pass")),
        "continuation": bool(summary.get("continuation", {}).get("enabled")),
        "completed_step": int(summary.get("completed_optimizer_steps", -1)) == expected_step,
        "dataset_path": Path(str(summary.get("dataset_run", ""))).resolve()
        == dataset_run.expanduser().resolve(),
        "dataset_hash": summary.get("dataset_manifest_sha256") == dataset_hash,
        "window_hash": summary.get("selected_windows_sha256") == window_hash,
        "execution_marker": (
            source_run / "markers/cvae_posterior_hierarchical_t64_execution.ok"
        ).is_file(),
        "continuation_marker": (
            source_run
            / "markers/cvae_posterior_hierarchical_t64_continuation_execution.ok"
        ).is_file(),
        "fit_marker": (
            source_run / "markers/cvae_posterior_hierarchical_t64_autoencode_fit.ok"
        ).is_file(),
        "checkpoint_format": checkpoint.get("format_version")
        == "sonic_posterior_hierarchical_t64_checkpoint_v1",
        "checkpoint_stage": checkpoint.get("stage") == "autoencode",
        "checkpoint_step": int(checkpoint.get("optimizer_step", -1)) == expected_step,
        "checkpoint_dataset_hash": checkpoint.get("dataset_manifest_sha256") == dataset_hash,
        "checkpoint_window_hash": checkpoint.get("selected_windows_sha256") == window_hash,
        "checkpoint_signature": checkpoint.get("model_signature")
        == _base_signature(model_config),
        "checkpoint_parameters": int(checkpoint.get("parameter_count", -1))
        == int(model_config["base_parameter_count"]),
        "model_state": isinstance(checkpoint.get("model"), dict),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"H50-A CPD source failed: {failed}")
    return checkpoint, {
        "run": str(source_run),
        "summary": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "optimizer_step": expected_step,
        "checks": checks,
        "model_only": True,
        "optimizer_scheduler_rng_restored": False,
    }


def initialize_from_h50_a(
    model: HierarchicalConditionalPriorTransformer,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    expected_missing = {
        name
        for name in model.state_dict()
        if model.is_conditional_prior_parameter(name)
    }
    checks = {
        "unexpected_keys_empty": not incompatible.unexpected_keys,
        "only_prior_keys_missing": set(incompatible.missing_keys) == expected_missing,
    }
    if not all(checks.values()):
        raise ValueError(f"H50-A to CPD migration failed: {checks}")
    model.initialize_conditional_prior_from_posterior()
    with torch.no_grad():
        checks.update({
            "state_value_columns_copied": torch.equal(
                model.conditional_prior_state_input.weight[:, : model.state_dim],
                model.state_input.weight[:, : model.state_dim],
            ),
            "action_value_columns_copied": torch.equal(
                model.conditional_prior_action_input.weight[:, :29],
                model.action_input.weight[:, :29],
            ),
            "state_mask_columns_zero": bool(
                torch.count_nonzero(
                    model.conditional_prior_state_input.weight[:, model.state_dim :]
                ) == 0
            ),
            "action_mask_columns_zero": bool(
                torch.count_nonzero(
                    model.conditional_prior_action_input.weight[:, 29:]
                ) == 0
            ),
        })
    if not all(checks.values()):
        raise RuntimeError(f"conditional prior initialization failed: {checks}")
    return {
        "strategy": "copy H50-A posterior; zero only the new Mask-input columns",
        "checks": checks,
        "prior_initialization_sha256": prior_state_sha256(model),
    }


def _lr_multiplier(step: int, warmup: int, maximum: int, minimum_ratio: float) -> float:
    if step < warmup:
        return max((step + 1) / max(warmup, 1), 1e-8)
    progress = min((step - warmup) / max(maximum - warmup, 1), 1.0)
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def phase_weights(step: int, phases: list[dict[str, Any]]) -> tuple[str, float, float]:
    for phase in phases:
        start, end = int(phase["start"]), int(phase["end"])
        if start < step <= end or (step == 0 and start == 0):
            fraction = min(max((step - start) / max(end - start, 1), 0.0), 1.0)
            latent = float(phase["latent_start"]) + fraction * (
                float(phase["latent_end"]) - float(phase["latent_start"])
            )
            reconstruction = float(phase["reconstruction_start"]) + fraction * (
                float(phase["reconstruction_end"])
                - float(phase["reconstruction_start"])
            )
            return str(phase["name"]), latent, reconstruction
    raise ValueError(f"optimizer step {step} is outside the registered CPD phases")


def configure_prior_optimizer(
    model: HierarchicalConditionalPriorTransformer,
    training: dict[str, Any],
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, dict[str, Any]]:
    counts = model.set_training_phase("prior")
    named = dict(model.named_parameters())
    head_prefixes = (
        "conditional_prior_state_input.", "conditional_prior_action_input.",
        "conditional_prior_type_embedding.", "conditional_prior_time_embedding.",
        "conditional_prior_cls", "conditional_prior_global_head.",
        "conditional_prior_local_head.", "conditional_prior_empty_local",
    )
    heads = [
        parameter for name, parameter in named.items()
        if parameter.requires_grad and any(name.startswith(prefix) for prefix in head_prefixes)
    ]
    encoder = [
        parameter for name, parameter in named.items()
        if parameter.requires_grad and name.startswith("conditional_prior_encoder.")
    ]
    if sum(parameter.numel() for parameter in heads + encoder) != counts["trainable"]:
        raise RuntimeError("conditional prior optimizer does not cover its allowlist exactly")
    encoder_lr = float(training["prior_encoder_learning_rate"])
    head_lr = float(training["prior_input_head_learning_rate"])
    minimum = float(training["minimum_learning_rate"])
    maximum = int(training["phases"][-1]["end"])
    warmup = int(training["warmup_steps"])
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": encoder_lr, "name": "conditional_prior_encoder"},
        {"params": heads, "lr": head_lr, "name": "conditional_prior_input_heads"},
    ], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [
            lambda step: _lr_multiplier(step, warmup, maximum, minimum / encoder_lr),
            lambda step: _lr_multiplier(step, warmup, maximum, minimum / head_lr),
        ],
    )
    return optimizer, scheduler, {"phase": "prior", "counts": counts}


def configure_adaptation_optimizer(
    model: HierarchicalConditionalPriorTransformer,
    training: dict[str, Any],
    stage: str,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, dict[str, Any]]:
    contract = training["decoder_adaptation"][stage]
    mode = "decoder_interface" if stage == "D1" else "decoder_full"
    counts = model.set_training_phase(mode)
    named = dict(model.named_parameters())
    prior = [p for n, p in named.items() if p.requires_grad and model.is_conditional_prior_parameter(n)]
    interface_prefixes = (
        "global_memory_projection.", "local_memory_projection.",
        "film_global_projection.", "film_local_projection.", "film_projection.",
    )
    interface = [
        p for n, p in named.items()
        if p.requires_grad and (
            any(n.startswith(prefix) for prefix in interface_prefixes)
            or (
                n.startswith("decoder.layers.")
                and (".cross_attention." in n or ".cross_query_norm." in n or ".cross_memory_norm." in n)
            )
        )
    ]
    selected = {id(parameter) for parameter in prior + interface}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in selected]
    groups = [
        {"params": prior, "lr": float(contract["prior_learning_rate"]), "name": "conditional_prior"},
        {"params": interface, "lr": float(contract["decoder_interface_learning_rate"]), "name": "decoder_latent_interface"},
    ]
    if stage == "D2":
        groups.append({
            "params": other,
            "lr": float(contract["other_decoder_learning_rate"]),
            "name": "decoder_other",
        })
    if sum(p.numel() for group in groups for p in group["params"]) != counts["trainable"]:
        raise RuntimeError("decoder adaptation optimizer does not cover its allowlist exactly")
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    maximum = int(contract["max_optimizer_steps"])
    warmup = int(training["decoder_adaptation"]["warmup_steps"])
    minimum = float(training["decoder_adaptation"]["minimum_learning_rate"])
    peaks = [float(group["lr"]) for group in groups]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [
            lambda step, peak=peak: _lr_multiplier(
                step, warmup, maximum, minimum / peak
            )
            for peak in peaks
        ],
    )
    return optimizer, scheduler, {"phase": stage, "counts": counts}


def full_and_masked_reconstruction_loss(
    output: Any,
    batch: dict[str, torch.Tensor],
    state_mask: torch.Tensor,
    action_mask: torch.Tensor,
) -> dict[str, Any]:
    full_state = batch["valid_state"].bool()[..., None].expand_as(batch["physical_state"])
    full_action = batch["valid_action"].bool()[..., None].expand_as(batch["action"])
    full = reconstruction_loss(output, batch, full_state, full_action)
    masked = reconstruction_loss(output, batch, state_mask, action_mask)
    return {
        "total": 0.5 * full["total"] + 0.5 * masked["total"],
        "state": 0.5 * full["state"] + 0.5 * masked["state"],
        "action": 0.5 * full["action"] + 0.5 * masked["action"],
        "contact": 0.5 * full["contact"] + 0.5 * masked["contact"],
        "full": full,
        "masked": masked,
    }


def latent_distillation_loss(
    predicted_global: torch.Tensor,
    predicted_local: torch.Tensor,
    teacher_global: torch.Tensor,
    teacher_local: torch.Tensor,
    global_std: torch.Tensor,
    local_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    global_loss = torch.square(
        (predicted_global - teacher_global) / global_std
    ).mean()
    local_loss = torch.square(
        (predicted_local - teacher_local) / local_std
    ).mean()
    return {
        "total": 0.5 * (global_loss + local_loss),
        "global": global_loss,
        "local": local_loss,
    }


@torch.no_grad()
def build_teacher_cache(
    model: HierarchicalConditionalPriorTransformer,
    base_loader: Iterable[dict[str, Any]],
    device: torch.device,
    output_run: Path,
    *,
    source_hash: str,
    dataset_hash: str,
    window_hash: str,
) -> dict[str, Any]:
    model.eval()
    globals_: list[torch.Tensor] = []
    locals_: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    window_indices: list[torch.Tensor] = []
    canonical_decoder_equivalent = True
    posterior_mask_invariant = True
    checked_alternate_mask = False
    for cpu_batch in base_loader:
        batch = _device_batch(cpu_batch, device)
        state_mask, action_mask, _ = make_autoencode_masks(cpu_batch)
        global_latent, local_latents = model.encode_posterior(
            batch, state_mask.to(device), action_mask.to(device)
        )
        decoded = model.decode_from_canonical_latents(
            global_latent,
            local_latents,
            valid_state=batch["valid_state"],
            valid_action=batch["valid_action"],
        )
        original = model.decode_from_hierarchical_latent(
            batch,
            state_mask.to(device),
            action_mask.to(device),
            global_latent,
            local_latents,
        )
        canonical_decoder_equivalent = bool(
            canonical_decoder_equivalent
            and torch.equal(decoded.physical_state, original.physical_state)
            and torch.equal(decoded.action, original.action)
            and torch.equal(
                decoded.state_contact_logits, original.state_contact_logits
            )
        )
        if not checked_alternate_mask:
            alternate_state, alternate_action, _ = make_physical_masks(
                cpu_batch, 20260830
            )
            alternate_global, alternate_local = model.encode_posterior(
                batch, alternate_state.to(device), alternate_action.to(device)
            )
            posterior_mask_invariant = bool(
                torch.equal(global_latent, alternate_global)
                and torch.equal(local_latents, alternate_local)
            )
            checked_alternate_mask = True
        globals_.append(global_latent.cpu())
        locals_.append(local_latents.cpu())
        states.append(decoded.physical_state.cpu())
        actions.append(decoded.action.cpu())
        logits.append(decoded.state_contact_logits.cpu())
        window_indices.append(cpu_batch["window_index"].long())
    global_all = torch.cat(globals_)
    local_all = torch.cat(locals_)
    indices = torch.cat(window_indices)
    expected = torch.arange(len(indices))
    if not torch.equal(indices, expected):
        raise RuntimeError("teacher cache lost deterministic window order")
    if not canonical_decoder_equivalent or not posterior_mask_invariant:
        raise RuntimeError(
            "H50-A canonical teacher/decoder invariants failed before CPD training"
        )
    cache = {
        "format_version": TEACHER_CACHE_FORMAT,
        "source_checkpoint_sha256": source_hash,
        "dataset_manifest_sha256": dataset_hash,
        "selected_windows_sha256": window_hash,
        "global_latent": global_all,
        "local_latents": local_all,
        "physical_state": torch.cat(states),
        "action": torch.cat(actions),
        "state_contact_logits": torch.cat(logits),
        "global_mean": global_all.mean(dim=0),
        "global_std": global_all.std(dim=0, unbiased=False).clamp_min(1e-3),
        "local_mean": local_all.mean(dim=0),
        "local_std": local_all.std(dim=0, unbiased=False).clamp_min(1e-3),
    }
    path = output_run / "data/teacher_latent_cache.pt"
    atomic_torch_save(path, cache)
    tensor_hash = file_sha256(path)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    roundtrip_checks = {
        name: torch.equal(cache[name], reloaded[name])
        for name in (
            "global_latent",
            "local_latents",
            "physical_state",
            "action",
            "state_contact_logits",
            "global_mean",
            "global_std",
            "local_mean",
            "local_std",
        )
    }
    if not all(roundtrip_checks.values()):
        raise RuntimeError("teacher latent/output cache failed exact readback")
    manifest = {
        "format_version": TEACHER_CACHE_FORMAT,
        "path": str(path),
        "sha256": tensor_hash,
        "source_checkpoint_sha256": source_hash,
        "dataset_manifest_sha256": dataset_hash,
        "selected_windows_sha256": window_hash,
        "window_count": len(indices),
        "global_shape": list(global_all.shape),
        "local_shape": list(local_all.shape),
        "standard_deviation_floor": 1e-3,
        "canonical_decoder_equivalent_to_full_both": canonical_decoder_equivalent,
        "posterior_latent_mask_invariant": posterior_mask_invariant,
        "exact_tensor_readback": roundtrip_checks,
    }
    atomic_write_json(output_run / "manifests/teacher_latent_cache.json", manifest)
    cache["manifest"] = manifest
    return cache


def teacher_for_batch(
    cache: dict[str, Any], batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = batch["window_index"].long()
    return (
        cache["global_latent"][indices].to(device),
        cache["local_latents"][indices].to(device),
    )


def _latent_gate(metrics: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
    ratios = {
        "global_standardized_rmse": metrics["global_standardized_rmse"]
        / thresholds["global_standardized_rmse"],
        "local_standardized_rmse": metrics["local_standardized_rmse"]
        / thresholds["local_standardized_rmse"],
        "global_cosine": thresholds["global_cosine"]
        / max(metrics["global_cosine"], 1e-12),
        "local_cosine": thresholds["local_cosine"]
        / max(metrics["local_cosine"], 1e-12),
        "cross_window_error_ratio": metrics["cross_window_error_ratio"]
        / thresholds["cross_window_error_ratio"],
        "cross_motion_error_ratio": metrics["cross_motion_error_ratio"]
        / thresholds["cross_motion_error_ratio"],
    }
    score = max(ratios.values())
    return {
        "passed": bool(math.isfinite(score) and score <= 1.0),
        "score": score,
        "thresholds": thresholds,
        "threshold_ratios": ratios,
    }


@torch.no_grad()
def evaluate_latent_alignment(
    model: HierarchicalConditionalPriorTransformer,
    loader: Iterable[dict[str, Any]],
    cache: dict[str, Any],
    device: torch.device,
    donor_maps: dict[str, list[int]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    model.eval()
    global_sse = local_sse = 0.0
    global_count = local_count = 0
    global_cosine: list[torch.Tensor] = []
    local_cosine: list[torch.Tensor] = []
    target_combined = cross_window_combined = cross_motion_combined = 0.0
    combined_count = 0
    global_std = cache["global_std"].to(device)
    local_std = cache["local_std"].to(device)
    for cpu_batch in loader:
        state_mask, action_mask, _ = make_heldout_random_token_masks(
            cpu_batch, int(cache["heldout_seed"])
        )
        batch = _device_batch(cpu_batch, device)
        predicted_global, predicted_local = model.encode_conditional_prior(
            batch, state_mask.to(device), action_mask.to(device)
        )
        teacher_global, teacher_local = teacher_for_batch(cache, cpu_batch, device)
        global_error = (predicted_global - teacher_global) / global_std
        local_error = (predicted_local - teacher_local) / local_std
        global_sse += float(torch.square(global_error).sum().cpu())
        local_sse += float(torch.square(local_error).sum().cpu())
        global_count += global_error.numel()
        local_count += local_error.numel()
        global_cosine.append(F.cosine_similarity(predicted_global, teacher_global, dim=-1).cpu())
        local_cosine.append(
            F.cosine_similarity(predicted_local, teacher_local, dim=-1).mean(dim=-1).cpu()
        )
        indices = cpu_batch["window_index"].tolist()
        cw = torch.tensor([donor_maps["cross_window"][int(i)] for i in indices])
        cm = torch.tensor([donor_maps["cross_motion"][int(i)] for i in indices])
        cw_global = cache["global_latent"][cw].to(device)
        cw_local = cache["local_latents"][cw].to(device)
        cm_global = cache["global_latent"][cm].to(device)
        cm_local = cache["local_latents"][cm].to(device)
        target_combined += float(torch.square(global_error).sum().cpu())
        target_combined += float(torch.square(local_error).sum().cpu())
        cross_window_combined += float(
            torch.square((cw_global - teacher_global) / global_std).sum().cpu()
            + torch.square((cw_local - teacher_local) / local_std).sum().cpu()
        )
        cross_motion_combined += float(
            torch.square((cm_global - teacher_global) / global_std).sum().cpu()
            + torch.square((cm_local - teacher_local) / local_std).sum().cpu()
        )
        combined_count += global_error.numel() + local_error.numel()
    target_rmse = math.sqrt(target_combined / combined_count)
    cross_window_rmse = math.sqrt(cross_window_combined / combined_count)
    cross_motion_rmse = math.sqrt(cross_motion_combined / combined_count)
    result = {
        "global_standardized_rmse": math.sqrt(global_sse / global_count),
        "local_standardized_rmse": math.sqrt(local_sse / local_count),
        "global_cosine": float(torch.cat(global_cosine).mean()),
        "local_cosine": float(torch.cat(local_cosine).mean()),
        "combined_standardized_rmse": target_rmse,
        "cross_window_donor_rmse": cross_window_rmse,
        "cross_motion_donor_rmse": cross_motion_rmse,
        "cross_window_error_ratio": target_rmse / max(cross_window_rmse, 1e-12),
        "cross_motion_error_ratio": target_rmse / max(cross_motion_rmse, 1e-12),
        "sample_count": len(loader.dataset),
    }
    result["gate"] = _latent_gate(result, thresholds)
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
def evaluate_latent_dependence(
    model: HierarchicalConditionalPriorTransformer,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    donor_maps: dict[str, list[int]],
    heldout_seed: int,
) -> dict[str, Any]:
    """Use one fixed random-both-65 query per window for donor diagnostics."""
    model.eval()
    globals_: list[torch.Tensor] = []
    locals_: list[torch.Tensor] = []
    batches: list[tuple[dict[str, Any], torch.Tensor, torch.Tensor]] = []
    for cpu_batch in loader:
        query_batch = dict(cpu_batch)
        query_batch["mask_slot"] = torch.full_like(cpu_batch["mask_slot"], 10)
        state_mask, action_mask, _ = make_heldout_random_token_masks(
            query_batch, heldout_seed
        )
        batch = _device_batch(cpu_batch, device)
        global_latent, local_latents = model.encode_conditional_prior(
            batch, state_mask.to(device), action_mask.to(device)
        )
        globals_.append(global_latent.cpu())
        locals_.append(local_latents.cpu())
        batches.append((cpu_batch, state_mask, action_mask))
    global_all = torch.cat(globals_)
    local_all = torch.cat(locals_)
    sums = {name: [0.0, 0] for name in ("correct", "zero", "cross_window", "cross_motion")}
    offset = 0
    for cpu_batch, state_mask, action_mask in batches:
        size = len(cpu_batch["motion_key"])
        indices = list(range(offset, offset + size))
        cw = [donor_maps["cross_window"][index] for index in indices]
        cm = [donor_maps["cross_motion"][index] for index in indices]
        choices = {
            "correct": (global_all[indices], local_all[indices]),
            "zero": (torch.zeros_like(global_all[indices]), torch.zeros_like(local_all[indices])),
            "cross_window": (global_all[cw], local_all[cw]),
            "cross_motion": (global_all[cm], local_all[cm]),
        }
        batch = _device_batch(cpu_batch, device)
        for name, (global_latent, local_latents) in choices.items():
            output = model.decode_from_canonical_latents(
                global_latent.to(device),
                local_latents.to(device),
                valid_state=batch["valid_state"],
                valid_action=batch["valid_action"],
            )
            total, count = _continuous_sse(
                output, batch, state_mask.to(device), action_mask.to(device)
            )
            sums[name][0] += total
            sums[name][1] += count
        offset += size
    rmse = {name: math.sqrt(total / count) for name, (total, count) in sums.items()}
    correct = max(rmse["correct"], 1e-12)
    ratios = {name: value / correct for name, value in rmse.items() if name != "correct"}
    return {
        "query": "held-out random_both_65",
        "rmse_excluding_contact": rmse,
        "main_ratios": ratios,
        "passed": all(ratios[name] >= 10.0 for name in ("zero", "cross_window", "cross_motion")),
    }


@torch.no_grad()
def evaluate_full_both_conditional_prior(
    model: HierarchicalConditionalPriorTransformer,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Report the deterministic no-information prior without gating quality."""
    model.eval()
    state_sse = action_sse = 0.0
    state_count = action_count = 0
    contact_correct = contact_count = 0
    absolute: list[torch.Tensor] = []
    for cpu_batch in loader:
        state_mask, action_mask, _ = make_autoencode_masks(cpu_batch)
        batch = _device_batch(cpu_batch, device)
        output = model(batch, state_mask.to(device), action_mask.to(device))
        state_error = (
            output.physical_state[..., :68] - batch["physical_state"][..., :68]
        ).masked_select(state_mask[..., :68].to(device))
        action_error = (output.action - batch["action"]).masked_select(
            action_mask.to(device)
        )
        state_sse += float(torch.square(state_error).sum().cpu())
        action_sse += float(torch.square(action_error).sum().cpu())
        state_count += state_error.numel()
        action_count += action_error.numel()
        absolute.extend((state_error.abs().cpu(), action_error.abs().cpu()))
        contact_mask = state_mask[..., 68:70].to(device)
        predicted = output.state_contact_logits.sigmoid() >= 0.5
        target = batch["physical_state"][..., 68:70] >= 0.5
        contact_correct += int((predicted == target).masked_select(contact_mask).sum().cpu())
        contact_count += int(contact_mask.sum().cpu())
    values = torch.cat(absolute)
    return {
        "global_state_rmse": math.sqrt(state_sse / state_count),
        "global_action_rmse": math.sqrt(action_sse / action_count),
        "continuous_p99_abs": deterministic_quantile(values, 0.99),
        "continuous_max_abs": float(values.max()),
        "contact_accuracy": contact_correct / contact_count,
        "gate_role": "non-identifiability diagnostic only; excluded from every PASS decision",
    }


@torch.no_grad()
def evaluate_teacher_latent_dependence(
    model: HierarchicalConditionalPriorTransformer,
    loader: Iterable[dict[str, Any]],
    cache: dict[str, Any],
    device: torch.device,
    donor_maps: dict[str, list[int]],
) -> dict[str, Any]:
    """Check that an adapted decoder still uses the canonical teacher latent."""
    model.eval()
    sums = {
        name: [0.0, 0]
        for name in ("correct", "zero", "cross_window", "cross_motion")
    }
    for cpu_batch in loader:
        batch = _device_batch(cpu_batch, device)
        indices = cpu_batch["window_index"].long()
        cross_window = torch.tensor(
            [donor_maps["cross_window"][int(index)] for index in indices]
        )
        cross_motion = torch.tensor(
            [donor_maps["cross_motion"][int(index)] for index in indices]
        )
        teacher_global, teacher_local = teacher_for_batch(cache, cpu_batch, device)
        choices = {
            "correct": (teacher_global, teacher_local),
            "zero": (torch.zeros_like(teacher_global), torch.zeros_like(teacher_local)),
            "cross_window": (
                cache["global_latent"][cross_window].to(device),
                cache["local_latents"][cross_window].to(device),
            ),
            "cross_motion": (
                cache["global_latent"][cross_motion].to(device),
                cache["local_latents"][cross_motion].to(device),
            ),
        }
        state_mask = batch["valid_state"].bool()[..., None].expand_as(
            batch["physical_state"]
        )
        action_mask = batch["valid_action"].bool()[..., None].expand_as(batch["action"])
        for name, (global_latent, local_latents) in choices.items():
            output = model.decode_from_canonical_latents(
                global_latent,
                local_latents,
                valid_state=batch["valid_state"],
                valid_action=batch["valid_action"],
            )
            total, count = _continuous_sse(output, batch, state_mask, action_mask)
            sums[name][0] += total
            sums[name][1] += count
    rmse = {name: math.sqrt(total / count) for name, (total, count) in sums.items()}
    correct = max(rmse["correct"], 1e-12)
    ratios = {
        name: value / correct for name, value in rmse.items() if name != "correct"
    }
    return {"rmse_excluding_contact": rmse, "main_ratios": ratios}


@torch.no_grad()
def evaluate_teacher_preservation(
    model: HierarchicalConditionalPriorTransformer,
    loader: Iterable[dict[str, Any]],
    cache: dict[str, Any],
    device: torch.device,
    fit_thresholds: dict[str, float],
    adaptation: dict[str, Any],
    donor_maps: dict[str, list[int]],
) -> dict[str, Any]:
    model.eval()
    truth_state_sse = truth_action_sse = 0.0
    state_count = action_count = 0
    worst_state = worst_action = 0.0
    functional_state_sse = functional_action_sse = 0.0
    truth_abs: list[torch.Tensor] = []
    functional_abs: list[torch.Tensor] = []
    contact_correct = contact_count = 0
    functional_contact_correct = functional_contact_count = 0
    for cpu_batch in loader:
        batch = _device_batch(cpu_batch, device)
        teacher_global, teacher_local = teacher_for_batch(cache, cpu_batch, device)
        output = model.decode_from_canonical_latents(
            teacher_global,
            teacher_local,
            valid_state=batch["valid_state"],
            valid_action=batch["valid_action"],
        )
        indices = cpu_batch["window_index"].long()
        reference_state = cache["physical_state"][indices].to(device)
        reference_action = cache["action"][indices].to(device)
        reference_logits = cache["state_contact_logits"][indices].to(device)
        valid_state = batch["valid_state"].bool()
        valid_action = batch["valid_action"].bool()
        state_error = output.physical_state[..., :68] - batch["physical_state"][..., :68]
        action_error = output.action - batch["action"]
        state_values = state_error.masked_select(valid_state[..., None])
        action_values = action_error.masked_select(valid_action[..., None])
        truth_state_sse += float(torch.square(state_values).sum().cpu())
        truth_action_sse += float(torch.square(action_values).sum().cpu())
        state_count += state_values.numel()
        action_count += action_values.numel()
        truth_abs.extend((state_values.abs().cpu(), action_values.abs().cpu()))
        for index in range(len(cpu_batch["motion_key"])):
            state_window = state_error[index].masked_select(
                valid_state[index, :, None]
            )
            action_window = action_error[index].masked_select(
                valid_action[index, :, None]
            )
            worst_state = max(
                worst_state,
                float(torch.sqrt(torch.square(state_window).mean()).cpu()),
            )
            worst_action = max(
                worst_action,
                float(torch.sqrt(torch.square(action_window).mean()).cpu()),
            )
        state_diff = (
            output.physical_state[..., :68] - reference_state[..., :68]
        ).masked_select(valid_state[..., None])
        action_diff = (output.action - reference_action).masked_select(valid_action[..., None])
        functional_state_sse += float(torch.square(state_diff).sum().cpu())
        functional_action_sse += float(torch.square(action_diff).sum().cpu())
        functional_abs.extend((state_diff.abs().cpu(), action_diff.abs().cpu()))
        predicted = output.state_contact_logits.sigmoid() >= 0.5
        target = batch["physical_state"][..., 68:70] >= 0.5
        contact_correct += int((predicted == target).masked_select(valid_state[..., None]).sum().cpu())
        contact_count += int(valid_state.sum()) * 2
        reference_contact = reference_logits.sigmoid() >= 0.5
        functional_contact_correct += int(
            (predicted == reference_contact)
            .masked_select(valid_state[..., None])
            .sum()
            .cpu()
        )
        functional_contact_count += int(valid_state.sum()) * 2
    baseline = cache["source_truth_metrics"]
    multiplier = float(adaptation["source_metric_multiplier"])
    metrics = {
        "global_state_rmse": math.sqrt(truth_state_sse / state_count),
        "global_action_rmse": math.sqrt(truth_action_sse / action_count),
        "worst_mask_state_rmse": worst_state,
        "worst_mask_action_rmse": worst_action,
        "continuous_p99_abs": deterministic_quantile(torch.cat(truth_abs), 0.99),
        "contact_accuracy": contact_correct / contact_count,
        "functional_contact_agreement": functional_contact_correct
        / functional_contact_count,
        "functional_state_rmse": math.sqrt(functional_state_sse / state_count),
        "functional_action_rmse": math.sqrt(functional_action_sse / action_count),
        "functional_p99_abs": deterministic_quantile(torch.cat(functional_abs), 0.99),
    }
    truth_limits = {
        key: min(float(baseline[key]) * multiplier, float(fit_thresholds[key]))
        for key in (
            "global_state_rmse", "global_action_rmse", "worst_mask_state_rmse",
            "worst_mask_action_rmse", "continuous_p99_abs",
        )
    }
    checks = {
        **{key: metrics[key] <= limit for key, limit in truth_limits.items()},
        "functional_state_rmse": metrics["functional_state_rmse"]
        <= float(adaptation["functional_state_rmse"]),
        "functional_action_rmse": metrics["functional_action_rmse"]
        <= float(adaptation["functional_action_rmse"]),
        "functional_p99_abs": metrics["functional_p99_abs"]
        <= float(adaptation["functional_p99_abs"]),
        "contact_accuracy": metrics["contact_accuracy"] == 1.0,
        "functional_contact_agreement": metrics["functional_contact_agreement"]
        == 1.0,
    }
    latent_dependence = evaluate_teacher_latent_dependence(
        model, loader, cache, device, donor_maps
    )
    for name, value in latent_dependence["main_ratios"].items():
        checks[f"latent_{name}_ratio"] = value >= float(
            fit_thresholds["latent_ratio"]
        )
    return {
        "passed": all(checks.values()),
        "metrics": metrics,
        "limits": truth_limits,
        "checks": checks,
        "latent_dependence": latent_dependence,
    }


def preservation_loss(
    model: HierarchicalConditionalPriorTransformer,
    batch: dict[str, torch.Tensor],
    cpu_batch: dict[str, Any],
    cache: dict[str, Any],
    teacher_global: torch.Tensor,
    teacher_local: torch.Tensor,
) -> torch.Tensor:
    output = model.decode_from_canonical_latents(
        teacher_global,
        teacher_local,
        valid_state=batch["valid_state"],
        valid_action=batch["valid_action"],
    )
    full_state = batch["valid_state"].bool()[..., None].expand_as(batch["physical_state"])
    full_action = batch["valid_action"].bool()[..., None].expand_as(batch["action"])
    truth = reconstruction_loss(output, batch, full_state, full_action)["total"]
    indices = cpu_batch["window_index"].long()
    reference_state = cache["physical_state"][indices].to(output.physical_state.device)
    reference_action = cache["action"][indices].to(output.action.device)
    reference_logits = cache["state_contact_logits"][indices].to(output.state_contact_logits.device)
    functional = torch.stack((
        torch.square(output.physical_state[..., :68] - reference_state[..., :68])
        .masked_select(full_state[..., :68]).mean(),
        torch.square(output.action - reference_action).masked_select(full_action).mean(),
        torch.square(output.state_contact_logits - reference_logits)
        .masked_select(full_state[..., 68:70]).mean(),
    )).mean()
    return truth + functional


def _checkpoint(
    model: HierarchicalConditionalPriorTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    *,
    mode: str,
    training_phase: str,
    optimizer_step: int,
    dataset_hash: str,
    window_hash: str,
    heldout_hash: str,
    fixed_hash: str,
    source_hash: str,
    teacher_cache_hash: str,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT,
        "mode": mode,
        "training_phase": training_phase,
        "optimizer_step": optimizer_step,
        "dataset_manifest_sha256": dataset_hash,
        "selected_windows_sha256": window_hash,
        "heldout_fixture_sha256": heldout_hash,
        "fixed_fixture_sha256": fixed_hash,
        "source_checkpoint_sha256": source_hash,
        "teacher_cache_sha256": teacher_cache_hash,
        "model_signature": _model_signature(config["model"]),
        "total_parameter_count": int(config["model"]["total_parameter_count"]),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }


def validate_checkpoint(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "format": checkpoint.get("format_version") == CHECKPOINT_FORMAT,
        "mode": checkpoint.get("mode") == expected["mode"],
        "dataset": checkpoint.get("dataset_manifest_sha256") == expected["dataset"],
        "windows": checkpoint.get("selected_windows_sha256") == expected["windows"],
        "source": checkpoint.get("source_checkpoint_sha256") == expected["source"],
        "teacher_cache": checkpoint.get("teacher_cache_sha256") == expected["teacher_cache"],
        "model": isinstance(checkpoint.get("model"), dict),
        "optimizer": isinstance(checkpoint.get("optimizer"), dict),
        "scheduler": isinstance(checkpoint.get("scheduler"), dict),
    }
    return {"passed": all(checks.values()), "checks": checks, "sha256": file_sha256(path)}


def decoder_adaptation_decision(
    latent_pass: bool,
    reconstruction_pass: bool,
    *,
    d1_scores: list[float] | None = None,
    d1_initial_score: float | None = None,
    preservation_pass: bool = True,
    minimum_improvement: float = 0.2,
    maximum_score: float = 1.5,
) -> str:
    if not latent_pass:
        return "STOP_CONDITIONAL_PRIOR_LATENT_ALIGNMENT_FAILED"
    if reconstruction_pass:
        return "FREEZE_KL0_BASELINE_AND_IMPLEMENT_KL_THREE_PATHS"
    if d1_scores is None:
        return "RUN_CONTROLLED_DECODER_ADAPTATION_D1"
    if not preservation_pass or len(d1_scores) < 3 or d1_initial_score is None:
        return "STOP_DECODER_ADAPTATION_REJECTED"
    median = sorted(d1_scores[-3:])[1]
    improvement = (d1_initial_score - median) / max(d1_initial_score, 1e-12)
    if improvement >= minimum_improvement and d1_scores[-1] <= maximum_score:
        return "RUN_CONTROLLED_DECODER_ADAPTATION_D2"
    return "STOP_DECODER_ADAPTATION_D1_NOT_PROMISING"


def _best_rank(result: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(result["joint_gate"]["score"]),
        float(result["latent_alignment"]["gate"]["score"]),
        float(result["random_masks"]["fit_gate"]["score"]),
    )


def render_cpd_plots(
    output_run: Path,
    records: list[dict[str, Any]],
) -> dict[str, str]:
    evaluations = [row for row in records if row.get("phase") == "evaluation"]
    train = [row for row in records if row.get("phase") == "train"]
    training_path = output_run / "plots/training_curves.svg"
    _svg(training_path, "H50-CPD training losses", "Latent teacher, full+masked reconstruction, and decoder preservation", [
        ("Objective", [(r["optimizer_step"], r["losses"]["objective"]) for r in train], "#08519c"),
        ("Latent", [(r["optimizer_step"], r["losses"]["latent"]) for r in train], "#238b45"),
        ("Reconstruction", [(r["optimizer_step"], r["losses"]["reconstruction"]) for r in train], "#d94801"),
        ("Keep", [(r["optimizer_step"], r["losses"]["keep"]) for r in train], "#756bb1"),
    ])
    gate_path = output_run / "plots/gate_curves.svg"
    _svg(gate_path, "H50-CPD quality gates", "All gates must pass together in P2 or accepted decoder adaptation", [
        ("Joint score", [(r["optimizer_step"], r["metrics"]["joint_gate"]["score"]) for r in evaluations], "#08519c"),
        ("Random Mask score", [(r["optimizer_step"], r["metrics"]["random_masks"]["fit_gate"]["score"]) for r in evaluations], "#41ab5d"),
        ("Fixed Mask score", [(r["optimizer_step"], r["metrics"]["fixed_physical_masks"]["fit_gate"]["score"]) for r in evaluations], "#fd8d3c"),
        ("PASS threshold", [(r["optimizer_step"], 1.0) for r in evaluations], "#555555"),
    ])
    path = output_run / "plots/latent_alignment.svg"
    _svg(path, "H50-CPD latent alignment", "Held-out complete-token random Masks", [
        ("Global standardized RMSE", [(r["optimizer_step"], r["metrics"]["latent_alignment"]["global_standardized_rmse"]) for r in evaluations], "#2171b5"),
        ("Local standardized RMSE", [(r["optimizer_step"], r["metrics"]["latent_alignment"]["local_standardized_rmse"]) for r in evaluations], "#41ab5d"),
        ("Latent gate threshold", [(r["optimizer_step"], 0.25) for r in evaluations], "#555555"),
    ])
    preservation = output_run / "plots/decoder_preservation.svg"
    _svg(preservation, "H50-A decoder preservation", "Teacher latent through current decoder", [
        ("Functional State RMSE", [(r["optimizer_step"], r["metrics"]["teacher_preservation"]["metrics"]["functional_state_rmse"]) for r in evaluations], "#756bb1"),
        ("Functional Action RMSE", [(r["optimizer_step"], r["metrics"]["teacher_preservation"]["metrics"]["functional_action_rmse"]) for r in evaluations], "#e6550d"),
    ])
    dependence_path = output_run / "plots/latent_dependence.svg"
    _svg(dependence_path, "H50-CPD latent dependence", "Student latent replacements on held-out random_both_65 queries", [
        ("Zero", [(r["optimizer_step"], r["metrics"]["latent_dependence"]["main_ratios"]["zero"]) for r in evaluations], "#cb181d"),
        ("Cross-window", [(r["optimizer_step"], r["metrics"]["latent_dependence"]["main_ratios"]["cross_window"]) for r in evaluations], "#6a51a3"),
        ("Cross-motion", [(r["optimizer_step"], r["metrics"]["latent_dependence"]["main_ratios"]["cross_motion"]) for r in evaluations], "#2171b5"),
        ("PASS threshold", [(r["optimizer_step"], 10.0) for r in evaluations], "#555555"),
    ])
    mask_path = output_run / "plots/random_mask_breakdown.svg"
    latest = evaluations[-1]["metrics"]["random_masks"] if evaluations else None
    mask_points = [] if latest is None else [
        (index, max(float(value["worst_state_rmse"]), float(value["worst_action_rmse"])))
        for index, value in enumerate(latest["cases"].values())
    ]
    _svg(mask_path, "Held-out random Token Mask breakdown", "Worst-window normalized RMSE", [
        ("Mask RMSE", mask_points, "#6a51a3")
    ], x_label="Mask family index")
    feature_path = output_run / "plots/feature_error.svg"
    feature_points = [] if latest is None else [
        (row["index"], row["normalized_rmse"]) for row in latest["feature_errors"]
    ]
    _svg(feature_path, "97 continuous feature errors", "Held-out random Token Mask normalized RMSE", [
        ("Feature RMSE", feature_points, "#2171b5")
    ], x_label="Continuous feature index")
    return {
        "training_curves": str(training_path),
        "gate_curves": str(gate_path),
        "latent_alignment": str(path),
        "decoder_preservation": str(preservation),
        "latent_dependence": str(dependence_path),
        "random_mask_breakdown": str(mask_path),
        "feature_error": str(feature_path),
    }


def run_experiment(
    dataset_run: Path,
    output_run: Path,
    source_run: Path,
    config: dict[str, Any],
    *,
    mode: str,
    init_run: Path | None,
    smoke: bool,
) -> dict[str, Any]:
    from .dataset import StateActionWindowDataset

    if mode not in MODES:
        raise ValueError(f"H50-CPD mode must be one of {MODES}")
    if smoke and mode != "train":
        raise ValueError("H50-CPD smoke exercises the frozen-decoder prior path")
    if mode == "adapt" and init_run is None:
        raise ValueError("decoder adaptation requires the completed prior run")
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
        raise FileNotFoundError("H50-CPD requires the dedicated overfit subset")
    data = config["data"]
    training = config["training"]
    model_config = config["model"]
    if (int(data["motion_count"]), int(data["window_transitions"]), int(data["stride"])) != (32, 64, 64):
        raise ValueError("H50-CPD requires 32 motions, T64, stride64")
    if training["random_training_mixture"] != TRAINING_MIXTURE:
        raise ValueError("H50-CPD training mixture differs from the registered protocol")
    if float(training["weight_decay"]) != 0.0 or float(training["kl_beta"]) != 0.0:
        raise ValueError("H50-CPD deterministic training requires weight_decay=KL=0")

    base = StateActionWindowDataset(
        dataset_run, "train", 64, 64, max_episodes=256, random_crop=False
    )
    motions = validate_motion_prefix(base, 32)
    if len(base.episodes) != 256 or len(motions) != 32:
        raise ValueError("H50-CPD requires exactly 32 motions x 8 variants")
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

    with torch.device("meta"):
        meta_model = build_model(model_config)
    if not isinstance(meta_model, HierarchicalConditionalPriorTransformer):
        raise TypeError("H50-CPD config built the wrong model")
    actual_parameters = sum(parameter.numel() for parameter in meta_model.parameters())
    del meta_model
    if actual_parameters != int(model_config["total_parameter_count"]):
        raise ValueError("H50-CPD parameter count differs from its reference config")

    seed_everything(int(config["initialization_seed"]))
    model = build_model(model_config)
    assert isinstance(model, HierarchicalConditionalPriorTransformer)
    initialization = initialize_from_h50_a(model, checkpoint)
    source_base_hash = base_state_sha256(model)
    initialization["base_state_sha256"] = source_base_hash

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    workers = 0 if smoke else int(data["num_workers"])
    micro_batch = int(training["micro_batch"])
    indexed: Dataset[dict[str, Any]] = MaskBankDataset(selected, 1)
    base_loader = DataLoader(
        indexed, batch_size=micro_batch, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    heldout_data = MaskBankDataset(selected, len(RANDOM_TOKEN_MASK_NAMES))
    heldout_loader = DataLoader(
        heldout_data, batch_size=micro_batch, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    fixed_data = MaskBankDataset(selected, len(PHYSICAL_MASK_NAMES))
    fixed_loader = DataLoader(
        fixed_data, batch_size=micro_batch, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    heldout_seed = int(config["heldout_mask_seed"])
    training_mask_seed = int(config["training_mask_seed"])
    heldout_maker = lambda value: make_heldout_random_token_masks(value, heldout_seed)
    fixed_maker = lambda value: make_physical_masks(value, 20260830)
    heldout_hash = mask_bank_sha256(heldout_loader, heldout_maker)
    fixed_hash = mask_bank_sha256(fixed_loader, fixed_maker)
    cache = build_teacher_cache(
        model,
        base_loader,
        device,
        output_run,
        source_hash=source["checkpoint_sha256"],
        dataset_hash=dataset_hash,
        window_hash=window_hash,
    )
    cache["heldout_seed"] = heldout_seed
    donor_maps = _donor_maps(windows)

    source_metrics = _source_truth_metrics(model, base_loader, cache, device)
    cache["source_truth_metrics"] = source_metrics
    cache_path = Path(cache["manifest"]["path"])
    cache_disk = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache_disk["source_truth_metrics"] = source_metrics
    atomic_torch_save(cache_path, cache_disk)
    cache["manifest"]["sha256"] = file_sha256(cache_path)
    atomic_write_json(output_run / "manifests/teacher_latent_cache.json", cache["manifest"])

    adaptation_source: dict[str, Any] | None = None
    if mode == "adapt":
        assert init_run is not None
        adaptation_source = load_prior_run_for_adaptation(
            model,
            init_run,
            model_config=model_config,
            dataset_hash=dataset_hash,
            window_hash=window_hash,
            heldout_hash=heldout_hash,
            fixed_hash=fixed_hash,
            source_hash=source["checkpoint_sha256"],
        )

    fit_thresholds = {key: float(value) for key, value in training["fit_thresholds"].items()}
    strict_thresholds = {
        "worst_state_rmse": 0.01, "worst_action_rmse": 0.01,
        "continuous_max_abs": 0.01, "contact_accuracy": 1.0, "latent_ratio": 10.0,
    }
    exact_thresholds = {
        "worst_state_rmse": 1e-4, "worst_action_rmse": 1e-4,
        "continuous_max_abs": 1e-3, "contact_accuracy": 1.0, "latent_ratio": 10.0,
    }
    latent_thresholds = {
        key: float(value) for key, value in training["latent_thresholds"].items()
    }
    state_std = torch.from_numpy(base.state_std)
    action_std = torch.from_numpy(base.action_std)
    metrics_path = output_run / "logs/metrics.jsonl"
    records: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_rank = (math.inf, math.inf, math.inf)
    best_latent_score = math.inf
    best_latent_step = -1
    best_step = -1
    current_training_phase = "step0"

    def run_evaluation(step: int) -> dict[str, Any]:
        started = time.perf_counter()
        random_metrics = evaluate(
            model, heldout_loader, base_loader, device, heldout_maker,
            fit_thresholds=fit_thresholds,
            strict_thresholds=strict_thresholds,
            exact_thresholds=exact_thresholds,
            state_std=state_std,
            action_std=action_std,
            latent_diagnostics=False,
            report_full_sequence=True,
        )
        fixed_metrics = evaluate(
            model, fixed_loader, base_loader, device, fixed_maker,
            fit_thresholds=fit_thresholds,
            strict_thresholds=strict_thresholds,
            exact_thresholds=exact_thresholds,
            state_std=state_std,
            action_std=action_std,
            latent_diagnostics=False,
        )
        alignment = evaluate_latent_alignment(
            model, heldout_loader, cache, device, donor_maps, latent_thresholds
        )
        dependence = evaluate_latent_dependence(
            model, base_loader, device, donor_maps, heldout_seed
        )
        preservation = evaluate_teacher_preservation(
            model,
            base_loader,
            cache,
            device,
            fit_thresholds,
            training["decoder_adaptation"],
            donor_maps,
        )
        full_both = evaluate_full_both_conditional_prior(
            model, base_loader, device
        )
        reconstruction_pass = bool(
            random_metrics["fit_gate"]["passed"]
            and fixed_metrics["fit_gate"]["passed"]
            and dependence["passed"]
        )
        joint_score = max(
            float(random_metrics["fit_gate"]["score"]),
            float(fixed_metrics["fit_gate"]["score"]),
            float(alignment["gate"]["score"]),
            max(10.0 / max(float(value), 1e-12) for value in dependence["main_ratios"].values()),
            1.0 if preservation["passed"] else 2.0,
        )
        result = {
            "random_masks": random_metrics,
            "fixed_physical_masks": fixed_metrics,
            "latent_alignment": alignment,
            "latent_dependence": dependence,
            "teacher_preservation": preservation,
            "full_both_conditional_prior": full_both,
            "reconstruction_pass": reconstruction_pass,
            "joint_gate": {
                "passed": bool(reconstruction_pass and alignment["gate"]["passed"] and preservation["passed"]),
                "score": joint_score,
            },
            "evaluation_seconds": time.perf_counter() - started,
            "evaluation_scope": "seen 32-motion T64 windows; held-out random-token bank plus fixed physical retention",
        }
        result["artifacts"] = write_evaluation_artifacts(output_run, step, random_metrics)
        row = {
            "phase": "evaluation", "mode": mode,
            "training_phase": current_training_phase,
            "optimizer_step": step, "metrics": result,
        }
        records.append(row)
        append_jsonl(metrics_path, row)
        return result

    initial = run_evaluation(0)
    if mode == "train":
        optimizer, scheduler, optimizer_contract = configure_prior_optimizer(model, training)
        stages = [("prior", 2 if smoke else int(training["phases"][-1]["end"]))]
    else:
        optimizer = scheduler = None
        optimizer_contract = {}
        stages = [("D1", int(training["decoder_adaptation"]["D1"]["max_optimizer_steps"]))]
    global_step = 0
    adaptation_records: dict[str, list[float]] = {"D1": [], "D2": []}
    d1_initial_score = float(initial["joint_gate"]["score"])
    rejected = False
    last_training_phase = "not_started"
    active_optimizer: torch.optim.Optimizer | None = None
    active_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    for stage, maximum in stages:
        if mode == "adapt":
            optimizer, scheduler, optimizer_contract = configure_adaptation_optimizer(
                model, training, stage
            )
        assert optimizer is not None and scheduler is not None
        active_optimizer, active_scheduler = optimizer, scheduler
        model.train()
        generator = torch.Generator().manual_seed(int(config["training_seed"]) + (0 if stage in {"prior", "D1"} else 1))
        train_loader = DataLoader(
            indexed, batch_size=micro_batch, shuffle=True, num_workers=workers,
            generator=generator, drop_last=not smoke,
            pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        )
        stream = _infinite(train_loader)
        accumulation = int(training["gradient_accumulation"])
        optimizer.zero_grad(set_to_none=True)
        stage_pass_streak = 0
        for stage_step in range(1, maximum + 1):
            global_step += 1
            if mode == "train":
                phase, latent_weight, reconstruction_weight = phase_weights(
                    global_step, training["phases"]
                )
            else:
                phase = stage
                contract = training["decoder_adaptation"][stage]
                latent_weight = float(contract["latent_weight"])
                reconstruction_weight = float(contract["reconstruction_weight"])
            current_training_phase = phase
            last_training_phase = phase
            started = time.perf_counter()
            aggregate = {
                "objective": 0.0, "latent": 0.0, "latent_global": 0.0,
                "latent_local": 0.0, "reconstruction": 0.0, "keep": 0.0,
            }
            sampled = hashlib.sha256()
            for slot in range(accumulation):
                cpu_batch = next(stream)
                state_mask, action_mask, names = make_dynamic_random_token_masks(
                    cpu_batch, training_mask_seed, global_step, slot
                )
                validate_complete_token_masks(cpu_batch, state_mask, action_mask)
                sampled.update(state_mask.numpy().tobytes(order="C"))
                sampled.update(action_mask.numpy().tobytes(order="C"))
                sampled.update(canonical_json_bytes(names))
                batch = _device_batch(cpu_batch, device)
                state_mask_d = state_mask.to(device)
                action_mask_d = action_mask.to(device)
                output = model(batch, state_mask_d, action_mask_d)
                teacher_global, teacher_local = teacher_for_batch(cache, cpu_batch, device)
                latent_losses = latent_distillation_loss(
                    output.global_latent,
                    output.local_latents,
                    teacher_global,
                    teacher_local,
                    cache["global_std"].to(device),
                    cache["local_std"].to(device),
                )
                reconstruction = full_and_masked_reconstruction_loss(
                    output, batch, state_mask_d, action_mask_d
                )
                keep = output.physical_state.sum() * 0.0
                keep_weight = 0.0
                if mode == "adapt":
                    keep = preservation_loss(
                        model, batch, cpu_batch, cache, teacher_global, teacher_local
                    )
                    keep_weight = float(training["decoder_adaptation"][stage]["keep_weight"])
                objective = (
                    latent_weight * latent_losses["total"]
                    + reconstruction_weight * reconstruction["total"]
                    + keep_weight * keep
                )
                (objective / accumulation).backward()
                values = {
                    "objective": objective, "latent": latent_losses["total"],
                    "latent_global": latent_losses["global"],
                    "latent_local": latent_losses["local"],
                    "reconstruction": reconstruction["total"], "keep": keep,
                }
                for name, value in values.items():
                    aggregate[name] += float(value.detach().cpu()) / accumulation
            trainable = [p for p in model.parameters() if p.requires_grad]
            gradient = torch.nn.utils.clip_grad_norm_(trainable, float(training["gradient_clip"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            row = {
                "phase": "train", "mode": mode, "training_phase": phase,
                "optimizer_step": global_step, "stage_optimizer_step": stage_step,
                "losses": aggregate,
                "loss_weights": {
                    "latent": latent_weight, "reconstruction": reconstruction_weight,
                    "keep": keep_weight,
                },
                "learning_rates": {group["name"]: float(group["lr"]) for group in optimizer.param_groups},
                "gradient_norm_before_clip": float(gradient),
                "gradient_was_clipped": float(gradient) > float(training["gradient_clip"]),
                "sample_mask_sha256": sampled.hexdigest(),
                "step_seconds": time.perf_counter() - started,
                "cuda_max_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
            }
            records.append(row)
            append_jsonl(metrics_path, row)
            interval = 2 if smoke else int(training["validation_interval"])
            if stage_step % interval and stage_step != maximum:
                continue
            result = run_evaluation(global_step)
            rank = _best_rank(result)
            latent_score = float(result["latent_alignment"]["gate"]["score"])
            checkpoint_is_eligible = bool(
                smoke or mode == "adapt" or phase == "P2"
            )
            checkpoint_is_eligible = bool(
                checkpoint_is_eligible
                and (
                    mode != "adapt"
                    or result["teacher_preservation"]["passed"]
                )
            )
            if checkpoint_is_eligible and latent_score < best_latent_score:
                best_latent_score = latent_score
                best_latent_step = global_step
                atomic_torch_save(
                    output_run / "checkpoints/best_latent.pt",
                    _checkpoint(
                        model, optimizer, scheduler, config, mode=mode,
                        training_phase=phase, optimizer_step=global_step,
                        dataset_hash=dataset_hash, window_hash=window_hash,
                        heldout_hash=heldout_hash, fixed_hash=fixed_hash,
                        source_hash=source["checkpoint_sha256"],
                        teacher_cache_hash=cache["manifest"]["sha256"],
                    ),
                )
            if checkpoint_is_eligible and rank < best_rank:
                best_rank, best_metrics, best_step = rank, result, global_step
                atomic_torch_save(
                    output_run / (
                        "checkpoints/best_decoder_adapted_fit.pt"
                        if mode == "adapt" else "checkpoints/best_prior_fit.pt"
                    ),
                    _checkpoint(
                        model, optimizer, scheduler, config, mode=mode,
                        training_phase=phase, optimizer_step=global_step,
                        dataset_hash=dataset_hash, window_hash=window_hash,
                        heldout_hash=heldout_hash, fixed_hash=fixed_hash,
                        source_hash=source["checkpoint_sha256"],
                        teacher_cache_hash=cache["manifest"]["sha256"],
                    ),
                )
            atomic_torch_save(
                output_run / "checkpoints/last.pt",
                _checkpoint(
                    model, optimizer, scheduler, config, mode=mode,
                    training_phase=phase, optimizer_step=global_step,
                    dataset_hash=dataset_hash, window_hash=window_hash,
                    heldout_hash=heldout_hash, fixed_hash=fixed_hash,
                    source_hash=source["checkpoint_sha256"],
                    teacher_cache_hash=cache["manifest"]["sha256"],
                ),
            )
            passed = bool(result["joint_gate"]["passed"])
            stage_pass_streak = stage_pass_streak + 1 if passed else 0
            if mode == "adapt":
                adaptation_records[stage].append(float(result["joint_gate"]["score"]))
                if not result["teacher_preservation"]["passed"]:
                    rejected = True
                    break
            render_cpd_plots(output_run, records)
            if (
                not smoke
                and stage_pass_streak >= int(training["required_pass_streak"])
                and (mode == "adapt" or phase == "P2")
            ):
                break
            model.train()
        if smoke or rejected or stage_pass_streak >= int(training["required_pass_streak"]):
            break
        if mode == "adapt" and stage == "D1":
            latest = [row["metrics"] for row in records if row["phase"] == "evaluation"][-1]
            decision = decoder_adaptation_decision(
                bool(latest["latent_alignment"]["gate"]["passed"]),
                bool(latest["joint_gate"]["passed"]),
                d1_scores=adaptation_records["D1"],
                d1_initial_score=d1_initial_score,
                preservation_pass=bool(latest["teacher_preservation"]["passed"]),
                minimum_improvement=float(training["decoder_adaptation"]["minimum_d1_improvement"]),
                maximum_score=float(training["decoder_adaptation"]["maximum_d1_fit_score"]),
            )
            if decision == "RUN_CONTROLLED_DECODER_ADAPTATION_D2":
                stages.append(("D2", int(training["decoder_adaptation"]["D2"]["max_optimizer_steps"])))
            else:
                break

    assert active_optimizer is not None and active_scheduler is not None
    if best_metrics is None:
        best_metrics = initial
        best_rank = _best_rank(initial)
        best_step = 0
    if not math.isfinite(best_latent_score):
        best_latent_score = float(initial["latent_alignment"]["gate"]["score"])
        best_latent_step = 0
    evaluations = [row["metrics"] for row in records if row["phase"] == "evaluation"]
    last_three = evaluations[-3:]
    quality_pass = bool(
        not smoke and len(last_three) == 3
        and all(row["joint_gate"]["passed"] for row in last_three)
    )
    latent_pass = bool(
        not smoke and len(last_three) == 3
        and all(row["latent_alignment"]["gate"]["passed"] for row in last_three)
    )
    reconstruction_pass = bool(
        not smoke and len(last_three) == 3
        and all(row["reconstruction_pass"] for row in last_three)
    )
    if mode == "train":
        next_step = decoder_adaptation_decision(latent_pass, reconstruction_pass)
    elif quality_pass:
        next_step = "FREEZE_ADAPTED_KL0_BASELINE_AND_IMPLEMENT_KL_THREE_PATHS"
    elif rejected:
        next_step = "STOP_DECODER_ADAPTATION_REJECTED"
    else:
        next_step = "STOP_DECODER_ADAPTATION_FAILED"
    frozen_base_gradients_absent = all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not model.is_conditional_prior_parameter(name)
    )
    if mode == "train" and not frozen_base_gradients_absent:
        raise RuntimeError("frozen H50-A base unexpectedly retained gradients")
    base_final_hash = base_state_sha256(model)
    if mode == "train" and base_final_hash != source_base_hash:
        raise RuntimeError("frozen H50-A base changed during prior training")
    plots = render_cpd_plots(output_run, records)
    checkpoint_readback = validate_checkpoint(
        output_run / "checkpoints/last.pt",
        {
            "mode": mode, "dataset": dataset_hash, "windows": window_hash,
            "source": source["checkpoint_sha256"],
            "teacher_cache": cache["manifest"]["sha256"],
        },
    )
    if not checkpoint_readback["passed"]:
        raise RuntimeError("H50-CPD checkpoint readback failed")
    summary = {
        "format_version": SUMMARY_FORMAT,
        "experiment": f"H50-CPD-{mode}",
        "mode": mode,
        "execution_pass": True,
        "smoke": smoke,
        "quality_pass": quality_pass,
        "latent_alignment_pass": latent_pass,
        "reconstruction_pass": reconstruction_pass,
        "decoder_adaptation_rejected": rejected,
        "dataset_run": str(dataset_run),
        "dataset_manifest_sha256": dataset_hash,
        "motion_count": 32,
        "episode_count": len(base.episodes),
        "selected_motion_keys": motions,
        "window_transitions": 64,
        "stride": 64,
        "window_count": len(selected),
        "selected_windows_sha256": window_hash,
        "heldout_fixture_count": len(heldout_data),
        "heldout_fixture_sha256": heldout_hash,
        "fixed_fixture_count": len(fixed_data),
        "fixed_fixture_sha256": fixed_hash,
        "mask_contract": {
            "granularity": "complete State or Action token only",
            "training_mixture": TRAINING_MIXTURE,
            "training_seed": training_mask_seed,
            "heldout_seed": heldout_seed,
            "full_both_training": False,
            "full_both_role": "non-identifiability diagnostic only",
        },
        "model_contract": {
            **model_config,
            "actual_parameter_count": actual_parameters,
            "decoder_receives_state_action_or_query_mask": False,
            "visible_condition_path": "conditional prior latent only",
        },
        "source": source,
        "initialization": initialization,
        "adaptation_source": adaptation_source,
        "teacher_cache": cache["manifest"],
        "source_truth_metrics": source_metrics,
        "optimizer_contract": optimizer_contract,
        "frozen_base_gradients_absent": frozen_base_gradients_absent,
        "completed_optimizer_steps": global_step,
        "last_training_phase": last_training_phase,
        "best_optimizer_step": best_step,
        "best_latent_optimizer_step": best_latent_step,
        "best_latent_score": best_latent_score,
        "best_rank": list(best_rank),
        "best_evaluation": best_metrics,
        "initial_evaluation": initial,
        "last_three_evaluations": last_three,
        "base_state_sha256_before": source_base_hash,
        "base_state_sha256_after": base_final_hash,
        "checkpoint_readback": checkpoint_readback,
        "plots": plots,
        "unique_next_step": next_step,
        "scope": "deterministic conditional-prior completion on seen 32-motion T64 windows; no KL, sampling, unseen-motion, or exhaustive-mask claim",
    }
    atomic_write_json(output_run / "manifests/posterior_conditional_prior_summary.json", summary)
    if smoke:
        atomic_write_text(output_run / "markers/cvae_posterior_conditional_prior_smoke.ok", "PASS\n")
    else:
        atomic_write_text(output_run / "markers/cvae_posterior_conditional_prior_execution.ok", "PASS\n")
        if latent_pass:
            atomic_write_text(output_run / "markers/cvae_posterior_conditional_prior_latent_alignment.ok", "PASS\n")
        if quality_pass:
            marker = (
                "cvae_posterior_conditional_prior_decoder_adapted_fit.ok"
                if mode == "adapt" else "cvae_posterior_conditional_prior_fit.ok"
            )
            atomic_write_text(output_run / "markers" / marker, "PASS\n")
        else:
            atomic_write_text(
                output_run / "markers/cvae.failed",
                f"QUALITY_FAIL execution_complete=true mode={mode} latent={str(latent_pass).lower()} reconstruction={str(reconstruction_pass).lower()}\n",
            )
    base.close()
    return summary


@torch.no_grad()
def _source_truth_metrics(
    model: HierarchicalConditionalPriorTransformer,
    loader: Iterable[dict[str, Any]],
    cache: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    state_sse = action_sse = 0.0
    state_count = action_count = 0
    worst_state = worst_action = 0.0
    all_abs: list[torch.Tensor] = []
    for cpu_batch in loader:
        batch = _device_batch(cpu_batch, device)
        global_latent, local_latents = teacher_for_batch(cache, cpu_batch, device)
        output = model.decode_from_canonical_latents(
            global_latent, local_latents,
            valid_state=batch["valid_state"], valid_action=batch["valid_action"],
        )
        state_error = output.physical_state[..., :68] - batch["physical_state"][..., :68]
        action_error = output.action - batch["action"]
        state_values = state_error.masked_select(batch["valid_state"].bool()[..., None])
        action_values = action_error.masked_select(batch["valid_action"].bool()[..., None])
        state_sse += float(torch.square(state_values).sum().cpu())
        action_sse += float(torch.square(action_values).sum().cpu())
        state_count += state_values.numel()
        action_count += action_values.numel()
        all_abs.extend((state_values.abs().cpu(), action_values.abs().cpu()))
        for index in range(len(cpu_batch["motion_key"])):
            state_window = state_error[index].masked_select(
                batch["valid_state"][index, :, None].bool()
            )
            action_window = action_error[index].masked_select(
                batch["valid_action"][index, :, None].bool()
            )
            worst_state = max(
                worst_state,
                float(torch.sqrt(torch.square(state_window).mean()).cpu()),
            )
            worst_action = max(
                worst_action,
                float(torch.sqrt(torch.square(action_window).mean()).cpu()),
            )
    return {
        "global_state_rmse": math.sqrt(state_sse / state_count),
        "global_action_rmse": math.sqrt(action_sse / action_count),
        "worst_mask_state_rmse": worst_state,
        "worst_mask_action_rmse": worst_action,
        "continuous_p99_abs": deterministic_quantile(torch.cat(all_abs), 0.99),
    }


def load_prior_run_for_adaptation(
    model: HierarchicalConditionalPriorTransformer,
    run: Path,
    *,
    model_config: dict[str, Any],
    dataset_hash: str,
    window_hash: str,
    heldout_hash: str,
    fixed_hash: str,
    source_hash: str,
) -> dict[str, Any]:
    run = run.expanduser().resolve()
    summary_path = run / "manifests/posterior_conditional_prior_summary.json"
    checkpoint_path = run / "checkpoints/best_latent.pt"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise ValueError("decoder adaptation requires prior summary and best_latent.pt")
    summary = load_json(summary_path)
    checks = {
        "mode": summary.get("mode") == "train",
        "formal": not bool(summary.get("smoke", True)),
        "execution": bool(summary.get("execution_pass")),
        "latent_pass": bool(summary.get("latent_alignment_pass")),
        "reconstruction_failed": not bool(summary.get("reconstruction_pass")),
        "decision": summary.get("unique_next_step") == "RUN_CONTROLLED_DECODER_ADAPTATION_D1",
        "execution_marker": (run / "markers/cvae_posterior_conditional_prior_execution.ok").is_file(),
        "latent_marker": (run / "markers/cvae_posterior_conditional_prior_latent_alignment.ok").is_file(),
        "quality_failure": (run / "markers/cvae.failed").is_file(),
        "dataset": summary.get("dataset_manifest_sha256") == dataset_hash,
        "windows": summary.get("selected_windows_sha256") == window_hash,
        "heldout_masks": summary.get("heldout_fixture_sha256") == heldout_hash,
        "fixed_masks": summary.get("fixed_fixture_sha256") == fixed_hash,
        "source": summary.get("source", {}).get("checkpoint_sha256") == source_hash,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"conditional-prior adaptation source failed: {failed}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_checks = {
        "format": checkpoint.get("format_version") == CHECKPOINT_FORMAT,
        "mode": checkpoint.get("mode") == "train",
        "phase": checkpoint.get("training_phase") == "P2",
        "dataset": checkpoint.get("dataset_manifest_sha256") == dataset_hash,
        "windows": checkpoint.get("selected_windows_sha256") == window_hash,
        "heldout_masks": checkpoint.get("heldout_fixture_sha256") == heldout_hash,
        "fixed_masks": checkpoint.get("fixed_fixture_sha256") == fixed_hash,
        "source": checkpoint.get("source_checkpoint_sha256") == source_hash,
        "best_latent_step": int(checkpoint.get("optimizer_step", -1))
        == int(summary.get("best_latent_optimizer_step", -2)),
        "signature": checkpoint.get("model_signature")
        == _model_signature(model_config),
        "parameters": int(checkpoint.get("total_parameter_count", -1))
        == int(model_config["total_parameter_count"]),
        "model": isinstance(checkpoint.get("model"), dict),
    }
    failed_checkpoint = [
        name for name, passed in checkpoint_checks.items() if not passed
    ]
    if failed_checkpoint:
        raise ValueError(
            f"conditional-prior adaptation checkpoint failed: {failed_checkpoint}"
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    return {
        "run": str(run), "summary": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
        "checks": checks,
        "checkpoint_checks": checkpoint_checks,
        "model_only": True,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run H50 conditional-prior distillation")
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=root / "configs/posterior_hierarchical_conditional_prior_h50.json",
    )
    parser.add_argument("--mode", choices=MODES, default="train")
    parser.add_argument("--init-run", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_experiment(
        args.dataset_run,
        args.output_run,
        args.source_run,
        load_config(args.config.resolve()),
        mode=args.mode,
        init_run=args.init_run,
        smoke=args.smoke,
    )
    print("Posterior H50-CPD conditional prior: PASS (execution complete)")
    print(json.dumps({
        "output_run": str(args.output_run.expanduser().resolve()),
        "mode": summary["mode"], "smoke": summary["smoke"],
        "quality_pass": summary["quality_pass"],
        "latent_alignment_pass": summary["latent_alignment_pass"],
        "completed_optimizer_steps": summary["completed_optimizer_steps"],
        "best_joint_score": summary["best_rank"][0],
        "unique_next_step": summary["unique_next_step"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
