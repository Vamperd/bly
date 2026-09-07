from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import PosteriorCapacityOutput, build_model, parameter_count
from .posterior_capacity import (
    DeterministicWindowSubset,
    FIXED_MASK_NAMES,
    MaskBankDataset,
    _device_batch,
    evaluate_exact,
    make_fixture_masks,
    reconstruction_loss,
    selected_window_identities,
    validate_motion_prefix,
)
from .posterior_capacity_ab import (
    COMPARISON_FORMAT,
    COMPARISON_MARKER,
    FORMAT_VERSION as AB_FORMAT_VERSION,
    _append_record,
    _infinite,
    _learning_rate_multiplier,
    _load_control_dt,
    _load_joint_names,
    _svg_series,
    batch_sample_identities,
    fixture_bitmap_sha256,
    identity_sha256,
    initial_decision,
    run_full_evaluation,
    select_fixed_curve_cases,
    validate_paired_runs,
    validate_step0,
)
from .posterior_capacity_tail import (
    EXPECTED_MOTIONS,
    EXPECTED_PARAMETERS,
    EXPECTED_WINDOW,
    evaluate_tail,
    validate_f4a_checkpoint,
    validate_output_isolation,
    validate_source_reproduction,
)
from .util import (
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    canonical_json_bytes,
    file_sha256,
    load_json,
    seed_everything,
)


FORMAT_VERSION = "sonic_posterior_autodecoder_summary_v1"
CHECKPOINT_FORMAT = "sonic_posterior_autodecoder_checkpoint_v1"
FAILURE_FORMAT = "sonic_posterior_autodecoder_failure_v1"
FIXTURE_SEED = 20260830
E1_SEED = 20260830
E2_SEED = 20260831
EXPECTED_WINDOWS = 80
EXPECTED_FIXTURES = EXPECTED_WINDOWS * len(FIXED_MASK_NAMES)
EXPECTED_CODE_PARAMETERS = EXPECTED_WINDOWS * 256
EXPECTED_TOTAL_PARAMETERS = EXPECTED_PARAMETERS + EXPECTED_CODE_PARAMETERS
EXPECTED_DATASET_RUN_NAME = "cvae_overfit_subset_20260828_234506"
EXPECTED_SOURCE_RUN_NAME = (
    "cvae_posterior_capacity_fixed_m4_t128_25m_s100000_gprogression_20260904_190425"
)
EXPECTED_F4A_RUN_NAME = "cvae_posterior_capacity_tail_diagnostic_f4a_20260905_200807"
EXPECTED_TRIGGER_RUN_NAME = "cvae_posterior_capacity_ab_comparison_20260907_102553"
SMOKE_MARKER = "cvae_posterior_autodecoder_smoke.ok"
EXECUTION_MARKER = "cvae_posterior_autodecoder_execution.ok"
E1_MARKER = "cvae_posterior_autodecoder_code_only.ok"
E2_MARKER = "cvae_posterior_autodecoder_coadapt.ok"


class WindowCodeAutoDecoder(nn.Module):
    """Identity-indexed code table feeding only the existing masked decoder."""

    def __init__(self, base_model: nn.Module, initial_codes: torch.Tensor) -> None:
        super().__init__()
        if tuple(initial_codes.shape) != (EXPECTED_WINDOWS, 256):
            raise ValueError("F4E requires exactly 80 shared 256-D window codes")
        self.base_model = base_model
        self.window_codes = nn.Embedding(EXPECTED_WINDOWS, 256)
        with torch.no_grad():
            self.window_codes.weight.copy_(initial_codes)

    @staticmethod
    def code_indices(batch: dict[str, Any]) -> torch.Tensor:
        indices = batch.get("window_index", batch.get("source_window_index"))
        if not isinstance(indices, torch.Tensor):
            raise ValueError("auto-decoder batch is missing stable window_index")
        indices = indices.long()
        if indices.ndim != 1 or bool((indices < 0).any()) or bool(
            (indices >= EXPECTED_WINDOWS).any()
        ):
            raise ValueError("auto-decoder window_index must be a vector in [0, 79]")
        return indices

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        state_mask: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        latent_override: torch.Tensor | None = None,
        use_prior: bool = False,
    ) -> PosteriorCapacityOutput:
        if use_prior:
            raise ValueError("F4E has no conditional-prior path")
        code = self.window_codes(self.code_indices(batch))
        latent = code if latent_override is None else latent_override
        if latent.shape != code.shape:
            raise ValueError("F4E latent override shape does not match window code")
        decoded = self.base_model.decode_from_global_latent(
            batch, state_mask, action_mask, latent
        )
        zeros = torch.zeros_like(code)
        return PosteriorCapacityOutput(
            physical_state=decoded.physical_state,
            action=decoded.action,
            state_contact_logits=decoded.state_contact_logits,
            posterior_mean=code,
            posterior_logvar=zeros,
            prior_mean=zeros,
            prior_logvar=zeros,
            latent=latent,
        )


def _all_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_ab_summary(record: dict[str, Any], expected_arm: str) -> dict[str, Any]:
    run = Path(str(record.get("run", ""))).expanduser().resolve()
    path = run / "manifests/posterior_ab_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"F4E trigger input summary is missing: {path}")
    if file_sha256(path) != record.get("summary_sha256"):
        raise ValueError(f"F4E trigger {expected_arm} summary hash changed")
    summary = load_json(path)
    if summary.get("format_version") != AB_FORMAT_VERSION:
        raise ValueError(f"F4E trigger {expected_arm} summary format is unsupported")
    if summary.get("arm") != expected_arm or not bool(summary.get("execution_pass")):
        raise ValueError(f"F4E trigger {expected_arm} is not a formal completed arm")
    if bool(summary.get("smoke")):
        raise ValueError("F4E trigger refuses smoke A/B/C runs")
    return summary


def validate_stop_trigger(
    trigger_run: Path,
    *,
    dataset_hash: str,
    source_checkpoint_hash: str,
) -> dict[str, Any]:
    run = trigger_run.expanduser().resolve()
    if run.name != EXPECTED_TRIGGER_RUN_NAME:
        raise ValueError("F4E requires the fixed final A/B/C comparison run")
    manifest_path = run / "manifests/posterior_ab_comparison.json"
    marker_path = run / f"markers/{COMPARISON_MARKER}"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise FileNotFoundError("F4E trigger comparison manifest or marker is missing")
    manifest = load_json(manifest_path)
    decision = manifest.get("decision", {})
    inputs = manifest.get("inputs", {})
    checks = {
        "format_version": manifest.get("format_version") == COMPARISON_FORMAT,
        "execution_pass": bool(manifest.get("execution_pass")),
        "comparison_phase": manifest.get("comparison_phase") == "initial",
        "decision": decision.get("decision") == "STOP_LOSS_LATENT_SEED_SEARCH",
        "implement_f4c": decision.get("implement_f4c") is False,
        "no_next_seed": decision.get("next_optimizer_seed") is None,
        "no_replication_arms": list(decision.get("replication_arms", [])) == [],
        "input_arms": set(inputs) == {"A", "B", "C"},
    }
    summaries = {
        arm: _load_ab_summary(inputs.get(arm, {}), arm) for arm in ("A", "B", "C")
    }
    checks["paired_B"] = bool(validate_paired_runs(summaries["A"], summaries["B"])["passed"])
    checks["paired_C"] = bool(validate_paired_runs(summaries["A"], summaries["C"])["passed"])
    recomputed = initial_decision(summaries["A"], summaries["B"], summaries["C"])
    checks["recomputed_decision"] = (
        recomputed.get("decision") == "STOP_LOSS_LATENT_SEED_SEARCH"
    )
    checks["dataset_hash"] = all(
        summary.get("dataset_manifest_sha256") == dataset_hash
        for summary in summaries.values()
    )
    checks["source_checkpoint_hash"] = all(
        summary.get("source", {}).get("checkpoint_sha256") == source_checkpoint_hash
        for summary in summaries.values()
    )
    checks["first_optimizer_seed"] = all(
        int(summary.get("optimizer_seed", -1)) == E1_SEED
        for summary in summaries.values()
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"F4E trigger comparison failed validation: {failed}")
    return {
        "run": str(run),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "decision": decision.get("decision"),
        "checks": checks,
        "recomputed_residuals": {
            arm: recomputed["candidate_comparisons"][arm]["residuals"]
            for arm in ("B", "C")
        },
    }


def validate_contract(
    *,
    dataset_run: Path,
    source_checkpoint: Path,
    f4a_run: Path,
    trigger_run: Path,
    output_run: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_run = validate_output_isolation(dataset_run, source_checkpoint, output_run)
    protected = [f4a_run.expanduser().resolve(), trigger_run.expanduser().resolve()]
    for root in protected:
        if output_run == root or output_run.is_relative_to(root):
            raise ValueError(f"F4E output must be isolated from protected run: {root}")
    names = {
        "dataset": (dataset_run.name, EXPECTED_DATASET_RUN_NAME),
        "source": (source_run.name, EXPECTED_SOURCE_RUN_NAME),
        "F4A": (f4a_run.name, EXPECTED_F4A_RUN_NAME),
        "trigger": (trigger_run.name, EXPECTED_TRIGGER_RUN_NAME),
    }
    wrong = [key for key, (actual, expected) in names.items() if actual != expected]
    if wrong:
        raise ValueError(f"F4E fixed source identity mismatch: {wrong}")
    if not (dataset_run / "markers/cvae_overfit_subset.ok").is_file():
        raise FileNotFoundError("F4E requires the dedicated overfit subset marker")
    if source_checkpoint.name != "best_progression.pt" or not source_checkpoint.is_file():
        raise FileNotFoundError("F4E requires F4D best_progression.pt")
    if not (f4a_run / "markers/cvae_posterior_capacity_tail_diagnostic.ok").is_file():
        raise FileNotFoundError("F4E requires the formal F4A execution marker")
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    source_summary = load_json(source_run / "manifests/posterior_capacity_summary.json")
    f4a_manifest = load_json(f4a_run / "manifests/posterior_tail_diagnostic.json")
    dataset_hash = file_sha256(dataset_run / "manifests/dataset_manifest.json")
    checkpoint_hash = file_sha256(source_checkpoint)
    validate_f4a_checkpoint(checkpoint, dataset_hash)
    if source_summary.get("passed") is not False:
        raise ValueError("F4E source must be the failed F4D run")
    if not bool(f4a_manifest.get("execution_pass")):
        raise ValueError("F4E F4A diagnostic did not complete")
    if f4a_manifest.get("checkpoint", {}).get("sha256") != checkpoint_hash:
        raise ValueError("F4E F4A and source checkpoint hashes differ")
    expected = {
        "format_version": "sonic_posterior_autodecoder_config_v1",
        "fixture_seed": FIXTURE_SEED,
        "data": {
            "motion_count": EXPECTED_MOTIONS,
            "window_transitions": EXPECTED_WINDOW,
            "max_windows": None,
        },
        "model": {
            "latent_dim": 256,
            "base_parameter_count": EXPECTED_PARAMETERS,
            "code_parameter_count": EXPECTED_CODE_PARAMETERS,
            "total_parameter_count": EXPECTED_TOTAL_PARAMETERS,
            "decoder_layer_latent_gates": False,
        },
        "training": {
            "micro_batch": 4,
            "gradient_accumulation": 16,
            "gradient_clip": 1.0,
            "weight_decay": 0.0,
            "kl_beta": 0.0,
            "free_bits": 0.0,
        },
    }
    failed: list[str] = []
    for key in ("format_version", "fixture_seed"):
        if config.get(key) != expected[key]:
            failed.append(key)
    for section in ("data", "model", "training"):
        for key, value in expected[section].items():
            if config.get(section, {}).get(key) != value:
                failed.append(f"{section}.{key}")
    e1 = config.get("training", {}).get("stage_e1", {})
    e2 = config.get("training", {}).get("stage_e2", {})
    required_stages = {
        "e1.seed": int(e1.get("optimizer_seed", -1)) == E1_SEED,
        "e1.steps": int(e1.get("max_optimizer_steps", -1)) == 5000,
        "e1.interval": int(e1.get("validation_interval", -1)) == 500,
        "e1.lr": float(e1.get("code_learning_rate", math.nan)) == 3e-4,
        "e1.min_lr": float(e1.get("code_minimum_learning_rate", math.nan)) == 1e-5,
        "e1.warmup": int(e1.get("warmup_steps", -1)) == 100,
        "e2.seed": int(e2.get("optimizer_seed", -1)) == E2_SEED,
        "e2.steps": int(e2.get("max_optimizer_steps", -1)) == 15000,
        "e2.interval": int(e2.get("validation_interval", -1)) == 1000,
        "e2.code_lr": float(e2.get("code_learning_rate", math.nan)) == 3e-4,
        "e2.code_min_lr": float(e2.get("code_minimum_learning_rate", math.nan)) == 1e-5,
        "e2.decoder_lr": float(e2.get("decoder_learning_rate", math.nan)) == 3e-5,
        "e2.decoder_min_lr": float(e2.get("decoder_minimum_learning_rate", math.nan)) == 1e-6,
        "e2.warmup": int(e2.get("warmup_steps", -1)) == 250,
    }
    failed.extend(name for name, passed in required_stages.items() if not passed)
    source_model = checkpoint["config"]["model"]
    for key in (
        "kind", "d_model", "encoder_layers", "decoder_layers", "heads",
        "ffn_dim", "latent_dim", "dropout",
    ):
        if config.get("model", {}).get(key) != source_model.get(key):
            failed.append(f"source_model.{key}")
    if failed:
        raise ValueError(f"F4E config contract mismatch: {failed}")
    trigger = validate_stop_trigger(
        trigger_run, dataset_hash=dataset_hash, source_checkpoint_hash=checkpoint_hash
    )
    return checkpoint, source_summary, f4a_manifest, trigger


@torch.no_grad()
def initialize_window_codes(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    encoded = torch.empty(EXPECTED_FIXTURES, 256, dtype=torch.float32)
    window_indices = torch.empty(EXPECTED_FIXTURES, dtype=torch.long)
    mask_slots = torch.empty(EXPECTED_FIXTURES, dtype=torch.long)
    filled = 0
    for cpu_batch in loader:
        state_mask, action_mask, _ = make_fixture_masks(cpu_batch, FIXTURE_SEED)
        batch = _device_batch(cpu_batch, device)
        output = model(batch, state_mask.to(device), action_mask.to(device))
        count = int(output.posterior_mean.shape[0])
        encoded[filled : filled + count] = output.posterior_mean.detach().cpu().float()
        window_indices[filled : filled + count] = cpu_batch["window_index"].long()
        mask_slots[filled : filled + count] = cpu_batch["mask_slot"].long()
        filled += count
    if filled != EXPECTED_FIXTURES:
        raise ValueError(f"F4E encoded {filled} fixtures instead of 800")
    centroids = torch.empty(EXPECTED_WINDOWS, 256, dtype=torch.float32)
    per_window: list[dict[str, Any]] = []
    for index in range(EXPECTED_WINDOWS):
        selected = window_indices == index
        slots = sorted(int(value) for value in mask_slots[selected].tolist())
        if slots != list(range(len(FIXED_MASK_NAMES))):
            raise ValueError(f"F4E window {index} does not have exactly the ten Mask slots")
        values = encoded[selected]
        centroid = values.mean(dim=0)
        centroids[index] = centroid
        delta = values - centroid
        per_window.append({
            "window_index": index,
            "posterior_count": int(values.shape[0]),
            "rms_distance_to_centroid": float(torch.sqrt(torch.square(delta).mean())),
            "max_abs_distance_to_centroid": float(delta.abs().max()),
            "centroid_l2_norm": float(torch.linalg.vector_norm(centroid)),
        })
    delta_all = encoded - centroids[window_indices]
    manifest = {
        "format_version": "sonic_posterior_autodecoder_code_initialization_v1",
        "fixture_seed": FIXTURE_SEED,
        "window_count": EXPECTED_WINDOWS,
        "fixture_count": EXPECTED_FIXTURES,
        "latent_dim": 256,
        "shared_code_per_window": True,
        "codes_per_fixture": False,
        "initialization": "arithmetic centroid of ten Mask-conditioned posterior means",
        "posterior_encoding_sha256": _tensor_sha256(encoded),
        "centroid_sha256": _tensor_sha256(centroids),
        "window_index_sha256": _tensor_sha256(window_indices),
        "mask_slot_sha256": _tensor_sha256(mask_slots),
        "global_rms_distance_to_centroid": float(torch.sqrt(torch.square(delta_all).mean())),
        "global_max_abs_distance_to_centroid": float(delta_all.abs().max()),
        "per_window": per_window,
    }
    tensors = {
        "posterior_mean_by_fixture": encoded,
        "window_index_by_fixture": window_indices,
        "mask_slot_by_fixture": mask_slots,
        "initial_window_codes": centroids,
    }
    return centroids, manifest, tensors


def configure_trainable_parameters(model: WindowCodeAutoDecoder, stage: str) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    allow_prefixes = ("window_codes.",)
    if stage == "E2":
        allow_prefixes += (
            "base_model.state_input.",
            "base_model.action_input.",
            "base_model.type_embedding.",
            "base_model.decoder_latent_token",
            "base_model.decoder.",
            "base_model.latent_projection.",
            "base_model.state_continuous_output.",
            "base_model.state_contact_output.",
            "base_model.action_output.",
        )
    if stage not in {"E1", "E2"}:
        raise ValueError("F4E stage must be E1 or E2")
    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        allowed = any(name.startswith(prefix) for prefix in allow_prefixes)
        parameter.requires_grad_(allowed)
        (trainable if allowed else frozen).append(name)
    if not trainable or any(
        name.startswith(("base_model.encoder", "base_model.posterior", "base_model.prior"))
        for name in trainable
    ):
        raise ValueError("F4E trainable-parameter allowlist violated encoder isolation")
    return {
        "stage": stage,
        "trainable_names": trainable,
        "frozen_names": frozen,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "sha256": hashlib.sha256(canonical_json_bytes(trainable)).hexdigest(),
    }


@torch.no_grad()
def evaluate_zero_code_dependence(
    model: WindowCodeAutoDecoder,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    accumulators = {
        name: {"continuous_squared": 0.0, "continuous_count": 0, "legacy_squared": 0.0, "legacy_count": 0}
        for name in ("correct", "zero")
    }
    for cpu_batch in loader:
        state_mask = cpu_batch["valid_state"].bool()[..., None].expand_as(
            cpu_batch["physical_state"]
        ).clone()
        action_mask = cpu_batch["valid_action"].bool()[..., None].expand_as(
            cpu_batch["action"]
        ).clone()
        batch = _device_batch(cpu_batch, device)
        state_mask_device = state_mask.to(device)
        action_mask_device = action_mask.to(device)
        correct = model(batch, state_mask_device, action_mask_device)
        zero = model(
            batch, state_mask_device, action_mask_device,
            latent_override=torch.zeros_like(correct.posterior_mean),
        )
        for name, output in (("correct", correct), ("zero", zero)):
            state_error = torch.square(
                output.physical_state[..., :68] - batch["physical_state"][..., :68]
            ).masked_select(state_mask_device[..., :68])
            action_error = torch.square(
                output.action - batch["action"]
            ).masked_select(action_mask_device)
            legacy_state = torch.square(
                output.physical_state - batch["physical_state"]
            ).masked_select(state_mask_device)
            item = accumulators[name]
            item["continuous_squared"] += float(state_error.sum().cpu()) + float(action_error.sum().cpu())
            item["continuous_count"] += int(state_error.numel() + action_error.numel())
            item["legacy_squared"] += float(legacy_state.sum().cpu()) + float(action_error.sum().cpu())
            item["legacy_count"] += int(legacy_state.numel() + action_error.numel())
    metrics = {
        name: {
            "continuous_combined_rmse_excluding_contact": math.sqrt(
                item["continuous_squared"] / item["continuous_count"]
            ),
            "legacy_combined_rmse_including_contact": math.sqrt(
                item["legacy_squared"] / item["legacy_count"]
            ),
            "continuous_count": item["continuous_count"],
            "legacy_count": item["legacy_count"],
        }
        for name, item in accumulators.items()
    }
    metrics["zero"]["continuous_ratio_to_correct"] = (
        metrics["zero"]["continuous_combined_rmse_excluding_contact"]
        / max(metrics["correct"]["continuous_combined_rmse_excluding_contact"], 1e-12)
    )
    metrics["zero"]["legacy_ratio_to_correct"] = (
        metrics["zero"]["legacy_combined_rmse_including_contact"]
        / max(metrics["correct"]["legacy_combined_rmse_including_contact"], 1e-12)
    )
    return {
        "aggregation": {
            "gate": "State[:68]+Action continuous combined RMSE; excludes contact",
            "legacy": "State[:70]+Action combined RMSE; includes contact probabilities",
        },
        "metrics": metrics,
    }


def _autodecoder_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    exact_gate = evaluation["exact"]["progression_gate"]
    donor = evaluation["latent_donors"]["metrics"]
    zero = float(
        evaluation["zero_code_dependence"]["metrics"]["zero"][
            "continuous_ratio_to_correct"
        ]
    )
    cross_window = float(donor["cross_window"]["continuous_ratio_to_correct"])
    cross_motion = float(donor["cross_motion"]["continuous_ratio_to_correct"])
    ratios = {
        **{
            key: float(value)
            for key, value in exact_gate["threshold_ratios"].items()
            if key != "zero_latent_dependence"
        },
        "zero_code_dependence": 10.0 / max(zero, 1e-12),
        "cross_window_code_dependence": 10.0 / max(cross_window, 1e-12),
        "cross_motion_code_dependence": 10.0 / max(cross_motion, 1e-12),
    }
    score = max(ratios.values())
    return {
        "passed": bool(math.isfinite(score) and score <= 1.0),
        "score": score,
        "threshold_ratios": ratios,
        "thresholds": {
            **exact_gate["thresholds"],
            "cross_window_ratio": 10.0,
            "cross_motion_ratio": 10.0,
        },
        "zero_ratio": zero,
        "legacy_zero_ratio_including_contact": float(
            evaluation["exact"]["latent_dependence"]["zero_ratio"]
        ),
        "cross_window_ratio": cross_window,
        "cross_motion_ratio": cross_motion,
    }


def evaluate_autodecoder(
    *,
    model: WindowCodeAutoDecoder,
    validation_loader: DataLoader[dict[str, Any]],
    base_loader: DataLoader[dict[str, Any]],
    selected_base: DeterministicWindowSubset,
    device: torch.device,
    config: dict[str, Any],
    joint_names: list[str],
    fixed_cases: list[dict[str, Any]],
    output_run: Path,
    step: int,
    stage: str,
    stage_step: int,
) -> dict[str, Any]:
    evaluation = run_full_evaluation(
        model=model,
        validation_loader=validation_loader,
        base_loader=base_loader,
        selected_base=selected_base,
        device=device,
        config=config,
        joint_names=joint_names,
        fixed_cases=fixed_cases,
        output_run=output_run,
        step=step,
    )
    zero_dependence = evaluate_zero_code_dependence(model, base_loader, device)
    zero_path = output_run / f"data/step_{step:05d}_zero_code_dependence.json"
    atomic_write_json(zero_path, zero_dependence)
    evaluation["zero_code_dependence"] = {
        **zero_dependence,
        "artifact": str(zero_path),
    }
    evaluation["stage"] = stage
    evaluation["stage_optimizer_step"] = int(stage_step)
    evaluation["autodecoder_progression_gate"] = _autodecoder_gate(evaluation)
    evaluation["evaluation_scope"] = (
        "identity-conditioned auto-decoder on the same 80 training windows and 800 fixed Masks"
    )
    return evaluation


def _last_three_pass(evaluations: list[dict[str, Any]], required_steps: Iterable[int]) -> bool:
    by_step = {int(row["optimizer_step"]): row for row in evaluations}
    return all(
        step in by_step and bool(by_step[step]["autodecoder_progression_gate"]["passed"])
        for step in required_steps
    )


def _categorical_svg(title: str, rows: list[dict[str, Any]]) -> str:
    width, height = 1280, 700
    left, right, top, bottom = 100.0, 30.0, 80.0, 150.0
    values = [max(float(row[key]), 1e-12) for row in rows for key in ("state", "action", "max_abs")]
    low = max(-12, math.floor(math.log10(min(values))) - 1)
    high = max(0, math.ceil(math.log10(max(values))))
    plot_height = height - top - bottom
    sy = lambda value: top + (high - math.log10(max(float(value), 1e-12))) / (high - low) * plot_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="38" font-size="24" font-weight="700">{html.escape(title)}</text>',
        f'<text x="24" y="{top + plot_height / 2}" transform="rotate(-90 24 {top + plot_height / 2})" text-anchor="middle">Value (log10 scale)</text>',
    ]
    for exponent in range(low, high + 1):
        y = sy(10.0**exponent)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end">10^{exponent}</text>')
    colors = {"state": "#2563eb", "action": "#059669", "max_abs": "#dc2626"}
    group = (width - left - right) / len(rows)
    bar = group * 0.2
    for index, row in enumerate(rows):
        center = left + (index + 0.5) * group
        for offset, key in enumerate(("state", "action", "max_abs")):
            x = center + (offset - 1) * bar
            y = sy(row[key])
            parts.append(f'<rect x="{x-bar*.4:.1f}" y="{y:.1f}" width="{bar*.8:.1f}" height="{top+plot_height-y:.1f}" fill="{colors[key]}"/>')
        label = html.escape(str(row["name"]))
        parts.append(f'<text x="{center:.1f}" y="{top+plot_height+24:.1f}" text-anchor="end" transform="rotate(-35 {center:.1f} {top+plot_height+24:.1f})" font-size="10">{label}</text>')
    for index, key in enumerate(("state", "action", "max_abs")):
        x = left + index * 180
        parts.append(f'<rect x="{x}" y="55" width="14" height="8" fill="{colors[key]}"/>')
        parts.append(f'<text x="{x+20}" y="64">{html.escape(key)}</text>')
    parts.append('<text x="100" y="675" font-size="11">Log plots clip non-positive values to 1e-12.</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def render_plots(
    output_run: Path,
    records: list[dict[str, Any]],
    code_manifest: dict[str, Any],
    best_evaluation: dict[str, Any],
) -> dict[str, str]:
    train = [row for row in records if row["phase"] == "train"]
    evaluations = [row for row in records if row["phase"] == "evaluation"]
    train_plot = train
    if len(train_plot) > 2000:
        indices = np.linspace(0, len(train_plot) - 1, 2000, dtype=np.int64)
        train_plot = [train_plot[int(index)] for index in indices]
    ema: list[tuple[float, float]] = []
    running: float | None = None
    for row in train:
        value = float(row["raw_reconstruction"]["total"])
        running = value if running is None else 0.05 * value + 0.95 * running
        ema.append((float(row["optimizer_step"]), running))
    training_path = output_run / "plots/training_curves.svg"
    gate_path = output_run / "plots/gate_curves.svg"
    mask_path = output_run / "plots/mask_breakdown.svg"
    code_path = output_run / "plots/code_statistics.svg"
    atomic_write_text(training_path, _svg_series(
        "F4E auto-decoder training and full fixed-fixture evaluation",
        [
            ("Train batch raw", [(r["optimizer_step"], r["raw_reconstruction"]["total"]) for r in train_plot], "#94a3b8"),
            ("Train EMA α=0.05", ema, "#2563eb"),
            ("Full fixed-fixture evaluation", [(r["optimizer_step"], r["exact"]["reconstruction_loss"]["total"]) for r in evaluations], "#dc2626"),
            ("Evaluation State MSE", [(r["optimizer_step"], r["exact"]["reconstruction_loss"]["state"]) for r in evaluations], "#0369a1"),
            ("Evaluation Action MSE", [(r["optimizer_step"], r["exact"]["reconstruction_loss"]["action"]) for r in evaluations], "#047857"),
            ("Evaluation contact BCE", [(r["optimizer_step"], r["exact"]["reconstruction_loss"]["contact"]) for r in evaluations], "#c2410c"),
            ("Peak learning rate", [(r["optimizer_step"], max(r["learning_rates"].values())) for r in train_plot], "#7c3aed"),
            ("Gradient norm", [(r["optimizer_step"], r["gradient_norm_before_clip"]) for r in train_plot], "#111827"),
        ],
        y_label="Value (log10 scale)", log_y=True, width=1900,
    ))
    atomic_write_text(gate_path, _svg_series(
        "F4E progression and code-dependence gates",
        [
            ("Worst State RMSE", [(r["optimizer_step"], r["exact"]["worst_state_rmse"]) for r in evaluations], "#2563eb"),
            ("Worst Action RMSE", [(r["optimizer_step"], r["exact"]["worst_action_rmse"]) for r in evaluations], "#059669"),
            ("Continuous max abs", [(r["optimizer_step"], r["exact"]["continuous_max_abs"]) for r in evaluations], "#dc2626"),
            ("Element exceed fraction", [(r["optimizer_step"], r["tail_global"]["threshold_exceed_fraction"]) for r in evaluations], "#7c3aed"),
            ("10 / zero-code ratio", [(r["optimizer_step"], 10.0 / max(r["autodecoder_progression_gate"]["zero_ratio"], 1e-12)) for r in evaluations], "#0f766e"),
            ("10 / cross-window ratio", [(r["optimizer_step"], 10.0 / max(r["autodecoder_progression_gate"]["cross_window_ratio"], 1e-12)) for r in evaluations], "#9333ea"),
            ("10 / cross-motion ratio", [(r["optimizer_step"], 10.0 / max(r["autodecoder_progression_gate"]["cross_motion_ratio"], 1e-12)) for r in evaluations], "#c2410c"),
        ],
        y_label="Value (log10 scale)", log_y=True,
        horizontal_lines=[
            ("progression RMSE/max = 1e-2", 1e-2, "#b91c1c"),
            ("exact max = 1e-3", 1e-3, "#d97706"),
            ("exact RMSE = 1e-4", 1e-4, "#059669"),
            ("code-dependence ratio gate = 1", 1.0, "#7c3aed"),
        ],
        width=1800,
    ))
    cases = best_evaluation["exact"]["cases"]
    atomic_write_text(mask_path, _categorical_svg(
        "F4E best checkpoint: ten fixed Mask types",
        [{
            "name": name,
            "state": cases[name]["worst_state_rmse"],
            "action": cases[name]["worst_action_rmse"],
            "max_abs": cases[name]["continuous_max_abs"],
        } for name in FIXED_MASK_NAMES],
    ))
    atomic_write_text(code_path, _svg_series(
        "F4E initial posterior variation across ten Masks per window",
        [
            ("RMS to centroid", [(row["window_index"], row["rms_distance_to_centroid"]) for row in code_manifest["per_window"]], "#2563eb"),
            ("Max abs to centroid", [(row["window_index"], row["max_abs_distance_to_centroid"]) for row in code_manifest["per_window"]], "#dc2626"),
        ],
        y_label="Value (log10 scale)", log_y=True,
    ).replace("Optimizer step", "Window index"))
    return {
        "training_curves": str(training_path),
        "gate_curves": str(gate_path),
        "mask_breakdown": str(mask_path),
        "code_statistics": str(code_path),
    }


def _checkpoint_payload(
    *,
    model: WindowCodeAutoDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    stage: str,
    global_step: int,
    stage_step: int,
    best_score: float,
    dataset_hash: str,
    source_hash: str,
    fixture_hash: str,
    code_initialization_hash: str,
    trainable_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT,
        "stage": stage,
        "optimizer_step": int(global_step),
        "stage_optimizer_step": int(stage_step),
        "best_progression_score": float(best_score),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "resolved_config": config,
        "dataset_manifest_sha256": dataset_hash,
        "source_checkpoint_sha256": source_hash,
        "fixture_bitmap_sha256": fixture_hash,
        "code_initialization_sha256": code_initialization_hash,
        "base_parameter_count": EXPECTED_PARAMETERS,
        "code_parameter_count": EXPECTED_CODE_PARAMETERS,
        "total_parameter_count": _all_parameter_count(model),
        "trainable_contract": trainable_contract,
    }


def validate_saved_checkpoint(
    path: Path,
    *,
    stage: str,
    expected_step: int,
    dataset_hash: str,
    source_hash: str,
    fixture_hash: str,
    code_initialization_hash: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    code = checkpoint.get("model", {}).get("window_codes.weight")
    checks = {
        "format_version": checkpoint.get("format_version") == CHECKPOINT_FORMAT,
        "stage": checkpoint.get("stage") == stage,
        "optimizer_step": int(checkpoint.get("optimizer_step", -1)) == expected_step,
        "dataset_hash": checkpoint.get("dataset_manifest_sha256") == dataset_hash,
        "source_hash": checkpoint.get("source_checkpoint_sha256") == source_hash,
        "fixture_hash": checkpoint.get("fixture_bitmap_sha256") == fixture_hash,
        "code_initialization_hash": checkpoint.get("code_initialization_sha256") == code_initialization_hash,
        "base_parameters": int(checkpoint.get("base_parameter_count", -1)) == EXPECTED_PARAMETERS,
        "code_parameters": int(checkpoint.get("code_parameter_count", -1)) == EXPECTED_CODE_PARAMETERS,
        "total_parameters": int(checkpoint.get("total_parameter_count", -1)) == EXPECTED_TOTAL_PARAMETERS,
        "code_shape": isinstance(code, torch.Tensor) and tuple(code.shape) == (80, 256),
        "model_state": bool(checkpoint.get("model")),
        "optimizer_state": bool(checkpoint.get("optimizer")),
        "scheduler_state": bool(checkpoint.get("scheduler")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"F4E checkpoint readback failed: {failed}")
    return {"passed": True, "checks": checks, "sha256": file_sha256(path)}


def _encoder_call_counters(base_model: nn.Module) -> tuple[dict[str, int], list[Any]]:
    counts = {"encoder": 0, "posterior_head": 0, "prior_head": 0}
    handles = [
        base_model.encoder.register_forward_pre_hook(
            lambda _module, _inputs: counts.__setitem__("encoder", counts["encoder"] + 1)
        ),
        base_model.posterior.register_forward_pre_hook(
            lambda _module, _inputs: counts.__setitem__("posterior_head", counts["posterior_head"] + 1)
        ),
        base_model.prior.register_forward_pre_hook(
            lambda _module, _inputs: counts.__setitem__("prior_head", counts["prior_head"] + 1)
        ),
    ]
    return counts, handles


def _assert_encoder_isolated(model: WindowCodeAutoDecoder, counts: dict[str, int]) -> dict[str, Any]:
    encoder_names = [
        name for name, _ in model.named_parameters()
        if name.startswith(("base_model.encoder_cls", "base_model.encoder.", "base_model.posterior.", "base_model.prior."))
    ]
    gradients_absent = all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name in encoder_names
    )
    result = {
        "call_counts": dict(counts),
        "zero_calls": all(value == 0 for value in counts.values()),
        "encoder_parameter_count": len(encoder_names),
        "gradients_absent": gradients_absent,
    }
    result["passed"] = bool(result["zero_calls"] and gradients_absent)
    if not result["passed"]:
        raise ValueError(f"F4E encoder isolation failed: {result}")
    return result


def _new_stage_optimizer(
    model: WindowCodeAutoDecoder,
    stage: str,
    config: dict[str, Any],
    max_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, dict[str, Any]]:
    contract = configure_trainable_parameters(model, stage)
    stage_config = config["training"]["stage_e1" if stage == "E1" else "stage_e2"]
    code_parameters = list(model.window_codes.parameters())
    groups: list[dict[str, Any]] = [{
        "params": code_parameters,
        "lr": float(stage_config["code_learning_rate"]),
        "name": "window_codes",
    }]
    lambdas = [lambda step: _learning_rate_multiplier(
        step,
        warmup_steps=int(stage_config["warmup_steps"]),
        max_steps=max_steps,
        minimum_ratio=float(stage_config["code_minimum_learning_rate"]) / float(stage_config["code_learning_rate"]),
    )]
    if stage == "E2":
        code_ids = {id(parameter) for parameter in code_parameters}
        decoder_parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in code_ids
        ]
        groups.append({
            "params": decoder_parameters,
            "lr": float(stage_config["decoder_learning_rate"]),
            "name": "decoder_side",
        })
        lambdas.append(lambda step: _learning_rate_multiplier(
            step,
            warmup_steps=int(stage_config["warmup_steps"]),
            max_steps=max_steps,
            minimum_ratio=float(stage_config["decoder_minimum_learning_rate"]) / float(stage_config["decoder_learning_rate"]),
        ))
    optimizer = torch.optim.AdamW(
        groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    return optimizer, scheduler, contract


def _stage_train_loader(
    fixture_data: MaskBankDataset,
    *,
    micro_batch: int,
    workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        fixture_data,
        batch_size=micro_batch,
        shuffle=True,
        num_workers=workers,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def run_autodecoder_experiment(
    *,
    dataset_run: Path,
    source_checkpoint: Path,
    f4a_run: Path,
    trigger_comparison_run: Path,
    output_run: Path,
    config: dict[str, Any],
    smoke: bool = False,
) -> dict[str, Any]:
    from .dataset import StateActionWindowDataset

    dataset_run = dataset_run.expanduser().resolve()
    source_checkpoint = source_checkpoint.expanduser().resolve()
    f4a_run = f4a_run.expanduser().resolve()
    trigger_comparison_run = trigger_comparison_run.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "plots", "videos"):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    checkpoint, source_summary, f4a_manifest, trigger = validate_contract(
        dataset_run=dataset_run,
        source_checkpoint=source_checkpoint,
        f4a_run=f4a_run,
        trigger_run=trigger_comparison_run,
        output_run=output_run,
        config=config,
    )
    dataset_hash = file_sha256(dataset_run / "manifests/dataset_manifest.json")
    source_hash = file_sha256(source_checkpoint)
    base = StateActionWindowDataset(
        dataset_run, "train", EXPECTED_WINDOW, EXPECTED_WINDOW,
        max_episodes=EXPECTED_MOTIONS * 8, random_crop=False,
    )
    try:
        selected_motions = validate_motion_prefix(base, EXPECTED_MOTIONS)
        selected_base = DeterministicWindowSubset(base, None)
        selected_windows = selected_window_identities(base, selected_base.indices)
        fixture_data = MaskBankDataset(selected_base, len(FIXED_MASK_NAMES))
        if len(selected_base) != EXPECTED_WINDOWS or len(fixture_data) != EXPECTED_FIXTURES:
            raise ValueError("F4E requires exactly 80 windows and 800 fixtures")
        if selected_motions != list(source_summary.get("selected_motion_keys", [])):
            raise ValueError("F4E selected motions differ from F4D")
        if selected_windows != list(source_summary.get("selected_windows", [])):
            raise ValueError("F4E selected windows differ from F4D")
        f4a_fixture = f4a_manifest.get("fixture_contract", {})
        if selected_windows != list(f4a_fixture.get("selected_windows", [])):
            raise ValueError("F4E selected windows differ from F4A")
        resolved = copy.deepcopy(config)
        resolved["model"]["state_dim"] = base.state_dim
        resolved["arm"] = "A"
        resolved["control_dt"] = _load_control_dt(base)
        resolved["training"]["tail_fraction"] = 0.2
        resolved["training"]["tail_mix"] = 0.5
        base_model = build_model(resolved["model"])
        if parameter_count(base_model) != EXPECTED_PARAMETERS:
            raise ValueError("F4E base model parameter count mismatch")
        base_model.load_state_dict(checkpoint["model"], strict=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model.to(device)
        workers = 0 if smoke else int(resolved["data"].get("num_workers", 4))
        micro_batch = int(resolved["training"]["micro_batch"])
        validation_loader = DataLoader(
            fixture_data, batch_size=micro_batch, shuffle=False, num_workers=workers,
            generator=torch.Generator().manual_seed(FIXTURE_SEED + 1), drop_last=False,
            pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        )
        base_loader = DataLoader(
            selected_base, batch_size=micro_batch, shuffle=False, num_workers=workers,
            generator=torch.Generator().manual_seed(FIXTURE_SEED + 2), drop_last=False,
            pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        )
        fixture_hash = fixture_bitmap_sha256(validation_loader, FIXTURE_SEED)
        identity_path = output_run / "manifests/posterior_autodecoder_identity_contract.json"
        atomic_write_json(identity_path, {
            "fixture_seed": FIXTURE_SEED,
            "selected_motion_keys": selected_motions,
            "window_transitions": EXPECTED_WINDOW,
            "selected_windows": selected_windows,
            "mask_names": list(FIXED_MASK_NAMES),
            "fixture_bitmap_sha256": fixture_hash,
            "shared_window_code": True,
        })
        joint_names = _load_joint_names(base)

        source_exact = evaluate_exact(
            base_model, validation_loader, device, FIXTURE_SEED, False,
            {key: float(value) for key, value in resolved["training"]["thresholds"].items()},
            {key: float(value) for key, value in resolved["training"]["progression_thresholds"].items()},
            "progression",
        )
        source_tail = evaluate_tail(
            base_model, validation_loader, device, seed=FIXTURE_SEED,
            state_mean=selected_base.base.state_mean,
            state_std=selected_base.base.state_std,
            action_mean=selected_base.base.action_mean,
            action_std=selected_base.base.action_std,
            joint_names=joint_names,
        )
        source_reproduction = validate_step0(
            source_exact, {"global": source_tail["global"]}, source_summary, checkpoint
        )
        legacy_source_reproduction = validate_source_reproduction(
            source_tail["global"], source_summary, checkpoint
        )

        centroids, code_manifest, code_tensors = initialize_window_codes(
            base_model, validation_loader, device
        )
        code_tensor_path = output_run / "data/posterior_codes_and_centroids.pt"
        code_manifest_path = output_run / "manifests/posterior_code_initialization.json"
        atomic_torch_save(code_tensor_path, code_tensors)
        code_manifest["tensor_artifact"] = str(code_tensor_path)
        code_manifest["tensor_artifact_sha256"] = file_sha256(code_tensor_path)
        atomic_write_json(code_manifest_path, code_manifest)

        model = WindowCodeAutoDecoder(base_model, centroids).to(device)
        if _all_parameter_count(model) != EXPECTED_TOTAL_PARAMETERS:
            raise ValueError("F4E total base+code parameter count mismatch")
        fixed_cases = select_fixed_curve_cases(
            model, selected_base, list(f4a_manifest.get("top_worst_windows", [])),
            device, joint_names,
        )
        fixed_case_hash = identity_sha256(fixed_cases)
        atomic_write_json(
            output_run / "manifests/fixed_velocity_cases.json",
            {"sha256": fixed_case_hash, "cases": fixed_cases},
        )

        metrics_path = output_run / "logs/metrics.jsonl"
        records: list[dict[str, Any]] = []
        evaluations: list[dict[str, Any]] = []
        step0 = evaluate_autodecoder(
            model=model, validation_loader=validation_loader, base_loader=base_loader,
            selected_base=selected_base, device=device, config=resolved,
            joint_names=joint_names, fixed_cases=fixed_cases, output_run=output_run,
            step=0, stage="E1", stage_step=0,
        )
        step0["training_context"] = {"learning_rates": None, "gradient_norm_before_clip": None}
        records.append(step0)
        evaluations.append(step0)
        _append_record(metrics_path, step0)
        best_evaluation = step0
        best_score = float(step0["autodecoder_progression_gate"]["score"])
        started = time.monotonic()
        encoder_counts, hook_handles = _encoder_call_counters(base_model)
        stage_results: dict[str, Any] = {}
        checkpoint_readbacks: dict[str, Any] = {}
        accumulation = int(resolved["training"]["gradient_accumulation"])
        gradient_clip = float(resolved["training"]["gradient_clip"])

        def train_stage(stage: str, global_start: int, formal_steps: int) -> tuple[int, bool, dict[str, Any]]:
            nonlocal best_evaluation, best_score
            stage_key = "stage_e1" if stage == "E1" else "stage_e2"
            stage_config = resolved["training"][stage_key]
            stage_steps = 2 if smoke and stage == "E1" else formal_steps
            seed = int(stage_config["optimizer_seed"])
            seed_everything(seed)
            train_loader = _stage_train_loader(
                fixture_data, micro_batch=micro_batch, workers=workers,
                device=device, seed=seed,
            )
            stream = _infinite(train_loader)
            optimizer, scheduler, trainable_contract = _new_stage_optimizer(
                model, stage, resolved, stage_steps
            )
            stage_best_score = math.inf
            stage_best_eval: dict[str, Any] | None = None
            stage_evaluations: list[dict[str, Any]] = []
            evaluation_interval = 2 if smoke else int(stage_config["validation_interval"])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            stage_started = time.monotonic()
            for stage_step in range(1, stage_steps + 1):
                global_step = global_start + stage_step
                step_started = time.monotonic()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                sums = {"total": 0.0, "state": 0.0, "action": 0.0, "contact": 0.0}
                sampled: list[dict[str, Any]] = []
                for _ in range(accumulation):
                    cpu_batch = next(stream)
                    sampled.extend(batch_sample_identities(cpu_batch))
                    state_mask, action_mask, _ = make_fixture_masks(cpu_batch, FIXTURE_SEED)
                    batch = _device_batch(cpu_batch, device)
                    state_mask = state_mask.to(device)
                    action_mask = action_mask.to(device)
                    output = model(batch, state_mask, action_mask)
                    loss = reconstruction_loss(output, batch, state_mask, action_mask)
                    if not bool(torch.isfinite(loss.total)):
                        raise FloatingPointError("F4E reconstruction loss became non-finite")
                    (loss.total / accumulation).backward()
                    for name in sums:
                        sums[name] += float(getattr(loss, name).detach().cpu()) / accumulation
                trainable_parameters = [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ]
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, gradient_clip)
                if not bool(torch.isfinite(gradient_norm)):
                    raise FloatingPointError("F4E gradient norm became non-finite")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                train_record = {
                    "phase": "train",
                    "stage": stage,
                    "optimizer_step": global_step,
                    "stage_optimizer_step": stage_step,
                    "sample_identity_sha256": identity_sha256(sampled),
                    "sample_count": len(sampled),
                    "raw_reconstruction": dict(sums),
                    "optimization_objective": dict(sums),
                    "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                    "gradient_clip_threshold": gradient_clip,
                    "gradient_was_clipped": float(gradient_norm.detach().cpu()) > gradient_clip,
                    "learning_rates": {
                        str(group.get("name", index)): float(group["lr"])
                        for index, group in enumerate(optimizer.param_groups)
                    },
                    "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                    "step_seconds": time.monotonic() - step_started,
                }
                records.append(train_record)
                _append_record(metrics_path, train_record)
                if stage_step % evaluation_interval != 0 and stage_step != stage_steps:
                    continue
                evaluation = evaluate_autodecoder(
                    model=model, validation_loader=validation_loader, base_loader=base_loader,
                    selected_base=selected_base, device=device, config=resolved,
                    joint_names=joint_names, fixed_cases=fixed_cases, output_run=output_run,
                    step=global_step, stage=stage, stage_step=stage_step,
                )
                evaluation["training_context"] = {
                    "learning_rates": train_record["learning_rates"],
                    "gradient_norm_before_clip": train_record["gradient_norm_before_clip"],
                    "gradient_was_clipped": train_record["gradient_was_clipped"],
                }
                evaluations.append(evaluation)
                stage_evaluations.append(evaluation)
                records.append(evaluation)
                _append_record(metrics_path, evaluation)
                score = float(evaluation["autodecoder_progression_gate"]["score"])
                if score < best_score:
                    best_score = score
                    best_evaluation = evaluation
                if score < stage_best_score:
                    stage_best_score = score
                    stage_best_eval = evaluation
                    name = "best_code_only.pt" if stage == "E1" else "best_coadapt.pt"
                    atomic_torch_save(output_run / f"checkpoints/{name}", _checkpoint_payload(
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        config=resolved, stage=stage, global_step=global_step,
                        stage_step=stage_step, best_score=stage_best_score,
                        dataset_hash=dataset_hash, source_hash=source_hash,
                        fixture_hash=fixture_hash,
                        code_initialization_hash=code_manifest["centroid_sha256"],
                        trainable_contract=trainable_contract,
                    ))
                last_name = "code_only_last.pt" if stage == "E1" else "last.pt"
                atomic_torch_save(output_run / f"checkpoints/{last_name}", _checkpoint_payload(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    config=resolved, stage=stage, global_step=global_step,
                    stage_step=stage_step, best_score=stage_best_score,
                    dataset_hash=dataset_hash, source_hash=source_hash,
                    fixture_hash=fixture_hash,
                    code_initialization_hash=code_manifest["centroid_sha256"],
                    trainable_contract=trainable_contract,
                ))
                render_plots(output_run, records, code_manifest, best_evaluation)
                model.train()
                if stage == "E2" and len(stage_evaluations) >= 3 and all(
                    bool(row["autodecoder_progression_gate"]["passed"])
                    for row in stage_evaluations[-3:]
                ):
                    break
            completed = stage_step
            global_completed = global_start + completed
            if stage == "E1" and not smoke:
                required = (4000, 4500, 5000)
                quality = _last_three_pass(stage_evaluations, required)
            elif stage == "E2" and not smoke:
                quality = len(stage_evaluations) >= 3 and all(
                    bool(row["autodecoder_progression_gate"]["passed"])
                    for row in stage_evaluations[-3:]
                )
            else:
                quality = False
            isolation = _assert_encoder_isolated(model, encoder_counts)
            stage_result = {
                "stage": stage,
                "optimizer_seed": seed,
                "optimizer_reset": True,
                "scheduler_reset": True,
                "loader_rng_reset": True,
                "completed_optimizer_steps": completed,
                "global_completed_optimizer_steps": global_completed,
                "formal_optimizer_steps": formal_steps,
                "validation_interval": evaluation_interval,
                "quality_pass": quality,
                "best_score": stage_best_score,
                "best_optimizer_step": int(stage_best_eval["optimizer_step"]) if stage_best_eval else None,
                "trainable_contract": trainable_contract,
                "encoder_isolation": isolation,
                "elapsed_seconds": time.monotonic() - stage_started,
            }
            return completed, quality, stage_result

        e1_completed, e1_pass, stage_results["E1"] = train_stage("E1", 0, 5000)
        e1_last = output_run / "checkpoints/code_only_last.pt"
        checkpoint_readbacks["E1"] = validate_saved_checkpoint(
            e1_last, stage="E1", expected_step=e1_completed,
            dataset_hash=dataset_hash, source_hash=source_hash,
            fixture_hash=fixture_hash,
            code_initialization_hash=code_manifest["centroid_sha256"],
        )
        final_stage = "E1"
        final_step = e1_completed
        e2_pass = False
        if smoke:
            atomic_torch_save(output_run / "checkpoints/last.pt", torch.load(
                e1_last, map_location="cpu", weights_only=False
            ))
        elif e1_pass:
            atomic_torch_save(output_run / "checkpoints/last.pt", torch.load(
                e1_last, map_location="cpu", weights_only=False
            ))
        else:
            e2_completed, e2_pass, stage_results["E2"] = train_stage("E2", e1_completed, 15000)
            final_stage = "E2"
            final_step = e1_completed + e2_completed
            checkpoint_readbacks["E2"] = validate_saved_checkpoint(
                output_run / "checkpoints/last.pt", stage="E2", expected_step=final_step,
                dataset_hash=dataset_hash, source_hash=source_hash,
                fixture_hash=fixture_hash,
                code_initialization_hash=code_manifest["centroid_sha256"],
            )
        for handle in hook_handles:
            handle.remove()
        plots = render_plots(output_run, records, code_manifest, best_evaluation)
        quality_pass = bool(e1_pass or e2_pass)
        if e1_pass:
            conclusion = "E1_PASS_ENCODER_CODE_BOTTLENECK"
            next_step = "POSTERIOR_TO_CODE_DISTILLATION_OR_AUTODECODER_THEN_ENCODER"
        elif e2_pass:
            conclusion = "E2_PASS_CODE_DECODER_COADAPTATION_BOTTLENECK"
            next_step = "STAGED_CVAE_AUTODECODER_THEN_FROZEN_DECODER_ENCODER"
        else:
            conclusion = "E1_E2_FAIL_GLOBAL_CODE_DECODER_CAPACITY_UNPROVEN"
            next_step = "COMPARE_LARGER_GLOBAL_LATENT_WITH_PER_TIME_LATENT_OR_REVIEW_MAX_ABS_GATE"
        train_records = [row for row in records if row["phase"] == "train"]
        summary = {
            "format_version": FORMAT_VERSION,
            "scope": (
                "seen 80-window identity-conditioned auto-decoder capacity only; "
                "no prior, held-out-Mask, unseen-motion, or State-to-Action inference claim"
            ),
            "execution_pass": True,
            "quality_pass": quality_pass,
            "smoke": bool(smoke),
            "dataset_run": str(dataset_run),
            "dataset_manifest_sha256": dataset_hash,
            "source": {
                "checkpoint": str(source_checkpoint),
                "checkpoint_sha256": source_hash,
                "f4a_run": str(f4a_run),
                "trigger_comparison": trigger,
                "source_reproduction": source_reproduction,
                "legacy_source_reproduction": legacy_source_reproduction,
                "source_exact": source_exact,
                "source_tail_global": source_tail["global"],
            },
            "data_contract": {
                "motion_count": EXPECTED_MOTIONS,
                "selected_motion_keys": selected_motions,
                "window_transitions": EXPECTED_WINDOW,
                "window_count": len(selected_base),
                "fixture_count": len(fixture_data),
                "fixture_bitmap_sha256": fixture_hash,
                "selected_windows_sha256": identity_sha256(selected_windows),
                "fixed_velocity_cases_sha256": fixed_case_hash,
                "identity_contract_sha256": file_sha256(identity_path),
            },
            "model_contract": {
                "base_parameter_count": EXPECTED_PARAMETERS,
                "code_parameter_count": EXPECTED_CODE_PARAMETERS,
                "total_parameter_count": EXPECTED_TOTAL_PARAMETERS,
                "latent_dim": 256,
                "shared_code_per_window": True,
                "per_fixture_code": False,
                "decoder_layer_latent_gates": False,
                "decoder_path_bypasses_encoders": True,
                "kl_beta": 0.0,
                "dropout": 0.0,
                "weight_decay": 0.0,
            },
            "code_initialization": {
                **{key: value for key, value in code_manifest.items() if key != "per_window"},
                "manifest": str(code_manifest_path),
            },
            "training_contract": {
                "micro_batch": micro_batch,
                "gradient_accumulation": accumulation,
                "effective_batch": micro_batch * accumulation,
                "precision": "FP32",
                "completed_optimizer_steps": final_step,
                "maximum_optimizer_steps": 20000,
                "stages": stage_results,
                "mean_train_step_seconds": statistics.fmean(float(row["step_seconds"]) for row in train_records),
            },
            "evaluations": evaluations,
            "best_optimizer_step": int(best_evaluation["optimizer_step"]),
            "best_progression_score": best_score,
            "best_evaluation": best_evaluation,
            "final_stage": final_stage,
            "checkpoint_readback": checkpoint_readbacks,
            "root_cause_assessment": conclusion,
            "unique_next_step": next_step,
            "artifacts": {
                "summary": str(output_run / "manifests/posterior_autodecoder_summary.json"),
                "metrics": str(metrics_path),
                "code_initialization": str(code_manifest_path),
                "code_tensors": str(code_tensor_path),
                "identity_contract": str(identity_path),
                "plots": plots,
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_write_json(output_run / "manifests/posterior_autodecoder_summary.json", summary)
        if smoke:
            atomic_write_text(output_run / f"markers/{SMOKE_MARKER}", "PASS execution_complete=true steps=2 stage=E1\n")
        else:
            atomic_write_text(output_run / f"markers/{EXECUTION_MARKER}", f"PASS execution_complete=true steps={final_step}\n")
            if e1_pass:
                atomic_write_text(output_run / f"markers/{E1_MARKER}", "PASS E1 code_only progression_and_code_dependence=true\n")
            if e2_pass:
                atomic_write_text(output_run / f"markers/{E2_MARKER}", "PASS E2 code_decoder_coadaptation progression_and_code_dependence=true\n")
            if not quality_pass:
                atomic_write_text(output_run / "markers/cvae.failed", "QUALITY_FAIL execution_complete=true E1=false E2=false\n")
        return summary
    finally:
        base.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F4E identity-conditioned auto-decoder diagnostic")
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--f4a-run", type=Path, required=True)
    parser.add_argument("--trigger-comparison-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _write_failure_manifest(output_run: Path, error: BaseException) -> None:
    output_run = output_run.expanduser().resolve()
    atomic_write_json(output_run / "manifests/posterior_autodecoder_failure.json", {
        "format_version": FAILURE_FORMAT,
        "execution_pass": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "metrics_path": str(output_run / "logs/metrics.jsonl"),
    })


def main() -> int:
    args = parse_args()
    try:
        result = run_autodecoder_experiment(
            dataset_run=args.dataset_run,
            source_checkpoint=args.source_checkpoint,
            f4a_run=args.f4a_run,
            trigger_comparison_run=args.trigger_comparison_run,
            output_run=args.output_run,
            config=load_json(args.config),
            smoke=args.smoke,
        )
        print("Posterior F4E auto-decoder: PASS (execution complete)")
        print(json.dumps({
            "output_run": str(args.output_run.expanduser().resolve()),
            "smoke": result["smoke"],
            "quality_pass": result["quality_pass"],
            "completed_optimizer_steps": result["training_contract"]["completed_optimizer_steps"],
            "root_cause_assessment": result["root_cause_assessment"],
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        _write_failure_manifest(args.output_run, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
