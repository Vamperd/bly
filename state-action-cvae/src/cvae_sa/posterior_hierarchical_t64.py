from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from .models import HierarchicalPosteriorTransformer, build_model, parameter_count
from .posterior_capacity import DeterministicWindowSubset, MaskBankDataset, validate_motion_prefix
from .posterior_direct_output import assert_output_isolated
from .posterior_t64_protocol import (
    PHYSICAL_MASK_NAMES,
    append_jsonl,
    evaluate,
    evaluate_diagnostic_masks,
    make_autoencode_masks,
    make_physical_masks,
    mask_bank_sha256,
    reconstruction_loss,
    render_plots,
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


STAGES = ("autoencode", "fixed", "random")


def hierarchical_next_step(profile: str, stage: str, quality_pass: bool) -> str:
    profile = profile.upper()
    if stage not in STAGES or profile not in {"H38", "H50"}:
        raise ValueError("invalid hierarchical decision state")
    if quality_pass:
        return {
            "autoencode": "RUN_HIERARCHICAL_FIXED_PHYSICAL_MASKS",
            "fixed": "RUN_HIERARCHICAL_RANDOM_HELDOUT_MASKS",
            "random": "FREEZE_KL0_BASELINE_THEN_IMPLEMENT_KL_THREE_PATHS",
        }[stage]
    if stage == "autoencode":
        return (
            "RUN_HIERARCHICAL_FIXED_PHYSICAL_MASKS_FROM_BEST_CHECKPOINT"
            if profile == "H38"
            else "STOP_MODEL_SCALING"
        )
    return {
        "fixed": "REVIEW_A_B_FAILURES_BEFORE_SINGLE_H50_REPLICATION",
        "random": "STOP_AND_DIAGNOSE_RANDOM_MASK_COVERAGE",
    }[stage]


def _infinite(loader: DataLoader[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _lr_multiplier(step: int, warmup: int, maximum: int, minimum_ratio: float) -> float:
    if step < warmup:
        return max((step + 1) / max(warmup, 1), 1e-8)
    progress = min((step - warmup) / max(maximum - warmup, 1), 1.0)
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def configure_optimizer(
    model: HierarchicalPosteriorTransformer,
    training: dict[str, Any],
    max_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, dict[str, Any]]:
    fast_prefixes = (
        "state_input.", "action_input.", "input_type_embedding.", "time_embedding.",
        "posterior_cls", "global_head.", "local_head.", "empty_local",
        "global_memory_projection.", "local_memory_projection.", "global_memory_type",
        "local_slot_embedding.", "decoder_query_base", "decoder_type_embedding.",
        "film_global_projection.", "film_local_projection.", "film_projection.",
        "state_continuous_output.", "state_contact_output.", "action_output.",
    )
    fast_names: list[str] = []
    slow_names: list[str] = []
    named = dict(model.named_parameters())
    for name in named:
        if any(name.startswith(prefix) for prefix in fast_prefixes) or ".cross_attention." in name:
            fast_names.append(name)
        else:
            slow_names.append(name)
    if not fast_names or not slow_names or set(fast_names) & set(slow_names):
        raise RuntimeError("hierarchical optimizer parameter partition is invalid")
    slow_lr = float(training["slow_learning_rate"])
    fast_lr = float(training["fast_learning_rate"])
    minimum = float(training["minimum_learning_rate"])
    optimizer = torch.optim.AdamW(
        [
            {"params": [named[name] for name in slow_names], "lr": slow_lr, "name": "encoders_self_attention_ffn"},
            {"params": [named[name] for name in fast_names], "lr": fast_lr, "name": "latent_cross_film_query_output"},
        ],
        weight_decay=0.0,
    )
    warmup = int(training["warmup_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [
            lambda step: _lr_multiplier(step, warmup, max_steps, minimum / slow_lr),
            lambda step: _lr_multiplier(step, warmup, max_steps, minimum / fast_lr),
        ],
    )
    contract = {
        "slow_names": slow_names,
        "fast_names": fast_names,
        "slow_parameter_count": sum(named[name].numel() for name in slow_names),
        "fast_parameter_count": sum(named[name].numel() for name in fast_names),
        "all_parameters_covered_once": len(slow_names) + len(fast_names) == len(named),
        "sha256": hashlib.sha256(canonical_json_bytes({"slow": slow_names, "fast": fast_names})).hexdigest(),
    }
    return optimizer, scheduler, contract


def configure_continuation_scheduler(
    optimizer: torch.optim.Optimizer,
    continuation: dict[str, Any],
    additional_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    peaks = (
        float(continuation["slow_peak_learning_rate"]),
        float(continuation["fast_peak_learning_rate"]),
    )
    minimum = float(continuation["minimum_learning_rate"])
    warmup = int(continuation["warmup_steps"])
    if len(optimizer.param_groups) != len(peaks):
        raise ValueError("H50 continuation expects exactly two optimizer groups")
    if not all(minimum <= peak for peak in peaks):
        raise ValueError("H50 continuation peak learning rates must be at least the minimum")
    for group, peak in zip(optimizer.param_groups, peaks, strict=True):
        group["lr"] = peak
        group["initial_lr"] = peak

    def multiplier(step: int, peak: float) -> float:
        minimum_ratio = minimum / peak
        if step < warmup:
            return minimum_ratio + (1.0 - minimum_ratio) * step / max(warmup, 1)
        progress = min((step - warmup) / max(additional_steps - warmup, 1), 1.0)
        return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [lambda step, peak=peak: multiplier(step, peak) for peak in peaks],
    )


def _model_signature(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "kind", "profile", "d_model", "posterior_encoder_layers", "condition_encoder_layers",
        "decoder_layers", "heads", "ffn_dim", "global_latent_dim", "local_latent_dim",
        "local_chunks", "chunk_transitions", "max_state_steps", "state_dim",
    )
    return {key: config.get(key) for key in keys}


def validate_source_checkpoint(
    path: Path,
    *,
    stage: str,
    dataset_hash: str,
    window_hash: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected_source_stage = "autoencode" if stage == "fixed" else "fixed"
    checks = {
        "format_version": checkpoint.get("format_version") == "sonic_posterior_hierarchical_t64_checkpoint_v1",
        "source_stage": checkpoint.get("stage") == expected_source_stage,
        "dataset_hash": checkpoint.get("dataset_manifest_sha256") == dataset_hash,
        "window_hash": checkpoint.get("selected_windows_sha256") == window_hash,
        "model_signature": checkpoint.get("model_signature") == _model_signature(config["model"]),
        "parameter_count": int(checkpoint.get("parameter_count", -1)) == int(config["model"]["parameter_count"]),
        "model_state": isinstance(checkpoint.get("model"), dict),
    }
    run = path.parents[1]
    summary_path = run / "manifests/posterior_hierarchical_t64_summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    execution_marker = run / "markers/cvae_posterior_hierarchical_t64_execution.ok"
    fit_marker = run / f"markers/cvae_posterior_hierarchical_t64_{expected_source_stage}_fit.ok"
    failure_marker = run / "markers/cvae.failed"
    source_quality_pass = bool(summary.get("quality_pass"))
    failure_marker_matches = bool(
        failure_marker.is_file()
        and failure_marker.read_text(encoding="utf-8").strip()
        == f"QUALITY_FAIL execution_complete=true stage={expected_source_stage} fit=false"
    )
    checks.update({
        "source_summary": bool(summary),
        "source_summary_stage": summary.get("stage") == expected_source_stage,
        "source_summary_profile": summary.get("profile") == config["model"]["profile"],
        "source_formal": not bool(summary.get("smoke", True)),
        "source_execution_pass": bool(summary.get("execution_pass")),
        "source_execution_marker": execution_marker.is_file(),
        "source_dataset_hash": summary.get("dataset_manifest_sha256") == dataset_hash,
        "source_window_hash": summary.get("selected_windows_sha256") == window_hash,
        "source_best_checkpoint_step": int(checkpoint.get("optimizer_step", -1))
        == int(summary.get("best_optimizer_step", -2)),
        "source_quality_marker_consistent": (
            fit_marker.is_file() if source_quality_pass else failure_marker_matches
        ),
    })
    # H38-A is a harder full-both latent stress test, so H38-B accepts either a
    # quality-passed or quality-failed but fully completed A checkpoint. H38-R and
    # every H50 continuation still require their source stage to pass quality.
    relaxed_h38_autoencode = stage == "fixed" and config["model"]["profile"] == "H38"
    if not relaxed_h38_autoencode:
        checks["source_fit_marker"] = fit_marker.is_file() and source_quality_pass
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f"hierarchical model-only source checkpoint failed: {failed}")
    return checkpoint, {
        "checkpoint": str(path),
        "checkpoint_sha256": file_sha256(path),
        "source_run": str(run),
        "source_stage": expected_source_stage,
        "source_summary": str(summary_path),
        "source_summary_sha256": file_sha256(summary_path),
        "source_quality_pass": source_quality_pass,
        "source_fit_marker": fit_marker.is_file(),
        "admission": (
            "completed_autoencode_best_checkpoint"
            if relaxed_h38_autoencode
            else f"quality_passed_{expected_source_stage}_best_checkpoint"
        ),
        "checks": checks,
        "model_only": True,
        "optimizer_scheduler_rng_restored": False,
    }


def validate_f4g_authorization(dataset_run: Path, f4g_run: Path) -> dict[str, Any]:
    f4g_run = f4g_run.expanduser().resolve()
    summary_path = f4g_run / "manifests/posterior_direct_output_summary.json"
    if not summary_path.is_file():
        raise ValueError("H38/H50 requires an F4G-O oracle summary")
    summary = load_json(summary_path)
    checks = {
        "formal": not bool(summary.get("smoke")),
        "oracle_target_copy": bool(summary.get("oracle_target_copy")),
        "oracle_marker": (f4g_run / "markers/cvae_posterior_direct_output_oracle.ok").is_file(),
        "fit_marker": (f4g_run / "markers/cvae_posterior_direct_output_fit.ok").is_file(),
        "execution_pass": bool(summary.get("execution_pass")),
        "quality_pass": bool(summary.get("quality_pass")),
        "motion_count": int(summary.get("motion_count", -1)) == 32,
        "window_transitions": int(summary.get("window_transitions", -1)) == 64,
        "dataset_path": Path(str(summary.get("dataset_run", ""))).resolve() == dataset_run.expanduser().resolve(),
        "dataset_hash": summary.get("dataset_manifest_sha256") == file_sha256(dataset_run / "manifests/dataset_manifest.json"),
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f"F4G-O authorization failed: {failed}")
    return {
        "run": str(f4g_run), "summary": str(summary_path),
        "summary_sha256": file_sha256(summary_path), "checks": checks,
    }


def validate_h50_authorization(
    dataset_run: Path, h38_failed_run: Path | None
) -> dict[str, Any]:
    if h38_failed_run is None:
        raise ValueError("H50-A requires the formal failed H38-B run after failed H38-A")
    run = h38_failed_run.expanduser().resolve()
    summary_path = run / "manifests/posterior_hierarchical_t64_summary.json"
    failure_path = run / "markers/cvae.failed"
    execution_path = run / "markers/cvae_posterior_hierarchical_t64_execution.ok"
    if not summary_path.is_file() or not failure_path.is_file():
        raise ValueError("H50-A authorization is missing H38-B summary or quality-failure marker")
    summary = load_json(summary_path)
    source_run_value = summary.get("initialization", {}).get("source_run")
    source_run = Path(str(source_run_value)).expanduser().resolve() if source_run_value else None
    source_summary_path = (
        source_run / "manifests/posterior_hierarchical_t64_summary.json"
        if source_run is not None else None
    )
    source_summary = (
        load_json(source_summary_path)
        if source_summary_path is not None and source_summary_path.is_file() else {}
    )
    checks = {
        "profile": summary.get("profile") == "H38",
        "stage": summary.get("stage") == "fixed",
        "formal": not bool(summary.get("smoke")),
        "execution_pass": bool(summary.get("execution_pass")),
        "execution_marker": execution_path.is_file(),
        "quality_failed": not bool(summary.get("quality_pass")),
        "failure_marker": failure_path.read_text(encoding="utf-8").strip()
        == "QUALITY_FAIL execution_complete=true stage=fixed fit=false",
        "dataset_path": Path(str(summary.get("dataset_run", ""))).resolve()
        == dataset_run.expanduser().resolve(),
        "source_was_failed_autoencode": (
            summary.get("initialization", {}).get("source_stage") == "autoencode"
            and summary.get("initialization", {}).get("source_quality_pass") is False
        ),
        "source_summary": bool(source_summary),
        "source_profile": source_summary.get("profile") == "H38",
        "source_stage": source_summary.get("stage") == "autoencode",
        "source_formal": not bool(source_summary.get("smoke", True)),
        "source_execution_pass": bool(source_summary.get("execution_pass")),
        "source_quality_failed": source_summary.get("quality_pass") is False,
        "source_failure_marker": bool(
            source_run is not None and (source_run / "markers/cvae.failed").is_file()
        ),
        "source_execution_marker": bool(
            source_run is not None
            and (source_run / "markers/cvae_posterior_hierarchical_t64_execution.ok").is_file()
        ),
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f"H50-A authorization failed: {failed}")
    return {
        "run": str(run), "summary": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "failed_autoencode_run": str(source_run),
        "failed_autoencode_summary": str(source_summary_path),
        "checks": checks,
    }


def validate_h50_continuation_source(
    dataset_run: Path,
    resume_run: Path | None,
    *,
    dataset_hash: str,
    window_hash: str,
    fixture_hash: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Admit only the completed, improving, narrowly failed formal H50-A run."""
    if resume_run is None:
        raise ValueError("H50-A continuation requires CVAE_POSTERIOR_HIERARCHICAL_RESUME_RUN")
    run = resume_run.expanduser().resolve()
    summary_path = run / "manifests/posterior_hierarchical_t64_summary.json"
    checkpoint_path = run / "checkpoints/last.pt"
    metrics_path = run / "logs/metrics.jsonl"
    execution_path = run / "markers/cvae_posterior_hierarchical_t64_execution.ok"
    failure_path = run / "markers/cvae.failed"
    if not summary_path.is_file() or not checkpoint_path.is_file() or not metrics_path.is_file():
        raise ValueError("H50-A continuation source is missing its summary, metrics, or last.pt")
    summary = load_json(summary_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = config["training"].get("continuation", {})
    source_step = int(contract.get("source_optimizer_step", -1))
    maximum_score = float(contract.get("maximum_source_fit_score", -1.0))
    last_three = summary.get("last_three_evaluations", [])
    scores = [float(row.get("fit_gate", {}).get("score", math.inf)) for row in last_three]
    log_evaluations: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("phase") == "evaluation":
                log_evaluations.append(row)
                if len(log_evaluations) > 3:
                    log_evaluations.pop(0)
    last_steps = [int(row.get("optimizer_step", -1)) for row in log_evaluations]
    log_scores = [
        float(row.get("metrics", {}).get("fit_gate", {}).get("score", math.inf))
        for row in log_evaluations
    ]
    expected_last_steps = [source_step - 2000, source_step - 1000, source_step]
    source_sha256 = file_sha256(checkpoint_path)
    source_authorization = summary.get("h50_authorization", {})
    authorization_checks = source_authorization.get("checks", {})
    expected_thresholds = {
        key: float(value) for key, value in config["training"]["fit_thresholds"].items()
    }
    checkpoint_scheduler = checkpoint.get("scheduler", {})
    checks = {
        "profile": summary.get("profile") == "H50",
        "stage": summary.get("stage") == "autoencode",
        "formal": not bool(summary.get("smoke", True)),
        "execution_pass": bool(summary.get("execution_pass")),
        "execution_marker": execution_path.is_file(),
        "quality_failed": summary.get("quality_pass") is False,
        "failure_marker": bool(
            failure_path.is_file()
            and failure_path.read_text(encoding="utf-8").strip()
            == "QUALITY_FAIL execution_complete=true stage=autoencode fit=false"
        ),
        "dataset_path": Path(str(summary.get("dataset_run", ""))).resolve()
        == dataset_run.expanduser().resolve(),
        "dataset_hash": summary.get("dataset_manifest_sha256") == dataset_hash,
        "window_hash": summary.get("selected_windows_sha256") == window_hash,
        "fixture_hash": summary.get("fixture_bitmap_sha256") == fixture_hash,
        "source_completed_step": int(summary.get("completed_optimizer_steps", -1)) == source_step,
        "source_best_at_last": int(summary.get("best_optimizer_step", -1)) == source_step,
        "last_three_steps": last_steps == expected_last_steps,
        "last_three_log_summary_match": len(log_scores) == len(scores) == 3 and all(
            math.isclose(log_score, summary_score, rel_tol=1e-12, abs_tol=1e-12)
            for log_score, summary_score in zip(log_scores, scores, strict=True)
        ),
        "last_three_failed": len(last_three) == 3 and all(
            not bool(row.get("fit_gate", {}).get("passed", True)) for row in last_three
        ),
        "last_three_improving": len(scores) == 3 and scores[0] > scores[1] > scores[2],
        "near_gate": len(scores) == 3 and math.isfinite(scores[-1]) and scores[-1] <= maximum_score,
        "fit_thresholds": summary.get("best_evaluation", {}).get("fit_gate", {}).get("thresholds")
        == expected_thresholds,
        "h50_authorization": bool(authorization_checks) and all(authorization_checks.values()),
        "checkpoint_format": checkpoint.get("format_version")
        == "sonic_posterior_hierarchical_t64_checkpoint_v1",
        "checkpoint_stage": checkpoint.get("stage") == "autoencode",
        "checkpoint_step": int(checkpoint.get("optimizer_step", -1)) == source_step,
        "checkpoint_dataset_hash": checkpoint.get("dataset_manifest_sha256") == dataset_hash,
        "checkpoint_window_hash": checkpoint.get("selected_windows_sha256") == window_hash,
        "checkpoint_fixture_hash": checkpoint.get("fixture_bitmap_sha256") == fixture_hash,
        "checkpoint_model_signature": checkpoint.get("model_signature")
        == _model_signature(config["model"]),
        "checkpoint_parameter_count": int(checkpoint.get("parameter_count", -1))
        == int(config["model"]["parameter_count"]),
        "model_state": isinstance(checkpoint.get("model"), dict),
        "optimizer_state": isinstance(checkpoint.get("optimizer"), dict)
        and bool(checkpoint.get("optimizer", {}).get("state")),
        "scheduler_state": isinstance(checkpoint_scheduler, dict),
        "scheduler_at_source_step": int(checkpoint_scheduler.get("last_epoch", -1)) == source_step,
        "checkpoint_readback_sha256": summary.get("checkpoint_readback", {}).get("sha256")
        == source_sha256,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f"H50-A continuation source failed: {failed}")
    return checkpoint, {
        "mode": "resume_optimizer_tail",
        "source_run": str(run),
        "source_summary": str(summary_path),
        "source_summary_sha256": file_sha256(summary_path),
        "source_metrics": str(metrics_path),
        "source_metrics_sha256": file_sha256(metrics_path),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": source_sha256,
        "source_optimizer_step": source_step,
        "source_best_fit_score": float(summary["best_fit_score"]),
        "source_last_three_scores": scores,
        "source_quality_pass": False,
        "model_only": False,
        "model_optimizer_restored": True,
        "source_scheduler_validated_then_replaced": True,
        "loader_rng_restored": False,
        "loader_rng_note": (
            "the v1 source checkpoint did not store DataLoader generator state; "
            "the continuation restarts ordering with its separately locked seed"
        ),
        "checks": checks,
    }


def compare_continuation_reproduction(
    actual: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    scalar_keys = (
        "global_state_rmse", "global_action_rmse", "worst_mask_state_rmse",
        "worst_mask_action_rmse", "continuous_p99_abs", "continuous_max_abs",
        "contact_accuracy",
    )
    checks = {
        key: math.isclose(
            float(actual.get(key, math.inf)), float(expected.get(key, -math.inf)),
            rel_tol=1e-5, abs_tol=1e-7,
        )
        for key in scalar_keys
    }
    checks["fit_score"] = math.isclose(
        float(actual.get("fit_gate", {}).get("score", math.inf)),
        float(expected.get("fit_gate", {}).get("score", -math.inf)),
        rel_tol=1e-5, abs_tol=1e-7,
    )
    actual_ratios = actual.get("latent_dependence", {}).get("main_ratios", {})
    expected_ratios = expected.get("latent_dependence", {}).get("main_ratios", {})
    for name in ("zero", "cross_window", "cross_motion"):
        checks[f"{name}_latent_ratio"] = math.isclose(
            float(actual_ratios.get(name, math.inf)),
            float(expected_ratios.get(name, -math.inf)),
            rel_tol=1e-5, abs_tol=1e-7,
        )
    return {"passed": all(checks.values()), "checks": checks}


def engineering_checks(
    model: HierarchicalPosteriorTransformer,
    cpu_batch: dict[str, Any],
    device: torch.device,
    fixture_seed: int,
) -> dict[str, Any]:
    model.eval()
    batch = _device_batch(cpu_batch, device)
    fixed_state, fixed_action, _ = make_physical_masks(cpu_batch, fixture_seed)
    full_state, full_action, _ = make_autoencode_masks(cpu_batch)
    fixed_state_d = fixed_state.to(device)
    fixed_action_d = fixed_action.to(device)
    full_state_d = full_state.to(device)
    full_action_d = full_action.to(device)
    with torch.no_grad():
        global_fixed, local_fixed = model.encode_posterior(batch, fixed_state_d, fixed_action_d)
        global_full, local_full = model.encode_posterior(batch, full_state_d, full_action_d)
        invariant = torch.equal(global_fixed, global_full) and torch.equal(local_fixed, local_full)
        decoded = model.decode_from_hierarchical_latent(
            batch, full_state_d, full_action_d, global_full, local_full
        )
        modified = dict(batch)
        modified["physical_state"] = batch["physical_state"].masked_fill(full_state_d, 12345.0)
        modified["action"] = batch["action"].masked_fill(full_action_d, -12345.0)
        isolated = model.decode_from_hierarchical_latent(
            modified, full_state_d, full_action_d, global_full, local_full
        )
        target_isolation = (
            torch.equal(decoded.physical_state, isolated.physical_state)
            and torch.equal(decoded.action, isolated.action)
            and torch.equal(decoded.state_contact_logits, isolated.state_contact_logits)
        )
    checks = {
        "posterior_mask_bitwise_invariant": invariant,
        "condition_masked_truth_isolated": target_isolation,
        "token_count": decoded.physical_state.shape[1] * 2 - 1 == 129,
        "state_shape": tuple(decoded.physical_state.shape[1:]) == (65, 70),
        "action_shape": tuple(decoded.action.shape[1:]) == (64, 29),
        "global_shape": tuple(global_full.shape[1:]) == (256,),
        "local_shape": tuple(local_full.shape[1:]) == (16, 128),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _checkpoint(
    model: HierarchicalPosteriorTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    stage: str,
    dataset_hash: str,
    window_hash: str,
    fixture_hash: str,
    step: int,
    score: float,
) -> dict[str, Any]:
    return {
        "format_version": "sonic_posterior_hierarchical_t64_checkpoint_v1",
        "stage": stage,
        "optimizer_step": step,
        "best_fit_score": score,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "model_signature": _model_signature(config["model"]),
        "dataset_manifest_sha256": dataset_hash,
        "selected_windows_sha256": window_hash,
        "fixture_bitmap_sha256": fixture_hash,
        "parameter_count": parameter_count(model),
    }


def validate_checkpoint(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "format_version": checkpoint.get("format_version") == "sonic_posterior_hierarchical_t64_checkpoint_v1",
        "stage": checkpoint.get("stage") == expected["stage"],
        "dataset_hash": checkpoint.get("dataset_manifest_sha256") == expected["dataset_hash"],
        "window_hash": checkpoint.get("selected_windows_sha256") == expected["window_hash"],
        "fixture_hash": checkpoint.get("fixture_bitmap_sha256") == expected["fixture_hash"],
        "parameter_count": int(checkpoint.get("parameter_count", -1)) == int(expected["parameter_count"]),
        "model_state": isinstance(checkpoint.get("model"), dict),
        "optimizer_state": isinstance(checkpoint.get("optimizer"), dict),
        "scheduler_state": isinstance(checkpoint.get("scheduler"), dict),
    }
    return {"passed": all(checks.values()), "checks": checks, "sha256": file_sha256(path)}


def run_experiment(
    dataset_run: Path,
    output_run: Path,
    config: dict[str, Any],
    *,
    stage: str,
    f4g_run: Path,
    init_checkpoint: Path | None,
    h38_failed_run: Path | None,
    resume_run: Path | None,
    smoke: bool,
) -> dict[str, Any]:
    from .dataset import StateActionWindowDataset

    if stage not in STAGES:
        raise ValueError(f"hierarchical stage must be one of {STAGES}")
    if smoke and stage != "autoencode":
        raise ValueError("hierarchical smoke always exercises the autoencode path")
    continuation = resume_run is not None
    if continuation and (smoke or stage != "autoencode" or config["model"]["profile"] != "H50"):
        raise ValueError("hierarchical continuation is restricted to formal H50 autoencode")
    if continuation and (init_checkpoint is not None or h38_failed_run is not None):
        raise ValueError("H50 continuation cannot also use init checkpoint or H38 authorization")
    dataset_run = dataset_run.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    protected = [dataset_run, f4g_run]
    if init_checkpoint is not None:
        protected.append(init_checkpoint.expanduser().resolve().parents[1])
    if h38_failed_run is not None:
        protected.append(h38_failed_run)
    if resume_run is not None:
        protected.append(resume_run)
    assert_output_isolated(output_run, protected)
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "plots", "videos"):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    if not (dataset_run / "markers/cvae_overfit_subset.ok").is_file():
        raise FileNotFoundError("hierarchical T64 requires the dedicated overfit subset")
    f4g = validate_f4g_authorization(dataset_run, f4g_run)
    data = config["data"]
    training = config["training"]
    if int(data["motion_count"]) != 32 or int(data["window_transitions"]) != 64 or int(data["stride"]) != 64:
        raise ValueError("hierarchical fixed data contract must be 32 motions, T64, stride64")
    if float(training.get("weight_decay", -1.0)) != 0.0 or float(training.get("kl_beta", -1.0)) != 0.0:
        raise ValueError("hierarchical capacity stages require weight_decay=0 and KL beta=0")
    base = StateActionWindowDataset(dataset_run, "train", 64, 64, max_episodes=256, random_crop=False)
    motions = validate_motion_prefix(base, 32)
    selected = DeterministicWindowSubset(base, 2 if smoke else data.get("max_windows"))
    indexed: Dataset[dict[str, Any]] = MaskBankDataset(selected, 1)
    windows = window_identity_rows(base, selected.indices)
    window_hash = rows_sha256(windows)
    dataset_hash = file_sha256(dataset_run / "manifests/dataset_manifest.json")
    model_config = config["model"]
    h50_authorization = None
    if model_config["profile"] == "H50" and stage == "autoencode" and not continuation:
        h50_authorization = validate_h50_authorization(dataset_run, h38_failed_run)
    elif h38_failed_run is not None:
        raise ValueError("H38 failure authorization is accepted only for H50-A")
    model_config["state_dim"] = base.state_dim
    model = build_model(model_config)
    if not isinstance(model, HierarchicalPosteriorTransformer):
        raise TypeError("hierarchical config built the wrong model kind")
    count = parameter_count(model)
    low, high = (int(value) for value in model_config["parameter_count_range"])
    if count != int(model_config["parameter_count"]) or not low <= count <= high:
        raise ValueError(f"hierarchical parameter count {count} violates the locked reference")
    initialization: dict[str, Any]
    if stage == "autoencode" and not continuation:
        if init_checkpoint is not None:
            raise ValueError("hierarchical autoencoding must start from random initialization")
        seed_everything(int(config["seed"]))
        # Rebuild after seeding so random initialization is governed solely by config.seed.
        model = build_model(model_config)
        assert isinstance(model, HierarchicalPosteriorTransformer)
        initialization = {
            "mode": "random", "seed": int(config["seed"]), "model_only": False,
            "optimizer_scheduler_rng_restored": False,
        }
    elif stage != "autoencode":
        if init_checkpoint is None:
            raise ValueError(f"hierarchical {stage} requires a best_fit.pt model-only source")
        checkpoint, initialization = validate_source_checkpoint(
            init_checkpoint, stage=stage, dataset_hash=dataset_hash,
            window_hash=window_hash, config=config,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        seed_everything(int(config["training_seed"]))
    else:
        initialization = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    workers = 0 if smoke else int(data.get("num_workers", 4))
    micro = int(training["micro_batch"])
    fixture_seed = int(config["fixture_seed"])
    heldout_seed = int(config["heldout_seed"])
    if stage == "autoencode":
        train_data: Dataset[dict[str, Any]] = indexed
        validation_data: Dataset[dict[str, Any]] = indexed
        train_maker: Callable[..., Any] = lambda batch, step=None: make_autoencode_masks(batch)
        eval_maker = lambda batch: make_autoencode_masks(batch)
        evaluation_scope = "same 32-motion T64 windows; full-both deterministic posterior autoencoding"
    elif stage == "fixed":
        train_data = MaskBankDataset(selected, len(PHYSICAL_MASK_NAMES))
        validation_data = train_data
        train_maker = lambda batch, step=None: make_physical_masks(batch, fixture_seed)
        eval_maker = lambda batch: make_physical_masks(batch, fixture_seed)
        evaluation_scope = "same 32-motion T64 windows and same eight fixed physical Masks"
    else:
        train_data = indexed
        validation_data = MaskBankDataset(selected, 16)
        train_maker = lambda batch, step: make_physical_masks(
            batch, fixture_seed, dynamic_step=int(step)
        )
        eval_maker = lambda batch: make_physical_masks(batch, heldout_seed, held_out=True)
        evaluation_scope = "seen 32-motion T64 windows; 16 deterministic held-out physical Masks per window"
    continuation_config = training.get("continuation", {}) if continuation else {}
    active_training_seed = int(
        continuation_config.get("training_seed", config["training_seed"])
    )
    generator = torch.Generator().manual_seed(active_training_seed)
    train_loader = DataLoader(
        train_data, batch_size=micro, shuffle=True, num_workers=workers, generator=generator,
        drop_last=not smoke, pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=micro, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    base_loader = DataLoader(indexed, batch_size=micro, shuffle=False, num_workers=workers)
    fixture_hash = mask_bank_sha256(validation_loader, eval_maker)
    resume_checkpoint: dict[str, Any] | None = None
    source_best_evaluation: dict[str, Any] | None = None
    if continuation:
        resume_checkpoint, initialization = validate_h50_continuation_source(
            dataset_run, resume_run, dataset_hash=dataset_hash, window_hash=window_hash,
            fixture_hash=fixture_hash, config=config,
        )
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        source_summary = load_json(Path(initialization["source_summary"]))
        source_best_evaluation = source_summary["best_evaluation"]
    first_batch = next(iter(base_loader))
    engineering = engineering_checks(model, first_batch, device, fixture_seed)
    if not engineering["passed"]:
        raise RuntimeError(f"hierarchical engineering contract failed: {engineering['checks']}")
    stage_config = training["stages"][stage]
    start_step = int(continuation_config.get("source_optimizer_step", 0))
    additional_steps = int(
        continuation_config.get("additional_optimizer_steps", stage_config["max_optimizer_steps"])
    )
    final_step = 2 if smoke else start_step + additional_steps
    validation_interval = 2 if smoke else int(
        continuation_config.get("validation_interval", stage_config["validation_interval"])
    )
    scheduler_horizon = int(stage_config["max_optimizer_steps"])
    optimizer, scheduler, optimizer_contract = configure_optimizer(
        model, training, scheduler_horizon
    )
    optimizer_restore: dict[str, Any] | None = None
    if continuation:
        assert resume_checkpoint is not None
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        minimum_lr = float(training["minimum_learning_rate"])
        restore_checks = {
            "optimizer_group_count": len(optimizer.param_groups) == 2,
            "optimizer_group_names": [group["name"] for group in optimizer.param_groups]
            == ["encoders_self_attention_ffn", "latent_cross_film_query_output"],
            "scheduler_last_epoch": int(scheduler.last_epoch) == start_step,
            "terminal_learning_rates": all(
                math.isclose(float(group["lr"]), minimum_lr, rel_tol=1e-6, abs_tol=1e-12)
                for group in optimizer.param_groups
            ),
        }
        if not all(restore_checks.values()):
            raise RuntimeError(f"H50 continuation optimizer restore failed: {restore_checks}")
        source_learning_rates = {
            group["name"]: float(group["lr"]) for group in optimizer.param_groups
        }
        scheduler = configure_continuation_scheduler(
            optimizer, continuation_config, additional_steps
        )
        optimizer_restore = {
            "passed": True,
            "checks": restore_checks,
            "source_scheduler_horizon": scheduler_horizon,
            "scheduler_policy": continuation_config["scheduler_policy"],
            "source_learning_rates": source_learning_rates,
            "tail_initial_learning_rates": {
                group["name"]: float(group["lr"]) for group in optimizer.param_groups
            },
        }
    # Fresh stages reset every RNG. The v1 checkpoint has no DataLoader-generator
    # state, so continuation restarts sample ordering with a separately locked seed.
    seed_everything(active_training_seed)
    records: list[dict[str, Any]] = []
    metrics_path = output_run / "logs/metrics.jsonl"
    state_std = torch.from_numpy(base.state_std)
    action_std = torch.from_numpy(base.action_std)
    best: dict[str, Any] | None = None
    best_score = math.inf
    best_step = -1
    pass_streak = 0

    def run_evaluation(step: int) -> dict[str, Any]:
        started = time.perf_counter()
        result = evaluate(
            model, validation_loader, base_loader, device, eval_maker,
            fit_thresholds={key: float(value) for key, value in training["fit_thresholds"].items()},
            strict_thresholds={key: float(value) for key, value in training["strict_memory_thresholds"].items()},
            exact_thresholds={key: float(value) for key, value in training["legacy_exact_thresholds"].items()},
            state_std=state_std, action_std=action_std, latent_diagnostics=True,
        )
        result["evaluation_seconds"] = time.perf_counter() - started
        result["evaluation_scope"] = evaluation_scope
        result["posterior_only_diagnostic_masks"] = evaluate_diagnostic_masks(
            model, base_loader, device
        )
        result["artifacts"] = write_evaluation_artifacts(output_run, step, result)
        row = {"phase": "evaluation", "stage": stage, "optimizer_step": step, "metrics": result}
        records.append(row)
        append_jsonl(metrics_path, row)
        return result

    initial = run_evaluation(start_step)
    source_reproduction: dict[str, Any] | None = None
    if continuation:
        assert source_best_evaluation is not None
        source_reproduction = compare_continuation_reproduction(
            initial, source_best_evaluation
        )
        if not source_reproduction["passed"]:
            raise RuntimeError(
                "H50 continuation failed to reproduce its source checkpoint: "
                f"{source_reproduction['checks']}"
            )
        best = initial
        best_score = float(initial["fit_gate"]["score"])
        best_step = start_step
    stream = _infinite(train_loader)
    accumulation = int(training["gradient_accumulation"])
    optimizer.zero_grad(set_to_none=True)
    if continuation:
        atomic_torch_save(
            output_run / "checkpoints/best_fit.pt",
            _checkpoint(
                model, optimizer, scheduler, config, stage, dataset_hash, window_hash,
                fixture_hash, start_step, best_score,
            ),
        )
        atomic_torch_save(
            output_run / "checkpoints/last.pt",
            _checkpoint(
                model, optimizer, scheduler, config, stage, dataset_hash, window_hash,
                fixture_hash, start_step, best_score,
            ),
        )
    for step in range(start_step + 1, final_step + 1):
        started = time.perf_counter()
        aggregate = {key: 0.0 for key in ("total", "state", "action", "contact")}
        sampled: list[dict[str, Any]] = []
        for _ in range(accumulation):
            cpu_batch = next(stream)
            state_mask, action_mask, names = train_maker(cpu_batch, step)
            batch = _device_batch(cpu_batch, device)
            state_mask_d = state_mask.to(device)
            action_mask_d = action_mask.to(device)
            output = model(batch, state_mask_d, action_mask_d)
            losses = reconstruction_loss(output, batch, state_mask_d, action_mask_d)
            (losses["total"] / accumulation).backward()
            for key in aggregate:
                aggregate[key] += float(losses[key].detach().cpu()) / accumulation
            for index, name in enumerate(names):
                sampled.append({
                    "motion_key": str(cpu_batch["motion_key"][index]),
                    "variant_id": int(cpu_batch["variant_id"][index]),
                    "window_start": int(cpu_batch["window_start"][index]),
                    "window_index": int(cpu_batch["window_index"][index]),
                    "mask_name": name,
                })
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        row = {
            "phase": "train", "stage": stage, "optimizer_step": step,
            "reconstruction": aggregate,
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "gradient_norm_before_clip": float(gradient),
            "gradient_was_clipped": float(gradient) > float(training["gradient_clip"]),
            "sample_identity_sha256": hashlib.sha256(canonical_json_bytes(sampled)).hexdigest(),
            "step_seconds": time.perf_counter() - started,
            "cuda_max_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        }
        records.append(row)
        append_jsonl(metrics_path, row)
        if step % validation_interval and step != final_step:
            continue
        result = run_evaluation(step)
        score = float(result["fit_gate"]["score"])
        if score < best_score:
            best, best_score, best_step = result, score, step
            atomic_torch_save(
                output_run / "checkpoints/best_fit.pt",
                _checkpoint(model, optimizer, scheduler, config, stage, dataset_hash, window_hash, fixture_hash, step, score),
            )
        atomic_torch_save(
            output_run / "checkpoints/last.pt",
            _checkpoint(model, optimizer, scheduler, config, stage, dataset_hash, window_hash, fixture_hash, step, best_score),
        )
        pass_streak = pass_streak + 1 if result["fit_gate"]["passed"] else 0
        render_plots(output_run, records, best)
        if not smoke and pass_streak >= int(training["required_pass_streak"]):
            break
        model.train()
    evaluation_rows = [row["metrics"] for row in records if row["phase"] == "evaluation"]
    last_three = evaluation_rows[-3:]
    quality_pass = bool(
        not smoke and len(last_three) == 3 and all(row["fit_gate"]["passed"] for row in last_three)
    )
    strict_pass = bool(
        quality_pass and all(row["strict_memory_gate"]["passed"] for row in last_three)
    )
    legacy_pass = bool(
        quality_pass and all(row["legacy_exact_gate"]["passed"] for row in last_three)
    )
    plots = render_plots(output_run, records, best)
    checkpoint_path = output_run / "checkpoints/last.pt"
    readback = validate_checkpoint(checkpoint_path, {
        "stage": stage, "dataset_hash": dataset_hash, "window_hash": window_hash,
        "fixture_hash": fixture_hash, "parameter_count": count,
    })
    if not readback["passed"]:
        raise RuntimeError("hierarchical checkpoint readback failed")
    next_step = hierarchical_next_step(model_config["profile"], stage, quality_pass)
    summary = {
        "format_version": "sonic_posterior_hierarchical_t64_summary_v1",
        "experiment": (
            f"{model_config['profile']}-{stage}-continue15k"
            if continuation else f"{model_config['profile']}-{stage}"
        ),
        "profile": model_config["profile"],
        "stage": stage,
        "execution_pass": True,
        "smoke": smoke,
        "quality_pass": quality_pass,
        "strict_memory_pass": strict_pass,
        "legacy_exact_pass": legacy_pass,
        "dataset_run": str(dataset_run),
        "dataset_manifest_sha256": dataset_hash,
        "motion_count": 32,
        "episode_count": len(base.episodes),
        "selected_motion_keys": motions,
        "window_transitions": 64,
        "stride": 64,
        "window_count": len(selected),
        "selected_windows": windows,
        "selected_windows_sha256": window_hash,
        "fixture_count": len(validation_data),
        "fixture_bitmap_sha256": fixture_hash,
        "active_mask_names": ["full_both"] if stage == "autoencode" else list(PHYSICAL_MASK_NAMES),
        "evaluation_scope": evaluation_scope,
        "model_contract": {**model_config, "actual_parameter_count": count},
        "optimizer_contract": optimizer_contract,
        "optimizer_restore": optimizer_restore,
        "regularization": {
            "dropout": float(model_config["dropout"]),
            "kl_beta": float(training["kl_beta"]),
            "weight_decay": float(training["weight_decay"]),
        },
        "initialization": initialization,
        "f4g_authorization": f4g,
        "h50_authorization": h50_authorization,
        "continuation": ({
            "enabled": True,
            "source_optimizer_step": start_step,
            "additional_optimizer_step_budget": additional_steps,
            "additional_completed_optimizer_steps": step - start_step,
            "final_optimizer_step_budget": final_step,
            "training_seed": active_training_seed,
            "model_optimizer_restored": True,
            "source_scheduler_validated_then_replaced": True,
            "loader_rng_restored": False,
            "source_reproduction": source_reproduction,
        } if continuation else {"enabled": False}),
        "engineering_checks": engineering,
        "completed_optimizer_steps": step,
        "best_optimizer_step": best_step,
        "best_fit_score": best_score,
        "best_evaluation": best,
        "initial_evaluation": initial,
        "step0_evaluation": initial if not continuation else None,
        "last_three_evaluations": last_three,
        "checkpoint_readback": readback,
        "plots": plots,
        "scope": "posterior memory on seen 32-motion T64 windows; no prior, sampling, unseen-motion, or deployment claim",
        "unique_next_step": "REVIEW_SMOKE_THEN_RUN_H38_AUTOENCODE" if smoke else next_step,
    }
    atomic_write_json(output_run / "manifests/posterior_hierarchical_t64_summary.json", summary)
    marker = "cvae_posterior_hierarchical_t64_smoke.ok" if smoke else "cvae_posterior_hierarchical_t64_execution.ok"
    atomic_write_text(output_run / "markers" / marker, "PASS\n")
    if continuation:
        atomic_write_text(
            output_run / "markers/cvae_posterior_hierarchical_t64_continuation_execution.ok",
            "PASS\n",
        )
    if quality_pass:
        atomic_write_text(
            output_run / f"markers/cvae_posterior_hierarchical_t64_{stage}_fit.ok", "PASS\n"
        )
    elif not smoke:
        atomic_write_text(
            output_run / "markers/cvae.failed",
            f"QUALITY_FAIL execution_complete=true stage={stage} fit=false\n",
        )
    if strict_pass:
        atomic_write_text(output_run / f"markers/cvae_posterior_hierarchical_t64_{stage}_strict_memory.ok", "PASS\n")
    if legacy_pass:
        atomic_write_text(output_run / f"markers/cvae_posterior_hierarchical_t64_{stage}_legacy_exact.ok", "PASS\n")
    base.close()
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run H38/H50 hierarchical posterior T64 capacity stages")
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--f4g-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs/posterior_hierarchical_t64_h38.json")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--h38-failed-run", type=Path)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    summary = run_experiment(
        args.dataset_run, args.output_run, config, stage=args.stage,
        f4g_run=args.f4g_run, init_checkpoint=args.init_checkpoint,
        h38_failed_run=args.h38_failed_run, resume_run=args.resume_run,
        smoke=args.smoke,
    )
    print("Posterior hierarchical T64: PASS (execution complete)")
    print(json.dumps({
        "output_run": str(args.output_run.expanduser().resolve()),
        "profile": summary["profile"], "stage": summary["stage"],
        "smoke": summary["smoke"], "quality_pass": summary["quality_pass"],
        "completed_optimizer_steps": summary["completed_optimizer_steps"],
        "best_fit_score": summary["best_fit_score"],
        "unique_next_step": summary["unique_next_step"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
