from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import time
from typing import Any, Iterable, Iterator

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .models import (
    HierarchicalGaussianLatent,
    HierarchicalStandardCVAETransformer,
    build_model,
)
from .posterior_direct_output import assert_output_isolated
from .posterior_t64_protocol import _svg, make_physical_masks
from .util import (
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    load_config,
    seed_everything,
)


CHECKPOINT_FORMAT = "sonic_65_token_hierarchical_standard_cvae_checkpoint_v1"
SUMMARY_FORMAT = "sonic_65_token_hierarchical_standard_cvae_summary_v1"
STAGES = ("fixed", "random", "kl")


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _infinite(loader: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def _model_signature(model_config: dict[str, Any], parameter_count: int | None = None) -> dict[str, Any]:
    return HierarchicalStandardCVAETransformer.architecture_signature(model_config, parameter_count)


def initialize_from_h50_a(model: HierarchicalStandardCVAETransformer, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Legacy entry retained only to fail explicitly instead of migrating weights."""
    del model, checkpoint
    raise ValueError("architecture signature mismatch: H50-A migration is disabled for the 65-token CVAE")


def hierarchical_kl(
    posterior: HierarchicalGaussianLatent,
    prior: HierarchicalGaussianLatent | None = None,
) -> dict[str, torch.Tensor]:
    """KL(q||N(0,I)); ``prior`` is ignored and exists only for old callers."""
    del prior
    global_kl = 0.5 * (torch.exp(posterior.global_logvar) + posterior.global_mean.square() - 1.0 - posterior.global_logvar).mean()
    local_kl = 0.5 * (torch.exp(posterior.local_logvar) + posterior.local_mean.square() - 1.0 - posterior.local_logvar).mean()
    return {"global": global_kl, "local": local_kl, "total": 0.5 * (global_kl + local_kl)}


def _full_target_masks(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
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
) -> dict[str, torch.Tensor]:
    """Return masked/full reconstruction terms without changing target semantics."""
    if abs(masked_weight + full_weight - 1.0) > 1e-6:
        raise ValueError("masked/full reconstruction weights must sum to one")

    def one(sm: torch.Tensor, am: torch.Tensor) -> dict[str, torch.Tensor]:
        state_mask_cont = sm[..., :68]
        state_values = (output.physical_state[..., :68] - batch["physical_state"][..., :68]).square().masked_select(state_mask_cont)
        action_values = (output.action - batch["action"]).square().masked_select(am)
        contact_values = F.binary_cross_entropy_with_logits(
            output.state_contact_logits, batch["physical_state"][..., 68:70], reduction="none"
        ).masked_select(sm[..., 68:70])
        zeros = output.action.sum() * 0.0
        state = state_values.mean() if state_values.numel() else zeros
        action = action_values.mean() if action_values.numel() else zeros
        contact = contact_values.mean() if contact_values.numel() else zeros
        available = [value for value in (state_values, action_values, contact_values) if value.numel()]
        if not available:
            raise ValueError("reconstruction target contains no valid elements")
        return {"total": torch.stack((state, action, contact)).mean(), "state": state, "action": action, "contact": contact}

    masked = one(state_mask, action_mask)
    full = one(*_full_target_masks(batch))
    return {
        "masked": masked,
        "full": full,
        **{name: masked_weight * masked[name] + full_weight * full[name] for name in ("total", "state", "action", "contact")},
    }


def _new_checkpoint(
    model: HierarchicalStandardCVAETransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: dict[str, Any],
    *,
    stage: str,
    step: int,
    training_contract: dict[str, Any] | None = None,
    best_step: int | None = None,
    best_metrics: dict[str, Any] | None = None,
    dataset_identity: dict[str, Any] | None = None,
    data_loader_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "format_version": CHECKPOINT_FORMAT,
        "architecture_version": model.ARCHITECTURE_VERSION,
        "stage": stage,
        "optimizer_step": int(step),
        "model_signature": _model_signature(config["model"], count),
        "parameter_count": count,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "training_contract": training_contract,
        "best_optimizer_step": best_step,
        "best_metrics": best_metrics,
        "dataset_identity": dataset_identity,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "data_loader_generator": (
                data_loader_generator.get_state()
                if data_loader_generator is not None
                else None
            ),
        },
    }


def validate_checkpoint(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("architecture signature mismatch: checkpoint is not the 65-token CVAE format")
    signature = checkpoint.get("model_signature", {})
    if signature.get("architecture_version") != HierarchicalStandardCVAETransformer.ARCHITECTURE_VERSION:
        raise ValueError("architecture signature mismatch: unsupported architecture_version")
    checks = {
        "format": True,
        "architecture_version": True,
        "model": isinstance(checkpoint.get("model"), dict),
        "parameter_count": int(checkpoint.get("parameter_count", -1)) == int(signature.get("parameter_count", -2)),
    }
    if expected:
        if "stage" in expected:
            checks["stage"] = checkpoint.get("stage") == expected["stage"]
        if "parameters" in expected:
            checks["parameters_expected"] = int(checkpoint.get("parameter_count", -1)) == int(expected["parameters"])
    return {"passed": all(checks.values()), "checks": checks, "sha256": file_sha256(path)}


def load_checkpoint(model: HierarchicalStandardCVAETransformer, path: Path, *, strict: bool = True) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    signature = checkpoint.get("model_signature", {})
    expected = model.architecture_signature(model.config, sum(parameter.numel() for parameter in model.parameters()))
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT or signature.get("architecture_version") != model.ARCHITECTURE_VERSION:
        raise ValueError("architecture signature mismatch: legacy H50 checkpoint cannot be loaded")
    if signature != expected:
        raise ValueError("architecture signature mismatch: checkpoint structure differs from current model")
    model.load_state_dict(checkpoint["model"], strict=strict)
    return checkpoint


def _stage_key(stage: str) -> tuple[str, str]:
    if stage == "fixed":
        return "A", "posterior"
    if stage == "random":
        return "B", "condition"
    if stage == "kl":
        return "C", "kl"
    if stage.upper() in {"A", "B", "C"}:
        code = stage.upper()
        return code, {"A": "posterior", "B": "condition", "C": "kl"}[code]
    raise ValueError("stage must be fixed/random/kl or A/B/C")


def _standard_reconstruction_loss(output: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    state_mask, action_mask = _full_target_masks(batch)
    state = (output.physical_state[..., :68] - batch["physical_state"][..., :68]).square().masked_select(state_mask[..., :68])
    action = (output.action - batch["action"]).square().masked_select(action_mask)
    contact = F.binary_cross_entropy_with_logits(
        output.state_contact_logits, batch["physical_state"][..., 68:70], reduction="none"
    ).masked_select(state_mask[..., 68:70])
    values = [value.mean() for value in (state, action, contact) if value.numel()]
    if not values:
        raise ValueError("reconstruction batch has no valid targets")
    return torch.stack(values).mean()


STATE_CONTINUOUS_FEATURE_NAMES = tuple(
    [f"joint_pos_{index}" for index in range(29)]
    + [f"joint_vel_{index}" for index in range(29)]
    + [f"base_lin_vel_{index}" for index in range(3)]
    + [f"base_ang_vel_{index}" for index in range(3)]
    + [f"gravity_robot_{index}" for index in range(3)]
    + ["base_height"]
)
ACTION_FEATURE_NAMES = tuple(f"action_{index}" for index in range(29))


def _batch_value(batch: dict[str, Any], key: str, index: int, default: Any = None) -> Any:
    value = batch.get(key, default)
    if isinstance(value, torch.Tensor):
        return value[index].item()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _window_identity(batch: dict[str, Any], index: int, ordinal: int) -> dict[str, Any]:
    return {
        "window_index": ordinal,
        "motion_key": str(_batch_value(batch, "motion_key", index, "unknown")),
        "episode_ref": str(_batch_value(batch, "episode_ref", index, "unknown")),
        "variant_id": int(_batch_value(batch, "variant_id", index, -1)),
        "window_start": int(_batch_value(batch, "window_start", index, -1)),
    }


@torch.no_grad()
def _evaluate_full_sequence(
    model: HierarchicalStandardCVAETransformer,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    stage: str = "A",
    mask_seed: int | None = None,
) -> dict[str, Any]:
    """Complete evaluation with global and tail-error diagnostics.

    Stage A evaluates the complete-sequence posterior path.  Stages B/C must
    evaluate the masked condition path; otherwise their metrics would silently
    measure the easier posterior reconstruction instead of the requested
    condition-encoder task.
    """
    stage = stage.upper()
    if stage not in {"A", "B", "C"}:
        raise ValueError("evaluation stage must be A, B or C")
    if stage != "A" and mask_seed is None:
        raise ValueError("masked condition evaluation requires mask_seed")
    model.eval()
    state_sse = action_sse = contact_loss = 0.0
    state_count = action_count = contact_count = 0
    max_abs = 0.0
    absolute_values: list[torch.Tensor] = []
    state_absolute_values: list[torch.Tensor] = []
    action_absolute_values: list[torch.Tensor] = []
    state_feature_sse = torch.zeros(68, dtype=torch.float64)
    state_feature_count = torch.zeros(68, dtype=torch.long)
    state_feature_max = torch.zeros(68, dtype=torch.float32)
    action_feature_sse = torch.zeros(29, dtype=torch.float64)
    action_feature_count = torch.zeros(29, dtype=torch.long)
    action_feature_max = torch.zeros(29, dtype=torch.float32)
    window_rows: list[dict[str, Any]] = []
    ordinal = 0
    for cpu_batch in loader:
        batch = _device_batch(cpu_batch, device)
        mask_names: list[str] | None = None
        if stage == "A":
            output = model(batch, stage="A")
        else:
            state_mask, action_mask, mask_names = make_physical_masks(
                cpu_batch, int(mask_seed),
            )
            output = model(
                batch,
                state_mask.to(device),
                action_mask.to(device),
                stage=stage,
            )
        state_error = output.physical_state[..., :68] - batch["physical_state"][..., :68]
        action_error = output.action - batch["action"]
        state_valid = batch["valid_state"].bool()[..., None].expand_as(state_error)
        action_valid = batch["valid_action"].bool()[..., None].expand_as(action_error)
        state_sse += float(state_error.square().masked_select(state_valid).sum().detach().cpu())
        action_sse += float(action_error.square().masked_select(action_valid).sum().detach().cpu())
        state_count += int(state_valid.sum())
        action_count += int(action_valid.sum())
        contact = F.binary_cross_entropy_with_logits(
            output.state_contact_logits,
            batch["physical_state"][..., 68:70],
            reduction="none",
        )
        contact_valid = batch["valid_state"].bool()[..., None].expand_as(contact)
        contact_loss += float(contact.masked_select(contact_valid).sum().detach().cpu())
        contact_count += int(contact_valid.sum())
        state_error_cpu = state_error.detach().float().cpu()
        action_error_cpu = action_error.detach().float().cpu()
        state_valid_cpu = state_valid.detach().cpu()
        action_valid_cpu = action_valid.detach().cpu()
        state_abs_cpu = state_error_cpu.abs().masked_select(state_valid_cpu)
        action_abs_cpu = action_error_cpu.abs().masked_select(action_valid_cpu)
        if state_abs_cpu.numel():
            state_absolute_values.append(state_abs_cpu)
            absolute_values.append(state_abs_cpu)
        if action_abs_cpu.numel():
            action_absolute_values.append(action_abs_cpu)
            absolute_values.append(action_abs_cpu)
        state_feature_sse += (state_error_cpu.square() * state_valid_cpu).sum(dim=(0, 1), dtype=torch.float64)
        state_feature_count += state_valid_cpu.sum(dim=(0, 1)).to(torch.long)
        state_feature_max = torch.maximum(
            state_feature_max,
            (state_error_cpu.abs() * state_valid_cpu).amax(dim=(0, 1)),
        )
        action_feature_sse += (action_error_cpu.square() * action_valid_cpu).sum(dim=(0, 1), dtype=torch.float64)
        action_feature_count += action_valid_cpu.sum(dim=(0, 1)).to(torch.long)
        action_feature_max = torch.maximum(
            action_feature_max,
            (action_error_cpu.abs() * action_valid_cpu).amax(dim=(0, 1)),
        )
        for index in range(state_error_cpu.shape[0]):
            state_values = state_error_cpu[index].square().masked_select(state_valid_cpu[index])
            action_values = action_error_cpu[index].square().masked_select(action_valid_cpu[index])
            state_mse = float(state_values.mean()) if state_values.numel() else 0.0
            action_mse = float(action_values.mean()) if action_values.numel() else 0.0
            window_rows.append({
                **_window_identity(cpu_batch, index, ordinal),
                "state_rmse": state_mse ** 0.5,
                "action_rmse": action_mse ** 0.5,
                "combined_rmse": ((state_mse + action_mse) / 2.0) ** 0.5,
                "mask_name": mask_names[index] if mask_names is not None else None,
                "max_abs": max(
                    float(state_error_cpu[index].abs().masked_select(state_valid_cpu[index]).max())
                    if state_values.numel() else 0.0,
                    float(action_error_cpu[index].abs().masked_select(action_valid_cpu[index]).max())
                    if action_values.numel() else 0.0,
                ),
            })
            ordinal += 1
        max_abs = max(
            max_abs,
            float(state_error.masked_select(state_valid).abs().max().detach().cpu()) if state_valid.any() else 0.0,
            float(action_error.masked_select(action_valid).abs().max().detach().cpu()) if action_valid.any() else 0.0,
        )
    state_rmse = (state_sse / max(state_count, 1)) ** 0.5
    action_rmse = (action_sse / max(action_count, 1)) ** 0.5
    contact_bce = contact_loss / max(contact_count, 1)
    all_absolute = torch.cat(absolute_values) if absolute_values else torch.zeros(1)
    state_absolute = torch.cat(state_absolute_values) if state_absolute_values else torch.zeros(1)
    action_absolute = torch.cat(action_absolute_values) if action_absolute_values else torch.zeros(1)

    def quantile(values: torch.Tensor, level: float) -> float:
        return float(torch.quantile(values, level).item())

    def feature_rows(
        sse: torch.Tensor,
        count: torch.Tensor,
        maximum: torch.Tensor,
        names: tuple[str, ...],
        domain: str,
    ) -> list[dict[str, Any]]:
        rows = []
        for index, name in enumerate(names):
            rows.append({
                "domain": domain,
                "feature_index": index,
                "name": name,
                "rmse": float((sse[index] / max(int(count[index]), 1)).sqrt()),
                "max_abs": float(maximum[index]),
                "count": int(count[index]),
            })
        return sorted(rows, key=lambda row: (row["rmse"], row["max_abs"]), reverse=True)

    state_features = feature_rows(
        state_feature_sse, state_feature_count, state_feature_max,
        STATE_CONTINUOUS_FEATURE_NAMES, "state",
    )
    action_features = feature_rows(
        action_feature_sse, action_feature_count, action_feature_max,
        ACTION_FEATURE_NAMES, "action",
    )
    worst_window = max(window_rows, key=lambda row: row["combined_rmse"], default=None)
    worst_window_by_max_abs = max(window_rows, key=lambda row: row["max_abs"], default=None)
    mask_breakdown: dict[str, dict[str, Any]] = {}
    for row in window_rows:
        name = row.get("mask_name")
        if not name:
            continue
        group = mask_breakdown.setdefault(
            str(name),
            {"count": 0, "state_rmse": [], "action_rmse": [], "combined_rmse": [], "max_abs": []},
        )
        group["count"] += 1
        for key in ("state_rmse", "action_rmse", "combined_rmse", "max_abs"):
            group[key].append(float(row[key]))
    for group in mask_breakdown.values():
        for key in ("state_rmse", "action_rmse", "combined_rmse", "max_abs"):
            values = group.pop(key)
            group[f"mean_{key}"] = sum(values) / max(len(values), 1)
            group[f"worst_{key}"] = max(values, default=0.0)
    return {
        "evaluation_stage": stage,
        "condition_mask_seed": int(mask_seed) if mask_seed is not None else None,
        "total_loss": (state_sse / max(state_count, 1) + action_sse / max(action_count, 1) + contact_bce) / 3.0,
        "state_rmse": state_rmse,
        "action_rmse": action_rmse,
        "contact_bce": contact_bce,
        "max_abs": max_abs,
        "continuous_abs_p95": quantile(all_absolute, 0.95),
        "continuous_abs_p99": quantile(all_absolute, 0.99),
        "state_abs_p95": quantile(state_absolute, 0.95),
        "state_abs_p99": quantile(state_absolute, 0.99),
        "action_abs_p95": quantile(action_absolute, 0.95),
        "action_abs_p99": quantile(action_absolute, 0.99),
        "worst_window": worst_window,
        "worst_window_by_max_abs": worst_window_by_max_abs,
        "worst_windows": sorted(window_rows, key=lambda row: row["combined_rmse"], reverse=True)[:10],
        "mask_breakdown": mask_breakdown,
        "worst_state_features": state_features[:10],
        "worst_action_features": action_features[:10],
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _checkpoint_contract(
    *,
    stage: str,
    maximum: int,
    micro_batch: int,
    learning_rate: float,
    schedule_name: str,
    warmup_steps: int,
    min_lr_ratio: float,
    validation_interval: int,
    checkpoint_interval: int,
    log_interval: int,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "max_steps": maximum,
        "micro_batch": micro_batch,
        "learning_rate": learning_rate,
        "lr_schedule": schedule_name,
        "warmup_steps": warmup_steps,
        "min_lr_ratio": min_lr_ratio,
        "validation_interval": validation_interval,
        "checkpoint_interval": checkpoint_interval,
        "log_interval": log_interval,
    }


def _render_training_plot(output_run: Path, records: list[dict[str, Any]]) -> str:
    train_rows = [row for row in records if row.get("phase") == "train"]
    eval_rows = [
        row for row in records
        if row.get("phase") == "evaluation" and "total_loss" in row
    ]
    path = output_run / "plots/training_curves.svg"
    _svg(
        path,
        "65-token Posterior training",
        "Train and complete-sequence evaluation losses (log10 scale)",
        [
            ("Train loss", [(row["optimizer_step"], row["loss"]) for row in train_rows], "#9ecae1"),
            ("Eval total", [(row["optimizer_step"], row["total_loss"]) for row in eval_rows], "#08519c"),
            ("Eval State RMSE", [(row["optimizer_step"], row["state_rmse"]) for row in eval_rows], "#238b45"),
            ("Eval Action RMSE", [(row["optimizer_step"], row["action_rmse"]) for row in eval_rows], "#d95f0e"),
        ],
    )
    return str(path)


def run_experiment(
    dataset_run: Path,
    output_run: Path,
    source_run: Path | None,
    config: dict[str, Any],
    *,
    stage: str,
    init_run: Path | None = None,
    smoke: bool = False,
    kl_beta_override: float | None = None,
    max_steps_override: int | None = None,
    micro_batch_override: int | None = None,
    learning_rate_override: float | None = None,
    lr_schedule: str | None = None,
    warmup_steps_override: int | None = None,
    min_lr_ratio_override: float | None = None,
    validation_interval_override: int | None = None,
    checkpoint_interval_override: int | None = None,
    log_interval_override: int | None = None,
    resume_run: Path | None = None,
) -> dict[str, Any]:
    """Run only the engineering-safe 65-token training path."""
    del source_run
    stage_code, stage_name = _stage_key(stage)
    if kl_beta_override is not None and stage_code != "C":
        raise ValueError("KL beta override is only valid for Stage C")
    dataset_run = Path(dataset_run).expanduser().resolve()
    output_run = Path(output_run).expanduser().resolve()
    assert_output_isolated(output_run, [dataset_run])
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "plots"):
        (output_run / child).mkdir(parents=True, exist_ok=True)

    from .dataset import StateActionWindowDataset

    data_cfg = config.get("data", {})
    window = int(data_cfg.get("window_transitions", 64))
    stride = int(data_cfg.get("stride", 64))
    if window != 64:
        raise ValueError("65-token CVAE requires 64 transitions per window")
    dataset = StateActionWindowDataset(
        dataset_run,
        "train",
        window,
        stride,
        max_episodes=data_cfg.get("max_episodes", 256),
        random_crop=False,
    )
    limit = 2 if smoke else data_cfg.get("max_windows")
    indices = list(range(len(dataset) if limit is None else min(int(limit), len(dataset))))
    if not indices:
        raise ValueError("dataset contains no training windows")

    model_cfg = dict(config["model"])
    model_cfg.setdefault("state_dim", 70)
    model_cfg.setdefault("state_input_dim", 99)
    model_cfg.setdefault("condition_input_dim", 198)
    model_cfg.setdefault("architecture_version", HierarchicalStandardCVAETransformer.ARCHITECTURE_VERSION)
    with torch.device("meta"):
        meta_model = build_model(model_cfg)
    if not isinstance(meta_model, HierarchicalStandardCVAETransformer):
        raise TypeError("65-token config built the wrong model")
    parameter_total = sum(parameter.numel() for parameter in meta_model.parameters())
    del meta_model

    seed_everything(int(config.get("initialization_seed", 20260921)))
    model = build_model(model_cfg)
    if not isinstance(model, HierarchicalStandardCVAETransformer):
        raise TypeError("65-token config built the wrong model")
    if init_run is not None:
        init_run = Path(init_run).expanduser().resolve()
        candidate = init_run / "checkpoints/best.pt"
        if not candidate.is_file():
            candidate = init_run / "checkpoints/last.pt"
        if not candidate.is_file():
            raise FileNotFoundError("new CVAE stage initialization checkpoint is missing")
        load_checkpoint(model, candidate)
    stage_counts = model.set_training_stage(stage_code)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training_cfg = config.get("training", {})
    loader_generator = torch.Generator().manual_seed(int(config.get("initialization_seed", 20260921)))
    micro_batch = int(
        micro_batch_override
        if micro_batch_override is not None
        else training_cfg.get("micro_batch", 2)
    )
    if micro_batch <= 0:
        raise ValueError("micro batch must be positive")
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=micro_batch,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=loader_generator,
    )
    eval_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=micro_batch,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    learning_rate = float(
        learning_rate_override
        if learning_rate_override is not None
        else training_cfg.get("learning_rate", 1e-4)
    )
    if learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    schedule_name = str(lr_schedule or training_cfg.get("lr_schedule", "constant")).lower()
    if schedule_name not in {"constant", "cosine", "linear"}:
        raise ValueError("lr schedule must be constant, cosine, or linear")
    maximum = int(
        max_steps_override
        if max_steps_override is not None
        else training_cfg.get(stage_name, {}).get("max_optimizer_steps", 1)
    )
    if maximum <= 0:
        raise ValueError("max optimizer steps must be positive")
    warmup_steps = int(
        warmup_steps_override
        if warmup_steps_override is not None
        else training_cfg.get("warmup_steps", 0)
    )
    if warmup_steps < 0:
        raise ValueError("warmup steps cannot be negative")
    min_lr_ratio = float(
        min_lr_ratio_override
        if min_lr_ratio_override is not None
        else training_cfg.get("min_lr_ratio", 0.01)
    )
    if not 0.0 < min_lr_ratio <= 1.0:
        raise ValueError("minimum LR ratio must be in (0, 1]")
    validation_interval = int(
        validation_interval_override
        if validation_interval_override is not None
        else training_cfg.get("validation_interval", 1000)
    )
    checkpoint_interval = int(
        checkpoint_interval_override
        if checkpoint_interval_override is not None
        else training_cfg.get("checkpoint_interval", 250)
    )
    log_interval = int(
        log_interval_override
        if log_interval_override is not None
        else training_cfg.get("log_interval", 20)
    )
    if validation_interval <= 0 or checkpoint_interval <= 0 or log_interval <= 0:
        raise ValueError("validation, checkpoint, and log intervals must be positive")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=0.0,
    )
    def lr_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        if schedule_name == "constant":
            return 1.0
        progress = min(
            max((step - warmup_steps) / max(maximum - warmup_steps, 1), 0.0),
            1.0,
        )
        if schedule_name == "linear":
            return 1.0 - progress * (1.0 - min_lr_ratio)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(progress * math.pi))
    if smoke:
        maximum = min(maximum, 2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    contract = _checkpoint_contract(
        stage=stage_code, maximum=maximum, micro_batch=micro_batch,
        learning_rate=learning_rate,
        schedule_name=schedule_name, warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio, validation_interval=validation_interval,
        checkpoint_interval=checkpoint_interval, log_interval=log_interval,
    )
    dataset_identity = {
        "dataset_run": str(dataset_run),
        "dataset_manifest_sha256": file_sha256(dataset_run / "manifests/dataset_manifest.json")
        if (dataset_run / "manifests/dataset_manifest.json").is_file() else None,
        "selected_window_count": len(indices),
        "window": window,
        "stride": stride,
    }
    metrics_path = output_run / "logs/metrics.jsonl"
    records = _read_jsonl(metrics_path)
    start_step = 0
    best_step: int | None = None
    best_metrics: dict[str, Any] | None = None
    best_score = math.inf
    resume_source = None
    if resume_run is not None:
        resume_source = Path(resume_run).expanduser().resolve()
        if resume_source != output_run:
            raise ValueError("resume-run must be the output run itself; refusing split-brain resume")
        resume_checkpoint = output_run / "checkpoints/last.pt"
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint is missing: {resume_checkpoint}")
        checkpoint = load_checkpoint(model, resume_checkpoint)
        if checkpoint.get("stage") != stage_code:
            raise ValueError("resume checkpoint stage does not match requested stage")
        if checkpoint.get("training_contract") != contract:
            raise ValueError("resume checkpoint training contract differs; keep CLI overrides unchanged")
        if checkpoint.get("dataset_identity") != dataset_identity:
            raise ValueError("resume checkpoint dataset identity differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint.get("optimizer_step", 0))
        if start_step >= maximum:
            raise ValueError(f"resume checkpoint is already at step {start_step}, maximum is {maximum}")
        best_step = checkpoint.get("best_optimizer_step")
        best_metrics = checkpoint.get("best_metrics")
        if best_metrics:
            best_score = float(best_metrics["total_loss"])
        rng_state = checkpoint.get("rng_state") or {}
        if rng_state.get("torch") is not None:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        if rng_state.get("data_loader_generator") is not None:
            loader_generator.set_state(rng_state["data_loader_generator"])
        records = [
            row for row in records
            if int(row.get("optimizer_step", 0)) <= start_step
        ]
        atomic_write_text(
            metrics_path,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        )
    elif records:
        raise ValueError("metrics.jsonl exists without --resume-run; refusing to overwrite a partial run")

    def save_last(step: int) -> None:
        checkpoint = _new_checkpoint(
            model, optimizer, scheduler, {"model": model_cfg}, stage=stage_code,
            step=step, training_contract=contract, best_step=best_step,
            best_metrics=best_metrics, dataset_identity=dataset_identity,
            data_loader_generator=loader_generator,
        )
        atomic_torch_save(output_run / "checkpoints/last.pt", checkpoint)

    def save_best(step: int, metrics: dict[str, Any]) -> None:
        checkpoint = _new_checkpoint(
            model, optimizer, scheduler, {"model": model_cfg}, stage=stage_code,
            step=step, training_contract=contract, best_step=step,
            best_metrics=metrics, dataset_identity=dataset_identity,
            data_loader_generator=loader_generator,
        )
        atomic_torch_save(output_run / "checkpoints/best.pt", checkpoint)

    iterator = _infinite(loader)
    if start_step == 0:
        model.eval()
        initial_metrics = _evaluate_full_sequence(
            model,
            eval_loader,
            device,
            stage=stage_code,
            mask_seed=int(config.get("training_mask_seed", 20260920)),
        )
        initial_metrics["optimizer_step"] = 0
        initial_metrics["evaluation_scope"] = (
            "complete selected-window Stage-A posterior sequence"
            if stage_code == "A"
            else f"complete selected-window Stage-{stage_code} masked-condition sequence"
        )
        initial_row = {"phase": "evaluation", **initial_metrics}
        _append_jsonl(metrics_path, initial_row)
        records.append(initial_row)
        best_score = float(initial_metrics["total_loss"])
        best_step = 0
        best_metrics = initial_metrics
        save_best(0, initial_metrics)
        save_last(0)
        start_row = {
            "phase": "lifecycle", "event": "training_started",
            "optimizer_step": 0, "max_steps": maximum,
            "validation_interval": validation_interval,
            "checkpoint_interval": checkpoint_interval,
            "log_interval": log_interval,
        }
        _append_jsonl(metrics_path, start_row)
        records.append(start_row)
        _render_training_plot(output_run, records)
        atomic_write_json(output_run / "manifests/progress.json", {
            "status": "running", "optimizer_step": 0, "max_steps": maximum,
            "best_step": best_step, "best_metrics": best_metrics,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        print(
            f"[stage={stage_code}] training started: steps={maximum} "
            f"validation_interval={validation_interval} checkpoint_interval={checkpoint_interval} "
            f"log_interval={log_interval}", flush=True,
        )
    else:
        resume_row = {
            "phase": "lifecycle", "event": "training_resumed",
            "optimizer_step": start_step, "max_steps": maximum,
        }
        _append_jsonl(metrics_path, resume_row)
        records.append(resume_row)
        print(f"[stage={stage_code}] resumed from step={start_step}/{maximum}", flush=True)
    model.train()
    last_step = start_step
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for step in range(start_step + 1, maximum + 1):
            started = time.perf_counter()
            cpu_batch = next(iterator)
            batch = _device_batch(cpu_batch, device)
            if stage_code == "A":
                output = model(batch, stage="A")
            else:
                state_mask, action_mask, _ = make_physical_masks(
                    cpu_batch, int(config.get("training_mask_seed", 20260920))
                )
                output = model(batch, state_mask.to(device), action_mask.to(device), stage=stage_code)
            reconstruction = _standard_reconstruction_loss(output, batch)
            if output.posterior is not None:
                kl = hierarchical_kl(output.posterior)
            else:
                zero = reconstruction * 0.0
                kl = {"total": zero, "global": zero, "local": zero}
            target_beta = float(kl_beta_override if kl_beta_override is not None else training_cfg.get("kl", {}).get("beta", 1e-3))
            beta_warmup = max(1, int(training_cfg.get("kl", {}).get("beta_warmup_steps", 1)))
            beta = 0.0 if stage_code != "C" else target_beta * min(step / beta_warmup, 1.0)
            loss = reconstruction + beta * kl["total"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at optimizer step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            last_step = step
            row = {
                "phase": "train", "optimizer_step": step,
                "loss": float(loss.detach().cpu()),
                "reconstruction": float(reconstruction.detach().cpu()),
                "kl": float(kl["total"].detach().cpu()),
                "kl_beta": beta, "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm_before_clip": float(gradient_norm),
                "step_seconds": time.perf_counter() - started,
                "cuda_max_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            }
            _append_jsonl(metrics_path, row)
            records.append(row)
            if step % log_interval == 0 or step == 1:
                print(
                    f"[stage={stage_code}] step={step}/{maximum} loss={row['loss']:.6g} "
                    f"reconstruction={row['reconstruction']:.6g} lr={row['learning_rate']:.3g} "
                    f"grad={row['gradient_norm_before_clip']:.3g} step_s={row['step_seconds']:.2f}",
                    flush=True,
                )
            if step % validation_interval == 0 or step == maximum:
                model.eval()
                eval_metrics = _evaluate_full_sequence(
                    model,
                    eval_loader,
                    device,
                    stage=stage_code,
                    mask_seed=int(config.get("training_mask_seed", 20260920)),
                )
                eval_metrics["optimizer_step"] = step
                eval_metrics["evaluation_scope"] = (
                    "complete selected-window Stage-A posterior sequence"
                    if stage_code == "A"
                    else f"complete selected-window Stage-{stage_code} masked-condition sequence"
                )
                _append_jsonl(metrics_path, {"phase": "evaluation", **eval_metrics})
                records.append({"phase": "evaluation", **eval_metrics})
                if float(eval_metrics["total_loss"]) < best_score:
                    best_score = float(eval_metrics["total_loss"])
                    best_step = step
                    best_metrics = eval_metrics
                    save_best(step, eval_metrics)
                save_last(step)
                _render_training_plot(output_run, records)
                atomic_write_json(output_run / "manifests/progress.json", {
                    "status": "running", "optimizer_step": step, "max_steps": maximum,
                    "best_step": best_step, "best_metrics": best_metrics,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                print(
                    f"[stage={stage_code}] eval step={step} total={eval_metrics['total_loss']:.6g} "
                    f"state_rmse={eval_metrics['state_rmse']:.6g} action_rmse={eval_metrics['action_rmse']:.6g} "
                    f"p99={eval_metrics['continuous_abs_p99']:.6g} "
                    f"max_abs={eval_metrics['max_abs']:.6g} best_step={best_step}", flush=True,
                )
                model.train()
            elif step % checkpoint_interval == 0:
                save_last(step)
                _render_training_plot(output_run, records)
                atomic_write_json(output_run / "manifests/progress.json", {
                    "status": "running", "optimizer_step": step, "max_steps": maximum,
                    "best_step": best_step, "best_metrics": best_metrics,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
    except KeyboardInterrupt:
        save_last(last_step)
        _render_training_plot(output_run, records)
        atomic_write_json(output_run / "manifests/progress.json", {
            "status": "interrupted", "optimizer_step": last_step, "max_steps": maximum,
            "best_step": best_step, "best_metrics": best_metrics,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        atomic_write_text(output_run / "markers/cvae.interrupted", "INTERRUPTED\n")
        raise
    except Exception:
        save_last(last_step)
        _render_training_plot(output_run, records)
        atomic_write_json(output_run / "manifests/progress.json", {
            "status": "failed", "optimizer_step": last_step, "max_steps": maximum,
            "best_step": best_step, "best_metrics": best_metrics,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

    save_last(last_step)
    _render_training_plot(output_run, records)
    atomic_write_json(output_run / "manifests/progress.json", {
        "status": "completed", "optimizer_step": last_step, "max_steps": maximum,
        "best_step": best_step, "best_metrics": best_metrics,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    readback = validate_checkpoint(
        output_run / "checkpoints/last.pt",
        {"stage": stage_code, "parameters": parameter_total},
    )
    if not readback["passed"]:
        raise RuntimeError("65-token CVAE checkpoint readback failed")
    best_readback = validate_checkpoint(
        output_run / "checkpoints/best.pt",
        {"stage": stage_code, "parameters": parameter_total},
    )
    if not best_readback["passed"]:
        raise RuntimeError("65-token CVAE best checkpoint readback failed")
    summary = {
        "format_version": SUMMARY_FORMAT,
        "architecture_version": model.ARCHITECTURE_VERSION,
        "stage": stage_code,
        "stage_name": stage_name,
        "execution_pass": True,
        "quality_pass": None,
        "smoke": smoke,
        "completed_optimizer_steps": last_step,
        "model_contract": {
            **model_cfg,
            "actual_parameter_count": parameter_total,
            "decoder_memory_length": 82,
            "posterior_memory_length": 17,
            "posterior_input_shape": ["B", 65, 99],
            "condition_input_shape": ["B", 65, 198],
            "output_state_shape": ["B", 65, 70],
            "output_action_shape": ["B", 64, 29],
            "terminal_action_policy": "zero_not_predicted",
            "hard_chunk_ranges": [[4 * i, 4 * i + 3] for i in range(15)] + [[60, 64]],
        },
        "stage_parameter_counts": stage_counts,
        "stage_route": {
            "posterior_encoder_executed": stage_code in {"A", "C"},
            "posterior_latent_used": stage_code in {"A", "C"},
            "condition_encoder_executed": stage_code in {"B", "C"},
            "condition_latent_used": stage_code in {"B", "C"},
            "condition_memory_used": stage_code in {"B", "C"},
            "film_used": True,
            "kl_enabled": stage_code == "C",
        },
        "training_overrides": {
            "max_steps": max_steps_override,
            "micro_batch": micro_batch,
            "learning_rate": learning_rate,
            "lr_schedule": schedule_name,
            "warmup_steps": warmup_steps,
            "min_lr_ratio": min_lr_ratio,
        },
        "records": records,
        "best_step": best_step,
        "best_metrics": best_metrics,
        "validation_interval": validation_interval,
        "checkpoint_interval": checkpoint_interval,
        "log_interval": log_interval,
        "resume_run": str(resume_source) if resume_source is not None else None,
        "training_plot": str(output_run / "plots/training_curves.svg"),
        "checkpoint_readback": readback,
        "best_checkpoint_readback": best_readback,
        "source_checkpoint": (
            str(resume_source / "checkpoints/last.pt") if resume_source is not None
            else (str(init_run) if init_run is not None else "none; random initialization")
        ),
        "unique_next_step": "ENGINEERING_REVIEW_ONLY",
    }
    atomic_write_json(output_run / "manifests/standard_cvae_summary.json", summary)
    atomic_write_json(output_run / "manifests/model_signature.json", _model_signature(model_cfg, parameter_total))
    for stale_marker in ("cvae.interrupted", "cvae.failed"):
        (output_run / "markers" / stale_marker).unlink(missing_ok=True)
    marker = "cvae_posterior_standard_cvae_smoke.ok" if smoke else "cvae_posterior_standard_cvae_execution.ok"
    atomic_write_text(output_run / "markers" / marker, "PASS\n")
    dataset.close()
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the 65-token hierarchical standard CVAE")
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs/posterior_hierarchical_standard_cvae_h50.json")
    parser.add_argument("--stage", choices=("fixed", "random", "kl", "A", "B", "C"), required=True)
    parser.add_argument("--init-run", type=Path)
    parser.add_argument("--kl-beta", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine", "linear"))
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--min-lr-ratio", type=float)
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--resume-run", type=Path)
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
        max_steps_override=args.max_steps,
        micro_batch_override=args.micro_batch,
        learning_rate_override=args.learning_rate,
        lr_schedule=args.lr_schedule,
        warmup_steps_override=args.warmup_steps,
        min_lr_ratio_override=args.min_lr_ratio,
        validation_interval_override=args.validation_interval,
        checkpoint_interval_override=args.checkpoint_interval,
        log_interval_override=args.log_interval,
        resume_run=args.resume_run,
    )
    print("65-token hierarchical standard CVAE: PASS (engineering execution complete)")
    print(json.dumps({
        "output_run": str(args.output_run.expanduser().resolve()),
        "stage": summary["stage"],
        "smoke": args.smoke,
        "parameter_count": summary["model_contract"]["actual_parameter_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
