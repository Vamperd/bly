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
from .cvae_protocol import CHECKPOINT as V2_CHECKPOINT


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
    if checkpoint.get("format_version") not in {CHECKPOINT_FORMAT, V2_CHECKPOINT}:
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
    if checkpoint.get("format_version") not in {CHECKPOINT_FORMAT, V2_CHECKPOINT} or signature.get("architecture_version") != model.ARCHITECTURE_VERSION:
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


def run_experiment(*args, **kwargs):
    """Public entry now exclusively uses the versioned, recoverable v2 trainer."""
    from .cvae_training import run_experiment as run_v2
    return run_v2(*args, **kwargs)


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
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--continue-checkpoint", type=Path)
    parser.add_argument("--additional-steps", type=int)
    parser.add_argument("--allow-legacy-identity", action="store_true")
    parser.add_argument("--mask-mode", choices=("fixed", "dynamic"), default="fixed")
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--beta-warmup-steps", type=int)
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
        init_checkpoint=args.init_checkpoint,
        continue_checkpoint=args.continue_checkpoint,
        additional_steps=args.additional_steps,
        allow_legacy_identity=args.allow_legacy_identity,
        mask_mode=args.mask_mode,
        eval_samples=args.eval_samples,
        beta_warmup_steps=args.beta_warmup_steps,
    )
    if summary.get("interrupted"):
        print("Training interrupted at a safe boundary; last.pt saved", flush=True)
        return 130
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
