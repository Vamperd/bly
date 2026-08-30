from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .util import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_bytes,
    file_sha256,
    load_json,
)


TASKS = (
    "forward_rollout",
    "inverse",
    "history_action",
    "arbitrary_state",
    "arbitrary_action",
)
COMPACT_GATE_TASKS = {
    "forward_rollout", "arbitrary_state", "arbitrary_action",
}
MASK_TENSOR_FIELDS = (
    "state_input",
    "previous_input",
    "action_input",
    "state_loss",
    "previous_loss",
    "action_loss",
    "forward_transition",
    "inverse_transition",
    "history_action_transition",
    "rollout_start",
    "rollout_horizon",
)
TASK_LOSS_COMPONENT = {
    "forward_rollout": ("forward", "rollout"),
    "inverse": ("inverse",),
    "history_action": ("history_action",),
    "arbitrary_state": ("masked",),
    "arbitrary_action": ("masked",),
}


def parse_checkpoint_kinds(value: str) -> tuple[str, ...]:
    kinds = tuple(item.strip() for item in value.split(",") if item.strip())
    if not kinds:
        raise ValueError("at least one fixture checkpoint kind is required")
    unsupported = sorted(set(kinds) - {"best", "last"})
    if unsupported:
        raise ValueError(f"unsupported fixture checkpoint kinds: {unsupported}")
    if len(set(kinds)) != len(kinds):
        raise ValueError("fixture checkpoint kinds must be unique")
    return kinds


def _tensor_bytes(value: Any, index: int) -> bytes:
    tensor = value[index].detach().cpu().contiguous()
    prefix = canonical_json_bytes({
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    })
    return prefix + b"\0" + tensor.numpy().tobytes()


def sample_mask_sha256(
    batch: dict[str, Any], masks: Any, index: int,
) -> tuple[str, str]:
    references = list(batch.get("episode_ref", ()))
    starts = batch.get("window_start")
    if starts is None or index >= len(references):
        raise ValueError("fixture batch is missing episode_ref/window_start identity")
    start = int(starts[index].detach().cpu()) if hasattr(starts[index], "detach") \
        else int(starts[index])
    window_key = f"{references[index]}|{start}"
    digest = hashlib.sha256()
    digest.update(window_key.encode("utf-8"))
    digest.update(canonical_json_bytes({
        "task_name": masks.task_name,
        "completion_name": masks.completion_name,
        "causal": bool(masks.causal),
    }))
    for field in MASK_TENSOR_FIELDS:
        digest.update(field.encode("utf-8"))
        value = getattr(masks, field)
        digest.update(b"<none>" if value is None else _tensor_bytes(value, index))
    return window_key, digest.hexdigest()


def fixture_sha256(sample_hashes: dict[str, str]) -> str:
    if not sample_hashes:
        raise ValueError("cannot hash an empty exact fixture")
    return hashlib.sha256(canonical_json_bytes(sorted(sample_hashes.items()))).hexdigest()


def validate_checkpoint_fixture_identity(
    checkpoint_results: dict[str, Any], expected_window_count: int,
) -> str:
    if not checkpoint_results:
        raise ValueError("exact fixture diagnosis produced no checkpoint results")
    hashes = {
        str(value["exact_training_fixture"]["fixture_sha256"])
        for value in checkpoint_results.values()
    }
    counts = {
        int(value["exact_training_fixture"]["window_count"])
        for value in checkpoint_results.values()
    }
    if len(hashes) != 1:
        raise ValueError("best/last exact fixture hashes differ")
    if counts != {int(expected_window_count)}:
        raise ValueError(
            "exact fixture did not cover every fixed window: "
            f"observed={sorted(counts)}, expected={expected_window_count}"
        )
    return next(iter(hashes))


def mask_batches_equal(left: Any, right: Any) -> bool:
    import torch

    if (
        left.task_id != right.task_id
        or left.task_name != right.task_name
        or left.completion_name != right.completion_name
        or left.causal != right.causal
    ):
        return False
    for field in MASK_TENSOR_FIELDS:
        first, second = getattr(left, field), getattr(right, field)
        if (first is None) != (second is None):
            return False
        if first is not None and not torch.equal(first, second):
            return False
    return True


def _squared_error(
    prediction: Any, target: Any, mask: Any | None = None,
) -> tuple[float, int]:
    import torch

    difference = prediction.float() - target.float()
    if mask is not None:
        values = torch.square(difference).masked_select(mask)
    else:
        values = torch.square(difference).reshape(-1)
    if not values.numel():
        return 0.0, 0
    return float(values.sum().detach().cpu()), int(values.numel())


def sample_metric_contributions(
    task: str,
    output: Any,
    batch: dict[str, Any],
    masks: Any,
    index: int,
) -> dict[str, tuple[float, int]]:
    if task == "forward_rollout":
        transition = masks.forward_transition[index]
        target_delta = (
            batch["physical_state"][index, 1:]
            - batch["physical_state"][index, :-1]
        )
        transition_mask = transition[:, None].expand_as(target_delta)
        result = {
            "forward_one_normalized_rmse": _squared_error(
                output.forward_delta[index], target_delta, transition_mask
            )
        }
        horizon = min(8, int(masks.rollout_horizon[index].item()))
        start = int(masks.rollout_start[index].item())
        available = min(
            horizon,
            int(output.rollout_state.shape[1]) if output.rollout_state is not None else 0,
            int(batch["physical_state"].shape[1]) - start - 1,
        )
        result["forward_rollout_8_normalized_rmse"] = (
            _squared_error(
                output.rollout_state[index, :available],
                batch["physical_state"][index, start + 1 : start + available + 1],
            )
            if available > 0 else (0.0, 0)
        )
        return result
    if task == "inverse":
        mask = masks.inverse_transition[index, :, None].expand_as(
            output.inverse_action[index]
        )
        return {
            "action_inverse_local_normalized_rmse": _squared_error(
                output.inverse_action[index], batch["action"][index], mask
            )
        }
    if task == "history_action":
        mask = masks.history_action_transition[index, :, None].expand_as(
            output.history_action[index]
        )
        return {
            "history_action_normalized_rmse": _squared_error(
                output.history_action[index], batch["action"][index], mask
            )
        }
    if task == "arbitrary_state":
        return {
            "arbitrary_state_normalized_rmse": _squared_error(
                output.physical_state[index],
                batch["physical_state"][index],
                masks.state_loss[index],
            )
        }
    if task == "arbitrary_action":
        return {
            "action_completion_macro_normalized_rmse": _squared_error(
                output.action[index], batch["action"][index], masks.action_loss[index]
            )
        }
    raise ValueError(f"unsupported exact fixture task {task!r}")


def _metric_result(
    batch_rmses: dict[str, list[float]],
    window_rmses: dict[str, list[float]],
    squared_errors: dict[str, float],
    element_counts: dict[str, int],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    primary: dict[str, float] = {}
    details: dict[str, dict[str, Any]] = {}
    for name in sorted(set(batch_rmses) | set(window_rmses)):
        batches = batch_rmses.get(name, [])
        windows = window_rmses.get(name, [])
        count = int(element_counts.get(name, 0))
        value = float(np.mean(batches)) if batches else math.inf
        primary[name] = value
        details[name] = {
            "normalized_rmse": value,
            "global_normalized_rmse": (
                math.sqrt(squared_errors[name] / count) if count else math.inf
            ),
            "mean_window_normalized_rmse": (
                float(np.mean(windows)) if windows else math.inf
            ),
            "batch_count": len(batches),
            "window_count": len(windows),
            "element_count": count,
        }
    return primary, details


def _model_dimensions_match_dataset(config: dict[str, Any], dataset: Any) -> None:
    expected = {
        "state_dim": dataset.state_dim,
        "include_previous_action": dataset.include_previous_action,
        "robot_info_dim": dataset.robot_info_dim,
        "dynamics_context_dim": dataset.dynamics_context_dim,
        "auxiliary_dim": dataset.auxiliary_dim,
        "joint_robot_info_dim": dataset.joint_robot_info_dim,
        "global_robot_info_dim": dataset.global_robot_info_dim,
        "actuator_type_count": len(dataset.actuator_type_to_id),
        "reference_available": dataset.reference_available,
    }
    mismatches = {
        name: {"checkpoint": config["model"].get(name), "dataset": value}
        for name, value in expected.items()
        if config["model"].get(name) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint model/dataset dimensions differ: {mismatches}")


def _load_checkpoint(
    path: Path,
    dataset_hash: str,
    task: str,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != "sonic_state_action_cvae_checkpoint_v2":
        raise ValueError(f"fixture diagnosis requires checkpoint v2: {path}")
    if checkpoint.get("dataset_manifest_sha256") != dataset_hash:
        raise ValueError(f"checkpoint dataset manifest hash differs: {path}")
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("training"), dict):
        raise ValueError(f"checkpoint is missing its training config: {path}")
    training = config["training"]
    if training.get("task_mode") != task:
        raise ValueError(f"checkpoint task differs from training summary: {path}")
    if training.get("fixed_training_masks") is not True:
        raise ValueError(f"checkpoint did not use fixed training Masks: {path}")
    if config.get("overfit_phase") != "capacity":
        raise ValueError(f"checkpoint is not from the capacity phase: {path}")
    if int(config["data"].get("window_transitions", -1)) != 128:
        raise ValueError(f"checkpoint did not use 128-transition windows: {path}")
    if bool(config["data"].get("random_crop", True)):
        raise ValueError(f"checkpoint used random crop: {path}")
    if bool(config["data"].get("shuffle_train", True)):
        raise ValueError(f"checkpoint shuffled fixed training windows: {path}")
    if config["masking"].get("action_step_curriculum"):
        raise ValueError(f"fixed fixture checkpoint has an Action curriculum: {path}")
    return checkpoint


def _training_run_specs(
    run_paths: Iterable[Path],
    checkpoint_kinds: tuple[str, ...],
    dataset_hash: str,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    observed_tasks: set[str] = set()
    seeds: set[int] = set()
    for raw_run in run_paths:
        run = raw_run.expanduser().resolve()
        summary_path = run / "manifests/training_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = load_json(summary_path)
        task = str(summary.get("task_mode"))
        if task not in TASKS or task in observed_tasks:
            raise ValueError(f"invalid or duplicate single-task run: {summary_path}")
        observed_tasks.add(task)
        if summary.get("memorization_benchmark") is not True:
            raise ValueError(f"run is not a memorization benchmark: {summary_path}")
        if summary.get("overfit_phase") != "capacity":
            raise ValueError(f"run is not a capacity experiment: {summary_path}")
        if summary.get("fixed_training_masks") is not True:
            raise ValueError(f"run did not use fixed training Masks: {summary_path}")
        if not str(summary.get("model_profile", "")).startswith("compact_single_"):
            raise ValueError(f"fixture diagnosis currently requires compact runs: {summary_path}")
        if int(summary.get("optimizer_steps", -1)) != 20_000:
            raise ValueError(f"single-task run did not complete 20k steps: {summary_path}")
        seeds.add(int(summary.get("seed", -1)))
        checkpoints: dict[str, dict[str, Any]] = {}
        config_hashes: set[str] = set()
        for kind in checkpoint_kinds:
            checkpoint_path = run / f"checkpoints/{kind}.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            checkpoint = _load_checkpoint(checkpoint_path, dataset_hash, task)
            config_hash = hashlib.sha256(
                canonical_json_bytes(checkpoint["config"])
            ).hexdigest()
            config_hashes.add(config_hash)
            checkpoints[kind] = {
                "path": checkpoint_path,
                "config": copy.deepcopy(checkpoint["config"]),
                "config_sha256": config_hash,
            }
        if len(config_hashes) != 1:
            raise ValueError(f"best/last checkpoint configs differ: {run}")
        specs.append({
            "run": run,
            "summary": summary,
            "task": task,
            "checkpoints": checkpoints,
        })
    missing = sorted(set(TASKS) - observed_tasks)
    extra = sorted(observed_tasks - set(TASKS))
    if missing or extra or len(specs) != len(TASKS):
        raise ValueError(f"fixture diagnosis requires exactly five tasks; missing={missing}")
    if len(seeds) != 1:
        raise ValueError(f"single-task runs use different seeds: {sorted(seeds)}")
    return sorted(specs, key=lambda item: TASKS.index(item["task"]))


def _evaluate_exact_checkpoint(
    model: Any,
    loader: Any,
    masker: Any,
    device: Any,
    config: dict[str, Any],
    task: str,
    checkpoint_step: int,
    checkpoint_kind: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    from .losses import compute_loss
    from .trainer import (
        _autocast,
        _device_batch,
        generate_training_masks,
        overfit_gate,
        overfit_task_thresholds,
    )

    training = config["training"]
    _, thresholds = overfit_task_thresholds(training)
    batch_rmses: dict[str, list[float]] = {}
    window_rmses: dict[str, list[float]] = {}
    squared_errors: dict[str, float] = {}
    element_counts: dict[str, int] = {}
    loss_sums: dict[str, float] = {}
    loss_batches = 0
    sample_hashes: dict[str, str] = {}
    window_records: list[dict[str, Any]] = []
    model.eval()
    for cpu_batch in loader:
        batch = _device_batch(cpu_batch, device)
        masker.set_step(1)
        first_step_masks = generate_training_masks(masker, batch, training)
        masker.set_step(checkpoint_step)
        masks = generate_training_masks(masker, batch, training)
        if not mask_batches_equal(first_step_masks, masks):
            raise ValueError(
                f"fixed fixture Mask changes between step 1 and {checkpoint_step}"
            )
        with _autocast(device, str(training["amp"])):
            output = model(
                batch, masks, sample_from_prior=False, deterministic=True
            )
            loss = compute_loss(output, batch, masks, training, kl_beta=0.0)
        for name, value in loss.detached().items():
            loss_sums[name] = loss_sums.get(name, 0.0) + value
        loss_batches += 1
        batch_contributions: dict[str, list[tuple[float, int]]] = {}
        for index in range(int(batch["physical_state"].shape[0])):
            window_key, mask_hash = sample_mask_sha256(batch, masks, index)
            if window_key in sample_hashes:
                raise ValueError(f"duplicate exact fixture window {window_key}")
            sample_hashes[window_key] = mask_hash
            contributions = sample_metric_contributions(
                task, output, batch, masks, index
            )
            record_metrics: dict[str, dict[str, Any]] = {}
            for name, (squared, count) in contributions.items():
                if count <= 0:
                    raise ValueError(f"exact fixture metric {name} has no targets")
                value = math.sqrt(squared / count)
                batch_contributions.setdefault(name, []).append((squared, count))
                window_rmses.setdefault(name, []).append(value)
                squared_errors[name] = squared_errors.get(name, 0.0) + squared
                element_counts[name] = element_counts.get(name, 0) + count
                record_metrics[name] = {
                    "normalized_rmse": value,
                    "element_count": count,
                }
            window_records.append({
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_step": checkpoint_step,
                "task_mode": task,
                "window_key": window_key,
                "mask_sha256": mask_hash,
                "completion_name": masks.completion_name,
                "metrics": record_metrics,
            })
        for name, contributions in batch_contributions.items():
            squared = sum(item[0] for item in contributions)
            count = sum(item[1] for item in contributions)
            batch_rmses.setdefault(name, []).append(math.sqrt(squared / count))
    metrics, metric_details = _metric_result(
        batch_rmses, window_rmses, squared_errors, element_counts
    )
    gate = overfit_gate(metrics, thresholds, require_complete=False)
    mean_losses = {
        name: value / max(loss_batches, 1) for name, value in loss_sums.items()
    }
    relevant_loss = max(
        (mean_losses.get(name, 0.0) for name in TASK_LOSS_COMPONENT[task]),
        default=0.0,
    )
    rmse_proxy = math.sqrt(max(2.0 * relevant_loss, 0.0))
    maximum_rmse = max(metrics.values(), default=math.inf)
    result = {
        "latent_mode": "posterior_mean",
        "mask_distribution": (
            "per-window deterministic mixed granularity used during training"
            if task.startswith("arbitrary_")
            else "per-window deterministic task fixture used during training"
        ),
        "metric_contract_note": (
            "action_completion_macro_normalized_rmse retains the legacy threshold key, "
            "but here aggregates the exact per-window mixed training Masks rather than "
            "averaging four newly generated granularity suites"
            if task == "arbitrary_action" else None
        ),
        "fixture_sha256": fixture_sha256(sample_hashes),
        "window_count": len(sample_hashes),
        "metrics": metrics,
        "metric_details": metric_details,
        "mean_training_loss": mean_losses,
        "task_loss_rmse_proxy": rmse_proxy,
        "max_metric_to_loss_proxy_ratio": (
            maximum_rmse / max(rmse_proxy, 1e-12)
        ),
        "objective_metric_gap_flag": bool(
            not gate["overfit_pass"]
            and math.isfinite(maximum_rmse)
            and maximum_rmse > 2.0 * max(rmse_proxy, 1e-12)
        ),
        "exact_pass": gate["overfit_pass"],
        "exact_score": gate["overfit_score"],
        "thresholds": gate["overfit_thresholds"],
        "ratios": gate["overfit_ratios"],
    }
    return result, window_records


def _evaluate_unseen_checkpoint(
    model: Any,
    loader: Any,
    masker: Any,
    device: Any,
    config: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    import torch

    from .trainer import overfit_gate, validate_overfit_suite

    training = config["training"]
    gate_mode = str(training["overfit_gate_latent_mode"])
    diagnostic_modes = tuple(training["overfit_diagnostic_latent_modes"])
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(config["seed"]) + 90_001)
        metrics = validate_overfit_suite(
            model,
            loader,
            masker,
            device,
            str(training["amp"]),
            int(training["validation_max_batches"]),
            gate_mode,
            diagnostic_modes,
        )
    gate = overfit_gate(metrics, thresholds, require_complete=False)
    return {
        "protocol": "existing fixed-seed unseen Mask suite; not the training fixture",
        "gate_latent_mode": gate_mode,
        "metrics": {name: metrics[name] for name in thresholds},
        "unseen_pass": gate["overfit_pass"],
        "unseen_score": gate["overfit_score"],
        "thresholds": gate["overfit_thresholds"],
        "ratios": gate["overfit_ratios"],
        "latent_diagnostics": metrics.get("latent_diagnostics", {}),
    }


def _evaluate_run(
    dataset_run: Path,
    spec: dict[str, Any],
    checkpoint_kinds: tuple[str, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from torch.utils.data import DataLoader

    from .dataset import StateActionWindowDataset, worker_seed
    from .masking import MaskGenerator
    from .models import build_model, parameter_count
    from .trainer import overfit_task_thresholds

    task = spec["task"]
    summary = spec["summary"]
    config = copy.deepcopy(spec["checkpoints"][checkpoint_kinds[0]]["config"])
    dataset = StateActionWindowDataset(
        dataset_run,
        "train",
        int(config["data"]["window_transitions"]),
        int(config["data"]["validation_stride"]),
        config["data"].get("max_train_episodes"),
        random_crop=False,
    )
    try:
        _model_dimensions_match_dataset(config, dataset)
        expected_windows = int(summary.get("fixed_window_count", -1))
        if len(dataset) != expected_windows:
            raise ValueError(
                f"fixture window count differs for {task}: {len(dataset)} != {expected_windows}"
            )
        loader = DataLoader(
            dataset,
            batch_size=int(config["training"]["micro_batch"]),
            shuffle=False,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=int(config["data"]["num_workers"]) > 0,
            worker_init_fn=worker_seed,
            drop_last=False,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, thresholds = overfit_task_thresholds(config["training"])
        checkpoint_results: dict[str, Any] = {}
        all_window_records: list[dict[str, Any]] = []
        for kind in checkpoint_kinds:
            checkpoint_path = spec["checkpoints"][kind]["path"]
            print(
                f"[exact-fixture] task={task} checkpoint={kind} stage=load",
                flush=True,
            )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            config_hash = hashlib.sha256(
                canonical_json_bytes(checkpoint["config"])
            ).hexdigest()
            if config_hash != spec["checkpoints"][kind]["config_sha256"]:
                raise ValueError(f"checkpoint config changed during diagnosis: {checkpoint_path}")
            model = build_model(copy.deepcopy(checkpoint["config"]["model"]))
            model.load_state_dict(checkpoint["model"], strict=True)
            checkpoint_step = int(checkpoint["step"])
            checkpoint_parameter_count = int(checkpoint["parameter_count"])
            if parameter_count(model) != checkpoint_parameter_count:
                raise ValueError(f"checkpoint parameter count differs: {checkpoint_path}")
            del checkpoint
            model.to(device).eval()
            exact_masker = MaskGenerator(copy.deepcopy(config["masking"]))
            print(
                f"[exact-fixture] task={task} checkpoint={kind} stage=exact",
                flush=True,
            )
            exact, records = _evaluate_exact_checkpoint(
                model,
                loader,
                exact_masker,
                device,
                config,
                task,
                checkpoint_step,
                kind,
            )
            checkpoint_hash = file_sha256(checkpoint_path)
            for record in records:
                record["training_run"] = str(spec["run"])
                record["checkpoint_sha256"] = checkpoint_hash
            all_window_records.extend(records)
            print(
                f"[exact-fixture] task={task} checkpoint={kind} stage=unseen",
                flush=True,
            )
            unseen = _evaluate_unseen_checkpoint(
                model,
                loader,
                MaskGenerator(copy.deepcopy(config["masking"])),
                device,
                config,
                thresholds,
            )
            checkpoint_results[kind] = {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "step": checkpoint_step,
                "exact_training_fixture": exact,
                "unseen_mask_diagnostic": unseen,
            }
            print(
                f"[exact-fixture] task={task} checkpoint={kind} "
                f"exact_score={exact['exact_score']:.6f} "
                f"unseen_score={unseen['unseen_score']:.6f}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        fixture_hash = validate_checkpoint_fixture_identity(
            checkpoint_results, len(dataset)
        )
        comparison: dict[str, Any] = {}
        if {"best", "last"}.issubset(checkpoint_results):
            best_score = checkpoint_results["best"]["exact_training_fixture"]["exact_score"]
            last_score = checkpoint_results["last"]["exact_training_fixture"]["exact_score"]
            comparison = {
                "last_exact_score_minus_best": last_score - best_score,
                "last_is_better_for_exact_fixture": bool(last_score < best_score),
                "checkpoint_selection_missed_better_exact_memory": bool(
                    last_score + 1e-12 < best_score
                ),
            }
        return ({
            "training_run": str(spec["run"]),
            "task_mode": task,
            "seed": int(summary["seed"]),
            "model_profile": summary["model_profile"],
            "parameter_count": int(summary["parameter_count"]),
            "fixed_window_count": len(dataset),
            "fixture_sha256": fixture_hash,
            "diagnostic_only": task in {"inverse", "history_action"},
            "checkpoints": checkpoint_results,
            "comparison": comparison,
        }, all_window_records)
    finally:
        dataset.close()


def classify_fixture_result(exact_pass: bool, unseen_pass: bool) -> str:
    if exact_pass and not unseen_pass:
        return "exact_memory_pass_unseen_mask_fail"
    if not exact_pass and not unseen_pass:
        return "exact_and_unseen_both_fail"
    if exact_pass and unseen_pass:
        return "exact_and_unseen_both_pass"
    return "unseen_pass_exact_fail_investigate_metric_contract"


def _interpretations(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for run in runs:
        for kind, checkpoint in run["checkpoints"].items():
            exact = checkpoint["exact_training_fixture"]
            unseen = checkpoint["unseen_mask_diagnostic"]
            values.append({
                "task_mode": run["task_mode"],
                "checkpoint_kind": kind,
                "classification": classify_fixture_result(
                    bool(exact["exact_pass"]), bool(unseen["unseen_pass"])
                ),
                "objective_metric_gap_flag": exact["objective_metric_gap_flag"],
            })
    return values


def _svg_axes(title: str, width: int = 1180, height: int = 680) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#0f172a}</style>',
        f'<text x="24" y="34" font-size="21" font-weight="700">{html.escape(title)}</text>',
    ]


def write_exact_vs_unseen_svg(result: dict[str, Any], path: Path) -> None:
    width, height = 1180, 680
    lines = _svg_axes("Exact training fixture vs unseen Mask", width, height)
    left, right, top, bottom = 100, 1140, 80, 585
    all_scores = []
    for run in result["runs"]:
        for checkpoint in run["checkpoints"].values():
            all_scores.extend((
                float(checkpoint["exact_training_fixture"]["exact_score"]),
                float(checkpoint["unseen_mask_diagnostic"]["unseen_score"]),
            ))
    maximum = max(1.25, min(max(all_scores, default=1.0) * 1.1, 20.0))
    y = lambda value: bottom - min(value, maximum) / maximum * (bottom - top)
    lines.extend((
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#64748b"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#64748b"/>',
        f'<line x1="{left}" y1="{y(1.0):.2f}" x2="{right}" y2="{y(1.0):.2f}" stroke="#dc2626" stroke-dasharray="6 4"/>',
        f'<text x="{right - 10}" y="{y(1.0) - 6:.2f}" text-anchor="end" font-size="11">threshold ratio = 1</text>',
    ))
    tasks = [run["task_mode"] for run in result["runs"]]
    group_width = (right - left) / max(len(tasks), 1)
    colors = {"exact": "#2563eb", "unseen": "#f97316"}
    kinds = list(result["checkpoint_kinds"])
    for task_index, run in enumerate(result["runs"]):
        center = left + group_width * (task_index + 0.5)
        offsets = np.linspace(-30, 30, max(2, len(kinds)))[:len(kinds)]
        for offset, kind in zip(offsets, kinds, strict=True):
            checkpoint = run["checkpoints"][kind]
            exact = float(checkpoint["exact_training_fixture"]["exact_score"])
            unseen = float(checkpoint["unseen_mask_diagnostic"]["unseen_score"])
            for delta, name, score in ((-7, "exact", exact), (7, "unseen", unseen)):
                x = center + float(offset) + delta
                lines.append(
                    f'<circle cx="{x:.2f}" cy="{y(score):.2f}" r="5" fill="{colors[name]}"/>'
                )
            lines.append(
                f'<text x="{center + float(offset):.2f}" y="{bottom + 18}" text-anchor="middle" font-size="10">{html.escape(kind)}</text>'
            )
        lines.append(
            f'<text x="{center:.2f}" y="{bottom + 40}" text-anchor="middle" font-size="11">{html.escape(run["task_mode"])}</text>'
        )
    lines.extend((
        '<circle cx="880" cy="48" r="5" fill="#2563eb"/><text x="892" y="52" font-size="11">exact training fixture</text>',
        '<circle cx="1020" cy="48" r="5" fill="#f97316"/><text x="1032" y="52" font-size="11">unseen Mask</text>',
        '<text x="24" y="335" font-size="13" transform="rotate(-90 24 335)">worst threshold ratio (lower is better)</text>',
        '</svg>',
    ))
    atomic_write_text(path, "".join(lines))


def write_best_vs_last_svg(result: dict[str, Any], path: Path) -> None:
    width, height = 1180, 680
    lines = _svg_axes("Exact fixture: best.pt vs last.pt", width, height)
    left, right, top, bottom = 100, 1140, 80, 585
    available = [
        kind for kind in ("best", "last") if kind in result["checkpoint_kinds"]
    ]
    scores = [
        float(run["checkpoints"][kind]["exact_training_fixture"]["exact_score"])
        for run in result["runs"] for kind in available
    ]
    maximum = max(1.25, min(max(scores, default=1.0) * 1.1, 20.0))
    y = lambda value: bottom - min(value, maximum) / maximum * (bottom - top)
    lines.extend((
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#64748b"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#64748b"/>',
        f'<line x1="{left}" y1="{y(1.0):.2f}" x2="{right}" y2="{y(1.0):.2f}" stroke="#dc2626" stroke-dasharray="6 4"/>',
    ))
    colors = {"best": "#2563eb", "last": "#059669"}
    group_width = (right - left) / max(len(result["runs"]), 1)
    for index, run in enumerate(result["runs"]):
        center = left + group_width * (index + 0.5)
        points = []
        for offset, kind in zip((-16, 16), available, strict=False):
            score = float(run["checkpoints"][kind]["exact_training_fixture"]["exact_score"])
            x = center + offset
            points.append((x, y(score)))
            lines.append(
                f'<circle cx="{x:.2f}" cy="{y(score):.2f}" r="6" fill="{colors[kind]}"/>'
            )
        if len(points) == 2:
            lines.append(
                f'<line x1="{points[0][0]:.2f}" y1="{points[0][1]:.2f}" x2="{points[1][0]:.2f}" y2="{points[1][1]:.2f}" stroke="#94a3b8"/>'
            )
        lines.append(
            f'<text x="{center:.2f}" y="{bottom + 30}" text-anchor="middle" font-size="11">{html.escape(run["task_mode"])}</text>'
        )
    lines.extend((
        '<circle cx="930" cy="48" r="5" fill="#2563eb"/><text x="942" y="52" font-size="11">best.pt</text>',
        '<circle cx="1030" cy="48" r="5" fill="#059669"/><text x="1042" y="52" font-size="11">last.pt</text>',
        '<text x="24" y="335" font-size="13" transform="rotate(-90 24 335)">exact worst threshold ratio</text>',
        '</svg>',
    ))
    atomic_write_text(path, "".join(lines))


def _write_report(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Exact Training Fixture 只读诊断报告",
        "",
        "> 该报告只判断训练样本与训练 Mask 的记忆，不代表泛化或部署性能。",
        "",
        f"- 执行完整：是",
        f"- 当前 overfit 门禁被修改：否",
        f"- best 正式任务全部 exact 通过：{result['quality_summary'].get('best_all_compact_gate_tasks_exact_pass')}",
        f"- last 正式任务全部 exact 通过：{result['quality_summary'].get('last_all_compact_gate_tasks_exact_pass')}",
        "",
        "| 任务 | checkpoint | exact ratio | unseen ratio | exact | unseen | 解释 |",
        "|---|---|---:|---:|---|---|---|",
    ]
    classifications = {
        (item["task_mode"], item["checkpoint_kind"]): item
        for item in result["interpretations"]
    }
    for run in result["runs"]:
        for kind, checkpoint in run["checkpoints"].items():
            exact = checkpoint["exact_training_fixture"]
            unseen = checkpoint["unseen_mask_diagnostic"]
            classification = classifications[(run["task_mode"], kind)]["classification"]
            lines.append(
                f"| {run['task_mode']} | {kind} | {exact['exact_score']:.4f} | "
                f"{unseen['unseen_score']:.4f} | "
                f"{'PASS' if exact['exact_pass'] else 'FAIL'} | "
                f"{'PASS' if unseen['unseen_pass'] else 'FAIL'} | {classification} |"
            )
    lines.extend((
        "",
        "## 解释规则",
        "",
        "- exact 通过而 unseen 失败：模型已记住训练 fixture，旧门禁测量的是未训练 Mask 组合迁移。",
        "- exact 与 unseen 都失败：继续检查损失、输出头、指标合同或真实容量。",
        "- last 优于 best：旧 unseen-Mask checkpoint 选择没有保留最佳训练记忆点。",
        "- objective_metric_gap_flag 为真：训练目标的 RMSE proxy 与 exact 指标存在超过2倍的启发式差距。",
        "- arbitrary Action 的 exact 指标沿用旧阈值字段名，但统计的是实际 mixed 训练 Mask，不是四类 unseen Mask 的 macro。",
    ))
    atomic_write_text(path, "\n".join(lines) + "\n")


def write_fixture_artifacts(
    result: dict[str, Any],
    output_run: Path,
    window_records: list[dict[str, Any]],
) -> None:
    atomic_write_json(output_run / "manifests/fixture_diagnostic.json", result)
    atomic_write_text(
        output_run / "data/exact_fixture_window_metrics.jsonl",
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in window_records
        ),
    )
    _write_report(result, output_run / "manifests/fixture_report_zh.md")
    write_exact_vs_unseen_svg(result, output_run / "videos/exact_vs_unseen.svg")
    write_best_vs_last_svg(result, output_run / "videos/best_vs_last.svg")
    atomic_write_text(
        output_run / "markers/cvae_overfit_fixture_diagnostic.ok", "PASS\n"
    )


def diagnose_overfit_fixture(
    dataset_run: Path,
    training_runs: Iterable[Path],
    output_run: Path,
    checkpoint_kinds: tuple[str, ...] = ("best", "last"),
) -> dict[str, Any]:
    dataset = dataset_run.expanduser().resolve()
    output = output_run.expanduser().resolve()
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "videos"):
        (output / child).mkdir(parents=True, exist_ok=True)
    marker = dataset / "markers/cvae_overfit_subset.ok"
    if not marker.is_file():
        raise FileNotFoundError(marker)
    manifest_path = dataset / "manifests/dataset_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("purpose") != "physics_state_action_32_motion_memorization":
        raise ValueError("fixture diagnosis requires the 32-motion memorization dataset")
    if (
        int(manifest.get("motion_count", -1)) != 32
        or int(manifest.get("canonical_episode_count", -1)) != 256
    ):
        raise ValueError("fixture diagnosis requires exactly 32 motions and 256 episodes")
    dataset_hash = file_sha256(manifest_path)
    kinds = tuple(checkpoint_kinds)
    if not kinds or any(kind not in {"best", "last"} for kind in kinds):
        raise ValueError("checkpoint kinds must contain only best and/or last")
    specs = _training_run_specs(training_runs, kinds, dataset_hash)
    runs: list[dict[str, Any]] = []
    window_records: list[dict[str, Any]] = []
    for spec in specs:
        print(f"[exact-fixture] task={spec['task']} stage=start", flush=True)
        run_result, records = _evaluate_run(dataset, spec, kinds)
        runs.append(run_result)
        window_records.extend(records)
        print(f"[exact-fixture] task={spec['task']} stage=complete", flush=True)
    quality_by_kind: dict[str, bool] = {}
    for kind in kinds:
        quality_by_kind[kind] = all(
            run["checkpoints"][kind]["exact_training_fixture"]["exact_pass"]
            for run in runs if run["task_mode"] in COMPACT_GATE_TASKS
        )
    result = {
        "format_version": "sonic_overfit_exact_fixture_diagnostic_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution_pass": True,
        "read_only_diagnostic": True,
        "source_read_only": True,
        "checkpoint_files_read_only": True,
        "changes_training_or_checkpoint_selection": False,
        "changes_existing_overfit_markers": False,
        "memorization_benchmark": True,
        "generalization_claim_allowed": False,
        "dataset_run": str(dataset),
        "dataset_manifest_sha256": dataset_hash,
        "checkpoint_kinds": list(kinds),
        "formal_compact_gate_tasks": sorted(COMPACT_GATE_TASKS),
        "inverse_history_without_reference_are_diagnostic_only": True,
        "quality_summary": {
            f"{kind}_all_compact_gate_tasks_exact_pass": value
            for kind, value in quality_by_kind.items()
        },
        "runs": runs,
    }
    result["interpretations"] = _interpretations(runs)
    write_fixture_artifacts(result, output, window_records)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare exact training fixtures with unseen overfit Masks"
    )
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--training-run", action="append", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--checkpoint-kinds", default="best,last")
    args = parser.parse_args()
    result = diagnose_overfit_fixture(
        args.dataset_run,
        args.training_run,
        args.output_run,
        parse_checkpoint_kinds(args.checkpoint_kinds),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
