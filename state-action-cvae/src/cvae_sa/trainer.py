from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import StateActionWindowDataset, worker_seed
from .losses import compute_loss
from .masking import MaskGenerator
from .models import build_model, parameter_count
from .overfit_diagnostics import evaluate_input_sensitivity
from .overfit_visualization import write_training_svg
from .util import (
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    load_config,
    seed_everything,
)


DEFAULT_OVERFIT_THRESHOLDS = {
    "forward_one_normalized_rmse": 0.05,
    "action_inverse_local_normalized_rmse": 0.05,
    "arbitrary_state_normalized_rmse": 0.05,
    "action_completion_macro_normalized_rmse": 0.05,
    "history_action_normalized_rmse": 0.08,
    "forward_rollout_8_normalized_rmse": 0.10,
}


def warmup_cosine_factor(
    step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.0
) -> float:
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if step < warmup_steps:
        return max(step, 1) / max(warmup_steps, 1)
    progress = min(max((step - warmup_steps) / max(max_steps - warmup_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def overfit_gate(
    metrics: dict[str, Any], thresholds: dict[str, float] | None = None
) -> dict[str, Any]:
    limits = dict(DEFAULT_OVERFIT_THRESHOLDS if thresholds is None else thresholds)
    missing = sorted(set(DEFAULT_OVERFIT_THRESHOLDS) - set(limits))
    if missing:
        raise ValueError(f"overfit thresholds are missing metrics: {missing}")
    ratios = {
        name: float(metrics.get(name, math.inf)) / max(float(limit), 1e-12)
        for name, limit in limits.items()
    }
    score = max(ratios.values(), default=math.inf)
    return {
        "overfit_pass": bool(math.isfinite(score) and score <= 1.0),
        "overfit_score": float(score),
        "overfit_thresholds": limits,
        "overfit_ratios": ratios,
    }


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _infinite(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def _autocast(device: torch.device, amp: str):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _masked_squared_error(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor | None:
    if not bool(mask.any()):
        return None
    return torch.square(prediction - target).masked_select(mask).mean()


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    masker: MaskGenerator,
    device: torch.device,
    amp: str,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    if getattr(model, "__class__", type(model)).__name__ == "PhysicsTransformerCVAE":
        result = _validate_physics(model, loader, masker, device, amp, max_batches)
        model.train()
        return result
    tasks = (
        ("forward", None),
        ("inverse", None),
        ("completion", "element"),
        ("completion", "step"),
        ("completion", "feature"),
    )
    task_errors: dict[str, list[float]] = {f"{a}:{b or 'none'}": [] for a, b in tasks}
    forward_errors: list[float] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        task, completion = tasks[batch_index % len(tasks)]
        batch = _device_batch(batch, device)
        masks = masker.generate(batch, force_task=task, force_completion=completion)
        with _autocast(device, amp):
            output = model(batch, masks, sample_from_prior=True, deterministic=True)
        group_errors = []
        for prediction, target, mask in (
            (output.physical_state, batch["physical_state"], masks.state_loss),
            (output.previous_action, batch["previous_action"], masks.previous_loss),
            (output.action, batch["action"], masks.action_loss),
        ):
            error = _masked_squared_error(prediction.float(), target.float(), mask)
            if error is not None:
                group_errors.append(error)
        if group_errors:
            task_errors[f"{task}:{completion or 'none'}"].append(
                float(torch.sqrt(torch.stack(group_errors).mean()).cpu())
            )
        if task == "forward":
            target_delta = batch["physical_state"][:, 1:] - batch["physical_state"][:, :-1]
            valid = batch["valid_action"][:, :, None].expand_as(target_delta)
            error = _masked_squared_error(output.forward_delta.float(), target_delta.float(), valid)
            if error is not None:
                forward_errors.append(float(torch.sqrt(error).cpu()))
    observed = [np.mean(values) for values in task_errors.values() if values]
    macro = float(np.mean(observed)) if observed else math.inf
    forward_rmse = float(np.mean(forward_errors)) if forward_errors else math.inf
    result = {
        "macro_masked_normalized_rmse": macro,
        "forward_normalized_rmse": forward_rmse,
        "selection_score": macro + forward_rmse,
    }
    result.update(
        {
            f"rmse/{name}": float(np.mean(values)) if values else math.nan
            for name, values in task_errors.items()
        }
    )
    model.train()
    return result


@torch.no_grad()
def _validate_physics(
    model: torch.nn.Module,
    loader: DataLoader,
    masker: MaskGenerator,
    device: torch.device,
    amp: str,
    max_batches: int,
) -> dict[str, float]:
    tasks = ("forward_one", "forward_rollout", "inverse", "history_action", "arbitrary")
    values: dict[str, list[float]] = {name: [] for name in tasks}
    masker.set_step(max(masker.optimizer_step, masker.rollout_start_step))
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        task = tasks[batch_index % len(tasks)]
        batch = _device_batch(cpu_batch, device)
        masks = masker.generate(batch, force_task=task)
        with _autocast(device, amp):
            output = model(batch, masks, sample_from_prior=True, deterministic=True)
        if task.startswith("forward"):
            if task == "forward_rollout" and output.rollout_state is not None:
                errors = []
                for index in range(output.rollout_state.shape[0]):
                    horizon = int(masks.rollout_horizon[index].item())
                    start = int(masks.rollout_start[index].item())
                    available = min(horizon, output.rollout_state.shape[1], batch["physical_state"].shape[1] - start - 1)
                    if available > 0:
                        errors.append(torch.square(
                            output.rollout_state[index, :available] -
                            batch["physical_state"][index, start + 1 : start + available + 1]
                        ).mean())
                if errors:
                    values[task].append(float(torch.sqrt(torch.stack(errors).mean()).cpu()))
            else:
                target = batch["physical_state"][:, 1:] - batch["physical_state"][:, :-1]
                mask = masks.forward_transition[:, :, None].expand_as(target)
                error = _masked_squared_error(output.forward_delta.float(), target.float(), mask)
                if error is not None:
                    values[task].append(float(torch.sqrt(error).cpu()))
        elif task == "inverse":
            mask = masks.inverse_transition[:, :, None].expand_as(output.inverse_action)
            error = _masked_squared_error(output.inverse_action.float(), batch["action"].float(), mask)
            if error is not None:
                values[task].append(float(torch.sqrt(error).cpu()))
        elif task == "history_action":
            mask = masks.history_action_transition[:, :, None].expand_as(output.history_action)
            error = _masked_squared_error(output.history_action.float(), batch["action"].float(), mask)
            if error is not None:
                values[task].append(float(torch.sqrt(error).cpu()))
        else:
            errors = []
            for prediction, target, mask in (
                (output.physical_state, batch["physical_state"], masks.state_loss),
                (output.action, batch["action"], masks.action_loss),
            ):
                error = _masked_squared_error(prediction.float(), target.float(), mask)
                if error is not None:
                    errors.append(error)
            if errors:
                values[task].append(float(torch.sqrt(torch.stack(errors).mean()).cpu()))
    means = {
        name: float(np.mean(items)) if items else math.inf for name, items in values.items()
    }
    score = (
        0.20 * means["forward_one"]
        + 0.20 * means["forward_rollout"]
        + 0.20 * means["inverse"]
        + 0.15 * means["history_action"]
        + 0.25 * means["arbitrary"]
    )
    result = {
        "selection_score": score,
        "forward_one_normalized_rmse": means["forward_one"],
        "forward_rollout_normalized_rmse": means["forward_rollout"],
        "inverse_action_normalized_rmse": means["inverse"],
        "history_action_normalized_rmse": means["history_action"],
        "arbitrary_mask_normalized_rmse": means["arbitrary"],
    }
    return result


@torch.no_grad()
def validate_action_finetune(
    model: torch.nn.Module,
    loader: DataLoader,
    masker: MaskGenerator,
    device: torch.device,
    amp: str,
    max_batches: int,
) -> dict[str, float]:
    """Fixed validation suite used for parent baselines and fine-tune selection."""
    cases = (
        ("forward_one", {}),
        ("forward_rollout", {}),
        ("inverse_local", {"force_task": "inverse", "force_length": 32}),
        ("inverse_full", {"force_task": "inverse", "force_length": 128}),
        ("history_action", {"force_task": "history_action", "force_length": 32}),
        ("completion_element", {"force_task": "arbitrary", "force_target": "action", "force_granularity": "element"}),
        ("completion_step", {"force_task": "arbitrary", "force_target": "action", "force_granularity": "step", "force_length": 32}),
        ("completion_feature", {"force_task": "arbitrary", "force_target": "action", "force_granularity": "feature"}),
        ("completion_semantic", {"force_task": "arbitrary", "force_target": "action", "force_granularity": "semantic"}),
        ("arbitrary_action", {"force_task": "arbitrary", "force_target": "action"}),
        ("arbitrary_state", {"force_task": "arbitrary", "force_target": "state"}),
        ("state_step32", {"force_task": "arbitrary", "force_target": "state", "force_granularity": "step", "force_length": 32}),
    )
    values: dict[str, list[float]] = {name: [] for name, _ in cases}
    model.eval()
    masker.set_step(max(masker.optimizer_step, 40_000))
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        name, arguments = cases[batch_index % len(cases)]
        batch = _device_batch(cpu_batch, device)
        force_task = name if name.startswith("forward_") else arguments.get("force_task")
        force_length = arguments.get("force_length")
        if name in {"inverse_local", "history_action"}:
            force_length = (1, 4, 8, 16, 32)[
                (batch_index // len(cases)) % 5
            ]
        masks = masker.generate(
            batch,
            force_task=force_task,
            force_length=force_length,
            force_target=arguments.get("force_target"),
            force_granularity=arguments.get("force_granularity"),
        )
        with _autocast(device, amp):
            output = model(batch, masks, sample_from_prior=True, deterministic=True)
        error: torch.Tensor | None = None
        if name == "forward_one":
            target = batch["physical_state"][:, 1:] - batch["physical_state"][:, :-1]
            mask = masks.forward_transition[:, :, None].expand_as(target)
            error = _masked_squared_error(output.forward_delta.float(), target.float(), mask)
        elif name == "forward_rollout":
            rollout_errors = []
            if output.rollout_state is not None:
                for index in range(output.rollout_state.shape[0]):
                    start = int(masks.rollout_start[index].item())
                    horizon = min(8, int(masks.rollout_horizon[index].item()))
                    if horizon > 0:
                        rollout_errors.append(torch.square(
                            output.rollout_state[index, :horizon].float()
                            - batch["physical_state"][index, start + 1 : start + horizon + 1].float()
                        ).mean())
            if rollout_errors:
                error = torch.stack(rollout_errors).mean()
        elif name.startswith("inverse_"):
            mask = masks.inverse_transition[:, :, None].expand_as(output.inverse_action)
            error = _masked_squared_error(output.inverse_action.float(), batch["action"].float(), mask)
        elif name == "history_action":
            mask = masks.history_action_transition[:, :, None].expand_as(output.history_action)
            error = _masked_squared_error(output.history_action.float(), batch["action"].float(), mask)
        elif name.startswith("completion_") or name == "arbitrary_action":
            error = _masked_squared_error(output.action.float(), batch["action"].float(), masks.action_loss)
        else:
            error = _masked_squared_error(
                output.physical_state.float(), batch["physical_state"].float(), masks.state_loss
            )
        if error is not None:
            values[name].append(float(torch.sqrt(error).cpu()))
    means = {name: float(np.mean(items)) if items else math.inf for name, items in values.items()}
    completion_macro = float(np.mean([
        means["completion_element"], means["completion_step"],
        means["completion_feature"], means["completion_semantic"],
    ]))
    score = (
        0.30 * means["inverse_local"]
        + 0.25 * means["inverse_full"]
        + 0.25 * completion_macro
        + 0.10 * means["history_action"]
        + 0.10 * means["arbitrary_action"]
    )
    result = {
        "selection_score": score,
        "action_inverse_local_normalized_rmse": means["inverse_local"],
        "action_inverse_full_128_normalized_rmse": means["inverse_full"],
        "action_completion_macro_normalized_rmse": completion_macro,
        "history_action_normalized_rmse": means["history_action"],
        "arbitrary_action_normalized_rmse": means["arbitrary_action"],
        "forward_one_normalized_rmse": means["forward_one"],
        "forward_rollout_8_normalized_rmse": means["forward_rollout"],
        "arbitrary_state_normalized_rmse": means["arbitrary_state"],
        "state_step_32_normalized_rmse": means["state_step32"],
        **{f"rmse/{name}": value for name, value in means.items()},
    }
    model.train()
    return result


ACTION_PARAMETER_PREFIXES = (
    "action_joint", "action_joint_pool", "action_fusion", "action_joint_decoder",
    "action_joint_output", "inverse_relation", "history_relation",
)
STATE_PARAMETER_PREFIXES = (
    "state_joint", "state_joint_pool", "state_base", "state_fusion",
    "state_joint_decoder", "state_joint_output", "state_base_output",
    "state_contact_output", "forward_relation", "forward_continuous",
    "forward_contact", "auxiliary_head",
)


def action_finetune_parameter_groups(
    model: torch.nn.Module, learning_rates: dict[str, float]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[torch.nn.Parameter]] = {"action": [], "shared": [], "state": []}
    names: dict[str, list[str]] = {"action": [], "shared": [], "state": []}
    for name, parameter in model.named_parameters():
        root = name.split(".", 1)[0]
        group = (
            "action" if root.startswith(ACTION_PARAMETER_PREFIXES)
            else "state" if root.startswith(STATE_PARAMETER_PREFIXES)
            else "shared"
        )
        grouped[group].append(parameter)
        names[group].append(name)
    if not all(grouped.values()):
        raise ValueError("fine-tune optimizer produced an empty parameter group")
    parameters = [parameter for values in grouped.values() for parameter in values]
    if len({id(parameter) for parameter in parameters}) != len(list(model.parameters())):
        raise RuntimeError("fine-tune parameter groups are not a disjoint model partition")
    result = [
        {"params": grouped[name], "lr": float(learning_rates[name]), "group_name": name}
        for name in ("action", "shared", "state")
    ]
    summary = {
        name: {
            "learning_rate": float(learning_rates[name]),
            "parameter_count": sum(parameter.numel() for parameter in grouped[name]),
            "parameter_names": names[name],
        }
        for name in grouped
    }
    return result, summary


def load_weight_only_initialization(
    model: torch.nn.Module,
    checkpoint_path: Path,
    dataset_manifest_hash: str,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != "sonic_state_action_cvae_checkpoint_v2":
        raise ValueError("weights-only initialization requires a PhysicsTransformer v2 checkpoint")
    if checkpoint.get("dataset_manifest_sha256") != dataset_manifest_hash:
        raise ValueError("parent checkpoint dataset manifest hash does not match")
    parent_model = checkpoint.get("config", {}).get("model", {})
    architecture_keys = (
        "kind", "d_model", "encoder_layers", "decoder_layers", "heads", "ffn_dim",
        "latent_dim", "joint_width", "state_dim", "joint_robot_info_dim",
        "global_robot_info_dim", "actuator_type_count", "context_mode",
        "include_previous_action", "robot_info_dim", "dynamics_context_dim",
        "auxiliary_dim", "token_layout", "robot_conditioning",
    )
    mismatches = {
        key: (parent_model.get(key), model_config.get(key))
        for key in architecture_keys
        if parent_model.get(key) != model_config.get(key)
    }
    if mismatches:
        raise ValueError(f"parent/model architecture mismatch: {mismatches}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return {
        "initialization": "weights_only",
        "parent_checkpoint": str(checkpoint_path),
        "parent_checkpoint_sha256": file_sha256(checkpoint_path),
        "parent_step": checkpoint.get("step"),
        "parent_best_score": checkpoint.get("best_score"),
        "optimizer_reinitialized": True,
        "scheduler_reinitialized": True,
        "amp_scaler_reinitialized": True,
        "rng_reinitialized": True,
    }


def _checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    step: int,
    best_score: float,
    dataset_manifest_hash: str,
    provenance: dict[str, Any] | None = None,
    curriculum_stage: dict[str, int] | None = None,
) -> dict[str, Any]:
    state = {
        "format_version": (
            "sonic_state_action_cvae_checkpoint_v2"
            if config["model"]["kind"] == "physics_transformer"
            else "sonic_state_action_cvae_checkpoint_v1"
        ),
        "step": step,
        "best_score": best_score,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "amp_scaler": scaler.state_dict(),
        "config": config,
        "dataset_manifest_sha256": dataset_manifest_hash,
        "parameter_count": parameter_count(model),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if provenance is not None:
        state["fine_tune"] = {
            **provenance,
            "fine_tune_local_step": step,
            "mask_curriculum_stage": curriculum_stage,
        }
    return state


def train(
    dataset_run: Path,
    output_run: Path,
    config: dict[str, Any],
    smoke: bool = False,
    init_checkpoint: Path | None = None,
    fine_tune_mode: str | None = None,
    overfit_phase: str | None = None,
) -> dict[str, Any]:
    dataset_run = dataset_run.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "videos"):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = config["data"]
    if overfit_phase not in {None, "capacity", "full"}:
        raise ValueError(f"unsupported overfit phase {overfit_phase!r}")
    if fine_tune_mode is not None and overfit_phase is not None:
        raise ValueError("Action fine-tune and overfit modes are mutually exclusive")
    train_split = str(data_config.get("train_split", "train"))
    validation_split = str(data_config.get("validation_split", "validation"))
    train_dataset = StateActionWindowDataset(
        dataset_run,
        train_split,
        int(data_config["window_transitions"]),
        int(data_config["validation_stride"]),
        data_config.get("max_train_episodes"),
        random_crop=bool(data_config.get("random_crop", True)),
        action_energy_crop_probability=float(
            data_config.get("action_energy_crop_probability", 0.0)
        ),
        action_energy_top_fraction=float(
            data_config.get("action_energy_top_fraction", 0.25)
        ),
    )
    validation_dataset = StateActionWindowDataset(
        dataset_run,
        validation_split,
        int(data_config["window_transitions"]),
        int(data_config["validation_stride"]),
        data_config.get("max_validation_episodes"),
        random_crop=False,
    )
    if (
        train_dataset.state_dim != validation_dataset.state_dim
        or train_dataset.include_previous_action != validation_dataset.include_previous_action
    ):
        raise ValueError("train/validation dataset representations differ")
    config["model"]["state_dim"] = train_dataset.state_dim
    config["model"]["include_previous_action"] = train_dataset.include_previous_action
    config["model"]["robot_info_dim"] = train_dataset.robot_info_dim
    config["model"]["dynamics_context_dim"] = train_dataset.dynamics_context_dim
    config["model"]["auxiliary_dim"] = train_dataset.auxiliary_dim
    config["model"]["joint_robot_info_dim"] = train_dataset.joint_robot_info_dim
    config["model"]["global_robot_info_dim"] = train_dataset.global_robot_info_dim
    config["model"]["actuator_type_count"] = len(train_dataset.actuator_type_to_id)
    if config["model"].get("context_mode", "hidden") in {"explicit", "oracle"} and not train_dataset.physics_v3:
        raise ValueError("oracle dynamics context requires a Physics State-Action v3 dataset")
    config["model"]["token_layout"] = (
        "interleaved" if train_dataset.physics_v3 else "grouped"
    )
    if config["model"]["kind"] == "physics_transformer" and not train_dataset.physics_v4:
        raise ValueError("physics_transformer requires a Physics State-Action CVAE v4 index")
    training = config["training"]
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if config["model"]["kind"] == "physics_transformer" and bool(
        data_config.get("status_balancing", True)
    ):
        statuses = [train_dataset.episodes[ref.episode_index]["status"] for ref in train_dataset.refs]
        failed = sum(status != "completed" for status in statuses)
        completed = len(statuses) - failed
        if failed and completed:
            weights = [
                0.10 / failed if status != "completed" else 0.90 / completed
                for status in statuses
            ]
            sampler = WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True, generator=generator
            )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["micro_batch"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(data_config["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(data_config["num_workers"]) > 0,
        worker_init_fn=worker_seed,
        generator=generator,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training["micro_batch"]),
        shuffle=False,
        num_workers=int(data_config["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(data_config["num_workers"]) > 0,
        worker_init_fn=worker_seed,
    )
    if len(train_loader) == 0:
        raise ValueError("training split is smaller than one micro batch")
    model = build_model(config["model"]).to(device)
    dataset_manifest = dataset_run / "manifests" / "dataset_manifest.json"
    dataset_hash = file_sha256(dataset_manifest)
    dataset_metadata = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    if overfit_phase is not None:
        if dataset_metadata.get("purpose") != "physics_state_action_32_motion_memorization":
            raise ValueError("overfit training requires a dedicated memorization subset")
        if (
            int(dataset_metadata.get("motion_count", -1)) != 32
            or int(dataset_metadata.get("canonical_episode_count", -1)) != 256
        ):
            raise ValueError("overfit training requires exactly 32 motions and 256 episodes")
        if validation_split != train_split or bool(data_config.get("random_crop", True)):
            raise ValueError("overfit training requires same-split evaluation and fixed windows")
        config["overfit_phase"] = overfit_phase
    actual_parameter_count = parameter_count(model)
    expected_parameter_count = config["model"].get("expected_parameter_count")
    if expected_parameter_count is not None and actual_parameter_count != int(expected_parameter_count):
        raise ValueError(
            f"model parameter count mismatch: expected {expected_parameter_count}, "
            f"found {actual_parameter_count}"
        )
    provenance: dict[str, Any] | None = None
    if fine_tune_mode is not None:
        if fine_tune_mode != "action":
            raise ValueError(f"unsupported fine-tune mode {fine_tune_mode!r}")
        if init_checkpoint is None:
            raise ValueError("Action fine-tune requires --init-checkpoint")
        if config["model"]["kind"] != "physics_transformer":
            raise ValueError("Action fine-tune requires physics_transformer")
        provenance = load_weight_only_initialization(
            model, init_checkpoint, dataset_hash, config["model"]
        )
    elif overfit_phase == "full":
        if init_checkpoint is None:
            raise ValueError("full overfit phase requires --init-checkpoint")
        provenance = load_weight_only_initialization(
            model, init_checkpoint, dataset_hash, config["model"]
        )
        provenance["overfit_phase"] = "full"
    elif init_checkpoint is not None:
        raise ValueError(
            "--init-checkpoint is accepted only with Action fine-tune or full overfit phase"
        )
    reference_config = dict(config["model"])
    if config["model"]["kind"] == "physics_transformer":
        reference_config["kind"] = None
        reference_parameter_count = None
    else:
        reference_config["kind"] = (
            "tcn" if config["model"]["kind"] == "transformer" else "transformer"
        )
        reference_model = build_model(reference_config)
        reference_parameter_count = parameter_count(reference_model)
        del reference_model
    optimizer_groups: dict[str, Any] | None = None
    if fine_tune_mode == "action":
        parameter_groups, optimizer_groups = action_finetune_parameter_groups(
            model, training["learning_rates"]
        )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=float(training["weight_decay"]),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
    max_steps = int(training["max_optimizer_steps"])
    scheduler_config = training.get("scheduler", {})
    scheduler_kind = str(scheduler_config.get("kind", "warmup_cosine"))
    if scheduler_kind != "warmup_cosine":
        raise ValueError(f"unsupported scheduler kind {scheduler_kind!r}")
    warmup = int(scheduler_config.get("warmup_steps", training["warmup_steps"]))
    min_lr_ratio = float(scheduler_config.get("min_lr_ratio", 0.0))

    def lr_factor(step: int) -> float:
        return warmup_cosine_factor(step, warmup, max_steps, min_lr_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and training["amp"] == "fp16"
    )
    masker = MaskGenerator(config["masking"])
    parent_validation: dict[str, float] | None = None
    if fine_tune_mode == "action":
        validation_devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda" else []
        )
        with torch.random.fork_rng(devices=validation_devices):
            torch.manual_seed(seed + 90_001)
            parent_validation = validate_action_finetune(
                model, validation_loader, masker, device, str(training["amp"]),
                int(training["validation_max_batches"]),
            )
        atomic_write_json(
            output_run / "manifests" / "parent_validation_baseline.json",
            parent_validation,
        )
        model.train()
    atomic_write_json(output_run / "manifests" / "config.json", config)
    atomic_write_json(
        output_run / "manifests" / "model.json",
        {
            "kind": config["model"]["kind"],
            "parameter_count": parameter_count(model),
            "comparison_kind": reference_config["kind"],
            "comparison_parameter_count": reference_parameter_count,
            "parameter_count_ratio": (
                parameter_count(model) / reference_parameter_count
                if reference_parameter_count else None
            ),
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "dataset_run": str(dataset_run),
            "dataset_manifest_sha256": dataset_hash,
            "fine_tune_mode": fine_tune_mode,
            "overfit_phase": overfit_phase,
            "model_profile": config["model"].get("profile"),
            "fixed_window_count": len(train_dataset),
            "optimizer_parameter_groups": optimizer_groups,
            "initialization": provenance,
        },
    )
    if provenance is not None:
        provenance_name = "overfit_provenance.json" if overfit_phase else "fine_tune_provenance.json"
        atomic_write_json(output_run / "manifests" / provenance_name, provenance)
    metrics_path = output_run / "logs" / "metrics.jsonl"
    stream = _infinite(train_loader)
    accumulation = int(training["gradient_accumulation"])
    best_score = math.inf
    best_unguarded_score = math.inf
    guarded_checkpoint_found = fine_tune_mode is None
    best_validation: dict[str, Any] | None = None
    validations_without_improvement = 0
    losses: list[float] = []
    validation_scores: list[float] = []
    overfit_pass_streak = 0
    overfit_consecutive_pass = False
    latent_mode = str(training.get("latent_mode", "sample"))
    if latent_mode not in {"sample", "posterior_mean"}:
        raise ValueError(f"unsupported latent mode {latent_mode!r}")
    fixed_window_count = len(train_dataset)
    samples_per_step = int(training["micro_batch"]) * accumulation
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for optimizer_step in range(1, max_steps + 1):
        masker.set_step(optimizer_step)
        accumulated: dict[str, float] = {}
        for _ in range(accumulation):
            batch = _device_batch(next(stream), device)
            masks = masker.generate(batch)
            beta = float(training["kl_beta"]) * min(
                optimizer_step / max(int(training["kl_warmup_steps"]), 1), 1.0
            )
            with _autocast(device, str(training["amp"])):
                output = model(batch, masks, deterministic=latent_mode == "posterior_mean")
                loss = compute_loss(output, batch, masks, training, beta)
                scaled_loss = loss.total / accumulation
            scaler.scale(scaled_loss).backward()
            for name, value in loss.detached().items():
                accumulated[name] = accumulated.get(name, 0.0) + value / accumulation
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip"])
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if not math.isfinite(accumulated["total"]):
            raise FloatingPointError(f"non-finite loss at optimizer step {optimizer_step}")
        losses.append(accumulated["total"])
        train_record = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "phase": "train",
            "optimizer_step": optimizer_step,
            "samples_seen": optimizer_step * samples_per_step,
            "fixed_window_count": fixed_window_count,
            "effective_epoch": optimizer_step * samples_per_step / fixed_window_count,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "learning_rates": {
                str(group.get("group_name", index)): group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
            "gradient_norm": float(gradient_norm),
            **accumulated,
        }
        with metrics_path.open("a", encoding="utf-8") as metrics_stream:
            metrics_stream.write(json.dumps(train_record, ensure_ascii=False) + "\n")

        validation_interval = int(training["validation_interval"])
        if optimizer_step % validation_interval == 0 or optimizer_step == max_steps:
            validation_devices = (
                [device.index if device.index is not None else torch.cuda.current_device()]
                if device.type == "cuda"
                else []
            )
            # Reuse the same validation masks at every checkpoint. Mixed task
            # losses and the rollout curriculum make unrelated random masks an
            # invalid smoke-progress comparison.
            with torch.random.fork_rng(devices=validation_devices):
                torch.manual_seed(seed + 90_001)
                validation = (
                    validate_action_finetune(
                        model, validation_loader, masker, device,
                        str(training["amp"]), int(training["validation_max_batches"]),
                    )
                    if fine_tune_mode == "action" or overfit_phase is not None
                    else validate(
                        model, validation_loader, masker, device,
                        str(training["amp"]), int(training["validation_max_batches"]),
                    )
                )
            state_guard_pass = True
            state_guard_ratios: dict[str, float] = {}
            if fine_tune_mode == "action":
                assert parent_validation is not None
                guard_keys = (
                    "forward_one_normalized_rmse",
                    "forward_rollout_8_normalized_rmse",
                    "arbitrary_state_normalized_rmse",
                    "state_step_32_normalized_rmse",
                )
                state_guard_ratios = {
                    key: validation[key] / max(parent_validation[key], 1e-12)
                    for key in guard_keys
                }
                state_guard_pass = all(
                    ratio <= float(training.get("state_guard_ratio", 1.05))
                    for ratio in state_guard_ratios.values()
                )
                validation["state_guard_pass"] = state_guard_pass
                validation["state_guard_ratios"] = state_guard_ratios
            if overfit_phase is not None:
                gate = overfit_gate(validation, training.get("overfit_thresholds"))
                validation.update(gate)
                overfit_pass_streak = overfit_pass_streak + 1 if gate["overfit_pass"] else 0
                validation["overfit_pass_streak"] = overfit_pass_streak
                score = float(gate["overfit_score"])
            else:
                score = float(validation["selection_score"])
            validation_record = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "phase": "validation",
                "optimizer_step": optimizer_step,
                **validation,
            }
            with metrics_path.open("a", encoding="utf-8") as metrics_stream:
                metrics_stream.write(json.dumps(validation_record, ensure_ascii=False) + "\n")
            validation_scores.append(float(score))
            checkpoint = _checkpoint_state(
                model, optimizer, scheduler, scaler, config, optimizer_step,
                score,
                dataset_hash, provenance, masker.curriculum_stage(),
            )
            if fine_tune_mode == "action" and score < best_unguarded_score:
                best_unguarded_score = score
                atomic_torch_save(
                    output_run / "checkpoints" / "best_unguarded.pt", checkpoint
                )
            if state_guard_pass and score < best_score:
                best_score = score
                guarded_checkpoint_found = True
                best_validation = dict(validation)
                validations_without_improvement = 0
                atomic_torch_save(
                    output_run / "checkpoints" / "best.pt",
                    _checkpoint_state(
                        model, optimizer, scheduler, scaler, config, optimizer_step,
                        best_score, dataset_hash, provenance, masker.curriculum_stage(),
                    ),
                )
            else:
                validations_without_improvement += 1
            atomic_torch_save(
                output_run / "checkpoints" / "last.pt",
                _checkpoint_state(
                    model, optimizer, scheduler, scaler, config, optimizer_step,
                    best_score, dataset_hash, provenance, masker.curriculum_stage(),
                ),
            )
            if overfit_phase is not None:
                write_training_svg(
                    metrics_path,
                    output_run / "videos/training_curves.svg",
                    training.get("overfit_thresholds", DEFAULT_OVERFIT_THRESHOLDS),
                )
                minimum_steps = int(training.get("minimum_optimizer_steps", 0))
                required_streak = int(training.get("overfit_consecutive_validations", 2))
                if optimizer_step >= minimum_steps and overfit_pass_streak >= required_streak:
                    overfit_consecutive_pass = True
                    break
            elif validations_without_improvement >= int(training["early_stopping_patience"]):
                break

    best_path = output_run / "checkpoints" / "best.pt"
    if fine_tune_mode == "action" and not guarded_checkpoint_found:
        if smoke and (output_run / "checkpoints" / "best_unguarded.pt").is_file():
            unguarded = torch.load(
                output_run / "checkpoints" / "best_unguarded.pt",
                map_location="cpu", weights_only=False,
            )
            atomic_torch_save(best_path, unguarded)
            best_score = float(unguarded["best_score"])
        else:
            diagnostics = {
                "passed": False,
                "reason": "no validation checkpoint satisfied all State guards",
                "parent_validation": parent_validation,
                "best_unguarded_score": best_unguarded_score,
                "diagnostic_checkpoint": str(
                    output_run / "checkpoints" / "best_unguarded.pt"
                ),
            }
            atomic_write_json(
                output_run / "manifests" / "training_summary.json", diagnostics
            )
            raise RuntimeError(diagnostics["reason"])
    reopened = torch.load(best_path, map_location="cpu", weights_only=False)
    expected_format = (
        "sonic_state_action_cvae_checkpoint_v2"
        if config["model"]["kind"] == "physics_transformer"
        else "sonic_state_action_cvae_checkpoint_v1"
    )
    if reopened.get("format_version") != expected_format:
        raise ValueError("atomically written checkpoint cannot be reopened")
    smoke_decreased = True
    train_window_decreased = True
    validation_score_improved = True
    if smoke and overfit_phase is None:
        span = min(20, len(losses) // 2)
        train_window_decreased = span > 0 and float(np.mean(losses[-span:])) < float(
            np.mean(losses[:span])
        )
        validation_score_improved = (
            len(validation_scores) >= 2
            and min(validation_scores[1:]) < validation_scores[0]
        )
        smoke_decreased = train_window_decreased or validation_score_improved
        if not smoke_decreased:
            atomic_write_json(
                output_run / "manifests" / "smoke_diagnostics.json",
                {
                    "train_window_decreased": train_window_decreased,
                    "validation_score_improved": validation_score_improved,
                    "validation_scores": validation_scores,
                    "first_train_window_mean": (
                        float(np.mean(losses[:span])) if span else None
                    ),
                    "last_train_window_mean": (
                        float(np.mean(losses[-span:])) if span else None
                    ),
                },
            )
            raise RuntimeError(
                "smoke showed no improvement in either the fixed validation suite "
                "or the train-loss windows"
            )
    summary = {
        "passed": True,
        "smoke": smoke,
        "model_kind": config["model"]["kind"],
        "optimizer_steps": len(losses),
        "best_validation_score": best_score,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "smoke_loss_decreased": smoke_decreased,
        "smoke_train_window_decreased": train_window_decreased,
        "smoke_validation_score_improved": validation_score_improved,
        "validation_scores": validation_scores,
        "parent_validation": parent_validation,
        "best_validation": best_validation,
        "state_guard_pass": (
            bool(best_validation and best_validation.get("state_guard_pass"))
            if fine_tune_mode == "action" else None
        ),
        "initialization": provenance,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(output_run / "checkpoints" / "last.pt"),
    }
    if overfit_phase is not None:
        summary.update({
            "passed": bool(overfit_consecutive_pass),
            "overfit_phase": overfit_phase,
            "model_profile": config["model"].get("profile", "unspecified"),
            "seed": seed,
            "parameter_count": actual_parameter_count,
            "best_overfit_score": best_score,
            "overfit_consecutive_pass": overfit_consecutive_pass,
            "required_consecutive_validations": int(
                training.get("overfit_consecutive_validations", 2)
            ),
            "fixed_window_count": fixed_window_count,
            "samples_seen": len(losses) * samples_per_step,
            "effective_epochs": len(losses) * samples_per_step / fixed_window_count,
            "memorization_benchmark": True,
            "generalization_claim_allowed": False,
            "scheduler": {
                "kind": scheduler_kind,
                "warmup_steps": warmup,
                "min_lr_ratio": min_lr_ratio,
            },
        })
        write_training_svg(
            metrics_path,
            output_run / "videos/training_curves.svg",
            training.get("overfit_thresholds", DEFAULT_OVERFIT_THRESHOLDS),
        )
        if overfit_phase == "full" and overfit_consecutive_pass:
            model.load_state_dict(reopened["model"], strict=True)
            validation_devices = (
                [device.index if device.index is not None else torch.cuda.current_device()]
                if device.type == "cuda" else []
            )
            with torch.random.fork_rng(devices=validation_devices):
                torch.manual_seed(seed + 120_001)
                summary["input_sensitivity"] = evaluate_input_sensitivity(
                    model,
                    validation_loader,
                    masker,
                    device,
                    list(dataset_metadata["joint_names"]),
                    output_run,
                    int(training.get("sensitivity_max_batches", 16)),
                )
    if fine_tune_mode == "action" and not smoke:
        assert parent_validation is not None and best_validation is not None
        improvement = {
            "inverse_local": 1.0 - best_validation["action_inverse_local_normalized_rmse"]
            / parent_validation["action_inverse_local_normalized_rmse"],
            "inverse_full_128": 1.0 - best_validation["action_inverse_full_128_normalized_rmse"]
            / parent_validation["action_inverse_full_128_normalized_rmse"],
            "action_completion_macro": 1.0 - best_validation["action_completion_macro_normalized_rmse"]
            / parent_validation["action_completion_macro_normalized_rmse"],
        }
        thresholds = {
            "inverse_local": 0.10,
            "inverse_full_128": 0.15,
            "action_completion_macro": 0.10,
        }
        summary["action_improvement"] = improvement
        summary["action_improvement_thresholds"] = thresholds
        summary["action_quality_pass"] = all(
            improvement[name] >= threshold for name, threshold in thresholds.items()
        )
        summary["passed"] = bool(summary["action_quality_pass"])
    atomic_write_json(output_run / "manifests" / "training_summary.json", summary)
    if fine_tune_mode == "action" and not smoke and not summary["action_quality_pass"]:
        raise RuntimeError("Action fine-tune did not satisfy all offline improvement gates")
    if overfit_phase is not None and not summary["passed"]:
        train_dataset.close()
        validation_dataset.close()
        raise RuntimeError(
            "overfit training did not satisfy the strict gate in consecutive validations"
        )
    marker_name = (
        "cvae_action_finetune_smoke.ok" if smoke else "cvae_action_finetune.ok"
    ) if fine_tune_mode == "action" else (
        ("cvae_overfit_smoke.ok" if smoke else f"cvae_overfit_{overfit_phase}.ok")
        if overfit_phase is not None else (
            "cvae_smoke_train.ok" if smoke else "cvae_train.ok"
        )
    )
    atomic_write_text(output_run / "markers" / marker_name, "PASS\n")
    train_dataset.close()
    validation_dataset.close()
    return summary


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train SONIC State-Action CVAE")
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-kind", choices=("transformer", "tcn", "physics_transformer"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--fine-tune-mode", choices=("action",))
    parser.add_argument("--overfit-phase", choices=("capacity", "full"))
    parser.add_argument("--context-mode", choices=("hidden", "oracle", "explicit"))
    parser.set_defaults(project_root=project_root)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_name = "smoke.json" if args.smoke else "default.json"
    default_path = args.project_root / "configs" / default_name
    config = load_config(default_path, args.config)
    if args.model_kind:
        config["model"]["kind"] = args.model_kind
    if args.seed is not None:
        config["seed"] = args.seed
    if args.context_mode is not None:
        config["model"]["context_mode"] = args.context_mode
    summary = train(
        args.dataset_run,
        args.output_run,
        config,
        smoke=args.smoke,
        init_checkpoint=args.init_checkpoint,
        fine_tune_mode=args.fine_tune_mode,
        overfit_phase=args.overfit_phase,
    )
    print("CVAE training: PASS")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
