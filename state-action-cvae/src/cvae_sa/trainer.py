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
from torch.utils.data import DataLoader

from .dataset import StateActionWindowDataset, worker_seed
from .losses import compute_loss
from .masking import MaskGenerator
from .models import build_model, parameter_count
from .util import (
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    load_config,
    seed_everything,
)


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


def _checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    step: int,
    best_score: float,
    dataset_manifest_hash: str,
) -> dict[str, Any]:
    return {
        "format_version": "sonic_state_action_cvae_checkpoint_v1",
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


def train(
    dataset_run: Path,
    output_run: Path,
    config: dict[str, Any],
    smoke: bool = False,
) -> dict[str, Any]:
    dataset_run = dataset_run.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "videos"):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = config["data"]
    train_dataset = StateActionWindowDataset(
        dataset_run,
        "train",
        int(data_config["window_transitions"]),
        int(data_config["validation_stride"]),
        data_config.get("max_train_episodes"),
        random_crop=True,
    )
    validation_dataset = StateActionWindowDataset(
        dataset_run,
        "validation",
        int(data_config["window_transitions"]),
        int(data_config["validation_stride"]),
        data_config.get("max_validation_episodes"),
        random_crop=False,
    )
    training = config["training"]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["micro_batch"]),
        shuffle=True,
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
    reference_config = dict(config["model"])
    reference_config["kind"] = (
        "tcn" if config["model"]["kind"] == "transformer" else "transformer"
    )
    reference_model = build_model(reference_config)
    reference_parameter_count = parameter_count(reference_model)
    del reference_model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    max_steps = int(training["max_optimizer_steps"])
    warmup = int(training["warmup_steps"])

    def lr_factor(step: int) -> float:
        if step < warmup:
            return max(step, 1) / max(warmup, 1)
        progress = (step - warmup) / max(max_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and training["amp"] == "fp16"
    )
    masker = MaskGenerator(config["masking"])
    dataset_manifest = dataset_run / "manifests" / "dataset_manifest.json"
    dataset_hash = file_sha256(dataset_manifest)
    atomic_write_json(output_run / "manifests" / "config.json", config)
    atomic_write_json(
        output_run / "manifests" / "model.json",
        {
            "kind": config["model"]["kind"],
            "parameter_count": parameter_count(model),
            "comparison_kind": reference_config["kind"],
            "comparison_parameter_count": reference_parameter_count,
            "parameter_count_ratio": parameter_count(model) / reference_parameter_count,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "dataset_run": str(dataset_run),
            "dataset_manifest_sha256": dataset_hash,
        },
    )
    metrics_path = output_run / "logs" / "metrics.jsonl"
    stream = _infinite(train_loader)
    accumulation = int(training["gradient_accumulation"])
    best_score = math.inf
    validations_without_improvement = 0
    losses: list[float] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for optimizer_step in range(1, max_steps + 1):
        accumulated: dict[str, float] = {}
        for _ in range(accumulation):
            batch = _device_batch(next(stream), device)
            masks = masker.generate(batch)
            beta = float(training["kl_beta"]) * min(
                optimizer_step / max(int(training["kl_warmup_steps"]), 1), 1.0
            )
            with _autocast(device, str(training["amp"])):
                output = model(batch, masks)
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
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm": float(gradient_norm),
            **accumulated,
        }
        with metrics_path.open("a", encoding="utf-8") as metrics_stream:
            metrics_stream.write(json.dumps(train_record, ensure_ascii=False) + "\n")

        validation_interval = int(training["validation_interval"])
        if optimizer_step % validation_interval == 0 or optimizer_step == max_steps:
            validation = validate(
                model,
                validation_loader,
                masker,
                device,
                str(training["amp"]),
                int(training["validation_max_batches"]),
            )
            validation_record = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "phase": "validation",
                "optimizer_step": optimizer_step,
                **validation,
            }
            with metrics_path.open("a", encoding="utf-8") as metrics_stream:
                metrics_stream.write(json.dumps(validation_record, ensure_ascii=False) + "\n")
            score = validation["selection_score"]
            if score < best_score:
                best_score = score
                validations_without_improvement = 0
                atomic_torch_save(
                    output_run / "checkpoints" / "best.pt",
                    _checkpoint_state(
                        model, optimizer, scheduler, scaler, config, optimizer_step, best_score, dataset_hash
                    ),
                )
            else:
                validations_without_improvement += 1
            atomic_torch_save(
                output_run / "checkpoints" / "last.pt",
                _checkpoint_state(
                    model, optimizer, scheduler, scaler, config, optimizer_step, best_score, dataset_hash
                ),
            )
            if validations_without_improvement >= int(training["early_stopping_patience"]):
                break

    best_path = output_run / "checkpoints" / "best.pt"
    reopened = torch.load(best_path, map_location="cpu", weights_only=False)
    if reopened.get("format_version") != "sonic_state_action_cvae_checkpoint_v1":
        raise ValueError("atomically written checkpoint cannot be reopened")
    smoke_decreased = True
    if smoke:
        span = min(20, len(losses) // 2)
        smoke_decreased = span > 0 and float(np.mean(losses[-span:])) < float(
            np.mean(losses[:span])
        )
        if not smoke_decreased:
            raise RuntimeError("smoke loss did not decrease from the first to last window")
    summary = {
        "passed": True,
        "smoke": smoke,
        "model_kind": config["model"]["kind"],
        "optimizer_steps": len(losses),
        "best_validation_score": best_score,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "smoke_loss_decreased": smoke_decreased,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(output_run / "checkpoints" / "last.pt"),
    }
    atomic_write_json(output_run / "manifests" / "training_summary.json", summary)
    marker_name = "cvae_smoke_train.ok" if smoke else "cvae_train.ok"
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
    parser.add_argument("--model-kind", choices=("transformer", "tcn"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
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
    summary = train(args.dataset_run, args.output_run, config, smoke=args.smoke)
    print("CVAE training: PASS")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
