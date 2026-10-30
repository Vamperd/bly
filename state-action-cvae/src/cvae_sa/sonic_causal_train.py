"""C-stage training entry point for the SONIC causal CVAE."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .sonic_causal_dataset import SonicCausalDataset, causal_collate
from .sonic_causal_model import SonicCausalCVAE
from .sonic_causal_objective import compose_causal_c_loss
from .util import atomic_torch_save, load_config, seed_everything


CHECKPOINT_FORMAT = "sonic_causal_cvae_checkpoint_v1"


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _load_model_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(value.get("model", value))


def save_checkpoint(
    path: str | Path,
    model: SonicCausalCVAE,
    optimizer: torch.optim.Optimizer,
    *,
    config: dict[str, Any],
    step: int,
    best_metric: float,
    dataset: SonicCausalDataset,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT,
        "architecture_version": model.ARCHITECTURE_VERSION,
        "stage": "C",
        "model_config": model.config,
        "model_signature": model.architecture_signature(
            model.config,
            sum(parameter.numel() for parameter in model.parameters()),
        ),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_step": int(step),
        "best_metric": float(best_metric),
        "dataset_run": str(dataset.dataset_run),
        "dataset_manifest": dataset.manifest,
        "train_motion_keys": dataset.motion_keys,
        "normalization": {
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
            "action_mean": dataset.action_mean,
            "action_std": dataset.action_std,
            "reference_mean": dataset.reference_mean,
            "reference_std": dataset.reference_std,
        },
        "training_config": config,
    }
    atomic_torch_save(path, payload)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    *,
    model_config: dict[str, Any] | None = None,
) -> tuple[SonicCausalCVAE, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported SONIC causal checkpoint: {path}")
    config = dict(model_config or payload.get("model_config") or {})
    model = SonicCausalCVAE(config).to(device)
    expected = model.architecture_signature(
        model.config, sum(parameter.numel() for parameter in model.parameters())
    )
    if payload.get("model_signature") != expected:
        raise ValueError("SONIC causal checkpoint architecture signature mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def _batch_loss(
    model: SonicCausalCVAE,
    batch: dict[str, Any],
    *,
    kl_beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    posterior_output = model(batch, stage="C")
    standard_output = model.infer_from_condition(
        batch,
        epsilon=torch.randn_like(posterior_output.global_latent),
    )
    if posterior_output.posterior is None:
        raise RuntimeError("C-stage output did not contain a posterior")
    return compose_causal_c_loss(
        posterior_action=posterior_output.action,
        standard_normal_action=standard_output.action,
        target_action=batch["target_action"],
        posterior=posterior_output.posterior,
        posterior_weight=1.0,
        standard_normal_weight=0.5,
        kl_beta=kl_beta,
    )


def _learning_rate_at(step: int, training: dict[str, Any], max_steps: int) -> float:
    peak = float(training.get("learning_rate", 1.0e-4))
    schedule = str(training.get("lr_schedule", "constant"))
    warmup = int(training.get("warmup_steps", 0))
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    if schedule == "constant":
        return peak
    if schedule != "cosine":
        raise ValueError(f"unsupported lr_schedule: {schedule}")
    floor = float(training.get("min_learning_rate", 0.0))
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def _kl_beta_at(step: int, training: dict[str, Any]) -> float:
    target = float(training.get("kl_beta", 0.002))
    start = float(training.get("kl_beta_start", target))
    warmup = int(training.get("kl_warmup_steps", 0))
    if warmup <= 0 or step >= warmup:
        return target
    return start + (target - start) * step / warmup


@torch.no_grad()
def _evaluate_routes(
    model: SonicCausalCVAE,
    loader: DataLoader,
    device: torch.device,
    *,
    latent_seed: int,
    max_batches: int,
) -> dict[str, float]:
    """Read back all deployment routes with one fixed standard-normal probe."""

    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(int(latent_seed))
    sums = {name: 0.0 for name in ("posterior_mean", "posterior_sample", "fixed_standard_normal")}
    count = 0
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index >= int(max_batches):
            break
        batch = _device_batch(cpu_batch, device)
        target = batch["target_action"].float()
        posterior_mean = model(batch, stage="A").action
        posterior_sample = model(
            batch,
            stage="C",
            epsilon=torch.randn(
                target.shape[0], model.global_latent_dim, generator=generator
            ).to(device),
        ).action
        fixed_standard = model.infer_from_condition(
            batch,
            epsilon=torch.randn(
                target.shape[0], model.global_latent_dim, generator=generator
            ).to(device),
        ).action
        for name, value in (
            ("posterior_mean", posterior_mean),
            ("posterior_sample", posterior_sample),
            ("fixed_standard_normal", fixed_standard),
        ):
            sums[name] += float((value.float() - target).square().sum())
        count += int(target.numel())
    return {name: (value / max(1, count)) ** 0.5 for name, value in sums.items()}


def train(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(int(config.get("seed", 20261101)))
    device = torch.device(str(config.get("device", "cuda:0")))
    data_config = dict(config["data"])
    train_dataset = SonicCausalDataset(
        data_config["dataset_run"],
        split="train",
        max_samples=data_config.get("max_samples"),
    )
    use_validation = bool(data_config.get("use_validation", True))
    readback_split = "validation" if use_validation else "train"
    readback_dataset = (
        SonicCausalDataset(
            data_config["dataset_run"],
            split="validation",
            max_samples=data_config.get("max_val_samples"),
        )
        if use_validation
        else train_dataset
    )
    batch_size = int(config.get("training", {}).get("micro_batch", 32))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=causal_collate,
    )
    val_loader = DataLoader(
        readback_dataset,
        batch_size=min(batch_size, 128),
        shuffle=False,
        num_workers=0,
        collate_fn=causal_collate,
    )
    model = SonicCausalCVAE(dict(config["model"])).to(device)
    training = dict(config.get("training", {}))
    learning_rate = float(training.get("learning_rate", 1.0e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    max_steps = int(training.get("max_optimizer_steps", 1000))
    kl_beta = float(training.get("kl_beta", 0.002))
    grad_clip = float(training.get("gradient_clip", 1.0))
    output_dir = Path(config.get("output_dir", "runs/sonic_causal_cvae"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoints" / "best.pt"
    latest_path = output_dir / "checkpoints" / "latest.pt"
    max_train_seconds = training.get("max_train_seconds")
    checkpoint_interval = int(training.get("checkpoint_interval", 0))
    started = time.monotonic()
    out_of_time = False
    best = math.inf
    step = 0
    model.train()
    while step < max_steps and not out_of_time:
        for cpu_batch in train_loader:
            batch = _device_batch(cpu_batch, device)
            current_lr = _learning_rate_at(step, training, max_steps)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            current_beta = _kl_beta_at(step, training)
            loss, terms = _batch_loss(model, batch, kl_beta=current_beta)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite causal C loss at step {step + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            step += 1
            elapsed = time.monotonic() - started
            if step % int(training.get("log_interval", 50)) == 0 or step == max_steps:
                print(
                    f"[sonic-causal] step={step}/{max_steps} loss={float(loss):.6g} "
                    f"post={float(terms['posterior_action']):.4g} "
                    f"std={float(terms['standard_normal_action']):.4g} "
                    f"kl={float(terms['kl']):.4g} beta={current_beta:.4g} lr={current_lr:.3g} "
                    f"elapsed={elapsed / 3600:.2f}h it/s={step / max(elapsed, 1e-9):.2f}",
                    flush=True,
                )
            if checkpoint_interval > 0 and step % checkpoint_interval == 0:
                save_checkpoint(
                    latest_path,
                    model,
                    optimizer,
                    config=config,
                    step=step,
                    best_metric=float(loss),
                    dataset=train_dataset,
                )
            if max_train_seconds is not None and elapsed >= float(max_train_seconds):
                print(f"[sonic-causal] time limit reached at step={step}", flush=True)
                out_of_time = True
                break
            if step >= max_steps:
                break
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for cpu_batch in val_loader:
            batch = _device_batch(cpu_batch, device)
            loss, _terms = _batch_loss(model, batch, kl_beta=kl_beta)
            values.append(float(loss))
            if len(values) >= int(training.get("max_val_batches", 16)):
                break
    val_loss = sum(values) / max(1, len(values))
    validation_routes = _evaluate_routes(
        model,
        val_loader,
        device,
        latent_seed=int(config.get("seed", 20261101)) + 700001,
        max_batches=int(training.get("max_val_batches", 16)),
    )
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        config=config,
        step=step,
        best_metric=val_loss,
        dataset=train_dataset,
    )
    summary = {
        "format_version": "sonic_causal_cvae_summary_v1",
        "execution_pass": True,
        "stage": "C",
        "optimizer_steps": step,
        "train_seconds": time.monotonic() - started,
        "stopped_by_time_limit": out_of_time,
        "readback_split": readback_split,
        f"{readback_split}_loss": val_loss,
        f"{readback_split}_action_rmse": validation_routes,
        "checkpoint": str(checkpoint_path),
        "dataset_run": str(train_dataset.dataset_run),
        "train_samples": len(train_dataset),
        "train_motion_keys": train_dataset.motion_keys,
        "validation_samples": len(readback_dataset) if use_validation else 0,
        "architecture_version": model.ARCHITECTURE_VERSION,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    markers = output_dir / "markers"
    markers.mkdir(parents=True, exist_ok=True)
    (markers / "sonic_causal_c_execution.ok").write_text(
        "execution_pass=true\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the SONIC causal CVAE")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, help="override training.max_optimizer_steps")
    parser.add_argument("--max-hours", type=float, help="override training.max_train_seconds")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    training = config.setdefault("training", {})
    if args.max_steps is not None:
        if args.max_steps <= 0:
            parser.error("--max-steps must be positive")
        training["max_optimizer_steps"] = args.max_steps
    if args.max_hours is not None:
        if args.max_hours <= 0:
            parser.error("--max-hours must be positive")
        training["max_train_seconds"] = args.max_hours * 3600.0
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

