"""C-stage training for the SONIC 10-history reference MLP-CVAE."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .sonic_causal_dataset import SonicCausalDataset, causal_collate
from .sonic_causal_objective import compose_causal_c_loss
from .sonic_reference_mlp_model import SonicReferenceMLPCVAE
from .util import atomic_torch_save, atomic_write_json, atomic_write_text, load_config, seed_everything

CHECKPOINT_FORMAT = "sonic_reference_mlp_cvae_checkpoint_v1"
SUMMARY_FORMAT = "sonic_reference_mlp_cvae_summary_v1"


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _save_checkpoint(path: Path, model: SonicReferenceMLPCVAE, optimizer: torch.optim.Optimizer, *, config: dict[str, Any], step: int, best_metric: float, dataset: SonicCausalDataset) -> None:
    payload = {
        "format_version": CHECKPOINT_FORMAT,
        "architecture_version": model.ARCHITECTURE_VERSION,
        "stage": "C",
        "model_config": model.config,
        "model_signature": model.architecture_instance_signature(),
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
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_torch_save(path, payload)


def _evaluate(model: SonicReferenceMLPCVAE, loader: DataLoader, device: torch.device, max_batches: int, seed: int) -> dict[str, float]:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    sums = {name: 0.0 for name in ("posterior_mean", "posterior_sample", "fixed_standard_normal")}
    count = 0
    with torch.inference_mode():
        for index, cpu_batch in enumerate(loader):
            if index >= int(max_batches):
                break
            batch = _device_batch(cpu_batch, device)
            target = batch["target_action"].float()
            mean = model(batch, stage="A").action
            posterior = model(batch, stage="C", epsilon=torch.randn(target.shape[0], model.global_latent_dim, generator=generator).to(device)).action
            standard = model.infer_from_condition(batch, epsilon=torch.randn(target.shape[0], model.global_latent_dim, generator=generator).to(device)).action
            sums["posterior_mean"] += float((mean.float() - target).square().sum())
            sums["posterior_sample"] += float((posterior.float() - target).square().sum())
            sums["fixed_standard_normal"] += float((standard.float() - target).square().sum())
            count += int(target.numel())
    return {key: math.sqrt(value / max(1, count)) for key, value in sums.items()}


def train(config: dict[str, Any], output_run: Path, *, smoke: bool = False) -> dict[str, Any]:
    seed_everything(int(config.get("seed", 20261102)))
    output = output_run.expanduser().resolve()
    for folder in ("checkpoints", "manifests", "markers", "logs", "plots", "data", "videos"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    data_cfg = dict(config.get("data", {}))
    dataset_run = os.environ.get("CVAE_DATASET_RUN", data_cfg.get("dataset_run"))
    if not dataset_run:
        raise ValueError("data.dataset_run or CVAE_DATASET_RUN is required")
    dataset = SonicCausalDataset(dataset_run, split="train", max_samples=data_cfg.get("max_samples"))
    train_motion_keys = list(dataset.motion_keys)
    batch_size = int(config.get("training", {}).get("micro_batch", 32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=causal_collate)
    eval_loader = DataLoader(dataset, batch_size=min(batch_size, 128), shuffle=False, num_workers=0, collate_fn=causal_collate)
    model = SonicReferenceMLPCVAE(dict(config["model"])).to(torch.device(str(config.get("device", "cuda:0"))))
    device = next(model.parameters()).device
    training = dict(config.get("training", {}))
    resume_path = os.environ.get("CVAE_RESUME_CHECKPOINT")
    start_step = 0
    if resume_path:
        payload = torch.load(Path(resume_path).expanduser().resolve(), map_location="cpu", weights_only=False)
        if payload.get("format_version") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported resume checkpoint format: {payload.get('format_version')!r}")
        if payload.get("architecture_version") != model.ARCHITECTURE_VERSION:
            raise ValueError("resume checkpoint architecture does not match the reference MLP-CVAE")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        start_step = int(payload.get("optimizer_step", 0))
    configured_max_steps = int(training.get("max_optimizer_steps", 245000))
    max_steps = min(configured_max_steps, 2) if smoke else int(os.environ.get("CVAE_MAX_STEPS_TOTAL", configured_max_steps))
    if max_steps <= start_step:
        raise ValueError(f"CVAE_MAX_STEPS_TOTAL={max_steps} must be greater than resume step {start_step}")
    max_seconds = 120.0 if smoke else float(os.environ.get("CVAE_MAX_TRAIN_SECONDS", training.get("max_train_seconds", 30600)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.get("learning_rate", 2.0e-4)), weight_decay=float(training.get("weight_decay", 0.0)))
    if resume_path:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    warmup = int(training.get("warmup_steps", 4000))
    min_lr = float(training.get("min_learning_rate", 1.0e-5))
    peak_lr = float(training.get("learning_rate", 2.0e-4))
    iterator = iter(loader)
    best_metric = float(payload.get("best_metric", float("inf"))) if resume_path else float("inf")
    previous_summary = output / "manifests" / "sonic_reference_mlp_cvae_summary.json"
    previous = json.loads(previous_summary.read_text()) if resume_path and previous_summary.is_file() else {}
    best_metrics: dict[str, float] | None = previous.get("best_metrics") or None
    last_metrics: dict[str, float] | None = previous.get("last_metrics") or None
    started = time.monotonic()
    interrupted = False
    stop_reason = "max_steps"

    def handler(_signum: int, _frame: Any) -> None:
        nonlocal interrupted, stop_reason
        interrupted = True
        stop_reason = "signal"

    old_handlers = {signal.SIGTERM: signal.signal(signal.SIGTERM, handler), signal.SIGINT: signal.signal(signal.SIGINT, handler)}
    try:
        for step in range(start_step + 1, max_steps + 1):
            if interrupted or time.monotonic() - started >= max_seconds:
                stop_reason = "wall_clock" if not interrupted else stop_reason
                break
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                cpu_batch = next(iterator)
            batch = _device_batch(cpu_batch, device)
            kl_beta_target = float(training.get("kl_beta", 0.002))
            kl_start = float(training.get("kl_beta_start", 0.0))
            kl_warmup = int(training.get("kl_warmup_steps", 20000))
            kl_beta = kl_beta_target if (kl_warmup <= 0 or step >= kl_warmup) else kl_start + (kl_beta_target - kl_start) * step / kl_warmup
            lr = peak_lr * (step + 1) / max(1, warmup) if step < warmup else min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, max_steps - warmup))))
            for group in optimizer.param_groups:
                group["lr"] = lr
            posterior = model(batch, stage="C")
            standard = model.infer_from_condition(batch, epsilon=torch.randn_like(posterior.global_latent))
            loss, terms = compose_causal_c_loss(
                posterior_action=posterior.action,
                standard_normal_action=standard.action,
                target_action=batch["target_action"],
                posterior=posterior.posterior,
                posterior_weight=1.0,
                standard_normal_weight=0.5,
                kl_beta=kl_beta,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite C loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip", 1.0)), error_if_nonfinite=True)
            optimizer.step()
            interval = 1 if smoke else int(training.get("checkpoint_interval", 5000))
            if step == start_step + 1 or step % interval == 0 or step == max_steps:
                metrics = _evaluate(model, eval_loader, device, int(training.get("max_val_batches", 128)), int(config.get("seed", 20261102)) + step)
                metrics.update({"loss": float(loss.detach().cpu()), "posterior_action_loss": float(terms["posterior_action"].detach().cpu()), "standard_normal_action_loss": float(terms["standard_normal_action"].detach().cpu()), "kl": float(terms["kl"].detach().cpu()), "kl_beta": kl_beta, "learning_rate": lr})
                last_metrics = metrics
                score = float(metrics["fixed_standard_normal"])
                if score < best_metric:
                    best_metric = score
                    best_metrics = dict(metrics)
                    _save_checkpoint(output / "checkpoints/best.pt", model, optimizer, config=config, step=step, best_metric=best_metric, dataset=dataset)
                if step % int(training.get("checkpoint_interval", 5000)) == 0 or step == max_steps or smoke:
                    _save_checkpoint(output / "checkpoints/last.pt", model, optimizer, config=config, step=step, best_metric=best_metric, dataset=dataset)
                print(f"[reference-mlp-cvae] step={step} fixed_normal={metrics['fixed_standard_normal']:.6g} posterior={metrics['posterior_mean']:.6g} kl={metrics['kl']:.6g}", flush=True)
            if interrupted:
                break
    finally:
        for sig, old in old_handlers.items():
            signal.signal(sig, old)
        dataset.close()
    completed = step if "step" in locals() else 0
    summary = {
        "summary_format": SUMMARY_FORMAT,
        "execution_pass": True,
        "quality_pass": None,
        "smoke": bool(smoke),
        "completed_optimizer_steps": int(completed),
        "requested_max_steps": int(max_steps),
        "stop_reason": stop_reason,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_signature": model.architecture_instance_signature(),
        "train_motion_keys": train_motion_keys,
        "best_metric": None if best_metrics is None else best_metric,
        "best_metrics": best_metrics,
        "last_metrics": last_metrics,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output / "manifests/sonic_reference_mlp_cvae_summary.json", summary)
    atomic_write_text(output / "markers" / ("sonic_reference_mlp_cvae_smoke.ok" if smoke else "sonic_reference_mlp_cvae_execution.ok"), "PASS\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    summary = train(load_config(args.config), args.output_run, smoke=args.smoke)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
