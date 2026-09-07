from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .models import HierarchicalDecodedOutput
from .posterior_capacity import DeterministicWindowSubset, MaskBankDataset, validate_motion_prefix
from .posterior_t64_protocol import (
    PHYSICAL_MASK_NAMES,
    append_jsonl,
    evaluate,
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
    file_sha256,
    load_config,
    seed_everything,
)


def direct_output_next_step(quality_pass: bool, *, smoke: bool = False) -> str:
    if smoke:
        return "REVIEW_SMOKE_ARTIFACTS_THEN_RUN_FORMAL_F4G"
    return "RUN_H38_ENGINEERING_SMOKE" if quality_pass else "INVESTIGATE_LOSS_MASK_EVALUATOR"


def assert_output_isolated(output_run: Path, protected: list[Path]) -> None:
    resolved = output_run.expanduser().resolve()
    for source in protected:
        source = source.expanduser().resolve()
        if resolved == source or resolved.is_relative_to(source):
            raise ValueError(f"output run overlaps protected source: {source}")


class DirectWindowOutput(nn.Module):
    """One mask-invariant State/Action output table entry per fixed window."""

    def __init__(self, window_count: int, state_steps: int = 65, action_steps: int = 64) -> None:
        super().__init__()
        self.window_count = int(window_count)
        self.state_continuous = nn.Parameter(torch.zeros(window_count, state_steps, 68))
        self.state_contact_logits = nn.Parameter(torch.zeros(window_count, state_steps, 2))
        self.action = nn.Parameter(torch.zeros(window_count, action_steps, 29))

    @staticmethod
    def indices(batch: dict[str, Any]) -> torch.Tensor:
        value = batch.get("window_index")
        if not isinstance(value, torch.Tensor):
            raise ValueError("direct-output batches require stable window_index tensors")
        return value.long()

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        state_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> HierarchicalDecodedOutput:
        if state_mask.shape != batch["physical_state"].shape or action_mask.shape != batch["action"].shape:
            raise ValueError("direct-output Mask shapes disagree with targets")
        indices = self.indices(batch)
        if bool((indices < 0).any()) or bool((indices >= self.window_count).any()):
            raise ValueError("direct-output window index is out of range")
        logits = self.state_contact_logits[indices]
        physical_state = torch.cat((self.state_continuous[indices], logits.sigmoid()), dim=-1)
        return HierarchicalDecodedOutput(
            physical_state=physical_state,
            action=self.action[indices],
            state_contact_logits=logits,
        )


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


def _checkpoint(
    model: DirectWindowOutput,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    dataset_hash: str,
    window_hash: str,
    step: int,
    score: float,
) -> dict[str, Any]:
    return {
        "format_version": "sonic_posterior_direct_output_t64_checkpoint_v1",
        "optimizer_step": step,
        "best_fit_score": score,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "dataset_manifest_sha256": dataset_hash,
        "selected_windows_sha256": window_hash,
        "window_count": model.window_count,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def validate_checkpoint(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "format_version": checkpoint.get("format_version") == "sonic_posterior_direct_output_t64_checkpoint_v1",
        "dataset_hash": checkpoint.get("dataset_manifest_sha256") == expected["dataset_hash"],
        "window_hash": checkpoint.get("selected_windows_sha256") == expected["window_hash"],
        "window_count": int(checkpoint.get("window_count", -1)) == int(expected["window_count"]),
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
    smoke: bool,
) -> dict[str, Any]:
    from .dataset import StateActionWindowDataset

    dataset_run = dataset_run.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    assert_output_isolated(output_run, [dataset_run])
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "plots", "videos"):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    if not (dataset_run / "markers/cvae_overfit_subset.ok").is_file():
        raise FileNotFoundError("F4G requires the dedicated 32-motion overfit subset")
    data = config["data"]
    training = config["training"]
    if int(data["motion_count"]) != 32 or int(data["window_transitions"]) != 64 or int(data["stride"]) != 64:
        raise ValueError("F4G fixed data contract must be 32 motions, T64, stride64")
    seed_everything(int(config["seed"]))
    base = StateActionWindowDataset(
        dataset_run, "train", 64, 64, max_episodes=256, random_crop=False
    )
    selected_motions = validate_motion_prefix(base, 32)
    max_windows = 2 if smoke else data.get("max_windows")
    selected = DeterministicWindowSubset(base, max_windows)
    windows = window_identity_rows(base, selected.indices)
    window_hash = rows_sha256(windows)
    fixtures: Dataset[dict[str, Any]] = MaskBankDataset(selected, len(PHYSICAL_MASK_NAMES))
    workers = 0 if smoke else int(data.get("num_workers", 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    micro_batch = min(int(training["micro_batch"]), len(fixtures))
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        fixtures, batch_size=micro_batch, shuffle=True, num_workers=workers,
        generator=generator, drop_last=not smoke,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        fixtures, batch_size=micro_batch, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    base_loader = DataLoader(selected, batch_size=min(micro_batch, len(selected)), shuffle=False, num_workers=workers)
    fixture_seed = int(config["fixture_seed"])
    mask_maker = lambda batch: make_physical_masks(batch, fixture_seed)
    fixture_hash = mask_bank_sha256(validation_loader, mask_maker)
    model = DirectWindowOutput(len(selected)).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_parameters = len(selected) * (65 * 70 + 64 * 29)
    if parameter_count != expected_parameters:
        raise RuntimeError("F4G output-table parameter count is inconsistent")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=0.0)
    max_steps = 2 if smoke else int(training["max_optimizer_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_multiplier(
            step, int(training["warmup_steps"]), max_steps,
            float(training["minimum_learning_rate"]) / float(training["learning_rate"]),
        ),
    )
    state_std = torch.from_numpy(base.state_std)
    action_std = torch.from_numpy(base.action_std)
    records: list[dict[str, Any]] = []
    metrics_path = output_run / "logs/metrics.jsonl"
    best: dict[str, Any] | None = None
    best_score = math.inf
    pass_streak = 0
    dataset_hash = file_sha256(dataset_run / "manifests/dataset_manifest.json")

    def run_evaluation(step: int) -> dict[str, Any]:
        started = time.perf_counter()
        result = evaluate(
            model, validation_loader, base_loader, device, mask_maker,
            fit_thresholds={key: float(value) for key, value in training["fit_thresholds"].items()},
            strict_thresholds={key: float(value) for key, value in training["strict_memory_thresholds"].items()},
            exact_thresholds={key: float(value) for key, value in training["legacy_exact_thresholds"].items()},
            state_std=state_std, action_std=action_std, latent_diagnostics=False,
        )
        result["evaluation_seconds"] = time.perf_counter() - started
        result["scope"] = "same 32-motion T64 windows; fixed physical Mask bank; mask-invariant direct output table"
        result["artifacts"] = write_evaluation_artifacts(output_run, step, result)
        row = {"phase": "evaluation", "optimizer_step": step, "metrics": result}
        records.append(row)
        append_jsonl(metrics_path, row)
        return result

    initial = run_evaluation(0)
    stream = _infinite(train_loader)
    accumulation = int(training["gradient_accumulation"])
    validation_interval = 2 if smoke else int(training["validation_interval"])
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, max_steps + 1):
        started = time.perf_counter()
        aggregate = {key: 0.0 for key in ("total", "state", "action", "contact")}
        for _ in range(accumulation):
            cpu_batch = next(stream)
            state_mask, action_mask, _ = mask_maker(cpu_batch)
            batch = _device_batch(cpu_batch, device)
            output = model(batch, state_mask.to(device), action_mask.to(device))
            losses = reconstruction_loss(output, batch, state_mask.to(device), action_mask.to(device))
            (losses["total"] / accumulation).backward()
            for key in aggregate:
                aggregate[key] += float(losses[key].detach().cpu()) / accumulation
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        row = {
            "phase": "train", "optimizer_step": step, "reconstruction": aggregate,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "gradient_norm_before_clip": float(gradient),
            "gradient_was_clipped": float(gradient) > float(training["gradient_clip"]),
            "step_seconds": time.perf_counter() - started,
            "cuda_max_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        }
        records.append(row)
        append_jsonl(metrics_path, row)
        if step % validation_interval and step != max_steps:
            continue
        result = run_evaluation(step)
        score = float(result["fit_gate"]["score"])
        if score < best_score:
            best_score, best = score, result
            atomic_torch_save(
                output_run / "checkpoints/best_fit.pt",
                _checkpoint(model, optimizer, scheduler, config, dataset_hash, window_hash, step, score),
            )
        atomic_torch_save(
            output_run / "checkpoints/last.pt",
            _checkpoint(model, optimizer, scheduler, config, dataset_hash, window_hash, step, best_score),
        )
        pass_streak = pass_streak + 1 if result["fit_gate"]["passed"] else 0
        render_plots(output_run, records, best)
        if not smoke and pass_streak >= int(training["required_pass_streak"]):
            break
        model.train()
    last_three = [row["metrics"] for row in records if row["phase"] == "evaluation"][-3:]
    quality_pass = bool(
        not smoke and len(last_three) == 3 and all(row["fit_gate"]["passed"] for row in last_three)
    )
    strict_pass = bool(
        quality_pass and all(row["strict_memory_gate"]["passed"] for row in last_three)
    )
    legacy_pass = bool(
        quality_pass and all(row["legacy_exact_gate"]["passed"] for row in last_three)
    )
    render_plots(output_run, records, best)
    checkpoint_path = output_run / "checkpoints/last.pt"
    readback = validate_checkpoint(checkpoint_path, {
        "dataset_hash": dataset_hash, "window_hash": window_hash, "window_count": len(selected),
    })
    if not readback["passed"]:
        raise RuntimeError("F4G checkpoint readback failed")
    summary = {
        "format_version": "sonic_posterior_direct_output_t64_summary_v1",
        "experiment": "F4G",
        "execution_pass": True,
        "smoke": smoke,
        "quality_gate_applicable": not smoke,
        "quality_pass": quality_pass,
        "strict_memory_pass": strict_pass,
        "legacy_exact_pass": legacy_pass,
        "dataset_run": str(dataset_run),
        "dataset_manifest_sha256": dataset_hash,
        "motion_count": 32,
        "episode_count": len(base.episodes),
        "selected_motion_keys": selected_motions,
        "window_transitions": 64,
        "stride": 64,
        "window_count": len(selected),
        "selected_windows": windows,
        "selected_windows_sha256": window_hash,
        "fixture_count": len(fixtures),
        "fixture_bitmap_sha256": fixture_hash,
        "mask_names": list(PHYSICAL_MASK_NAMES),
        "shared_output_per_window": True,
        "per_fixture_output": False,
        "parameter_count": parameter_count,
        "completed_optimizer_steps": step,
        "best_fit_score": best_score,
        "best_evaluation": best,
        "step0_evaluation": initial,
        "last_three_evaluations": last_three,
        "checkpoint_readback": readback,
        "plots": render_plots(output_run, records, best),
        "scope": "loss/evaluator ceiling only; no encoder, latent, decoder, prior, or generalization claim",
        "unique_next_step": direct_output_next_step(quality_pass, smoke=smoke),
    }
    atomic_write_json(output_run / "manifests/posterior_direct_output_summary.json", summary)
    marker = "cvae_posterior_direct_output_smoke.ok" if smoke else "cvae_posterior_direct_output_execution.ok"
    atomic_write_text(output_run / "markers" / marker, "PASS\n")
    if quality_pass:
        atomic_write_text(output_run / "markers/cvae_posterior_direct_output_fit.ok", "PASS\n")
    elif not smoke:
        atomic_write_text(output_run / "markers/cvae.failed", "QUALITY_FAIL execution_complete=true fit=false\n")
    if strict_pass:
        atomic_write_text(output_run / "markers/cvae_posterior_direct_output_strict_memory.ok", "PASS\n")
    if legacy_pass:
        atomic_write_text(output_run / "markers/cvae_posterior_direct_output_legacy_exact.ok", "PASS\n")
    base.close()
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the F4G T64 direct-output ceiling")
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs/posterior_direct_output_t64.json")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    summary = run_experiment(args.dataset_run, args.output_run, config, smoke=args.smoke)
    print("Posterior F4G direct output: PASS (execution complete)")
    print(json.dumps({
        "output_run": str(args.output_run.expanduser().resolve()),
        "smoke": summary["smoke"],
        "quality_pass": summary["quality_pass"],
        "completed_optimizer_steps": summary["completed_optimizer_steps"],
        "best_fit_score": summary["best_fit_score"],
        "unique_next_step": summary["unique_next_step"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
