from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .posterior_capacity import DeterministicWindowSubset, MaskBankDataset, validate_motion_prefix
from .posterior_direct_output import (
    DirectWindowOutput,
    assert_output_isolated,
    direct_output_next_step,
    initialize_direct_output_from_targets,
)
from .posterior_t64_protocol import (
    PHYSICAL_MASK_NAMES,
    append_jsonl,
    evaluate,
    make_physical_masks,
    mask_bank_sha256,
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
    seed_everything,
)


def _metric_fingerprint(metrics: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in metrics.items()
        if key not in {"evaluation_seconds", "artifacts", "scope"}
    }
    import hashlib

    return hashlib.sha256(canonical_json_bytes(stable)).hexdigest()


def _checkpoint(
    model: DirectWindowOutput,
    config: dict[str, Any],
    dataset_hash: str,
    window_hash: str,
    fixture_hash: str,
    initialization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": "sonic_posterior_direct_output_t64_oracle_checkpoint_v1",
        "optimizer_step": 0,
        "model": model.state_dict(),
        "config": config,
        "dataset_manifest_sha256": dataset_hash,
        "selected_windows_sha256": window_hash,
        "fixture_bitmap_sha256": fixture_hash,
        "window_count": model.window_count,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initialization": initialization,
        "optimizer_state": None,
        "scheduler_state": None,
    }


def _validate_checkpoint(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "format_version": checkpoint.get("format_version")
        == "sonic_posterior_direct_output_t64_oracle_checkpoint_v1",
        "optimizer_step": int(checkpoint.get("optimizer_step", -1)) == 0,
        "dataset_hash": checkpoint.get("dataset_manifest_sha256") == expected["dataset_hash"],
        "window_hash": checkpoint.get("selected_windows_sha256") == expected["window_hash"],
        "fixture_hash": checkpoint.get("fixture_bitmap_sha256") == expected["fixture_hash"],
        "window_count": int(checkpoint.get("window_count", -1)) == int(expected["window_count"]),
        "parameter_count": int(checkpoint.get("parameter_count", -1))
        == int(expected["parameter_count"]),
        "model_state": isinstance(checkpoint.get("model"), dict),
        "optimizer_absent": checkpoint.get("optimizer_state") is None,
        "scheduler_absent": checkpoint.get("scheduler_state") is None,
    }
    return {"passed": all(checks.values()), "checks": checks, "sha256": file_sha256(path)}


def run_oracle(
    dataset_run: Path,
    output_run: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    from .dataset import StateActionWindowDataset

    dataset_run = dataset_run.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    assert_output_isolated(output_run, [dataset_run])
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "plots", "videos"):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    if not (dataset_run / "markers/cvae_overfit_subset.ok").is_file():
        raise FileNotFoundError("F4G oracle requires the dedicated 32-motion overfit subset")
    data = config["data"]
    training = config["training"]
    if int(data["motion_count"]) != 32 or int(data["window_transitions"]) != 64 or int(data["stride"]) != 64:
        raise ValueError("F4G oracle data contract must be 32 motions, T64, stride64")
    seed_everything(int(config["seed"]))
    base = StateActionWindowDataset(
        dataset_run, "train", 64, 64, max_episodes=256, random_crop=False
    )
    selected_motions = validate_motion_prefix(base, 32)
    selected = DeterministicWindowSubset(base, data.get("max_windows"))
    windows = window_identity_rows(base, selected.indices)
    window_hash = rows_sha256(windows)
    fixtures: Dataset[dict[str, Any]] = MaskBankDataset(selected, len(PHYSICAL_MASK_NAMES))
    workers = int(data.get("num_workers", 4))
    batch_size = min(int(training["micro_batch"]), len(fixtures))
    validation_loader = DataLoader(
        fixtures,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    base_loader = DataLoader(
        selected,
        batch_size=min(batch_size, len(selected)),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    mask_maker = lambda batch: make_physical_masks(batch, int(config["fixture_seed"]))
    fixture_hash = mask_bank_sha256(validation_loader, mask_maker)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectWindowOutput(len(selected)).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_parameters = len(selected) * (65 * 70 + 64 * 29)
    if parameter_count != expected_parameters:
        raise RuntimeError("F4G oracle output-table parameter count is inconsistent")
    initialization = initialize_direct_output_from_targets(model, selected)
    if not initialization["exact_parameter_copy"]:
        raise RuntimeError("F4G oracle target copy was not exact")
    atomic_write_json(output_run / "manifests/direct_output_oracle_initialization.json", initialization)

    state_std = torch.from_numpy(base.state_std)
    action_std = torch.from_numpy(base.action_std)
    records: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    metric_hashes: list[str] = []
    metrics_path = output_run / "logs/metrics.jsonl"
    for repetition in range(3):
        started = time.perf_counter()
        metrics = evaluate(
            model,
            validation_loader,
            base_loader,
            device,
            mask_maker,
            fit_thresholds={key: float(value) for key, value in training["fit_thresholds"].items()},
            strict_thresholds={key: float(value) for key, value in training["strict_memory_thresholds"].items()},
            exact_thresholds={key: float(value) for key, value in training["legacy_exact_thresholds"].items()},
            state_std=state_std,
            action_std=action_std,
            latent_diagnostics=False,
        )
        metrics["evaluation_seconds"] = time.perf_counter() - started
        metrics["scope"] = "analytic target-copy ceiling on all 32-motion T64 fixed physical fixtures"
        metrics["artifacts"] = write_evaluation_artifacts(output_run, repetition, metrics)
        metric_hash = _metric_fingerprint(metrics)
        metric_hashes.append(metric_hash)
        row = {
            "phase": "evaluation",
            "mode": "oracle_target_copy",
            "optimizer_step": 0,
            "evaluation_repetition": repetition + 1,
            "metrics": metrics,
            "metric_fingerprint": metric_hash,
        }
        records.append(row)
        evaluations.append(metrics)
        append_jsonl(metrics_path, row)
    deterministic = len(set(metric_hashes)) == 1
    quality_pass = deterministic and all(result["fit_gate"]["passed"] for result in evaluations)
    strict_pass = quality_pass and all(result["strict_memory_gate"]["passed"] for result in evaluations)
    legacy_pass = quality_pass and all(result["legacy_exact_gate"]["passed"] for result in evaluations)
    best = min(evaluations, key=lambda result: float(result["fit_gate"]["score"]))
    plots = render_plots(output_run, records, best)
    dataset_hash = file_sha256(dataset_run / "manifests/dataset_manifest.json")
    checkpoint_path = output_run / "checkpoints/oracle.pt"
    atomic_torch_save(
        checkpoint_path,
        _checkpoint(model, config, dataset_hash, window_hash, fixture_hash, initialization),
    )
    readback = _validate_checkpoint(
        checkpoint_path,
        {
            "dataset_hash": dataset_hash,
            "window_hash": window_hash,
            "fixture_hash": fixture_hash,
            "window_count": len(selected),
            "parameter_count": parameter_count,
        },
    )
    if not readback["passed"]:
        raise RuntimeError("F4G oracle checkpoint readback failed")
    summary = {
        "format_version": "sonic_posterior_direct_output_t64_oracle_summary_v1",
        "experiment": "F4G-O",
        "execution_pass": True,
        "smoke": False,
        "oracle_target_copy": True,
        "quality_gate_applicable": True,
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
        "completed_optimizer_steps": 0,
        "evaluation_repetitions": 3,
        "deterministic_metric_repetition": deterministic,
        "metric_fingerprints": metric_hashes,
        "initialization": initialization,
        "best_fit_score": float(best["fit_gate"]["score"]),
        "best_evaluation": best,
        "last_three_evaluations": evaluations,
        "checkpoint_readback": readback,
        "plots": plots,
        "scope": "analytic loss/Mask/evaluator ceiling only; no optimization, encoder, latent, decoder, prior, or generalization claim",
        "unique_next_step": direct_output_next_step(quality_pass),
    }
    atomic_write_json(output_run / "manifests/posterior_direct_output_summary.json", summary)
    atomic_write_text(output_run / "markers/cvae_posterior_direct_output_execution.ok", "PASS\n")
    if quality_pass:
        atomic_write_text(output_run / "markers/cvae_posterior_direct_output_oracle.ok", "PASS\n")
        atomic_write_text(output_run / "markers/cvae_posterior_direct_output_fit.ok", "PASS\n")
    else:
        atomic_write_text(
            output_run / "markers/cvae.failed",
            "QUALITY_FAIL execution_complete=true oracle_target_copy_fit=false\n",
        )
    if strict_pass:
        atomic_write_text(output_run / "markers/cvae_posterior_direct_output_strict_memory.ok", "PASS\n")
    if legacy_pass:
        atomic_write_text(output_run / "markers/cvae_posterior_direct_output_legacy_exact.ok", "PASS\n")
    base.close()
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the F4G-O analytic target-copy ceiling")
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs/posterior_direct_output_t64.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_oracle(
        args.dataset_run,
        args.output_run,
        load_config(args.config.resolve()),
    )
    print("Posterior F4G-O direct-output oracle: PASS (execution complete)")
    print(
        json.dumps(
            {
                "output_run": str(args.output_run.expanduser().resolve()),
                "quality_pass": summary["quality_pass"],
                "completed_optimizer_steps": 0,
                "best_fit_score": summary["best_fit_score"],
                "unique_next_step": summary["unique_next_step"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
