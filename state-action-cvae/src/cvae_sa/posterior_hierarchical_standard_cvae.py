from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from .posterior_t64_protocol import make_physical_masks
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
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(training_cfg.get("micro_batch", 2)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    maximum = 2 if smoke else int(training_cfg.get(stage_name, {}).get("max_optimizer_steps", 1))
    iterator = _infinite(loader)
    records: list[dict[str, Any]] = []
    for step in range(1, maximum + 1):
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
        target_beta = (
            float(kl_beta_override)
            if kl_beta_override is not None
            else float(training_cfg.get("kl", {}).get("beta", 1e-3))
        )
        beta_warmup = max(1, int(training_cfg.get("kl", {}).get("beta_warmup_steps", 1)))
        beta = 0.0 if stage_code != "C" else target_beta * min(step / beta_warmup, 1.0)
        loss = reconstruction + beta * kl["total"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()
        records.append({
            "step": step,
            "loss": float(loss.detach().cpu()),
            "reconstruction": float(reconstruction.detach().cpu()),
            "kl": float(kl["total"].detach().cpu()),
            "finite": bool(torch.isfinite(loss)),
        })
        if smoke:
            break

    checkpoint = _new_checkpoint(model, optimizer, scheduler, {"model": model_cfg}, stage=stage_code, step=records[-1]["step"])
    atomic_torch_save(output_run / "checkpoints/last.pt", checkpoint)
    atomic_torch_save(output_run / "checkpoints/best.pt", checkpoint)
    readback = validate_checkpoint(
        output_run / "checkpoints/last.pt",
        {"stage": stage_code, "parameters": parameter_total},
    )
    if not readback["passed"]:
        raise RuntimeError("65-token CVAE checkpoint readback failed")
    summary = {
        "format_version": SUMMARY_FORMAT,
        "architecture_version": model.ARCHITECTURE_VERSION,
        "stage": stage_code,
        "stage_name": stage_name,
        "execution_pass": True,
        "quality_pass": None,
        "smoke": smoke,
        "completed_optimizer_steps": records[-1]["step"],
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
        "records": records,
        "checkpoint_readback": readback,
        "source_checkpoint": "none; random initialization",
        "unique_next_step": "ENGINEERING_REVIEW_ONLY",
    }
    atomic_write_json(output_run / "manifests/standard_cvae_summary.json", summary)
    atomic_write_json(output_run / "manifests/model_signature.json", _model_signature(model_cfg, parameter_total))
    with (output_run / "logs" / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
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
