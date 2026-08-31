from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .models import PosteriorCapacityOutput, build_model, parameter_count
from .util import (
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    load_json,
    seed_everything,
)


FIXED_MASK_NAMES = (
    "full_state",
    "full_action",
    "full_both",
    "element_both_10",
    "element_both_50",
    "state_time_50",
    "action_time_50",
    "state_feature_50",
    "action_feature_50",
    "semantic_both",
)
JOINT_GROUPS = ((0, 6), (6, 12), (12, 15), (15, 22), (22, 29))


class MaskBankDataset(Dataset[dict[str, Any]]):
    """Repeat every immutable data window once per deterministic mask slot."""

    def __init__(self, base: Dataset[dict[str, Any]], slots: int) -> None:
        if slots <= 0:
            raise ValueError("mask bank must contain at least one slot")
        self.base = base
        self.slots = int(slots)

    def __len__(self) -> int:
        return len(self.base) * self.slots

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_index, slot = divmod(int(index), self.slots)
        item = dict(self.base[base_index])
        item["window_index"] = base_index
        item["mask_slot"] = slot
        return item


class DeterministicWindowSubset(Dataset[dict[str, Any]]):
    """Use the first N fixed windows without changing their underlying identities."""

    def __init__(self, base: Dataset[dict[str, Any]], max_windows: int | None) -> None:
        if max_windows is not None and int(max_windows) <= 0:
            raise ValueError("max_windows must be positive when provided")
        self.base = base
        limit = len(base) if max_windows is None else min(int(max_windows), len(base))
        self.indices = tuple(range(limit))
        if not self.indices:
            raise ValueError("posterior capacity window subset is empty")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = self.indices[int(index)]
        item = dict(self.base[source_index])
        item["source_window_index"] = source_index
        return item


def selected_window_identities(dataset: Any, indices: tuple[int, ...]) -> list[dict[str, Any]]:
    """Describe fixed window refs without opening the underlying HDF5 files."""
    identities: list[dict[str, Any]] = []
    for source_index in indices:
        ref = dataset.refs[source_index]
        if ref.fixed_start is None:
            raise ValueError("posterior capacity requires deterministic fixed window starts")
        episode = dataset.episodes[ref.episode_index]
        identities.append({
            "source_window_index": int(source_index),
            "motion_key": str(episode["motion_key"]),
            "variant_id": int(episode["variant_id"]),
            "episode": str(episode["episode"]),
            "episode_ref": f"{episode['source_run']}::{episode['episode']}",
            "window_start": int(ref.fixed_start),
        })
    return identities


@dataclass(frozen=True)
class ReconstructionLoss:
    total: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    contact: torch.Tensor


def _stable_seed(*values: object) -> int:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def validate_motion_prefix(dataset: Any, motion_count: int) -> list[str]:
    expected_episodes = int(motion_count) * 8
    if len(dataset.episodes) != expected_episodes:
        raise ValueError(
            f"posterior capacity requires exactly {expected_episodes} selected episodes; "
            f"found {len(dataset.episodes)}"
        )
    by_motion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset.episodes:
        by_motion[str(row["motion_key"])].append(row)
    if len(by_motion) != motion_count:
        raise ValueError(
            f"requested {motion_count} complete motions but prefix contains {len(by_motion)}"
        )
    for motion_key, rows in by_motion.items():
        variants = {int(row["variant_id"]) for row in rows}
        if len(rows) != 8 or variants != set(range(8)):
            raise ValueError(f"{motion_key}: expected exactly variants 0..7")
    return sorted(by_motion)


def _identity(batch: dict[str, Any], index: int) -> tuple[object, ...]:
    return (
        batch["motion_key"][index],
        int(batch["variant_id"][index]),
        int(batch["window_start"][index]),
        int(batch.get("window_index", torch.arange(len(batch["motion_key"])))[index]),
    )


def _ensure_target(
    state_mask: torch.Tensor,
    action_mask: torch.Tensor,
    valid_state: torch.Tensor,
    valid_action: torch.Tensor,
) -> None:
    if bool(state_mask.any()) or bool(action_mask.any()):
        return
    locations = torch.nonzero(valid_state, as_tuple=False).flatten()
    if locations.numel():
        state_mask[int(locations[0]), 0] = True
        return
    locations = torch.nonzero(valid_action, as_tuple=False).flatten()
    if locations.numel():
        action_mask[int(locations[0]), 0] = True
        return
    raise ValueError("window contains no valid posterior reconstruction target")


def make_fixture_masks(
    batch: dict[str, Any],
    seed: int,
    *,
    held_out: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Build the ten exact masks or a deterministic held-out extension of them."""
    valid_state = batch["valid_state"].bool()
    valid_action = batch["valid_action"].bool()
    state_mask = torch.zeros_like(batch["physical_state"], dtype=torch.bool)
    action_mask = torch.zeros_like(batch["action"], dtype=torch.bool)
    raw_slots = batch["mask_slot"].tolist()
    names: list[str] = []
    for index, raw_slot in enumerate(raw_slots):
        slot = int(raw_slot) % len(FIXED_MASK_NAMES)
        name = FIXED_MASK_NAMES[slot]
        names.append(name)
        state_valid = valid_state[index]
        action_valid = valid_action[index]
        local_seed = _stable_seed(seed, "heldout" if held_out else "exact", *_identity(batch, index), raw_slot)
        generator = torch.Generator().manual_seed(local_seed)
        state = state_mask[index]
        action = action_mask[index]
        if slot == 0:
            state[:] = state_valid[:, None]
        elif slot == 1:
            action[:] = action_valid[:, None]
        elif slot == 2:
            state[:] = state_valid[:, None]
            action[:] = action_valid[:, None]
        elif slot in {3, 4}:
            fraction = 0.10 if slot == 3 else 0.50
            if held_out:
                fraction = float(torch.empty(()).uniform_(0.01, 1.0, generator=generator))
            state[:] = (torch.rand(state.shape, generator=generator) < fraction) & state_valid[:, None]
            action[:] = (torch.rand(action.shape, generator=generator) < fraction) & action_valid[:, None]
            for target, valid in ((state, state_valid), (action, action_valid)):
                if not bool(target.any()) and bool(valid.any()):
                    target[int(torch.nonzero(valid, as_tuple=False)[0]), 0] = True
        elif slot in {5, 6}:
            valid = state_valid if slot == 5 else action_valid
            target = state if slot == 5 else action
            count = int(valid.sum())
            fraction = 0.50 if not held_out else float(
                torch.empty(()).uniform_(0.01, 1.0, generator=generator)
            )
            length = max(1, min(count, int(math.ceil(count * fraction))))
            start = int(torch.randint(0, count - length + 1, (), generator=generator))
            target[start : start + length] = True
        elif slot in {7, 8}:
            target = state if slot == 7 else action
            valid = state_valid if slot == 7 else action_valid
            width = target.shape[-1]
            fraction = 0.50 if not held_out else float(
                torch.empty(()).uniform_(0.01, 1.0, generator=generator)
            )
            count = max(1, min(width, int(round(width * fraction))))
            dimensions = torch.randperm(width, generator=generator)[:count]
            target[:, dimensions] = valid[:, None]
        else:
            low, high = JOINT_GROUPS[int(torch.randint(0, len(JOINT_GROUPS), (), generator=generator))]
            state[:, low:high] = state_valid[:, None]
            state[:, 29 + low : 29 + high] = state_valid[:, None]
            action[:, low:high] = action_valid[:, None]
        state &= state_valid[:, None]
        action &= action_valid[:, None]
        _ensure_target(state, action, state_valid, action_valid)
    return state_mask, action_mask, names


def make_random_masks(
    batch: dict[str, Any], seed: int, optimizer_step: int
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    varied = dict(batch)
    size = batch["physical_state"].shape[0]
    generator = torch.Generator().manual_seed(_stable_seed(seed, "train", optimizer_step))
    varied["mask_slot"] = torch.randint(0, len(FIXED_MASK_NAMES), (size,), generator=generator)
    varied["window_index"] = torch.arange(size)
    return make_fixture_masks(varied, seed + optimizer_step, held_out=True)


def reconstruction_loss(
    output: PosteriorCapacityOutput,
    batch: dict[str, torch.Tensor],
    state_mask: torch.Tensor,
    action_mask: torch.Tensor,
) -> ReconstructionLoss:
    terms: list[torch.Tensor] = []
    state_continuous_mask = state_mask[..., :68]
    if bool(state_continuous_mask.any()):
        state = F.mse_loss(
            output.physical_state[..., :68], batch["physical_state"][..., :68], reduction="none"
        ).masked_select(state_continuous_mask).mean()
        terms.append(state)
    else:
        state = output.physical_state.sum() * 0.0
    if bool(action_mask.any()):
        action = F.mse_loss(output.action, batch["action"], reduction="none").masked_select(
            action_mask
        ).mean()
        terms.append(action)
    else:
        action = output.action.sum() * 0.0
    contact_mask = state_mask[..., 68:70]
    if bool(contact_mask.any()):
        contact = F.binary_cross_entropy_with_logits(
            output.state_contact_logits,
            batch["physical_state"][..., 68:70],
            reduction="none",
        ).masked_select(contact_mask).mean()
        terms.append(contact)
    else:
        contact = output.state_contact_logits.sum() * 0.0
    if not terms:
        raise ValueError("posterior reconstruction batch contains no targets")
    return ReconstructionLoss(torch.stack(terms).mean(), state, action, contact)


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _infinite(loader: DataLoader[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def _masked_rmse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> float | None:
    if not bool(mask.any()):
        return None
    return float(torch.sqrt(torch.square(prediction - target).masked_select(mask).mean()).cpu())


@torch.no_grad()
def evaluate_exact(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    seed: int,
    held_out: bool,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    model.eval()
    cases: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "windows": 0.0, "worst_state_rmse": 0.0, "worst_action_rmse": 0.0,
            "continuous_max_abs": 0.0, "contact_correct": 0.0, "contact_count": 0.0,
        }
    )
    worst_state = worst_action = worst_max = 0.0
    contact_correct = contact_count = 0
    latent_sse = {"correct": 0.0, "zero": 0.0, "swapped": 0.0}
    latent_count = 0
    for cpu_batch in loader:
        state_mask, action_mask, names = make_fixture_masks(cpu_batch, seed, held_out=held_out)
        batch = _device_batch(cpu_batch, device)
        state_mask = state_mask.to(device)
        action_mask = action_mask.to(device)
        output = model(batch, state_mask, action_mask)
        full_both = torch.tensor([name == "full_both" for name in names], device=device)
        zero_output = swapped_output = None
        if bool(full_both.any()):
            zero_output = model(
                batch, state_mask, action_mask,
                latent_override=torch.zeros_like(output.posterior_mean),
            )
            swapped_output = model(
                batch, state_mask, action_mask,
                latent_override=output.posterior_mean.flip(0),
            )
        for index, name in enumerate(names):
            state_continuous_mask = state_mask[index, :, :68]
            action_target_mask = action_mask[index]
            state_rmse = _masked_rmse(
                output.physical_state[index, :, :68],
                batch["physical_state"][index, :, :68],
                state_continuous_mask,
            )
            action_rmse = _masked_rmse(
                output.action[index], batch["action"][index], action_target_mask
            )
            errors = []
            if bool(state_continuous_mask.any()):
                errors.append(torch.abs(
                    output.physical_state[index, :, :68] - batch["physical_state"][index, :, :68]
                ).masked_select(state_continuous_mask).max())
            if bool(action_target_mask.any()):
                errors.append(torch.abs(output.action[index] - batch["action"][index]).masked_select(
                    action_target_mask
                ).max())
            maximum = float(torch.stack(errors).max().cpu()) if errors else 0.0
            contact_mask = state_mask[index, :, 68:70]
            correct = count = 0
            if bool(contact_mask.any()):
                predicted = output.state_contact_logits[index] >= 0
                target = batch["physical_state"][index, :, 68:70] >= 0.5
                correct = int((predicted == target).masked_select(contact_mask).sum())
                count = int(contact_mask.sum())
            item = cases[name]
            item["windows"] += 1
            item["worst_state_rmse"] = max(item["worst_state_rmse"], state_rmse or 0.0)
            item["worst_action_rmse"] = max(item["worst_action_rmse"], action_rmse or 0.0)
            item["continuous_max_abs"] = max(item["continuous_max_abs"], maximum)
            item["contact_correct"] += correct
            item["contact_count"] += count
            worst_state = max(worst_state, state_rmse or 0.0)
            worst_action = max(worst_action, action_rmse or 0.0)
            worst_max = max(worst_max, maximum)
            contact_correct += correct
            contact_count += count
            if name == "full_both":
                assert zero_output is not None and swapped_output is not None
                target_parts = (batch["physical_state"][index], batch["action"][index])
                mask_parts = (state_mask[index], action_mask[index])
                for label, candidate in (
                    ("correct", output), ("zero", zero_output), ("swapped", swapped_output)
                ):
                    prediction_parts = (candidate.physical_state[index], candidate.action[index])
                    for prediction, target, mask in zip(prediction_parts, target_parts, mask_parts):
                        latent_sse[label] += float(
                            torch.square(prediction - target).masked_select(mask).sum().cpu()
                        )
                latent_count += int(state_mask[index].sum() + action_mask[index].sum())
    if not cases:
        raise ValueError("exact posterior evaluation produced no fixtures")
    for item in cases.values():
        item["contact_accuracy"] = (
            item["contact_correct"] / item["contact_count"] if item["contact_count"] else 1.0
        )
    correct_rmse = math.sqrt(latent_sse["correct"] / max(latent_count, 1))
    zero_rmse = math.sqrt(latent_sse["zero"] / max(latent_count, 1))
    swapped_rmse = math.sqrt(latent_sse["swapped"] / max(latent_count, 1))
    zero_ratio = zero_rmse / max(correct_rmse, 1e-12)
    swapped_ratio = swapped_rmse / max(correct_rmse, 1e-12)
    contact_accuracy = contact_correct / contact_count if contact_count else 1.0
    ratios = {
        "state_rmse": worst_state / thresholds["state_rmse"],
        "action_rmse": worst_action / thresholds["action_rmse"],
        "continuous_max_abs": worst_max / thresholds["continuous_max_abs"],
        "contact_accuracy": 2.0 - contact_accuracy,
        "zero_latent_dependence": thresholds["latent_ratio"] / max(zero_ratio, 1e-12),
    }
    score = max(ratios.values())
    return {
        "passed": bool(math.isfinite(score) and score <= 1.0),
        "score": score,
        "threshold_ratios": ratios,
        "worst_state_rmse": worst_state,
        "worst_action_rmse": worst_action,
        "continuous_max_abs": worst_max,
        "contact_accuracy": contact_accuracy,
        "latent_dependence": {
            "correct_rmse": correct_rmse,
            "zero_rmse": zero_rmse,
            "swapped_rmse": swapped_rmse,
            "zero_ratio": zero_ratio,
            "swapped_ratio": swapped_ratio,
        },
        "cases": dict(cases),
    }


def _checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    step: int,
    score: float,
    dataset_hash: str,
) -> dict[str, Any]:
    return {
        "format_version": "sonic_posterior_capacity_checkpoint_v1",
        "step": step,
        "best_score": score,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "dataset_manifest_sha256": dataset_hash,
        "parameter_count": parameter_count(model),
    }


def run_experiment(
    dataset_run: Path,
    output_run: Path,
    config: dict[str, Any],
    *,
    init_checkpoint: Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    from .dataset import StateActionWindowDataset

    dataset_run = dataset_run.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "videos"):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    if not (dataset_run / "markers/cvae_overfit_subset.ok").is_file():
        raise FileNotFoundError("posterior capacity requires the dedicated overfit subset marker")
    seed = int(config["seed"])
    seed_everything(seed)
    data = config["data"]
    training = config["training"]
    motion_count = int(data["motion_count"])
    window = int(data["window_transitions"])
    configured_max_windows = data.get("max_windows")
    max_windows = (
        None if configured_max_windows is None else int(configured_max_windows)
    )
    if max_windows is not None and max_windows <= 0:
        raise ValueError("data.max_windows must be positive when provided")
    phase = str(training.get("mask_phase", "fixed"))
    if phase not in {"fixed", "generalization"}:
        raise ValueError("mask_phase must be fixed or generalization")
    base = StateActionWindowDataset(
        dataset_run,
        "train",
        window,
        window,
        max_episodes=motion_count * 8,
        random_crop=False,
    )
    selected_motions = validate_motion_prefix(base, motion_count)
    config["model"]["state_dim"] = base.state_dim
    if config["model"].get("kind") != "physics_posterior_transformer":
        raise ValueError("posterior capacity requires physics_posterior_transformer")
    model = build_model(config["model"])
    count = parameter_count(model)
    lower, upper = config["model"].get("parameter_count_range", (6_500_000, 12_000_000))
    if not int(lower) <= count <= int(upper):
        raise ValueError(f"posterior capacity parameter count {count} is outside [{lower}, {upper}]")
    dataset_hash = file_sha256(dataset_run / "manifests/dataset_manifest.json")
    if phase == "generalization" and init_checkpoint is None:
        raise ValueError("generalization phase requires a fixed-mask best_exact checkpoint")
    if phase == "fixed" and init_checkpoint is not None:
        raise ValueError("fixed posterior capacity must start from random initialization")
    if init_checkpoint is not None:
        checkpoint = torch.load(init_checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != "sonic_posterior_capacity_checkpoint_v1":
            raise ValueError("unsupported posterior capacity initialization checkpoint")
        if checkpoint.get("dataset_manifest_sha256") != dataset_hash:
            raise ValueError("posterior checkpoint dataset manifest hash mismatch")
        checkpoint_config = checkpoint.get("config", {})
        if checkpoint_config.get("training", {}).get("mask_phase") != "fixed":
            raise ValueError("generalization must initialize from a fixed-mask checkpoint")
        checkpoint_data = checkpoint_config.get("data", {})
        checkpoint_max_windows = checkpoint_data.get("max_windows")
        checkpoint_max_windows = (
            None if checkpoint_max_windows is None else int(checkpoint_max_windows)
        )
        if (
            int(checkpoint_data.get("motion_count", -1)) != motion_count
            or int(checkpoint_data.get("window_transitions", -1)) != window
            or checkpoint_max_windows != max_windows
        ):
            raise ValueError(
                "generalization must keep the checkpoint motion count, window length, "
                "and deterministic window subset"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    fixed_slots = len(FIXED_MASK_NAMES)
    selected_base = DeterministicWindowSubset(base, max_windows)
    selected_windows = selected_window_identities(base, selected_base.indices)
    train_data: Dataset[dict[str, Any]] = (
        MaskBankDataset(selected_base, fixed_slots) if phase == "fixed" else selected_base
    )
    validation_data = MaskBankDataset(
        selected_base, fixed_slots if phase == "fixed" else 16
    )
    workers = 0 if smoke else int(data.get("num_workers", 4))
    generator = torch.Generator().manual_seed(seed)
    micro_batch = int(training["micro_batch"])
    train_loader = DataLoader(
        train_data, batch_size=micro_batch, shuffle=True, num_workers=workers,
        generator=generator, drop_last=False, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=micro_batch, shuffle=False, num_workers=workers,
        drop_last=False, pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=0.0
    )
    max_steps = 2 if smoke else int(training["max_optimizer_steps"])
    warmup = int(training["warmup_steps"])
    minimum_ratio = float(training.get("min_lr_ratio", 1.0 / 300.0))

    def lr_multiplier(step: int) -> float:
        if step < warmup:
            return max((step + 1) / max(warmup, 1), 1e-8)
        progress = min((step - warmup) / max(max_steps - warmup, 1), 1.0)
        return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    accumulation = int(training["gradient_accumulation"])
    validation_interval = 1 if smoke else int(training["validation_interval"])
    thresholds = {key: float(value) for key, value in training["thresholds"].items()}
    stream = _infinite(train_loader)
    metrics_path = output_run / "logs/metrics.jsonl"
    best_score = math.inf
    best_metrics: dict[str, Any] | None = None
    pass_streak = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for optimizer_step in range(1, max_steps + 1):
        accumulated = {"total": 0.0, "state": 0.0, "action": 0.0, "contact": 0.0}
        for _ in range(accumulation):
            cpu_batch = next(stream)
            if phase == "fixed":
                state_mask, action_mask, _ = make_fixture_masks(cpu_batch, seed)
            else:
                state_mask, action_mask, _ = make_random_masks(cpu_batch, seed, optimizer_step)
            batch = _device_batch(cpu_batch, device)
            state_mask = state_mask.to(device)
            action_mask = action_mask.to(device)
            output = model(batch, state_mask, action_mask)
            loss = reconstruction_loss(output, batch, state_mask, action_mask)
            (loss.total / accumulation).backward()
            for key in accumulated:
                accumulated[key] += float(getattr(loss, key).detach().cpu()) / accumulation
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip"])
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        record = {
            "phase": "train", "optimizer_step": optimizer_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm": float(gradient_norm), **accumulated,
        }
        with metrics_path.open("a", encoding="utf-8") as stream_file:
            stream_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if optimizer_step % validation_interval != 0 and optimizer_step != max_steps:
            continue
        metrics = evaluate_exact(
            model, validation_loader, device, seed + 700_001,
            held_out=phase == "generalization", thresholds=thresholds,
        )
        metrics["optimizer_step"] = optimizer_step
        with metrics_path.open("a", encoding="utf-8") as stream_file:
            stream_file.write(json.dumps({"phase": "validation", **metrics}, ensure_ascii=False) + "\n")
        score = float(metrics["score"])
        if score < best_score:
            best_score = score
            best_metrics = metrics
            atomic_torch_save(
                output_run / "checkpoints/best_exact.pt",
                _checkpoint(model, optimizer, scheduler, config, optimizer_step, score, dataset_hash),
            )
        atomic_torch_save(
            output_run / "checkpoints/last.pt",
            _checkpoint(model, optimizer, scheduler, config, optimizer_step, best_score, dataset_hash),
        )
        pass_streak = pass_streak + 1 if metrics["passed"] else 0
        if not smoke and pass_streak >= int(training.get("required_pass_streak", 3)):
            break
        model.train()
    passed = bool(best_metrics and best_metrics["passed"] and pass_streak >= int(
        training.get("required_pass_streak", 3)
    ))
    summary = {
        "format_version": "sonic_posterior_capacity_summary_v1",
        "scope": "posterior memorization only; no conditional-direction claim",
        "mask_phase": phase,
        "smoke": smoke,
        "passed": passed if not smoke else True,
        "motion_count": motion_count,
        "selected_motion_keys": selected_motions,
        "episode_count": len(base.episodes),
        "available_window_count": len(base),
        "max_windows": max_windows,
        "window_count": len(selected_base),
        "selected_windows": selected_windows,
        "window_transitions": window,
        "mask_fixture_count": len(validation_data),
        "parameter_count": count,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "checkpoint": str(output_run / "checkpoints/best_exact.pt"),
    }
    atomic_write_json(output_run / "manifests/posterior_capacity_summary.json", summary)
    if smoke:
        atomic_write_text(output_run / "markers/cvae_posterior_capacity_smoke.ok", "PASS\n")
    elif passed:
        marker = (
            "cvae_posterior_capacity.ok" if phase == "fixed"
            else "cvae_posterior_mask_generalization.ok"
        )
        atomic_write_text(output_run / "markers" / marker, "PASS\n")
    base.close()
    if not smoke and not passed:
        raise RuntimeError("posterior capacity experiment did not satisfy every exact gate")
    return summary


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the minimal posterior Transformer capacity experiment")
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=project_root / "configs/posterior_capacity_minimal.json",
    )
    parser.add_argument("--motions", type=int)
    parser.add_argument("--window-transitions", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--mask-phase", choices=("fixed", "generalization"))
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.config.resolve())
    if args.motions is not None:
        config["data"]["motion_count"] = args.motions
    if args.window_transitions is not None:
        config["data"]["window_transitions"] = args.window_transitions
    if args.max_windows is not None:
        config["data"]["max_windows"] = args.max_windows
    if args.mask_phase is not None:
        config["training"]["mask_phase"] = args.mask_phase
    if args.seed is not None:
        config["seed"] = args.seed
    window = int(config["data"]["window_transitions"])
    if window <= 32:
        micro_batch, accumulation = 16, 4
    elif window <= 64:
        micro_batch, accumulation = 8, 8
    else:
        micro_batch, accumulation = 4, 16
    config["training"]["micro_batch"] = micro_batch
    config["training"]["gradient_accumulation"] = accumulation
    summary = run_experiment(
        args.dataset_run, args.output_run, config,
        init_checkpoint=args.init_checkpoint, smoke=args.smoke,
    )
    print("Posterior capacity experiment: PASS")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
