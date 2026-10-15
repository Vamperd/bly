"""Offline 192-step evaluation for the two contiguous T64 inference methods.

This module deliberately keeps the simulator/replay worker separate.  It
produces sealed normalized and canonical Action predictions plus paired block
metrics that can be consumed by the existing replay/render pipeline.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate

from .action_masks import relative_to_raw_action
from .cvae_diagnostics import epsilon_for, route_output
from .cvae_protocol import digest, isolated_rng
from .long_horizon_dataset import LongHorizonWindowDataset
from .models import build_model
from .posterior_hierarchical_standard_cvae import load_checkpoint
from .util import atomic_write_json, atomic_write_text, file_sha256


VERSION = "sonic_long_horizon_inference_v1"
CHUNK = 64
BLOCK = 32
HORIZON = 192


def _finite(name: str, value: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    value = np.asarray(value)
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} shape {value.shape} != {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return value


def _row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["segment_group_id"]), int(row["group_chunk_index"])


def _metrics(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    truth = _finite("truth", truth)
    pred = _finite("prediction", pred)
    mask = np.asarray(mask, dtype=bool)
    if truth.shape != pred.shape or mask.shape != truth.shape:
        raise ValueError("metric arrays must have identical shapes")
    values = (pred - truth)[mask]
    if values.size == 0:
        return {"count": 0, "rmse": None, "mae": None, "p95": None, "max_abs": None}
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "mae": float(np.mean(absolute)),
        "p95": float(np.quantile(absolute, 0.95)),
        "max_abs": float(np.max(absolute)),
    }


def _mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(mask, dtype=np.bool_).tobytes(order="C")).hexdigest()


class LongHorizonEvaluator:
    def __init__(self, dataset_run: Path, checkpoint_path: Path, device: str | None = None,
                 split: str = "train") -> None:
        self.dataset_run = dataset_run.expanduser().resolve()
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        saved = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        config = saved.get("config") or {}
        model_config = config.get("model") or saved.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError("checkpoint does not contain a model configuration")
        self.model = build_model(model_config)
        load_checkpoint(self.model, self.checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.dataset = LongHorizonWindowDataset(self.dataset_run, split, include_tail=True)
        self._groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, row in enumerate(self.dataset.windows):
            self._groups[str(row["segment_group_id"])].append((index, row))
        for rows in self._groups.values():
            rows.sort(key=lambda item: int(item[1]["group_chunk_index"]))
        self.complete_group_ids = {
            group_id for group_id, rows in self._groups.items()
            if len(rows) == HORIZON // CHUNK and all(int(row["valid_steps"]) == CHUNK for _, row in rows)
        }

    def close(self) -> None:
        self.dataset.close()

    def _epsilon(self, batch: dict[str, Any], seed: int, sample_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return epsilon_for(self.model, batch, sample_index, seed, self.device)

    def _predict(self, sample: dict[str, Any], state_mask: torch.Tensor, action_mask: torch.Tensor,
                 *, seed: int, sample_index: int) -> tuple[np.ndarray, np.ndarray]:
        batch = default_collate([sample])
        batch["stable_window_id"] = [digest((sample["episode_ref"], int(sample["window_start"]), sample.get("segment_group_id", "")))]
        batch["window_index"] = torch.tensor([0], dtype=torch.long)
        batch["mask_slot"] = torch.tensor([0], dtype=torch.long)
        batch_device = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()}
        state_mask = state_mask.to(self.device)
        action_mask = action_mask.to(self.device)
        batch_device["physical_state"] = batch_device["physical_state"].masked_fill(state_mask, 0.0)
        batch_device["action"] = batch_device["action"].masked_fill(action_mask, 0.0)
        epsilon = self._epsilon(batch, seed, sample_index)
        with isolated_rng(), torch.inference_mode():
            output = route_output(self.model, batch_device, state_mask, action_mask,
                                  "standard_normal", epsilon)
        state = output.physical_state[0].detach().float().cpu().numpy().copy()
        state[:, 68:] = output.state_contact_logits[0].sigmoid().detach().float().cpu().numpy()
        action = output.action[0].detach().float().cpu().numpy().copy()
        return _finite("predicted state", state, (65, 70)), _finite("predicted action", action, (64, 29))

    def method_a(self, group_id: str, *, seed: int = 20261015) -> dict[str, Any]:
        rows = self._groups[group_id]
        if len(rows) != 3:
            raise ValueError("method A requires a complete three-chunk group")
        truth_actions: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        truth_states: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for index, row in rows:
            sample = self.dataset[index]
            valid_state = sample["valid_state"].numpy().astype(bool)
            valid_action = sample["valid_action"].numpy().astype(bool)
            sm = torch.zeros_like(sample["physical_state"], dtype=torch.bool)[None]
            am = torch.zeros_like(sample["action"], dtype=torch.bool)[None]
            am[0] = torch.from_numpy(valid_action[:, None]).expand_as(am[0])
            _, prediction = self._predict(sample, sm, am, seed=seed, sample_index=int(row["group_chunk_index"]))
            truth_states.append(sample["physical_state"].numpy())
            truth_actions.append(sample["action"].numpy())
            predictions.append(prediction)
            masks.append(np.broadcast_to(valid_action[:, None], prediction.shape).copy())
        state = np.concatenate([part[:-1] for part in truth_states] + [truth_states[-1][-1:]], axis=0)
        truth = np.concatenate(truth_actions, axis=0)
        prediction = np.concatenate(predictions, axis=0)
        mask = np.concatenate(masks, axis=0)
        return {"method": "full_action_t64", "truth_state": state, "truth_action": truth,
                "predicted_action": prediction, "action_mask": mask,
                "blocks": [_metrics(truth[i:i + BLOCK], prediction[i:i + BLOCK], mask[i:i + BLOCK])
                            for i in range(0, len(truth), BLOCK)]}

    def method_b(self, group_id: str, *, seed: int = 20261015) -> dict[str, Any]:
        rows = self._groups[group_id]
        first = self.dataset[rows[0][0]]
        episode_index = int(rows[0][1]["episode_index"])
        group_start = int(rows[0][1]["window_start"])
        group_valid = int(rows[0][1]["group_valid_steps"])
        truth = np.zeros((HORIZON, 29), dtype=np.float32)
        prediction = np.zeros_like(truth)
        mask = np.zeros_like(truth, dtype=bool)
        block_rows: list[dict[str, Any]] = []
        for block_start in range(0, HORIZON, BLOCK):
            absolute_target = group_start + block_start
            context_start = absolute_target - BLOCK
            valid_target = max(0, min(BLOCK, group_valid - block_start))
            if valid_target <= 0:
                continue
            if context_start < 0:
                block_rows.append({"block_start": block_start, "context_available": False,
                                   "status": "warmup_unavailable", "metrics": None})
                continue
            sample = self.dataset.source.get_window(episode_index, context_start)
            target_truth = sample["action"].numpy()[BLOCK:BLOCK + valid_target]
            sm = torch.zeros_like(sample["physical_state"], dtype=torch.bool)[None]
            am = torch.zeros_like(sample["action"], dtype=torch.bool)[None]
            am[0, BLOCK:BLOCK + valid_target] = True
            _, output = self._predict(sample, sm, am, seed=seed, sample_index=block_start // BLOCK)
            target_prediction = output[BLOCK:BLOCK + valid_target]
            truth[block_start:block_start + valid_target] = target_truth
            prediction[block_start:block_start + valid_target] = target_prediction
            mask[block_start:block_start + valid_target] = True
            block_rows.append({"block_start": block_start, "context_available": True,
                               "status": "scored", "metrics": _metrics(target_truth, target_prediction,
                                                                            np.ones_like(target_truth, dtype=bool))})
        return {"method": "tail32_context_next32", "truth_state": first["physical_state"].numpy(),
                "truth_action": truth, "predicted_action": prediction, "action_mask": mask,
                "blocks": block_rows}

    def evaluate_group(self, group_id: str, *, seed: int = 20261015) -> dict[str, Any]:
        a = self.method_a(group_id, seed=seed)
        b = self.method_b(group_id, seed=seed)
        episode_index = int(self._groups[group_id][0][1]["episode_index"])
        mapping = self.dataset.source.physics_action_mapping(episode_index)
        action_mean = self.dataset.source.action_mean
        action_std = self.dataset.source.action_std
        canonical = {
            "truth": a["truth_action"] * action_std + action_mean,
            "a": a["predicted_action"] * action_std + action_mean,
            "b": b["predicted_action"] * action_std + action_mean,
        }
        raw = {}
        achieved = {}
        for key, value in canonical.items():
            raw[key], achieved[key], _ = relative_to_raw_action(
                value, mapping["default"], mapping["scale"], mapping["offset"],
                mapping["clip"], mapping["wrapper_clip"],
            )
        result = {
            "version": VERSION,
            "segment_group_id": group_id,
            "dataset_run": str(self.dataset_run),
            "checkpoint": str(self.checkpoint_path),
            "methods": {},
        }
        for name, value in (("full_action_t64", a), ("tail32_context_next32", b)):
            result["methods"][name] = {
                "action_metrics": _metrics(value["truth_action"], value["predicted_action"], value["action_mask"]),
                "blocks": value["blocks"],
                "warmup_blocks": sum(block.get("status") == "warmup_unavailable" for block in value["blocks"]),
                "action_mask_sha256": _mask_hash(value["action_mask"]),
                "masked_action_elements": int(value["action_mask"].sum()),
                "visible_action_elements": int((~value["action_mask"]).sum()),
            }
        return result, a, b, canonical, raw, achieved


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_run.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output run must be new and empty: {output}")
    for folder in ("data", "manifests", "markers", "plots", "videos", "logs"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    evaluator = LongHorizonEvaluator(args.dataset_run, args.checkpoint, args.device, args.split)
    try:
        group_ids = sorted(evaluator.complete_group_ids)
        if args.group_id:
            if args.group_id not in evaluator.complete_group_ids:
                raise ValueError("--group-id must identify a complete 192-step group")
            group_ids = [args.group_id]
        if args.limit_groups:
            group_ids = group_ids[:args.limit_groups]
        if not group_ids:
            raise ValueError("no groups selected")
        rows = []
        for index, group_id in enumerate(group_ids):
            summary, method_a, method_b, canonical, raw, achieved = evaluator.evaluate_group(group_id, seed=args.seed)
            rows.append(summary)
            np.savez_compressed(
                output / "data" / f"{index:06d}.npz",
                group_id=np.asarray(group_id),
                truth_state=method_a["truth_state"],
                truth_action=method_a["truth_action"],
                action_a=method_a["predicted_action"],
                action_b=method_b["predicted_action"],
                truth_action_canonical=canonical["truth"].astype(np.float32),
                action_a_canonical=canonical["a"].astype(np.float32),
                action_b_canonical=canonical["b"].astype(np.float32),
                truth_action_raw=raw["truth"].astype(np.float32),
                action_a_raw=raw["a"].astype(np.float32),
                action_b_raw=raw["b"].astype(np.float32),
                truth_action_achieved=achieved["truth"].astype(np.float32),
                action_a_achieved=achieved["a"].astype(np.float32),
                action_b_achieved=achieved["b"].astype(np.float32),
                mask_a=method_a["action_mask"],
                mask_b=method_b["action_mask"],
            )
        atomic_write_json(output / "manifests" / "long_horizon_eval.json", {
            "version": VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_run": str(evaluator.dataset_run), "checkpoint": str(evaluator.checkpoint_path),
            "split": args.split,
            "group_ids": group_ids, "seed": args.seed, "results": rows,
            "quality_pass": None,
        })
        atomic_write_text(output / "markers" / "cvae_long_horizon_execution.ok", "PASS\n")
        return {"output_run": str(output), "group_count": len(rows), "quality_pass": None}
    finally:
        evaluator.close()


def render_action_video(run: Path, output_name: str = "comparison_grid.mp4", output_dir: Path | None = None) -> Path:
    """Render an offline Action/error comparison video when matplotlib is available."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    arrays = sorted((run / "data").glob("*.npz"))
    if not arrays:
        raise FileNotFoundError("evaluation has no NPZ group outputs")
    with np.load(arrays[0], allow_pickle=False) as values:
        truth = values["truth_action"]
        action_a = values["action_a"]
        action_b = values["action_b"]
    frames = truth.shape[0] + 1
    output = (output_dir or (run / "videos")) / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    writer = FFMpegWriter(fps=50, metadata={"title": VERSION})
    with writer.saving(fig, str(output), dpi=100):
        for frame in range(frames):
            axes[0].clear(); axes[1].clear()
            stop = min(frame, truth.shape[0])
            x = np.arange(stop)
            if stop:
                axes[0].plot(x, truth[:stop, 0], label="truth action[0]", color="black")
                axes[0].plot(x, action_a[:stop, 0], label="method A", color="tab:blue")
                axes[0].plot(x, action_b[:stop, 0], label="method B", color="tab:orange")
                axes[1].plot(x, np.sqrt(np.mean((action_a[:stop] - truth[:stop]) ** 2, axis=1)), label="A RMSE")
                axes[1].plot(x, np.sqrt(np.mean((action_b[:stop] - truth[:stop]) ** 2, axis=1)), label="B RMSE")
            for boundary in (32, 64, 96, 128, 160):
                axes[0].axvline(boundary, color="gray", alpha=.3)
                axes[1].axvline(boundary, color="gray", alpha=.3)
            axes[0].set_title(f"{VERSION} | frame {frame}/{truth.shape[0]}")
            axes[0].legend(loc="upper right")
            axes[1].set_xlabel("transition"); axes[1].set_ylabel("Action RMSE")
            axes[1].legend(loc="upper right")
            writer.grab_frame()
    return output


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--group-id")
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--device")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_evaluation(args)
    if args.render:
        render_action_video(args.output_run.expanduser().resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
