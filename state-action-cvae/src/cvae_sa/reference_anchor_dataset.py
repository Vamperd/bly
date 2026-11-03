"""Physics v5 windows whose future States 33..64 hold the reference motion.

The derived run copies a long-horizon manifest (physical T64 windows) and adds
reference anchors laid out exactly like the DAgger rollout condition
(``dagger.runtime.cvae_window.build_condition``): left-padded history in
slots ``[32-h, 32)``, the current State at slot 32, and reference frames
``motion_step[t]+1..+32`` in the continuous dims of slots 33..64.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .cvae_protocol import digest
from .long_horizon_dataset import VERSION as LONG_HORIZON_VERSION, _jsonl, _write_jsonl
from .posterior_capacity import _stable_seed
from .util import atomic_write_json, atomic_write_text, file_sha256, load_json


VERSION = "cvae_v5_reference_mixed_dataset_v1"
REFERENCE_TABLE_VERSION = "cvae_reference_table_v1"
PHYSICS_V5 = "sonic_physics_state_action_cvae_dataset_v5"
ANCHOR = 32
FUTURE = 32
STATE_STEPS = 65
ACTION_STEPS = 64
CONTINUOUS = 68
MARKER = "cvae_reference_mixed_dataset.ok"
LONG_HORIZON_COPIES = (
    "dataset_manifest.json", "long_horizon_groups.jsonl", "long_horizon_windows.jsonl",
    "episodes.jsonl", "source_hashes.json", "coverage.json",
)
STATE_ALIGNED = ("physical_state", "previous_action", "progress", "valid_state")
ACTION_ALIGNED = ("action", "auxiliary_transition", "reference_future", "valid_action")


def _prepare_output(output_run: Path, allow_precreated_output: bool) -> None:
    folders = ("data", "manifests", "markers", "logs", "checkpoints", "videos")
    if output_run.exists():
        if not allow_precreated_output:
            raise FileExistsError(f"output run must be new and empty: {output_run}")
        # cvae_repro.sh precreates the run, provenance files, and the build log.
        allowed = {"logs/build_reference_mixed_dataset.log"} | {
            f"manifests/{name}" for name in (
                "command.txt", "source_commit.txt", "source_status.txt", "sonic_commit.txt",
                "sonic_status.txt", "isaaclab_commit.txt", "isaaclab_status.txt",
                "environment_freeze.txt", "nvidia-smi.txt")}
        if any(entry.name not in folders or not entry.is_dir() for entry in output_run.iterdir()):
            raise FileExistsError(f"output run must be new and empty: {output_run}")
        for path in output_run.rglob("*"):
            if path.is_file() and path.relative_to(output_run).as_posix() not in allowed:
                raise FileExistsError(f"output run must be new and empty: {output_run}")
    else:
        output_run.mkdir(parents=True, exist_ok=False)
    for folder in folders:
        (output_run / folder).mkdir(parents=True, exist_ok=True)


def history_steps(anchor_step: int, anchor_id: str, *, seed: int, truncation_probability: float) -> tuple[int, int, bool]:
    """Natural history min(t, 32), deterministically truncated to U[0, h]."""
    natural = min(int(anchor_step), ANCHOR)
    rng = np.random.default_rng(_stable_seed(seed, "history_truncation", anchor_id))
    if rng.random() < truncation_probability:
        return natural, int(rng.integers(0, natural + 1)), True
    return natural, natural, False


def normalization_difference(current: Path, other: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"current": str(current), "other": str(other)}
    with np.load(current) as a, np.load(other) as b:
        for prefix in ("physical_state", "action"):
            mean_a, std_a = a[f"{prefix}_mean"].astype(np.float64), a[f"{prefix}_std"].astype(np.float64)
            mean_b, std_b = b[f"{prefix}_mean"].astype(np.float64), b[f"{prefix}_std"].astype(np.float64)
            shift = np.abs(mean_a - mean_b) / np.maximum(std_b, 1e-6)
            ratio = std_a / np.maximum(std_b, 1e-6)
            report[prefix] = {
                "max_mean_shift_in_other_std": float(shift.max()),
                "argmax_mean_shift": int(shift.argmax()),
                "std_ratio_min": float(ratio.min()),
                "std_ratio_max": float(ratio.max()),
            }
    return report


def build_reference_mixed_dataset(
    long_horizon_run: Path,
    reference_table_run: Path,
    output_run: Path,
    *,
    anchor_stride: int = 4,
    history_truncation_probability: float = 0.25,
    seed: int = 20261102,
    compare_normalization: Path | None = None,
    allow_precreated_output: bool = False,
) -> dict[str, Any]:
    import h5py

    if anchor_stride < 1 or not 0.0 <= history_truncation_probability <= 1.0:
        raise ValueError("anchor_stride must be positive and truncation probability in [0, 1]")
    long_horizon_run = long_horizon_run.expanduser().resolve()
    reference_table_run = reference_table_run.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    for marker in (long_horizon_run / "markers/cvae_long_horizon_dataset.ok",
                   reference_table_run / "markers/cvae_reference_table.ok"):
        if not marker.is_file():
            raise FileNotFoundError(marker)
    long_manifest = load_json(long_horizon_run / "manifests/dataset_manifest.json")
    if long_manifest.get("schema_version") != LONG_HORIZON_VERSION:
        raise ValueError("long_horizon_run is not a long-horizon dataset")
    physics_run = Path(long_manifest["source_dataset_run"])
    physics_manifest_path = physics_run / "manifests/dataset_manifest.json"
    if load_json(physics_manifest_path).get("schema_version") != PHYSICS_V5:
        raise ValueError("reference anchors require a Physics v5 source dataset")
    table_manifest = load_json(reference_table_run / "manifests/reference_table_manifest.json")
    if table_manifest.get("schema_version") != REFERENCE_TABLE_VERSION:
        raise ValueError("unsupported reference table manifest")
    if not table_manifest.get("alignment", {}).get("pass"):
        raise ValueError("reference table alignment check did not pass")
    table_path = reference_table_run / table_manifest["table"]
    if file_sha256(table_path) != table_manifest["table_sha256"]:
        raise ValueError("reference table sha256 does not match its manifest")
    if file_sha256(physics_manifest_path) != table_manifest["physics_dataset_manifest_sha256"]:
        raise ValueError("reference table was exported from a different Physics dataset")
    with np.load(table_path) as table:
        keys = [str(value) for value in table["motion_keys"]]
        num_steps = table["num_steps"].astype(np.int64)

    _prepare_output(output_run, allow_precreated_output)
    for name in LONG_HORIZON_COPIES:
        shutil.copy2(long_horizon_run / "manifests" / name, output_run / "manifests" / name)
    shutil.copy2(long_horizon_run / "data/normalization.npz", output_run / "data/normalization.npz")
    shutil.copy2(table_path, output_run / "data/reference_states.npz")
    shutil.copy2(reference_table_run / "manifests/reference_table_manifest.json",
                 output_run / "manifests/reference_table_manifest.json")

    key_index = {key: index for index, key in enumerate(keys)}
    episodes = _jsonl(output_run / "manifests/episodes.jsonl")
    anchors: list[dict[str, Any]] = []
    handles: dict[str, Any] = {}
    try:
        for episode_index, episode in enumerate(episodes):
            key = str(episode["motion_key"])
            if key not in key_index:
                raise KeyError(f"motion {key} is missing from the reference table")
            path = str(episode["hdf5_path"])
            if path not in handles:
                handles[path] = h5py.File(path, "r")
            motion_step = np.asarray(handles[path][f"data/{episode['episode']}/motion/motion_step"],
                                     dtype=np.int64).reshape(-1)
            steps = int(episode["steps"])
            if motion_step.shape[0] < steps:
                raise ValueError(f"{episode['episode']}: motion_step has {motion_step.shape[0]} rows for {steps} steps")
            episode_ref = f"{episode['source_run']}::{episode['episode']}"
            for t in range(0, steps, anchor_stride):
                anchor_id = digest({"episode_ref": episode_ref, "anchor_step": t})
                natural, h, truncated = history_steps(
                    t, anchor_id, seed=seed, truncation_probability=history_truncation_probability)
                anchors.append({
                    "anchor_id": anchor_id, "episode_index": episode_index, "episode_ref": episode_ref,
                    "episode": episode["episode"], "split": episode["split"], "motion_key": key,
                    "motion_index": key_index[key], "variant_id": int(episode["variant_id"]),
                    "episode_steps": steps, "anchor_step": t, "motion_step": int(motion_step[t]),
                    "natural_history_steps": natural, "history_steps": h, "history_truncated": truncated,
                })
    finally:
        for handle in handles.values():
            handle.close()
    if len({row["anchor_id"] for row in anchors}) != len(anchors):
        raise ValueError("duplicate anchor identities")
    _write_jsonl(output_run / "manifests/anchor_windows.jsonl", anchors)

    comparison = None
    if compare_normalization is not None:
        comparison = normalization_difference(output_run / "data/normalization.npz",
                                              compare_normalization.expanduser().resolve())
        print(f"[reference_mixed] normalization_difference={json.dumps(comparison, sort_keys=True)}", flush=True)
    out_of_range = sum(row["motion_step"] < 0 or row["motion_step"] >= int(num_steps[row["motion_index"]])
                       for row in anchors)
    manifest = {
        "schema_version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "long_horizon_run": str(long_horizon_run),
        "long_horizon_manifest_sha256": file_sha256(long_horizon_run / "manifests/dataset_manifest.json"),
        "physics_dataset_run": str(physics_run),
        "reference_table_run": str(reference_table_run),
        "reference_table_sha256": table_manifest["table_sha256"],
        "anchor_stride": anchor_stride,
        "history_truncation_probability": history_truncation_probability,
        "history_truncation_seed": seed,
        "anchor_count": len(anchors),
        "anchor_count_by_split": dict(Counter(str(row["split"]) for row in anchors)),
        "truncated_count": sum(row["history_truncated"] for row in anchors),
        "full_history_count": sum(row["history_steps"] == ANCHOR for row in anchors),
        "anchor_motion_step_out_of_range": int(out_of_range),
        "anchor_windows_sha256": file_sha256(output_run / "manifests/anchor_windows.jsonl"),
        "normalization_sha256": file_sha256(output_run / "data/normalization.npz"),
        "train_motion_keys": sorted({str(row["motion_key"]) for row in anchors if row["split"] == "train"}),
        "normalization_difference": comparison,
    }
    atomic_write_json(output_run / "manifests/reference_mixed_manifest.json", manifest)
    atomic_write_text(output_run / "markers" / MARKER, "PASS\n")
    return manifest


def _shift(value: torch.Tensor, shift: int) -> torch.Tensor:
    """Move time index k to k+shift, dropping the overflow and zero-filling the left."""
    out = torch.zeros_like(value)
    if shift < value.shape[0]:
        out[shift:] = value[: value.shape[0] - shift]
    return out


class ReferenceAnchorDataset(Dataset[dict[str, Any]]):
    """Reference-conditioned anchors with the rollout layout and hidden real targets."""

    def __init__(self, dataset_run: Path, split: str = "train") -> None:
        from .dataset import StateActionWindowDataset

        self.dataset_run = Path(dataset_run).expanduser().resolve()
        if not (self.dataset_run / "markers" / MARKER).is_file():
            raise FileNotFoundError(self.dataset_run / "markers" / MARKER)
        self.manifest = load_json(self.dataset_run / "manifests/reference_mixed_manifest.json")
        if self.manifest.get("schema_version") != VERSION:
            raise ValueError("unsupported reference mixed dataset manifest")
        anchors_path = self.dataset_run / "manifests/anchor_windows.jsonl"
        if file_sha256(anchors_path) != self.manifest["anchor_windows_sha256"]:
            raise ValueError("anchor_windows.jsonl no longer matches its manifest")
        self.anchors = [row for row in _jsonl(anchors_path) if row["split"] == split]
        if not self.anchors:
            raise ValueError(f"split {split!r} has no reference anchors")
        with np.load(self.dataset_run / "data/reference_states.npz") as table:
            self.reference_states = table["states"].astype(np.float32)
            self.motion_offsets = table["motion_offsets"].astype(np.int64)
            self.num_steps = table["num_steps"].astype(np.int64)
        long_manifest = load_json(self.dataset_run / "manifests/dataset_manifest.json")
        self.source = StateActionWindowDataset(Path(long_manifest["source_dataset_run"]), split, 64, 64,
                                               max_episodes=None, random_crop=False)
        if file_sha256(self.dataset_run / "data/normalization.npz") != file_sha256(
                self.source.dataset_run / "data/normalization.npz"):
            raise ValueError("reference mixed normalization differs from the Physics source")
        self.episodes = self.source.episodes
        self._episode_indices = {
            f"{row['source_run']}::{row['episode']}": index for index, row in enumerate(self.source.episodes)
        }
        missing = {row["episode_ref"] for row in self.anchors} - set(self._episode_indices)
        if missing:
            raise ValueError(f"anchor episodes are not available in split: {sorted(missing)[:3]}")
        self.state_mean = torch.from_numpy(self.source.state_mean[:CONTINUOUS])
        self.state_std = torch.from_numpy(self.source.state_std[:CONTINUOUS])

    def __len__(self) -> int:
        return len(self.anchors)

    def reference_frames(self, row: dict[str, Any]) -> torch.Tensor:
        motion = int(row["motion_index"])
        frames = np.clip(int(row["motion_step"]) + np.arange(1, FUTURE + 1), 0, int(self.num_steps[motion]) - 1)
        return torch.from_numpy(self.reference_states[self.motion_offsets[motion] + frames])

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.anchors[int(index)]
        t, h = int(row["anchor_step"]), int(row["history_steps"])
        window = self.source.get_window(self._episode_indices[row["episode_ref"]], t - h)
        shift = ANCHOR - h
        sample = dict(window)
        for key in STATE_ALIGNED:
            sample[key] = _shift(window[key], shift)
        for key in ACTION_ALIGNED:
            sample[key] = _shift(window[key], shift)
        target_state_valid = sample["valid_state"].clone()
        target_action_valid = sample["valid_action"].clone()
        state = sample["physical_state"]
        reference = self.reference_frames(row)
        state[ANCHOR + 1:, :CONTINUOUS] = (reference[:, :CONTINUOUS] - self.state_mean) / self.state_std
        state[ANCHOR + 1:, CONTINUOUS:] *= target_state_valid[ANCHOR + 1:, None].to(state.dtype)
        sample["valid_state"][ANCHOR + 1:] = True
        sample["valid_action"][ANCHOR:] = True
        sample.update({
            "window_start": t,
            "target_action_valid": target_action_valid,
            "target_contact_valid": target_state_valid,
            "condition_kind": "reference",
            "anchor_id": row["anchor_id"],
            "anchor_step": t,
            "history_steps": h,
            "history_truncated": bool(row["history_truncated"]),
        })
        return sample

    def close(self) -> None:
        self.source.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-horizon-run", type=Path, required=True)
    parser.add_argument("--reference-table-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--anchor-stride", type=int, default=4)
    parser.add_argument("--history-truncation-probability", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--compare-normalization", type=Path, default=None)
    parser.add_argument("--allow-precreated-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = build_reference_mixed_dataset(
        args.long_horizon_run, args.reference_table_run, args.output_run,
        anchor_stride=args.anchor_stride,
        history_truncation_probability=args.history_truncation_probability,
        seed=args.seed, compare_normalization=args.compare_normalization,
        allow_precreated_output=args.allow_precreated_output,
    )
    print(json.dumps({"output_run": str(args.output_run.resolve()), **manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
