"""Derived contiguous T64 groups for long-horizon State--Action experiments.

The source Physics dataset and its HDF5 files remain read-only.  This module
only writes a small derived index, manifests, and a copy of the source
normalization archive so the normal training identity checks can still be used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from typing import Any

import torch
from torch.utils.data import Dataset

from .util import atomic_write_json, atomic_write_text, file_sha256, load_json


VERSION = "sonic_long_horizon_state_action_dataset_v1"
CHUNK_TRANSITIONS = 64
TARGET_HORIZON = 192
CONTEXT_STEPS = 32


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _source_hashes(dataset_run: Path, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = dataset_run / "manifests" / "dataset_manifest.json"
    episodes_path = dataset_run / "manifests" / "episodes.jsonl"
    normalization = dataset_run / "data" / "normalization.npz"
    hdf5_paths = sorted({str(Path(row["hdf5_path"]).resolve()) for row in episodes})
    schema_paths = sorted({str(Path(row["schema_path"]).resolve()) for row in episodes})
    return {
        "dataset_run": str(dataset_run),
        "dataset_manifest_sha256": file_sha256(manifest),
        "episodes_index_sha256": file_sha256(episodes_path),
        "normalization_sha256": file_sha256(normalization),
        "hdf5": [{"path": path, "sha256": file_sha256(Path(path))} for path in hdf5_paths],
        "schema": [{"path": path, "sha256": file_sha256(Path(path))} for path in schema_paths],
    }


def _validate_source(dataset_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_run = dataset_run.expanduser().resolve()
    required = (
        dataset_run / "markers" / "cvae_dataset.ok",
        dataset_run / "manifests" / "dataset_manifest.json",
        dataset_run / "manifests" / "episodes.jsonl",
        dataset_run / "data" / "normalization.npz",
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    manifest = load_json(required[1])
    if not str(manifest.get("schema_version", "")).startswith("sonic_physics_state_action_cvae_dataset_v"):
        raise ValueError("long-horizon dataset requires a Physics v3/v4/v5 source")
    episodes = _jsonl(required[2])
    if not episodes:
        raise ValueError("source dataset has no episodes")
    for row in episodes:
        for key in ("episode", "steps", "hdf5_path", "schema_path", "source_run", "motion_key", "variant_id", "split"):
            if key not in row:
                raise ValueError(f"source episode row is missing {key!r}")
        if int(row["steps"]) < 0:
            raise ValueError("episode steps must be non-negative")
        if not Path(row["hdf5_path"]).is_file():
            raise FileNotFoundError(row["hdf5_path"])
    return manifest, episodes


def build_long_horizon_dataset(
    source_dataset_run: Path,
    output_run: Path,
    *,
    target_horizon: int = TARGET_HORIZON,
    chunk_transitions: int = CHUNK_TRANSITIONS,
    context_steps: int = CONTEXT_STEPS,
    include_tail: bool = True,
) -> dict[str, Any]:
    """Create a manifest-only contiguous episode grouping."""
    if target_horizon <= 0 or chunk_transitions <= 0 or target_horizon % chunk_transitions:
        raise ValueError("target_horizon must be a positive multiple of chunk_transitions")
    if context_steps < 0 or context_steps >= chunk_transitions:
        raise ValueError("context_steps must be in [0, chunk_transitions)")
    source_dataset_run = source_dataset_run.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    if output_run.exists() and any(output_run.iterdir()):
        raise FileExistsError(f"output run must be new and empty: {output_run}")
    for folder in ("data", "manifests", "markers", "logs", "checkpoints", "videos"):
        (output_run / folder).mkdir(parents=True, exist_ok=True)

    source_manifest, source_episodes = _validate_source(source_dataset_run)
    source_hash = _source_hashes(source_dataset_run, source_episodes)
    groups: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {"episodes": {}, "total_transitions": 0, "covered_transitions": 0}
    group_count_by_split: dict[str, int] = defaultdict(int)
    window_count_by_split: dict[str, int] = defaultdict(int)

    for episode_index, episode in enumerate(source_episodes):
        steps = int(episode["steps"])
        episode_ref = f"{episode['source_run']}::{episode['episode']}"
        starts = list(range(0, steps, target_horizon)) if steps else []
        if not include_tail:
            starts = [start for start in starts if start + target_horizon <= steps]
        group_rows: list[dict[str, Any]] = []
        covered = 0
        for group_index, start in enumerate(starts):
            stop = min(start + target_horizon, steps)
            valid_steps = max(0, stop - start)
            if valid_steps <= 0:
                continue
            group_id = f"{episode_ref}::segment_{group_index:05d}_start_{start:08d}"
            context_start = start - context_steps
            group = {
                "segment_group_id": group_id,
                "episode_index": episode_index,
                "episode_ref": episode_ref,
                "source_run": episode["source_run"],
                "hdf5_path": episode["hdf5_path"],
                "schema_path": episode["schema_path"],
                "episode": episode["episode"],
                "split": episode["split"],
                "motion_key": episode["motion_key"],
                "variant_id": int(episode["variant_id"]),
                "target_start": start,
                "target_stop": stop,
                "target_steps": valid_steps,
                "target_horizon_steps": target_horizon,
                "chunk_transitions": chunk_transitions,
                "chunk_count": target_horizon // chunk_transitions,
                "context_steps": context_steps,
                "context_start": context_start,
                "context_available": context_start >= 0,
                "tail_group": valid_steps < target_horizon,
                "valid_steps": valid_steps,
                "chunk_starts": [],
                "chunk_valid_steps": [],
            }
            for chunk_index in range(target_horizon // chunk_transitions):
                chunk_start = start + chunk_index * chunk_transitions
                chunk_valid = max(0, min(chunk_transitions, steps - chunk_start))
                group["chunk_starts"].append(chunk_start)
                group["chunk_valid_steps"].append(chunk_valid)
                if chunk_valid <= 0:
                    continue
                window = {
                    "segment_group_id": group_id,
                    "episode_index": episode_index,
                    "episode_ref": episode_ref,
                    "source_run": episode["source_run"],
                    "hdf5_path": episode["hdf5_path"],
                    "schema_path": episode["schema_path"],
                    "episode": episode["episode"],
                    "split": episode["split"],
                    "motion_key": episode["motion_key"],
                    "variant_id": int(episode["variant_id"]),
                    "group_chunk_index": chunk_index,
                    "window_start": chunk_start,
                    "window_stop": chunk_start + chunk_valid,
                    "group_valid_steps": valid_steps,
                    "valid_steps": chunk_valid,
                    "context_start": context_start,
                    "context_available": context_start >= 0,
                    "tail_group": valid_steps < target_horizon,
                }
                windows.append(window)
                group["chunk_valid_steps"][-1] = chunk_valid
                covered += chunk_valid
            group_rows.append(group)
            groups.append(group)
            group_count_by_split[str(episode["split"])] += 1
            window_count_by_split[str(episode["split"])] += sum(v > 0 for v in group["chunk_valid_steps"])
        if starts:
            expected = list(range(0, steps))
            ranges: list[int] = []
            for group in group_rows:
                ranges.extend(range(int(group["target_start"]), int(group["target_stop"])))
            if include_tail and ranges != expected:
                raise ValueError(f"coverage mismatch for {episode_ref}")
            covered = len(ranges)
        coverage["episodes"][episode_ref] = {
            "motion_key": episode["motion_key"],
            "variant_id": int(episode["variant_id"]),
            "split": episode["split"],
            "steps": steps,
            "covered_steps": covered,
            "coverage_pass": covered == steps,
            "coverage_expected": bool(include_tail),
            "group_count": len(group_rows),
        }
        coverage["total_transitions"] += steps
        coverage["covered_transitions"] += covered

    if include_tail and coverage["total_transitions"] != coverage["covered_transitions"]:
        raise ValueError("global transition coverage is incomplete")

    _write_jsonl(output_run / "manifests" / "long_horizon_groups.jsonl", groups)
    _write_jsonl(output_run / "manifests" / "long_horizon_windows.jsonl", windows)
    _write_jsonl(output_run / "manifests" / "episodes.jsonl", source_episodes)
    shutil.copy2(source_dataset_run / "data" / "normalization.npz", output_run / "data" / "normalization.npz")
    atomic_write_json(output_run / "manifests" / "source_hashes.json", source_hash)
    atomic_write_json(output_run / "manifests" / "coverage.json", coverage)
    manifest = {
        "schema_version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset_run": str(source_dataset_run),
        "source_dataset_manifest": source_manifest,
        "source_hashes": source_hash,
        "source_read_only": True,
        "identity_key": "segment_group_id",
        "chunk_transitions": chunk_transitions,
        "target_horizon_steps": target_horizon,
        "context_steps": context_steps,
        "group_stride": target_horizon,
        "include_tail_groups": bool(include_tail),
        "group_count": len(groups),
        "window_count": len(windows),
        "group_count_by_split": dict(group_count_by_split),
        "window_count_by_split": dict(window_count_by_split),
        "coverage_sha256": file_sha256(output_run / "manifests" / "coverage.json"),
        "groups_sha256": file_sha256(output_run / "manifests" / "long_horizon_groups.jsonl"),
        "windows_sha256": file_sha256(output_run / "manifests" / "long_horizon_windows.jsonl"),
        "normalization_sha256": file_sha256(output_run / "data" / "normalization.npz"),
    }
    atomic_write_json(output_run / "manifests" / "dataset_manifest.json", manifest)
    atomic_write_text(output_run / "markers" / "cvae_long_horizon_dataset.ok", "PASS\n")
    return manifest


class LongHorizonWindowDataset(Dataset[dict[str, Any]]):
    """T64 samples addressed by the derived long-horizon window manifest."""

    def __init__(self, dataset_run: Path, split: str = "train", include_tail: bool = True) -> None:
        self.dataset_run = dataset_run.expanduser().resolve()
        self.manifest = load_json(self.dataset_run / "manifests" / "dataset_manifest.json")
        if self.manifest.get("schema_version") != VERSION:
            raise ValueError("unsupported long-horizon dataset manifest")
        self.groups = _jsonl(self.dataset_run / "manifests" / "long_horizon_groups.jsonl")
        rows = _jsonl(self.dataset_run / "manifests" / "long_horizon_windows.jsonl")
        self.windows = [row for row in rows if row["split"] == split and (include_tail or not row["tail_group"])]
        if not self.windows:
            raise ValueError(f"split {split!r} has no long-horizon windows")
        from .dataset import StateActionWindowDataset
        source_run = Path(self.manifest["source_dataset_run"])
        self.source = StateActionWindowDataset(source_run, split, 64, 64, max_episodes=None, random_crop=False)
        self._episode_indices = {
            f"{row['source_run']}::{row['episode']}": index
            for index, row in enumerate(self.source.episodes)
        }
        self.group_count = len({row["segment_group_id"] for row in self.windows})
        self.long_horizon_manifest = str(self.dataset_run / "manifests" / "long_horizon_windows.jsonl")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.windows[int(index)]
        episode_index = self._episode_indices.get(str(row["episode_ref"]))
        if episode_index is None:
            raise ValueError(f"source episode is not available in split: {row['episode_ref']}")
        sample = self.source.get_window(episode_index, int(row["window_start"]))
        sample.update({
            "segment_group_id": row["segment_group_id"],
            "group_chunk_index": int(row["group_chunk_index"]),
            "group_valid_steps": int(row["group_valid_steps"]),
            "context_start": int(row["context_start"]),
            "context_available": bool(row["context_available"]),
            "tail_group": bool(row["tail_group"]),
        })
        return sample

    def close(self) -> None:
        self.source.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--target-horizon", type=int, default=TARGET_HORIZON)
    parser.add_argument("--chunk-transitions", type=int, default=CHUNK_TRANSITIONS)
    parser.add_argument("--context-steps", type=int, default=CONTEXT_STEPS)
    parser.add_argument("--exclude-tail", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = build_long_horizon_dataset(
        args.source_dataset_run,
        args.output_run,
        target_horizon=args.target_horizon,
        chunk_transitions=args.chunk_transitions,
        context_steps=args.context_steps,
        include_tail=not args.exclude_tail,
    )
    print(json.dumps({"output_run": str(args.output_run.resolve()), **manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
