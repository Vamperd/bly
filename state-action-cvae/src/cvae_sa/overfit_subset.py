from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .constants import ACTION_DIM, PACKAGES
from .indexer import RunningStats
from .physics_schema import (
    AUXILIARY_TRANSITION_DIM,
    DYNAMICS_CONTEXT_DIM,
    GLOBAL_ROBOT_INFO_DIM,
    JOINT_ROBOT_INFO_DIM,
    PHYSICS_STATE_DIM,
    ROBOT_INFO_DIM,
    dynamics_context_vector,
    load_physics_schema,
    read_auxiliary_transitions,
    read_physics_states,
    robot_information_vector,
    structured_robot_information,
)
from .util import atomic_write_json, atomic_write_text, file_sha256, load_json


PURPOSE = "physics_state_action_32_motion_memorization"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _select_motion_keys(
    rows: list[dict[str, Any]], motions_per_package: int, seed: int
) -> dict[str, list[str]]:
    by_motion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") == "train":
            by_motion[str(row["motion_key"])].append(row)
    eligible: dict[str, list[str]] = {package: [] for package in PACKAGES}
    for motion_key, motion_rows in by_motion.items():
        packages = {str(row["package"]) for row in motion_rows}
        variants = {int(row["variant_id"]) for row in motion_rows}
        completed = all(str(row["status"]) == "completed" for row in motion_rows)
        if len(packages) != 1:
            raise ValueError(f"{motion_key}: package changes across variants")
        package = next(iter(packages))
        if package not in eligible:
            raise ValueError(f"{motion_key}: unsupported package {package!r}")
        if variants == set(range(8)) and len(motion_rows) == 8 and completed:
            eligible[package].append(motion_key)
    rng = np.random.default_rng(seed)
    selected: dict[str, list[str]] = {}
    for package in PACKAGES:
        candidates = sorted(eligible[package])
        if len(candidates) < motions_per_package:
            raise ValueError(
                f"{package}: requires {motions_per_package} train motions with eight completed "
                f"variants, found {len(candidates)}"
            )
        indices = rng.choice(len(candidates), size=motions_per_package, replace=False)
        selected[package] = sorted(candidates[int(index)] for index in indices)
    return selected


def _normalization(
    rows: list[dict[str, Any]], actuator_type_to_id: dict[str, int]
) -> dict[str, np.ndarray]:
    state_stats = RunningStats(PHYSICS_STATE_DIM)
    action_stats = RunningStats(ACTION_DIM)
    robot_stats = RunningStats(ROBOT_INFO_DIM)
    joint_robot_stats = RunningStats(JOINT_ROBOT_INFO_DIM)
    global_robot_stats = RunningStats(GLOBAL_ROBOT_INFO_DIM)
    dynamics_stats = RunningStats(DYNAMICS_CONTEXT_DIM)
    auxiliary_stats = RunningStats(AUXILIARY_TRANSITION_DIM)
    handles: dict[str, h5py.File] = {}
    schemas: dict[str, dict[str, Any]] = {}
    try:
        for row in rows:
            hdf5_path = str(row["hdf5_path"])
            schema_path = str(row["schema_path"])
            stream = handles.get(hdf5_path)
            if stream is None:
                stream = h5py.File(hdf5_path, "r")
                handles[hdf5_path] = stream
            schema = schemas.get(schema_path)
            if schema is None:
                schema = load_physics_schema(Path(schema_path))
                schemas[schema_path] = schema
            episode = stream[f"data/{row['episode']}"]
            states = read_physics_states(episode["states"])
            actions = np.asarray(
                episode["actions/action_target_canonical"], dtype=np.float32
            )
            auxiliary = read_auxiliary_transitions(episode["diagnostics"])
            context = stream[f"contexts/{row['context_id']}"]
            robot = robot_information_vector(schema, context, int(row["env_id"]))
            joint_robot, _, global_robot = structured_robot_information(
                schema, context, int(row["env_id"]), actuator_type_to_id
            )
            state_stats.update(states)
            action_stats.update(actions)
            robot_stats.update(robot[None])
            joint_robot_stats.update(joint_robot)
            global_robot_stats.update(global_robot[None])
            dynamics_stats.update(dynamics_context_vector(context)[None])
            auxiliary_stats.update(auxiliary)
    finally:
        for stream in handles.values():
            stream.close()
    state_mean, state_std = state_stats.finalize()
    state_mean[64:67], state_std[64:67] = 0.0, 1.0
    state_mean[68:70], state_std[68:70] = 0.0, 1.0
    result: dict[str, np.ndarray] = {
        "physical_state_mean": state_mean,
        "physical_state_std": state_std,
    }
    for prefix, stats in (
        ("action", action_stats),
        ("robot_info", robot_stats),
        ("joint_robot_info", joint_robot_stats),
        ("global_robot_info", global_robot_stats),
        ("dynamics_context", dynamics_stats),
        ("auxiliary_transition", auxiliary_stats),
    ):
        mean, std = stats.finalize()
        result[f"{prefix}_mean"] = mean
        result[f"{prefix}_std"] = std
    return result


def build_overfit_subset(
    parent_dataset_run: Path,
    output_run: Path,
    motions_per_package: int = 4,
    seed: int = 20260828,
) -> dict[str, Any]:
    parent = parent_dataset_run.expanduser().resolve()
    output = output_run.expanduser().resolve()
    if motions_per_package <= 0:
        raise ValueError("motions_per_package must be positive")
    parent_manifest_path = parent / "manifests/dataset_manifest.json"
    parent_episodes_path = parent / "manifests/episodes.jsonl"
    for path in (
        parent / "markers/cvae_dataset.ok",
        parent / "markers/cvae_physics_dataset.ok",
        parent_manifest_path,
        parent_episodes_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    parent_manifest = load_json(parent_manifest_path)
    if parent_manifest.get("schema_version") != "sonic_physics_state_action_cvae_dataset_v4":
        raise ValueError("overfit subset requires a Physics v4 parent dataset")
    expected_hash = parent_manifest.get("episodes_index_sha256")
    if expected_hash and file_sha256(parent_episodes_path) != expected_hash:
        raise ValueError("parent episodes index hash mismatch")
    parent_rows = _read_rows(parent_episodes_path)
    selected_by_package = _select_motion_keys(parent_rows, motions_per_package, seed)
    selected_keys = sorted(key for values in selected_by_package.values() for key in values)
    selected_set = set(selected_keys)
    rows = [dict(row, split="train") for row in parent_rows if row["motion_key"] in selected_set]
    rows.sort(key=lambda row: (str(row["motion_key"]), int(row["variant_id"])))
    expected_motions = motions_per_package * len(PACKAGES)
    expected_episodes = expected_motions * 8
    if len(selected_keys) != expected_motions or len(rows) != expected_episodes:
        raise ValueError(
            f"subset cardinality mismatch: motions={len(selected_keys)}, episodes={len(rows)}"
        )
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "videos"):
        (output / child).mkdir(parents=True, exist_ok=True)
    vocabulary = parent_manifest["representations"]["joint_actuator_type"]["vocabulary"]
    actuator_type_to_id = {str(name): index for index, name in enumerate(vocabulary)}
    normalization = _normalization(rows, actuator_type_to_id)
    normalization_path = output / "data/normalization.npz"
    temporary = normalization_path.with_name(f".{normalization_path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **normalization)
    os.replace(temporary, normalization_path)
    episodes_path = output / "manifests/episodes.jsonl"
    atomic_write_text(
        episodes_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    split_manifest = {"train": selected_keys, "validation": [], "test": []}
    atomic_write_json(output / "manifests/split_motion_keys.json", split_manifest)
    selection = {
        "purpose": PURPOSE,
        "seed": seed,
        "motions_per_package": motions_per_package,
        "eligibility": "parent train split; variants exactly 0..7; every variant completed",
        "selected_by_package": selected_by_package,
        "selected_motion_keys": selected_keys,
    }
    atomic_write_json(output / "manifests/overfit_selection.json", selection)
    manifest = {
        **parent_manifest,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": PURPOSE,
        "memorization_benchmark": True,
        "generalization_claim_allowed": False,
        "seed": seed,
        "parent_dataset_run": str(parent),
        "parent_dataset_manifest_sha256": file_sha256(parent_manifest_path),
        "source_read_only": True,
        "motion_count": expected_motions,
        "canonical_episode_count": expected_episodes,
        "transition_count": int(sum(int(row["steps"]) for row in rows)),
        "package_counts": {
            package: len(selected_by_package[package]) for package in PACKAGES
        },
        "split_motion_counts": {"train": expected_motions, "validation": 0, "test": 0},
        "split_episode_counts": {"train": expected_episodes, "validation": 0, "test": 0},
        "window_transitions": 128,
        "validation_stride": 64,
        "normalization": {
            "path": str(normalization_path),
            "training_split_only": True,
            "subset_recomputed": True,
            "gravity": "unit_vector",
            "contact": "binary",
        },
        "episodes_index_sha256": file_sha256(episodes_path),
        "selection_manifest": str(output / "manifests/overfit_selection.json"),
    }
    atomic_write_json(output / "manifests/dataset_manifest.json", manifest)
    atomic_write_text(output / "markers/cvae_dataset.ok", "PASS\n")
    atomic_write_text(output / "markers/cvae_physics_dataset.ok", "PASS\n")
    atomic_write_text(output / "markers/cvae_overfit_subset.ok", "PASS\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a read-only 32-motion overfit subset")
    parser.add_argument("--parent-dataset-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--motions-per-package", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    result = build_overfit_subset(
        args.parent_dataset_run, args.output_run, args.motions_per_package, args.seed
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
