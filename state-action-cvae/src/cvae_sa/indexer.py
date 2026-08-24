from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from .constants import ACTION_DIM, CONTROL_DT, PACKAGES, PHYSICAL_STATE_DIM, SPLITS
from .schema import (
    SchemaError,
    load_schema,
    raw_action_to_relative,
    read_physical_state,
)
from .util import atomic_write_json, atomic_write_text, file_sha256, load_json


@dataclass(frozen=True)
class SourceRun:
    run_dir: Path
    dataset: Path
    schema_path: Path
    summary_path: Path
    index_path: Path
    schema: dict[str, Any]
    profile: str


class RunningStats:
    def __init__(self, width: int) -> None:
        self.count = 0
        self.sum = np.zeros(width, dtype=np.float64)
        self.square_sum = np.zeros(width, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.sum.size)
        if values.size == 0:
            return
        self.count += values.shape[0]
        self.sum += values.sum(axis=0)
        self.square_sum += np.square(values).sum(axis=0)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("cannot finalize empty normalization statistics")
        mean = self.sum / self.count
        variance = np.maximum(self.square_sum / self.count - np.square(mean), 1e-12)
        std = np.sqrt(variance)
        std[std < 1e-6] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)


def _constant_scalar(group: h5py.Group, path: str, steps: int) -> int:
    values = np.asarray(group[path], dtype=np.int64)
    if values.shape != (steps,) or not np.all(values == values[0]):
        raise ValueError(f"{group.name}/{path} is not constant with shape [{steps}]")
    return int(values[0])


def _source_run(path: Path) -> SourceRun:
    run_dir = path.expanduser().resolve()
    marker = run_dir / "markers" / "collect_state_action.ok"
    summary_path = run_dir / "manifests" / "collection_summary.json"
    schema_path = run_dir / "manifests" / "state_action_schema.json"
    index_path = run_dir / "manifests" / "canonical_episode_index.json"
    dataset = run_dir / "data" / "sonic_minimal_sa.hdf5"
    for required in (marker, summary_path, schema_path, index_path, dataset):
        if not required.is_file() or (required != marker and required.stat().st_size == 0):
            raise FileNotFoundError(f"required collection artifact is missing: {required}")
    summary = load_json(summary_path)
    if summary.get("passed") is not True:
        raise ValueError(f"collection summary is not PASS: {summary_path}")
    schema = load_schema(schema_path)
    collection = schema.get("motion_collection", {})
    profile = str(collection.get("randomization_profile", ""))
    if profile not in {"startup", "initial_state_mild"}:
        raise ValueError(f"unsupported randomization profile {profile!r} in {schema_path}")
    return SourceRun(run_dir, dataset, schema_path, summary_path, index_path, schema, profile)


def _hash_order(seed: int, value: str) -> str:
    import hashlib

    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def stratified_motion_split(
    motions: list[dict[str, Any]],
    target_counts: tuple[int, int, int],
    seed: int,
) -> dict[str, str]:
    if sum(target_counts) != len(motions):
        raise ValueError(
            f"split targets sum to {sum(target_counts)}, found {len(motions)} motions"
        )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for motion in motions:
        grouped[(motion["cohort"], motion["package"])].append(motion)
    fractions = np.asarray(target_counts, dtype=np.float64) / len(motions)
    allocation: dict[tuple[str, str], np.ndarray] = {}
    remaining_by_group: dict[tuple[str, str], int] = {}
    allocated_totals = np.zeros(3, dtype=np.int64)
    for key, rows in grouped.items():
        ideal = len(rows) * fractions
        base = np.floor(ideal).astype(np.int64)
        allocation[key] = base
        remaining_by_group[key] = len(rows) - int(base.sum())
        allocated_totals += base

    target = np.asarray(target_counts, dtype=np.int64)
    while int(sum(remaining_by_group.values())):
        candidates: list[tuple[float, str, tuple[str, str], int]] = []
        for group_key, remaining in remaining_by_group.items():
            if remaining <= 0:
                continue
            size = len(grouped[group_key])
            ideal = size * fractions
            for split_index in range(3):
                if allocated_totals[split_index] >= target[split_index]:
                    continue
                deficit = float(ideal[split_index] - allocation[group_key][split_index])
                tie = _hash_order(seed, f"{group_key}:{SPLITS[split_index]}")
                candidates.append((deficit, tie, group_key, split_index))
        if not candidates:
            raise RuntimeError("could not satisfy exact stratified split quotas")
        _, _, group_key, split_index = max(candidates)
        allocation[group_key][split_index] += 1
        allocated_totals[split_index] += 1
        remaining_by_group[group_key] -= 1

    if tuple(int(value) for value in allocated_totals) != target_counts:
        raise RuntimeError(f"split allocation mismatch: {allocated_totals}")
    result: dict[str, str] = {}
    for group_key, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: _hash_order(seed, row["motion_key"]))
        cursor = 0
        for split_index, split_name in enumerate(SPLITS):
            count = int(allocation[group_key][split_index])
            for row in ordered[cursor : cursor + count]:
                result[row["motion_key"]] = split_name
            cursor += count
    return result


def _high_frequency_energy(actions: np.ndarray) -> float:
    if actions.shape[0] < 4:
        return 0.0
    centered = actions - actions.mean(axis=0, keepdims=True)
    spectrum = np.square(np.abs(np.fft.rfft(centered, axis=0)))
    start = max(1, int(math.ceil(spectrum.shape[0] * 0.75)))
    total = float(spectrum[1:].sum())
    return float(spectrum[start:].sum() / total) if total > 1e-12 else 0.0


def _physical_sequence(episode: h5py.Group) -> np.ndarray:
    current = read_physical_state(episode["state_t"])
    next_state = read_physical_state(episode["state_tp1"])
    if current.shape != next_state.shape:
        raise ValueError(f"{episode.name}: state_t/state_tp1 shape mismatch")
    if current.shape[0] > 1 and not np.allclose(
        next_state[:-1], current[1:], rtol=1e-4, atol=1e-5
    ):
        raise ValueError(f"{episode.name}: physical state continuity failed")
    return np.concatenate((current, next_state[-1:]), axis=0)


def _mapped_previous_sequence(
    episode: h5py.Group, schema: dict[str, Any], env_id: int
) -> np.ndarray:
    previous_raw = np.asarray(episode["state_t/previous_action"], dtype=np.float32)
    next_previous_raw = np.asarray(
        episode["state_tp1/previous_action"], dtype=np.float32
    )
    actions = np.asarray(episode["actions"], dtype=np.float32)
    if previous_raw.shape != actions.shape or next_previous_raw.shape != actions.shape:
        raise ValueError(f"{episode.name}: previous action shape mismatch")
    if actions.shape[0] > 1 and not np.allclose(
        previous_raw[1:], actions[:-1], rtol=1e-5, atol=1e-6
    ):
        raise ValueError(f"{episode.name}: previous action shift failed")
    if not np.allclose(next_previous_raw, actions, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{episode.name}: state_tp1 previous action mismatch")
    current_mapped, _ = raw_action_to_relative(previous_raw, schema, env_id)
    terminal_mapped, _ = raw_action_to_relative(next_previous_raw[-1:], schema, env_id)
    return np.concatenate((current_mapped, terminal_mapped), axis=0)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source_hash_record(source: SourceRun) -> dict[str, Any]:
    paths = {
        "dataset": source.dataset,
        "schema": source.schema_path,
        "summary": source.summary_path,
        "canonical_index": source.index_path,
    }
    return {
        "run_dir": str(source.run_dir),
        "files": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in paths.items()
        },
    }


def build_index(
    source_paths: Iterable[Path],
    output_run: Path,
    expected_motions: int = 768,
    expected_episodes: int = 5120,
    split_counts: tuple[int, int, int] = (616, 76, 76),
    seed: int = 20260824,
) -> dict[str, Any]:
    output_run = output_run.expanduser().resolve()
    output_run.mkdir(parents=True, exist_ok=False) if not output_run.exists() else None
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "videos"):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    sources = [_source_run(path) for path in source_paths]
    if len(sources) != 4 and expected_motions == 768:
        raise ValueError(f"production index requires four source runs, found {len(sources)}")

    records: list[dict[str, Any]] = []
    motion_meta: dict[str, dict[str, Any]] = {}
    pair_seen: set[tuple[str, int]] = set()
    joint_names: list[str] | None = None
    for source_index, source in enumerate(sources):
        current_names = list(source.schema["joint_names"])
        if joint_names is None:
            joint_names = current_names
        elif joint_names != current_names:
            raise ValueError(f"joint order differs in {source.schema_path}")
        configured_offset = int(
            source.schema.get("motion_collection", {}).get("variant_offset", -1)
        )
        configured_repeat = int(
            source.schema.get("motion_collection", {}).get("eval_motion_repeat", -1)
        )
        expected_variants = set(
            range(configured_offset, configured_offset + configured_repeat)
        )
        index = load_json(source.index_path).get("episodes", [])
        with h5py.File(source.dataset, "r") as stream:
            data = stream.get("data")
            if data is None:
                raise ValueError(f"{source.dataset} has no data group")
            for item in index:
                if int(item.get("attempt_id", -1)) != 0:
                    raise ValueError(f"canonical index contains nonzero attempt: {item}")
                episode_name = str(item["episode"])
                if episode_name not in data:
                    raise ValueError(f"{source.dataset} is missing {episode_name}")
                episode = data[episode_name]
                actions = np.asarray(episode["actions"], dtype=np.float32)
                steps = int(actions.shape[0])
                if steps <= 0 or actions.shape != (steps, ACTION_DIM):
                    raise ValueError(f"{episode.name}: invalid action shape {actions.shape}")
                if int(episode.attrs.get("num_samples", -1)) != steps:
                    raise ValueError(f"{episode.name}: num_samples mismatch")
                terminated = np.asarray(episode["outcome/terminated"], dtype=bool)
                truncated = np.asarray(episode["outcome/truncated"], dtype=bool)
                if terminated.shape != (steps,) or truncated.shape != (steps,):
                    raise ValueError(f"{episode.name}: invalid termination shape")
                done = terminated | truncated
                if bool(done[:-1].any()) or not bool(done[-1]):
                    raise ValueError(f"{episode.name}: canonical boundary is not final-only")
                status = "failed" if bool(terminated[-1]) else "completed"
                if str(item.get("status", "")) != status:
                    raise ValueError(f"{episode.name}: canonical status/outcome mismatch")
                env_id = _constant_scalar(episode, "motion/env_id", steps)
                variant_id = _constant_scalar(episode, "motion/variant_id", steps)
                attempt_id = _constant_scalar(episode, "motion/attempt_id", steps)
                global_id = _constant_scalar(episode, "motion/global_motion_id", steps)
                if attempt_id != 0 or variant_id not in expected_variants:
                    raise ValueError(f"{episode.name}: invalid canonical identity")
                motion_key = str(
                    item.get("motion_key")
                    or source.schema["motion_id_to_key"].get(str(global_id), "")
                )
                if not motion_key:
                    raise ValueError(f"{episode.name}: empty motion_key")
                if source.schema["motion_id_to_key"].get(str(global_id)) != motion_key:
                    raise ValueError(f"{episode.name}: schema motion identity mismatch")
                pair = (motion_key, variant_id)
                if pair in pair_seen:
                    raise ValueError(f"duplicate cross-run canonical identity {pair}")
                pair_seen.add(pair)
                package = str(item.get("package", ""))
                if package not in PACKAGES:
                    raise ValueError(f"{episode.name}: unsupported package {package!r}")
                meta = motion_meta.setdefault(
                    motion_key,
                    {"motion_key": motion_key, "package": package, "variants": set()},
                )
                if meta["package"] != package:
                    raise ValueError(f"package changes across variants for {motion_key}")
                meta["variants"].add(variant_id)
                action_rel, scale = raw_action_to_relative(actions, source.schema, env_id)
                physical = _physical_sequence(episode)
                previous = _mapped_previous_sequence(episode, source.schema, env_id)
                if not np.isfinite(action_rel).all() or not np.isfinite(physical).all():
                    raise ValueError(f"{episode.name}: non-finite model data")
                joint_velocity = physical[:-1, 29:58]
                action_diff = np.diff(action_rel, axis=0)
                record = {
                    "source_index": source_index,
                    "source_run": str(source.run_dir),
                    "hdf5_path": str(source.dataset),
                    "schema_path": str(source.schema_path),
                    "episode": episode_name,
                    "steps": steps,
                    "env_id": env_id,
                    "global_motion_id": global_id,
                    "motion_key": motion_key,
                    "package": package,
                    "variant_id": variant_id,
                    "attempt_id": 0,
                    "profile": source.profile,
                    "status": status,
                    "terminated": bool(item.get("terminated", False)),
                    "truncated": bool(item.get("truncated", False)),
                    "action_scale": scale.tolist(),
                    "features": {
                        "duration_seconds": steps * float(
                            source.schema.get("control_dt", CONTROL_DT)
                        ),
                        "action_rms": float(np.sqrt(np.mean(np.square(action_rel)))),
                        "action_derivative_rms": float(
                            np.sqrt(np.mean(np.square(action_diff)))
                        )
                        if action_diff.size
                        else 0.0,
                        "joint_velocity_rms": float(
                            np.sqrt(np.mean(np.square(joint_velocity)))
                        ),
                        "high_frequency_energy": _high_frequency_energy(action_rel),
                    },
                }
                records.append(record)

    if len(records) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} canonical episodes, found {len(records)}")
    if len(motion_meta) != expected_motions:
        raise ValueError(f"expected {expected_motions} motions, found {len(motion_meta)}")
    if expected_motions == 768:
        variant_histogram = Counter(len(meta["variants"]) for meta in motion_meta.values())
        if variant_histogram != Counter({4: 256, 8: 512}):
            raise ValueError(f"unexpected variant coverage {dict(variant_histogram)}")
    for meta in motion_meta.values():
        count = len(meta["variants"])
        meta["cohort"] = "old256" if count == 4 else "new512" if count == 8 else f"v{count}"
        meta["variants"] = sorted(meta["variants"])

    motions = sorted(motion_meta.values(), key=lambda item: item["motion_key"])
    split_by_key = stratified_motion_split(motions, split_counts, seed)
    state_stats = RunningStats(61)
    previous_stats = RunningStats(ACTION_DIM)
    action_stats = RunningStats(ACTION_DIM)
    schema_cache = {str(source.schema_path): source.schema for source in sources}
    handles: dict[str, h5py.File] = {}
    try:
        for record in records:
            record["cohort"] = motion_meta[record["motion_key"]]["cohort"]
            record["split"] = split_by_key[record["motion_key"]]
            if record["split"] != "train":
                continue
            hdf_path = record["hdf5_path"]
            stream = handles.setdefault(hdf_path, h5py.File(hdf_path, "r"))
            episode = stream[f"data/{record['episode']}"]
            schema = schema_cache[record["schema_path"]]
            physical = _physical_sequence(episode)
            previous = _mapped_previous_sequence(episode, schema, record["env_id"])
            actions, _ = raw_action_to_relative(
                np.asarray(episode["actions"], dtype=np.float32), schema, record["env_id"]
            )
            state_stats.update(physical[:, :61])
            previous_stats.update(previous)
            action_stats.update(actions)
    finally:
        for stream in handles.values():
            stream.close()

    state_mean61, state_std61 = state_stats.finalize()
    state_mean = np.concatenate((state_mean61, np.zeros(3, dtype=np.float32)))
    state_std = np.concatenate((state_std61, np.ones(3, dtype=np.float32)))
    previous_mean, previous_std = previous_stats.finalize()
    action_mean, action_std = action_stats.finalize()
    normalization_path = output_run / "data" / "normalization.npz"
    temporary_npz = normalization_path.with_name(
        f".{normalization_path.name}.tmp.{os.getpid()}"
    )
    with temporary_npz.open("wb") as stream:
        np.savez_compressed(
            stream,
            physical_state_mean=state_mean,
            physical_state_std=state_std,
            previous_action_mean=previous_mean,
            previous_action_std=previous_std,
            action_mean=action_mean,
            action_std=action_std,
        )
    os.replace(temporary_npz, normalization_path)

    records.sort(key=lambda item: (item["motion_key"], item["variant_id"]))
    episodes_path = output_run / "manifests" / "episodes.jsonl"
    atomic_write_text(
        episodes_path,
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
    )
    split_manifest = {
        split: sorted(key for key, assigned in split_by_key.items() if assigned == split)
        for split in SPLITS
    }
    atomic_write_json(output_run / "manifests" / "split_motion_keys.json", split_manifest)
    source_hashes = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [_source_hash_record(source) for source in sources],
    }
    atomic_write_json(output_run / "manifests" / "source_hashes.json", source_hashes)

    audit: dict[str, Any] = {"by_split": {}, "by_package": {}}
    feature_names = tuple(records[0]["features"])
    for split in SPLITS:
        rows = [item for item in records if item["split"] == split]
        audit["by_split"][split] = {
            name: {
                "mean": float(np.mean([row["features"][name] for row in rows])),
                "std": float(np.std([row["features"][name] for row in rows])),
            }
            for name in feature_names
        }
    for package in PACKAGES:
        audit["by_package"][package] = {
            split: sum(
                item["package"] == package and item["split"] == split for item in records
            )
            for split in SPLITS
        }
    atomic_write_json(output_run / "data" / "action_feature_audit.json", audit)

    manifest = {
        "schema_version": "sonic_state_action_cvae_dataset_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "source_runs": [str(source.run_dir) for source in sources],
        "source_read_only": True,
        "identity_key": "motion_key",
        "canonical_attempt_id": 0,
        "include_failed_canonical": True,
        "motion_count": len(motions),
        "canonical_episode_count": len(records),
        "transition_count": int(sum(item["steps"] for item in records)),
        "cohort_counts": dict(Counter(item["cohort"] for item in motions)),
        "package_counts": dict(Counter(item["package"] for item in motions)),
        "split_motion_counts": {
            split: len(split_manifest[split]) for split in SPLITS
        },
        "split_episode_counts": dict(Counter(item["split"] for item in records)),
        "window_transitions": 128,
        "validation_stride": 64,
        "control_dt": CONTROL_DT,
        "joint_names": joint_names,
        "representations": {
            "physical_state": {
                "dimension": 64,
                "fields": ["joint_pos", "joint_vel", "base_ang_vel", "gravity_robot"],
            },
            "previous_action": {
                "dimension": 29,
                "mapping": "clip(raw*scale+offset)-default_joint_pos",
            },
            "action_rel": {
                "dimension": 29,
                "mapping": "clip(raw*scale+offset)-default_joint_pos",
            },
            "action_scale": {"dimension": 29, "normalization": "none"},
        },
        "normalization": {
            "path": str(normalization_path),
            "training_split_only": True,
            "gravity": "unit_vector_not_standardized",
        },
        "episodes_index_sha256": file_sha256(episodes_path),
        "source_hashes_sha256": file_sha256(
            output_run / "manifests" / "source_hashes.json"
        ),
    }
    atomic_write_json(output_run / "manifests" / "dataset_manifest.json", manifest)
    atomic_write_text(output_run / "markers" / "cvae_dataset.ok", "PASS\n")
    return manifest


def _parse_split_counts(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 3 or min(parts) < 0:
        raise argparse.ArgumentTypeError("split counts must be train,validation,test")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a read-only SONIC CVAE index")
    parser.add_argument("--source-run", action="append", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--expected-motions", type=int, default=768)
    parser.add_argument("--expected-episodes", type=int, default=5120)
    parser.add_argument("--split-counts", type=_parse_split_counts, default=(616, 76, 76))
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_index(
            args.source_run,
            args.output_run,
            args.expected_motions,
            args.expected_episodes,
            args.split_counts,
            args.seed,
        )
    except (OSError, ValueError, KeyError, SchemaError) as error:
        print(f"CVAE dataset index: FAIL\n{type(error).__name__}: {error}")
        return 1
    print("CVAE dataset index: PASS")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
