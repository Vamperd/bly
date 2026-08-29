from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from .constants import ACTION_DIM, PACKAGES, SPLITS
from .indexer import RunningStats, _high_frequency_energy, stratified_motion_split
from .physics_schema import (
    AUXILIARY_TRANSITION_DIM,
    DYNAMICS_CONTEXT_DIM,
    GLOBAL_ROBOT_INFO_DIM,
    JOINT_ROBOT_INFO_DIM,
    PHYSICS_STATE_DIM,
    REFERENCE_DIM,
    REFERENCE_FRAMES,
    ROBOT_INFO_DIM,
    actuator_type_names,
    dynamics_context_vector,
    load_physics_schema,
    read_physics_states,
    read_reference_future,
    read_auxiliary_transitions,
    robot_information_vector,
    structured_robot_information,
)
from .util import atomic_write_json, atomic_write_text, file_sha256, load_json


@dataclass(frozen=True)
class PhysicsSource:
    run_dir: Path
    dataset: Path
    schema_path: Path
    summary_path: Path
    index_path: Path
    schema: dict[str, Any]
    profile: str
    cohort: str
    reference_available: bool


def _source(path: Path) -> PhysicsSource:
    run = path.expanduser().resolve()
    reference_dataset = run / "data/sonic_physics_sa_v5.hdf5"
    legacy_dataset = run / "data/sonic_physics_sa_v3.hdf5"
    dataset = reference_dataset if reference_dataset.is_file() else legacy_dataset
    required = {
        "marker": run / "markers/collect_physics_state_action.ok",
        "dataset": dataset,
        "schema": run / "manifests/physics_state_action_schema.json",
        "summary": run / "manifests/collection_summary.json",
        "index": run / "manifests/canonical_episode_index.json",
    }
    for name, item in required.items():
        if not item.is_file() or (name != "marker" and item.stat().st_size == 0):
            raise FileNotFoundError(item)
    summary = load_json(required["summary"])
    if summary.get("passed") is not True:
        raise ValueError(f"source summary is not PASS: {required['summary']}")
    schema = load_physics_schema(required["schema"])
    profile = str(schema["motion_collection"]["randomization_profile"])
    motion_count = len(schema["motion_id_to_key"])
    cohort = "old256" if motion_count == 256 else "new512" if motion_count == 512 else f"m{motion_count}"
    reference_available = schema.get("schema_version") == "sonic_physics_sa_v5"
    return PhysicsSource(
        run, required["dataset"], required["schema"], required["summary"],
        required["index"], schema, profile, cohort, reference_available
    )


def _feature_record(states: np.ndarray, actions: np.ndarray, control_dt: float) -> dict[str, float]:
    difference = np.diff(actions, axis=0)
    return {
        "duration_seconds": actions.shape[0] * control_dt,
        "action_rms": float(np.sqrt(np.mean(np.square(actions)))),
        "action_derivative_rms": float(np.sqrt(np.mean(np.square(difference)))) if difference.size else 0.0,
        "joint_velocity_rms": float(np.sqrt(np.mean(np.square(states[:-1, 29:58])))),
        "high_frequency_energy": _high_frequency_energy(actions),
    }


def _source_hash(source: PhysicsSource) -> dict[str, Any]:
    return {
        "run_dir": str(source.run_dir),
        "files": {
            name: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for name, path in (
                ("dataset", source.dataset), ("schema", source.schema_path),
                ("summary", source.summary_path), ("canonical_index", source.index_path),
            )
        },
    }


def build_physics_index(
    source_paths: Iterable[Path],
    output_run: Path,
    expected_motions: int = 768,
    expected_episodes: int = 6144,
    split_counts: tuple[int, int, int] = (616, 76, 76),
    seed: int = 20260824,
) -> dict[str, Any]:
    output_run = output_run.expanduser().resolve()
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "videos"):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    sources = [_source(path) for path in source_paths]
    if not sources:
        raise ValueError("at least one Physics State-Action source run is required")
    reference_modes = {source.reference_available for source in sources}
    if len(reference_modes) != 1:
        raise ValueError("Physics sources cannot mix reference-aware v5 and legacy v3 data")
    reference_available = reference_modes == {True}
    if expected_motions == 768 and len(sources) != 4:
        raise ValueError(f"production physics index requires four source runs, found {len(sources)}")
    if expected_motions == 768:
        signatures = Counter((source.cohort, source.profile) for source in sources)
        expected_signatures = Counter(
            {
                ("old256", "startup"): 1,
                ("old256", "initial_state_mild"): 1,
                ("new512", "startup"): 1,
                ("new512", "initial_state_mild"): 1,
            }
        )
        if signatures != expected_signatures:
            raise ValueError(
                f"physics source cohort/profile composition is invalid: {signatures}"
            )
    if expected_motions == 2048:
        signatures = Counter((source.cohort, source.profile) for source in sources)
        expected_signatures = Counter(
            {
                ("m2048", "startup"): 1,
                ("m2048", "initial_state_mild"): 1,
            }
        )
        if signatures != expected_signatures:
            raise ValueError(
                "2048-motion physics index requires exactly one startup and one "
                f"initial_state_mild source: {signatures}"
            )
    actuator_types = sorted(
        {
            name
            for source in sources
            for name in actuator_type_names(source.schema)
        }
    )
    if "unknown" not in actuator_types:
        actuator_types.insert(0, "unknown")
    actuator_type_to_id = {name: index for index, name in enumerate(actuator_types)}
    observed_max_delay = max([
        0,
        *(
            int(group.get("max_delay", 0))
            for source in sources
            for group in source.schema.get("actuator_groups", {}).values()
        ),
    ])
    records: list[dict[str, Any]] = []
    motion_meta: dict[str, dict[str, Any]] = {}
    identities: set[tuple[str, int]] = set()
    joint_names: list[str] | None = None
    for source_index, source in enumerate(sources):
        names = list(source.schema["joint_names"])
        if joint_names is None:
            joint_names = names
        elif joint_names != names:
            raise ValueError("joint order differs across physics sources")
        offset = int(source.schema["motion_collection"]["variant_offset"])
        repeat = int(source.schema["motion_collection"]["eval_motion_repeat"])
        if repeat != 4:
            raise ValueError(f"{source.run_dir}: expected four variants, found {repeat}")
        expected_profile = "startup" if offset == 0 else "initial_state_mild" if offset == 4 else None
        if expected_profile is None or source.profile != expected_profile:
            raise ValueError(
                f"{source.run_dir}: profile {source.profile!r} does not match offset {offset}"
            )
        expected_variants = set(range(offset, offset + repeat))
        rows = load_json(source.index_path)["episodes"]
        with h5py.File(source.dataset, "r") as stream:
            for item in rows:
                if int(item["attempt_id"]) != 0:
                    raise ValueError("canonical physics index contains an additional attempt")
                episode_name = str(item["episode"])
                episode = stream[f"data/{episode_name}"]
                steps = int(episode.attrs["num_transitions"])
                states = read_physics_states(episode["states"])
                actions = np.asarray(episode["actions/action_target_canonical"], dtype=np.float32)
                auxiliary = read_auxiliary_transitions(episode["diagnostics"])
                if reference_available:
                    reference, offsets = read_reference_future(episode)
                    expected_offsets = np.asarray(
                        source.schema["reference_future"]["time_offsets_seconds"],
                        dtype=np.float32,
                    )
                    if reference.shape != (steps, REFERENCE_FRAMES, REFERENCE_DIM):
                        raise ValueError(f"{episode.name}: invalid reference length")
                    if not np.array_equal(offsets, expected_offsets):
                        raise ValueError(f"{episode.name}: reference offsets differ from schema")
                if (
                    states.shape != (steps + 1, PHYSICS_STATE_DIM)
                    or actions.shape != (steps, ACTION_DIM)
                    or auxiliary.shape != (steps, AUXILIARY_TRANSITION_DIM)
                ):
                    raise ValueError(f"{episode.name}: invalid State/Action lengths")
                motion_key = str(item["motion_key"])
                variant_id = int(item["variant_id"])
                if variant_id not in expected_variants:
                    raise ValueError(f"{episode.name}: variant {variant_id} outside source range")
                identity = (motion_key, variant_id)
                if identity in identities:
                    raise ValueError(f"duplicate physics identity {identity}")
                identities.add(identity)
                package = str(item["package"])
                if package not in PACKAGES:
                    raise ValueError(f"unsupported package {package!r}")
                meta = motion_meta.setdefault(
                    motion_key,
                    {"motion_key": motion_key, "package": package, "cohort": source.cohort, "variants": set()},
                )
                if meta["package"] != package or meta["cohort"] != source.cohort:
                    raise ValueError(f"motion metadata changes for {motion_key}")
                meta["variants"].add(variant_id)
                records.append(
                    {
                        "source_index": source_index,
                        "source_run": str(source.run_dir),
                        "hdf5_path": str(source.dataset),
                        "schema_path": str(source.schema_path),
                        "episode": episode_name,
                        "steps": steps,
                        "env_id": int(item["env_id"]),
                        "context_id": str(item["context_id"]),
                        "global_motion_id": int(item["global_motion_id"]),
                        "motion_key": motion_key,
                        "package": package,
                        "cohort": source.cohort,
                        "variant_id": variant_id,
                        "attempt_id": 0,
                        "profile": source.profile,
                        "status": str(item["status"]),
                        "features": _feature_record(
                            states, actions, float(source.schema["simulation"]["control_dt"])
                        ),
                    }
                )
    if len(records) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} episodes, found {len(records)}")
    if len(motion_meta) != expected_motions:
        raise ValueError(f"expected {expected_motions} motions, found {len(motion_meta)}")
    for meta in motion_meta.values():
        if meta["variants"] != set(range(8)):
            raise ValueError(f"{meta['motion_key']}: expected variants 0..7")
        meta["variants"] = sorted(meta["variants"])
    if expected_motions == 768:
        cohort_counts = Counter(meta["cohort"] for meta in motion_meta.values())
        package_counts = Counter(meta["package"] for meta in motion_meta.values())
        if cohort_counts != {"old256": 256, "new512": 512}:
            raise ValueError(f"unexpected cohort balance: {cohort_counts}")
        if package_counts != {package: 96 for package in PACKAGES}:
            raise ValueError(f"unexpected package balance: {package_counts}")
    if expected_motions == 2048:
        package_counts = Counter(meta["package"] for meta in motion_meta.values())
        if package_counts != {package: 256 for package in PACKAGES}:
            raise ValueError(f"unexpected 2048-motion package balance: {package_counts}")
    motions = sorted(motion_meta.values(), key=lambda item: item["motion_key"])
    split_by_key = stratified_motion_split(motions, split_counts, seed)
    state_stats = RunningStats(PHYSICS_STATE_DIM)
    action_stats = RunningStats(ACTION_DIM)
    robot_stats = RunningStats(ROBOT_INFO_DIM)
    joint_robot_stats = RunningStats(JOINT_ROBOT_INFO_DIM)
    global_robot_stats = RunningStats(GLOBAL_ROBOT_INFO_DIM)
    dynamics_stats = RunningStats(DYNAMICS_CONTEXT_DIM)
    auxiliary_stats = RunningStats(AUXILIARY_TRANSITION_DIM)
    reference_stats = RunningStats(REFERENCE_DIM) if reference_available else None
    handles: dict[str, h5py.File] = {}
    schemas = {str(source.schema_path): source.schema for source in sources}
    try:
        for record in records:
            record["split"] = split_by_key[record["motion_key"]]
            if record["split"] != "train":
                continue
            stream = handles.setdefault(record["hdf5_path"], h5py.File(record["hdf5_path"], "r"))
            episode = stream[f"data/{record['episode']}"]
            schema = schemas[record["schema_path"]]
            states = read_physics_states(episode["states"])
            actions = np.asarray(episode["actions/action_target_canonical"], dtype=np.float32)
            context = stream[f"contexts/{record['context_id']}"]
            robot_info = robot_information_vector(schema, context, record["env_id"])
            joint_robot, _, global_robot = structured_robot_information(
                schema, context, record["env_id"], actuator_type_to_id
            )
            dynamics_context = dynamics_context_vector(context)
            auxiliary = read_auxiliary_transitions(episode["diagnostics"])
            reference = read_reference_future(episode)[0] if reference_available else None
            state_stats.update(states)
            action_stats.update(actions)
            robot_stats.update(robot_info[None])
            joint_robot_stats.update(joint_robot)
            global_robot_stats.update(global_robot[None])
            dynamics_stats.update(dynamics_context[None])
            auxiliary_stats.update(auxiliary)
            if reference_stats is not None and reference is not None:
                reference_stats.update(reference.reshape(-1, REFERENCE_DIM))
    finally:
        for stream in handles.values():
            stream.close()
    state_mean, state_std = state_stats.finalize()
    state_mean[64:67] = 0.0
    state_std[64:67] = 1.0
    state_mean[68:70] = 0.0
    state_std[68:70] = 1.0
    action_mean, action_std = action_stats.finalize()
    robot_mean, robot_std = robot_stats.finalize()
    joint_robot_mean, joint_robot_std = joint_robot_stats.finalize()
    global_robot_mean, global_robot_std = global_robot_stats.finalize()
    dynamics_mean, dynamics_std = dynamics_stats.finalize()
    auxiliary_mean, auxiliary_std = auxiliary_stats.finalize()
    if reference_stats is not None:
        reference_mean, reference_std = reference_stats.finalize()
    else:
        reference_mean = np.zeros(REFERENCE_DIM, dtype=np.float32)
        reference_std = np.ones(REFERENCE_DIM, dtype=np.float32)
    normalization_path = output_run / "data/normalization.npz"
    temporary = normalization_path.with_name(f".{normalization_path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            physical_state_mean=state_mean,
            physical_state_std=state_std,
            action_mean=action_mean,
            action_std=action_std,
            robot_info_mean=robot_mean,
            robot_info_std=robot_std,
            joint_robot_info_mean=joint_robot_mean,
            joint_robot_info_std=joint_robot_std,
            global_robot_info_mean=global_robot_mean,
            global_robot_info_std=global_robot_std,
            dynamics_context_mean=dynamics_mean,
            dynamics_context_std=dynamics_std,
            auxiliary_transition_mean=auxiliary_mean,
            auxiliary_transition_std=auxiliary_std,
            reference_future_mean=reference_mean,
            reference_future_std=reference_std,
        )
    os.replace(temporary, normalization_path)
    records.sort(key=lambda item: (item["motion_key"], item["variant_id"]))
    episodes_path = output_run / "manifests/episodes.jsonl"
    atomic_write_text(episodes_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    split_manifest = {
        split: sorted(key for key, assigned in split_by_key.items() if assigned == split)
        for split in SPLITS
    }
    atomic_write_json(output_run / "manifests/split_motion_keys.json", split_manifest)
    atomic_write_json(
        output_run / "manifests/source_hashes.json",
        {"generated_at": datetime.now(timezone.utc).isoformat(), "sources": [_source_hash(s) for s in sources]},
    )
    audit: dict[str, Any] = {"by_split": {}, "by_package": {}}
    for split in SPLITS:
        rows = [row for row in records if row["split"] == split]
        audit["by_split"][split] = {
            name: {"mean": float(np.mean([row["features"][name] for row in rows])), "std": float(np.std([row["features"][name] for row in rows]))}
            for name in records[0]["features"]
        }
    for package in PACKAGES:
        audit["by_package"][package] = {
            split: sum(row["package"] == package and row["split"] == split for row in records)
            for split in SPLITS
        }
    atomic_write_json(output_run / "data/action_feature_audit.json", audit)
    manifest = {
        "schema_version": (
            "sonic_physics_state_action_cvae_dataset_v5"
            if reference_available else "sonic_physics_state_action_cvae_dataset_v4"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "source_runs": [str(source.run_dir) for source in sources],
        "source_read_only": True,
        "identity_key": "motion_key",
        "canonical_attempt_id": 0,
        "motion_count": len(motions),
        "canonical_episode_count": len(records),
        "transition_count": int(sum(row["steps"] for row in records)),
        "cohort_counts": dict(Counter(row["cohort"] for row in motions)),
        "package_counts": dict(Counter(row["package"] for row in motions)),
        "split_motion_counts": {split: len(split_manifest[split]) for split in SPLITS},
        "split_episode_counts": dict(Counter(row["split"] for row in records)),
        "window_transitions": 128,
        "validation_stride": 64,
        "joint_names": joint_names,
        "representations": {
            "physical_state": {"dimension": 70, "fields": [name for name, _ in (("joint_pos_canonical",29),("joint_vel",29),("base_lin_vel_robot",3),("base_ang_vel_robot",3),("gravity_robot",3),("base_height",1),("foot_contact",2))]},
            "previous_action": {"dimension": 0, "duplicated": False},
            "action": {"dimension": 29, "definition": "processed_joint_target_abs - nominal_default_joint_pos"},
            "robot_information": {
                "dimension": ROBOT_INFO_DIM,
                "legacy_compatibility": True,
                "model_input": False,
            },
            "joint_robot_information": {
                "shape": [29, JOINT_ROBOT_INFO_DIM],
                "fields": [
                    "nominal_joint_pos", "position_limit_low_canonical",
                    "position_limit_high_canonical", "velocity_limit", "effort_limit",
                    "stiffness_kp", "damping_kd", "armature", "joint_friction",
                    "actuator_min_delay_control_steps",
                    "actuator_max_delay_control_steps",
                ],
                "excludes": ["action_scale", "action_offset", "action_clip"],
            },
            "joint_actuator_type": {
                "shape": [29],
                "vocabulary": actuator_types,
            },
            "global_robot_information": {
                "dimension": GLOBAL_ROBOT_INFO_DIM,
                "fields": [
                    "sim_dt", "control_dt", "decimation", "gravity_w_x",
                    "gravity_w_y", "gravity_w_z", "solver_position_iterations",
                    "solver_velocity_iterations", "foot_contact_threshold_n",
                ],
            },
            "dynamics_context": {
                "dimension": DYNAMICS_CONTEXT_DIM,
                "fields": [
                    "body_mass", "body_inertia", "body_com", "body_material",
                    "ground_material",
                ],
                "default_visibility": "hidden",
            },
            "auxiliary_transition": {
                "dimension": AUXILIARY_TRANSITION_DIM,
                "fields": ["applied_joint_torque_mean", "foot_contact_impulse"],
                "model_input": False,
            },
            "action_before_window": {
                "dimension": 29,
                "definition": "actions[start-1] or initial_processed_target_canonical",
            },
            "known_action_queue": {
                "source": "causal_history_of_sent_processed_targets",
                "observed_max_delay_control_steps": observed_max_delay,
            },
            "reference_future": {
                "available": reference_available,
                "shape_per_transition": [REFERENCE_FRAMES, REFERENCE_DIM],
                "joint_pos_vel_dimension": 58,
                "root_orientation_dimension": 6,
                "source": (
                    "command_manager_runtime_observation"
                    if reference_available else None
                ),
                "model_visibility": "action_branch_only",
            },
            "token_layout": "A_before_S0_A0_S1_A1_to_ST",
        },
        "input_provenance": {
            "physical_state": "measured",
            "action": "known_command",
            "action_before_window": "known_command",
            "joint_robot_information": "configured_or_calibrated",
            "global_robot_information": "configured",
            "reference_future": (
                "known_runtime_command" if reference_available else "unavailable"
            ),
            "dynamics_context": "oracle_only",
            "auxiliary_transition": "supervision_only",
        },
        "input_provenance_classification": {
            "physical_state": "measured",
            "action_and_queue": "configured",
            "joint_robot_information": "configured",
            "global_robot_information": "configured",
            "reference_future": "configured" if reference_available else "unavailable",
            "causal_dynamics_embedding": "causally_estimated",
            "dynamics_context": "oracle_only",
            "auxiliary_transition": "supervision_only",
        },
        "normalization": {"path": str(normalization_path), "training_split_only": True, "gravity": "unit_vector", "contact": "binary"},
        "episodes_index_sha256": file_sha256(episodes_path),
    }
    atomic_write_json(output_run / "manifests/dataset_manifest.json", manifest)
    atomic_write_text(output_run / "markers/cvae_dataset.ok", "PASS\n")
    atomic_write_text(output_run / "markers/cvae_physics_dataset.ok", "PASS\n")
    return manifest


def _split_counts(value: str) -> tuple[int, int, int]:
    result = tuple(int(item) for item in value.split(","))
    if len(result) != 3 or min(result) < 0:
        raise argparse.ArgumentTypeError("split counts must be train,validation,test")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the SONIC Physics State-Action v3/v5 index")
    parser.add_argument("--source-run", action="append", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--expected-motions", type=int, default=768)
    parser.add_argument("--expected-episodes", type=int, default=6144)
    parser.add_argument("--split-counts", type=_split_counts, default=(616, 76, 76))
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    result = build_physics_index(
        args.source_run, args.output_run, args.expected_motions, args.expected_episodes,
        args.split_counts, args.seed
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
