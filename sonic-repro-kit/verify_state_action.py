#!/usr/bin/env python3
"""Validate and index a SONIC multi-motion, multi-variant collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


STATE_FIELDS = {
    "joint_pos": 29,
    "joint_vel": 29,
    "base_ang_vel": 3,
    "gravity_robot": 3,
    "previous_action": 29,
}
GOAL_FIELDS = {
    "reference_joint_pos": 29,
    "reference_joint_vel": 29,
    "gravity_reference": 3,
    "relative_heading": 2,
}
CONTEXT_FIELDS = {
    "reset_root_pose_delta": 6,
    "reset_root_velocity_delta": 6,
    "reset_joint_pos_delta": 29,
    "reset_joint_vel_delta": 29,
}
IDENTITY_FIELDS = (
    "env_id",
    "motion_id",
    "global_motion_id",
    "variant_id",
    "batch_id",
    "attempt_id",
    "motion_step",
)
CORE_TERMINATION_TERMS = (
    "anchor_pos",
    "anchor_ori_full",
    "ee_body_pos",
    "motion_time_out",
)
PACKAGES = (
    "Locomotion",
    "Communication",
    "Interactions",
    "Dances",
    "Gaming",
    "Everyday",
    "Sport",
    "Other",
)
MILD_LIMITS = {
    "reset_root_pose_delta": np.array([0.025, 0.025, 0.005, 0.05, 0.05, 0.1]),
    "reset_root_velocity_delta": np.array([0.25, 0.25, 0.1, 0.26, 0.26, 0.39]),
    "reset_joint_pos_delta": np.full(29, 0.05),
    "reset_joint_vel_delta": np.zeros(29),
}


def parameter_for_env(
    entry: dict[str, object], env_id: int, tail_shape: tuple[int, ...]
) -> np.ndarray:
    """Resolve a global or per-environment runtime parameter from schema metadata."""
    scope = entry.get("scope")
    values = np.asarray(entry.get("values"), dtype=np.float64)
    if scope == "global":
        expected_shape = tail_shape
        resolved = values
    elif scope == "per_environment":
        expected_shape = (values.shape[0], *tail_shape)
        if env_id < 0 or env_id >= values.shape[0]:
            raise ValueError(f"env_id={env_id} is outside per-environment parameter rows")
        resolved = values[env_id]
    else:
        raise ValueError(f"unsupported parameter scope: {scope!r}")
    if tuple(values.shape) != expected_shape:
        raise ValueError(f"expected parameter shape {expected_shape}, found {values.shape}")
    return resolved


def validate_parameter_entry(
    entry: object, tail_shape: tuple[int, ...]
) -> tuple[bool, str]:
    """Validate runtime parameter metadata without assuming a global scope."""
    if not isinstance(entry, dict):
        return False, f"expected object, found {type(entry).__name__}"
    scope = entry.get("scope")
    try:
        values = np.asarray(entry.get("values"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        return False, str(error)
    if scope == "global":
        shape_valid = values.shape == tail_shape
    elif scope == "per_environment":
        shape_valid = values.ndim == len(tail_shape) + 1 and values.shape[1:] == tail_shape
        shape_valid = shape_valid and values.shape[0] > 0
    else:
        return False, f"unsupported scope={scope!r}"
    finite = bool(np.isfinite(values).all())
    return bool(shape_valid and finite), f"scope={scope}, shape={values.shape}, finite={finite}"


def validate_runtime_entry(entry: object, num_envs: int) -> tuple[bool, str]:
    """Validate an uncompressed per-environment physics tensor."""
    if not isinstance(entry, dict):
        return False, f"expected object, found {type(entry).__name__}"
    try:
        values = np.asarray(entry.get("values"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        return False, str(error)
    passed = (
        entry.get("scope") == "per_environment"
        and values.ndim >= 2
        and values.shape[0] == num_envs
        and np.isfinite(values).all()
    )
    return bool(passed), f"scope={entry.get('scope')}, shape={values.shape}"


def load_motion_manifest(path: Path | None) -> tuple[dict[str, dict], dict[int, dict]]:
    """Load the JSONL subset manifest keyed by both motion key and stable ID."""
    if path is None:
        return {}, {}
    by_key: dict[str, dict] = {}
    by_id: dict[int, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            motion_id = int(entry["global_motion_id"])
            motion_key = str(entry["motion_key"])
            if motion_id in by_id or motion_key in by_key:
                raise ValueError(f"duplicate motion manifest entry at line {line_number}")
            by_id[motion_id] = entry
            by_key[motion_key] = entry
    return by_key, by_id


def file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _constant_scalar(episode: h5py.Group, path: str, steps: int) -> int:
    if path not in episode:
        raise ValueError(f"missing {path}")
    values = np.asarray(episode[path], dtype=np.int64)
    if values.shape != (steps,):
        raise ValueError(f"{path} must have shape {(steps,)}, found {values.shape}")
    if not np.all(values == values[0]):
        raise ValueError(f"{path} changes inside episode")
    return int(values[0])


def _runtime_signature(schema: dict, env_id: int) -> bytes:
    """Build an exact signature of startup-randomized parameters for one env."""
    arrays = [
        parameter_for_env(schema["default_joint_pos"], env_id, (29,)),
        parameter_for_env(schema["action_offset"], env_id, (29,)),
    ]
    runtime = schema["runtime_physics"]
    for name in ("body_com", "material_properties"):
        entry = runtime[name]
        values = np.asarray(entry["values"])
        arrays.append(parameter_for_env(entry, env_id, tuple(values.shape[1:])))
    return b"".join(np.ascontiguousarray(value, dtype=np.float64).tobytes() for value in arrays)


def _package_for_motion(
    global_motion_id: int,
    schema: dict,
    manifest_by_key: dict[str, dict],
) -> tuple[str, str]:
    motion_key = str(schema["motion_id_to_key"].get(str(global_motion_id), ""))
    manifest_entry = manifest_by_key.get(motion_key)
    package = str(manifest_entry.get("package", "unclassified")) if manifest_entry else "unclassified"
    return motion_key, package


def _completion_status(terminated: np.ndarray, truncated: np.ndarray) -> str:
    if bool(truncated[-1]) and not bool(terminated[-1]):
        return "completed"
    if bool(terminated[-1]):
        return "failed"
    return "interrupted"


def resolve_termination_schema(
    schema: dict,
) -> tuple[tuple[str, ...], bool, tuple[str, ...]]:
    """Resolve output terms and the narrowly-scoped legacy timeout fallback."""
    active_terms = schema.get("termination_terms")
    if (
        not isinstance(active_terms, list)
        or not all(isinstance(name, str) and name for name in active_terms)
        or len(set(active_terms)) != len(active_terms)
    ):
        raise ValueError("schema termination_terms must be a unique list of names")

    configured_mapping = schema.get("termination_term_mapping")
    if configured_mapping is not None:
        if (
            not isinstance(configured_mapping, dict)
            or not configured_mapping
            or not all(
                isinstance(output_name, str)
                and output_name
                and isinstance(runtime_name, str)
                and runtime_name
                for output_name, runtime_name in configured_mapping.items()
            )
        ):
            raise ValueError("termination_term_mapping must map non-empty strings")
        if len(set(configured_mapping.values())) != len(configured_mapping):
            raise ValueError("termination_term_mapping runtime names must be unique")
        if set(configured_mapping.values()) != set(active_terms):
            raise ValueError(
                "termination_term_mapping runtime names do not match termination_terms"
            )
        missing = sorted(set(CORE_TERMINATION_TERMS) - set(configured_mapping))
        if missing:
            raise ValueError(f"termination_term_mapping is missing core terms: {missing}")
        return tuple(configured_mapping), False, ()

    runtime_terms = set(active_terms)
    timeout_runtime_name = (
        "motion_time_out" if "motion_time_out" in runtime_terms else "time_out"
    )
    required_runtime_terms = {
        "anchor_pos",
        "anchor_ori_full",
        "ee_body_pos",
        timeout_runtime_name,
    }
    missing = sorted(required_runtime_terms - runtime_terms)
    if missing:
        raise ValueError(f"legacy schema is missing runtime termination terms: {missing}")
    legacy_timeout_recovery = (
        timeout_runtime_name == "time_out" and "motion_time_out" not in runtime_terms
    )
    unrecorded_runtime_terms = tuple(sorted(runtime_terms - required_runtime_terms))
    return CORE_TERMINATION_TERMS, legacy_timeout_recovery, unrecorded_runtime_terms


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="sonic_minimal_sa")
    parser.add_argument("--expected-motion-count", type=int, required=True)
    parser.add_argument("--expected-variants-per-motion", type=int, required=True)
    parser.add_argument("--variant-offset", type=int, default=0)
    parser.add_argument(
        "--randomization-profile",
        choices=("startup", "initial_state_mild"),
        required=True,
    )
    parser.add_argument("--motion-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    dataset_filename = (
        args.dataset_name if args.dataset_name.endswith(".hdf5") else f"{args.dataset_name}.hdf5"
    )
    dataset_path = run_dir / "data" / dataset_filename
    schema_path = run_dir / "manifests" / "state_action_schema.json"
    summary_path = run_dir / "manifests" / "collection_summary.json"
    canonical_index_path = run_dir / "manifests" / "canonical_episode_index.json"
    attempt_index_path = run_dir / "manifests" / "additional_attempt_index.json"
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, evidence: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    check("run_dir", run_dir.is_dir(), str(run_dir))
    check("dataset", dataset_path.is_file() and dataset_path.stat().st_size > 0, str(dataset_path))
    check("schema", schema_path.is_file() and schema_path.stat().st_size > 0, str(schema_path))
    if not dataset_path.is_file() or not schema_path.is_file():
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "passed": False,
            "checks": checks,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 1

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    manifest_path = args.motion_manifest
    if manifest_path is None:
        configured_manifest = schema.get("motion_collection", {}).get("motion_manifest")
        if configured_manifest:
            manifest_path = Path(configured_manifest)
    if manifest_path is not None:
        manifest_path = manifest_path.expanduser().resolve()
    try:
        manifest_by_key, manifest_by_id = load_motion_manifest(manifest_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        manifest_by_key, manifest_by_id = {}, {}
        check("motion_manifest", False, str(error))
    else:
        manifest_ok = manifest_path is None or len(manifest_by_id) == args.expected_motion_count
        evidence = (
            "not supplied"
            if manifest_path is None
            else f"{manifest_path}: {len(manifest_by_id)} entries"
        )
        check("motion_manifest", manifest_ok, evidence)

    check(
        "schema_version",
        schema.get("schema_version") == "sonic_minimal_sa_v2",
        str(schema.get("schema_version")),
    )
    check(
        "schema_dimensions",
        schema.get("dimensions") == {"state": 93, "goal": 63, "action": 29},
        str(schema.get("dimensions")),
    )
    check(
        "action_term_type",
        schema.get("action_term_type") == "JointPositionAction",
        str(schema.get("action_term_type")),
    )
    check(
        "wrapper_action_transform",
        schema.get("wrapper_action_transform_enabled") is False,
        str(schema.get("wrapper_action_transform_enabled")),
    )
    joint_names = schema.get("joint_names", [])
    check(
        "joint_names",
        len(joint_names) == 29 and len(set(joint_names)) == 29,
        f"count={len(joint_names)}",
    )
    for name in ("default_joint_pos", "default_joint_vel", "action_scale", "action_offset"):
        valid, evidence = validate_parameter_entry(schema.get(name), (29,))
        check(name, valid, evidence)
    if schema.get("action_clip") is not None:
        clip_valid, clip_evidence = validate_parameter_entry(
            schema.get("action_clip"), (29, 2)
        )
        check("action_clip", clip_valid, clip_evidence)
    check(
        "control_dt",
        np.isclose(float(schema.get("control_dt", -1.0)), 0.02),
        str(schema.get("control_dt")),
    )
    check(
        "sim_dt",
        np.isclose(float(schema.get("sim_dt", -1.0)), 0.005),
        str(schema.get("sim_dt")),
    )

    collection = schema.get("motion_collection", {})
    check(
        "eval_motion_repeat",
        collection.get("eval_motion_repeat") == args.expected_variants_per_motion,
        str(collection.get("eval_motion_repeat")),
    )
    check(
        "eval_require_full_batch",
        collection.get("eval_require_full_batch") is True,
        str(collection.get("eval_require_full_batch")),
    )
    check(
        "variant_offset",
        collection.get("variant_offset") == args.variant_offset,
        str(collection.get("variant_offset")),
    )
    check(
        "randomization_profile",
        collection.get("randomization_profile") == args.randomization_profile,
        str(collection.get("randomization_profile")),
    )
    motion_key_map = schema.get("motion_id_to_key", {})
    check(
        "motion_key_map",
        len(motion_key_map) == args.expected_motion_count,
        f"count={len(motion_key_map)}",
    )
    if manifest_path is not None and manifest_by_id:
        manifest_identity_ok = set(manifest_by_id) == set(range(args.expected_motion_count))
        manifest_identity_ok = manifest_identity_ok and all(
            str(manifest_by_id[motion_id]["motion_key"])
            == str(motion_key_map.get(str(motion_id)))
            for motion_id in range(args.expected_motion_count)
        )
        check(
            "manifest_global_identity",
            manifest_identity_ok,
            "manifest IDs and runtime motion keys must match exactly",
        )
        package_counts = Counter(
            str(entry.get("package", "")) for entry in manifest_by_id.values()
        )
        expected_per_package = args.expected_motion_count // len(PACKAGES)
        package_balance_ok = (
            args.expected_motion_count % len(PACKAGES) == 0
            and set(package_counts) == set(PACKAGES)
            and all(package_counts[name] == expected_per_package for name in PACKAGES)
        )
        check(
            "manifest_package_balance",
            package_balance_ok,
            str(dict(package_counts)),
        )
    runtime_physics = schema.get("runtime_physics", {})
    body_com_entry = runtime_physics.get("body_com")
    material_entry = runtime_physics.get("material_properties")
    num_runtime_envs = len(collection.get("env_to_variant", []))
    body_com_valid, body_com_evidence = validate_runtime_entry(
        body_com_entry, num_runtime_envs
    )
    material_valid, material_evidence = validate_runtime_entry(
        material_entry, num_runtime_envs
    )
    check(
        "runtime_body_com",
        body_com_valid,
        body_com_evidence,
    )
    check(
        "runtime_material",
        material_valid,
        material_evidence,
    )

    try:
        (
            termination_output_terms,
            legacy_timeout_recovery,
            legacy_unrecorded_runtime_terms,
        ) = resolve_termination_schema(schema)
    except ValueError as error:
        termination_output_terms = CORE_TERMINATION_TERMS
        legacy_timeout_recovery = False
        legacy_unrecorded_runtime_terms = ()
        check("termination_term_mapping", False, str(error))
    else:
        if legacy_timeout_recovery:
            mapping_evidence = (
                "legacy schema: recover motion_time_out from outcome/truncated; "
                f"unrecorded runtime terms={list(legacy_unrecorded_runtime_terms)}"
            )
        else:
            mapping_evidence = f"output terms={list(termination_output_terms)}"
        check("termination_term_mapping", True, mapping_evidence)

    episodes: list[dict[str, object]] = []
    canonical_by_pair: dict[tuple[int, int], dict] = {}
    reference_by_motion: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    reset_context_by_pair: dict[tuple[int, int], bytes] = {}
    motion_ids_seen: set[int] = set()
    legacy_timeout_recovered_episodes = 0
    try:
        with h5py.File(dataset_path, "r") as stream:
            if "data" not in stream:
                raise ValueError("HDF5 file does not contain top-level 'data' group")
            data_group = stream["data"]
            for episode_name in sorted(data_group.keys()):
                episode = data_group[episode_name]
                if "actions" not in episode:
                    raise ValueError(f"{episode_name}: missing actions")
                actions = np.asarray(episode["actions"])
                if actions.ndim != 2 or actions.shape[1] != 29 or actions.shape[0] == 0:
                    raise ValueError(f"{episode_name}: invalid actions shape {actions.shape}")
                if not np.isfinite(actions).all():
                    raise ValueError(f"{episode_name}: actions contain non-finite values")
                wrapper_clip = schema.get("wrapper_action_clip")
                if wrapper_clip is not None and float(wrapper_clip) > 0:
                    max_action = float(np.max(np.abs(actions)))
                    if max_action > float(wrapper_clip) + 1e-6:
                        raise ValueError(
                            f"{episode_name}: action exceeds wrapper clip "
                            f"({max_action} > {wrapper_clip})"
                        )
                steps = actions.shape[0]
                if int(episode.attrs.get("num_samples", -1)) != steps:
                    raise ValueError(f"{episode_name}: num_samples does not match actions")

                groups = (
                    ("state_t", STATE_FIELDS),
                    ("state_tp1", STATE_FIELDS),
                    ("goal_t", GOAL_FIELDS),
                    ("context_t", CONTEXT_FIELDS),
                )
                for group_name, fields in groups:
                    if group_name not in episode:
                        raise ValueError(f"{episode_name}: missing {group_name}")
                    for field_name, width in fields.items():
                        path = f"{group_name}/{field_name}"
                        if path not in episode:
                            raise ValueError(f"{episode_name}: missing {path}")
                        values = np.asarray(episode[path])
                        if values.shape != (steps, width) or not np.isfinite(values).all():
                            raise ValueError(
                                f"{episode_name}: invalid {path} shape/data {values.shape}"
                            )

                identity = {
                    name: _constant_scalar(episode, f"motion/{name}", steps)
                    for name in IDENTITY_FIELDS
                    if name != "motion_step"
                }
                motion_steps = np.asarray(episode["motion/motion_step"], dtype=np.int64)
                if motion_steps.shape != (steps,) or (
                    steps > 1 and np.any(np.diff(motion_steps) < 0)
                ):
                    raise ValueError(f"{episode_name}: invalid motion_step sequence")
                env_id = identity["env_id"]
                expected_variant = collection["env_to_variant"][env_id]
                expected_slot = collection["env_to_motion_slot"][env_id]
                if identity["variant_id"] != expected_variant:
                    raise ValueError(f"{episode_name}: env-to-variant mapping mismatch")
                if identity["motion_id"] != expected_slot:
                    raise ValueError(f"{episode_name}: env-to-motion-slot mapping mismatch")
                batch_motion_count = max(collection["env_to_motion_slot"]) + 1
                expected_global_id = identity["batch_id"] * batch_motion_count + expected_slot
                if identity["global_motion_id"] != expected_global_id:
                    raise ValueError(
                        f"{episode_name}: batch start/global motion identity mismatch"
                    )
                motion_ids_seen.add(identity["global_motion_id"])

                terminated = np.asarray(episode["outcome/terminated"], dtype=bool)
                truncated = np.asarray(episode["outcome/truncated"], dtype=bool)
                if terminated.shape != (steps,) or truncated.shape != (steps,):
                    raise ValueError(f"{episode_name}: invalid outcome shape")
                done = terminated | truncated
                if done[:-1].any():
                    raise ValueError(f"{episode_name}: terminal marker before final frame")
                for term_name in termination_output_terms:
                    path = f"outcome/termination_terms/{term_name}"
                    if path in episode:
                        term_values = np.asarray(episode[path])
                    elif term_name == "motion_time_out" and legacy_timeout_recovery:
                        term_values = truncated
                        legacy_timeout_recovered_episodes += 1
                    else:
                        raise ValueError(f"{episode_name}: missing or invalid {path}")
                    if term_values.shape != (steps,):
                        raise ValueError(f"{episode_name}: missing or invalid {path}")

                previous_action = np.asarray(episode["state_t/previous_action"])
                next_previous_action = np.asarray(episode["state_tp1/previous_action"])
                if not np.allclose(previous_action[0], 0.0, rtol=1e-5, atol=1e-6):
                    raise ValueError(f"{episode_name}: first previous_action is not zero")
                if steps > 1 and not np.allclose(
                    previous_action[1:], actions[:-1], rtol=1e-5, atol=1e-6
                ):
                    raise ValueError(f"{episode_name}: previous_action shift failed")
                if not np.allclose(next_previous_action, actions, rtol=1e-5, atol=1e-6):
                    raise ValueError(f"{episode_name}: state_tp1 previous_action failed")
                for field_name in STATE_FIELDS:
                    if steps > 1 and not np.allclose(
                        episode[f"state_tp1/{field_name}"][:-1],
                        episode[f"state_t/{field_name}"][1:],
                        rtol=1e-4,
                        atol=1e-5,
                    ):
                        raise ValueError(
                            f"{episode_name}: state continuity failed for {field_name}"
                        )
                norm_paths = (
                    "state_t/gravity_robot",
                    "state_tp1/gravity_robot",
                    "goal_t/gravity_reference",
                    "goal_t/relative_heading",
                )
                for path in norm_paths:
                    if not np.allclose(
                        np.linalg.norm(episode[path], axis=-1),
                        1.0,
                        rtol=1e-4,
                        atol=1e-4,
                    ):
                        raise ValueError(f"{episode_name}: {path} is not unit length")

                for context_name, limit in MILD_LIMITS.items():
                    context = np.asarray(
                        episode[f"context_t/{context_name}"], dtype=np.float64
                    )
                    if not np.allclose(context, context[0], rtol=0.0, atol=1e-7):
                        raise ValueError(f"{episode_name}: {context_name} changes inside episode")
                    expected_limit = (
                        limit
                        if args.randomization_profile == "initial_state_mild"
                        else np.zeros_like(limit)
                    )
                    if np.any(np.abs(context[0]) > expected_limit + 1e-6):
                        raise ValueError(f"{episode_name}: {context_name} exceeds profile range")

                default_pos = parameter_for_env(schema["default_joint_pos"], env_id, (29,))
                scale = parameter_for_env(schema["action_scale"], env_id, (29,))
                offset = parameter_for_env(schema["action_offset"], env_id, (29,))
                processed = actions * scale + offset
                clip_entry = schema.get("action_clip")
                if clip_entry is not None:
                    clip = parameter_for_env(clip_entry, env_id, (29, 2))
                    processed = np.clip(processed, clip[:, 0], clip[:, 1])
                if not np.isfinite(processed).all():
                    raise ValueError(
                        f"{episode_name}: reconstructed action target is non-finite"
                    )
                if clip_entry is None and not np.allclose(
                    processed - default_pos,
                    actions * scale,
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    raise ValueError(f"{episode_name}: relative action mapping failed")

                motion_key, package = _package_for_motion(
                    identity["global_motion_id"], schema, manifest_by_key
                )
                status = _completion_status(terminated, truncated)
                item = {
                    "episode": episode_name,
                    "steps": steps,
                    **identity,
                    "motion_key": motion_key,
                    "package": package,
                    "status": status,
                    "terminated": bool(terminated[-1]),
                    "truncated": bool(truncated[-1]),
                }
                episodes.append(item)
                if identity["attempt_id"] == 0:
                    if not done[-1]:
                        raise ValueError(f"{episode_name}: canonical episode is incomplete")
                    pair = (identity["global_motion_id"], identity["variant_id"])
                    if pair in canonical_by_pair:
                        raise ValueError(f"duplicate canonical pair {pair}")
                    canonical_by_pair[pair] = item
                    reset_context_by_pair[pair] = b"".join(
                        np.ascontiguousarray(
                            episode[f"context_t/{context_name}"][0], dtype=np.float64
                        ).tobytes()
                        for context_name in CONTEXT_FIELDS
                    )
                    absolute_reference = (
                        np.asarray(episode["goal_t/reference_joint_pos"]) + default_pos
                    )
                    if identity["global_motion_id"] in reference_by_motion:
                        prior_steps, prior_reference = reference_by_motion[
                            identity["global_motion_id"]
                        ]
                        common = min(len(prior_steps), len(motion_steps))
                        same_steps = np.array_equal(
                            prior_steps[:common], motion_steps[:common]
                        )
                        same_reference = np.allclose(
                            prior_reference[:common],
                            absolute_reference[:common],
                            rtol=1e-5,
                            atol=1e-5,
                        )
                        if not same_steps or not same_reference:
                            raise ValueError(
                                f"{episode_name}: absolute reference differs across variants"
                            )
                    else:
                        reference_by_motion[identity["global_motion_id"]] = (
                            motion_steps,
                            absolute_reference,
                        )
            if legacy_timeout_recovery and legacy_timeout_recovered_episodes not in (
                0,
                len(episodes),
            ):
                raise ValueError(
                    "legacy timeout recovery is inconsistently required across episodes"
                )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        check("hdf5_validation", False, str(error))
    else:
        check("hdf5_validation", True, f"validated {len(episodes)} exported episodes")
        if legacy_timeout_recovery:
            check(
                "legacy_timeout_recovery",
                legacy_timeout_recovered_episodes == len(episodes),
                f"recovered {legacy_timeout_recovered_episodes}/{len(episodes)} episodes "
                "from outcome/truncated without rewriting HDF5",
            )

    expected_pairs = {
        (motion_id, variant_id)
        for motion_id in range(args.expected_motion_count)
        for variant_id in range(
            args.variant_offset,
            args.variant_offset + args.expected_variants_per_motion,
        )
    }
    actual_pairs = set(canonical_by_pair)
    check(
        "canonical_episode_count",
        len(actual_pairs) == len(expected_pairs),
        f"actual={len(actual_pairs)}, expected={len(expected_pairs)}",
    )
    check(
        "canonical_pair_coverage",
        actual_pairs == expected_pairs,
        f"missing={sorted(expected_pairs - actual_pairs)[:10]}, "
        f"extra={sorted(actual_pairs - expected_pairs)[:10]}",
    )
    check(
        "distinct_global_motions",
        motion_ids_seen == set(range(args.expected_motion_count)),
        f"count={len(motion_ids_seen)}",
    )

    if (
        args.randomization_profile == "startup"
        and args.expected_variants_per_motion > 1
        and canonical_by_pair
        and body_com_valid
        and material_valid
    ):
        diversity_failures = []
        for motion_id in range(args.expected_motion_count):
            signatures = {
                _runtime_signature(
                    schema,
                    int(canonical_by_pair[(motion_id, variant_id)]["env_id"]),
                )
                for variant_id in range(
                    args.variant_offset,
                    args.variant_offset + args.expected_variants_per_motion,
                )
                if (motion_id, variant_id) in canonical_by_pair
            }
            if len(signatures) < 2:
                diversity_failures.append(motion_id)
        check(
            "startup_randomization_diversity",
            not diversity_failures,
            f"motions_without_diversity={diversity_failures[:10]}",
        )
    if args.randomization_profile == "initial_state_mild" and canonical_by_pair:
        mild_diversity_failures = []
        for motion_id in range(args.expected_motion_count):
            signatures = {
                reset_context_by_pair[(motion_id, variant_id)]
                for variant_id in range(
                    args.variant_offset,
                    args.variant_offset + args.expected_variants_per_motion,
                )
                if (motion_id, variant_id) in reset_context_by_pair
            }
            if len(signatures) != args.expected_variants_per_motion:
                mild_diversity_failures.append(motion_id)
        check(
            "mild_reset_randomization_diversity",
            not mild_diversity_failures,
            f"motions_without_distinct_resets={mild_diversity_failures[:10]}",
        )

    canonical = [canonical_by_pair[pair] for pair in sorted(canonical_by_pair)]
    additional_attempts = [item for item in episodes if int(item["attempt_id"]) > 0]
    completion_by_package: dict[str, dict[str, float | int]] = {}
    package_rows: dict[str, list[dict]] = defaultdict(list)
    for item in canonical:
        package_rows[str(item["package"])].append(item)
    for package, items in sorted(package_rows.items()):
        completed = sum(item["status"] == "completed" for item in items)
        completion_by_package[package] = {
            "canonical_episodes": len(items),
            "completed": completed,
            "completion_rate": completed / len(items),
        }
    completed_total = sum(item["status"] == "completed" for item in canonical)
    completion_rate = completed_total / len(canonical) if canonical else 0.0

    canonical_index_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_index_path.write_text(
        json.dumps({"episodes": canonical}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    attempt_index_path.write_text(
        json.dumps({"episodes": additional_attempts}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    passed = all(bool(item["passed"]) for item in checks)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "dataset": str(dataset_path),
        "passed": passed,
        "randomization_profile": args.randomization_profile,
        "canonical_episode_count": len(canonical),
        "additional_attempt_count": len(additional_attempts),
        "legacy_timeout_recovery": {
            "applied": legacy_timeout_recovered_episodes > 0,
            "source": "outcome/truncated" if legacy_timeout_recovered_episodes > 0 else None,
            "episode_count": legacy_timeout_recovered_episodes,
            "unrecorded_runtime_terms": list(legacy_unrecorded_runtime_terms),
        },
        "completed_canonical_episodes": completed_total,
        "completion_rate": completion_rate,
        "completion_by_package": completion_by_package,
        "motion_manifest": str(manifest_path) if manifest_path is not None else None,
        "motion_manifest_sha256": file_sha256(manifest_path),
        "canonical_index": str(canonical_index_path),
        "additional_attempt_index": str(attempt_index_path),
        "checks": checks,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"SONIC multi-variant state-action dataset: {'PASS' if passed else 'FAIL'}")
    for item in checks:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{status}] {item['name']}: {item['evidence']}")
    print(f"Completion: {completed_total}/{len(canonical)} ({completion_rate:.1%})")
    print(f"Summary: {summary_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
