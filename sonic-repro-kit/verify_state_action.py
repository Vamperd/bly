#!/usr/bin/env python3
"""Validate a SONIC minimal state-goal-action HDF5 dataset."""

from __future__ import annotations

import argparse
import json
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="sonic_minimal_sa")
    parser.add_argument("--expected-min-episodes", type=int, default=1)
    parser.add_argument("--expected-min-motions", type=int, default=1)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    dataset_filename = (
        args.dataset_name if args.dataset_name.endswith(".hdf5") else f"{args.dataset_name}.hdf5"
    )
    dataset_path = run_dir / "data" / dataset_filename
    schema_path = run_dir / "manifests" / "state_action_schema.json"
    summary_path = run_dir / "manifests" / "collection_summary.json"
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
    check(
        "schema_version",
        schema.get("schema_version") == "sonic_minimal_sa_v1",
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
    default_pos_valid, default_pos_evidence = validate_parameter_entry(
        schema.get("default_joint_pos"), (29,)
    )
    check("default_joint_pos", default_pos_valid, default_pos_evidence)
    default_vel_valid, default_vel_evidence = validate_parameter_entry(
        schema.get("default_joint_vel"), (29,)
    )
    check("default_joint_vel", default_vel_valid, default_vel_evidence)
    check(
        "control_dt",
        np.isclose(float(schema.get("control_dt", -1.0)), 0.02),
        str(schema.get("control_dt")),
    )
    check("sim_dt", np.isclose(float(schema.get("sim_dt", -1.0)), 0.005), str(schema.get("sim_dt")))

    episode_summaries: list[dict[str, object]] = []
    motion_ids_seen: set[int] = set()
    try:
        with h5py.File(dataset_path, "r") as stream:
            if "data" not in stream:
                raise ValueError("HDF5 file does not contain top-level 'data' group")
            data_group = stream["data"]
            episode_names = sorted(data_group.keys())
            check(
                "episode_count",
                len(episode_names) >= args.expected_min_episodes,
                f"count={len(episode_names)}",
            )

            for episode_name in episode_names:
                episode = data_group[episode_name]
                if "actions" not in episode:
                    raise ValueError(f"{episode_name}: missing actions")
                actions = np.asarray(episode["actions"])
                if actions.ndim != 2 or actions.shape[1] != 29:
                    raise ValueError(
                        f"{episode_name}: actions shape is {actions.shape}, expected (T, 29)"
                    )
                steps = actions.shape[0]
                if steps == 0:
                    raise ValueError(f"{episode_name}: empty episode")
                if not np.isfinite(actions).all():
                    raise ValueError(f"{episode_name}: actions contain non-finite values")
                wrapper_clip = schema.get("wrapper_action_clip")
                if wrapper_clip is not None and float(wrapper_clip) > 0:
                    max_action = float(np.max(np.abs(actions)))
                    if max_action > float(wrapper_clip) + 1.0e-6:
                        raise ValueError(
                            f"{episode_name}: raw action exceeds wrapper clip; "
                            f"max={max_action}, clip={wrapper_clip}"
                        )
                if int(episode.attrs.get("num_samples", -1)) != steps:
                    raise ValueError(
                        f"{episode_name}: num_samples attribute does not match actions"
                    )

                groups_and_fields = (
                    ("state_t", STATE_FIELDS),
                    ("state_tp1", STATE_FIELDS),
                    ("goal_t", GOAL_FIELDS),
                )
                for group_name, fields in groups_and_fields:
                    if group_name not in episode:
                        raise ValueError(f"{episode_name}: missing {group_name}")
                    group = episode[group_name]
                    for field_name, width in fields.items():
                        if field_name not in group:
                            raise ValueError(f"{episode_name}: missing {group_name}/{field_name}")
                        array = np.asarray(group[field_name])
                        if array.shape != (steps, width):
                            raise ValueError(
                                f"{episode_name}: {group_name}/{field_name} shape {array.shape}, "
                                f"expected {(steps, width)}"
                            )
                        if not np.isfinite(array).all():
                            raise ValueError(
                                f"{episode_name}: non-finite values in "
                                f"{group_name}/{field_name}"
                            )

                scalar_paths = (
                    "outcome/terminated",
                    "outcome/truncated",
                    "motion/env_id",
                    "motion/motion_id",
                    "motion/motion_step",
                )
                for path in scalar_paths:
                    if path not in episode:
                        raise ValueError(f"{episode_name}: missing {path}")
                    if np.asarray(episode[path]).shape != (steps,):
                        raise ValueError(f"{episode_name}: {path} must have shape {(steps,)}")

                state_t = episode["state_t"]
                state_tp1 = episode["state_tp1"]
                previous_action = np.asarray(state_t["previous_action"])
                next_previous_action = np.asarray(state_tp1["previous_action"])
                if not np.allclose(previous_action[0], 0.0, rtol=1.0e-5, atol=1.0e-6):
                    raise ValueError(f"{episode_name}: first previous_action is not zero")
                if steps > 1 and not np.allclose(
                    previous_action[1:], actions[:-1], rtol=1.0e-5, atol=1.0e-6
                ):
                    raise ValueError(
                        f"{episode_name}: previous_action is not actions shifted by one frame"
                    )
                if not np.allclose(next_previous_action, actions, rtol=1.0e-5, atol=1.0e-6):
                    raise ValueError(
                        f"{episode_name}: state_tp1.previous_action does not equal actions"
                    )

                terminated = np.asarray(episode["outcome/terminated"], dtype=bool)
                truncated = np.asarray(episode["outcome/truncated"], dtype=bool)
                done = terminated | truncated
                if not done[-1]:
                    raise ValueError(
                        f"{episode_name}: exported episode does not end with a terminal frame"
                    )
                if done[:-1].any():
                    raise ValueError(
                        f"{episode_name}: terminal marker appears before the final frame"
                    )
                for field_name in STATE_FIELDS:
                    if steps <= 1:
                        continue
                    lhs = np.asarray(state_tp1[field_name][:-1])
                    rhs = np.asarray(state_t[field_name][1:])
                    if not np.allclose(lhs, rhs, rtol=1.0e-4, atol=1.0e-5):
                        max_error = float(np.max(np.abs(lhs - rhs)))
                        raise ValueError(
                            f"{episode_name}: state continuity failed for {field_name}; "
                            f"max_error={max_error}"
                        )

                gravity_paths = (
                    "state_t/gravity_robot",
                    "state_tp1/gravity_robot",
                    "goal_t/gravity_reference",
                )
                for path in gravity_paths:
                    norms = np.linalg.norm(np.asarray(episode[path]), axis=-1)
                    if not np.allclose(norms, 1.0, rtol=1.0e-4, atol=1.0e-4):
                        raise ValueError(f"{episode_name}: {path} is not unit length")
                heading_norms = np.linalg.norm(
                    np.asarray(episode["goal_t/relative_heading"]), axis=-1
                )
                if not np.allclose(heading_norms, 1.0, rtol=1.0e-4, atol=1.0e-4):
                    raise ValueError(f"{episode_name}: relative_heading is not unit length")

                env_ids = np.asarray(episode["motion/env_id"], dtype=np.int64)
                motion_ids = np.asarray(episode["motion/motion_id"], dtype=np.int64)
                motion_steps = np.asarray(episode["motion/motion_step"], dtype=np.int64)
                if not np.all(env_ids == env_ids[0]):
                    raise ValueError(f"{episode_name}: env_id changes inside episode")
                if not np.all(motion_ids == motion_ids[0]):
                    raise ValueError(f"{episode_name}: motion_id changes inside episode")
                if steps > 1 and np.any(np.diff(motion_steps) < 0):
                    raise ValueError(f"{episode_name}: motion_step moves backwards")
                env_id = int(env_ids[0])
                motion_id = int(motion_ids[0])
                motion_ids_seen.add(motion_id)

                default_joint_pos = parameter_for_env(
                    schema["default_joint_pos"], env_id, (29,)
                )
                scale = parameter_for_env(schema["action_scale"], env_id, (29,))
                offset = parameter_for_env(schema["action_offset"], env_id, (29,))
                processed = actions * scale + offset
                clip_entry = schema.get("action_clip")
                if clip_entry is not None:
                    clip = parameter_for_env(clip_entry, env_id, (29, 2))
                    processed = np.clip(processed, clip[:, 0], clip[:, 1])
                if not np.isfinite(processed).all():
                    raise ValueError(
                        f"{episode_name}: reconstructed processed actions are non-finite"
                    )
                if clip_entry is None and not np.allclose(
                    processed - default_joint_pos, actions * scale, rtol=1.0e-5, atol=1.0e-6
                ):
                    raise ValueError(
                        f"{episode_name}: relative action target reconstruction failed"
                    )

                episode_summaries.append(
                    {
                        "episode": episode_name,
                        "steps": steps,
                        "env_id": env_id,
                        "motion_id": motion_id,
                        "terminated": bool(terminated[-1]),
                        "truncated": bool(truncated[-1]),
                    }
                )
    except (OSError, ValueError, KeyError, TypeError) as error:
        check("hdf5_validation", False, str(error))
    else:
        check("hdf5_validation", True, f"validated {len(episode_summaries)} episodes")

    check(
        "distinct_motions",
        len(motion_ids_seen) >= args.expected_min_motions,
        f"count={len(motion_ids_seen)}, ids={sorted(motion_ids_seen)}",
    )
    passed = all(bool(item["passed"]) for item in checks)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "dataset": str(dataset_path),
        "passed": passed,
        "episode_count": len(episode_summaries),
        "distinct_motion_ids": sorted(motion_ids_seen),
        "episodes": episode_summaries,
        "checks": checks,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SONIC minimal state-action dataset: {'PASS' if passed else 'FAIL'}")
    for item in checks:
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['name']}: {item['evidence']}")
    print(f"Summary: {summary_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
