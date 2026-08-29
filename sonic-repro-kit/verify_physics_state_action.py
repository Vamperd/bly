from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from consolidate_physics_state_action import ACTION_FIELDS, REPLAY_FIELDS, STATE_FIELDS


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parameter(entry: dict[str, Any], env_id: int) -> np.ndarray:
    values = np.asarray(entry["values"], dtype=np.float64)
    if entry["scope"] == "global":
        return values
    if entry["scope"] == "per_environment":
        return values[env_id]
    raise ValueError(f"unsupported scope {entry.get('scope')!r}")


def _scalar(group: h5py.Group, name: str) -> int:
    values = np.asarray(group[name])
    if values.shape != ():
        raise ValueError(f"{group.name}/{name}: expected scalar, found {values.shape}")
    return int(values)


def _manifest_rows(path: Path | None) -> tuple[dict[str, dict], str | None]:
    if path is None:
        return {}, None
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {str(row["motion_key"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("motion manifest contains duplicate motion_key values")
    return result, _sha256(path)


def verify(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    schema_path = run_dir / "manifests" / "physics_state_action_schema.json"
    summary_path = run_dir / "manifests" / "collection_summary.json"
    canonical_path = run_dir / "manifests" / "canonical_episode_index.json"
    attempts_path = run_dir / "manifests" / "additional_attempt_index.json"
    schema = _load_json(schema_path)
    schema_version = schema.get("schema_version")
    if schema_version not in {"sonic_physics_sa_v3", "sonic_physics_sa_v5"}:
        raise ValueError(f"unsupported Physics schema {schema_version!r}")
    reference_available = schema_version == "sonic_physics_sa_v5"
    dataset_path = run_dir / "data" / f"{schema_version}.hdf5"
    manifest_path = args.motion_manifest.expanduser().resolve() if args.motion_manifest else None
    manifest, manifest_sha = _manifest_rows(manifest_path)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    check("schema_version", schema_version in {"sonic_physics_sa_v3", "sonic_physics_sa_v5"}, str(schema_version))
    check("dimensions", schema.get("dimensions") == {"state": 70, "action": 29}, str(schema.get("dimensions")))
    check("storage_no_duplicates", schema.get("storage", {}).get("state_tp1_duplicate") is False, str(schema.get("storage")))
    check("action_term", schema.get("action_term_type") == "JointPositionAction", str(schema.get("action_term_type")))
    check("wrapper_transform", schema.get("wrapper_action_transform_enabled") is False, str(schema.get("wrapper_action_transform_enabled")))
    check("timing", np.isclose(schema["simulation"]["sim_dt"], 0.005) and np.isclose(schema["simulation"]["control_dt"], 0.02) and int(schema["simulation"]["decimation"]) == 4, str(schema["simulation"]))
    check("joint_names", len(schema.get("joint_names", [])) == 29 and len(set(schema["joint_names"])) == 29, f"count={len(schema.get('joint_names', []))}")
    active_events = {
        name
        for names in schema.get("active_events", {}).values()
        for name in names
    }
    forbidden_wrench_events = {"push_robot", "compliance_force_push"} & active_events
    check(
        "external_wrench_events_disabled",
        not forbidden_wrench_events,
        f"active={sorted(active_events)}",
    )

    motion_key_map = {int(key): str(value) for key, value in schema["motion_id_to_key"].items()}
    expected_pairs = {
        (motion_id, variant_id)
        for motion_id in range(args.expected_motion_count)
        for variant_id in range(args.variant_offset, args.variant_offset + args.expected_variants_per_motion)
    }
    canonical: dict[tuple[int, int], dict[str, Any]] = {}
    additional: list[dict[str, Any]] = []
    completion_by_package: dict[str, list[bool]] = defaultdict(list)
    max_roundtrip = 0.0
    episode_count = 0
    state_count = 0
    context_ids: set[str] = set()
    try:
        with h5py.File(dataset_path, "r") as stream:
            if stream.attrs.get("schema_version") != schema_version:
                raise ValueError("HDF5 schema_version attribute differs")
            expected_contexts = len(schema["motion_collection"]["env_to_variant"])
            if len(stream["contexts"]) != expected_contexts:
                raise ValueError(
                    f"expected {expected_contexts} contexts, found {len(stream['contexts'])}"
                )
            body_count = len(schema["body_names"])
            for context_id, context in stream["contexts"].items():
                context_ids.add(context_id)
                for dataset in context.values():
                    if not np.isfinite(np.asarray(dataset)).all():
                        raise ValueError(f"{dataset.name}: NaN/Inf")
                expected_shapes = {
                    "runtime_default_joint_pos": (29,),
                    "action_offset": (29,),
                    "joint_position_limits": (29, 2),
                    "joint_velocity_limits": (29,),
                    "joint_effort_limits": (29,),
                    "joint_stiffness": (29,),
                    "joint_damping": (29,),
                    "joint_armature": (29,),
                    "joint_friction": (29,),
                    "body_mass": (body_count,),
                    "body_inertia": (body_count, 9),
                    "body_com": (body_count, 7),
                    "ground_material": (3,),
                }
                for name, shape in expected_shapes.items():
                    if name not in context or context[name].shape != shape:
                        found = context[name].shape if name in context else None
                        raise ValueError(
                            f"{context.name}/{name}: expected {shape}, found {found}"
                        )
                if (
                    "body_material" not in context
                    or context["body_material"].ndim != 2
                    or context["body_material"].shape[1] != 3
                ):
                    raise ValueError(f"{context.name}/body_material: invalid shape")
                env_id = int(context.attrs["env_id"])
                expected_variant = int(
                    schema["motion_collection"]["env_to_variant"][env_id]
                )
                if int(context.attrs["variant_id"]) != expected_variant:
                    raise ValueError(f"{context.name}: variant_id mismatch")
                if int(context.attrs["startup_randomization_seed"]) != int(
                    schema["motion_collection"]["seed"]
                ):
                    raise ValueError(f"{context.name}: seed mismatch")
            for episode_name in sorted(stream["data"].keys()):
                episode = stream[f"data/{episode_name}"]
                steps = int(episode.attrs["num_transitions"])
                if int(episode.attrs["num_samples"]) != steps:
                    raise ValueError(f"{episode.name}: num_samples changed")
                context_id = str(episode.attrs["context_id"])
                if context_id not in context_ids:
                    raise ValueError(f"{episode.name}: missing context {context_id}")
                context = stream[f"contexts/{context_id}"]
                env_id = int(context.attrs["env_id"])
                arrays = []
                for name, width in STATE_FIELDS.items():
                    values = np.asarray(episode[f"states/{name}"])
                    if values.shape != (steps + 1, width):
                        raise ValueError(f"{episode.name}/states/{name}: {values.shape}")
                    if not np.isfinite(values).all():
                        raise ValueError(f"{episode.name}/states/{name}: NaN/Inf")
                    arrays.append(values.astype(np.float64))
                if np.concatenate(arrays, axis=1).shape != (steps + 1, 70):
                    raise ValueError(f"{episode.name}: State is not 70-D")
                state_count += steps + 1
                gravity = np.asarray(episode["states/gravity_robot"], dtype=np.float64)
                if not np.allclose(np.linalg.norm(gravity, axis=1), 1.0, atol=2.0e-3):
                    raise ValueError(f"{episode.name}: gravity is not unit length")
                contact = np.asarray(episode["states/foot_contact"])
                if not np.isin(contact, (0, 1, False, True)).all():
                    raise ValueError(f"{episode.name}: foot_contact is not binary")
                height = np.asarray(episode["states/base_height"], dtype=np.float64)
                if np.any(height < -0.2) or np.any(height > 2.5):
                    raise ValueError(f"{episode.name}: implausible base_height")
                for name, width in REPLAY_FIELDS.items():
                    values = np.asarray(episode[f"replay/{name}"])
                    if values.shape[0] != steps + 1 or (width is not None and values.shape[1:] != (width,)):
                        raise ValueError(f"{episode.name}/replay/{name}: {values.shape}")
                    if not np.isfinite(values).all():
                        raise ValueError(f"{episode.name}/replay/{name}: NaN/Inf")
                for name, width in ACTION_FIELDS.items():
                    values = np.asarray(episode[f"actions/{name}"])
                    if values.shape != (steps, width) or not np.isfinite(values).all():
                        raise ValueError(f"{episode.name}/actions/{name}: {values.shape}")
                initial_target = np.asarray(
                    episode["actions/initial_processed_target_canonical"]
                )
                if initial_target.shape != (29,) or not np.isfinite(initial_target).all():
                    raise ValueError(f"{episode.name}: invalid initial processed target")
                for name, shape in {
                    "reset_root_pose_delta": (6,),
                    "reset_root_velocity_delta": (6,),
                    "reset_joint_pos_delta": (29,),
                    "reset_joint_vel_delta": (29,),
                }.items():
                    values = np.asarray(episode[f"episode_context/{name}"])
                    if values.shape != shape or not np.isfinite(values).all():
                        raise ValueError(
                            f"{episode.name}/episode_context/{name}: {values.shape}"
                        )
                raw = np.asarray(episode["actions/raw_policy_action"], dtype=np.float64)
                processed = np.asarray(episode["actions/processed_joint_target_abs"], dtype=np.float64)
                canonical_action = np.asarray(episode["actions/action_target_canonical"], dtype=np.float64)
                scale = _parameter(schema["action_scale"], env_id)
                offset = np.asarray(context["action_offset"], dtype=np.float64)
                expected_processed = raw * scale + offset
                if schema.get("action_clip") is not None:
                    clip = _parameter(schema["action_clip"], env_id)
                    expected_processed = np.clip(expected_processed, clip[:, 0], clip[:, 1])
                nominal = _parameter(schema["nominal_default_joint_pos"], env_id)
                error = max(
                    float(np.max(np.abs(expected_processed - processed))),
                    float(np.max(np.abs(canonical_action + nominal - processed))),
                )
                max_roundtrip = max(max_roundtrip, error)
                if error > 1.0e-6:
                    raise ValueError(f"{episode.name}: Action roundtrip error {error}")
                torque = np.asarray(episode["diagnostics/applied_joint_torque_mean"])
                impulse = np.asarray(episode["diagnostics/foot_contact_impulse"])
                if torque.shape != (steps, 29) or not np.isfinite(torque).all():
                    raise ValueError(f"{episode.name}: torque diagnostic shape")
                if impulse.shape != (steps, 2, 3) or not np.isfinite(impulse).all():
                    raise ValueError(f"{episode.name}: impulse diagnostic shape")
                if reference_available:
                    joint_reference = np.asarray(
                        episode["reference/joint_pos_vel_future"]
                    )
                    root_reference = np.asarray(
                        episode["reference/root_orientation_future"]
                    )
                    reference_offsets = np.asarray(episode["reference/time_offsets"])
                    expected_offsets = np.asarray(
                        schema["reference_future"]["time_offsets_seconds"],
                        dtype=np.float32,
                    )
                    if joint_reference.shape != (steps, 10, 58):
                        raise ValueError(f"{episode.name}: joint reference shape")
                    if root_reference.shape != (steps, 10, 6):
                        raise ValueError(f"{episode.name}: root reference shape")
                    if not np.isfinite(joint_reference).all() or not np.isfinite(
                        root_reference
                    ).all():
                        raise ValueError(f"{episode.name}: reference NaN/Inf")
                    if reference_offsets.shape != (10,) or not np.allclose(
                        reference_offsets, expected_offsets, rtol=0.0, atol=1.0e-7
                    ):
                        raise ValueError(f"{episode.name}: reference offsets mismatch")
                elif "reference" in episode:
                    raise ValueError(f"{episode.name}: legacy v3 unexpectedly contains reference")
                if np.asarray(episode["transition_context/external_wrench_events"]).shape != (0, 8):
                    raise ValueError(f"{episode.name}: baseline external wrench events must be empty")

                global_id = _scalar(episode["motion"], "global_motion_id")
                variant_id = _scalar(episode["motion"], "variant_id")
                attempt_id = _scalar(episode["motion"], "attempt_id")
                motion_key = motion_key_map[global_id]
                row = manifest.get(motion_key, {})
                package = str(row.get("package", "Unknown"))
                terminated = bool(np.asarray(episode["outcome/terminated"])[-1])
                truncated = bool(np.asarray(episode["outcome/truncated"])[-1])
                timeout = bool(np.asarray(episode["outcome/termination_terms/motion_time_out"])[-1])
                completed = (truncated or timeout) and not terminated
                record = {
                    "episode": episode_name,
                    "hdf5_path": str(dataset_path),
                    "schema_path": str(schema_path),
                    "motion_key": motion_key,
                    "package": package,
                    "global_motion_id": global_id,
                    "variant_id": variant_id,
                    "attempt_id": attempt_id,
                    "env_id": env_id,
                    "context_id": context_id,
                    "steps": steps,
                    "status": "completed" if completed else "failed",
                }
                if attempt_id == 0:
                    pair = (global_id, variant_id)
                    if pair in canonical:
                        raise ValueError(f"duplicate canonical pair {pair}")
                    canonical[pair] = record
                    completion_by_package[package].append(completed)
                else:
                    additional.append(record)
                episode_count += 1
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        check("hdf5_validation", False, str(error))
    else:
        check("hdf5_validation", True, f"validated {episode_count} episodes and {state_count} states")
        check("action_roundtrip", max_roundtrip <= 1.0e-6, f"max_abs={max_roundtrip:.9g}")

    actual_pairs = set(canonical)
    check("canonical_episode_count", len(actual_pairs) == len(expected_pairs), f"actual={len(actual_pairs)}, expected={len(expected_pairs)}")
    check("canonical_pair_coverage", actual_pairs == expected_pairs, f"missing={sorted(expected_pairs-actual_pairs)[:10]}, extra={sorted(actual_pairs-expected_pairs)[:10]}")
    check("motion_manifest", not manifest or set(motion_key_map.values()) == set(manifest), f"runtime={len(motion_key_map)}, manifest={len(manifest)}")
    profile = schema["motion_collection"]["randomization_profile"]
    check("randomization_profile", profile == args.randomization_profile, f"schema={profile}, expected={args.randomization_profile}")

    canonical_rows = [canonical[pair] for pair in sorted(canonical)]
    completed_total = sum(row["status"] == "completed" for row in canonical_rows)
    package_summary = {
        package: {
            "canonical_episodes": len(values),
            "completed": sum(values),
            "completion_rate": float(np.mean(values)),
        }
        for package, values in sorted(completion_by_package.items())
    }
    _write_json(canonical_path, {"episodes": canonical_rows})
    _write_json(attempts_path, {"episodes": additional})
    passed = all(item["passed"] for item in checks)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "schema_version": schema_version,
        "dataset": str(dataset_path),
        "schema": str(schema_path),
        "passed": passed,
        "randomization_profile": profile,
        "canonical_episode_count": len(canonical_rows),
        "additional_attempt_count": len(additional),
        "completed_canonical_episodes": completed_total,
        "completion_rate": completed_total / len(canonical_rows) if canonical_rows else 0.0,
        "completion_by_package": package_summary,
        "motion_manifest": str(manifest_path) if manifest_path else None,
        "motion_manifest_sha256": manifest_sha,
        "canonical_index": str(canonical_path),
        "additional_attempt_index": str(attempts_path),
        "checks": checks,
    }
    _write_json(summary_path, summary)
    print(
        f"SONIC Physics State-Action {schema_version.rsplit('_', 1)[-1]}: "
        f"{'PASS' if passed else 'FAIL'}"
    )
    for item in checks:
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['name']}: {item['evidence']}")
    print(f"Completion: {completed_total}/{len(canonical_rows)}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SONIC Physics State-Action v3/v5")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-motion-count", type=int, required=True)
    parser.add_argument("--expected-variants-per-motion", type=int, required=True)
    parser.add_argument("--variant-offset", type=int, required=True)
    parser.add_argument("--randomization-profile", choices=("startup", "initial_state_mild"), required=True)
    parser.add_argument("--motion-manifest", type=Path)
    args = parser.parse_args()
    return 0 if verify(args)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
