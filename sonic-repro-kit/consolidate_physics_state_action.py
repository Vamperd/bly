from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


STATE_FIELDS = {
    "joint_pos_canonical": 29,
    "joint_vel": 29,
    "base_lin_vel_robot": 3,
    "base_ang_vel_robot": 3,
    "gravity_robot": 3,
    "base_height": 1,
    "foot_contact": 2,
}
REPLAY_FIELDS = {
    "root_pos_w": 3,
    "root_quat_w": 4,
    "body_pos_w": None,
}
ACTION_FIELDS = {
    "action_target_canonical": 29,
    "raw_policy_action": 29,
    "processed_joint_target_abs": 29,
}
DIAGNOSTIC_FIELDS = {
    "applied_joint_torque_mean": (29,),
    "foot_contact_impulse": (2, 3),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _parameter(entry: dict[str, Any], env_id: int) -> np.ndarray:
    values = np.asarray(entry["values"])
    if entry["scope"] == "global":
        return values
    if entry["scope"] == "per_environment":
        return values[env_id]
    raise ValueError(f"unsupported parameter scope {entry.get('scope')!r}")


def _write_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    options = {}
    if values.size and values.ndim:
        options = {"compression": "gzip", "shuffle": True}
    group.create_dataset(name, data=values, **options)


def _transition_count(episode: h5py.Group) -> int:
    """Infer T from the v3 Action stream instead of Isaac Lab's generic attribute.

    Isaac Lab only populates ``num_samples`` from a top-level ``actions`` tensor.
    The raw v3 recorder deliberately stores several Action representations below
    ``physics_action``, so the generic attribute is zero even for valid episodes.
    """
    action_path = "physics_action/action_target_canonical"
    if action_path not in episode:
        raise ValueError(f"{episode.name}: missing {action_path}")
    action_shape = episode[action_path].shape
    if len(action_shape) != 2 or action_shape[1:] != (29,):
        raise ValueError(
            f"{episode.name}/{action_path}: expected [T,29], found {action_shape}"
        )
    steps = int(action_shape[0])
    if steps <= 0:
        raise ValueError(f"{episode.name}: empty episode")

    stored_steps = int(episode.attrs.get("num_samples", 0))
    if stored_steps not in (0, steps):
        raise ValueError(
            f"{episode.name}: num_samples={stored_steps} differs from Action length {steps}"
        )
    return steps


def _copy_attributes(source: h5py.Group, target: h5py.Group) -> None:
    for name, value in source.attrs.items():
        target.attrs[name] = value


def _unique_sequence(
    episode: h5py.Group, name: str, width: int | None
) -> np.ndarray:
    current = np.asarray(episode[f"physics_state_t/{name}"])
    following = np.asarray(episode[f"physics_state_tp1/{name}"])
    if current.shape != following.shape or current.ndim < 2:
        raise ValueError(f"{episode.name}/{name}: invalid pre/post shapes")
    if width is not None and current.shape[1:] != (width,):
        raise ValueError(f"{episode.name}/{name}: expected [T,{width}], found {current.shape}")
    if current.shape[0] > 1 and not np.allclose(
        following[:-1], current[1:], rtol=1.0e-4, atol=1.0e-5
    ):
        error = float(np.max(np.abs(following[:-1] - current[1:])))
        raise ValueError(f"{episode.name}/{name}: transition continuity failed ({error})")
    return np.concatenate((current, following[-1:]), axis=0)


def _constant_first(episode: h5py.Group, path: str, steps: int) -> np.ndarray:
    values = np.asarray(episode[path])
    if values.shape[0] != steps:
        raise ValueError(f"{episode.name}/{path}: expected leading dimension {steps}")
    if steps > 1 and not np.allclose(values[1:], values[:1], rtol=1.0e-5, atol=1.0e-6):
        raise ValueError(f"{episode.name}/{path}: expected episode-constant values")
    return values[0]


def _write_contexts(target: h5py.File, schema: dict[str, Any]) -> None:
    contexts = target.create_group("contexts")
    collection = schema["motion_collection"]
    env_to_variant = list(collection["env_to_variant"])
    runtime_joint = schema["runtime_joint"]
    rigid = schema["runtime_rigid_body"]
    for env_id, variant_id in enumerate(env_to_variant):
        group = contexts.create_group(f"env_{env_id:04d}")
        group.attrs["env_id"] = env_id
        group.attrs["variant_id"] = int(variant_id)
        group.attrs["startup_randomization_seed"] = int(collection["seed"])
        group.attrs["randomization_profile"] = str(
            collection["randomization_profile"]
        )
        for name, entry in (
            ("runtime_default_joint_pos", runtime_joint["default_joint_pos"]),
            ("action_offset", schema["action_offset"]),
            ("joint_position_limits", runtime_joint["position_limits"]),
            ("joint_velocity_limits", runtime_joint["velocity_limits"]),
            ("joint_effort_limits", runtime_joint["effort_limits"]),
            ("joint_stiffness", runtime_joint["stiffness"]),
            ("joint_damping", runtime_joint["damping"]),
            ("joint_armature", runtime_joint["armature"]),
            ("joint_friction", runtime_joint["friction"]),
            ("body_mass", rigid["mass"]),
            ("body_inertia", rigid["inertia"]),
            ("body_com", rigid["com"]),
            ("body_material", rigid["material"]),
        ):
            _write_dataset(group, name, _parameter(entry, env_id))
        terrain = schema["terrain"]
        _write_dataset(
            group,
            "ground_material",
            np.asarray(
                [
                    terrain["static_friction"],
                    terrain["dynamic_friction"],
                    terrain["restitution"],
                ],
                dtype=np.float32,
            ),
        )
        group.attrs["friction_combine_mode"] = str(terrain["friction_combine_mode"])
        group.attrs["restitution_combine_mode"] = str(
            terrain["restitution_combine_mode"]
        )


def consolidate(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    raw_path = run_dir / "data" / "sonic_physics_sa_raw.hdf5"
    raw_schema_path = run_dir / "manifests" / "physics_state_action_raw_schema.json"
    output_path = run_dir / "data" / "sonic_physics_sa_v3.hdf5"
    schema_path = run_dir / "manifests" / "physics_state_action_schema.json"
    for required in (raw_path, raw_schema_path):
        if not required.is_file() or required.stat().st_size == 0:
            raise FileNotFoundError(required)
    if output_path.exists() or schema_path.exists():
        raise FileExistsError("refusing to overwrite an existing v3 artifact")
    schema = _load_json(raw_schema_path)
    if schema.get("schema_version") != "sonic_physics_sa_raw_v3":
        raise ValueError("raw schema version is not sonic_physics_sa_raw_v3")
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    episode_count = 0
    try:
        with h5py.File(raw_path, "r") as source, h5py.File(temporary, "w") as target:
            target.attrs["schema_version"] = "sonic_physics_sa_v3"
            target.attrs["state_semantics"] = "S[0:T+1], A[0:T]"
            _write_contexts(target, schema)
            output_data = target.create_group("data")
            for episode_name in sorted(source["data"].keys()):
                source_episode = source[f"data/{episode_name}"]
                steps = _transition_count(source_episode)
                episode = output_data.create_group(episode_name)
                _copy_attributes(source_episode, episode)
                episode.attrs["raw_num_samples"] = int(
                    source_episode.attrs.get("num_samples", 0)
                )
                episode.attrs["num_samples"] = steps
                episode.attrs["num_transitions"] = steps
                env_values = np.asarray(source_episode["motion/env_id"], dtype=np.int64)
                if env_values.shape != (steps,) or not np.all(env_values == env_values[0]):
                    raise ValueError(f"{source_episode.name}: env_id is not constant")
                env_id = int(env_values[0])
                episode.attrs["context_id"] = f"env_{env_id:04d}"

                states = episode.create_group("states")
                for name, width in STATE_FIELDS.items():
                    _write_dataset(states, name, _unique_sequence(source_episode, name, width))

                replay = episode.create_group("replay")
                for name, width in REPLAY_FIELDS.items():
                    _write_dataset(replay, name, _unique_sequence(source_episode, name, width))

                actions = episode.create_group("actions")
                for name, width in ACTION_FIELDS.items():
                    values = np.asarray(source_episode[f"physics_action/{name}"])
                    if values.shape != (steps, width):
                        raise ValueError(f"{source_episode.name}/{name}: found {values.shape}")
                    _write_dataset(actions, name, values)
                # This raw field is derived from prev_action and therefore changes after
                # t=0. Only its first row is the episode's initial processed target.
                initial_values = np.asarray(
                    source_episode[
                        "physics_action/initial_processed_target_canonical"
                    ]
                )
                if initial_values.shape != (steps, 29):
                    raise ValueError(
                        f"{source_episode.name}/initial_processed_target_canonical: "
                        f"found {initial_values.shape}"
                    )
                initial = initial_values[0]
                _write_dataset(actions, "initial_processed_target_canonical", initial)

                diagnostics = episode.create_group("diagnostics")
                for name, tail in DIAGNOSTIC_FIELDS.items():
                    values = np.asarray(source_episode[f"physics_transition/{name}"])
                    if values.shape != (steps, *tail):
                        raise ValueError(f"{source_episode.name}/{name}: found {values.shape}")
                    _write_dataset(diagnostics, name, values)

                transition_context = episode.create_group("transition_context")
                _write_dataset(
                    transition_context,
                    "external_wrench_events",
                    np.empty((0, 8), dtype=np.float32),
                )

                episode_context = episode.create_group("episode_context")
                for name in (
                    "reset_root_pose_delta",
                    "reset_root_velocity_delta",
                    "reset_joint_pos_delta",
                    "reset_joint_vel_delta",
                ):
                    _write_dataset(
                        episode_context,
                        name,
                        _constant_first(source_episode, f"context_t/{name}", steps),
                    )

                motion = episode.create_group("motion")
                for name in (
                    "env_id", "motion_id", "global_motion_id", "variant_id",
                    "batch_id", "attempt_id",
                ):
                    _write_dataset(
                        motion, name, _constant_first(source_episode, f"motion/{name}", steps)
                    )
                _write_dataset(motion, "motion_step", np.asarray(source_episode["motion/motion_step"]))

                outcome = episode.create_group("outcome")
                for name in ("terminated", "truncated"):
                    _write_dataset(outcome, name, np.asarray(source_episode[f"outcome/{name}"]))
                terms = outcome.create_group("termination_terms")
                for name, dataset in source_episode["outcome/termination_terms"].items():
                    _write_dataset(terms, name, np.asarray(dataset))
                episode_count += 1
            target.flush()
        os.replace(temporary, output_path)
    except Exception:
        # Preserve the failed temporary artifact for post-mortem inspection.
        raise

    final_schema = dict(schema)
    final_schema["schema_version"] = "sonic_physics_sa_v3"
    final_schema["dataset_file"] = output_path.name
    final_schema["storage"] = {
        "states": "data/<episode>/states/* with leading length T+1",
        "actions": "data/<episode>/actions/action_target_canonical with leading length T",
        "transition": "states[t], actions[t] -> states[t+1]",
        "state_tp1_duplicate": False,
        "previous_action_duplicate": False,
        "context_table": "contexts/<context_id>",
    }
    commits = {}
    for name in ("sonic_commit", "isaaclab_commit", "source_commit"):
        path = run_dir / "manifests" / f"{name}.txt"
        commits[name] = path.read_text(encoding="utf-8").strip() if path.is_file() else None
    final_schema["commits"] = commits
    _atomic_json(schema_path, final_schema)
    return {
        "dataset": str(output_path),
        "schema": str(schema_path),
        "episode_count": episode_count,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolidate raw SONIC physics transitions")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(consolidate(args.run_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
