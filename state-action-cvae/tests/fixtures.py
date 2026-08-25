from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def write_collection_run(
    root: Path,
    motion_key: str,
    package: str,
    variants: tuple[int, ...],
    per_environment: bool = False,
) -> Path:
    run = root / motion_key
    (run / "data").mkdir(parents=True)
    (run / "manifests").mkdir()
    (run / "markers").mkdir()
    (run / "markers" / "collect_state_action.ok").write_text("PASS\n")
    env_count = len(variants)
    default_values = [[0.1] * 29 for _ in variants] if per_environment else [0.1] * 29
    offset_values = [[0.1] * 29 for _ in variants] if per_environment else [0.1] * 29
    parameter_scope = "per_environment" if per_environment else "global"
    schema = {
        "schema_version": "sonic_minimal_sa_v2",
        "dimensions": {"state": 93, "goal": 63, "action": 29},
        "joint_names": [f"joint_{index}" for index in range(29)],
        "default_joint_pos": {"scope": parameter_scope, "values": default_values},
        "default_joint_vel": {"scope": "global", "values": [0.0] * 29},
        "action_scale": {"scope": "global", "values": [0.5] * 29},
        "action_offset": {"scope": parameter_scope, "values": offset_values},
        "action_clip": None,
        "wrapper_action_transform_enabled": False,
        "action_term_type": "JointPositionAction",
        "control_dt": 0.02,
        "sim_dt": 0.005,
        "motion_id_to_key": {"0": motion_key},
        "motion_collection": {
            "randomization_profile": "startup",
            "variant_offset": min(variants),
            "eval_motion_repeat": len(variants),
        },
    }
    schema_path = run / "manifests" / "state_action_schema.json"
    schema_path.write_text(json.dumps(schema))
    episodes = []
    with h5py.File(run / "data" / "sonic_minimal_sa.hdf5", "w") as stream:
        data = stream.create_group("data")
        for env_id, variant in enumerate(variants):
            name = f"demo_{env_id}"
            steps = 6
            episode = data.create_group(name)
            episode.attrs["num_samples"] = steps
            actions = np.full((steps, 29), 0.01 * (variant + 1), dtype=np.float32)
            episode.create_dataset("actions", data=actions)
            outcome = episode.create_group("outcome")
            outcome.create_dataset("terminated", data=np.zeros(steps, dtype=bool))
            truncated = np.zeros(steps, dtype=bool)
            truncated[-1] = True
            outcome.create_dataset("truncated", data=truncated)
            state_t = episode.create_group("state_t")
            state_tp1 = episode.create_group("state_tp1")
            joint_pos = np.arange(steps * 29, dtype=np.float32).reshape(steps, 29) / 1000
            joint_vel = np.arange(steps * 29, dtype=np.float32).reshape(steps, 29) / 100
            base = np.arange(steps * 3, dtype=np.float32).reshape(steps, 3) / 100
            gravity = np.tile(np.array([[0.0, 0.0, -1.0]], np.float32), (steps, 1))
            previous = np.concatenate((np.zeros((1, 29), np.float32), actions[:-1]), axis=0)
            for field, current, final in (
                ("joint_pos", joint_pos, joint_pos[-1:] + 0.001),
                ("joint_vel", joint_vel, joint_vel[-1:]),
                ("base_ang_vel", base, base[-1:]),
                ("gravity_robot", gravity, gravity[-1:]),
                ("previous_action", previous, actions[-1:]),
            ):
                state_t.create_dataset(field, data=current)
                state_tp1.create_dataset(
                    field, data=np.concatenate((current[1:], final), axis=0)
                )
            motion = episode.create_group("motion")
            for field, value in (
                ("env_id", env_id),
                ("global_motion_id", 0),
                ("variant_id", variant),
                ("attempt_id", 0),
            ):
                motion.create_dataset(field, data=np.full(steps, value, dtype=np.int64))
            episodes.append(
                {
                    "episode": name,
                    "steps": steps,
                    "env_id": env_id,
                    "global_motion_id": 0,
                    "variant_id": variant,
                    "attempt_id": 0,
                    "motion_key": motion_key,
                    "package": package,
                    "status": "completed",
                    "terminated": False,
                    "truncated": True,
                }
            )
    (run / "manifests" / "canonical_episode_index.json").write_text(
        json.dumps({"episodes": episodes})
    )
    (run / "manifests" / "collection_summary.json").write_text(
        json.dumps({"passed": True, "canonical_episode_count": len(episodes)})
    )
    return run


def write_physics_collection_run(
    root: Path,
    name: str,
    motion_key: str,
    package: str,
    variants: tuple[int, ...],
) -> Path:
    run = root / name
    (run / "data").mkdir(parents=True)
    (run / "manifests").mkdir()
    (run / "markers").mkdir()
    (run / "markers/collect_physics_state_action.ok").write_text("PASS\n")
    env_count = len(variants)

    def parameter(values, scope="global"):
        return {"scope": scope, "values": values}

    per_env_29 = [[0.0] * 29 for _ in variants]
    per_env_limits = [[[-2.0, 2.0] for _ in range(29)] for _ in variants]
    schema = {
        "schema_version": "sonic_physics_sa_v3",
        "dimensions": {"state": 70, "action": 29},
        "storage": {"state_tp1_duplicate": False, "previous_action_duplicate": False},
        "joint_names": [f"joint_{index}" for index in range(29)],
        "body_names": ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"],
        "nominal_default_joint_pos": parameter([0.0] * 29),
        "action_scale": parameter([0.5] * 29),
        "action_offset": parameter(per_env_29, "per_environment"),
        "action_clip": None,
        "wrapper_action_transform_enabled": False,
        "action_term_type": "JointPositionAction",
        "simulation": {
            "sim_dt": 0.005,
            "control_dt": 0.02,
            "decimation": 4,
            "gravity_w": [0.0, 0.0, -9.81],
            "solver_position_iteration_count": 8,
            "solver_velocity_iteration_count": 4,
        },
        "contact": {"threshold_n": 10.0},
        "actuator_groups": {
            "all": {
                "type": "ImplicitActuator",
                "joint_names": [f"joint_{index}" for index in range(29)],
                "min_delay": 0,
                "max_delay": 0,
            }
        },
        "motion_id_to_key": {"0": motion_key},
        "motion_collection": {
            "randomization_profile": "startup" if min(variants) == 0 else "initial_state_mild",
            "variant_offset": min(variants),
            "eval_motion_repeat": len(variants),
            "env_to_variant": list(variants),
        },
    }
    schema_path = run / "manifests/physics_state_action_schema.json"
    schema_path.write_text(json.dumps(schema))
    index = []
    with h5py.File(run / "data/sonic_physics_sa_v3.hdf5", "w") as stream:
        stream.attrs["schema_version"] = "sonic_physics_sa_v3"
        contexts = stream.create_group("contexts")
        data = stream.create_group("data")
        for env_id, variant in enumerate(variants):
            context_id = f"env_{env_id:04d}"
            context = contexts.create_group(context_id)
            context.attrs["env_id"] = env_id
            context.attrs["variant_id"] = variant
            context.create_dataset("body_mass", data=np.ones(30))
            context.create_dataset("body_inertia", data=np.ones((30, 9)))
            body_com = np.zeros((30, 7))
            body_com[:, 3] = 1.0
            context.create_dataset("body_com", data=body_com)
            context.create_dataset("body_material", data=np.ones((45, 3)) * 0.5)
            context.create_dataset("ground_material", data=np.asarray([0.8, 0.6, 0.0]))
            for field, values in (
                ("joint_stiffness", np.ones(29) * 40),
                ("joint_damping", np.ones(29) * 2),
                ("joint_armature", np.ones(29) * 0.01),
                ("joint_friction", np.zeros(29)),
                ("joint_effort_limits", np.ones(29) * 50),
                ("joint_velocity_limits", np.ones(29) * 30),
                ("joint_position_limits", np.asarray(per_env_limits[env_id])),
            ):
                context.create_dataset(field, data=values)
            episode_name = f"demo_{env_id}"
            episode = data.create_group(episode_name)
            steps = 6
            episode.attrs["num_transitions"] = steps
            episode.attrs["num_samples"] = steps
            episode.attrs["context_id"] = context_id
            states = episode.create_group("states")
            state_values = {
                "joint_pos_canonical": np.arange((steps + 1) * 29).reshape(steps + 1, 29) / 1000,
                "joint_vel": np.arange((steps + 1) * 29).reshape(steps + 1, 29) / 100,
                "base_lin_vel_robot": np.zeros((steps + 1, 3)),
                "base_ang_vel_robot": np.zeros((steps + 1, 3)),
                "gravity_robot": np.tile([0.0, 0.0, -1.0], (steps + 1, 1)),
                "base_height": np.ones((steps + 1, 1)) * 0.75,
                "foot_contact": np.tile([1, 1], (steps + 1, 1)),
            }
            for field, values in state_values.items():
                states.create_dataset(field, data=np.asarray(values, dtype=np.float32))
            actions = episode.create_group("actions")
            action = np.ones((steps, 29), dtype=np.float32) * (variant + 1) / 100
            actions.create_dataset("action_target_canonical", data=action)
            actions.create_dataset(
                "initial_processed_target_canonical",
                data=np.ones(29, dtype=np.float32) * variant / 100,
            )
            diagnostics = episode.create_group("diagnostics")
            diagnostics.create_dataset(
                "applied_joint_torque_mean", data=np.ones((steps, 29), dtype=np.float32)
            )
            diagnostics.create_dataset(
                "foot_contact_impulse", data=np.zeros((steps, 2, 3), dtype=np.float32)
            )
            motion = episode.create_group("motion")
            for field, value in (
                ("global_motion_id", 0), ("variant_id", variant), ("attempt_id", 0),
            ):
                motion.create_dataset(field, data=np.asarray(value, dtype=np.int64))
            index.append(
                {
                    "episode": episode_name,
                    "steps": steps,
                    "env_id": env_id,
                    "context_id": context_id,
                    "global_motion_id": 0,
                    "variant_id": variant,
                    "attempt_id": 0,
                    "motion_key": motion_key,
                    "package": package,
                    "status": "completed",
                }
            )
    (run / "manifests/canonical_episode_index.json").write_text(json.dumps({"episodes": index}))
    (run / "manifests/collection_summary.json").write_text(json.dumps({"passed": True}))
    return run
