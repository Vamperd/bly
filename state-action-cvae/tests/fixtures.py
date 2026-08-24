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
