from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .util import load_json


PHYSICS_STATE_FIELDS = (
    ("joint_pos_canonical", 29),
    ("joint_vel", 29),
    ("base_lin_vel_robot", 3),
    ("base_ang_vel_robot", 3),
    ("gravity_robot", 3),
    ("base_height", 1),
    ("foot_contact", 2),
)
PHYSICS_STATE_DIM = 70
ROBOT_INFO_DIM = 293
DYNAMICS_CONTEXT_DIM = 648
AUXILIARY_TRANSITION_DIM = 35


def resolve_parameter(entry: dict[str, Any], env_id: int) -> np.ndarray:
    values = np.asarray(entry["values"], dtype=np.float32)
    if entry["scope"] == "global":
        result = values
    elif entry["scope"] == "per_environment":
        result = values[env_id]
    else:
        raise ValueError(f"unsupported parameter scope {entry.get('scope')!r}")
    if not np.isfinite(result).all():
        raise ValueError("schema parameter contains NaN/Inf")
    return np.asarray(result, dtype=np.float32)


def validate_physics_schema(schema: dict[str, Any]) -> None:
    if schema.get("schema_version") != "sonic_physics_sa_v3":
        raise ValueError(f"unsupported physics schema {schema.get('schema_version')!r}")
    if schema.get("dimensions") != {"state": 70, "action": 29}:
        raise ValueError(f"unexpected physics dimensions {schema.get('dimensions')!r}")
    if schema.get("storage", {}).get("state_tp1_duplicate") is not False:
        raise ValueError("physics dataset must contain one unique State sequence")
    if schema.get("action_term_type") != "JointPositionAction":
        raise ValueError("physics dataset requires JointPositionAction")
    if schema.get("wrapper_action_transform_enabled") is not False:
        raise ValueError("wrapper Action transform must be disabled")
    names = schema.get("joint_names", [])
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("physics schema must contain 29 unique joint names")
    timing = schema.get("simulation", {})
    if not np.isclose(timing.get("sim_dt", -1), 0.005):
        raise ValueError("physics sim_dt must be 0.005")
    if not np.isclose(timing.get("control_dt", -1), 0.02):
        raise ValueError("physics control_dt must be 0.02")
    if int(timing.get("decimation", -1)) != 4:
        raise ValueError("physics decimation must be 4")


def load_physics_schema(path: Path) -> dict[str, Any]:
    schema = load_json(path)
    validate_physics_schema(schema)
    return schema


def read_physics_states(group: Any, start: int | None = None, stop: int | None = None) -> np.ndarray:
    arrays = []
    for name, width in PHYSICS_STATE_FIELDS:
        values = np.asarray(group[name][slice(start, stop)], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError(f"{group.name}/{name}: invalid shape {values.shape}")
        arrays.append(values)
    result = np.concatenate(arrays, axis=-1)
    if not np.isfinite(result).all():
        raise ValueError(f"{group.name}: State contains NaN/Inf")
    return result


def robot_information_vector(
    schema: dict[str, Any], context: Any, env_id: int
) -> np.ndarray:
    timing = schema["simulation"]
    values = [
        resolve_parameter(schema["nominal_default_joint_pos"], env_id).reshape(-1),
        resolve_parameter(schema["action_scale"], env_id).reshape(-1),
        np.asarray(context["joint_stiffness"], dtype=np.float32).reshape(-1),
        np.asarray(context["joint_damping"], dtype=np.float32).reshape(-1),
        np.asarray(context["joint_armature"], dtype=np.float32).reshape(-1),
        np.asarray(context["joint_friction"], dtype=np.float32).reshape(-1),
        np.asarray(context["joint_effort_limits"], dtype=np.float32).reshape(-1),
        np.asarray(context["joint_velocity_limits"], dtype=np.float32).reshape(-1),
        np.asarray(context["joint_position_limits"], dtype=np.float32).reshape(-1),
        np.asarray(
            [timing["sim_dt"], timing["control_dt"], timing["decimation"]],
            dtype=np.float32,
        ),
    ]
    result = np.concatenate(values)
    if result.shape != (ROBOT_INFO_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"invalid robot information vector {result.shape}")
    return result


def dynamics_context_vector(context: Any) -> np.ndarray:
    result = np.concatenate(
        [
            np.asarray(context[name], dtype=np.float32).reshape(-1)
            for name in (
                "body_mass",
                "body_inertia",
                "body_com",
                "body_material",
                "ground_material",
            )
        ]
    )
    if result.shape != (DYNAMICS_CONTEXT_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"invalid randomized dynamics context {result.shape}")
    return result


def read_auxiliary_transitions(
    group: Any, start: int | None = None, stop: int | None = None
) -> np.ndarray:
    torque = np.asarray(
        group["applied_joint_torque_mean"][slice(start, stop)], dtype=np.float32
    )
    impulse = np.asarray(
        group["foot_contact_impulse"][slice(start, stop)], dtype=np.float32
    )
    if torque.ndim != 2 or torque.shape[1] != 29:
        raise ValueError(f"{group.name}/applied_joint_torque_mean: {torque.shape}")
    if impulse.shape != (torque.shape[0], 2, 3):
        raise ValueError(f"{group.name}/foot_contact_impulse: {impulse.shape}")
    result = np.concatenate((torque, impulse.reshape(torque.shape[0], 6)), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError(f"{group.name}: auxiliary transition contains NaN/Inf")
    return result
