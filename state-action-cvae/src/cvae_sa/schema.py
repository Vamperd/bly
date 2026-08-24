from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .constants import ACTION_DIM, PHYSICAL_STATE_FIELDS
from .util import load_json


class SchemaError(ValueError):
    pass


def resolve_parameter(
    entry: dict[str, Any], env_id: int, tail_shape: tuple[int, ...]
) -> np.ndarray:
    scope = entry.get("scope")
    values = np.asarray(entry.get("values"), dtype=np.float32)
    if scope == "global":
        expected = tail_shape
        result = values
    elif scope == "per_environment":
        expected = (values.shape[0], *tail_shape)
        if not 0 <= env_id < values.shape[0]:
            raise SchemaError(f"env_id {env_id} is outside {values.shape[0]} rows")
        result = values[env_id]
    else:
        raise SchemaError(f"unsupported parameter scope {scope!r}")
    if tuple(values.shape) != expected:
        raise SchemaError(f"expected parameter shape {expected}, found {values.shape}")
    if not np.isfinite(result).all():
        raise SchemaError("runtime parameter contains NaN or Inf")
    return np.asarray(result, dtype=np.float32)


def validate_schema(schema: dict[str, Any]) -> None:
    if schema.get("schema_version") != "sonic_minimal_sa_v2":
        raise SchemaError(f"unsupported schema version {schema.get('schema_version')!r}")
    if schema.get("dimensions") != {"state": 93, "goal": 63, "action": 29}:
        raise SchemaError(f"unexpected dimensions {schema.get('dimensions')!r}")
    if schema.get("action_term_type") != "JointPositionAction":
        raise SchemaError("only JointPositionAction datasets are supported")
    if schema.get("wrapper_action_transform_enabled") is not False:
        raise SchemaError("wrapper action transform must be disabled")
    if not np.isclose(float(schema.get("control_dt", -1.0)), 0.02):
        raise SchemaError(f"control_dt must be 0.02, found {schema.get('control_dt')!r}")
    if not np.isclose(float(schema.get("sim_dt", -1.0)), 0.005):
        raise SchemaError(f"sim_dt must be 0.005, found {schema.get('sim_dt')!r}")
    names = schema.get("joint_names")
    if not isinstance(names, list) or len(names) != ACTION_DIM or len(set(names)) != ACTION_DIM:
        raise SchemaError("schema must contain 29 unique joint names")
    for name in ("default_joint_pos", "action_scale", "action_offset"):
        resolve_parameter(schema[name], 0, (ACTION_DIM,))
    if schema.get("action_clip") is not None:
        resolve_parameter(schema["action_clip"], 0, (ACTION_DIM, 2))


def load_schema(path: Path) -> dict[str, Any]:
    schema = load_json(path)
    validate_schema(schema)
    return schema


def raw_action_to_relative(
    actions: np.ndarray, schema: dict[str, Any], env_id: int
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise SchemaError(f"raw actions must have shape [T,29], found {actions.shape}")
    scale = resolve_parameter(schema["action_scale"], env_id, (ACTION_DIM,))
    offset = resolve_parameter(schema["action_offset"], env_id, (ACTION_DIM,))
    default = resolve_parameter(schema["default_joint_pos"], env_id, (ACTION_DIM,))
    processed = actions * scale + offset
    if schema.get("action_clip") is not None:
        clip = resolve_parameter(schema["action_clip"], env_id, (ACTION_DIM, 2))
        if np.any(clip[:, 0] > clip[:, 1]):
            raise SchemaError("action_clip lower bound exceeds upper bound")
        processed = np.clip(processed, clip[:, 0], clip[:, 1])
    relative = processed - default
    if not np.isfinite(relative).all():
        raise SchemaError("mapped relative action contains NaN or Inf")
    return relative.astype(np.float32, copy=False), scale


def read_physical_state(group: Any) -> np.ndarray:
    arrays = []
    for name, width in PHYSICAL_STATE_FIELDS:
        values = np.asarray(group[name], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != width:
            raise SchemaError(f"{group.name}/{name} has invalid shape {values.shape}")
        arrays.append(values)
    result = np.concatenate(arrays, axis=-1)
    if not np.isfinite(result).all():
        raise SchemaError(f"{group.name} contains NaN or Inf")
    return result
