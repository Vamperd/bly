#!/usr/bin/env python3
"""Restore one recorded Physics-v3 environment before external Action replay.

This module is deliberately imported only when ``external_replay_initialization_path``
is configured.  Ordinary SONIC evaluation and historical Action replay are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


FORMAT_VERSION = "sonic_exact_replay_initialization_v1"
HASH_EXCLUDED_KEYS = {"payload_sha256"}
PHYSICS_V3_FIELDS = (
    "joint_pos_canonical",
    "joint_vel",
    "base_lin_vel_robot",
    "base_ang_vel_robot",
    "gravity_robot",
    "base_height",
    "foot_contact",
)
PHYSICS_STATE_DIM = 70
REQUIRED_ARRAYS: dict[str, tuple[int, ...] | None] = {
    "joint_names": (29,),
    "joint_pos": (29,),
    "joint_vel": (29,),
    "root_pos_relative": (3,),
    "root_quat_wxyz": (4,),
    "root_lin_vel_world": (3,),
    "root_ang_vel_world": (3,),
    "body_pos_relative": None,
    "physics_state_v3": (70,),
    "nominal_default_joint_pos": (29,),
    "runtime_default_joint_pos": (29,),
    "action_scale": (29,),
    "action_offset": (29,),
    "action_clip": None,
    "wrapper_action_clip": (),
    "joint_position_limits": (29, 2),
    "joint_velocity_limits": (29,),
    "joint_effort_limits": (29,),
    "joint_stiffness": (29,),
    "joint_damping": (29,),
    "joint_armature": (29,),
    "joint_friction": (29,),
    "body_mass": None,
    "body_inertia": None,
    "body_com": None,
    "body_material": None,
    "ground_material": (3,),
    "previous_raw_action": (29,),
    "previous_processed_action": (29,),
    "initial_joint_target_abs": (29,),
    "format_version": (),
    "payload_sha256": (),
}


def _array_digest(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(key for key in values if key not in HASH_EXCLUDED_KEYS):
        value = np.ascontiguousarray(values[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _scalar_text(value: np.ndarray) -> str:
    item = np.asarray(value).tolist()
    if isinstance(item, list):
        raise ValueError("expected a scalar string")
    return str(item)


def load_initialization(path: Path) -> tuple[dict[str, np.ndarray], str]:
    """Load, shape-check, and cryptographically validate an initialization package."""

    resolved = path.expanduser().resolve(strict=True)
    with np.load(resolved, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_ARRAYS).difference(archive.files))
        if missing:
            raise ValueError(f"exact initialization is missing arrays: {missing}")
        values = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if _scalar_text(values["format_version"]) != FORMAT_VERSION:
        raise ValueError("unsupported exact initialization format")
    for name, expected_shape in REQUIRED_ARRAYS.items():
        value = values[name]
        if expected_shape is not None and value.shape != expected_shape:
            raise ValueError(
                f"exact initialization {name} has shape {value.shape}, expected {expected_shape}"
            )
        # A NaN scalar is the explicit representation of an absent wrapper clip.
        # Every other floating-point payload must remain finite.
        allow_absent_wrapper_clip = (
            name == "wrapper_action_clip"
            and value.shape == ()
            and bool(np.isnan(value))
        )
        if (
            value.dtype.kind in "fc"
            and not allow_absent_wrapper_clip
            and not np.isfinite(value).all()
        ):
            raise ValueError(f"exact initialization {name} contains NaN/Inf")
    if values["body_pos_relative"].ndim != 2 or values["body_pos_relative"].shape[1] != 3:
        raise ValueError("body_pos_relative must have shape [body,3]")
    body_count = values["body_pos_relative"].shape[0]
    if values["body_mass"].shape != (body_count,):
        raise ValueError("body_mass does not match body count")
    if values["body_inertia"].shape != (body_count, 9):
        raise ValueError("body_inertia does not match body count")
    if values["body_com"].shape != (body_count, 7):
        raise ValueError("body_com does not match body count")
    if values["body_material"].ndim != 2 or values["body_material"].shape[1] != 3:
        raise ValueError("body_material must have shape [shape,3]")
    if values["action_clip"].shape not in {(0,), (29, 2)}:
        raise ValueError("action_clip must be empty or [29,2]")
    expected = _scalar_text(values["payload_sha256"])
    actual = _array_digest(values)
    if actual != expected:
        raise ValueError(
            f"exact initialization payload SHA256 mismatch: expected {expected}, found {actual}"
        )
    return values, actual


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _max_abs(reference: np.ndarray, value: np.ndarray) -> float:
    reference = np.asarray(reference)
    value = np.asarray(value)
    if reference.shape != value.shape:
        return float("inf")
    if reference.dtype.kind in "US" or value.dtype.kind in "US":
        return 0.0 if np.array_equal(reference, value) else float("inf")
    if np.array_equal(reference, value, equal_nan=True):
        return 0.0
    difference = np.abs(reference.astype(np.float64) - value.astype(np.float64))
    if not np.isfinite(difference).all():
        return float("inf")
    return float(np.max(difference)) if difference.size else 0.0


def _quaternion_error_degrees(reference: np.ndarray, value: np.ndarray) -> float:
    reference64 = np.asarray(reference, dtype=np.float64)
    value64 = np.asarray(value, dtype=np.float64)
    reference64 /= max(float(np.linalg.norm(reference64)), 1.0e-12)
    value64 /= max(float(np.linalg.norm(value64)), 1.0e-12)
    dot = float(np.clip(abs(np.dot(reference64, value64)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _joint_ids(term: Any) -> list[int] | slice:
    ids = getattr(term, "_joint_ids")
    if isinstance(ids, slice):
        return ids
    return [int(value) for value in ids]


def _action_clip_array(term: Any) -> np.ndarray:
    clip = getattr(term, "_clip", None)
    if clip is None:
        return np.empty((0,), dtype=np.float32)
    value = _numpy(torch.as_tensor(clip))
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    return value.astype(np.float32, copy=False)


def _set_actuator_parameters(robot: Any, values: dict[str, np.ndarray]) -> None:
    """Keep actuator-side mirrors aligned with the PhysX parameters we restore."""

    fields = {
        "stiffness": "joint_stiffness",
        "damping": "joint_damping",
        "armature": "joint_armature",
        "friction": "joint_friction",
        "effort_limit": "joint_effort_limits",
        "effort_limit_sim": "joint_effort_limits",
        "velocity_limit": "joint_velocity_limits",
        "velocity_limit_sim": "joint_velocity_limits",
    }
    for actuator in robot.actuators.values():
        indices = actuator.joint_indices
        if isinstance(indices, slice):
            indices = list(range(robot.num_joints))[indices]
        indices = [int(index) for index in indices]
        for attribute, source in fields.items():
            current = getattr(actuator, attribute, None)
            if not isinstance(current, torch.Tensor):
                continue
            restored = torch.as_tensor(
                values[source][indices], device=current.device, dtype=current.dtype
            ).reshape(1, -1)
            current[:] = restored


def _set_ground_material(raw_env: Any, ground: np.ndarray) -> np.ndarray:
    """Restore the plane's USD material and return the immediate USD readback."""

    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdPhysics

    material_cfg = raw_env.cfg.scene.terrain.physics_material
    material_cfg.static_friction = float(ground[0])
    material_cfg.dynamic_friction = float(ground[1])
    material_cfg.restitution = float(ground[2])
    prim_path = f"{raw_env.cfg.scene.terrain.prim_path}/physicsMaterial"
    prim = get_current_stage().GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"ground physics material prim is missing: {prim_path}")
    api = UsdPhysics.MaterialAPI(prim)
    if not api:
        api = UsdPhysics.MaterialAPI.Apply(prim)
    api.GetStaticFrictionAttr().Set(float(ground[0]))
    api.GetDynamicFrictionAttr().Set(float(ground[1]))
    api.GetRestitutionAttr().Set(float(ground[2]))
    return np.asarray(
        [
            api.GetStaticFrictionAttr().Get(),
            api.GetDynamicFrictionAttr().Get(),
            api.GetRestitutionAttr().Get(),
        ],
        dtype=np.float32,
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def apply_exact_replay_initialization(
    wrapped_env: Any,
    initialization_path: str | Path,
    report_path: str | Path,
) -> dict[str, torch.Tensor]:
    """Apply an initialization after reset and return freshly computed wrapper observations."""

    from gear_sonic.envs.manager_env.mdp.recorders import _action_term, _physics_state

    values, payload_sha256 = load_initialization(Path(initialization_path))
    raw = wrapped_env.env
    if raw.num_envs != 1:
        raise ValueError("exact replay initialization requires num_envs=1")
    term = _action_term(raw)
    runtime_names = tuple(str(name) for name in getattr(term, "_joint_names", ()))
    expected_names = tuple(str(name) for name in values["joint_names"].tolist())
    if runtime_names != expected_names:
        raise ValueError("exact replay initialization joint order differs from runtime")
    joint_ids = _joint_ids(term)
    robot = raw.scene["robot"]
    env_ids_device = torch.tensor([0], dtype=torch.long, device=raw.device)
    env_ids_cpu = env_ids_device.cpu()

    def device_tensor(name: str) -> torch.Tensor:
        return torch.as_tensor(values[name], device=raw.device, dtype=torch.float32).unsqueeze(0)

    # Runtime calibration and all PhysX joint parameters are restored before state.
    nominal = device_tensor("nominal_default_joint_pos")[0]
    robot.data.default_joint_pos_nominal = nominal.clone()
    robot.data.default_joint_pos[0, joint_ids] = device_tensor("runtime_default_joint_pos")[0]
    term._scale = device_tensor("action_scale")  # noqa: SLF001
    term._offset = device_tensor("action_offset")  # noqa: SLF001
    if values["action_clip"].size:
        term._clip = torch.as_tensor(  # noqa: SLF001
            values["action_clip"], device=raw.device, dtype=torch.float32
        ).unsqueeze(0)
    else:
        term._clip = None  # noqa: SLF001
    wrapper_clip = float(values["wrapper_action_clip"])
    runtime_wrapper = getattr(raw, "wrapper", wrapped_env)
    runtime_wrapper.config["action_clip_value"] = (
        None if np.isnan(wrapper_clip) else wrapper_clip
    )
    if runtime_wrapper is not wrapped_env:
        wrapped_env.config["action_clip_value"] = runtime_wrapper.config[
            "action_clip_value"
        ]

    robot.write_joint_position_limit_to_sim(
        device_tensor("joint_position_limits"), joint_ids=joint_ids, env_ids=env_ids_device
    )
    robot.write_joint_velocity_limit_to_sim(
        device_tensor("joint_velocity_limits"), joint_ids=joint_ids, env_ids=env_ids_device
    )
    robot.write_joint_effort_limit_to_sim(
        device_tensor("joint_effort_limits"), joint_ids=joint_ids, env_ids=env_ids_device
    )
    robot.write_joint_stiffness_to_sim(
        device_tensor("joint_stiffness"), joint_ids=joint_ids, env_ids=env_ids_device
    )
    robot.write_joint_damping_to_sim(
        device_tensor("joint_damping"), joint_ids=joint_ids, env_ids=env_ids_device
    )
    robot.write_joint_armature_to_sim(
        device_tensor("joint_armature"), joint_ids=joint_ids, env_ids=env_ids_device
    )
    robot.write_joint_friction_coefficient_to_sim(
        device_tensor("joint_friction"), joint_ids=joint_ids, env_ids=env_ids_device
    )
    _set_actuator_parameters(robot, values)

    for getter, setter, name in (
        (robot.root_physx_view.get_masses, robot.root_physx_view.set_masses, "body_mass"),
        (robot.root_physx_view.get_inertias, robot.root_physx_view.set_inertias, "body_inertia"),
        (robot.root_physx_view.get_coms, robot.root_physx_view.set_coms, "body_com"),
        (
            robot.root_physx_view.get_material_properties,
            robot.root_physx_view.set_material_properties,
            "body_material",
        ),
    ):
        current = getter().clone()
        current[0] = torch.as_tensor(values[name], device=current.device, dtype=current.dtype)
        setter(current, env_ids_cpu)
    ground_readback = _set_ground_material(raw, values["ground_material"])

    origin = _numpy(raw.scene.env_origins[0])
    root_position = values["root_pos_relative"].copy()
    root_position += origin
    root_state = np.concatenate(
        (
            root_position,
            values["root_quat_wxyz"],
            values["root_lin_vel_world"],
            values["root_ang_vel_world"],
        )
    ).astype(np.float32)
    robot.write_root_state_to_sim(
        torch.as_tensor(root_state, device=raw.device).unsqueeze(0), env_ids=env_ids_device
    )
    robot.write_joint_state_to_sim(
        device_tensor("joint_pos"),
        device_tensor("joint_vel"),
        joint_ids=joint_ids,
        env_ids=env_ids_device,
    )

    previous_raw = device_tensor("previous_raw_action")
    term.process_actions(previous_raw)
    raw.action_manager._action[:] = previous_raw  # noqa: SLF001
    raw.action_manager._prev_action[:] = previous_raw  # noqa: SLF001
    robot.set_joint_position_target(
        device_tensor("initial_joint_target_abs"), joint_ids=joint_ids, env_ids=env_ids_device
    )
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    raw.scene.update(dt=0.0)

    state = _physics_state(raw)
    state_flat = torch.cat(
        [state[name].to(dtype=torch.float32) for name in PHYSICS_V3_FIELDS], dim=-1
    )
    if state_flat.shape != (1, PHYSICS_STATE_DIM):
        raise RuntimeError(f"unexpected initialized Physics State shape {tuple(state_flat.shape)}")
    actual_root_relative = _numpy(robot.data.root_pos_w[0]) - origin
    actual_body_relative = _numpy(robot.data.body_pos_w[0]) - origin[None]
    actual_context = {
        "nominal_default_joint_pos": _numpy(robot.data.default_joint_pos_nominal),
        "runtime_default_joint_pos": _numpy(robot.data.default_joint_pos[0, joint_ids]),
        "action_scale": _numpy(torch.as_tensor(term._scale)[0]),  # noqa: SLF001
        "action_offset": _numpy(torch.as_tensor(term._offset)[0]),  # noqa: SLF001
        "action_clip": _action_clip_array(term),
        "wrapper_action_clip": np.asarray(
            np.nan
            if runtime_wrapper.config.get("action_clip_value", None) is None
            else runtime_wrapper.config["action_clip_value"],
            dtype=np.float32,
        ),
        "joint_position_limits": _numpy(robot.data.joint_pos_limits[0, joint_ids]),
        "joint_velocity_limits": _numpy(robot.data.joint_vel_limits[0, joint_ids]),
        "joint_effort_limits": _numpy(robot.data.joint_effort_limits[0, joint_ids]),
        "joint_stiffness": _numpy(robot.data.joint_stiffness[0, joint_ids]),
        "joint_damping": _numpy(robot.data.joint_damping[0, joint_ids]),
        "joint_armature": _numpy(robot.data.joint_armature[0, joint_ids]),
        "joint_friction": _numpy(robot.data.joint_friction_coeff[0, joint_ids]),
        "body_mass": _numpy(robot.root_physx_view.get_masses()[0]),
        "body_inertia": _numpy(robot.root_physx_view.get_inertias()[0]),
        "body_com": _numpy(robot.root_physx_view.get_coms()[0]),
        "body_material": _numpy(robot.root_physx_view.get_material_properties()[0]),
        "ground_material": ground_readback,
    }
    context_fields = tuple(actual_context)
    context_errors = {
        name: _max_abs(values[name], actual_context[name]) for name in context_fields
    }
    readback = {
        "physics_state_v3": _numpy(state_flat[0]),
        "joint_pos": _numpy(robot.data.joint_pos[0, joint_ids]),
        "joint_vel": _numpy(robot.data.joint_vel[0, joint_ids]),
        "root_pos_relative": actual_root_relative,
        "root_quat_wxyz": _numpy(robot.data.root_quat_w[0]),
        "root_lin_vel_world": _numpy(robot.data.root_lin_vel_w[0]),
        "root_ang_vel_world": _numpy(robot.data.root_ang_vel_w[0]),
        "body_pos_relative": actual_body_relative,
        "previous_raw_action": _numpy(raw.action_manager.prev_action[0]),
        "current_raw_action": _numpy(raw.action_manager.action[0]),
        "previous_processed_action": _numpy(term.processed_actions[0]),
        "initial_joint_target_abs": _numpy(robot.data.joint_pos_target[0, joint_ids]),
    }
    errors = {
        "physics_state_v3_max_abs": _max_abs(
            values["physics_state_v3"], readback["physics_state_v3"]
        ),
        "joint_pos_max_abs_rad": _max_abs(values["joint_pos"], readback["joint_pos"]),
        "joint_vel_max_abs_rad_s": _max_abs(values["joint_vel"], readback["joint_vel"]),
        "root_pos_max_abs_m": _max_abs(
            values["root_pos_relative"], readback["root_pos_relative"]
        ),
        "root_orientation_error_deg": _quaternion_error_degrees(
            values["root_quat_wxyz"], readback["root_quat_wxyz"]
        ),
        "root_lin_vel_max_abs_m_s": _max_abs(
            values["root_lin_vel_world"], readback["root_lin_vel_world"]
        ),
        "root_ang_vel_max_abs_rad_s": _max_abs(
            values["root_ang_vel_world"], readback["root_ang_vel_world"]
        ),
        "body_pos_max_abs_m": _max_abs(
            values["body_pos_relative"], readback["body_pos_relative"]
        ),
        "previous_raw_action_max_abs": _max_abs(
            values["previous_raw_action"], readback["previous_raw_action"]
        ),
        "current_raw_action_max_abs": _max_abs(
            values["previous_raw_action"], readback["current_raw_action"]
        ),
        "previous_processed_action_max_abs": _max_abs(
            values["previous_processed_action"], readback["previous_processed_action"]
        ),
        "initial_joint_target_max_abs": _max_abs(
            values["initial_joint_target_abs"], readback["initial_joint_target_abs"]
        ),
        "runtime_context_max_abs": max(context_errors.values()),
    }
    report = {
        "format_version": "sonic_exact_replay_initialization_readback_v1",
        "initialization_path": str(Path(initialization_path).expanduser().resolve()),
        "initialization_file_sha256": hashlib.sha256(
            Path(initialization_path).expanduser().resolve().read_bytes()
        ).hexdigest(),
        "payload_sha256": payload_sha256,
        "environment_count": 1,
        "joint_names": list(runtime_names),
        "environment_origin": origin.tolist(),
        "errors": errors,
        "runtime_context_errors": context_errors,
        "runtime_context_readback": {
            name: (
                None
                if name == "wrapper_action_clip" and bool(np.isnan(value))
                else value.tolist()
            )
            for name, value in actual_context.items()
        },
        "readback": {name: value.tolist() for name, value in readback.items()},
        "application_complete": True,
    }
    _atomic_json(Path(report_path).expanduser().resolve(), report)

    raw.observation_manager.reset(env_ids_device)
    raw.obs_buf = raw.observation_manager.compute(update_history=True)
    observations = wrapped_env.process_raw_obs(raw.obs_buf, flatten_dict_obs=True)
    if hasattr(wrapped_env, "_last_obs_dict"):
        wrapped_env._last_obs_dict = observations  # noqa: SLF001
    return observations
