"""Recorder used by generic CVAE Action-mask source capture and physics replay."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from isaaclab.managers import manager_term_cfg, recorder_manager
from isaaclab.utils import configclass
import numpy as np
import torch

from gear_sonic.envs.manager_env.mdp.recorders import (
    PHYSICS_CONTACT_THRESHOLD_N,
    _action_term,
    _minimal_state,
    _physics_nominal_joint_pos,
    _physics_state,
)


ACTION_DIM = 29
PHYSICAL_FIELDS = ("joint_pos", "joint_vel", "base_ang_vel", "gravity_robot")
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
JOINT_ROBOT_INFO_DIM = 11
GLOBAL_ROBOT_INFO_DIM = 9
DYNAMICS_CONTEXT_DIM = 648


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _parameter_for_env(
    value: Any,
    env_id: int,
    num_envs: int,
    tail_shape: tuple[int, ...],
) -> np.ndarray:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim == 0:
        tensor = tensor.expand(tail_shape)
    elif tuple(tensor.shape) == tail_shape:
        pass
    elif tuple(tensor.shape) == (num_envs, *tail_shape):
        tensor = tensor[env_id]
    else:
        try:
            tensor = torch.broadcast_to(tensor, tail_shape)
        except RuntimeError as error:
            raise ValueError(
                f"cannot resolve runtime parameter shape {tuple(tensor.shape)} as {tail_shape}"
            ) from error
    result = tensor.numpy().astype(np.float32, copy=True)
    if result.shape != tail_shape or not np.isfinite(result).all():
        raise ValueError(f"invalid runtime parameter shape/value: {result.shape}")
    return result


class ActionReplayTrajectoryRecorderTerm(recorder_manager.RecorderTerm):
    """Capture model inputs, raw controls, and render-complete Isaac states."""

    cfg: ActionReplayTrajectoryRecorderCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.cfg = cfg
        self._env = env
        self._closed = False
        self._initialized = False
        self._states: list[dict[str, list[np.ndarray]]] = []
        self._actions: list[dict[str, list[np.ndarray]]] = []

    def _initialize(self) -> None:
        if self._initialized:
            return
        if not self.cfg.save_path:
            raise ValueError("Action replay recorder save_path is required")
        if self.cfg.environment_id_offset < 0:
            raise ValueError("environment_id_offset must be non-negative")
        term = _action_term(self._env)
        joint_names = list(getattr(term, "_joint_names", []))
        if len(joint_names) != ACTION_DIM or len(set(joint_names)) != ACTION_DIM:
            raise ValueError(f"expected 29 unique Action joints, found {joint_names}")
        self._joint_names = tuple(joint_names)
        self._joint_ids = getattr(term, "_joint_ids")
        robot = self._env.scene["robot"]
        nominal = getattr(robot.data, "default_joint_pos_nominal", None)
        if nominal is None:
            # Action-mask capture intentionally disables startup randomization.
            # In that case the runtime default is exactly the canonical nominal.
            robot.data.default_joint_pos_nominal = robot.data.default_joint_pos[0].detach().clone()
            self._nominal_source = "runtime_default_no_randomization"
        else:
            self._nominal_source = "default_joint_pos_nominal"
        wrapper = getattr(self._env, "wrapper", None)
        if wrapper is not None and getattr(wrapper, "action_transform_module", None) is not None:
            raise RuntimeError("External Action replay does not support a wrapper Action transform")
        self._states = [
            {
                "physical_state": [],
                "physics_state_v3": [],
                "previous_action_rel": [],
                "joint_pos": [],
                "joint_vel": [],
                "root_pos": [],
                "root_quat": [],
                "root_lin_vel": [],
                "root_ang_vel": [],
                "body_pos": [],
                "body_quat": [],
            }
            for _ in range(self._env.num_envs)
        ]
        self._actions = [
            {
                "raw_action": [],
                "processed_action": [],
                "action_rel": [],
                "action_target_canonical": [],
            }
            for _ in range(self._env.num_envs)
        ]
        self._mapping_cache = [
            self._resolve_mapping(env_id) for env_id in range(self._env.num_envs)
        ]
        Path(self.cfg.save_path).mkdir(parents=True, exist_ok=True)
        self._initialized = True

    def _resolve_mapping(self, env_id: int) -> dict[str, Any]:
        term = _action_term(self._env)
        robot = self._env.scene["robot"]
        default = _to_numpy(robot.data.default_joint_pos[env_id, self._joint_ids])
        nominal = _to_numpy(
            _physics_nominal_joint_pos(self._env, self._joint_ids)[env_id]
        ).astype(np.float32)
        scale = _parameter_for_env(
            getattr(term, "_scale", 1.0), env_id, self._env.num_envs, (ACTION_DIM,)
        )
        offset = _parameter_for_env(
            getattr(term, "_offset", 0.0), env_id, self._env.num_envs, (ACTION_DIM,)
        )
        clip_value = getattr(term, "_clip", None)
        clip = (
            _parameter_for_env(clip_value, env_id, self._env.num_envs, (ACTION_DIM, 2))
            if clip_value is not None
            else np.empty((0,), dtype=np.float32)
        )
        wrapper = getattr(self._env, "wrapper", None)
        wrapper_clip = wrapper.config.get("action_clip_value", None) if wrapper is not None else None
        limits = _to_numpy(robot.data.joint_pos_limits[env_id, self._joint_ids]).astype(
            np.float32
        )
        actuator_type_names = np.full(ACTION_DIM, "unknown", dtype="U64")
        actuator_delay = np.zeros((ACTION_DIM, 2), dtype=np.float32)
        joint_to_index = {name: index for index, name in enumerate(self._joint_names)}
        assigned: set[int] = set()
        for actuator in robot.actuators.values():
            actuator_type = type(actuator).__name__
            delay = (
                float(getattr(actuator.cfg, "min_delay", 0)),
                float(getattr(actuator.cfg, "max_delay", 0)),
            )
            for joint_name in actuator.joint_names:
                if joint_name not in joint_to_index:
                    continue
                index = joint_to_index[joint_name]
                if index in assigned:
                    raise RuntimeError(
                        f"Action joint {joint_name!r} belongs to multiple actuator groups"
                    )
                assigned.add(index)
                actuator_type_names[index] = actuator_type
                actuator_delay[index] = delay
        joint_robot_information = np.column_stack(
            (
                nominal,
                limits[:, 0] - nominal,
                limits[:, 1] - nominal,
                _to_numpy(robot.data.joint_vel_limits[env_id, self._joint_ids]),
                _to_numpy(robot.data.joint_effort_limits[env_id, self._joint_ids]),
                _to_numpy(robot.data.joint_stiffness[env_id, self._joint_ids]),
                _to_numpy(robot.data.joint_damping[env_id, self._joint_ids]),
                _to_numpy(robot.data.joint_armature[env_id, self._joint_ids]),
                _to_numpy(robot.data.joint_friction_coeff[env_id, self._joint_ids]),
                actuator_delay[:, 0],
                actuator_delay[:, 1],
            )
        ).astype(np.float32)
        articulation_props = getattr(getattr(robot.cfg, "spawn", None), "articulation_props", None)
        gravity = tuple(getattr(self._env.cfg.sim, "gravity", (0.0, 0.0, -9.81)))
        global_robot_information = np.asarray(
            [
                float(self._env.physics_dt),
                float(self._env.step_dt),
                float(self._env.cfg.decimation),
                *[float(value) for value in gravity],
                float(getattr(articulation_props, "solver_position_iteration_count", 0) or 0),
                float(getattr(articulation_props, "solver_velocity_iteration_count", 0) or 0),
                float(PHYSICS_CONTACT_THRESHOLD_N),
            ],
            dtype=np.float32,
        )
        terrain_cfg = getattr(self._env.cfg.scene, "terrain", None)
        terrain_material = getattr(terrain_cfg, "physics_material", None)
        ground_material = np.asarray(
            [
                float(getattr(terrain_material, "static_friction", 0.0) or 0.0),
                float(getattr(terrain_material, "dynamic_friction", 0.0) or 0.0),
                float(getattr(terrain_material, "restitution", 0.0) or 0.0),
            ],
            dtype=np.float32,
        )
        dynamics_context = np.concatenate(
            (
                _to_numpy(robot.root_physx_view.get_masses()[env_id]).reshape(-1),
                _to_numpy(robot.root_physx_view.get_inertias()[env_id]).reshape(-1),
                _to_numpy(robot.root_physx_view.get_coms()[env_id]).reshape(-1),
                _to_numpy(
                    robot.root_physx_view.get_material_properties()[env_id]
                ).reshape(-1),
                ground_material,
            )
        ).astype(np.float32)
        if joint_robot_information.shape != (ACTION_DIM, JOINT_ROBOT_INFO_DIM):
            raise RuntimeError(
                f"Unexpected joint RobotInfo shape {joint_robot_information.shape}"
            )
        if global_robot_information.shape != (GLOBAL_ROBOT_INFO_DIM,):
            raise RuntimeError(
                f"Unexpected global RobotInfo shape {global_robot_information.shape}"
            )
        if dynamics_context.shape != (DYNAMICS_CONTEXT_DIM,):
            raise RuntimeError(
                f"Unexpected dynamics context shape {dynamics_context.shape}"
            )
        initial_raw = _to_numpy(self._env.action_manager.prev_action[env_id]).astype(
            np.float32
        )
        initial_processed = initial_raw * scale + offset
        if clip.size:
            initial_processed = np.clip(
                initial_processed, clip[:, 0], clip[:, 1]
            )
        return {
            "default": default.astype(np.float32),
            "nominal": nominal,
            "scale": scale,
            "offset": offset,
            "clip": clip,
            "wrapper_clip": np.float32(np.nan if wrapper_clip is None else wrapper_clip),
            "joint_robot_information": joint_robot_information,
            "joint_actuator_type_names": actuator_type_names,
            "global_robot_information": global_robot_information,
            "dynamics_context": dynamics_context,
            "initial_processed_target_canonical": (
                initial_processed - nominal
            ).astype(np.float32),
        }

    def _mapping(self, env_id: int) -> dict[str, Any]:
        return self._mapping_cache[env_id]

    def _relative_from_raw(self, raw: torch.Tensor, env_id: int) -> np.ndarray:
        mapping = self._mapping(env_id)
        raw_value = _to_numpy(raw[env_id]).astype(np.float32)
        processed = raw_value * mapping["scale"] + mapping["offset"]
        if mapping["clip"].size:
            processed = np.clip(processed, mapping["clip"][:, 0], mapping["clip"][:, 1])
        return (processed - mapping["default"]).astype(np.float32)

    def _canonical_from_raw(self, raw: torch.Tensor, env_id: int) -> np.ndarray:
        mapping = self._mapping(env_id)
        raw_value = _to_numpy(raw[env_id]).astype(np.float32)
        processed = raw_value * mapping["scale"] + mapping["offset"]
        if mapping["clip"].size:
            processed = np.clip(processed, mapping["clip"][:, 0], mapping["clip"][:, 1])
        return (processed - mapping["nominal"]).astype(np.float32)

    def _append_state(self, previous_raw: torch.Tensor) -> None:
        state = _minimal_state(self._env, previous_raw)
        physical = torch.cat([state[name] for name in PHYSICAL_FIELDS], dim=-1)
        physics_v3 = _physics_state(self._env)
        physics_v3_flat = torch.cat(
            [physics_v3[name].to(dtype=torch.float32) for name in PHYSICS_V3_FIELDS],
            dim=-1,
        )
        if physics_v3_flat.shape != (self._env.num_envs, PHYSICS_STATE_DIM):
            raise RuntimeError(
                f"Unexpected Physics v3 State shape {tuple(physics_v3_flat.shape)}"
            )
        robot = self._env.scene["robot"]
        origins = self._env.scene.env_origins
        for env_id in range(self._env.num_envs):
            target = self._states[env_id]
            target["physical_state"].append(_to_numpy(physical[env_id]))
            target["physics_state_v3"].append(_to_numpy(physics_v3_flat[env_id]))
            target["previous_action_rel"].append(self._relative_from_raw(previous_raw, env_id))
            target["joint_pos"].append(_to_numpy(robot.data.joint_pos[env_id, self._joint_ids]))
            target["joint_vel"].append(_to_numpy(robot.data.joint_vel[env_id, self._joint_ids]))
            target["root_pos"].append(
                _to_numpy(robot.data.root_pos_w[env_id] - origins[env_id])
            )
            target["root_quat"].append(_to_numpy(robot.data.root_quat_w[env_id]))
            target["root_lin_vel"].append(_to_numpy(robot.data.root_lin_vel_w[env_id]))
            target["root_ang_vel"].append(_to_numpy(robot.data.root_ang_vel_w[env_id]))
            target["body_pos"].append(
                _to_numpy(robot.data.body_pos_w[env_id] - origins[env_id][None])
            )
            target["body_quat"].append(_to_numpy(robot.data.body_quat_w[env_id]))

    def record_pre_step(self):
        self._initialize()
        action_manager = self._env.action_manager
        if not self._states[0]["physical_state"]:
            self._append_state(action_manager.prev_action)
        term = _action_term(self._env)
        raw = action_manager.action
        processed = term.processed_actions
        robot = self._env.scene["robot"]
        default = robot.data.default_joint_pos[:, self._joint_ids]
        nominal = _physics_nominal_joint_pos(self._env, self._joint_ids)
        for env_id in range(self._env.num_envs):
            target = self._actions[env_id]
            target["raw_action"].append(_to_numpy(raw[env_id]))
            target["processed_action"].append(_to_numpy(processed[env_id]))
            target["action_rel"].append(_to_numpy(processed[env_id] - default[env_id]))
            target["action_target_canonical"].append(
                _to_numpy(processed[env_id] - nominal[env_id])
            )
        return None, None

    def record_post_step(self):
        self._initialize()
        self._append_state(self._env.action_manager.action)
        return None, None

    def close(self, file_path: str):
        del file_path
        if self._closed:
            return
        self._closed = True
        if not self._initialized:
            return
        output = Path(self.cfg.save_path)
        output.mkdir(parents=True, exist_ok=True)
        for env_id in range(self._env.num_envs):
            state = self._states[env_id]
            action = self._actions[env_id]
            state_count = len(state["physical_state"])
            action_count = min(len(action["raw_action"]), max(0, state_count - 1))
            if action_count <= 0:
                continue
            mapping = self._mapping(env_id)
            arrays = {
                name: np.asarray(values[: action_count + 1], dtype=np.float32)
                for name, values in state.items()
            }
            arrays.update(
                {
                    name: np.asarray(values[:action_count], dtype=np.float32)
                    for name, values in action.items()
                }
            )
            artifact_id = env_id + self.cfg.environment_id_offset
            replay_path = output / f"{artifact_id:06d}.replay.npz"
            temporary = output / f".{artifact_id:06d}.replay.tmp.npz"
            np.savez_compressed(
                temporary,
                **arrays,
                joint_names=np.asarray(self._joint_names),
                action_default=mapping["default"],
                nominal_default_joint_pos=mapping["nominal"],
                action_scale=mapping["scale"],
                action_offset=mapping["offset"],
                action_clip=mapping["clip"],
                wrapper_action_clip=mapping["wrapper_clip"],
                control_dt=np.float32(self._env.step_dt),
                sim_dt=np.float32(self._env.physics_dt),
                replay_schema_version=np.asarray("sonic_action_replay_npz_v2"),
                nominal_source=np.asarray(self._nominal_source),
                initial_processed_target_canonical=mapping[
                    "initial_processed_target_canonical"
                ],
                joint_robot_information=mapping["joint_robot_information"],
                joint_actuator_type_names=mapping["joint_actuator_type_names"],
                global_robot_information=mapping["global_robot_information"],
                dynamics_context=mapping["dynamics_context"],
            )
            os.replace(temporary, replay_path)
            trajectory = {
                "dof_pos": arrays["joint_pos"],
                "root_pos_w": arrays["root_pos"],
                "root_quat_w": arrays["root_quat"],
                "quat_format": "wxyz",
                "fps": 1.0 / float(self._env.step_dt),
                "num_joints": ACTION_DIM,
                "total_frames": int(action_count + 1),
                "object_pos_w": None,
                "object_quat_w": None,
                "table_pos_w": None,
                "table_quat_w": None,
            }
            trajectory_path = output / f"{artifact_id:06d}.trajectory.pkl"
            temporary_trajectory = output / f".{artifact_id:06d}.trajectory.tmp.pkl"
            with temporary_trajectory.open("wb") as stream:
                pickle.dump(trajectory, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_trajectory, trajectory_path)

    def close_writers(self) -> None:
        """Match SONIC's trajectory-recorder shutdown interface."""
        self.close("")


@configclass
class ActionReplayTrajectoryRecorderCfg(manager_term_cfg.RecorderTermCfg):
    class_type = ActionReplayTrajectoryRecorderTerm
    save_path: str | None = None
    environment_id_offset: int = 0
