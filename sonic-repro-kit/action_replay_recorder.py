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

from gear_sonic.envs.manager_env.mdp.recorders import _action_term, _minimal_state


ACTION_DIM = 29
PHYSICAL_FIELDS = ("joint_pos", "joint_vel", "base_ang_vel", "gravity_robot")


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
        term = _action_term(self._env)
        joint_names = list(getattr(term, "_joint_names", []))
        if len(joint_names) != ACTION_DIM or len(set(joint_names)) != ACTION_DIM:
            raise ValueError(f"expected 29 unique Action joints, found {joint_names}")
        self._joint_names = tuple(joint_names)
        self._joint_ids = getattr(term, "_joint_ids")
        wrapper = getattr(self._env, "wrapper", None)
        if wrapper is not None and getattr(wrapper, "action_transform_module", None) is not None:
            raise RuntimeError("External Action replay does not support a wrapper Action transform")
        self._states = [
            {
                "physical_state": [],
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
            {"raw_action": [], "processed_action": [], "action_rel": []}
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
        return {
            "default": default.astype(np.float32),
            "scale": scale,
            "offset": offset,
            "clip": clip,
            "wrapper_clip": np.float32(np.nan if wrapper_clip is None else wrapper_clip),
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

    def _append_state(self, previous_raw: torch.Tensor) -> None:
        state = _minimal_state(self._env, previous_raw)
        physical = torch.cat([state[name] for name in PHYSICAL_FIELDS], dim=-1)
        robot = self._env.scene["robot"]
        origins = self._env.scene.env_origins
        for env_id in range(self._env.num_envs):
            target = self._states[env_id]
            target["physical_state"].append(_to_numpy(physical[env_id]))
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
        for env_id in range(self._env.num_envs):
            target = self._actions[env_id]
            target["raw_action"].append(_to_numpy(raw[env_id]))
            target["processed_action"].append(_to_numpy(processed[env_id]))
            target["action_rel"].append(_to_numpy(processed[env_id] - default[env_id]))
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
            replay_path = output / f"{env_id:06d}.replay.npz"
            temporary = output / f".{env_id:06d}.replay.tmp.npz"
            np.savez_compressed(
                temporary,
                **arrays,
                joint_names=np.asarray(self._joint_names),
                action_default=mapping["default"],
                action_scale=mapping["scale"],
                action_offset=mapping["offset"],
                action_clip=mapping["clip"],
                wrapper_action_clip=mapping["wrapper_clip"],
                control_dt=np.float32(self._env.step_dt),
                sim_dt=np.float32(self._env.physics_dt),
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
            trajectory_path = output / f"{env_id:06d}.trajectory.pkl"
            temporary_trajectory = output / f".{env_id:06d}.trajectory.tmp.pkl"
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
