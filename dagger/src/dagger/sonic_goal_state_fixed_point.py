"""SONIC goal/state fixed-point probe for the 65-token DAgger contract.

The probe keeps SONIC's released actor and Isaac environment unchanged.  At
iteration zero it uses the native SONIC reference observation.  At later
iterations it replaces only the two G1 tokenizer goal terms with the previous
iteration's simulated State trajectory, projected to SONIC's joint-position /
joint-velocity / relative-orientation goal representation.

Only converged samples write the compact database: goal trajectory, current
State trajectory, and previous State trajectory.  Actions are intentionally
not persisted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from .adapters.sonic_release_teacher import SonicReleaseTeacher
from .cli import _create_release_env, _launch_app, _prepare_release_env

STATE_DIM = 70
GOAL_DIM = 64
JOINT_DIM = 29
GOAL_JOINT_VEL_DIM = 58
FUTURE_FRAMES = 10
IDENTITY_ROT6 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _load_motion_keys(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys = [str(value) for value in payload.get("motion_keys", [])]
    if len(keys) < 32:
        raise ValueError(f"motion manifest contains {len(keys)} motions; expected at least 32")
    return keys[:32]


def _raw_env(env: Any) -> Any:
    return getattr(env.env, "env", env.env)


def _native_goal(env: Any, observation: dict[str, torch.Tensor]) -> torch.Tensor:
    """Extract current-frame native SONIC goal as [B,64]."""
    tokenizer = observation["tokenizer"]
    command_start, command_end = env._reference_slices["command_multi_future_nonflat"]
    orientation_start, orientation_end = env._reference_slices[
        "motion_anchor_ori_b_mf_nonflat"
    ]
    command = tokenizer[:, command_start:command_end].reshape(
        env.num_envs, FUTURE_FRAMES, GOAL_JOINT_VEL_DIM
    )
    orientation = tokenizer[:, orientation_start:orientation_end].reshape(
        env.num_envs, FUTURE_FRAMES, 6
    )
    return torch.cat((command[:, 0], orientation[:, 0]), dim=-1).detach()


def _state_to_goal(states: torch.Tensor) -> torch.Tensor:
    """Project Physics State [B,70] to SONIC policy goal [B,64]."""
    if states.ndim < 2 or states.shape[-1] != STATE_DIM:
        raise ValueError(f"expected State [...,70], found {tuple(states.shape)}")
    identity = torch.tensor(IDENTITY_ROT6, device=states.device, dtype=states.dtype)
    identity = identity.expand(*states.shape[:-1], 6)
    return torch.cat((states[..., :GOAL_JOINT_VEL_DIM], identity), dim=-1)


def _goal_from_previous_states(previous: torch.Tensor, step: int) -> torch.Tensor:
    """Build [B,10,64] future goals from the previous [32,B,70] trajectory."""
    horizon, batch, width = previous.shape
    offsets = torch.arange(FUTURE_FRAMES, device=previous.device)
    indices = torch.clamp(step + offsets, max=horizon - 1)
    future = previous.index_select(0, indices)
    return _state_to_goal(future).permute(1, 0, 2).contiguous()


def _patch_goal(
    env: Any, observation: dict[str, torch.Tensor], goal_future: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Replace only SONIC's G1 future joint/orientation tokenizer terms."""
    if tuple(goal_future.shape) != (env.num_envs, FUTURE_FRAMES, GOAL_DIM):
        raise ValueError(f"expected goal future [B,10,64], found {tuple(goal_future.shape)}")
    tokenizer = observation["tokenizer"].clone()
    command_start, command_end = env._reference_slices["command_multi_future_nonflat"]
    orientation_start, orientation_end = env._reference_slices[
        "motion_anchor_ori_b_mf_nonflat"
    ]
    command = goal_future[..., :GOAL_JOINT_VEL_DIM].reshape(env.num_envs, -1)
    orientation = goal_future[..., GOAL_JOINT_VEL_DIM:].reshape(env.num_envs, -1)
    if command.shape[-1] != command_end - command_start:
        raise ValueError("SONIC command goal slice width changed")
    if orientation.shape[-1] != orientation_end - orientation_start:
        raise ValueError("SONIC orientation goal slice width changed")
    tokenizer[:, command_start:command_end] = command
    tokenizer[:, orientation_start:orientation_end] = orientation
    result = dict(observation)
    result["tokenizer"] = tokenizer
    return result


def _restore_baseline(
    env: Any,
    baseline: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    raw = _raw_env(env)
    motion = env.motion_term
    motion.set_motion_state(
        baseline["motion_ids"], baseline["time_steps"], baseline["motion_start_time_steps"]
    )
    robot = raw.scene["robot"]
    env_ids = baseline["env_ids"]
    robot.write_root_state_to_sim(baseline["root_state"], env_ids=env_ids)
    robot.write_joint_state_to_sim(
        baseline["joint_pos"], baseline["joint_vel"], env_ids=env_ids
    )
    action_manager = getattr(raw, "action_manager", None)
    if action_manager is not None:
        reset = getattr(action_manager, "reset", None)
        if callable(reset):
            reset(env_ids)
        previous = getattr(action_manager, "prev_action", None)
        if isinstance(previous, torch.Tensor) and previous.shape == baseline["prev_action"].shape:
            previous.copy_(baseline["prev_action"])
    for name in ("episode_length_buf", "reset_buf", "terminated", "truncated"):
        value = getattr(raw, name, None)
        if isinstance(value, torch.Tensor) and value.shape[0] == env.num_envs:
            value.zero_()
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    raw.scene.update(dt=0.0)
    observer = getattr(raw, "observation_manager", None)
    reset_obs = getattr(observer, "reset", None)
    if callable(reset_obs):
        reset_obs(env_ids)
    raw.obs_buf = observer.compute(update_history=True)
    observation = env.env.process_raw_obs(raw.obs_buf, flatten_dict_obs=True)
    if hasattr(env.env, "_last_obs_dict"):
        env.env._last_obs_dict = observation
    return env._augment(observation)


def _baseline(env: Any) -> dict[str, torch.Tensor]:
    raw = _raw_env(env)
    robot = raw.scene["robot"]
    action_manager = getattr(raw, "action_manager", None)
    previous = getattr(action_manager, "prev_action", None)
    if not isinstance(previous, torch.Tensor):
        previous = torch.zeros(
            (env.num_envs, int(env.action_space.shape[-1])), device=env.device, dtype=torch.float32
        )
    return {
        "env_ids": torch.arange(env.num_envs, device=env.device, dtype=torch.long),
        "motion_ids": env.motion_term.motion_ids.clone(),
        "time_steps": env.motion_term.time_steps.clone(),
        "motion_start_time_steps": env.motion_term.motion_start_time_steps.clone(),
        "root_state": robot.data.root_state_w.clone(),
        "joint_pos": robot.data.joint_pos.clone(),
        "joint_vel": robot.data.joint_vel.clone(),
        "prev_action": previous.clone(),
    }


def _metric(current: torch.Tensor, goal: torch.Tensor, previous: torch.Tensor | None) -> dict[str, float]:
    current_goal = _state_to_goal(current)
    goal_error = current_goal - goal
    joint_error = goal_error[..., :GOAL_JOINT_VEL_DIM]
    orientation_error = goal_error[..., GOAL_JOINT_VEL_DIM:]
    goal_joint_rmse = float(joint_error.square().mean().sqrt().item())
    goal_orientation_rmse = float(orientation_error.square().mean().sqrt().item())
    goal_scale = float(goal[..., :GOAL_JOINT_VEL_DIM].square().mean().sqrt().item())
    if previous is None:
        update_joint_rmse = float("nan")
        update_orientation_rmse = float("nan")
        update_scale = float("nan")
    else:
        update = current_goal - _state_to_goal(previous)
        update_joint_rmse = float(update[..., :GOAL_JOINT_VEL_DIM].square().mean().sqrt().item())
        update_orientation_rmse = float(update[..., GOAL_JOINT_VEL_DIM:].square().mean().sqrt().item())
        update_scale = float(_state_to_goal(previous).square().mean().sqrt().item())
    goal_relative = goal_joint_rmse / (goal_scale + 1.0e-3)
    update_relative = (
        update_joint_rmse / (update_scale + 1.0e-3)
        if previous is not None
        else float("nan")
    )
    return {
        "goal_joint_vel_rmse": goal_joint_rmse,
        "goal_orientation_rmse": goal_orientation_rmse,
        "goal_relative_rmse": goal_relative,
        "update_joint_vel_rmse": update_joint_rmse,
        "update_orientation_rmse": update_orientation_rmse,
        "update_relative_rmse": update_relative,
    }


def _finite_or_none(value: float) -> float | None:
    return value if np.isfinite(value) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sonic-root", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--motion-manifest", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--absolute-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    if args.horizon < 1 or args.max_iterations < 1:
        raise ValueError("horizon and max-iterations must be positive")
    keys = _load_motion_keys(args.motion_manifest)
    output = args.output_run.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output run must be new and empty: {output}")
    (output / "manifests").mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "markers").mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config)
    config.mode = "sonic_goal_state_fixed_point"
    config.seed = int(args.seed)
    config.device = str(args.device)
    config.sonic_repo_root = str(args.sonic_root.expanduser().resolve())
    config.teacher_checkpoint = str(args.teacher_checkpoint.expanduser().resolve())
    config.run_root = str(output.parent)
    config.dagger.num_envs = len(keys)
    config.dagger.motion_file = str(args.motion_file.expanduser().resolve())
    config.dagger.motion_keys = keys
    config.dagger.motion_count = 0
    config.dagger.max_unique_motions = len(keys)
    config.dagger.sample_unique_motions = True
    config.dagger.disable_motion_augmentation = True

    original_argv = sys.argv
    sys.argv = [original_argv[0]]
    launcher = None
    env = None
    metrics: dict[str, Any] = {
        "format_version": "sonic_goal_state_fixed_point_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "motion_keys_requested": keys,
        "horizon": int(args.horizon),
        "max_iterations": int(args.max_iterations),
        "goal_representation": "[joint_pos_canonical(29), joint_vel(29), relative_orientation_6d(6)]",
        "state_representation": "Physics State v2 [70]",
        "relative_tolerance": float(args.relative_tolerance),
        "absolute_tolerance": float(args.absolute_tolerance),
        "iterations": [],
        "converged": [],
    }
    try:
        os.chdir(config.sonic_repo_root)
        prepared = _prepare_release_env(config, output)
        terminations = prepared["release_config"].manager_env.terminations
        for name in list(terminations.keys()):
            if name not in {"_target_", "time_out"}:
                del terminations[name]
        launcher, args_cli = _launch_app(config)
        env = _create_release_env(prepared, torch.device(str(args.device)), args_cli)
        observation = env.reset()
        env.disable_auto_reset()
        teacher = SonicReleaseTeacher(
            config=prepared["release_config"],
            env=env,
            checkpoint=str(args.teacher_checkpoint.expanduser().resolve()),
            device=torch.device(str(args.device)),
        )
        baseline = _baseline(env)
        previous_states: torch.Tensor | None = None
        converged_records: list[dict[str, Any]] = []
        motion_keys_loaded = env.motion_keys
        metrics["motion_keys_loaded"] = motion_keys_loaded
        if motion_keys_loaded != keys:
            raise RuntimeError(f"motion order mismatch: requested={keys}, loaded={motion_keys_loaded}")

        for iteration in range(int(args.max_iterations)):
            if iteration > 0:
                observation = _restore_baseline(env, baseline)
            teacher.reset(env.num_envs)
            done = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            state_steps: list[torch.Tensor] = []
            goal_steps: list[torch.Tensor] = []
            for step in range(int(args.horizon)):
                if iteration == 0:
                    policy_observation = observation
                    goal64 = _native_goal(env, observation)
                    goal_future = None
                else:
                    if previous_states is None:
                        raise RuntimeError("missing previous state trajectory")
                    goal_future = _goal_from_previous_states(previous_states, step).to(
                        device=env.device
                    )
                    policy_observation = _patch_goal(env, observation, goal_future)
                    goal64 = goal_future[:, 0]
                goal_steps.append(goal64.detach().cpu())
                action = teacher.act(policy_observation, done)
                observation, _reward, next_done, _info = env.step({"actions": action})
                done = next_done.to(device=env.device).bool()
                state_steps.append(env.physics_state().detach().cpu())
                if bool(done.any()):
                    metrics.setdefault("termination_events", []).append(
                        {"iteration": iteration, "step": step, "env_ids": torch.nonzero(done).flatten().tolist()}
                    )
                    break
            current_states = torch.stack(state_steps, dim=0)
            goals = torch.stack(goal_steps, dim=0)
            if current_states.shape[0] != int(args.horizon):
                raise RuntimeError(
                    f"rollout terminated before horizon: iteration={iteration} steps={current_states.shape[0]}"
                )
            iteration_metrics: dict[str, Any] = {"iteration": iteration, "steps": int(current_states.shape[0])}
            per_motion: list[dict[str, Any]] = []
            for env_id, key in enumerate(keys):
                previous_for_motion = None if previous_states is None else previous_states[:, env_id]
                metric = _metric(current_states[:, env_id], goals[:, env_id], previous_for_motion)
                small_goal = (
                    metric["goal_joint_vel_rmse"] <= args.absolute_tolerance
                    and metric["goal_orientation_rmse"] <= args.absolute_tolerance
                    and metric["goal_relative_rmse"] <= args.relative_tolerance
                )
                small_update = previous_states is not None and (
                    metric["update_joint_vel_rmse"] <= args.absolute_tolerance
                    and metric["update_orientation_rmse"] <= args.absolute_tolerance
                    and metric["update_relative_rmse"] <= args.relative_tolerance
                )
                converged = bool(small_goal and small_update)
                item = {"env_id": env_id, "motion_key": key, **metric, "small_goal": small_goal, "small_update": small_update, "converged": converged}
                per_motion.append(item)
                if converged and not any(r["motion_key"] == key for r in converged_records):
                    converged_records.append(
                        {
                            "env_id": env_id,
                            "motion_key": key,
                            "iteration": iteration,
                            "goal": goals[:, env_id].numpy().astype(np.float32),
                            "current_state": current_states[:, env_id].numpy().astype(np.float32),
                            "previous_state": previous_states[:, env_id].numpy().astype(np.float32),
                        }
                    )
            iteration_metrics["per_motion"] = per_motion
            iteration_metrics["converged_count"] = sum(bool(item["converged"]) for item in per_motion)
            iteration_metrics["max_goal_relative_rmse"] = max(item["goal_relative_rmse"] for item in per_motion)
            iteration_metrics["max_update_relative_rmse"] = (
                None
                if previous_states is None
                else max(item["update_relative_rmse"] for item in per_motion)
            )
            metrics["iterations"].append(iteration_metrics)
            previous_states = current_states
            if len(converged_records) == len(keys):
                break

        metrics["converged"] = [
            {k: value for k, value in record.items() if k not in {"goal", "current_state", "previous_state"}}
            for record in converged_records
        ]
        metrics["converged_count"] = len(converged_records)
        metrics["failed_count"] = len(keys) - len(converged_records)
        metrics["fixed_point_pass"] = len(converged_records) > 0
        metrics["all_motions_converged"] = len(converged_records) == len(keys)
        _write_json(output / "manifests" / "goal_state_fixed_point.json", metrics)
        if converged_records:
            np.savez_compressed(
                output / "data" / "converged_goal_current_previous_state.npz",
                motion_keys=np.asarray([r["motion_key"] for r in converged_records]),
                iteration=np.asarray([r["iteration"] for r in converged_records], dtype=np.int32),
                goal=np.asarray([r["goal"] for r in converged_records], dtype=np.float32),
                current_state=np.asarray([r["current_state"] for r in converged_records], dtype=np.float32),
                previous_state=np.asarray([r["previous_state"] for r in converged_records], dtype=np.float32),
            )
        (output / "markers" / "sonic_goal_state_fixed_point.execution.ok").write_text("PASS\n", encoding="utf-8")
        (output / "markers" / ("sonic_goal_state_fixed_point.converged.ok" if converged_records else "sonic_goal_state_fixed_point.failed.ok")).write_text("PASS\n", encoding="utf-8")
        print(json.dumps({k: metrics[k] for k in ("converged_count", "failed_count", "fixed_point_pass", "all_motions_converged")}, indent=2), flush=True)
        return 0
    finally:
        sys.argv = original_argv
        if env is not None:
            env.close()
        if launcher is not None:
            close = getattr(launcher, "close", None)
            if callable(close):
                close()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
