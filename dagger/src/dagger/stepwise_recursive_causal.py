"""Exact-init, per-step recursive causal replay for the 65-token CVAE.

This entry intentionally executes one Action per model call.  It never uses the
32-Action open-loop ``recursive_state_feedback_b`` contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from .runtime.sonic_env import SonicActionContract


def _ensure_kit_path(config) -> None:
    kit = Path(
        str(config.get("sonic_kit_root", Path.home() / "bly" / "sonic-repro-kit"))
    ).expanduser().resolve()
    if not (kit / "exact_replay_initializer.py").is_file():
        raise FileNotFoundError(f"sonic-repro-kit is missing: {kit}")
    if str(kit) not in sys.path:
        sys.path.insert(0, str(kit))


def _load_source(case: dict) -> dict:
    with np.load(Path(case["source_replay"]), allow_pickle=False) as archive:
        state = np.asarray(archive["physical_state_v3"], dtype=np.float32)
        action = np.asarray(
            archive["action_target_canonical"], dtype=np.float32
        )
        root_pos = np.asarray(
            archive["root_pos"] if "root_pos" in archive else archive["root_pos_w"],
            dtype=np.float64,
        )
        root_quat = np.asarray(
            archive["root_quat"]
            if "root_quat" in archive
            else archive["root_quat_w"],
            dtype=np.float64,
        )
        body_pos = (
            np.asarray(archive["body_pos"], dtype=np.float64)
            if "body_pos" in archive
            else None
        )
    return {
        "physical_state": state,
        "canonical_action": action,
        "root_pos_w": root_pos,
        "root_quat_w": root_quat,
        "body_pos_w": body_pos,
        "source_trajectory": case["source_trajectory"],
    }


def _robot_frame(env, contract, physics_state: torch.Tensor) -> dict:
    raw = getattr(env.env, "env", env.env)
    robot = raw.scene["robot"]
    joint_ids = getattr(contract.term, "_joint_ids")
    return {
        "dof_pos": robot.data.joint_pos[:, joint_ids][0].detach().cpu().numpy(),
        "root_pos_w": robot.data.root_pos_w[0].detach().cpu().numpy(),
        "root_quat_w": robot.data.root_quat_w[0].detach().cpu().numpy(),
        "body_pos_w": robot.data.body_pos_w[0].detach().cpu().numpy(),
        "physics_state": physics_state[0].detach().cpu().numpy(),
    }


def _stack_frames(frames: list[dict], actions: list[np.ndarray]) -> dict:
    result = {
        key: np.stack([frame[key] for frame in frames]).astype(
            np.float32 if key == "physics_state" else np.float64
        )
        for key in (
            "dof_pos",
            "root_pos_w",
            "root_quat_w",
            "body_pos_w",
            "physics_state",
        )
    }
    result["canonical_action"] = np.stack(actions).astype(np.float32)
    result["fps"] = 50.0
    return result


def _rmse(value: np.ndarray) -> float | None:
    if value.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def _metrics(source: dict, policy: dict, fall_frame: int | None) -> dict:
    frames = min(
        source["physical_state"].shape[0],
        policy["physics_state"].shape[0],
    )
    if frames <= 0:
        raise ValueError("stepwise replay produced no common frames")
    source_state = source["physical_state"][:frames]
    policy_state = policy["physics_state"][:frames]
    state_error = policy_state - source_state
    contact_error = np.abs(
        (policy_state[:, 68:70] >= 0.5).astype(np.float32)
        - (source_state[:, 68:70] >= 0.5).astype(np.float32)
    )
    root_error = policy["root_pos_w"][:frames] - source["root_pos_w"][:frames]
    action_frames = min(
        source["canonical_action"].shape[0],
        policy["canonical_action"].shape[0],
    )
    action_error = (
        policy["canonical_action"][:action_frames]
        - source["canonical_action"][:action_frames]
    )
    body_mpjpe = None
    if (
        source.get("body_pos_w") is not None
        and policy["body_pos_w"].shape[1:] == source["body_pos_w"].shape[1:]
    ):
        body_error = (
            policy["body_pos_w"][:frames] - source["body_pos_w"][:frames]
        )
        body_mpjpe = float(
            np.linalg.norm(body_error, axis=-1).mean()
        )
    return {
        "frames": int(frames),
        "source_frames": int(source["physical_state"].shape[0]),
        "policy_frames": int(policy["physics_state"].shape[0]),
        "state_rmse_70": _rmse(state_error),
        "state_rmse_68": _rmse(state_error[:, :68]),
        "state_max_abs_70": float(np.abs(state_error).max()),
        "contact_disagreement": float(contact_error.mean()),
        "root_position_rmse_m": _rmse(root_error),
        "body_mpjpe_m": body_mpjpe,
        "action_canonical_rmse": _rmse(action_error),
        "action_frames": int(action_frames),
        "fall_frame": fall_frame,
    }


def _run_policy(
    *,
    env,
    adapter,
    contract,
    exact_init: Path,
    readback_path: Path,
    horizon: int,
    action_clip_value: float | None,
) -> tuple[dict, list[np.ndarray], int | None]:
    from exact_replay_initializer import apply_exact_replay_initialization

    env.reset()
    apply_exact_replay_initialization(env.env, exact_init, readback_path)
    current_state = env.physics_state()
    contract = SonicActionContract(env)
    adapter.action_contract = contract
    adapter.set_action_clip_value(action_clip_value)
    adapter.set_latent_mode("fixed_standard_normal")
    adapter.begin_evaluation()
    adapter.reset(1, current_state)
    frames = [_robot_frame(env, contract, current_state)]
    actions: list[np.ndarray] = []
    fall_frame = None
    for step in range(int(horizon)):
        rollout = adapter.observe()
        raw = rollout.action_raw
        actions.append(rollout.action_canonical[0].detach().cpu().numpy())
        _next_observation, _reward, done, _info = env.step({"actions": raw})
        done_value = bool(done.any().item())
        next_state = env.physics_state()
        if done_value:
            fall_frame = len(frames) - 1
            adapter.after_step(contract.raw_to_canonical(raw), next_state, done)
            break
        frames.append(_robot_frame(env, contract, next_state))
        adapter.after_step(contract.raw_to_canonical(raw), next_state, done)
    return _stack_frames(frames, actions), actions, fall_frame


def run(args: argparse.Namespace) -> dict:
    from .adapters.cvae_student import CvaeStudentAdapter
    from .cli import (
        _close_cvae_env,
        _create_cvae_env_for_keys,
        _launch_app,
        _prepare_release_env,
        _write_json,
    )

    config = OmegaConf.load(args.config)
    _ensure_kit_path(config)
    output = Path(args.output_run).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output run must be new and empty: {output}")
    for folder in ("data", "trajectories", "manifests", "markers", "logs", "videos"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    cases_path = Path(str(config.stepwise.cases_manifest)).expanduser().resolve()
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases.get("cases"), list) or not cases["cases"]:
        raise ValueError("stepwise cases manifest is empty")
    horizon = int(config.stepwise.get("horizon_steps", 192))
    sample_seed = int(config.stepwise.get("sample_seed", 20261023))
    action_clip_value = config.stepwise.get("action_clip_value", 20.0)
    device = torch.device(str(config.get("device", "cuda:0")))
    run_dir = output

    sonic_root = Path(str(config.sonic_repo_root)).expanduser().resolve()
    os.chdir(sonic_root)
    prepared = _prepare_release_env(config, run_dir)
    _launcher, args_cli = _launch_app(config)
    adapters = {
        "original_c": CvaeStudentAdapter(
            checkpoint=config.stepwise.original_checkpoint,
            model_config=config.stepwise.model_config,
            normalization_run=config.stepwise.normalization_run,
            action_contract=None,
            device=device,
            cvae_root=config.stepwise.cvae_root,
            seed=sample_seed,
            source_mix=0.0,
            source_cache_size=0,
            source_dataset_kind="base",
            action_clip_value=action_clip_value,
            raise_on_clip_error=False,
            lr_schedule="constant",
            warmup_steps=0,
            train_steps=1,
            batch_size=1,
        ),
        "finetuned_c": CvaeStudentAdapter(
            checkpoint=config.stepwise.finetuned_checkpoint,
            model_config=config.stepwise.model_config,
            normalization_run=config.stepwise.normalization_run,
            action_contract=None,
            device=device,
            cvae_root=config.stepwise.cvae_root,
            seed=sample_seed,
            source_mix=0.0,
            source_cache_size=0,
            source_dataset_kind="base",
            action_clip_value=action_clip_value,
            raise_on_clip_error=False,
            lr_schedule="constant",
            warmup_steps=0,
            train_steps=1,
            batch_size=1,
        ),
    }

    manifest_cases = []
    for case_index, case in enumerate(cases["cases"]):
        motion_key = str(case["motion_key"])
        exact_init = Path(case["exact_init"]).expanduser().resolve()
        source = _load_source(case)
        env, contract, _teacher = _create_cvae_env_for_keys(
            config, prepared, run_dir, [motion_key], args_cli, device
        )
        try:
            policy_records = {}
            for label, adapter in adapters.items():
                readback = (
                    run_dir
                    / "manifests"
                    / f"exact_init_readback_{case_index:02d}_{label}.json"
                )
                trajectory, _actions, fall_frame = _run_policy(
                    env=env,
                    adapter=adapter,
                    contract=contract,
                    exact_init=exact_init,
                    readback_path=readback,
                    horizon=horizon,
                    action_clip_value=action_clip_value,
                )
                trajectory_path = (
                    run_dir / "trajectories" / label / f"{motion_key}.trajectory.pkl"
                )
                trajectory_path.parent.mkdir(parents=True, exist_ok=True)
                trajectory.update(
                    {
                        "policy": label,
                        "motion_key": motion_key,
                        "fall_frame": fall_frame,
                    }
                )
                with trajectory_path.open("wb") as stream:
                    pickle.dump(trajectory, stream, protocol=pickle.HIGHEST_PROTOCOL)
                policy_records[label] = {
                    "trajectory": str(trajectory_path),
                    "metrics": _metrics(source, trajectory, fall_frame),
                    "readback": str(readback),
                }
        finally:
            _close_cvae_env(env)
        manifest_cases.append(
            {
                "motion_key": motion_key,
                "exact_init": str(exact_init),
                "source_replay": str(Path(case["source_replay"]).resolve()),
                "source_trajectory": str(
                    Path(case["source_trajectory"]).resolve()
                ),
                "policies": policy_records,
            }
        )
        print(f"[stepwise] {motion_key} complete", flush=True)

    manifest = {
        "format_version": "dagger_stepwise_recursive_causal_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "horizon_steps": horizon,
        "execution_contract": "one_action_per_step_causal_prefix",
        "source_mode": "recorded_hdf",
        "cases_manifest": str(cases_path),
        "cases": manifest_cases,
        "quality_note": (
            "Kinematic/Isaac rollouts from one exact initialization; source is "
            "recorded HDF, not a live teacher rollout."
        ),
    }
    _write_json(run_dir / "manifests" / "stepwise_recursive_causal.json", manifest)
    (run_dir / "markers" / "stepwise_recursive_causal.ok").write_text(
        "PASS\n", encoding="utf-8"
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
