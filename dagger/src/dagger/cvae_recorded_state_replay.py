"""Recorded-state CVAE replay with a 32-action prefix and 32-action receding window."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import torch


STATE_STEPS = 65
ACTION_STEPS = 64
PREFIX_ACTIONS = 32
HORIZON_ACTIONS = 192


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _case(cases_path: Path, index: int) -> dict:
    payload = _read_json(cases_path)
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"empty cases manifest: {cases_path}")
    if index < 0 or index >= len(cases):
        raise ValueError(f"case index {index} outside 0..{len(cases) - 1}")
    return cases[index]


def _source_arrays(case: dict) -> tuple[np.ndarray, np.ndarray]:
    path = Path(case["source_replay"]).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        state_key = (
            "physical_state_v3"
            if "physical_state_v3" in archive
            else "physical_state"
        )
        states = np.asarray(archive[state_key], dtype=np.float32)
        actions = np.asarray(
            archive["action_target_canonical"], dtype=np.float32
        )
    if states.shape[0] < HORIZON_ACTIONS + 1 or actions.shape[0] < HORIZON_ACTIONS:
        raise ValueError(
            f"source window too short: states={states.shape}, actions={actions.shape}"
        )
    return states[: HORIZON_ACTIONS + 1], actions[:HORIZON_ACTIONS]


def _initial_window(
    source_states: np.ndarray,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    device: torch.device,
):
    states = torch.as_tensor(source_states[:STATE_STEPS], dtype=torch.float32, device=device)
    states = (states - state_mean) / state_std.clamp_min(1.0e-6)
    actions = torch.zeros(ACTION_STEPS, 29, dtype=torch.float32, device=device)
    valid_state = torch.ones(STATE_STEPS, dtype=torch.bool, device=device)
    valid_action = torch.ones(ACTION_STEPS, dtype=torch.bool, device=device)
    batch = {
        "physical_state": states.unsqueeze(0),
        "action": actions.unsqueeze(0),
        "action_before_window": actions[0].unsqueeze(0),
        "valid_state": valid_state.unsqueeze(0),
        "valid_action": valid_action.unsqueeze(0),
    }
    state_mask = torch.zeros(1, STATE_STEPS, 70, dtype=torch.bool, device=device)
    action_mask = torch.ones(1, ACTION_STEPS, 29, dtype=torch.bool, device=device)
    return batch, state_mask, action_mask, action_mean, action_std


def _receding_window(
    real_states: list[np.ndarray],
    source_states: np.ndarray,
    executed_actions: list[np.ndarray],
    action_index: int,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    device: torch.device,
):
    if action_index < PREFIX_ACTIONS or action_index >= HORIZON_ACTIONS:
        raise ValueError("receding action index must be in [32, 191]")
    real = np.asarray(real_states[action_index - PREFIX_ACTIONS : action_index + 1])
    if real.shape != (PREFIX_ACTIONS + 1, 70):
        raise ValueError(f"real state window shape is {real.shape}, expected (33,70)")
    expected = np.asarray(
        [
            source_states[min(value, HORIZON_ACTIONS)]
            for value in range(action_index + 1, action_index + 33)
        ],
        dtype=np.float32,
    )
    states = np.concatenate((real, expected), axis=0)
    if states.shape != (STATE_STEPS, 70):
        raise ValueError(f"receding state window shape is {states.shape}")

    past = np.asarray(
        executed_actions[action_index - PREFIX_ACTIONS : action_index],
        dtype=np.float32,
    )
    if past.shape != (PREFIX_ACTIONS, 29):
        raise ValueError(f"executed action window shape is {past.shape}")
    actions = np.concatenate(
        (past, np.zeros((PREFIX_ACTIONS, 29), dtype=np.float32)), axis=0
    )

    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    states_t = (states_t - state_mean) / state_std.clamp_min(1.0e-6)
    actions_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    actions_t = (actions_t - action_mean) / action_std.clamp_min(1.0e-6)
    valid_state = torch.ones(STATE_STEPS, dtype=torch.bool, device=device)
    valid_action = torch.ones(ACTION_STEPS, dtype=torch.bool, device=device)
    batch = {
        "physical_state": states_t.unsqueeze(0),
        "action": actions_t.unsqueeze(0),
        "action_before_window": actions_t[0].unsqueeze(0),
        "valid_state": valid_state.unsqueeze(0),
        "valid_action": valid_action.unsqueeze(0),
    }
    state_mask = torch.zeros(1, STATE_STEPS, 70, dtype=torch.bool, device=device)
    action_mask = torch.zeros(1, ACTION_STEPS, 29, dtype=torch.bool, device=device)
    action_mask[:, PREFIX_ACTIONS:] = True
    return batch, state_mask, action_mask, action_mean, action_std


def _predict_action(
    adapter,
    batch,
    state_mask,
    action_mask,
    action_index: int,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    contract,
):
    with torch.no_grad():
        output = adapter.model.infer_from_condition(
            batch, state_mask, action_mask, sample=False
        )
    normalized = output.action[0, action_index].float()
    canonical = normalized * action_std + action_mean
    raw, achieved = contract.canonical_to_raw(canonical.unsqueeze(0))
    if not torch.isfinite(raw).all():
        raise FloatingPointError("recorded-state replay produced non-finite Action")
    return raw, achieved[0]


def _run_lane(args: argparse.Namespace) -> dict:
    from omegaconf import OmegaConf

    from .adapters.cvae_student import CvaeStudentAdapter
    from .cli import (
        _close_cvae_env,
        _create_cvae_env_for_keys,
        _launch_app,
        _prepare_release_env,
    )
    from .runtime.sonic_env import SonicActionContract
    from .stepwise_recursive_causal import (
        _ensure_kit_path,
        _load_source,
        _metrics,
        _robot_frame,
        _stack_frames,
    )

    case = _case(Path(args.cases_manifest), int(args.case_index))
    motion_key = str(case["motion_key"])
    source_states, _source_actions = _source_arrays(case)
    source = _load_source(case)
    lane_name = str(args.lane)
    checkpoint = (
        args.original_checkpoint
        if lane_name == "original_c"
        else args.finetuned_checkpoint
    )
    output = Path(args.output_run).expanduser().resolve()
    lane_output = Path(args.lane_output).expanduser().resolve()
    if lane_output.exists() and any(lane_output.iterdir()):
        raise FileExistsError(f"lane output must be new and empty: {lane_output}")
    lane_output.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config)
    _ensure_kit_path(config)
    from exact_replay_initializer import apply_exact_replay_initialization

    device = torch.device(str(config.get("device", "cuda:0")))
    sonic_root = Path(str(config.sonic_repo_root)).expanduser().resolve()
    os.chdir(sonic_root)
    prepared = _prepare_release_env(config, lane_output)
    _launcher, args_cli = _launch_app(config)
    env, contract, _teacher = _create_cvae_env_for_keys(
        config, prepared, lane_output, [motion_key], args_cli, device
    )
    env.reset()
    env.reset_motion_to_first_frame()
    env.disable_auto_reset()
    apply_exact_replay_initialization(
        env.env,
        Path(case["exact_init"]).expanduser().resolve(),
        lane_output / "manifests" / "exact_initialization_readback.json",
    )

    adapter = CvaeStudentAdapter(
        checkpoint=checkpoint,
        model_config=config.stepwise.model_config,
        normalization_run=config.stepwise.normalization_run,
        action_contract=None,
        device=device,
        cvae_root=config.stepwise.cvae_root,
        seed=int(config.seed),
        source_mix=0.0,
        source_cache_size=0,
        source_dataset_kind="base",
        raise_on_clip_error=False,
        lr_schedule="constant",
        warmup_steps=0,
        train_steps=1,
        batch_size=1,
    )
    adapter.action_contract = contract
    adapter.model.eval()

    current_state = env.physics_state()
    real_states = [current_state[0].detach().cpu().numpy()]
    frames = [_robot_frame(env, contract, current_state)]
    executed_actions: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    try:
        batch, state_mask, action_mask, mean, std = _initial_window(
            source_states,
            adapter.state_mean,
            adapter.state_std,
            adapter.action_mean,
            adapter.action_std,
            device,
        )
        with torch.no_grad():
            output = adapter.model.infer_from_condition(
                batch, state_mask, action_mask, sample=False
            )
        initial_actions = output.action[0, :PREFIX_ACTIONS].float()
        for index in range(PREFIX_ACTIONS):
            normalized = initial_actions[index]
            canonical = normalized * adapter.action_std + adapter.action_mean
            raw, achieved = contract.canonical_to_raw(canonical.unsqueeze(0))
            env.step({"actions": raw})
            executed_actions.append(achieved[0].detach().cpu().numpy())
            actions.append(achieved[0].detach().cpu().numpy())
            current_state = env.physics_state()
            real_states.append(current_state[0].detach().cpu().numpy())
            frames.append(_robot_frame(env, contract, current_state))

        for action_index in range(PREFIX_ACTIONS, HORIZON_ACTIONS):
            batch, state_mask, action_mask, mean, std = _receding_window(
                real_states,
                source_states,
                executed_actions,
                action_index,
                adapter.state_mean,
                adapter.state_std,
                adapter.action_mean,
                adapter.action_std,
                device,
            )
            raw, achieved = _predict_action(
                adapter,
                batch,
                state_mask,
                action_mask,
                PREFIX_ACTIONS,
                mean,
                std,
                contract,
            )
            env.step({"actions": raw})
            executed_actions.append(achieved.detach().cpu().numpy())
            actions.append(achieved.detach().cpu().numpy())
            current_state = env.physics_state()
            real_states.append(current_state[0].detach().cpu().numpy())
            frames.append(_robot_frame(env, contract, current_state))
    finally:
        _close_cvae_env(env)

    trajectory = _stack_frames(frames, actions)
    trajectory_path = lane_output / f"{lane_name}.trajectory.pkl"
    with trajectory_path.open("wb") as stream:
        pickle.dump(trajectory, stream, protocol=pickle.HIGHEST_PROTOCOL)
    metrics = _metrics(source, trajectory, None)
    payload = {
        "format_version": "dagger_cvae_recorded_state_lane_v1",
        "lane": lane_name,
        "motion_key": motion_key,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "trajectory": str(trajectory_path),
        "metrics": metrics,
        "actions": int(len(actions)),
        "states": int(len(frames)),
    }
    _write_json(lane_output / f"{lane_name}.json", payload)
    print(
        f"[recorded] {motion_key} {lane_name} actions={len(actions)} "
        f"states={len(frames)}",
        flush=True,
    )
    return payload


def _run_parent(args: argparse.Namespace) -> dict:
    cases_path = Path(args.cases_manifest).expanduser().resolve()
    case = _case(cases_path, int(args.case_index))
    motion_key = str(case["motion_key"])
    output = Path(args.output_run).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output run must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("manifests", "trajectories", "videos", "markers"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    lanes = {}
    for lane_name in ("original_c", "finetuned_c"):
        lane_output = output / lane_name
        command = [
            sys.executable,
            "-m",
            "dagger.cvae_recorded_state_replay",
            "--cases-manifest",
            str(cases_path),
            "--case-index",
            str(args.case_index),
            "--config",
            str(Path(args.config).expanduser().resolve()),
            "--original-checkpoint",
            str(Path(args.original_checkpoint).expanduser().resolve()),
            "--finetuned-checkpoint",
            str(Path(args.finetuned_checkpoint).expanduser().resolve()),
            "--output-run",
            str(output),
            "--lane",
            lane_name,
            "--lane-output",
            str(lane_output),
        ]
        subprocess.run(command, check=True)
        lanes[lane_name] = _read_json(lane_output / f"{lane_name}.json")

    manifest = {
        "format_version": "dagger_cvae_recorded_state_replay_v1",
        "motion_key": motion_key,
        "horizon_actions": HORIZON_ACTIONS,
        "horizon_states": HORIZON_ACTIONS + 1,
        "window_contract": "1+64 initial, 33 real+32 expected receding",
        "source_trajectory": str(Path(case["source_trajectory"]).resolve()),
        "source_replay": str(Path(case["source_replay"]).resolve()),
        "policies": lanes,
        "cases": [
            {
                "motion_key": motion_key,
                "source_trajectory": str(Path(case["source_trajectory"]).resolve()),
                "source_replay": str(Path(case["source_replay"]).resolve()),
                "exact_init": str(Path(case["exact_init"]).resolve()),
                "policies": lanes,
            }
        ],
    }
    _write_json(output / "manifests" / "recorded_state_replay.json", manifest)
    (output / "markers" / "recorded_state_replay.ok").write_text(
        "PASS\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-manifest", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--lane", choices=("original_c", "finetuned_c"))
    parser.add_argument("--lane-output", type=Path)
    args = parser.parse_args(argv)
    if args.lane is not None:
        if args.lane_output is None:
            parser.error("--lane requires --lane-output")
        _run_lane(args)
    else:
        _run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
