"""Isaac one-Action-at-a-time replay for the sampled 11/10 MLP CVAE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch

from cvae_sa.dataset import StateActionWindowDataset
from cvae_sa.mlp_action11 import build_recursive_query
from cvae_sa.mlp_cvae11 import MLPConditionalVAE11

HORIZON_ACTIONS = 192


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def _case(cases_path: Path, index: int) -> dict[str, Any]:
    cases = _read_json(cases_path).get("cases")
    if not isinstance(cases, list) or not 0 <= int(index) < len(cases):
        raise ValueError(f"invalid case index {index}: {cases_path}")
    return dict(cases[int(index)])


def _load_source(case: dict[str, Any]) -> dict[str, np.ndarray]:
    path = Path(case["source_replay"]).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        state_key = "physical_state_v3" if "physical_state_v3" in archive else "physical_state"
        states = np.asarray(archive[state_key], dtype=np.float32)
        actions = np.asarray(archive["action_target_canonical"], dtype=np.float32)
    if states.shape != (HORIZON_ACTIONS + 1, 70):
        raise ValueError(f"source State must be [193,70], found {states.shape}")
    if actions.shape != (HORIZON_ACTIONS, 29):
        raise ValueError(f"source Action must be [192,29], found {actions.shape}")
    return {"physical_state": states, "canonical_action": actions}


def _load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") not in {
        "sonic_11_frame_mlp_cvae_checkpoint_v1",
        "sonic_11_frame_mlp_cvae_dagger_checkpoint_v1",
    }:
        raise ValueError("checkpoint is not an 11-frame MLP CVAE checkpoint")
    config = checkpoint.get("config")
    model_config = config.get("model") if isinstance(config, dict) else checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("CVAE checkpoint has no model config")
    model = MLPConditionalVAE11(model_config).to(device)
    state_dict = checkpoint.get("model") or checkpoint.get("model_state_dict")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint


def _load_normalization(dataset_run: Path) -> dict[str, np.ndarray]:
    dataset = StateActionWindowDataset(
        dataset_run.expanduser().resolve(), "train", window_transitions=10,
        validation_stride=1, random_crop=False,
    )
    try:
        return {
            "state_mean": np.asarray(dataset.state_mean, dtype=np.float32).copy(),
            "state_std": np.asarray(dataset.state_std, dtype=np.float32).copy(),
            "action_mean": np.asarray(dataset.action_mean, dtype=np.float32).copy(),
            "action_std": np.asarray(dataset.action_std, dtype=np.float32).copy(),
        }
    finally:
        dataset.close()


def _predict_prior_action(model: MLPConditionalVAE11, query: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    state = query["state"].to(device)
    action = query["action"].to(device)
    state_visible = query["state_visible"].to(device).any(dim=-1)
    action_visible = query["action_visible"].to(device).any(dim=-1)
    state_valid = query["state_valid"].to(device)
    action_valid = query["action_valid"].to(device)
    target_state = torch.zeros_like(state_valid)
    target_action = query["target_action"].to(device).any(dim=-1)
    before = query["action_before"].to(device)
    before_valid = query["action_before_valid"].to(device)
    with torch.inference_mode():
        output = model.prior_forward(
            condition_state=state,
            condition_action=action,
            state_visible=state_visible,
            action_visible=action_visible,
            state_valid=state_valid,
            action_valid=action_valid,
            target_state=target_state,
            target_action=target_action,
            action_before=before,
            action_before_valid=before_valid,
            sample=True,
        )
    return output["p_action"][:, -1]


def _run_lane(args: argparse.Namespace) -> dict[str, Any]:
    from omegaconf import OmegaConf
    from .cli import _close_cvae_env, _create_cvae_env_for_keys, _launch_app, _prepare_release_env
    from .stepwise_recursive_causal import _ensure_kit_path, _load_source as _load_replay_source, _metrics, _robot_frame, _stack_frames

    case = _case(args.cases_manifest.expanduser().resolve(), args.case_index)
    motion_key = str(case["motion_key"])
    source_np = _load_source(case)
    output = args.output_run.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"CVAE replay output must be new and empty: {output}")
    for folder in ("data", "manifests", "markers", "videos", "logs"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    device = torch.device(str(args.device or ("cuda" if torch.cuda.is_available() else "cpu")))
    model, checkpoint = _load_checkpoint(args.checkpoint, device)
    normalization = _load_normalization(args.dataset_run)
    config = OmegaConf.load(args.config.expanduser().resolve())
    import os
    configured_kit = os.environ.get("SONIC_KIT_ROOT")
    if configured_kit:
        config.sonic_kit_root = configured_kit
    _ensure_kit_path(config)
    sonic_root = Path(str(config.sonic_repo_root)).expanduser().resolve()
    os.chdir(sonic_root)
    prepared = _prepare_release_env(config, output)
    _launcher, args_cli = _launch_app(config)
    env, contract, _teacher = _create_cvae_env_for_keys(config, prepared, output, [motion_key], args_cli, device)
    from exact_replay_initializer import apply_exact_replay_initialization

    try:
        env.reset()
        env.reset_motion_to_first_frame()
        env.disable_auto_reset()
        apply_exact_replay_initialization(
            env.env, Path(case["exact_init"]).expanduser().resolve(),
            output / "manifests/exact_initialization_readback.json",
        )
        current = env.physics_state()[0].detach().cpu().numpy().astype(np.float32)
        actual_states: list[np.ndarray] = [current]
        generated_actions: list[np.ndarray] = []
        frames = [_robot_frame(env, contract, env.physics_state())]
        for target_step in range(HORIZON_ACTIONS):
            query = build_recursive_query(
                expected_states=source_np["physical_state"], actual_states=actual_states,
                executed_actions=generated_actions, target_step=target_step,
                state_mean=normalization["state_mean"], state_std=normalization["state_std"],
                action_mean=normalization["action_mean"], action_std=normalization["action_std"],
            )
            normalized = _predict_prior_action(model, query, device)
            canonical = normalized * torch.as_tensor(normalization["action_std"], device=device)[None]
            canonical = canonical + torch.as_tensor(normalization["action_mean"], device=device)[None]
            raw, achieved = contract.canonical_to_raw(canonical)
            if not torch.isfinite(raw).all() or not torch.isfinite(achieved).all():
                raise FloatingPointError(f"MLP CVAE11 produced non-finite Action at step {target_step}")
            env.step({"actions": raw})
            generated_actions.append(achieved[0].detach().cpu().numpy().astype(np.float32))
            physics = env.physics_state()
            actual_states.append(physics[0].detach().cpu().numpy().astype(np.float32))
            frames.append(_robot_frame(env, contract, physics))
    finally:
        _close_cvae_env(env)

    trajectory = _stack_frames(frames, generated_actions)
    trajectory_path = output / "data/mlp_cvae11.trajectory.pkl"
    with trajectory_path.open("wb") as handle:
        pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
    source = _load_replay_source(case)
    metrics = _metrics(source, trajectory, None)
    payload = {
        "format_version": "dagger_mlp_cvae11_recorded_state_replay_v1",
        "motion_key": motion_key,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "dataset_run": str(args.dataset_run.expanduser().resolve()),
        "source_replay": str(Path(case["source_replay"]).expanduser().resolve()),
        "exact_initialization": str(Path(case["exact_init"]).expanduser().resolve()),
        "trajectory": str(trajectory_path), "horizon_actions": HORIZON_ACTIONS,
        "horizon_states": HORIZON_ACTIONS + 1,
        "input_contract": "render_only_right_aligned_current_real_state_expected_next_state_unknown_prefix",
        "target_action_slot": 9, "a10_policy": "not represented in the 11/10 training contract",
        "inference_route": "prior_gaussian_sample_then_shared_decoder",
        "metrics": metrics, "actions": len(generated_actions), "states": len(frames),
        "checkpoint_step": checkpoint.get("optimizer_step"),
    }
    _write_json(output / "manifests/mlp_cvae11_replay.json", payload)
    (output / "markers/mlp_cvae11_replay.ok").write_text("PASS\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-manifest", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    _run_lane(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
