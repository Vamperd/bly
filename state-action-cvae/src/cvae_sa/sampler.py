from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import TASK_NAMES
from .dataset import StateActionWindowDataset, WindowRef, read_episode_index
from .masking import MaskBatch, MaskGenerator
from .models import build_model
from .util import atomic_write_json, atomic_write_text, file_sha256, seed_everything


SAMPLE_TASKS = TASK_NAMES + (
    "forward_one", "forward_rollout", "forward_cold", "history_action", "arbitrary"
)


def _batch_sample(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value[None].to(device) if isinstance(value, torch.Tensor) else [value]
        for key, value in sample.items()
    }


def _sample_from_npz(
    path: Path, dataset: StateActionWindowDataset
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    with np.load(path) as values:
        action_key = "action" if "action" in values else "action_rel"
        if dataset.physics_v3 and (
            "robot_information" not in values or "dynamics_context" not in values
        ):
            raise ValueError(
                "Physics v3 NPZ inputs must include robot_information and "
                "dynamics_context; use an indexed HDF5 episode when possible"
            )
        state = np.asarray(values["physical_state"], dtype=np.float32)
        previous = np.asarray(
            values.get("previous_action", np.empty((state.shape[0], 0))),
            dtype=np.float32,
        )
        action = np.asarray(values[action_key], dtype=np.float32)
        scale = np.asarray(values["action_scale"], dtype=np.float32)
        robot_information = np.asarray(
            values.get("robot_information", np.empty((dataset.robot_info_dim,))),
            dtype=np.float32,
        )
        dynamics_context = np.asarray(
            values.get("dynamics_context", np.empty((dataset.dynamics_context_dim,))),
            dtype=np.float32,
        )
        action_before = np.asarray(
            values.get("action_before_window", np.zeros(29)), dtype=np.float32
        )
        joint_robot = np.asarray(
            values.get("joint_robot_information", np.empty((29, dataset.joint_robot_info_dim))),
            dtype=np.float32,
        )
        actuator_type = np.asarray(
            values.get("joint_actuator_type", np.zeros(29)), dtype=np.int64
        )
        global_robot = np.asarray(
            values.get("global_robot_information", np.empty((dataset.global_robot_info_dim,))),
            dtype=np.float32,
        )
        if dataset.physics_v4 and any(
            name not in values
            for name in (
                "action_before_window", "joint_robot_information",
                "joint_actuator_type", "global_robot_information",
            )
        ):
            raise ValueError(
                "Physics CVAE v4 NPZ requires action_before_window and structured robot information"
            )
        normalized = bool(np.asarray(values.get("normalized", False)).item())
        valid_state = np.asarray(
            values.get("valid_state", np.ones(state.shape[0], dtype=bool)), dtype=bool
        )
        valid_action = np.asarray(
            values.get("valid_action", np.ones(action.shape[0], dtype=bool)), dtype=bool
        )
        progress = np.asarray(
            values.get("progress", np.linspace(0.0, 1.0, state.shape[0])),
            dtype=np.float32,
        )
        state_mask = values.get("mask_physical_state")
        action_mask = values.get("mask_action")
        if state_mask is not None and action_mask is not None:
            previous_mask = values.get(
                "mask_previous_action", np.zeros_like(previous, dtype=bool)
            )
            explicit_masks = tuple(
                np.asarray(value, dtype=bool)
                for value in (state_mask, previous_mask, action_mask)
            )
        else:
            explicit_masks = None
    expected = {
        "physical_state": (state.shape[0], dataset.state_dim),
        "previous_action": (state.shape[0], dataset.previous_mean.size),
        "action": (state.shape[0] - 1, 29),
        "action_scale": (29,),
        "robot_information": (dataset.robot_info_dim,),
        "dynamics_context": (dataset.dynamics_context_dim,),
        "action_before_window": (29,),
        "joint_robot_information": (29, dataset.joint_robot_info_dim),
        "joint_actuator_type": (29,),
        "global_robot_information": (dataset.global_robot_info_dim,),
    }
    actual = {
        "physical_state": state.shape,
        "previous_action": previous.shape,
        "action": action.shape,
        "action_scale": scale.shape,
        "robot_information": robot_information.shape,
        "dynamics_context": dynamics_context.shape,
        "action_before_window": action_before.shape,
        "joint_robot_information": joint_robot.shape,
        "joint_actuator_type": actuator_type.shape,
        "global_robot_information": global_robot.shape,
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name}: expected {shape}, found {actual[name]}")
    if not normalized:
        state = (state - dataset.state_mean) / dataset.state_std
        previous = (previous - dataset.previous_mean) / dataset.previous_std
        action = (action - dataset.action_mean) / dataset.action_std
        robot_information = (
            robot_information - dataset.robot_mean
        ) / dataset.robot_std
        dynamics_context = (
            dynamics_context - dataset.dynamics_mean
        ) / dataset.dynamics_std
        action_before = (action_before - dataset.action_mean) / dataset.action_std
        if dataset.physics_v4:
            joint_robot = (
                joint_robot - dataset.joint_robot_mean
            ) / dataset.joint_robot_std
            global_robot = (
                global_robot - dataset.global_robot_mean
            ) / dataset.global_robot_std
    sample = {
        "physical_state": torch.from_numpy(state),
        "previous_action": torch.from_numpy(previous),
        "action": torch.from_numpy(action),
        "action_before_window": torch.from_numpy(action_before),
        "action_scale": torch.from_numpy(scale),
        "robot_information": torch.from_numpy(robot_information),
        "joint_robot_information": torch.from_numpy(joint_robot),
        "joint_actuator_type": torch.from_numpy(actuator_type),
        "global_robot_information": torch.from_numpy(global_robot),
        "dynamics_context": torch.from_numpy(dynamics_context),
        "auxiliary_transition": torch.zeros((action.shape[0], dataset.auxiliary_dim)),
        "valid_state": torch.from_numpy(valid_state),
        "valid_action": torch.from_numpy(valid_action),
        "progress": torch.from_numpy(progress),
        "motion_key": path.stem,
        "package": "external",
        "status": "external",
        "variant_id": -1,
        "window_start": 0,
        "episode_ref": str(path.resolve()),
    }
    return sample, explicit_masks


def _indexed_hdf_sample(
    input_path: Path,
    episode_name: str,
    start: int,
    dataset: StateActionWindowDataset,
) -> dict[str, Any]:
    resolved = str(input_path.expanduser().resolve())
    matches = [
        item
        for item in read_episode_index(dataset.dataset_run)
        if str(Path(item["hdf5_path"]).resolve()) == resolved
        and item["episode"] == episode_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one indexed record for {resolved}::{episode_name}, found {len(matches)}"
        )
    dataset.episodes = matches
    dataset.refs = [WindowRef(0, max(0, min(start, int(matches[0]["steps"]) - 1)))]
    return dataset[0]


def _explicit_mask_batch(
    batch: dict[str, Any],
    masks: tuple[np.ndarray, np.ndarray, np.ndarray],
    task: str,
) -> MaskBatch:
    state, previous, action = (
        torch.from_numpy(value)[None].to(batch["physical_state"].device) for value in masks
    )
    state &= batch["valid_state"][:, :, None]
    previous &= batch["valid_state"][:, :, None]
    action &= batch["valid_action"][:, :, None]
    previous_input = previous.clone()
    if previous_input.shape[-1]:
        previous_input[:, 1:] |= action
    previous_loss = previous.clone()
    if previous_loss.shape[-1]:
        previous_loss[:, 1:] &= ~action
    return MaskBatch(
        state,
        previous_input,
        action,
        state.clone(),
        previous_loss,
        action.clone(),
        0 if task.startswith("forward") else 1 if task in {"inverse", "history_action"} else 2,
        task,
        "external",
        task.startswith("forward") or task == "history_action",
    )


@torch.no_grad()
def sample(
    dataset_run: Path,
    checkpoint_path: Path,
    output_run: Path,
    input_path: Path | None,
    episode: str | None,
    start: int,
    task: str,
    completion: str,
    latent_samples: int,
) -> dict[str, Any]:
    output_run = output_run.expanduser().resolve()
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "videos"):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed_everything(int(config["seed"]))
    dataset_run = dataset_run.expanduser().resolve()
    if file_sha256(dataset_run / "manifests" / "dataset_manifest.json") != checkpoint[
        "dataset_manifest_sha256"
    ]:
        raise ValueError("checkpoint and dataset manifest hashes differ")
    dataset = StateActionWindowDataset(
        dataset_run,
        "test",
        int(config["data"]["window_transitions"]),
        int(config["data"]["validation_stride"]),
        random_crop=False,
    )
    explicit_masks = None
    if input_path is None:
        sample_value = dataset[0]
        input_description = sample_value["episode_ref"]
    elif input_path.suffix.lower() == ".npz":
        sample_value, explicit_masks = _sample_from_npz(input_path, dataset)
        input_description = str(input_path.expanduser().resolve())
    elif input_path.suffix.lower() in {".h5", ".hdf5"}:
        if not episode:
            raise ValueError("--episode is required for an HDF5 input")
        sample_value = _indexed_hdf_sample(input_path, episode, start, dataset)
        input_description = f"{input_path.expanduser().resolve()}::{episode}@{start}"
    else:
        raise ValueError("sample input must be NPZ or indexed HDF5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = _batch_sample(sample_value, device)
    masker = MaskGenerator(config["masking"])
    masks = (
        _explicit_mask_batch(batch, explicit_masks, task)
        if explicit_masks is not None
        else masker.generate(batch, force_task=task, force_completion=completion)
    )
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    generated_state = []
    generated_previous = []
    generated_action = []
    for _ in range(latent_samples):
        output = model(batch, masks, sample_from_prior=True, deterministic=False)
        state_prediction = output.physical_state
        if output.forward_contact_logits is not None and masks.forward_transition is not None:
            forward_steps = masks.forward_transition[:, :, None].expand(
                -1, -1, state_prediction.shape[-1]
            )
            forward_state = batch["physical_state"][:, :-1] + output.forward_delta
            state_prediction = state_prediction.clone()
            state_prediction[:, 1:] = torch.where(
                forward_steps, forward_state, state_prediction[:, 1:]
            )
        action_prediction = output.action
        if output.inverse_action is not None and masks.inverse_transition is not None:
            inverse_steps = masks.inverse_transition[:, :, None].expand_as(action_prediction)
            action_prediction = torch.where(
                inverse_steps, output.inverse_action, action_prediction
            )
        if output.history_action is not None and masks.history_action_transition is not None:
            history_steps = masks.history_action_transition[:, :, None].expand_as(
                action_prediction
            )
            action_prediction = torch.where(
                history_steps, output.history_action, action_prediction
            )
        completed_state = torch.where(
            masks.state_input, state_prediction, batch["physical_state"]
        )
        completed_previous = torch.where(
            masks.previous_input, output.previous_action, batch["previous_action"]
        )
        completed_action = torch.where(masks.action_input, action_prediction, batch["action"])
        generated_state.append(completed_state[0].float().cpu().numpy())
        generated_previous.append(completed_previous[0].float().cpu().numpy())
        generated_action.append(completed_action[0].float().cpu().numpy())
    state_normalized = np.stack(generated_state)
    previous_normalized = np.stack(generated_previous)
    action_normalized = np.stack(generated_action)
    state_values = state_normalized * dataset.state_std + dataset.state_mean
    previous_values = previous_normalized * dataset.previous_std + dataset.previous_mean
    action_values = action_normalized * dataset.action_std + dataset.action_mean
    if previous_values.shape[-1]:
        previous_values[:, 1:] = action_values
    output_path = output_run / "data" / "samples.npz"
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            physical_state_samples=state_values,
            physical_state_mean=state_values.mean(axis=0),
            physical_state_variance=state_values.var(axis=0),
            previous_action_samples=previous_values,
            previous_action_mean=previous_values.mean(axis=0),
            previous_action_variance=previous_values.var(axis=0),
            action_samples=action_values,
            action_mean=action_values.mean(axis=0),
            action_variance=action_values.var(axis=0),
            mask_physical_state=masks.state_input[0].cpu().numpy(),
            mask_previous_action=masks.previous_input[0].cpu().numpy(),
            mask_action=masks.action_input[0].cpu().numpy(),
            valid_state=batch["valid_state"][0].cpu().numpy(),
            valid_action=batch["valid_action"][0].cpu().numpy(),
            action_scale=batch["action_scale"][0].cpu().numpy(),
            robot_information=batch["robot_information"][0].cpu().numpy(),
            action_before_window=batch["action_before_window"][0].cpu().numpy(),
            joint_robot_information=batch["joint_robot_information"][0].cpu().numpy(),
            joint_actuator_type=batch["joint_actuator_type"][0].cpu().numpy(),
            global_robot_information=batch["global_robot_information"][0].cpu().numpy(),
            dynamics_context=batch["dynamics_context"][0].cpu().numpy(),
        )
    os.replace(temporary, output_path)
    manifest = {
        "passed": True,
        "checkpoint": str(checkpoint_path.expanduser().resolve()),
        "dataset_run": str(dataset_run),
        "input": input_description,
        "task": task,
        "completion": completion if task == "completion" else None,
        "latent_samples": latent_samples,
        "output": str(output_path),
        "units": {
            "joint_position": "rad",
            "joint_velocity": "rad/s",
            "base_angular_velocity": "rad/s",
            "base_linear_velocity": "m/s",
            "base_height": "m",
            "foot_contact": "binary",
            "gravity": "unit vector",
            "action": "relative target joint angle, rad",
        },
    }
    atomic_write_json(output_run / "manifests" / "sample.json", manifest)
    atomic_write_text(output_run / "markers" / "cvae_sample.ok", "PASS\n")
    dataset.close()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample masked CVAE completions")
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--episode")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--task", choices=SAMPLE_TASKS, default="completion")
    parser.add_argument("--completion", choices=("element", "step", "feature"), default="step")
    parser.add_argument("--latent-samples", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = sample(
        args.dataset_run,
        args.checkpoint,
        args.output_run,
        args.input,
        args.episode,
        args.start,
        args.task,
        args.completion,
        args.latent_samples,
    )
    print("CVAE sampling: PASS")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
