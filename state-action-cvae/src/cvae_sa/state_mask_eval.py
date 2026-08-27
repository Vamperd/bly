from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import ACTION_DIM, DEFAULT_WINDOW_TRANSITIONS
from .masking import MaskBatch
from .models import build_model
from .physics_schema import (
    PHYSICS_STATE_DIM,
    dynamics_context_vector,
    load_physics_schema,
    read_physics_states,
    resolve_parameter,
    robot_information_vector,
    structured_robot_information,
)
from .state_masks import (
    PRESET_NAME,
    STATE_FIELD_SLICES,
    StateMaskScenario,
    build_default_scenarios,
    scenario_mask,
)
from .util import atomic_write_json, atomic_write_text, file_sha256, load_json, seed_everything


DATASET_SCHEMA = "sonic_physics_state_action_cvae_dataset_v4"
CHECKPOINT_FORMAT = "sonic_state_action_cvae_checkpoint_v2"
MIN_EPISODE_STEPS = DEFAULT_WINDOW_TRANSITIONS + 64
COMPLETIONS_PATH = Path("data/state_predictions.npz")
SCENARIOS_PATH = Path("manifests/state_mask_scenarios.jsonl")
METRICS_PATH = Path("manifests/state_prediction_metrics.json")
REPRESENTATIVE_VIDEOS = (
    "state_source_recorded.mp4",
    "state_truth_reconstruction.mp4",
    "forward_rollout_8_comparison.mp4",
    "forward_rollout_32_ood_comparison.mp4",
    "state_step_8_comparison.mp4",
    "state_feature_random_25_comparison.mp4",
    "state_semantic_base_motion_comparison.mp4",
    "state_semantic_left_leg_comparison.mp4",
    "all_state_predictions_grid.mp4",
)


def _stable_order(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _read_episode_index(dataset_run: Path) -> list[dict[str, Any]]:
    path = dataset_run / "manifests/episodes.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_npz(path: Path, **values: Any) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _select_record(
    dataset_run: Path,
    split: str,
    package: str,
    motion_key: str,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    if split != "validation":
        raise ValueError("State-mask video validation is restricted to the validation split")
    records = [
        item
        for item in _read_episode_index(dataset_run)
        if item["split"] == split
        and item["package"] == package
        and item["status"] == "completed"
        and int(item["steps"]) >= MIN_EPISODE_STEPS
    ]
    if motion_key != "auto":
        records = [item for item in records if item["motion_key"] == motion_key]
    if variant != "auto":
        variant_id = int(variant)
        if not 0 <= variant_id <= 7:
            raise ValueError("CVAE_STATE_VARIANT must be auto or an integer in 0..7")
        records = [item for item in records if int(item["variant_id"]) == variant_id]
    if not records:
        raise ValueError("no completed validation episode matches the State selection")
    if motion_key == "auto":
        selected_key = min(
            {str(item["motion_key"]) for item in records},
            key=lambda value: _stable_order(seed, value),
        )
        records = [item for item in records if item["motion_key"] == selected_key]
    return min(
        records,
        key=lambda item: _stable_order(
            seed,
            f"{item['motion_key']}:{item['variant_id']}:{item['source_run']}:{item['episode']}",
        ),
    )


def _select_window(actions: np.ndarray) -> tuple[int, int]:
    steps = actions.shape[0]
    first = 32
    last = steps - DEFAULT_WINDOW_TRANSITIONS - 32
    if last < first:
        raise ValueError("episode is too short for a 128-step window with 32-step margins")
    derivative = np.square(np.diff(actions, axis=0)).sum(axis=-1)
    starts = np.arange(first, last + 1)
    scores = np.asarray(
        [derivative[start : start + DEFAULT_WINDOW_TRANSITIONS - 1].mean() for start in starts]
    )
    window_start = int(starts[int(np.argmax(scores))])
    local = derivative[window_start : window_start + DEFAULT_WINDOW_TRANSITIONS - 1]
    block = np.convolve(local, np.ones(32, dtype=np.float32), mode="valid")
    peak_state_start = int(np.argmax(block))
    return window_start, peak_state_start


def _load_source(
    dataset_run: Path, record: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import h5py

    manifest = load_json(dataset_run / "manifests/dataset_manifest.json")
    if manifest.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("State video validation requires the Physics v4 dataset")
    vocabulary = tuple(
        str(value)
        for value in manifest["representations"]["joint_actuator_type"]["vocabulary"]
    )
    actuator_to_id = {name: index for index, name in enumerate(vocabulary)}
    schema = load_physics_schema(Path(record["schema_path"]))
    env_id = int(record["env_id"])
    with h5py.File(record["hdf5_path"], "r") as stream:
        episode = stream[f"data/{record['episode']}"]
        states = read_physics_states(episode["states"])
        actions = np.asarray(
            episode["actions/action_target_canonical"], dtype=np.float32
        )
        initial_action = np.asarray(
            episode["actions/initial_processed_target_canonical"], dtype=np.float32
        )
        root_pos = np.asarray(episode["replay/root_pos_w"], dtype=np.float32)
        root_quat = np.asarray(episode["replay/root_quat_w"], dtype=np.float32)
        context = stream[f"contexts/{record['context_id']}"]
        robot = robot_information_vector(schema, context, env_id)
        joint_robot, actuator_ids, global_robot = structured_robot_information(
            schema, context, env_id, actuator_to_id
        )
        dynamics = dynamics_context_vector(context)
    steps = actions.shape[0]
    if states.shape != (steps + 1, PHYSICS_STATE_DIM):
        raise ValueError("source State and Action lengths are inconsistent")
    if root_pos.shape != (steps + 1, 3) or root_quat.shape != (steps + 1, 4):
        raise ValueError("source replay root trajectory has an invalid shape")
    nominal = resolve_parameter(schema["nominal_default_joint_pos"], env_id).reshape(ACTION_DIM)
    scale = resolve_parameter(schema["action_scale"], env_id).reshape(ACTION_DIM)
    if initial_action.shape != (ACTION_DIM,):
        raise ValueError("initial processed Action must be a 29-vector")
    source = {
        "states": states,
        "actions": actions,
        "initial_action": initial_action,
        "root_pos": root_pos,
        "root_quat": root_quat,
        "joint_names": tuple(str(value) for value in schema["joint_names"]),
        "nominal": nominal.astype(np.float32),
        "action_scale": scale.astype(np.float32),
        "robot": robot.astype(np.float32),
        "joint_robot": joint_robot.astype(np.float32),
        "actuator_ids": actuator_ids.astype(np.int64),
        "global_robot": global_robot.astype(np.float32),
        "dynamics": dynamics.astype(np.float32),
        "control_dt": float(schema["simulation"]["control_dt"]),
    }
    with np.load(dataset_run / "data/normalization.npz") as values:
        normalization = {
            name: np.asarray(values[name], dtype=np.float32)
            for name in (
                "physical_state_mean", "physical_state_std", "action_mean", "action_std",
                "robot_info_mean", "robot_info_std", "joint_robot_info_mean",
                "joint_robot_info_std", "global_robot_info_mean", "global_robot_info_std",
                "dynamics_context_mean", "dynamics_context_std",
            )
        }
    return source, normalization


def _make_batch(
    source: dict[str, Any],
    normalization: dict[str, np.ndarray],
    start: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    stop = start + DEFAULT_WINDOW_TRANSITIONS
    states = source["states"][start : stop + 1]
    actions = source["actions"][start:stop]
    action_before = source["actions"][start - 1] if start > 0 else source["initial_action"]
    state_mean = normalization["physical_state_mean"]
    state_std = normalization["physical_state_std"]
    action_mean = normalization["action_mean"]
    action_std = normalization["action_std"]
    progress = np.arange(start, stop + 1, dtype=np.float32) / max(
        source["actions"].shape[0], 1
    )
    return {
        "physical_state": torch.from_numpy((states - state_mean) / state_std)[None].to(device),
        "previous_action": torch.empty((1, 129, 0), dtype=torch.float32, device=device),
        "action": torch.from_numpy((actions - action_mean) / action_std)[None].to(device),
        "action_before_window": torch.from_numpy((action_before - action_mean) / action_std)[None].to(device),
        "action_scale": torch.from_numpy(source["action_scale"])[None].to(device),
        "robot_information": torch.from_numpy(
            (source["robot"] - normalization["robot_info_mean"])
            / normalization["robot_info_std"]
        )[None].to(device),
        "joint_robot_information": torch.from_numpy(
            (source["joint_robot"] - normalization["joint_robot_info_mean"])
            / normalization["joint_robot_info_std"]
        )[None].to(device),
        "joint_actuator_type": torch.from_numpy(source["actuator_ids"])[None].to(device),
        "global_robot_information": torch.from_numpy(
            (source["global_robot"] - normalization["global_robot_info_mean"])
            / normalization["global_robot_info_std"]
        )[None].to(device),
        "dynamics_context": torch.from_numpy(
            (source["dynamics"] - normalization["dynamics_context_mean"])
            / normalization["dynamics_context_std"]
        )[None].to(device),
        "auxiliary_transition": torch.zeros((1, 128, 35), dtype=torch.float32, device=device),
        "valid_state": torch.ones((1, 129), dtype=torch.bool, device=device),
        "valid_action": torch.ones((1, 128), dtype=torch.bool, device=device),
        "progress": torch.from_numpy(progress)[None].to(device),
    }


def completion_mask_batch(mask: np.ndarray, device: torch.device) -> MaskBatch:
    state = torch.from_numpy(mask)[None].to(device)
    action = torch.zeros((1, 128, ACTION_DIM), dtype=torch.bool, device=device)
    previous = torch.empty((1, 129, 0), dtype=torch.bool, device=device)
    transitions = torch.zeros((1, 128), dtype=torch.bool, device=device)
    return MaskBatch(
        state, previous, action, state.clone(), previous.clone(), action.clone(),
        2, "arbitrary", "external_state", False,
        transitions.clone(), transitions.clone(), transitions.clone(),
        torch.zeros(1, dtype=torch.long, device=device),
        torch.zeros(1, dtype=torch.long, device=device),
    )


def forward_mask_batch(
    start: int, horizon: int, device: torch.device
) -> MaskBatch:
    if start < 0 or start + horizon > 128 or not 1 <= horizon <= 8:
        raise ValueError("one forward segment must contain 1..8 valid transitions")
    state = torch.zeros((1, 129, PHYSICS_STATE_DIM), dtype=torch.bool, device=device)
    state[:, start + 1 :] = True
    action = torch.zeros((1, 128, ACTION_DIM), dtype=torch.bool, device=device)
    previous = torch.empty((1, 129, 0), dtype=torch.bool, device=device)
    transitions = torch.zeros((1, 128), dtype=torch.bool, device=device)
    transitions[:, start : start + horizon] = True
    return MaskBatch(
        state, previous, action, state.clone(), previous.clone(), action.clone(),
        0, "forward_rollout", None, True,
        transitions, torch.zeros_like(transitions), torch.zeros_like(transitions),
        torch.as_tensor([start], dtype=torch.long, device=device),
        torch.as_tensor([horizon], dtype=torch.long, device=device),
    )


@torch.no_grad()
def segmented_forward_rollout(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    start: int,
    horizon: int,
) -> torch.Tensor:
    if horizon <= 0 or start < 0 or start + horizon > batch["action"].shape[1]:
        raise ValueError("forward rollout does not fit the model window")
    working = dict(batch)
    states = batch["physical_state"].clone()
    offset = 0
    while offset < horizon:
        length = min(8, horizon - offset)
        segment_start = start + offset
        working["physical_state"] = states
        output = model(
            working,
            forward_mask_batch(segment_start, length, states.device),
            sample_from_prior=True,
            deterministic=True,
        )
        if output.rollout_state is None or output.rollout_state.shape[1] < length:
            raise RuntimeError("model did not return the requested eight-step rollout segment")
        states[:, segment_start + 1 : segment_start + length + 1] = output.rollout_state[:, :length]
        offset += length
    return states[:, start + 1 : start + horizon + 1]


def _hold_last(
    truth: np.ndarray, mask: np.ndarray, boundary_before: np.ndarray | None = None
) -> np.ndarray:
    result = truth.copy()
    for dimension in range(truth.shape[1]):
        visible = np.flatnonzero(~mask[:, dimension])
        for index in np.flatnonzero(mask[:, dimension]):
            before = visible[visible < index]
            if before.size:
                result[index, dimension] = truth[int(before[-1]), dimension]
            elif boundary_before is not None:
                result[index, dimension] = boundary_before[dimension]
            elif visible.size:
                result[index, dimension] = truth[int(visible[0]), dimension]
    return result


def _linear_interpolation(
    truth: np.ndarray,
    mask: np.ndarray,
    boundary_before: np.ndarray | None = None,
    boundary_after: np.ndarray | None = None,
) -> np.ndarray:
    result = truth.copy()
    times = np.arange(truth.shape[0])
    for dimension in range(truth.shape[1]):
        visible = ~mask[:, dimension]
        xp = times[visible].astype(np.float64)
        fp = truth[visible, dimension].astype(np.float64)
        if boundary_before is not None:
            xp = np.concatenate(([-1.0], xp))
            fp = np.concatenate(([boundary_before[dimension]], fp))
        if boundary_after is not None:
            xp = np.concatenate((xp, [float(truth.shape[0])]))
            fp = np.concatenate((fp, [boundary_after[dimension]]))
        if xp.size:
            result[mask[:, dimension], dimension] = np.interp(
                times[mask[:, dimension]], xp, fp
            )
    return result


def _constant_velocity(
    truth: np.ndarray, start: int, horizon: int, control_dt: float
) -> np.ndarray:
    result = truth.copy()
    anchor = truth[start].copy()
    for offset in range(1, horizon + 1):
        value = anchor.copy()
        value[:29] = anchor[:29] + anchor[29:58] * control_dt * offset
        result[start + offset] = value
    return result


def _error_summary(truth: np.ndarray, value: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = (value - truth)[mask]
    if selected.size == 0:
        raise ValueError("State metric cannot use an empty mask")
    return {
        "mae": float(np.mean(np.abs(selected))),
        "rmse": float(np.sqrt(np.mean(np.square(selected)))),
        "max_abs": float(np.max(np.abs(selected))),
    }


def _state_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    state_std: np.ndarray,
    uncertainty: np.ndarray,
    baseline_a: tuple[str, np.ndarray],
    baseline_b: tuple[str, np.ndarray],
    nominal: np.ndarray,
    joint_limits: np.ndarray,
) -> dict[str, Any]:
    normalized_truth = truth / state_std
    normalized_prediction = prediction / state_std
    result: dict[str, Any] = {
        "masked_elements": int(mask.sum()),
        "masked_fraction": float(mask.mean()),
        "cvae_physical": _error_summary(truth, prediction, mask),
        "cvae_normalized": _error_summary(normalized_truth, normalized_prediction, mask),
        baseline_a[0]: _error_summary(truth, baseline_a[1], mask),
        baseline_b[0]: _error_summary(truth, baseline_b[1], mask),
        "latent_std_mean": float(np.mean(uncertainty[mask])) if mask.any() else 0.0,
        "latent_std_max": float(np.max(uncertainty[mask])) if mask.any() else 0.0,
        "fields": {},
    }
    for name, (low, high) in STATE_FIELD_SLICES.items():
        if name == "base_motion":
            continue
        selected_mask = mask[:, low:high]
        if selected_mask.any():
            result["fields"][name] = _error_summary(
                truth[:, low:high], prediction[:, low:high], selected_mask
            )
    gravity_rows = mask[:, 64:67].any(axis=-1)
    if gravity_rows.any():
        reference = truth[gravity_rows, 64:67]
        candidate = prediction[gravity_rows, 64:67]
        reference /= np.linalg.norm(reference, axis=-1, keepdims=True).clip(1.0e-12)
        candidate /= np.linalg.norm(candidate, axis=-1, keepdims=True).clip(1.0e-12)
        angle = np.degrees(np.arccos(np.sum(reference * candidate, axis=-1).clip(-1.0, 1.0)))
        result["gravity_angle_deg"] = {
            "mean": float(angle.mean()), "max": float(angle.max())
        }
    contact_mask = mask[:, 68:70]
    if contact_mask.any():
        result["contact_accuracy"] = float(
            np.mean((prediction[:, 68:70][contact_mask] >= 0.5) == (truth[:, 68:70][contact_mask] >= 0.5))
        )
    absolute_q = prediction[:, :29] + nominal
    violated = (absolute_q < joint_limits[:, 0]) | (absolute_q > joint_limits[:, 1])
    result["joint_limit_violation_fraction"] = float(violated.mean())
    result["gravity_norm_max_abs_error"] = float(
        np.max(np.abs(np.linalg.norm(prediction[:, 64:67], axis=-1) - 1.0))
    )
    result["beats_baselines"] = {
        baseline_a[0]: result["cvae_physical"]["rmse"] < result[baseline_a[0]]["rmse"],
        baseline_b[0]: result["cvae_physical"]["rmse"] < result[baseline_b[0]]["rmse"],
    }
    return result


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ], dtype=np.float64,
    )


def _yaw_from_quat(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w = quaternion[0]
    xyz = quaternion[1:]
    return (
        vector * (2 * w * w - 1)
        + 2 * xyz * np.dot(xyz, vector)
        + 2 * w * np.cross(xyz, vector)
    )


def reconstruct_root_trajectory(
    states: np.ndarray,
    source_root_pos: np.ndarray,
    source_root_quat: np.ndarray,
    anchor: int,
    control_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    if states.shape[0] != source_root_pos.shape[0] or source_root_quat.shape != (states.shape[0], 4):
        raise ValueError("State and root trajectory lengths differ")
    if not 0 <= anchor < states.shape[0]:
        raise ValueError("root reconstruction anchor is outside the episode")
    position = source_root_pos.astype(np.float64, copy=True)
    quaternion = source_root_quat.astype(np.float64, copy=True)
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(1.0e-12)
    anchor_height = float(states[anchor, 67])
    for index in range(anchor, states.shape[0] - 1):
        current = quaternion[index]
        omega = states[index, 61:64].astype(np.float64)
        angle = float(np.linalg.norm(omega) * control_dt)
        if angle > 1.0e-12:
            axis = omega / np.linalg.norm(omega)
            delta = np.concatenate(([math.cos(angle / 2)], axis * math.sin(angle / 2)))
            integrated = _quat_multiply(current, delta)
        else:
            integrated = current.copy()
        integrated /= np.linalg.norm(integrated).clip(1.0e-12)
        gravity = states[index + 1, 64:67].astype(np.float64)
        gravity /= np.linalg.norm(gravity).clip(1.0e-12)
        pitch = math.asin(float(np.clip(gravity[0], -1.0, 1.0)))
        roll = math.atan2(float(-gravity[1]), float(-gravity[2]))
        next_quat = _quat_from_euler(roll, pitch, _yaw_from_quat(integrated))
        if np.dot(next_quat, current) < 0:
            next_quat = -next_quat
        quaternion[index + 1] = next_quat
        world_velocity = _quat_rotate(current, states[index, 58:61].astype(np.float64))
        position[index + 1, :2] = position[index, :2] + world_velocity[:2] * control_dt
        position[index + 1, 2] = source_root_pos[anchor, 2] + states[index + 1, 67] - anchor_height
    return position.astype(np.float32), quaternion.astype(np.float32)


def _quaternion_error_degrees(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    reference = reference.astype(np.float64)
    value = value.astype(np.float64)
    reference /= np.linalg.norm(reference, axis=-1, keepdims=True).clip(1.0e-12)
    value /= np.linalg.norm(value, axis=-1, keepdims=True).clip(1.0e-12)
    dot = np.abs(np.sum(reference * value, axis=-1)).clip(0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def _root_metrics(
    reference_pos: np.ndarray,
    reference_quat: np.ndarray,
    value_pos: np.ndarray,
    value_quat: np.ndarray,
    start: int,
) -> dict[str, float]:
    difference = value_pos[start:] - reference_pos[start:]
    angle = _quaternion_error_degrees(reference_quat[start:], value_quat[start:])
    return {
        "position_rmse_m": float(np.sqrt(np.mean(np.square(difference)))),
        "position_max_m": float(np.max(np.linalg.norm(difference, axis=-1))),
        "orientation_mean_deg": float(angle.mean()),
        "orientation_max_deg": float(angle.max()),
    }


@torch.no_grad()
def evaluate(
    dataset_run: Path,
    checkpoint_path: Path,
    output_run: Path,
    split: str,
    package: str,
    motion_key: str,
    variant: str,
    preset: str,
    latent_mode: str,
    latent_samples: int,
    render_mode: str,
    root_mode: str,
    seed: int,
) -> dict[str, Any]:
    dataset_run = dataset_run.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    if preset != PRESET_NAME:
        raise ValueError(f"unsupported State mask preset {preset!r}")
    if latent_mode != "prior_mean":
        raise ValueError("State video validation only permits honest prior_mean inference")
    if latent_samples <= 0:
        raise ValueError("latent sample count must be positive")
    if render_mode not in {"none", "representatives", "all"}:
        raise ValueError("State render mode must be none, representatives, or all")
    if root_mode != "integrate_predicted":
        raise ValueError("State root mode must be integrate_predicted")
    marker = dataset_run / "markers/cvae_physics_dataset.ok"
    if not marker.is_file():
        raise FileNotFoundError(marker)
    manifest_path = dataset_run / "manifests/dataset_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("State video validation requires a Physics v4 index")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("State video validation requires a v2 Physics checkpoint")
    if checkpoint.get("config", {}).get("model", {}).get("kind") != "physics_transformer":
        raise ValueError("State video validation requires physics_transformer")
    if checkpoint.get("dataset_manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("checkpoint and dataset manifest hashes differ")
    record = _select_record(dataset_run, split, package, motion_key, variant, seed)
    source, normalization = _load_source(dataset_run, record)
    window_start, peak_state_start = _select_window(source["actions"])
    scenarios = build_default_scenarios(
        window_start, peak_state_start, source["joint_names"], seed
    )
    atomic_write_text(
        output_run / SCENARIOS_PATH,
        "".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in scenarios),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    batch = _make_batch(source, normalization, window_start, device)
    state_mean = normalization["physical_state_mean"]
    state_std = normalization["physical_state_std"]
    truth_window = source["states"][window_start : window_start + 129]
    boundary_before = source["states"][window_start - 1]
    boundary_after = source["states"][window_start + 129]
    joint_limits = source["joint_robot"][:, 1:3] + source["nominal"][:, None]

    completed_states: list[np.ndarray] = []
    absolute_masks: list[np.ndarray] = []
    latent_stds: list[np.ndarray] = []
    predicted_root_pos: list[np.ndarray] = []
    predicted_root_quat: list[np.ndarray] = []
    metrics: dict[str, Any] = {}
    common_anchor = window_start - 1
    truth_root_pos, truth_root_quat = reconstruct_root_trajectory(
        source["states"], source["root_pos"], source["root_quat"],
        common_anchor, source["control_dt"],
    )
    reconstruction_floor = _root_metrics(
        source["root_pos"], source["root_quat"], truth_root_pos, truth_root_quat,
        common_anchor,
    )

    for scenario in scenarios:
        local_mask = scenario_mask(scenario, source["joint_names"])
        seed_everything(scenario.seed)
        if scenario.task == "forward_rollout":
            rollout_start = int(scenario.temporal_selector["start_in_window"]) - 1
            prediction_normalized = segmented_forward_rollout(
                model, batch, rollout_start, scenario.rollout_horizon
            )[0].float().cpu().numpy()
            prediction = prediction_normalized * state_std + state_mean
            predicted_window = truth_window.copy()
            predicted_window[
                rollout_start + 1 : rollout_start + scenario.rollout_horizon + 1
            ] = prediction
            uncertainty = np.zeros_like(truth_window)
            hold = _hold_last(truth_window, local_mask, boundary_before)
            constant = _constant_velocity(
                truth_window, rollout_start, scenario.rollout_horizon, source["control_dt"]
            )
            baselines = (("hold_last", hold), ("constant_velocity", constant))
            effective_candidates = 1
        else:
            masks = completion_mask_batch(local_mask, device)
            output = model(batch, masks, sample_from_prior=True, deterministic=True)
            predicted_window = (
                output.physical_state[0].float().cpu().numpy() * state_std + state_mean
            )
            predicted_window = np.where(local_mask, predicted_window, truth_window)
            samples = []
            for index in range(latent_samples):
                seed_everything(scenario.seed + index + 1)
                sampled = model(batch, masks, sample_from_prior=True, deterministic=False)
                sampled_state = sampled.physical_state[0].float().cpu().numpy() * state_std + state_mean
                samples.append(np.where(local_mask, sampled_state, truth_window))
            uncertainty = np.stack(samples).std(axis=0)
            baselines = (
                ("hold_last", _hold_last(truth_window, local_mask, boundary_before)),
                (
                    "linear_interpolation",
                    _linear_interpolation(
                        truth_window, local_mask, boundary_before, boundary_after
                    ),
                ),
            )
            effective_candidates = latent_samples
        visible_change = float(np.max(np.abs(predicted_window[~local_mask] - truth_window[~local_mask])))
        if visible_change != 0.0:
            raise RuntimeError(f"{scenario.name} changed a visible State value")
        full_state = source["states"].copy()
        full_mask = np.zeros_like(full_state, dtype=bool)
        full_state[window_start : window_start + 129] = predicted_window
        full_mask[window_start : window_start + 129] = local_mask
        root_pos, root_quat = reconstruct_root_trajectory(
            full_state, source["root_pos"], source["root_quat"],
            common_anchor, source["control_dt"],
        )
        first_mask = int(np.flatnonzero(local_mask.any(axis=-1))[0]) + window_start
        scenario_metrics = _state_metrics(
            truth_window, predicted_window, local_mask, state_std, uncertainty,
            baselines[0], baselines[1], source["nominal"], joint_limits,
        )
        scenario_metrics.update(
            {
                "scenario": scenario.to_dict(),
                "visible_state_max_abs_change": visible_change,
                "effective_latent_candidate_count": effective_candidates,
                "oracle_uses_ground_truth_state": False,
                "kinematic_trajectory_vs_truth_reconstruction": _root_metrics(
                    truth_root_pos, truth_root_quat, root_pos, root_quat, first_mask
                ),
            }
        )
        metrics[scenario.name] = scenario_metrics
        completed_states.append(full_state)
        absolute_masks.append(full_mask)
        full_uncertainty = np.zeros_like(full_state)
        full_uncertainty[window_start : window_start + 129] = uncertainty
        latent_stds.append(full_uncertainty)
        predicted_root_pos.append(root_pos)
        predicted_root_quat.append(root_quat)

    _atomic_npz(
        output_run / COMPLETIONS_PATH,
        scenario_names=np.asarray([item.name for item in scenarios]),
        source_state=source["states"],
        completed_state=np.stack(completed_states),
        mask_state=np.stack(absolute_masks),
        latent_state_std=np.stack(latent_stds),
        source_root_pos=source["root_pos"],
        source_root_quat=source["root_quat"],
        truth_reconstructed_root_pos=truth_root_pos,
        truth_reconstructed_root_quat=truth_root_quat,
        predicted_root_pos=np.stack(predicted_root_pos),
        predicted_root_quat=np.stack(predicted_root_quat),
        nominal_joint_pos=source["nominal"],
        joint_names=np.asarray(source["joint_names"]),
        control_dt=np.float32(source["control_dt"]),
        fps=np.float32(1.0 / source["control_dt"]),
        window_start=np.int64(window_start),
        window_length=np.int64(DEFAULT_WINDOW_TRANSITIONS),
        common_anchor=np.int64(common_anchor),
    )
    request = {
        "schema_version": "sonic_state_mask_eval_request_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_run": str(dataset_run),
        "checkpoint": str(checkpoint_path),
        "split": split,
        "package": package,
        "motion_key": record["motion_key"],
        "variant_id": int(record["variant_id"]),
        "source_episode": record["episode"],
        "source_hdf5": record["hdf5_path"],
        "preset": preset,
        "latent_mode": latent_mode,
        "latent_samples": latent_samples,
        "render_mode": render_mode,
        "root_mode": root_mode,
        "seed": seed,
        "window_start": window_start,
        "test_split_consumed": False,
    }
    atomic_write_json(output_run / "manifests/state_mask_request.json", request)
    report = {
        "schema_version": "sonic_state_prediction_metrics_v1",
        "passed": True,
        "engineering_pass": True,
        "model_quality_is_a_gate": False,
        "latent_mode": latent_mode,
        "oracle_uses_ground_truth_state": False,
        "representation_reconstruction_floor": reconstruction_floor,
        "scenario_count": len(scenarios),
        "scenarios": metrics,
    }
    atomic_write_json(output_run / METRICS_PATH, report)
    atomic_write_text(output_run / "markers/state_mask_evaluation.ok", "PASS\n")
    return request


def finalize(output_run: Path, render_mode: str) -> dict[str, Any]:
    output_run = output_run.expanduser().resolve()
    for path in (
        output_run / "markers/state_mask_evaluation.ok",
        output_run / COMPLETIONS_PATH,
        output_run / METRICS_PATH,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    videos: list[str] = []
    if render_mode != "none":
        render_manifest = output_run / "manifests/state_mask_render.json"
        if not render_manifest.is_file():
            raise FileNotFoundError(render_manifest)
        required = REPRESENTATIVE_VIDEOS if render_mode == "representatives" else (
            "state_source_recorded.mp4", "state_truth_reconstruction.mp4",
            "all_state_predictions_grid.mp4",
        )
        for name in required:
            path = output_run / "videos" / name
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
            videos.append(str(path))
    summary = {
        "schema_version": "sonic_state_mask_eval_summary_v1",
        "passed": True,
        "pipeline_completed": True,
        "model_quality_is_a_gate": False,
        "render_mode": render_mode,
        "videos": videos,
    }
    atomic_write_json(output_run / "manifests/state_mask_summary.json", summary)
    atomic_write_text(output_run / "markers/cvae_state_mask_video.ok", "PASS\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Physics v4 State masks and video trajectories")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--dataset-run", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--output-run", type=Path, required=True)
    evaluate_parser.add_argument("--split", default="validation")
    evaluate_parser.add_argument("--package", default="Locomotion")
    evaluate_parser.add_argument("--motion-key", default="auto")
    evaluate_parser.add_argument("--variant", default="auto")
    evaluate_parser.add_argument("--preset", default=PRESET_NAME)
    evaluate_parser.add_argument("--latent-mode", default="prior_mean")
    evaluate_parser.add_argument("--latent-samples", type=int, default=8)
    evaluate_parser.add_argument("--render-mode", choices=("none", "representatives", "all"), default="representatives")
    evaluate_parser.add_argument("--root-mode", default="integrate_predicted")
    evaluate_parser.add_argument("--seed", type=int, default=20260830)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output-run", type=Path, required=True)
    finalize_parser.add_argument("--render-mode", choices=("none", "representatives", "all"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "evaluate":
        result = evaluate(
            args.dataset_run, args.checkpoint, args.output_run, args.split,
            args.package, args.motion_key, args.variant, args.preset,
            args.latent_mode, args.latent_samples, args.render_mode,
            args.root_mode, args.seed,
        )
    else:
        result = finalize(args.output_run, args.render_mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
