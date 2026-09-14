from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .action_masks import relative_to_raw_action
from .models import HierarchicalPosteriorTransformer, build_model, parameter_count
from .physics_schema import read_physics_states, resolve_parameter
from .posterior_capacity import validate_motion_prefix
from .posterior_h50_action_replay import (
    _actuator_type_names,
    _device_batch,
    _error_metrics,
    _select_seen_window,
    _source_contract,
    _write_npz,
    _write_trajectory,
)
from .posterior_t64_protocol import make_autoencode_masks, rows_sha256, window_identity_rows
from .util import atomic_write_json, atomic_write_text, file_sha256, load_json


WINDOW_TRANSITIONS = 64
FORMAT_VERSION = "sonic_exact_replay_initialization_v1"
SCENARIOS = ("original", "h50a_posterior_full_both")
INIT_RELATIVE_PATH = Path("data/exact_initialization.npz")
SOURCE_RELATIVE_PATH = Path("data/recorded_hdf.replay.npz")
SOURCE_TRAJECTORY_RELATIVE_PATH = Path("data/recorded_hdf.trajectory.pkl")
REPLAY_ACTIONS_RELATIVE_PATH = Path("data/exact_replay_actions.npz")
READBACK_RELATIVE_PATHS = (
    Path("manifests/exact_initialization_readback_000000.json"),
    Path("manifests/exact_initialization_readback_000001.json"),
)
VIDEO_RELATIVE_PATHS = (
    Path("videos/recorded_hdf.mp4"),
    Path("videos/original_action_exact_init.mp4"),
    Path("videos/h50a_action_exact_init.mp4"),
)


def _quaternion_error_degrees(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    reference_input = np.asarray(reference)
    value_input = np.asarray(value)
    if reference_input.shape != value_input.shape or reference_input.shape[-1] != 4:
        raise ValueError("quaternion arrays must have matching [...,4] shapes")
    exactly_equal = np.all(reference_input == value_input, axis=-1)
    reference64 = reference_input.astype(np.float64, copy=False)
    value64 = value_input.astype(np.float64, copy=False)
    reference64 /= np.linalg.norm(reference64, axis=-1, keepdims=True).clip(1.0e-12)
    value64 /= np.linalg.norm(value64, axis=-1, keepdims=True).clip(1.0e-12)
    dot = np.abs(np.sum(reference64 * value64, axis=-1)).clip(0.0, 1.0)
    angle = np.degrees(2.0 * np.arccos(dot))
    return np.where(exactly_equal, 0.0, angle)


def _trajectory_metrics(
    reference: dict[str, np.ndarray], value: dict[str, np.ndarray]
) -> dict[str, float]:
    length = min(reference["joint_pos"].shape[0], value["joint_pos"].shape[0])
    if length <= 1:
        raise ValueError("replay trajectory is too short")
    joint_difference = value["joint_pos"][:length] - reference["joint_pos"][:length]
    velocity_difference = value["joint_vel"][:length] - reference["joint_vel"][:length]
    root_difference = value["root_pos"][:length] - reference["root_pos"][:length]
    root_angle = _quaternion_error_degrees(
        reference["root_quat"][:length], value["root_quat"][:length]
    )
    state_dim = reference["physical_state"].shape[-1]
    if value["physical_state"].shape[-1] != state_dim or state_dim != 70:
        raise ValueError("exact replay requires matching 70-dimensional Physics State")
    gravity_reference = reference["physical_state"][:length, 64:67]
    gravity_value = value["physical_state"][:length, 64:67]
    gravity_dot = np.sum(gravity_reference * gravity_value, axis=-1).clip(-1.0, 1.0)
    body_difference = value["body_pos"][:length] - reference["body_pos"][:length]
    contact_reference = reference["physical_state"][:length, 68:70] >= 0.5
    contact_value = value["physical_state"][:length, 68:70] >= 0.5
    return {
        "aligned_state_count": length,
        "physical_state_dimension": state_dim,
        "joint_position_rmse_rad": float(np.sqrt(np.mean(np.square(joint_difference)))),
        "joint_velocity_rmse_rad_s": float(
            np.sqrt(np.mean(np.square(velocity_difference)))
        ),
        "root_position_rmse_m": float(np.sqrt(np.mean(np.square(root_difference)))),
        "root_position_max_m": float(np.max(np.linalg.norm(root_difference, axis=-1))),
        "root_orientation_mean_deg": float(np.mean(root_angle)),
        "root_orientation_max_deg": float(np.max(root_angle)),
        "gravity_mean_deg": float(np.mean(np.degrees(np.arccos(gravity_dot)))),
        "body_mpjpe_m": float(np.mean(np.linalg.norm(body_difference, axis=-1))),
        "minimum_root_height_m": float(np.min(value["root_pos"][:length, 2])),
        "base_linear_velocity_rmse_m_s": float(
            np.sqrt(
                np.mean(
                    np.square(
                        value["physical_state"][:length, 58:61]
                        - reference["physical_state"][:length, 58:61]
                    )
                )
            )
        ),
        "base_angular_velocity_rmse_rad_s": float(
            np.sqrt(
                np.mean(
                    np.square(
                        value["physical_state"][:length, 61:64]
                        - reference["physical_state"][:length, 61:64]
                    )
                )
            )
        ),
        "base_height_rmse_m": float(
            np.sqrt(
                np.mean(
                    np.square(
                        value["physical_state"][:length, 67]
                        - reference["physical_state"][:length, 67]
                    )
                )
            )
        ),
        "foot_contact_accuracy": float(np.mean(contact_reference == contact_value)),
    }


def _payload_sha256(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(key for key in values if key != "payload_sha256"):
        value = np.ascontiguousarray(values[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _identity_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rotate_body_to_world(quaternion_wxyz: np.ndarray, vector_body: np.ndarray) -> np.ndarray:
    """Rotate a vector from the robot frame into world coordinates."""

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    vector = np.asarray(vector_body, dtype=np.float64)
    if quaternion.shape != (4,) or vector.shape != (3,):
        raise ValueError("quaternion/vector shapes must be [4] and [3]")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("zero quaternion cannot rotate a vector")
    w, x, y, z = quaternion / norm
    xyz = np.asarray([x, y, z], dtype=np.float64)
    rotated = vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) + w * vector)
    return rotated.astype(np.float32)


def _raw_from_processed(
    processed: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
    clip: np.ndarray | None,
    wrapper_clip: float | None,
) -> np.ndarray:
    if np.any(np.abs(scale) <= 1.0e-12):
        raise ValueError("cannot invert an Action mapping with zero scale")
    raw = (processed - offset) / scale
    if wrapper_clip is not None:
        raw = np.clip(raw, -wrapper_clip, wrapper_clip)
    reconstructed = raw * scale + offset
    if clip is not None:
        reconstructed = np.clip(reconstructed, clip[:, 0], clip[:, 1])
    error = float(np.max(np.abs(reconstructed - processed)))
    if error > 1.0e-6:
        raise ValueError(f"initial Action target cannot be inverted exactly; max error={error}")
    return raw.astype(np.float32)


def _write_exact_initialization(
    path: Path, arrays: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    normalized["format_version"] = np.asarray(FORMAT_VERSION)
    payload_sha = _payload_sha256(normalized)
    normalized["payload_sha256"] = np.asarray(payload_sha)
    _write_npz(path, **normalized)
    manifest = {
        "format_version": FORMAT_VERSION,
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "payload_sha256": payload_sha,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_sha256(value),
            }
            for name, value in sorted(normalized.items())
            if name != "payload_sha256"
        },
    }
    return payload_sha, manifest


def _load_replay(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        result = {
            name: np.asarray(values[name], dtype=np.float32)
            for name in (
                "physics_state_v3",
                "joint_pos",
                "joint_vel",
                "root_pos",
                "root_quat",
                "root_lin_vel",
                "root_ang_vel",
                "body_pos",
                "raw_action",
            )
        }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"replay contains NaN/Inf: {path}")
    result["physical_state"] = result["physics_state_v3"]
    return result


def _window_context_hash(context_arrays: dict[str, np.ndarray]) -> str:
    return _payload_sha256(context_arrays)


def _recorded_hdf_identity(
    dataset_run: Path, record: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the already-recorded source HDF hash without rehashing a large file."""

    subset_manifest = load_json(dataset_run / "manifests/dataset_manifest.json")
    parent_run = Path(str(subset_manifest["parent_dataset_run"])).expanduser().resolve()
    parent_manifest_path = parent_run / "manifests/dataset_manifest.json"
    expected_parent_manifest_hash = str(
        subset_manifest.get("parent_dataset_manifest_sha256", "")
    )
    actual_parent_manifest_hash = file_sha256(parent_manifest_path)
    if (
        not expected_parent_manifest_hash
        or actual_parent_manifest_hash != expected_parent_manifest_hash
    ):
        raise ValueError("parent dataset manifest does not match the overfit subset identity")
    parent_manifest = load_json(parent_manifest_path)
    source_hashes_path = parent_run / "manifests/source_hashes.json"
    actual_source_hashes_hash = file_sha256(source_hashes_path)
    expected_source_hashes_hash = str(parent_manifest.get("source_hashes_sha256", ""))
    if (
        expected_source_hashes_hash
        and actual_source_hashes_hash != expected_source_hashes_hash
    ):
        raise ValueError("parent source-hash manifest does not match its recorded SHA256")
    recorded_path = Path(str(record["hdf5_path"])).expanduser().resolve()
    recorded_run = Path(str(record["source_run"])).expanduser().resolve()
    for source in load_json(source_hashes_path).get("sources", []):
        if Path(str(source.get("run_dir", ""))).expanduser().resolve() != recorded_run:
            continue
        dataset_file = source.get("files", {}).get("dataset", {})
        if Path(str(dataset_file.get("path", ""))).expanduser().resolve() != recorded_path:
            continue
        if int(dataset_file.get("size_bytes", -1)) != recorded_path.stat().st_size:
            raise ValueError("recorded HDF size differs from the indexed source identity")
        return {
            "path": str(recorded_path),
            "size_bytes": int(dataset_file["size_bytes"]),
            "sha256": str(dataset_file["sha256"]),
            "source_run": str(recorded_run),
            "source_hashes_manifest": str(source_hashes_path),
            "source_hashes_manifest_sha256": actual_source_hashes_hash,
            "source_hashes_manifest_parent_anchor": (
                expected_source_hashes_hash if expected_source_hashes_hash else None
            ),
            "parent_dataset_manifest": str(parent_manifest_path),
            "parent_dataset_manifest_sha256": actual_parent_manifest_hash,
        }
    raise ValueError("selected HDF does not have a matching indexed source SHA256 record")


def prepare(
    dataset_run: Path,
    source_run: Path,
    checkpoint_path: Path,
    output_run: Path,
    *,
    motion_key: str,
    variant_id: int,
    window_start: int,
    seed: int,
) -> dict[str, Any]:
    import h5py
    from torch.utils.data import default_collate

    from .action_mask_eval import _resolve_motion_file
    from .dataset import StateActionWindowDataset
    from .physics_schema import load_physics_schema

    dataset_run = dataset_run.expanduser().resolve()
    source_run = source_run.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    if variant_id < 0 or window_start < 0:
        raise ValueError("variant and window start must be non-negative")
    for child in (
        "data/replay",
        "data/replay_action_slices",
        "manifests",
        "markers",
        "logs",
        "videos",
        "plots",
    ):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    if not (dataset_run / "markers/cvae_overfit_subset.ok").is_file():
        raise FileNotFoundError("exact replay requires the 32-motion overfit subset")
    dataset_manifest_path = dataset_run / "manifests/dataset_manifest.json"
    dataset_hash = file_sha256(dataset_manifest_path)

    base = StateActionWindowDataset(
        dataset_run,
        "train",
        WINDOW_TRANSITIONS,
        WINDOW_TRANSITIONS,
        max_episodes=256,
        random_crop=False,
    )
    motions = validate_motion_prefix(base, 32)
    windows = window_identity_rows(base, range(len(base)))
    windows_hash = rows_sha256(windows)
    checkpoint, source = _source_contract(
        dataset_run,
        source_run,
        checkpoint_path,
        dataset_hash=dataset_hash,
        windows_hash=windows_hash,
    )
    source_index, selected, selection_rule = _select_seen_window(
        windows, motion_key, variant_id, window_start
    )
    item = base[source_index]
    batch_cpu = default_collate([item])
    state_mask_cpu, action_mask_cpu, names = make_autoencode_masks(batch_cpu)
    if names != ["full_both"] or not bool(state_mask_cpu.all()) or not bool(action_mask_cpu.all()):
        raise RuntimeError("H50-A replay must use its trained full-both Mask")

    model = build_model(dict(checkpoint["config"]["model"]))
    if not isinstance(model, HierarchicalPosteriorTransformer):
        raise TypeError("H50-A replay checkpoint built the wrong model type")
    if parameter_count(model) != 51_005_283:
        raise ValueError("H50-A replay model parameter count is not 51,005,283")
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    batch = _device_batch(batch_cpu, device)
    with torch.no_grad():
        output = model(
            batch,
            state_mask_cpu.to(device),
            action_mask_cpu.to(device),
        )
        canonical = model.decode_from_canonical_latents(
            output.global_latent,
            output.local_latents,
            valid_state=batch["valid_state"],
            valid_action=batch["valid_action"],
        )
    canonical_checks = {
        "state": torch.allclose(output.physical_state, canonical.physical_state, rtol=1e-5, atol=1e-7),
        "action": torch.allclose(output.action, canonical.action, rtol=1e-5, atol=1e-7),
        "contact": torch.allclose(
            output.state_contact_logits, canonical.state_contact_logits, rtol=1e-5, atol=1e-7
        ),
    }
    if not all(canonical_checks.values()):
        raise RuntimeError(f"H50-A trained/canonical decode paths disagree: {canonical_checks}")
    predicted_action_normalized = output.action[0].detach().cpu().numpy().astype(np.float32)
    target_action_normalized = batch_cpu["action"][0].numpy().astype(np.float32)
    predicted_action = base.denormalize_action(output.action[0]).detach().cpu().numpy().astype(np.float32)
    target_action = base.denormalize_action(batch["action"][0]).detach().cpu().numpy().astype(np.float32)

    ref = base.refs[source_index]
    record = base.episodes[ref.episode_index]
    if int(ref.fixed_start if ref.fixed_start is not None else -1) != window_start:
        raise RuntimeError("selected dataset ref does not preserve the requested fixed start")
    schema = load_physics_schema(Path(record["schema_path"]))
    env_id = int(record["env_id"])
    window_stop = window_start + WINDOW_TRANSITIONS
    with h5py.File(record["hdf5_path"], "r") as stream:
        episode = stream[f"data/{record['episode']}"]
        if window_stop > int(record["steps"]):
            raise ValueError("selected H50-A window is shorter than T64")
        states = read_physics_states(episode["states"], window_start, window_stop + 1)
        original_canonical = np.asarray(
            episode["actions/action_target_canonical"][window_start:window_stop], dtype=np.float32
        )
        original_raw = np.asarray(
            episode["actions/raw_policy_action"][window_start:window_stop], dtype=np.float32
        )
        processed_abs = np.asarray(
            episode["actions/processed_joint_target_abs"][window_start:window_stop], dtype=np.float32
        )
        root_pos_world = np.asarray(
            episode["replay/root_pos_w"][window_start : window_stop + 1], dtype=np.float32
        )
        root_quat = np.asarray(
            episode["replay/root_quat_w"][window_start : window_stop + 1], dtype=np.float32
        )
        body_pos_world = np.asarray(
            episode["replay/body_pos_w"][window_start : window_stop + 1], dtype=np.float32
        )
        context = stream[f"contexts/{record['context_id']}"]
        context_arrays = {
            name: np.asarray(context[name], dtype=np.float32)
            for name in (
                "runtime_default_joint_pos",
                "action_offset",
                "joint_position_limits",
                "joint_velocity_limits",
                "joint_effort_limits",
                "joint_stiffness",
                "joint_damping",
                "joint_armature",
                "joint_friction",
                "body_mass",
                "body_inertia",
                "body_com",
                "body_material",
                "ground_material",
            )
        }
        episode_context = {
            name: np.asarray(episode[f"episode_context/{name}"], dtype=np.float32)
            for name in (
                "reset_root_pose_delta",
                "reset_root_velocity_delta",
                "reset_joint_pos_delta",
                "reset_joint_vel_delta",
            )
        }
        initial_target_canonical = np.asarray(
            episode["actions/initial_processed_target_canonical"], dtype=np.float32
        )

    nominal = resolve_parameter(schema["nominal_default_joint_pos"], env_id).reshape(29)
    action_scale = resolve_parameter(schema["action_scale"], env_id).reshape(29)
    action_clip_entry = schema.get("action_clip")
    action_clip = (
        None
        if action_clip_entry is None
        else resolve_parameter(action_clip_entry, env_id).reshape(29, 2)
    )
    wrapper_value = schema.get("wrapper_action_clip")
    wrapper_clip = None if wrapper_value is None else float(wrapper_value)
    predicted_raw, achieved_action, saturated = relative_to_raw_action(
        predicted_action,
        nominal,
        action_scale,
        context_arrays["action_offset"],
        action_clip,
        wrapper_clip,
    )
    if window_start == 0:
        previous_processed = initial_target_canonical + nominal
        previous_raw = _raw_from_processed(
            previous_processed,
            action_scale,
            context_arrays["action_offset"],
            action_clip,
            wrapper_clip,
        )
        previous_source = "episode initial processed target, inverted through recorded Action mapping"
    else:
        import h5py

        with h5py.File(record["hdf5_path"], "r") as stream:
            episode = stream[f"data/{record['episode']}"]
            previous_raw = np.asarray(
                episode["actions/raw_policy_action"][window_start - 1], dtype=np.float32
            )
            previous_processed = np.asarray(
                episode["actions/processed_joint_target_abs"][window_start - 1], dtype=np.float32
            )
        previous_source = "recorded Action immediately before the selected window"

    translation = np.asarray([root_pos_world[0, 0], root_pos_world[0, 1], 0.0], dtype=np.float32)
    root_pos = root_pos_world - translation
    body_pos = body_pos_world - translation[None, None]
    joint_pos = states[:, :29] + nominal[None]
    root_lin_vel_world = rotate_body_to_world(root_quat[0], states[0, 58:61])
    root_ang_vel_world = rotate_body_to_world(root_quat[0], states[0, 61:64])

    hdf_identity = {
        **_recorded_hdf_identity(dataset_run, record),
        "episode": str(record["episode"]),
        "context_id": str(record["context_id"]),
    }
    episode_arrays = {
        "states": states,
        "original_raw": original_raw,
        "processed_abs": processed_abs,
        "root_pos": root_pos,
        "root_quat": root_quat,
        "body_pos": body_pos,
        **{f"episode_context_{name}": value for name, value in episode_context.items()},
    }
    window_identity = {
        "dataset_index": int(source_index),
        "motion_key": str(selected["motion_key"]),
        "variant_id": int(selected["variant_id"]),
        "window_start": int(selected["window_start"]),
        "window_transitions": WINDOW_TRANSITIONS,
        "episode": str(record["episode"]),
        "context_id": str(record["context_id"]),
    }
    init_arrays: dict[str, Any] = {
        "joint_names": np.asarray(schema["joint_names"]),
        "joint_pos": joint_pos[0],
        "joint_vel": states[0, 29:58],
        "root_pos_relative": root_pos[0],
        "root_quat_wxyz": root_quat[0],
        "root_lin_vel_world": root_lin_vel_world,
        "root_ang_vel_world": root_ang_vel_world,
        "body_pos_relative": body_pos[0],
        "physics_state_v3": states[0],
        "nominal_default_joint_pos": nominal,
        "runtime_default_joint_pos": context_arrays["runtime_default_joint_pos"],
        "action_scale": action_scale,
        "action_offset": context_arrays["action_offset"],
        "action_clip": (
            np.empty((0,), dtype=np.float32) if action_clip is None else action_clip
        ),
        "wrapper_action_clip": np.float32(
            np.nan if wrapper_clip is None else wrapper_clip
        ),
        "joint_position_limits": context_arrays["joint_position_limits"],
        "joint_velocity_limits": context_arrays["joint_velocity_limits"],
        "joint_effort_limits": context_arrays["joint_effort_limits"],
        "joint_stiffness": context_arrays["joint_stiffness"],
        "joint_damping": context_arrays["joint_damping"],
        "joint_armature": context_arrays["joint_armature"],
        "joint_friction": context_arrays["joint_friction"],
        "body_mass": context_arrays["body_mass"],
        "body_inertia": context_arrays["body_inertia"],
        "body_com": context_arrays["body_com"],
        "body_material": context_arrays["body_material"],
        "ground_material": context_arrays["ground_material"],
        "previous_raw_action": previous_raw,
        "previous_processed_action": previous_processed,
        "initial_joint_target_abs": previous_processed,
        **{name: value for name, value in episode_context.items()},
        "dataset_manifest_sha256": np.asarray(dataset_hash),
        "selected_windows_sha256": np.asarray(windows_hash),
        "window_identity_sha256": np.asarray(_identity_sha256(window_identity)),
        "hdf_identity_sha256": np.asarray(_identity_sha256(hdf_identity)),
        "episode_arrays_sha256": np.asarray(_window_context_hash(episode_arrays)),
        "context_arrays_sha256": np.asarray(_window_context_hash(context_arrays)),
        "source_checkpoint_sha256": np.asarray(source["checkpoint_sha256"]),
    }
    initialization_path = output_run / INIT_RELATIVE_PATH
    payload_sha, initialization_manifest = _write_exact_initialization(
        initialization_path, init_arrays
    )
    initialization_manifest.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_identity": window_identity,
            "window_identity_sha256": _identity_sha256(window_identity),
            "hdf_identity": hdf_identity,
            "hdf_identity_sha256": _identity_sha256(hdf_identity),
            "episode_arrays_sha256": _window_context_hash(episode_arrays),
            "context_arrays_sha256": _window_context_hash(context_arrays),
            "source_world_xy_translation_removed": translation.tolist(),
            "previous_action_source": previous_source,
            "base_velocity_transform": "world_vector = quaternion_wxyz rotate(body_vector)",
        }
    )
    atomic_write_json(
        output_run / "manifests/exact_initialization.json", initialization_manifest
    )

    fps = 1.0 / float(schema["simulation"]["control_dt"])
    source_npz = output_run / SOURCE_RELATIVE_PATH
    _write_npz(
        source_npz,
        physical_state=states,
        physics_state_v3=states,
        joint_pos=joint_pos,
        joint_vel=states[:, 29:58],
        root_pos=root_pos,
        root_quat=root_quat,
        root_lin_vel=np.vstack(
            [rotate_body_to_world(q, v) for q, v in zip(root_quat, states[:, 58:61], strict=True)]
        ),
        root_ang_vel=np.vstack(
            [rotate_body_to_world(q, v) for q, v in zip(root_quat, states[:, 61:64], strict=True)]
        ),
        body_pos=body_pos,
        raw_action=original_raw,
        processed_action=processed_abs,
        action_rel=original_canonical,
        action_target_canonical=original_canonical,
        joint_names=np.asarray(schema["joint_names"]),
        action_default=context_arrays["runtime_default_joint_pos"],
        nominal_default_joint_pos=nominal,
        action_scale=action_scale,
        action_offset=context_arrays["action_offset"],
        action_clip=(np.empty((0,), dtype=np.float32) if action_clip is None else action_clip),
        wrapper_action_clip=np.float32(np.nan if wrapper_clip is None else wrapper_clip),
        control_dt=np.float32(schema["simulation"]["control_dt"]),
        sim_dt=np.float32(schema["simulation"]["sim_dt"]),
        initial_processed_target_canonical=previous_processed - nominal,
        joint_robot_information=np.column_stack(
            (
                nominal,
                context_arrays["joint_position_limits"][:, 0] - nominal,
                context_arrays["joint_position_limits"][:, 1] - nominal,
                context_arrays["joint_velocity_limits"],
                context_arrays["joint_effort_limits"],
                context_arrays["joint_stiffness"],
                context_arrays["joint_damping"],
                context_arrays["joint_armature"],
                context_arrays["joint_friction"],
                np.zeros(29, dtype=np.float32),
                np.zeros(29, dtype=np.float32),
            )
        ).astype(np.float32),
        joint_actuator_type_names=_actuator_type_names(schema),
        global_robot_information=np.asarray(
            [
                schema["simulation"]["sim_dt"],
                schema["simulation"]["control_dt"],
                schema["simulation"]["decimation"],
                *schema["simulation"].get("gravity_w", [0.0, 0.0, -9.81]),
                schema["simulation"].get("solver_position_iteration_count", 0),
                schema["simulation"].get("solver_velocity_iteration_count", 0),
                schema.get("contact", {}).get("threshold_n", 10.0),
            ],
            dtype=np.float32,
        ),
        dynamics_context=np.concatenate(
            [
                context_arrays[name].reshape(-1)
                for name in ("body_mass", "body_inertia", "body_com", "body_material", "ground_material")
            ]
        ).astype(np.float32),
        replay_schema_version=np.asarray("sonic_h50a_exact_recorded_hdf_v1"),
        nominal_source=np.asarray("training_hdf5_variant_context"),
    )
    _write_trajectory(
        output_run / SOURCE_TRAJECTORY_RELATIVE_PATH,
        joint_pos=joint_pos,
        root_pos=root_pos,
        root_quat=root_quat,
        fps=fps,
    )
    _write_npz(
        output_run / REPLAY_ACTIONS_RELATIVE_PATH,
        raw_actions=np.stack((original_raw, predicted_raw), axis=1),
        scenario_names=np.asarray(SCENARIOS),
    )

    motion_path, motion_provenance = _resolve_motion_file(record)
    request = {
        "schema_version": "sonic_h50a_exact_init_action_replay_request_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_run": str(dataset_run),
        "dataset_manifest_sha256": dataset_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": source["checkpoint_sha256"],
        **window_identity,
        "mask_name": "full_both",
        "replay_steps": WINDOW_TRANSITIONS,
        "post_window_replay_steps": 0,
        "seed": int(seed),
        "motion_file": str(motion_path),
        "motion_file_sha256": motion_provenance["sha256"],
        "motion_manifest": motion_provenance["manifest"],
        "selection_rule": selection_rule,
        "render_mode": "three_independent_videos_and_triptych",
    }
    atomic_write_json(output_run / "manifests/action_mask_request.json", request)
    replay_request = {
        "schema_version": "sonic_action_replay_request_exact_init_v1",
        "representation": "physics_v4",
        "motion_key": str(selected["motion_key"]),
        "motion_file": str(motion_path),
        "motion_file_sha256": motion_provenance["sha256"],
        "raw_actions_file": str((output_run / REPLAY_ACTIONS_RELATIVE_PATH).resolve()),
        "raw_actions_sha256": file_sha256(output_run / REPLAY_ACTIONS_RELATIVE_PATH),
        "steps": WINDOW_TRANSITIONS,
        "num_envs": 2,
        "scenario_names": list(SCENARIOS),
        "source_capture": str(source_npz.resolve()),
        "control_dt": float(schema["simulation"]["control_dt"]),
        "latent_mode": "posterior_mean",
        "execution_mode": "two_serial_independent_single_environment_processes",
        "exact_initialization_file": str(initialization_path.resolve()),
        "exact_initialization_file_sha256": initialization_manifest["file_sha256"],
        "exact_initialization_payload_sha256": payload_sha,
        "exact_initialization_report_paths": [
            str((output_run / path).resolve()) for path in READBACK_RELATIVE_PATHS
        ],
    }
    atomic_write_json(output_run / "manifests/action_replay_request.json", replay_request)
    offline = {
        "format_version": "sonic_h50a_exact_init_offline_metrics_v1",
        "selection": {**window_identity, "rule": selection_rule},
        "checkpoint": source,
        "dataset": {
            "motion_count": len(motions),
            "episode_count": len(base.episodes),
            "window_count": len(base),
            "selected_windows_sha256": windows_hash,
        },
        "model": {
            "parameter_count": parameter_count(model),
            "posterior_global_shape": list(output.global_latent.shape),
            "posterior_local_shape": list(output.local_latents.shape),
            "trained_and_canonical_decode_match": canonical_checks,
        },
        "action": {
            "normalized": _error_metrics(predicted_action_normalized, target_action_normalized),
            "physical_rad": _error_metrics(achieved_action, target_action),
            "requested_rad": _error_metrics(predicted_action, target_action),
            "saturated_element_count": int(saturated.sum()),
            "saturated_element_fraction": float(saturated.mean()),
        },
        "initialization": initialization_manifest,
        "scope": (
            "one seen H50-A T64 posterior/full-both window; exact historical initialization; "
            "not conditional-prior, KL, sampling, or unseen-motion evaluation"
        ),
    }
    atomic_write_json(output_run / "manifests/h50a_exact_init_offline_metrics.json", offline)
    for marker in ("action_mask_prepare.ok", "action_mask_source.ok", "action_mask_completion.ok"):
        atomic_write_text(output_run / "markers" / marker, "PASS\n")
    return {
        "output_run": str(output_run),
        "selected_window": window_identity,
        "checkpoint_step": source["optimizer_step"],
        "initialization_payload_sha256": payload_sha,
        "offline_action_rmse_rad": offline["action"]["physical_rad"]["rmse"],
        "replay_steps": WINDOW_TRANSITIONS,
    }


def _readback_array(report: dict[str, Any], name: str) -> np.ndarray:
    return np.asarray(report["readback"][name], dtype=np.float64)


def _readback_identity_metrics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    fields = tuple(sorted(set(reports[0]["readback"]).intersection(reports[1]["readback"])))
    differences = {
        name: float(np.max(np.abs(_readback_array(reports[0], name) - _readback_array(reports[1], name))))
        for name in fields
    }
    orientation_error = float(
        np.max(
            _quaternion_error_degrees(
                _readback_array(reports[0], "root_quat_wxyz")[None],
                _readback_array(reports[1], "root_quat_wxyz")[None],
            )
        )
    )
    context_fields = tuple(
        sorted(
            set(reports[0]["runtime_context_readback"]).intersection(
                reports[1]["runtime_context_readback"]
            )
        )
    )
    context_differences = {}
    for name in context_fields:
        left_value = reports[0]["runtime_context_readback"][name]
        right_value = reports[1]["runtime_context_readback"][name]
        if left_value is None or right_value is None:
            context_differences[name] = (
                0.0 if left_value is None and right_value is None else float("inf")
            )
            continue
        left = np.asarray(left_value)
        right = np.asarray(right_value)
        if np.array_equal(left, right, equal_nan=True):
            context_differences[name] = 0.0
        else:
            difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
            context_differences[name] = (
                float(np.max(difference))
                if difference.size and np.isfinite(difference).all()
                else float("inf")
            )
    return {
        "per_field_max_abs": differences,
        "runtime_context_per_field_max_abs": context_differences,
        "max_abs": max((*differences.values(), *context_differences.values())),
        "root_orientation_error_deg": orientation_error,
    }


def _per_frame_errors(reference: dict[str, np.ndarray], value: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    length = min(reference["joint_pos"].shape[0], value["joint_pos"].shape[0])
    contact_reference = reference["physical_state"][:length, 68:70] >= 0.5
    contact_value = value["physical_state"][:length, 68:70] >= 0.5
    return {
        "joint_position_rmse_rad": np.sqrt(
            np.mean(np.square(value["joint_pos"][:length] - reference["joint_pos"][:length]), axis=-1)
        ),
        "root_position_error_m": np.linalg.norm(
            value["root_pos"][:length] - reference["root_pos"][:length], axis=-1
        ),
        "root_orientation_error_deg": _quaternion_error_degrees(
            reference["root_quat"][:length], value["root_quat"][:length]
        ),
        "body_mpjpe_m": np.mean(
            np.linalg.norm(value["body_pos"][:length] - reference["body_pos"][:length], axis=-1),
            axis=-1,
        ),
        "contact_agreement": np.mean(contact_reference == contact_value, axis=-1),
    }


def _first_threshold_crossings(errors: dict[str, np.ndarray]) -> dict[str, int | None]:
    thresholds = {
        "joint_position_rmse_rad": (0.02, "above"),
        "root_position_error_m": (0.05, "above"),
        "root_orientation_error_deg": (5.0, "above"),
        "body_mpjpe_m": (0.05, "above"),
        "contact_agreement": (0.95, "below"),
    }
    result: dict[str, int | None] = {}
    for name, (threshold, direction) in thresholds.items():
        values = errors[name]
        indices = np.flatnonzero(values > threshold if direction == "above" else values < threshold)
        result[name] = int(indices[0]) if indices.size else None
    return result


def _write_error_svg(path: Path, errors: dict[str, np.ndarray], fps: float) -> None:
    width, height = 1100, 650
    margin_left, margin_right, margin_top, margin_bottom = 100, 30, 45, 65
    plot_width = width - margin_left - margin_right
    panel_height = (height - margin_top - margin_bottom) / len(errors)
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c"]
    thresholds = [0.02, 0.05, 5.0, 0.05, 0.95]
    labels = [
        "Joint RMSE (rad)",
        "Root position error (m)",
        "Root orientation error (deg)",
        "Body MPJPE (m)",
        "Contact agreement",
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="550" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">Recorded HDF vs exact-init original Action replay</text>',
    ]
    frame_count = max(len(value) for value in errors.values())
    for index, ((_, values), color, threshold, label) in enumerate(
        zip(errors.items(), colors, thresholds, labels, strict=True)
    ):
        y0 = margin_top + index * panel_height
        y1 = y0 + panel_height - 18
        maximum = max(float(np.max(values)), threshold, 1.0e-12) * 1.1
        points = []
        for frame, value in enumerate(values):
            x = margin_left + plot_width * frame / max(frame_count - 1, 1)
            y = y1 - (y1 - y0) * float(value) / maximum
            points.append(f"{x:.2f},{y:.2f}")
        threshold_y = y1 - (y1 - y0) * threshold / maximum
        parts.extend(
            [
                f'<line x1="{margin_left}" y1="{y1:.2f}" x2="{width-margin_right}" y2="{y1:.2f}" stroke="#999"/>',
                f'<line x1="{margin_left}" y1="{threshold_y:.2f}" x2="{width-margin_right}" y2="{threshold_y:.2f}" stroke="#555" stroke-dasharray="5 4"/>',
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>',
                f'<text x="{margin_left-8}" y="{(y0+y1)/2:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{label}</text>',
                f'<text x="{width-margin_right-4}" y="{threshold_y-3:.2f}" text-anchor="end" font-family="sans-serif" font-size="10">gate {threshold:g}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="{margin_left+plot_width/2:.2f}" y="{height-22}" text-anchor="middle" font-family="sans-serif" font-size="13">Frame at {fps:g} Hz</text>',
            f'<text x="{margin_left}" y="{height-42}" font-family="sans-serif" font-size="10">0</text>',
            f'<text x="{width-margin_right}" y="{height-42}" text-anchor="end" font-family="sans-serif" font-size="10">{frame_count-1}</text>',
            "</svg>",
        ]
    )
    atomic_write_text(path, "\n".join(parts) + "\n")


def finalize(output_run: Path) -> dict[str, Any]:
    output_run = output_run.expanduser().resolve()
    replay_request = load_json(output_run / "manifests/action_replay_request.json")
    initialization = load_json(output_run / "manifests/exact_initialization.json")
    required = [
        output_run / "markers/action_mask_replay.ok",
        output_run / "markers/h50a_exact_action_replay_render.ok",
        output_run / SOURCE_RELATIVE_PATH,
        *(output_run / path for path in READBACK_RELATIVE_PATHS),
        *(output_run / path for path in VIDEO_RELATIVE_PATHS),
        output_run / "data/replay/000000.replay.npz",
        output_run / "data/replay/000001.replay.npz",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"exact-init H50-A Action replay is incomplete: {missing}")
    if file_sha256(output_run / INIT_RELATIVE_PATH) != initialization["file_sha256"]:
        raise ValueError("exact initialization file changed after preparation")

    reports = [load_json(output_run / path) for path in READBACK_RELATIVE_PATHS]
    report_identity = {
        "same_payload_sha256": len({report["payload_sha256"] for report in reports}) == 1,
        "same_initialization_file_sha256": (
            len({report["initialization_file_sha256"] for report in reports}) == 1
        ),
        "matches_prepared_payload": all(
            report["payload_sha256"] == initialization["payload_sha256"] for report in reports
        ),
        "matches_prepared_file": all(
            report["initialization_file_sha256"] == initialization["file_sha256"]
            for report in reports
        ),
        "application_complete": all(bool(report.get("application_complete")) for report in reports),
    }
    readback_identity = _readback_identity_metrics(reports)
    per_run_init_checks = []
    for report in reports:
        errors = report["errors"]
        per_run_init_checks.append(
            {
                "physics_state": float(errors["physics_state_v3_max_abs"]) <= 1.0e-5,
                "joint_state": max(
                    float(errors["joint_pos_max_abs_rad"]),
                    float(errors["joint_vel_max_abs_rad_s"]),
                )
                <= 1.0e-5,
                "root_state": max(
                    float(errors["root_pos_max_abs_m"]),
                    float(errors["root_lin_vel_max_abs_m_s"]),
                    float(errors["root_ang_vel_max_abs_rad_s"]),
                )
                <= 1.0e-5,
                "root_orientation": float(errors["root_orientation_error_deg"]) <= 1.0e-4,
                "body_pose": float(errors["body_pos_max_abs_m"]) <= 1.0e-5,
                "runtime_context": float(errors["runtime_context_max_abs"]) <= 1.0e-5,
                "action_history_and_target": max(
                    float(errors["previous_raw_action_max_abs"]),
                    float(errors["current_raw_action_max_abs"]),
                    float(errors["previous_processed_action_max_abs"]),
                    float(errors["initial_joint_target_max_abs"]),
                )
                <= 1.0e-6,
            }
        )
    identity_checks = {
        **report_identity,
        "two_readbacks_max_abs": readback_identity["max_abs"] <= 1.0e-6,
        "two_readbacks_orientation": readback_identity["root_orientation_error_deg"] <= 1.0e-4,
        "each_readback_matches_hdf": all(all(checks.values()) for checks in per_run_init_checks),
    }
    initialization_identity_pass = all(identity_checks.values())

    source = _load_replay(output_run / SOURCE_RELATIVE_PATH)
    replay = [
        _load_replay(output_run / f"data/replay/{index:06d}.replay.npz")
        for index in range(2)
    ]
    with np.load(output_run / REPLAY_ACTIONS_RELATIVE_PATH, allow_pickle=False) as values:
        planned_raw = np.asarray(values["raw_actions"], dtype=np.float32)
    executed_raw = np.stack((replay[0]["raw_action"], replay[1]["raw_action"]), axis=1)
    if planned_raw.shape != (WINDOW_TRANSITIONS, 2, 29) or executed_raw.shape != planned_raw.shape:
        raise ValueError(
            f"exact replay Action shapes are invalid: planned={planned_raw.shape}, executed={executed_raw.shape}"
        )
    action_execution_max_abs = float(np.max(np.abs(planned_raw - executed_raw)))
    source_to_original = _trajectory_metrics(source, replay[0])
    source_to_h50a = _trajectory_metrics(source, replay[1])
    original_to_h50a = _trajectory_metrics(replay[0], replay[1])
    baseline_thresholds = {
        "joint_position_rmse_rad": 0.02,
        "root_position_rmse_m": 0.05,
        "root_orientation_max_deg": 5.0,
        "body_mpjpe_m": 0.05,
        "foot_contact_accuracy": 0.95,
    }
    baseline_metric_checks = {
        name: (
            float(source_to_original[name]) >= threshold
            if name == "foot_contact_accuracy"
            else float(source_to_original[name]) <= threshold
        )
        for name, threshold in baseline_thresholds.items()
    }
    recorded_baseline_pass = initialization_identity_pass and all(
        baseline_metric_checks.values()
    )
    h50a_metric_checks = {
        name: (
            float(source_to_h50a[name]) >= threshold
            if name == "foot_contact_accuracy"
            else float(source_to_h50a[name]) <= threshold
        )
        for name, threshold in baseline_thresholds.items()
    }
    h50a_replay_pass = recorded_baseline_pass and all(h50a_metric_checks.values())

    errors = _per_frame_errors(source, replay[0])
    crossings = _first_threshold_crossings(errors)
    error_npz = output_run / "data/recorded_to_original_per_frame_errors.npz"
    _write_npz(error_npz, **{name: value.astype(np.float32) for name, value in errors.items()})
    error_svg = output_run / "plots/recorded_to_original_error_curves.svg"
    _write_error_svg(error_svg, errors, fps=50.0)

    video_checks = {
        str(path): (output_run / path).is_file() and (output_run / path).stat().st_size > 0
        for path in VIDEO_RELATIVE_PATHS
    }
    execution_checks = {
        "two_serial_replay_artifacts": all(
            replay[index]["joint_pos"].shape[0] == WINDOW_TRANSITIONS + 1
            for index in range(2)
        ),
        "planned_raw_actions_executed": action_execution_max_abs <= 1.0e-6,
        "two_initialization_reports": len(reports) == 2,
        "three_independent_videos": all(video_checks.values()),
        "per_frame_error_artifacts": error_npz.is_file() and error_svg.is_file(),
    }
    if not all(execution_checks.values()):
        raise RuntimeError(f"exact-init Action replay execution failed: {execution_checks}")

    if not initialization_identity_pass:
        conclusion = "INITIALIZATION_GATE_FAILED_CANNOT_EVALUATE_REPLAY"
        next_step = "FIX_EXACT_INITIALIZATION_OR_CONFIRM_MISSING_RECORDED_STATE"
    elif not recorded_baseline_pass:
        conclusion = "RECORDED_HDF_NOT_REPRODUCIBLE_WITH_RECORDED_INITIALIZATION"
        next_step = "STOP_MODEL_ATTRIBUTION_AND_AUDIT_UNRECORDED_PHYSX_STATE"
    else:
        conclusion = (
            "ORIGINAL_AND_H50A_ACTION_REPLAY_PASS_ON_ONE_SEEN_WINDOW"
            if h50a_replay_pass
            else "ORIGINAL_REPLAY_PASSES_H50A_ACTION_REPLAY_FAILS"
        )
        next_step = (
            "REPORT_ONE_WINDOW_POSTERIOR_ACTION_REPLAY_SUPPORT_ONLY"
            if h50a_replay_pass
            else "ATTRIBUTE_REPLAY_GAP_TO_H50A_ACTION_ERROR_ON_THIS_WINDOW"
        )

    summary = {
        "format_version": "sonic_h50a_exact_init_action_replay_summary_v1",
        "execution_pass": True,
        "initialization_identity_pass": initialization_identity_pass,
        "recorded_action_baseline_pass": recorded_baseline_pass,
        "request": load_json(output_run / "manifests/action_mask_request.json"),
        "offline": load_json(output_run / "manifests/h50a_exact_init_offline_metrics.json"),
        "initialization": {
            "manifest": initialization,
            "identity_checks": identity_checks,
            "per_run_checks": per_run_init_checks,
            "cross_run_readback": readback_identity,
            "reports": [str(output_run / path) for path in READBACK_RELATIVE_PATHS],
        },
        "physics": {
            "recorded_hdf_to_original_action_exact_init": source_to_original,
            "recorded_hdf_to_h50a_action_exact_init": source_to_h50a,
            "original_action_to_h50a_action_exact_init": original_to_h50a,
            "baseline_thresholds": baseline_thresholds,
            "baseline_metric_checks": baseline_metric_checks,
            "h50a_metric_checks": h50a_metric_checks,
            "h50a_replay_pass_after_valid_baseline": h50a_replay_pass,
            "planned_executed_raw_action_max_abs": action_execution_max_abs,
            "first_threshold_crossing_frame": crossings,
            "per_frame_errors": str(error_npz),
            "error_curves": str(error_svg),
        },
        "videos": {
            path.stem: {
                "path": str(output_run / path),
                "sha256": file_sha256(output_run / path),
            }
            for path in VIDEO_RELATIVE_PATHS
        },
        "optional_triptych": str(output_run / "videos/exact_init_comparison.mp4"),
        "execution_checks": execution_checks,
        "conclusion": conclusion,
        "unique_next_step": next_step,
        "scope": (
            "one seen H50-A T64 posterior/full-both window; no conditional prior, random Mask, "
            "sampling, KL, or unseen-motion claim"
        ),
    }
    atomic_write_json(
        output_run / "manifests/h50a_exact_init_action_replay_summary.json", summary
    )
    atomic_write_text(
        output_run / "markers/cvae_posterior_h50_action_replay_exact_execution.ok", "PASS\n"
    )
    if initialization_identity_pass:
        atomic_write_text(
            output_run
            / "markers/cvae_posterior_h50_action_replay_initialization_identity.ok",
            "PASS\n",
        )
    if recorded_baseline_pass:
        atomic_write_text(
            output_run / "markers/cvae_posterior_h50_action_replay_recorded_baseline.ok",
            "PASS\n",
        )
    return {
        "output_run": str(output_run),
        "execution_pass": True,
        "initialization_identity_pass": initialization_identity_pass,
        "recorded_action_baseline_pass": recorded_baseline_pass,
        "conclusion": conclusion,
        "unique_next_step": next_step,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one seen H50-A window from its recorded exact initialization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--dataset-run", type=Path, required=True)
    prepare_parser.add_argument("--source-run", type=Path, required=True)
    prepare_parser.add_argument("--checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--output-run", type=Path, required=True)
    prepare_parser.add_argument("--motion-key", default="auto")
    prepare_parser.add_argument("--variant-id", type=int, default=0)
    prepare_parser.add_argument("--window-start", type=int, default=0)
    prepare_parser.add_argument("--seed", type=int, default=20260834)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output-run", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        result = prepare(
            args.dataset_run,
            args.source_run,
            args.checkpoint,
            args.output_run,
            motion_key=args.motion_key,
            variant_id=args.variant_id,
            window_start=args.window_start,
            seed=args.seed,
        )
    else:
        result = finalize(args.output_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
