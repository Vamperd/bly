from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch

from .action_masks import relative_to_raw_action
from .models import HierarchicalPosteriorTransformer, build_model, parameter_count
from .physics_schema import (
    dynamics_context_vector,
    read_physics_states,
    resolve_parameter,
    structured_robot_information,
)
from .posterior_capacity import validate_motion_prefix
from .posterior_t64_protocol import make_autoencode_masks, rows_sha256, window_identity_rows
from .util import atomic_write_json, atomic_write_text, file_sha256, load_json


WINDOW_TRANSITIONS = 64
SCENARIO_NAME = "h50a_posterior_full_both"
SOURCE_RELATIVE_PATH = Path("data/source/000000.replay.npz")
SOURCE_TRAJECTORY_RELATIVE_PATH = Path("data/source/000000.trajectory.pkl")
REPLAY_ACTIONS_RELATIVE_PATH = Path("data/replay_actions.npz")
COMPLETIONS_RELATIVE_PATH = Path("data/completed_actions.npz")
PRIMARY_VIDEO_RELATIVE_PATH = Path("videos/h50a_seen_window_action_replay.mp4")


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _rmse(difference: np.ndarray) -> float:
    values = np.asarray(difference, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values))))


def _error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError(
            f"metric arrays must be non-empty and shape-identical; found "
            f"{prediction.shape} and {target.shape}"
        )
    absolute = np.abs(prediction.astype(np.float64) - target.astype(np.float64))
    return {
        "rmse": _rmse(prediction - target),
        "mean_abs": float(np.mean(absolute)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "max_abs": float(np.max(absolute)),
    }


def _select_seen_window(
    windows: list[dict[str, Any]],
    motion_key: str,
    variant_id: int,
    window_start: int,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Select a pre-registered training window without performance cherry-picking."""

    candidates = [
        (index, row)
        for index, row in enumerate(windows)
        if int(row["variant_id"]) == int(variant_id)
        and int(row["window_start"]) == int(window_start)
        and (motion_key == "auto" or str(row["motion_key"]) == motion_key)
    ]
    if not candidates:
        raise ValueError(
            f"no seen H50-A window matches motion={motion_key!r}, "
            f"variant={variant_id}, start={window_start}"
        )
    selected_index, selected = min(
        candidates,
        key=lambda pair: (str(pair[1]["motion_key"]), int(pair[0])),
    )
    rule = {
        "motion_selection": (
            "lexicographically first matching motion_key"
            if motion_key == "auto"
            else "explicit motion_key"
        ),
        "performance_metrics_used_for_selection": False,
        "variant_id": int(variant_id),
        "window_start": int(window_start),
    }
    return selected_index, selected, rule


def _source_contract(
    dataset_run: Path,
    source_run: Path,
    checkpoint_path: Path,
    *,
    dataset_hash: str,
    windows_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = source_run / "manifests/posterior_hierarchical_t64_summary.json"
    execution_marker = source_run / "markers/cvae_posterior_hierarchical_t64_execution.ok"
    continuation_marker = (
        source_run / "markers/cvae_posterior_hierarchical_t64_continuation_execution.ok"
    )
    fit_marker = source_run / "markers/cvae_posterior_hierarchical_t64_autoencode_fit.ok"
    required = (summary_path, execution_marker, continuation_marker, fit_marker, checkpoint_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"H50-A replay source is incomplete: {missing}")
    summary = load_json(summary_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("config", {}).get("model", {})
    checkpoint_step = int(checkpoint.get("optimizer_step", -1))
    completed_step = int(summary.get("completed_optimizer_steps", -2))
    best_step = int(summary.get("best_optimizer_step", -3))
    checks = {
        "formal_continuation_run": (
            not bool(summary.get("smoke", True))
            and bool(summary.get("continuation", {}).get("enabled"))
        ),
        "profile_h50": summary.get("profile") == "H50",
        "stage_autoencode": summary.get("stage") == "autoencode",
        "execution_pass": bool(summary.get("execution_pass")),
        "quality_pass": bool(summary.get("quality_pass")),
        "dataset_path": Path(str(summary.get("dataset_run", ""))).resolve()
        == dataset_run.resolve(),
        "dataset_hash": summary.get("dataset_manifest_sha256") == dataset_hash,
        "window_hash": summary.get("selected_windows_sha256") == windows_hash,
        "window_count": int(summary.get("window_count", -1)) == 1504,
        "checkpoint_format": checkpoint.get("format_version")
        == "sonic_posterior_hierarchical_t64_checkpoint_v1",
        "checkpoint_profile": model_config.get("profile") == "H50",
        "checkpoint_stage": checkpoint.get("stage") == "autoencode",
        "checkpoint_step": checkpoint_step in {completed_step, best_step},
        "checkpoint_dataset_hash": checkpoint.get("dataset_manifest_sha256") == dataset_hash,
        "checkpoint_window_hash": checkpoint.get("selected_windows_sha256") == windows_hash,
        "checkpoint_parameter_count": int(checkpoint.get("parameter_count", -1))
        == 51_005_283,
        "model_state": isinstance(checkpoint.get("model"), dict),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"H50-A replay source contract failed: {failed}")
    return checkpoint, {
        "run": str(source_run),
        "summary": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "optimizer_step": checkpoint_step,
        "checkpoint_choice": (
            "last.pt at step 34000; raw reconstruction metrics improved beyond the "
            "score-floor-selected step-32000 best_fit.pt"
            if checkpoint_step == completed_step
            else "best_fit.pt selected by the source run's fit-score checkpoint policy"
        ),
        "checks": checks,
    }


def _actuator_type_names(schema: dict[str, Any]) -> np.ndarray:
    joint_names = [str(value) for value in schema["joint_names"]]
    result = np.full(len(joint_names), "unknown", dtype="U64")
    index = {name: position for position, name in enumerate(joint_names)}
    for group in schema.get("actuator_groups", {}).values():
        actuator_type = str(group.get("type", "unknown"))
        for joint_name in group.get("joint_names", []):
            if joint_name in index:
                result[index[joint_name]] = actuator_type
    return result


def _write_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _write_trajectory(
    path: Path,
    *,
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    fps: float,
) -> None:
    value = {
        "dof_pos": np.asarray(joint_pos, dtype=np.float32),
        "root_pos_w": np.asarray(root_pos, dtype=np.float32),
        "root_quat_w": np.asarray(root_quat, dtype=np.float32),
        "quat_format": "wxyz",
        "fps": float(fps),
        "num_joints": 29,
        "total_frames": int(joint_pos.shape[0]),
        "object_pos_w": None,
        "object_quat_w": None,
        "table_pos_w": None,
        "table_quat_w": None,
    }
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.pkl")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def prepare(
    dataset_run: Path,
    source_run: Path,
    checkpoint_path: Path,
    output_run: Path,
    *,
    motion_key: str,
    variant_id: int,
    window_start: int,
    post_steps: int,
    seed: int,
) -> dict[str, Any]:
    # These imports remain lazy so documentation/static tests do not require h5py.
    import h5py
    from torch.utils.data import default_collate

    from .action_mask_eval import _resolve_motion_file
    from .dataset import StateActionWindowDataset
    from .physics_schema import load_physics_schema

    dataset_run = dataset_run.expanduser().resolve()
    source_run = source_run.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    if variant_id < 0 or post_steps < 0 or window_start < 0:
        raise ValueError("variant, window start, and post steps must be non-negative")
    for child in ("data/source", "data/replay", "manifests", "markers", "logs", "videos"):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    if not (dataset_run / "markers/cvae_overfit_subset.ok").is_file():
        raise FileNotFoundError("H50-A replay requires the 32-motion overfit subset")
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

    model_config = dict(checkpoint["config"]["model"])
    model = build_model(model_config)
    if not isinstance(model, HierarchicalPosteriorTransformer):
        raise TypeError("H50-A replay checkpoint built the wrong model type")
    if parameter_count(model) != 51_005_283:
        raise ValueError("H50-A replay model parameter count is not 51,005,283")
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    batch = _device_batch(batch_cpu, device)
    state_mask = state_mask_cpu.to(device)
    action_mask = action_mask_cpu.to(device)
    with torch.no_grad():
        output = model(batch, state_mask, action_mask)
        canonical_output = model.decode_from_canonical_latents(
            output.global_latent,
            output.local_latents,
            valid_state=batch["valid_state"],
            valid_action=batch["valid_action"],
        )
    canonical_checks = {
        "state": torch.allclose(
            output.physical_state,
            canonical_output.physical_state,
            rtol=1.0e-5,
            atol=1.0e-7,
        ),
        "action": torch.allclose(
            output.action, canonical_output.action, rtol=1.0e-5, atol=1.0e-7
        ),
        "contact_logits": torch.allclose(
            output.state_contact_logits,
            canonical_output.state_contact_logits,
            rtol=1.0e-5,
            atol=1.0e-7,
        ),
    }
    if not all(canonical_checks.values()):
        raise RuntimeError(f"H50-A trained/canonical decode paths disagree: {canonical_checks}")

    predicted_action_normalized = output.action[0].detach().cpu().numpy().astype(np.float32)
    target_action_normalized = batch_cpu["action"][0].numpy().astype(np.float32)
    predicted_action = base.denormalize_action(output.action[0]).detach().cpu().numpy().astype(np.float32)
    target_action = base.denormalize_action(batch["action"][0]).detach().cpu().numpy().astype(np.float32)
    predicted_state_normalized = output.physical_state[0].detach().cpu().numpy().astype(np.float32)
    target_state_normalized = batch_cpu["physical_state"][0].numpy().astype(np.float32)
    predicted_state = base.denormalize_state(output.physical_state[0]).detach().cpu().numpy().astype(np.float32)
    target_state = base.denormalize_state(batch["physical_state"][0]).detach().cpu().numpy().astype(np.float32)
    contact_accuracy = float(
        np.mean((predicted_state_normalized[:, 68:70] >= 0.5) == (target_state_normalized[:, 68:70] >= 0.5))
    )

    ref = base.refs[source_index]
    record = base.episodes[ref.episode_index]
    if int(ref.fixed_start if ref.fixed_start is not None else -1) != int(window_start):
        raise RuntimeError("selected dataset ref does not preserve the requested fixed start")
    schema = load_physics_schema(Path(record["schema_path"]))
    env_id = int(record["env_id"])
    with h5py.File(record["hdf5_path"], "r") as stream:
        episode = stream[f"data/{record['episode']}"]
        steps = int(record["steps"])
        window_stop = window_start + WINDOW_TRANSITIONS
        if window_stop > steps:
            raise ValueError("selected H50-A window is shorter than T64")
        replay_steps = min(steps, window_stop + post_steps)
        states = read_physics_states(episode["states"], 0, replay_steps + 1)
        original_canonical = np.asarray(
            episode["actions/action_target_canonical"][:replay_steps], dtype=np.float32
        )
        original_raw = np.asarray(
            episode["actions/raw_policy_action"][:replay_steps], dtype=np.float32
        )
        processed_abs = np.asarray(
            episode["actions/processed_joint_target_abs"][:replay_steps], dtype=np.float32
        )
        initial_target = np.asarray(
            episode["actions/initial_processed_target_canonical"], dtype=np.float32
        )
        root_pos_world = np.asarray(episode["replay/root_pos_w"][: replay_steps + 1], dtype=np.float32)
        root_quat = np.asarray(episode["replay/root_quat_w"][: replay_steps + 1], dtype=np.float32)
        body_pos_world = np.asarray(episode["replay/body_pos_w"][: replay_steps + 1], dtype=np.float32)
        context = stream[f"contexts/{record['context_id']}"]
        runtime_default = np.asarray(context["runtime_default_joint_pos"], dtype=np.float32)
        action_offset = np.asarray(context["action_offset"], dtype=np.float32)
        joint_robot, _, global_robot = structured_robot_information(
            schema, context, env_id, base.actuator_type_to_id
        )
        dynamics_context = dynamics_context_vector(context)

    nominal = resolve_parameter(schema["nominal_default_joint_pos"], env_id).reshape(29)
    action_scale = resolve_parameter(schema["action_scale"], env_id).reshape(29)
    clip_entry = schema.get("action_clip")
    action_clip = (
        None if clip_entry is None else resolve_parameter(clip_entry, env_id).reshape(29, 2)
    )
    wrapper_value = schema.get("wrapper_action_clip")
    wrapper_clip = None if wrapper_value is None else float(wrapper_value)
    predicted_raw_window, achieved_window, saturated_window = relative_to_raw_action(
        predicted_action,
        nominal,
        action_scale,
        action_offset,
        action_clip,
        wrapper_clip,
    )
    completed_canonical = original_canonical.copy()
    completed_raw = original_raw.copy()
    completed_canonical[window_start:window_stop] = achieved_window
    completed_raw[window_start:window_stop] = predicted_raw_window
    mask = np.zeros_like(original_canonical, dtype=bool)
    mask[window_start:window_stop] = True
    saturation = np.zeros_like(mask)
    saturation[window_start:window_stop] = saturated_window

    # Collection states contain the Isaac environment grid offset.  Recenter only
    # world x/y so recorded and one-environment replay share a common visual frame.
    translation = np.asarray(
        [root_pos_world[0, 0], root_pos_world[0, 1], 0.0], dtype=np.float32
    )
    root_pos = root_pos_world - translation
    body_pos = body_pos_world - translation[None, None]
    joint_pos = states[:, :29] + nominal[None]
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
        body_pos=body_pos,
        raw_action=original_raw,
        action_rel=original_canonical,
        action_target_canonical=original_canonical,
        processed_action=processed_abs,
        processed_joint_target_abs=processed_abs,
        joint_names=np.asarray(schema["joint_names"]),
        action_default=runtime_default,
        nominal_default_joint_pos=nominal,
        action_scale=action_scale,
        action_offset=action_offset,
        action_clip=(
            np.empty((0,), dtype=np.float32)
            if action_clip is None
            else action_clip.astype(np.float32)
        ),
        wrapper_action_clip=np.float32(np.nan if wrapper_clip is None else wrapper_clip),
        control_dt=np.float32(schema["simulation"]["control_dt"]),
        sim_dt=np.float32(schema["simulation"]["sim_dt"]),
        initial_processed_target_canonical=initial_target,
        joint_robot_information=joint_robot,
        joint_actuator_type_names=_actuator_type_names(schema),
        global_robot_information=global_robot,
        dynamics_context=dynamics_context,
        replay_schema_version=np.asarray("sonic_h50a_seen_window_source_v1"),
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
        output_run / COMPLETIONS_RELATIVE_PATH,
        scenario_names=np.asarray([SCENARIO_NAME]),
        original_action_rel=original_canonical,
        original_raw_action=original_raw,
        completed_action_rel=completed_canonical[None],
        completed_raw_action=completed_raw[None],
        mask_action=mask[None],
        action_saturated=saturation[None],
        window_start=np.int64(window_start),
        window_length=np.int64(WINDOW_TRANSITIONS),
    )
    replay_raw = np.stack((original_raw, completed_raw), axis=1)
    _write_npz(
        output_run / REPLAY_ACTIONS_RELATIVE_PATH,
        raw_actions=replay_raw,
        scenario_names=np.asarray(["original", SCENARIO_NAME]),
    )

    motion_path, motion_provenance = _resolve_motion_file(record)
    request = {
        "schema_version": "sonic_h50a_seen_window_action_replay_request_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_run": str(dataset_run),
        "dataset_manifest_sha256": dataset_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": source["checkpoint_sha256"],
        "motion_key": selected["motion_key"],
        "variant_id": int(selected["variant_id"]),
        "window_start": int(selected["window_start"]),
        "window_transitions": WINDOW_TRANSITIONS,
        "mask_name": "full_both",
        "mask_semantics": (
            "posterior encoder reads the exact complete seen window; decoder condition "
            "contains all-masked State and Action tokens"
        ),
        "post_window_replay_steps": int(replay_steps - window_stop),
        "replay_steps": int(replay_steps),
        "seed": int(seed),
        "render_mode": "all",
        "motion_file": str(motion_path),
        "motion_file_sha256": motion_provenance["sha256"],
        "motion_manifest": motion_provenance["manifest"],
        "source_episode": str(record["episode"]),
        "source_collection_run": str(record["source_run"]),
        "selection_rule": selection_rule,
    }
    atomic_write_json(output_run / "manifests/action_mask_request.json", request)
    replay_request = {
        "schema_version": "sonic_action_replay_request_v2",
        "representation": "physics_v4",
        "motion_key": selected["motion_key"],
        "motion_file": str(motion_path),
        "motion_file_sha256": motion_provenance["sha256"],
        "raw_actions_file": str((output_run / REPLAY_ACTIONS_RELATIVE_PATH).resolve()),
        "raw_actions_sha256": file_sha256(output_run / REPLAY_ACTIONS_RELATIVE_PATH),
        "completed_actions_sha256": file_sha256(output_run / COMPLETIONS_RELATIVE_PATH),
        "steps": int(replay_steps),
        "num_envs": 2,
        "scenario_names": ["original", SCENARIO_NAME],
        "source_capture": str(source_npz.resolve()),
        "control_dt": float(schema["simulation"]["control_dt"]),
        "latent_mode": "posterior_mean",
    }
    atomic_write_json(output_run / "manifests/action_replay_request.json", replay_request)
    atomic_write_text(
        output_run / "manifests/action_mask_scenarios.jsonl",
        json.dumps(
            {
                "name": SCENARIO_NAME,
                "mask": "all 29 Action dimensions for the exact 64-transition seen window",
                "latent": "H50-A canonical posterior mean from the complete State-Action window",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
    )
    offline = {
        "format_version": "sonic_h50a_seen_window_offline_metrics_v1",
        "selection": {**selected, "dataset_index": int(source_index), "rule": selection_rule},
        "mask_name": "full_both",
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
        "normalized": {
            "state_continuous": _error_metrics(
                predicted_state_normalized[:, :68], target_state_normalized[:, :68]
            ),
            "action": _error_metrics(predicted_action_normalized, target_action_normalized),
            "contact_accuracy": contact_accuracy,
        },
        "physical": {
            "state_continuous": _error_metrics(predicted_state[:, :68], target_state[:, :68]),
            "action_rad": _error_metrics(achieved_window, target_action),
            "requested_action_rad": _error_metrics(predicted_action, target_action),
            "saturated_element_count": int(saturated_window.sum()),
            "saturated_element_fraction": float(saturated_window.mean()),
        },
        "replay": {
            "replaced_action_steps": WINDOW_TRANSITIONS,
            "replay_steps": int(replay_steps),
            "post_window_steps": int(replay_steps - window_stop),
            "source_world_xy_translation_removed": translation.tolist(),
            "outside_window_raw_action_bitwise_unchanged": bool(
                np.array_equal(completed_raw[~mask], original_raw[~mask])
            ),
        },
        "scope": (
            "single seen H50-A training window; posterior mean/full-both only; "
            "not conditional inference or unseen-Mask generalization"
        ),
    }
    atomic_write_json(output_run / "manifests/h50a_seen_window_offline_metrics.json", offline)
    atomic_write_text(output_run / "markers/action_mask_prepare.ok", "PASS\n")
    atomic_write_text(output_run / "markers/action_mask_source.ok", "PASS\n")
    atomic_write_text(output_run / "markers/action_mask_completion.ok", "PASS\n")
    return {
        "output_run": str(output_run),
        "selected_window": offline["selection"],
        "checkpoint_step": source["optimizer_step"],
        "offline_metrics": offline["normalized"],
        "replay_steps": int(replay_steps),
    }


def finalize(output_run: Path) -> dict[str, Any]:
    from .action_mask_eval import (
        _load_replay_trajectory,
        _load_runtime_mapping,
        _mapping_max_abs,
        _runtime_context_max_abs,
        _trajectory_metrics,
    )

    output_run = output_run.expanduser().resolve()
    required = (
        "markers/action_mask_prepare.ok",
        "markers/action_mask_source.ok",
        "markers/action_mask_completion.ok",
        "markers/action_mask_replay.ok",
        "markers/h50a_action_replay_render.ok",
    )
    missing = [name for name in required if not (output_run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"H50-A Action replay stages are incomplete: {missing}")
    request = load_json(output_run / "manifests/action_replay_request.json")
    source = _load_replay_trajectory(output_run / SOURCE_RELATIVE_PATH)
    replay_paths = [
        output_run / f"data/replay/{index:06d}.replay.npz" for index in range(2)
    ]
    replay = [_load_replay_trajectory(path) for path in replay_paths]
    mapping_raw = [_load_runtime_mapping(path) for path in replay_paths]
    mapping_difference = _mapping_max_abs(mapping_raw[0][0], mapping_raw[1][0])
    context_difference = _runtime_context_max_abs(mapping_raw[0][2], mapping_raw[1][2])
    with np.load(output_run / REPLAY_ACTIONS_RELATIVE_PATH, allow_pickle=False) as values:
        planned_raw = np.asarray(values["raw_actions"], dtype=np.float32)
    executed_raw = np.stack((mapping_raw[0][1], mapping_raw[1][1]), axis=1)
    if planned_raw.shape != executed_raw.shape:
        raise ValueError(
            f"planned/executed Action shapes differ: {planned_raw.shape} vs {executed_raw.shape}"
        )
    action_execution_max_abs = float(np.max(np.abs(planned_raw - executed_raw)))
    recorded_to_original = _trajectory_metrics(source, replay[0])
    original_to_h50a = _trajectory_metrics(replay[0], replay[1])
    recorded_to_h50a = _trajectory_metrics(source, replay[1])
    baseline_thresholds = {
        "joint_position_rmse_rad": 0.05,
        "root_position_rmse_m": 0.10,
        "root_orientation_max_deg": 10.0,
    }
    baseline_fidelity = all(
        float(recorded_to_original[name]) <= limit
        for name, limit in baseline_thresholds.items()
    )
    video_path = output_run / PRIMARY_VIDEO_RELATIVE_PATH
    execution_checks = {
        "two_replay_trajectories": all(path.is_file() and path.stat().st_size > 0 for path in replay_paths),
        "planned_action_executed": action_execution_max_abs <= 1.0e-6,
        "controlled_mapping_identity": mapping_difference <= 1.0e-6,
        "controlled_context_identity": context_difference <= 1.0e-6,
        "primary_video": video_path.is_file() and video_path.stat().st_size > 0,
    }
    if not all(execution_checks.values()):
        raise RuntimeError(f"H50-A Action replay execution failed: {execution_checks}")
    summary = {
        "format_version": "sonic_h50a_seen_window_action_replay_summary_v1",
        "execution_pass": True,
        "model_quality_gate": "not defined for this single qualitative diagnostic",
        "request": load_json(output_run / "manifests/action_mask_request.json"),
        "offline_metrics": load_json(
            output_run / "manifests/h50a_seen_window_offline_metrics.json"
        ),
        "physics": {
            "recorded_training_to_original_action_replay": recorded_to_original,
            "original_action_replay_to_h50a_action_replay": original_to_h50a,
            "recorded_training_to_h50a_action_replay": recorded_to_h50a,
            "recorded_baseline_fidelity_pass": baseline_fidelity,
            "recorded_baseline_fidelity_thresholds": baseline_thresholds,
            "planned_executed_raw_action_max_abs": action_execution_max_abs,
            "controlled_mapping_max_abs": mapping_difference,
            "controlled_context_max_abs": context_difference,
        },
        "execution_checks": execution_checks,
        "primary_video": str(video_path),
        "primary_video_sha256": file_sha256(video_path),
        "interpretation": (
            "recorded, original-replay, and H50-A replay can be compared directly"
            if baseline_fidelity
            else "original-vs-H50-A is a controlled Action comparison, but the recorded-vs-replay "
            "gap must be treated as environment/reset/context mismatch"
        ),
        "scope": (
            "H50-A posterior reconstruction on one seen full-both training window; this does not "
            "test conditional prior, random Mask completion, sampling, KL, or unseen motions"
        ),
    }
    atomic_write_json(output_run / "manifests/h50a_seen_window_action_replay_summary.json", summary)
    atomic_write_text(output_run / "markers/cvae_posterior_h50_action_replay.ok", "PASS\n")
    return {
        "output_run": str(output_run),
        "execution_pass": True,
        "recorded_baseline_fidelity_pass": baseline_fidelity,
        "primary_video": str(video_path),
        "action_rmse_rad": summary["offline_metrics"]["physical"]["action_rad"]["rmse"],
        "physics_joint_rmse_vs_original_rad": original_to_h50a["joint_position_rmse_rad"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one exact seen H50-A full-both posterior Action window"
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
    prepare_parser.add_argument("--post-steps", type=int, default=50)
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
            post_steps=args.post_steps,
            seed=args.seed,
        )
    else:
        result = finalize(args.output_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
