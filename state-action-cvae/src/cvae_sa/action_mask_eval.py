from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .action_masks import (
    PRESET_NAME,
    ActionMaskScenario,
    build_default_scenarios,
    load_scenarios,
    masked_baselines,
    relative_to_raw_action,
    scenario_mask,
    write_scenarios,
)
from .action_selection import LATENT_MODES, select_latent_action_window
from .constants import ACTION_DIM, DEFAULT_WINDOW_TRANSITIONS, PHYSICAL_STATE_DIM, TASK_NAMES
from .dataset import read_episode_index
from .masking import MaskBatch
from .models import build_model
from .util import atomic_write_json, atomic_write_text, file_sha256, load_json, seed_everything


MIN_EPISODE_STEPS = DEFAULT_WINDOW_TRANSITIONS + 64
SOURCE_RELATIVE_PATH = Path("data/source/000000.replay.npz")
REPLAY_ACTIONS_RELATIVE_PATH = Path("data/replay_actions.npz")
COMPLETIONS_RELATIVE_PATH = Path("data/completed_actions.npz")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stable_order(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _resolve_motion_file(record: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    source_run = Path(record["source_run"])
    manifest = source_run / "manifests/motion_manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"collection motion manifest is missing: {manifest}")
    matches = [row for row in _jsonl(manifest) if row.get("motion_key") == record["motion_key"]]
    if len(matches) != 1:
        raise ValueError(
            f"expected one manifest row for {record['motion_key']}, found {len(matches)} in {manifest}"
        )
    row = matches[0]
    candidates = [row.get("final_pkl"), row.get("converted_pkl")]
    path = next((Path(value) for value in candidates if value and Path(value).is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"motion PKL for {record['motion_key']} is unavailable; checked {candidates}"
        )
    expected_hash = row.get("final_sha256") or row.get("converted_sha256")
    actual_hash = file_sha256(path)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(f"motion PKL hash mismatch for {path}")
    return path.resolve(), {"manifest": str(manifest.resolve()), "row": row, "sha256": actual_hash}


def prepare(
    dataset_run: Path,
    checkpoint: Path,
    output_run: Path,
    split: str,
    package: str,
    motion_key: str,
    seed: int,
    preset: str,
    custom_scenarios: Path | None,
    latent_mode: str,
    latent_samples: int,
    render_mode: str,
) -> dict[str, Any]:
    dataset_run = dataset_run.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    if split != "validation":
        raise ValueError("Action-mask physics replay is restricted to the validation split")
    if latent_mode not in LATENT_MODES:
        raise ValueError(
            f"unsupported latent_mode {latent_mode!r}; expected one of {sorted(LATENT_MODES)}"
        )
    if latent_samples <= 0:
        raise ValueError("latent_samples must be positive")
    if render_mode not in {"representatives", "all", "none"}:
        raise ValueError(f"unsupported render mode {render_mode!r}")
    if preset != PRESET_NAME and custom_scenarios is None:
        raise ValueError(f"unsupported Action mask preset {preset!r}")
    marker = dataset_run / "markers/cvae_dataset.ok"
    if not marker.is_file():
        raise FileNotFoundError(f"dataset marker is missing: {marker}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CVAE checkpoint is missing: {checkpoint}")

    records = [
        row
        for row in read_episode_index(dataset_run)
        if row["split"] == split
        and row["package"] == package
        and row["status"] == "completed"
        and int(row["steps"]) >= MIN_EPISODE_STEPS
    ]
    if motion_key != "auto":
        records = [row for row in records if row["motion_key"] == motion_key]
    if not records:
        raise ValueError(
            f"no completed {split}/{package} motion has at least {MIN_EPISODE_STEPS} steps"
        )
    by_key: dict[str, dict[str, Any]] = {}
    for row in records:
        by_key.setdefault(row["motion_key"], row)
    selected = min(by_key.values(), key=lambda row: _stable_order(seed, row["motion_key"]))
    motion_path, motion_provenance = _resolve_motion_file(selected)

    custom_path = custom_scenarios.expanduser().resolve() if custom_scenarios else None
    if custom_path is not None:
        if not custom_path.is_file():
            raise FileNotFoundError(f"custom Action mask scenarios are missing: {custom_path}")
        # Parse now so an invalid user file fails before Isaac starts.
        load_scenarios(custom_path)

    request = {
        "schema_version": "sonic_action_mask_eval_request_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_run": str(dataset_run),
        "dataset_manifest_sha256": file_sha256(dataset_run / "manifests/dataset_manifest.json"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "split": split,
        "package": package,
        "motion_key": selected["motion_key"],
        "indexed_steps": int(selected["steps"]),
        "source_episode": selected["episode"],
        "source_collection_run": selected["source_run"],
        "motion_file": str(motion_path),
        "motion_file_sha256": motion_provenance["sha256"],
        "motion_manifest": motion_provenance["manifest"],
        "preset": preset,
        "custom_scenarios": str(custom_path) if custom_path else None,
        "latent_mode": latent_mode,
        "latent_samples": latent_samples,
        "render_mode": render_mode,
        "seed": seed,
        "minimum_episode_steps": MIN_EPISODE_STEPS,
        "test_split_consumed": False,
    }
    atomic_write_json(output_run / "manifests/action_mask_request.json", request)
    atomic_write_text(output_run / "markers/action_mask_prepare.ok", "PASS\n")
    return request


def _required_array(values: Any, name: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    if name not in values:
        raise ValueError(f"source replay artifact is missing {name}")
    result = np.asarray(values[name])
    if result.ndim != len(shape_tail) + 1 or tuple(result.shape[1:]) != shape_tail:
        raise ValueError(f"{name} must have shape [T,{','.join(map(str, shape_tail))}], found {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result.astype(np.float32)


def _load_source(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as values:
        source = {
            "physical_state": _required_array(values, "physical_state", (PHYSICAL_STATE_DIM,)),
            "previous_action_rel": _required_array(values, "previous_action_rel", (ACTION_DIM,)),
            "raw_action": _required_array(values, "raw_action", (ACTION_DIM,)),
            "action_rel": _required_array(values, "action_rel", (ACTION_DIM,)),
            "joint_names": tuple(str(value) for value in values["joint_names"].tolist()),
            "action_default": np.asarray(values["action_default"], dtype=np.float32),
            "action_scale": np.asarray(values["action_scale"], dtype=np.float32),
            "action_offset": np.asarray(values["action_offset"], dtype=np.float32),
            "action_clip": np.asarray(values["action_clip"], dtype=np.float32),
            "wrapper_action_clip": float(np.asarray(values["wrapper_action_clip"]).item()),
            "control_dt": float(np.asarray(values["control_dt"]).item()),
        }
    steps = source["raw_action"].shape[0]
    if source["action_rel"].shape[0] != steps:
        raise ValueError("raw and relative source Action lengths differ")
    if source["physical_state"].shape[0] != steps + 1:
        raise ValueError("source capture must contain T+1 physical states for T Actions")
    if source["previous_action_rel"].shape[0] != steps + 1:
        raise ValueError("source capture must contain T+1 previous Actions")
    if len(source["joint_names"]) != ACTION_DIM:
        raise ValueError("source capture does not contain 29 Action joint names")
    for name in ("action_default", "action_scale", "action_offset"):
        if source[name].shape != (ACTION_DIM,):
            raise ValueError(f"{name} must be a 29-vector")
    if source["action_clip"].size == 0:
        source["action_clip"] = None
    elif source["action_clip"].shape != (ACTION_DIM, 2):
        raise ValueError("action_clip must be empty or [29,2]")
    if not math.isfinite(source["wrapper_action_clip"]):
        source["wrapper_action_clip"] = None
    if not np.isclose(source["control_dt"], 0.02):
        raise ValueError(f"source control_dt must be 0.02, found {source['control_dt']}")
    return source


def _select_window(actions: np.ndarray) -> tuple[int, int]:
    steps = actions.shape[0]
    minimum_start = 32
    maximum_start = steps - DEFAULT_WINDOW_TRANSITIONS - 32
    if maximum_start < minimum_start:
        raise ValueError(f"source trajectory has {steps} steps; at least {MIN_EPISODE_STEPS} are required")
    derivative = np.square(np.diff(actions, axis=0)).sum(axis=1)
    scores = np.asarray(
        [derivative[start : start + DEFAULT_WINDOW_TRANSITIONS - 1].mean() for start in range(minimum_start, maximum_start + 1)]
    )
    window_start = minimum_start + int(np.argmax(scores))
    local = derivative[window_start : window_start + DEFAULT_WINDOW_TRANSITIONS - 1]
    block_scores = np.convolve(local, np.ones(8, dtype=np.float32), mode="valid")
    peak_block_start = int(np.argmax(block_scores))
    return window_start, peak_block_start


def _make_batch(source: dict[str, Any], dataset_run: Path, start: int, device: torch.device):
    with np.load(dataset_run / "data/normalization.npz") as values:
        state_mean = values["physical_state_mean"].astype(np.float32)
        state_std = values["physical_state_std"].astype(np.float32)
        previous_mean = values["previous_action_mean"].astype(np.float32)
        previous_std = values["previous_action_std"].astype(np.float32)
        action_mean = values["action_mean"].astype(np.float32)
        action_std = values["action_std"].astype(np.float32)
    stop = start + DEFAULT_WINDOW_TRANSITIONS
    physical = source["physical_state"][start : stop + 1]
    previous = source["previous_action_rel"][start : stop + 1]
    action = source["action_rel"][start:stop]
    progress = np.arange(start, stop + 1, dtype=np.float32) / max(source["raw_action"].shape[0], 1)
    batch = {
        "physical_state": torch.from_numpy((physical - state_mean) / state_std)[None].to(device),
        "previous_action": torch.from_numpy((previous - previous_mean) / previous_std)[None].to(device),
        "action": torch.from_numpy((action - action_mean) / action_std)[None].to(device),
        "action_scale": torch.from_numpy(source["action_scale"])[None].to(device),
        "valid_state": torch.ones((1, DEFAULT_WINDOW_TRANSITIONS + 1), dtype=torch.bool, device=device),
        "valid_action": torch.ones((1, DEFAULT_WINDOW_TRANSITIONS), dtype=torch.bool, device=device),
        "progress": torch.from_numpy(progress)[None].to(device),
    }
    normalization = {
        "action_mean": action_mean,
        "action_std": action_std,
    }
    return batch, normalization


def _mask_batch(mask: np.ndarray, task: str, device: torch.device) -> MaskBatch:
    action = torch.from_numpy(mask)[None].to(device)
    state = torch.zeros((1, DEFAULT_WINDOW_TRANSITIONS + 1, PHYSICAL_STATE_DIM), dtype=torch.bool, device=device)
    previous_input = torch.zeros(
        (1, DEFAULT_WINDOW_TRANSITIONS + 1, ACTION_DIM), dtype=torch.bool, device=device
    )
    previous_input[:, 1:] = action
    previous_loss = torch.zeros_like(previous_input)
    return MaskBatch(
        state_input=state,
        previous_input=previous_input,
        action_input=action,
        state_loss=state.clone(),
        previous_loss=previous_loss,
        action_loss=action.clone(),
        task_id=TASK_NAMES.index(task),
        task_name=task,
        completion_name="external",
        causal=False,
    )


def _masked_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    hold: np.ndarray,
    linear: np.ndarray,
    uncertainty: np.ndarray,
    joint_names: tuple[str, ...],
    saturated: np.ndarray,
) -> dict[str, Any]:
    if not mask.any():
        raise ValueError("cannot compute metrics for an empty Action mask")

    def errors(value: np.ndarray) -> dict[str, float]:
        difference = value - truth
        selected = difference[mask]
        return {
            "mae_rad": float(np.mean(np.abs(selected))),
            "rmse_rad": float(np.sqrt(np.mean(np.square(selected)))),
            "max_abs_rad": float(np.max(np.abs(selected))),
        }

    transition_mask = mask[1:] | mask[:-1]
    derivative_error = np.diff(prediction, axis=0) - np.diff(truth, axis=0)
    derivative_selected = derivative_error[transition_mask]
    per_joint = {}
    for index, name in enumerate(joint_names):
        selected = mask[:, index]
        if selected.any():
            difference = prediction[selected, index] - truth[selected, index]
            per_joint[name] = {
                "count": int(selected.sum()),
                "mae_rad": float(np.mean(np.abs(difference))),
                "rmse_rad": float(np.sqrt(np.mean(np.square(difference)))),
            }
    result = {
        "masked_elements": int(mask.sum()),
        "masked_fraction": float(mask.mean()),
        "cvae": errors(prediction),
        "hold_last": errors(hold),
        "linear_interpolation": errors(linear),
        "action_derivative_rmse_rad": (
            float(np.sqrt(np.mean(np.square(derivative_selected))))
            if derivative_selected.size
            else 0.0
        ),
        "latent_std_mean_rad": float(np.mean(uncertainty[mask])),
        "latent_std_max_rad": float(np.max(uncertainty[mask])),
        "saturation_fraction": float(saturated[mask].mean()),
        "per_joint": per_joint,
    }
    result["beats_hold_last"] = result["cvae"]["rmse_rad"] < result["hold_last"]["rmse_rad"]
    result["beats_linear_interpolation"] = (
        result["cvae"]["rmse_rad"] < result["linear_interpolation"]["rmse_rad"]
    )
    return result


def _scan_window_starts(action_steps: int, stride: int = 16) -> list[int]:
    first = 32
    last = action_steps - DEFAULT_WINDOW_TRANSITIONS - 32
    if last < first:
        raise ValueError("trajectory is too short for a 128-step window with 32-step margins")
    starts = list(range(first, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _aggregate_scan_values(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("multi-position Action-mask scan produced invalid values")
    return {
        "windows": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


@torch.no_grad()
def _scan_action_masks(
    model: torch.nn.Module,
    source: dict[str, Any],
    dataset_run: Path,
    device: torch.device,
    scenarios: list[ActionMaskScenario],
    custom_scenarios: bool,
    seed: int,
) -> dict[str, Any]:
    starts = _scan_window_starts(source["raw_action"].shape[0])
    values: dict[str, dict[str, list[float]]] = {
        scenario.name: {
            "cvae_rmse_rad": [],
            "hold_last_rmse_rad": [],
            "linear_interpolation_rmse_rad": [],
            "saturation_fraction": [],
        }
        for scenario in scenarios
    }
    for start in starts:
        stop = start + DEFAULT_WINDOW_TRANSITIONS
        source_window = source["action_rel"][start:stop]
        derivative = np.square(np.diff(source_window, axis=0)).sum(axis=1)
        block_scores = np.convolve(derivative, np.ones(8, dtype=np.float32), mode="valid")
        peak_block_start = int(np.argmax(block_scores))
        if custom_scenarios:
            window_scenarios = [replace(scenario, window_start=start) for scenario in scenarios]
        else:
            window_scenarios = build_default_scenarios(
                start, peak_block_start, source["joint_names"], seed
            )
        if [item.name for item in window_scenarios] != [item.name for item in scenarios]:
            raise RuntimeError("multi-position scan changed the Action-mask scenario identities")
        batch, normalization = _make_batch(source, dataset_run, start, device)
        for scenario in window_scenarios:
            mask = scenario_mask(scenario, source["joint_names"])
            output = model(
                batch,
                _mask_batch(mask, scenario.task, device),
                sample_from_prior=True,
                deterministic=True,
            )
            prediction = (
                output.action[0].float().cpu().numpy() * normalization["action_std"]
                + normalization["action_mean"]
            )
            completed = np.where(mask, prediction, source_window)
            _, achieved, saturated = relative_to_raw_action(
                completed,
                source["action_default"],
                source["action_scale"],
                source["action_offset"],
                source["action_clip"],
                source["wrapper_action_clip"],
            )
            hold, linear = masked_baselines(source_window, mask)

            def rmse(candidate: np.ndarray) -> float:
                return float(
                    np.sqrt(np.mean(np.square(candidate[mask] - source_window[mask])))
                )

            target = values[scenario.name]
            target["cvae_rmse_rad"].append(rmse(achieved))
            target["hold_last_rmse_rad"].append(rmse(hold))
            target["linear_interpolation_rmse_rad"].append(rmse(linear))
            target["saturation_fraction"].append(float(saturated[mask].mean()))
    return {
        "stride_steps": 16,
        "stride_seconds": 0.32,
        "window_starts": starts,
        "window_count": len(starts),
        "scenarios": {
            name: {metric: _aggregate_scan_values(samples) for metric, samples in metrics.items()}
            for name, metrics in values.items()
        },
    }


@torch.no_grad()
def complete(
    dataset_run: Path,
    checkpoint_path: Path,
    output_run: Path,
    custom_scenarios: Path | None,
    latent_samples: int,
    seed: int,
) -> dict[str, Any]:
    output_run = output_run.expanduser().resolve()
    dataset_run = dataset_run.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    request = load_json(output_run / "manifests/action_mask_request.json")
    latent_mode = request["latent_mode"]
    if latent_mode not in LATENT_MODES:
        raise ValueError(f"unsupported latent_mode {latent_mode!r} in request")
    if int(request["latent_samples"]) != latent_samples:
        raise ValueError("prepare and complete latent sample counts differ")
    source_path = output_run / SOURCE_RELATIVE_PATH
    if not source_path.is_file():
        raise FileNotFoundError(f"SONIC source capture is missing: {source_path}")
    source = _load_source(source_path)
    window_start, peak_block_start = _select_window(source["action_rel"])
    scenario_path = custom_scenarios or (
        Path(request["custom_scenarios"]) if request.get("custom_scenarios") else None
    )
    if scenario_path is None:
        scenarios = build_default_scenarios(
            window_start, peak_block_start, source["joint_names"], seed
        )
    else:
        scenarios = load_scenarios(scenario_path)
        starts = {scenario.window_start for scenario in scenarios}
        if len(starts) != 1:
            raise ValueError("all custom scenarios must use one shared window_start")
        window_start = starts.pop()
        if (
            window_start < 32
            or window_start + DEFAULT_WINDOW_TRANSITIONS + 32
            > source["raw_action"].shape[0]
        ):
            raise ValueError("custom scenario window must preserve 32-step prefix/suffix margins")
    write_scenarios(output_run / "manifests/action_mask_scenarios.jsonl", scenarios)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset_hash = file_sha256(dataset_run / "manifests/dataset_manifest.json")
    if checkpoint["dataset_manifest_sha256"] != dataset_hash:
        raise ValueError("checkpoint and CVAE dataset manifest hashes differ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    batch, normalization = _make_batch(source, dataset_run, window_start, device)
    window_stop = window_start + DEFAULT_WINDOW_TRANSITIONS
    source_window = source["action_rel"][window_start:window_stop]

    scenario_names: list[str] = []
    completed_relative = []
    completed_raw = []
    full_masks = []
    uncertainty_values = []
    saturation_values = []
    selected_candidate_indices = []
    candidate_error_values = []
    offline_metrics: dict[str, Any] = {}
    mapping_round_trip_max = 0.0

    for scenario in scenarios:
        mask = scenario_mask(scenario, source["joint_names"])
        masks = _mask_batch(mask, scenario.task, device)
        seed_everything(scenario.seed)
        deterministic_output = model(batch, masks, sample_from_prior=True, deterministic=True)
        predicted_window = (
            deterministic_output.action[0].float().cpu().numpy()
            * normalization["action_std"]
            + normalization["action_mean"]
        )
        prior_mean_window = np.where(mask, predicted_window, source_window)

        stochastic_windows = []
        for sample_index in range(latent_samples):
            seed_everything(scenario.seed + sample_index + 1)
            sampled = model(batch, masks, sample_from_prior=True, deterministic=False)
            sampled_action = (
                sampled.action[0].float().cpu().numpy()
                * normalization["action_std"]
                + normalization["action_mean"]
            )
            stochastic_windows.append(np.where(mask, sampled_action, source_window))
        candidate_windows = np.stack(stochastic_windows)
        uncertainty = candidate_windows.std(axis=0)
        completed_window, selected_candidate_index, candidate_errors = (
            select_latent_action_window(
                prior_mean_window,
                candidate_windows,
                source_window,
                mask,
                latent_mode,
            )
        )
        prior_mean_rmse = float(
            np.sqrt(np.mean(np.square(prior_mean_window[mask] - source_window[mask])))
        )

        full_relative = source["action_rel"].copy()
        absolute_mask = np.zeros_like(full_relative, dtype=bool)
        full_relative[window_start:window_stop] = completed_window
        absolute_mask[window_start:window_stop] = mask
        raw, achieved, saturated = relative_to_raw_action(
            full_relative,
            source["action_default"],
            source["action_scale"],
            source["action_offset"],
            source["action_clip"],
            source["wrapper_action_clip"],
        )
        raw[~absolute_mask] = source["raw_action"][~absolute_mask]
        achieved[~absolute_mask] = source["action_rel"][~absolute_mask]
        if not np.array_equal(raw[~absolute_mask], source["raw_action"][~absolute_mask]):
            raise RuntimeError(f"{scenario.name} changed an Action outside its declared mask")
        processed_round_trip = raw * source["action_scale"] + source["action_offset"]
        if source["action_clip"] is not None:
            processed_round_trip = np.clip(
                processed_round_trip,
                source["action_clip"][:, 0],
                source["action_clip"][:, 1],
            )
        round_trip_error = float(
            np.max(
                np.abs(
                    (processed_round_trip - source["action_default"])
                    - achieved
                )
            )
        )
        mapping_round_trip_max = max(mapping_round_trip_max, round_trip_error)
        hold, linear = masked_baselines(source_window, mask)
        offline_metrics[scenario.name] = {
            "scenario": scenario.to_dict(),
            "latent_selection_mode": latent_mode,
            "latent_candidate_seeds": [
                scenario.seed + sample_index + 1 for sample_index in range(latent_samples)
            ],
            "selected_candidate_index": selected_candidate_index,
            "candidate_masked_rmse_rad": candidate_errors.tolist(),
            "selected_candidate_pre_mapping_rmse_rad": (
                float(candidate_errors[selected_candidate_index])
                if selected_candidate_index >= 0
                else None
            ),
            "prior_mean_pre_mapping_rmse_rad": prior_mean_rmse,
            **_masked_metrics(
                source_window,
                achieved[window_start:window_stop],
                mask,
                hold,
                linear,
                uncertainty,
                source["joint_names"],
                saturated[window_start:window_stop],
            ),
        }
        scenario_names.append(scenario.name)
        completed_relative.append(achieved)
        completed_raw.append(raw)
        full_masks.append(absolute_mask)
        uncertainty_full = np.zeros_like(full_relative)
        uncertainty_full[window_start:window_stop] = uncertainty
        uncertainty_values.append(uncertainty_full)
        saturation_values.append(saturated)
        selected_candidate_indices.append(selected_candidate_index)
        candidate_error_values.append(candidate_errors)

    relative_array = np.stack(completed_relative)
    raw_array = np.stack(completed_raw)
    mask_array = np.stack(full_masks)
    uncertainty_array = np.stack(uncertainty_values)
    saturation_array = np.stack(saturation_values)
    candidate_error_array = np.stack(candidate_error_values)
    temporary = output_run / "data/.completed_actions.tmp.npz"
    np.savez_compressed(
        temporary,
        scenario_names=np.asarray(scenario_names),
        original_action_rel=source["action_rel"],
        original_raw_action=source["raw_action"],
        completed_action_rel=relative_array,
        completed_raw_action=raw_array,
        mask_action=mask_array,
        latent_action_std=uncertainty_array,
        latent_candidate_masked_rmse_rad=candidate_error_array,
        selected_latent_candidate_index=np.asarray(selected_candidate_indices, dtype=np.int64),
        action_saturated=saturation_array,
        window_start=np.int64(window_start),
        window_length=np.int64(DEFAULT_WINDOW_TRANSITIONS),
        peak_block_start=np.int64(peak_block_start),
    )
    os.replace(temporary, output_run / COMPLETIONS_RELATIVE_PATH)
    replay_raw = np.concatenate((source["raw_action"][None], raw_array), axis=0).transpose(1, 0, 2)
    temporary = output_run / "data/.replay_actions.tmp.npz"
    np.savez_compressed(
        temporary,
        raw_actions=replay_raw,
        scenario_names=np.asarray(["original", *scenario_names]),
    )
    os.replace(temporary, output_run / REPLAY_ACTIONS_RELATIVE_PATH)

    multi_position_scan = _scan_action_masks(
        model,
        source,
        dataset_run,
        device,
        scenarios,
        custom_scenarios=scenario_path is not None,
        seed=seed,
    )

    aggregate = {
        "schema_version": "sonic_action_completion_metrics_v1",
        "motion_key": request["motion_key"],
        "window_start": window_start,
        "window_length": DEFAULT_WINDOW_TRANSITIONS,
        "peak_block_start": peak_block_start,
        "latent_mode": latent_mode,
        "latent_samples_for_uncertainty": latent_samples,
        "latent_candidate_count": latent_samples,
        "oracle_uses_ground_truth_action": latent_mode == "oracle_best_of_n",
        "multi_position_scan_latent_mode": "prior_mean",
        "relative_raw_processed_round_trip_max_abs": mapping_round_trip_max,
        "relative_raw_processed_round_trip_threshold": 1.0e-6,
        "visible_action_max_abs_change": 0.0,
        "scenario_count": len(scenarios),
        "by_distribution_status": dict(Counter(scenario.distribution_status for scenario in scenarios)),
        "scenarios": offline_metrics,
        "multi_position_scan": multi_position_scan,
    }
    atomic_write_json(output_run / "manifests/action_completion_metrics.json", aggregate)
    replay_request = {
        "schema_version": "sonic_action_replay_request_v1",
        "motion_key": request["motion_key"],
        "motion_file": request["motion_file"],
        "motion_file_sha256": request["motion_file_sha256"],
        "raw_actions_file": str((output_run / REPLAY_ACTIONS_RELATIVE_PATH).resolve()),
        "raw_actions_sha256": file_sha256(output_run / REPLAY_ACTIONS_RELATIVE_PATH),
        "completed_actions_sha256": file_sha256(output_run / COMPLETIONS_RELATIVE_PATH),
        "scenario_manifest_sha256": file_sha256(
            output_run / "manifests/action_mask_scenarios.jsonl"
        ),
        "steps": int(replay_raw.shape[0]),
        "num_envs": int(replay_raw.shape[1]),
        "scenario_names": ["original", *scenario_names],
        "latent_mode": latent_mode,
        "latent_candidate_count": latent_samples,
        "oracle_uses_ground_truth_action": latent_mode == "oracle_best_of_n",
        "source_capture": str(source_path.resolve()),
        "control_dt": source["control_dt"],
    }
    atomic_write_json(output_run / "manifests/action_replay_request.json", replay_request)
    atomic_write_text(output_run / "markers/action_mask_completion.ok", "PASS\n")
    return replay_request


def _quaternion_error_degrees(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    reference = reference / np.linalg.norm(reference, axis=-1, keepdims=True).clip(1.0e-8)
    value = value / np.linalg.norm(value, axis=-1, keepdims=True).clip(1.0e-8)
    dot = np.abs(np.sum(reference * value, axis=-1)).clip(0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def _trajectory_metrics(reference: dict[str, np.ndarray], value: dict[str, np.ndarray]) -> dict[str, float]:
    length = min(reference["joint_pos"].shape[0], value["joint_pos"].shape[0])
    if length <= 1:
        raise ValueError("replay trajectory is too short")
    joint_difference = value["joint_pos"][:length] - reference["joint_pos"][:length]
    velocity_difference = value["joint_vel"][:length] - reference["joint_vel"][:length]
    root_difference = value["root_pos"][:length] - reference["root_pos"][:length]
    root_angle = _quaternion_error_degrees(reference["root_quat"][:length], value["root_quat"][:length])
    gravity_ref = reference["physical_state"][:length, 61:64]
    gravity_value = value["physical_state"][:length, 61:64]
    gravity_dot = np.sum(gravity_ref * gravity_value, axis=-1).clip(-1.0, 1.0)
    gravity_angle = np.degrees(np.arccos(gravity_dot))
    body_difference = value["body_pos"][:length] - reference["body_pos"][:length]
    return {
        "aligned_state_count": length,
        "joint_position_rmse_rad": float(np.sqrt(np.mean(np.square(joint_difference)))),
        "joint_velocity_rmse_rad_s": float(np.sqrt(np.mean(np.square(velocity_difference)))),
        "root_position_rmse_m": float(np.sqrt(np.mean(np.square(root_difference)))),
        "root_position_max_m": float(np.max(np.linalg.norm(root_difference, axis=-1))),
        "root_orientation_mean_deg": float(np.mean(root_angle)),
        "root_orientation_max_deg": float(np.max(root_angle)),
        "gravity_mean_deg": float(np.mean(gravity_angle)),
        "body_mpjpe_m": float(np.mean(np.linalg.norm(body_difference, axis=-1))),
        "minimum_root_height_m": float(np.min(value["root_pos"][:length, 2])),
    }


def _load_replay_trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        names = ("joint_pos", "joint_vel", "root_pos", "root_quat", "body_pos", "physical_state")
        result = {name: np.asarray(values[name], dtype=np.float32) for name in names}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"trajectory contains NaN or Inf: {path}")
    return result


def _load_runtime_mapping(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        mapping = {
            name: np.asarray(values[name], dtype=np.float32)
            for name in (
                "action_default",
                "action_scale",
                "action_offset",
                "action_clip",
                "wrapper_action_clip",
            )
        }
        raw_action = np.asarray(values["raw_action"], dtype=np.float32)
    return mapping, raw_action


def _mapping_max_abs(reference: dict[str, np.ndarray], value: dict[str, np.ndarray]) -> float:
    maximum = 0.0
    for name in reference:
        if reference[name].shape != value[name].shape:
            return float("inf")
        difference = np.abs(reference[name] - value[name])
        finite = difference[np.isfinite(difference)]
        if finite.size:
            maximum = max(maximum, float(finite.max()))
        if not np.array_equal(np.isnan(reference[name]), np.isnan(value[name])):
            return float("inf")
    return maximum


def _trajectory_slice(
    trajectory: dict[str, np.ndarray], start: int, stop: int | None = None
) -> dict[str, np.ndarray]:
    return {name: value[start:stop] for name, value in trajectory.items()}


def _post_mask_metrics(
    reference: dict[str, np.ndarray],
    value: dict[str, np.ndarray],
    first_mask_action: int,
    control_dt: float,
) -> dict[str, Any]:
    aligned = min(reference["joint_pos"].shape[0], value["joint_pos"].shape[0])
    first_affected_state = min(first_mask_action + 1, aligned - 1)
    half_second = max(1, round(0.5 / control_dt))
    one_second = max(1, round(1.0 / control_dt))

    def period(count: int | None) -> dict[str, float] | None:
        stop = aligned if count is None else min(aligned, first_affected_state + count)
        if stop - first_affected_state <= 1:
            return None
        return _trajectory_metrics(
            _trajectory_slice(reference, first_affected_state, stop),
            _trajectory_slice(value, first_affected_state, stop),
        )

    joint_error = value["joint_pos"][:aligned] - reference["joint_pos"][:aligned]
    per_state_rmse = np.sqrt(np.mean(np.square(joint_error), axis=-1))
    post_error = per_state_rmse[first_affected_state:]
    if post_error.size >= 2:
        times = np.arange(post_error.size, dtype=np.float64) * control_dt
        growth_rate = float(np.polyfit(times, post_error.astype(np.float64), 1)[0])
    else:
        growth_rate = 0.0
    root_height_difference = (
        value["root_pos"][:aligned, 2] - reference["root_pos"][:aligned, 2]
    )
    relative_fall = bool(
        np.any(
            (value["root_pos"][:aligned, 2] < 0.35)
            & (reference["root_pos"][:aligned, 2] > 0.55)
        )
    )
    return {
        "post_mask_0_5_seconds": period(half_second),
        "post_mask_1_0_seconds": period(one_second),
        "post_mask_remaining_episode": period(None),
        "joint_error_growth_rate_rad_s": growth_rate,
        "joint_error_peak_rmse_rad": float(np.max(post_error)) if post_error.size else 0.0,
        "max_root_height_deviation_m": float(np.max(np.abs(root_height_difference))),
        "fell_relative_to_original": relative_fall,
        "early_termination": value["joint_pos"].shape[0] < reference["joint_pos"].shape[0],
    }


def physics_metrics(output_run: Path) -> dict[str, Any]:
    output_run = output_run.expanduser().resolve()
    request = load_json(output_run / "manifests/action_replay_request.json")
    replay_execution_manifest = output_run / "manifests/action_replay_slices.json"
    replay_execution_mode = (
        load_json(replay_execution_manifest)["execution_mode"]
        if replay_execution_manifest.is_file()
        else "legacy_parallel_environment"
    )
    source = _load_replay_trajectory(output_run / SOURCE_RELATIVE_PATH)
    replay_dir = output_run / "data/replay"
    trajectories = [
        _load_replay_trajectory(replay_dir / f"{index:06d}.replay.npz")
        for index in range(request["num_envs"])
    ]
    source_mapping, source_raw = _load_runtime_mapping(output_run / SOURCE_RELATIVE_PATH)
    replay_mapping_raw = [
        _load_runtime_mapping(replay_dir / f"{index:06d}.replay.npz")
        for index in range(request["num_envs"])
    ]
    mapping_errors = [
        _mapping_max_abs(source_mapping, mapping) for mapping, _ in replay_mapping_raw
    ]
    mapping_pass = max(mapping_errors) <= 1.0e-6
    with np.load(output_run / REPLAY_ACTIONS_RELATIVE_PATH, allow_pickle=False) as expected_values:
        expected_raw = np.asarray(expected_values["raw_actions"], dtype=np.float32)
    executed_raw = np.stack([raw for _, raw in replay_mapping_raw], axis=1)
    if expected_raw.shape != executed_raw.shape:
        raise ValueError(
            f"planned/executed raw Action shapes differ: {expected_raw.shape} vs {executed_raw.shape}"
        )
    executed_action_error = float(np.max(np.abs(expected_raw - executed_raw)))
    action_execution_pass = executed_action_error <= 1.0e-6
    if source_raw.shape != expected_raw[:, 0].shape:
        raise ValueError("source raw Action length differs from original replay plan")
    baseline = trajectories[0]
    source_fidelity = _trajectory_metrics(source, baseline)
    fidelity_pass = (
        source_fidelity["joint_position_rmse_rad"] <= 1.0e-3
        and source_fidelity["root_position_rmse_m"] <= 1.0e-3
        and source_fidelity["root_orientation_max_deg"] <= 0.1
    )
    with np.load(output_run / COMPLETIONS_RELATIVE_PATH, allow_pickle=False) as completion_values:
        masks = np.asarray(completion_values["mask_action"], dtype=bool)
    by_scenario = {}
    pre_mask_pass = True
    for index, name in enumerate(request["scenario_names"][1:], start=1):
        metrics = _trajectory_metrics(baseline, trajectories[index])
        masked_steps = np.flatnonzero(masks[index - 1].any(axis=-1))
        first_mask = int(masked_steps[0]) if masked_steps.size else 0
        pre_length = min(
            first_mask + 1,
            baseline["joint_pos"].shape[0],
            trajectories[index]["joint_pos"].shape[0],
        )
        pre_error = trajectories[index]["joint_pos"][:pre_length] - baseline["joint_pos"][:pre_length]
        pre_velocity_error = (
            trajectories[index]["joint_vel"][:pre_length]
            - baseline["joint_vel"][:pre_length]
        )
        pre_root_error = (
            trajectories[index]["root_pos"][:pre_length]
            - baseline["root_pos"][:pre_length]
        )
        pre_body_error = (
            trajectories[index]["body_pos"][:pre_length]
            - baseline["body_pos"][:pre_length]
        )
        pre_root_angle = _quaternion_error_degrees(
            baseline["root_quat"][:pre_length],
            trajectories[index]["root_quat"][:pre_length],
        )
        metrics["first_mask_action_step"] = first_mask
        metrics["pre_mask_joint_max_abs_rad"] = (
            float(np.max(np.abs(pre_error))) if pre_error.size else 0.0
        )
        metrics["pre_mask_joint_velocity_max_abs_rad_s"] = (
            float(np.max(np.abs(pre_velocity_error))) if pre_velocity_error.size else 0.0
        )
        metrics["pre_mask_root_position_max_abs_m"] = (
            float(np.max(np.abs(pre_root_error))) if pre_root_error.size else 0.0
        )
        metrics["pre_mask_root_orientation_max_deg"] = (
            float(np.max(pre_root_angle)) if pre_root_angle.size else 0.0
        )
        metrics["pre_mask_body_position_max_abs_m"] = (
            float(np.max(np.abs(pre_body_error))) if pre_body_error.size else 0.0
        )
        metrics.update(
            _post_mask_metrics(
                baseline,
                trajectories[index],
                first_mask,
                float(request["control_dt"]),
            )
        )
        pre_mask_pass &= (
            metrics["pre_mask_joint_max_abs_rad"] <= 1.0e-6
            and metrics["pre_mask_joint_velocity_max_abs_rad_s"] <= 1.0e-6
            and metrics["pre_mask_root_position_max_abs_m"] <= 1.0e-6
            and metrics["pre_mask_root_orientation_max_deg"] <= 1.0e-4
            and metrics["pre_mask_body_position_max_abs_m"] <= 1.0e-6
        )
        by_scenario[name] = metrics
    offline = load_json(output_run / "manifests/action_completion_metrics.json")
    quality = {
        name: {
            "beats_hold_last": values["beats_hold_last"],
            "beats_linear_interpolation": values["beats_linear_interpolation"],
        }
        for name, values in offline["scenarios"].items()
    }
    report = {
        "schema_version": "sonic_action_replay_physics_metrics_v1",
        "replay_execution_mode": replay_execution_mode,
        "latent_mode": offline["latent_mode"],
        "latent_candidate_count": offline["latent_candidate_count"],
        "oracle_uses_ground_truth_action": offline["oracle_uses_ground_truth_action"],
        "passed": fidelity_pass and pre_mask_pass and mapping_pass and action_execution_pass,
        "source_replay_fidelity": source_fidelity,
        "source_replay_fidelity_thresholds": {
            "joint_position_rmse_rad": 1.0e-3,
            "root_position_rmse_m": 1.0e-3,
            "root_orientation_max_deg": 0.1,
        },
        "scenarios": by_scenario,
        "pre_mask_identity_pass": pre_mask_pass,
        "pre_mask_identity_thresholds": {
            "joint_position_rad": 1.0e-6,
            "joint_velocity_rad_s": 1.0e-6,
            "root_position_m": 1.0e-6,
            "root_orientation_deg": 1.0e-4,
            "body_position_m": 1.0e-6,
        },
        "runtime_mapping_max_abs_by_environment": mapping_errors,
        "runtime_mapping_identity_pass": mapping_pass,
        "runtime_mapping_max_abs_threshold": 1.0e-6,
        "planned_executed_raw_action_max_abs": executed_action_error,
        "planned_executed_raw_action_pass": action_execution_pass,
        "model_quality": quality,
        "model_quality_pass": all(
            value["beats_hold_last"] and value["beats_linear_interpolation"]
            for value in quality.values()
        ),
    }
    atomic_write_json(output_run / "manifests/replay_physics_metrics.json", report)
    atomic_write_text(output_run / "markers/action_mask_physics_metrics.ok", "PASS\n")
    return report


def finalize(output_run: Path, render_mode: str) -> dict[str, Any]:
    output_run = output_run.expanduser().resolve()
    required = (
        "markers/action_mask_prepare.ok",
        "markers/action_mask_source.ok",
        "markers/action_mask_completion.ok",
        "markers/action_mask_replay.ok",
        "markers/action_mask_physics_metrics.ok",
    )
    missing = [name for name in required if not (output_run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Action mask evaluation stages are incomplete: {missing}")
    physics = load_json(output_run / "manifests/replay_physics_metrics.json")
    videos = sorted(path.name for path in (output_run / "videos").glob("*.mp4") if path.stat().st_size)
    if render_mode != "none" and not (output_run / "markers/action_mask_render.ok").is_file():
        raise FileNotFoundError("rendering was requested but action_mask_render.ok is missing")
    if render_mode != "none" and not videos:
        raise RuntimeError("rendering completed without a non-empty MP4")
    summary = {
        "schema_version": "sonic_action_mask_eval_summary_v1",
        "passed": bool(physics["passed"]),
        "pipeline_completed": True,
        "latent_mode": physics["latent_mode"],
        "latent_candidate_count": physics["latent_candidate_count"],
        "oracle_uses_ground_truth_action": physics["oracle_uses_ground_truth_action"],
        "model_quality_pass": bool(physics["model_quality_pass"]),
        "render_mode": render_mode,
        "videos": videos,
        "source_replay_fidelity": physics["source_replay_fidelity"],
    }
    atomic_write_json(output_run / "manifests/action_mask_eval_summary.json", summary)
    if not summary["passed"]:
        raise RuntimeError("source Action replay fidelity gate failed; comparison is not valid")
    atomic_write_text(output_run / "markers/cvae_action_mask_replay.ok", "PASS\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and validate generic Action-only CVAE masks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--dataset-run", type=Path, required=True)
    prepare_parser.add_argument("--checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--output-run", type=Path, required=True)
    prepare_parser.add_argument("--split", default="validation")
    prepare_parser.add_argument("--package", default="Locomotion")
    prepare_parser.add_argument("--motion-key", default="auto")
    prepare_parser.add_argument("--seed", type=int, default=20260824)
    prepare_parser.add_argument("--preset", default=PRESET_NAME)
    prepare_parser.add_argument("--custom-scenarios", type=Path)
    prepare_parser.add_argument("--latent-mode", default="prior_mean")
    prepare_parser.add_argument("--latent-samples", type=int, default=8)
    prepare_parser.add_argument("--render-mode", default="representatives")

    complete_parser = subparsers.add_parser("complete")
    complete_parser.add_argument("--dataset-run", type=Path, required=True)
    complete_parser.add_argument("--checkpoint", type=Path, required=True)
    complete_parser.add_argument("--output-run", type=Path, required=True)
    complete_parser.add_argument("--custom-scenarios", type=Path)
    complete_parser.add_argument("--latent-samples", type=int, default=8)
    complete_parser.add_argument("--seed", type=int, default=20260824)

    metrics_parser = subparsers.add_parser("physics-metrics")
    metrics_parser.add_argument("--output-run", type=Path, required=True)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output-run", type=Path, required=True)
    finalize_parser.add_argument("--render-mode", default="representatives")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        result = prepare(
            args.dataset_run,
            args.checkpoint,
            args.output_run,
            args.split,
            args.package,
            args.motion_key,
            args.seed,
            args.preset,
            args.custom_scenarios,
            args.latent_mode,
            args.latent_samples,
            args.render_mode,
        )
    elif args.command == "complete":
        result = complete(
            args.dataset_run,
            args.checkpoint,
            args.output_run,
            args.custom_scenarios,
            args.latent_samples,
            args.seed,
        )
    elif args.command == "physics-metrics":
        result = physics_metrics(args.output_run)
    else:
        result = finalize(args.output_run, args.render_mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
