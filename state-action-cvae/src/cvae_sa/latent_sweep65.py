"""C-stage latent sensitivity and multi-seed full-action replay.

This module is deliberately separate from training and from ``replay65``.  It
uses the C deployment contract (condition encoder plus standard-normal
latents), while also recording a full-truth posterior-mean reference for
diagnostics.  Every command operates on a new run and never changes a
checkpoint or a dataset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import torch
from torch.utils.data import default_collate

from .cvae_diagnostics import epsilon_for, read_normalization
from .cvae_protocol import CHECKPOINT, Fixtures, digest, isolated_rng, run_lock
from .cvae_training import (
    check_source_identity,
    data_identity,
    make_dataset,
    provenance,
    source_info,
)
from .models import HierarchicalStandardCVAETransformer, build_model
from .posterior_direct_output import assert_output_isolated
from .posterior_h50_action_replay import _write_npz, _write_trajectory
from .posterior_hierarchical_standard_cvae import load_checkpoint
from .posterior_h50_action_replay_exact_init import (
    _first_threshold_crossings,
    _load_replay,
    _per_frame_errors,
    _trajectory_metrics,
)
from .posterior_t64_protocol import PHYSICAL_MASK_NAMES
from .replay65 import (
    KIT,
    ROOT,
    CONTEXT_FIELDS,
    execution_actions,
    finite,
    launch,
    load_source,
    checkpoint_config,
    raw_to_target,
    initialization_checks,
    verify_worker,
)
from .util import atomic_write_json, atomic_write_text, file_sha256, load_json


VERSION = "65-token-latent-sweep-v1"
MASK = "full_action"
MASK_SLOT = PHYSICAL_MASK_NAMES.index(MASK)
DEFAULT_SEEDS = (20260923, 20260924, 20260925, 20260926, 20260927)
HARD_WINDOW = 844
HARD_MOTION = "jump_right_004__A029"
SIMULATION_SEED = 20260930
LATENT_NAMES = (
    "global_mean",
    "local_mean",
    "global_logvar",
    "local_logvar",
    "global_std",
    "local_std",
)


def _device_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _finite_array(name: str, value: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    value = np.asarray(value)
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name}: expected {shape}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name}: non-finite values")
    return value


def _physical_state(normalized: np.ndarray, norm: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    result = normalized * norm["state"][1] + norm["state"][0]
    # Contacts are probabilities, not z-scored continuous values.
    result[..., 68:] = normalized[..., 68:]
    return result.astype(np.float32)


def _prediction_arrays(
    output,
    norm: dict[str, tuple[np.ndarray, np.ndarray]],
    truth_state: np.ndarray,
    truth_action: np.ndarray,
    source: dict,
    init: dict,
    state_mask: np.ndarray,
    action_mask: np.ndarray,
    *,
    full_truth: bool,
    q: object | None = None,
    epsilon: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, np.ndarray]:
    state_normalized = output.physical_state[0].detach().cpu().numpy().copy()
    action_normalized = output.action[0].detach().cpu().numpy().copy()
    state = _physical_state(state_normalized, norm)
    action = action_normalized * norm["action"][1] + norm["action"][0]
    state = _finite_array("predicted State", state, (65, 70)).astype(np.float32)
    action = _finite_array("predicted Action", action, (64, 29)).astype(np.float32)
    raw, achieved, mapping = execution_actions(
        source["raw_action"],
        truth_action,
        action,
        action_mask,
        init["nominal_default_joint_pos"],
        init["action_scale"],
        init["action_offset"],
        init["action_clip"] if init["action_clip"].size else None,
        None if np.isnan(init["wrapper_action_clip"]) else float(init["wrapper_action_clip"]),
        full_prediction=full_truth,
    )
    completed_state = np.where(state_mask, state, truth_state)
    completed_action = np.where(action_mask, action, truth_action)
    result: dict[str, np.ndarray] = {
        "truth_state": truth_state.astype(np.float32),
        "truth_action": truth_action.astype(np.float32),
        "predicted_state": state,
        "predicted_action": action.astype(np.float32),
        "predicted_state_normalized": state_normalized.astype(np.float32),
        "predicted_action_normalized": action_normalized.astype(np.float32),
        "completed_state": completed_state.astype(np.float32),
        "completed_action": completed_action.astype(np.float32),
        "state_mask": state_mask.astype(bool),
        "action_mask": action_mask.astype(bool),
        "executed_raw": raw.astype(np.float32),
        "achieved_action": achieved.astype(np.float32),
        "valid_state": np.ones(65, dtype=bool),
        "valid_action": np.ones(64, dtype=bool),
        "mapping_replaced_elements": np.asarray(mapping["replaced_elements"], dtype=np.int64),
        "mapping_saturated_elements": np.asarray(mapping["saturated_elements"], dtype=np.int64),
    }
    if epsilon is not None:
        result["epsilon_global"] = epsilon[0][0].detach().cpu().numpy().astype(np.float32)
        result["epsilon_local"] = epsilon[1][0].detach().cpu().numpy().astype(np.float32)
        result["sampled_global"] = result["epsilon_global"].copy()
        result["sampled_local"] = result["epsilon_local"].copy()
    if q is not None:
        result.update(
            posterior_global_mean=q.global_mean[0].detach().cpu().numpy().astype(np.float32),
            posterior_local_mean=q.local_mean[0].detach().cpu().numpy().astype(np.float32),
            posterior_global_logvar=q.global_logvar[0].detach().cpu().numpy().astype(np.float32),
            posterior_local_logvar=q.local_logvar[0].detach().cpu().numpy().astype(np.float32),
            posterior_global_std=torch.exp(0.5 * q.global_logvar[0]).detach().cpu().numpy().astype(np.float32),
            posterior_local_std=torch.exp(0.5 * q.local_logvar[0]).detach().cpu().numpy().astype(np.float32),
        )
    return result


def _write_prediction(path: Path, arrays: dict[str, np.ndarray], source: dict, init: dict) -> None:
    _write_npz(path, **arrays)
    if "predicted_state" in arrays:
        truth_pos, truth_quat = source["root_pos"], source["root_quat"]
        # Use the same state integration utility as replay65 when available.
        from .state_mask_eval import reconstruct_root_trajectory

        truth_pos, truth_quat = reconstruct_root_trajectory(
            arrays["truth_state"], source["root_pos"], source["root_quat"], 0, .02
        )
        pred_pos, pred_quat = reconstruct_root_trajectory(
            arrays["completed_state"], source["root_pos"], source["root_quat"], 0, .02
        )
        base = path.parent
        _write_trajectory(
            base / "truth_integrated.trajectory.pkl",
            joint_pos=arrays["truth_state"][:, :29] + init["nominal_default_joint_pos"],
            root_pos=truth_pos,
            root_quat=truth_quat,
            fps=50.,
        )
        _write_trajectory(
            base / "predicted_integrated.trajectory.pkl",
            joint_pos=arrays["completed_state"][:, :29] + init["nominal_default_joint_pos"],
            root_pos=pred_pos,
            root_quat=pred_quat,
            fps=50.,
        )


def _selection(dataset, indices: list[int], fixtures: Fixtures, *, hardest: int, motion: str,
               motion_count: int, seed: int) -> list[dict]:
    rows = fixtures.manifest()
    by_index = {int(row["window_index"]): row for row in rows}
    if hardest not in by_index:
        raise ValueError(f"hardest window index {hardest} is not in this dataset")
    hard = dict(by_index[hardest])
    if str(hard["motion_key"]) != motion:
        raise ValueError(f"window {hardest} is {hard['motion_key']!r}, expected {motion!r}")
    if int(hard["valid_actions"]) != 64 or int(hard["valid_states"]) != 65:
        raise ValueError("hardest window is not a complete T64 window")
    if motion_count < 1:
        raise ValueError("motion-count must be positive")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["motion_key"] == motion:
            continue
        if int(row["valid_actions"]) != 64 or int(row["valid_states"]) != 65:
            continue
        grouped.setdefault(str(row["motion_key"]), []).append(dict(row))
    needed = motion_count - 1
    if len(grouped) < needed:
        raise ValueError("not enough complete distinct motions for selection")
    generator = np.random.default_rng(int(seed))
    keys = sorted(grouped)
    chosen_keys = [keys[int(i)] for i in generator.permutation(len(keys))[:needed]]
    selected = [hard]
    for key in chosen_keys:
        candidates = sorted(grouped[key], key=lambda row: int(row["window_index"]))
        selected.append(candidates[int(generator.integers(0, len(candidates)))])
    for row in selected:
        if int(row["valid_actions"]) != 64 or int(row["valid_states"]) != 65:
            raise ValueError("selected window is not a complete T64 window")
    return selected


def _motion_manifest(source: dict, source_meta: dict, row: dict, init_manifest: dict, simulation_seed: int) -> dict:
    return {
        "version": VERSION,
        "window": row,
        "route": "C",
        "mask": MASK,
        "simulation_seed": int(simulation_seed),
        "source": source_meta,
        "initialization": init_manifest,
        "entries": [],
        "limitations": [
            "posterior mean is a full-truth posterior reference, not physical ground truth",
            "full_action State video retains recorded State outside the Action replay",
        ],
    }


def _source_identity(checkpoint: dict, dataset_run: Path, rows: list[dict], allow_recovered: bool) -> dict:
    identity = data_identity(dataset_run, rows)
    try:
        return check_source_identity(
            checkpoint,
            identity,
            allow_unknown=False,
            allow_manifest_mismatch=allow_recovered,
        )
    except ValueError:
        if not allow_recovered:
            raise
        old = checkpoint.get("dataset_identity") or {}
        keys = ("dataset_manifest_sha256", "episodes_index_sha256", "normalization_sha256",
                "selected_windows_sha256", "selected_window_count")
        mismatches = [key for key in keys if old.get(key) is not None and old.get(key) != identity[key]]
        missing = [key for key in keys if old.get(key) is None]
        if missing:
            raise ValueError("recovered dataset still requires checkpoint identity fields: " + ", ".join(missing))
        return {
            "verified_fields": [], "unknown_fields": [], "exact_identity_verified": False,
            "recovered_manifest_mismatch": "dataset_manifest_sha256" in mismatches,
            "recovered_identity_mismatches": mismatches,
            "identity_warning": "explicit recovered-dataset mode allowed identity mismatch: " + ", ".join(mismatches),
        }


def prepare(args) -> None:
    run = args.output_run.resolve()
    checkpoint_path = args.checkpoint.resolve()
    dataset_run = args.dataset_run.resolve()
    assert_output_isolated(run, [dataset_run, checkpoint_path.parent.parent, ROOT / "state-action-cvae", KIT])
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("prepare requires a new empty output run")
    if args.mask != MASK:
        raise ValueError("latent sweep currently requires --mask full_action")
    seeds = tuple(int(seed) for seed in args.sample_seeds)
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("sample seeds must be non-empty and unique")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    signature = checkpoint.get("model_signature", {})
    if signature.get("architecture_version") != HierarchicalStandardCVAETransformer.ARCHITECTURE_VERSION:
        raise ValueError("architecture signature mismatch: old checkpoints are not supported")
    if checkpoint.get("format_version") != CHECKPOINT or checkpoint.get("stage") != "C":
        raise ValueError("latent sweep requires a v2 Stage-C checkpoint")
    config = checkpoint_config(checkpoint)
    model = build_model(config["model"])
    load_checkpoint(model, checkpoint_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    dataset, indices = make_dataset(dataset_run, config)
    try:
        fixtures = Fixtures(dataset, indices, expand=True)
        rows = fixtures.manifest()
        identity_check = _source_identity(checkpoint, dataset_run, rows, args.allow_recovered_dataset)
        selected = _selection(dataset, indices, fixtures, hardest=args.hardest_window_index,
                              motion=args.hardest_motion_key, motion_count=args.motion_count,
                              seed=args.selection_seed)
        norm = read_normalization(dataset_run / "data/normalization.npz")
        run.mkdir(parents=True, exist_ok=True)
        for folder in ("manifests", "motions", "aggregate", "plots", "videos", "markers", "logs"):
            (run / folder).mkdir(parents=True, exist_ok=True)
        selection_manifest = {
            "selection_version": "latent_sensitivity_motion_v1",
            "hardest_window_index": int(args.hardest_window_index),
            "hardest_motion_key": args.hardest_motion_key,
            "selection_seed": int(args.selection_seed),
            "motion_count": int(args.motion_count),
            "mask": MASK,
            "windows": selected,
        }
        atomic_write_json(run / "manifests/selection.json", selection_manifest)
        atomic_write_json(run / "manifests/sweep.json", {
            "version": VERSION, "checkpoint": source_info(checkpoint_path, checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint_path), "dataset_run": str(dataset_run),
            "mask": MASK, "mask_slot": MASK_SLOT, "sample_seeds": list(seeds),
            "sample_index": int(args.sample_index), "simulation_seed": int(args.simulation_seed),
            "selection_seed": int(args.selection_seed), "identity_check": identity_check,
            "recovered_dataset": bool(args.allow_recovered_dataset),
            "exact_identity_verified": bool(identity_check["exact_identity_verified"]),
            "posterior_reference": "full_truth_posterior_mean",
        })
        atomic_write_json(run / "manifests/source_identity.json", {
            "dataset_identity": data_identity(dataset_run, rows), "identity_check": identity_check,
            "recovered_dataset": bool(args.allow_recovered_dataset),
        })
        shutil.copy2(dataset_run / "data/normalization.npz", run / "data_normalization.npz")
        for ordinal, row in enumerate(selected):
            index = int(row["window_index"])
            source, init, schema, source_meta = load_source(dataset, index)
            motion_dir = run / "motions" / f"m{ordinal:03d}_window{index:05d}"
            # Do not create a simulation child here.  An existing child without
            # its worker completion seal is deliberately treated as incomplete
            # by _simulation_child, so only create its parent container.
            for folder in ("reference", "samples", "baseline", "metrics", "videos", "data", "manifests", "logs", "markers"):
                (motion_dir / folder).mkdir(parents=True, exist_ok=True)
            _write_npz(motion_dir / "data/recorded_hdf.replay.npz", **source)
            _write_trajectory(motion_dir / "data/recorded_hdf.trajectory.pkl",
                              joint_pos=source["joint_pos"], root_pos=source["root_pos"],
                              root_quat=source["root_quat"], fps=50.)
            from .posterior_h50_action_replay_exact_init import _write_exact_initialization
            _, init_manifest = _write_exact_initialization(motion_dir / "data/exact_initialization.npz", init)
            atomic_write_json(motion_dir / "manifests/exact_initialization.json", init_manifest)
            atomic_write_json(motion_dir / "manifests/source_schema.json", schema)
            motion_meta = _motion_manifest(source, source_meta, row, init_manifest, args.simulation_seed)
            atomic_write_json(motion_dir / "manifests/replay65.json", motion_meta)
            sample = dict(dataset[index])
            sample.update(stable_window_id=row["stable_window_id"], window_index=index,
                          fixture_index=ordinal, mask_slot=MASK_SLOT)
            cpu = default_collate([sample])
            valid_state = cpu["valid_state"].bool()
            valid_action = cpu["valid_action"].bool()
            state_mask = torch.zeros_like(cpu["physical_state"], dtype=torch.bool)
            action_mask = valid_action[..., None].expand_as(cpu["action"]).clone()
            truth_batch = _device_batch(cpu, device)
            condition_batch = dict(truth_batch)
            condition_batch["action"] = condition_batch["action"].masked_fill(action_mask.to(device), 0.)
            state_mask_d, action_mask_d = state_mask.to(device), action_mask.to(device)
            truth_state, truth_action = source["physical_state"], source["action_target_canonical"]
            with isolated_rng(), torch.inference_mode():
                posterior = model.encode_posterior_distribution(truth_batch)
                if tuple(posterior.global_mean.shape) != (1, 256) or tuple(posterior.local_mean.shape) != (1, 16, 128):
                    raise ValueError(
                        "latent sweep requires C global/local shapes [256] and [16,128]; "
                        f"got {tuple(posterior.global_mean.shape)} and {tuple(posterior.local_mean.shape)}"
                    )
                condition = model.encode_condition(condition_batch, state_mask_d, action_mask_d)
                fused = model.fuse_latents((posterior.global_mean, posterior.local_mean), condition)
                reference_out = model.decode_from_fused_latents(condition_batch, fused, condition)
            reference_arrays = _prediction_arrays(reference_out, norm, truth_state, truth_action, source, init,
                                                  state_mask[0].numpy(), action_mask[0].numpy(), full_truth=True, q=posterior)
            _write_prediction(motion_dir / "reference/posterior_mean.npz", reference_arrays, source, init)
            atomic_write_json(motion_dir / "reference/manifest.json", {
                "name": "full_truth_posterior_mean", "condition_mask": MASK,
                "posterior_input": "complete physical_state and action", "shape_global": [256],
                "shape_local": [16, 128],
            })
            for seed in seeds:
                sample_dir = motion_dir / "samples" / f"seed_{seed}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                with isolated_rng(), torch.inference_mode():
                    epsilon = epsilon_for(model, cpu, args.sample_index, seed, device)
                    out = model.infer_from_condition(condition_batch, state_mask_d, action_mask_d, epsilon=epsilon)
                arrays = _prediction_arrays(out, norm, truth_state, truth_action, source, init,
                                            state_mask[0].numpy(), action_mask[0].numpy(), full_truth=True,
                                            epsilon=epsilon)
                _write_prediction(sample_dir / "prediction.npz", arrays, source, init)
                atomic_write_json(sample_dir / "manifest.json", {
                    "seed": int(seed), "sample_index": int(args.sample_index), "mask": MASK,
                    "simulation_seed": int(args.simulation_seed), "latent_source": "standard_normal",
                    "posterior_encoder_executed": False,
                })
            baseline = motion_dir / "baseline"
            _write_npz(baseline / "raw_actions.npz", raw_actions=np.stack([source["raw_action"]] * 2, axis=1),
                       scenario_names=np.asarray(["original_1", "original_2"]))
            atomic_write_json(motion_dir / "manifests/prepared.json", {
                "window": row, "motion_index": ordinal, "sample_seeds": list(seeds),
                "reference": "reference/posterior_mean.npz", "baseline": "baseline/raw_actions.npz",
                "source": source_meta,
            })
        # Seal only read-only prepared inputs. Simulation/render/report artifacts are intentionally excluded.
        sealed = {}
        for p in run.rglob("*"):
            if p.is_file() and p.name not in {"progress.json"} and ("simulations" not in p.parts) and ("videos" not in p.parts):
                sealed[str(p.relative_to(run))] = file_sha256(p)
        atomic_write_json(run / "manifests/prepared_hashes.json", sealed)
        atomic_write_text(run / "markers/latent_sweep_prepared.ok", "PREPARED\n")
    finally:
        dataset.close()


def _verify_prepared(run: Path) -> None:
    hashes = load_json(run / "manifests/prepared_hashes.json")
    for relative, expected in hashes.items():
        path = (run / relative).resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"prepared artifact changed or missing: {relative}")


def _simulation_child(motion_dir: Path, tag: str, names: list[str], raws: list[np.ndarray]) -> Path:
    if tag == "baseline":
        child = motion_dir / "baseline/simulations"
    elif tag == "posterior_mean":
        child = motion_dir / "reference/simulations/full_action"
    else:
        child = motion_dir / f"samples/{tag}/simulations/full_action"
    if child.exists():
        complete = child / "manifests/replay65_worker_complete.json"
        if complete.is_file():
            verify_worker(child)
            return child
        raise FileExistsError(f"incomplete/changed simulation is not overwritten: {child}")
    for folder in ("data", "manifests", "logs", "markers"):
        (child / folder).mkdir(parents=True, exist_ok=True)
    shutil.copy2(motion_dir / "data/exact_initialization.npz", child / "data/exact_initialization.npz")
    meta = load_json(motion_dir / "manifests/replay65.json")
    raw_path = child / "data/raw_actions.npz"
    _write_npz(raw_path, raw_actions=np.stack(raws, axis=1), scenario_names=np.asarray(names))
    request = {
        "schema_version": "65-token-replay-v1", "representation": "physics_v4", "replay65_guard": True,
        "motion_file": meta["source"]["motion_file"], "motion_file_sha256": meta["source"]["motion_file_sha256"],
        "motion_key": meta["window"]["motion_key"], "raw_actions_file": str(raw_path),
        "raw_actions_sha256": file_sha256(raw_path), "steps": 64, "num_envs": len(names),
        "scenario_names": names, "control_dt": .02,
        "exact_initialization_file": str(child / "data/exact_initialization.npz"),
        "exact_initialization_file_sha256": meta["initialization"]["file_sha256"],
        "exact_initialization_payload_sha256": meta["initialization"]["payload_sha256"],
        "exact_initialization_report_paths": [str(child / f"manifests/exact_initialization_readback_{i:06d}.json") for i in range(len(names))],
    }
    atomic_write_json(child / "manifests/action_replay_request.json", request)
    atomic_write_json(child / "manifests/action_mask_request.json", {"seed": meta["simulation_seed"]})
    env = dict(os.environ, ACTION_MASK_RUN_DIR=str(child), PYTHONUNBUFFERED="1")
    launch(["bash", str(KIT / "sonic_repro.sh"), "replay-action-mask"], child / "logs/worker.log", env)
    hashes = {p: file_sha256(child / p) for p in ("data/raw_actions.npz", "data/exact_initialization.npz",
                                                  "manifests/action_replay_request.json", "manifests/action_mask_request.json")}
    for i in range(len(names)):
        for rel in (f"data/replay/{i:06d}.replay.npz", f"data/replay/{i:06d}.trajectory.pkl",
                    f"manifests/exact_initialization_readback_{i:06d}.json", f"data/replay/{i:06d}.runtime.json"):
            hashes[rel] = file_sha256(child / rel)
    atomic_write_json(child / "manifests/replay65_worker_complete.json", {"hashes": hashes})
    return child


def simulate(args) -> None:
    run = args.run.resolve()
    _verify_prepared(run)
    sweep = load_json(run / "manifests/sweep.json")
    seeds = [int(v) for v in sweep["sample_seeds"]]
    for motion_dir in sorted((run / "motions").glob("m*_window*")):
        source = _load_replay(motion_dir / "data/recorded_hdf.replay.npz")
        _simulation_child(motion_dir, "baseline", ["original_1", "original_2"], [source["raw_action"]] * 2)
        reference = np.load(motion_dir / "reference/posterior_mean.npz", allow_pickle=False)["executed_raw"].copy()
        _simulation_child(motion_dir, "posterior_mean", ["posterior_mean_reference"], [reference])
        for seed in seeds:
            raw = np.load(motion_dir / "samples" / f"seed_{seed}" / "prediction.npz", allow_pickle=False)["executed_raw"].copy()
            # Keep the simulation directory aligned with the prepared sample
            # directory and with render/report lookup: samples/seed_<N>/...
            _simulation_child(motion_dir, f"seed_{seed}", [f"seed_{seed}"], [raw])
    atomic_write_text(run / "markers/latent_sweep_simulation_complete.ok", "SIMULATION COMPLETE\n")


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))))


def _correlation(x: list[float], y: list[float]) -> dict:
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return {"count": int(len(a)), "pearson": None, "spearman": None}
    rank_a = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
    rank_b = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(np.float64)
    return {"count": int(len(a)), "pearson": float(np.corrcoef(a, b)[0, 1]),
            "spearman": float(np.corrcoef(rank_a, rank_b)[0, 1])}


def _latent_metrics(sample: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict:
    result = {}
    for name, axes in (("global", ("sampled_global", "posterior_global_mean")), ("local", ("sampled_local", "posterior_local_mean"))):
        a, b = sample[axes[0]].astype(np.float64), reference[axes[1]].astype(np.float64)
        diff = a - b
        flat_a, flat_b = a.reshape(-1), b.reshape(-1)
        denom = max(float(np.linalg.norm(flat_b)), 1e-12)
        cosine = float(np.dot(flat_a, flat_b) / max(np.linalg.norm(flat_a) * np.linalg.norm(flat_b), 1e-12))
        result[name] = {
            "rmse": _rmse(a, b), "mae": float(np.mean(np.abs(diff))),
            "l2_normalized": float(np.linalg.norm(diff) / denom), "cosine_similarity": cosine,
            "max_abs": float(np.abs(diff).max()),
        }
    result["local_chunk_rmse"] = [_rmse(sample["sampled_local"][i], reference["posterior_local_mean"][i]) for i in range(16)]
    result["local_chunk_l2"] = [float(np.linalg.norm(sample["sampled_local"][i] - reference["posterior_local_mean"][i]) /
                                         max(np.linalg.norm(reference["posterior_local_mean"][i]), 1e-12)) for i in range(16)]
    result["posterior_std_normalized_gap"] = {
        "global": float(np.sqrt(np.mean(np.square((sample["sampled_global"] - reference["posterior_global_mean"]) /
                                                   np.maximum(reference["posterior_global_std"], 1e-8))))),
        "local": float(np.sqrt(np.mean(np.square((sample["sampled_local"] - reference["posterior_local_mean"]) /
                                                  np.maximum(reference["posterior_local_std"], 1e-8))))),
    }
    return result


def _output_metrics(sample: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict:
    result = {"action_saturation_count": int(np.asarray(sample.get("mapping_saturated_elements", 0)).item())}
    for name in ("state", "action"):
        pred = sample[f"predicted_{name}"].astype(np.float64)
        ref = reference[f"predicted_{name}"].astype(np.float64)
        truth = sample[f"truth_{name}"].astype(np.float64)
        diff = pred - ref
        result[name] = {
            "full_rmse": _rmse(pred, ref), "mae": float(np.mean(np.abs(diff))),
            "max_abs": float(np.abs(diff).max()), "truth_rmse": _rmse(pred, truth),
            "per_timestep_rmse": [float(np.sqrt(np.mean(np.square(diff[t])))) for t in range(len(diff))],
            "per_feature_rmse": [float(np.sqrt(np.mean(np.square(diff[..., f])))) for f in range(diff.shape[-1])],
        }
    return result


def _physical_for_motion(motion_dir: Path, seeds: list[int]) -> dict:
    result = {}
    seed_replays = []
    baseline_dir = motion_dir / "baseline/simulations"
    base0 = _load_replay(baseline_dir / "data/replay/000000.replay.npz") if (baseline_dir / "data/replay/000000.replay.npz").is_file() else None
    base1 = _load_replay(baseline_dir / "data/replay/000001.replay.npz") if (baseline_dir / "data/replay/000001.replay.npz").is_file() else None
    baseline_valid = False
    if base0 is not None and base1 is not None:
        init_checks = []
        for index in (0, 1):
            readback = motion_dir / f"baseline/simulations/manifests/exact_initialization_readback_{index:06d}.json"
            runtime = motion_dir / f"baseline/simulations/data/replay/{index:06d}.runtime.json"
            if readback.is_file() and runtime.is_file():
                init_checks.append(initialization_checks(load_json(readback), load_json(motion_dir / "manifests/replay65.json")["initialization"], load_json(runtime)))
        baseline_valid = len(init_checks) == 2 and all(all(check.values()) for check in init_checks)
    result["baseline_valid"] = baseline_valid if base0 is not None and base1 is not None else None
    result["model_physical_quality"] = "UNDETERMINED" if result["baseline_valid"] is not True else "DIAGNOSTIC_ONLY"
    if base0 is not None and base1 is not None:
        result["original_repeatability"] = _trajectory_metrics(base0, base1)
    for label, child, idx in [("posterior_mean_reference", motion_dir / "reference/simulations/full_action", 0)] + [
        (f"seed_{seed}", motion_dir / f"samples/seed_{seed}/simulations/full_action", 0) for seed in seeds
    ]:
        path = child / f"data/replay/{idx:06d}.replay.npz"
        if path.is_file() and base0 is not None:
            replay = _load_replay(path)
            if label.startswith("seed_"):
                seed_replays.append(replay)
            result[label] = _trajectory_metrics(base0, replay)
            result[label]["joint_position_rmse_rad"] = float(_rmse(base0["joint_pos"], replay["joint_pos"]))
            result[label]["root_position_rmse_m"] = float(_rmse(base0["root_pos"], replay["root_pos"]))
            result[label]["first_threshold_crossing_frame"] = _first_threshold_crossings(_per_frame_errors(base0, replay))
    if len(seed_replays) == len(seeds):
        result["realized_seed_spread"] = {
            "realized_joint_position_seed_std": np.stack([row["joint_pos"] for row in seed_replays]).std(axis=0).tolist(),
            "realized_root_position_seed_std": np.stack([row["root_pos"] for row in seed_replays]).std(axis=0).tolist(),
            "realized_body_position_seed_std": np.stack([row["body_pos"] for row in seed_replays]).std(axis=0).tolist(),
        }
    return result


def _write_plots(run: Path, rows: list[dict], physical_rows: list[dict] | None = None,
                 spread_rows: list[dict] | None = None) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as error:
        atomic_write_text(run / "plots/README.txt", f"matplotlib unavailable: {error}\n")
        return
    labels = [f"{r['motion_key']}\nseed={r['seed']}" for r in rows]
    values = [r["latent"]["global"]["rmse"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4)); ax.bar(np.arange(len(values)), values); ax.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right"); ax.set_ylabel("global latent RMSE vs posterior mean"); fig.tight_layout(); fig.savefig(run / "plots/latent_global_distance.svg"); plt.close(fig)
    values = [r["output"]["action"]["full_rmse"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4)); ax.bar(np.arange(len(values)), values); ax.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right"); ax.set_ylabel("Action RMSE vs posterior mean"); fig.tight_layout(); fig.savefig(run / "plots/action_output_distance.svg"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, latent_name in zip(axes, ("global", "local")):
        x = [row["latent"][latent_name]["rmse"] for row in rows]
        y = [row["output"]["action"]["full_rmse"] for row in rows]
        ax.scatter(x, y, s=20); ax.set_xlabel(f"{latent_name} latent RMSE"); ax.set_ylabel("Action RMSE")
    fig.tight_layout(); fig.savefig(run / "plots/latent_to_action_correlation.svg"); plt.close(fig)
    physical_by_motion = {row["motion_key"]: row["physical"] for row in (physical_rows or [])}
    realized_points = [(row, physical_by_motion.get(row["motion_key"], {}).get(f"seed_{row['seed']}", {}).get("body_mpjpe_m")) for row in rows]
    realized_points = [(row, value) for row, value in realized_points if value is not None]
    if realized_points:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, latent_name in zip(axes, ("global", "local")):
            ax.scatter([row["latent"][latent_name]["rmse"] for row, _ in realized_points], [value for _, value in realized_points], s=20)
            ax.set_xlabel(f"{latent_name} latent RMSE"); ax.set_ylabel("realized body MPJPE (m)")
        fig.tight_layout(); fig.savefig(run / "plots/latent_to_realized_state_correlation.svg"); plt.close(fig)
    local_chunks = np.asarray([row["latent"]["local_chunk_rmse"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4)); image = ax.imshow(local_chunks, aspect="auto"); ax.set_xlabel("local chunk"); ax.set_ylabel("motion/seed row"); fig.colorbar(image, ax=ax, label="RMSE vs posterior mean"); fig.tight_layout(); fig.savefig(run / "plots/local_latent_chunk_distance_heatmap.svg"); plt.close(fig)
    if spread_rows:
        for domain, filename in (("action", "action_seed_spread_heatmap.svg"), ("state", "state_seed_spread_heatmap.svg")):
            arrays = np.stack([np.asarray(row[f"{domain}_seed_std"], dtype=float) for row in spread_rows])
            values = arrays.mean(axis=0).T
            fig, ax = plt.subplots(figsize=(10, 5)); image = ax.imshow(values, aspect="auto", origin="lower"); ax.set_xlabel("frame"); ax.set_ylabel("joint" if domain == "action" else "State feature"); fig.colorbar(image, ax=ax, label="seed std"); fig.tight_layout(); fig.savefig(run / "plots" / filename); plt.close(fig)
    for domain, length, filename in (("action", 64, "action_seed_spread.svg"), ("state", 65, "state_seed_spread.svg")):
        fig, ax = plt.subplots(figsize=(10, 4))
        grouped = {}
        for row in rows:
            values = np.asarray(row["output"][domain]["per_timestep_rmse"], dtype=float)
            grouped.setdefault(row["motion_key"], []).append(values)
        for motion, arrays in grouped.items():
            stack = np.stack(arrays)
            ax.plot(np.arange(stack.shape[1]), stack.mean(axis=0), label=motion)
        ax.set_xlabel("frame"); ax.set_ylabel(f"{domain} seed spread"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(run / "plots" / filename); plt.close(fig)


def report(args) -> dict:
    run = args.run.resolve()
    _verify_prepared(run)
    (run / "plots").mkdir(parents=True, exist_ok=True)
    (run / "aggregate").mkdir(parents=True, exist_ok=True)
    sweep = load_json(run / "manifests/sweep.json")
    seeds = [int(v) for v in sweep["sample_seeds"]]
    latent_rows, output_rows, physical_rows, timestep_rows, spread_rows = [], [], [], [], []
    for motion_dir in sorted((run / "motions").glob("m*_window*")):
        selection = load_json(motion_dir / "manifests/prepared.json")["window"]
        with np.load(motion_dir / "reference/posterior_mean.npz", allow_pickle=False) as archive:
            reference = {k: archive[k].copy() for k in archive.files}
        sample_predictions = {}
        for seed in seeds:
            with np.load(motion_dir / "samples" / f"seed_{seed}/prediction.npz", allow_pickle=False) as archive:
                sample = {k: archive[k].copy() for k in archive.files}
            latent = _latent_metrics(sample, reference)
            output = _output_metrics(sample, reference)
            row = {"motion_key": selection["motion_key"], "window_index": int(selection["window_index"]), "seed": seed,
                   "latent": latent, "output": output}
            latent_rows.append({k: row[k] for k in ("motion_key", "window_index", "seed", "latent")})
            output_rows.append({k: row[k] for k in ("motion_key", "window_index", "seed", "latent", "output")})
            for t, value in enumerate(output["action"]["per_timestep_rmse"]):
                timestep_rows.append({"motion_key": selection["motion_key"], "window_index": int(selection["window_index"]), "seed": seed, "domain": "action", "t": t, "rmse_vs_posterior_mean": value})
            sample_predictions[seed] = sample
        # Seed spread is recorded at motion level and as a JSONL row per motion.
        stack_state = np.stack([sample_predictions[s]["predicted_state"] for s in seeds])
        stack_action = np.stack([sample_predictions[s]["predicted_action"] for s in seeds])
        spread = {"motion_key": selection["motion_key"], "window_index": int(selection["window_index"]),
                  "state_seed_std": stack_state.std(axis=0).tolist(), "action_seed_std": stack_action.std(axis=0).tolist(),
                  "state_pairwise_rmse": [_rmse(stack_state[i], stack_state[j]) for i in range(len(seeds)) for j in range(i + 1, len(seeds))],
                  "action_pairwise_rmse": [_rmse(stack_action[i], stack_action[j]) for i in range(len(seeds)) for j in range(i + 1, len(seeds))],
                  "state_pairwise_max": [float(np.abs(stack_state[i] - stack_state[j]).max()) for i in range(len(seeds)) for j in range(i + 1, len(seeds))],
                  "action_pairwise_max": [float(np.abs(stack_action[i] - stack_action[j]).max()) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]}
        spread_rows.append(spread)
        atomic_write_json(motion_dir / "metrics/seed_spread.json", spread)
        physical_rows.append({"motion_key": selection["motion_key"], "window_index": int(selection["window_index"]),
                              "physical": _physical_for_motion(motion_dir, seeds)})
    physical_by_motion = {row["motion_key"]: row["physical"] for row in physical_rows}
    correlation_groups: dict[str, list[dict]] = {"all": []}
    for latent_row, output_row in zip(latent_rows, output_rows):
        motion = latent_row["motion_key"]; seed = latent_row["seed"]
        realized = physical_by_motion.get(motion, {}).get(f"seed_{seed}", {})
        point = {"global_distance": latent_row["latent"]["global"]["rmse"],
                 "local_distance": latent_row["latent"]["local"]["rmse"],
                 "action_rmse": output_row["output"]["action"]["full_rmse"],
                 "realized_state_divergence": realized.get("body_mpjpe_m")}
        correlation_groups["all"].append(point); correlation_groups.setdefault(motion, []).append(point)
    correlations = {}
    for group, points in correlation_groups.items():
        valid_physical = [point for point in points if point["realized_state_divergence"] is not None]
        correlations[group] = {
            "global_to_action": _correlation([p["global_distance"] for p in points], [p["action_rmse"] for p in points]),
            "local_to_action": _correlation([p["local_distance"] for p in points], [p["action_rmse"] for p in points]),
            "global_to_realized_state": _correlation([p["global_distance"] for p in valid_physical], [p["realized_state_divergence"] for p in valid_physical]),
            "local_to_realized_state": _correlation([p["local_distance"] for p in valid_physical], [p["realized_state_divergence"] for p in valid_physical]),
        }
    atomic_write_json(run / "aggregate/latent_output_correlations.json", correlations)
    aggregate = {
        "version": VERSION, "mask": MASK, "sample_seeds": seeds,
        "recovered_dataset": bool(sweep["recovered_dataset"]),
        "exact_identity_verified": bool(sweep["exact_identity_verified"]),
        "latent_rows": len(latent_rows), "output_rows": len(output_rows),
        "physical_rows": len(physical_rows), "correlations": correlations, "limitations": [
            "This is a latent-sensitivity diagnostic; it is not a training-quality gate.",
            "If original Action replay is invalid, physical model quality is UNDETERMINED.",
        ],
    }
    atomic_write_json(run / "manifests/latent_sweep_report.json", aggregate)
    for path, rows in ((run / "aggregate/latent_metrics.jsonl", latent_rows), (run / "aggregate/output_metrics.jsonl", output_rows), (run / "aggregate/physical_metrics.jsonl", physical_rows), (run / "aggregate/per_timestep_metrics.jsonl", timestep_rows)):
        path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    _write_plots(run, output_rows, physical_rows, spread_rows)
    # Report is complete once all prediction and simulation artifacts are present; rendering is checked separately.
    required = []
    for motion_dir in sorted((run / "motions").glob("m*_window*")):
        required.extend([motion_dir / "reference/posterior_mean.npz", motion_dir / "reference/simulations/full_action/data/replay/000000.replay.npz", motion_dir / "baseline/simulations/data/replay/000000.replay.npz", motion_dir / "baseline/simulations/data/replay/000001.replay.npz"])
        for seed in seeds:
            required.append(motion_dir / f"samples/seed_{seed}/prediction.npz")
    simulations_complete = all(path.is_file() and path.stat().st_size > 0 for path in required)
    videos = list(run.glob("motions/*/videos/*.mp4"))
    render_complete = len(videos) == len(list((run / "motions").glob("m*_window*"))) * (2 + len(seeds)) and all(p.stat().st_size > 0 for p in videos)
    aggregate["simulation_complete"] = simulations_complete
    aggregate["render_complete"] = render_complete
    aggregate["execution_complete"] = bool(simulations_complete and render_complete)
    atomic_write_json(run / "manifests/latent_sweep_report.json", aggregate)
    if aggregate["execution_complete"]:
        atomic_write_text(run / "markers/latent_sweep_execution.ok", "EXECUTION COMPLETE; quality is diagnostic only\n")
    if args.export:
        output = (args.output or (run / "latent_sweep_report.zip")).resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite report archive: {output}")
        files = [p for p in run.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl", ".log", ".svg", ".npz"}]
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, str(path.relative_to(run)))
            archive.writestr("report_index.json", json.dumps({"hashes": {str(p.relative_to(run)): file_sha256(p) for p in files}, "excluded": ["videos", "trajectory pickle", "HDF5"]}))
        print(f"REPORT={output}", flush=True)
    return aggregate


def _render(args) -> None:
    # Rendering follows the same writer/camera contract as render_replay65 but
    # uses the sweep's explicit sample directories and labels.
    run = args.run.resolve(); _verify_prepared(run)
    os.environ["MUJOCO_GL"] = args.gl
    import cv2
    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    import mujoco
    if str(KIT) not in sys.path:
        sys.path.insert(0, str(KIT))
    from render_h50a_exact_action_replays import _validate_joint_order, _writer
    from render_mujoco_trajectory import G1_ISAACLAB_JOINT_NAMES, build_mujoco_qpos, load_trajectory, prepare_runtime_xml
    sweep = load_json(run / "manifests/sweep.json"); seeds = [int(v) for v in sweep["sample_seeds"]]
    for motion_dir in sorted((run / "motions").glob("m*_window*")):
        xml = motion_dir / "manifests/replay65_render.xml"
        prepare_runtime_xml(args.model.resolve(), xml, 640, 480)
        model = mujoco.MjModel.from_xml_path(str(xml)); _validate_joint_order(model)
        data = mujoco.MjData(model); renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera); camera.type = mujoco.mjtCamera.mjCAMERA_FREE; camera.azimuth = 135; camera.elevation = -15; camera.distance = 3.5
        truth = motion_dir / "data/recorded_hdf.trajectory.pkl"; baseline = motion_dir / "baseline/simulations/data/replay"
        jobs = [("original_repeatability", [truth, baseline / "000000.trajectory.pkl", baseline / "000001.trajectory.pkl"], ["Recorded HDF pose", "Original Action replay 1", "Original Action replay 2"], None)]
        jobs.append(("posterior_mean_full_action_action", [truth, baseline / "000000.trajectory.pkl", motion_dir / "reference/simulations/full_action/data/replay/000000.trajectory.pkl"], ["Recorded HDF pose", "Original Action replay", "C | full_action | posterior_mean_reference"], "posterior"))
        for seed in seeds:
            jobs.append((f"seed_{seed}_full_action_action", [truth, baseline / "000000.trajectory.pkl", motion_dir / f"samples/seed_{seed}/simulations/full_action/data/replay/000000.trajectory.pkl"], ["Recorded HDF pose", "Original Action replay", f"C | full_action | seed={seed}"], "sample"))
        try:
            for name, paths, labels, kind in jobs:
                if any(not path.is_file() for path in paths):
                    raise FileNotFoundError(f"render input missing for {name}")
                trajectories = [load_trajectory(path) for path in paths]; qposes = [build_mujoco_qpos(t) for t in trajectories]
                if any(q.shape != (65, 36) or not np.isfinite(q).all() for q in qposes):
                    raise ValueError("video requires 65 finite frames")
                output = motion_dir / "videos" / f"{name}.mp4"; writer = _writer(imageio, output, 50.)
                try:
                    for t in range(65):
                        panels = []
                        for qpos, label in zip(qposes, labels):
                            data.qpos[:] = qpos[t]; data.qvel[:] = 0; mujoco.mj_forward(model, data); camera.lookat[:] = qposes[0][t, :3]; renderer.update_scene(data, camera=camera); frame = renderer.render().copy()
                            cv2.rectangle(frame, (0, 0), (640, 84), (25, 25, 25), -1)
                            cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .52, (245, 245, 245), 1)
                            cv2.putText(frame, "LATENT SENSITIVITY / diagnostic only", (10, 47), cv2.FONT_HERSHEY_SIMPLEX, .40, (80, 190, 255), 1)
                            cv2.putText(frame, f"C | full_action | frame {t}/64 | 50 Hz", (10, 71), cv2.FONT_HERSHEY_SIMPLEX, .43, (230, 230, 230), 1)
                            panels.append(frame)
                        writer.append_data(np.concatenate(panels, axis=1))
                finally:
                    writer.close()
                frames, seconds = count_frames_and_secs(str(output))
                if frames != 65 or output.stat().st_size <= 0: raise RuntimeError(f"invalid video {output}")
        finally:
            renderer.close()
    atomic_write_json(run / "manifests/render.json", {"version": VERSION, "videos": [str(p.relative_to(run)) for p in run.glob("motions/*/videos/*.mp4")]})


def render(args) -> None:
    _render(args)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for flag in ("checkpoint", "dataset-run", "output-run"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--hardest-window-index", type=int, default=HARD_WINDOW)
    p.add_argument("--hardest-motion-key", default=HARD_MOTION)
    p.add_argument("--motion-count", type=int, default=3)
    p.add_argument("--selection-seed", type=int, default=20260928)
    p.add_argument("--sample-seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--mask", choices=(MASK,), default=MASK)
    p.add_argument("--simulation-seed", type=int, default=SIMULATION_SEED)
    p.add_argument("--allow-recovered-dataset", action="store_true")
    p.add_argument("--device")
    s = sub.add_parser("simulate"); s.add_argument("--run", type=Path, required=True)
    r = sub.add_parser("render"); r.add_argument("--run", type=Path, required=True); r.add_argument("--model", type=Path, required=True); r.add_argument("--gl", choices=("egl", "osmesa", "glfw"), default="egl")
    q = sub.add_parser("report"); q.add_argument("--run", type=Path, required=True); q.add_argument("--export", action="store_true"); q.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    run_path = getattr(args, "output_run", getattr(args, "run", None)).resolve()
    if sys.platform == "linux":
        runs_root = Path("/home/helloworld/bly/runs").resolve()
        if run_path == runs_root or not run_path.is_relative_to(runs_root):
            parser.error("Ubuntu latent sweep outputs must be children of /home/helloworld/bly/runs")
    if args.command == "prepare": prepare(args)
    elif args.command == "simulate":
        with run_lock(args.run.resolve()): simulate(args)
    elif args.command == "render": render(args)
    elif args.command == "report": report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
