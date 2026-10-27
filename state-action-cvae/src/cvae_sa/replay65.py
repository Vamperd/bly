"""Read-only 65-token replay preparation, isolated simulation, rendering and reporting.

All outputs live in a NEW run. No training source or checkpoint is modified.
Isaac/MuJoCo are imported only by the separately launched Ubuntu workers.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import torch
from torch.utils.data import default_collate

from .action_masks import relative_to_raw_action
from .cvae_diagnostics import epsilon_for, read_normalization, route_output, stats
from .cvae_protocol import CHECKPOINT, Fixtures, digest, isolated_rng, run_lock, sample_identity
from .cvae_training import check_source_identity, data_identity, make_dataset, provenance, source_info
from .models import HierarchicalStandardCVAETransformer, build_model
from .posterior_direct_output import assert_output_isolated
from .posterior_hierarchical_standard_cvae import load_checkpoint
from .posterior_h50_action_replay import _write_npz, _write_trajectory
from .posterior_h50_action_replay_exact_init import (
    _first_threshold_crossings, _load_replay, _per_frame_errors,
    _raw_from_processed, _trajectory_metrics, _trajectory_region, _write_error_svg,
    _write_exact_initialization, rotate_body_to_world,
)
from .posterior_t64_protocol import (
    DIAGNOSTIC_MASK_NAMES,
    PHYSICAL_MASK_NAMES,
    make_diagnostic_masks,
    make_physical_masks,
)
from .state_mask_eval import reconstruct_root_trajectory
from .util import atomic_write_json, atomic_write_text, file_sha256, load_json

VERSION = "65-token-replay-v2"
CHUNK = 64
DEFAULT_HORIZON = 192
# Closed-loop capture vs open-loop replay of the captured raw Actions.  Anything
# above float32 noise means the simulator itself is not reproducible.
DETERMINISM_LIMITS = {"joint_pos": 1e-4, "root_pos": 1e-4, "body_pos": 1e-4}
DETERMINISM_FIELDS = ("joint_pos", "joint_vel", "root_pos", "root_quat", "body_pos", "raw_action")
THRESHOLDS = {"joint_position_rmse_rad": .02, "root_position_rmse_m": .05,
              "root_orientation_max_deg": 5., "body_mpjpe_m": .05, "foot_contact_accuracy": .95}
# These are deliberately relative to the same-window original Action baseline;
# they are not a substitute for the baseline validity gate.  A replay can be
# numerically complete while remaining physically unassessable when the source
# environment itself cannot be reproduced.
MODEL_QUALITY_RATIO = 1.2
MODEL_QUALITY_METRICS = ("joint_position_rmse_rad", "root_position_rmse_m", "body_mpjpe_m")
REPRESENTATIVES = ("state_gap_16", "action_gap_16", "full_action", "joint_gap_8")
CONTEXT_FIELDS = ("runtime_default_joint_pos", "action_offset", "joint_position_limits",
    "joint_velocity_limits", "joint_effort_limits", "joint_stiffness", "joint_damping",
    "joint_armature", "joint_friction", "body_mass", "body_inertia", "body_com",
    "body_material", "ground_material")
ROOT = Path(__file__).resolve().parents[3]
KIT = ROOT / "sonic-repro-kit"


def finite(name, value, shape=None):
    value = np.asarray(value)
    if (shape is not None and value.shape != shape) or not np.isfinite(value).all():
        raise ValueError(f"invalid {name}: shape={value.shape}, expected={shape}, finite required")
    return value


def raw_to_target(raw, scale, offset, clip, wrapper):
    raw = finite("raw Action", raw)
    if wrapper is not None and wrapper > 0:
        raw = np.clip(raw, -wrapper, wrapper)
    target = raw * scale + offset
    return np.clip(target, clip[:, 0], clip[:, 1]) if clip is not None else target


def execution_actions(original_raw, original_canonical, predicted, mask, nominal, scale, offset,
                      clip=None, wrapper=None, *, full_prediction=False):
    """Preserve recorded raw bits outside the requested mask, including clipping aliases."""
    shape = np.shape(original_raw)
    if len(shape) != 2 or shape[1] != 29 or shape[0] < CHUNK or shape[0] % CHUNK:
        raise ValueError(f"Action windows must be whole [64k,29] chunks, found {shape}")
    for name, value in (("raw", original_raw), ("truth", original_canonical), ("prediction", predicted)):
        finite(name, value, shape)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"Action mask must be {list(shape)}")
    baseline = raw_to_target(original_raw, scale, offset, clip, wrapper) - nominal
    if not np.allclose(baseline, original_canonical, atol=2e-6, rtol=1e-5):
        raise ValueError("recorded raw/processed/canonical Action mapping mismatch")
    replace = np.ones_like(mask) if full_prediction else mask
    requested = np.where(replace, predicted, original_canonical)
    raw, _, saturated = relative_to_raw_action(requested, nominal, scale, offset, clip, wrapper)
    raw = np.where(replace, raw, original_raw).astype(np.float32)
    achieved = raw_to_target(raw, scale, offset, clip, wrapper) - nominal
    if not np.array_equal(raw[~replace], original_raw[~replace]):
        raise RuntimeError("visible raw Action changed")
    return raw, achieved, {"replaced_elements": int(replace.sum()),
        "saturated_elements": int((saturated & replace).sum()),
        "visible_raw_bitwise_equal": True,
        "requested_vs_achieved_max_rad": float(np.abs(achieved - requested).max()),
        "recorded_mapping_max_rad": float(np.abs(baseline - original_canonical).max())}


def predict(model, cpu, route, mask_seed, sample_seed, sample_index=0, mask_name=None):
    """Only truth returned to the evaluator; B/C model inputs have hidden values zeroed."""
    device = next(model.parameters()).device
    if mask_name in DIAGNOSTIC_MASK_NAMES:
        sm, am = make_diagnostic_masks(cpu, mask_name)
        names = [mask_name]
    else:
        sm, am, names = make_physical_masks(cpu, mask_seed)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cpu.items()}
    sm, am = sm.to(device), am.to(device)
    if route != "A":
        batch = dict(batch)
        batch["physical_state"] = batch["physical_state"].masked_fill(sm, 0.)
        batch["action"] = batch["action"].masked_fill(am, 0.)
    with isolated_rng(), torch.inference_mode():
        model.eval()
        eps = epsilon_for(model, cpu, sample_index, sample_seed, device) if route == "C" else None
        out = route_output(model, batch, sm, am, "standard_normal" if route == "C" else route, eps)
    state = out.physical_state[0].detach().cpu().numpy().copy()
    # Contacts are Bernoulli logits, not normalized continuous output.
    state[... , 68:] = out.state_contact_logits[0].sigmoid().cpu().numpy()
    action = out.action[0].detach().cpu().numpy().copy()
    finite("prediction State", state, (65, 70)); finite("prediction Action", action, (64, 29))
    return state, action, sm[0].cpu().numpy(), am[0].cpu().numpy(), names[0], eps


def offline_metrics(truth_s, truth_a, pred_s, pred_a, sm, am):
    result = {}
    for scope in ("full", "masked", "visible"):
        result[scope] = {}
        for domain, truth, pred, mask in (("state", truth_s[:, :68], pred_s[:, :68], sm[:, :68]),
                                          ("action", truth_a, pred_a, am)):
            use = np.ones_like(mask) if scope == "full" else mask if scope == "masked" else ~mask
            result[scope][domain] = stats((pred - truth)[use])
        use = np.ones_like(sm[:,68:]) if scope == "full" else sm[:,68:] if scope == "masked" else ~sm[:,68:]
        target = truth_s[:,68:][use] >= .5
        probability = np.clip(pred_s[:,68:][use],1e-7,1-1e-7)
        predicted = probability >= .5
        result[scope]["contact"] = {"count":int(use.sum()),
            "bce":float(np.mean(-target.astype(float)*np.log(probability)-(~target).astype(float)*np.log1p(-probability))) if target.size else None,
            "accuracy":float(np.mean(predicted==target)) if target.size else None,
            "tp":int((predicted & target).sum()),"fp":int((predicted & ~target).sum()),
            "tn":int((~predicted & ~target).sum()),"fn":int((~predicted & target).sum())}
    return result


def tail_coordinates(truth, prediction, mean, std, start, state=False):
    from .cvae_diagnostics import STATE_NAMES, UNITS
    count = 68 if state else 29
    errors = ((prediction-truth)/std)[:,:count]
    order = np.argsort(-np.abs(errors).reshape(-1),kind="stable")[:100]
    result = []
    for flat in order:
        t,f = np.unravel_index(flat,errors.shape)
        result.append({"relative_frame":int(t),"episode_frame":int(start+t),"feature_index":int(f),
            "feature_name":STATE_NAMES[f] if state else f"action_{f}","unit":UNITS[f] if state else "rad",
            "signed_normalized_error":float(errors[t,f]),"absolute_normalized_error":float(abs(errors[t,f])),
            "prediction_physical":float(prediction[t,f]),"truth_physical":float(truth[t,f]),
            "signed_physical_error":float(prediction[t,f]-truth[t,f]),"mean":float(mean[f]),"std":float(std[f])})
    return result


def physical_groups(truth, pred):
    return {name: stats((pred[:, a:b] - truth[:, a:b]).reshape(-1)) for name, a, b in (
        ("joint_position_rad", 0, 29), ("joint_velocity_rad_s", 29, 58),
        ("base_linear_velocity_m_s", 58, 61), ("base_angular_velocity_rad_s", 61, 64),
        ("gravity_unitless", 64, 67), ("height_m", 67, 68))}


def resolve_window_start(requested, steps, horizon):
    """Keep the requested chunk start unless the horizon would run past the episode."""
    if horizon < CHUNK or horizon % CHUNK:
        raise ValueError("horizon must be a positive multiple of 64 transitions")
    requested, steps = int(requested), int(steps)
    if requested + horizon <= steps:
        return requested, None
    start = (steps - horizon) // CHUNK * CHUNK
    if start < 0:
        raise ValueError(f"episode has {steps} transitions; horizon {horizon} does not fit")
    return start, f"requested start {requested} + {horizon} exceeds episode steps {steps}; shifted to {start}"


def chunk_indices(dataset, index, start, chunks):
    """Dataset windows that tile [start, start+64*chunks) on the same episode."""
    episode = dataset.refs[index].episode_index
    lookup = {int(ref.fixed_start): i for i, ref in enumerate(dataset.refs)
              if ref.episode_index == episode and ref.fixed_start is not None}
    wanted = [int(start) + CHUNK * k for k in range(chunks)]
    missing = [value for value in wanted if value not in lookup]
    if missing:
        raise ValueError(f"no fixed T64 dataset window at episode starts {missing}; stride must divide 64")
    return [lookup[value] for value in wanted]


def load_source(dataset, index, horizon=CHUNK, start=None):
    """Only the requested slice plus its preceding Action; never copy an entire HDF5."""
    import h5py
    from .physics_schema import load_physics_schema, read_physics_states, resolve_parameter
    from .action_mask_eval import _resolve_motion_file
    ref = dataset.refs[index]
    record = dataset.episodes[ref.episode_index]
    start = int(ref.fixed_start if start is None else start)
    horizon = int(horizon)
    if start < 0 or horizon < 1 or start + horizon > int(record["steps"]):
        raise ValueError(f"replay requires {horizon} complete transitions from frame {start}")
    stop = start + horizon
    schema = load_physics_schema(Path(record["schema_path"]))
    sim = schema["simulation"]
    terrain = schema.get("terrain") or {}
    missing_terrain_contract = [
        name
        for name in ("friction_combine_mode", "restitution_combine_mode")
        if terrain.get(name) in (None, "")
    ]
    if missing_terrain_contract:
        raise ValueError(
            "replay65 requires recorded terrain combine modes; missing "
            + ", ".join(missing_terrain_contract)
        )
    if not (np.isclose(sim["sim_dt"], .005) and np.isclose(sim["control_dt"], .02) and int(sim["decimation"]) == 4):
        raise ValueError("replay65 requires recorded 50 Hz, decimation=4")
    if len(schema["joint_names"]) != 29 or len(set(schema["joint_names"])) != 29:
        raise ValueError("29 unique ordered joint names required")
    env_id = int(record["env_id"])
    nominal = resolve_parameter(schema["nominal_default_joint_pos"], env_id).reshape(29)
    scale = resolve_parameter(schema["action_scale"], env_id).reshape(29)
    clip = None if schema.get("action_clip") is None else resolve_parameter(schema["action_clip"], env_id).reshape(29, 2)
    wrapper = schema.get("wrapper_action_clip")
    with h5py.File(record["hdf5_path"], "r") as stream:
        ep = stream[f"data/{record['episode']}"]
        context = stream[f"contexts/{record['context_id']}"]
        ctx = {name: np.asarray(context[name], dtype=np.float32) for name in CONTEXT_FIELDS}
        states = read_physics_states(ep["states"], start, stop + 1)
        canonical = np.asarray(ep["actions/action_target_canonical"][start:stop], dtype=np.float32)
        processed = np.asarray(ep["actions/processed_joint_target_abs"][start:stop], dtype=np.float32)
        if "raw_policy_action" in ep["actions"]:
            raw = np.asarray(ep["actions/raw_policy_action"][start:stop], dtype=np.float32)
            raw_source = "recorded_raw_policy_action"
        else:
            raw, achieved, _ = relative_to_raw_action(canonical, nominal, scale, ctx["action_offset"], clip, wrapper)
            if not np.allclose(achieved, canonical, atol=2e-6, rtol=1e-5):
                raise ValueError("recorded processed targets are not invertible")
            raw_source = "verified_inverse_mapping_raw_missing"
        pos = np.asarray(ep["replay/root_pos_w"][start:stop+1], dtype=np.float32)
        quat = np.asarray(ep["replay/root_quat_w"][start:stop+1], dtype=np.float32)
        body = np.asarray(ep["replay/body_pos_w"][start:stop+1], dtype=np.float32)
        previous = np.asarray(ep["actions/processed_joint_target_abs"][start-1], dtype=np.float32) if start else np.asarray(ep["actions/initial_processed_target_canonical"], dtype=np.float32) + nominal
        previous_raw = np.asarray(ep["actions/raw_policy_action"][start-1], dtype=np.float32) if start and "raw_policy_action" in ep["actions"] else _raw_from_processed(previous, scale, ctx["action_offset"], clip, wrapper)
    for name, value, shape in (("State", states, (horizon+1,70)), ("Action", canonical, (horizon,29)),
                               ("raw", raw, (horizon,29)), ("root", pos, (horizon+1,3)), ("quaternion", quat, (horizon+1,4))):
        finite(name, value, shape)
    if not np.allclose(processed, canonical + nominal, atol=2e-6, rtol=1e-5):
        raise ValueError("recorded absolute/canonical targets disagree")
    translation = np.array([pos[0,0], pos[0,1], 0], dtype=np.float32)
    pos, body = pos - translation, body - translation[None, None]
    lin = np.stack([rotate_body_to_world(q, v) for q, v in zip(quat, states[:,58:61])])
    ang = np.stack([rotate_body_to_world(q, v) for q, v in zip(quat, states[:,61:64])])
    source = dict(physics_state_v3=states, physical_state=states, joint_pos=states[:,:29]+nominal,
        joint_vel=states[:,29:58], root_pos=pos, root_quat=quat, body_pos=body, root_lin_vel=lin,
        root_ang_vel=ang, raw_action=raw, processed_action=processed, action_target_canonical=canonical,
        joint_names=np.asarray(schema["joint_names"]))
    init = dict(ctx, joint_names=np.asarray(schema["joint_names"]), joint_pos=source["joint_pos"][0],
        joint_vel=states[0,29:58], root_pos_relative=pos[0], root_quat_wxyz=quat[0],
        root_lin_vel_world=lin[0], root_ang_vel_world=ang[0], body_pos_relative=body[0],
        physics_state_v3=states[0], nominal_default_joint_pos=nominal, action_scale=scale,
        action_clip=np.empty((0,), dtype=np.float32) if clip is None else clip,
        wrapper_action_clip=np.float32(np.nan if wrapper is None else wrapper), previous_raw_action=previous_raw,
        previous_processed_action=previous, initial_joint_target_abs=previous,
        replay65_contract=np.asarray(json.dumps({"version": VERSION, "simulation": sim,
            "actuators": schema.get("actuator_groups", "unknown"), "active_events": schema.get("active_events", "unknown"),
            "asset":schema.get("asset"),"contact":schema.get("contact"),
            "ground_material": {
                "friction_combine_mode": terrain["friction_combine_mode"],
                "restitution_combine_mode": terrain["restitution_combine_mode"],
            },
            "window_start": start, "horizon_steps": horizon, "hidden_simulator_state": "not_recorded"})))
    motion, motion_info = _resolve_motion_file(record)
    return source, init, schema, {"record": record, "raw_source": raw_source, "motion_file": str(motion),
        "motion_file_sha256": motion_info["sha256"], "removed_world_xy": translation.tolist()}


def progress(run, status, **kwargs):
    row = {"protocol_version": VERSION, "status": status, "pid": os.getpid(),
           "updated_at": datetime.now(timezone.utc).isoformat(), **kwargs}
    atomic_write_json(run / "manifests/progress.json", row)
    print(json.dumps({"run": str(run), **row}, ensure_ascii=False), flush=True)


def stitch_states(parts):
    """Chunk k>0 drops its first State: the shared boundary belongs to chunk k-1."""
    return np.concatenate([parts[0], *[part[1:] for part in parts[1:]]], axis=0)


def state_only_continuation(args, chunk):
    """Return whether a chunk must be inferred from State inputs only.

    The first 64-transition chunk is the selected checkpoint window and keeps
    the historical replay contract.  With ``--state-only-continuation``, every
    later chunk reuses that one dataset window as a model-input template while
    its source State is supplied from the captured/recorded trajectory.  The
    target Action is retained only for scoring and physical replay; it is
    replaced by zeros before the model call, as is the preceding Action
    context.  This keeps the single-window comparison explicit and prevents a
    missing stride-64 HDF5 window from becoming an implicit input.
    """
    return bool(getattr(args, "state_only_continuation", False)) and int(chunk) > 0


def mapping_args(init):
    return (init["nominal_default_joint_pos"], init["action_scale"], init["action_offset"],
            init["action_clip"] if init["action_clip"].size else None,
            None if np.isnan(init["wrapper_action_clip"]) else float(init["wrapper_action_clip"]))


def load_capture(capture_run):
    """Sealed closed-loop source: frame 0 .. window_start + horizon."""
    capture_run = Path(capture_run).resolve()
    complete = verify_capture(capture_run)
    meta = load_json(capture_run / "manifests/capture.json")
    path = capture_run / "data/source/000000.replay.npz"
    full = _load_replay(path)
    with np.load(path, allow_pickle=False) as archive:
        processed = np.asarray(archive["processed_action"], dtype=np.float32)
    with np.load(capture_run / "data/exact_initialization.npz", allow_pickle=False) as archive:
        init = {name: archive[name].copy() for name in archive.files}
    return {"run": capture_run, "meta": meta, "complete": complete, "full": full,
            "canonical": processed - init["nominal_default_joint_pos"][None], "init": init}


def verify_capture(capture_run):
    capture_run = Path(capture_run).resolve()
    path = capture_run / "manifests/capture_complete.json"
    if not path.is_file():
        raise FileNotFoundError(f"capture is incomplete (no {path}); rerun replay65 capture")
    complete = load_json(path)
    for relative, expected in complete["hashes"].items():
        target = (capture_run / relative).resolve()
        if not target.is_relative_to(capture_run) or file_sha256(target) != expected:
            raise ValueError(f"capture artifact changed: {relative}")
    return complete


def prepare_window(args, dataset, fixtures, model, index, run, norm, rows, capture=None):
    row = next(r for r in rows if r["window_index"] == index)
    by_index = {r["window_index"]: r for r in rows}
    horizon = int(args.horizon_steps)
    chunks = horizon // CHUNK
    for folder in ("manifests", "data", "plots", "videos", "logs", "markers"):
        (run / folder).mkdir(parents=True, exist_ok=True)
    if capture is None:
        steps = int(dataset.episodes[dataset.refs[index].episode_index]["steps"])
        start, shift = resolve_window_start(row["window_start"], steps, horizon)
        full, init, schema, source_meta = load_source(dataset, index, horizon, start)
        offset, seed = 0, args.simulation_seed
        truth_a, prefix = full["action_target_canonical"], np.zeros((0, 29), np.float32)
        _write_npz(run / "data/source.replay.npz", **full)
        _write_trajectory(run / "data/source.trajectory.pkl", joint_pos=full["joint_pos"],
            root_pos=full["root_pos"], root_quat=full["root_quat"], fps=50.)
        _, init_manifest = _write_exact_initialization(run / "data/exact_initialization.npz", init)
        atomic_write_json(run / "manifests/source_schema.json", schema)
        source_meta.update(mode="dataset", training_seen=True, deterministic=None)
        before = None
    else:
        meta = capture["meta"]
        if meta["window"]["stable_window_id"] != row["stable_window_id"]:
            raise ValueError("capture run was recorded for a different dataset window")
        if int(meta["horizon_steps"]) != horizon:
            raise ValueError("--horizon-steps must equal the capture horizon")
        start, shift = int(meta["window_start"]), meta.get("window_start_shift")
        offset, seed, init = start, int(meta["simulation_seed"]), capture["init"]
        full = capture["full"]
        truth_a = capture["canonical"][start:start + horizon]
        prefix = full["raw_action"][:start]
        source_dir = capture["run"] / "data/source"
        shutil.copy2(source_dir / "000000.replay.npz", run / "data/source.replay.npz")
        shutil.copy2(source_dir / "000000.trajectory.pkl", run / "data/source.trajectory.pkl")
        shutil.copy2(capture["run"] / "data/exact_initialization.npz", run / "data/exact_initialization.npz")
        shutil.copy2(capture["run"] / "manifests/source_schema.json", run / "manifests/source_schema.json")
        init_manifest = dict(meta["initialization"], path=str(run / "data/exact_initialization.npz"))
        # The capture's original-Action determinism replay is the baseline; reuse it bit for bit.
        shutil.copytree(capture["run"] / "simulations/baseline", run / "simulations/baseline",
                        symlinks=True, ignore=shutil.ignore_patterns("checkpoints", "replay_motion"))
        nominal = init["nominal_default_joint_pos"]
        before = [capture["canonical"][start + CHUNK * k - 1] if start + CHUNK * k > 0
                  else init["previous_processed_action"] - nominal for k in range(chunks)]
        source_meta = {"mode": "recapture", "capture_run": str(capture["run"]),
            "capture_complete_sha256": file_sha256(capture["run"] / "manifests/capture_complete.json"),
            "motion_file": meta["motion_file"], "motion_file_sha256": meta["motion_file_sha256"],
            "deterministic": bool(capture["complete"]["deterministic"]), "training_seen": False,
            "raw_source": "closed_loop_capture"}
    atomic_write_json(run / "manifests/exact_initialization.json", init_manifest)
    nominal = init["nominal_default_joint_pos"]
    truth_s = full["physical_state"][offset:offset + horizon + 1]
    finite("source State window", truth_s, (horizon + 1, 70))
    finite("source Action window", truth_a, (horizon, 29))
    s_mean, s_std = norm["state"]
    a_mean, a_std = norm["action"]
    single_window = bool(getattr(args, "state_only_continuation", False))
    # In single-window continuation mode all chunks use the selected window's
    # mask/fixture identity.  The later source States come from the recorded
    # trajectory, so no consecutive T64 dataset windows are required.
    chunk_ids = ([index] * chunks if single_window else
                 chunk_indices(dataset, index, start, chunks))
    entries = []
    diagnostic_only = any(name in DIAGNOSTIC_MASK_NAMES for name in args.masks)
    mask_names = ([name for name in args.masks if name in DIAGNOSTIC_MASK_NAMES]
                  if diagnostic_only else list(PHYSICAL_MASK_NAMES))
    for slot, name in enumerate(mask_names):
        parts = {key: [] for key in ("ps", "pa", "sm", "am", "truth_n", "truth_an", "eps_g", "eps_l")}
        chunk_rows = []
        for k, ci in enumerate(chunk_ids):
            sample = dict(dataset[ci])
            chunk_row = by_index.get(ci) or {**sample_identity(sample), "window_index": ci,
                                             "stable_window_id": digest(sample_identity(sample))}
            sample.update(stable_window_id=chunk_row["stable_window_id"], window_index=ci, mask_slot=slot,
                          fixture_index=fixtures.indices.index(ci) * 8 + slot if ci in fixtures.indices else -1)
            lo = CHUNK * k
            truth_n = ((truth_s[lo:lo + CHUNK + 1] - s_mean) / s_std).astype(np.float32)
            truth_an = ((truth_a[lo:lo + CHUNK] - a_mean) / a_std).astype(np.float32)
            continuation_state_only = state_only_continuation(args, k)
            if capture is None and not single_window:
                # Compare HDF targets with the normalized dataset before interpreting predictions.
                if not np.allclose(sample["physical_state"].numpy()[:, :68], truth_n[:, :68], atol=2e-5, rtol=1e-5):
                    raise ValueError(f"dataset/HDF State window mismatch at chunk {k}")
                if not np.allclose(sample["action"].numpy(), truth_an, atol=2e-5, rtol=1e-5):
                    raise ValueError(f"dataset/HDF Action window mismatch at chunk {k}")
            elif capture is not None or single_window:
                # Identity stays the dataset window's (masks/epsilon); tensors come from the capture.
                sample["physical_state"] = torch.from_numpy(truth_n)
                sample["action"] = torch.from_numpy(
                    np.zeros_like(truth_an) if continuation_state_only else truth_an
                )
                if capture is not None:
                    previous = before[k]
                    sample["action_before_window"] = torch.from_numpy(
                        np.zeros_like(previous) if continuation_state_only else
                        ((previous - a_mean) / a_std).astype(np.float32)
                    )
                elif continuation_state_only:
                    sample["action_before_window"] = torch.zeros_like(sample["action_before_window"])
            cpu = default_collate([sample])
            ps, pa, sm, am, actual_name, eps = predict(
                model, cpu, args.route, args.mask_seed, args.sample_seed,
                args.sample_index, mask_name=name,
            )
            assert actual_name == name
            for key, value in (("ps", ps), ("pa", pa), ("sm", sm), ("am", am),
                               ("truth_n", truth_n.copy()), ("truth_an", truth_an.copy())):
                parts[key].append(value)
            if eps is not None:
                parts["eps_g"].append(eps[0][0].cpu().numpy()); parts["eps_l"].append(eps[1][0].cpu().numpy())
            chunk_rows.append({"chunk": k, "window_index": ci, "stable_window_id": chunk_row["stable_window_id"],
                               "episode_start": start + lo,
                               "input_contract": "state_only" if continuation_state_only else "selected_window"})
        ps, sm, truth_n = (stitch_states(parts[key]) for key in ("ps", "sm", "truth_n"))
        pa, am, truth_an = (np.concatenate(parts[key], axis=0) for key in ("pa", "am", "truth_an"))
        # Adjacent chunks both predict the shared boundary State; report their disagreement.
        boundary = [float(np.abs((parts["ps"][k][-1, :68] - parts["ps"][k + 1][0, :68]) * s_std[:68]).max())
                    for k in range(chunks - 1)]
        state = ps * s_std + s_mean
        state[:, 68:] = ps[:, 68:]
        action = pa * a_std + a_mean
        completed_s = state if args.route == "A" else np.where(sm, state, truth_s)
        completed_a = action if args.route == "A" else np.where(am, action, truth_a)
        raw_window, achieved, mapping = execution_actions(full["raw_action"][offset:offset + horizon], truth_a,
            action, am, *mapping_args(init),
            full_prediction=args.route == "A" or args.action_mode == "full-prediction")
        executed_raw = np.concatenate([prefix, raw_window], axis=0).astype(np.float32)
        metrics = offline_metrics(truth_n, truth_an, ps, pa, sm, am)
        chunk_metrics = [offline_metrics(parts["truth_n"][k], parts["truth_an"][k], parts["ps"][k],
                                         parts["pa"][k], parts["sm"][k], parts["am"][k]) for k in range(chunks)]
        directory = run / "data" / name
        directory.mkdir()
        arrays = dict(truth_state=truth_s, truth_action=truth_a, predicted_state=state, predicted_action=action,
            predicted_state_normalized=ps, predicted_action_normalized=pa, completed_state=completed_s,
            completed_action=completed_a, state_mask=sm, action_mask=am, executed_raw=executed_raw,
            window_executed_raw=raw_window, achieved_action=achieved,
            valid_state=np.ones(horizon + 1, bool), valid_action=np.ones(horizon, bool))
        if parts["eps_g"]:
            arrays.update(epsilon_global=np.stack(parts["eps_g"]), epsilon_local=np.stack(parts["eps_l"]))
        _write_npz(directory / "prediction.npz", **arrays)
        root_pos = full["root_pos"][offset:offset + horizon + 1]
        root_quat = full["root_quat"][offset:offset + horizon + 1]
        truth_pos, truth_quat = reconstruct_root_trajectory(truth_s, root_pos, root_quat, 0, .02)
        pred_pos, pred_quat = reconstruct_root_trajectory(completed_s, root_pos, root_quat, 0, .02)
        for tag, st, pos, quat in (("truth_integrated",truth_s,truth_pos,truth_quat), ("predicted_integrated",completed_s,pred_pos,pred_quat)):
            _write_trajectory(directory / f"{tag}.trajectory.pkl", joint_pos=st[:,:29]+nominal, root_pos=pos, root_quat=quat, fps=50.)
        # Render every requested representative State view for every route.
        # A is full prediction, so its Action trajectory is identical for all
        # Mask labels; keep one physical Action replay there and avoid four
        # duplicate Isaac jobs while retaining all four State MP4s.
        representative = name in args.masks
        action_model_replay = representative and mapping["replaced_elements"] > 0 \
            and (args.route != "A" or name == args.masks[0])
        entries.append({"mask": name, "fixture_id": digest([row["stable_window_id"], slot, args.mask_seed]),
            "offline": metrics, "physical_groups": physical_groups(truth_s,state), "mapping": mapping,
            "chunks": [dict(c, offline=m) for c, m in zip(chunk_rows, chunk_metrics)],
            "chunk_boundary_state_max_abs": boundary,
            "state_top100":tail_coordinates(truth_s,state,s_mean,s_std,start,state=True),
            "action_top100":tail_coordinates(truth_a,action,a_mean,a_std,start),
            "representative": representative, "action_model_replay": action_model_replay,
            "hidden_initial_state": bool(sm[0].any()), "simulation_initial_state": "recorded_truth_not_model_input",
            "visualization": {"gravity_normalized": True, "contact_not_used_as_pose_constraint": True,
                "truth_integration_root_rmse_m": float(np.sqrt(np.mean((truth_pos-root_pos)**2))),
                "predicted_gravity_norm_min": float(np.linalg.norm(completed_s[:,64:67],axis=-1).min()),
                "predicted_height_anchor_offset_m": float(root_pos[0,2]-completed_s[0,67]),
                "anchor": "shared_recorded_root_pose_frame_zero_only"}})
    manifest = {"version": VERSION, "window": row, "route": args.route, "action_mode": args.action_mode,
        "mask_seed": args.mask_seed, "sample_seed": args.sample_seed, "sample_index": args.sample_index,
        "simulation_seed": seed, "source": source_meta, "source_mode": source_meta["mode"],
        "training_seen": source_meta["training_seen"], "horizon_steps": horizon, "chunk_count": chunks,
        "chunk_windows": chunk_rows, "window_start": start, "requested_window_start": int(row["window_start"]),
        "window_start_shift": shift, "score_offset": offset, "replay_steps": offset + horizon,
        "single_window_continuation": single_window,
        "continuation_input_contract": "state_only_after_first_chunk" if single_window else "dataset_window_each_chunk",
        "entries": entries,
        "initialization": init_manifest, "baseline_thresholds": THRESHOLDS,
        "model_quality": {"ratio_threshold": MODEL_QUALITY_RATIO, "metrics": MODEL_QUALITY_METRICS},
        "quality_pass": None,
        "limitations": ["solver/contact warm-start state not recorded", "actual historical delay/queues may be unknown",
                         "kinematic State rendering is not a dynamics validation"]}
    atomic_write_json(run / "manifests/replay65.json", manifest)
    return manifest


def prepare(args):
    run, checkpoint_path, dataset_run = args.output_run.resolve(), args.checkpoint.resolve(), args.dataset_run.resolve()
    assert_output_isolated(run, [dataset_run, checkpoint_path.parent.parent, ROOT / "state-action-cvae", KIT])
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("prepare requires a new empty output run")
    checkpoint_hash = file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_signature",{}).get("architecture_version") != HierarchicalStandardCVAETransformer.ARCHITECTURE_VERSION:
        raise ValueError("architecture signature mismatch: replay65 does not migrate old H50 checkpoints")
    config = checkpoint_config(checkpoint)
    training_mask_seed = checkpoint.get("training_contract", {}).get("training_mask_seed") if checkpoint.get("training_contract") else None
    if args.mask_seed is None:
        if args.route != "A" and training_mask_seed is None:
            raise ValueError("checkpoint has no fixture seed; specify --mask-seed explicitly")
        args.mask_seed = int(training_mask_seed if training_mask_seed is not None else 20260920)
    if args.route in {"B","C"} and (checkpoint.get("format_version") != CHECKPOINT or checkpoint.get("stage") != args.route):
        raise ValueError("B/C replay requires a v2 checkpoint of that stage; legacy B is not condition-only")
    if args.route == "A" and checkpoint.get("stage") != "A":
        raise ValueError("A replay requires a Stage-A checkpoint")
    if args.route == "A":
        args.action_mode = "full-prediction"
    capture_run = getattr(args, "capture_run", None)
    args.source = getattr(args, "source", None) or ("recapture" if capture_run else "dataset")
    capture = None
    if args.source == "recapture":
        if capture_run is None:
            raise ValueError("--source recapture requires --capture-run")
        capture = load_capture(capture_run)
        captured = int(capture["meta"]["window"]["window_index"])
        if args.window_index is not None and args.window_index != captured:
            raise ValueError(f"--window-index {args.window_index} differs from the captured window {captured}")
        args.window_index = captured
        if getattr(args, "horizon_steps", None) is None:
            args.horizon_steps = int(capture["meta"]["horizon_steps"])
    elif capture_run is not None:
        raise ValueError("--capture-run is only valid with --source recapture")
    if getattr(args, "horizon_steps", None) is None:
        args.horizon_steps = DEFAULT_HORIZON
    if args.horizon_steps < CHUNK or args.horizon_steps % CHUNK:
        raise ValueError("--horizon-steps must be a positive multiple of 64")
    if args.window_index is None and getattr(args, "selection", None) is None:
        raise ValueError("specify --window-index, --selection first-suite or --capture-run")
    model = build_model(config["model"])
    load_checkpoint(model, checkpoint_path)
    model.to(args.device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
    dataset, indices = make_dataset(dataset_run, config)
    try:
        fixtures = Fixtures(dataset, indices, expand=True)
        rows = fixtures.manifest()
        identity = data_identity(dataset_run, rows)
        identity_check = check_source_identity(
            checkpoint,
            identity,
            allow_unknown=args.route == "A",
            allow_manifest_mismatch=getattr(args, "allow_recovered_dataset", False),
            allow_identity_mismatch=getattr(args, "allow_recovered_identity", False),
        )
        norm = read_normalization(dataset_run / "data/normalization.npz")
        run.mkdir(parents=True, exist_ok=True)
        progress(run, "preparing")
        atomic_write_json(run / "manifests/selected_windows.json", rows)
        if args.window_index is not None:
            selected = [args.window_index]
            if args.window_index not in indices:
                raise ValueError("window index is not in checkpoint's selected training windows")
            selection = "explicit_checkpoint_seen_window"
        else:
            # Predefined representative selection; no manual cherry-picking after viewing videos.
            selected = []
            for predicate in (lambda r:r["window_start"] == 0, lambda r:r["window_start"] > 0):
                match = next((r["window_index"] for r in rows if predicate(r)), None)
                if match is not None:
                    selected.append(match)
            scores = []
            for i, index in enumerate(indices):
                mse = []
                for slot in range(8):
                    cpu = default_collate([fixtures[i*8+slot]])
                    ps,pa,sm,am,_,_ = predict(model,cpu,args.route,args.mask_seed,args.sample_seed,args.sample_index)
                    errors = np.concatenate(((ps[:,:68]-cpu["physical_state"][0].numpy()[:,:68])[sm[:,:68]],
                                             (pa-cpu["action"][0].numpy())[am]))
                    mse.append(float(np.mean(errors.astype(np.float64)**2)))
                scores.append({"window_index":index,"masked_continuous_mse":float(np.mean(mse))})
                progress(run,"selecting",windows_done=i+1,windows_total=len(indices))
            selected.append(max(scores,key=lambda r:r["masked_continuous_mse"])["window_index"])
            selected = list(dict.fromkeys(selected))
            atomic_write_json(run / "manifests/selection_scores.json",scores)
            selection = "first_episode_start_first_nonzero_start_worst_equal_family_masked_mse"
        paths = []
        for index in selected:
            child = run / "windows" / f"w{index:06d}"
            prepare_window(args,dataset,fixtures,model,index,child,norm,rows,capture)
            paths.append(str(child.relative_to(run)))
        if file_sha256(checkpoint_path) != checkpoint_hash:
            raise RuntimeError("source checkpoint changed during preparation; use a completed fixed checkpoint and a new run")
        (run / "data").mkdir(exist_ok=True)
        shutil.copy2(dataset_run / "data/normalization.npz",run / "data/normalization.npz")
        prov = provenance(config,vars_serializable(args),identity,source_info(checkpoint_path,checkpoint))
        replay_sources = [Path(__file__), *[Path(__file__).with_name(n) for n in (
            "action_masks.py","physics_schema.py","state_mask_eval.py","posterior_h50_action_replay.py","posterior_h50_action_replay_exact_init.py")],
            *[KIT/n for n in ("exact_replay_initializer.py","replay65_runtime.py","replay65_recorder.py",
                "render_replay65.py","render_replay65_compare.py","render_mujoco_trajectory.py",
                "render_h50a_exact_action_replays.py","sonic_repro.sh")]]
        for p in replay_sources:
            prov["source_hashes"][str(p.relative_to(ROOT))] = file_sha256(p)
        if capture is not None:
            prov["capture"] = {"run": str(capture["run"]),
                "capture_complete_sha256": file_sha256(capture["run"] / "manifests/capture_complete.json")}
        atomic_write_json(run / "manifests/provenance.json",prov)
        atomic_write_json(run / "manifests/replay65.json",{"version":VERSION,"windows":paths,
            "source_mode":args.source,"horizon_steps":int(args.horizon_steps),
            "single_window_continuation": bool(getattr(args, "state_only_continuation", False)),
            "capture_run":None if capture is None else str(capture["run"]),
            "selection":selection,"identity_check":identity_check,"source_checkpoint":source_info(checkpoint_path,checkpoint),
            "training_mask_seed":training_mask_seed,"replay_mask_seed":args.mask_seed,
            "exact_training_fixture": args.route == "B" and args.mask_seed == training_mask_seed and checkpoint.get("training_contract",{}).get("mask_mode")=="fixed",
            "recovered_dataset": bool(getattr(args, "allow_recovered_dataset", False)),
            "recovered_identity_override": bool(getattr(args, "allow_recovered_identity", False)),
        })
        # Seal all prepared input files. Reports and simulation outputs are intentionally not sealed here.
        hashes = {str(p.relative_to(run)):file_sha256(p) for p in run.rglob("*") if p.is_file() and p.name != "progress.json"}
        atomic_write_json(run / "manifests/prepared_hashes.json",hashes)
        progress(run,"prepared",windows=paths)
    finally:
        dataset.close()


def prepare_method_b(args):
    """Prepare a physical replay from an offline tail32/context-next32 eval.

    Method B predicts only masked target Action blocks.  The visible Action
    context is copied from the recorded source before the combined raw Action
    sequence is sent to the replay worker.  A prepared Method-A run is used as
    the immutable source/baseline template so initialization and simulator
    identity remain identical.
    """
    template = args.template_run.expanduser().resolve()
    offline = args.offline_run.expanduser().resolve()
    output = args.output_run.expanduser().resolve()
    label = str(args.label)
    if not label or "/" in label or "\\" in label or label in {"baseline", ".", ".."}:
        raise ValueError("invalid Method B replay label")
    verify_prepared(template, code=True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output run must be new and empty: {output}")
    offline_manifest = load_json(offline / "manifests" / "long_horizon_eval.json")
    group_ids = offline_manifest.get("group_ids") or []
    if len(group_ids) != 1:
        raise ValueError("Method B replay requires exactly one offline evaluation group")
    arrays = sorted((offline / "data").glob("*.npz"))
    if len(arrays) != 1:
        raise ValueError("Method B replay requires exactly one offline NPZ group output")
    with np.load(arrays[0], allow_pickle=False) as values:
        required = ("truth_action_raw", "action_b_raw", "truth_action_canonical",
                    "action_b_achieved", "mask_b", "truth_state", "truth_action")
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError("offline Method B output is missing: " + ", ".join(missing))
        truth_raw = np.asarray(values["truth_action_raw"], dtype=np.float32)
        predicted_raw = np.asarray(values["action_b_raw"], dtype=np.float32)
        truth_canonical = np.asarray(values["truth_action_canonical"], dtype=np.float32)
        predicted_canonical = np.asarray(values["action_b_achieved"], dtype=np.float32)
        action_mask = np.asarray(values["mask_b"], dtype=bool)
        truth_state = np.asarray(values["truth_state"], dtype=np.float32)
        truth_action = np.asarray(values["truth_action"], dtype=np.float32)
    if truth_raw.shape != (HORIZON, 29) or predicted_raw.shape != truth_raw.shape:
        raise ValueError("Method B raw Action output must have shape [192,29]")
    if action_mask.shape != truth_raw.shape:
        raise ValueError("Method B Action mask shape mismatch")
    if int(action_mask.sum()) == 0 or int(action_mask.sum()) >= int(action_mask.size):
        raise ValueError("Method B mask must contain both visible and hidden Action elements")
    executed_raw = np.where(action_mask, predicted_raw, truth_raw).astype(np.float32)
    completed_canonical = np.where(action_mask, predicted_canonical, truth_canonical).astype(np.float32)

    template_root = load_json(template / "manifests" / "replay65.json")
    if len(template_root.get("windows", [])) != 1:
        raise ValueError("Method B template must contain exactly one replay window")
    relative = Path(template_root["windows"][0])
    template_window = template / relative
    template_meta = load_json(template_window / "manifests" / "replay65.json")
    if int(template_meta.get("horizon_steps", CHUNK)) != HORIZON:
        raise ValueError("Method B template must use a 192-transition horizon")
    source = _load_replay(template_window / "data/source.replay.npz")
    if not np.allclose(source["raw_action"], truth_raw, atol=1e-6, rtol=0):
        raise ValueError("offline Method B group does not match template source Action")

    for folder in ("data", "manifests", "markers", "plots", "videos", "logs"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    (output / relative / "data" / label).mkdir(parents=True, exist_ok=True)
    (output / relative / "simulations").mkdir(parents=True, exist_ok=True)
    for name in ("source.replay.npz", "source.trajectory.pkl", "exact_initialization.npz"):
        shutil.copy2(template_window / "data" / name, output / relative / "data" / name)
    for name in ("source_schema.json", "exact_initialization.json"):
        shutil.copy2(template_window / "manifests" / name, output / relative / "manifests" / name)
    shutil.copytree(
        template_window / "simulations" / "baseline",
        output / relative / "simulations" / "baseline",
    )
    if (template / "data" / "normalization.npz").is_file():
        shutil.copy2(template / "data" / "normalization.npz", output / "data" / "normalization.npz")
    shutil.copy2(template / "manifests" / "provenance.json", output / "manifests" / "provenance.json")

    prediction_dir = output / relative / "data" / label
    _write_npz(
        prediction_dir / "prediction.npz",
        truth_state=truth_state,
        truth_action=truth_canonical,
        predicted_state=truth_state,
        predicted_action=completed_canonical,
        predicted_state_normalized=truth_state,
        predicted_action_normalized=truth_action,
        completed_state=truth_state,
        completed_action=completed_canonical,
        state_mask=np.zeros_like(truth_state, dtype=bool),
        action_mask=action_mask,
        executed_raw=executed_raw,
        window_executed_raw=executed_raw,
        achieved_action=completed_canonical,
        valid_state=np.ones((HORIZON + 1, 70), dtype=bool),
        valid_action=np.ones((HORIZON, 29), dtype=bool),
    )
    entry = {
        "mask": label,
        "fixture_id": digest([group_ids[0], label, offline_manifest.get("seed")]),
        "offline": {"method": "tail32_context_next32", "offline_run": str(offline),
                    "group_id": group_ids[0], "masked_action_elements": int(action_mask.sum())},
        "physical_groups": {}, "chunks": [], "chunk_boundary_state_max_abs": [],
        "state_top100": [], "action_top100": [], "representative": True,
        "action_model_replay": True, "hidden_initial_state": False,
        "simulation_initial_state": "recorded_truth_not_model_input",
        "visualization": {"method": "tail32_context_next32"},
    }
    window_meta = deepcopy(template_meta)
    window_meta["route"] = "C"
    window_meta["action_mode"] = "tail32_context_next32"
    window_meta["entries"] = [entry]
    window_meta["method_b_offline_run"] = str(offline)
    window_meta["method_b_group_id"] = group_ids[0]
    atomic_write_json(output / relative / "manifests" / "replay65.json", window_meta)

    root_meta = deepcopy(template_root)
    root_meta["windows"] = [str(relative)]
    root_meta["method"] = "tail32_context_next32"
    root_meta["method_b_offline_run"] = str(offline)
    root_meta["method_b_group_id"] = group_ids[0]
    root_meta["quality_pass"] = None
    atomic_write_json(output / "manifests" / "replay65.json", root_meta)
    atomic_write_text(output / "markers" / "replay65_method_b_prepared.ok", "PASS\n")
    hashes = {
        str(path.relative_to(output)): file_sha256(path)
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"progress.json", "prepared_hashes.json"}
    }
    atomic_write_json(output / "manifests" / "prepared_hashes.json", hashes)
    progress(output, "prepared_method_b", windows=[str(relative)], group_id=group_ids[0], label=label)
    print(json.dumps({"output_run": str(output), "group_id": group_ids[0], "label": label}, ensure_ascii=False))


def vars_serializable(args):
    return {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}


def checkpoint_config(checkpoint):
    if checkpoint.get("config"):
        return checkpoint["config"]
    # Historical 65-token A wrote only the structural signature, unlike v2.
    model = checkpoint.get("model_config",checkpoint.get("model_signature"))
    if not model:
        raise ValueError("checkpoint has no model configuration/signature")
    return {"model":model,"data":{"window_transitions":64,"stride":64,"max_episodes":256,"num_workers":0}}


def verify_prepared(run, *, code=False):
    for relative, expected in load_json(run / "manifests/prepared_hashes.json").items():
        path = (run / relative).resolve()
        if not path.is_relative_to(run.resolve()) or file_sha256(path) != expected:
            raise ValueError(f"prepared artifact changed: {relative}")
    if code:
        for relative, expected in load_json(run/"manifests/provenance.json")["source_hashes"].items():
            if file_sha256(ROOT/relative) != expected:
                raise ValueError(f"source changed since prepare: {relative}; prepare a new run")


def verify_worker(child):
    hashes=load_json(child/"manifests/replay65_worker_complete.json")["hashes"]
    if not hashes:
        raise ValueError("empty worker artifact seal")
    for relative,expected in hashes.items():
        path=(child/relative).resolve()
        if not path.is_relative_to(child.resolve()) or file_sha256(path)!=expected:
            raise ValueError(f"simulation artifact changed: {path}")


def launch(command, log, env=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a",encoding="utf-8") as output:
        output.write(json.dumps(command)+"\n"); output.flush()
        process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,
                                   start_new_session=os.name != "nt")
        try:
            for line in process.stdout:
                print(line,end="",flush=True); output.write(line); output.flush()
            code = process.wait()
        except BaseException:
            if os.name != "nt":
                os.killpg(process.pid,signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(process.pid,signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=10)
            raise
    if code:
        raise RuntimeError(f"worker exit {code}; see {log}")


def window_context(window):
    manifest = load_json(window / "manifests/replay65.json")
    return {"motion_file": manifest["source"]["motion_file"],
            "motion_file_sha256": manifest["source"]["motion_file_sha256"],
            "motion_key": manifest["window"]["motion_key"], "seed": int(manifest["simulation_seed"]),
            "init_path": window / "data/exact_initialization.npz", "initialization": manifest["initialization"],
            "steps": int(manifest.get("replay_steps", CHUNK))}


def simulate_group(window, tag, names, raws, context):
    child = window / "simulations" / tag
    if child.exists():
        completion = child / "manifests/replay65_worker_complete.json"
        if completion.is_file():
            verify_worker(child)
            return
        raise FileExistsError(f"incomplete/changed simulation is not overwritten: {child}; prepare a new run")
    steps = int(context["steps"])
    for raw in raws:
        finite("planned raw Action", raw, (steps, 29))
    # The shell worker writes exit codes and completion markers after Isaac
    # exits. Create the complete child layout before launching it; creating
    # only data leaves mark_stage unable to write action_mask_replay.ok.
    for folder in ("data", "manifests", "logs", "markers"):
        (child / folder).mkdir(parents=True, exist_ok=True)
    shutil.copy2(context["init_path"], child / "data/exact_initialization.npz")
    init = context["initialization"]
    actions = child / "data/raw_actions.npz"
    _write_npz(actions,raw_actions=np.stack(raws,axis=1),scenario_names=np.asarray(names))
    request = {"schema_version":VERSION,"representation":"physics_v4","replay65_guard":True,
        "motion_file":context["motion_file"],"motion_file_sha256":context["motion_file_sha256"],
        "motion_key":context["motion_key"],"raw_actions_file":str(actions),"raw_actions_sha256":file_sha256(actions),
        "steps":steps,"num_envs":len(names),"scenario_names":names,"control_dt":.02,
        "exact_initialization_file":str(child / "data/exact_initialization.npz"),
        "exact_initialization_file_sha256":init["file_sha256"],"exact_initialization_payload_sha256":init["payload_sha256"],
        "exact_initialization_report_paths":[str(child / f"manifests/exact_initialization_readback_{i:06d}.json") for i in range(len(names))]}
    atomic_write_json(child / "manifests/action_replay_request.json",request)
    atomic_write_json(child / "manifests/action_mask_request.json",{"seed":context["seed"]})
    env = dict(os.environ,ACTION_MASK_RUN_DIR=str(child),PYTHONUNBUFFERED="1")
    launch(["bash",str(KIT / "sonic_repro.sh"),"replay-action-mask"],child / "logs/worker.log",env)
    hashes = {p:file_sha256(child/p) for p in ("data/raw_actions.npz","data/exact_initialization.npz",
        "manifests/action_replay_request.json","manifests/action_mask_request.json")}
    for i in range(len(names)):
        for rel in (f"data/replay/{i:06d}.replay.npz", f"data/replay/{i:06d}.trajectory.pkl",
                    f"manifests/exact_initialization_readback_{i:06d}.json",f"data/replay/{i:06d}.runtime.json"):
            hashes[rel] = file_sha256(child / rel)
    atomic_write_json(child / "manifests/replay65_worker_complete.json",{"hashes":hashes})


def simulate(args):
    run = args.run.resolve(); verify_prepared(run,code=True)
    for relative in load_json(run / "manifests/replay65.json")["windows"]:
        window = run / relative
        meta = load_json(window / "manifests/replay65.json")
        context = window_context(window)
        progress(run,"simulating",window=relative,phase="original_repeat_baseline")
        if meta.get("source_mode", "dataset") == "recapture":
            # Copied from the capture's determinism gate and sealed at prepare.
            verify_worker(window / "simulations/baseline")
        else:
            source = _load_replay(window / "data/source.replay.npz")
            simulate_group(window,"baseline",["original_1","original_2"],[source["raw_action"]]*2,context)
        report_window(window)
        if args.baseline_only:
            continue
        for entry in meta["entries"]:
            if entry["action_model_replay"]:
                progress(run,"simulating",window=relative,phase=entry["mask"])
                with np.load(window / "data" / entry["mask"] / "prediction.npz",allow_pickle=False) as data:
                    raw = data["executed_raw"].copy()
                simulate_group(window,entry["mask"],[entry["mask"]],[raw],context)
        report_window(window)
    report(args)
    progress(run,"baseline_complete" if args.baseline_only else "simulation_complete")


def passes(metrics):
    return all(metrics[k]>=v if k=="foot_contact_accuracy" else metrics[k]<=v for k,v in THRESHOLDS.items())


def model_quality_against_baseline(baseline, model):
    """Compare a model Action replay to the same-window original replay.

    The source comparison is used for both trajectories so common simulator
    drift is not mistaken for a model regression.  The small denominator floor
    makes a near-perfect source replay a stricter, but finite, reference.
    """
    ratios = {}
    for key in MODEL_QUALITY_METRICS:
        reference = float(baseline[key])
        value = float(model[key])
        ratios[key] = value / max(abs(reference), 1.0e-6)
    return {"pass": all(value <= MODEL_QUALITY_RATIO for value in ratios.values()),
            "ratio_threshold": MODEL_QUALITY_RATIO, "ratios": ratios,
            "baseline_metrics": {key: float(baseline[key]) for key in MODEL_QUALITY_METRICS},
            "model_metrics": {key: float(model[key]) for key in MODEL_QUALITY_METRICS}}


def absolute_model_quality(model):
    return {"pass": passes(model), "mode": "absolute_thresholds",
            "model_metrics": {key: float(model[key]) for key in THRESHOLDS}}


def initialization_checks(readback, expected, runtime):
    errors = readback.get("errors",{})
    limits = {"joint_pos_max_abs_rad":1e-5,"joint_vel_max_abs_rad_s":1e-5,
        "root_pos_max_abs_m":1e-5,"root_orientation_error_deg":1e-4,"root_lin_vel_max_abs_m_s":1e-5,
        "root_ang_vel_max_abs_rad_s":1e-5,"body_pos_max_abs_m":1e-5,"runtime_context_max_abs":1e-5,
        "previous_raw_action_max_abs":1e-6,"current_raw_action_max_abs":1e-6,
        "previous_processed_action_max_abs":1e-6,"initial_joint_target_max_abs":1e-6}
    # Contact sensors are measured, not writable state; cold-start disagreement is reported separately.
    return {"payload":readback.get("payload_sha256")==expected["payload_sha256"],
        "file":readback.get("initialization_file_sha256")==expected["file_sha256"],
        "application_complete":readback.get("application_complete") is True,
        **{k:bool(np.isfinite(errors.get(k,np.inf)) and errors.get(k,np.inf)<=v) for k,v in limits.items()},
        "runtime_contract":runtime.get("contract_verified") is True,
        "no_midrun_reset":runtime.get("midrun_reset") is False}


def prefix_errors(reference, value, stop):
    """Max deviation on the shared closed-loop prefix (identical raw Actions)."""
    return {key: float(np.abs(value[key][:stop + 1] - reference[key][:stop + 1]).max())
            for key in ("joint_pos", "root_pos", "body_pos")}


def report_window(window):
    meta = load_json(window / "manifests/replay65.json")
    horizon = int(meta.get("horizon_steps", CHUNK))
    offset = int(meta.get("score_offset", 0))
    steps = int(meta.get("replay_steps", offset + horizon))
    episode_start = int(meta.get("window_start", meta["window"]["window_start"]))
    recapture = meta.get("source_mode", "dataset") == "recapture"
    deterministic = meta.get("source", {}).get("deterministic")
    source_full = _load_replay(window / "data/source.replay.npz")
    if source_full["joint_pos"].shape[0] != steps + 1:
        raise ValueError(f"source trajectory must contain {steps + 1} States")
    source = _trajectory_region(source_full, offset, offset + horizon)
    with np.load(window / "data/exact_initialization.npz",allow_pickle=False) as archive:
        mapping = {k:archive[k].copy() for k in ("action_scale","action_offset","action_clip","wrapper_action_clip")}
    trajectories, init_ok, execution = {}, {}, {}
    comparisons = {}
    for tag in ["baseline"]+[e["mask"] for e in meta["entries"] if e["action_model_replay"]]:
        child = window / "simulations" / tag
        if not (child / "manifests/replay65_worker_complete.json").exists():
            continue
        verify_worker(child)
        for i in range(2 if tag=="baseline" else 1):
            name = f"original_{i+1}" if tag=="baseline" else tag
            full_replay = _load_replay(child / f"data/replay/{i:06d}.replay.npz")
            if full_replay["joint_pos"].shape != (steps+1,29) or full_replay["raw_action"].shape != (steps,29):
                raise ValueError(f"partial/reset replay cannot be scored as a complete {steps}-step replay")
            with np.load(child / "data/raw_actions.npz",allow_pickle=False) as planned:
                if not np.allclose(full_replay["raw_action"],planned["raw_actions"][:,i],atol=1e-6,rtol=0):
                    raise ValueError("planned/executed raw Action mismatch")
            replay = _trajectory_region(full_replay, offset, offset + horizon)
            with np.load(child / f"data/replay/{i:06d}.replay.npz",allow_pickle=False) as archive:
                processed = finite("executed processed targets",archive["processed_action"],(steps,29))
                planned_processed = raw_to_target(full_replay["raw_action"],mapping["action_scale"],mapping["action_offset"],
                    mapping["action_clip"] if mapping["action_clip"].size else None,
                    None if np.isnan(mapping["wrapper_action_clip"]) else float(mapping["wrapper_action_clip"]))
                processed_error = float(np.abs(processed-planned_processed).max())
                if not np.allclose(processed,planned_processed,atol=2e-6,rtol=1e-5):
                    raise ValueError("planned/executed processed target mismatch")
            rb = load_json(child / f"manifests/exact_initialization_readback_{i:06d}.json")
            runtime = load_json(child / f"data/replay/{i:06d}.runtime.json")
            init_ok[name] = initialization_checks(rb,meta["initialization"],runtime)
            execution[name] = {"initialization":init_ok[name],"runtime":runtime,
                "processed_target_max_abs_rad":processed_error,
                "initial_contact_sensor_max_abs":rb["errors"].get("physics_state_v3_max_abs")}
            trajectories[name] = replay
            key = f"source_to_{name}"
            comparisons[key] = _trajectory_metrics(source,replay)
            comparisons[key]["chunks"] = [_trajectory_metrics(_trajectory_region(source, CHUNK*k, CHUNK*(k+1)),
                _trajectory_region(replay, CHUNK*k, CHUNK*(k+1))) for k in range(horizon // CHUNK)]
            if offset:
                comparisons[key]["prefix_max_abs"] = prefix_errors(source_full, full_replay, offset)
            errors = _per_frame_errors(source,replay)
            plot_errors = dict(errors)
            errors["joint_velocity_rmse_rad_s"] = np.sqrt(np.mean((replay["joint_vel"]-source["joint_vel"])**2,axis=-1))
            _write_npz(window / "data" / f"{key}_errors.npz",**errors)
            (window / "plots").mkdir(exist_ok=True)
            _write_error_svg(window / "plots" / f"{key}.svg",plot_errors,50.,
                title=f"{'Closed-loop capture' if recapture else 'Recorded HDF'} vs {name} (episode frames "
                      f"{episode_start}-{episode_start + horizon})")
            crossings = _first_threshold_crossings(errors)
            comparisons[key]["first_crossing_relative_frame"] = crossings
            comparisons[key]["first_crossing_episode_frame"] = {k:None if t is None else t+episode_start for k,t in crossings.items()}
    baseline_valid = None
    if "original_2" in trajectories:
        comparisons["original_repeatability"] = _trajectory_metrics(trajectories["original_1"],trajectories["original_2"])
        baseline_valid = (all(all(init_ok[n].values()) for n in ("original_1","original_2"))
            and all(passes(comparisons[k]) for k in ("source_to_original_1","source_to_original_2","original_repeatability"))
            and deterministic is not False)
    for name,replay in trajectories.items():
        if name.startswith("original_"):
            continue
        comparisons[f"original_to_{name}"] = _trajectory_metrics(trajectories["original_1"],replay)
        with np.load(window / "data" / name / "prediction.npz",allow_pickle=False) as values:
            comparisons[f"predicted_state_to_realized_{name}"] = physical_groups(values["predicted_state"],replay["physical_state"])
            comparisons[f"completed_state_to_realized_{name}"] = physical_groups(values["completed_state"],replay["physical_state"])
        if not all(init_ok[name].values()):
            execution[name]["model_physical_quality"] = "UNDETERMINED_INITIALIZATION_INVALID"
    expected_names = {"original_1","original_2"}|{e["mask"] for e in meta["entries"] if e["action_model_replay"]}
    execution_complete = expected_names <= set(trajectories)
    model_names = [name for name in trajectories if not name.startswith("original_")]
    model_quality_results = {}
    if baseline_valid and model_names:
        baseline_metrics = comparisons["source_to_original_1"]
        for name in model_names:
            # A deterministic capture makes the baseline ~0, so a ratio is meaningless there.
            result = None if not (init_ok.get(name) and all(init_ok[name].values())) else (
                absolute_model_quality(comparisons[f"source_to_{name}"]) if recapture else
                model_quality_against_baseline(baseline_metrics, comparisons[f"source_to_{name}"]))
            model_quality_results[name] = result
            execution[name]["model_quality"] = result or {
                "pass": None, "reason": "initialization_or_runtime_contract_invalid"
            }
    model_quality_pass = (
        all(result is not None and result["pass"] for result in model_quality_results.values())
        if execution_complete and baseline_valid and model_names and len(model_quality_results) == len(model_names)
        else None
    )
    banner = (
        "BASELINE_INVALID / MODEL_QUALITY_UNDETERMINED" if baseline_valid is False else
        "BASELINE_NOT_RUN / MODEL_QUALITY_UNDETERMINED" if baseline_valid is None else
        "BASELINE_VALID / MODEL_QUALITY_PASS" if model_quality_pass is True else
        "BASELINE_VALID / MODEL_QUALITY_FAIL" if model_quality_pass is False else
        "BASELINE_VALID / QUALITY_UNASSESSED"
    )
    report = {"version":VERSION,"window":meta["window"],"execution_complete":execution_complete,
        "baseline_valid":baseline_valid,"quality_pass":model_quality_pass,
        "model_quality": model_quality_results,
        "model_physical_quality": (
            "PASS" if model_quality_pass is True else
            ("FAIL_ABSOLUTE_THRESHOLDS" if recapture else "FAIL_RELATIVE_TO_BASELINE") if model_quality_pass is False else
            "UNASSESSED_NO_MODEL_ACTION_REPLAY" if baseline_valid else
            "MODEL_QUALITY_UNDETERMINED"
        ),
        "banner":banner,"source_mode":"recapture" if recapture else "dataset","deterministic_source":deterministic,
        "horizon_steps":horizon,"score_offset":offset,"replay_steps":steps,"window_start":episode_start,
        "threshold_version":"window-baseline-v2+" + ("model-absolute-v1" if recapture else "model-relative-v1"),
        "thresholds":THRESHOLDS,
        "model_quality_thresholds":({"mode":"absolute_thresholds","thresholds":THRESHOLDS} if recapture else
            {"mode":"relative_to_baseline","ratio_threshold":MODEL_QUALITY_RATIO,"metrics":MODEL_QUALITY_METRICS}),
        "checks":execution,"comparisons":comparisons,"missing_scenarios":sorted(expected_names-set(trajectories))}
    atomic_write_json(window / "manifests/replay65_report.json",report)
    return report


def report(args):
    run=args.run.resolve(); verify_prepared(run)
    reports={p:report_window(run/p) for p in load_json(run/"manifests/replay65.json")["windows"]}
    simulation_complete = all(r["execution_complete"] for r in reports.values())
    rendered = run/"manifests/replay65_render.json"
    render_complete = False
    if rendered.is_file():
        videos=load_json(rendered)["videos"]
        expected=0
        for relative in reports:
            entries=load_json(run/relative/"manifests/replay65.json")["entries"]
            expected += 1+sum(e["representative"] for e in entries)+sum(e["action_model_replay"] for e in entries)
        render_complete=len(videos)==expected and all((run/v["path"]).is_file() and file_sha256(run/v["path"])==v["sha256"] for v in videos)
    quality_values = [report["quality_pass"] for report in reports.values()]
    baseline_values = [report["baseline_valid"] for report in reports.values()]
    quality_pass = True if quality_values and all(value is True for value in quality_values) else (
        False if any(value is False for value in quality_values) else None
    )
    baseline_valid = True if baseline_values and all(value is True for value in baseline_values) else (
        False if any(value is False for value in baseline_values) else None
    )
    summary={"version":VERSION,"execution_complete":simulation_complete and render_complete,
             "simulation_complete":simulation_complete,"render_complete":render_complete,
             "baseline_valid":baseline_valid,"quality_pass":quality_pass,"windows":reports}
    atomic_write_json(run/"manifests/replay65_report.json",summary)
    if summary["execution_complete"]:
        atomic_write_text(
            run / "markers/replay65_execution.ok",
            f"EXECUTION COMPLETE; baseline_valid={summary['baseline_valid']} quality_pass={summary['quality_pass']}\n",
        )
        quality_text = {True: "MODEL QUALITY PASS", False: "MODEL QUALITY FAIL", None: "MODEL QUALITY UNASSESSED"}[summary["quality_pass"]]
        atomic_write_text(run / "markers/replay65_model_quality.status", quality_text + "\n")
    if getattr(args,"export",False):
        output=getattr(args,"output",None) or run/"replay65_report.zip"
        output=output.resolve()
        files=[p for p in run.rglob("*") if p.is_file() and p.suffix in {".json",".jsonl",".log",".svg",".npz"}]
        with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_DEFLATED) as archive:
            for p in files:
                archive.write(p,str(p.relative_to(run)))
            archive.writestr("report_index.json",json.dumps({"hashes":{str(p.relative_to(run)):file_sha256(p) for p in files},"excluded":["checkpoints","HDF5","videos","trajectory pickle"]}))
        print(f"REPORT={output}",flush=True)
    return summary


def render(args):
    # Renderer-only overlays may legitimately evolve after preparation.  The
    # explicit flag keeps normal render calls tied to the sealed source while
    # allowing a read-only annotation refresh on an existing prepared run.
    verify_prepared(args.run.resolve(),code=not getattr(args, "allow_render_code_change", False))
    report(args)
    run=args.run.resolve()
    command = [sys.executable, str(KIT / "render_replay65.py"), "--run", str(run),
               "--model", str(args.model.resolve()), "--gl", args.gl]
    command.extend(["--robot-profile", getattr(args, "robot_profile", "g1")])
    if getattr(args, "mesh_dir", None) is not None:
        command.extend(["--mesh-dir", str(args.mesh_dir.resolve())])
    command.extend(["--ground-style", getattr(args, "ground_style", "checker")])
    if getattr(args, "origin_only", False):
        command.append("--origin-only")
    launch(command, run / "logs/render.log")
    if getattr(args, "origin_only", False):
        progress(run, "origin_render_complete")
    else:
        report(args)
        progress(run,"render_complete")


def mapping_error(path, init, steps):
    with np.load(path, allow_pickle=False) as archive:
        raw = finite("captured raw Action", archive["raw_action"], (steps, 29))
        processed = finite("captured processed target", archive["processed_action"], (steps, 29))
    _, scale, offset, clip, wrapper = mapping_args(init)
    planned = raw_to_target(raw, scale, offset, clip, wrapper)
    if not np.allclose(processed, planned, atol=2e-6, rtol=1e-5):
        raise ValueError("captured raw/processed Action mapping mismatch")
    return float(np.abs(processed - planned).max())


def capture_source(args, run):
    """Frame 0 .. window_start + horizon of the HDF episode; only frame 0 seeds the capture."""
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    config = checkpoint_config(checkpoint)
    dataset, indices = make_dataset(args.dataset_run.resolve(), config)
    try:
        if args.window_index not in indices:
            raise ValueError("window index is not in checkpoint's selected training windows")
        rows = Fixtures(dataset, indices, expand=True).manifest()
        row = next(r for r in rows if r["window_index"] == args.window_index)
        steps = int(dataset.episodes[dataset.refs[args.window_index].episode_index]["steps"])
        start, shift = resolve_window_start(row["window_start"], steps, args.horizon_steps)
        if not bool(getattr(args, "state_only_continuation", False)):
            chunk_indices(dataset, args.window_index, start, args.horizon_steps // CHUNK)
        source, init, schema, source_meta = load_source(dataset, args.window_index, start + args.horizon_steps, start=0)
    finally:
        dataset.close()
    _write_npz(run / "data/dataset_source.replay.npz", **source)
    _write_trajectory(run / "data/dataset_source.trajectory.pkl", joint_pos=source["joint_pos"],
        root_pos=source["root_pos"], root_quat=source["root_quat"], fps=50.)
    _, init_manifest = _write_exact_initialization(run / "data/exact_initialization.npz", init)
    atomic_write_json(run / "manifests/source_schema.json", schema)
    shutil.copy2(args.dataset_run.resolve() / "data/normalization.npz", run / "data/normalization.npz")
    meta = {"version": VERSION, "window": row, "window_start": start, "requested_window_start": int(row["window_start"]),
        "window_start_shift": shift, "horizon_steps": int(args.horizon_steps), "episode_steps": steps,
        "simulation_seed": int(args.simulation_seed), "motion_file": source_meta["motion_file"],
        "motion_file_sha256": source_meta["motion_file_sha256"], "raw_source": source_meta["raw_source"],
        "initialization": init_manifest, "checkpoint": str(args.checkpoint.resolve()),
        "single_window_continuation": bool(getattr(args, "state_only_continuation", False)),
        "dataset_run": str(args.dataset_run.resolve())}
    atomic_write_json(run / "manifests/capture.json", meta)
    return meta


def determinism_metrics(a, b, start, stop):
    return {"max_abs": {key: float(np.abs(a[key] - b[key]).max()) for key in DETERMINISM_FIELDS},
            "full": _trajectory_metrics(a, b),
            "scored": _trajectory_metrics(_trajectory_region(a, start, stop), _trajectory_region(b, start, stop))}


def capture_context(run, meta):
    return {"motion_file": meta["motion_file"], "motion_file_sha256": meta["motion_file_sha256"],
            "motion_key": meta["window"]["motion_key"], "seed": int(meta["simulation_seed"]),
            "init_path": run / "data/exact_initialization.npz", "initialization": meta["initialization"],
            "steps": int(meta["window_start"]) + int(meta["horizon_steps"])}


def capture(args):
    """Closed-loop rollout from HDF frame 0, then two open-loop replays of its raw Actions."""
    run = args.output_run.resolve()
    assert_output_isolated(run, [args.dataset_run.resolve(), args.checkpoint.resolve().parent.parent,
                                 ROOT / "state-action-cvae", KIT])
    horizon = int(args.horizon_steps)
    if horizon < CHUNK or horizon % CHUNK:
        raise ValueError("--horizon-steps must be a positive multiple of 64")
    if (run / "manifests/capture_complete.json").is_file():
        complete = verify_capture(run)
        print(f"CAPTURE={run} deterministic={complete['deterministic']} (already sealed)", flush=True)
        return complete
    meta_path = run / "manifests/capture.json"
    if meta_path.is_file():
        meta = load_json(meta_path)
        expected = (int(meta["window"]["window_index"]), int(meta["horizon_steps"]),
                    int(meta["simulation_seed"]), bool(meta.get("single_window_continuation", False)))
        actual = (args.window_index, horizon, args.simulation_seed,
                  bool(getattr(args, "state_only_continuation", False)))
        if expected != actual:
            raise ValueError(f"capture run was started with window/horizon/seed/mode {expected}; use a new --output-run")
    else:
        if run.exists() and any(run.iterdir()):
            raise FileExistsError("capture requires a new empty output run")
        for folder in ("manifests", "data", "plots", "logs", "markers"):
            (run / folder).mkdir(parents=True, exist_ok=True)
        meta = capture_source(args, run)
    progress(run, "capturing", window_index=args.window_index, horizon_steps=horizon)
    context = capture_context(run, meta)
    start, steps = int(meta["window_start"]), context["steps"]
    readback = run / "manifests/exact_initialization_readback_capture.json"
    request = {"schema_version": VERSION, "motion_file": meta["motion_file"],
        "motion_file_sha256": meta["motion_file_sha256"], "motion_key": context["motion_key"],
        "seed": context["seed"], "steps": steps, "exact_initialization_file": str(context["init_path"]),
        "exact_initialization_file_sha256": meta["initialization"]["file_sha256"],
        "exact_initialization_report_path": str(readback)}
    atomic_write_json(run / "manifests/replay65_capture_request.json", request)
    source_dir = run / "data/source"
    if not (run / "markers/replay65_capture.ok").is_file():
        if source_dir.exists() and any(source_dir.iterdir()):
            raise FileExistsError(f"partial closed-loop capture is not overwritten: {source_dir}; use a new --output-run")
        env = dict(os.environ, ACTION_MASK_RUN_DIR=str(run), PYTHONUNBUFFERED="1")
        launch(["bash", str(KIT / "sonic_repro.sh"), "capture-replay65-source"], run / "logs/capture_worker.log", env)
    captured = _load_replay(source_dir / "000000.replay.npz")
    if captured["joint_pos"].shape != (steps + 1, 29) or captured["raw_action"].shape != (steps, 29):
        raise ValueError(f"closed-loop capture must contain {steps} Actions and {steps + 1} States (reset or early stop?)")
    runtime = load_json(source_dir / "000000.runtime.json")
    if runtime.get("control_steps") not in (None, steps):
        raise ValueError(f"capture recorded {runtime.get('control_steps')} control steps, expected {steps}")
    capture_checks = initialization_checks(load_json(readback), meta["initialization"], runtime)
    with np.load(context["init_path"], allow_pickle=False) as archive:
        init = {name: archive[name].copy() for name in archive.files}
    mapping = mapping_error(source_dir / "000000.replay.npz", init, steps)
    progress(run, "capture_determinism_gate", steps=steps)
    simulate_group(run, "baseline", ["original_1", "original_2"], [captured["raw_action"]] * 2, context)
    baseline = run / "simulations/baseline"
    replays, replay_checks = [], {}
    for i in range(2):
        replays.append(_load_replay(baseline / f"data/replay/{i:06d}.replay.npz"))
        replay_checks[f"original_{i + 1}"] = initialization_checks(
            load_json(baseline / f"manifests/exact_initialization_readback_{i:06d}.json"), meta["initialization"],
            load_json(baseline / f"data/replay/{i:06d}.runtime.json"))
    stop = start + horizon
    pairs = {"capture_to_original_1": determinism_metrics(captured, replays[0], start, stop),
             "capture_to_original_2": determinism_metrics(captured, replays[1], start, stop),
             "original_1_to_original_2": determinism_metrics(replays[0], replays[1], start, stop)}
    within = {name: {key: pair["max_abs"][key] <= limit for key, limit in DETERMINISM_LIMITS.items()}
              for name, pair in pairs.items()}
    init_valid = all(capture_checks.values()) and all(all(c.values()) for c in replay_checks.values())
    deterministic = init_valid and all(all(v.values()) for v in within.values())
    # Informational only: how far the closed-loop policy drifts from the recorded HDF episode.
    dataset = _load_replay(run / "data/dataset_source.replay.npz")
    scored_dataset, scored_capture = _trajectory_region(dataset, start, stop), _trajectory_region(captured, start, stop)
    _write_error_svg(run / "plots/dataset_to_capture.svg", _per_frame_errors(scored_dataset, scored_capture), 50.,
                     title=f"Recorded HDF vs closed-loop capture (episode frames {start}-{stop})")
    determinism = {"version": VERSION, "deterministic": deterministic, "limits": DETERMINISM_LIMITS,
        "scored_region": [start, stop], "pairs": pairs, "within_limits": within,
        "initialization": {"capture": capture_checks, **replay_checks}, "capture_runtime": runtime,
        "capture_processed_target_max_abs_rad": mapping,
        "dataset_to_capture": {"full": _trajectory_metrics(dataset, captured),
                               "scored": _trajectory_metrics(scored_dataset, scored_capture)}}
    atomic_write_json(run / "manifests/capture_determinism.json", determinism)
    sealed = [*sorted(str(p.relative_to(run)) for p in source_dir.iterdir() if p.is_file() and not p.is_symlink()),
              "data/exact_initialization.npz", "data/dataset_source.replay.npz", "data/normalization.npz",
              "manifests/capture.json", "manifests/source_schema.json", "manifests/replay65_capture_request.json",
              str(readback.relative_to(run)), "simulations/baseline/manifests/replay65_worker_complete.json",
              "manifests/capture_determinism.json"]
    atomic_write_json(run / "manifests/capture_complete.json", {"version": VERSION, "deterministic": deterministic,
        "hashes": {relative: file_sha256(run / relative) for relative in sealed}})
    atomic_write_text(run / "markers/replay65_capture.status", ("DETERMINISTIC" if deterministic else "NON_DETERMINISTIC") + "\n")
    progress(run, "capture_complete", deterministic=deterministic)
    print(f"CAPTURE={run} deterministic={deterministic}", flush=True)
    for name, pair in pairs.items():
        print(f"  {name}: " + " ".join(f"{k}={pair['max_abs'][k]:.3g}" for k in DETERMINISM_LIMITS), flush=True)
    return load_json(run / "manifests/capture_complete.json")


COMPARE_COLORS = ((66, 135, 245), (60, 200, 90), (255, 150, 40), (200, 80, 200))
FALL_TILT_DEG = 60.
FALL_ROOT_HEIGHT_M = .35


def first_fall_frame(trajectory):
    quat = np.asarray(trajectory["root_quat"], np.float64)  # wxyz
    up_z = 1. - 2. * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = (up_z < np.cos(np.deg2rad(FALL_TILT_DEG))) | (np.asarray(trajectory["root_pos"])[:, 2] < FALL_ROOT_HEIGHT_M)
    hits = np.flatnonzero(fallen)
    return None if hits.size == 0 else int(hits[0])


def compare_entry(meta, mask):
    replays = [entry for entry in meta["entries"] if entry["action_model_replay"]]
    if meta["route"] == "A":
        # A is full prediction: its single Action replay is mask independent.
        if len(replays) != 1:
            raise ValueError("route A run must contain exactly one Action replay")
        return replays[0]
    entry = next((entry for entry in replays if entry["mask"] == mask), None)
    if entry is None:
        raise ValueError(f"route {meta['route']} run has no Action replay for {mask}; prepare it with --masks {mask}")
    return entry


def write_compare_svg(path, routes, title, fps=50.):
    keys = ("joint_position_rmse_rad", "root_position_error_m", "root_orientation_error_deg", "body_mpjpe_m")
    width, height, left, right, top, bottom = 1100, 700, 100, 150, 45, 50
    panel = (height - top - bottom) / len(keys)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width // 2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>']
    frames = max(len(route["per_frame"][keys[0]]) for route in routes)
    for row, key in enumerate(keys):
        y0, y1 = top + row * panel, top + (row + 1) * panel - 18
        peak = max(max(max(route["per_frame"][key]), 1e-12) for route in routes) * 1.1
        parts.append(f'<rect x="{left}" y="{y0:.1f}" width="{width - left - right}" height="{y1 - y0:.1f}" fill="none" stroke="#999"/>')
        parts.append(f'<text x="{left - 8}" y="{(y0 + y1) / 2:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{key}</text>')
        parts.append(f'<text x="{left - 8}" y="{y0 + 10:.1f}" text-anchor="end" font-family="sans-serif" font-size="10">{peak:.3g}</text>')
        for route in routes:
            color = "#%02x%02x%02x" % tuple(route["color"])
            points = " ".join(f"{left + t * (width - left - right) / max(frames - 1, 1):.1f},{y1 - v / peak * (y1 - y0):.1f}"
                              for t, v in enumerate(route["per_frame"][key]))
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{points}"/>')
            if route["fall_frame"] is not None:
                x = left + route["fall_frame"] * (width - left - right) / max(frames - 1, 1)
                parts.append(f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}" stroke="{color}" stroke-dasharray="4 3"/>')
    for i, route in enumerate(routes):
        color = "#%02x%02x%02x" % tuple(route["color"])
        parts.append(f'<rect x="{width - right + 15}" y="{top + 20 * i}" width="14" height="10" fill="{color}"/>')
        parts.append(f'<text x="{width - right + 35}" y="{top + 20 * i + 10}" font-family="sans-serif" font-size="12">'
                     f'{html.escape(route["label"])}</text>')
    parts.append(f'<text x="{width // 2}" y="{height - 15}" text-anchor="middle" font-family="sans-serif" font-size="12">'
                 f'frame (50 Hz, {frames / fps:.2f} s); dashed line = first fall frame</text></svg>')
    atomic_write_text(path, "\n".join(parts) + "\n")


def compare(args):
    """Same window, same source, one Action replay per route; renders ghost overlay and/or 2x2 grid."""
    output = args.output_run.resolve()
    if len(args.labels) != len(args.runs) or not 1 <= len(args.runs) <= len(COMPARE_COLORS):
        raise ValueError(f"--labels must match --runs (1-{len(COMPARE_COLORS)} runs)")
    route_count = len(args.runs) + (1 if getattr(args, "include_origin", False) else 0)
    if route_count > 3:
        raise ValueError("Source plus routes must fit the 2x2 grid (at most three routes)")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("compare requires a new empty output run")
    reference, routes, source_window = None, [], None
    masks = list(args.masks) if getattr(args, "masks", None) else [args.mask] * len(args.runs)
    if len(masks) != len(args.runs):
        raise ValueError("--masks must match --runs")
    color_offset = 1 if getattr(args, "include_origin", False) else 0
    for run_index, (run_path, label, mask) in enumerate(zip(args.runs, args.labels, masks)):
        color = COMPARE_COLORS[run_index + color_offset]
        run = run_path.resolve()
        verify_prepared(run)
        window = run / load_json(run / "manifests/replay65.json")["windows"][0]
        meta = load_json(window / "manifests/replay65.json")
        entry = compare_entry(meta, mask)
        child = window / "simulations" / entry["mask"]
        if not (child / "manifests/replay65_worker_complete.json").is_file():
            raise FileNotFoundError(f"route {label} has not simulated {entry['mask']}; run replay65 simulate --run {run}")
        verify_worker(child)
        key = {"stable_window_id": meta["window"]["stable_window_id"], "horizon_steps": int(meta.get("horizon_steps", CHUNK)),
               "score_offset": int(meta.get("score_offset", 0)), "simulation_seed": int(meta["simulation_seed"]),
               "source_sha256": file_sha256(window / "data/source.replay.npz")}
        if reference is None:
            reference, source_window = key, window
        elif key != reference:
            diff = {k: (reference[k], key[k]) for k in key if key[k] != reference[k]}
            raise ValueError(f"route {label} does not replay the same window/source as {args.labels[0]}: {diff}")
        horizon, offset = key["horizon_steps"], key["score_offset"]
        source = _trajectory_region(_load_replay(window / "data/source.replay.npz"), offset, offset + horizon)
        replay = _trajectory_region(_load_replay(child / "data/replay/000000.replay.npz"), offset, offset + horizon)
        errors = _per_frame_errors(source, replay)
        routes.append({"label": label, "route": meta["route"], "mask": entry["mask"], "color": list(color),
            "run": str(run), "window": str(window), "trajectory": str(child / "data/replay/000000.trajectory.pkl"),
            "trajectory_sha256": file_sha256(child / "data/replay/000000.trajectory.pkl"),
            "metrics": _trajectory_metrics(source, replay), "quality_pass": passes(_trajectory_metrics(source, replay)),
            "fall_frame": first_fall_frame(replay),
            "per_frame": {k: [float(v) for v in values] for k, values in errors.items()}})
    for folder in ("manifests", "plots", "videos", "logs"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    meta = load_json(source_window / "manifests/replay65.json")
    start = int(meta.get("window_start", meta["window"]["window_start"]))
    source_full = _load_replay(source_window / "data/source.replay.npz")
    if getattr(args, "include_origin", False):
        baseline_child = source_window / "simulations" / "baseline"
        verify_worker(baseline_child)
        origin_path = baseline_child / "data" / "replay" / "000000.replay.npz"
        origin = _trajectory_region(_load_replay(origin_path), reference["score_offset"],
                                    reference["score_offset"] + reference["horizon_steps"])
        source_region = _trajectory_region(source_full, reference["score_offset"],
                                           reference["score_offset"] + reference["horizon_steps"])
        origin_errors = _per_frame_errors(source_region, origin)
        origin_metrics = _trajectory_metrics(source_region, origin)
        routes.insert(0, {
            "label": "Origin",
            "route": "origin",
            "mask": "origin",
            "color": list(COMPARE_COLORS[0]),
            "run": str(source_window.parent.parent),
            "window": str(source_window),
            "trajectory": str(baseline_child / "data" / "replay" / "000000.trajectory.pkl"),
            "trajectory_sha256": file_sha256(baseline_child / "data" / "replay" / "000000.trajectory.pkl"),
            "metrics": origin_metrics,
            "quality_pass": passes(origin_metrics),
            "fall_frame": first_fall_frame(origin),
            "per_frame": {k: [float(v) for v in values] for k, values in origin_errors.items()},
        })
    manifest = {"version": VERSION, "mask": args.mask, "masks": masks, "window": meta["window"], "source_mode": meta.get("source_mode", "dataset"),
        "include_origin": bool(getattr(args, "include_origin", False)),
        "single_window_continuation": bool(meta.get("single_window_continuation", False)),
        "continuation_input_contract": meta.get("continuation_input_contract"),
        "horizon_steps": reference["horizon_steps"], "score_offset": reference["score_offset"], "window_start": start,
        "source_trajectory": str(source_window / "data/source.trajectory.pkl"),
        "source_trajectory_sha256": file_sha256(source_window / "data/source.trajectory.pkl"),
        "source_fall_frame": first_fall_frame(_trajectory_region(source_full, reference["score_offset"],
                                                                 reference["score_offset"] + reference["horizon_steps"])),
        "fall_rule": {"tilt_deg": FALL_TILT_DEG, "root_height_m": FALL_ROOT_HEIGHT_M}, "routes": routes}
    atomic_write_json(output / "manifests/compare.json", manifest)
    write_compare_svg(output / "plots/compare_errors.svg", routes,
                      f"Source vs routes {'/'.join(args.labels)} ({args.mask}, episode frames {start}-{start + reference['horizon_steps']})")
    for route in routes:
        fall = "none" if route["fall_frame"] is None else f"frame {route['fall_frame']}"
        m = route["metrics"]
        print(f"{route['label']}: route={route['route']} mask={route['mask']} joint_rmse={m['joint_position_rmse_rad']:.4f} "
              f"root_rmse={m['root_position_rmse_m']:.4f} mpjpe={m['body_mpjpe_m']:.4f} fall={fall}", flush=True)
    if args.skip_render:
        return manifest
    launch([sys.executable, str(KIT / "render_replay65_compare.py"), "--manifest", str(output / "manifests/compare.json"),
            "--model", str(args.model.resolve()), "--gl", args.gl, "--layout", args.layout,
            "--robot-profile", getattr(args, "robot_profile", "g1"),
            "--ground-style", getattr(args, "ground_style", "checker"),
            *(["--mesh-dir", str(args.mesh_dir.resolve())] if getattr(args, "mesh_dir", None) is not None else [])],
           output / "logs/render_compare.log")
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare")
    for flag in ("checkpoint","dataset-run","output-run"):
        p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--route",choices=("A","B","C"),required=True)
    select=p.add_mutually_exclusive_group()
    select.add_argument("--window-index",type=int)
    select.add_argument("--selection",choices=("first-suite",))
    p.add_argument("--source",choices=("dataset","recapture"),
                   help="default: recapture when --capture-run is given, else dataset")
    p.add_argument("--capture-run",type=Path,help="sealed run from 'replay65 capture'")
    p.add_argument("--horizon-steps",type=int,help=f"multiple of {CHUNK}; default: capture horizon or {DEFAULT_HORIZON}")
    p.add_argument(
        "--state-only-continuation",
        action="store_true",
        help=(
            "use the selected T64 window as the only dataset template; for chunks "
            "after the first, provide State only and zero Action/previous-Action inputs"
        ),
    )
    p.add_argument("--masks",nargs="+",
                   choices=tuple(PHYSICAL_MASK_NAMES) + tuple(DIAGNOSTIC_MASK_NAMES),
                   default=list(REPRESENTATIVES),
                   help="physical masks plus full_state/full_both diagnostic masks")
    p.add_argument("--mask-seed",type=int,help="default: checkpoint training fixture seed (A legacy: 20260920)")
    p.add_argument("--sample-seed",type=int,default=20260923)
    p.add_argument("--sample-index",type=int,default=0)
    p.add_argument("--simulation-seed",type=int,default=20260923)
    p.add_argument("--action-mode",choices=("masked-completion","full-prediction"),default="masked-completion")
    p.add_argument(
        "--allow-recovered-dataset",
        action="store_true",
        help=(
            "allow only a dataset_manifest hash mismatch for a deliberately rebuilt "
            "dataset; episodes, normalization, selected windows and count must still match"
        ),
    )
    p.add_argument(
        "--allow-recovered-identity",
        action="store_true",
        help=(
            "diagnostic-only override for explicit dataset identity drift beyond the manifest "
            "(normalization/index/window identity); report will set exact_identity_verified=false"
        ),
    )
    p.add_argument("--device")
    b=sub.add_parser("prepare-method-b", help="prepare legacy recorded-context tail32/context-next32 replay (not recursive B)")
    b.add_argument("--template-run", type=Path, required=True,
                    help="prepared 192-step replay run with the same source/baseline")
    b.add_argument("--offline-run", type=Path, required=True,
                    help="single-group cvae_sa.long_horizon_eval output")
    b.add_argument("--output-run", type=Path, required=True)
    b.add_argument("--label", default="method_b_tail32_context_next32")
    s=sub.add_parser("simulate"); s.add_argument("--run",type=Path,required=True); s.add_argument("--baseline-only",action="store_true")
    r=sub.add_parser("render"); r.add_argument("--run",type=Path,required=True); r.add_argument("--model",type=Path,required=True); r.add_argument("--gl",choices=("egl","osmesa","glfw"),default="egl")
    r.add_argument("--robot-profile", choices=("g1", "g1_plus_locked29"), default="g1")
    r.add_argument("--mesh-dir", type=Path)
    r.add_argument("--ground-style", choices=("checker", "black"), default="checker")
    r.add_argument("--origin-only",action="store_true",help="render only Source vs Origin Action repeatability")
    r.add_argument("--allow-render-code-change",action="store_true",
                   help="read-only renderer refresh after a renderer overlay change; preparation inputs remain sealed")
    r=sub.add_parser("report"); r.add_argument("--run",type=Path,required=True); r.add_argument("--export",action="store_true"); r.add_argument("--output",type=Path)
    c=sub.add_parser("capture",help="closed-loop source capture + determinism gate")
    for flag in ("checkpoint","dataset-run","output-run"):
        c.add_argument("--"+flag,type=Path,required=True)
    c.add_argument("--window-index",type=int,required=True)
    c.add_argument("--horizon-steps",type=int,default=DEFAULT_HORIZON)
    c.add_argument(
        "--state-only-continuation",
        action="store_true",
        help="capture a longer source trajectory without requiring consecutive T64 dataset windows",
    )
    c.add_argument("--simulation-seed",type=int,default=20260923)
    m=sub.add_parser("compare",help="ghost overlay / 2x2 grid of several routes on one window")
    m.add_argument("--runs",type=Path,nargs="+",required=True)
    m.add_argument("--labels",nargs="+",help="default: A B C (one per run)")
    m.add_argument("--mask",choices=PHYSICAL_MASK_NAMES,default="full_action",help="B/C mask; A uses its single replay")
    m.add_argument("--masks",nargs="+",choices=tuple(PHYSICAL_MASK_NAMES) + ("recursive_state_feedback_b",),
                   help="optional per-run mask labels; overrides --mask and must match --runs")
    m.add_argument("--layout",choices=("ghost","grid","both"),default="both")
    m.add_argument("--include-origin",action="store_true",
                   help="add the deterministic Origin Action replay as a comparison route")
    m.add_argument("--output-run",type=Path,required=True)
    m.add_argument("--model",type=Path)
    m.add_argument("--gl",choices=("egl","osmesa","glfw"),default="egl")
    m.add_argument("--robot-profile", choices=("g1", "g1_plus_locked29"), default="g1")
    m.add_argument("--mesh-dir", type=Path)
    m.add_argument("--ground-style", choices=("checker", "black"), default="checker")
    m.add_argument("--skip-render",action="store_true")
    args=parser.parse_args(argv)
    if args.command == "prepare" and (args.sample_index < 0 or (args.window_index is not None and args.window_index < 0)):
        parser.error("sample-index and window-index must be non-negative")
    if args.command == "capture" and args.window_index < 0:
        parser.error("window-index must be non-negative")
    if args.command == "compare":
        args.labels = args.labels or ["A", "B", "C", "D"][:len(args.runs)]
        if len(args.labels) != len(args.runs):
            parser.error("--labels must match --runs")
        if args.masks is not None and len(args.masks) != len(args.runs):
            parser.error("--masks must match --runs")
        if not args.skip_render and args.model is None:
            parser.error("compare requires --model unless --skip-render")
    if sys.platform == "linux":
        run_path = getattr(args,"output_run",getattr(args,"run",None)).resolve()
        runs_root = Path("/home/helloworld/bly/runs").resolve()
        if run_path == runs_root or not run_path.is_relative_to(runs_root):
            parser.error("Ubuntu replay outputs must be a child run under /home/helloworld/bly/runs")
    try:
        if args.command == "prepare-method-b":
            prepare_method_b(args)
        elif args.command == "simulate":
            verify_prepared(args.run.resolve(),code=True)
            with run_lock(args.run.resolve()):
                simulate(args)
        else:
            globals()[args.command](args)
    except (Exception,KeyboardInterrupt) as error:
        run=getattr(args,"run",getattr(args,"output_run",None))
        # Never overwrite unrelated or source paths on a failed preparation.
        marker = "manifests/capture.json" if args.command == "capture" else "manifests/replay65.json"
        if args.command not in ("prepare", "compare") and run and (run/marker).is_file():
            progress(run,"interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error=str(error))
        raise
    return 0


if __name__=="__main__":
    raise SystemExit(main())
