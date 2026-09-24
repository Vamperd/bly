"""Read-only 65-token replay preparation, isolated simulation, rendering and reporting.

All outputs live in a NEW run. No training source or checkpoint is modified.
Isaac/MuJoCo are imported only by the separately launched Ubuntu workers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from .cvae_protocol import CHECKPOINT, Fixtures, digest, isolated_rng, run_lock
from .cvae_training import check_source_identity, data_identity, make_dataset, provenance, source_info
from .models import HierarchicalStandardCVAETransformer, build_model
from .posterior_direct_output import assert_output_isolated
from .posterior_hierarchical_standard_cvae import load_checkpoint
from .posterior_h50_action_replay import _write_npz, _write_trajectory
from .posterior_h50_action_replay_exact_init import (
    _first_threshold_crossings, _load_replay, _per_frame_errors,
    _raw_from_processed, _trajectory_metrics, _write_error_svg,
    _write_exact_initialization, rotate_body_to_world,
)
from .posterior_t64_protocol import PHYSICAL_MASK_NAMES, make_physical_masks
from .state_mask_eval import reconstruct_root_trajectory
from .util import atomic_write_json, atomic_write_text, file_sha256, load_json

VERSION = "65-token-replay-v1"
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
    for name, value in (("raw", original_raw), ("truth", original_canonical), ("prediction", predicted)):
        finite(name, value, (64, 29))
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (64, 29):
        raise ValueError("Action mask must be [64,29]")
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


def predict(model, cpu, route, mask_seed, sample_seed, sample_index=0):
    """Only truth returned to the evaluator; B/C model inputs have hidden values zeroed."""
    device = next(model.parameters()).device
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


def load_source(dataset, index):
    """Only one T64 slice plus its preceding Action; never copy an entire HDF5."""
    import h5py
    from .physics_schema import load_physics_schema, read_physics_states, resolve_parameter
    from .action_mask_eval import _resolve_motion_file
    ref = dataset.refs[index]
    record = dataset.episodes[ref.episode_index]
    start = int(ref.fixed_start)
    if start + 64 > int(record["steps"]):
        raise ValueError("replay requires a complete 64-transition window")
    schema = load_physics_schema(Path(record["schema_path"]))
    sim = schema["simulation"]
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
        states = read_physics_states(ep["states"], start, start + 65)
        canonical = np.asarray(ep["actions/action_target_canonical"][start:start+64], dtype=np.float32)
        processed = np.asarray(ep["actions/processed_joint_target_abs"][start:start+64], dtype=np.float32)
        if "raw_policy_action" in ep["actions"]:
            raw = np.asarray(ep["actions/raw_policy_action"][start:start+64], dtype=np.float32)
            raw_source = "recorded_raw_policy_action"
        else:
            raw, achieved, _ = relative_to_raw_action(canonical, nominal, scale, ctx["action_offset"], clip, wrapper)
            if not np.allclose(achieved, canonical, atol=2e-6, rtol=1e-5):
                raise ValueError("recorded processed targets are not invertible")
            raw_source = "verified_inverse_mapping_raw_missing"
        pos = np.asarray(ep["replay/root_pos_w"][start:start+65], dtype=np.float32)
        quat = np.asarray(ep["replay/root_quat_w"][start:start+65], dtype=np.float32)
        body = np.asarray(ep["replay/body_pos_w"][start:start+65], dtype=np.float32)
        previous = np.asarray(ep["actions/processed_joint_target_abs"][start-1], dtype=np.float32) if start else np.asarray(ep["actions/initial_processed_target_canonical"], dtype=np.float32) + nominal
        previous_raw = np.asarray(ep["actions/raw_policy_action"][start-1], dtype=np.float32) if start and "raw_policy_action" in ep["actions"] else _raw_from_processed(previous, scale, ctx["action_offset"], clip, wrapper)
    for name, value, shape in (("State", states, (65,70)), ("Action", canonical, (64,29)),
                               ("raw", raw, (64,29)), ("root", pos, (65,3)), ("quaternion", quat, (65,4))):
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
            "window_start": start, "hidden_simulator_state": "not_recorded"})))
    motion, motion_info = _resolve_motion_file(record)
    return source, init, schema, {"record": record, "raw_source": raw_source, "motion_file": str(motion),
        "motion_file_sha256": motion_info["sha256"], "removed_world_xy": translation.tolist()}


def progress(run, status, **kwargs):
    row = {"protocol_version": VERSION, "status": status, "pid": os.getpid(),
           "updated_at": datetime.now(timezone.utc).isoformat(), **kwargs}
    atomic_write_json(run / "manifests/progress.json", row)
    print(json.dumps({"run": str(run), **row}, ensure_ascii=False), flush=True)


def prepare_window(args, dataset, fixtures, model, index, run, norm):
    row = next(r for r in fixtures.manifest() if r["window_index"] == index)
    source, init, schema, source_meta = load_source(dataset, index)
    for folder in ("manifests", "data", "plots", "videos", "logs", "markers"):
        (run / folder).mkdir(parents=True, exist_ok=True)
    _write_npz(run / "data/recorded_hdf.replay.npz", **source)
    _write_trajectory(run / "data/recorded_hdf.trajectory.pkl", joint_pos=source["joint_pos"],
        root_pos=source["root_pos"], root_quat=source["root_quat"], fps=50.)
    _, init_manifest = _write_exact_initialization(run / "data/exact_initialization.npz", init)
    atomic_write_json(run / "manifests/exact_initialization.json", init_manifest)
    atomic_write_json(run / "manifests/source_schema.json", schema)
    truth_s, truth_a = source["physical_state"], source["action_target_canonical"]
    entries = []
    for slot, name in enumerate(PHYSICAL_MASK_NAMES):
        sample = dict(dataset[index], stable_window_id=row["stable_window_id"], window_index=index,
                      fixture_index=fixtures.indices.index(index)*8+slot, mask_slot=slot)
        cpu = default_collate([sample])
        ps, pa, sm, am, actual_name, eps = predict(model, cpu, args.route, args.mask_seed, args.sample_seed, args.sample_index)
        assert actual_name == name
        state = ps * norm["state"][1] + norm["state"][0]
        state[:,68:] = ps[:,68:]
        action = pa * norm["action"][1] + norm["action"][0]
        # Compare HDF targets with the normalized dataset before interpreting predictions.
        if not np.allclose(cpu["physical_state"][0].numpy()[:,:68], ((truth_s-norm["state"][0])/norm["state"][1])[:,:68], atol=2e-5, rtol=1e-5):
            raise ValueError("dataset/HDF State window mismatch")
        if not np.allclose(cpu["action"][0].numpy(), (truth_a-norm["action"][0])/norm["action"][1], atol=2e-5, rtol=1e-5):
            raise ValueError("dataset/HDF Action window mismatch")
        completed_s = state if args.route == "A" else np.where(sm, state, truth_s)
        completed_a = action if args.route == "A" else np.where(am, action, truth_a)
        raw, achieved, mapping = execution_actions(source["raw_action"], truth_a, action, am,
            init["nominal_default_joint_pos"], init["action_scale"], init["action_offset"],
            init["action_clip"] if init["action_clip"].size else None,
            None if np.isnan(init["wrapper_action_clip"]) else float(init["wrapper_action_clip"]),
            full_prediction=args.route == "A" or args.action_mode == "full-prediction")
        metrics = offline_metrics(cpu["physical_state"][0].numpy(), cpu["action"][0].numpy(), ps, pa, sm, am)
        directory = run / "data" / name
        directory.mkdir()
        arrays = dict(truth_state=truth_s, truth_action=truth_a, predicted_state=state, predicted_action=action,
            predicted_state_normalized=ps, predicted_action_normalized=pa, completed_state=completed_s,
            completed_action=completed_a, state_mask=sm, action_mask=am, executed_raw=raw,
            achieved_action=achieved, valid_state=np.ones(65,bool), valid_action=np.ones(64,bool))
        if eps is not None:
            arrays.update(epsilon_global=eps[0][0].cpu().numpy(), epsilon_local=eps[1][0].cpu().numpy())
        _write_npz(directory / "prediction.npz", **arrays)
        truth_pos, truth_quat = reconstruct_root_trajectory(truth_s, source["root_pos"], source["root_quat"], 0, .02)
        pred_pos, pred_quat = reconstruct_root_trajectory(completed_s, source["root_pos"], source["root_quat"], 0, .02)
        for tag, st, pos, quat in (("truth_integrated",truth_s,truth_pos,truth_quat), ("predicted_integrated",completed_s,pred_pos,pred_quat)):
            _write_trajectory(directory / f"{tag}.trajectory.pkl", joint_pos=st[:,:29]+init["nominal_default_joint_pos"], root_pos=pos, root_quat=quat, fps=50.)
        representative = name in args.masks and (args.route != "A" or name == args.masks[0])
        entries.append({"mask": name, "fixture_id": digest([row["stable_window_id"], slot, args.mask_seed]),
            "offline": metrics, "physical_groups": physical_groups(truth_s,state), "mapping": mapping,
            "state_top100":tail_coordinates(truth_s,state,*norm["state"],row["window_start"],state=True),
            "action_top100":tail_coordinates(truth_a,action,*norm["action"],row["window_start"]),
            "representative": representative, "action_model_replay": representative and mapping["replaced_elements"] > 0,
            "hidden_initial_state": bool(sm[0].any()), "simulation_initial_state": "recorded_truth_not_model_input",
            "visualization": {"gravity_normalized": True, "contact_not_used_as_pose_constraint": True,
                "truth_integration_root_rmse_m": float(np.sqrt(np.mean((truth_pos-source["root_pos"])**2))),
                "predicted_gravity_norm_min": float(np.linalg.norm(completed_s[:,64:67],axis=-1).min()),
                "predicted_height_anchor_offset_m": float(source["root_pos"][0,2]-completed_s[0,67]),
                "anchor": "shared_recorded_root_pose_frame_zero_only"}})
    manifest = {"version": VERSION, "window": row, "route": args.route, "action_mode": args.action_mode,
        "mask_seed": args.mask_seed, "sample_seed": args.sample_seed, "sample_index": args.sample_index,
        "simulation_seed": args.simulation_seed, "source": source_meta, "entries": entries,
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
    model = build_model(config["model"])
    load_checkpoint(model, checkpoint_path)
    model.to(args.device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
    dataset, indices = make_dataset(dataset_run, config)
    try:
        fixtures = Fixtures(dataset, indices, expand=True)
        rows = fixtures.manifest()
        identity = data_identity(dataset_run, rows)
        identity_check = check_source_identity(checkpoint, identity, allow_unknown=args.route == "A")
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
            prepare_window(args,dataset,fixtures,model,index,child,norm)
            paths.append(str(child.relative_to(run)))
        if file_sha256(checkpoint_path) != checkpoint_hash:
            raise RuntimeError("source checkpoint changed during preparation; use a completed fixed checkpoint and a new run")
        (run / "data").mkdir(exist_ok=True)
        shutil.copy2(dataset_run / "data/normalization.npz",run / "data/normalization.npz")
        prov = provenance(config,vars_serializable(args),identity,source_info(checkpoint_path,checkpoint))
        replay_sources = [Path(__file__), *[Path(__file__).with_name(n) for n in (
            "action_masks.py","physics_schema.py","state_mask_eval.py","posterior_h50_action_replay.py","posterior_h50_action_replay_exact_init.py")],
            *[KIT/n for n in ("exact_replay_initializer.py","replay65_runtime.py","replay65_recorder.py",
                "render_replay65.py","render_mujoco_trajectory.py","render_h50a_exact_action_replays.py","sonic_repro.sh")]]
        for p in replay_sources:
            prov["source_hashes"][str(p.relative_to(ROOT))] = file_sha256(p)
        atomic_write_json(run / "manifests/provenance.json",prov)
        atomic_write_json(run / "manifests/replay65.json",{"version":VERSION,"windows":paths,
            "selection":selection,"identity_check":identity_check,"source_checkpoint":source_info(checkpoint_path,checkpoint),
            "training_mask_seed":training_mask_seed,"replay_mask_seed":args.mask_seed,
            "exact_training_fixture": args.route == "B" and args.mask_seed == training_mask_seed and checkpoint.get("training_contract",{}).get("mask_mode")=="fixed"})
        # Seal all prepared input files. Reports and simulation outputs are intentionally not sealed here.
        hashes = {str(p.relative_to(run)):file_sha256(p) for p in run.rglob("*") if p.is_file() and p.name != "progress.json"}
        atomic_write_json(run / "manifests/prepared_hashes.json",hashes)
        progress(run,"prepared",windows=paths)
    finally:
        dataset.close()


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


def simulate_group(parent, window, tag, names, raws):
    child = window / "simulations" / tag
    if child.exists():
        completion = child / "manifests/replay65_worker_complete.json"
        if completion.is_file():
            verify_worker(child)
            return
        raise FileExistsError(f"incomplete/changed simulation is not overwritten: {child}; prepare a new run")
    manifest = load_json(window / "manifests/replay65.json")
    (child / "data").mkdir(parents=True)
    shutil.copy2(window / "data/exact_initialization.npz", child / "data/exact_initialization.npz")
    init = manifest["initialization"]
    actions = child / "data/raw_actions.npz"
    _write_npz(actions,raw_actions=np.stack(raws,axis=1),scenario_names=np.asarray(names))
    request = {"schema_version":VERSION,"representation":"physics_v4","replay65_guard":True,
        "motion_file":manifest["source"]["motion_file"],"motion_file_sha256":manifest["source"]["motion_file_sha256"],
        "motion_key":manifest["window"]["motion_key"],"raw_actions_file":str(actions),"raw_actions_sha256":file_sha256(actions),
        "steps":64,"num_envs":len(names),"scenario_names":names,"control_dt":.02,
        "exact_initialization_file":str(child / "data/exact_initialization.npz"),
        "exact_initialization_file_sha256":init["file_sha256"],"exact_initialization_payload_sha256":init["payload_sha256"],
        "exact_initialization_report_paths":[str(child / f"manifests/exact_initialization_readback_{i:06d}.json") for i in range(len(names))]}
    atomic_write_json(child / "manifests/action_replay_request.json",request)
    atomic_write_json(child / "manifests/action_mask_request.json",{"seed":manifest["simulation_seed"]})
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
        source = _load_replay(window / "data/recorded_hdf.replay.npz")
        progress(run,"simulating",window=relative,phase="original_repeat_baseline")
        simulate_group(run,window,"baseline",["original_1","original_2"],[source["raw_action"]]*2)
        report_window(window)
        if args.baseline_only:
            continue
        for entry in meta["entries"]:
            if entry["action_model_replay"]:
                progress(run,"simulating",window=relative,phase=entry["mask"])
                with np.load(window / "data" / entry["mask"] / "prediction.npz",allow_pickle=False) as data:
                    raw = data["executed_raw"].copy()
                simulate_group(run,window,entry["mask"],[entry["mask"]],[raw])
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


def report_window(window):
    meta = load_json(window / "manifests/replay65.json")
    source = _load_replay(window / "data/recorded_hdf.replay.npz")
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
            replay = _load_replay(child / f"data/replay/{i:06d}.replay.npz")
            if replay["joint_pos"].shape != (65,29) or replay["raw_action"].shape != (64,29):
                raise ValueError("partial/reset replay cannot be scored as a complete T64 window")
            with np.load(child / "data/raw_actions.npz",allow_pickle=False) as planned:
                if not np.allclose(replay["raw_action"],planned["raw_actions"][:,i],atol=1e-6,rtol=0):
                    raise ValueError("planned/executed raw Action mismatch")
            with np.load(child / f"data/replay/{i:06d}.replay.npz",allow_pickle=False) as archive:
                processed = finite("executed processed targets",archive["processed_action"],(64,29))
                planned_processed = raw_to_target(replay["raw_action"],mapping["action_scale"],mapping["action_offset"],
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
            comparisons[f"recorded_to_{name}"] = _trajectory_metrics(source,replay)
            errors = _per_frame_errors(source,replay)
            plot_errors = dict(errors)
            errors["joint_velocity_rmse_rad_s"] = np.sqrt(np.mean((replay["joint_vel"]-source["joint_vel"])**2,axis=-1))
            _write_npz(window / "data" / f"recorded_to_{name}_errors.npz",**errors)
            (window / "plots").mkdir(exist_ok=True)
            _write_error_svg(window / "plots" / f"recorded_to_{name}.svg",plot_errors,50.)
            comparisons[f"recorded_to_{name}"]["first_crossing_relative_frame"] = _first_threshold_crossings(errors)
            comparisons[f"recorded_to_{name}"]["first_crossing_episode_frame"] = {k:None if t is None else t+meta["window"]["window_start"] for k,t in _first_threshold_crossings(errors).items()}
    baseline_valid = None
    if "original_2" in trajectories:
        comparisons["original_repeatability"] = _trajectory_metrics(trajectories["original_1"],trajectories["original_2"])
        baseline_valid = all(all(init_ok[n].values()) for n in ("original_1","original_2")) and all(passes(comparisons[k]) for k in ("recorded_to_original_1","recorded_to_original_2","original_repeatability"))
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
        baseline_metrics = comparisons["recorded_to_original_1"]
        for name in model_names:
            result = model_quality_against_baseline(
                baseline_metrics, comparisons[f"recorded_to_{name}"]
            ) if init_ok.get(name) and all(init_ok[name].values()) else None
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
            "FAIL_RELATIVE_TO_BASELINE" if model_quality_pass is False else
            "UNASSESSED_NO_MODEL_ACTION_REPLAY" if baseline_valid else
            "MODEL_QUALITY_UNDETERMINED"
        ),
        "banner":banner,
        "threshold_version":"legacy-window-baseline-v1+model-relative-v1","thresholds":THRESHOLDS,
        "model_quality_thresholds":{"ratio_threshold":MODEL_QUALITY_RATIO,"metrics":MODEL_QUALITY_METRICS},
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
    verify_prepared(args.run.resolve(),code=True)
    report(args)
    run=args.run.resolve()
    launch([sys.executable,str(KIT/"render_replay65.py"),"--run",str(run),"--model",str(args.model.resolve()),"--gl",args.gl],run/"logs/render.log")
    report(args)
    progress(run,"render_complete")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare")
    for flag in ("checkpoint","dataset-run","output-run"):
        p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--route",choices=("A","B","C"),required=True)
    select=p.add_mutually_exclusive_group(required=True)
    select.add_argument("--window-index",type=int)
    select.add_argument("--selection",choices=("first-suite",))
    p.add_argument("--masks",nargs="+",choices=PHYSICAL_MASK_NAMES,default=list(REPRESENTATIVES))
    p.add_argument("--mask-seed",type=int,help="default: checkpoint training fixture seed (A legacy: 20260920)")
    p.add_argument("--sample-seed",type=int,default=20260923)
    p.add_argument("--sample-index",type=int,default=0)
    p.add_argument("--simulation-seed",type=int,default=20260923)
    p.add_argument("--action-mode",choices=("masked-completion","full-prediction"),default="masked-completion")
    p.add_argument("--device")
    s=sub.add_parser("simulate"); s.add_argument("--run",type=Path,required=True); s.add_argument("--baseline-only",action="store_true")
    r=sub.add_parser("render"); r.add_argument("--run",type=Path,required=True); r.add_argument("--model",type=Path,required=True); r.add_argument("--gl",choices=("egl","osmesa","glfw"),default="egl")
    r=sub.add_parser("report"); r.add_argument("--run",type=Path,required=True); r.add_argument("--export",action="store_true"); r.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    if args.command == "prepare" and (args.sample_index < 0 or (args.window_index is not None and args.window_index < 0)):
        parser.error("sample-index and window-index must be non-negative")
    if sys.platform == "linux":
        run_path = getattr(args,"output_run",getattr(args,"run",None)).resolve()
        runs_root = Path("/home/helloworld/bly/runs").resolve()
        if run_path == runs_root or not run_path.is_relative_to(runs_root):
            parser.error("Ubuntu replay outputs must be a child run under /home/helloworld/bly/runs")
    try:
        if args.command == "simulate":
            verify_prepared(args.run.resolve(),code=True)
            with run_lock(args.run.resolve()):
                simulate(args)
        else:
            globals()[args.command](args)
    except (Exception,KeyboardInterrupt) as error:
        run=getattr(args,"run",getattr(args,"output_run",None))
        # Never overwrite unrelated or source paths on a failed preparation.
        if args.command != "prepare" and run and (run/"manifests/replay65.json").is_file():
            progress(run,"interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error=str(error))
        raise
    return 0


if __name__=="__main__":
    raise SystemExit(main())
