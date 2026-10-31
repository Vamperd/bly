"""Standalone DAgger CLI; SONIC is loaded only as an external teacher/runtime."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback

import torch
from omegaconf import OmegaConf

from .adapters.plain_mlp_student import PlainMlpStudent
from .adapters.sonic_release_teacher import SonicReleaseTeacher
from .core.cvae_objective import cvae_early_stop_decision
from .core.dataset import AggregatedDataset
from .core.loop import DaggerLoop
from .core.metrics import estimate_teacher_identifiability, same_motion_memorization_pass
from .core.motion_selection import select_motion_keys
from .runtime.sonic_env import SonicActionContract, compose_release_config, create_release_env


def _launch_app(config):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Standalone DAgger runner")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    args_cli.num_envs = int(config.dagger.num_envs)
    args_cli.seed = int(config.seed)
    args_cli.headless = True
    args_cli.multi_gpu = False
    args_cli.distributed = False
    args_cli.device = str(config.get("device", "cuda:0"))
    args_cli.kit_args = "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
    from filelock import FileLock

    print("[dagger] waiting for Isaac app launcher lock", flush=True)
    with FileLock("/tmp/isaaclab_app_launcher.lock"):
        print("[dagger] Isaac app launcher lock acquired", flush=True)
        launcher = AppLauncher(args_cli)
    print("[dagger] AppLauncher returned", flush=True)
    return launcher, args_cli


def _new_run_dir(config) -> Path:
    root = Path(os.path.expanduser(str(config.run_root))).resolve()
    run = root / f"dagger_{config.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for child in ("logs", "manifests", "data/dagger", "checkpoints", "markers"):
        (run / child).mkdir(parents=True, exist_ok=True)
    return run


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _motion_info(config, sonic_root: Path) -> dict:
    motion_file = config.dagger.get("motion_file", None)
    if motion_file is None:
        motion_file = os.environ.get("DAGGER_MOTION_FILE", None)
    if motion_file is None:
        motion_path = sonic_root / "data" / "motion_lib_bones_seed" / "robot_filtered"
    else:
        motion_path = Path(str(motion_file)).expanduser()
        if not motion_path.is_absolute():
            motion_path = sonic_root / motion_path
        motion_path = motion_path.resolve()
    motion_key = config.dagger.get("motion_key", None)
    if motion_key:
        requested = [str(motion_key)]
    else:
        requested = [str(value) for value in (config.dagger.get("motion_keys", []) or [])]
    explicit = bool(requested)
    motion_count = int(config.dagger.get("motion_count", 0) or 0)
    motion_seed = int(config.dagger.get("motion_seed", int(config.seed)))
    if not requested and motion_count > 0:
        requested = select_motion_keys(motion_path, motion_count, motion_seed)
    max_unique = config.dagger.get("max_unique_motions", 1)
    if requested:
        if max_unique is None:
            max_unique = len(requested)
        elif int(max_unique) < len(requested):
            raise ValueError(
                f"max_unique_motions={max_unique} would drop requested motion keys; "
                f"requested {len(requested)}"
            )
    sample_unique = config.dagger.get("sample_unique_motions", None)
    sample_unique = None if sample_unique is None else bool(sample_unique)
    if sample_unique and requested and int(config.dagger.num_envs) > len(requested):
        raise ValueError(
            f"sample_unique_motions requires num_envs <= requested motions "
            f"({int(config.dagger.num_envs)} > {len(requested)})"
        )
    return {
        "motion_file": str(motion_path),
        "requested_keys": requested,
        "selection_method": "explicit" if explicit else ("seeded_random" if requested else "library_default"),
        "selection_seed": motion_seed,
        "requested_count": len(requested),
        "max_unique_motions": None if max_unique is None else int(max_unique),
        "sample_unique_motions": sample_unique,
        "observation_corruption": bool(config.dagger.get("observation_corruption", False)),
        "events": str(config.dagger.get("events", "tracking/level0_4")),
        "terminations": str(config.dagger.get("terminations", "tracking/base_adaptive_strict_ori_foot_xyz")),
        "smpl_motion_file": str(config.dagger.get("smpl_motion_file", "zeros")),
    }


def _render_disables_tracking_terminations(config) -> bool:
    if str(config.mode) != "mlp_render":
        return False
    value = str(config.dagger.get("render_disable_tracking_terminations", False)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _prepare_release_env(config, run_dir: Path) -> dict:
    sonic_root = Path(str(config.sonic_repo_root)).expanduser().resolve()
    motion = _motion_info(config, sonic_root)
    print(
        f"[dagger] motion method={motion['selection_method']} file={motion['motion_file']} "
        f"requested={motion['requested_keys']}",
        flush=True,
    )
    print("[dagger] composing SONIC release config", flush=True)
    release_config = compose_release_config(
        sonic_root,
        num_envs=int(config.dagger.num_envs),
        seed=int(config.seed),
        motion_file=motion["motion_file"],
        smpl_motion_file=motion["smpl_motion_file"],
        motion_keys=motion["requested_keys"] or None,
        max_unique_motions=motion["max_unique_motions"],
        sample_unique_motions=motion["sample_unique_motions"],
        observation_corruption=motion["observation_corruption"],
        events=motion["events"],
        terminations=motion["terminations"],
    )
    print("[dagger] SONIC release config composed", flush=True)
    if _render_disables_tracking_terminations(config):
        # Isaac auto-resets terminated envs, so ignoring done would jump the robot back to the start.
        terminations_cfg = release_config.manager_env.terminations
        dropped = [key for key in terminations_cfg if key not in ("_target_", "time_out")]
        for key in dropped:
            del terminations_cfg[key]
        print(f"[dagger] render tracking terminations disabled: {dropped}", flush=True)
    release_config.manager_env.config.experiment_dir = str(run_dir)
    release_config.manager_env.config.save_rendering_dir = str(run_dir / "renderings")
    reference_terms = [
        str(value) for value in release_config.algo.config.actor.backbone.encoders.g1.inputs
    ]
    print(
        f"[dagger] release env prepared reference_terms={reference_terms}",
        flush=True,
    )
    return {
        "sonic_root": sonic_root,
        "release_config": release_config,
        "motion": motion,
        "reference_terms": reference_terms,
    }


def _create_release_env(prepared: dict, device: torch.device, args_cli) -> object:
    motion = prepared["motion"]
    print("[dagger] creating env", flush=True)
    env = create_release_env(
        prepared["release_config"],
        device=str(device),
        args_cli=args_cli,
        reference_terms=prepared["reference_terms"],
    )
    print("[dagger] create_release_env returned", flush=True)
    motion["loaded_keys"] = env.motion_keys
    print(
        f"[dagger] env ready loaded_motion_keys={motion['loaded_keys']} "
        f"reference_terms={prepared['reference_terms']}",
        flush=True,
    )
    if motion["requested_keys"]:
        missing = [key for key in motion["requested_keys"] if key not in motion["loaded_keys"]]
        if missing:
            raise RuntimeError(f"requested motion keys were not loaded: {missing}")
    return env


def _motion_weights_from_eval(eval_metrics: dict, keys: list[str]) -> dict[str, float] | None:
    by_motion = eval_metrics.get("student_only", {}).get(
        "teacher_action_rmse_by_motion", {}
    )
    termination_rates = [
        float(by_motion.get(key, {}).get("termination_rate", 0.0)) for key in keys
    ]
    maximum = max(termination_rates) if termination_rates else 0.0
    if maximum <= 0.0:
        return None
    return {
        key: 1.0 + 3.0 * rate / maximum
        for key, rate in zip(keys, termination_rates)
    }


def _run_plain(config, run_dir: Path, device: torch.device, args_cli, prepared: dict):
    sonic_root = prepared["sonic_root"]
    release_config = prepared["release_config"]
    motion_info = prepared["motion"]
    reference_terms = prepared["reference_terms"]
    env = _create_release_env(prepared, device, args_cli)
    print("[dagger] resetting env for rollout", flush=True)
    observation = env.reset()
    print("[dagger] env reset returned", flush=True)
    print(f"[dagger] observation keys={list(observation.keys())}", flush=True)
    actor_obs = observation["actor_obs"]
    reference_features = observation["student_reference_features"]
    print(
        f"[dagger] actor_obs shape={tuple(actor_obs.shape)} "
        f"reference shape={tuple(reference_features.shape)}",
        flush=True,
    )
    observation_dim = int(actor_obs.shape[-1] + reference_features.shape[-1])
    action_dim = int(env.action_space.shape[-1])
    action_low = env.action_low
    action_high = env.action_high
    finite_bounds = bool(torch.isfinite(action_low).all() and torch.isfinite(action_high).all())
    clip_value = config.dagger.get("action_clip_value", None)
    if finite_bounds:
        clip_value = None
    elif clip_value is None:
        clip_value = 20.0
    print(
        f"[dagger] action_space shape={tuple(env.action_space.shape)} "
        f"action_low shape={tuple(action_low.shape)} "
        f"action_high shape={tuple(action_high.shape)} finite_bounds={finite_bounds} "
        f"clip_value={clip_value}",
        flush=True,
    )
    input_terms = ("actor_obs", *reference_terms)
    print("[dagger] resolving hidden dims", flush=True)
    hidden_dims = tuple(
        int(value)
        for value in config.student.get("hidden_dims", (2048, 2048, 1024, 1024, 512, 512))
    )
    print(
        f"[dagger] observation_dim={observation_dim} action_dim={action_dim} "
        f"input_terms={input_terms} hidden_dims={hidden_dims}",
        flush=True,
    )
    print("[dagger] loading SONIC release teacher", flush=True)
    teacher = SonicReleaseTeacher(
        config=release_config,
        env=env,
        checkpoint=str(config.teacher_checkpoint),
        device=device,
    )
    print(
        f"[dagger] teacher ready params={teacher.parameter_count} "
        f"action_dim={teacher.action_dim}",
        flush=True,
    )
    student = PlainMlpStudent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        device=device,
        hidden_dims=hidden_dims,
        learning_rate=float(config.student.learning_rate),
        weight_decay=float(config.student.weight_decay),
        train_steps=int(config.student.train_steps_per_round),
        batch_size=int(config.student.get("batch_size", config.student.get("micro_batch", 256))),
        seed=int(config.seed),
        input_terms=input_terms,
        freeze_normalization=bool(config.student.get("freeze_normalization", False)),
        motion_keys=motion_info["loaded_keys"],
        layer_norm=bool(config.student.get("layer_norm", False)),
        dropout=float(config.student.get("dropout", 0.0)),
        lr_schedule=str(config.student.get("lr_schedule", "constant")),
        warmup_steps=int(config.student.get("warmup_steps", 0)),
        min_lr_ratio=float(config.student.get("min_lr_ratio", 0.01)),
    )
    print(
        f"[dagger] student ready params={sum(p.numel() for p in student.model.parameters())} "
        f"normalization_mode={'frozen' if student.freeze_normalization else 'recomputed'} "
        f"layer_norm={student.layer_norm} lr_schedule={student.lr_schedule}",
        flush=True,
    )
    loop = DaggerLoop(
        env=env,
        teacher=teacher,
        student=student,
        num_envs=int(config.dagger.num_envs),
        rollout_steps=int(config.dagger.rollout_steps),
        action_clip_value=clip_value,
        run_dir=run_dir,
        seed=int(config.seed),
        action_low=action_low if finite_bounds else None,
        action_high=action_high if finite_bounds else None,
    )
    eval_steps = int(config.dagger.get("eval_steps", config.dagger.rollout_steps) or config.dagger.rollout_steps)
    beta_schedule = [float(value) for value in config.dagger.beta_schedule]
    rounds = int(config.dagger.get("rounds", len(beta_schedule)) or len(beta_schedule))
    beta_schedule = (beta_schedule + [beta_schedule[-1]] * rounds)[:rounds]
    train_steps_per_round = int(config.student.train_steps_per_round)
    total_training_steps = len(beta_schedule) * train_steps_per_round
    motion_weights: dict[str, float] | None = None
    data_paths: list[Path] = []
    for index in range(len(beta_schedule)):
        data_path = run_dir / "data" / "dagger" / f"round_{index:02d}.pt"
        checkpoint_path = run_dir / "checkpoints" / f"round_{index:02d}.pt"
        if data_path.is_file() and checkpoint_path.is_file():
            data_paths.append(data_path)
        else:
            break
    start_round = len(data_paths)
    metrics: list[dict] = []
    if start_round:
        partial_path = run_dir / "manifests" / "dagger_mlp_metrics.partial.json"
        if partial_path.is_file():
            loaded_metrics = json.loads(partial_path.read_text(encoding="utf-8"))
            if isinstance(loaded_metrics, list):
                metrics = loaded_metrics[:start_round]
        if len(metrics) != start_round:
            metrics = []
            for index in range(start_round):
                log_path = run_dir / "logs" / f"round_{index:02d}.json"
                if not log_path.is_file():
                    raise RuntimeError(
                        f"cannot resume: completed round {index} has no metrics record"
                    )
                metrics.append(json.loads(log_path.read_text(encoding="utf-8")))
        if len(metrics) != start_round:
            raise RuntimeError("cannot resume: metrics count does not match completed rounds")
        loop.dataset = AggregatedDataset.load_rounds(data_paths)
        last_checkpoint = run_dir / "checkpoints" / f"round_{start_round - 1:02d}.pt"
        payload = student.load(last_checkpoint)
        if payload.get("input_terms") and tuple(payload["input_terms"]) != input_terms:
            raise ValueError("resume checkpoint input_terms do not match current contract")
        if payload.get("motion_keys") and list(payload["motion_keys"]) != motion_info["loaded_keys"]:
            raise ValueError("resume checkpoint motion_keys do not match current motion selection")
        print(
            f"[dagger] resumed from round={start_round - 1} "
            f"dataset_samples={loop.dataset.count}",
            flush=True,
        )
    if metrics:
        motion_weights = _motion_weights_from_eval(
            metrics[-1], motion_info["loaded_keys"]
        )
    rounds_per_process = int(os.environ.get("DAGGER_ROUNDS_PER_PROCESS", "0") or 0)
    max_hours = config.dagger.get("max_train_hours", None)
    max_seconds = None if max_hours is None else float(max_hours) * 3600.0
    deadline_text = os.environ.get("DAGGER_DEADLINE_EPOCH", "").strip()
    deadline_epoch = float(deadline_text) if deadline_text else None
    stopped_by_time_limit = False
    processed = 0
    for round_index, beta in enumerate(beta_schedule):
        if round_index < start_round:
            continue
        if rounds_per_process > 0 and processed >= rounds_per_process:
            break
        if max_seconds is not None and metrics:
            # Summed over resumed shards; excludes Isaac startup between shards.
            elapsed = sum(float(m.get("round_seconds", 0.0)) for m in metrics)
            if elapsed + elapsed / len(metrics) > max_seconds:
                print(
                    f"[dagger] time budget {max_hours}h: stop before round {round_index} "
                    f"(elapsed={elapsed / 3600:.2f}h, mean_round={elapsed / len(metrics) / 60:.1f}min)",
                    flush=True,
                )
                stopped_by_time_limit = True
                break
        if deadline_epoch is not None and metrics:
            mean_round = sum(float(m.get("round_seconds", 0.0)) for m in metrics) / len(metrics)
            if time.time() + mean_round > deadline_epoch:
                print(
                    f"[dagger] wall deadline {time.strftime('%H:%M', time.localtime(deadline_epoch))}: "
                    f"stop before round {round_index} (mean_round={mean_round / 60:.1f}min)",
                    flush=True,
                )
                stopped_by_time_limit = True
                break
        round_started = time.perf_counter()
        print(f"[dagger] round={round_index} beta={float(beta)} collecting", flush=True)
        collect_started = time.perf_counter()
        round_data = loop.collect_round(round_index, float(beta))
        collect_seconds = time.perf_counter() - collect_started
        print(f"[dagger] round={round_index} collected samples={round_data.count}", flush=True)
        train_started = time.perf_counter()
        round_metrics = loop.train_student(
            round_index=round_index,
            motion_weights=motion_weights,
            global_step_offset=round_index * train_steps_per_round,
            total_training_steps=total_training_steps,
        )
        train_seconds = time.perf_counter() - train_started
        round_metrics["collect_seconds"] = collect_seconds
        round_metrics["train_seconds"] = train_seconds
        round_metrics["motion_sample_weights"] = motion_weights
        print(f"[dagger] round={round_index} trained", flush=True)
        round_metrics["teacher_action_rmse_rollout"] = float(
            torch.sqrt(
                torch.mean(
                    (round_data.teacher_action - round_data.student_action).square()
                )
            ).item()
        )
        round_metrics["teacher_action_count"] = int(round_data.mixture_teacher.sum().item())
        round_metrics["student_action_count"] = int((~round_data.mixture_teacher).sum().item())
        round_metrics["teacher_action_abs_max"] = float(round_data.teacher_action.abs().max().item())
        round_metrics["student_action_abs_max"] = float(round_data.student_action.abs().max().item())
        round_metrics["executed_action_abs_max"] = float(round_data.executed_action.abs().max().item())
        ids = round_data.motion_id.cpu()
        per_motion_samples = {}
        keys = motion_info["loaded_keys"]
        for index, key in enumerate(keys):
            per_motion_samples[key] = int((ids == index).sum().item())
        if not keys:
            for value in sorted(set(int(item) for item in ids.tolist())):
                per_motion_samples[f"motion_{value}"] = int((ids == value).sum().item())
        round_metrics["per_motion_samples"] = per_motion_samples
        print(f"[dagger] round={round_index} evaluating steps={eval_steps}", flush=True)
        eval_started = time.perf_counter()
        round_metrics["student_only"] = loop.evaluate_student_only(
            eval_steps, round_index=round_index
        )
        round_metrics["eval_seconds"] = time.perf_counter() - eval_started
        motion_weights = _motion_weights_from_eval(
            round_metrics, motion_info["loaded_keys"]
        )
        round_metrics["next_motion_weights"] = motion_weights
        print(
            f"[dagger] round={round_index} eval_rmse="
            f"{round_metrics['student_only'].get('teacher_action_rmse')}",
            flush=True,
        )
        checkpoint = run_dir / "checkpoints" / f"round_{round_index:02d}.pt"
        student.save(str(checkpoint), extra_metadata={"round_index": round_index})
        round_metrics["round_seconds"] = time.perf_counter() - round_started
        metrics.append(
            {
                "round": round_index,
                "beta": float(beta),
                "samples": round_data.count,
                "checkpoint": str(checkpoint),
                **round_metrics,
            }
        )
        _write_json(run_dir / "manifests" / "dagger_mlp_metrics.partial.json", metrics)
        _write_json(run_dir / "logs" / f"round_{round_index:02d}.json", metrics[-1])
        print(
            f"[dagger] round={round_index} partial metrics written",
            flush=True,
        )
        student_only = round_metrics["student_only"]
        print(
            f"[dagger] round={round_index} beta={float(beta):.2f} "
            f"eval_rmse={float(student_only.get('teacher_action_rmse', float('nan'))):.4f} "
            f"term_rate={float(student_only.get('termination_rate', float('nan'))):.4f} "
            f"survival={float(student_only.get('survival_ratio', float('nan'))):.2f} "
            f"round_min={round_metrics['round_seconds'] / 60:.1f}",
            flush=True,
        )
        processed += 1
    all_rounds_complete = stopped_by_time_limit or (start_round + processed) >= len(beta_schedule)
    memorization_pass = same_motion_memorization_pass(metrics)
    manifest = {
        "format_version": "dagger_mlp_manifest_v1",
        "mode": str(config.mode),
        "sonic_repo_root": str(sonic_root),
        "sonic_commit": _git_head(sonic_root),
        "dagger_commit": _git_head(Path(__file__).resolve().parents[3]),
        "teacher_checkpoint": str(Path(str(config.teacher_checkpoint)).expanduser().resolve()),
        "teacher_checkpoint_sha256": teacher.checkpoint_sha256,
        "teacher_encoder": "g1",
        "teacher_parameter_count": teacher.parameter_count,
        "teacher_action_dim": teacher.action_dim,
        "student_architecture": "plain_mlp",
        "student_hidden_dims": list(student.hidden_dims),
        "student_layer_norm": student.layer_norm,
        "student_dropout": student.dropout,
        "student_lr_schedule": student.lr_schedule,
        "student_warmup_steps": student.warmup_steps,
        "student_peak_learning_rate": student.peak_learning_rate,
        "student_minimum_learning_rate": student.minimum_learning_rate,
        "motion_weighting": "termination_rate",
        "student_parameter_count": sum(parameter.numel() for parameter in student.model.parameters()),
        "student_observation_dim": observation_dim,
        "input_terms": list(input_terms),
        "reference_terms": list(reference_terms),
        "action_dim": action_dim,
        "action_space": "raw",
        "action_bounds": {
            "finite": finite_bounds,
            "clip_value": clip_value,
            "low": action_low.cpu().reshape(-1).tolist() if finite_bounds else None,
            "high": action_high.cpu().reshape(-1).tolist() if finite_bounds else None,
        },
        "normalization_mode": "frozen" if student.freeze_normalization else "recomputed_per_round",
        "motion": motion_info,
        "motion_keys_requested": motion_info["requested_keys"],
        "motion_keys_loaded": motion_info["loaded_keys"],
        "beta_schedule": [float(value) for value in config.dagger.beta_schedule],
        "num_envs": int(config.dagger.num_envs),
        "rollout_steps": int(config.dagger.rollout_steps),
        "train_steps_per_round": train_steps_per_round,
        "total_training_steps": total_training_steps,
        "eval_steps": eval_steps,
        "seed": int(config.seed),
        "resume_run": os.environ.get("DAGGER_RESUME_RUN") or None,
        "start_round": start_round,
        "rounds_per_process": rounds_per_process,
        "completed_rounds": len(metrics),
        "complete": all_rounds_complete,
        "max_train_hours": None if max_hours is None else float(max_hours),
        "deadline_epoch": deadline_epoch,
        "stopped_by_time_limit": stopped_by_time_limit,
        "rounds": metrics,
        "execution_pass": True,
        "quality_pass": None,
        "memorization_pass": memorization_pass,
        "memorization_scope": "same_motion_set",
        "quality_note": "Same-motion DAgger memorization only; no held-out or new-motion generalization claim.",
    }
    _write_json(run_dir / "manifests" / "dagger_mlp_manifest.partial.json", manifest)
    _write_json(run_dir / "manifests" / "dagger_mlp_metrics.partial.json", metrics)
    if all_rounds_complete:
        _write_json(run_dir / "manifests" / "dagger_mlp_manifest.json", manifest)
        _write_json(run_dir / "manifests" / "dagger_mlp_metrics.json", metrics)
        _write_json(run_dir / "manifests" / "dagger_manifest.json", manifest)
        _write_json(run_dir / "manifests" / "dagger_metrics.json", metrics)
        print(
            f"[dagger] complete manifest written memorization_pass={memorization_pass}",
            flush=True,
        )
        if memorization_pass:
            (run_dir / "markers" / "dagger_mlp_memorization.ok").write_text(
                "memorization_pass=true\n", encoding="utf-8"
            )
    else:
        print(
            f"[dagger] shard complete completed_rounds={len(metrics)}/{len(beta_schedule)}; "
            "final manifest deferred",
            flush=True,
        )
    return env, manifest


def _run_mlp_render(config, run_dir: Path, device: torch.device, args_cli, prepared: dict):
    import numpy as np

    from gear_sonic.envs.env_utils.joint_utils import get_body_joint_indices

    env = _create_release_env(prepared, device, args_cli)
    observation = env.reset()
    actor_obs = observation["actor_obs"]
    reference_features = observation["student_reference_features"]
    observation_dim = int(actor_obs.shape[-1] + reference_features.shape[-1])
    action_dim = int(env.action_space.shape[-1])
    input_terms = ("actor_obs", *prepared["reference_terms"])
    action_low = env.action_low
    action_high = env.action_high
    finite_bounds = bool(torch.isfinite(action_low).all() and torch.isfinite(action_high).all())
    clip_value = config.dagger.get("action_clip_value", None)
    if finite_bounds:
        clip_value = None
    elif clip_value is None:
        clip_value = 20.0
    policy_kind = str(config.dagger.get("render_policy", "mlp")).lower()
    if policy_kind not in {"mlp", "sonic", "sonic_causal"}:
        raise ValueError("render_policy must be mlp, sonic or sonic_causal")
    video_name = f"{policy_kind}_policy.mp4"
    extra_manifest: dict = {}

    def after_step(executed_action, done_value):
        return None

    if policy_kind == "sonic_causal":
        from .adapters.sonic_causal_student import SonicCausalStudentAdapter

        checkpoint = Path(str(config.student.checkpoint)).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"render checkpoint is missing: {checkpoint}")
        contract = SonicActionContract(env)
        causal = SonicCausalStudentAdapter(
            checkpoint=checkpoint,
            device=device,
            action_contract=contract,
            normalization_run=config.student.get("normalization_run"),
            seed=int(config.seed),
        )
        unseen = sorted(set(prepared["motion"]["loaded_keys"]) - set(causal.train_motion_keys))
        if unseen:
            raise ValueError(f"render motion is not among the C training motions: {unseen}")
        latent_mode = str(config.dagger.get("render_latent_mode", "fixed_standard_normal"))
        causal.set_latent_mode(latent_mode)
        actual = getattr(env.env, "env", env.env)
        previous = getattr(getattr(actual, "action_manager", None), "prev_action", None)
        initial_action = (
            contract.raw_to_canonical(previous.detach())
            if isinstance(previous, torch.Tensor) and tuple(previous.shape) == (env.num_envs, action_dim)
            else None
        )
        causal.reset(env.num_envs, env.physics_state(), initial_action=initial_action)
        if observation.get("student_reference_future") is None:
            raise RuntimeError("SONIC release observation has no [10,64] reference future")

        def policy_action(observation_value, done_value):
            return causal.observe(observation_value["student_reference_future"]).action_raw

        def after_step(executed_action, done_value):
            causal.after_step(contract.raw_to_canonical(executed_action), env.physics_state(), done_value)

        checkpoint_sha256 = causal.checkpoint_sha256
        policy_source = "sonic_causal_cvae_policy_rollout"
        extra_manifest = {
            "latent_mode": latent_mode,
            "c_train_motion_keys": list(causal.train_motion_keys),
            "architecture_version": causal.model.ARCHITECTURE_VERSION,
            "initial_action_from_env": initial_action is not None,
        }
    elif policy_kind == "mlp":
        hidden_dims = tuple(
            int(value)
            for value in config.student.get("hidden_dims", (2048, 2048, 1024, 1024, 512, 512))
        )
        checkpoint = Path(str(config.student.checkpoint)).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"render checkpoint is missing: {checkpoint}")
        student = PlainMlpStudent(
            observation_dim=observation_dim,
            action_dim=action_dim,
            device=device,
            hidden_dims=hidden_dims,
            train_steps=0,
            batch_size=1,
            seed=int(config.seed),
            input_terms=input_terms,
            freeze_normalization=True,
            motion_keys=prepared["motion"]["loaded_keys"],
            layer_norm=bool(config.student.get("layer_norm", False)),
            dropout=float(config.student.get("dropout", 0.0)),
        )
        payload = student.load(checkpoint)
        if payload.get("input_terms") and tuple(payload["input_terms"]) != input_terms:
            raise ValueError("render checkpoint input_terms do not match current contract")
        checkpoint_motions = set(payload.get("motion_keys") or [])
        selected_motions = set(prepared["motion"]["loaded_keys"])
        if checkpoint_motions and selected_motions and not selected_motions.issubset(checkpoint_motions):
            raise ValueError(
                "render motion is not among the checkpoint training motions: "
                f"{sorted(selected_motions - checkpoint_motions)}"
            )

        def policy_action(observation_value, done_value):
            return student.act(student.observe(observation_value))

        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        policy_source = "dagger_mlp_policy_rollout"
    else:
        checkpoint = Path(str(config.teacher_checkpoint)).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SONIC teacher checkpoint is missing: {checkpoint}")
        teacher = SonicReleaseTeacher(
            config=prepared["release_config"],
            env=env,
            checkpoint=str(checkpoint),
            device=device,
        )
        teacher.reset(env.num_envs)

        def policy_action(observation_value, done_value):
            return teacher.act(observation_value, done_value)

        checkpoint_sha256 = teacher.checkpoint_sha256
        policy_source = "sonic_release_policy_rollout"

    actual_env = getattr(env.env, "env", env.env)
    scene = actual_env.scene
    robot = scene["robot"]
    joint_ids = get_body_joint_indices(robot)
    if len(joint_ids) != action_dim:
        raise ValueError(f"expected {action_dim} body joints for rendering, found {len(joint_ids)}")
    fps = 1.0 / float(getattr(actual_env, "step_dt", 0.02))
    steps = int(config.dagger.get("render_steps", 192))

    def frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dof_pos = robot.data.joint_pos[:, joint_ids][0].detach().cpu().numpy().copy()
        root_pos = (
            (robot.data.root_pos_w[0] - scene.env_origins[0]).detach().cpu().numpy().copy()
        )
        root_quat = robot.data.root_quat_w[0].detach().cpu().numpy().copy()
        return dof_pos, root_pos, root_quat

    dof_frames: list[np.ndarray] = []
    root_pos_frames: list[np.ndarray] = []
    root_quat_frames: list[np.ndarray] = []
    terminated = False
    done = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    executed_max = 0.0
    for _ in range(steps):
        dof_pos, root_pos, root_quat = frame()
        dof_frames.append(dof_pos)
        root_pos_frames.append(root_pos)
        root_quat_frames.append(root_quat)
        action = policy_action(observation, done)
        if finite_bounds:
            low = action_low[..., : action.shape[-1]]
            high = action_high[..., : action.shape[-1]]
            action = torch.maximum(torch.minimum(action, high), low)
        elif clip_value is not None and clip_value > 0:
            action = torch.clamp(action, -float(clip_value), float(clip_value))
        if not torch.isfinite(action).all():
            raise FloatingPointError("non-finite render action")
        executed_max = max(executed_max, float(action.abs().max().item()))
        observation, _reward, next_done, _info = env.step({"actions": action})
        done = next_done.to(device=env.device).bool()
        after_step(action, done)
        if bool(done.any()):
            terminated = True
            break
    if not terminated:
        dof_pos, root_pos, root_quat = frame()
        dof_frames.append(dof_pos)
        root_pos_frames.append(root_pos)
        root_quat_frames.append(root_quat)

    trajectory = {
        "dof_pos": np.stack(dof_frames, axis=0),
        "root_pos_w": np.stack(root_pos_frames, axis=0),
        "root_quat_w": np.stack(root_quat_frames, axis=0),
        "quat_format": "wxyz",
        "fps": fps,
        "num_joints": int(dof_frames[0].shape[0]),
        "total_frames": len(dof_frames),
        "source": policy_source,
    }
    trajectory_path = run_dir / "data" / "000000.trajectory.pkl"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    with trajectory_path.open("wb") as handle:
        pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
    video_path = run_dir / "videos" / video_name
    manifest = {
        "format_version": "dagger_policy_render_manifest_v1",
        "mode": str(config.mode),
        "policy_kind": policy_kind,
        "sonic_repo_root": str(prepared["sonic_root"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "motion_keys": prepared["motion"]["loaded_keys"],
        "input_terms": list(input_terms),
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "action_bounds": {
            "finite": finite_bounds,
            "clip_value": clip_value,
            "low": action_low.cpu().reshape(-1).tolist() if finite_bounds else None,
            "high": action_high.cpu().reshape(-1).tolist() if finite_bounds else None,
        },
        "requested_steps": steps,
        "frames": len(dof_frames),
        "fps": fps,
        "terminated": terminated,
        "tracking_terminations_disabled": _render_disables_tracking_terminations(config),
        "executed_action_abs_max": executed_max,
        "trajectory": str(trajectory_path),
        "video": str(video_path),
        "render_kind": "isaac_rollout_then_mujoco_kinematic_render",
        "execution_pass": True,
        "quality_pass": None,
        **extra_manifest,
    }
    _write_json(run_dir / "manifests" / "dagger_policy_render_manifest.json", manifest)
    print(
        f"[dagger] render rollout written trajectory={trajectory_path} "
        f"frames={len(dof_frames)} terminated={terminated} video={video_path}",
        flush=True,
    )
    return env, manifest


def _run_cvae(config, run_dir: Path, device: torch.device, args_cli, prepared: dict):
    from .adapters.cvae_student import CvaeStudentAdapter
    from .core.cvae_loop import CvaeDaggerLoop

    sonic_root = prepared["sonic_root"]
    release_config = prepared["release_config"]
    env = _create_release_env(prepared, device, args_cli)
    print("[dagger] resetting CVAE env for rollout", flush=True)
    observation = env.reset()
    print("[dagger] CVAE env reset returned", flush=True)
    del observation
    contract = SonicActionContract(env)
    student_checkpoint = Path(str(config.student.checkpoint)).expanduser().resolve()
    model_config = Path(str(config.student.model_config)).expanduser().resolve()
    normalization_run = Path(str(config.student.normalization_run)).expanduser().resolve()
    student = CvaeStudentAdapter(
        checkpoint=student_checkpoint,
        model_config=model_config,
        normalization_run=normalization_run,
        action_contract=contract,
        device=device,
        seed=int(config.seed),
        learning_rate=float(config.student.learning_rate),
        weight_decay=float(config.student.weight_decay),
        train_steps=int(config.student.train_steps_per_round),
        batch_size=int(config.student.micro_batch),
        source_mix=float(config.student.source_mix),
        mask_seed=int(config.student.mask_seed),
        kl_beta=float(config.student.kl_beta),
        kl_warmup=int(config.student.kl_warmup),
    )
    teacher = SonicReleaseTeacher(config=release_config, env=env, checkpoint=str(config.teacher_checkpoint), device=device)
    current_state = env.physics_state()
    loop = CvaeDaggerLoop(
        env=env, teacher=teacher, student=student, action_contract=contract,
        num_envs=int(config.dagger.num_envs), rollout_steps=int(config.dagger.rollout_steps),
        run_dir=run_dir, seed=int(config.seed), state_mean=student.state_mean,
        state_std=student.state_std, action_mean=student.action_mean, action_std=student.action_std,
    )
    if tuple(current_state.shape) != (int(config.dagger.num_envs), 70):
        raise ValueError(f"SONIC physics State shape mismatch: {tuple(current_state.shape)}")
    metrics = []
    for round_index, beta in enumerate(config.dagger.beta_schedule):
        round_data = loop.collect_round(round_index, float(beta))
        train_metrics = student.train_on_dataset(loop.dataset)
        if "mean_future_state" in train_metrics:
            train_metrics["future_state_rmse_train"] = float(train_metrics["mean_future_state"] ** 0.5)
        eval_metrics = {
            mode: loop.evaluate_student_only(int(config.dagger.rollout_steps), latent_mode=mode)
            for mode in ("fixed_standard_normal", "zero", "posterior_mean")
        }
        teacher_rmse_normalized = torch.sqrt(
            torch.mean((round_data.tensors["teacher_action"] - round_data.tensors["student_action"]) ** 2)
        ).item()
        teacher_rmse_canonical = torch.sqrt(
            torch.mean(
                (round_data.tensors["teacher_action_canonical"] - round_data.tensors["student_action_canonical"]) ** 2
            )
        ).item()
        teacher_rmse_raw = torch.sqrt(
            torch.mean(
                (round_data.tensors["teacher_action_raw"] - round_data.tensors["student_action_raw"]) ** 2
            )
        ).item()
        checkpoint = run_dir / "checkpoints" / f"round_{round_index:02d}.pt"
        student.save(checkpoint, round_index=round_index)
        metrics.append({
            "round": round_index, "beta": float(beta), "samples": round_data.count,
            "normalization_out_of_range": float(round_data.tensors["normalization_out_of_range"].mean()),
            "teacher_action_rmse_normalized": teacher_rmse_normalized,
            "teacher_action_rmse_canonical": teacher_rmse_canonical,
            "teacher_action_rmse_raw": teacher_rmse_raw,
            **train_metrics, "student_only": eval_metrics,
        })
    mapping_payload = {
        "scale": contract.scale.detach().cpu().tolist(),
        "offset": contract.offset.detach().cpu().tolist(),
        "nominal": contract.nominal.detach().cpu().tolist(),
        "clip": None if contract.clip is None else contract.clip.detach().cpu().tolist(),
    }
    mapping_hash = hashlib.sha256(json.dumps(mapping_payload, sort_keys=True).encode()).hexdigest()
    information_contract = estimate_teacher_identifiability(loop.dataset)
    information_contract_pass = (
        None if not information_contract["available"] else not information_contract["teacher_not_identifiable"]
    )
    manifest = {
        "format_version": "dagger_cvae_manifest_v1", "mode": str(config.mode),
        "dagger_root": str(Path(__file__).resolve().parents[3]),
        "dagger_commit": _git_head(Path(__file__).resolve().parents[3]),
        "sonic_repo_root": str(sonic_root), "sonic_commit": _git_head(sonic_root),
        "state_action_cvae_root": str(Path(str(config.student.cvae_root)).expanduser().resolve()),
        "state_action_cvae_commit": _git_head(Path(str(config.student.cvae_root)).expanduser().resolve()),
        "teacher_checkpoint": str(Path(str(config.teacher_checkpoint)).expanduser().resolve()),
        "teacher_checkpoint_sha256": teacher.checkpoint_sha256,
        "teacher_parameter_count": teacher.parameter_count, "teacher_action_dim": teacher.action_dim,
        "student_checkpoint": str(student_checkpoint), "student_checkpoint_sha256": student.checkpoint_sha256,
        "student_checkpoint_stage": student.checkpoint_stage,
        "student_parameter_count": sum(parameter.numel() for parameter in student.model.parameters()),
        "student_architecture": student.model_signature,
        "normalization_run": str(normalization_run), "normalization_sha256": student.normalization_sha256,
        "action_mapping_sha256": mapping_hash, "teacher_encoder": "g1",
        "anchor": 32, "state_shape": [65, 70], "action_shape": [64, 29],
        "beta_schedule": [float(value) for value in config.dagger.beta_schedule],
        "num_envs": int(config.dagger.num_envs), "rollout_steps": int(config.dagger.rollout_steps),
        "disturbance": "level0_4", "observation_corruption": False,
        "rounds": metrics, "execution_pass": True,
        "information_contract": information_contract,
        "information_contract_pass": information_contract_pass,
        "quality_pass": None,
        "quality_note": "No-reference CVAE is a dynamic-completion diagnostic; teacher action identifiability requires a separate probe.",
    }
    _write_json(run_dir / "manifests" / "dagger_cvae_manifest.json", manifest)
    _write_json(run_dir / "manifests" / "dagger_cvae_metrics.json", metrics)
    return env, manifest


def _select_sonic_causal_motions(config, checkpoint=None) -> list[str]:
    """Pin DAgger motions to keys the C-stage student actually trained on."""

    import random

    checkpoint = Path(str(checkpoint or config.student.checkpoint)).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    seen = sorted(str(key) for key in payload.get("train_motion_keys") or [])
    if not seen:
        raise ValueError(
            f"{checkpoint} has no train_motion_keys; retrain C with the current sonic_causal_train"
        )
    requested = [str(key) for key in (config.dagger.get("motion_keys", []) or [])]
    if requested:
        unseen = sorted(set(requested) - set(seen))
        if unseen:
            raise ValueError(f"DAgger motion keys were not seen in C training: {unseen}")
    else:
        count = int(config.dagger.get("motion_count", 0) or 0)
        if count <= 0 or count > len(seen):
            raise ValueError(
                f"motion_count={count} must be in [1, {len(seen)}] C-trained motions"
            )
        seed = int(config.dagger.get("motion_seed", int(config.seed)))
        requested = random.Random(seed).sample(seen, count)
    config.dagger.motion_keys = requested
    config.dagger.max_unique_motions = len(requested)
    print(
        f"[dagger] sonic_causal motions={requested} (from {len(seen)} C-trained motions)",
        flush=True,
    )
    return seen


def _run_sonic_causal(config, run_dir: Path, device: torch.device, args_cli, prepared: dict):
    """Run the independent 10-history/10-reference single-step DAgger path."""

    from .adapters.sonic_causal_student import SonicCausalStudentAdapter
    from .core.sonic_causal_loop import SonicCausalDaggerLoop
    from cvae_sa.sonic_causal_dataset import SonicCausalDataset

    env = _create_release_env(prepared, device, args_cli)
    try:
        observation = env.reset()
        current_state = env.physics_state()
        if observation.get("student_reference_future") is None:
            raise RuntimeError("SONIC release observation has no [10,64] reference future")
        contract = SonicActionContract(env)
        checkpoint = Path(str(config.student.checkpoint)).expanduser().resolve()
        student = SonicCausalStudentAdapter(
            checkpoint=checkpoint,
            device=device,
            action_contract=contract,
            normalization_run=config.student.get("normalization_run"),
            seed=int(config.seed),
            learning_rate=float(config.student.get("learning_rate", 1.0e-5)),
            weight_decay=float(config.student.get("weight_decay", 0.0)),
            action_clip_value=config.student.get("action_clip_value", 20.0),
            kl_beta=float(config.student.get("kl_beta", 0.002)),
        )
        loaded_motions = [str(key) for key in (getattr(env, "motion_keys", None) or [])]
        unseen = sorted(set(loaded_motions) - set(student.train_motion_keys))
        if not loaded_motions or unseen:
            raise RuntimeError(
                f"DAgger env loaded motions not seen in C training: {unseen or 'none loaded'}"
            )
        teacher = SonicReleaseTeacher(
            config=prepared["release_config"],
            env=env,
            checkpoint=str(config.teacher_checkpoint),
            device=device,
        )
        source_dataset = SonicCausalDataset(
            config.student.source_dataset_run,
            split="train",
            max_samples=config.student.get("source_max_samples"),
        )
        loop = SonicCausalDaggerLoop(
            env=env,
            teacher=teacher,
            student=student,
            action_contract=contract,
            num_envs=int(config.dagger.num_envs),
            rollout_steps=int(config.dagger.rollout_steps),
            run_dir=run_dir,
            seed=int(config.seed),
        )
        beta_schedule = [float(value) for value in config.dagger.beta_schedule]
        rounds = int(config.dagger.get("rounds", len(beta_schedule)) or len(beta_schedule))
        if len(beta_schedule) < rounds:
            beta_schedule.extend([beta_schedule[-1]] * (rounds - len(beta_schedule)))
        beta_schedule = beta_schedule[:rounds]
        max_hours = config.dagger.get("max_train_hours", None)
        max_seconds = None if max_hours is None else float(max_hours) * 3600.0
        started = time.monotonic()
        stopped_by_time_limit = False
        metrics = []
        for round_index, beta in enumerate(beta_schedule):
            if max_seconds is not None and round_index > 0:
                elapsed = time.monotonic() - started
                if elapsed + elapsed / round_index > max_seconds:
                    print(
                        f"[sonic-causal] time budget {max_hours}h: stop before round {round_index} "
                        f"(elapsed={elapsed / 3600:.2f}h, mean_round={elapsed / round_index / 60:.1f}min)",
                        flush=True,
                    )
                    stopped_by_time_limit = True
                    break
            dynamic = loop.collect_round(round_index, beta)
            data_path = run_dir / "data" / "dagger" / f"round_{round_index:02d}.pt"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data_tmp = data_path.with_name(f"{data_path.name}.tmp.{__import__('os').getpid()}")
            torch.save(
                {
                    "format_version": "sonic_causal_dagger_samples_v1",
                    "round_index": int(round_index),
                    "beta": float(beta),
                    "architecture_version": student.model.ARCHITECTURE_VERSION,
                    "samples": dynamic,
                },
                data_tmp,
            )
            data_tmp.replace(data_path)
            train_metrics = student.train_on_samples(
                loop.samples,
                source_dataset=source_dataset,
                steps=int(config.student.get("train_steps_per_round", 1000)),
                batch_size=int(config.student.get("micro_batch", 32)),
                source_mix=float(config.student.get("source_mix", 0.5)),
                generator=torch.Generator(device="cpu").manual_seed(
                    int(config.seed) + round_index * 1009
                ),
            )
            eval_metrics = loop.evaluate_student_only(
                int(config.dagger.get("eval_steps", 192))
            )
            checkpoint_path = run_dir / "checkpoints" / f"round_{round_index:02d}.pt"
            student.save(
                checkpoint_path,
                round_index=round_index,
                metadata={
                    "beta": beta,
                    "motion_keys": list(getattr(env, "motion_keys", []) or []),
                    "eval_metrics": eval_metrics,
                },
            )
            metrics.append(
                {
                    "round": round_index,
                    "beta": beta,
                    "round_samples": len(dynamic),
                    "samples": len(loop.samples),
                    "train": train_metrics,
                    "student_only": eval_metrics,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            _write_json(run_dir / "manifests" / "sonic_causal_metrics.partial.json", metrics)
            print(
                f"[sonic-causal] round={round_index} beta={beta:.2f} samples={len(loop.samples)} "
                f"train_loss={train_metrics['train_loss']:.4g} "
                f"eval_rmse={eval_metrics['teacher_action_rmse']:.4f} "
                f"term_rate={eval_metrics['termination_rate']:.4f} "
                f"survival={eval_metrics['survival_ratio']:.2f} "
                f"elapsed={(time.monotonic() - started) / 3600:.2f}h",
                flush=True,
            )
        baseline = metrics[0]["student_only"] if metrics else None
        final = metrics[-1]["student_only"] if metrics else None
        baseline_by_motion = (baseline or {}).get("teacher_action_rmse_by_motion", {})
        final_by_motion = (final or {}).get("teacher_action_rmse_by_motion", {})
        motion_ratios = {
            key: float(final_by_motion[key]["teacher_action_rmse"])
            / max(float(baseline_by_motion[key]["teacher_action_rmse"]), 1.0e-12)
            for key in sorted(set(baseline_by_motion) & set(final_by_motion))
        }
        motion_pass_count = sum(value <= 0.7 for value in motion_ratios.values())
        expected_motion_count = int(config.dagger.get("motion_count", 4))
        gate = {
            "pass": bool(
                baseline
                and final
                and float(final["teacher_action_rmse"]) <= 0.5 * float(baseline["teacher_action_rmse"])
                and expected_motion_count >= 4
                and motion_pass_count >= 3
                and float(final["termination_rate"]) <= float(baseline["termination_rate"])
                and float(final["action_saturation_ratio"]) < 0.05
            ),
            "scope": "same_motion_teacher_action_imitation",
            "overall_rmse_ratio": None
            if not baseline or not final
            else float(final["teacher_action_rmse"])
            / max(float(baseline["teacher_action_rmse"]), 1.0e-12),
            "per_motion_rmse_ratio": motion_ratios,
            "per_motion_pass_count": int(motion_pass_count),
            "per_motion_required": 3,
            "motion_gate_complete": expected_motion_count >= 4,
        }
        manifest = {
            "format_version": "sonic_causal_dagger_manifest_v1",
            "mode": str(config.mode),
            "execution_pass": True,
            "quality_pass": gate["pass"],
            "quality_scope": "same_motion_teacher_action_imitation",
            "architecture_version": student.model.ARCHITECTURE_VERSION,
            "history_state_shape": [10, 70],
            "history_action_shape": [10, 29],
            "reference_shape": [10, 64],
            "action_shape": [29],
            "dagger_commit": _git_head(Path(__file__).resolve().parents[3]),
            "sonic_commit": _git_head(prepared["sonic_root"]),
            "teacher_checkpoint": str(Path(str(config.teacher_checkpoint)).expanduser().resolve()),
            "teacher_checkpoint_sha256": teacher.checkpoint_sha256,
            "c_train_motion_keys": list(student.train_motion_keys),
            "dagger_motion_keys": loaded_motions,
            "student_checkpoint": str(checkpoint),
            "student_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "source_dataset_run": str(source_dataset.dataset_run),
            "source_dataset_manifest_sha256": hashlib.sha256(
                (source_dataset.dataset_run / "manifests" / "dataset_manifest.json").read_bytes()
            ).hexdigest(),
            "data_root": str(run_dir / "data" / "dagger"),
            "sample_format": "sonic_causal_dagger_samples_v1",
            "rounds": metrics,
            "max_train_hours": max_hours,
            "stopped_by_time_limit": stopped_by_time_limit,
            "imitation_gate": gate,
            "teacher_information_contract": {
                "teacher_input_keys": sorted(
                    str(key) for key in observation.keys() if not str(key).startswith("student_")
                ),
                "excluded_student_keys": sorted(
                    str(key) for key in observation.keys() if str(key).startswith("student_")
                ),
                "reference_key": "student_reference_future",
                "reference_shape": [10, 64],
                "teacher_receives_student_reference": False,
            },
            "information_contract_pass": True,
            "quality_note": "Same-motion DAgger imitation only; no new-motion generalization claim.",
        }
        _write_json(run_dir / "manifests" / "sonic_causal_manifest.json", manifest)
        (run_dir / "markers" / "sonic_causal_execution.ok").write_text(
            "execution_pass=true\n", encoding="utf-8"
        )
        if gate["pass"]:
            (run_dir / "markers" / "sonic_causal_imitation.ok").write_text(
                "quality_pass=true\n", encoding="utf-8"
            )
        return env, manifest
    except BaseException:
        _close_cvae_env(env)
        raise


def _load_kl2_motion_manifest(config) -> dict:
    raw = config.dagger.get("motion_manifest", None)
    if raw is None:
        raise ValueError("cvae_kl2 requires dagger.motion_manifest")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[2] / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CVAE KL2 motion manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    keys = [str(value) for value in manifest.get("motion_keys", [])]
    groups = [
        [str(value) for value in group]
        for group in manifest.get("motion_groups", [])
    ]
    pilot = [str(value) for value in manifest.get("pilot_motion_keys", [])]
    if len(keys) != 32:
        raise ValueError(f"expected 32 aligned motion keys, found {len(keys)}")
    if not groups or any(len(group) != 8 for group in groups):
        raise ValueError("motion_groups must contain 8-key groups")
    flattened = [key for group in groups for key in group]
    if sorted(flattened) != sorted(keys):
        raise ValueError("motion_groups do not partition the 32 motion keys")
    if not pilot or not set(pilot).issubset(set(keys)):
        raise ValueError("pilot_motion_keys must be a non-empty subset of motion_keys")
    motion_file = config.dagger.get("motion_file", None)
    if motion_file is not None:
        motion_dir = Path(str(motion_file)).expanduser().resolve()
        if motion_dir.is_dir():
            missing = [
                key for key in keys if not (motion_dir / f"{key}.pkl").is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"motion files are missing from {motion_dir}: {missing}"
                )
            manifest["_motion_file"] = str(motion_dir)
    manifest["_path"] = str(path)
    return manifest


def _kl2_keys_for_round(mode: str, manifest: dict, round_index: int) -> list[str]:
    if "smoke" in mode:
        return [str(manifest["pilot_motion_keys"][0])]
    if "pilot" in mode:
        return [str(value) for value in manifest["pilot_motion_keys"]]
    groups = manifest["motion_groups"]
    return [str(value) for value in groups[int(round_index) % len(groups)]]


def _create_cvae_env_for_keys(
    config,
    prepared: dict,
    run_dir: Path,
    keys: list[str],
    args_cli,
    device: torch.device,
):
    motion = prepared["motion"]
    release_config = compose_release_config(
        prepared["sonic_root"],
        num_envs=len(keys),
        seed=int(config.seed),
        motion_file=motion["motion_file"],
        smpl_motion_file=motion["smpl_motion_file"],
        motion_keys=keys,
        max_unique_motions=len(keys),
        sample_unique_motions=True,
        observation_corruption=motion["observation_corruption"],
        events=motion["events"],
        terminations=motion["terminations"],
    )
    release_config.manager_env.config.experiment_dir = str(run_dir)
    release_config.manager_env.config.save_rendering_dir = str(run_dir / "renderings")
    args_cli.num_envs = len(keys)
    env = create_release_env(
        release_config,
        device=str(device),
        args_cli=args_cli,
        reference_terms=prepared["reference_terms"],
    )
    if int(env.num_envs) != len(keys):
        raise RuntimeError(
            f"created env has {env.num_envs} envs, expected {len(keys)}"
        )
    contract = SonicActionContract(env)
    teacher = SonicReleaseTeacher(
        config=release_config,
        env=env,
        checkpoint=str(config.teacher_checkpoint),
        device=device,
    )
    return env, contract, teacher


def _check_cvae_action_mapping(student, contract) -> dict:
    """Compare the live env Action mapping with the recorded CVAE dataset."""

    dataset = getattr(student, "_source_dataset", None)
    if dataset is None:
        return {"available": False, "pass": None, "reason": "no source dataset"}
    source = getattr(dataset, "source", dataset)
    windows = getattr(dataset, "windows", None)
    if windows:
        episode_index = int(windows[0]["episode_index"])
    else:
        episodes = getattr(source, "episodes", None) or []
        if not episodes:
            return {"available": False, "pass": None, "reason": "no source episodes"}
        episode_index = 0
    if not hasattr(source, "physics_action_mapping"):
        return {"available": False, "pass": None, "reason": "mapping API unavailable"}
    mapping = source.physics_action_mapping(episode_index)
    recorded = {
        "nominal": torch.as_tensor(mapping["default"], dtype=torch.float32),
        "scale": torch.as_tensor(mapping["scale"], dtype=torch.float32),
        "offset": torch.as_tensor(mapping["offset"], dtype=torch.float32),
    }
    recorded_clip = mapping.get("clip")
    live = {
        "nominal": contract.nominal[0].detach().cpu().float(),
        "scale": contract.scale[0].detach().cpu().float(),
        "offset": contract.offset[0].detach().cpu().float(),
    }
    live_clip = None if contract.clip is None else contract.clip[0].detach().cpu().float()
    checks: dict[str, dict] = {}
    tolerances = {
        # add_joint_default_pos samples the action-term offset again at startup.
        # The model's canonical Action is relative to the original nominal, so
        # the offset may differ per episode; two independent startup samples can
        # differ by at most the configured [-0.01, 0.01] range.
        "nominal": 1.0e-6,
        "scale": 1.0e-6,
        "offset": 3.0e-2,
    }
    for name in ("nominal", "scale", "offset"):
        difference = float((recorded[name] - live[name]).abs().max().item())
        tolerance = tolerances[name]
        checks[name] = {
            "max_abs": difference,
            "tolerance": tolerance,
            "pass": bool(difference <= tolerance),
        }
    if recorded_clip is None and live_clip is None:
        checks["clip"] = {"max_abs": 0.0, "pass": True}
    elif recorded_clip is not None and live_clip is not None:
        difference = float(
            (
                torch.as_tensor(recorded_clip, dtype=torch.float32)
                - live_clip
            )
            .abs()
            .max()
            .item()
        )
        checks["clip"] = {
            "max_abs": difference,
            "tolerance": 1.0e-6,
            "pass": bool(difference <= 1.0e-6),
        }
    else:
        checks["clip"] = {
            "max_abs": None,
            "pass": False,
            "reason": "clip presence mismatch",
        }
    passed = all(bool(item.get("pass")) for item in checks.values())
    result = {
        "available": True,
        "pass": passed,
        "episode_index": episode_index,
        "checks": checks,
        "note": (
            "scale/nominal/clip are strict; action offset is resampled by the "
            "startup default-joint event and does not change canonical Action "
            "semantics."
        ),
    }
    if not passed:
        raise ValueError(f"CVAE Action mapping mismatch: {checks}")
    return result


def _close_cvae_env(env) -> None:
    if env is None:
        return
    try:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def _cvae_kl2_resume_state(run_dir: Path):
    data_dir = run_dir / "data" / "dagger"
    checkpoint_dir = run_dir / "checkpoints"
    completed = []
    round_index = 0
    while True:
        data_path = data_dir / f"round_{round_index:02d}.pt"
        checkpoint_path = checkpoint_dir / f"round_{round_index:02d}.pt"
        if data_path.is_file() and checkpoint_path.is_file():
            completed.append((round_index, data_path, checkpoint_path))
            round_index += 1
        else:
            break
    metrics_path = run_dir / "manifests" / "dagger_cvae_kl2_metrics.partial.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else []
    if len(metrics) < len(completed):
        for round_index, _data_path, checkpoint_path in completed[len(metrics) :]:
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            metrics.append(
                {
                    "round": int(round_index),
                    "beta": payload.get("beta"),
                    "motion_keys": list(payload.get("motion_keys") or []),
                    "samples": payload.get("sample_count"),
                    "teacher_executed_ratio": None,
                    "student_executed_ratio": None,
                    "history_steps_mean": None,
                    "train": None,
                    "student_only": payload.get("eval_metrics") or {},
                    "motion_coverage": payload.get("motion_coverage") or {},
                    "reconstructed_from_checkpoint": True,
                }
            )
    return completed, metrics


def _cvae_kl2_partial_round(run_dir: Path) -> dict | None:
    path = run_dir / "manifests" / "dagger_cvae_kl2_round_progress.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _merge_cvae_train_chunks(chunks: list[dict]) -> dict:
    if not chunks:
        return {}
    merged = dict(chunks[-1])
    merged["source_steps"] = sum(
        float(chunk.get("source_steps", 0.0)) for chunk in chunks
    )
    merged["dagger_steps"] = sum(
        float(chunk.get("dagger_steps", 0.0)) for chunk in chunks
    )
    merged["chunk_metrics"] = chunks
    merged["chunks"] = len(chunks)
    return merged


def _cvae_group_reference_rmse(
    metrics: dict | None, keys: list[str]
) -> float | None:
    if not metrics:
        return None
    by_motion = metrics.get("teacher_action_rmse_by_motion", {})
    values = []
    for key in keys:
        value = by_motion.get(str(key))
        if value is None:
            continue
        score = float(value.get("teacher_action_rmse", 0.0))
        if score > 0.0:
            values.append(score)
    if values:
        return sum(values) / len(values)
    overall = metrics.get("teacher_action_rmse")
    return float(overall) if overall is not None and float(overall) > 0.0 else None


def _cvae_motion_weights(eval_metrics: dict | None) -> dict[str, float] | None:
    by_motion = (eval_metrics or {}).get("teacher_action_rmse_by_motion", {})
    rates = {
        str(key): float(value.get("termination_rate", 0.0))
        for key, value in by_motion.items()
    }
    maximum = max(rates.values()) if rates else 0.0
    if maximum <= 0.0:
        return None
    return {key: 1.0 + 3.0 * rate / maximum for key, rate in rates.items()}


def _merge_cvae_eval_metrics(results: list[dict]) -> dict | None:
    if not results:
        return None
    weights = [
        float(result.get("steps", 0.0)) * max(1.0, float(result.get("num_envs", 1.0)))
        for result in results
    ]
    total_weight = sum(weights) or 1.0
    merged_by_motion: dict[str, dict] = {}
    for result in results:
        for key, value in result.get("teacher_action_rmse_by_motion", {}).items():
            merged_by_motion[str(key)] = value

    def weighted(key: str) -> float:
        return sum(
            float(result.get(key, 0.0)) * weight
            for result, weight in zip(results, weights)
        ) / total_weight

    return {
        "steps": max(float(result.get("steps", 0.0)) for result in results),
        "num_envs": sum(int(result.get("num_envs", 0)) for result in results),
        "done_count": sum(float(result.get("done_count", 0.0)) for result in results),
        "mean_reward": weighted("mean_reward"),
        "teacher_action_rmse": weighted("teacher_action_rmse"),
        "teacher_action_rmse_raw": weighted("teacher_action_rmse_raw"),
        "teacher_action_rmse_normalized": weighted("teacher_action_rmse_normalized"),
        "teacher_action_rmse_by_motion": merged_by_motion,
        "termination_rate": weighted("termination_rate"),
        "mean_episode_length": weighted("mean_episode_length"),
        "max_episode_length": max(
            float(result.get("max_episode_length", 0.0)) for result in results
        ),
        "survival_ratio": weighted("survival_ratio"),
        "action_saturation_ratio": weighted("action_saturation_ratio"),
        "student_action_abs_max": max(
            float(result.get("student_action_abs_max", 0.0)) for result in results
        ),
        "teacher_action_abs_max": max(
            float(result.get("teacher_action_abs_max", 0.0)) for result in results
        ),
        "executed_action_abs_max": max(
            float(result.get("executed_action_abs_max", 0.0)) for result in results
        ),
        "action_clip_error_max": max(
            float(result.get("action_clip_error_max", 0.0)) for result in results
        ),
        "grouped_eval": True,
        "group_count": len(results),
    }


def _cvae_imitation_gate(
    baseline: dict | None,
    final: dict | None,
    *,
    required_motions: int,
) -> dict:
    if not baseline or not final:
        return {"pass": False, "reason": "missing baseline or final metrics"}
    baseline_rmse = baseline.get("teacher_action_rmse")
    final_rmse = final.get("teacher_action_rmse")
    if baseline_rmse is None or final_rmse is None or float(baseline_rmse) <= 0.0:
        return {"pass": False, "reason": "invalid teacher RMSE"}
    overall_ratio = float(final_rmse) / float(baseline_rmse)
    baseline_by = baseline.get("teacher_action_rmse_by_motion", {})
    final_by = final.get("teacher_action_rmse_by_motion", {})
    per_motion = {}
    improved = 0
    for key, baseline_value in baseline_by.items():
        final_value = final_by.get(key)
        if final_value is None:
            continue
        base = float(baseline_value.get("teacher_action_rmse", 0.0))
        value = float(final_value.get("teacher_action_rmse", 0.0))
        ratio = value / base if base > 0.0 else float("inf")
        passed = bool(base > 0.0 and ratio <= 0.7)
        if passed:
            improved += 1
        per_motion[key] = {
            "baseline_teacher_action_rmse": base,
            "final_teacher_action_rmse": value,
            "ratio": ratio,
            "pass": passed,
        }
    termination_ok = float(final.get("termination_rate", 1.0)) <= float(
        baseline.get("termination_rate", 0.0)
    ) + 1.0e-9
    saturation_ok = float(final.get("action_saturation_ratio", 1.0)) < 0.05
    passed = bool(
        overall_ratio <= 0.5
        and improved >= int(required_motions)
        and termination_ok
        and saturation_ok
    )
    return {
        "pass": passed,
        "overall_ratio": overall_ratio,
        "overall_rule": "final <= 0.5 * baseline",
        "required_motion_improvements": int(required_motions),
        "motion_improvements": improved,
        "per_motion_rule": "final <= 0.7 * baseline",
        "termination_ok": termination_ok,
        "saturation_ok": saturation_ok,
        "per_motion": per_motion,
    }


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(
        error
    ).lower()


def _evaluate_cvae_full(
    config,
    prepared: dict,
    run_dir: Path,
    args_cli,
    device: torch.device,
    student,
    manifest: dict,
    dataset,
    eval_steps: int,
    *,
    label: str,
) -> dict:
    from .core.cvae_loop import CvaeDaggerLoop

    keys = [str(value) for value in manifest["motion_keys"]]

    def evaluate(env, contract, teacher, motion_group: list[str]) -> dict:
        student.action_contract = contract
        student.set_action_clip_value(config.student.get("action_clip_value", 20.0))
        loop = CvaeDaggerLoop(
            env=env,
            teacher=teacher,
            student=student,
            action_contract=contract,
            num_envs=int(env.num_envs),
            rollout_steps=int(eval_steps),
            run_dir=run_dir,
            seed=int(config.seed),
            state_mean=student.state_mean,
            state_std=student.state_std,
            action_mean=student.action_mean,
            action_std=student.action_std,
            dataset=dataset,
            motion_group=motion_group,
        )
        return loop.evaluate_student_only(
            int(eval_steps), latent_mode="fixed_standard_normal"
        )

    try:
        env, contract, teacher = _create_cvae_env_for_keys(
            config, prepared, run_dir, keys, args_cli, device
        )
        try:
            metrics = evaluate(env, contract, teacher, keys)
            return {"label": label, "mode": "32env", "fallback": False, "metrics": metrics}
        finally:
            _close_cvae_env(env)
    except Exception as error:
        groups = []
        for group in manifest["motion_groups"]:
            motion_group = [str(value) for value in group]
            env, contract, teacher = _create_cvae_env_for_keys(
                config, prepared, run_dir, motion_group, args_cli, device
            )
            try:
                groups.append(evaluate(env, contract, teacher, motion_group))
            finally:
                _close_cvae_env(env)
        return {
            "label": label,
            "mode": "4x8_fallback",
            "fallback": True,
            "oom": _is_oom(error),
            "reason": str(error),
            "groups": groups,
            "metrics": _merge_cvae_eval_metrics(groups),
        }


def _run_cvae_kl2(config, run_dir: Path, device: torch.device, args_cli, prepared: dict):
    from .adapters.cvae_student import CvaeStudentAdapter
    from .core.cvae_dataset import CvaeDaggerDataset
    from .core.cvae_loop import CvaeDaggerLoop

    manifest = _load_kl2_motion_manifest(config)
    mode = str(config.mode)
    early_stop_marker = run_dir / "markers" / "dagger_cvae_kl2_early_stop.ok"
    if early_stop_marker.is_file():
        early_stop_path = (
            run_dir / "manifests" / "dagger_cvae_kl2_early_stop.json"
        )
        payload = (
            json.loads(early_stop_path.read_text(encoding="utf-8"))
            if early_stop_path.is_file()
            else {}
        )
        return None, {
            "marker": "dagger_cvae_kl2_early_stop.ok",
            "complete": False,
            "early_stop": payload,
        }
    smoke = "smoke" in mode
    pilot = "pilot" in mode
    formal = not smoke and not pilot
    beta_schedule = [float(value) for value in config.dagger.beta_schedule]
    if not beta_schedule:
        raise ValueError("beta_schedule must not be empty")
    requested_rounds = int(
        config.dagger.get("rounds", len(beta_schedule)) or len(beta_schedule)
    )
    if requested_rounds <= 0:
        raise ValueError("dagger.rounds must be positive")
    if requested_rounds <= len(beta_schedule):
        beta_schedule = beta_schedule[:requested_rounds]
    else:
        extension = str(config.dagger.get("beta_schedule_extension", "hold_last"))
        if extension == "hold_last":
            beta_schedule = beta_schedule + [beta_schedule[-1]] * (
                requested_rounds - len(beta_schedule)
            )
        elif extension == "repeat":
            repeats = (requested_rounds + len(beta_schedule) - 1) // len(beta_schedule)
            beta_schedule = (beta_schedule * repeats)[:requested_rounds]
        else:
            raise ValueError("beta_schedule_extension must be hold_last or repeat")
    early_stop_config = config.dagger.get("early_stop", None)
    early_stop_enabled = bool(
        early_stop_config.get("enabled", False)
    ) if early_stop_config else False
    early_stop_rounds = {
        int(value)
        for value in (
            early_stop_config.get("check_rounds", []) if early_stop_config else []
        )
    }
    early_stop_baseline_round = int(
        early_stop_config.get("baseline_round", 0)
    ) if early_stop_config else 0
    early_stop_min_ratio = float(
        early_stop_config.get("min_teacher_improvement_ratio", 0.98)
    ) if early_stop_config else 0.98
    early_stop_term_tol = float(
        early_stop_config.get("termination_tolerance", 0.0)
    ) if early_stop_config else 0.0
    if early_stop_enabled and not early_stop_rounds:
        raise ValueError("early_stop.check_rounds must not be empty")
    if early_stop_enabled and any(value < 0 for value in early_stop_rounds):
        raise ValueError("early_stop.check_rounds must be non-negative")
    train_steps_per_round = int(config.student.train_steps_per_round)
    train_chunk_steps = int(config.student.get("train_chunk_steps", 0) or 0)
    if train_chunk_steps <= 0 or train_chunk_steps >= train_steps_per_round:
        train_chunk_steps = train_steps_per_round
    chunks_total = (
        train_steps_per_round + train_chunk_steps - 1
    ) // train_chunk_steps
    eval_steps = int(config.dagger.get("eval_steps", 192))
    rounds_per_process = int(os.environ.get("DAGGER_ROUNDS_PER_PROCESS", "0") or 0)
    if rounds_per_process != 1:
        print(
            "[dagger] forcing one round per process for CVAE KL2; "
            "use run_dagger_auto.sh for multi-round execution",
            flush=True,
        )
        rounds_per_process = 1
    resume = bool(os.environ.get("DAGGER_RESUME_RUN", "").strip())
    completed, metrics = _cvae_kl2_resume_state(run_dir) if resume else ([], [])
    start_round = len(completed)
    total_rounds = len(beta_schedule)
    if start_round > total_rounds:
        raise ValueError("resume has more completed rounds than beta_schedule")
    if start_round and len(metrics) < start_round:
        raise RuntimeError(
            "partial metrics are missing for completed rounds; refusing to resume "
            "with an incomplete gate history"
        )
    metrics = metrics[:start_round]
    if start_round >= total_rounds:
        full_manifest = run_dir / "manifests" / "dagger_cvae_kl2_manifest.json"
        if full_manifest.is_file():
            return None, {
                "marker": "dagger_cvae_kl2_execution.ok",
                "complete": True,
                "manifest": str(full_manifest),
            }
        raise RuntimeError("all rounds completed but final manifest is missing")
    end_round = (
        total_rounds
        if rounds_per_process <= 0
        else min(total_rounds, start_round + rounds_per_process)
    )
    print(
        f"[dagger] cvae shard start_round={start_round} "
        f"end_round={end_round} total_rounds={total_rounds} "
        f"rounds_per_process={rounds_per_process}",
        flush=True,
    )
    dataset = (
        CvaeDaggerDataset.load_rounds([item[1] for item in completed])
        if completed
        else CvaeDaggerDataset()
    )
    student = CvaeStudentAdapter(
        checkpoint=config.student.checkpoint,
        model_config=config.student.model_config,
        normalization_run=config.student.normalization_run,
        action_contract=None,
        device=device,
        cvae_root=config.student.get("cvae_root", None),
        seed=int(config.seed),
        learning_rate=float(config.student.learning_rate),
        weight_decay=float(config.student.weight_decay),
        train_steps=train_steps_per_round,
        batch_size=int(config.student.micro_batch),
        source_mix=float(config.student.source_mix),
        source_mix_mode=str(config.student.get("source_mix_mode", "random")),
        mask_seed=int(config.student.get("mask_seed", 20261024)),
        kl_beta=float(config.student.kl_beta),
        kl_warmup=int(config.student.kl_warmup),
        source_dataset_run=config.student.get(
            "source_dataset_run", config.student.normalization_run
        ),
        source_dataset_kind=str(
            config.student.get("source_dataset_kind", "long_horizon")
        ),
        source_mask_mode=str(config.student.get("source_mask_mode", "fixed")),
        source_causal_fraction=float(
            config.student.get("source_causal_fraction", 0.25)
        ),
        teacher_mode=str(config.student.get("teacher_mode", "dual")),
        source_identity=manifest,
        source_cache_size=int(config.student.get("source_cache_size", 0)),
        source_cache_path=(
            run_dir / "data" / "source_cache.pt"
            if int(config.student.get("source_cache_size", 0)) > 0
            else None
        ),
        state_weight=float(config.student.get("state_weight", 1.0)),
        contact_weight=float(config.student.get("contact_weight", 0.1)),
        teacher_sn_weight=float(config.student.get("teacher_sn_weight", 0.5)),
        teacher_pm_weight=float(config.student.get("teacher_pm_weight", 0.5)),
        action_clip_value=config.student.get("action_clip_value", 20.0),
        lr_schedule=str(config.student.get("lr_schedule", "cosine")),
        warmup_steps=int(config.student.get("warmup_steps", 500)),
        min_lr_ratio=float(config.student.get("min_lr_ratio", 0.01)),
        load_optimizer=False,
        raise_on_clip_error=bool(
            config.student.get("raise_on_clip_error", False)
        ),
        future_reference=bool(config.student.get("future_reference", False)),
    )
    if completed:
        student.load_round_checkpoint(completed[-1][2], load_optimizer=True)
    source_contract = student.source_identity.get("training_contract") or {}
    source_mask_mode = source_contract.get("mask_mode")
    if source_mask_mode and str(source_mask_mode) != str(
        config.student.get("source_mask_mode", "fixed")
    ):
        raise ValueError(
            "configured source_mask_mode differs from the KL2 checkpoint: "
            f"{config.student.get('source_mask_mode')} != {source_mask_mode}"
        )
    source_beta = source_contract.get("kl_beta")
    if source_beta is not None and abs(
        float(source_beta) - float(config.student.kl_beta)
    ) > 1.0e-12:
        raise ValueError(
            "configured KL beta differs from the KL2 checkpoint: "
            f"{config.student.kl_beta} != {source_beta}"
        )
    if float(config.student.source_mix) > 0:
        student._load_source_dataset()
    round_progress_path = (
        run_dir / "manifests" / "dagger_cvae_kl2_round_progress.json"
    )
    partial = _cvae_kl2_partial_round(run_dir)
    resume_partial = False
    partial_start_chunk = 0
    partial_train_chunks: list[dict] = []
    partial_keys: list[str] = []
    partial_beta = float(beta_schedule[start_round])
    partial_motion_weights: dict[str, float] | None = None
    partial_data_path = run_dir / "data" / "dagger" / f"round_{start_round:02d}.pt"
    partial_checkpoint_path = (
        run_dir / "checkpoints" / f"round_{start_round:02d}_partial.pt"
    )
    partial_global_step = start_round * train_steps_per_round
    if partial:
        partial_round = int(partial.get("round_index", -1))
        if partial_round < start_round:
            round_progress_path.unlink(missing_ok=True)
            leftover = Path(str(partial.get("checkpoint_path", "")))
            if leftover.is_file():
                leftover.unlink()
            partial = None
        elif partial_round > start_round:
            raise RuntimeError(
                "round progress points past the next uncompleted round: "
                f"{partial_round} > {start_round}"
            )
        else:
            partial_data_path = Path(
                str(partial.get("data_path", partial_data_path))
            ).expanduser()
            partial_checkpoint_path = Path(
                str(partial.get("checkpoint_path", partial_checkpoint_path))
            ).expanduser()
            if not partial_data_path.is_file() or not partial_checkpoint_path.is_file():
                raise RuntimeError(
                    "partial round data or chunk checkpoint is missing: "
                    f"{partial_data_path}, {partial_checkpoint_path}"
                )
            partial_dataset = CvaeDaggerDataset.load_rounds([partial_data_path])
            if not partial_dataset.rounds:
                raise RuntimeError("partial round data is empty")
            dataset.append(partial_dataset.rounds[0])
            checkpoint_payload = student.load_round_checkpoint(
                partial_checkpoint_path, load_optimizer=True
            )
            checkpoint_round = checkpoint_payload.get("round_index")
            if checkpoint_round is not None and int(checkpoint_round) != start_round:
                raise RuntimeError(
                    "partial checkpoint round mismatch: "
                    f"{checkpoint_round} != {start_round}"
                )
            checkpoint_chunk = checkpoint_payload.get("chunk_index")
            if checkpoint_chunk is not None:
                partial_start_chunk = int(checkpoint_chunk) + 1
            else:
                partial_start_chunk = int(partial.get("next_chunk_index", 0))
            if not 0 <= partial_start_chunk <= chunks_total:
                raise RuntimeError(
                    f"partial chunk index out of range: {partial_start_chunk}/{chunks_total}"
                )
            partial_global_step = int(
                checkpoint_payload.get(
                    "global_step",
                    partial.get("global_step", start_round * train_steps_per_round),
                )
            )
            round_start_step = start_round * train_steps_per_round
            if not (
                round_start_step
                <= partial_global_step
                <= round_start_step + train_steps_per_round
            ):
                raise RuntimeError(
                    "partial global step is outside the current round: "
                    f"{partial_global_step} not in "
                    f"[{round_start_step}, {round_start_step + train_steps_per_round}]"
                )
            partial_keys = [
                str(value)
                for value in (
                    checkpoint_payload.get("motion_keys")
                    or partial.get("motion_keys")
                    or []
                )
            ]
            if not partial_keys:
                partial_keys = _kl2_keys_for_round(mode, manifest, start_round)
            partial_beta = float(
                checkpoint_payload.get(
                    "beta", partial.get("beta", beta_schedule[start_round])
                )
            )
            partial_motion_weights = partial.get("motion_weights")
            partial_train_chunks = list(
                checkpoint_payload.get("train_chunks")
                or partial.get("train_chunks")
                or []
            )
            if len(partial_train_chunks) > partial_start_chunk:
                partial_train_chunks = partial_train_chunks[:partial_start_chunk]
            resume_partial = True
    total_train_steps = total_rounds * train_steps_per_round
    global_step_offset = (
        partial_global_step
        if resume_partial
        else start_round * train_steps_per_round
    )
    partial_manifest_path = (
        run_dir / "manifests" / "dagger_cvae_kl2_manifest.partial.json"
    )
    source_baseline_32 = None
    best_teacher_rmse = None
    best_survival = None
    action_mapping_check = None
    if partial_manifest_path.is_file():
        try:
            previous = json.loads(
                partial_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            previous = {}
        source_baseline_32 = previous.get("source_baseline_32")
        best_teacher_rmse = previous.get("best_teacher_rmse")
        best_survival = previous.get("best_survival")
    if formal and start_round == 0 and source_baseline_32 is None:
        if resume_partial:
            raise RuntimeError(
                "source baseline is missing for a resumed partial round"
            )
        source_baseline_32 = _evaluate_cvae_full(
            config,
            prepared,
            run_dir,
            args_cli,
            device,
            student,
            manifest,
            dataset,
            eval_steps,
            label="source_baseline_32",
        )
        _write_json(
            partial_manifest_path,
            {
                "status": "running",
                "start_round": start_round,
                "source_identity": student.source_identity,
                "source_baseline_32": source_baseline_32,
                "best_teacher_rmse": best_teacher_rmse,
                "best_survival": best_survival,
                "action_mapping_check": action_mapping_check,
                "motion_manifest": manifest,
            },
        )
    formal_reference_metrics = (
        (source_baseline_32 or {}).get("metrics") if formal else None
    )

    for round_index in range(start_round, end_round):
        round_started = time.perf_counter()
        resume_this_round = bool(resume_partial and round_index == start_round)
        if resume_this_round:
            keys = list(partial_keys)
            beta = float(partial_beta)
        else:
            keys = _kl2_keys_for_round(mode, manifest, round_index)
            beta = float(beta_schedule[round_index])
        env, contract, teacher = _create_cvae_env_for_keys(
            config, prepared, run_dir, keys, args_cli, device
        )
        post_fall_continuation = bool(
            config.dagger.get("post_fall_continuation", False)
        )
        if post_fall_continuation:
            env.disable_auto_reset()
        env_setup_seconds = time.perf_counter() - round_started
        try:
            student.action_contract = contract
            student.set_action_clip_value(
                config.student.get("action_clip_value", 20.0)
            )
            if action_mapping_check is None:
                action_mapping_check = _check_cvae_action_mapping(student, contract)
            loop = CvaeDaggerLoop(
                env=env,
                teacher=teacher,
                student=student,
                action_contract=contract,
                num_envs=len(keys),
                rollout_steps=int(config.dagger.rollout_steps),
                run_dir=run_dir,
                seed=int(config.seed),
                state_mean=student.state_mean,
                state_std=student.state_std,
                action_mean=student.action_mean,
                action_std=student.action_std,
                dataset=dataset,
                motion_group=keys,
                mixture_mode=str(config.dagger.get("mixture_mode", "per_step")),
                post_fall_continuation=post_fall_continuation,
                post_fall_baseline_steps=int(
                    config.dagger.get("post_fall_baseline_steps", 10)
                ),
                post_fall_max_error_ratio=float(
                    config.dagger.get("post_fall_max_error_ratio", 3.0)
                ),
                post_fall_down_steps=int(
                    config.dagger.get("post_fall_down_steps", 50)
                ),
                episode_mix=bool(config.dagger.get("episode_mix", False)),
                teacher_prefix_steps=int(
                    config.dagger.get("teacher_prefix_steps", 32)
                ),
                student_block_steps=int(
                    config.dagger.get("student_block_steps", 16)
                ),
                teacher_episode_period=int(
                    config.dagger.get("teacher_episode_period", 4)
                ),
            )
            if resume_this_round:
                round_data = dataset.rounds[-1]
                collect_seconds = 0.0
                chunk_index = int(partial_start_chunk)
                train_chunks = list(partial_train_chunks)
                motion_weights = partial_motion_weights
                data_path = partial_data_path
                partial_path = partial_checkpoint_path
                resume_partial = False
            else:
                collect_started = time.perf_counter()
                round_data = loop.collect_round(round_index, beta)
                collect_seconds = time.perf_counter() - collect_started
                if post_fall_continuation:
                    env.enable_auto_reset()
                chunk_index = 0
                train_chunks = []
                if formal:
                    motion_weights = _cvae_motion_weights(
                        (source_baseline_32 or {}).get("metrics")
                    )
                else:
                    motion_weights = (
                        _cvae_motion_weights(metrics[-1].get("student_only"))
                        if metrics
                        else None
                    )
                data_path = (
                    run_dir / "data" / "dagger" / f"round_{round_index:02d}.pt"
                )
                partial_path = (
                    run_dir
                    / "checkpoints"
                    / f"round_{round_index:02d}_partial.pt"
                )
            train_started = time.perf_counter()
            while chunk_index < chunks_total:
                steps = min(
                    train_chunk_steps,
                    train_steps_per_round - chunk_index * train_chunk_steps,
                )
                chunk_metrics = student.train_on_dataset(
                    dataset,
                    round_index=round_index,
                    global_step_offset=global_step_offset,
                    total_training_steps=total_train_steps,
                    motion_weights=motion_weights,
                    steps=steps,
                )
                global_step_offset += steps
                train_chunks.append(chunk_metrics)
                student.save(
                    partial_path,
                    round_index=round_index,
                    beta=beta,
                    motion_keys=keys,
                    eval_metrics=None,
                    sample_count=int(round_data.count),
                    motion_coverage=dataset.motion_coverage(),
                    extra_metadata={
                        "chunk_index": chunk_index,
                        "train_steps_completed": global_step_offset
                        - round_index * train_steps_per_round,
                        "train_steps_total": train_steps_per_round,
                        "train_chunks": train_chunks,
                    },
                )
                _write_json(
                    round_progress_path,
                    {
                        "round_index": round_index,
                        "next_chunk_index": chunk_index + 1,
                        "chunks_total": chunks_total,
                        "train_steps_completed": global_step_offset
                        - round_index * train_steps_per_round,
                        "train_steps_total": train_steps_per_round,
                        "global_step": global_step_offset,
                        "data_path": str(data_path),
                        "checkpoint_path": str(partial_path),
                        "beta": beta,
                        "motion_keys": keys,
                        "motion_weights": motion_weights,
                        "train_chunks": train_chunks,
                    },
                )
                chunk_index += 1
            train_seconds = time.perf_counter() - train_started
            train_metrics = _merge_cvae_train_chunks(train_chunks)
            eval_started = time.perf_counter()
            eval_metrics = loop.evaluate_student_only(
                eval_steps,
                latent_mode="fixed_standard_normal",
                round_index=round_index,
            )
            eval_seconds = time.perf_counter() - eval_started
            round_seconds = time.perf_counter() - round_started
            raw_teacher_score = float(
                eval_metrics.get("teacher_action_rmse", float("inf"))
            )
            group_baseline = _cvae_group_reference_rmse(
                formal_reference_metrics, keys
            )
            if formal and group_baseline is not None and group_baseline > 0.0:
                teacher_score = raw_teacher_score / group_baseline
                selection_scope = "source_baseline_group_relative"
            else:
                teacher_score = raw_teacher_score
                selection_scope = (
                    "same_motion_absolute" if not formal else "absolute_fallback"
                )
            if best_teacher_rmse is None or teacher_score < float(
                best_teacher_rmse.get("score", float("inf"))
            ):
                best_teacher_rmse = {
                    "round": round_index,
                    "global_step": int(student.global_step),
                    "score": teacher_score,
                    "raw_teacher_action_rmse": raw_teacher_score,
                    "group_baseline_rmse": group_baseline,
                    "selection_scope": selection_scope,
                    "metrics": eval_metrics,
                }
            survival_key = (
                float(eval_metrics.get("survival_ratio", 0.0)),
                float(eval_metrics.get("mean_episode_length", 0.0)),
                -float(eval_metrics.get("termination_rate", 1.0)),
            )
            previous_survival_key = (
                (
                    float(best_survival.get("metrics", {}).get("survival_ratio", 0.0)),
                    float(
                        best_survival.get("metrics", {}).get(
                            "mean_episode_length", 0.0
                        )
                    ),
                    -float(
                        best_survival.get("metrics", {}).get(
                            "termination_rate", 1.0
                        )
                    ),
                )
                if best_survival
                else None
            )
            if previous_survival_key is None or survival_key > previous_survival_key:
                best_survival = {
                    "round": round_index,
                    "global_step": int(student.global_step),
                    "survival_ratio": survival_key[0],
                    "mean_episode_length": survival_key[1],
                    "termination_rate": -survival_key[2],
                    "metrics": eval_metrics,
                }
            teacher_ratio = float(
                round_data.tensors["mixture_teacher"].float().mean()
            )
            history_mean = (
                float(round_data.tensors["history_steps"].float().mean())
                if "history_steps" in round_data.tensors
                else None
            )
            metrics.append(
                {
                    "round": round_index,
                    "beta": beta,
                    "motion_keys": keys,
                    "samples": int(round_data.count),
                    "env_setup_seconds": env_setup_seconds,
                    "collect_seconds": collect_seconds,
                    "train_seconds": train_seconds,
                    "eval_seconds": eval_seconds,
                    "round_seconds": round_seconds,
                    "train_optimizer_steps": int(train_steps_per_round),
                    "rollout_steps": int(config.dagger.rollout_steps),
                    "teacher_executed_ratio": teacher_ratio,
                    "student_executed_ratio": 1.0 - teacher_ratio,
                    "history_steps_mean": history_mean,
                    "train": train_metrics,
                    "train_chunks": train_chunks,
                    "student_only": eval_metrics,
                    "motion_coverage": dataset.motion_coverage(),
                }
            )
            early_stop_payload = None
            if early_stop_enabled and round_index in early_stop_rounds:
                baseline_entry = next(
                    (
                        item
                        for item in metrics
                        if int(item.get("round", -1))
                        == early_stop_baseline_round
                    ),
                    None,
                )
                if baseline_entry is None:
                    raise RuntimeError(
                        "early-stop baseline round is not available: "
                        f"{early_stop_baseline_round}"
                    )
                decision = cvae_early_stop_decision(
                    baseline_entry["student_only"],
                    eval_metrics,
                    min_teacher_improvement_ratio=early_stop_min_ratio,
                    termination_tolerance=early_stop_term_tol,
                )
                metrics[-1]["early_stop_check"] = decision
                if not decision["pass"]:
                    early_stop_payload = {
                        "format_version": "dagger_cvae_kl2_early_stop_v1",
                        "round": round_index,
                        "baseline_round": early_stop_baseline_round,
                        "decision": decision,
                        "baseline": baseline_entry["student_only"],
                        "current": eval_metrics,
                        "quality_note": (
                            "Stopped because teacher RMSE did not improve enough "
                            "or termination became worse."
                        ),
                    }
            global_step_offset = (round_index + 1) * train_steps_per_round
            _write_json(
                run_dir / "manifests" / "dagger_cvae_kl2_metrics.partial.json",
                metrics,
            )
            _write_json(
                partial_manifest_path,
                {
                    "status": "running",
                    "start_round": start_round,
                    "completed_round": round_index,
                    "source_identity": student.source_identity,
                    "source_baseline_32": source_baseline_32,
                    "best_teacher_rmse": best_teacher_rmse,
                    "best_survival": best_survival,
                    "action_mapping_check": action_mapping_check,
                    "motion_manifest": manifest,
                },
            )
            if early_stop_payload is not None:
                _write_json(
                    run_dir
                    / "manifests"
                    / "dagger_cvae_kl2_early_stop.json",
                    early_stop_payload,
                )
                early_stop_marker.write_text(
                    "EARLY STOP\n", encoding="utf-8"
                )
            final_path = (
                run_dir / "checkpoints" / f"round_{round_index:02d}.pt"
            )
            if partial_path.is_file():
                os.replace(partial_path, final_path)
            else:
                student.save(
                    final_path,
                    round_index=round_index,
                    beta=beta,
                    motion_keys=keys,
                    eval_metrics=eval_metrics,
                    sample_count=int(round_data.count),
                    motion_coverage=dataset.motion_coverage(),
                )
            round_progress_path.unlink(missing_ok=True)
            if early_stop_payload is not None:
                return None, {
                    "marker": "dagger_cvae_kl2_early_stop.ok",
                    "complete": False,
                    "early_stop": early_stop_payload,
                }
        finally:
            _close_cvae_env(env)

    complete = end_round >= total_rounds
    final_32 = None
    gate = None
    if complete:
        if formal:
            final_32 = _evaluate_cvae_full(
                config,
                prepared,
                run_dir,
                args_cli,
                device,
                student,
                manifest,
                dataset,
                eval_steps,
                label="final_32",
            )
            gate = _cvae_imitation_gate(
                (source_baseline_32 or {}).get("metrics"),
                (final_32 or {}).get("metrics"),
                required_motions=24,
            )
        else:
            first = metrics[0].get("student_only") if metrics else None
            last = metrics[-1].get("student_only") if metrics else None
            gate = _cvae_imitation_gate(first, last, required_motions=3)
    best_checkpoints: dict[str, str] = {}
    if complete and not smoke:
        for name, info in (
            ("best_teacher_rmse.pt", best_teacher_rmse),
            ("best_survival.pt", best_survival),
        ):
            if not info:
                continue
            source = (
                run_dir / "checkpoints" / f"round_{int(info['round']):02d}.pt"
            )
            if not source.is_file():
                continue
            destination = run_dir / "checkpoints" / name
            shutil.copy2(source, destination)
            best_checkpoints[name] = str(destination)
    information_contract = (
        estimate_teacher_identifiability(dataset)
        if dataset.rounds
        else {
            "available": False,
            "group_count": 0,
            "repeated_group_count": 0,
            "teacher_not_identifiable": None,
            "note": "no DAgger samples",
        }
    )
    information_contract_pass = (
        None
        if not information_contract.get("available")
        else not bool(information_contract.get("teacher_not_identifiable"))
    )
    manifest_payload = {
        "format_version": "dagger_cvae_kl2_manifest_v1",
        "mode": mode,
        "execution_pass": True,
        "complete": bool(complete),
        "quality_pass": bool(gate["pass"]) if gate else None,
        "quality_scope": "same_motion_teacher_action_imitation",
        "rounds_completed": [int(item["round"]) for item in metrics],
        "beta_schedule": beta_schedule,
        "motion_manifest": manifest,
        "source_identity": student.source_identity,
        "source_dataset_run": str(student.source_dataset_run),
        "normalization_run": str(student.normalization_run),
        "normalization_sha256": student.normalization_sha256,
        "source_mix_exact": bool(student.source_mix_exact),
        "source_hash_mismatch": student.source_hash_mismatch,
        "student_checkpoint": str(student.checkpoint_path),
        "student_checkpoint_sha256": student.checkpoint_sha256,
        "teacher_checkpoint": str(
            Path(str(config.teacher_checkpoint)).expanduser().resolve()
        ),
        "eval_steps": int(eval_steps),
        "source_baseline_32": source_baseline_32,
        "final_32": final_32,
        "action_mapping_check": action_mapping_check,
        "best_teacher_rmse": best_teacher_rmse,
        "best_survival": best_survival,
        "best_checkpoints": best_checkpoints,
        "imitation_gate": gate,
        "information_contract": information_contract,
        "information_contract_pass": information_contract_pass,
        "motion_coverage": dataset.motion_coverage(),
        "rounds": metrics,
        "quality_note": (
            "No-reference CVAE; same-motion teacher-Action imitation only. "
            "The 192-step Isaac survival metric is diagnostic."
        ),
    }
    _write_json(
        run_dir / "manifests" / "dagger_cvae_kl2_metrics.json", metrics
    )
    _write_json(
        run_dir / "manifests" / "dagger_cvae_kl2_manifest.json", manifest_payload
    )
    if not complete:
        (run_dir / "markers" / "dagger_cvae_kl2_shard.ok").write_text(
            "execution_pass=true\n", encoding="utf-8"
        )
        return None, {
            "marker": "dagger_cvae_kl2_shard.ok",
            "complete": False,
            "rounds_completed": [int(item["round"]) for item in metrics],
        }
    (run_dir / "markers" / "dagger_cvae_kl2_execution.ok").write_text(
        "execution_pass=true\n", encoding="utf-8"
    )
    if gate and gate.get("pass"):
        (run_dir / "markers" / "dagger_cvae_kl2_imitation.ok").write_text(
            "quality_pass=true\n", encoding="utf-8"
        )
    return None, {
        "marker": (
            "dagger_cvae_kl2_smoke.ok" if smoke else "dagger_cvae_kl2_execution.ok"
        ),
        "complete": True,
        "quality_pass": manifest_payload["quality_pass"],
        "imitation_gate": gate,
    }


def main() -> None:
    import faulthandler
    import signal

    faulthandler.enable()
    try:
        faulthandler.register(signal.SIGUSR1)
    except (AttributeError, OSError, ValueError):
        pass
    faulthandler_seconds = int(os.environ.get("DAGGER_FAULTHANDLER_SECONDS", "120"))
    if faulthandler_seconds > 0:
        faulthandler.dump_traceback_later(faulthandler_seconds, repeat=True)
    print(
        f"[dagger] faulthandler enabled every {faulthandler_seconds}s "
        f"(SIGUSR1 supported where available)",
        flush=True,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    sys.argv = [sys.argv[0]]
    resume_run = os.environ.get("DAGGER_RESUME_RUN", "").strip()
    if resume_run:
        run_dir = Path(resume_run).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"DAGGER_RESUME_RUN does not exist: {run_dir}")
        for child in ("logs", "manifests", "data/dagger", "checkpoints", "markers"):
            (run_dir / child).mkdir(parents=True, exist_ok=True)
        print(f"[dagger] resuming run_dir={run_dir}", flush=True)
    else:
        run_dir = _new_run_dir(config)
    device = torch.device(str(config.get("device", "cuda:0")))
    sonic_root = Path(str(config.sonic_repo_root)).expanduser().resolve()
    if not sonic_root.is_dir():
        raise FileNotFoundError(f"SONIC_REPO_ROOT does not exist: {sonic_root}")
    os.chdir(sonic_root)
    print(
        f"[dagger] mode={config.mode} run_dir={run_dir} sonic_root={sonic_root} "
        f"device={device}",
        flush=True,
    )
    if str(config.mode).startswith("sonic_causal"):
        _select_sonic_causal_motions(config)
    if (
        str(config.mode) == "mlp_render"
        and str(config.dagger.get("render_policy", "mlp")).lower() == "sonic_causal"
    ):
        render_key = str(config.dagger.get("motion_key", "") or "")
        config.dagger.motion_keys = [render_key] if render_key else []
        config.dagger.motion_key = None
        config.dagger.motion_count = 1
        _select_sonic_causal_motions(config)
    motion_source = str(config.dagger.get("motion_source_checkpoint", "") or "")
    if str(config.mode).startswith("mlp") and str(config.mode) != "mlp_render" and motion_source:
        # Same checkpoint + motion_seed + motion_count as sonic_causal_pilot -> identical motion set.
        _select_sonic_causal_motions(config, checkpoint=motion_source)
    prepared = _prepare_release_env(config, run_dir)
    print("[dagger] launching Isaac app", flush=True)
    launcher, args_cli = _launch_app(config)
    print("[dagger] Isaac app ready", flush=True)
    env = None
    exit_code = 1
    try:
        if str(config.mode).startswith("cvae_kl2"):
            env, manifest = _run_cvae_kl2(config, run_dir, device, args_cli, prepared)
            marker_name = manifest.get("marker", "dagger_cvae_kl2_execution.ok")
        elif str(config.mode).startswith("sonic_causal"):
            env, manifest = _run_sonic_causal(config, run_dir, device, args_cli, prepared)
            marker_name = "sonic_causal_execution.ok"
        elif str(config.mode).startswith("cvae"):
            env, manifest = _run_cvae(config, run_dir, device, args_cli, prepared)
            marker_name = "dagger_cvae_smoke.ok" if str(config.mode) == "cvae_smoke" else "dagger_cvae_execution.ok"
        elif str(config.mode) == "mlp_render":
            env, manifest = _run_mlp_render(config, run_dir, device, args_cli, prepared)
            marker_name = "dagger_mlp_render.ok"
        elif str(config.mode).startswith("mlp"):
            env, manifest = _run_plain(config, run_dir, device, args_cli, prepared)
            if manifest.get("complete", True):
                marker_name = "dagger_mlp_smoke.ok" if str(config.mode) == "mlp_smoke" else "dagger_mlp_execution.ok"
            else:
                marker_name = "dagger_mlp_shard.ok"
        else:
            env, manifest = _run_plain(config, run_dir, device, args_cli, prepared)
            marker_name = "dagger_smoke.ok" if str(config.mode) == "smoke" else "dagger_execution.ok"
        (run_dir / "markers" / marker_name).write_text("execution_pass=true\n", encoding="utf-8")
        (run_dir / marker_name).write_text("execution_pass=true\n", encoding="utf-8")
        print(json.dumps({"run_dir": str(run_dir), "manifest": manifest}, indent=2, default=str))
        exit_code = 0
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()

