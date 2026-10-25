"""Standalone DAgger CLI; SONIC is loaded only as an external teacher/runtime."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch
from omegaconf import OmegaConf

from .adapters.plain_mlp_student import PlainMlpStudent
from .adapters.sonic_release_teacher import SonicReleaseTeacher
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


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
    )
    print(
        f"[dagger] student ready params={sum(p.numel() for p in student.model.parameters())} "
        f"normalization_mode={'frozen' if student.freeze_normalization else 'recomputed'}",
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
    metrics = []
    for round_index, beta in enumerate(config.dagger.beta_schedule):
        print(f"[dagger] round={round_index} beta={float(beta)} collecting", flush=True)
        round_data = loop.collect_round(round_index, float(beta))
        print(f"[dagger] round={round_index} collected samples={round_data.count}", flush=True)
        round_metrics = loop.train_student()
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
        round_metrics["student_only"] = loop.evaluate_student_only(eval_steps)
        print(
            f"[dagger] round={round_index} eval_rmse="
            f"{round_metrics['student_only'].get('teacher_action_rmse')}",
            flush=True,
        )
        checkpoint = run_dir / "checkpoints" / f"round_{round_index:02d}.pt"
        student.save(str(checkpoint), extra_metadata={"round_index": round_index})
        metrics.append(
            {
                "round": round_index,
                "beta": float(beta),
                "samples": round_data.count,
                "checkpoint": str(checkpoint),
                **round_metrics,
            }
        )
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
        "eval_steps": eval_steps,
        "seed": int(config.seed),
        "rounds": metrics,
        "execution_pass": True,
        "quality_pass": None,
        "memorization_pass": memorization_pass,
        "memorization_scope": "same_motion_set",
        "quality_note": "Same-motion DAgger memorization only; no held-out or new-motion generalization claim.",
    }
    _write_json(run_dir / "manifests" / "dagger_mlp_manifest.json", manifest)
    _write_json(run_dir / "manifests" / "dagger_mlp_metrics.json", metrics)
    _write_json(run_dir / "manifests" / "dagger_manifest.json", manifest)
    _write_json(run_dir / "manifests" / "dagger_metrics.json", metrics)
    print(
        f"[dagger] manifest written memorization_pass={memorization_pass}",
        flush=True,
    )
    if memorization_pass:
        (run_dir / "markers" / "dagger_mlp_memorization.ok").write_text(
            "memorization_pass=true\n", encoding="utf-8"
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
    prepared = _prepare_release_env(config, run_dir)
    print("[dagger] launching Isaac app", flush=True)
    launcher, args_cli = _launch_app(config)
    print("[dagger] Isaac app ready", flush=True)
    env = None
    exit_code = 1
    try:
        if str(config.mode).startswith("cvae"):
            env, manifest = _run_cvae(config, run_dir, device, args_cli, prepared)
            marker_name = "dagger_cvae_smoke.ok" if str(config.mode) == "cvae_smoke" else "dagger_cvae_execution.ok"
        elif str(config.mode).startswith("mlp"):
            env, manifest = _run_plain(config, run_dir, device, args_cli, prepared)
            marker_name = "dagger_mlp_smoke.ok" if str(config.mode) == "mlp_smoke" else "dagger_mlp_execution.ok"
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

