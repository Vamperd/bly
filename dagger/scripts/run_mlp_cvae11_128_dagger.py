"""Scheduler-owned duration stage; the shell entry point documents its interface."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


def load(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def atomic(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def jobs():
    result = subprocess.run(["gpu", "ps", "-N", "500", "--json"], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["jobs"]


def committed(run: Path):
    rows = load(run / "manifests/sonic_causal_metrics.partial.json", [])
    for index, row in enumerate(rows):
        if int(row["round"]) != index:
            raise RuntimeError("DAgger commit metrics have noncontiguous numeric rounds")
        for child in ("checkpoints", "data/dagger"):
            path = run / child / f"round_{index:02d}.pt"
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f"committed round is missing artifact: {path}")
    return rows


def shard(config: Path, log_path: Path, lease_seconds: int) -> int:
    # This process starts only after scheduler allocation, not while queued.
    os.environ["DAGGER_DEADLINE_EPOCH"] = str(time.time() + lease_seconds - 30)
    with log_path.open("ab", buffering=0) as stream:
        return subprocess.call([sys.executable, "-m", "dagger.cli", "--config", str(config)], stdout=stream, stderr=subprocess.STDOUT)


def run_stage(c_run: Path, migration: Path, state_dir: Path) -> None:
    from omegaconf import OmegaConf
    import torch

    code = Path(__file__).resolve().parents[2]
    c_checkpoint = c_run / "checkpoints/best.pt"
    c_hash = hashlib.sha256(c_checkpoint.read_bytes()).hexdigest()
    payload = torch.load(c_checkpoint, map_location="cpu", weights_only=False)
    keys = [str(key) for key in (payload.get("selection") or {}).get("selected_motion_keys", payload.get("train_motion_keys", []))]
    if len(keys) != 128 or len(set(keys)) != 128:
        raise ValueError(f"C checkpoint must contain exactly 128 distinct trained motions, got {len(keys)}")
    if payload.get("format_version") != "sonic_11_frame_mlp_cvae_checkpoint_v1":
        raise ValueError("C best.pt is not a trained mlp_cvae11 checkpoint")
    del payload
    sonic = Path(os.environ.get("SONIC_REPO_ROOT", code / "sonic-repro/GR00T-WholeBodyControl")).resolve()
    dataset = migration / "cvae_physics_dataset_20260825_235244"
    motion = migration / "bones_seed_ingest_2048_20260825_132447/data/robot_filtered"
    if not dataset.is_dir() or not motion.is_dir():
        raise FileNotFoundError(f"migration dataset/motion library missing beneath {migration}")
    smoke = os.environ.get("DAGGER_STAGE_SMOKE", "0") == "1"
    for name in ("HOURS", "ROUNDS", "ROLLOUT_STEPS", "TRAIN_STEPS", "EVAL_STEPS"):
        if not smoke and f"DAGGER_STAGE_{name}" in os.environ:
            raise ValueError(f"DAGGER_STAGE_{name} requires DAGGER_STAGE_SMOKE=1")
    hours = float(os.environ.get("DAGGER_STAGE_HOURS", "12")) if smoke else 12.0
    rounds = int(os.environ.get("DAGGER_STAGE_ROUNDS", "2")) if smoke else 1000000
    shard_rounds = int(os.environ.get("DAGGER_SHARD_ROUNDS", "1" if smoke else "2"))
    lease = int(os.environ.get("DAGGER_STAGE_LEASE_SECONDS", "7200"))
    if min(hours, rounds, shard_rounds) <= 0 or lease <= 240:
        raise ValueError("duration/rounds must be positive and shard lease must exceed 240s")
    env = dict(os.environ)
    env.update({
        "BLY_PROJECT_ROOT": str(code), "SONIC_RUNS_ROOT": str(code.parent / "runs"),
        "SONIC_REPO_ROOT": str(sonic), "SONIC_KIT_ROOT": str(code / "sonic-repro-kit"),
        "SONIC_TEACHER_CHECKPOINT": str(sonic / "sonic_release/last.pt"),
        "DAGGER_MOTION_FILE": str(motion), "MLP_MOTION_SOURCE_CHECKPOINT": str(c_checkpoint),
        "MLP_CVAE11_CHECKPOINT": str(c_checkpoint), "CVAE_DATASET_RUN": str(dataset),
        "CVAE_DATASET_PATH_MAP": json.dumps({"/home/helloworld/bly/runs": str(migration)}),
        "PYTHONPATH": ":".join(map(str, (code / "dagger/src", code / "state-action-cvae/src", code / "sonic-repro-kit", sonic))),
        "PYTHONUNBUFFERED": "1", "DAGGER_ROUNDS_PER_PROCESS": str(shard_rounds),
        "DAGGER_LEASE_BUFFER_SECONDS": os.environ.get("DAGGER_LEASE_BUFFER_SECONDS", "180"),
        "TMPDIR": str(code.parent / "isaac_tmp"), "XDG_CACHE_HOME": str(code.parent / "isaac_tmp"),
    })
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    os.environ.update(env)
    path = state_dir / "dagger_stage.json"
    state = load(path)
    if state is None or state.get("initialization_failed", False):
        dagger_run = state_dir / "dagger_run"
        for child in ("logs", "manifests", "data/dagger", "checkpoints", "markers"):
            (dagger_run / child).mkdir(parents=True, exist_ok=True)
        state = {
            "dagger_run": str(dagger_run), "c_run": str(c_run), "migration_root": str(migration),
            "c_checkpoint": str(c_checkpoint), "c_checkpoint_sha256": c_hash,
            "train_motion_keys": keys, "completed_training_seconds": 0.0,
            "completed_rounds": 0, "complete": False, "latest_checkpoint": None,
            "state": "ready", "smoke": smoke, "target_training_seconds": hours * 3600,
            "target_rounds": rounds, "shard_index": 0, "job_ref": None,
        }
        atomic(path, state)
        config = OmegaConf.load(code / "dagger/configs/sonic_causal_mlp_cvae11_128_12h.yaml")
        config.dagger.max_train_hours = hours
        config.dagger.rounds = rounds
        config.dagger.motion_keys = keys
        if smoke:
            config.dagger.rollout_steps = int(os.environ.get("DAGGER_STAGE_ROLLOUT_STEPS", "2"))
            config.dagger.eval_steps = int(os.environ.get("DAGGER_STAGE_EVAL_STEPS", "2"))
            config.student.train_steps_per_round = int(os.environ.get("DAGGER_STAGE_TRAIN_STEPS", "1"))
        atomic(state_dir / "dagger_config.json", OmegaConf.to_container(config, resolve=True))
    elif any((state["c_checkpoint_sha256"] != c_hash, state["c_run"] != str(c_run),
              state["migration_root"] != str(migration), state["smoke"] != smoke)):
        raise ValueError("existing DAgger state belongs to different C checkpoint/migration/smoke mode")
    config_path = state_dir / "dagger_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"immutable stage config missing: {config_path}")
    env["DAGGER_RESUME_RUN"] = state["dagger_run"]
    poll = float(os.environ.get("DAGGER_STAGE_POLL_SECONDS", "15"))

    def refresh():
        rows = committed(Path(state["dagger_run"]))
        if any(set(row.get("collected_motion_keys", [])) != set(keys) for row in rows):
            raise RuntimeError("committed DAgger round lacks actual full 128-motion sample coverage")
        state["completed_rounds"] = len(rows)
        state["completed_training_seconds"] = float(rows[-1]["completed_training_seconds"]) if rows else 0.0
        state["latest_checkpoint"] = str(Path(state["dagger_run"]) / "checkpoints" / f"round_{len(rows) - 1:02d}.pt") if rows else None
        state["collected_motion_keys"] = sorted(set().union(*(set(row["collected_motion_keys"]) for row in rows))) if rows else []
        state["updated_at"] = time.time()
        return rows

    try:
        while not state["complete"]:
            rows = refresh()
            if not state.get("job_ref"):
                # Recover the narrow submit/persist window without a duplicate job.
                existing = [job for job in jobs() if state.get("job_name") and job["name"] == state["job_name"]]
                if existing:
                    state["job_ref"] = existing[-1]["job_ref"]
                else:
                    state["shard_index"] += 1
                    state["rounds_before_shard"] = len(rows)
                    state["job_name"] = f"mlp11-128-dagger-{uuid.uuid4().hex[:12]}"
                    state["state"] = "submitting"
                    atomic(path, state)
                    log = Path(state["dagger_run"]) / "logs" / f"shard_{state['shard_index']:05d}.log"
                    command = ["gpu", "1", "-c", os.environ.get("DAGGER_STAGE_CPUS", "16"),
                               "-m", os.environ.get("DAGGER_STAGE_RAM", "48G"), "-t", f"{lease}s", "-n", state["job_name"],
                               "env", *[f"{key}={value}" for key, value in env.items()],
                               sys.executable, str(Path(__file__).resolve()), "--shard", str(config_path), str(log), str(lease)]
                    submitted = subprocess.run(command, check=True, capture_output=True, text=True)
                    (state_dir / f"submit_{state['shard_index']:05d}.log").write_text(submitted.stdout + submitted.stderr)
                    matches = re.findall(r"\b\d{4}-\d{4,}\b", submitted.stdout + submitted.stderr)
                    if matches:
                        state["job_ref"] = matches[-1]
                    else:
                        matching = [job for job in jobs() if job["name"] == state["job_name"]]
                        if not matching:
                            raise RuntimeError("gpu submitted no discoverable job; inspect submit log")
                        state["job_ref"] = matching[-1]["job_ref"]
                state["state"] = "queued"
                atomic(path, state)
            job = next((job for job in jobs() if job["job_ref"] == state["job_ref"]), None)
            if job is None:
                raise RuntimeError(f"scheduler lost DAgger job {state['job_ref']}")
            terminal = job["state"] in {"SUCCEEDED", "FAILED", "CANCELLED", "PREEMPTED"}
            state["scheduler_state"] = job["state"]
            state["stop_reason"] = job.get("stop_reason")
            state["state"] = "running" if job["state"] == "RUNNING" else job["state"].lower()
            rows = refresh()
            atomic(path, state)
            if not terminal:
                time.sleep(poll)
                continue
            timeout = job["state"] == "CANCELLED" and job.get("stop_reason") == "TIME_LIMIT"
            if job["state"] != "SUCCEEDED" and not timeout:
                raise RuntimeError(f"DAgger shard {state['job_ref']} ended {job['state']} ({job.get('stop_reason')})")
            if len(rows) <= state["rounds_before_shard"]:
                raise RuntimeError(f"DAgger shard {state['job_ref']} saved no new committed round; inspect shard log")
            latest = torch.load(state["latest_checkpoint"], map_location="cpu", weights_only=False)
            if int(latest.get("round_index", -1)) != len(rows) - 1 or not latest.get("optimizer"):
                raise RuntimeError("latest committed checkpoint has no matching round/optimizer")
            if latest.get("source_checkpoint_sha256") != c_hash or set(latest.get("motion_keys", [])) != set(keys):
                raise RuntimeError("latest DAgger checkpoint C provenance or 128-motion assignment mismatch")
            del latest
            budget_complete = state["completed_training_seconds"] >= state["target_training_seconds"]
            smoke_complete = state["smoke"] and len(rows) >= state["target_rounds"]
            if budget_complete or smoke_complete:
                state.update(complete=True, state="complete", completion_reason="time_budget" if budget_complete else "smoke_rounds")
            elif len(rows) >= state["target_rounds"]:
                raise RuntimeError("round ceiling reached before the full training duration")
            else:
                state["last_job_ref"] = state["job_ref"]
                state["job_ref"] = None
                state["job_name"] = None
                state["state"] = "ready"
            atomic(path, state)
    except BaseException as error:
        state.update(complete=False, state="error", error=f"{type(error).__name__}: {error}")
        atomic(path, state)
        raise


def main() -> int:
    if sys.argv[1:2] == ["--shard"]:
        return shard(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
    c_run, migration, state_dir = (Path(value).expanduser().resolve() for value in sys.argv[1:])
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / ".dagger_stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run_stage(c_run, migration, state_dir)
        except BaseException as error:
            path = state_dir / "dagger_stage.json"
            state = load(path, {"dagger_run": str(state_dir / "dagger_run"), "initialization_failed": True})
            state.update(complete=False, state="error", error=f"{type(error).__name__}: {error}")
            atomic(path, state)
            raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
