#!/usr/bin/env python3
"""Wait for the 128-motion C budget, then run DAgger and four trained-motion ghosts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


ACTIVE = {"QUEUED", "PENDING", "RUNNING", "STARTING", "STOPPING"}


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "updated_at": datetime.now(timezone.utc).isoformat()}
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _jobs() -> dict:
    return json.loads(subprocess.check_output(["gpu", "ps", "--json", "-N", "500"], text=True, timeout=30))


def _job(ref: str) -> dict:
    jobs = _jobs()["jobs"]
    return next((row for row in jobs if row["job_ref"] == ref), {"state": "MISSING"})


def _validate_c(c_run: Path, job: dict, *, expected_seconds: float) -> dict:
    summary = _read(c_run / "manifests/mlp_cvae11_summary.json")
    if not summary.get("execution_pass") or summary.get("smoke"):
        raise RuntimeError("C has no complete non-smoke execution summary")
    elapsed = float(summary.get("elapsed_seconds", 0.0))
    if elapsed < expected_seconds - 120.0:
        raise RuntimeError(f"C budget incomplete: {elapsed:.1f}/{expected_seconds:.1f} seconds")
    state = job["state"]
    if state == "CANCELLED":
        if job.get("stop_reason") != "TIME_LIMIT" or summary.get("stop_reason") != "signal":
            raise RuntimeError(f"C cancellation is not an expected lease expiry: {job}")
    elif state != "SUCCEEDED":
        raise RuntimeError(f"C did not finish safely: {job}")
    selection = _read(c_run / "manifests/motion_selection.json")
    keys = selection.get("selected_motion_keys", [])
    if len(keys) != 128 or len(set(keys)) != 128:
        raise RuntimeError("C does not contain exactly 128 distinct selected motions")
    if not (c_run / "markers/cvae_mlp_cvae11_execution.ok").is_file():
        raise RuntimeError("C execution marker is missing")
    checkpoint = c_run / "checkpoints/best.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError("C best checkpoint is missing")
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format_version") != "sonic_11_frame_mlp_cvae_checkpoint_v1":
        raise RuntimeError("C best checkpoint format mismatch")
    if saved.get("selection", {}).get("selected_motion_keys") != keys:
        raise RuntimeError("C best checkpoint motion identity differs from selection manifest")
    metrics = saved.get("metrics", {})
    for key in ("prior_state_rmse", "prior_action_rmse", "prior_causal_action_rmse"):
        if not math.isfinite(float(metrics.get(key, float("nan")))):
            raise RuntimeError(f"C best checkpoint metric missing or nonfinite: {key}")
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"job_state": state, "job_stop_reason": job.get("stop_reason"), "elapsed_seconds": elapsed,
            "completed_optimizer_steps": summary.get("completed_optimizer_steps"),
            "checkpoint": str(checkpoint), "checkpoint_step": saved["optimizer_step"],
            "checkpoint_sha256": digest, "motion_count": len(keys), "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("c_job")
    parser.add_argument("c_run", type=Path)
    parser.add_argument("migration", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    c_run = args.c_run.resolve()
    state_dir = (args.state_dir or c_run / "pipeline").resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / "pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code = Path(__file__).resolve().parents[2]
    state_path = state_dir / "pipeline_state.json"
    state = _read(state_path)
    state.update({"c_run": str(c_run), "c_job": args.c_job, "state_dir": str(state_dir), "pid": os.getpid()})
    try:
        if state.get("status") == "complete":
            print("Pipeline already complete", flush=True)
            return 0
        if not state.get("c_validation"):
            while True:
                job = _job(args.c_job)
                state.update({"status": "waiting_c", "c_job_state": job["state"]})
                _write(state_path, state)
                print(f"C_JOB={job['state']}", flush=True)
                if job["state"] not in ACTIVE:
                    # Scheduler termination and summary writes can race for a few seconds.
                    summary_path = c_run / "manifests/mlp_cvae11_summary.json"
                    deadline = time.monotonic() + 90.0
                    while not summary_path.exists() and time.monotonic() < deadline:
                        time.sleep(5.0)
                    state["c_validation"] = _validate_c(c_run, job, expected_seconds=43200.0)
                    _write(state_path, state)
                    break
                time.sleep(args.poll_seconds)
        state["status"] = "dagger"
        _write(state_path, state)
        dagger_state = _read(state_dir / "dagger_stage.json")
        if not dagger_state.get("complete"):
            subprocess.run([
                str(code / "dagger/scripts/run_mlp_cvae11_128_dagger.sh"),
                str(c_run), str(args.migration.resolve()), str(state_dir),
            ], check=True)
            dagger_state = _read(state_dir / "dagger_stage.json")
        if not dagger_state.get("complete") or not dagger_state.get("dagger_run"):
            raise RuntimeError(f"DAgger stage did not complete: {dagger_state}")
        state.update({"status": "rendering", "dagger_run": dagger_state["dagger_run"]})
        _write(state_path, state)
        ghost_state = _read(state_dir / "ghost_stage.json")
        if not ghost_state.get("complete"):
            if not state.get("ghost_job"):
                env_values = {
                    "PYTHON": str(code / "sonic-repro/.venv-sonic/bin/python"),
                    "BLY_PROJECT_ROOT": str(code),
                    "SONIC_RUNS_ROOT": str(c_run.parent),
                    "SONIC_REPO_ROOT": str(code / "sonic-repro/GR00T-WholeBodyControl"),
                    "SONIC_KIT_ROOT": str(code / "sonic-repro-kit"),
                    "CVAE_DATASET_PATH_MAP": json.dumps({"/home/helloworld/bly/runs": str(args.migration.resolve())}),
                    "TMPDIR": str(state_dir / "isaac_tmp"),
                    "XDG_CACHE_HOME": str(state_dir / "isaac_tmp"),
                }
                Path(env_values["TMPDIR"]).mkdir(parents=True, exist_ok=True)
                command = ["gpu", "1", "-c16", "-m48G", "-t4h", "-n", "mlp-cvae11-128-ghost4", "env"]
                command.extend(f"{key}={value}" for key, value in env_values.items())
                command.extend([
                    str(code / "dagger/scripts/render_mlp_cvae11_128_final_ghost4.sh"),
                    str(c_run), dagger_state["dagger_run"], str(state_dir), str(args.migration.resolve()),
                ])
                submission = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
                print(submission, flush=True)
                match = re.search(r"Submitted job (\d{4}-\d+)", submission)
                if match is None:
                    raise RuntimeError(f"Cannot parse ghost allocation: {submission}")
                state["ghost_job"] = match.group(1)
                _write(state_path, state)
            while True:
                job = _job(state["ghost_job"])
                print(f"GHOST_JOB={job['state']}", flush=True)
                if job["state"] not in ACTIVE:
                    if job["state"] != "SUCCEEDED":
                        raise RuntimeError(f"Ghost job failed: {job}")
                    break
                time.sleep(args.poll_seconds)
            ghost_state = _read(state_dir / "ghost_stage.json")
        if not ghost_state.get("complete"):
            raise RuntimeError(f"Ghost job lacks validated output: {ghost_state}")
        state.update({"status": "complete", "ghost_stage": ghost_state})
        _write(state_path, state)
        print("PIPELINE_COMPLETE", flush=True)
        return 0
    except Exception as error:
        state.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
        _write(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
