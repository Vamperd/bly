#!/usr/bin/env python3
"""Post-DAgger ghost selection/rendering and pull-only local delivery."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time


LAYERS = ["sonic", "dagger_before", "dagger_after"]
ROOT_MARKER = "markers/mlp_cvae11_trained_motion_ghost4.ok"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def motion_keys(payload):
    selection = payload.get("selection") or {}
    values = selection.get("selected_motion_keys", payload.get("train_motion_keys", []))
    if not isinstance(values, list) or any(not isinstance(key, str) or not key or "\n" in key for key in values):
        raise ValueError("invalid or unknown declared training motion set")
    if len(values) != 128 or len(set(values)) != 128:
        raise ValueError(f"expected exactly 128 distinct declared training motions, got {len(values)}")
    return set(values)


def select_inputs(c_run, d_run):
    import torch

    before = c_run / "checkpoints/best.pt"
    c_payload = torch.load(before, map_location="cpu", weights_only=False)
    if c_payload.get("format_version") != "sonic_11_frame_mlp_cvae_checkpoint_v1":
        raise ValueError("BEFORE is not an original Stage-C MLP CVAE11 checkpoint")
    c_keys = motion_keys(c_payload)
    del c_payload
    before_hash = sha256(before)
    metrics_path = d_run / "manifests/sonic_causal_metrics.partial.json"
    if metrics_path.is_file():
        metrics = load_json(metrics_path)
    else:
        metrics = load_json(d_run / "manifests/sonic_causal_manifest.json")["rounds"]
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("DAgger has no committed training metrics")
    committed = {int(row["round"]) for row in metrics}
    checkpoints = {}
    for path in (d_run / "checkpoints").glob("round_*.pt"):
        match = re.fullmatch(r"round_(\d+)\.pt", path.name)
        if match and int(match[1]) in committed:
            index = int(match[1])
            if index in checkpoints:
                raise ValueError(f"ambiguous numeric round checkpoint {index}")
            checkpoints[index] = path
    if not checkpoints or set(checkpoints) != committed:
        raise ValueError("committed DAgger metrics/checkpoints disagree")
    coverage = Counter()
    evidence = []
    last_index = max(checkpoints)
    after = checkpoints[last_index]
    for index in sorted(checkpoints):
        path = checkpoints[index]
        stat_before = path.stat()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if path.stat().st_mtime_ns != stat_before.st_mtime_ns or path.stat().st_size != stat_before.st_size:
            raise ValueError(f"checkpoint changed while reading: {path}")
        if payload.get("format_version") != "sonic_11_frame_mlp_cvae_dagger_checkpoint_v1" or int(payload.get("round_index", -1)) != index:
            raise ValueError(f"invalid completed round checkpoint: {path}")
        if payload.get("source_checkpoint_sha256") != before_hash:
            raise ValueError(f"DAgger source checkpoint is not the supplied C best: {path}")
        source_path = Path(payload.get("source_checkpoint", "")).expanduser().resolve()
        if source_path != before.resolve():
            raise ValueError(f"DAgger source checkpoint path differs from C best: {source_path}")
        if motion_keys(payload) != c_keys:
            raise ValueError("C and DAgger declared training selections differ")
        keys = payload.get("motion_keys")
        if not isinstance(keys, list) or not keys or len(set(keys)) != len(keys) or not set(keys).issubset(c_keys):
            raise ValueError(f"unknown/empty/non-C round motion map: {path}")
        data_path = d_run / "data/dagger" / path.name
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        samples = data.get("samples")
        if int(data.get("round_index", -1)) != index or not isinstance(samples, list) or not samples:
            raise ValueError(f"empty or invalid round samples: {data_path}")
        round_coverage = Counter()
        for sample in samples:
            value = sample.get("motion_id")
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise ValueError(f"missing scalar motion_id in {data_path}")
            motion_id = int(value.item())
            if not 0 <= motion_id < len(keys):
                raise ValueError(f"unknown motion_id={motion_id} in {data_path}")
            round_coverage[keys[motion_id]] += 1
        coverage.update(round_coverage)
        evidence.append({"round": index, "data": str(data_path), "samples": len(samples),
                         "motion_samples": dict(sorted(round_coverage.items())), "checkpoint": str(path)})
        del data, samples, payload
    eligible = sorted(set(coverage) & c_keys)
    if len(eligible) < 4:
        raise ValueError(f"fewer than four actually trained C/DAgger motions: {len(eligible)}")
    chosen = eligible[:4]
    return {
        "motion_keys": chosen, "selection_rule": "lexicographically_first_four_actual_sample_keys_intersect_C",
        "declared_training_motion_count": 128, "actual_covered_motion_count": len(eligible),
        "actual_sample_counts": dict(sorted(coverage.items())), "coverage_rounds": evidence,
        "before_checkpoint": {"path": str(before), "sha256": before_hash},
        "after_checkpoint": {"path": str(after), "sha256": sha256(after), "round_index": last_index},
    }


def relative_file(path, root):
    relative = PurePosixPath(path).relative_to(PurePosixPath(root))
    if not relative.parts or ".." in relative.parts or any("\n" in part or "\r" in part for part in relative.parts):
        raise ValueError(f"unsafe delivery path: {path}")
    return relative.as_posix()


def delivery_files(manifest, output_root):
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 4 or len({case["motion_key"] for case in cases}) != 4:
        raise ValueError("ghost manifest must have four distinct motion cases")
    if manifest.get("layers") != LAYERS or int(manifest.get("horizon_states", 0)) != 193:
        raise ValueError("ghost manifest layer/horizon contract mismatch")
    files = {"manifest.json", ROOT_MARKER}
    videos = []
    for case in cases:
        ghost = case["ghost"]
        if int(ghost.get("frames", 0)) != 193 or [layer["name"] for layer in ghost["layers"]] != LAYERS:
            raise ValueError("ghost must contain 193 display frames and three policy layers")
        if abs(float(ghost.get("fps", 0.0)) - 50.0) > 1.0e-6:
            raise ValueError("ghost must use the 50Hz rollout frame rate")
        if ghost.get("engines") != {"rollout": "Isaac Sim", "display": "MuJoCo kinematic replay"}:
            raise ValueError("missing engine evidence")
        expected_counts = {layer["name"]: int(layer["frames"]) for layer in ghost["layers"]}
        if ghost.get("original_frame_counts") != expected_counts:
            raise ValueError("ghost original-frame manifest mismatch")
        padding = ghost.get("display_padding", {})
        if (padding.get("method") != "repeat_final_frame_only"
                or padding.get("trajectories_modified") is not False
                or padding.get("frames_by_layer") != {name: 193 - count for name, count in expected_counts.items()}):
            raise ValueError("ghost display-padding contract mismatch")
        for layer in ghost["layers"]:
            evidence = case["rollouts"][layer["name"]]
            original_frames = int(layer["frames"])
            if not 1 <= original_frames <= 193 or int(evidence["frames"]) != original_frames:
                raise ValueError("invalid original frame counts")
            if evidence.get("execution_pass") is not True or evidence.get("render_kind") != "isaac_rollout_then_mujoco_kinematic_render":
                raise ValueError("invalid rollout evidence")
            if evidence.get("motion_keys") != [case["motion_key"]]:
                raise ValueError("rollout key differs from case key")
        video = relative_file(ghost["video"], output_root)
        if PurePosixPath(video).suffix != ".mp4":
            raise ValueError("unexpected ghost video extension")
        ghost_dir = PurePosixPath(video).parent.parent
        case_dir = ghost_dir.parent
        files.update([video, str(ghost_dir / "manifests/policy_ghost.json"),
                      str(ghost_dir / "markers/policy_ghost.ok"), str(case_dir / "case.json")])
        videos.append({"path": video, "sha256": ghost["video_sha256"], "frames": 193, "motion_key": case["motion_key"]})
    if len({video["path"] for video in videos}) != 4:
        raise ValueError("expected four different output videos")
    return sorted(files), videos


def validate_identities(manifest, state):
    for label, role in (("before_checkpoint", "dagger_before"), ("after_checkpoint", "dagger_after")):
        identity = state[label]
        if manifest.get(label) != identity["path"]:
            raise ValueError(f"root manifest {label} differs from selected checkpoint")
        for case in manifest["cases"]:
            evidence = case["rollouts"][role]
            if evidence.get("checkpoint") != identity["path"] or evidence.get("checkpoint_sha256") != identity["sha256"]:
                raise ValueError(f"{role} rollout checkpoint identity mismatch")


def render_stage(args):
    c_run, d_run, state_dir, migration = [Path(value).expanduser().resolve() for value in args.paths]
    state_file = state_dir / "ghost_stage.json"
    output_root = state_dir / "ghost4"
    state = {"format_version": "mlp_cvae11_128_ghost_stage_v1", "complete": False, "status": "running",
             "started_at": timestamp(), "output_root": str(output_root), "c_run": str(c_run), "dagger_run": str(d_run)}
    if state_file.exists() and load_json(state_file).get("complete") is True:
        raise FileExistsError(f"ghost stage is already complete: {state_file}")
    atomic_json(state_file, state)
    try:
        selected = select_inputs(c_run, d_run)
        state.update(selected)
        atomic_json(state_file, state)
        keys_path = state_dir / "ghost_motion_keys.json"
        atomic_json(keys_path, selected["motion_keys"])
        atomic_json(state_dir / "ghost_selection.json", selected)
        code = Path(__file__).resolve().parents[2]
        state["frame_verifier_python"] = str(Path(sys.executable).absolute())
        environment = os.environ.copy()
        # multiprocessing.Manager uses AF_UNIX sockets under TMPDIR; deep run
        # paths exceed its 108-byte address limit.
        short_tmp = code.parent / "isaac_tmp"
        short_tmp.mkdir(parents=True, exist_ok=True)
        environment["TMPDIR"] = str(short_tmp)
        environment["GHOST_OUTPUT_ROOT"] = str(output_root)
        environment.setdefault("CVAE_DATASET_PATH_MAP", json.dumps({"/home/helloworld/bly/runs": str(migration)}))
        command = ["bash", str(code / "dagger/scripts/render_mlp_cvae11_trained_motion_ghost4.sh"),
                   selected["before_checkpoint"]["path"], selected["after_checkpoint"]["path"], str(keys_path),
                   str(migration / "bones_seed_ingest_2048_20260825_132447/data/robot_filtered"),
                   str(migration / "cvae_physics_dataset_20260825_235244"), "192"]
        subprocess.run(command, env=environment, check=True)
        manifest_path = output_root / "manifest.json"
        manifest = load_json(manifest_path)
        if [case["motion_key"] for case in manifest["cases"]] != selected["motion_keys"]:
            raise ValueError("rendered keys differ from selected actual coverage")
        validate_identities(manifest, state)
        files, videos = delivery_files(manifest, str(output_root))
        for video in videos:
            if sha256(output_root / video["path"]) != video["sha256"]:
                raise ValueError("rendered video SHA256 mismatch")
        state.update({"complete": True, "status": "complete", "completed_at": timestamp(),
                      "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
                      "delivery_files": {name: sha256(output_root / name) for name in files}, "videos": videos})
        atomic_json(state_file, state)
        print(json.dumps(state), flush=True)
    except BaseException as exc:
        state.update({"complete": False, "status": "error", "error": f"{type(exc).__name__}: {exc}", "failed_at": timestamp()})
        atomic_json(state_file, state)
        raise


REMOTE_STATE_PROGRAM = """import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
result={}
for name in ('ghost_stage.json','pipeline_state.json','pipeline.json','pipeline_error.json'):
    path=root/name
    if path.is_file(): result[name]=json.loads(path.read_text())
print(json.dumps(result))
"""


def inspect_remote(target, remote_state):
    command = shlex.join(["python3", "-c", REMOTE_STATE_PROGRAM, remote_state])
    result = subprocess.run(["ssh", "-oBatchMode=yes", "-oConnectTimeout=20", "-oServerAliveInterval=15",
                             "-oServerAliveCountMax=2", "--", target, command],
                            check=True, capture_output=True, text=True, timeout=75)
    return json.loads(result.stdout)


def count_video_frames(path):
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("local video verification requires ffprobe on PATH")
    result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                             "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True, timeout=300)
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"expected one video stream: {path}")
    return int(streams[0]["nb_read_frames"])


REMOTE_VIDEO_PROGRAM = """import hashlib,json,pathlib,sys
from imageio_ffmpeg import count_frames_and_secs
path=pathlib.Path(sys.argv[1])
def digest():
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()
before=digest()
frames,seconds=count_frames_and_secs(str(path))
after=digest()
if before!=after: raise RuntimeError('remote video changed during frame verification')
print(json.dumps({'frames':int(frames),'seconds':float(seconds),'sha256':after}))
"""


def remote_video_evidence(target, python, path):
    if not python or not PurePosixPath(python).is_absolute():
        raise ValueError("remote frame verification requires absolute stage frame_verifier_python or --remote-python")
    command = shlex.join([python, "-c", REMOTE_VIDEO_PROGRAM, str(path)])
    result = subprocess.run(["ssh", "-oBatchMode=yes", "-oConnectTimeout=20", "-oServerAliveInterval=15",
                             "-oServerAliveCountMax=2", "--", target, command],
                            check=True, capture_output=True, text=True, timeout=300)
    return json.loads(result.stdout)


def receive_stage(args):
    target, remote_state, destination = args.paths
    if target.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_.@-]+", target):
        raise ValueError("SSH_TARGET must be a host alias or user@host, not SSH options")
    if not PurePosixPath(remote_state).is_absolute():
        raise ValueError("REMOTE_STATE_DIR must be absolute")
    if args.poll_seconds <= 0 or args.max_wait_seconds <= 0:
        raise ValueError("poll and deadline must be positive")
    if shutil.which("rsync") is None:
        raise RuntimeError("receiver requires local rsync")
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.max_wait_seconds
    consecutive_errors = 0
    print(f"[ghost-receiver] READY target={target} state={remote_state} local={destination}", flush=True)
    while time.monotonic() < deadline:
        try:
            snapshot = inspect_remote(target, remote_state)
            consecutive_errors = 0
        except (subprocess.SubprocessError, ValueError) as exc:
            consecutive_errors += 1
            if consecutive_errors >= 10:
                raise RuntimeError("ten consecutive remote state read failures") from exc
            print(f"[ghost-receiver] state read failed ({consecutive_errors}/10): {exc}", flush=True)
            time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
            continue
        for name, payload in snapshot.items():
            if name != "ghost_stage.json" and (payload.get("status") in {"error", "failed", "cancelled"} or payload.get("error")):
                raise RuntimeError(f"remote pipeline failed ({name}): {payload}")
        state = snapshot.get("ghost_stage.json", {})
        if state.get("status") == "error" or state.get("error"):
            raise RuntimeError(f"remote ghost stage failed: {state}")
        if state.get("complete") is True:
            break
        time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
    else:
        raise TimeoutError("remote ghost stage did not finish before receiver deadline")
    root = state["output_root"]
    if not PurePosixPath(root).is_absolute():
        raise ValueError("remote output root is not absolute")
    files = state["delivery_files"]
    if not isinstance(files, dict) or not files or len(files) > 100:
        raise ValueError("invalid delivery file inventory")
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\n" in name or "\r" in name:
            raise ValueError(f"unsafe relative transfer path: {name}")
        if path.suffix not in {".mp4", ".json", ".ok"}:
            raise ValueError(f"refusing non-video/manifest/marker transfer: {name}")
    # Exact file allowlist: never copy rollout pickles, checkpoints, or datasets.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as file_list:
        file_list.write("\n".join(sorted(files)) + "\n")
        file_list.flush()
        subprocess.run(["rsync", "-a", "--protect-args", "--timeout=120", "--files-from", file_list.name,
                        "-e", "ssh -oBatchMode=yes -oConnectTimeout=20 -oServerAliveInterval=15 -oServerAliveCountMax=2",
                        "--", f"{target}:{root.rstrip('/')}/", str(destination) + "/"], check=True, timeout=3600)
    for name, expected in files.items():
        if sha256(destination / name) != expected:
            raise ValueError(f"local file SHA256 mismatch: {name}")
    manifest_path = destination / "manifest.json"
    if sha256(manifest_path) != state["manifest_sha256"]:
        raise ValueError("local root manifest SHA256 mismatch")
    manifest = load_json(manifest_path)
    expected_files, videos = delivery_files(manifest, root)
    if set(expected_files) != set(files) or videos != state["videos"]:
        raise ValueError("stage inventory disagrees with ghost manifest")
    if [case["motion_key"] for case in manifest["cases"]] != state["motion_keys"]:
        raise ValueError("stage motion keys disagree with manifest")
    validate_identities(manifest, state)
    results = []
    for video in videos:
        path = destination / video["path"]
        local_sha = sha256(path)
        if local_sha != video["sha256"]:
            raise ValueError(f"local video SHA256 validation failed: {path}")
        if shutil.which("ffprobe") is not None:
            verified_frames = count_video_frames(path)
            verification_location = "local_ffprobe"
            evidence = {"sha256": local_sha}
        else:
            remote_python = args.remote_python or state.get("frame_verifier_python")
            evidence = remote_video_evidence(target, remote_python, PurePosixPath(root) / video["path"])
            if evidence["sha256"] != local_sha:
                raise ValueError(f"remote/local video SHA256 differs: {path}")
            verified_frames = int(evidence["frames"])
            verification_location = "remote_hash_identical/localcopy"
        if verified_frames != 193:
            raise ValueError(f"video frame validation failed: {path}: {verified_frames}")
        results.append({**video, "local_path": str(path), "verified_frames": verified_frames,
                        "verification_location": verification_location, "verification_evidence": evidence})
    atomic_json(destination / "ghost_stage.json", state)
    atomic_json(destination / "receipt.json", {"format_version": "mlp_cvae11_128_ghost_delivery_receipt_v1",
                "complete": True, "received_at": timestamp(), "ssh_target": target, "remote_state_dir": remote_state,
                "remote_output_root": root, "local_replay_dir": str(destination), "manifest_sha256": state["manifest_sha256"],
                "motion_keys": state["motion_keys"], "before_checkpoint": state["before_checkpoint"],
                "after_checkpoint": state["after_checkpoint"], "verified_files": files, "videos": results})
    print(f"[ghost-receiver] COMPLETE receipt={destination / 'receipt.json'} videos=4 frames=193 SHA256=PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    render = subparsers.add_parser("render")
    render.add_argument("paths", nargs=4, metavar="PATH", help="C_RUN D_RUN OUTPUT_STATE_DIR MIG")
    receive = subparsers.add_parser("receive")
    receive.add_argument("paths", nargs=3, metavar="PATH", help="SSH_TARGET REMOTE_STATE_DIR LOCAL_REPLAY_DIR")
    receive.add_argument("--poll-seconds", type=float, default=60)
    receive.add_argument("--max-wait-seconds", type=float, default=72 * 3600)
    receive.add_argument("--remote-python", help="Absolute remote Python with imageio_ffmpeg; default: stage's recorded interpreter")
    args = parser.parse_args()
    (render_stage if args.mode == "render" else receive_stage)(args)


if __name__ == "__main__":
    main()
