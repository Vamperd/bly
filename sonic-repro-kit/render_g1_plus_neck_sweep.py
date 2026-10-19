#!/usr/bin/env python3
"""Render a deterministic G1+ unlocked31 neck sweep over a held body pose."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from render_h50a_exact_action_replays import _writer
from render_mujoco_trajectory import (
    build_mujoco_qpos,
    load_trajectory,
    prepare_runtime_xml,
    validate_model_profile,
)


PROFILE = "g1_plus_unlocked31"
NECK_NAMES = ("neck_pitch_joint", "neck_yaw_joint")
FPS = 50.0
LIMIT_MARGIN = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _joint_id(model, name: str) -> int:
    import mujoco

    try:
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    except Exception as error:
        raise ValueError(f"MuJoCo model has no joint named {name!r}") from error
    if joint_id < 0:
        raise ValueError(f"MuJoCo model has no joint named {name!r}")
    return joint_id


def _joint_qpos_addresses(model, names: tuple[str, ...]) -> dict[str, int]:
    return {name: int(model.jnt_qposadr[_joint_id(model, name)]) for name in names}


def _limited_sweep(model, name: str, horizon: int) -> tuple[np.ndarray, dict[str, float]]:
    joint_id = _joint_id(model, name)
    if int(model.jnt_limited[joint_id]) != 1:
        raise ValueError(f"neck joint {name!r} has no finite MuJoCo range")
    lower, upper = map(float, model.jnt_range[joint_id])
    if not np.isfinite([lower, upper]).all() or not upper > lower:
        raise ValueError(f"invalid MuJoCo range for {name!r}: {(lower, upper)}")
    margin = LIMIT_MARGIN * (upper - lower)
    safe_lower, safe_upper = lower + margin, upper - margin
    return np.linspace(safe_lower, safe_upper, horizon + 1, dtype=np.float64), {
        "model_lower": lower,
        "model_upper": upper,
        "safe_lower": safe_lower,
        "safe_upper": safe_upper,
        "margin_fraction": LIMIT_MARGIN,
    }


def _select_body_frame(state_path: Path | None, requested: int, trajectory_length: int) -> tuple[int, dict[str, object]]:
    if requested >= 0:
        if requested >= trajectory_length:
            raise ValueError(f"body frame {requested} is outside trajectory length {trajectory_length}")
        return requested, {"mode": "explicit", "frame": requested}
    if state_path is None:
        raise ValueError("automatic upright selection requires --body-state")
    with np.load(state_path, allow_pickle=False) as values:
        if "physical_state" not in values:
            raise ValueError(f"body state archive has no physical_state: {state_path}")
        state = np.asarray(values["physical_state"], dtype=np.float64)
    if state.ndim != 2 or state.shape[1] < 70:
        raise ValueError(f"expected physical_state shape (T, >=70), found {state.shape}")
    count = min(trajectory_length, state.shape[0])
    lin = np.linalg.norm(state[:count, 58:61], axis=1)
    ang = np.linalg.norm(state[:count, 61:64], axis=1)
    contacts = state[:count, 68:70]
    finite = np.isfinite(state[:count, :70]).all(axis=1)
    candidates = np.flatnonzero(
        finite & (contacts[:, 0] >= 0.5) & (contacts[:, 1] >= 0.5)
        & (lin <= 0.05) & (ang <= 0.05)
    )
    if len(candidates) == 0:
        raise ValueError("no upright frame with bilateral contact and near-zero base velocity")
    frame = int(candidates[0])
    return frame, {
        "mode": "automatic_upright",
        "frame": frame,
        "base_lin_speed": float(lin[frame]),
        "base_ang_speed": float(ang[frame]),
        "foot_contact": contacts[frame].tolist(),
        "state_source": str(state_path),
        "state_source_sha256": sha256(state_path),
    }


def _load_body_frame(path: Path, frame: int) -> dict[str, np.ndarray]:
    trajectory = load_trajectory(path)
    required = ("dof_pos", "root_pos_w", "root_quat_w")
    missing = [name for name in required if name not in trajectory]
    if missing:
        raise ValueError(f"body trajectory is missing fields: {missing}")
    count = int(np.asarray(trajectory["dof_pos"]).shape[0])
    if not 0 <= frame < count:
        raise ValueError(f"body frame {frame} is outside trajectory length {count}")
    body = {
        name: np.asarray(trajectory[name])[frame : frame + 1].copy()
        for name in required
    }
    if body["dof_pos"].shape != (1, 29):
        raise ValueError(f"expected body dof_pos shape (1, 29), found {body['dof_pos'].shape}")
    return body


def render(args: argparse.Namespace) -> dict[str, object]:
    import cv2
    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    import mujoco

    output = args.output_run.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output run must be new and empty: {output}")
    if args.horizon_steps <= 0:
        raise ValueError("horizon-steps must be positive")
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("assets", "data", "diagnostics", "logs", "manifests", "videos"):
        (output / folder).mkdir()

    args.model = args.model.expanduser().resolve(strict=True)
    args.body_trajectory = args.body_trajectory.expanduser().resolve(strict=True)
    xml_path = output / "manifests" / "g1_plus_unlocked31_runtime.xml"
    xml_details = prepare_runtime_xml(
        args.model,
        xml_path,
        width=640,
        height=480,
        mesh_dir=args.mesh_dir,
        ground_style=args.ground_style,
    )

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    profile = validate_model_profile(model, PROFILE)
    addresses = _joint_qpos_addresses(model, NECK_NAMES)
    yaw, yaw_limits = _limited_sweep(model, "neck_yaw_joint", args.horizon_steps)
    pitch, pitch_limits = _limited_sweep(model, "neck_pitch_joint", args.horizon_steps)

    trajectory_length = int(load_trajectory(args.body_trajectory)["dof_pos"].shape[0])
    body_frame, body_selection = _select_body_frame(args.body_state, args.body_frame, trajectory_length)
    body = _load_body_frame(args.body_trajectory, body_frame)
    body_repeat = {
        name: np.repeat(value, args.horizon_steps + 1, axis=0)
        for name, value in body.items()
    }
    qpos = build_mujoco_qpos(
        body_repeat,
        profile=PROFILE,
        model=model,
        extra_joint_values={
            "neck_pitch_joint": pitch,
            "neck_yaw_joint": yaw,
        },
    )

    neck_addresses = set(addresses.values())
    body_addresses = [index for index in range(model.nq) if index not in neck_addresses]
    if not np.allclose(qpos[:, body_addresses], qpos[0, body_addresses]):
        raise RuntimeError("body qpos changed during the neck-only sweep")
    if not np.isfinite(qpos).all():
        raise RuntimeError("neck sweep qpos contains NaN or infinite values")

    np.savez_compressed(
        output / "data" / "neck_sweep.npz",
        qpos=qpos,
        neck_pitch=pitch,
        neck_yaw=yaw,
        body_qpos=qpos[:, body_addresses],
    )

    model_manifest = {
        "version": "g1_plus_unlocked31_model_manifest_v1",
        "profile": PROFILE,
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "model_njnt": int(model.njnt),
        "mesh_count": int(model.nmesh),
        "model_joint_names": profile["model_joint_names"],
        "body_policy_joint_names": profile["policy_joint_names"],
        "neck_joint_names": list(NECK_NAMES),
        "neck_qpos_addresses": addresses,
        "neck_joint_ids": {
            name: _joint_id(model, name)
            for name in NECK_NAMES
        },
        "neck_joint_ranges": {
            name: [float(value) for value in model.jnt_range[_joint_id(model, name)]]
            for name in NECK_NAMES
        },
        "fixed_joint_names": profile["fixed_joint_names"],
    }
    (output / "manifests" / "unlocked31_model_manifest.json").write_text(
        json.dumps(model_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 135
    camera.elevation = -25
    camera.distance = 3.5
    camera.lookat[:] = qpos[0, :3]

    video = output / "videos" / "neck_sweep.mp4"
    writer = _writer(imageio, video, FPS)
    try:
        for frame_index in range(args.horizon_steps + 1):
            data.qpos[:] = qpos[frame_index]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = renderer.render().copy()
            cv2.rectangle(frame, (0, 0), (640, 126), (25, 25, 25), -1)
            cv2.putText(frame, f"G1+ unlocked31 | frame {frame_index}/{args.horizon_steps}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .52, (245, 245, 245), 1)
            cv2.putText(frame, f"neck_yaw={yaw[frame_index]:+.4f} rad | left -> right",
                        (10, 49), cv2.FONT_HERSHEY_SIMPLEX, .43, (80, 220, 255), 1)
            cv2.putText(frame, f"neck_pitch={pitch[frame_index]:+.4f} rad | up -> down",
                        (10, 72), cv2.FONT_HERSHEY_SIMPLEX, .43, (80, 220, 255), 1)
            cv2.putText(frame, f"nq={model.nq} nv={model.nv} | 50 Hz | black ground",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, .43, (230, 230, 230), 1)
            cv2.putText(frame, f"pitch range [{pitch_limits['safe_lower']:+.3f}, {pitch_limits['safe_upper']:+.3f}] | "
                                  f"yaw range [{yaw_limits['safe_lower']:+.3f}, {yaw_limits['safe_upper']:+.3f}]",
                        (10, 118), cv2.FONT_HERSHEY_SIMPLEX, .34, (230, 230, 230), 1)
            writer.append_data(frame)
    finally:
        writer.close()
        renderer.close()

    frames, seconds = count_frames_and_secs(str(video))
    if frames != args.horizon_steps + 1 or video.stat().st_size <= 0:
        raise RuntimeError(f"video validation failed: frames={frames}, size={video.stat().st_size}")

    manifest = {
        "version": "g1_plus_unlocked31_neck_sweep_v1",
        "profile": PROFILE,
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "runtime_xml": str(xml_path),
        "runtime_xml_sha256": sha256(xml_path),
        "body_source": str(args.body_trajectory),
        "body_source_sha256": sha256(args.body_trajectory),
        "body_frame": int(body_frame),
        "body_frame_selection": body_selection,
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "model_njnt": int(model.njnt),
        "model_joint_names": profile["model_joint_names"],
        "body_policy_joint_names": profile["policy_joint_names"],
        "neck_joint_names": list(NECK_NAMES),
        "neck_qpos_addresses": addresses,
        "joint_limits": {"neck_pitch_joint": pitch_limits, "neck_yaw_joint": yaw_limits},
        "sweep": {
            "horizon_steps": int(args.horizon_steps),
            "frames": int(frames),
            "fps": FPS,
            "seconds": float(seconds),
            "yaw_start": float(yaw[0]),
            "yaw_end": float(yaw[-1]),
            "pitch_start": float(pitch[0]),
            "pitch_end": float(pitch[-1]),
            "body_qpos_constant": True,
        },
        "ground_style": args.ground_style,
        "runtime_ground": xml_details["ground"],
        "replaced_runtime_elements": xml_details["replaced_runtime_elements"],
        "video": {"path": str(video), "sha256": sha256(video), "frames": int(frames), "seconds": float(seconds)},
    }
    (output / "manifests" / "neck_sweep.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--body-trajectory", type=Path, required=True)
    parser.add_argument("--body-state", type=Path)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path)
    parser.add_argument("--body-frame", type=int, default=-1,
                        help="explicit body frame; default selects the first upright bilateral-contact frame")
    parser.add_argument("--horizon-steps", type=int, default=192)
    parser.add_argument("--ground-style", choices=("black", "checker"), default="black")
    parser.add_argument("--gl", choices=("egl", "osmesa", "glfw"), default="egl")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import os
    os.environ["MUJOCO_GL"] = args.gl
    render(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
