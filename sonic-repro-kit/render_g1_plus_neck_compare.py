#!/usr/bin/env python3
"""Render one full 29-DoF body trajectory with fixed and moving G1+ necks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _joint_id(model, name: str) -> int:
    import mujoco

    joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    if joint_id < 0:
        raise ValueError(f"MuJoCo model has no joint named {name!r}")
    return joint_id


def _neck_qpos_addresses(model) -> dict[str, int]:
    return {
        name: int(model.jnt_qposadr[_joint_id(model, name)])
        for name in NECK_NAMES
    }


def _neck_traces(model, frames: int, amplitude_fraction: float) -> dict[str, np.ndarray]:
    if frames <= 1:
        raise ValueError("trajectory must contain at least two frames")
    if not 0.0 < amplitude_fraction < 1.0:
        raise ValueError("neck amplitude fraction must be in (0, 1)")

    addresses = _neck_qpos_addresses(model)
    moving: dict[str, np.ndarray] = {}
    phase = np.linspace(0.0, 2.0 * np.pi, frames, dtype=np.float64)

    for name in NECK_NAMES:
        joint_id = _joint_id(model, name)
        lower, upper = map(float, model.jnt_range[joint_id])
        if not np.isfinite([lower, upper]).all() or not upper > lower:
            raise ValueError(f"invalid range for {name}: {(lower, upper)}")
        center = float(model.qpos0[addresses[name]])
        half_range = 0.5 * (upper - lower)
        amplitude = amplitude_fraction * half_range
        values = center + amplitude * np.sin(
            phase + (np.pi / 2.0 if name == "neck_pitch_joint" else 0.0)
        )
        if np.any(values <= lower) or np.any(values >= upper):
            raise ValueError(f"generated {name} trace exceeds joint limits")
        moving[name] = values

    return moving


def _slice_body_trajectory(trajectory: dict[str, object], frames: int) -> dict[str, object]:
    required = ("dof_pos", "root_pos_w", "root_quat_w")
    missing = [name for name in required if name not in trajectory]
    if missing:
        raise ValueError(f"trajectory is missing fields: {missing}")

    dof_pos = np.asarray(trajectory["dof_pos"])
    if dof_pos.ndim != 2 or dof_pos.shape[1] != 29:
        raise ValueError(f"expected dof_pos shape (T, 29), found {dof_pos.shape}")
    if dof_pos.shape[0] < frames:
        raise ValueError(
            f"trajectory has {dof_pos.shape[0]} frames; need at least {frames}"
        )

    result = {
        name: np.asarray(trajectory[name])[:frames].copy()
        for name in required
    }
    result["fps"] = float(trajectory.get("fps", FPS))
    if not np.isfinite(result["fps"]) or abs(result["fps"] - FPS) > 1.0e-6:
        raise ValueError(f"expected {FPS} Hz trajectory, found {result['fps']}")
    return result


def _render_variant(
    *,
    imageio,
    cv2,
    count_frames_and_secs,
    mujoco,
    model,
    qpos: np.ndarray,
    neck_values: dict[str, np.ndarray],
    name: str,
    output: Path,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
) -> dict[str, object]:
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=540, width=960)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = camera_azimuth
    camera.elevation = camera_elevation
    camera.distance = camera_distance
    camera.lookat[:] = qpos[0, :3]

    writer = _writer(imageio, output, FPS)
    try:
        for frame_index in range(qpos.shape[0]):
            data.qpos[:] = qpos[frame_index]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            camera.lookat[0:2] = qpos[frame_index, 0:2]
            camera.lookat[2] = qpos[0, 2]
            renderer.update_scene(data, camera=camera)
            frame = renderer.render().copy()
            cv2.rectangle(frame, (0, 0), (960, 116), (25, 25, 25), -1)
            cv2.putText(
                frame,
                f"G1+ unlocked31 | {name} | frame {frame_index}/{qpos.shape[0] - 1}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                1,
            )
            cv2.putText(
                frame,
                f"neck_pitch={neck_values['neck_pitch_joint'][frame_index]:+.4f} rad",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (80, 220, 255),
                1,
            )
            cv2.putText(
                frame,
                f"neck_yaw={neck_values['neck_yaw_joint'][frame_index]:+.4f} rad",
                (10, 73),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (80, 220, 255),
                1,
            )
            cv2.putText(
                frame,
                "kinematic replay | 50 Hz | body source unchanged",
                (10, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (230, 230, 230),
                1,
            )
            writer.append_data(frame)
    finally:
        writer.close()
        renderer.close()

    frames, seconds = count_frames_and_secs(str(output))
    if frames != qpos.shape[0] or output.stat().st_size <= 0:
        raise RuntimeError(
            f"video validation failed for {output}: frames={frames}, "
            f"expected={qpos.shape[0]}, size={output.stat().st_size}"
        )
    return {
        "path": str(output),
        "sha256": sha256(output),
        "frames": int(frames),
        "seconds": float(seconds),
        "variant": name,
    }


def render(args: argparse.Namespace) -> dict[str, object]:
    if args.horizon_steps <= 0:
        raise ValueError("horizon-steps must be positive")
    output = args.output_run.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output run must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("data", "diagnostics", "manifests", "videos"):
        (output / folder).mkdir()

    model_path = args.model.expanduser().resolve(strict=True)
    trajectory_path = args.body_trajectory.expanduser().resolve(strict=True)
    frames = args.horizon_steps + 1

    os.environ["MUJOCO_GL"] = args.gl
    import cv2
    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    import mujoco

    runtime_xml = output / "manifests" / "g1_plus_unlocked31_runtime.xml"
    xml_details = prepare_runtime_xml(
        model_path,
        runtime_xml,
        width=960,
        height=540,
        mesh_dir=args.mesh_dir,
        ground_style=args.ground_style,
    )
    model = mujoco.MjModel.from_xml_path(str(runtime_xml))
    profile = validate_model_profile(model, PROFILE)

    source = _slice_body_trajectory(load_trajectory(trajectory_path), frames)
    neck_moving = _neck_traces(model, frames, args.neck_amplitude_fraction)
    neck_static = {
        name: np.full(frames, float(model.qpos0[_neck_qpos_addresses(model)[name]]))
        for name in NECK_NAMES
    }

    qpos_static = build_mujoco_qpos(
        source,
        profile=PROFILE,
        model=model,
        extra_joint_values=neck_static,
    )
    qpos_moving = build_mujoco_qpos(
        source,
        profile=PROFILE,
        model=model,
        extra_joint_values=neck_moving,
    )
    if not np.allclose(qpos_static[:, :7], qpos_moving[:, :7]):
        raise RuntimeError("static and moving variants changed the body root trajectory")

    np.savez_compressed(
        output / "data" / "qpos_variants.npz",
        qpos_static=qpos_static,
        qpos_moving=qpos_moving,
        neck_static_pitch=neck_static["neck_pitch_joint"],
        neck_static_yaw=neck_static["neck_yaw_joint"],
        neck_moving_pitch=neck_moving["neck_pitch_joint"],
        neck_moving_yaw=neck_moving["neck_yaw_joint"],
    )

    videos = [
        _render_variant(
            imageio=imageio,
            cv2=cv2,
            count_frames_and_secs=count_frames_and_secs,
            mujoco=mujoco,
            model=model,
            qpos=qpos_static,
            neck_values=neck_static,
            name="neck_static",
            output=output / "videos" / "neck_static.mp4",
            camera_distance=args.camera_distance,
            camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
        ),
        _render_variant(
            imageio=imageio,
            cv2=cv2,
            count_frames_and_secs=count_frames_and_secs,
            mujoco=mujoco,
            model=model,
            qpos=qpos_moving,
            neck_values=neck_moving,
            name="neck_moving",
            output=output / "videos" / "neck_moving.mp4",
            camera_distance=args.camera_distance,
            camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
        ),
    ]

    manifest = {
        "version": "g1_plus_unlocked31_full_trajectory_neck_compare_v1",
        "profile": PROFILE,
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "runtime_xml": str(runtime_xml),
        "runtime_xml_sha256": sha256(runtime_xml),
        "body_trajectory": str(trajectory_path),
        "body_trajectory_sha256": sha256(trajectory_path),
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "model_njnt": int(model.njnt),
        "model_joint_names": profile["model_joint_names"],
        "horizon_steps": int(args.horizon_steps),
        "frames": int(frames),
        "fps": FPS,
        "neck_amplitude_fraction": float(args.neck_amplitude_fraction),
        "neck_static": {
            name: float(values[0]) for name, values in neck_static.items()
        },
        "neck_moving_ranges": {
            name: [float(values.min()), float(values.max())]
            for name, values in neck_moving.items()
        },
        "body_root_identical_between_variants": True,
        "kinematic_only": True,
        "quality_pass": None,
        "runtime_ground": xml_details["ground"],
        "videos": videos,
    }
    (output / "manifests" / "neck_compare.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--body-trajectory", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path)
    parser.add_argument("--horizon-steps", type=int, default=192)
    parser.add_argument("--neck-amplitude-fraction", type=float, default=0.35)
    parser.add_argument("--ground-style", choices=("black", "checker"), default="black")
    parser.add_argument("--gl", choices=("egl", "osmesa", "glfw"), default="egl")
    parser.add_argument("--camera-distance", type=float, default=3.5)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-25.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    render(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
