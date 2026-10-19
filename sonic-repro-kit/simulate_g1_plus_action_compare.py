#!/usr/bin/env python3
"""Run and render a real MuJoCo rollout with zero and changing controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
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


def joint_id(model, name: str) -> int:
    import mujoco

    value = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    if value < 0:
        raise ValueError(f"model has no joint named {name!r}")
    return value


def actuator_ids_for_joint(model, name: str) -> list[int]:
    jid = joint_id(model, name)
    return [
        index
        for index in range(int(model.nu))
        if int(model.actuator_trnid[index, 0]) == jid
    ]


def add_collision_ground(path: Path) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("runtime XML has no worldbody")
    floor = worldbody.find("./geom[@name='sonic_floor']")
    if floor is None:
        raise RuntimeError("runtime XML has no sonic_floor geom")
    floor.set("contype", "1")
    floor.set("conaffinity", "1")
    floor.set("friction", "1 0.005 0.0001")
    floor.set("condim", "3")
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def source_initial_qpos(model, trajectory_path: Path) -> np.ndarray:
    source = load_trajectory(trajectory_path)
    required = ("dof_pos", "root_pos_w", "root_quat_w")
    missing = [name for name in required if name not in source]
    if missing:
        raise ValueError(f"source trajectory is missing fields: {missing}")
    values = {
        name: np.asarray(source[name])[:1].copy()
        for name in required
    }
    if np.asarray(values["dof_pos"]).shape != (1, 29):
        raise ValueError("source trajectory must contain 29 DoF positions")
    neck = {
        name: np.asarray([float(model.qpos0[int(model.jnt_qposadr[joint_id(model, name)])])])
        for name in NECK_NAMES
    }
    qpos = build_mujoco_qpos(
        values,
        profile=PROFILE,
        model=model,
        extra_joint_values=neck,
    )
    return qpos[0]


def action_traces(model, horizon: int, scope: str, amplitude_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 < amplitude_fraction <= 1.0:
        raise ValueError("amplitude fraction must be in (0, 1]")

    static = np.zeros((horizon, int(model.nu)), dtype=np.float64)
    moving = np.zeros_like(static)
    normalized = np.linspace(
        -amplitude_fraction,
        amplitude_fraction,
        horizon,
        dtype=np.float64,
    )

    if scope == "neck":
        names = NECK_NAMES
    elif scope == "all":
        names = tuple(str(model.actuator(index).name) for index in range(int(model.nu)))
    else:
        raise ValueError(f"unknown moving scope: {scope}")

    if scope == "all":
        selected = range(int(model.nu))
    else:
        selected = [
            actuator_id
            for name in names
            for actuator_id in actuator_ids_for_joint(model, name)
        ]

    for actuator_id in selected:
        low, high = map(float, model.actuator_ctrlrange[actuator_id])
        center = 0.5 * (low + high)
        half = 0.5 * (high - low)
        moving[:, actuator_id] = center + half * normalized

    return static, moving


def rollout(model, initial_qpos: np.ndarray, actions: np.ndarray) -> np.ndarray:
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    qposes = np.empty((actions.shape[0] + 1, int(model.nq)), dtype=np.float64)
    qposes[0] = data.qpos
    for step, control in enumerate(actions):
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f"non-finite MuJoCo state at step {step}")
        qposes[step + 1] = data.qpos
    return qposes


def render_variant(
    *, imageio, cv2, count_frames_and_secs, mujoco, model, qposes, actions,
    name: str, output: Path, camera_distance: float, camera_azimuth: float,
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
    camera.lookat[:] = qposes[0, :3]

    neck_addresses = {
        name: int(model.jnt_qposadr[joint_id(model, name)])
        for name in NECK_NAMES
    }
    writer = _writer(imageio, output, FPS)
    try:
        for frame_index, qpos in enumerate(qposes):
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            camera.lookat[0:2] = qpos[0:2]
            camera.lookat[2] = qposes[0, 2]
            renderer.update_scene(data, camera=camera)
            frame = renderer.render().copy()
            cv2.rectangle(frame, (0, 0), (960, 140), (25, 25, 25), -1)
            cv2.putText(
                frame,
                f"G1+ unlocked31 | dynamics | {name} | frame {frame_index}/{len(qposes)-1}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1,
            )
            cv2.putText(
                frame,
                f"neck_pitch={qpos[neck_addresses['neck_pitch_joint']]:+.4f} rad",
                (10, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 220, 255), 1,
            )
            cv2.putText(
                frame,
                f"neck_yaw={qpos[neck_addresses['neck_yaw_joint']]:+.4f} rad",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 220, 255), 1,
            )
            if frame_index < actions.shape[0]:
                max_control = float(np.max(np.abs(actions[frame_index])))
            else:
                max_control = 0.0
            cv2.putText(
                frame,
                f"mj_step | max_ctrl={max_control:.4f} | 50 Hz | collision ground",
                (10, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1,
            )
            cv2.putText(
                frame,
                "initial qvel=0; actual MuJoCo rollout",
                (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1,
            )
            writer.append_data(frame)
    finally:
        writer.close()
        renderer.close()

    frames, seconds = count_frames_and_secs(str(output))
    if frames != len(qposes) or output.stat().st_size <= 0:
        raise RuntimeError(
            f"video validation failed: {output}, frames={frames}, expected={len(qposes)}"
        )
    return {
        "path": str(output),
        "sha256": sha256(output),
        "frames": int(frames),
        "seconds": float(seconds),
        "variant": name,
    }


def render(args: argparse.Namespace) -> dict[str, object]:
    output = args.output_run.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output run must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("data", "diagnostics", "manifests", "videos"):
        (output / folder).mkdir()

    model_path = args.model.expanduser().resolve(strict=True)
    source_path = args.initial_trajectory.expanduser().resolve(strict=True)
    horizon = int(args.horizon_steps)

    os.environ["MUJOCO_GL"] = args.gl
    import cv2
    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    import mujoco

    runtime_xml = output / "manifests" / "g1_plus_unlocked31_dynamics_runtime.xml"
    xml_details = prepare_runtime_xml(
        model_path,
        runtime_xml,
        width=960,
        height=540,
        mesh_dir=args.mesh_dir,
        ground_style="black",
    )
    add_collision_ground(runtime_xml)
    model = mujoco.MjModel.from_xml_path(str(runtime_xml))
    profile = validate_model_profile(model, PROFILE)
    initial_qpos = source_initial_qpos(model, source_path)
    static_actions, moving_actions = action_traces(
        model, horizon, args.moving_scope, args.amplitude_fraction
    )
    if np.max(np.abs(static_actions)) != 0.0:
        raise RuntimeError("static action trace is not exactly zero")

    qpos_static = rollout(model, initial_qpos, static_actions)
    qpos_moving = rollout(model, initial_qpos, moving_actions)
    np.savez_compressed(
        output / "data" / "action_rollouts.npz",
        static_actions=static_actions,
        moving_actions=moving_actions,
        static_qpos=qpos_static,
        moving_qpos=qpos_moving,
    )

    videos = [
        render_variant(
            imageio=imageio, cv2=cv2, count_frames_and_secs=count_frames_and_secs,
            mujoco=mujoco, model=model, qposes=qpos_static, actions=static_actions,
            name="action_static_zero", output=output / "videos" / "action_static_zero.mp4",
            camera_distance=args.camera_distance, camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
        ),
        render_variant(
            imageio=imageio, cv2=cv2, count_frames_and_secs=count_frames_and_secs,
            mujoco=mujoco, model=model, qposes=qpos_moving, actions=moving_actions,
            name=f"action_moving_{args.moving_scope}", output=output / "videos" / "action_moving.mp4",
            camera_distance=args.camera_distance, camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
        ),
    ]

    neck_action_ids = {
        name: actuator_ids_for_joint(model, name)
        for name in NECK_NAMES
    }
    manifest = {
        "version": "g1_plus_unlocked31_action_dynamics_compare_v1",
        "profile": PROFILE,
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "runtime_xml": str(runtime_xml),
        "runtime_xml_sha256": sha256(runtime_xml),
        "initial_trajectory": str(source_path),
        "initial_trajectory_sha256": sha256(source_path),
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "model_njnt": int(model.njnt),
        "model_nu": int(model.nu),
        "horizon_steps": horizon,
        "frames": horizon + 1,
        "fps": FPS,
        "control_timestep": float(model.opt.timestep),
        "initial_qvel": "zero",
        "action_units": "MuJoCo actuator ctrl; not canonical 29-D Action",
        "static_action_max_abs": float(np.max(np.abs(static_actions))),
        "moving_scope": args.moving_scope,
        "moving_amplitude_fraction": float(args.amplitude_fraction),
        "neck_actuator_ids": neck_action_ids,
        "moving_action_min": moving_actions.min(axis=0).tolist(),
        "moving_action_max": moving_actions.max(axis=0).tolist(),
        "gravity": np.asarray(model.opt.gravity, dtype=float).tolist(),
        "collision_ground": True,
        "kinematic_only": False,
        "quality_pass": None,
        "runtime_ground": xml_details["ground"],
        "videos": videos,
    }
    (output / "manifests" / "action_dynamics_compare.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--initial-trajectory", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path)
    parser.add_argument("--horizon-steps", type=int, default=192)
    parser.add_argument("--moving-scope", choices=("neck", "all"), default="neck")
    parser.add_argument("--amplitude-fraction", type=float, default=0.35)
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
