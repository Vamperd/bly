#!/usr/bin/env python3
"""Render SONIC trajectory recorder output with MuJoCo's off-screen renderer."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np


G1_ISAACLAB_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

# Indexing an Isaac Lab ordered vector with this list produces MuJoCo joint order.
G1_ISAACLAB_TO_MUJOCO_DOF = np.asarray(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int64,
)
G1_MUJOCO_JOINT_NAMES = tuple(
    G1_ISAACLAB_JOINT_NAMES[index] for index in G1_ISAACLAB_TO_MUJOCO_DOF
)


def load_trajectory(path: Path) -> dict[str, Any]:
    """Load trusted, locally generated TrajectoryRecorderTerm output."""
    with path.open("rb") as stream:
        trajectory = pickle.load(stream)  # noqa: S301 - input is a local SONIC artifact
    if not isinstance(trajectory, dict):
        raise ValueError(f"Trajectory must be a dictionary: {path}")
    return trajectory


def build_mujoco_qpos(trajectory: dict[str, Any]) -> np.ndarray:
    """Convert recorded Isaac Lab state into MuJoCo's 36-value qpos layout."""
    missing = [key for key in ("dof_pos", "root_pos_w", "root_quat_w") if key not in trajectory]
    if missing:
        raise ValueError(f"Trajectory is missing required fields: {', '.join(missing)}")

    dof_pos = np.asarray(trajectory["dof_pos"], dtype=np.float64)
    root_pos = np.asarray(trajectory["root_pos_w"], dtype=np.float64)
    root_quat = np.asarray(trajectory["root_quat_w"], dtype=np.float64)

    if dof_pos.ndim != 2 or dof_pos.shape[1] != 29:
        raise ValueError(f"Expected dof_pos shape (T, 29), found {dof_pos.shape}")
    if root_pos.shape != (dof_pos.shape[0], 3):
        raise ValueError(f"Expected root_pos_w shape {(dof_pos.shape[0], 3)}, found {root_pos.shape}")
    if root_quat.shape != (dof_pos.shape[0], 4):
        raise ValueError(
            f"Expected root_quat_w shape {(dof_pos.shape[0], 4)}, found {root_quat.shape}"
        )
    if dof_pos.shape[0] == 0:
        raise ValueError("Trajectory contains no frames")

    for name, values in (
        ("dof_pos", dof_pos),
        ("root_pos_w", root_pos),
        ("root_quat_w", root_quat),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"Trajectory field {name} contains NaN or infinite values")

    quat_norm = np.linalg.norm(root_quat, axis=1)
    if np.any(quat_norm < 1.0e-8):
        raise ValueError("Trajectory contains a zero-length root quaternion")
    if np.any(np.abs(quat_norm - 1.0) > 5.0e-3):
        worst = float(np.max(np.abs(quat_norm - 1.0)))
        raise ValueError(f"Root quaternion norm differs from 1 by as much as {worst:.6f}")
    root_quat = root_quat / quat_norm[:, None]

    if sorted(G1_ISAACLAB_TO_MUJOCO_DOF.tolist()) != list(range(29)):
        raise RuntimeError("G1 joint mapping is not a permutation of 0..28")
    mujoco_joints = dof_pos[:, G1_ISAACLAB_TO_MUJOCO_DOF]
    qpos = np.concatenate((root_pos, root_quat, mujoco_joints), axis=1)
    if qpos.shape != (dof_pos.shape[0], 36):
        raise RuntimeError(f"Internal qpos shape error: {qpos.shape}")
    return qpos


def _remove_matching_children(root: ET.Element, tag: str, attribute: str, value: str) -> int:
    removed = 0
    for parent in root.iter():
        for child in list(parent):
            if child.tag == tag and child.get(attribute) == value:
                parent.remove(child)
                removed += 1
    return removed


def prepare_runtime_xml(source_path: Path, destination_path: Path) -> dict[str, Any]:
    """Create a run-local model copy with absolute robot meshes and no missing terrain mesh."""
    source_path = source_path.resolve(strict=True)
    tree = ET.parse(source_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    mesh_dir = (source_path.parent / "meshes").resolve(strict=True)
    compiler.set("meshdir", str(mesh_dir))

    removed_meshes = _remove_matching_children(root, "mesh", "name", "terrain_mesh")
    removed_geoms = _remove_matching_children(root, "geom", "mesh", "terrain_mesh")
    removed_bodies = _remove_matching_children(root, "body", "name", "terrain_body")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(destination_path, encoding="utf-8", xml_declaration=True)
    return {
        "mesh_dir": str(mesh_dir),
        "removed_terrain_meshes": removed_meshes,
        "removed_terrain_geoms": removed_geoms,
        "removed_terrain_bodies": removed_bodies,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--gl", choices=("osmesa", "egl", "glfw"), default="osmesa")
    parser.add_argument("--camera-distance", type=float, default=2.0)
    parser.add_argument("--camera-azimuth", type=float, default=120.0)
    parser.add_argument("--camera-elevation", type=float, default=-30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        raise ValueError("Video width and height must be positive even integers")
    if args.camera_distance <= 0:
        raise ValueError("Camera distance must be positive")

    trajectory_path = args.trajectory.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    output_path = args.output.resolve(strict=False)
    manifest_dir = args.manifest_dir.resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing video: {output_path}")

    # MuJoCo selects its OpenGL implementation at import time.
    os.environ["MUJOCO_GL"] = args.gl
    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    import mujoco

    trajectory = load_trajectory(trajectory_path)
    qpos_frames = build_mujoco_qpos(trajectory)
    fps = float(trajectory.get("fps", 0.0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Trajectory FPS must be positive and finite, found {fps}")

    stem = trajectory_path.name.removesuffix(".trajectory.pkl")
    runtime_xml = manifest_dir / f"g1_offline_render_{stem}.xml"
    xml_details = prepare_runtime_xml(model_path, runtime_xml)

    model = mujoco.MjModel.from_xml_path(str(runtime_xml))
    if model.nq != 36:
        raise ValueError(f"Expected MuJoCo model nq=36, found {model.nq}")
    if qpos_frames.shape[1] != model.nq:
        raise ValueError(f"Trajectory qpos width {qpos_frames.shape[1]} != model nq {model.nq}")
    model_joint_names = tuple(
        model.joint(index).name
        for index in range(model.njnt)
        if model.joint(index).name != "floating_base_joint"
    )
    if model_joint_names != G1_MUJOCO_JOINT_NAMES:
        raise ValueError(
            "MuJoCo model joint order does not match SONIC's canonical G1 order: "
            f"{model_joint_names}"
        )

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise ValueError("MuJoCo model does not contain a pelvis body")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = pelvis_id
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    camera.distance = args.camera_distance
    camera.lookat[:] = qpos_frames[0, :3]

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=5,
        pixelformat="yuv420p",
        macro_block_size=2,
    )
    try:
        for frame_qpos in qpos_frames:
            data.qpos[:] = frame_qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Renderer did not produce a non-empty video: {output_path}")
    encoded_frames, encoded_duration = count_frames_and_secs(str(output_path))
    if encoded_frames != qpos_frames.shape[0]:
        raise RuntimeError(
            f"Encoded video has {encoded_frames} frames; expected {qpos_frames.shape[0]}"
        )

    manifest = {
        "renderer": "mujoco_offscreen",
        "gl_backend": args.gl,
        "trajectory": str(trajectory_path),
        "source_model": str(model_path),
        "runtime_model": str(runtime_xml.resolve()),
        "output": str(output_path),
        "frames": int(qpos_frames.shape[0]),
        "encoded_frames": int(encoded_frames),
        "fps": fps,
        "duration_seconds": float(qpos_frames.shape[0] / fps),
        "encoded_duration_seconds": float(encoded_duration),
        "width": args.width,
        "height": args.height,
        "camera": {
            "type": "tracking",
            "body": "pelvis",
            "distance": args.camera_distance,
            "azimuth": args.camera_azimuth,
            "elevation": args.camera_elevation,
        },
        "quaternion_format": "wxyz",
        "isaaclab_joint_order": list(G1_ISAACLAB_JOINT_NAMES),
        "mujoco_joint_order": list(model_joint_names),
        "joint_mapping": G1_ISAACLAB_TO_MUJOCO_DOF.tolist(),
        "xml_patch": xml_details,
        "video_size_bytes": output_path.stat().st_size,
    }
    manifest_path = manifest_dir / f"{stem}.render.json"
    write_json_atomic(manifest_path, manifest)
    print(
        f"Rendered {qpos_frames.shape[0]} frames at {fps:.3f} FPS "
        f"with MUJOCO_GL={args.gl}: {output_path}"
    )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
