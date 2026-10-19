#!/usr/bin/env python3
"""Ghost-compare source and C body replay trajectories on the unlocked31 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from render_g1_plus_neck_compare import _neck_traces, _slice_body_trajectory
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
GHOST_ALPHA = 0.55
STATIC_COLOR = (40, 190, 255)
MOVING_COLOR = (255, 100, 180)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robot_mask(segmentation, geom_types, plane_type, geom_objtype):
    ids, types = segmentation[..., 0], segmentation[..., 1]
    mask = (types == geom_objtype) & (ids >= 0)
    mask[mask] = geom_types[ids[mask]] != plane_type
    return mask


def blend(base, layer, mask, color):
    tinted = 0.45 * layer.astype(np.float32) + 0.55 * np.asarray(color, np.float32)
    out = base.astype(np.float32)
    out[mask] = (1.0 - GHOST_ALPHA) * out[mask] + GHOST_ALPHA * tinted[mask]
    return out.clip(0, 255).astype(np.uint8)


def build_qposes(model, trajectory_path: Path, frames: int, neck_values):
    body = _slice_body_trajectory(load_trajectory(trajectory_path), frames)
    return build_mujoco_qpos(
        body,
        profile=PROFILE,
        model=model,
        extra_joint_values=neck_values,
    )


def render_variant(
    *, imageio, cv2, count_frames_and_secs, mujoco, model,
    source_qpos, static_qpos, moving_qpos, output, camera_distance,
    camera_azimuth, camera_elevation,
):
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = camera_azimuth
    camera.elevation = camera_elevation
    camera.distance = camera_distance

    plane_type = int(mujoco.mjtGeom.mjGEOM_PLANE)
    geom_objtype = int(mujoco.mjtObj.mjOBJ_GEOM)
    geom_types = np.asarray(model.geom_type)

    def draw(qpos, segment=False):
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if segment:
            renderer.enable_segmentation_rendering()
        try:
            renderer.update_scene(data, camera=camera)
            return renderer.render().copy()
        finally:
            if segment:
                renderer.disable_segmentation_rendering()

    writer = _writer(imageio, output, FPS)
    try:
        for frame_index in range(source_qpos.shape[0]):
            camera.lookat[0:2] = source_qpos[frame_index, 0:2]
            camera.lookat[2] = source_qpos[0, 2]
            base = draw(source_qpos[frame_index])
            static_layer = draw(static_qpos[frame_index])
            static_segmentation = draw(static_qpos[frame_index], segment=True)
            static_mask = robot_mask(
                static_segmentation, geom_types, plane_type, geom_objtype
            )
            frame = blend(base, static_layer, static_mask, STATIC_COLOR)

            moving_layer = draw(moving_qpos[frame_index])
            moving_segmentation = draw(moving_qpos[frame_index], segment=True)
            moving_mask = robot_mask(
                moving_segmentation, geom_types, plane_type, geom_objtype
            )
            frame = blend(frame, moving_layer, moving_mask, MOVING_COLOR)

            cv2.rectangle(frame, (0, 0), (1280, 108), (25, 25, 25), -1)
            cv2.putText(
                frame,
                f"G1+ unlocked31 | Ghost source/static/moving | frame {frame_index}/{source_qpos.shape[0] - 1}",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1,
            )
            cv2.rectangle(frame, (10, 43), (29, 61), (235, 235, 235), -1)
            cv2.putText(
                frame,
                "source (opaque)",
                (38, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (235, 235, 235), 1,
            )
            cv2.rectangle(frame, (250, 43), (269, 61), STATIC_COLOR, -1)
            cv2.putText(
                frame,
                "C replay + neck static",
                (278, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (235, 235, 235), 1,
            )
            cv2.rectangle(frame, (520, 43), (539, 61), MOVING_COLOR, -1)
            cv2.putText(
                frame,
                "C replay + neck moving",
                (548, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (235, 235, 235), 1,
            )
            cv2.putText(
                frame,
                "same camera | kinematic replay | 50 Hz",
                (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1,
            )
            writer.append_data(frame)
    finally:
        writer.close()
        renderer.close()

    frames, seconds = count_frames_and_secs(str(output))
    if frames != source_qpos.shape[0] or output.stat().st_size <= 0:
        raise RuntimeError(
            f"ghost video validation failed: {output}, frames={frames}, "
            f"expected={source_qpos.shape[0]}"
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
    for folder in ("data", "manifests", "videos"):
        (output / folder).mkdir()

    model_path = args.model.expanduser().resolve(strict=True)
    source_path = args.source_trajectory.expanduser().resolve(strict=True)
    route_path = args.route_trajectory.expanduser().resolve(strict=True)
    frames = args.horizon_steps + 1

    os.environ["MUJOCO_GL"] = args.gl
    import cv2
    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    import mujoco

    runtime_xml = output / "manifests" / "g1_plus_unlocked31_ghost_runtime.xml"
    xml_details = prepare_runtime_xml(
        model_path,
        runtime_xml,
        width=1280,
        height=720,
        mesh_dir=args.mesh_dir,
        ground_style="black",
    )
    model = mujoco.MjModel.from_xml_path(str(runtime_xml))
    profile = validate_model_profile(model, PROFILE)

    neck_moving = _neck_traces(model, frames, args.neck_amplitude_fraction)
    addresses = {
        name: int(model.jnt_qposadr[int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))])
        for name in NECK_NAMES
    }
    neck_static = {
        name: np.full(frames, float(model.qpos0[addresses[name]]))
        for name in NECK_NAMES
    }

    source_static = build_qposes(model, source_path, frames, neck_static)
    route_static = build_qposes(model, route_path, frames, neck_static)
    route_moving = build_qposes(model, route_path, frames, neck_moving)

    np.savez_compressed(
        output / "data" / "ghost_qposes.npz",
        source_static=source_static,
        route_static=route_static,
        route_moving=route_moving,
    )

    videos = [
        render_variant(
            imageio=imageio, cv2=cv2, count_frames_and_secs=count_frames_and_secs,
            mujoco=mujoco, model=model, source_qpos=source_static,
            static_qpos=route_static, moving_qpos=route_moving,
            output=output / "videos" / "ghost_source_static_moving.mp4",
            camera_distance=args.camera_distance, camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
        )
    ]

    manifest = {
        "version": "g1_plus_unlocked31_ghost_compare_v1",
        "profile": PROFILE,
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "runtime_xml": str(runtime_xml),
        "runtime_xml_sha256": sha256(runtime_xml),
        "source_trajectory": str(source_path),
        "source_trajectory_sha256": sha256(source_path),
        "route_trajectory": str(route_path),
        "route_trajectory_sha256": sha256(route_path),
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "model_njnt": int(model.njnt),
        "horizon_steps": int(args.horizon_steps),
        "frames": int(frames),
        "fps": FPS,
        "neck_amplitude_fraction": float(args.neck_amplitude_fraction),
        "ghost_alpha": GHOST_ALPHA,
        "source_variant": "source with neck static",
        "static_variant": "C replay with neck static",
        "moving_variant": "C replay with neck moving",
        "route_colors": {
            "static": list(STATIC_COLOR),
            "moving": list(MOVING_COLOR),
        },
        "same_camera": True,
        "kinematic_only": True,
        "quality_pass": None,
        "runtime_ground": xml_details["ground"],
        "videos": videos,
    }
    (output / "manifests" / "ghost_compare.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-trajectory", type=Path, required=True)
    parser.add_argument("--route-trajectory", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path)
    parser.add_argument("--horizon-steps", type=int, default=192)
    parser.add_argument("--neck-amplitude-fraction", type=float, default=0.35)
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
