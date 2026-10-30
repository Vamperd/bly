#!/usr/bin/env python3
"""N-layer ghost video from Isaac policy-rollout trajectory pickles.

The first --layer is drawn opaque and the camera follows it; the remaining
layers are drawn translucent with distinct colors.  Kinematic replay only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from render_mujoco_trajectory import (
    build_mujoco_qpos,
    load_trajectory,
    prepare_runtime_xml,
    validate_model_profile,
    write_json_atomic,
)
from render_three_way_ghost import GHOST_SIZE, _blend, _robot_mask, _sha256


BASE_COLOR = (235, 235, 235)
LAYER_COLORS = ((70, 165, 255), (90, 220, 110), (255, 140, 60), (220, 90, 220))


def _parse_layers(values: list[str]) -> list[tuple[str, Path]]:
    layers = []
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--layer must be NAME=TRAJECTORY.pkl, got {value!r}")
        layers.append((name, Path(path).expanduser().resolve()))
    if len(layers) < 2 or len(layers) > 1 + len(LAYER_COLORS):
        raise ValueError(f"need 2..{1 + len(LAYER_COLORS)} layers, got {len(layers)}")
    return layers


def render(args: argparse.Namespace) -> dict:
    layers = _parse_layers(args.layer)
    output = args.output_run.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output run must be new and empty: {output}")
    for folder in ("manifests", "videos", "markers"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    os.environ["MUJOCO_GL"] = args.gl
    import cv2
    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    import mujoco

    xml = output / "manifests" / "policy_ghost.xml"
    xml_details = prepare_runtime_xml(
        args.model.expanduser().resolve(),
        xml,
        *GHOST_SIZE,
        mesh_dir=args.mesh_dir,
        ground_style=args.ground_style,
        ground_size_m=args.ground_size,
    )
    model = mujoco.MjModel.from_xml_path(str(xml))
    profile_details = validate_model_profile(model, args.robot_profile)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=GHOST_SIZE[1], width=GHOST_SIZE[0])
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 135.0
    camera.elevation = -25.0
    plane_type = int(mujoco.mjtGeom.mjGEOM_PLANE)
    geom_objtype = int(mujoco.mjtObj.mjOBJ_GEOM)
    geom_types = np.asarray(model.geom_type)
    geom_body_ids = np.asarray(model.geom_bodyid)
    robot_geom_ids = np.flatnonzero(geom_body_ids != 0)

    def draw(qpos, lookat, segment=False):
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        camera.lookat[:] = lookat
        if segment:
            renderer.enable_segmentation_rendering()
        try:
            renderer.update_scene(data, camera=camera)
            return renderer.render().copy()
        finally:
            if segment:
                renderer.disable_segmentation_rendering()

    def lowest_robot_z(qpos):
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        return float(data.geom_xpos[robot_geom_ids, 2].min())

    trajectories = [load_trajectory(path) for _name, path in layers]
    fps_values = {name: float(item.get("fps", 0.0)) for (name, _), item in zip(layers, trajectories)}
    fps = fps_values[layers[0][0]]
    if any(abs(value - fps) > 1e-6 for value in fps_values.values()):
        raise ValueError(f"layers have different fps: {fps_values}")
    qposes = [
        build_mujoco_qpos(item, profile=args.robot_profile, model=model) for item in trajectories
    ]
    frame_counts = {name: int(qpos.shape[0]) for (name, _), qpos in zip(layers, qposes)}
    if any(value <= 0 for value in frame_counts.values()):
        raise ValueError(f"empty trajectory: {frame_counts}")
    frames = max(frame_counts.values())
    alignment = {}
    for index in range(1, len(qposes)):
        offset = qposes[0][0, :3] - qposes[index][0, :3]
        qposes[index][:, :3] += offset
        alignment[layers[index][0]] = offset.tolist()
    lookat_z = float(qposes[0][0, 2])
    names = [name for name, _ in layers]
    colors = [BASE_COLOR, *LAYER_COLORS[: len(layers) - 1]]
    title = args.title or "ghost"
    video_path = output / "videos" / f"ghost_{title}.mp4"
    writer = imageio.get_writer(
        str(video_path),
        fps=fps,
        codec="libx264",
        quality=5,
        pixelformat="yuv420p",
        macro_block_size=8,
        ffmpeg_params=["-movflags", "+faststart"],
    )
    legend_step = max(1, (GHOST_SIZE[0] - 20) // len(layers))
    try:
        for frame_index in range(frames):
            base_qpos = qposes[0][min(frame_index, frame_counts[names[0]] - 1)]
            base_ground_z = lowest_robot_z(base_qpos)
            frame = draw(base_qpos, np.asarray([base_qpos[0], base_qpos[1], lookat_z]))
            if np.ptp(frame) < 2:
                raise RuntimeError(f"base frame {frame_index} is blank; check MuJoCo GL/scene")
            for index in range(1, len(layers)):
                qpos = np.asarray(
                    qposes[index][min(frame_index, frame_counts[names[index]] - 1)],
                    dtype=np.float64,
                ).copy()
                qpos[2] += base_ground_z - lowest_robot_z(qpos)
                lookat = np.asarray([qpos[0], qpos[1], qpos[2]])
                layer = draw(qpos, lookat)
                mask = _robot_mask(
                    draw(qpos, lookat, segment=True),
                    geom_types,
                    geom_body_ids,
                    plane_type,
                    geom_objtype,
                )
                frame = _blend(frame, layer, mask, colors[index])
            cv2.rectangle(frame, (0, 0), (GHOST_SIZE[0], 106), (25, 25, 25), -1)
            cv2.putText(
                frame,
                f"Ghost | {title} | frame {frame_index}/{frames - 1} | {fps:g} Hz",
                (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (245, 245, 245),
                1,
            )
            for index, (name, color) in enumerate(zip(names, colors)):
                x = 10 + index * legend_step
                label = name if index else f"{name} (opaque)"
                if frame_index >= frame_counts[name]:
                    label += f" ended {frame_counts[name] - 1}"
                cv2.rectangle(frame, (x, 43), (x + 19, 61), color, -1)
                cv2.putText(
                    frame, label, (x + 28, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (235, 235, 235), 1
                )
            cv2.putText(
                frame,
                "Isaac rollouts replayed kinematically | camera follows opaque layer",
                (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (230, 230, 230),
                1,
            )
            writer.append_data(frame)
    finally:
        writer.close()
        renderer.close()
    rendered, seconds = count_frames_and_secs(str(video_path))
    if rendered != frames or video_path.stat().st_size <= 0:
        raise RuntimeError(f"ghost video validation failed: {video_path} frames={rendered}/{frames}")
    result = {
        "format_version": "sonic_policy_ghost_v1",
        "video": str(video_path),
        "video_sha256": _sha256(video_path),
        "frames": int(rendered),
        "seconds": float(seconds),
        "fps": fps,
        "layers": [
            {
                "name": name,
                "trajectory": str(path),
                "trajectory_sha256": _sha256(path),
                "frames": frame_counts[name],
                "color_rgb": list(color),
                "opaque": index == 0,
            }
            for index, ((name, path), color) in enumerate(zip(layers, colors))
        ],
        "initial_xyz_alignment": alignment,
        "vertical_alignment": "translucent layer lowest robot geom z follows opaque layer",
        "model": str(args.model.expanduser().resolve()),
        "model_details": xml_details,
        "profile": profile_details,
    }
    write_json_atomic(output / "manifests" / "policy_ghost.json", result)
    (output / "markers" / "policy_ghost.ok").write_text("PASS\n", encoding="utf-8")
    print(f"[ghost] frames={frames} layers={frame_counts} video={video_path}", flush=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", action="append", required=True, help="NAME=trajectory.pkl; first is opaque")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--robot-profile", default="g1")
    parser.add_argument("--mesh-dir", type=Path)
    parser.add_argument("--ground-style", default="black", choices=("checker", "black"))
    parser.add_argument("--ground-size", type=float, default=10.0)
    parser.add_argument("--gl", default="egl", choices=("egl", "osmesa", "glfw"))
    render(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
