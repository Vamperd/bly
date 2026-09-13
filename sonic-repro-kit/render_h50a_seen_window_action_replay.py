#!/usr/bin/env python3
"""Render recorded, original-Action, and H50-A-Action trajectories side by side."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from render_mujoco_trajectory import (
    G1_MUJOCO_JOINT_NAMES,
    build_mujoco_qpos,
    load_trajectory,
    prepare_runtime_xml,
    write_json_atomic,
)


def _validate_joint_order(model: Any) -> None:
    names = tuple(
        model.joint(index).name
        for index in range(model.njnt)
        if model.joint(index).name != "floating_base_joint"
    )
    if names != G1_MUJOCO_JOINT_NAMES:
        raise ValueError("MuJoCo model joint order does not match SONIC canonical order")


def _label(cv2: Any, frame: np.ndarray, text: str, ended: bool) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 38), (20, 20, 20), -1)
    suffix = "  [trajectory ended]" if ended else ""
    cv2.putText(
        result,
        text + suffix,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return result


def _timeline(
    cv2: Any,
    frame: np.ndarray,
    *,
    frame_index: int,
    action_steps: int,
    window_start: int,
    window_stop: int,
) -> np.ndarray:
    result = frame.copy()
    height, width = result.shape[:2]
    y0 = height - 18
    cv2.rectangle(result, (0, y0), (width - 1, height - 1), (35, 35, 35), -1)
    x0 = int(window_start * width / max(action_steps, 1))
    x1 = max(x0 + 1, int(window_stop * width / max(action_steps, 1)))
    cv2.rectangle(result, (x0, y0), (min(width - 1, x1), height - 1), (220, 30, 30), -1)
    action_step = min(max(frame_index - 1, 0), max(action_steps - 1, 0))
    cursor = min(width - 1, int(action_step * width / max(action_steps, 1)))
    cv2.line(result, (cursor, y0 - 5), (cursor, height - 1), (255, 255, 255), 2)
    cv2.putText(
        result,
        "red = 64 H50-A reconstructed Action steps",
        (10, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return result


def _writer(imageio: Any, path: Path, fps: float):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite video: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=5,
        pixelformat="yuv420p",
        macro_block_size=2,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--gl", choices=("osmesa", "egl", "glfw"), default="egl")
    parser.add_argument("--camera-distance", type=float, default=2.0)
    parser.add_argument("--camera-azimuth", type=float, default=120.0)
    parser.add_argument("--camera-elevation", type=float, default=-25.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 6 or args.height % 2:
        raise ValueError("width must be divisible by 6 and height must be a positive even integer")
    if args.camera_distance <= 0:
        raise ValueError("camera distance must be positive")
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    model_path = args.model.expanduser().resolve(strict=True)
    request = json.loads(
        (run_dir / "manifests/action_mask_request.json").read_text(encoding="utf-8")
    )
    replay_request = json.loads(
        (run_dir / "manifests/action_replay_request.json").read_text(encoding="utf-8")
    )
    if replay_request.get("scenario_names") != ["original", "h50a_posterior_full_both"]:
        raise ValueError("H50-A replay must contain exactly original and posterior scenarios")

    trajectory_paths = [
        run_dir / "data/source/000000.trajectory.pkl",
        run_dir / "data/replay/000000.trajectory.pkl",
        run_dir / "data/replay/000001.trajectory.pkl",
    ]
    trajectories = [load_trajectory(path) for path in trajectory_paths]
    qpos = [build_mujoco_qpos(value) for value in trajectories]
    lengths = [int(value.shape[0]) for value in qpos]
    if any(length <= 0 for length in lengths):
        raise ValueError(f"all trajectories must contain frames; found {lengths}")
    frame_count = lengths[0]
    fps_values = [float(value.get("fps", 0.0)) for value in trajectories]
    if not np.isclose(fps_values[0], 50.0) or any(
        not np.isclose(value, fps_values[0]) for value in fps_values[1:]
    ):
        raise ValueError(f"all trajectories must use synchronized 50 Hz timing: {fps_values}")

    os.environ["MUJOCO_GL"] = args.gl
    import cv2
    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    import mujoco

    runtime_xml = run_dir / "manifests/g1_h50a_action_replay_render.xml"
    panel_width = args.width // 3
    xml_details = prepare_runtime_xml(model_path, runtime_xml, panel_width, args.height)
    model = mujoco.MjModel.from_xml_path(str(runtime_xml))
    if model.nq != 36:
        raise ValueError(f"expected MuJoCo model nq=36, found {model.nq}")
    _validate_joint_order(model)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    camera.distance = args.camera_distance
    renderer = mujoco.Renderer(model, height=args.height, width=panel_width)
    output_path = run_dir / "videos/h50a_seen_window_action_replay.mp4"
    writer = _writer(imageio, output_path, fps_values[0])
    labels = (
        "Recorded training trajectory (seen)",
        "Original recorded Action replay",
        "H50-A posterior Action replay",
    )

    def render(value: np.ndarray, lookat: np.ndarray) -> np.ndarray:
        data.qpos[:] = value
        data.qvel[:] = 0.0
        camera.lookat[:] = lookat
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()

    try:
        for frame_index in range(frame_count):
            original_index = min(frame_index, lengths[1] - 1)
            lookat = qpos[1][original_index, :3]
            panels = []
            for trajectory_index, (value, label) in enumerate(zip(qpos, labels, strict=True)):
                value_index = min(frame_index, lengths[trajectory_index] - 1)
                panel = render(value[value_index], lookat)
                panels.append(
                    _label(
                        cv2,
                        panel,
                        label,
                        ended=frame_index >= lengths[trajectory_index],
                    )
                )
            combined = np.concatenate(panels, axis=1)
            combined = _timeline(
                cv2,
                combined,
                frame_index=frame_index,
                action_steps=int(request["replay_steps"]),
                window_start=int(request["window_start"]),
                window_stop=int(request["window_start"]) + int(request["window_transitions"]),
            )
            writer.append_data(combined)
    finally:
        writer.close()
        renderer.close()

    encoded_frames, duration = count_frames_and_secs(str(output_path))
    if encoded_frames != frame_count or output_path.stat().st_size <= 0:
        raise RuntimeError(
            f"H50-A video encoded {encoded_frames} frames; expected {frame_count}"
        )
    manifest = {
        "schema_version": "sonic_h50a_seen_window_action_replay_render_v1",
        "output": str(output_path),
        "frames": int(encoded_frames),
        "duration_seconds": float(duration),
        "fps": fps_values[0],
        "width": args.width,
        "height": args.height,
        "source_and_replay_frame_counts": lengths,
        "short_replay_policy": "hold final pose and label trajectory ended",
        "camera": {
            "lookat": "original Action replay root",
            "distance": args.camera_distance,
            "azimuth": args.camera_azimuth,
            "elevation": args.camera_elevation,
        },
        "xml_patch": xml_details,
    }
    write_json_atomic(
        run_dir / "manifests/h50a_seen_window_action_replay_render.json", manifest
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
