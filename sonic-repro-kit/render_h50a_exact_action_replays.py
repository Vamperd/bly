#!/usr/bin/env python3
"""Render three independent videos for the H50-A exact-initialization replay."""

from __future__ import annotations

import argparse
import hashlib
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


TRAJECTORIES = (
    ("recorded_hdf", "data/recorded_hdf.trajectory.pkl", "videos/recorded_hdf.mp4"),
    (
        "original_action_exact_init",
        "data/replay/000000.trajectory.pkl",
        "videos/original_action_exact_init.mp4",
    ),
    (
        "h50a_action_exact_init",
        "data/replay/000001.trajectory.pkl",
        "videos/h50a_action_exact_init.mp4",
    ),
)
LABELS = {
    "recorded_hdf": "Recorded training HDF (no simulation)",
    "original_action_exact_init": "Original Action replay (exact init)",
    "h50a_action_exact_init": "H50-A hybrid replay (red interval predicted)",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_joint_order(model: Any) -> None:
    names = tuple(
        model.joint(index).name
        for index in range(model.njnt)
        if model.joint(index).name != "floating_base_joint"
    )
    if names != G1_MUJOCO_JOINT_NAMES:
        raise ValueError("MuJoCo model joint order does not match SONIC canonical order")


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


def progress_bar_geometry(
    width: int,
    total_action_steps: int,
    masked_window_start: int,
    masked_window_stop: int,
) -> dict[str, int]:
    if width <= 40 or total_action_steps <= 0:
        raise ValueError("video width and total Action steps must be positive")
    if not (
        0 <= masked_window_start < masked_window_stop <= total_action_steps
    ):
        raise ValueError("Mask progress interval is outside the full episode")
    left, right = 14, width - 14
    span = right - left

    def position(action_index: int) -> int:
        return left + round(span * action_index / total_action_steps)

    return {
        "left": left,
        "right": right,
        "masked_left": position(masked_window_start),
        "masked_right": position(masked_window_stop),
    }


def _annotate(
    cv2: Any,
    frame: np.ndarray,
    label: str,
    frame_index: int,
    *,
    total_action_steps: int,
    masked_window_start: int,
    masked_window_stop: int,
) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 38), (20, 20, 20), -1)
    cv2.putText(
        output,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    digits = len(str(total_action_steps))
    cv2.putText(
        output,
        f"frame {frame_index:0{digits}d}/{total_action_steps}  |  50 Hz",
        (14, output.shape[0] - 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    geometry = progress_bar_geometry(
        output.shape[1],
        total_action_steps,
        masked_window_start,
        masked_window_stop,
    )
    bar_top, bar_bottom = output.shape[0] - 19, output.shape[0] - 10
    cv2.rectangle(
        output,
        (geometry["left"], bar_top),
        (geometry["right"], bar_bottom),
        (45, 45, 45),
        -1,
    )
    current = geometry["left"] + round(
        (geometry["right"] - geometry["left"])
        * min(max(frame_index, 0), total_action_steps)
        / total_action_steps
    )
    cv2.rectangle(
        output,
        (geometry["left"], bar_top),
        (current, bar_bottom),
        (210, 145, 45),
        -1,
    )
    completed_mask_right = min(current, geometry["masked_right"])
    if completed_mask_right > geometry["masked_left"]:
        cv2.rectangle(
            output,
            (geometry["masked_left"], bar_top),
            (completed_mask_right, bar_bottom),
            (35, 35, 230),
            -1,
        )
    cv2.rectangle(
        output,
        (geometry["masked_left"], bar_top - 1),
        (geometry["masked_right"], bar_bottom + 1),
        (35, 35, 230),
        1,
    )
    cv2.line(output, (current, bar_top - 2), (current, bar_bottom + 2), (255, 255, 255), 1)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--gl", choices=("osmesa", "egl", "glfw"), default="egl")
    parser.add_argument("--camera-distance", type=float, default=2.0)
    parser.add_argument("--camera-azimuth", type=float, default=120.0)
    parser.add_argument("--camera-elevation", type=float, default=-25.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        raise ValueError("width and height must be positive even integers")
    if args.camera_distance <= 0:
        raise ValueError("camera distance must be positive")
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    model_path = args.model.expanduser().resolve(strict=True)
    request = json.loads(
        (run_dir / "manifests/action_replay_request.json").read_text(encoding="utf-8")
    )
    if request.get("scenario_names") != ["original", "h50a_posterior_full_both"]:
        raise ValueError("exact H50-A replay requires original and posterior scenarios")
    action_steps = int(request.get("steps", -1))
    replay_scope = str(request.get("replay_scope", "full_episode"))
    if replay_scope not in {"full_episode", "window"}:
        raise ValueError("exact H50-A renderer received an invalid replay scope")
    masked_window_start = int(request.get("masked_window_start", -1))
    masked_window_stop = int(request.get("masked_window_stop", -1))
    if action_steps <= 0:
        raise ValueError("exact H50-A renderer requires a non-empty replay")
    if masked_window_stop - masked_window_start != 64:
        raise ValueError("exact H50-A renderer requires one 64-Action Mask window")
    progress_bar_geometry(
        args.width, action_steps, masked_window_start, masked_window_stop
    )
    total_frames = action_steps + 1
    labels = dict(LABELS)
    if replay_scope == "window":
        labels["h50a_action_exact_init"] = "H50-A short T64 Action replay"

    entries = []
    for name, trajectory_relative, video_relative in TRAJECTORIES:
        trajectory_path = run_dir / trajectory_relative
        trajectory = load_trajectory(trajectory_path)
        qpos = build_mujoco_qpos(trajectory)
        fps = float(trajectory.get("fps", 0.0))
        if qpos.shape[0] != total_frames or not np.isclose(fps, 50.0):
            raise ValueError(
                f"{name} must contain {total_frames} frames at 50 Hz; "
                f"found {qpos.shape[0]} at {fps}"
            )
        entries.append(
            {
                "name": name,
                "trajectory": trajectory_path,
                "video": run_dir / video_relative,
                "qpos": qpos,
                "fps": fps,
            }
        )
    camera_qpos = entries[0]["qpos"]

    os.environ["MUJOCO_GL"] = args.gl
    import cv2
    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    import mujoco

    runtime_xml = run_dir / "manifests/g1_h50a_exact_action_replay_render.xml"
    xml_details = prepare_runtime_xml(model_path, runtime_xml, args.width, args.height)
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
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    encoded: list[dict[str, Any]] = []
    try:
        # Render each trajectory into its own file.  Physics replay has already
        # completed; rendering cannot couple the two Isaac executions.
        for entry in entries:
            writer = _writer(imageio, entry["video"], entry["fps"])
            try:
                for frame_index in range(total_frames):
                    data.qpos[:] = entry["qpos"][frame_index]
                    data.qvel[:] = 0.0
                    camera.lookat[:] = camera_qpos[frame_index, :3]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    frame = _annotate(
                        cv2,
                        renderer.render().copy(),
                        labels[entry["name"]],
                        frame_index,
                        total_action_steps=action_steps,
                        masked_window_start=masked_window_start,
                        masked_window_stop=masked_window_stop,
                    )
                    writer.append_data(frame)
            finally:
                writer.close()
            frame_count, duration = count_frames_and_secs(str(entry["video"]))
            if int(frame_count) != total_frames or entry["video"].stat().st_size <= 0:
                raise RuntimeError(
                    f"{entry['name']} encoded {frame_count} frames; expected {total_frames}"
                )
            encoded.append(
                {
                    "name": entry["name"],
                    "trajectory": str(entry["trajectory"]),
                    "trajectory_sha256": _sha256(entry["trajectory"]),
                    "video": str(entry["video"]),
                    "video_sha256": _sha256(entry["video"]),
                    "frames": int(frame_count),
                    "duration_seconds": float(duration),
                }
            )
    finally:
        renderer.close()

    # The optional comparison is composed only after all three independent
    # videos exist; it never drives or synchronizes the simulations.
    triptych_path = run_dir / "videos/exact_init_comparison.mp4"
    readers = [imageio.get_reader(str(entry["video"])) for entry in entries]
    triptych_writer = _writer(imageio, triptych_path, 50.0)
    try:
        for frame_index in range(total_frames):
            frames = [reader.get_data(frame_index) for reader in readers]
            triptych_writer.append_data(np.concatenate(frames, axis=1))
    finally:
        triptych_writer.close()
        for reader in readers:
            reader.close()
    triptych_frames, triptych_duration = count_frames_and_secs(str(triptych_path))
    if int(triptych_frames) != total_frames:
        raise RuntimeError(
            f"triptych encoded {triptych_frames} frames; expected {total_frames}"
        )

    manifest = {
        "schema_version": "sonic_h50a_exact_action_replay_render_v2",
        "independent_videos": encoded,
        "optional_triptych": {
            "video": str(triptych_path),
            "video_sha256": _sha256(triptych_path),
            "frames": int(triptych_frames),
            "duration_seconds": float(triptych_duration),
        },
        "fps": 50.0,
        "replay_scope": replay_scope,
        "replay_action_steps": action_steps,
        "replay_frames": total_frames,
        "source_episode_action_steps": int(
            request.get("episode_steps", action_steps)
        ),
        "masked_window_progress_segment": {
            "action_start_inclusive": masked_window_start,
            "action_stop_exclusive": masked_window_stop,
            "transitions": masked_window_stop - masked_window_start,
            "color": "red",
            "meaning": "H50-A posterior Action replacement interval in the hybrid replay",
        },
        "width": args.width,
        "height": args.height,
        "camera": {
            "lookat": "recorded HDF root trajectory, shared across all videos",
            "distance": args.camera_distance,
            "azimuth": args.camera_azimuth,
            "elevation": args.camera_elevation,
        },
        "execution_semantics": (
            "three trajectories rendered after two serial isolated Isaac runs; the red progress "
            f"segment marks the H50-A T64 interval; replay_scope={replay_scope}"
        ),
        "xml_patch": xml_details,
    }
    write_json_atomic(run_dir / "manifests/h50a_exact_action_replay_render.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
