#!/usr/bin/env python3
"""Render synchronized SONIC/CVAE Action-mask physics trajectories with MuJoCo."""

from __future__ import annotations

import argparse
import json
import math
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


REPRESENTATIVE_OUTPUTS = {
    "element_25": "element_25_comparison.mp4",
    "step_contiguous_8": "step_8_comparison.mp4",
    "feature_random_25": "feature_random_25_comparison.mp4",
    "semantic_left_leg": "semantic_left_leg_comparison.mp4",
    "inverse_full_128": "inverse_full_128_comparison.mp4",
}


def comparison_output_name(scenario: str) -> str:
    return REPRESENTATIVE_OUTPUTS.get(scenario, f"{scenario}_comparison.mp4")


def grid_shape(panel_count: int) -> tuple[int, int]:
    if panel_count <= 0:
        raise ValueError("panel_count must be positive")
    columns = min(5, panel_count)
    rows = math.ceil(panel_count / columns)
    return rows, columns


def _video_writer(imageio: Any, path: Path, fps: float):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing video: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=5,
        pixelformat="yuv420p",
        macro_block_size=2,
    )


def _label(cv2: Any, frame: np.ndarray, text: str) -> np.ndarray:
    result = frame.copy()
    compact = result.shape[1] < 300
    bar_height = 26 if compact else 34
    font_scale = 0.40 if compact else 0.62
    cv2.rectangle(result, (0, 0), (result.shape[1], bar_height), (20, 20, 20), -1)
    cv2.putText(
        result,
        text,
        (6 if compact else 12, 18 if compact else 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return result


def _mask_timeline(
    cv2: Any,
    frame: np.ndarray,
    action_mask: np.ndarray,
    state_frame: int,
) -> np.ndarray:
    result = frame.copy()
    height, width = result.shape[:2]
    bar_y0 = height - 14
    cv2.rectangle(result, (0, bar_y0), (width - 1, height - 1), (35, 35, 35), -1)
    active_steps = np.flatnonzero(action_mask.any(axis=-1))
    total_steps = action_mask.shape[0]
    for step in active_steps:
        x0 = int(step * width / max(total_steps, 1))
        x1 = max(x0 + 1, int((step + 1) * width / max(total_steps, 1)))
        cv2.rectangle(result, (x0, bar_y0), (x1, height - 1), (230, 20, 20), -1)
    action_step = min(max(state_frame - 1, 0), max(total_steps - 1, 0))
    current_x = min(width - 1, int(action_step * width / max(total_steps, 1)))
    cv2.line(result, (current_x, bar_y0 - 4), (current_x, height - 1), (255, 255, 255), 2)
    return result


def _compose_pair(
    cv2: Any,
    original: np.ndarray,
    completed: np.ndarray,
    scenario: str,
    mask: np.ndarray,
    frame_index: int,
    width: int,
    height: int,
) -> np.ndarray:
    half = width // 2
    left = cv2.resize(original, (half, height), interpolation=cv2.INTER_AREA)
    right = cv2.resize(completed, (width - half, height), interpolation=cv2.INTER_AREA)
    result = np.concatenate(
        (_label(cv2, left, "Original Action replay"), _label(cv2, right, scenario)), axis=1
    )
    return _mask_timeline(cv2, result, mask, frame_index)


def _compose_grid(
    cv2: Any,
    frames: list[np.ndarray],
    labels: list[str],
    masks: list[np.ndarray],
    frame_index: int,
    width: int,
    height: int,
) -> np.ndarray:
    rows, columns = grid_shape(len(frames))
    panel_width = width // columns
    panel_height = height // rows
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    for index, (frame, label, mask) in enumerate(zip(frames, labels, masks, strict=True)):
        row, column = divmod(index, columns)
        x0, y0 = column * panel_width, row * panel_height
        x1 = width if column == columns - 1 else (column + 1) * panel_width
        y1 = height if row == rows - 1 else (row + 1) * panel_height
        panel = cv2.resize(frame, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        panel = _label(cv2, panel, label)
        panel = _mask_timeline(cv2, panel, mask, frame_index)
        canvas[y0:y1, x0:x1] = panel
    return canvas


def _validate_joint_order(model: Any) -> None:
    model_joint_names = tuple(
        model.joint(index).name
        for index in range(model.njnt)
        if model.joint(index).name != "floating_base_joint"
    )
    if model_joint_names != G1_MUJOCO_JOINT_NAMES:
        raise ValueError("MuJoCo model joint order does not match SONIC's canonical G1 order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--render-mode", choices=("representatives", "all"), required=True)
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
        raise ValueError("Video width and height must be positive even integers")
    if args.camera_distance <= 0:
        raise ValueError("Camera distance must be positive")
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    model_path = args.model.expanduser().resolve(strict=True)
    videos_dir = run_dir / "videos"
    manifests_dir = run_dir / "manifests"
    replay_request = json.loads(
        (manifests_dir / "action_replay_request.json").read_text(encoding="utf-8")
    )
    scenario_names = [str(value) for value in replay_request["scenario_names"]]
    if not scenario_names or scenario_names[0] != "original":
        raise ValueError("Action replay request must begin with the original environment")
    with np.load(run_dir / "data/completed_actions.npz", allow_pickle=False) as values:
        masks = np.asarray(values["mask_action"], dtype=bool)
        completion_names = [str(value) for value in values["scenario_names"].tolist()]
    if completion_names != scenario_names[1:] or masks.shape[0] != len(completion_names):
        raise ValueError("Completion masks and replay scenario identities differ")

    source_trajectory = load_trajectory(run_dir / "data/source/000000.trajectory.pkl")
    replay_trajectories = [
        load_trajectory(run_dir / f"data/replay/{index:06d}.trajectory.pkl")
        for index in range(len(scenario_names))
    ]
    source_qpos = build_mujoco_qpos(source_trajectory)
    replay_qpos = [build_mujoco_qpos(value) for value in replay_trajectories]
    frame_count = source_qpos.shape[0]
    lengths = [value.shape[0] for value in replay_qpos]
    if any(length != frame_count for length in lengths):
        raise ValueError(f"Source/replay frame counts are not synchronized: {[frame_count, *lengths]}")
    if masks.shape[1] != frame_count - 1:
        raise ValueError("Action masks must contain exactly one fewer step than state frames")
    fps = float(source_trajectory.get("fps", 0.0))
    replay_fps = [float(value.get("fps", 0.0)) for value in replay_trajectories]
    if not np.isclose(fps, 50.0) or any(not np.isclose(value, fps) for value in replay_fps):
        raise ValueError(f"All trajectories must be synchronized at 50 FPS; found {[fps, *replay_fps]}")

    selected = (
        completion_names
        if args.render_mode == "all"
        else [name for name in REPRESENTATIVE_OUTPUTS if name in completion_names]
    )
    if args.render_mode == "representatives" and set(selected) != set(REPRESENTATIVE_OUTPUTS):
        missing = sorted(set(REPRESENTATIVE_OUTPUTS) - set(selected))
        raise ValueError(f"Representative Action-mask scenarios are missing: {missing}")

    os.environ["MUJOCO_GL"] = args.gl
    import cv2
    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    import mujoco

    runtime_xml = manifests_dir / "g1_action_mask_render.xml"
    xml_details = prepare_runtime_xml(model_path, runtime_xml, args.width, args.height)
    model = mujoco.MjModel.from_xml_path(str(runtime_xml))
    if model.nq != 36:
        raise ValueError(f"Expected MuJoCo model nq=36, found {model.nq}")
    _validate_joint_order(model)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    camera.distance = args.camera_distance
    full_renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    panel_renderer = mujoco.Renderer(model, height=args.height, width=args.width // 2)

    source_path = videos_dir / "sonic_source.mp4"
    original_path = videos_dir / "original_action_replay.mp4"
    grid_path = videos_dir / "all_action_masks_grid.mp4"
    writers = {
        "source": _video_writer(imageio, source_path, fps),
        "original": _video_writer(imageio, original_path, fps),
        "grid": _video_writer(imageio, grid_path, fps),
    }
    comparison_paths = {name: videos_dir / comparison_output_name(name) for name in selected}
    for name, path in comparison_paths.items():
        writers[name] = _video_writer(imageio, path, fps)

    zero_mask = np.zeros_like(masks[0])

    def render(qpos: np.ndarray, renderer: Any, lookat: np.ndarray) -> np.ndarray:
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        camera.lookat[:] = lookat
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()

    try:
        for frame_index in range(frame_count):
            common_lookat = replay_qpos[0][frame_index, :3]
            source_frame = render(source_qpos[frame_index], full_renderer, common_lookat)
            original_full = render(replay_qpos[0][frame_index], full_renderer, common_lookat)
            original_panel = render(replay_qpos[0][frame_index], panel_renderer, common_lookat)
            scenario_frames = [
                render(value[frame_index], panel_renderer, common_lookat)
                for value in replay_qpos[1:]
            ]
            writers["source"].append_data(_label(cv2, source_frame, "SONIC source policy"))
            writers["original"].append_data(
                _label(cv2, original_full, "Original Action physics replay")
            )
            for name in selected:
                scenario_index = completion_names.index(name)
                writers[name].append_data(
                    _compose_pair(
                        cv2,
                        original_panel,
                        scenario_frames[scenario_index],
                        name,
                        masks[scenario_index],
                        frame_index,
                        args.width,
                        args.height,
                    )
                )
            writers["grid"].append_data(
                _compose_grid(
                    cv2,
                    [original_panel, *scenario_frames],
                    scenario_names,
                    [zero_mask, *list(masks)],
                    frame_index,
                    args.width,
                    args.height,
                )
            )
    finally:
        for writer in writers.values():
            writer.close()
        panel_renderer.close()
        full_renderer.close()

    outputs = [source_path, original_path, grid_path, *comparison_paths.values()]
    encoded = {}
    for path in outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Renderer did not produce a non-empty video: {path}")
        encoded_frames, duration = count_frames_and_secs(str(path))
        if encoded_frames != frame_count:
            raise RuntimeError(f"{path.name} has {encoded_frames} frames; expected {frame_count}")
        encoded[path.name] = {
            "frames": int(encoded_frames),
            "duration_seconds": float(duration),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": "sonic_action_mask_render_v1",
        "renderer": "mujoco_offscreen_common_world_camera",
        "gl_backend": args.gl,
        "run_dir": str(run_dir),
        "source_model": str(model_path),
        "runtime_model": str(runtime_xml.resolve()),
        "render_mode": args.render_mode,
        "scenario_names": scenario_names,
        "comparison_scenarios": selected,
        "frames": frame_count,
        "fps": fps,
        "width": args.width,
        "height": args.height,
        "camera": {
            "type": "free",
            "lookat_source": "original_action_replay_root_per_frame",
            "distance": args.camera_distance,
            "azimuth": args.camera_azimuth,
            "elevation": args.camera_elevation,
        },
        "mask_overlay": "red per-action-step timeline; white current-step cursor",
        "videos": encoded,
        "xml_patch": xml_details,
    }
    write_json_atomic(manifests_dir / "action_mask_render.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
