#!/usr/bin/env python3
"""Render synchronized recorded, truth-State, and predicted-State trajectories."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from render_mujoco_trajectory import (
    G1_ISAACLAB_TO_MUJOCO_DOF,
    G1_MUJOCO_JOINT_NAMES,
    prepare_runtime_xml,
    write_json_atomic,
)


REPRESENTATIVE_OUTPUTS = {
    "forward_rollout_8": "forward_rollout_8_comparison.mp4",
    "forward_rollout_32": "forward_rollout_32_ood_comparison.mp4",
    "step_contiguous_32": "state_step_32_comparison.mp4",
    "feature_random_25": "state_feature_random_25_comparison.mp4",
    "semantic_base_motion": "state_semantic_base_motion_comparison.mp4",
    "semantic_left_leg": "state_semantic_left_leg_comparison.mp4",
}

NON_REPRESENTATIVE_OUTPUTS = {
    "step_contiguous_8": "state_step_8_comparison.mp4",
}


def comparison_output_name(scenario: str) -> str:
    return REPRESENTATIVE_OUTPUTS.get(
        scenario,
        NON_REPRESENTATIVE_OUTPUTS.get(scenario, f"state_{scenario}_comparison.mp4"),
    )


def grid_shape(panel_count: int) -> tuple[int, int]:
    if panel_count <= 0:
        raise ValueError("panel_count must be positive")
    columns = min(5, panel_count)
    return math.ceil(panel_count / columns), columns


def _writer(imageio: Any, path: Path, fps: float):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing video: {path}")
    return imageio.get_writer(
        path, fps=fps, codec="libx264", quality=5, pixelformat="yuv420p",
        macro_block_size=2,
    )


def _label(cv2: Any, frame: np.ndarray, text: str) -> np.ndarray:
    result = frame.copy()
    compact = result.shape[1] < 320
    height = 25 if compact else 34
    scale = 0.38 if compact else 0.58
    cv2.rectangle(result, (0, 0), (result.shape[1], height), (20, 20, 20), -1)
    cv2.putText(
        result, text, (6 if compact else 10, 18 if compact else 24),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA,
    )
    return result


def _timeline(cv2: Any, frame: np.ndarray, mask: np.ndarray, frame_index: int) -> np.ndarray:
    result = frame.copy()
    height, width = result.shape[:2]
    y0 = height - 14
    cv2.rectangle(result, (0, y0), (width - 1, height - 1), (35, 35, 35), -1)
    active = np.flatnonzero(mask.any(axis=-1))
    total = mask.shape[0]
    for state_index in active:
        x0 = int(state_index * width / max(total, 1))
        x1 = max(x0 + 1, int((state_index + 1) * width / max(total, 1)))
        cv2.rectangle(result, (x0, y0), (x1, height - 1), (230, 20, 20), -1)
    x = min(width - 1, int(frame_index * width / max(total, 1)))
    cv2.line(result, (x, y0 - 4), (x, height - 1), (255, 255, 255), 2)
    return result


def _triple(
    cv2: Any,
    source: np.ndarray,
    truth: np.ndarray,
    predicted: np.ndarray,
    scenario: str,
    mask: np.ndarray,
    frame_index: int,
    width: int,
    height: int,
) -> np.ndarray:
    widths = (width // 3, width // 3, width - 2 * (width // 3))
    panels = []
    for frame, label, panel_width in zip(
        (source, truth, predicted),
        ("Recorded source", "Truth State reconstruction", scenario),
        widths,
        strict=True,
    ):
        panel = cv2.resize(frame, (panel_width, height), interpolation=cv2.INTER_AREA)
        panels.append(_label(cv2, panel, label))
    return _timeline(cv2, np.concatenate(panels, axis=1), mask, frame_index)


def _grid(
    cv2: Any,
    frames: list[np.ndarray],
    labels: list[str],
    masks: list[np.ndarray],
    frame_index: int,
    width: int,
    height: int,
) -> np.ndarray:
    rows, columns = grid_shape(len(frames))
    panel_width, panel_height = width // columns, height // rows
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    for index, (frame, label, mask) in enumerate(zip(frames, labels, masks, strict=True)):
        row, column = divmod(index, columns)
        x0, y0 = column * panel_width, row * panel_height
        x1 = width if column == columns - 1 else (column + 1) * panel_width
        y1 = height if row == rows - 1 else (row + 1) * panel_height
        panel = cv2.resize(frame, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        panel = _timeline(cv2, _label(cv2, panel, label), mask, frame_index)
        canvas[y0:y1, x0:x1] = panel
    return canvas


def _qpos(root_pos: np.ndarray, root_quat: np.ndarray, joints: np.ndarray) -> np.ndarray:
    if root_pos.shape != (joints.shape[0], 3) or root_quat.shape != (joints.shape[0], 4):
        raise ValueError("root and joint trajectory lengths differ")
    quaternion = root_quat.astype(np.float64)
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(1.0e-12)
    return np.concatenate(
        (root_pos, quaternion, joints[:, G1_ISAACLAB_TO_MUJOCO_DOF]), axis=-1
    ).astype(np.float64)


def _validate_model(model: Any) -> None:
    names = tuple(
        model.joint(index).name
        for index in range(model.njnt)
        if model.joint(index).name != "floating_base_joint"
    )
    if names != G1_MUJOCO_JOINT_NAMES:
        raise ValueError("MuJoCo joint order does not match the canonical G1 order")


def _body_positions(model: Any, qpos: np.ndarray) -> np.ndarray:
    import mujoco

    data = mujoco.MjData(model)
    values = []
    for frame in qpos:
        data.qpos[:] = frame
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        values.append(data.xpos[1:].copy())
    return np.asarray(values)


def _body_metrics(reference: np.ndarray, value: np.ndarray) -> dict[str, float]:
    difference = value - reference
    per_body = np.linalg.norm(difference, axis=-1)
    return {
        "body_mpjpe_m": float(per_body.mean()),
        "body_max_error_m": float(per_body.max()),
    }


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
        raise ValueError("video width and height must be positive even integers")
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    model_path = args.model.expanduser().resolve(strict=True)
    videos = run_dir / "videos"
    manifests = run_dir / "manifests"
    request = json.loads((manifests / "state_mask_request.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "data/state_predictions.npz", allow_pickle=False) as values:
        names = [str(value) for value in values["scenario_names"].tolist()]
        source_state = np.asarray(values["source_state"], dtype=np.float32)
        completed_state = np.asarray(values["completed_state"], dtype=np.float32)
        masks = np.asarray(values["mask_state"], dtype=bool)
        nominal = np.asarray(values["nominal_joint_pos"], dtype=np.float32)
        source_root_pos = np.asarray(values["source_root_pos"], dtype=np.float32)
        source_root_quat = np.asarray(values["source_root_quat"], dtype=np.float32)
        truth_root_pos = np.asarray(values["truth_reconstructed_root_pos"], dtype=np.float32)
        truth_root_quat = np.asarray(values["truth_reconstructed_root_quat"], dtype=np.float32)
        predicted_root_pos = np.asarray(values["predicted_root_pos"], dtype=np.float32)
        predicted_root_quat = np.asarray(values["predicted_root_quat"], dtype=np.float32)
        fps = float(values["fps"])
    frame_count = source_state.shape[0]
    expected = (len(names), frame_count, 70)
    if completed_state.shape != expected or masks.shape != expected:
        raise ValueError("State prediction arrays have inconsistent shapes")
    if predicted_root_pos.shape != (len(names), frame_count, 3):
        raise ValueError("predicted root positions have an invalid shape")
    if predicted_root_quat.shape != (len(names), frame_count, 4):
        raise ValueError("predicted root orientations have an invalid shape")
    if nominal.shape != (29,) or not np.isclose(fps, 50.0):
        raise ValueError("State renderer requires 29 joints and 50 FPS")
    selected = names if args.render_mode == "all" else [name for name in REPRESENTATIVE_OUTPUTS if name in names]
    if args.render_mode == "representatives" and set(selected) != set(REPRESENTATIVE_OUTPUTS):
        raise ValueError("representative State scenarios are missing")

    source_joints = source_state[:, :29] + nominal
    predicted_joints = completed_state[:, :, :29] + nominal[None, None]
    source_qpos = _qpos(source_root_pos, source_root_quat, source_joints)
    truth_qpos = _qpos(truth_root_pos, truth_root_quat, source_joints)
    predicted_qpos = [
        _qpos(predicted_root_pos[index], predicted_root_quat[index], predicted_joints[index])
        for index in range(len(names))
    ]

    os.environ["MUJOCO_GL"] = args.gl
    import cv2
    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    import mujoco

    runtime_xml = manifests / "g1_state_mask_render.xml"
    xml_details = prepare_runtime_xml(model_path, runtime_xml, args.width, args.height)
    model = mujoco.MjModel.from_xml_path(str(runtime_xml))
    if model.nq != 36:
        raise ValueError(f"expected MuJoCo nq=36, found {model.nq}")
    _validate_model(model)
    source_body = _body_positions(model, source_qpos)
    truth_body = _body_positions(model, truth_qpos)
    predicted_body = [_body_positions(model, value) for value in predicted_qpos]
    body_report = {
        "schema_version": "sonic_state_kinematic_metrics_v1",
        "truth_reconstruction_vs_recorded": _body_metrics(source_body, truth_body),
        "scenarios": {
            name: _body_metrics(truth_body, predicted_body[index])
            for index, name in enumerate(names)
        },
    }
    write_json_atomic(manifests / "state_kinematic_metrics.json", body_report)

    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    camera.distance = args.camera_distance
    full_renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    panel_renderer = mujoco.Renderer(model, height=args.height, width=max(2, args.width // 3))

    source_path = videos / "state_source_recorded.mp4"
    truth_path = videos / "state_truth_reconstruction.mp4"
    grid_path = videos / "all_state_predictions_grid.mp4"
    writers = {
        "source": _writer(imageio, source_path, fps),
        "truth": _writer(imageio, truth_path, fps),
        "grid": _writer(imageio, grid_path, fps),
    }
    comparison_paths = {name: videos / comparison_output_name(name) for name in selected}
    for name, path in comparison_paths.items():
        writers[name] = _writer(imageio, path, fps)
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
            lookat = source_root_pos[frame_index]
            source_full = render(source_qpos[frame_index], full_renderer, lookat)
            truth_full = render(truth_qpos[frame_index], full_renderer, lookat)
            source_panel = render(source_qpos[frame_index], panel_renderer, lookat)
            truth_panel = render(truth_qpos[frame_index], panel_renderer, lookat)
            predicted_panels = [
                render(value[frame_index], panel_renderer, lookat) for value in predicted_qpos
            ]
            writers["source"].append_data(_label(cv2, source_full, "Recorded source trajectory"))
            writers["truth"].append_data(_label(cv2, truth_full, "Truth State integration reconstruction"))
            for name in selected:
                index = names.index(name)
                writers[name].append_data(
                    _triple(
                        cv2, source_panel, truth_panel, predicted_panels[index], name,
                        masks[index], frame_index, args.width, args.height,
                    )
                )
            writers["grid"].append_data(
                _grid(
                    cv2,
                    [source_panel, truth_panel, *predicted_panels],
                    ["Recorded", "Truth State", *names],
                    [zero_mask, zero_mask, *list(masks)],
                    frame_index, args.width, args.height,
                )
            )
    finally:
        for writer in writers.values():
            writer.close()
        panel_renderer.close()
        full_renderer.close()

    outputs = [source_path, truth_path, grid_path, *comparison_paths.values()]
    encoded: dict[str, Any] = {}
    for path in outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"State renderer did not create {path}")
        frames, duration = count_frames_and_secs(str(path))
        if frames != frame_count:
            raise RuntimeError(f"{path.name} has {frames} frames; expected {frame_count}")
        encoded[path.name] = {
            "frames": int(frames), "duration_seconds": float(duration),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": "sonic_state_mask_render_v1",
        "renderer": "mujoco_forward_kinematics_common_world_camera",
        "render_mode": args.render_mode,
        "motion_key": request["motion_key"],
        "scenario_names": names,
        "comparison_scenarios": selected,
        "frames": frame_count,
        "fps": fps,
        "width": args.width,
        "height": args.height,
        "camera": {"lookat_source": "recorded_source_root_per_frame"},
        "videos": encoded,
        "xml_patch": xml_details,
    }
    write_json_atomic(manifests / "state_mask_render.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
