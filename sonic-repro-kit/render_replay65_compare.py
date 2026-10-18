"""Ghost overlay and 2x2 grid of several replay65 routes on one window (kinematic playback only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from render_mujoco_trajectory import (G1_ISAACLAB_JOINT_NAMES, build_mujoco_qpos,
    load_trajectory, prepare_runtime_xml, validate_model_profile, write_json_atomic)
from render_h50a_exact_action_replays import _writer
from render_replay65 import scored_frames
from replay65_render_utils import FRAME_PROVENANCE_VERSION, draw_provenance_timeline, frame_provenance

GHOST_SIZE = (1280, 720)
PANEL_SIZE = (640, 480)
GHOST_ALPHA = .55
ERROR_KEYS = (("joint_position_rmse_rad", "joint", "rad"), ("root_position_error_m", "root", "m"),
              ("body_mpjpe_m", "mpjpe", "m"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def robot_mask(segmentation, geom_types, plane_type, geom_objtype):
    """Pixels whose geom is part of the robot (not floor/skybox)."""
    ids, types = segmentation[..., 0], segmentation[..., 1]
    mask = (types == geom_objtype) & (ids >= 0)
    mask[mask] = geom_types[ids[mask]] != plane_type
    return mask


def blend(base, layer, mask, color, alpha=GHOST_ALPHA):
    tinted = .45 * layer.astype(np.float32) + .55 * np.asarray(color, np.float32)
    out = base.astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * tinted[mask]
    return out.clip(0, 255).astype(np.uint8)


def fall_strip(cv2, frame, fall_frame, t, frames, y):
    width = frame.shape[1] - 20
    cv2.rectangle(frame, (10, y), (10 + width, y + 10), (70, 110, 70), -1)
    if fall_frame is not None:
        x0 = 10 + fall_frame * width // max(frames - 1, 1)
        cv2.rectangle(frame, (x0, y), (10 + width, y + 10), (220, 50, 50), -1)
    x = 10 + t * width // max(frames - 1, 1)
    cv2.line(frame, (x, y - 4), (x, y + 14), (255, 255, 255), 2)


def error_text(route, t):
    return "  ".join(f"{name}={route['per_frame'][key][t]:.3f}{unit}" for key, name, unit in ERROR_KEYS)


def load_inputs(manifest, *, model, robot_profile):
    horizon, offset = int(manifest["horizon_steps"]), int(manifest["score_offset"])
    source_path = Path(manifest["source_trajectory"])
    with np.load(source_path.with_name("source.replay.npz"), allow_pickle=False) as archive:
        if tuple(archive["joint_names"].tolist()) != G1_ISAACLAB_JOINT_NAMES:
            raise ValueError("recorded Isaac joint order does not match the renderer permutation")
    paths = [(source_path, manifest["source_trajectory_sha256"])] + [
        (Path(r["trajectory"]), r["trajectory_sha256"]) for r in manifest["routes"]]
    qposes = []
    for path, expected in paths:
        if sha(path) != expected:
            raise ValueError(f"trajectory changed since compare manifest was written: {path}")
        trajectory = load_trajectory(path)
        if not np.isclose(trajectory["fps"], 50):
            raise ValueError(f"expected 50 Hz trajectory: {path}")
        qpos = scored_frames(build_mujoco_qpos(trajectory, profile=robot_profile, model=model), horizon, offset)
        if qpos.shape != (horizon + 1, int(model.nq)) or not np.isfinite(qpos).all():
            raise ValueError(f"partial/non-finite trajectory cannot be compared: {path}")
        qposes.append(qpos)
    return qposes, {str(path): expected for path, expected in paths}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--robot-profile", choices=("g1", "g1_plus_locked29"), default="g1")
    p.add_argument("--mesh-dir", type=Path)
    p.add_argument("--gl", default="egl", choices=("egl", "osmesa", "glfw"))
    p.add_argument("--layout", default="both", choices=("ghost", "grid", "both"))
    args = p.parse_args(argv)
    os.environ["MUJOCO_GL"] = args.gl
    import cv2
    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    import mujoco
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = manifest_path.parent.parent
    routes, horizon = manifest["routes"], int(manifest["horizon_steps"])
    provenance_meta = dict(manifest)
    source_trajectory = Path(manifest["source_trajectory"])
    window_meta_path = source_trajectory.parent.parent / "manifests" / "replay65.json"
    if window_meta_path.is_file():
        provenance_meta.update(json.loads(window_meta_path.read_text(encoding="utf-8")))
    provenance = frame_provenance(provenance_meta, horizon)
    frames_total, start = horizon + 1, int(manifest["window_start"])
    xml = output / "manifests/compare_render.xml"
    xml_details = prepare_runtime_xml(args.model.resolve(), xml, *GHOST_SIZE, mesh_dir=args.mesh_dir)
    model = mujoco.MjModel.from_xml_path(str(xml))
    profile_details = validate_model_profile(model, args.robot_profile)
    qposes, input_hashes = load_inputs(manifest, model=model, robot_profile=args.robot_profile)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 135, -15
    plane, geom_obj = int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtObj.mjOBJ_GEOM)
    geom_types = np.asarray(model.geom_type)

    def draw(renderer, qpos, lookat, segment=False):
        data.qpos[:] = qpos; data.qvel[:] = 0
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
    def ghost_frame(renderer, t):
        lookat = qposes[0][t, :3]
        frame = draw(renderer, qposes[0][t], lookat)
        for route, qpos in zip(routes, qposes[1:]):
            layer = draw(renderer, qpos[t], lookat)
            mask = robot_mask(draw(renderer, qpos[t], lookat, segment=True), geom_types, plane, geom_obj)
            frame = blend(frame, layer, mask, route["color"])
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 64 + 24 * len(routes)), (25, 25, 25), -1)
        cv2.putText(frame, f"Ghost overlay | source opaque | {manifest['mask']} | frame {t}/{horizon} (ep {start + t}) | 50 Hz",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 240), 1)
        for i, route in enumerate(routes):
            fallen = route["fall_frame"] is not None and t >= route["fall_frame"]
            y = 52 + 24 * i
            cv2.rectangle(frame, (10, y - 13), (28, y + 3), tuple(route["color"]), -1)
            cv2.putText(frame, f"{route['label']} (route {route['route']}, {route['mask']})  {error_text(route, t)}"
                        + ("  FALLEN" if fallen else ""), (36, y), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (255, 90, 90) if fallen else (235, 235, 235), 1)
        cv2.putText(frame, provenance["label"], (10, 64 + 24 * len(routes) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (230, 230, 230), 1)
        draw_provenance_timeline(cv2, frame, t, horizon, provenance, frame.shape[0] - 20)
        return frame

    def grid_frame(renderer, t):
        lookat = qposes[0][t, :3]
        panels = []
        rows = [{"label": "Source", "route": "-", "mask": manifest.get("source_mode", "source"),
                 "color": [230, 230, 230], "fall_frame": manifest.get("source_fall_frame"), "per_frame": None}, *routes]
        for row, qpos in zip(rows, qposes):
            frame = draw(renderer, qpos[t], lookat)
            fallen = row["fall_frame"] is not None and t >= row["fall_frame"]
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 98), (25, 25, 25), -1)
            cv2.rectangle(frame, (8, 8), (24, 24), tuple(row["color"]), -1)
            cv2.putText(frame, f"{row['label']}  route {row['route']} | {row['mask']} | frame {t}/{horizon}",
                        (32, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1)
            if row["per_frame"] is not None:
                cv2.putText(frame, error_text(row, t), (10, 48), cv2.FONT_HERSHEY_SIMPLEX, .45, (210, 210, 210), 1)
            cv2.putText(frame, provenance["label"], (10, 78), cv2.FONT_HERSHEY_SIMPLEX, .30, (230, 230, 230), 1)
            if fallen:
                cv2.putText(frame, "FALLEN", (frame.shape[1] - 130, 110), cv2.FONT_HERSHEY_SIMPLEX, 1., (230, 40, 40), 2)
            draw_provenance_timeline(cv2, frame, t, horizon, provenance, 440)
            fall_strip(cv2, frame, row["fall_frame"], t, frames_total, frame.shape[0] - 20)
            panels.append(frame)
        while len(panels) < 4:
            panels.append(np.zeros_like(panels[0]))
        return np.concatenate([np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:4], axis=1)], axis=0)

    jobs = []
    if args.layout in ("ghost", "both"):
        jobs.append(("compare_ghost", GHOST_SIZE, 3.5, ghost_frame))
    if args.layout in ("grid", "both"):
        if len(routes) > 3:
            raise ValueError("2x2 grid holds Source plus at most three routes")
        jobs.append(("compare_grid", PANEL_SIZE, 3.5, grid_frame))
    videos = []
    for name, (width, height), distance, make_frame in jobs:
        path = output / "videos" / f"{name}.mp4"
        camera.distance = distance
        renderer = mujoco.Renderer(model, height=height, width=width)
        writer = _writer(imageio, path, 50.)
        try:
            for t in range(frames_total):
                writer.append_data(make_frame(renderer, t))
        finally:
            writer.close()
            renderer.close()
        frames, seconds = count_frames_and_secs(str(path))
        if frames != frames_total or path.stat().st_size <= 0:
            raise RuntimeError(f"encoded frame count mismatch for {path}: {frames} != {frames_total}")
        videos.append({"path": str(path.relative_to(output)), "sha256": sha(path), "frames": frames, "seconds": seconds})
        print(f"Rendered {path} ({frames} frames)", flush=True)
    write_json_atomic(output / "manifests/compare_render.json", {"version": "65-token-replay-compare-v2",
        "robot_profile": args.robot_profile,
        "model_nq": int(model.nq), "model_nv": int(model.nv),
        "model_joint_names": profile_details["model_joint_names"],
        "fixed_joint_names": profile_details["fixed_joint_names"],
        "model_to_policy_mapping": profile_details["model_to_policy_mapping"],
        "policy_to_model_mapping": profile_details["policy_to_model_mapping"],
        "videos": videos, "input_hashes": input_hashes, "compare_manifest_sha256": sha(manifest_path),
        "frame_provenance_version": FRAME_PROVENANCE_VERSION, "frame_provenance": provenance,
        "xml": xml_details, "camera": "follows source root; same camera for every route",
        "ghost": {"alpha": GHOST_ALPHA, "note": "2D compositing; occlusion between ghosts is not depth sorted"},
        "kinematic_only": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
