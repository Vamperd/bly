#!/usr/bin/env python3
"""Three-layer ghost video: recorded source, original C, fine-tuned C.

The three trajectories must be 50 Hz kinematic trajectory-pickles produced by
the stepwise exact-init causal replay.  This is visual comparison only, not a
torque-physics replay.
"""

from __future__ import annotations

import argparse
import hashlib
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


GHOST_SIZE = (1280, 720)
GHOST_ALPHA = 0.55
SOURCE_COLOR = (235, 235, 235)
ORIGINAL_COLOR = (70, 165, 255)
FINETUNED_COLOR = (255, 140, 60)
FPS = 50.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _robot_mask(segmentation, geom_types, plane_type, geom_objtype):
    ids, types = segmentation[..., 0], segmentation[..., 1]
    mask = (types == geom_objtype) & (ids >= 0)
    mask[mask] = geom_types[ids[mask]] != plane_type
    return mask


def _blend(base, layer, mask, color):
    tinted = 0.45 * layer.astype(np.float32) + 0.55 * np.asarray(color, np.float32)
    out = base.astype(np.float32)
    out[mask] = (1.0 - GHOST_ALPHA) * out[mask] + GHOST_ALPHA * tinted[mask]
    return out.clip(0, 255).astype(np.uint8)


def _qposes(model, paths: list[Path], profile: str, frames: int):
    return [
        build_mujoco_qpos(load_trajectory(path), profile=profile, model=model)[:frames]
        for path in paths
    ]


def render(args: argparse.Namespace) -> dict:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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

    xml = output / "manifests" / "three_way_ghost.xml"
    xml_details = prepare_runtime_xml(
        args.model.expanduser().resolve(),
        xml,
        *GHOST_SIZE,
        mesh_dir=args.mesh_dir,
        ground_style=args.ground_style,
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

    results = []
    try:
        for case in manifest["cases"]:
            motion = str(case["motion_key"])
            policies = case["policies"]
            source_path = Path(case["source_trajectory"])
            original_path = Path(policies["original_c"]["trajectory"])
            finetuned_path = Path(policies["finetuned_c"]["trajectory"])
            trajectories = [
                load_trajectory(source_path),
                load_trajectory(original_path),
                load_trajectory(finetuned_path),
            ]
            frames = min(int(item["dof_pos"].shape[0]) for item in trajectories)
            if frames <= 0:
                raise ValueError(f"empty trajectory for {motion}")
            qposes = _qposes(
                model,
                [source_path, original_path, finetuned_path],
                args.robot_profile,
                frames,
            )
            lookat_z = float(qposes[0][0, 2])
            video_path = output / "videos" / f"ghost_source_original_finetuned_{motion}.mp4"
            writer = imageio.get_writer(
                str(video_path), fps=FPS, codec="libx264", quality=8,
                macro_block_size=8,
            )
            try:
                for frame_index in range(frames):
                    lookat = np.asarray(
                        [
                            qposes[0][frame_index, 0],
                            qposes[0][frame_index, 1],
                            lookat_z,
                        ]
                    )
                    frame = draw(qposes[0][frame_index], lookat)
                    for qpos, color in (
                        (qposes[1][frame_index], ORIGINAL_COLOR),
                        (qposes[2][frame_index], FINETUNED_COLOR),
                    ):
                        layer = draw(qpos, lookat)
                        segmentation = draw(qpos, lookat, segment=True)
                        mask = _robot_mask(
                            segmentation, geom_types, plane_type, geom_objtype
                        )
                        frame = _blend(frame, layer, mask, color)
                    cv2.rectangle(
                        frame, (0, 0), (GHOST_SIZE[0], 106), (25, 25, 25), -1
                    )
                    cv2.putText(
                        frame,
                        f"Ghost | source opaque | {motion} | frame {frame_index}/{frames - 1} | 50 Hz",
                        (10, 27),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (245, 245, 245),
                        1,
                    )
                    for offset, color, label in (
                        (0, SOURCE_COLOR, "source recorded"),
                        (215, ORIGINAL_COLOR, "C original"),
                        (430, FINETUNED_COLOR, "C fine-tuned"),
                    ):
                        cv2.rectangle(
                            frame, (10 + offset, 43), (29 + offset, 61), color, -1
                        )
                        cv2.putText(
                            frame,
                            label,
                            (38 + offset, 59),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.47,
                            (235, 235, 235),
                            1,
                        )
                    cv2.putText(
                        frame,
                        "kinematic ghost | same camera follows source | not torque replay",
                        (10, 88),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (230, 230, 230),
                        1,
                    )
                    writer.append_data(frame)
            finally:
                writer.close()
            rendered_frames, seconds = count_frames_and_secs(str(video_path))
            if rendered_frames != frames or video_path.stat().st_size <= 0:
                raise RuntimeError(
                    f"ghost video validation failed: {video_path}, "
                    f"frames={rendered_frames}, expected={frames}"
                )
            results.append(
                {
                    "motion_key": motion,
                    "path": str(video_path),
                    "sha256": _sha256(video_path),
                    "frames": int(rendered_frames),
                    "seconds": float(seconds),
                    "source_trajectory": str(source_path),
                    "original_trajectory": str(original_path),
                    "finetuned_trajectory": str(finetuned_path),
                }
            )
            print(f"[ghost] rendered {motion}: {video_path}", flush=True)
    finally:
        renderer.close()

    result = {
        "format_version": "sonic_three_way_ghost_v1",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "model": str(args.model.expanduser().resolve()),
        "model_details": xml_details,
        "profile": profile_details,
        "ghost_alpha": GHOST_ALPHA,
        "layout": "source opaque, original C and fine-tuned C translucent",
        "videos": results,
    }
    write_json_atomic(output / "manifests" / "three_way_ghost.json", result)
    (output / "markers" / "three_way_ghost.ok").write_text(
        "PASS\n", encoding="utf-8"
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--robot-profile", default="g1")
    parser.add_argument("--mesh-dir", type=Path)
    parser.add_argument("--ground-style", default="checker", choices=("checker", "black"))
    parser.add_argument("--gl", default="egl", choices=("egl", "osmesa", "glfw"))
    args = parser.parse_args(argv)
    render(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
