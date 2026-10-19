#!/usr/bin/env python3
"""Run a short MuJoCo actuator smoke for the unlocked G1+ neck joints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from render_mujoco_trajectory import validate_model_profile


PROFILE = "g1_plus_unlocked31"
NECK_NAMES = ("neck_pitch_joint", "neck_yaw_joint")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    import mujoco

    model_path = args.model.expanduser().resolve(strict=True)
    output = args.output_run.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = output / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    profile = validate_model_profile(model, PROFILE)

    joint_ids = {
        name: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        for name in NECK_NAMES
    }
    actuator_map: dict[str, list[int]] = {}
    for name, joint_id in joint_ids.items():
        actuator_map[name] = [
            index
            for index in range(int(model.nu))
            if int(model.actuator_trnid[index, 0]) == joint_id
        ]
        if not actuator_map[name]:
            raise RuntimeError(f"no actuator is transmitted to {name}")

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0
    original_gravity = np.asarray(model.opt.gravity).copy()
    model.opt.gravity[:] = 0
    mujoco.mj_forward(model, data)

    body_qpos_addresses = set(range(int(model.nq)))
    for name in NECK_NAMES:
        body_qpos_addresses.discard(int(model.jnt_qposadr[joint_ids[name]]))
    body_initial = data.qpos[list(sorted(body_qpos_addresses))].copy()
    initial_neck = {
        name: float(data.qpos[int(model.jnt_qposadr[joint_ids[name]])])
        for name in NECK_NAMES
    }

    actuator_details = {}
    trajectories = {name: [] for name in NECK_NAMES}
    for name in NECK_NAMES:
        actuator_ids = actuator_map[name]
        actuator_details[name] = []
        for actuator_id in actuator_ids:
            limited = bool(model.actuator_ctrllimited[actuator_id])
            ctrlrange = np.asarray(model.actuator_ctrlrange[actuator_id], dtype=float)
            if limited and np.isfinite(ctrlrange).all() and ctrlrange[1] > ctrlrange[0]:
                low, high = map(float, ctrlrange)
                amplitude = 0.5 * (high - low)
                center = 0.5 * (high + low)
                signal = np.linspace(center - 0.5 * amplitude, center + 0.5 * amplitude, args.steps)
            else:
                low, high = -0.1, 0.1
                signal = 0.1 * np.sin(np.linspace(0.0, 2.0 * np.pi, args.steps))
            actuator_details[name].append({
                "actuator_id": int(actuator_id),
                "actuator_name": str(model.actuator(actuator_id).name),
                "joint_id": joint_ids[name],
                "ctrl_limited": limited,
                "ctrlrange": ctrlrange.tolist(),
                "test_control_low": low,
                "test_control_high": high,
            })

        for step in range(args.steps):
            data.ctrl[:] = 0
            for actuator_id, detail in zip(actuator_ids, actuator_details[name]):
                low = float(detail["test_control_low"])
                high = float(detail["test_control_high"])
                data.ctrl[actuator_id] = np.linspace(low, high, args.steps)[step]
            mujoco.mj_step(model, data)
            for joint_name in NECK_NAMES:
                address = int(model.jnt_qposadr[joint_ids[joint_name]])
                trajectories[joint_name].append(float(data.qpos[address]))

    model.opt.gravity[:] = original_gravity
    final_neck = {
        name: float(data.qpos[int(model.jnt_qposadr[joint_ids[name]])])
        for name in NECK_NAMES
    }
    body_final = data.qpos[list(sorted(body_qpos_addresses))]
    response = {
        name: float(max(trajectories[name]) - min(trajectories[name]))
        for name in NECK_NAMES
    }
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    body_drift = float(np.max(np.abs(body_final - body_initial))) if body_initial.size else 0.0
    result = {
        "version": "g1_plus_unlocked31_neck_actuator_smoke_v1",
        "profile": PROFILE,
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "neck_joint_ids": joint_ids,
        "actuators": actuator_details,
        "steps": int(args.steps),
        "timestep": float(model.opt.timestep),
        "initial_neck_qpos": initial_neck,
        "final_neck_qpos": final_neck,
        "neck_response_rad": response,
        "body_qpos_max_drift": body_drift,
        "finite_qpos_qvel": finite,
        "response_pass": all(value > args.min_response for value in response.values()),
        "body_drift_pass": body_drift < args.max_body_drift,
        "pass": finite
        and all(value > args.min_response for value in response.values())
        and body_drift < args.max_body_drift,
    }
    path = diagnostics / "neck_actuator_smoke.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["pass"]:
        raise RuntimeError(f"neck actuator smoke failed; see {path}")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--min-response", type=float, default=1.0e-5)
    parser.add_argument("--max-body-drift", type=float, default=1.0e-3)
    parser.add_argument("--gl", choices=("egl", "osmesa", "glfw"), default="egl")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import os
    os.environ["MUJOCO_GL"] = args.gl
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
