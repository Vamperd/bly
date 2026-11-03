"""Export per-frame 70-D reference States for the motions of a Physics v5 dataset.

Runs as ``python -m dagger.export_reference_table`` (not ``dagger.cli``) so the
run_dagger.sh process guard of a concurrent DAgger run never matches it.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch


SCHEMA_VERSION = "cvae_reference_table_v1"
TABLE_NAME = "reference_states.npz"
OFFSETS = (-1, 0, 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def read_physics_episodes(dataset_run: Path) -> list[dict]:
    path = dataset_run / "manifests" / "episodes.jsonl"
    episodes = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not episodes:
        raise ValueError(f"no episodes in {path}")
    return episodes


def joint_index(joint_ids, width: int) -> np.ndarray:
    if isinstance(joint_ids, slice):
        return np.arange(width, dtype=np.int64)[joint_ids]
    return np.asarray(torch.as_tensor(joint_ids).cpu(), dtype=np.int64).reshape(-1)


def alignment_check(
    episodes: list[dict],
    table: dict[str, np.ndarray],
    *,
    tolerance: float,
) -> dict:
    """Compare table frame motion_step[t]+d with HDF5 reference frame 0 at t.

    The recorder reshapes SONIC's flat [pos 10x29 | vel 10x29] command to
    [10,58], so frame-0 pos is flat[:29] and frame-0 vel is flat[290:319].
    The recorded reference is the motion lib's absolute dof order, so it is
    compared both raw and reordered by the action joint ids.
    """

    import h5py

    keys = [str(value) for value in table["motion_keys"]]
    key_index = {key: index for index, key in enumerate(keys)}
    starts = table["motion_offsets"]
    num_steps = table["num_steps"]
    states = table["states"]
    nominal = table["nominal_joint_pos"].astype(np.float64)
    order = table["joint_ids"]
    columns_by_name = {"recorded_reindexed": order, "recorded_raw": slice(None)}
    errors = {
        name: {offset: {"pos": 0.0, "vel": 0.0} for offset in OFFSETS}
        for name in columns_by_name
    }
    sums = {name: {offset: 0.0 for offset in OFFSETS} for name in columns_by_name}
    motion_max: dict[str, float] = {}
    rows = 0
    clipped = 0
    handles: dict[str, h5py.File] = {}
    try:
        for item in episodes:
            key = str(item["motion_key"])
            if key not in key_index:
                raise KeyError(f"episode {item['episode']} uses motion {key} missing from the table")
            motion = key_index[key]
            path = str(item["hdf5_path"])
            if path not in handles:
                handles[path] = h5py.File(path, "r")
            group = handles[path][f"data/{item['episode']}"]
            motion_step = np.asarray(group["motion/motion_step"], dtype=np.int64).reshape(-1)
            future = np.asarray(group["reference/joint_pos_vel_future"], dtype=np.float64)
            if future.shape != (motion_step.shape[0], 10, 58):
                raise ValueError(f"{item['episode']}: unexpected reference shape {future.shape}")
            flat = future.reshape(future.shape[0], -1)
            recorded_pos = flat[:, :29]
            recorded_vel = flat[:, 290:319]
            last = int(num_steps[motion]) - 1
            clipped += int(np.count_nonzero((motion_step < 0) | (motion_step > last)))
            rows += int(motion_step.shape[0])
            for offset in OFFSETS:
                frame = np.clip(motion_step + offset, 0, last) + int(starts[motion])
                pos = states[frame, :29].astype(np.float64) + nominal
                vel = states[frame, 29:58].astype(np.float64)
                for name, columns in columns_by_name.items():
                    pos_diff = np.abs(pos - recorded_pos[:, columns])
                    vel_diff = np.abs(vel - recorded_vel[:, columns])
                    sums[name][offset] += float(pos_diff.sum())
                    slot = errors[name][offset]
                    slot["pos"] = max(slot["pos"], float(pos_diff.max()))
                    slot["vel"] = max(slot["vel"], float(vel_diff.max()))
                    if offset == 0 and name == "recorded_reindexed":
                        motion_max[key] = max(motion_max.get(key, 0.0), float(pos_diff.max()))
    finally:
        for handle in handles.values():
            handle.close()
    matched = [
        name for name in errors
        if errors[name][0]["pos"] <= tolerance and errors[name][0]["vel"] <= tolerance
    ]
    mean_abs = {
        name: {offset: sums[name][offset] / max(rows * 29, 1) for offset in OFFSETS}
        for name in columns_by_name
    }
    ranked = mean_abs[matched[0] if matched else "recorded_reindexed"]
    report = {
        "tolerance": float(tolerance),
        "episodes": len(episodes),
        "rows": rows,
        "motion_step_out_of_range": clipped,
        "max_abs_error": {
            name: {str(offset): value for offset, value in by_offset.items()}
            for name, by_offset in errors.items()
        },
        "joint_pos_mean_abs_error": {
            name: {str(offset): value for offset, value in by_offset.items()}
            for name, by_offset in mean_abs.items()
        },
        "matched_orders": matched,
        "best_offset": int(min(OFFSETS, key=lambda value: ranked[value])),
        "motions_over_tolerance": sum(value > tolerance for value in motion_max.values()),
        "worst_motions_pos": dict(sorted(motion_max.items(), key=lambda kv: -kv[1])[:10]),
    }
    report["pass"] = bool(matched) and report["best_offset"] == 0
    return report


def _export(args) -> Path:
    from omegaconf import OmegaConf

    from .cli import _git_head, _launch_app
    from .runtime.sonic_env import compose_release_config, create_release_env, reference_states_at

    dataset_run = Path(args.physics_dataset).expanduser().resolve()
    sonic_root = Path(args.sonic_root).expanduser().resolve()
    motion_file = Path(args.motion_file).expanduser().resolve()
    if not motion_file.is_dir():
        raise FileNotFoundError(f"motion file directory is missing: {motion_file}")
    dataset_manifest = dataset_run / "manifests" / "dataset_manifest.json"
    schema = json.loads(dataset_manifest.read_text(encoding="utf-8")).get("schema_version")
    if schema != "sonic_physics_state_action_cvae_dataset_v5":
        raise ValueError(f"expected a Physics v5 dataset, found {schema}")
    episodes = read_physics_episodes(dataset_run)
    keys = sorted({str(item["motion_key"]) for item in episodes})
    missing = [key for key in keys if not (motion_file / f"{key}.pkl").is_file()]
    if missing:
        raise FileNotFoundError(f"motion files missing from {motion_file}: {missing}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root).expanduser().resolve() / f"cvae_reference_table_v5_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for child in ("data", "manifests", "markers"):
        (run_dir / child).mkdir()
    print(f"[reference_table] run_dir={run_dir} motions={len(keys)} episodes={len(episodes)}", flush=True)

    config = OmegaConf.create(
        {"dagger": {"num_envs": len(keys)}, "seed": int(args.seed), "device": str(args.device)}
    )
    os.chdir(sonic_root)
    _, args_cli = _launch_app(config)
    release_config = compose_release_config(
        sonic_root,
        num_envs=len(keys),
        seed=int(args.seed),
        motion_file=motion_file,
        motion_keys=keys,
        max_unique_motions=len(keys),
        sample_unique_motions=True,
    )
    release_config.manager_env.config.experiment_dir = str(run_dir)
    release_config.manager_env.config.save_rendering_dir = str(run_dir / "renderings")
    # Collection loads motions with is_evaluation=True; the training load path used
    # here would otherwise apply sonic_release's freeze-frame / upper-body augments.
    motion_cfg = release_config.manager_env.commands.motion
    for flag in ("freeze_frame_aug", "cat_upper_body_poses", "randomize_wrist_poses", "randomize_heading"):
        motion_cfg[flag] = False
    reference_terms = [str(v) for v in release_config.algo.config.actor.backbone.encoders.g1.inputs]
    env = create_release_env(
        release_config, device=str(args.device), args_cli=args_cli, reference_terms=reference_terms
    )
    lib = env.motion_term.motion_lib
    loaded = [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in np.asarray(lib._motion_data_keys)[lib._curr_motion_ids.cpu().numpy()].reshape(-1)
    ]
    if sorted(loaded) != keys or len(set(loaded)) != len(loaded):
        raise RuntimeError(f"motion lib loaded {loaded}, expected each of {keys} once")
    nominal_rows = env.nominal_joint_pos()
    down_rows = env.motion_term.down_dir.to(env.device)
    if not torch.allclose(nominal_rows, nominal_rows[:1].expand_as(nominal_rows)):
        raise RuntimeError("nominal joint positions differ across envs")
    if not torch.allclose(down_rows, down_rows[:1].expand_as(down_rows)):
        raise RuntimeError("down_dir differs across envs")
    joint_ids = joint_index(env.action_joint_ids(), int(lib.dof_pos.shape[-1]))

    local = torch.arange(len(loaded), device=env.device, dtype=torch.long)
    num_steps = lib.get_motion_num_steps(local).long()
    num_frames = lib.get_time_step_total(local).long()
    if bool((num_steps > num_frames).any()):
        raise RuntimeError("motion num_steps exceed stored frames; sim and motion fps differ")
    chunks = []
    for motion in range(len(loaded)):
        count = int(num_steps[motion])
        for start in range(0, count, int(args.chunk)):
            frames = torch.arange(start, min(start + int(args.chunk), count), device=env.device)
            value = reference_states_at(
                env,
                torch.full_like(frames, motion),
                frames[:, None],
                nominal=nominal_rows[:1].expand(frames.numel(), -1),
                down=down_rows[:1].expand(frames.numel(), -1),
            )
            chunks.append(value[:, 0].cpu().numpy())
    order = np.argsort(np.asarray(loaded))
    steps_np = num_steps.cpu().numpy().astype(np.int64)
    per_motion = np.split(np.concatenate(chunks, axis=0), np.cumsum(steps_np)[:-1])
    states = np.concatenate([per_motion[index] for index in order], axis=0).astype(np.float32)
    sorted_steps = steps_np[order]
    table = {
        "states": states,
        "motion_keys": np.asarray([loaded[index] for index in order]),
        "motion_offsets": np.concatenate(([0], np.cumsum(sorted_steps))).astype(np.int64),
        "num_steps": sorted_steps,
        "nominal_joint_pos": nominal_rows[0].cpu().numpy().astype(np.float32),
        "joint_ids": joint_ids,
    }
    table_path = run_dir / "data" / TABLE_NAME
    np.savez(table_path, **table)
    report = alignment_check(episodes, table, tolerance=float(args.tolerance))
    _write_json(run_dir / "manifests" / "alignment_check.json", report)
    print(f"[reference_table] alignment={json.dumps(report, sort_keys=True)}", flush=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "table": f"data/{TABLE_NAME}",
        "table_sha256": _sha256(table_path),
        "state_layout": "joint_pos-nominal[29], joint_vel[29], base_lin_vel_b[3], "
        "base_ang_vel_b[3], gravity_b[3], base_height[1], contact[2]=0",
        "frame_index": "row motion_offsets[m] + motion lib step; clamp steps to num_steps-1",
        "motion_file": str(motion_file),
        "physics_dataset_run": str(dataset_run),
        "physics_dataset_manifest_sha256": _sha256(dataset_manifest),
        "physics_episodes_sha256": _sha256(dataset_run / "manifests" / "episodes.jsonl"),
        "motions": [
            {"motion_key": str(key), "num_steps": int(steps)}
            for key, steps in zip(table["motion_keys"], table["num_steps"])
        ],
        "total_frames": int(states.shape[0]),
        "alignment": report,
        "dagger_git_head": _git_head(Path(__file__).resolve().parents[2]),
    }
    _write_json(run_dir / "manifests" / "reference_table_manifest.json", manifest)
    if not report["pass"]:
        raise RuntimeError(f"reference table alignment check failed: {report}")
    (run_dir / "markers" / "cvae_reference_table.ok").write_text("alignment_pass=true\n", encoding="utf-8")
    print(f"[reference_table] done run_dir={run_dir}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-dataset", required=True)
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--sonic-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    parser.add_argument("--chunk", type=int, default=4096)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    exit_code = 1
    try:
        _export(args)
        exit_code = 0
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
