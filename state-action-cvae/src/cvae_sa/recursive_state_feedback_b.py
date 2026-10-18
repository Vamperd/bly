"""Recursive state-feedback B inference with blockwise physical replay.

The first T64 call predicts a provisional 64-Action sequence from the
recorded 65 States.  Only its first 32 Actions are executed.  Every later
32-Action block is predicted from the actually executed State history, the
previous generated Actions, and the recorded expected future States, then
executed before the next block is inferred.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from .cvae_protocol import digest
from .long_horizon_eval import CHUNK, HORIZON, BLOCK, LongHorizonEvaluator
from .state_mask_eval import reconstruct_root_trajectory
from .replay65 import (
    _load_replay,
    _write_exact_initialization,
    _write_npz,
    _write_trajectory,
    file_sha256,
    relative_to_raw_action,
    simulate_group,
    verify_worker,
)
from .util import atomic_write_json, atomic_write_text, load_json


VERSION = "sonic_recursive_state_feedback_b_v1"
LABEL = "recursive_state_feedback_b"


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: np.asarray(values[name]).copy() for name in values.files}


def _update_initialization(template: dict[str, np.ndarray], replay_path: Path,
                           init_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    values = {name: np.asarray(value).copy() for name, value in template.items()}
    replay = _load_replay(replay_path)
    raw = _load_arrays(replay_path)
    last = -1
    values.update({
        "physics_state_v3": replay["physics_state_v3"][last],
        "joint_pos": replay["joint_pos"][last],
        "joint_vel": replay["joint_vel"][last],
        "root_pos_relative": replay["root_pos"][last],
        "root_quat_wxyz": replay["root_quat"][last],
        "root_lin_vel_world": replay["root_lin_vel"][last],
        "root_ang_vel_world": replay["root_ang_vel"][last],
        "body_pos_relative": replay["body_pos"][last],
        "previous_raw_action": replay["raw_action"][last],
        "previous_processed_action": raw["processed_action"][last],
        "initial_joint_target_abs": raw["processed_action"][last],
    })
    payload, manifest = _write_exact_initialization(init_path, values)
    manifest["payload_sha256"] = payload
    return values, manifest


class RecursiveStateFeedbackB:
    def __init__(self, dataset_run: Path, checkpoint: Path, template_run: Path,
                 group_id: str, seed: int, device: str | None = None) -> None:
        self.dataset_run = dataset_run.expanduser().resolve()
        self.checkpoint = checkpoint.expanduser().resolve()
        self.template_run = template_run.expanduser().resolve()
        self.group_id = str(group_id)
        self.seed = int(seed)
        self.evaluator = LongHorizonEvaluator(
            self.dataset_run, self.checkpoint, device=device, split="train"
        )
        if self.group_id not in self.evaluator.complete_group_ids:
            raise ValueError("group must be a complete 192-transition group")
        self.rows = self.evaluator._groups[self.group_id]
        self.first_index = int(self.rows[0][0])
        self.episode_index = int(self.rows[0][1]["episode_index"])
        template_root = load_json(self.template_run / "manifests/replay65.json")
        self.source_window = self.template_run / template_root["windows"][0]
        template_meta = load_json(self.source_window / "manifests/replay65.json")
        verify_worker(self.source_window / "simulations" / "baseline")
        template_row = template_meta.get("window", {})
        dataset_row = self.rows[0][1]
        for key in ("episode_ref", "motion_key", "variant_id", "window_start"):
            if key in template_row and str(template_row[key]) != str(dataset_row[key]):
                raise ValueError(
                    f"template/group identity mismatch for {key}: "
                    f"{template_row[key]!r} != {dataset_row[key]!r}"
                )
        self.source = _load_replay(self.source_window / "data/source.replay.npz")
        source_arrays = _load_arrays(self.source_window / "data/source.replay.npz")
        self.truth_action = np.asarray(
            source_arrays["action_target_canonical"], dtype=np.float32
        )
        if self.truth_action.shape != (HORIZON, 29):
            raise ValueError("template source must contain 192 canonical Actions")
        self.truth_state = self.source["physical_state"].astype(np.float32)
        sample = self.evaluator.dataset[self.first_index]
        sample_state = sample["physical_state"].numpy() * self.evaluator.dataset.source.state_std + self.evaluator.dataset.source.state_mean
        sample_action = sample["action"].numpy() * self.evaluator.dataset.source.action_std + self.evaluator.dataset.source.action_mean
        if not np.allclose(self.truth_state[:CHUNK + 1], sample_state, atol=2e-5, rtol=1e-5):
            raise ValueError("template source State does not match the selected long-horizon group")
        if not np.allclose(self.truth_action[:CHUNK], sample_action, atol=2e-5, rtol=1e-5):
            raise ValueError("template source Action does not match the selected long-horizon group")
        self.action_mean = self.evaluator.dataset.source.action_mean
        self.action_std = self.evaluator.dataset.source.action_std
        self.state_mean = self.evaluator.dataset.source.state_mean
        self.state_std = self.evaluator.dataset.source.state_std

    def close(self) -> None:
        self.evaluator.close()

    def initial_actions(self) -> np.ndarray:
        sample = dict(self.evaluator.dataset[self.first_index])
        state_mask = torch.zeros((1, 65, 70), dtype=torch.bool)
        action_mask = torch.ones((1, 64, 29), dtype=torch.bool)
        _, action = self.evaluator._predict(
            sample, state_mask, action_mask, seed=self.seed, sample_index=0
        )
        return action * self.action_std + self.action_mean

    def feedback_actions(self, target_start: int, actual_states: np.ndarray,
                         generated_actions: np.ndarray) -> np.ndarray:
        if target_start < BLOCK or target_start % BLOCK:
            raise ValueError("feedback target must be a positive 32-step boundary")
        if actual_states.shape[0] != target_start + 1:
            raise ValueError("actual State history must end at the feedback boundary")
        context_start = target_start - BLOCK
        context_states = actual_states[context_start:target_start + 1]
        expected_future = self.truth_state[target_start + 1:target_start + BLOCK + 1]
        if context_states.shape != (BLOCK + 1, 70) or expected_future.shape != (BLOCK, 70):
            raise ValueError("feedback State slices do not have the T64 contract")
        physical_states = np.concatenate((context_states, expected_future), axis=0)
        normalized_states = (physical_states - self.state_mean) / self.state_std
        context_actions = generated_actions[context_start:target_start]
        normalized_actions = (context_actions - self.action_mean) / self.action_std
        normalized_actions = np.concatenate(
            (normalized_actions, np.zeros((BLOCK, 29), dtype=np.float32)), axis=0
        ).astype(np.float32)
        sample = dict(self.evaluator.dataset[self.first_index])
        sample["physical_state"] = torch.from_numpy(normalized_states.astype(np.float32))
        sample["action"] = torch.from_numpy(normalized_actions)
        if context_start > 0:
            previous = generated_actions[context_start - 1]
            sample["action_before_window"] = torch.from_numpy(
                ((previous - self.action_mean) / self.action_std).astype(np.float32)
            )
        state_mask = torch.zeros((1, 65, 70), dtype=torch.bool)
        action_mask = torch.zeros((1, 64, 29), dtype=torch.bool)
        action_mask[:, BLOCK:] = True
        _, action = self.evaluator._predict(
            sample, state_mask, action_mask, seed=self.seed,
            sample_index=target_start // BLOCK
        )
        return action[BLOCK:] * self.action_std + self.action_mean


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_run.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output run must be new and empty: {output}")
    engine = RecursiveStateFeedbackB(
        args.dataset_run, args.checkpoint, args.template_run,
        args.group_id, args.seed, args.device,
    )
    try:
        template_init_path = engine.source_window / "data/exact_initialization.npz"
        initial_values = _load_arrays(template_init_path)
        initial_manifest = load_json(
            engine.source_window / "manifests/exact_initialization.json"
        )
        current_values = initial_values
        current_manifest = initial_manifest
        actual_states = None
        generated_parts: list[np.ndarray] = []
        segment_replays: list[dict[str, np.ndarray]] = []
        segment_raw: list[np.ndarray] = []
        segment_dirs: list[Path] = []
        segment_records: list[dict[str, Any]] = []

        provisional = engine.initial_actions()
        for segment_index, target_start in enumerate(range(0, HORIZON, BLOCK)):
            if segment_index == 0:
                canonical = provisional[:BLOCK]
            else:
                canonical = engine.feedback_actions(
                    target_start, actual_states, np.concatenate(generated_parts, axis=0)
                )
            nominal = current_values["nominal_default_joint_pos"]
            clip = current_values["action_clip"]
            clip = clip if clip.size else None
            wrapper = current_values["wrapper_action_clip"]
            wrapper = None if np.isnan(wrapper) else float(wrapper)
            raw, achieved, _ = relative_to_raw_action(
                canonical, nominal, current_values["action_scale"],
                current_values["action_offset"], clip, wrapper,
            )
            if not np.isfinite(raw).all():
                raise ValueError("recursive predicted raw Action contains NaN/Inf")
            segment = output / "segments" / f"segment_{segment_index:02d}"
            for folder in ("data", "manifests", "logs", "markers"):
                (segment / folder).mkdir(parents=True, exist_ok=True)
            init_path = segment / "data/exact_initialization.npz"
            if segment_index == 0:
                _, current_manifest = _write_exact_initialization(init_path, current_values)
            else:
                if not init_path.is_file():
                    raise FileNotFoundError(f"missing chained initialization: {init_path}")
            context = {
                "motion_file": load_json(engine.source_window / "manifests/replay65.json")["source"]["motion_file"],
                "motion_file_sha256": load_json(engine.source_window / "manifests/replay65.json")["source"]["motion_file_sha256"],
                "motion_key": load_json(engine.source_window / "manifests/replay65.json")["window"]["motion_key"],
                "seed": 20261023,
                "init_path": init_path,
                "initialization": current_manifest,
                "steps": BLOCK,
            }
            simulate_group(segment, "recursive_model", ["recursive_model"], [raw], context)
            child = segment / "simulations/recursive_model"
            replay_path = child / "data/replay/000000.replay.npz"
            replay = _load_replay(replay_path)
            segment_replays.append(replay)
            segment_raw.append(raw)
            segment_dirs.append(segment)
            segment_records.append({
                "segment_index": segment_index,
                "target_start": target_start,
                "target_stop": target_start + BLOCK,
                "input_contract": (
                    "full_t64_state_all_action_mask"
                    if segment_index == 0 else
                    "actual_state_33_plus_expected_state_32_generated_action_32"
                ),
                "action_mask_contract": (
                    "all_64_hidden"
                    if segment_index == 0 else
                    "first_32_visible_last_32_hidden"
                ),
                "executed_state_start": target_start,
                "executed_state_stop": target_start + BLOCK,
                "expected_state_start": None if segment_index == 0 else target_start + 1,
                "expected_state_stop": None if segment_index == 0 else target_start + BLOCK + 1,
                "replay": str(replay_path),
            })
            if actual_states is None:
                actual_states = replay["physics_state_v3"].copy()
            else:
                actual_states = np.concatenate((actual_states, replay["physics_state_v3"][1:]), axis=0)
            generated_parts.append(achieved.astype(np.float32))
            if segment_index < HORIZON // BLOCK - 1:
                current_values, current_manifest = _update_initialization(
                    current_values,
                    replay_path,
                    output / "segments" / f"segment_{segment_index + 1:02d}" / "data/exact_initialization.npz",
                )

        generated_action = np.concatenate(generated_parts, axis=0)
        raw_action = np.concatenate(segment_raw, axis=0)
        if actual_states is None or actual_states.shape != (HORIZON + 1, 70):
            raise ValueError("recursive replay did not produce 193 States")

        relative = Path("windows/w000000")
        window = output / relative
        for folder in ("data", "manifests", "simulations", "plots", "videos", "logs", "markers"):
            (window / folder).mkdir(parents=True, exist_ok=True)
        for name in ("source.replay.npz", "source.trajectory.pkl", "exact_initialization.npz"):
            shutil.copy2(engine.source_window / "data" / name, window / "data" / name)
        for name in ("source_schema.json", "exact_initialization.json"):
            shutil.copy2(engine.source_window / "manifests" / name, window / "manifests" / name)
        shutil.copytree(
            engine.source_window / "simulations" / "baseline",
            window / "simulations" / "baseline",
        )
        shutil.copy2(engine.template_run / "data/normalization.npz", output / "data/normalization.npz")
        shutil.copy2(engine.template_run / "manifests/provenance.json", output / "manifests/provenance.json")

        model_child = window / "simulations" / LABEL
        for folder in ("data/replay", "manifests", "logs", "markers"):
            (model_child / folder).mkdir(parents=True, exist_ok=True)
        exact_init = window / "data/exact_initialization.npz"
        shutil.copy2(exact_init, model_child / "data/exact_initialization.npz")
        _write_npz(
            model_child / "data/raw_actions.npz",
            raw_actions=raw_action[:, None],
            scenario_names=np.asarray([LABEL]),
        )
        atomic_write_json(model_child / "manifests/action_replay_request.json", {
            "schema_version": "sonic_action_replay_request_v2",
            "motion_file": load_json(engine.source_window / "manifests/replay65.json")["source"]["motion_file"],
            "motion_file_sha256": load_json(engine.source_window / "manifests/replay65.json")["source"]["motion_file_sha256"],
            "motion_key": load_json(engine.source_window / "manifests/replay65.json")["window"]["motion_key"],
            "raw_actions_file": str(model_child / "data/raw_actions.npz"),
            "steps": HORIZON, "num_envs": 1, "scenario_names": [LABEL],
            "control_dt": 0.02,
        })
        atomic_write_json(model_child / "manifests/action_mask_request.json", {"seed": 20261023})
        template_baseline = engine.source_window / "simulations/baseline"
        shutil.copy2(
            template_baseline / "manifests/exact_initialization_readback_000000.json",
            model_child / "manifests/exact_initialization_readback_000000.json",
        )
        runtime = load_json(template_baseline / "data/replay/000000.runtime.json")
        runtime["control_steps"] = HORIZON
        atomic_write_json(model_child / "data/replay/000000.runtime.json", runtime)

        combined = {}
        for field in ("physics_state_v3", "joint_pos", "joint_vel", "root_pos",
                      "root_quat", "root_lin_vel", "root_ang_vel", "body_pos"):
            combined[field] = np.concatenate(
                [r[field] if i == 0 else r[field][1:] for i, r in enumerate(segment_replays)], axis=0
            )
        combined["raw_action"] = raw_action
        processed = np.concatenate([
            _load_arrays(segment_dirs[i] / "simulations/recursive_model/data/replay/000000.replay.npz")["processed_action"]
            for i in range(len(segment_dirs))
        ], axis=0)
        combined["processed_action"] = processed
        _write_npz(model_child / "data/replay/000000.replay.npz", **combined)
        _write_trajectory(
            model_child / "data/replay/000000.trajectory.pkl",
            joint_pos=combined["joint_pos"], root_pos=combined["root_pos"],
            root_quat=combined["root_quat"], fps=50.0,
        )
        worker_hashes = {
            "data/raw_actions.npz": file_sha256(model_child / "data/raw_actions.npz"),
            "data/exact_initialization.npz": file_sha256(model_child / "data/exact_initialization.npz"),
            "manifests/action_replay_request.json": file_sha256(model_child / "manifests/action_replay_request.json"),
            "manifests/action_mask_request.json": file_sha256(model_child / "manifests/action_mask_request.json"),
            "manifests/exact_initialization_readback_000000.json": file_sha256(model_child / "manifests/exact_initialization_readback_000000.json"),
            "data/replay/000000.replay.npz": file_sha256(model_child / "data/replay/000000.replay.npz"),
            "data/replay/000000.trajectory.pkl": file_sha256(model_child / "data/replay/000000.trajectory.pkl"),
            "data/replay/000000.runtime.json": file_sha256(model_child / "data/replay/000000.runtime.json"),
        }
        atomic_write_json(model_child / "manifests/replay65_worker_complete.json", {"hashes": worker_hashes})

        action_mask = np.zeros((HORIZON, 29), dtype=bool)
        action_mask[:BLOCK] = True
        for start in range(BLOCK, HORIZON, BLOCK):
            action_mask[start:start + BLOCK] = True
        prediction_dir = window / "data" / LABEL
        prediction_dir.mkdir(parents=True, exist_ok=True)
        block_action_masks = np.zeros((HORIZON // BLOCK, CHUNK, 29), dtype=bool)
        block_action_masks[0, :, :] = True
        block_action_masks[1:, BLOCK:, :] = True
        nominal = initial_values["nominal_default_joint_pos"]
        source_root_pos = engine.source["root_pos"]
        source_root_quat = engine.source["root_quat"]
        truth_root_pos, truth_root_quat = reconstruct_root_trajectory(
            engine.truth_state, source_root_pos, source_root_quat, 0, 0.02
        )
        predicted_root_pos, predicted_root_quat = reconstruct_root_trajectory(
            actual_states, source_root_pos, source_root_quat, 0, 0.02
        )
        _write_trajectory(
            prediction_dir / "truth_integrated.trajectory.pkl",
            joint_pos=engine.truth_state[:, :29] + nominal,
            root_pos=truth_root_pos, root_quat=truth_root_quat, fps=50.0,
        )
        _write_trajectory(
            prediction_dir / "predicted_integrated.trajectory.pkl",
            joint_pos=actual_states[:, :29] + nominal,
            root_pos=predicted_root_pos, root_quat=predicted_root_quat, fps=50.0,
        )
        _write_npz(
            prediction_dir / "prediction.npz",
            truth_state=engine.truth_state,
            truth_action=engine.truth_action,
            predicted_state=actual_states,
            predicted_action=generated_action,
            completed_state=actual_states,
            completed_action=generated_action,
            state_mask=np.zeros((HORIZON + 1, 70), dtype=bool),
            action_mask=action_mask,
            block_action_masks=block_action_masks,
            executed_raw=raw_action,
            window_executed_raw=raw_action,
            achieved_action=generated_action,
            valid_state=np.ones((HORIZON + 1, 70), dtype=bool),
            valid_action=np.ones((HORIZON, 29), dtype=bool),
        )
        template_meta = load_json(engine.source_window / "manifests/replay65.json")
        entry = {
            "mask": LABEL, "fixture_id": digest([engine.group_id, LABEL, engine.seed]),
            "offline": {"method": VERSION, "group_id": engine.group_id},
            "physical_groups": {}, "chunks": [], "chunk_boundary_state_max_abs": [],
            "state_top100": [], "action_top100": [], "representative": True,
            "action_model_replay": True,
            "simulation_initial_state": "exact_initialization_with_recursive_feedback",
            "visualization": {"method": VERSION},
        }
        template_meta["route"] = "C"
        template_meta["action_mode"] = VERSION
        template_meta["entries"] = [entry]
        template_meta["recursive_feedback_contract"] = {
            "initial_provisional_horizon": 64,
            "initial_executed_block": 32,
            "feedback_block": 32,
            "actual_state_context": "executed_state_history_through_boundary",
            "expected_state_context": "recorded_future_state_block",
        }
        template_meta["checkpoint"] = str(engine.checkpoint)
        template_meta["dataset_run"] = str(engine.dataset_run)
        atomic_write_json(window / "manifests/replay65.json", template_meta)
        root_meta = deepcopy(load_json(engine.template_run / "manifests/replay65.json"))
        root_meta["windows"] = [str(relative)]
        root_meta["method"] = VERSION
        root_meta["checkpoint"] = str(engine.checkpoint)
        root_meta["dataset_run"] = str(engine.dataset_run)
        root_meta["quality_pass"] = None
        atomic_write_json(output / "manifests/replay65.json", root_meta)
        atomic_write_text(output / "markers/recursive_state_feedback_b.ok", "PASS\n")
        hashes = {
            str(path.relative_to(output)): file_sha256(path)
            for path in output.rglob("*")
            if path.is_file() and path.name not in {"prepared_hashes.json", "progress.json"}
        }
        atomic_write_json(output / "manifests/prepared_hashes.json", hashes)
        atomic_write_json(output / "manifests/recursive_state_feedback_b.json", {
            "version": VERSION, "group_id": engine.group_id,
            "checkpoint": str(engine.checkpoint), "dataset_run": str(engine.dataset_run),
            "segment_count": len(segment_dirs), "horizon_steps": HORIZON,
            "block_steps": BLOCK, "segments": segment_records,
        })
        return {"output_run": str(output), "group_id": engine.group_id,
                "segments": len(segment_dirs), "horizon_steps": HORIZON}
    finally:
        engine.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--template-run", type=Path, required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
