from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .action_masks import semantic_joint_groups
from .physics_schema import (
    load_physics_schema,
    read_physics_states,
    structured_robot_information,
)
from .overfit_visualization import write_identifiability_svg
from .util import atomic_write_json, atomic_write_text, file_sha256, load_json


STATE_GROUPS = {
    "joint_position": np.arange(0, 29),
    "joint_velocity": np.arange(29, 58),
    "base_linear_velocity": np.arange(58, 61),
    "base_angular_velocity": np.arange(61, 64),
    "gravity": np.arange(64, 67),
    "base_height": np.arange(67, 68),
    "foot_contact": np.arange(68, 70),
}
TASKS = (
    "forward_rollout", "inverse", "history_action",
    "arbitrary_state", "arbitrary_action",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _percentiles(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"p50": math.inf, "p90": math.inf}
    return {
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
    }


def _load_unique_episodes(
    dataset_run: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    import h5py

    manifest = load_json(dataset_run / "manifests/dataset_manifest.json")
    if manifest.get("memorization_benchmark") is not True:
        raise ValueError("identifiability analysis requires the dedicated memorization dataset")
    episodes_path = dataset_run / "manifests/episodes.jsonl"
    expected_hash = manifest.get("episodes_index_sha256")
    if expected_hash and file_sha256(episodes_path) != expected_hash:
        raise ValueError("episodes index hash does not match the dataset manifest")
    rows = [row for row in _rows(episodes_path) if row.get("split") == "train"]
    if not rows:
        raise ValueError("memorization dataset has no train episodes")
    with np.load(dataset_run / "data/normalization.npz") as values:
        state_mean = values["physical_state_mean"].astype(np.float32)
        state_std = values["physical_state_std"].astype(np.float32)
        action_mean = values["action_mean"].astype(np.float32)
        action_std = values["action_std"].astype(np.float32)
    handles: dict[str, Any] = {}
    schemas: dict[str, dict[str, Any]] = {}
    episodes: list[dict[str, Any]] = []
    joint_robot_values: list[np.ndarray] = []
    global_robot_values: list[np.ndarray] = []
    actuator_values: list[np.ndarray] = []
    vocabulary = manifest["representations"]["joint_actuator_type"]["vocabulary"]
    actuator_type_to_id = {str(name): index for index, name in enumerate(vocabulary)}
    try:
        for episode_index, row in enumerate(rows):
            hdf5_path = str(row["hdf5_path"])
            schema_path = str(row["schema_path"])
            if hdf5_path not in handles:
                handles[hdf5_path] = h5py.File(hdf5_path, "r")
            if schema_path not in schemas:
                schemas[schema_path] = load_physics_schema(Path(schema_path))
            stream = handles[hdf5_path]
            schema = schemas[schema_path]
            group = stream[f"data/{row['episode']}"]
            states = read_physics_states(group["states"])
            actions = np.asarray(
                group["actions/action_target_canonical"], dtype=np.float32
            )
            if states.shape[0] != actions.shape[0] + 1:
                raise ValueError(f"{group.name}: State/Action sequence is not contiguous")
            context = stream[f"contexts/{row['context_id']}"]
            joint, actuator, global_info = structured_robot_information(
                schema, context, int(row["env_id"]), actuator_type_to_id
            )
            joint_robot_values.append(joint)
            actuator_values.append(actuator)
            global_robot_values.append(global_info)
            episodes.append({
                "episode_index": episode_index,
                "episode_ref": f"{row['source_run']}::{row['episode']}",
                "states": ((states - state_mean) / state_std).astype(np.float32),
                "actions": ((actions - action_mean) / action_std).astype(np.float32),
            })
    finally:
        for handle in handles.values():
            handle.close()
    joint_array = np.stack(joint_robot_values)
    global_array = np.stack(global_robot_values)
    actuator_array = np.stack(actuator_values)
    robot_variance = {
        "episode_count": len(episodes),
        "joint_field_mean_variance": {
            name: float(np.var(joint_array[:, :, index], axis=0).mean())
            for index, name in enumerate(
                manifest["representations"]["joint_robot_information"]["fields"]
            )
        },
        "global_field_variance": {
            name: float(np.var(global_array[:, index]))
            for index, name in enumerate(
                manifest["representations"]["global_robot_information"]["fields"]
            )
        },
        "actuator_type_unique_count_by_joint": [
            int(np.unique(actuator_array[:, index]).size) for index in range(29)
        ],
        "interpretation": (
            "Variance is measured across the selected episodes. Zero variance means the "
            "field cannot explain episode-to-episode residuals in this benchmark."
        ),
    }
    return episodes, manifest, robot_variance


def _action_groups(joint_names: Iterable[str]) -> dict[str, np.ndarray]:
    names = [str(name) for name in joint_names]
    try:
        semantic = semantic_joint_groups(names)
    except ValueError:
        return {"all_action_joints": np.arange(29)}
    index = {name: position for position, name in enumerate(names)}
    return {
        name: np.asarray([index[joint] for joint in joints], dtype=np.int64)
        for name, joints in semantic.items()
    }


def _samples_for_task(
    episodes: list[dict[str, Any]], task: str, history_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    episode_ids: list[int] = []
    frame_ids: list[int] = []
    for episode in episodes:
        states, actions = episode["states"], episode["actions"]
        for frame in range(history_steps, actions.shape[0]):
            state_history = states[frame - history_steps + 1 : frame + 1].reshape(-1)
            previous_actions = actions[frame - history_steps : frame].reshape(-1)
            if task == "forward_rollout":
                current_actions = actions[
                    frame - history_steps + 1 : frame + 1
                ].reshape(-1)
                feature = np.concatenate((state_history, current_actions))
                target = states[frame + 1] - states[frame]
            elif task == "inverse":
                feature = np.concatenate((state_history, previous_actions, states[frame + 1]))
                target = actions[frame]
            elif task == "history_action":
                feature = np.concatenate((state_history, previous_actions))
                target = actions[frame]
            else:
                raise ValueError(task)
            features.append(feature.astype(np.float32, copy=False))
            targets.append(target.astype(np.float32, copy=False))
            episode_ids.append(int(episode["episode_index"]))
            frame_ids.append(frame)
    if not features:
        return (
            np.empty((0, 0), np.float32), np.empty((0, 0), np.float32),
            np.empty(0, np.int64), np.empty(0, np.int64),
        )
    return (
        np.stack(features), np.stack(targets),
        np.asarray(episode_ids, dtype=np.int64), np.asarray(frame_ids, dtype=np.int64),
    )


def nearest_target_dispersions(
    episodes: list[dict[str, Any]],
    joint_names: Iterable[str],
    history_steps: Iterable[int],
    max_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Compute a deterministic empirical ambiguity indicator, not a lower bound."""
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    action_groups = _action_groups(joint_names)
    for history in history_steps:
        for task in ("forward_rollout", "inverse", "history_action"):
            feature, target, episode_id, frame_id = _samples_for_task(
                episodes, task, int(history)
            )
            if not len(feature):
                output.append({
                    "task": task,
                    "history_steps": int(history),
                    "sample_count": 0,
                    "target_disagreement_normalized_rmse_p50": math.inf,
                    "target_disagreement_normalized_rmse_p90": math.inf,
                    "status": "insufficient_episode_length",
                })
                continue
            if len(feature) > max_samples:
                selection = np.sort(rng.choice(len(feature), max_samples, replace=False))
                feature = feature[selection]
                target = target[selection]
                episode_id = episode_id[selection]
                frame_id = frame_id[selection]
            mean = feature.mean(axis=0, keepdims=True)
            std = feature.std(axis=0, keepdims=True)
            standardized = (feature - mean) / np.maximum(std, 1e-6)
            projection_dim = min(64, standardized.shape[1])
            projection = rng.normal(
                0.0, 1.0 / math.sqrt(projection_dim),
                size=(standardized.shape[1], projection_dim),
            ).astype(np.float32)
            embedded = standardized @ projection
            embedded = (embedded - embedded.mean(0, keepdims=True)) / np.maximum(
                embedded.std(0, keepdims=True), 1e-6
            )
            nearest = np.full(len(embedded), -1, dtype=np.int64)
            nearest_distance = np.full(len(embedded), np.inf, dtype=np.float32)
            norms = np.square(embedded).sum(axis=1)
            exclusion_radius = int(history) + 8
            for start in range(0, len(embedded), 256):
                stop = min(start + 256, len(embedded))
                distance = (
                    norms[start:stop, None] + norms[None]
                    - 2.0 * embedded[start:stop] @ embedded.T
                )
                same_neighborhood = (
                    episode_id[start:stop, None] == episode_id[None]
                ) & (
                    np.abs(frame_id[start:stop, None] - frame_id[None])
                    <= exclusion_radius
                )
                distance[same_neighborhood] = np.inf
                indices = np.argmin(distance, axis=1)
                values = distance[np.arange(stop - start), indices]
                nearest[start:stop] = indices
                nearest_distance[start:stop] = np.sqrt(np.maximum(values, 0.0))
            valid = np.isfinite(nearest_distance) & (nearest >= 0)
            target_difference = target[valid] - target[nearest[valid]]
            disagreement = np.sqrt(np.mean(np.square(target_difference), axis=1))
            distance_summary = _percentiles(nearest_distance[valid])
            disagreement_summary = _percentiles(disagreement)
            groups = STATE_GROUPS if task == "forward_rollout" else action_groups
            group_disagreement = {
                name: _percentiles(np.sqrt(np.mean(
                    np.square(target_difference[:, indices]), axis=1
                )))
                for name, indices in groups.items()
            }
            output.append({
                "task": task,
                "history_steps": int(history),
                "sample_count": int(valid.sum()),
                "projected_input_distance_p50": distance_summary["p50"],
                "projected_input_distance_p90": distance_summary["p90"],
                "target_disagreement_normalized_rmse_p50": disagreement_summary["p50"],
                "target_disagreement_normalized_rmse_p90": disagreement_summary["p90"],
                "half_pair_disagreement_proxy_p50": disagreement_summary["p50"] / 2.0,
                "semantic_group_disagreement": group_disagreement,
                "status": "empirical_ambiguity_indicator",
            })
    return output


def _model_diagnostics(
    dataset_run: Path,
    checkpoint_path: Path,
    output_run: Path,
    seed: int,
    max_batches: int,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from .dataset import StateActionWindowDataset
    from .losses import compute_loss
    from .masking import MaskGenerator
    from .models import build_model, parameter_count
    from .overfit_diagnostics import evaluate_input_sensitivity
    from .trainer import generate_training_masks

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != "sonic_state_action_cvae_checkpoint_v2":
        raise ValueError("analysis requires a Physics checkpoint v2")
    manifest_hash = file_sha256(dataset_run / "manifests/dataset_manifest.json")
    if checkpoint.get("dataset_manifest_sha256") != manifest_hash:
        raise ValueError("checkpoint and analysis dataset manifest hashes differ")
    config = checkpoint["config"]
    dataset = StateActionWindowDataset(
        dataset_run, "train",
        window_transitions=int(config["data"]["window_transitions"]),
        validation_stride=int(config["data"].get("validation_stride", 64)),
        random_crop=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    masker = MaskGenerator(config["masking"])
    cpu_batch = next(iter(loader))
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in cpu_batch.items()
    }
    stem_prefixes = (
        "joint_id", "actuator_type", "robot_joint", "robot_encoder.layers.0",
        "global_robot", "robot_pool", "state_joint", "action_joint",
        "joint_spatial.layers.0", "state_pool", "action_pool", "state_base",
        "state_fusion", "action_fusion",
    )
    selected = [
        (name, parameter) for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(stem_prefixes)
    ]
    if not selected:
        raise ValueError("no shared-stem parameters selected for gradient diagnostics")
    gradients: dict[str, torch.Tensor] = {}
    loss_values: dict[str, float] = {}
    for task in TASKS:
        model.zero_grad(set_to_none=True)
        masks = generate_training_masks(masker, batch, {
            "task_mode": task,
            "fixed_training_masks": True,
            "fixed_mask_seed": seed,
        }).to(device)
        output = model(batch, masks, sample_from_prior=False, deterministic=True)
        loss = compute_loss(output, batch, masks, config["loss"], kl_beta=0.0)
        loss.total.backward()
        gradients[task] = torch.cat([
            torch.zeros(parameter.numel())
            if parameter.grad is None else parameter.grad.detach().float().cpu().reshape(-1)
            for _, parameter in selected
        ])
        loss_values[task] = float(loss.total.detach().cpu())
    cosines: dict[str, dict[str, float]] = {}
    for task, gradient in gradients.items():
        cosines[task] = {}
        for other, other_gradient in gradients.items():
            denominator = float(gradient.norm() * other_gradient.norm())
            cosines[task][other] = (
                float(torch.dot(gradient, other_gradient) / denominator)
                if denominator > 0.0 else 0.0
            )
    sensitivity = evaluate_input_sensitivity(
        model, loader, masker, device,
        list(dataset.manifest["joint_names"]), output_run,
        max_batches=max_batches,
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "model_kind": config["model"]["kind"],
        "parameter_count": parameter_count(model),
        "gradient_scope": "shared input/robot/joint stem only",
        "gradient_parameter_count": int(sum(parameter.numel() for _, parameter in selected)),
        "task_losses": loss_values,
        "gradient_cosines": cosines,
        "input_sensitivity": sensitivity,
    }
    dataset.close()
    return result


def analyze_overfit_dataset(
    dataset_run: Path,
    output_run: Path,
    checkpoint: Path | None = None,
    history_steps: tuple[int, ...] = (1, 4, 10, 32),
    max_samples: int = 4096,
    max_sensitivity_batches: int = 16,
    seed: int = 20260828,
) -> dict[str, Any]:
    dataset = dataset_run.expanduser().resolve()
    output = output_run.expanduser().resolve()
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "videos"):
        (output / child).mkdir(parents=True, exist_ok=True)
    if max_samples < 2 or any(step < 1 for step in history_steps):
        raise ValueError("analysis sample/history limits must be positive")
    episodes, manifest, robot_variance = _load_unique_episodes(dataset)
    ambiguity = nearest_target_dispersions(
        episodes, manifest["joint_names"], history_steps, max_samples, seed
    )
    model_diagnostics: dict[str, Any] | None = None
    if checkpoint is not None:
        model_diagnostics = _model_diagnostics(
            dataset, checkpoint.expanduser().resolve(), output, seed,
            max_sensitivity_batches,
        )
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_run": str(dataset),
        "dataset_manifest_sha256": file_sha256(
            dataset / "manifests/dataset_manifest.json"
        ),
        "source_read_only": True,
        "uses_unique_transitions_not_overlapping_windows": True,
        "generalization_claim_allowed": False,
        "nearest_neighbor_interpretation": (
            "Empirical ambiguity indicator only, not a mathematical error lower bound. "
            "Self matches and same-episode frames within history+8 steps are excluded."
        ),
        "history_steps": list(history_steps),
        "max_samples_per_task_history": max_samples,
        "robot_information_variance": robot_variance,
        "nearest_neighbor_target_dispersion": ambiguity,
        "model_diagnostics": model_diagnostics,
    }
    atomic_write_json(output / "manifests/overfit_analysis.json", result)
    write_identifiability_svg(
        ambiguity,
        (model_diagnostics or {}).get("gradient_cosines", {}),
        output / "videos/identifiability_heatmap.svg",
    )
    atomic_write_text(output / "markers/cvae_overfit_analysis.ok", "PASS\n")
    return result


def _history_steps(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if not result or min(result) < 1:
        raise argparse.ArgumentTypeError("history steps must be positive comma-separated integers")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze 32-motion empirical identifiability")
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--history-steps", type=_history_steps, default=(1, 4, 10, 32))
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--max-sensitivity-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    result = analyze_overfit_dataset(
        args.dataset_run, args.output_run, args.checkpoint, args.history_steps,
        args.max_samples, args.max_sensitivity_batches, args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
