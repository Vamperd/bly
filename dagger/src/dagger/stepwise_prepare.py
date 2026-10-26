"""Prepare exact-initialization source templates for stepwise causal replay.

The heavy lifting for source State/Action extraction and exact initialization is
already implemented by ``cvae_sa.replay65.prepare_window``.  This module selects
one start-at-zero 192-step group per motion and writes a compact case manifest
for ``dagger.stepwise_recursive_causal``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


def _ensure_cvae_path(cvae_root: Path) -> None:
    source = cvae_root / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"CVAE source directory is missing: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _window_args(args: argparse.Namespace, mask_seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        route="C",
        horizon_steps=int(args.horizon_steps),
        simulation_seed=int(args.simulation_seed),
        mask_seed=mask_seed,
        sample_seed=int(args.sample_seed),
        sample_index=0,
        action_mode="masked-completion",
        state_only_continuation=True,
        source="dataset",
        capture_run=None,
        allow_recovered_dataset=False,
        allow_recovered_identity=False,
        masks=[],
    )


def prepare(args: argparse.Namespace) -> dict:
    cvae_root = Path(args.cvae_root).expanduser().resolve()
    _ensure_cvae_path(cvae_root)
    from cvae_sa.models import build_model
    from cvae_sa.posterior_hierarchical_standard_cvae import load_checkpoint
    from cvae_sa.replay65 import (
        Fixtures,
        checkpoint_config,
        make_dataset,
        prepare_window,
        read_normalization,
    )

    output = Path(args.output_run).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output run must be new and empty: {output}")
    for folder in ("data", "manifests", "markers", "logs"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_run = Path(args.dataset_run).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint_config(checkpoint)
    signature = checkpoint.get("model_signature", {})
    if signature.get("architecture_version") != "65-token-hierarchical-standard-cvae-v1":
        raise ValueError("stepwise prepare requires the 65-token standard CVAE")
    if checkpoint.get("stage") not in {"C", "KL"}:
        raise ValueError("stepwise prepare requires a Stage-C checkpoint")
    training_contract = checkpoint.get("training_contract") or {}
    mask_seed = int(
        args.mask_seed
        if args.mask_seed is not None
        else training_contract.get("training_mask_seed", 20261016)
    )
    model = build_model(config["model"])
    load_checkpoint(model, checkpoint_path)
    model.to(args.device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()

    dataset, indices = make_dataset(dataset_run, config)
    fixtures = Fixtures(dataset, indices, expand=True)
    rows = fixtures.manifest()
    norm = read_normalization(dataset_run / "data/normalization.npz")
    requested = [str(value) for value in args.motion_key]
    if not requested:
        raise ValueError("at least one --motion-key is required")
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate --motion-key values are not allowed")

    window_args = _window_args(args, mask_seed)
    cases = []
    for motion_key in requested:
        row = next(
            (
                item
                for item in rows
                if str(item["motion_key"]) == motion_key
                and int(item["window_start"]) == 0
            ),
            None,
        )
        if row is None:
            raise ValueError(
                f"no start-at-zero window for motion {motion_key}; "
                "stepwise ghost requires frame-0 initialization"
            )
        index = int(row["window_index"])
        episode_steps = int(
            dataset.episodes[dataset.refs[index].episode_index]["steps"]
        )
        if int(args.horizon_steps) > episode_steps:
            raise ValueError(
                f"motion {motion_key} episode has only {episode_steps} steps; "
                f"cannot start at frame 0 for {args.horizon_steps} steps"
            )
        window_args.window_index = index
        child = output / "windows" / motion_key
        prepare_window(
            window_args,
            dataset,
            fixtures,
            model,
            index,
            child,
            norm,
            rows,
            capture=None,
        )
        cases.append(
            {
                "motion_key": motion_key,
                "window_index": index,
                "window_start": 0,
                "episode_ref": str(row.get("episode_ref", "")),
                "template_run": str(child),
                "exact_init": str(child / "data" / "exact_initialization.npz"),
                "source_replay": str(child / "data" / "source.replay.npz"),
                "source_trajectory": str(child / "data" / "source.trajectory.pkl"),
            }
        )
        print(f"[stepwise_prepare] {motion_key} window={index}", flush=True)

    manifest = {
        "format_version": "dagger_stepwise_prepare_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_run": str(dataset_run),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": __import__("hashlib").sha256(
            checkpoint_path.read_bytes()
        ).hexdigest(),
        "mask_seed": mask_seed,
        "sample_seed": int(args.sample_seed),
        "simulation_seed": int(args.simulation_seed),
        "horizon_steps": int(args.horizon_steps),
        "cases": cases,
    }
    _write_json(output / "manifests" / "cases.json", manifest)
    (output / "markers" / "stepwise_prepare.ok").write_text(
        "PASS\n", encoding="utf-8"
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--cvae-root", type=Path, required=True)
    parser.add_argument("--motion-key", action="append", required=True)
    parser.add_argument("--horizon-steps", type=int, default=192)
    parser.add_argument("--mask-seed", type=int)
    parser.add_argument("--sample-seed", type=int, default=20261023)
    parser.add_argument("--simulation-seed", type=int, default=20261023)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    if args.horizon_steps < 64 or args.horizon_steps % 64:
        parser.error("--horizon-steps must be a positive multiple of 64")
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
