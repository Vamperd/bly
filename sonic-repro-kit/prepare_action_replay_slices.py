#!/usr/bin/env python3
"""Split a batched external Action replay into deterministic one-env inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ACTION_DIM = 29


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_replay(source: Path, output_dir: Path, expected_num_envs: int) -> dict:
    source = source.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"batched Action replay is missing: {source}")
    if expected_num_envs <= 0:
        raise ValueError("expected_num_envs must be positive")
    with np.load(source, allow_pickle=False) as values:
        if "raw_actions" not in values:
            raise ValueError("batched Action replay does not contain raw_actions")
        raw_actions = np.asarray(values["raw_actions"], dtype=np.float32)
    expected_tail = (expected_num_envs, ACTION_DIM)
    if raw_actions.ndim != 3 or tuple(raw_actions.shape[1:]) != expected_tail:
        raise ValueError(
            f"raw_actions must have shape [T,{expected_num_envs},{ACTION_DIM}], "
            f"found {raw_actions.shape}"
        )
    if raw_actions.shape[0] == 0 or not np.isfinite(raw_actions).all():
        raise ValueError("raw_actions must be non-empty and finite")

    output_dir.mkdir(parents=True, exist_ok=True)
    slices = []
    for env_id in range(expected_num_envs):
        target = output_dir / f"{env_id:06d}.actions.npz"
        temporary = output_dir / f".{env_id:06d}.actions.tmp.npz"
        np.savez_compressed(temporary, raw_actions=raw_actions[:, env_id : env_id + 1])
        os.replace(temporary, target)
        slices.append(
            {
                "environment_id": env_id,
                "path": str(target),
                "sha256": _sha256(target),
                "shape": [int(raw_actions.shape[0]), 1, ACTION_DIM],
            }
        )

    manifest = {
        "schema_version": "sonic_serial_action_replay_slices_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": _sha256(source),
        "source_shape": list(raw_actions.shape),
        "execution_mode": "serial_single_environment",
        "slices": slices,
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-num-envs", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = split_replay(args.source, args.output_dir, args.expected_num_envs)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_name(f".{args.manifest.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
