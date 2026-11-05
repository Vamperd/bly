"""Explicit read-time relocation without rewriting dataset bytes or identities."""
from __future__ import annotations

import json
import os
from pathlib import Path


def dataset_path_mapping() -> dict[str, str]:
    raw = os.environ.get("CVAE_DATASET_PATH_MAP", "")
    if not raw:
        return {}
    mapping = json.loads(raw)
    if not isinstance(mapping, dict):
        raise ValueError("CVAE_DATASET_PATH_MAP must be a JSON object of absolute path prefixes")
    for source, destination in mapping.items():
        if not isinstance(source, str) or not isinstance(destination, str):
            raise ValueError("dataset path prefixes must be strings")
        if not Path(source).is_absolute() or not Path(destination).is_absolute():
            raise ValueError("dataset path prefixes must be absolute")
        if ".." in Path(source).parts or ".." in Path(destination).parts:
            raise ValueError("dataset path prefixes must not contain '..'")
    return mapping


def resolve_dataset_path(value: str | Path) -> Path:
    """Map the longest matching path prefix once; never change logical IDs."""
    path = Path(value).expanduser()
    mapping = dataset_path_mapping()
    for source, destination in sorted(mapping.items(), key=lambda item: len(Path(item[0]).parts), reverse=True):
        try:
            relative = path.relative_to(Path(source))
        except ValueError:
            continue
        return Path(destination) / relative
    return path
