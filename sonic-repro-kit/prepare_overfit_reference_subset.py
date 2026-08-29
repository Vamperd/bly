from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare_reference_motion_subset(
    selection_run: Path, ingest_run: Path, output_run: Path
) -> dict[str, Any]:
    selection = selection_run.expanduser().resolve()
    ingest = ingest_run.expanduser().resolve()
    output = output_run.expanduser().resolve()
    for required in (
        selection / "markers/cvae_overfit_subset.ok",
        selection / "manifests/overfit_selection.json",
        ingest / "markers/prepare_bones_subset.ok",
        ingest / "manifests/motion_manifest.jsonl",
        ingest / "data/robot_filtered",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    selection_manifest = _json(selection / "manifests/overfit_selection.json")
    selected_keys = [str(value) for value in selection_manifest["selected_motion_keys"]]
    if len(selected_keys) != 32 or len(set(selected_keys)) != 32:
        raise ValueError("reference collection requires exactly the selected 32 unique motions")
    source_manifest_path = ingest / "manifests/motion_manifest.jsonl"
    source_rows = {str(row["motion_key"]): row for row in _jsonl(source_manifest_path)}
    missing = sorted(set(selected_keys) - set(source_rows))
    if missing:
        raise ValueError(f"selected motions are absent from the ingest manifest: {missing}")
    package_counts = Counter(str(source_rows[key].get("package", "")) for key in selected_keys)
    if len(package_counts) != 8 or set(package_counts.values()) != {4}:
        raise ValueError(
            f"selected reference motions must contain eight packages x four: {package_counts}"
        )
    for child in ("data/robot_filtered", "manifests", "markers", "logs", "videos", "checkpoints"):
        (output / child).mkdir(parents=True, exist_ok=True)
    if any((output / "data/robot_filtered").iterdir()):
        raise FileExistsError("reference motion subset destination must be empty")
    output_rows = []
    for global_motion_id, motion_key in enumerate(sorted(selected_keys)):
        source_row = source_rows[motion_key]
        source = Path(str(source_row.get("final_pkl", ""))).expanduser()
        if not source.is_absolute() or not source.is_file():
            source = ingest / "data/robot_filtered" / f"{motion_key}.pkl"
        source = source.resolve()
        if not source.is_relative_to(ingest) or not source.is_file():
            raise ValueError(f"motion source escapes or is missing from ingest run: {source}")
        source_hash = _sha256(source)
        declared_hash = source_row.get("final_sha256")
        if declared_hash and source_hash != declared_hash:
            raise ValueError(f"source motion hash changed: {motion_key}")
        destination = output / "data/robot_filtered" / f"{motion_key}.pkl"
        destination.symlink_to(source)
        output_rows.append({
            **source_row,
            "global_motion_id": global_motion_id,
            "motion_key": motion_key,
            "final_pkl": str(destination),
            "final_sha256": source_hash,
            "read_only_symlink_source": str(source),
        })
    manifest_path = output / "manifests/motion_manifest.jsonl"
    _atomic_text(
        manifest_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
    )
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "same_32_motion_reference_aware_collection_input",
        "motion_count": 32,
        "source_read_only": True,
        "materialization": "absolute_symlink",
        "selection_run": str(selection),
        "selection_manifest_sha256": _sha256(
            selection / "manifests/overfit_selection.json"
        ),
        "ingest_run": str(ingest),
        "source_motion_manifest_sha256": _sha256(source_manifest_path),
        "motion_manifest": str(manifest_path),
        "motion_manifest_sha256": _sha256(manifest_path),
        "package_counts": dict(sorted(package_counts.items())),
        "selected_motion_keys": sorted(selected_keys),
    }
    _atomic_json(output / "manifests/reference_motion_subset.json", result)
    _atomic_text(output / "markers/prepare_overfit_reference_subset.ok", "PASS\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare read-only links for the exact selected 32 motions"
    )
    parser.add_argument("--selection-run", type=Path, required=True)
    parser.add_argument("--ingest-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_reference_motion_subset(
        args.selection_run, args.ingest_run, args.output_run
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
