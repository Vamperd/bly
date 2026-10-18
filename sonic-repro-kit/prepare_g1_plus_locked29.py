#!/usr/bin/env python3
"""Audit a transferred G1+ asset bundle and create a run-local locked-neck URDF.

The source asset directory is treated as read-only.  The command writes only to
the requested run directory, so it is safe to use on the Ubuntu execution host
after the Windows asset bundle has been transferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


NECK_JOINTS = ("neck_pitch_joint", "neck_yaw_joint")
EXPECTED_MOVABLE_COUNT = 31
EXPECTED_LOCKED_MOVABLE_COUNT = 29
URDF_NAME = "g1_pl.urdf"
MOTOR_INFO_NAME = "motor_info.md"


class AssetError(ValueError):
    """Raised when a G1+ asset bundle violates the expected contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _repo_file(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _parse_urdf(path: Path) -> ET.ElementTree:
    try:
        return ET.parse(path)
    except (ET.ParseError, OSError) as error:
        raise AssetError(f"failed to parse URDF {path}: {error}") from error


def movable_joint_names(tree: ET.ElementTree) -> tuple[str, ...]:
    names = tuple(
        str(joint.get("name"))
        for joint in tree.getroot().findall("joint")
        if joint.get("type") != "fixed"
    )
    if any(name in {"None", ""} for name in names):
        raise AssetError("URDF contains a movable joint without a name")
    if len(set(names)) != len(names):
        raise AssetError("URDF contains duplicate movable joint names")
    return names


def mesh_references(tree: ET.ElementTree) -> tuple[str, ...]:
    references: set[str] = set()
    for mesh in tree.getroot().iter("mesh"):
        filename = mesh.get("filename")
        if filename:
            references.add(filename)
    return tuple(sorted(references))


def _resolve_mesh(root: Path, reference: str) -> Path:
    normalized = reference.replace("\\", "/")
    if normalized.startswith("package://"):
        normalized = normalized.split("/", 3)[-1]
    if normalized.startswith("file://"):
        normalized = normalized[7:]
    candidate = (root / normalized).resolve()
    if candidate.is_file():
        return candidate
    # G1+ URDFs use meshes/<name>, while the transferred bundle keeps the
    # meshes directory beside g1_pl.urdf.
    candidate = (root / "meshes" / Path(normalized).name).resolve()
    return candidate


def audit_asset(asset_root: Path, *, source_commit: str | None = None) -> dict[str, Any]:
    asset_root = asset_root.expanduser().resolve(strict=True)
    urdf_path = asset_root / URDF_NAME
    motor_info = asset_root / MOTOR_INFO_NAME
    if not urdf_path.is_file() or urdf_path.stat().st_size == 0:
        raise AssetError(f"missing or empty {urdf_path}")
    if not motor_info.is_file() or motor_info.stat().st_size == 0:
        raise AssetError(f"missing or empty {motor_info}")

    tree = _parse_urdf(urdf_path)
    movable = movable_joint_names(tree)
    if len(movable) != EXPECTED_MOVABLE_COUNT:
        raise AssetError(f"expected {EXPECTED_MOVABLE_COUNT} movable joints, found {len(movable)}")
    missing_neck = [name for name in NECK_JOINTS if name not in movable]
    if missing_neck:
        raise AssetError(f"missing G1+ neck joints: {missing_neck}")

    refs = mesh_references(tree)
    missing_meshes = [str(path) for path in (_resolve_mesh(asset_root, ref) for ref in refs) if not path.is_file()]
    if missing_meshes:
        raise AssetError(f"missing mesh files: {missing_meshes}")

    files = []
    for path in sorted(asset_root.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(asset_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    return {
        "version": "g1_plus_asset_manifest_v1",
        "asset_root": str(asset_root),
        "source_commit": source_commit,
        "nested_repo_root": _repo_file(asset_root),
        "nested_repo_commit": _git_commit(Path(_repo_file(asset_root))) if _repo_file(asset_root) else None,
        "urdf": {
            "path": URDF_NAME,
            "sha256": sha256(urdf_path),
            "movable_joint_count": len(movable),
            "movable_joint_names": list(movable),
            "neck_joints": list(NECK_JOINTS),
            "mesh_references": list(refs),
            "missing_mesh_references": [],
        },
        "motor_info": {"path": MOTOR_INFO_NAME, "sha256": sha256(motor_info)},
        "file_count": len(files),
        "files": files,
    }


def lock_neck_tree(tree: ET.ElementTree) -> dict[str, Any]:
    root = tree.getroot()
    changed: list[str] = []
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in NECK_JOINTS:
            continue
        if joint.get("type") == "fixed":
            continue
        if joint.get("type") != "revolute":
            raise AssetError(f"{name} must be revolute before locking, found {joint.get('type')!r}")
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "dynamics"):
            child = joint.find(tag)
            if child is not None:
                joint.remove(child)
        changed.append(str(name))

    if sorted(changed) != sorted(NECK_JOINTS):
        raise AssetError(f"expected to lock exactly {list(NECK_JOINTS)}, changed {changed}")
    locked_names = movable_joint_names(tree)
    if len(locked_names) != EXPECTED_LOCKED_MOVABLE_COUNT:
        raise AssetError(f"locked URDF has {len(locked_names)} movable joints, expected {EXPECTED_LOCKED_MOVABLE_COUNT}")

    return {
        "locked_joints": changed,
        "movable_joint_count": len(locked_names),
        "movable_joint_names": list(locked_names),
    }


def lock_neck_urdf(source: Path, destination: Path) -> dict[str, Any]:
    source = source.expanduser().resolve(strict=True)
    destination = destination.expanduser().resolve()
    tree = _parse_urdf(source)
    locked = lock_neck_tree(tree)

    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return {
        "source": str(source),
        "destination": str(destination),
        **locked,
        "sha256": sha256(destination),
    }


def run(asset_root: Path, run_root: Path, *, source_commit: str | None = None) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    manifest_dir = run_root / "manifests"
    assets_dir = run_root / "assets"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    manifest = audit_asset(asset_root, source_commit=source_commit)
    locked = lock_neck_urdf(Path(manifest["asset_root"]) / URDF_NAME, assets_dir / "g1_pl_locked29.urdf")
    manifest["locked29"] = locked
    manifest["locked29"]["source_asset_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    output = manifest_dir / "g1_plus_asset_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-commit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run(args.asset_root, args.run_root, source_commit=args.source_commit)
    print(json.dumps({
        "asset_manifest": str((args.run_root / "manifests/g1_plus_asset_manifest.json").resolve()),
        "locked_urdf": manifest["locked29"]["destination"],
        "movable_joint_count": manifest["urdf"]["movable_joint_count"],
        "locked_movable_joint_count": manifest["locked29"]["movable_joint_count"],
        "file_count": manifest["file_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssetError as error:
        print(f"G1+ asset audit failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
