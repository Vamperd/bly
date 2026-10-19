#!/usr/bin/env python3
"""Create a run-local, unlocked G1+ 31-DoF URDF copy.

The transferred G1+ asset bundle remains read-only.  This command audits the
bundle, verifies that the two neck joints are still revolute, and copies the
original URDF into a fresh run directory for later URDF-to-MJCF conversion.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from prepare_g1_plus_locked29 import (
    AssetError,
    EXPECTED_MOVABLE_COUNT,
    MOTOR_INFO_NAME,
    NECK_JOINTS,
    URDF_NAME,
    _parse_urdf,
    _resolve_mesh,
    audit_asset,
    movable_joint_names,
    sha256,
)


UNLOCKED_NAME = "g1_pl_unlocked31.urdf"
PAIR_NAME = "g1_pl_unlocked31_converter.urdf"
FLAT_NAME = "g1_pl_unlocked31_converter_flat.urdf"


def validate_unlocked_tree(tree: ET.ElementTree) -> dict[str, object]:
    root = tree.getroot()
    movable = movable_joint_names(tree)
    if len(movable) != EXPECTED_MOVABLE_COUNT:
        raise AssetError(
            f"unlocked URDF expected {EXPECTED_MOVABLE_COUNT} movable joints, found {len(movable)}"
        )

    joint_types: dict[str, str] = {}
    for joint in root.findall("joint"):
        name = str(joint.get("name"))
        if name in NECK_JOINTS:
            joint_types[name] = str(joint.get("type"))

    missing = [name for name in NECK_JOINTS if name not in joint_types]
    if missing:
        raise AssetError(f"unlocked URDF is missing neck joints: {missing}")
    invalid = {
        name: kind
        for name, kind in joint_types.items()
        if kind not in {"revolute", "continuous"}
    }
    if invalid:
        raise AssetError(f"neck joints must remain revolute/continuous: {invalid}")

    return {
        "movable_joint_count": len(movable),
        "movable_joint_names": list(movable),
        "neck_joint_names": list(NECK_JOINTS),
        "neck_joint_types": joint_types,
    }


def _ensure_visual_collision_pairs(root: ET.Element) -> list[dict[str, object]]:
    changes: list[dict[str, object]] = []
    for link in root.findall("link"):
        visuals = list(link.findall("visual"))
        collisions = list(link.findall("collision"))
        if visuals and not collisions:
            geometry = copy.deepcopy(visuals[0])
            geometry.tag = "collision"
            link.append(geometry)
            changes.append({"link": link.get("name"), "added_collision": 1})
        elif collisions and not visuals:
            geometry = copy.deepcopy(collisions[0])
            geometry.tag = "visual"
            link.append(geometry)
            changes.append({"link": link.get("name"), "added_visual": 1})
    return changes


def _prepare_converter_urdf(
    source: Path,
    pair_destination: Path,
    flat_destination: Path,
    asset_root: Path,
    mesh_workspace: Path,
) -> dict[str, object]:
    tree = _parse_urdf(source)
    pair_changes = _ensure_visual_collision_pairs(tree.getroot())

    mesh_workspace.mkdir(parents=True, exist_ok=True)
    mesh_map: dict[str, str] = {}
    for mesh in tree.getroot().iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        basename = Path(filename.replace("\\", "/")).name
        if not basename:
            raise AssetError(f"empty mesh basename in {source}")
        mesh_path = _resolve_mesh(asset_root, filename)
        if not mesh_path.is_file():
            raise AssetError(f"mesh reference cannot be resolved: {filename}")
        destination = mesh_workspace / basename
        if destination.exists():
            if not os.path.samefile(mesh_path, destination):
                raise AssetError(f"duplicate mesh basename maps to different files: {basename}")
        else:
            try:
                os.link(mesh_path, destination)
            except OSError:
                shutil.copy2(mesh_path, destination)
        mesh.set("filename", basename)
        mesh_map[filename] = basename

    mujoco = tree.getroot().find("mujoco")
    if mujoco is None:
        mujoco = ET.Element("mujoco")
        tree.getroot().insert(0, mujoco)
    compiler = mujoco.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco, "compiler")
    compiler.set("meshdir", ".")

    pair_destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(pair_destination, encoding="utf-8", xml_declaration=True)
    shutil.copy2(pair_destination, flat_destination)
    return {
        "pair_destination": str(pair_destination),
        "flat_destination": str(flat_destination),
        "mesh_workspace": str(mesh_workspace),
        "unique_mesh_count": len(set(mesh_map.values())),
        "pair_changes": pair_changes,
        "mesh_map": mesh_map,
    }


def prepare(asset_root: Path, run_root: Path, *, source_commit: str | None = None) -> dict[str, object]:
    run_root = run_root.expanduser().resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"run root must be new and empty: {run_root}")

    asset_root = asset_root.expanduser().resolve(strict=True)
    manifest = audit_asset(asset_root, source_commit=source_commit)
    source = asset_root / URDF_NAME
    source_tree = _parse_urdf(source)
    unlocked = validate_unlocked_tree(source_tree)

    assets = run_root / "assets"
    manifests = run_root / "manifests"
    diagnostics = run_root / "diagnostics"
    assets.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)

    destination = assets / UNLOCKED_NAME
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise AssetError("run-local unlocked URDF hash differs from the source URDF")

    converter = _prepare_converter_urdf(
        destination,
        assets / PAIR_NAME,
        assets / FLAT_NAME,
        asset_root,
        assets,
    )

    manifest["unlocked31"] = {
        "source": str(source),
        "destination": str(destination),
        "source_sha256": sha256(source),
        "destination_sha256": sha256(destination),
        **unlocked,
        "source_is_read_only": True,
        "converter": converter,
    }
    manifest_path = manifests / "g1_plus_asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    result = {
        "asset_manifest": str(manifest_path),
        "unlocked_urdf": str(destination),
        "movable_joint_count": unlocked["movable_joint_count"],
        "neck_joint_names": unlocked["neck_joint_names"],
        "neck_joint_types": unlocked["neck_joint_types"],
        "source_sha256": sha256(source),
        "destination_sha256": sha256(destination),
        "converter_urdf": str(assets / PAIR_NAME),
        "converter_flat_urdf": str(assets / FLAT_NAME),
        "motor_info": str(asset_root / MOTOR_INFO_NAME),
    }
    (diagnostics / "unlocked31_prepare.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-commit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = prepare(args.asset_root, args.run_root, source_commit=args.source_commit)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssetError as error:
        print(f"G1+ unlocked31 preparation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
