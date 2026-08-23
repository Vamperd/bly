#!/usr/bin/env python3
"""Prepare a deterministic, package-balanced BONES-SEED G1 subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


PACKAGES = (
    "Locomotion",
    "Communication",
    "Interactions",
    "Dances",
    "Gaming",
    "Everyday",
    "Sport",
    "Other",
)
FILTER_KEYWORDS = (
    "bed",
    "bike",
    "chair",
    "climb",
    "com_up_50cm",
    "sitting",
    "step_on",
    "seat",
    "table",
    "_sit_",
    "sit_",
    "ladder",
    "crutch",
    "_bed_",
    "_ride_",
    "scooter",
    "stepdown",
    "acrobatics_",
    "box_hspu",
    "cartwheel",
    "50cm_box_",
    "on_box",
    "fall_from",
    "handstand_ff_",
    "on_1m",
    "form_box",
    "off_1m",
    "230m",
    "jump_over_obstacle_",
    "lift_crate_come_up_",
    "jump_to_shoulder_roll",
    "kozak_dance",
    "stair",
    "handstand",
    "box_jump",
    "monkey_jump",
    "safety_roll",
    "box_dips",
    "walking_on_edge",
    "push_obstacle",
)
G1_PATH_COLUMNS = (
    "move_g1_path",
    "move_g1_mujoco_path",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def free_gib(path: Path) -> int:
    return shutil.disk_usage(path).free // (1024**3)


def require_free(path: Path, minimum: int, stage: str) -> int:
    available = free_gib(path)
    if available < minimum:
        raise RuntimeError(
            f"{stage} requires at least {minimum} GiB free, found {available} GiB at {path}"
        )
    return available


def ensure_new_output_tree(run_dir: Path) -> None:
    for relative in (
        "data/extracted",
        "data/converted",
        "data/robot_filtered_candidates",
        "data/robot_filtered",
    ):
        path = run_dir / relative
        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            raise RuntimeError(f"Output directory is not empty; use a new ingest run: {path}")
    for relative in ("logs", "manifests", "markers"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)


def load_metadata(source_dir: Path) -> tuple[list[dict[str, object]], Path]:
    csv_path = source_dir / "metadata" / "seed_metadata_v004.csv"
    parquet_path = source_dir / "metadata" / "seed_metadata_v004.parquet"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream)), csv_path
    if parquet_path.is_file():
        try:
            import pandas as pd

            frame = pd.read_parquet(parquet_path)
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "Parquet support is unavailable in the pinned environment. Download "
                "metadata/seed_metadata_v004.csv; do not change dependencies."
            ) from error
        return frame.fillna("").to_dict(orient="records"), parquet_path
    raise FileNotFoundError(
        f"Expected seed_metadata_v004.csv or .parquet under {source_dir / 'metadata'}"
    )


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalized_member_name(value: str) -> str:
    parts = [part for part in PurePosixPath(value).parts if part not in ("", ".")]
    if ".." in parts:
        raise ValueError(f"unsafe archive path: {value}")
    return "/".join(parts)


def resolve_g1_path_column(row: dict[str, object]) -> str:
    for column in G1_PATH_COLUMNS:
        if column in row:
            return column
    expected = ", ".join(G1_PATH_COLUMNS)
    raise ValueError(
        f"metadata is missing a G1 path column; expected one of: {expected}"
    )


def eligible_rows(rows: list[dict[str, object]]) -> tuple[dict[str, list[dict]], Counter]:
    by_package = {package: [] for package in PACKAGES}
    rejection_counts: Counter = Counter()
    required = {
        "filename",
        "move_duration_frames",
        "package",
        "is_mirror",
    }
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0] if rows else {}))
        raise ValueError(f"metadata is missing required columns: {missing}")
    g1_path_column = resolve_g1_path_column(rows[0])
    for row in rows:
        package = str(row["package"]).strip()
        if package not in by_package:
            rejection_counts["unknown_package"] += 1
            continue
        filename = str(row["filename"]).strip()
        source_path = normalized_member_name(str(row[g1_path_column]))
        try:
            duration_frames = int(float(row["move_duration_frames"]))
        except (TypeError, ValueError):
            rejection_counts["invalid_duration"] += 1
            continue
        if parse_bool(row["is_mirror"]) or filename.endswith("_M"):
            rejection_counts["mirror"] += 1
            continue
        if duration_frames < 240 or duration_frames > 2400:
            rejection_counts["duration"] += 1
            continue
        searchable = f"{source_path}/{filename}".lower()
        if any(keyword in searchable for keyword in FILTER_KEYWORDS):
            rejection_counts["sonic_keyword"] += 1
            continue
        if not source_path.endswith(".csv"):
            rejection_counts["not_g1_csv"] += 1
            continue
        candidate = dict(row)
        candidate["filename"] = filename
        candidate["package"] = package
        candidate["move_duration_frames"] = duration_frames
        candidate["move_g1_mujoco_path"] = source_path
        by_package[package].append(candidate)
    return by_package, rejection_counts


def choose_candidates(
    by_package: dict[str, list[dict]], seed: int, count: int
) -> list[dict]:
    selected: list[dict] = []
    shortages = {
        package: len(rows)
        for package, rows in by_package.items()
        if len(rows) < count
    }
    if shortages:
        raise RuntimeError(f"packages below {count} eligible candidates: {shortages}")
    for package_index, package in enumerate(PACKAGES):
        rng = random.Random(seed + package_index)
        candidates = sorted(by_package[package], key=lambda row: str(row["filename"]))
        for rank, row in enumerate(rng.sample(candidates, count)):
            item = dict(row)
            item["candidate_rank"] = rank
            selected.append(item)
    return selected


def extract_selected_csvs(
    archive_path: Path,
    candidates: list[dict],
    extracted_dir: Path,
) -> dict[str, Path]:
    selected_paths = {
        normalized_member_name(str(row["move_g1_mujoco_path"])): row
        for row in candidates
    }
    extracted: dict[str, Path] = {}
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            member_name = normalized_member_name(member.name)
            if member_name not in selected_paths:
                continue
            if not member.isfile():
                raise RuntimeError(f"selected archive member is not a file: {member_name}")
            parts = PurePosixPath(member_name).parts
            try:
                csv_index = parts.index("csv")
            except ValueError as error:
                raise RuntimeError(f"unexpected G1 archive path: {member_name}") from error
            relative_parts = parts[csv_index + 1 :]
            if len(relative_parts) < 2:
                raise RuntimeError(f"G1 CSV path lacks a session directory: {member_name}")
            destination = extracted_dir.joinpath(*relative_parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read archive member: {member_name}")
            with source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            extracted[member_name] = destination
            if len(extracted) == len(selected_paths):
                break
    missing = sorted(set(selected_paths) - set(extracted))
    if missing:
        raise RuntimeError(f"archive did not contain {len(missing)} selected CSVs: {missing[:5]}")
    return extracted


def run_checked(command: list[str], log_path: Path, cwd: Path) -> None:
    with log_path.open("x", encoding="utf-8") as log_stream:
        process = subprocess.run(
            command,
            cwd=cwd,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"command failed with exit {process.returncode}; see {log_path}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-run", type=Path, required=True)
    parser.add_argument("--sonic-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--candidate-per-package", type=int, default=40)
    parser.add_argument("--final-per-package", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-free-gib", type=int, default=40)
    args = parser.parse_args()

    run_dir = args.ingest_run.expanduser().resolve()
    sonic_dir = args.sonic_dir.expanduser().resolve()
    allowed_root = (Path.home() / "bly" / "runs").resolve()
    if run_dir == allowed_root or allowed_root not in run_dir.parents:
        raise RuntimeError(f"ingest run must be a child of {allowed_root}; found {run_dir}")
    source_dir = run_dir / "data" / "source"
    archive_path = source_dir / "g1.tar.gz"
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing BONES G1 archive: {archive_path}")
    report_path = run_dir / "manifests" / "bones_subset_report.json"
    if report_path.exists():
        raise RuntimeError(f"Ingest run already has a report; create a new run: {report_path}")
    ensure_new_output_tree(run_dir)
    start_free = require_free(run_dir, args.min_free_gib, "prepare start")
    rows, metadata_path = load_metadata(source_dir)
    metadata_g1_path_column = resolve_g1_path_column(rows[0] if rows else {})
    by_package, rejection_counts = eligible_rows(rows)
    eligibility_counts = {package: len(by_package[package]) for package in PACKAGES}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "candidate_per_package": args.candidate_per_package,
        "final_per_package": args.final_per_package,
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "metadata_g1_path_column": metadata_g1_path_column,
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "eligibility_counts": eligibility_counts,
        "rejection_counts": dict(rejection_counts),
        "free_gib_start": start_free,
        "passed": False,
    }
    try:
        candidates = choose_candidates(by_package, args.seed, args.candidate_per_package)
        extracted = extract_selected_csvs(
            archive_path,
            candidates,
            run_dir / "data" / "extracted",
        )
        require_free(run_dir, args.min_free_gib, "selective extraction")
        candidate_rows = []
        for row in candidates:
            item = dict(row)
            csv_path = extracted[str(row["move_g1_mujoco_path"])]
            item["extracted_csv"] = str(csv_path)
            item["source_sha256"] = sha256(csv_path)
            candidate_rows.append(item)
        write_jsonl(run_dir / "manifests" / "candidate_manifest.jsonl", candidate_rows)

        converter = sonic_dir / "gear_sonic" / "data_process" / "convert_soma_csv_to_motion_lib.py"
        filter_script = sonic_dir / "gear_sonic" / "data_process" / "filter_and_copy_bones_data.py"
        run_checked(
            [
                sys.executable,
                str(converter),
                "--input",
                str(run_dir / "data" / "extracted"),
                "--output",
                str(run_dir / "data" / "converted"),
                "--fps",
                "30",
                "--fps_source",
                "120",
                "--individual",
                "--num_workers",
                str(args.workers),
            ],
            run_dir / "logs" / "convert_bones_subset.log",
            sonic_dir,
        )
        require_free(run_dir, args.min_free_gib, "conversion")
        run_checked(
            [
                sys.executable,
                str(filter_script),
                "--source",
                str(run_dir / "data" / "converted"),
                "--dest",
                str(run_dir / "data" / "robot_filtered_candidates"),
                "--workers",
                str(args.workers),
            ],
            run_dir / "logs" / "filter_bones_subset.log",
            sonic_dir,
        )

        available_candidates: dict[str, list[dict]] = {package: [] for package in PACKAGES}
        for row in candidate_rows:
            matches = list(
                (run_dir / "data" / "robot_filtered_candidates").glob(
                    f"*/{row['filename']}.pkl"
                )
            )
            if len(matches) == 1:
                item = dict(row)
                item["converted_pkl"] = str(matches[0])
                available_candidates[str(row["package"])].append(item)
        insufficient = {
            package: len(items)
            for package, items in available_candidates.items()
            if len(items) < args.final_per_package
        }
        if insufficient:
            raise RuntimeError(
                f"packages below {args.final_per_package} converted/filtered motions: {insufficient}"
            )

        final_candidates = []
        for package in PACKAGES:
            ordered = sorted(
                available_candidates[package], key=lambda row: int(row["candidate_rank"])
            )
            final_candidates.extend(ordered[: args.final_per_package])
        filenames = [str(row["filename"]) for row in final_candidates]
        if len(set(filenames)) != len(filenames):
            raise RuntimeError("selected motion filenames are not globally unique")

        final_rows = []
        final_dir = run_dir / "data" / "robot_filtered"
        for global_motion_id, row in enumerate(
            sorted(final_candidates, key=lambda item: str(item["filename"]))
        ):
            converted_path = Path(str(row["converted_pkl"]))
            destination = final_dir / f"{row['filename']}.pkl"
            shutil.copy2(converted_path, destination)
            item = dict(row)
            item.update(
                {
                    "global_motion_id": global_motion_id,
                    "motion_key": str(row["filename"]),
                    "converted_sha256": sha256(converted_path),
                    "final_pkl": str(destination),
                    "final_sha256": sha256(destination),
                }
            )
            final_rows.append(item)
        package_counts = Counter(str(row["package"]) for row in final_rows)
        expected_total = len(PACKAGES) * args.final_per_package
        if len(final_rows) != expected_total or any(
            package_counts[package] != args.final_per_package for package in PACKAGES
        ):
            raise RuntimeError(f"final balance check failed: {dict(package_counts)}")
        write_jsonl(run_dir / "manifests" / "motion_manifest.jsonl", final_rows)
        attribution = {
            "dataset": "BONES-SEED: Skeletal Everyday Embodiment Dataset",
            "repository": "https://huggingface.co/datasets/bones-studio/seed",
            "license": "https://huggingface.co/datasets/bones-studio/seed/blob/main/LICENSE.md",
            "notice": (
                "BONES-SEED and these derived files remain subject to the BONES-SEED "
                "license. Access permission is granted by Bones Studio, not by this tool."
            ),
        }
        (run_dir / "manifests" / "license_attribution.json").write_text(
            json.dumps(attribution, indent=2) + "\n", encoding="utf-8"
        )
        report.update(
            {
                "passed": True,
                "candidate_count": len(candidate_rows),
                "final_count": len(final_rows),
                "package_counts": dict(package_counts),
                "free_gib_end": require_free(run_dir, args.min_free_gib, "finalization"),
                "motion_manifest": str(run_dir / "manifests" / "motion_manifest.jsonl"),
                "motion_directory": str(final_dir),
            }
        )
    except Exception as error:
        report["error"] = str(error)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
