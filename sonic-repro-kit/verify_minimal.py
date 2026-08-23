#!/usr/bin/env python3
"""Audit authoritative artifacts from the SONIC minimal reproduction phases."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


FATAL_PATTERNS = {
    "traceback": re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    "cuda_oom": re.compile(r"CUDA out of memory|OutOfMemoryError", re.IGNORECASE),
    "missing_file": re.compile(r"FileNotFoundError", re.IGNORECASE),
    "missing_module": re.compile(r"ModuleNotFoundError", re.IGNORECASE),
    "segfault": re.compile(r"segmentation fault", re.IGNORECASE),
    "nan_token": re.compile(r"(?<![A-Za-z0-9_])nan(?![A-Za-z0-9_])", re.IGNORECASE),
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--expected-sonic-commit", required=True)
    parser.add_argument("--expected-isaaclab-commit", required=True)
    args = parser.parse_args()

    state_dir = args.work_root / "state"
    latest_file = state_dir / "latest_run_dir.txt"
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, evidence: str) -> None:
        checks.append({"name": name, "passed": passed, "evidence": evidence})

    if not latest_file.is_file():
        check("latest_run_pointer", False, f"missing: {latest_file}")
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "passed": False,
            "checks": checks,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    run_dir = Path(read_text(latest_file).strip()).expanduser().resolve()
    expected_runs_root = (args.work_root / "GR00T-WholeBodyControl" / "runs").resolve()
    try:
        run_dir.relative_to(expected_runs_root)
        run_path_valid = True
    except ValueError:
        run_path_valid = False
    check("latest_run_pointer", run_path_valid and run_dir.is_dir(), str(run_dir))

    if not run_path_valid or not run_dir.is_dir():
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "passed": False,
            "checks": checks,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    expected_files = {
        "metrics_log": run_dir / "metrics.log",
        "render_log": run_dir / "render.log",
        "smoke_train_log": run_dir / "train_smoke.log",
        "environment_freeze": run_dir / "environment_freeze.txt",
        "gpu_manifest": run_dir / "nvidia-smi.txt",
    }
    for name, path in expected_files.items():
        check(name, path.is_file() and path.stat().st_size > 0, str(path))

    for name in ("eval.ok", "render.ok", "smoke_train.ok"):
        path = run_dir / name
        check(f"success_marker:{name}", path.is_file(), str(path))

    sonic_commit_file = run_dir / "sonic_commit.txt"
    sonic_actual = read_text(sonic_commit_file).strip() if sonic_commit_file.is_file() else "missing"
    check(
        "sonic_commit",
        sonic_actual == args.expected_sonic_commit,
        f"expected={args.expected_sonic_commit}, actual={sonic_actual}",
    )

    isaaclab_commit_file = run_dir / "isaaclab_commit.txt"
    isaaclab_actual = (
        read_text(isaaclab_commit_file).strip() if isaaclab_commit_file.is_file() else "missing"
    )
    check(
        "isaaclab_commit",
        isaaclab_actual == args.expected_isaaclab_commit,
        f"expected={args.expected_isaaclab_commit}, actual={isaaclab_actual}",
    )

    freeze_file = expected_files["environment_freeze"]
    freeze = read_text(freeze_file).lower() if freeze_file.is_file() else ""
    check("dependency:torch", "torch==2.7.0" in freeze, "torch==2.7.0")
    check("dependency:isaacsim", "isaacsim==5.1.0" in freeze, "isaacsim==5.1.0")

    videos = [path for path in (run_dir / "renders").glob("*.mp4") if path.stat().st_size > 0]
    check("rendered_mp4", bool(videos), ", ".join(str(path) for path in videos) or "none")

    for log_name in ("metrics.log", "render.log", "train_smoke.log"):
        path = run_dir / log_name
        if not path.is_file():
            continue
        text = read_text(path)
        for pattern_name, pattern in FATAL_PATTERNS.items():
            match = pattern.search(text)
            check(
                f"fatal_scan:{log_name}:{pattern_name}",
                match is None,
                "not found" if match is None else match.group(0),
            )

    passed = all(bool(item["passed"]) for item in checks)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "passed": passed,
        "checks": checks,
    }
    report_path = run_dir / "verification_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"SONIC minimal reproduction: {'PASS' if passed else 'FAIL'}")
    for item in checks:
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['name']}: {item['evidence']}")
    print(f"Report: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
