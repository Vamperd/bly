#!/usr/bin/env python3
"""Enforce the startup-collection quality gate before mild reset perturbations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--overall-min", type=float, default=0.80)
    parser.add_argument("--package-min", type=float, default=0.60)
    parser.add_argument("--expected-canonical", type=int)
    parser.add_argument("--expected-package-count", type=int, default=8)
    parser.add_argument("--expected-motion-manifest", type=Path)
    args = parser.parse_args()

    summary_path = args.summary.expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if summary.get("passed") is not True:
        errors.append("baseline verifier did not pass")
    if summary.get("randomization_profile") != "startup":
        errors.append("baseline randomization profile is not startup")
    if args.expected_canonical is not None:
        actual = int(summary.get("canonical_episode_count", -1))
        if actual != args.expected_canonical:
            errors.append(
                f"canonical episode count is {actual}, expected {args.expected_canonical}"
            )
    if args.expected_motion_manifest is not None:
        digest = hashlib.sha256(
            args.expected_motion_manifest.expanduser().resolve().read_bytes()
        ).hexdigest()
        if summary.get("motion_manifest_sha256") != digest:
            errors.append("baseline motion manifest does not match the requested motion set")

    overall = float(summary.get("completion_rate", -1.0))
    if overall < args.overall_min:
        errors.append(f"overall completion rate {overall:.3f} < {args.overall_min:.3f}")
    package_rows = summary.get("completion_by_package", {})
    if len(package_rows) != args.expected_package_count:
        errors.append(
            f"package count is {len(package_rows)}, expected {args.expected_package_count}"
        )
    for package, row in sorted(package_rows.items()):
        rate = float(row.get("completion_rate", -1.0))
        if rate < args.package_min:
            errors.append(
                f"package {package!r} completion rate {rate:.3f} < {args.package_min:.3f}"
            )

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print(
        f"[PASS] startup gate: overall={overall:.3f}, "
        f"packages={len(package_rows)}, source={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
