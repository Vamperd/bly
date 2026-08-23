from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main, mock

import check_collection_gate


class CheckCollectionGateTest(TestCase):
    def _run(self, package_rate: float) -> int:
        with TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.json"
            packages = {
                name: {"completion_rate": package_rate}
                for name in (
                    "Locomotion",
                    "Communication",
                    "Interactions",
                    "Dances",
                    "Gaming",
                    "Everyday",
                    "Sport",
                    "Other",
                )
            }
            summary_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "randomization_profile": "startup",
                        "canonical_episode_count": 1024,
                        "completion_rate": 0.85,
                        "completion_by_package": packages,
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "check_collection_gate.py",
                "--summary",
                str(summary_path),
                "--expected-canonical",
                "1024",
            ]
            with mock.patch.object(sys, "argv", argv):
                return check_collection_gate.main()

    def test_accepts_passing_startup_summary(self):
        self.assertEqual(self._run(0.70), 0)

    def test_rejects_package_below_gate(self):
        self.assertEqual(self._run(0.50), 1)


if __name__ == "__main__":
    main()
