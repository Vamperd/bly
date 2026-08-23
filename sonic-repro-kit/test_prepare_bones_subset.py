from __future__ import annotations

import io
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import prepare_bones_subset


class PrepareBonesSubsetTest(TestCase):
    def _row(self, package: str, index: int, **overrides) -> dict:
        filename = f"{package.lower()}_{index:03d}"
        row = {
            "filename": filename,
            "move_duration_frames": "600",
            "package": package,
            "is_mirror": "False",
            "move_g1_mujoco_path": f"g1/csv/240101/{filename}.csv",
        }
        row.update(overrides)
        return row

    def test_balanced_candidate_selection_is_deterministic(self):
        rows = [
            self._row(package, index)
            for package in prepare_bones_subset.PACKAGES
            for index in range(45)
        ]
        eligible, rejected = prepare_bones_subset.eligible_rows(rows)
        self.assertFalse(rejected)
        first = prepare_bones_subset.choose_candidates(eligible, 20260823, 40)
        second = prepare_bones_subset.choose_candidates(eligible, 20260823, 40)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 320)

    def test_filtering_rejects_mirror_duration_and_sonic_keyword(self):
        rows = [
            self._row("Sport", 0, is_mirror="True"),
            self._row("Sport", 1, move_duration_frames="120"),
            self._row("Sport", 2, filename="chair_turn"),
        ]
        eligible, rejected = prepare_bones_subset.eligible_rows(rows)
        self.assertEqual(eligible["Sport"], [])
        self.assertEqual(rejected["mirror"], 1)
        self.assertEqual(rejected["duration"], 1)
        self.assertEqual(rejected["sonic_keyword"], 1)

    def test_selective_tar_extraction_preserves_session(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "g1.tar.gz"
            wanted = "g1/csv/240101/walk.csv"
            with tarfile.open(archive_path, "w:gz") as archive:
                for name in (wanted, "g1/csv/240101/skip.csv"):
                    payload = name.encode()
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
            output = root / "output"
            output.mkdir()
            extracted = prepare_bones_subset.extract_selected_csvs(
                archive_path,
                [{"move_g1_mujoco_path": wanted}],
                output,
            )
            self.assertEqual(extracted[wanted], output / "240101" / "walk.csv")
            self.assertFalse((output / "240101" / "skip.csv").exists())


if __name__ == "__main__":
    main()
