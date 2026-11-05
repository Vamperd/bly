import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from cvae_sa.path_mapping import dataset_path_mapping, resolve_dataset_path


class DatasetPathMappingTests(unittest.TestCase):
    root = Path.cwd().anchor
    old = Path(root) / "old" / "runs"
    new = Path(root) / "server" / "migration"

    def mapping(self, values):
        return patch.dict(os.environ, {"CVAE_DATASET_PATH_MAP": json.dumps(values)})

    def test_no_mapping_preserves_path(self):
        with patch.dict(os.environ, {"CVAE_DATASET_PATH_MAP": ""}):
            self.assertEqual(resolve_dataset_path(self.old / "a"), self.old / "a")

    def test_file_and_root_mapping(self):
        with self.mapping({str(self.old): str(self.new)}):
            self.assertEqual(resolve_dataset_path(self.old), self.new)
            self.assertEqual(resolve_dataset_path(self.old / "a/data.hdf5"), self.new / "a/data.hdf5")

    def test_path_component_boundary(self):
        with self.mapping({str(self.old): str(self.new)}):
            path = self.old.parent / "runs_other/a"
            self.assertEqual(resolve_dataset_path(path), path)

    def test_longest_match_and_single_application(self):
        with self.mapping({str(self.old): str(self.new), str(self.old / "a"): str(self.new / "special"),
                           str(self.new): str(self.old)}):
            self.assertEqual(resolve_dataset_path(self.old / "a/file"), self.new / "special/file")

    def test_identity_strings_are_not_rewritten(self):
        logical = f"{self.old}/collection::demo_1"
        with self.mapping({str(self.old): str(self.new)}):
            # Callers only map IO paths; episode identities and manifests remain verbatim.
            row = {"episode_ref": logical, "source_run": str(self.old / "collection")}
            before = json.dumps(row)
            resolve_dataset_path(self.old / "collection/data.hdf5")
            self.assertEqual(json.dumps(row), before)

    def test_invalid_maps_fail_closed(self):
        for mapping in ([], {"relative": str(self.new)}, {str(self.old): 5},
                        {str(self.old): str(self.new / "../other")}):
            with self.subTest(mapping=mapping), self.mapping(mapping):
                with self.assertRaises(ValueError):
                    dataset_path_mapping()


if __name__ == "__main__":
    unittest.main()
