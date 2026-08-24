from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cvae_sa.dataset import StateActionWindowDataset
from cvae_sa.indexer import build_index
from tests.fixtures import write_collection_run


class IndexerDatasetTest(unittest.TestCase):
    def test_builds_leak_free_index_and_padded_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = write_collection_run(
                root / "sources", "motion_a", "Locomotion", (0, 1), per_environment=True
            )
            source_b = write_collection_run(
                root / "sources", "motion_b", "Dances", (0, 1)
            )
            output = root / "cvae_dataset"
            manifest = build_index(
                (source_a, source_b),
                output,
                expected_motions=2,
                expected_episodes=4,
                split_counts=(1, 1, 0),
                seed=7,
            )
            self.assertEqual(manifest["motion_count"], 2)
            self.assertEqual(manifest["canonical_episode_count"], 4)
            split = json.loads(
                (output / "manifests" / "split_motion_keys.json").read_text()
            )
            self.assertFalse(set(split["train"]) & set(split["validation"]))
            dataset = StateActionWindowDataset(output, "train", window_transitions=8)
            value = dataset[0]
            self.assertEqual(tuple(value["physical_state"].shape), (9, 64))
            self.assertEqual(tuple(value["action"].shape), (8, 29))
            self.assertEqual(int(value["valid_action"].sum()), 6)
            self.assertTrue(np.isfinite(value["physical_state"].numpy()).all())
            dataset.close()


if __name__ == "__main__":
    unittest.main()

