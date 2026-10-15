import json
import tempfile
import unittest
from pathlib import Path

try:
    import h5py  # noqa: F401
except ImportError:  # Windows static environments intentionally lack HDF5.
    h5py = None

if h5py is not None:
    from cvae_sa.long_horizon_dataset import build_long_horizon_dataset
    from cvae_sa.long_horizon_dataset import LongHorizonWindowDataset
    from tests.fixtures import write_physics_collection_run
    from cvae_sa.physics_indexer import build_physics_index


@unittest.skipUnless(h5py is not None, "requires h5py")
class LongHorizonDatasetTests(unittest.TestCase):
    def test_non_overlapping_groups_cover_tail_and_preserve_chunk_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_physics_collection_run(
                root / "sources", "source", "motion_a", "Locomotion", (0, 1, 2, 3)
            )
            parent = root / "physics"
            build_physics_index(
                (source,), parent, expected_motions=1, expected_episodes=4,
                split_counts=(1, 0, 0), seed=17,
            )
            output = root / "long"
            manifest = build_long_horizon_dataset(parent, output)
            self.assertEqual(manifest["schema_version"], "sonic_long_horizon_state_action_dataset_v1")
            coverage = json.loads((output / "manifests" / "coverage.json").read_text())
            self.assertEqual(coverage["total_transitions"], coverage["covered_transitions"])
            groups = [json.loads(line) for line in (output / "manifests/long_horizon_groups.jsonl").read_text().splitlines()]
            windows = [json.loads(line) for line in (output / "manifests/long_horizon_windows.jsonl").read_text().splitlines()]
            self.assertTrue(groups)
            self.assertTrue(all(len(row["chunk_starts"]) == 3 for row in groups))
            self.assertTrue(all(row["window_start"] % 64 == 0 for row in windows))
            dataset = LongHorizonWindowDataset(output, "train")
            sample = dataset[0]
            self.assertEqual(tuple(sample["physical_state"].shape), (65, 70))
            self.assertEqual(tuple(sample["action"].shape), (64, 29))
            self.assertIn("segment_group_id", sample)
            dataset.close()


if __name__ == "__main__":
    unittest.main()
