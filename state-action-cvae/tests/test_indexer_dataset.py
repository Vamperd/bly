from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cvae_sa.dataset import StateActionWindowDataset
from cvae_sa.indexer import build_index
from cvae_sa.physics_indexer import build_physics_index
from tests.fixtures import write_collection_run, write_physics_collection_run


class IndexerDatasetTest(unittest.TestCase):
    def test_action_energy_candidates_select_top_quarter(self) -> None:
        actions = np.zeros((40, 29), dtype=np.float32)
        actions[20:, 0] = np.arange(20, dtype=np.float32) * 10.0
        starts = StateActionWindowDataset.high_energy_window_starts(
            actions, window_transitions=8, top_fraction=0.25
        )
        self.assertEqual(len(starts), int(np.ceil((40 - 8 + 1) * 0.25)))
        derivative = np.square(np.diff(actions, axis=0)).sum(axis=-1)
        scores = np.convolve(derivative, np.ones(7), mode="valid") / 7
        cutoff = np.partition(scores, -len(starts))[-len(starts)]
        self.assertTrue(bool((scores[starts] >= cutoff).all()))

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

    def test_builds_physics_v3_index_and_unique_state_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = (
                write_physics_collection_run(root / "sources", "a0", "motion_a", "Locomotion", (0, 1, 2, 3)),
                write_physics_collection_run(root / "sources", "a1", "motion_a", "Locomotion", (4, 5, 6, 7)),
                write_physics_collection_run(root / "sources", "b0", "motion_b", "Dances", (0, 1, 2, 3)),
                write_physics_collection_run(root / "sources", "b1", "motion_b", "Dances", (4, 5, 6, 7)),
            )
            output = root / "physics_dataset"
            manifest = build_physics_index(
                sources,
                output,
                expected_motions=2,
                expected_episodes=16,
                split_counts=(1, 1, 0),
                seed=11,
            )
            self.assertEqual(manifest["canonical_episode_count"], 16)
            dataset = StateActionWindowDataset(output, "train", window_transitions=8)
            value = dataset[0]
            self.assertEqual(tuple(value["physical_state"].shape), (9, 70))
            self.assertEqual(tuple(value["previous_action"].shape), (9, 0))
            self.assertEqual(tuple(value["action"].shape), (8, 29))
            self.assertEqual(tuple(value["robot_information"].shape), (293,))
            self.assertEqual(tuple(value["joint_robot_information"].shape), (29, 11))
            self.assertEqual(tuple(value["joint_actuator_type"].shape), (29,))
            self.assertEqual(tuple(value["global_robot_information"].shape), (9,))
            self.assertEqual(tuple(value["action_before_window"].shape), (29,))
            restored_before = dataset.denormalize_action(value["action_before_window"])
            self.assertTrue(np.isfinite(restored_before.numpy()).all())
            self.assertEqual(tuple(value["dynamics_context"].shape), (648,))
            self.assertEqual(tuple(value["auxiliary_transition"].shape), (8, 35))
            self.assertEqual(int(value["valid_state"].sum()), 7)
            self.assertEqual(int(value["valid_action"].sum()), 6)
            dataset.close()


if __name__ == "__main__":
    unittest.main()
