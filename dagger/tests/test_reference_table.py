import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from dagger.export_reference_table import alignment_check, joint_index


def _table(order: np.ndarray) -> dict:
    steps = np.asarray([20, 15], dtype=np.int64)
    rng = np.random.default_rng(0)
    states = rng.normal(size=(int(steps.sum()), 70)).astype(np.float32)
    return {
        "states": states,
        "motion_keys": np.asarray(["a", "b"]),
        "motion_offsets": np.concatenate(([0], np.cumsum(steps))).astype(np.int64),
        "num_steps": steps,
        "nominal_joint_pos": np.linspace(-0.2, 0.2, 29).astype(np.float32),
        "joint_ids": order,
    }


def _write_episode(stream, name, table, motion, motion_step, shift):
    start = int(table["motion_offsets"][motion])
    last = int(table["num_steps"][motion]) - 1
    frames = np.clip(motion_step + shift, 0, last) + start
    action_pos = table["states"][frames, :29] + table["nominal_joint_pos"]
    action_vel = table["states"][frames, 29:58]
    inverse = np.argsort(table["joint_ids"])
    flat = np.zeros((motion_step.size, 580), dtype=np.float32)
    flat[:, :29] = action_pos[:, inverse]
    flat[:, 290:319] = action_vel[:, inverse]
    recorded = flat.reshape(motion_step.size, 10, 58)
    group = stream.create_group(f"data/{name}")
    group.create_dataset("motion/motion_step", data=motion_step.astype(np.int64))
    group.create_dataset("reference/joint_pos_vel_future", data=recorded)


class ReferenceTableAlignmentTest(unittest.TestCase):
    def _run(self, order: np.ndarray, shift: int) -> dict:
        table = _table(order)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "v5.hdf5"
            with h5py.File(path, "w") as stream:
                _write_episode(stream, "demo_0", table, 0, np.arange(3, 15), shift)
                _write_episode(stream, "demo_1", table, 1, np.arange(0, 14), shift)
            episodes = [
                {"episode": "demo_0", "motion_key": "a", "hdf5_path": str(path)},
                {"episode": "demo_1", "motion_key": "b", "hdf5_path": str(path)},
            ]
            return alignment_check(episodes, table, tolerance=1.0e-5)

    def test_identity_order_zero_offset_passes(self):
        report = self._run(np.arange(29), 0)
        self.assertTrue(report["pass"])
        self.assertEqual(report["best_offset"], 0)
        self.assertIn("recorded_reindexed", report["matched_orders"])
        self.assertGreater(report["max_abs_error"]["recorded_reindexed"]["1"]["pos"], 1.0e-3)

    def test_off_by_one_fails(self):
        report = self._run(np.arange(29), 1)
        self.assertFalse(report["pass"])
        self.assertEqual(report["best_offset"], 1)

    def test_permuted_joint_order_matches_reindexed(self):
        order = np.random.default_rng(1).permutation(29)
        report = self._run(order, 0)
        self.assertTrue(report["pass"])
        self.assertEqual(report["matched_orders"], ["recorded_reindexed"])

    def test_joint_index_slice(self):
        np.testing.assert_array_equal(joint_index(slice(None), 29), np.arange(29))
        np.testing.assert_array_equal(joint_index([2, 0, 1], 3), np.asarray([2, 0, 1]))


if __name__ == "__main__":
    unittest.main()
