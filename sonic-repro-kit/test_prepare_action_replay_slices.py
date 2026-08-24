from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from prepare_action_replay_slices import split_replay


class PrepareActionReplaySlicesTest(unittest.TestCase):
    def test_each_output_contains_exactly_one_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "batched.npz"
            raw = np.arange(5 * 3 * 29, dtype=np.float32).reshape(5, 3, 29)
            np.savez_compressed(source, raw_actions=raw)

            manifest = split_replay(source, root / "slices", 3)

            self.assertEqual(manifest["execution_mode"], "serial_single_environment")
            self.assertEqual(len(manifest["slices"]), 3)
            for env_id, record in enumerate(manifest["slices"]):
                with np.load(record["path"], allow_pickle=False) as values:
                    actual = np.asarray(values["raw_actions"])
                self.assertEqual(actual.shape, (5, 1, 29))
                np.testing.assert_array_equal(actual[:, 0], raw[:, env_id])


if __name__ == "__main__":
    unittest.main()
