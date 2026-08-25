from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from consolidate_physics_state_action import _transition_count


class TransitionCountTest(unittest.TestCase):
    def _episode(self, path: Path, steps: int, num_samples: int):
        stream = h5py.File(path, "w")
        episode = stream.create_group("data/demo_0")
        episode.attrs["num_samples"] = num_samples
        episode.create_dataset(
            "physics_action/action_target_canonical",
            data=np.zeros((steps, 29), dtype=np.float32),
        )
        return stream, episode

    def test_uses_action_length_when_generic_attribute_is_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            stream, episode = self._episode(Path(directory) / "raw.hdf5", 17, 0)
            with stream:
                self.assertEqual(_transition_count(episode), 17)

    def test_accepts_matching_generic_attribute(self):
        with tempfile.TemporaryDirectory() as directory:
            stream, episode = self._episode(Path(directory) / "raw.hdf5", 17, 17)
            with stream:
                self.assertEqual(_transition_count(episode), 17)

    def test_rejects_nonzero_mismatched_attribute(self):
        with tempfile.TemporaryDirectory() as directory:
            stream, episode = self._episode(Path(directory) / "raw.hdf5", 17, 16)
            with stream:
                with self.assertRaisesRegex(ValueError, "differs from Action length"):
                    _transition_count(episode)

    def test_rejects_truly_empty_action_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            stream, episode = self._episode(Path(directory) / "raw.hdf5", 0, 0)
            with stream:
                with self.assertRaisesRegex(ValueError, "empty episode"):
                    _transition_count(episode)


if __name__ == "__main__":
    unittest.main()
