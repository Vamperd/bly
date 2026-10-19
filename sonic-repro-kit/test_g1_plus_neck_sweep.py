from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from render_g1_plus_neck_sweep import _select_body_frame


class NeckSweepSelectionTests(unittest.TestCase):
    def test_selects_first_upright_bilateral_contact_frame(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "source.replay.npz"
            state = np.zeros((4, 70), dtype=np.float32)
            state[1, 58:61] = [0.2, 0.0, 0.0]
            state[1, 68:70] = [1.0, 1.0]
            state[2, 68:70] = [1.0, 1.0]
            np.savez(path, physical_state=state)

            frame, details = _select_body_frame(path, -1, 4)

            self.assertEqual(frame, 2)
            self.assertEqual(details["mode"], "automatic_upright")
            self.assertEqual(details["foot_contact"], [1.0, 1.0])

    def test_explicit_frame_does_not_require_state_archive(self) -> None:
        frame, details = _select_body_frame(None, 3, 4)
        self.assertEqual(frame, 3)
        self.assertEqual(details["mode"], "explicit")


if __name__ == "__main__":
    unittest.main()
