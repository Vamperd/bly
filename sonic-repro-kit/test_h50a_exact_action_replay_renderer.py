from __future__ import annotations

import unittest

from render_h50a_exact_action_replays import progress_bar_geometry


class H50ExactActionReplayRendererTests(unittest.TestCase):
    def test_progress_bar_marks_mask_interval_inside_full_episode(self) -> None:
        geometry = progress_bar_geometry(
            width=960,
            total_action_steps=320,
            masked_window_start=64,
            masked_window_stop=128,
        )
        self.assertLess(geometry["left"], geometry["masked_left"])
        self.assertLess(geometry["masked_left"], geometry["masked_right"])
        self.assertLess(geometry["masked_right"], geometry["right"])
        self.assertAlmostEqual(
            (geometry["masked_right"] - geometry["masked_left"])
            / (geometry["right"] - geometry["left"]),
            64 / 320,
            delta=0.002,
        )

    def test_progress_bar_rejects_invalid_mask_interval(self) -> None:
        with self.assertRaises(ValueError):
            progress_bar_geometry(960, 100, 60, 124)


if __name__ == "__main__":
    unittest.main()
