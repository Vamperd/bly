from __future__ import annotations

import unittest

from render_action_mask_comparison import (
    REPRESENTATIVE_OUTPUTS,
    comparison_output_name,
    grid_shape,
)


class ActionMaskRendererTest(unittest.TestCase):
    def test_representative_video_names_match_public_contract(self) -> None:
        self.assertEqual(comparison_output_name("step_contiguous_8"), "step_8_comparison.mp4")
        self.assertEqual(
            comparison_output_name("semantic_left_leg"),
            "semantic_left_leg_comparison.mp4",
        )
        self.assertEqual(len(REPRESENTATIVE_OUTPUTS), 5)

    def test_grid_uses_five_columns_for_original_plus_default_suite(self) -> None:
        self.assertEqual(grid_shape(17), (4, 5))
        self.assertEqual(grid_shape(3), (1, 3))


if __name__ == "__main__":
    unittest.main()
