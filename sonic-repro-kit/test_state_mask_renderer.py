import unittest
from pathlib import Path

from render_state_mask_comparison import comparison_output_name, grid_shape


class StateMaskRendererTests(unittest.TestCase):
    def test_representative_names_are_stable(self) -> None:
        self.assertEqual(
            comparison_output_name("forward_rollout_32"),
            "forward_rollout_32_ood_comparison.mp4",
        )
        self.assertEqual(
            comparison_output_name("step_contiguous_32"),
            "state_step_32_comparison.mp4",
        )
        self.assertEqual(
            comparison_output_name("step_contiguous_8"),
            "state_step_8_comparison.mp4",
        )

    def test_grid_shape(self) -> None:
        self.assertEqual(grid_shape(2), (1, 2))
        self.assertEqual(grid_shape(17), (4, 5))

    def test_action_render_phase_has_no_state_dependencies(self) -> None:
        root = Path(__file__).resolve().parent
        script = (root / "sonic_repro.sh").read_text(encoding="utf-8")
        action_start = script.index("phase_render_action_mask()")
        state_start = script.index("phase_render_state_mask()")
        action_body = script[action_start:state_start]
        self.assertNotIn("STATE_MASK_", action_body)
        action_renderer = (root / "render_action_mask_comparison.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("state_predictions.npz", action_renderer)


if __name__ == "__main__":
    unittest.main()
