from __future__ import annotations

import unittest

import numpy as np
import torch

from cvae_sa.action_mask_eval import (
    _mask_batch,
    _post_mask_metrics,
    _scan_window_starts,
)


class ActionMaskEvaluationTest(unittest.TestCase):
    def test_action_mask_hides_next_previous_action_without_state_mask(self) -> None:
        mask = np.zeros((128, 29), dtype=bool)
        mask[20:28] = True
        masks = _mask_batch(mask, "completion", torch.device("cpu"))
        self.assertFalse(masks.state_input.any().item())
        self.assertTrue(masks.action_input[0, 20:28].all().item())
        self.assertTrue(masks.previous_input[0, 21:29].all().item())
        self.assertFalse(masks.previous_loss.any().item())
        self.assertTrue(masks.action_loss.equal(masks.action_input))

    def test_stride_scan_preserves_margins_and_includes_last_window(self) -> None:
        starts = _scan_window_starts(400)
        self.assertEqual(starts[0], 32)
        self.assertEqual(starts[-1], 400 - 128 - 32)
        self.assertTrue(all(next_value > value for value, next_value in zip(starts, starts[1:])))

    def test_post_mask_metrics_reports_controlled_joint_drift(self) -> None:
        states = 80
        reference = {
            "joint_pos": np.zeros((states, 29), dtype=np.float32),
            "joint_vel": np.zeros((states, 29), dtype=np.float32),
            "root_pos": np.tile(np.array([[0.0, 0.0, 0.8]], dtype=np.float32), (states, 1)),
            "root_quat": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (states, 1)),
            "body_pos": np.zeros((states, 30, 3), dtype=np.float32),
            "physical_state": np.tile(
                np.concatenate((np.zeros(61), np.array([0.0, 0.0, -1.0])))[None],
                (states, 1),
            ).astype(np.float32),
        }
        value = {name: array.copy() for name, array in reference.items()}
        value["joint_pos"][21:, 0] = np.linspace(0.0, 0.2, states - 21)
        metrics = _post_mask_metrics(reference, value, first_mask_action=20, control_dt=0.02)
        self.assertGreater(metrics["joint_error_growth_rate_rad_s"], 0.0)
        self.assertFalse(metrics["fell_relative_to_original"])
        self.assertIsNotNone(metrics["post_mask_0_5_seconds"])


if __name__ == "__main__":
    unittest.main()
