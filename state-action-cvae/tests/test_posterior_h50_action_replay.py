from __future__ import annotations

import unittest

import numpy as np

from cvae_sa.posterior_h50_action_replay import _error_metrics, _select_seen_window


class H50SeenWindowReplayTests(unittest.TestCase):
    def test_auto_selection_is_deterministic_and_not_metric_driven(self) -> None:
        windows = [
            {"motion_key": "z_motion", "variant_id": 0, "window_start": 0},
            {"motion_key": "a_motion", "variant_id": 0, "window_start": 64},
            {"motion_key": "b_motion", "variant_id": 0, "window_start": 0},
            {"motion_key": "a_motion", "variant_id": 0, "window_start": 0},
        ]
        index, selected, rule = _select_seen_window(windows, "auto", 0, 0)
        self.assertEqual(index, 3)
        self.assertEqual(selected["motion_key"], "a_motion")
        self.assertFalse(rule["performance_metrics_used_for_selection"])

    def test_explicit_selection_requires_exact_seen_identity(self) -> None:
        windows = [
            {"motion_key": "motion", "variant_id": 0, "window_start": 0},
            {"motion_key": "motion", "variant_id": 1, "window_start": 0},
        ]
        index, selected, _ = _select_seen_window(windows, "motion", 1, 0)
        self.assertEqual(index, 1)
        self.assertEqual(selected["variant_id"], 1)
        with self.assertRaisesRegex(ValueError, "no seen H50-A window"):
            _select_seen_window(windows, "missing", 0, 0)

    def test_error_metrics_have_direct_units(self) -> None:
        target = np.zeros((2, 2), dtype=np.float32)
        prediction = np.asarray([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        metrics = _error_metrics(prediction, target)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(5.0 / 4.0))
        self.assertAlmostEqual(metrics["mean_abs"], 0.75)
        self.assertEqual(metrics["max_abs"], 2.0)
        self.assertGreaterEqual(metrics["p99_abs"], 1.9)


if __name__ == "__main__":
    unittest.main()
