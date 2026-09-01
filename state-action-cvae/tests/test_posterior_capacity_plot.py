from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from cvae_sa.posterior_capacity import FIXED_MASK_NAMES
from cvae_sa.posterior_capacity_plot import render_posterior_capacity_plots


class PosteriorCapacityPlotTest(unittest.TestCase):
    def test_clear_log_axes_scopes_thresholds_and_zero_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "logs").mkdir()
            cases = {
                name: {
                    "worst_state_rmse": 1e-3,
                    "worst_action_rmse": 2e-3,
                    "continuous_max_abs": 5e-3,
                    "contact_accuracy": 1.0,
                }
                for name in FIXED_MASK_NAMES
            }
            thresholds = {
                "state_rmse": 1e-4,
                "action_rmse": 1e-4,
                "continuous_max_abs": 1e-3,
                "latent_ratio": 10.0,
            }
            progression = {
                "state_rmse": 1e-2,
                "action_rmse": 1e-2,
                "continuous_max_abs": 1e-2,
                "latent_ratio": 10.0,
            }
            records = [
                {
                    "phase": "train",
                    "optimizer_step": 1,
                    "learning_rate": 3e-4,
                    "gradient_norm": 0.0,
                    "total": 0.0,
                    "state": 1e-5,
                    "action": 2e-5,
                    "contact": 3e-5,
                },
                {
                    "phase": "validation",
                    "optimizer_step": 1,
                    "evaluation_scope": "full fixed-fixture evaluation on the same training windows and masks",
                    "acceptance_gate": "progression",
                    "score": 0.5,
                    "worst_state_rmse": 1e-3,
                    "worst_action_rmse": 2e-3,
                    "continuous_max_abs": 5e-3,
                    "contact_accuracy": 1.0,
                    "reconstruction_loss": {
                        "total": 2e-5,
                        "state": 1e-5,
                        "action": 2e-5,
                        "contact": 3e-5,
                    },
                    "latent_dependence": {"zero_ratio": 20.0, "swapped_ratio": 18.0},
                    "exact_gate": {"thresholds": thresholds},
                    "progression_gate": {"thresholds": progression},
                    "cases": cases,
                },
            ]
            metrics = run_dir / "logs/metrics.jsonl"
            metrics.write_text(
                "".join(json.dumps(record) + "\n" for record in records) + "{partial",
                encoding="utf-8",
            )
            paths = render_posterior_capacity_plots(run_dir)
            self.assertEqual(len(paths), 3)
            for path in paths:
                self.assertTrue(path.is_file())
                ET.parse(path)
            training = paths[0].read_text(encoding="utf-8")
            gates = paths[1].read_text(encoding="utf-8")
            masks = paths[2].read_text(encoding="utf-8")
            self.assertIn("Optimizer step", training)
            self.assertIn("Value (log10 scale)", training)
            self.assertIn("Train batch raw", training)
            self.assertIn("Full fixed-fixture evaluation", training)
            self.assertIn("same training windows and masks", training)
            self.assertIn("1e-12", training)
            self.assertIn("10^-", training)
            self.assertIn("exact", gates)
            self.assertIn("progression", gates)
            self.assertIn("PASS &lt;= 1", gates)
            self.assertIn("Gate ratio (log10 scale)", masks)
            self.assertIn("full_both", masks)
            records[-1]["evaluation_scope"] = "held-out-mask evaluation on seen training windows"
            metrics.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            held_out_training = render_posterior_capacity_plots(run_dir)[0].read_text(
                encoding="utf-8"
            )
            self.assertIn("Held-out-mask evaluation", held_out_training)
            self.assertIn("seen training windows", held_out_training)


if __name__ == "__main__":
    unittest.main()
