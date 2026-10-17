from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import torch

from cvae_sa.cvae_diagnostics import Diagnostics, cached_route_output, epsilon_for, route_output
from cvae_sa.cvae_training import evaluation_profile_for_step
from cvae_sa.models import build_model
from cvae_sa.posterior_t64_protocol import make_physical_masks

from test_posterior_hierarchical_standard_cvae import batch, config


class EvaluationOptimizationTests(unittest.TestCase):
    def test_schedule_has_summary_steps_and_full_boundaries(self):
        self.assertIsNone(evaluation_profile_for_step(1, 300000, 5000, 50000))
        self.assertEqual(evaluation_profile_for_step(5000, 300000, 5000, 50000), "summary")
        self.assertEqual(evaluation_profile_for_step(45000, 300000, 5000, 50000), "summary")
        self.assertEqual(evaluation_profile_for_step(50000, 300000, 5000, 50000), "diagnostic")
        self.assertEqual(evaluation_profile_for_step(300000, 300000, 5000, 50000), "diagnostic")

    def test_route_cache_matches_uncached_eager_outputs(self):
        value = batch(2)
        model = build_model(config()).eval()
        state_mask, action_mask, _ = make_physical_masks(value, 17)
        epsilon = epsilon_for(model, value, 0, 17, "cpu")
        for route in ("posterior_mean", "posterior_sample", "standard_normal", "zero"):
            uncached = route_output(model, value, state_mask, action_mask, route, epsilon)
            cached = cached_route_output(model, value, state_mask, action_mask, route, epsilon, {})
            torch.testing.assert_close(uncached.physical_state, cached.physical_state, rtol=0, atol=0)
            torch.testing.assert_close(uncached.action, cached.action, rtol=0, atol=0)
            torch.testing.assert_close(uncached.state_contact_logits, cached.state_contact_logits, rtol=0, atol=0)

    def test_summary_diagnostics_preserve_partitions(self):
        value = batch(3)
        output = type("Output", (), {
            "physical_state": value["physical_state"] + 0.1,
            "action": value["action"] - 0.2,
            "state_contact_logits": torch.zeros(3, 65, 2),
        })()
        state_mask, action_mask, names = make_physical_masks(value, 3)
        normalization = {
            "state": (np.zeros(70), np.ones(70)),
            "action": (np.zeros(29), np.ones(29)),
        }
        full = Diagnostics(normalization)
        summary = Diagnostics(normalization, summary_only=True)
        full.add(value, output, state_mask, action_mask, names)
        summary.add(value, output, state_mask, action_mask, names)
        expected, actual = full.finish(), summary.finish()
        for part in ("full", "masked", "visible"):
            for domain in ("state", "action"):
                self.assertEqual(
                    expected["partitions"][part][domain]["count"],
                    actual["partitions"][part][domain]["count"],
                )
                self.assertAlmostEqual(
                    expected["partitions"][part][domain]["mse"],
                    actual["partitions"][part][domain]["mse"],
                )
        self.assertEqual(set(expected["mask_families"]), set(actual["mask_families"]))

    def test_long_horizon_config_preserves_training_batch(self):
        path = Path(__file__).resolve().parents[1] / "configs/posterior_hierarchical_standard_cvae_long_horizon_fp32.json"
        configuration = json.loads(path.read_text(encoding="utf-8"))
        training = configuration["training"]
        self.assertEqual(training["micro_batch"], 32)
        self.assertEqual(training["eval_micro_batch"], 128)
        self.assertEqual(training["validation_interval"], 5000)
        self.assertEqual(training["full_validation_interval"], 50000)
        self.assertTrue(training["skip_initial_eval"])


if __name__ == "__main__":
    unittest.main()
