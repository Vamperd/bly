from __future__ import annotations

import unittest

import numpy as np

from cvae_sa.latent_distribution65 import GROUP_NAMES, _group_summary, kl_per_dimension


class LatentDistribution65Test(unittest.TestCase):
    def test_seventeen_group_contract_and_finite_kl(self):
        self.assertEqual(len(GROUP_NAMES), 17)
        self.assertEqual(GROUP_NAMES[0], "global")
        self.assertEqual(GROUP_NAMES[-1], "local_15")
        mean = np.zeros((3, 4), dtype=np.float32)
        logvar = np.zeros((3, 4), dtype=np.float32)
        np.testing.assert_allclose(kl_per_dimension(mean, logvar), 0.0)

    def test_group_summary_exposes_partial_kl_and_normal_gap(self):
        mean = np.zeros((5, 4), dtype=np.float32)
        mean[:, 0] = 2.0
        logvar = np.zeros_like(mean)
        logvar[:, 1] = -2.0
        samples = np.stack([mean + np.exp(0.5 * logvar) * offset for offset in (-1.0, 0.0, 1.0)])
        summary = _group_summary("local_03", mean, logvar, samples)
        self.assertEqual(summary["window_count"], 5)
        self.assertEqual(summary["latent_dim"], 4)
        self.assertGreater(summary["kl_dim_fraction_gt_1e-1"], 0.0)
        self.assertGreater(summary["standard_normal_gap"]["sample_mean_abs_gap"], 0.0)
        self.assertEqual(len(summary["per_dimension"]["kl_mean"]), 4)

    def test_group_summary_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            _group_summary(
                "global",
                np.zeros((2, 3), dtype=np.float32),
                np.zeros((2, 3), dtype=np.float32),
                np.zeros((2, 3, 3), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
