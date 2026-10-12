from __future__ import annotations

import unittest

import numpy as np

from cvae_sa.latent_normality65 import dimension_statistics


class LatentNormality65Test(unittest.TestCase):
    def test_standard_normal_columns_are_close(self):
        rng = np.random.default_rng(7)
        values = rng.standard_normal((4096, 3)).astype(np.float32)
        result = dimension_statistics(values)
        self.assertEqual(result["mean"].shape, (3,))
        self.assertGreaterEqual(float(np.mean(result["near_standard_normal"])), 0.5)

    def test_shifted_column_is_not_near_standard_normal(self):
        rng = np.random.default_rng(8)
        values = rng.standard_normal((4096, 3)).astype(np.float32)
        values[:, 1] += 2.0
        result = dimension_statistics(values)
        self.assertFalse(result["near_standard_normal"][1])
        self.assertGreater(float(result["mean_gap"][1]), 1.0)


if __name__ == "__main__":
    unittest.main()
