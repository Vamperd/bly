from __future__ import annotations

import unittest

import numpy as np

from cvae_sa.action_selection import select_latent_action_window


class ActionSelectionTest(unittest.TestCase):
    def test_oracle_selects_one_whole_candidate_with_lowest_masked_error(self) -> None:
        truth = np.zeros((4, 3), dtype=np.float32)
        mask = np.zeros_like(truth, dtype=bool)
        mask[1:3, 1:] = True
        prior_mean = np.where(mask, 0.5, truth).astype(np.float32)
        candidates = np.stack(
            [
                np.where(mask, 0.3, truth),
                np.where(mask, 0.1, truth),
                np.where(mask, -0.2, truth),
            ]
        ).astype(np.float32)

        selected, selected_index, errors = select_latent_action_window(
            prior_mean, candidates, truth, mask, "oracle_best_of_n"
        )

        self.assertEqual(selected_index, 1)
        np.testing.assert_array_equal(selected, candidates[1])
        np.testing.assert_allclose(errors, [0.3, 0.1, 0.2], atol=1.0e-6)

    def test_prior_mean_does_not_select_using_ground_truth(self) -> None:
        truth = np.zeros((2, 2), dtype=np.float32)
        mask = np.ones_like(truth, dtype=bool)
        prior_mean = np.full_like(truth, 0.25)
        candidates = np.stack([np.zeros_like(truth), np.ones_like(truth)])

        selected, selected_index, _ = select_latent_action_window(
            prior_mean, candidates, truth, mask, "prior_mean"
        )

        self.assertEqual(selected_index, -1)
        np.testing.assert_array_equal(selected, prior_mean)


if __name__ == "__main__":
    unittest.main()
