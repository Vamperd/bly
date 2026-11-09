from __future__ import annotations

import unittest

import torch

from cvae_sa.mlp_action11 import ACTION_DIM, ACTION_STEPS, STATE_DIM, STATE_STEPS
from cvae_sa.mlp_cvae11 import CausalBlockMaskSampler, MLPConditionalVAE11, MODEL_INPUT_KEYS


class MLPCVAE11Tests(unittest.TestCase):
    def config(self) -> dict:
        return {
            "latent_dim": 16,
            "encoder_widths": [64, 32],
            "decoder_widths": [64, 32],
        }

    def batch(self, size: int = 4) -> dict[str, torch.Tensor]:
        return {
            "physical_state": torch.randn(size, STATE_STEPS, STATE_DIM),
            "action": torch.randn(size, ACTION_STEPS, ACTION_DIM),
            "action_before_window": torch.randn(size, ACTION_DIM),
            "action_before_valid": torch.ones(size, dtype=torch.bool),
        }

    def test_forward_shapes_and_sampled_prior_are_finite(self) -> None:
        model = MLPConditionalVAE11(self.config())
        raw = CausalBlockMaskSampler(17).sample(self.batch())
        values = {key: value for key, value in raw.items() if isinstance(value, torch.Tensor)}
        output = model(**{key: values[key] for key in MODEL_INPUT_KEYS}, sample=True)
        prior = model.prior_forward(
            condition_state=values["condition_state"],
            condition_action=values["condition_action"],
            state_visible=values["state_visible"],
            action_visible=values["action_visible"],
            state_valid=values["state_valid"],
            action_valid=values["action_valid"],
            target_state=values["target_state"],
            target_action=values["target_action"],
            action_before=values["action_before"],
            action_before_valid=values["action_before_valid"],
            sample=False,
        )
        self.assertEqual(tuple(prior["p_action"].shape), (4, ACTION_STEPS, ACTION_DIM))
        self.assertTrue(torch.isfinite(prior["p_action"]).all())
        self.assertEqual(tuple(output["q_output"].shape), (4, STATE_STEPS, STATE_DIM))
        self.assertEqual(tuple(output["p_action"].shape), (4, ACTION_STEPS, ACTION_DIM))
        for value in output.values():
            self.assertTrue(torch.isfinite(value).all())
        self.assertGreater(model.architecture_signature()["parameter_count"], 0)
        self.assertNotEqual(
            model.posterior_encoder[0].weight.data_ptr(),
            model.prior_encoder[0].weight.data_ptr(),
        )

    def test_mask_families_and_causal_target_alignment(self) -> None:
        batch = self.batch(4000)
        sampler = CausalBlockMaskSampler(20261112)
        masks = sampler.sample(batch)
        families = masks["family"]
        counts = {family: families.count(family) for family in {"causal", "state_only", "action_only", "both"}}
        for family in counts:
            self.assertGreater(counts[family], 500)
        causal_rows = [index for index, family in enumerate(families) if family == "causal"]
        self.assertTrue(causal_rows)
        for row in causal_rows[:20]:
            self.assertTrue(masks["target_action"][row, -1])
            self.assertFalse(masks["action_visible"][row, -1])
            self.assertTrue(masks["action_valid"][row, -1])
            self.assertTrue(torch.equal(masks["truth_action"][row, -1], masks["posterior_action"][row, -1]))
            self.assertTrue(torch.equal(masks["condition_action"][row, -1], torch.zeros(ACTION_DIM)))
            self.assertTrue(torch.equal(masks["action_before"][row], batch["action_before_window"][row]))
            self.assertEqual(bool(masks["action_before_valid"][row]), bool(batch["action_before_valid"][row]))

    def test_posterior_sees_masked_truth_while_prior_sees_condition(self) -> None:
        batch = self.batch(128)
        masks = CausalBlockMaskSampler(31).sample(batch)
        row = next(index for index, family in enumerate(masks["family"]) if family == "action_only")
        model = MLPConditionalVAE11(self.config())
        condition = model._condition_vector(
            masks["condition_state"][row : row + 1], masks["condition_action"][row : row + 1],
            masks["state_visible"][row : row + 1], masks["action_visible"][row : row + 1],
            masks["state_valid"][row : row + 1], masks["action_valid"][row : row + 1],
            masks["target_state"][row : row + 1], masks["target_action"][row : row + 1],
            masks["action_before"][row : row + 1], masks["action_before_valid"][row : row + 1],
        )
        posterior = model._condition_vector(
            masks["posterior_state"][row : row + 1], masks["posterior_action"][row : row + 1],
            masks["state_visible"][row : row + 1], masks["action_visible"][row : row + 1],
            masks["state_valid"][row : row + 1], masks["action_valid"][row : row + 1],
            masks["target_state"][row : row + 1], masks["target_action"][row : row + 1],
            masks["action_before"][row : row + 1], masks["action_before_valid"][row : row + 1],
            use_visibility=False,
        )
        self.assertFalse(torch.allclose(condition, posterior))

    def test_initial_action_has_invalid_history_marker(self) -> None:
        batch = self.batch(100)
        masks = CausalBlockMaskSampler(9).sample(batch)
        causal_rows = [index for index, family in enumerate(masks["family"]) if family == "causal"]
        self.assertTrue(causal_rows)
        row = causal_rows[0]
        if not masks["action_before_valid"][row]:
            self.assertTrue(torch.equal(masks["action_before"][row], torch.zeros(ACTION_DIM)))


if __name__ == "__main__":
    unittest.main()
