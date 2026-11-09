from __future__ import annotations

import unittest

import torch

from cvae_sa.sonic_reference_mlp_model import SonicReferenceMLPCVAE


class SonicReferenceMLPTests(unittest.TestCase):
    def config(self) -> dict:
        return {
            "global_latent_dim": 16,
            "encoder_widths": [64, 32],
            "decoder_widths": [64, 32],
        }

    def batch(self, size: int = 3) -> dict[str, torch.Tensor]:
        return {
            "state_history": torch.randn(size, 10, 70),
            "action_history": torch.randn(size, 10, 29),
            "reference_future": torch.randn(size, 10, 64),
            "target_action": torch.randn(size, 29),
            "state_valid": torch.ones(size, 10, dtype=torch.bool),
            "action_valid": torch.ones(size, 10, dtype=torch.bool),
            "reference_time_offsets": torch.arange(10, dtype=torch.float32)[None].expand(size, -1),
        }

    def test_c_and_infer_shapes(self) -> None:
        model = SonicReferenceMLPCVAE(self.config())
        batch = self.batch()
        c = model(batch, stage="C")
        infer = model.infer_from_condition(batch, sample=False)
        self.assertEqual(tuple(c.action.shape), (3, 29))
        self.assertEqual(tuple(infer.action.shape), (3, 29))
        self.assertIsNotNone(c.posterior)
        self.assertTrue(torch.isfinite(c.action).all())
        self.assertTrue(torch.isfinite(infer.action).all())
        self.assertGreater(model.architecture_instance_signature()["parameter_count"], 0)

    def test_posterior_receives_target_but_condition_does_not(self) -> None:
        model = SonicReferenceMLPCVAE(self.config())
        batch = self.batch()
        batch["target_action"].zero_()
        before = model.encode_posterior_distribution(batch).mean
        batch["target_action"].fill_(1.0)
        after = model.encode_posterior_distribution(batch).mean
        self.assertFalse(torch.allclose(before, after))
        self.assertEqual(tuple(model.encode_condition(batch).shape), (3, model.condition_encoder[-1].out_features))


if __name__ == "__main__":
    unittest.main()
