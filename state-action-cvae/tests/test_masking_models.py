from __future__ import annotations

import unittest

import torch

from cvae_sa.losses import compute_loss
from cvae_sa.masking import MaskGenerator
from cvae_sa.models import build_model


def batch(batch_size: int = 2, transitions: int = 8) -> dict[str, torch.Tensor]:
    return {
        "physical_state": torch.randn(batch_size, transitions + 1, 64),
        "previous_action": torch.randn(batch_size, transitions + 1, 29),
        "action": torch.randn(batch_size, transitions, 29),
        "action_scale": torch.ones(batch_size, 29),
        "valid_state": torch.ones(batch_size, transitions + 1, dtype=torch.bool),
        "valid_action": torch.ones(batch_size, transitions, dtype=torch.bool),
        "progress": torch.linspace(0, 1, transitions + 1).repeat(batch_size, 1),
    }


MASK_CONFIG = {
    "task_probabilities": [0.35, 0.25, 0.40],
    "completion_probabilities": [0.40, 0.30, 0.30],
    "element_fraction": [0.20, 0.20],
    "step_count": [2, 2],
    "feature_fraction": [0.20, 0.20],
}


MODEL_CONFIG = {
    "kind": "transformer",
    "d_model": 32,
    "encoder_layers": 1,
    "decoder_layers": 1,
    "heads": 4,
    "ffn_dim": 64,
    "latent_dim": 8,
    "dropout": 0.0,
    "tcn_channels": 32,
    "tcn_encoder_layers": 1,
    "tcn_decoder_layers": 1,
    "tcn_kernel_size": 3,
}


class MaskingModelTest(unittest.TestCase):
    def test_action_mask_hides_duplicate_without_double_loss(self) -> None:
        value = batch()
        masks = MaskGenerator(MASK_CONFIG).generate(value, force_task="inverse")
        self.assertTrue(torch.equal(masks.previous_input[:, 1:], masks.action_input))
        self.assertFalse(bool(masks.previous_loss.any()))

    def test_transformer_and_tcn_shapes_are_finite(self) -> None:
        value = batch()
        masks = MaskGenerator(MASK_CONFIG).generate(
            value, force_task="completion", force_completion="feature"
        )
        for kind in ("transformer", "tcn"):
            config = dict(MODEL_CONFIG, kind=kind)
            model = build_model(config)
            output = model(value, masks, deterministic=True)
            self.assertEqual(tuple(output.physical_state.shape), (2, 9, 64))
            self.assertEqual(tuple(output.previous_action.shape), (2, 9, 29))
            self.assertEqual(tuple(output.action.shape), (2, 8, 29))
            self.assertEqual(tuple(output.forward_delta.shape), (2, 8, 64))
            self.assertTrue(torch.isfinite(output.physical_state).all())

    def test_tiny_batch_can_overfit_inverse_mapping(self) -> None:
        torch.manual_seed(3)
        value = batch(batch_size=2, transitions=4)
        masks = MaskGenerator(MASK_CONFIG).generate(value, force_task="inverse")
        model = build_model(MODEL_CONFIG)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        training = {
            "free_bits": 0.0,
            "forward_weight": 2.0,
            "inverse_weight": 1.0,
            "gravity_weight": 0.1,
        }
        losses = []
        for _ in range(30):
            output = model(value, masks, deterministic=True)
            loss = compute_loss(output, value, masks, training, kl_beta=0.0).total
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(sum(losses[-5:]), sum(losses[:5]))


if __name__ == "__main__":
    unittest.main()

