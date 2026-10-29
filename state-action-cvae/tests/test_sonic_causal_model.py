import unittest

import torch

from cvae_sa.sonic_causal_model import SonicCausalCVAE
from cvae_sa.sonic_causal_objective import compose_causal_c_loss


class SonicCausalModelTest(unittest.TestCase):
    def _model(self):
        return SonicCausalCVAE(
            {
                "d_model": 64,
                "heads": 4,
                "ffn_dim": 128,
                "posterior_encoder_layers": 1,
                "condition_encoder_layers": 1,
                "decoder_layers": 1,
                "global_latent_dim": 16,
                "dropout": 0.0,
            }
        )

    def _batch(self):
        return {
            "state_history": torch.randn(2, 10, 70),
            "action_history": torch.randn(2, 10, 29),
            "reference_future": torch.randn(2, 10, 64),
            "target_action": torch.randn(2, 29),
            "reference_time_offsets": torch.arange(10).float()[None].expand(2, -1) * 0.1,
        }

    def test_contract_and_inference_routes(self):
        model = self._model()
        batch = self._batch()
        posterior = model(batch, stage="C")
        standard = model.infer_from_condition(
            batch, epsilon=torch.zeros(2, 16)
        )
        self.assertEqual(tuple(posterior.action.shape), (2, 29))
        self.assertEqual(tuple(standard.action.shape), (2, 29))
        self.assertIsNotNone(posterior.posterior)
        self.assertEqual(posterior.global_latent.shape, (2, 16))

    def test_condition_path_does_not_read_target_action(self):
        model = self._model()
        batch = self._batch()
        epsilon = torch.zeros(2, 16)
        first = model.infer_from_condition(batch, epsilon=epsilon).action
        changed = dict(batch)
        changed["target_action"] = batch["target_action"] + 100.0
        second = model.infer_from_condition(changed, epsilon=epsilon).action
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=1e-6))

    def test_layout_has_ten_history_ten_reference_and_one_query(self):
        model = self._model()
        tokens, valid, _ = model._encode_tokens(
            self._batch(), target_visible=False, condition=True
        )
        self.assertEqual(tuple(tokens.shape), (2, 21, 64))
        self.assertEqual(tuple(valid.shape), (2, 21))
        self.assertTrue(bool(valid.all()))

    def test_c_loss_contains_only_action_and_kl_terms(self):
        model = self._model()
        batch = self._batch()
        posterior = model(batch, stage="C")
        standard = model.infer_from_condition(batch, epsilon=torch.zeros(2, 16))
        loss, terms = compose_causal_c_loss(
            posterior_action=posterior.action,
            standard_normal_action=standard.action,
            target_action=batch["target_action"],
            posterior=posterior.posterior,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(terms), {"posterior_action", "standard_normal_action", "kl"})


if __name__ == "__main__":
    unittest.main()

