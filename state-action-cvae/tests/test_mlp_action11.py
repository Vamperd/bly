from __future__ import annotations

import unittest

import numpy as np
import torch

from cvae_sa.mlp_action11 import (
    ACTION_DIM,
    ACTION_STEPS,
    STATE_DIM,
    STATE_STEPS,
    CompletionMaskSampler,
    MLPActionCompletion11,
    build_recursive_query,
    masked_mse,
    predict_recursive_action,
)


class MLPAction11Tests(unittest.TestCase):
    def config(self) -> dict:
        return {
            "kind": "mlp_action_completion_11",
            "state_steps": STATE_STEPS,
            "action_steps": ACTION_STEPS,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "latent_dim": 16,
            "encoder_widths": [64, 32, 16],
            "decoder_widths": [32, 64, 16],
        }

    def test_forward_shape_and_finite_loss(self) -> None:
        model = MLPActionCompletion11(self.config())
        batch = 3
        sampler = CompletionMaskSampler(17)
        masks = sampler.sample(batch)
        state = torch.randn(batch, STATE_STEPS, STATE_DIM)
        action = torch.randn(batch, ACTION_STEPS, ACTION_DIM)
        before = torch.randn(batch, ACTION_DIM)
        output = model(
            state * masks.state_visible,
            action * masks.action_visible,
            masks.state_visible,
            masks.action_visible,
            masks.state_valid,
            masks.action_valid,
            masks.target_action,
            before,
            torch.ones(batch, dtype=torch.bool),
        )
        self.assertEqual(tuple(output.shape), (batch, ACTION_STEPS, ACTION_DIM))
        self.assertTrue(torch.isfinite(masked_mse(output, action, masks.target_action)))
        self.assertGreater(model.architecture_signature()["parameter_count"], 0)

    def test_mask_ratio_and_reasonable_target(self) -> None:
        masks = CompletionMaskSampler(20261110).sample(1000)
        step_ratio = sum(value.startswith("step_") for value in masks.family) / len(masks.family)
        self.assertGreater(step_ratio, 0.60)
        self.assertLess(step_ratio, 0.80)
        self.assertTrue(masks.target_action.any(dim=(1, 2)).all())
        for index in range(1000):
            target_rows = masks.target_action[index].any(dim=-1)
            self.assertTrue(torch.all(~masks.action_visible[index][masks.target_action[index]]))
            self.assertTrue(torch.all(masks.action_valid[index][target_rows]))

    def test_recursive_query_right_aligns_and_never_represents_a10(self) -> None:
        expected = np.arange(193 * STATE_DIM, dtype=np.float32).reshape(193, STATE_DIM)
        actual = [expected[0].copy()]
        query = build_recursive_query(
            expected_states=expected,
            actual_states=actual,
            executed_actions=[],
            target_step=0,
            state_mean=np.zeros(STATE_DIM, dtype=np.float32),
            state_std=np.ones(STATE_DIM, dtype=np.float32),
            action_mean=np.zeros(ACTION_DIM, dtype=np.float32),
            action_std=np.ones(ACTION_DIM, dtype=np.float32),
        )
        self.assertEqual(tuple(query["state"].shape), (1, STATE_STEPS, STATE_DIM))
        self.assertEqual(tuple(query["action"].shape), (1, ACTION_STEPS, ACTION_DIM))
        self.assertTrue(torch.equal(query["state_valid"][0, -2:], torch.ones(2, dtype=torch.bool)))
        self.assertFalse(query["state_valid"][0, :-2].any())
        self.assertTrue(query["target_action"][0, -1].all())
        self.assertFalse(query["action_before_valid"][0])

        actual = [expected[index].copy() for index in range(10)]
        executed = [np.full(ACTION_DIM, index, dtype=np.float32) for index in range(9)]
        query = build_recursive_query(
            expected_states=expected,
            actual_states=actual,
            executed_actions=executed,
            target_step=9,
            state_mean=np.zeros(STATE_DIM, dtype=np.float32),
            state_std=np.ones(STATE_DIM, dtype=np.float32),
            action_mean=np.zeros(ACTION_DIM, dtype=np.float32),
            action_std=np.ones(ACTION_DIM, dtype=np.float32),
        )
        self.assertTrue(query["state_valid"].all())
        self.assertTrue(query["action_visible"][0, :9].all())
        self.assertFalse(query["action_visible"][0, 9].any())
        self.assertTrue(query["target_action"][0, 9].all())
        self.assertFalse(query["action_before_valid"][0])

        model = MLPActionCompletion11(self.config()).eval()
        predicted = predict_recursive_action(model, query, torch.device("cpu"))
        self.assertEqual(tuple(predicted.shape), (1, ACTION_DIM))
        self.assertTrue(torch.isfinite(predicted).all())


if __name__ == "__main__":
    unittest.main()
