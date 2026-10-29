import unittest

import torch

from dagger.runtime.sonic_causal_window import SonicCausalWindowBuffer


class SonicCausalWindowTest(unittest.TestCase):
    def test_reset_and_advance_keep_ten_history_steps(self):
        buffer = SonicCausalWindowBuffer(device=torch.device("cpu"), num_envs=1)
        state = torch.zeros(1, 70)
        action = torch.ones(1, 29)
        buffer.reset(state, action)
        self.assertEqual(buffer.history_steps().tolist(), [10])
        buffer.advance(torch.full((1, 29), 2.0), torch.ones(1, 70), torch.zeros(1, dtype=torch.bool))
        window = buffer.build_condition(
            reference_future=torch.zeros(1, 10, 64),
            state_mean=torch.zeros(70),
            state_std=torch.ones(70),
            action_mean=torch.zeros(29),
            action_std=torch.ones(29),
            reference_mean=torch.zeros(64),
            reference_std=torch.ones(64),
        )
        self.assertEqual(tuple(window.batch["state_history"].shape), (1, 10, 70))
        self.assertEqual(tuple(window.batch["action_history"].shape), (1, 10, 29))
        self.assertEqual(tuple(window.batch["reference_future"].shape), (1, 10, 64))
        self.assertEqual(window.batch["state_valid"].tolist(), [[False] * 8 + [True, True]])
        self.assertEqual(window.batch["action_valid"].tolist(), [[False] * 9 + [True]])
        self.assertEqual(tuple(window.batch["history_validity"].shape), (1, 10, 2))
        self.assertTrue(torch.equal(window.batch["state_history"][0, -1], torch.ones(70)))
        self.assertTrue(torch.equal(window.batch["action_history"][0, -1], torch.full((29,), 2.0)))

    def test_done_resets_to_padding_and_initial_action(self):
        buffer = SonicCausalWindowBuffer(device=torch.device("cpu"), num_envs=1)
        initial_state = torch.zeros(1, 70)
        initial_action = torch.full((1, 29), 3.0)
        buffer.reset(initial_state, initial_action)
        buffer.advance(
            torch.full((1, 29), 4.0),
            torch.full((1, 70), 5.0),
            torch.ones(1, dtype=torch.bool),
        )
        window = buffer.build_condition(
            reference_future=torch.zeros(1, 10, 64),
            state_mean=torch.zeros(70), state_std=torch.ones(70),
            action_mean=torch.zeros(29), action_std=torch.ones(29),
            reference_mean=torch.zeros(64), reference_std=torch.ones(64),
        )
        self.assertEqual(window.batch["state_valid"].tolist(), [[False] * 9 + [True]])
        self.assertEqual(window.batch["action_valid"].tolist(), [[False] * 10])
        self.assertTrue(torch.equal(window.batch["state_history"][0, -1], torch.full((70,), 5.0)))
        self.assertTrue(torch.equal(window.batch["action_history"][0, -1], torch.full((29,), 3.0)))


if __name__ == "__main__":
    unittest.main()

