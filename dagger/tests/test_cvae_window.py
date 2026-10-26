import unittest

import torch

from dagger.runtime.cvae_window import ANCHOR, CvaeWindowBuffer, build_training_window


class CvaeWindowTest(unittest.TestCase):
    def test_rollout_window_hides_future_but_keeps_current(self):
        buffer = CvaeWindowBuffer(device=torch.device("cpu"), num_envs=1)
        buffer.reset(torch.zeros(1, 70))
        window = buffer.build_condition(
            state_mean=torch.zeros(70), state_std=torch.ones(70),
            action_mean=torch.zeros(29), action_std=torch.ones(29),
        )
        self.assertEqual(tuple(window.batch["physical_state"].shape), (1, 65, 70))
        self.assertEqual(tuple(window.batch["action"].shape), (1, 64, 29))
        self.assertTrue(torch.all(window.state_mask[:, ANCHOR + 1:]))
        self.assertTrue(torch.all(window.action_mask[:, ANCHOR:]))
        self.assertFalse(bool(window.state_mask[:, ANCHOR].any()))
        self.assertEqual(int(window.loss_action_mask.sum()), 29)

    def test_done_history_can_be_preserved_for_continuation(self):
        buffer = CvaeWindowBuffer(device=torch.device("cpu"), num_envs=1)
        buffer.reset(torch.zeros(1, 70))
        buffer.advance(
            torch.ones(1, 29),
            torch.ones(1, 70),
            torch.tensor([True]),
            clear_on_done=False,
        )
        self.assertEqual(int(buffer.history_steps().item()), 1)
        buffer.advance(
            torch.ones(1, 29),
            torch.ones(1, 70),
            torch.tensor([True]),
            clear_on_done=True,
        )
        self.assertEqual(int(buffer.history_steps().item()), 0)

    def test_training_target_retains_future_values(self):
        states = torch.arange(100 * 70, dtype=torch.float32).reshape(100, 70)
        actions = torch.zeros(99, 29)
        sample = build_training_window(
            states, actions, torch.ones(29), torch.zeros(29), step=50,
            state_mean=torch.zeros(70), state_std=torch.ones(70),
            action_mean=torch.zeros(29), action_std=torch.ones(29),
        )
        self.assertEqual(tuple(sample["physical_state"].shape), (65, 70))
        self.assertTrue(torch.all(sample["state_mask"][ANCHOR + 1:]))
        # The target remains available for the training loss even though the
        # condition encoder will mask it.
        self.assertTrue(torch.equal(sample["physical_state"][ANCHOR + 1], states[51]))
        self.assertEqual(int(sample["loss_action_mask"].sum()), 29)
        self.assertTrue(torch.allclose(sample["teacher_action"], torch.ones(29)))


if __name__ == "__main__":
    unittest.main()

