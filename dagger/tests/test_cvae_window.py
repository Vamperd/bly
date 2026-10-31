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

    def test_reference_condition_shows_future_states_hides_contacts_and_actions(self):
        buffer = CvaeWindowBuffer(device=torch.device("cpu"), num_envs=2)
        buffer.reset(torch.zeros(2, 70))
        reference = torch.full((2, 32, 70), 3.0)
        window = buffer.build_condition(
            state_mean=torch.zeros(70), state_std=torch.ones(70),
            action_mean=torch.zeros(29), action_std=torch.ones(29),
            reference_future=reference,
        )
        future = window.batch["physical_state"][:, ANCHOR + 1:]
        self.assertTrue(torch.all(future[..., :68] == 3.0))
        self.assertTrue(torch.all(future[..., 68:] == 0.0))
        self.assertFalse(bool(window.state_mask[:, :, :68].any()))
        self.assertTrue(torch.all(window.state_mask[:, ANCHOR + 1:, 68:]))
        self.assertTrue(torch.all(window.action_mask[:, ANCHOR:]))
        self.assertFalse(bool(window.action_mask[:, :ANCHOR].any()))
        self.assertEqual(int(window.loss_action_mask.sum()), 2 * 29)
        self.assertFalse(bool(window.loss_state_mask.any()))

    def test_reference_training_window_matches_condition_layout(self):
        states = torch.arange(60 * 70, dtype=torch.float32).reshape(60, 70)
        actions = torch.zeros(59, 29)
        reference = torch.full((32, 70), -2.0)
        sample = build_training_window(
            states, actions, torch.ones(29), torch.zeros(29), step=40,
            state_mean=torch.zeros(70), state_std=torch.ones(70),
            action_mean=torch.zeros(29), action_std=torch.ones(29),
            reference_future=reference,
        )
        future = sample["physical_state"][ANCHOR + 1:]
        self.assertTrue(torch.all(future[:, :68] == -2.0))
        # Real contacts stay as hidden targets while the simulator rollout exists.
        self.assertTrue(torch.equal(future[:19, 68:], states[41:60, 68:]))
        self.assertTrue(torch.all(future[19:, 68:] == 0.0))
        self.assertTrue(torch.all(sample["valid_state"][ANCHOR + 1:]))
        self.assertFalse(bool(sample["state_mask"][:, :68].any()))
        self.assertTrue(torch.all(sample["state_mask"][ANCHOR + 1:, 68:]))
        self.assertFalse(bool(sample["loss_state_mask"][:, :68].any()))
        self.assertTrue(torch.all(sample["loss_state_mask"][ANCHOR + 1:ANCHOR + 20, 68:]))
        self.assertFalse(bool(sample["loss_state_mask"][ANCHOR + 20:].any()))
        self.assertEqual(int(sample["loss_action_mask"].sum()), 29)


if __name__ == "__main__":
    unittest.main()

