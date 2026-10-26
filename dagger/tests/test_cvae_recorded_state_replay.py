import unittest

import numpy as np
import torch

from dagger.cvae_recorded_state_replay import (
    ACTION_STEPS,
    HORIZON_ACTIONS,
    PREFIX_ACTIONS,
    STATE_STEPS,
    _initial_window,
    _receding_window,
)


class RecordedStateReplayTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.mean_state = torch.zeros(70)
        self.std_state = torch.ones(70)
        self.mean_action = torch.zeros(29)
        self.std_action = torch.ones(29)
        self.source = np.zeros((HORIZON_ACTIONS + 1, 70), dtype=np.float32)
        self.source[:, 67] = 0.8
        self.initial = np.zeros((HORIZON_ACTIONS + 1, 70), dtype=np.float32)
        self.actions = [np.zeros(29, dtype=np.float32) for _ in range(HORIZON_ACTIONS)]

    def test_initial_window_masks_every_action(self):
        batch, state_mask, action_mask, _, _ = _initial_window(
            self.source,
            self.mean_state,
            self.std_state,
            self.mean_action,
            self.std_action,
            self.device,
        )
        self.assertEqual(tuple(batch["physical_state"].shape), (1, STATE_STEPS, 70))
        self.assertEqual(tuple(batch["action"].shape), (1, ACTION_STEPS, 29))
        self.assertFalse(bool(state_mask.any()))
        self.assertTrue(bool(action_mask.all()))

    def test_receding_window_uses_33_real_and_32_expected(self):
        batch, state_mask, action_mask, _, _ = _receding_window(
            self.initial.tolist(),
            self.source,
            self.actions,
            PREFIX_ACTIONS,
            self.mean_state,
            self.std_state,
            self.mean_action,
            self.std_action,
            self.device,
        )
        self.assertEqual(tuple(batch["physical_state"].shape), (1, STATE_STEPS, 70))
        self.assertEqual(tuple(batch["action"].shape), (1, ACTION_STEPS, 29))
        self.assertFalse(bool(state_mask.any()))
        self.assertTrue(bool(action_mask[:, :PREFIX_ACTIONS].any() == False))
        self.assertTrue(bool(action_mask[:, PREFIX_ACTIONS:].all()))

    def test_receding_window_rejects_invalid_action_index(self):
        with self.assertRaises(ValueError):
            _receding_window(
                self.initial.tolist(),
                self.source,
                self.actions,
                HORIZON_ACTIONS,
                self.mean_state,
                self.std_state,
                self.mean_action,
                self.std_action,
                self.device,
            )


if __name__ == "__main__":
    unittest.main()
