import unittest
from types import SimpleNamespace

import numpy as np
import torch

from cvae_sa.cvae_diagnostics import epsilon_for
from cvae_sa.latent_sweep65 import (
    POST_ACTION_STEPS,
    REPLAY_ACTION_STEPS,
    TOTAL_ACTION_STEPS,
    TOTAL_STATE_FRAMES,
    _continuation_mode,
    _extend_action_sequence,
    _latent_metrics,
    _output_metrics,
    _pad_recorded_trajectory,
    _selection,
)


class _FixtureRows:
    def __init__(self, rows):
        self._rows = rows

    def manifest(self):
        return list(self._rows)


class LatentSweep65Tests(unittest.TestCase):
    def test_epsilon_is_reproducible_and_seed_sensitive(self):
        model = SimpleNamespace(global_latent_dim=8, local_latent_dim=4)
        batch = {
            "stable_window_id": ["window-a"], "mask_slot": torch.tensor([5]),
            "window_index": torch.tensor([844]), "fixture_index": torch.tensor([0]),
            "episode_ref": ["episode"], "motion_key": ["motion"],
            "variant_id": torch.tensor([4]), "window_start": torch.tensor([0]),
            "physical_state": torch.zeros(1, 65, 70),
        }
        a = epsilon_for(model, batch, 0, 20260923, torch.device("cpu"))
        b = epsilon_for(model, batch, 0, 20260923, torch.device("cpu"))
        c = epsilon_for(model, batch, 0, 20260924, torch.device("cpu"))
        self.assertTrue(torch.equal(a[0], b[0])); self.assertTrue(torch.equal(a[1], b[1]))
        self.assertFalse(torch.equal(a[0], c[0])); self.assertFalse(torch.equal(a[1], c[1]))
        self.assertEqual(tuple(a[0].shape), (1, 8)); self.assertEqual(tuple(a[1].shape), (1, 16, 4))

    def test_selection_contains_hardest_and_two_distinct_motions(self):
        rows = []
        for index, motion in ((844, "jump_right_004__A029"), (2, "motion-b"), (3, "motion-b"), (9, "motion-c"), (10, "motion-c"), (17, "motion-d")):
            rows.append({"window_index": index, "motion_key": motion, "variant_id": 0,
                         "episode_ref": f"e{index}", "window_start": 0,
                         "valid_states": 65, "valid_actions": 64,
                         "stable_window_id": f"id-{index}"})
        selected = _selection(None, [row["window_index"] for row in rows], _FixtureRows(rows),
                              hardest=844, motion="jump_right_004__A029", motion_count=3, seed=20260928)
        self.assertEqual(selected[0]["window_index"], 844)
        self.assertEqual(len({row["motion_key"] for row in selected}), 3)
        self.assertEqual(selected, _selection(None, [], _FixtureRows(rows), hardest=844,
                                              motion="jump_right_004__A029", motion_count=3, seed=20260928))

    def test_metric_shapes_and_finiteness(self):
        rng = np.random.default_rng(2)
        reference = {
            "sampled_global": rng.normal(size=8).astype(np.float32),
            "sampled_local": rng.normal(size=(16, 4)).astype(np.float32),
            "posterior_global_mean": rng.normal(size=8).astype(np.float32),
            "posterior_local_mean": rng.normal(size=(16, 4)).astype(np.float32),
            "posterior_global_std": np.ones(8, dtype=np.float32),
            "posterior_local_std": np.ones((16, 4), dtype=np.float32),
            "predicted_state": rng.normal(size=(65, 70)).astype(np.float32),
            "predicted_action": rng.normal(size=(64, 29)).astype(np.float32),
        }
        sample = {**reference,
                  "sampled_global": reference["sampled_global"] + .1,
                  "sampled_local": reference["sampled_local"] + .1,
                  "predicted_state": reference["predicted_state"] + .1,
                  "predicted_action": reference["predicted_action"] + .1,
                  "truth_state": reference["predicted_state"],
                  "truth_action": reference["predicted_action"]}
        latent = _latent_metrics(sample, reference); output = _output_metrics(sample, reference)
        self.assertEqual(len(latent["local_chunk_rmse"]), 16)
        self.assertEqual(len(output["state"]["per_timestep_rmse"]), 65)
        self.assertTrue(np.isfinite(latent["global"]["rmse"]))
        self.assertTrue(np.isfinite(output["action"]["full_rmse"]))

    def test_t64_replay_preserves_the_source_action_window(self):
        actions = np.arange(REPLAY_ACTION_STEPS * 29, dtype=np.float32).reshape(REPLAY_ACTION_STEPS, 29)
        replay = _extend_action_sequence(actions)
        self.assertEqual(replay.shape, (TOTAL_ACTION_STEPS, 29))
        np.testing.assert_array_equal(replay, actions)
        self.assertEqual(_continuation_mode([actions, actions.copy()]), "none")
        self.assertEqual(POST_ACTION_STEPS, 0)

    def test_recorded_reference_is_already_the_65_frame_video(self):
        trajectory = {
            "dof_pos": np.zeros((REPLAY_ACTION_STEPS + 1, 29), dtype=np.float32),
            "root_pos_w": np.zeros((REPLAY_ACTION_STEPS + 1, 3), dtype=np.float32),
            "root_quat_w": np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (REPLAY_ACTION_STEPS + 1, 1)),
            "total_frames": REPLAY_ACTION_STEPS + 1,
        }
        padded = _pad_recorded_trajectory(trajectory)
        self.assertEqual(padded["dof_pos"].shape, (TOTAL_STATE_FRAMES, 29))
        self.assertEqual(padded["root_pos_w"].shape, (TOTAL_STATE_FRAMES, 3))
        self.assertEqual(padded["total_frames"], TOTAL_STATE_FRAMES)


if __name__ == "__main__":
    unittest.main()
