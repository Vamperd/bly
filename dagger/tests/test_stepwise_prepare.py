import unittest
from types import SimpleNamespace

from dagger.stepwise_prepare import _utc_now_iso, _window_args


class StepwisePrepareTest(unittest.TestCase):
    def test_utc_timestamp_is_available(self):
        self.assertIn("+00:00", _utc_now_iso())

    def test_window_args_match_replay65_contract(self):
        args = SimpleNamespace(
            horizon_steps=192,
            simulation_seed=20261023,
            sample_seed=20261023,
        )
        window = _window_args(args, mask_seed=20261016)
        self.assertEqual(window.route, "C")
        self.assertEqual(window.horizon_steps, 192)
        self.assertEqual(window.mask_seed, 20261016)
        self.assertEqual(window.sample_seed, 20261023)
        self.assertEqual(window.masks, [])
        self.assertTrue(window.state_only_continuation)


if __name__ == "__main__":
    unittest.main()
