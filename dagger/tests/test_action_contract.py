import unittest

import torch

from dagger.runtime.sonic_env import SonicActionContract


class ActionContractTest(unittest.TestCase):
    def _contract(self):
        contract = SonicActionContract.__new__(SonicActionContract)
        contract.device = torch.device("cpu")
        contract.num_envs = 1
        contract.scale = torch.full((1, 29), 2.0)
        contract.offset = torch.full((1, 29), 0.25)
        contract.nominal = torch.full((1, 29), -0.5)
        contract.clip = None
        contract.raw_low = torch.full((1, 29), -10.0)
        contract.raw_high = torch.full((1, 29), 10.0)
        contract.raw_action_clip_value = None
        contract.term = object()
        return contract

    def test_canonical_raw_round_trip(self):
        contract = self._contract()
        canonical = torch.full((1, 29), 0.75)
        raw, achieved = contract.canonical_to_raw(canonical)
        self.assertTrue(torch.allclose(raw, torch.zeros(1, 29)))
        self.assertTrue(torch.allclose(achieved, canonical))

    def test_clip_is_explicitly_visible(self):
        contract = self._contract()
        contract.clip = torch.stack((torch.full((1, 29), -0.25), torch.full((1, 29), 0.25)), dim=-1)
        canonical = torch.full((1, 29), 5.0)
        _raw, achieved = contract.canonical_to_raw(canonical)
        self.assertTrue(torch.all(achieved < canonical))


if __name__ == "__main__":
    unittest.main()

