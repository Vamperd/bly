import unittest

import torch

from dagger.core.contracts import DaggerRound


class CoreContractTest(unittest.TestCase):
    def test_round_shape_contract(self):
        round_data = DaggerRound(
            student_obs=torch.zeros(2, 3),
            teacher_action=torch.zeros(2, 1),
            student_action=torch.zeros(2, 1),
            executed_action=torch.zeros(2, 1),
            done=torch.zeros(2, dtype=torch.bool),
            episode_id=torch.zeros(2, dtype=torch.long),
            timestep=torch.zeros(2, dtype=torch.long),
            motion_id=torch.zeros(2, dtype=torch.long),
            mixture_teacher=torch.ones(2, dtype=torch.bool),
        )
        self.assertEqual(round_data.count, 2)


if __name__ == "__main__":
    unittest.main()
