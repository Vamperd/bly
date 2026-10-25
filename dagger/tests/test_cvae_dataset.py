import unittest

import torch

from dagger.core.cvae_dataset import CvaeDaggerDataset, CvaeDaggerRound
from dagger.core.metrics import estimate_teacher_identifiability


def _round(count=3):
    return CvaeDaggerRound({
        "physical_state": torch.zeros(count, 65, 70),
        "action": torch.zeros(count, 64, 29),
        "valid_state": torch.ones(count, 65, dtype=torch.bool),
        "valid_action": torch.ones(count, 64, dtype=torch.bool),
        "state_mask": torch.zeros(count, 65, 70, dtype=torch.bool),
        "action_mask": torch.zeros(count, 64, 29, dtype=torch.bool),
        "loss_state_mask": torch.ones(count, 65, 70, dtype=torch.bool),
        "loss_action_mask": torch.ones(count, 64, 29, dtype=torch.bool),
        "teacher_action": torch.zeros(count, 29),
        "student_action": torch.zeros(count, 29),
    })


class CvaeDatasetTest(unittest.TestCase):
    def test_round_balanced_sample_shape(self):
        dataset = CvaeDaggerDataset()
        dataset.append(_round(2))
        dataset.append(_round(4))
        sample = dataset.sample(5, generator=torch.Generator().manual_seed(3))
        self.assertEqual(dataset.count, 6)
        self.assertEqual(tuple(sample["physical_state"].shape), (5, 65, 70))
        self.assertEqual(tuple(sample["teacher_action"].shape), (5, 29))

    def test_information_probe_detects_conflicting_teacher_labels(self):
        dataset = CvaeDaggerDataset()
        data = _round(2)
        data.tensors["teacher_action"][1].fill_(1.0)
        dataset.append(data)
        result = estimate_teacher_identifiability(dataset)
        self.assertTrue(result["available"])
        self.assertTrue(result["teacher_not_identifiable"])


if __name__ == "__main__":
    unittest.main()

