import unittest

import torch

from dagger.adapters.plain_mlp_student import PlainMlpStudent


class PlainMlpTest(unittest.TestCase):
    def test_observation_dimension_is_explicit(self):
        student = PlainMlpStudent(
            observation_dim=5,
            action_dim=2,
            device=torch.device("cpu"),
            hidden_dims=(8,),
            train_steps=1,
            batch_size=2,
        )
        observation = {
            "actor_obs": torch.zeros(2, 3),
            "tokenizer": torch.zeros(2, 1),
            "student_reference_features": torch.zeros(2, 2),
        }
        self.assertEqual(tuple(student.observe(observation).shape), (2, 5))


if __name__ == "__main__":
    unittest.main()
