import unittest
from types import SimpleNamespace

import torch

from dagger.core.sonic_causal_loop import SonicCausalDaggerLoop


class _Motion:
    motion_ids = torch.tensor([0])


class _Commands:
    def get_term(self, name):
        if name != "motion":
            raise KeyError(name)
        return _Motion()


class _Env:
    device = torch.device("cpu")
    num_envs = 1
    motion_keys = ["motion_a"]
    command_manager = _Commands()

    def __init__(self):
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return {"student_reference_future": torch.zeros(1, 10, 64), "tokenizer": torch.zeros(1, 1)}

    def physics_state(self):
        return torch.full((1, 70), float(self.step_count))

    def step(self, action):
        del action
        self.step_count += 1
        return self.reset_observation(), torch.zeros(1), torch.zeros(1, dtype=torch.bool), {}

    def reset_observation(self):
        return {"student_reference_future": torch.zeros(1, 10, 64), "tokenizer": torch.zeros(1, 1)}


class _Contract:
    def clip_raw(self, value):
        return value

    def raw_to_canonical(self, value):
        return value

    def canonical_to_raw(self, value):
        return value, value


class _Teacher:
    def reset(self, num_envs):
        self.num_envs = num_envs

    def act(self, observation, done):
        del observation, done
        return torch.ones(1, 29)


class _Student:
    action_mean = torch.zeros(29)
    action_std = torch.ones(29)

    def reset(self, num_envs, state, initial_action=None):
        del num_envs, state, initial_action

    def observe(self, reference):
        del reference
        action = torch.zeros(1, 29)
        return SimpleNamespace(
            action_canonical=action,
            requested_action_canonical=action,
            action_raw=action,
            batch={
                "state_history": torch.zeros(1, 10, 70),
                "action_history": torch.zeros(1, 10, 29),
                "reference_future": torch.zeros(1, 10, 64),
            },
        )

    def after_step(self, action, state, done):
        del action, state, done


class SonicCausalLoopTest(unittest.TestCase):
    def test_collect_round_records_one_action_per_environment_step(self):
        loop = SonicCausalDaggerLoop(
            env=_Env(), teacher=_Teacher(), student=_Student(),
            action_contract=_Contract(), num_envs=1, rollout_steps=3,
            run_dir=".", seed=7,
        )
        samples = loop.collect_round(0, beta=0.0)
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(tuple(row["target_action"].shape) == (29,) for row in samples))
        self.assertTrue(all("executed_action_raw" in row for row in samples))


if __name__ == "__main__":
    unittest.main()
