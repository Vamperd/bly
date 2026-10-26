import unittest

import torch

from dagger.runtime.sonic_env import SonicDaggerEnv


class _ObsConfig:
    group_obs_names = {"tokenizer": ["term_a", "term_b"]}
    group_obs_dims = {"tokenizer": {"term_a": (2,), "term_b": (3,)}}


class _Config:
    obs = _ObsConfig()


class _MotionLib:
    _motion_data_keys = ["motion_a", "motion_b"]


class _Motion:
    motion_lib = _MotionLib()
    motion_ids = torch.tensor([0])
    time_steps = torch.tensor([5])
    motion_start_time_steps = torch.tensor([9])
    motion_num_steps = torch.tensor([10], dtype=torch.int32)

    def set_motion_state(self, motion_ids, time_steps, motion_start_time_steps=None):
        self.motion_ids = motion_ids
        self.time_steps = time_steps
        self.motion_start_time_steps = (
            motion_start_time_steps
            if motion_start_time_steps is not None
            else torch.zeros_like(motion_ids)
        )


class _CommandManager:
    def __init__(self):
        self.motion = _Motion()

    def get_term(self, name):
        if name != "motion":
            raise KeyError(name)
        return self.motion


class _Space:
    low = [-1.0, -1.0]
    high = [1.0, 1.0]


class _Env:
    device = torch.device("cpu")
    num_envs = 1
    config = _Config()
    action_space = _Space()
    command_manager = _CommandManager()
    closed = False

    def close(self):
        self.closed = True


class _Wrapper:
    device = torch.device("cpu")
    num_envs = 1
    config = _Config()
    action_space = _Space()
    command_manager = _CommandManager()

    def __init__(self):
        self.env = _Env()


class SonicEnvContractTest(unittest.TestCase):
    def test_reference_term_selection_and_motion_keys(self):
        env = SonicDaggerEnv(_Env(), reference_terms=("term_b",))
        tokenizer = torch.arange(5, dtype=torch.float32).reshape(1, 5)
        augmented = env._augment({"actor_obs": torch.zeros(1, 2), "tokenizer": tokenizer})
        self.assertTrue(torch.equal(augmented["student_reference_features"], torch.tensor([[2.0, 3.0, 4.0]])))
        self.assertEqual(env.motion_keys, ["motion_a", "motion_b"])
        self.assertTrue(torch.equal(env.action_low, torch.tensor([[-1.0, -1.0]])))
        self.assertTrue(torch.equal(env.action_high, torch.tensor([[1.0, 1.0]])))

    def test_close_delegates_to_raw_environment(self):
        raw = _Env()
        env = SonicDaggerEnv(raw, reference_terms=("term_b",))
        env.close()
        self.assertTrue(raw.closed)

    def test_close_delegates_through_wrapper(self):
        wrapper = _Wrapper()
        env = SonicDaggerEnv(wrapper, reference_terms=("term_b",))
        env.close()
        self.assertTrue(wrapper.env.closed)

    def test_reset_motion_to_first_frame(self):
        env = SonicDaggerEnv(_Env(), reference_terms=("term_b",))
        env.reset_motion_to_first_frame()
        motion = env.motion_term
        self.assertEqual(motion.time_steps.tolist(), [0])
        self.assertEqual(motion.motion_start_time_steps.tolist(), [0])
        self.assertEqual(motion.motion_num_steps.dtype, torch.int32)

    def test_missing_reference_term_is_reported(self):
        with self.assertRaises(KeyError):
            SonicDaggerEnv(_Env(), reference_terms=("missing_term",))


if __name__ == "__main__":
    unittest.main()
