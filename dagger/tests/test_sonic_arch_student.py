import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from dagger.adapters.sonic_arch_student import SonicArchStudent, tokenizer_layout

NAMES = ["encoder_index", "command_multi_future_nonflat", "command_z", "motion_anchor_ori_b_mf_nonflat"]
DIMS = {
    "encoder_index": (3,),
    "command_multi_future_nonflat": (2, 3),
    "command_z": (2,),
    "motion_anchor_ori_b_mf_nonflat": (2, 2),
}
REFERENCE = ("command_multi_future_nonflat", "motion_anchor_ori_b_mf_nonflat")
ACTOR_OBS = 4


class FakeActor(nn.Module):
    """Reads only G1 inputs plus encoder_index, like the SONIC g1 path."""

    def __init__(self, layout):
        super().__init__()
        self.layout = layout
        self.net = nn.Sequential(nn.Linear(ACTOR_OBS + 10, 16), nn.SiLU(), nn.Linear(16, 2))

    def forward(self, obs_dict):
        actor_obs, tokenizer = obs_dict["actor_obs"], obs_dict["tokenizer"]
        assert actor_obs.ndim == 3 and tokenizer.shape[-1] == self.layout["total_dim"]
        start = self.layout["encoder_index_slice"][0]
        assert torch.all(tokenizer[..., start] == 1)
        parts = [tokenizer[..., a:b] for a, b in self.layout["reference_slices"]]
        return self.net(torch.cat([actor_obs, *parts], dim=-1))


class FakeDataset:
    def __init__(self, observations, actions):
        self._obs, self._actions, self.rounds = observations, actions, [0]

    def tensors(self):
        return self._obs, self._actions

    def motion_ids(self):
        return torch.zeros(self._obs.shape[0], dtype=torch.long)


def make_student(seed=0, train_steps=200):
    torch.manual_seed(seed)
    layout = tokenizer_layout(NAMES, DIMS, REFERENCE, ["g1", "teleop", "smpl"])
    return SonicArchStudent(
        model=FakeActor(layout),
        layout=layout,
        actor_obs_dim=ACTOR_OBS,
        action_dim=2,
        device=torch.device("cpu"),
        learning_rate=1.0e-2,
        train_steps=train_steps,
        batch_size=32,
    )


def live_observation(batch=5):
    tokenizer = torch.randn(batch, 3 + 6 + 2 + 4)
    tokenizer[:, 0:3] = torch.tensor([1.0, 0.0, 0.0])
    reference = torch.cat([tokenizer[:, 3:9], tokenizer[:, 11:15]], dim=-1)
    return {
        "actor_obs": torch.randn(batch, ACTOR_OBS),
        "tokenizer": tokenizer,
        "student_reference_features": reference,
    }


class SonicArchStudentTest(unittest.TestCase):
    def test_layout_offsets(self):
        layout = tokenizer_layout(NAMES, DIMS, REFERENCE, ["g1", "teleop", "smpl"])
        self.assertEqual(layout["total_dim"], 15)
        self.assertEqual(layout["reference_slices"], [[3, 9], [11, 15]])
        self.assertEqual(layout["encoder_index_slice"], [0, 3])
        self.assertEqual(layout["g1_column"], 0)

    def test_rebuilt_tokenizer_matches_live_g1_path(self):
        student = make_student()
        observation = live_observation()
        rebuilt = student.build_tokenizer(observation["student_reference_features"])
        live = observation["tokenizer"]
        self.assertTrue(torch.equal(rebuilt[:, 3:9], live[:, 3:9]))
        self.assertTrue(torch.equal(rebuilt[:, 11:15], live[:, 11:15]))
        self.assertTrue(torch.equal(rebuilt[:, 0:3], live[:, 0:3]))
        self.assertTrue(torch.all(rebuilt[:, 9:11] == 0))
        full = student.act_full_tokenizer(observation)
        rebuilt_action = student.act(student.observe(observation))
        self.assertLess(float((full - rebuilt_action).abs().max()), 1.0e-6)

    def test_training_reduces_mse_and_checkpoint_round_trips(self):
        student = make_student()
        observations = torch.randn(256, ACTOR_OBS + 10)
        actions = torch.tanh(observations[:, :2])
        dataset = FakeDataset(observations, actions)
        before = float(torch.mean((student.act(observations) - actions).square()))
        metrics = student.train_on_dataset(dataset)
        self.assertLess(metrics["teacher_action_rmse"] ** 2, before)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "round_00.pt"
            student.save(str(path), extra_metadata={"round_index": 0})
            other = make_student(seed=1)
            payload = other.load(path)
            self.assertEqual(payload["normalization_mode"], "none")
            self.assertTrue(torch.equal(other.act(observations), student.act(observations)))
            other.layout = dict(other.layout, total_dim=99)
            with self.assertRaises(ValueError):
                other.load(path)

    def test_rejects_wrong_input_dim(self):
        student = make_student()
        observation = live_observation()
        observation["student_reference_features"] = observation["student_reference_features"][:, :3]
        with self.assertRaises(ValueError):
            student.observe(observation)


if __name__ == "__main__":
    unittest.main()
