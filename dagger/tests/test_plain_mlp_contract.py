import tempfile
import unittest
from pathlib import Path

import torch

from dagger.adapters.plain_mlp_student import PlainMlpStudent
from dagger.core.contracts import DaggerRound
from dagger.core.dataset import AggregatedDataset
from dagger.core.loop import DaggerLoop
from dagger.core.metrics import same_motion_memorization_pass
from dagger.core.motion_selection import select_motion_keys


def _round(observations: torch.Tensor, actions: torch.Tensor) -> DaggerRound:
    count = int(observations.shape[0])
    return DaggerRound(
        student_obs=observations,
        teacher_action=actions,
        student_action=torch.zeros_like(actions),
        executed_action=torch.zeros_like(actions),
        done=torch.zeros(count, dtype=torch.bool),
        episode_id=torch.arange(count),
        timestep=torch.zeros(count, dtype=torch.long),
        motion_id=torch.zeros(count, dtype=torch.long),
        mixture_teacher=torch.ones(count, dtype=torch.bool),
    )


class PlainMlpContractTest(unittest.TestCase):
    def _student(self) -> PlainMlpStudent:
        return PlainMlpStudent(
            observation_dim=3,
            action_dim=2,
            device=torch.device("cpu"),
            hidden_dims=(8, 4),
            train_steps=1,
            batch_size=2,
            input_terms=("actor_obs", "reference"),
        )

    def test_observe_returns_raw_values(self):
        student = self._student()
        observation = {
            "actor_obs": torch.tensor([[1.0, 2.0]]),
            "student_reference_features": torch.tensor([[3.0]]),
        }
        value = student.observe(observation)
        self.assertTrue(torch.equal(value, torch.tensor([[1.0, 2.0, 3.0]])))

    def test_normalization_is_applied_once_across_rounds(self):
        student = self._student()
        dataset = AggregatedDataset()
        first_observations = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        first_actions = torch.zeros(2, 2)
        dataset.append(_round(first_observations, first_actions))
        student.train_on_dataset(dataset)
        self.assertTrue(torch.allclose(student._mean, first_observations.mean(dim=0)))
        self.assertTrue(torch.allclose(student._std, first_observations.std(dim=0)))

        second_observations = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        dataset.append(_round(second_observations, torch.zeros(2, 2)))
        combined = torch.cat((first_observations, second_observations), dim=0)
        student.train_on_dataset(dataset)
        self.assertTrue(torch.allclose(student._mean, combined.mean(dim=0)))
        self.assertTrue(torch.allclose(student._std, combined.std(dim=0)))

    def test_frozen_normalization_stays_fixed(self):
        student = PlainMlpStudent(
            observation_dim=3,
            action_dim=2,
            device=torch.device("cpu"),
            hidden_dims=(8, 4),
            train_steps=1,
            batch_size=2,
            freeze_normalization=True,
        )
        dataset = AggregatedDataset()
        dataset.append(_round(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), torch.zeros(2, 2)))
        student.train_on_dataset(dataset)
        mean_before = student._mean.clone()
        std_before = student._std.clone()
        dataset.append(_round(torch.tensor([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]), torch.zeros(2, 2)))
        student.train_on_dataset(dataset)
        self.assertTrue(torch.equal(student._mean, mean_before))
        self.assertTrue(torch.equal(student._std, std_before))

    def test_checkpoint_round_trip(self):
        student = self._student()
        dataset = AggregatedDataset()
        observations = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dataset.append(_round(observations, torch.ones(2, 2)))
        student.train_on_dataset(dataset)
        student.motion_keys = ["motion_a", "motion_b"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "student.pt"
            student.save(str(path))
            restored = self._student()
            payload = restored.load(path)
        self.assertEqual(payload["format_version"], "dagger_plain_mlp_student_checkpoint_v1")
        for key, value in student.model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.model.state_dict()[key]))
        self.assertTrue(torch.allclose(student._mean, restored._mean))
        self.assertTrue(torch.allclose(student._std, restored._std))
        self.assertEqual(restored.input_terms, ("actor_obs", "reference"))
        self.assertEqual(restored.motion_keys, ["motion_a", "motion_b"])

    def test_action_bounds_clip(self):
        loop = DaggerLoop.__new__(DaggerLoop)
        loop.action_low = torch.tensor([[-1.0, 0.0]])
        loop.action_high = torch.tensor([[1.0, 2.0]])
        loop.action_clip_value = None
        clipped = loop._clip(torch.tensor([[-5.0, 3.0]]))
        self.assertTrue(torch.equal(clipped, torch.tensor([[-1.0, 2.0]])))

        loop.action_low = None
        loop.action_high = None
        loop.action_clip_value = 2.0
        clipped = loop._clip(torch.tensor([[-5.0, 3.0]]))
        self.assertTrue(torch.equal(clipped, torch.tensor([[-2.0, 2.0]])))


class _FakeMotion:
    def __init__(self, motion_ids: torch.Tensor) -> None:
        self.motion_ids = motion_ids


class _FakeCommandManager:
    def __init__(self, motion_ids: torch.Tensor) -> None:
        self._motion = _FakeMotion(motion_ids)

    def get_term(self, name: str):
        if name != "motion":
            raise KeyError(name)
        return self._motion


class _FakeEnv:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.num_envs = 2
        self.motion_keys = ["motion_a", "motion_b"]
        self.command_manager = _FakeCommandManager(torch.tensor([0, 1]))
        self.steps = 0
        self.action_space = type(
            "Space", (), {"low": torch.tensor([-1.0]), "high": torch.tensor([1.0])}
        )()

    def reset(self):
        return {
            "actor_obs": torch.zeros(self.num_envs, 2),
            "student_reference_features": torch.zeros(self.num_envs, 1),
        }

    def step(self, actions):
        self.steps += 1
        observation = self.reset()
        return observation, torch.zeros(self.num_envs), torch.zeros(self.num_envs, dtype=torch.bool), {}


class _FakeTeacher:
    def reset(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)

    def act(self, observation, done):
        return torch.zeros(observation["actor_obs"].shape[0], 1)


class EvaluationLoopTest(unittest.TestCase):
    def test_evaluation_reports_teacher_rmse_by_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            env = _FakeEnv()
            student = PlainMlpStudent(
                observation_dim=3,
                action_dim=1,
                device=torch.device("cpu"),
                hidden_dims=(4,),
            )
            loop = DaggerLoop(
                env=env,
                teacher=_FakeTeacher(),
                student=student,
                num_envs=2,
                rollout_steps=1,
                action_clip_value=None,
                run_dir=directory,
                seed=0,
                action_low=torch.tensor([[-1.0]]),
                action_high=torch.tensor([[1.0]]),
            )
            metrics = loop.evaluate_student_only(2)
        self.assertIsNotNone(metrics["teacher_action_rmse"])
        self.assertEqual(set(metrics["teacher_action_rmse_by_motion"]), {"motion_a", "motion_b"})
        self.assertEqual(metrics["num_envs"], 2.0)


class MotionSelectionTest(unittest.TestCase):
    def test_selection_is_deterministic_and_excludes_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("motion_a.pkl", "motion_b.pkl", "motion_c.pkl", "metadata.pkl"):
                (root / name).write_bytes(b"")
            first = select_motion_keys(root, 2, 7)
            second = select_motion_keys(root, 2, 7)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 2)
            self.assertNotIn("metadata", first)

    def test_selection_rejects_too_many(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "motion_a.pkl").write_bytes(b"")
            with self.assertRaises(ValueError):
                select_motion_keys(root, 2, 7)


class MemorizationGateTest(unittest.TestCase):
    def _metrics(self, final_scale: float = 0.4) -> list[dict]:
        first = {
            "teacher_action_rmse": 1.0,
            "teacher_action_rmse_by_motion": {
                f"m{index}": {"teacher_action_rmse": 1.0} for index in range(4)
            },
            "termination_rate": 0.2,
            "action_saturation_ratio": 0.01,
        }
        last = {
            "teacher_action_rmse": final_scale,
            "teacher_action_rmse_by_motion": {
                f"m{index}": {"teacher_action_rmse": final_scale} for index in range(4)
            },
            "termination_rate": 0.1,
            "action_saturation_ratio": 0.01,
        }
        return [{"student_only": first}, {"student_only": last}]

    def test_memorization_pass(self):
        self.assertTrue(same_motion_memorization_pass(self._metrics()))
        self.assertFalse(same_motion_memorization_pass(self._metrics(final_scale=0.8)))


if __name__ == "__main__":
    unittest.main()
