import tempfile
import unittest
from pathlib import Path

import torch

from dagger.adapters.cvae_student import CvaeRollout
from dagger.core.cvae_dataset import CvaeDaggerDataset, CvaeDaggerRound
from dagger.core.cvae_loop import CvaeDaggerLoop
from dagger.core.cvae_objective import (
    compose_dagger_loss,
    cvae_early_stop_decision,
    masked_future_contact_bce,
    masked_future_state_mse,
    teacher_action_mse,
)
from dagger.core.motion_groups import build_motion_groups, select_pilot_keys
from dagger.runtime.cvae_window import (
    ANCHOR,
    CvaeWindow,
    CvaeWindowBuffer,
    causal_masks,
)
from dagger.runtime.sonic_env import SonicActionContract


class MotionGroupTest(unittest.TestCase):
    def test_groups_partition_sorted_keys(self):
        keys = [f"motion_{index:02d}" for index in range(32)]
        groups = build_motion_groups(keys, 8)
        self.assertEqual(len(groups), 4)
        self.assertEqual(sorted(key for group in groups for key in group), sorted(keys))
        self.assertTrue(all(len(group) == 8 for group in groups))

    def test_pilot_selection_is_deterministic_and_prefers_existing_keys(self):
        keys = [f"motion_{index:02d}" for index in range(32)]
        first = select_pilot_keys(keys, count=4, seed=20261024)
        second = select_pilot_keys(keys, count=4, seed=20261024)
        self.assertEqual(first, second)
        preferred = select_pilot_keys(
            keys, count=4, seed=20261024, preferred=["motion_03", "missing"]
        )
        self.assertEqual(preferred[0], "motion_03")
        self.assertEqual(len(preferred), 4)


class CausalMaskTest(unittest.TestCase):
    def test_causal_masks_hide_future_state_and_action(self):
        state_mask, action_mask, loss_state, loss_action = causal_masks(
            2,
            device=torch.device("cpu"),
            valid_state=torch.ones(2, 65, dtype=torch.bool),
            valid_action=torch.ones(2, 64, dtype=torch.bool),
        )
        self.assertTrue(torch.all(state_mask[:, ANCHOR + 1 :]))
        self.assertTrue(torch.all(action_mask[:, ANCHOR:]))
        self.assertFalse(bool(state_mask[:, ANCHOR].any()))
        self.assertTrue(torch.all(loss_state[:, ANCHOR + 1 :]))
        self.assertEqual(int(loss_action.sum()), 2 * 29)
        self.assertTrue(torch.all(loss_action[:, ANCHOR]))

    def test_history_steps_tracks_real_history(self):
        buffer = CvaeWindowBuffer(device=torch.device("cpu"), num_envs=2)
        buffer.reset(torch.zeros(2, 70))
        self.assertEqual(buffer.history_steps().tolist(), [0, 0])
        buffer.advance(
            torch.zeros(2, 29),
            torch.zeros(2, 70),
            torch.zeros(2, dtype=torch.bool),
        )
        self.assertEqual(buffer.history_steps().tolist(), [1, 1])


class MotionWeightedDatasetTest(unittest.TestCase):
    def _round(self, motion_ids):
        count = len(motion_ids)
        return CvaeDaggerRound(
            {
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
                "motion_id": torch.tensor(motion_ids, dtype=torch.long),
            },
            metadata={"motion_keys": ["a", "b"]},
        )

    def test_motion_weights_change_row_sampling(self):
        dataset = CvaeDaggerDataset()
        dataset.append(self._round([0, 1, 0, 1]))
        sample = dataset.sample(
            32,
            generator=torch.Generator().manual_seed(7),
            motion_weights={"a": 100.0, "b": 0.0},
        )
        self.assertTrue(torch.all(sample["motion_id"] == 0))
        self.assertEqual(dataset.motion_coverage(), {"a": 2, "b": 2})

    def test_shard_round_trip_preserves_metadata(self):
        import tempfile
        from pathlib import Path

        dataset = CvaeDaggerDataset()
        dataset.append(self._round([0, 1]))
        with tempfile.TemporaryDirectory() as temporary:
            path = dataset.save_round(0, Path(temporary))
            loaded = CvaeDaggerDataset.load_rounds([path])
        self.assertEqual(loaded.rounds[0].metadata["motion_keys"], ["a", "b"])


class ActionFallbackClipTest(unittest.TestCase):
    def _contract(self):
        contract = SonicActionContract.__new__(SonicActionContract)
        contract.device = torch.device("cpu")
        contract.num_envs = 1
        contract.scale = torch.ones(1, 29)
        contract.offset = torch.zeros(1, 29)
        contract.nominal = torch.zeros(1, 29)
        contract.clip = None
        contract.raw_low = torch.full((1, 29), float("-inf"))
        contract.raw_high = torch.full((1, 29), float("inf"))
        contract.raw_action_clip_value = 20.0
        contract.term = object()
        return contract

    def test_fallback_clip_applies_when_bounds_are_infinite(self):
        contract = self._contract()
        raw = contract.clip_raw(torch.full((1, 29), 100.0))
        self.assertTrue(torch.allclose(raw, torch.full((1, 29), 20.0)))

    def test_canonical_to_raw_returns_finite_achieved_action(self):
        contract = self._contract()
        raw, achieved = contract.canonical_to_raw(torch.full((1, 29), 100.0))
        self.assertTrue(torch.isfinite(raw).all())
        self.assertTrue(torch.isfinite(achieved).all())
        self.assertTrue(torch.all(raw <= 20.0 + 1.0e-6))


class CvaeObjectiveTest(unittest.TestCase):
    def test_teacher_and_masked_objectives(self):
        predicted = torch.zeros(2, 64, 29)
        teacher = torch.ones(2, 29)
        self.assertAlmostEqual(
            float(teacher_action_mse(predicted, teacher, ANCHOR).item()), 1.0
        )
        state = torch.zeros(1, 65, 70)
        target = torch.ones(1, 65, 70)
        mask = torch.zeros(1, 65, 70, dtype=torch.bool)
        mask[:, ANCHOR + 1 :] = True
        self.assertAlmostEqual(
            float(masked_future_state_mse(state, target, mask).item()), 1.0
        )
        logits = torch.zeros(1, 65, 2)
        contact_target = torch.zeros(1, 65, 70)
        contact_target[..., 68:70] = 0.5
        contact_mask = torch.zeros(1, 65, 70, dtype=torch.bool)
        contact_mask[:, ANCHOR + 1 :] = True
        self.assertGreater(
            float(masked_future_contact_bce(logits, contact_target, contact_mask).item()),
            0.0,
        )

    def test_early_stop_passes_with_improvement_and_stable_termination(self):
        decision = cvae_early_stop_decision(
            {"teacher_action_rmse": 0.30, "termination_rate": 0.10},
            {"teacher_action_rmse": 0.29, "termination_rate": 0.10},
            min_teacher_improvement_ratio=0.98,
        )
        self.assertTrue(decision["pass"])
        self.assertTrue(decision["teacher_ok"])
        self.assertTrue(decision["termination_ok"])

    def test_early_stop_fails_without_teacher_improvement(self):
        decision = cvae_early_stop_decision(
            {"teacher_action_rmse": 0.30, "termination_rate": 0.10},
            {"teacher_action_rmse": 0.30, "termination_rate": 0.10},
            min_teacher_improvement_ratio=0.98,
        )
        self.assertFalse(decision["pass"])
        self.assertFalse(decision["teacher_ok"])
        self.assertTrue(decision["termination_ok"])

    def test_early_stop_fails_when_termination_worsens(self):
        decision = cvae_early_stop_decision(
            {"teacher_action_rmse": 0.30, "termination_rate": 0.10},
            {"teacher_action_rmse": 0.20, "termination_rate": 0.20},
            min_teacher_improvement_ratio=0.98,
        )
        self.assertFalse(decision["pass"])
        self.assertTrue(decision["teacher_ok"])
        self.assertFalse(decision["termination_ok"])

    def test_compose_dagger_loss_weights(self):
        zero = torch.zeros(())
        one = torch.ones(())
        loss = compose_dagger_loss(
            future_state=one,
            future_contact=one,
            teacher_standard_normal=one,
            teacher_posterior_mean=one,
            kl=one,
            state_weight=1.0,
            contact_weight=0.1,
            standard_normal_weight=0.5,
            posterior_mean_weight=0.5,
            kl_beta=0.002,
        )
        self.assertAlmostEqual(float(loss.item()), 2.102, places=6)
        self.assertEqual(float(zero.item()), 0.0)


class _FakeMotion:
    def __init__(self, num_envs: int) -> None:
        self.motion_ids = torch.arange(num_envs, dtype=torch.long)
        self.metrics = {
            "error_anchor_pos": torch.zeros(num_envs),
            "error_body_pos": torch.zeros(num_envs),
        }


class _FakeCommandManager:
    def __init__(self, num_envs: int) -> None:
        self._motion = _FakeMotion(num_envs)

    def get_term(self, name: str):
        return self._motion


class _FakeEnv:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.motion_keys = [f"motion_{index}" for index in range(num_envs)]
        self.command_manager = _FakeCommandManager(num_envs)

    def reset(self):
        return {"actor_obs": torch.zeros(self.num_envs, 1)}

    def physics_state(self):
        return torch.zeros(self.num_envs, 70)

    def step(self, actions):
        del actions
        return (
            {"actor_obs": torch.zeros(self.num_envs, 1)},
            torch.zeros(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {},
        )


class _FakeTeacher:
    def reset(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)

    def act(self, observation, done):
        del observation, done
        return torch.zeros(self.num_envs, 29)


class _FakeStudent:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)
        self.latent_mode = "fixed_standard_normal"

    def reset(self, num_envs: int, current_state: torch.Tensor) -> None:
        del current_state
        self.num_envs = int(num_envs)

    def set_latent_mode(self, mode: str) -> None:
        self.latent_mode = str(mode)

    def _window(self) -> CvaeWindow:
        count = self.num_envs
        return CvaeWindow(
            batch={
                "physical_state": torch.zeros(count, 65, 70),
                "action": torch.zeros(count, 64, 29),
                "action_before_window": torch.zeros(count, 29),
                "valid_state": torch.ones(count, 65, dtype=torch.bool),
                "valid_action": torch.ones(count, 64, dtype=torch.bool),
            },
            state_mask=torch.zeros(count, 65, 70, dtype=torch.bool),
            action_mask=torch.zeros(count, 64, 29, dtype=torch.bool),
            loss_state_mask=torch.zeros(count, 65, 70, dtype=torch.bool),
            loss_action_mask=torch.zeros(count, 64, 29, dtype=torch.bool),
            history_steps=torch.zeros(count, dtype=torch.long),
        )

    def observe(self) -> CvaeRollout:
        count = self.num_envs
        return CvaeRollout(
            window=self._window(),
            action_normalized=torch.zeros(count, 29),
            action_canonical=torch.zeros(count, 29),
            action_raw=torch.zeros(count, 29),
            predicted_state=torch.zeros(count, 65, 70),
            predicted_action=torch.zeros(count, 64, 29),
            epsilon_global=torch.zeros(count, 256),
            epsilon_local=torch.zeros(count, 16, 128),
            action_clip_error=torch.zeros(count),
        )

    def after_step(
        self,
        executed_action_canonical,
        next_state,
        done,
        *,
        clear_history_on_done: bool = True,
    ) -> None:
        del executed_action_canonical, next_state, done, clear_history_on_done


def _fake_contract() -> SonicActionContract:
    contract = SonicActionContract.__new__(SonicActionContract)
    contract.device = torch.device("cpu")
    contract.num_envs = 2
    contract.scale = torch.ones(1, 29)
    contract.offset = torch.zeros(1, 29)
    contract.nominal = torch.zeros(1, 29)
    contract.clip = None
    contract.raw_low = torch.full((1, 29), -10.0)
    contract.raw_high = torch.full((1, 29), 10.0)
    contract.raw_action_clip_value = None
    contract.term = object()
    return contract


class CvaeLoopContractTest(unittest.TestCase):
    def _loop(self, directory: str, beta_steps: int = 8) -> CvaeDaggerLoop:
        env = _FakeEnv(2)
        return CvaeDaggerLoop(
            env=env,
            teacher=_FakeTeacher(),
            student=_FakeStudent(2),
            action_contract=_fake_contract(),
            num_envs=2,
            rollout_steps=beta_steps,
            run_dir=Path(directory),
            seed=11,
            state_mean=torch.zeros(70),
            state_std=torch.ones(70),
            action_mean=torch.zeros(29),
            action_std=torch.ones(29),
            motion_group=["motion_0", "motion_1"],
            mixture_mode="per_step",
        )

    def test_record_normalizes_initial_state_to_cpu(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        record = CvaeDaggerLoop._new_record(torch.zeros(70, device=device))
        self.assertEqual(record["states"][0].device.type, "cpu")

    def test_per_step_mixture_records_teacher_and_student(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = self._loop(directory)
            round_data = loop.collect_round(0, 0.0)
            self.assertFalse(bool(round_data.tensors["mixture_teacher"].any()))
            round_data = loop.collect_round(1, 1.0)
            self.assertTrue(bool(round_data.tensors["mixture_teacher"].all()))
            self.assertEqual(round_data.tensors["teacher_action_canonical"].shape[0], 16)
            self.assertEqual(
                round_data.metadata["motion_keys"], ["motion_0", "motion_1"]
            )

    def test_post_fall_continuation_keeps_collecting_after_done(self):
        class _DoneEnv(_FakeEnv):
            def step(self, actions):
                observation, reward, _done, info = super().step(actions)
                return observation, reward, torch.ones_like(_done), info

        with tempfile.TemporaryDirectory() as directory:
            env = _DoneEnv(2)
            loop = CvaeDaggerLoop(
                env=env,
                teacher=_FakeTeacher(),
                student=_FakeStudent(2),
                action_contract=_fake_contract(),
                num_envs=2,
                rollout_steps=4,
                run_dir=Path(directory),
                seed=11,
                state_mean=torch.zeros(70),
                state_std=torch.ones(70),
                action_mean=torch.zeros(29),
                action_std=torch.ones(29),
                motion_group=["motion_0", "motion_1"],
                mixture_mode="per_step",
                post_fall_continuation=True,
                post_fall_baseline_steps=2,
                post_fall_max_error_ratio=3.0,
                post_fall_down_steps=2,
            )
            round_data = loop.collect_round(0, 0.0)
        self.assertEqual(round_data.tensors["teacher_action_canonical"].shape[0], 8)
        self.assertTrue(bool(round_data.tensors["done"].all()))

    def test_evaluation_reports_teacher_and_survival_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = self._loop(directory)
            metrics = loop.evaluate_student_only(4, round_index=0)
        self.assertEqual(metrics["num_envs"], 2.0)
        self.assertEqual(metrics["steps"], 4.0)
        self.assertAlmostEqual(metrics["teacher_action_rmse"], 0.0)
        self.assertEqual(set(metrics["teacher_action_rmse_by_motion"]), {"motion_0", "motion_1"})
        self.assertAlmostEqual(metrics["survival_ratio"], 1.0)
        self.assertEqual(metrics["latent_mode"], "fixed_standard_normal")


if __name__ == "__main__":
    unittest.main()
