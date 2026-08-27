from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from cvae_sa.models import build_model
from cvae_sa.state_mask_eval import (
    _hold_last,
    _linear_interpolation,
    completion_mask_batch,
    reconstruct_root_trajectory,
    segmented_forward_rollout,
)
from cvae_sa.state_masks import build_default_scenarios, scenario_mask


JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


class FakeRolloutModel:
    def __call__(self, batch, masks, sample_from_prior, deterministic):
        del sample_from_prior, deterministic
        start = int(masks.rollout_start[0])
        current = batch["physical_state"][:, start]
        values = torch.stack([current + float(index + 1) for index in range(8)], dim=1)
        return SimpleNamespace(rollout_state=values)


class StateMaskTests(unittest.TestCase):
    @staticmethod
    def _physics_batch() -> dict[str, torch.Tensor]:
        state = torch.randn(1, 129, 70)
        state[..., 64:67] = torch.nn.functional.normalize(state[..., 64:67], dim=-1)
        state[..., 68:70] = torch.randint(0, 2, (1, 129, 2)).float()
        return {
            "physical_state": state,
            "previous_action": torch.empty(1, 129, 0),
            "action": torch.randn(1, 128, 29),
            "action_before_window": torch.randn(1, 29),
            "action_scale": torch.ones(1, 29),
            "robot_information": torch.randn(1, 293),
            "joint_robot_information": torch.randn(1, 29, 11),
            "joint_actuator_type": torch.zeros(1, 29, dtype=torch.long),
            "global_robot_information": torch.randn(1, 9),
            "dynamics_context": torch.randn(1, 648),
            "auxiliary_transition": torch.zeros(1, 128, 35),
            "valid_state": torch.ones(1, 129, dtype=torch.bool),
            "valid_action": torch.ones(1, 128, dtype=torch.bool),
            "progress": torch.linspace(0, 1, 129)[None],
        }

    @staticmethod
    def _physics_model() -> torch.nn.Module:
        return build_model(
            {
                "kind": "physics_transformer", "d_model": 32,
                "encoder_layers": 1, "decoder_layers": 1, "heads": 4,
                "ffn_dim": 64, "latent_dim": 8, "dropout": 0.0,
                "joint_width": 16, "context_mode": "hidden",
                "joint_robot_info_dim": 11, "global_robot_info_dim": 9,
                "actuator_type_count": 1, "dynamics_context_dim": 648,
                "auxiliary_dim": 35,
            }
        ).eval()

    def test_default_masks_only_target_state(self) -> None:
        scenarios = build_default_scenarios(32, 40, JOINT_NAMES, 20260830)
        self.assertIn("forward_rollout_32", [item.name for item in scenarios])
        semantic = next(item for item in scenarios if item.name == "semantic_left_leg")
        mask = scenario_mask(semantic, JOINT_NAMES)
        self.assertEqual(mask.shape, (129, 70))
        self.assertEqual(int(mask[0].sum()), 12)
        masks = completion_mask_batch(mask, torch.device("cpu"))
        self.assertFalse(bool(masks.action_input.any()))
        self.assertEqual(tuple(masks.previous_input.shape), (1, 129, 0))

    def test_segmented_32_step_rollout_does_not_modify_input(self) -> None:
        state = torch.zeros((1, 129, 70))
        batch = {
            "physical_state": state,
            "action": torch.zeros((1, 128, 29)),
        }
        result = segmented_forward_rollout(FakeRolloutModel(), batch, 32, 32)
        self.assertEqual(tuple(result.shape), (1, 32, 70))
        self.assertTrue(torch.all(result[:, -1] == 32.0))
        self.assertTrue(torch.all(batch["physical_state"] == 0.0))

    def test_real_physics_rollout_is_finite_and_cannot_read_future_state(self) -> None:
        torch.manual_seed(7)
        model = self._physics_model()
        batch = self._physics_batch()
        original = segmented_forward_rollout(model, batch, 32, 8)
        changed = dict(batch)
        changed["physical_state"] = batch["physical_state"].clone()
        changed["physical_state"][:, 33:] += 100.0
        second = segmented_forward_rollout(model, changed, 32, 8)
        self.assertTrue(torch.isfinite(original).all())
        self.assertTrue(torch.allclose(original, second, atol=1e-6))

    def test_constant_velocity_root_integration(self) -> None:
        states = np.zeros((11, 70), dtype=np.float32)
        states[:, 58] = 1.0
        states[:, 64:67] = (0.0, 0.0, -1.0)
        states[:, 67] = 0.8
        root_pos = np.zeros((11, 3), dtype=np.float32)
        root_pos[:, 2] = 0.8
        root_quat = np.zeros((11, 4), dtype=np.float32)
        root_quat[:, 0] = 1.0
        position, quaternion = reconstruct_root_trajectory(
            states, root_pos, root_quat, anchor=0, control_dt=0.02
        )
        np.testing.assert_allclose(position[:, 0], np.arange(11) * 0.02, atol=1e-6)
        np.testing.assert_allclose(position[:, 1:], root_pos[:, 1:], atol=1e-6)
        np.testing.assert_allclose(quaternion[:, 0], 1.0, atol=1e-6)

    def test_yaw_integration_uses_body_angular_velocity(self) -> None:
        states = np.zeros((6, 70), dtype=np.float32)
        states[:, 63] = 1.0
        states[:, 64:67] = (0.0, 0.0, -1.0)
        states[:, 67] = 0.8
        root_pos = np.zeros((6, 3), dtype=np.float32)
        root_pos[:, 2] = 0.8
        root_quat = np.zeros((6, 4), dtype=np.float32)
        root_quat[:, 0] = 1.0
        _, quaternion = reconstruct_root_trajectory(
            states, root_pos, root_quat, anchor=0, control_dt=0.02
        )
        yaw = 2.0 * np.arctan2(quaternion[:, 3], quaternion[:, 0])
        np.testing.assert_allclose(yaw, np.arange(6) * 0.02, atol=1e-5)

    def test_feature_baselines_use_only_visible_boundaries(self) -> None:
        truth = np.arange(15, dtype=np.float32).reshape(5, 3)
        mask = np.zeros_like(truth, dtype=bool)
        mask[:, 1] = True
        before = np.asarray((-1.0, 10.0, -1.0), dtype=np.float32)
        after = np.asarray((99.0, 22.0, 99.0), dtype=np.float32)
        hold = _hold_last(truth, mask, before)
        linear = _linear_interpolation(truth, mask, before, after)
        np.testing.assert_allclose(hold[:, 1], 10.0)
        self.assertFalse(np.array_equal(linear[:, 1], truth[:, 1]))
        np.testing.assert_allclose(linear[~mask], truth[~mask])


if __name__ == "__main__":
    unittest.main()
