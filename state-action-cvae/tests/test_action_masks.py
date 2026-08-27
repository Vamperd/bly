from __future__ import annotations

import unittest

import numpy as np

from cvae_sa.action_masks import (
    add_combined_limb_scenarios,
    build_default_scenarios,
    masked_baselines,
    relative_to_raw_action,
    scenario_mask,
    semantic_joint_groups,
)
from cvae_sa.constants import ACTION_DIM


JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)


class ActionMaskTest(unittest.TestCase):
    def test_default_action_mask_suite_is_complete_and_exact(self) -> None:
        scenarios = build_default_scenarios(32, 60, JOINT_NAMES, 20260824)
        self.assertEqual(len(scenarios), 16)
        masks = {scenario.name: scenario_mask(scenario, JOINT_NAMES) for scenario in scenarios}
        self.assertEqual(masks["element_10"].sum(), round(128 * ACTION_DIM * 0.10))
        self.assertEqual(masks["element_25"].sum(), round(128 * ACTION_DIM * 0.25))
        self.assertEqual(masks["element_40"].sum(), round(128 * ACTION_DIM * 0.40))
        for length in (1, 2, 4, 8):
            self.assertEqual(masks[f"step_contiguous_{length}"].sum(), length * ACTION_DIM)
        self.assertEqual(masks["feature_random_10"].sum(), 128 * round(ACTION_DIM * 0.10))
        self.assertEqual(masks["feature_random_25"].sum(), 128 * round(ACTION_DIM * 0.25))
        self.assertEqual(masks["feature_random_40"].sum(), 128 * round(ACTION_DIM * 0.40))
        self.assertEqual(masks["semantic_waist"].sum(), 128 * 3)
        self.assertEqual(masks["semantic_left_leg"].sum(), 128 * 6)
        self.assertEqual(masks["semantic_right_leg"].sum(), 128 * 6)
        self.assertEqual(masks["semantic_left_arm"].sum(), 128 * 7)
        self.assertEqual(masks["semantic_right_arm"].sum(), 128 * 7)
        self.assertTrue(masks["inverse_full_128"].all())
        inverse = next(item for item in scenarios if item.name == "inverse_full_128")
        self.assertEqual(inverse.distribution_status, "out_of_distribution")
        trained = build_default_scenarios(
            32, 60, JOINT_NAMES, 20260824, inverse_full_in_distribution=True
        )
        trained_inverse = next(
            item for item in trained if item.name == "inverse_full_128"
        )
        self.assertEqual(trained_inverse.distribution_status, "in_distribution")

    def test_semantic_groups_partition_all_joints_and_optional_groups_are_labeled(
        self,
    ) -> None:
        groups = semantic_joint_groups(JOINT_NAMES)
        flattened = [name for values in groups.values() for name in values]
        self.assertEqual(len(flattened), ACTION_DIM)
        self.assertEqual(set(flattened), set(JOINT_NAMES))
        base = build_default_scenarios(32, 60, JOINT_NAMES, 7)
        extended = add_combined_limb_scenarios(base, JOINT_NAMES, 7)
        self.assertEqual(extended[-2].distribution_status, "near_out_of_distribution")
        self.assertEqual(extended[-1].distribution_status, "out_of_distribution")

    def test_masked_baselines_and_action_mapping_preserve_visible_values(self) -> None:
        actions = np.arange(12 * ACTION_DIM, dtype=np.float32).reshape(12, ACTION_DIM) / 100.0
        mask = np.zeros_like(actions, dtype=bool)
        mask[3:6, 0] = True
        hold, linear = masked_baselines(actions, mask)
        np.testing.assert_array_equal(hold[~mask], actions[~mask])
        np.testing.assert_array_equal(linear[~mask], actions[~mask])
        self.assertTrue(np.isclose(hold[4, 0], actions[2, 0]))
        self.assertTrue(actions[2, 0] < linear[4, 0] < actions[6, 0])

        default = np.linspace(-0.2, 0.2, ACTION_DIM, dtype=np.float32)
        scale = np.linspace(0.5, 1.5, ACTION_DIM, dtype=np.float32)
        offset = default.copy()
        clip = np.stack((default - 0.4, default + 0.4), axis=-1)
        relative = np.linspace(-0.6, 0.6, 4 * ACTION_DIM, dtype=np.float32).reshape(
            4, ACTION_DIM
        )
        raw, achieved, saturated = relative_to_raw_action(
            relative, default, scale, offset, clip, wrapper_clip=20.0
        )
        processed = np.clip(raw * scale + offset, clip[:, 0], clip[:, 1]) - default
        np.testing.assert_allclose(processed, achieved, atol=1.0e-6)
        self.assertTrue(saturated.any())


if __name__ == "__main__":
    unittest.main()
