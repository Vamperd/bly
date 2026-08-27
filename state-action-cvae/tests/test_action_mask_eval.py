from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from cvae_sa.action_mask_eval import (
    _dataset_representation,
    _load_source,
    _make_batch,
    _mask_batch,
    _post_mask_metrics,
    _predicted_action,
    _quaternion_error_degrees,
    _scan_window_starts,
)


class ActionMaskEvaluationTest(unittest.TestCase):
    def test_identical_float32_quaternion_has_exactly_zero_error(self) -> None:
        quaternion = np.asarray(
            [[0.70710677, 0.0, 0.70710677, 0.0]], dtype=np.float32
        )
        error = _quaternion_error_degrees(quaternion, quaternion.copy())
        self.assertEqual(float(error[0]), 0.0)

    def test_dataset_representation_auto_detects_legacy_and_physics_v4(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifests").mkdir()
            manifest = root / "manifests/dataset_manifest.json"
            manifest.write_text(
                json.dumps({"schema_version": "sonic_state_action_cvae_dataset_v1"}),
                encoding="utf-8",
            )
            self.assertFalse(_dataset_representation(root)[1])
            manifest.write_text(
                json.dumps(
                    {"schema_version": "sonic_physics_state_action_cvae_dataset_v4"}
                ),
                encoding="utf-8",
            )
            self.assertTrue(_dataset_representation(root)[1])

    def test_action_mask_hides_next_previous_action_without_state_mask(self) -> None:
        mask = np.zeros((128, 29), dtype=bool)
        mask[20:28] = True
        masks = _mask_batch(mask, "completion", torch.device("cpu"))
        self.assertFalse(masks.state_input.any().item())
        self.assertTrue(masks.action_input[0, 20:28].all().item())
        self.assertTrue(masks.previous_input[0, 21:29].all().item())
        self.assertFalse(masks.previous_loss.any().item())
        self.assertTrue(masks.action_loss.equal(masks.action_input))

    def test_stride_scan_preserves_margins_and_includes_last_window(self) -> None:
        starts = _scan_window_starts(400)
        self.assertEqual(starts[0], 32)
        self.assertEqual(starts[-1], 400 - 128 - 32)
        self.assertTrue(all(next_value > value for value, next_value in zip(starts, starts[1:])))

    def test_physics_v4_mask_has_empty_previous_action_and_inverse_transitions(self) -> None:
        mask = np.ones((128, 29), dtype=bool)
        masks = _mask_batch(
            mask, "inverse", torch.device("cpu"), physics_v4=True
        )
        self.assertEqual(tuple(masks.state_input.shape), (1, 129, 70))
        self.assertEqual(tuple(masks.previous_input.shape), (1, 129, 0))
        self.assertFalse(masks.state_input.any().item())
        self.assertTrue(masks.inverse_transition.all().item())

    def test_physics_v4_uses_dedicated_inverse_head(self) -> None:
        reconstruction = torch.zeros((1, 128, 29))
        inverse = torch.ones((1, 128, 29))
        output = SimpleNamespace(action=reconstruction, inverse_action=inverse)
        self.assertIs(_predicted_action(output, "inverse", True), inverse)
        self.assertIs(_predicted_action(output, "completion", True), reconstruction)

    def test_physics_v4_source_and_batch_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.npz"
            steps = 200
            action_values = np.repeat(
                (np.arange(steps, dtype=np.float32) * 0.001)[:, None], 29, axis=1
            )
            np.savez(
                source_path,
                physics_state_v3=np.zeros((steps + 1, 70), dtype=np.float32),
                previous_action_rel=np.zeros((steps + 1, 29), dtype=np.float32),
                raw_action=action_values,
                action_rel=action_values,
                action_target_canonical=action_values,
                joint_names=np.asarray([f"joint_{index}" for index in range(29)]),
                action_default=np.zeros(29, dtype=np.float32),
                nominal_default_joint_pos=np.zeros(29, dtype=np.float32),
                action_scale=np.ones(29, dtype=np.float32),
                action_offset=np.zeros(29, dtype=np.float32),
                action_clip=np.empty((0,), dtype=np.float32),
                wrapper_action_clip=np.float32(np.nan),
                control_dt=np.float32(0.02),
                initial_processed_target_canonical=np.zeros(29, dtype=np.float32),
                joint_robot_information=np.zeros((29, 11), dtype=np.float32),
                joint_actuator_type_names=np.asarray(["ImplicitActuator"] * 29),
                global_robot_information=np.zeros(9, dtype=np.float32),
                dynamics_context=np.zeros(648, dtype=np.float32),
            )
            (root / "manifests").mkdir()
            (root / "data").mkdir()
            (root / "manifests/dataset_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "sonic_physics_state_action_cvae_dataset_v4",
                        "representations": {
                            "joint_actuator_type": {
                                "vocabulary": ["unknown", "ImplicitActuator"]
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            np.savez(
                root / "data/normalization.npz",
                physical_state_mean=np.zeros(70, dtype=np.float32),
                physical_state_std=np.ones(70, dtype=np.float32),
                action_mean=np.zeros(29, dtype=np.float32),
                action_std=np.ones(29, dtype=np.float32),
                joint_robot_info_mean=np.zeros(11, dtype=np.float32),
                joint_robot_info_std=np.ones(11, dtype=np.float32),
                global_robot_info_mean=np.zeros(9, dtype=np.float32),
                global_robot_info_std=np.ones(9, dtype=np.float32),
                dynamics_context_mean=np.zeros(648, dtype=np.float32),
                dynamics_context_std=np.ones(648, dtype=np.float32),
            )
            source = _load_source(source_path, physics_v4=True)
            source["joint_actuator_type_names"] = (
                "UnseenActuator",
                *source["joint_actuator_type_names"][1:],
            )
            batch, _ = _make_batch(
                source, root, 32, torch.device("cpu"), physics_v4=True
            )
            self.assertEqual(tuple(batch["physical_state"].shape), (1, 129, 70))
            self.assertEqual(tuple(batch["action"].shape), (1, 128, 29))
            self.assertEqual(tuple(batch["previous_action"].shape), (1, 129, 0))
            self.assertEqual(tuple(batch["joint_robot_information"].shape), (1, 29, 11))
            self.assertEqual(tuple(batch["global_robot_information"].shape), (1, 9))
            self.assertEqual(tuple(batch["dynamics_context"].shape), (1, 648))
            self.assertEqual(int(batch["joint_actuator_type"][0, 0]), 0)
            self.assertTrue((batch["joint_actuator_type"][0, 1:] == 1).all().item())
            self.assertTrue(
                torch.allclose(
                    batch["action_before_window"], torch.full((1, 29), 0.031)
                )
            )

    def test_post_mask_metrics_reports_controlled_joint_drift(self) -> None:
        states = 80
        reference = {
            "joint_pos": np.zeros((states, 29), dtype=np.float32),
            "joint_vel": np.zeros((states, 29), dtype=np.float32),
            "root_pos": np.tile(np.array([[0.0, 0.0, 0.8]], dtype=np.float32), (states, 1)),
            "root_quat": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (states, 1)),
            "body_pos": np.zeros((states, 30, 3), dtype=np.float32),
            "physical_state": np.tile(
                np.concatenate((np.zeros(61), np.array([0.0, 0.0, -1.0])))[None],
                (states, 1),
            ).astype(np.float32),
        }
        value = {name: array.copy() for name, array in reference.items()}
        value["joint_pos"][21:, 0] = np.linspace(0.0, 0.2, states - 21)
        metrics = _post_mask_metrics(reference, value, first_mask_action=20, control_dt=0.02)
        self.assertGreater(metrics["joint_error_growth_rate_rad_s"], 0.0)
        self.assertFalse(metrics["fell_relative_to_original"])
        self.assertIsNotNone(metrics["post_mask_0_5_seconds"])

    def test_physics_v4_metrics_include_base_height_and_contact(self) -> None:
        states = 20
        physical = np.zeros((states, 70), dtype=np.float32)
        physical[:, 66] = -1.0
        physical[:, 67] = 0.8
        physical[:, 68:70] = 1.0
        reference = {
            "joint_pos": np.zeros((states, 29), dtype=np.float32),
            "joint_vel": np.zeros((states, 29), dtype=np.float32),
            "root_pos": np.tile(np.array([[0.0, 0.0, 0.8]], dtype=np.float32), (states, 1)),
            "root_quat": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (states, 1)),
            "body_pos": np.zeros((states, 30, 3), dtype=np.float32),
            "physical_state": physical,
        }
        value = {name: array.copy() for name, array in reference.items()}
        value["physical_state"][:, 58] = 0.2
        value["physical_state"][:, 67] = 0.75
        value["physical_state"][:, 68] = 0.0
        metrics = _post_mask_metrics(reference, value, 0, 0.02)[
            "post_mask_remaining_episode"
        ]
        self.assertEqual(metrics["physical_state_dimension"], 70)
        self.assertGreater(metrics["base_linear_velocity_rmse_m_s"], 0.0)
        self.assertGreater(metrics["base_height_rmse_m"], 0.0)
        self.assertEqual(metrics["foot_contact_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
