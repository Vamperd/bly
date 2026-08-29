from __future__ import annotations

import unittest

import numpy as np

from cvae_sa.schema import raw_action_to_relative, resolve_parameter
from cvae_sa.physics_schema import structured_robot_information, validate_physics_schema


class SchemaTest(unittest.TestCase):
    def test_v5_reference_contract_requires_runtime_provenance_and_ten_offsets(self) -> None:
        schema = {
            "schema_version": "sonic_physics_sa_v5",
            "dimensions": {"state": 70, "action": 29},
            "storage": {"state_tp1_duplicate": False},
            "joint_names": [f"joint_{index}" for index in range(29)],
            "action_term_type": "JointPositionAction",
            "wrapper_action_transform_enabled": False,
            "simulation": {"sim_dt": 0.005, "control_dt": 0.02, "decimation": 4},
            "reference_future": {
                "frames": 10,
                "joint_pos_vel_dimension": 58,
                "root_orientation_dimension": 6,
                "time_offsets_seconds": [0.1 * index for index in range(10)],
                "source": "command_manager_runtime_observation",
            },
        }
        validate_physics_schema(schema)
        invalid = {**schema, "reference_future": {
            **schema["reference_future"], "source": "motion_id_reconstruction",
        }}
        with self.assertRaisesRegex(ValueError, "deployable reference contract"):
            validate_physics_schema(invalid)
        invalid_offsets = {**schema, "reference_future": {
            **schema["reference_future"], "time_offsets_seconds": [0.0] * 9,
        }}
        with self.assertRaisesRegex(ValueError, "ten finite"):
            validate_physics_schema(invalid_offsets)

    def test_resolves_per_environment_and_maps_clipped_action(self) -> None:
        schema = {
            "action_scale": {"scope": "global", "values": [2.0] * 29},
            "action_offset": {
                "scope": "per_environment",
                "values": [[0.0] * 29, [1.0] * 29],
            },
            "default_joint_pos": {"scope": "global", "values": [0.5] * 29},
            "action_clip": {
                "scope": "global",
                "values": [[-1.0, 1.5]] * 29,
            },
        }
        actions = np.array([[1.0] * 29], dtype=np.float32)
        relative, scale = raw_action_to_relative(actions, schema, env_id=1)
        np.testing.assert_allclose(scale, 2.0)
        np.testing.assert_allclose(relative, 1.0)
        np.testing.assert_allclose(
            resolve_parameter(schema["action_offset"], 1, (29,)), 1.0
        )

    def test_structured_robot_information_excludes_raw_action_scale(self) -> None:
        names = [f"joint_{index}" for index in range(29)]
        schema = {
            "joint_names": names,
            "nominal_default_joint_pos": {"scope": "global", "values": [0.1] * 29},
            "action_scale": {"scope": "global", "values": [999.0] * 29},
            "actuator_groups": {
                "all": {
                    "type": "ImplicitActuator",
                    "joint_names": names,
                    "min_delay": 1,
                    "max_delay": 2,
                }
            },
            "simulation": {
                "sim_dt": 0.005,
                "control_dt": 0.02,
                "decimation": 4,
                "gravity_w": [0.0, 0.0, -9.81],
                "solver_position_iteration_count": 8,
                "solver_velocity_iteration_count": 4,
            },
            "contact": {"threshold_n": 10.0},
        }
        context = {
            "joint_position_limits": np.tile([-2.0, 2.0], (29, 1)),
            "joint_velocity_limits": np.ones(29) * 30,
            "joint_effort_limits": np.ones(29) * 50,
            "joint_stiffness": np.ones(29) * 40,
            "joint_damping": np.ones(29) * 2,
            "joint_armature": np.ones(29) * 0.01,
            "joint_friction": np.zeros(29),
        }
        joint, actuator, global_info = structured_robot_information(
            schema, context, 0, {"unknown": 0, "ImplicitActuator": 1}
        )
        self.assertEqual(joint.shape, (29, 11))
        self.assertEqual(actuator.shape, (29,))
        self.assertEqual(global_info.shape, (9,))
        np.testing.assert_allclose(joint[:, 0], 0.1)
        np.testing.assert_allclose(joint[:, 9:], np.tile([1.0, 2.0], (29, 1)))
        self.assertFalse(np.any(joint == 999.0))


if __name__ == "__main__":
    unittest.main()
