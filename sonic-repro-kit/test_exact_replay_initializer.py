from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from exact_replay_initializer import (
    FORMAT_VERSION,
    _array_digest,
    load_initialization,
)


def _valid_payload() -> dict[str, np.ndarray]:
    body_count = 3
    values = {
        "joint_names": np.asarray([f"joint_{index}" for index in range(29)]),
        "joint_pos": np.zeros(29, dtype=np.float32),
        "joint_vel": np.zeros(29, dtype=np.float32),
        "root_pos_relative": np.zeros(3, dtype=np.float32),
        "root_quat_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "root_lin_vel_world": np.zeros(3, dtype=np.float32),
        "root_ang_vel_world": np.zeros(3, dtype=np.float32),
        "body_pos_relative": np.zeros((body_count, 3), dtype=np.float32),
        "physics_state_v3": np.zeros(70, dtype=np.float32),
        "nominal_default_joint_pos": np.zeros(29, dtype=np.float32),
        "runtime_default_joint_pos": np.zeros(29, dtype=np.float32),
        "action_scale": np.ones(29, dtype=np.float32),
        "action_offset": np.zeros(29, dtype=np.float32),
        "action_clip": np.empty((0,), dtype=np.float32),
        "wrapper_action_clip": np.asarray(np.nan, dtype=np.float32),
        "joint_position_limits": np.zeros((29, 2), dtype=np.float32),
        "joint_velocity_limits": np.ones(29, dtype=np.float32),
        "joint_effort_limits": np.ones(29, dtype=np.float32),
        "joint_stiffness": np.ones(29, dtype=np.float32),
        "joint_damping": np.ones(29, dtype=np.float32),
        "joint_armature": np.ones(29, dtype=np.float32),
        "joint_friction": np.ones(29, dtype=np.float32),
        "body_mass": np.ones(body_count, dtype=np.float32),
        "body_inertia": np.ones((body_count, 9), dtype=np.float32),
        "body_com": np.ones((body_count, 7), dtype=np.float32),
        "body_material": np.ones((4, 3), dtype=np.float32),
        "ground_material": np.ones(3, dtype=np.float32),
        "previous_raw_action": np.zeros(29, dtype=np.float32),
        "previous_processed_action": np.zeros(29, dtype=np.float32),
        "initial_joint_target_abs": np.zeros(29, dtype=np.float32),
        "format_version": np.asarray(FORMAT_VERSION),
    }
    values["payload_sha256"] = np.asarray(_array_digest(values))
    return values


class ExactReplayInitializationTest(unittest.TestCase):
    def test_load_accepts_absent_wrapper_clip_and_validates_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "initialization.npz"
            values = _valid_payload()
            np.savez_compressed(path, **values)
            loaded, digest = load_initialization(path)
            self.assertTrue(np.isnan(loaded["wrapper_action_clip"]))
            self.assertEqual(digest, str(values["payload_sha256"].tolist()))

    def test_corrupted_initialization_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "initialization.npz"
            values = _valid_payload()
            values["joint_pos"] = np.ones(29, dtype=np.float32)
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_initialization(path)

    def test_runtime_hook_restores_state_dynamics_and_action_history_before_readback(self) -> None:
        source = (Path(__file__).resolve().parent / "exact_replay_initializer.py").read_text(
            encoding="utf-8"
        )
        required_operations = (
            "write_joint_position_limit_to_sim",
            "write_joint_velocity_limit_to_sim",
            "write_joint_effort_limit_to_sim",
            "write_joint_stiffness_to_sim",
            "write_joint_damping_to_sim",
            "write_joint_armature_to_sim",
            "write_joint_friction_coefficient_to_sim",
            "root_physx_view.set_masses",
            "root_physx_view.set_inertias",
            "root_physx_view.set_coms",
            "root_physx_view.set_material_properties",
            "_set_ground_material",
            "exactReplayPhysicsMaterial",
            "get_first_matching_child_prim",
            "bind_physics_material",
            "write_root_state_to_sim",
            "write_joint_state_to_sim",
            "term.process_actions(previous_raw)",
            "raw.action_manager._action[:] = previous_raw",
            "raw.action_manager._prev_action[:] = previous_raw",
            "set_joint_position_target",
            "raw.scene.write_data_to_sim()",
            "raw.sim.forward()",
        )
        for operation in required_operations:
            with self.subTest(operation=operation):
                self.assertIn(operation, source)
        self.assertLess(source.index("write_root_state_to_sim"), source.index("raw.sim.forward()"))
        self.assertLess(source.index("term.process_actions(previous_raw)"), source.index("raw.sim.forward()"))
        self.assertLess(source.index("raw.sim.forward()"), source.index('"application_complete": True'))
        self.assertNotIn(
            'ground physics material prim is missing: {prim_path}',
            source,
        )


if __name__ == "__main__":
    unittest.main()
