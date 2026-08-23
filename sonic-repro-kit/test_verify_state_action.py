from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main, mock

import h5py
import numpy as np

import verify_state_action


class VerifyStateActionTest(TestCase):
    def test_resolves_per_environment_parameter(self):
        entry = {
            "scope": "per_environment",
            "values": [[0.0] * 29, [1.0] * 29],
        }
        resolved = verify_state_action.parameter_for_env(entry, 1, (29,))
        np.testing.assert_array_equal(resolved, np.ones(29))

    def _write_fixture(
        self,
        run_dir: Path,
        corrupt_previous_action: bool = False,
        legacy_timeout_schema: bool = False,
    ) -> None:
        (run_dir / "data").mkdir(parents=True)
        (run_dir / "manifests").mkdir()
        actions = np.arange(3 * 29, dtype=np.float32).reshape(3, 29) / 100.0
        previous_action = np.vstack((np.zeros((1, 29), dtype=np.float32), actions[:-1]))
        if corrupt_previous_action:
            previous_action[1, 0] += 1.0

        state_t = {
            "joint_pos": np.arange(3 * 29, dtype=np.float32).reshape(3, 29) / 1000.0,
            "joint_vel": np.arange(3 * 29, dtype=np.float32).reshape(3, 29) / 100.0,
            "base_ang_vel": np.arange(9, dtype=np.float32).reshape(3, 3) / 100.0,
            "gravity_robot": np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (3, 1)),
            "previous_action": previous_action,
        }
        state_tp1 = {
            name: np.concatenate((values[1:], values[-1:]), axis=0)
            for name, values in state_t.items()
        }
        state_tp1["previous_action"] = actions.copy()
        goal_t = {
            "reference_joint_pos": np.zeros((3, 29), dtype=np.float32),
            "reference_joint_vel": np.zeros((3, 29), dtype=np.float32),
            "gravity_reference": np.tile(
                np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (3, 1)
            ),
            "relative_heading": np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (3, 1)),
        }

        with h5py.File(run_dir / "data" / "sonic_minimal_sa.hdf5", "w") as stream:
            data = stream.create_group("data")
            episode = data.create_group("demo_0")
            episode.attrs["num_samples"] = 3
            episode.create_dataset("actions", data=actions)
            for group_name, values_by_name in (
                ("state_t", state_t),
                ("state_tp1", state_tp1),
                ("goal_t", goal_t),
            ):
                group = episode.create_group(group_name)
                for name, values in values_by_name.items():
                    group.create_dataset(name, data=values)
            outcome = episode.create_group("outcome")
            outcome.create_dataset("terminated", data=np.array([False, False, True]))
            outcome.create_dataset("truncated", data=np.array([False, False, False]))
            termination_terms = outcome.create_group("termination_terms")
            for name in ("anchor_pos", "anchor_ori_full", "ee_body_pos"):
                termination_terms.create_dataset(
                    name, data=np.array([False, False, name == "anchor_pos"])
                )
            if not legacy_timeout_schema:
                termination_terms.create_dataset(
                    "foot_pos_xyz", data=np.array([False, False, False])
                )
                termination_terms.create_dataset(
                    "motion_time_out", data=np.array([False, False, False])
                )
            motion = episode.create_group("motion")
            motion.create_dataset("env_id", data=np.zeros(3, dtype=np.int64))
            motion.create_dataset("motion_id", data=np.zeros(3, dtype=np.int64))
            motion.create_dataset("global_motion_id", data=np.zeros(3, dtype=np.int64))
            motion.create_dataset("variant_id", data=np.zeros(3, dtype=np.int64))
            motion.create_dataset("batch_id", data=np.zeros(3, dtype=np.int64))
            motion.create_dataset("attempt_id", data=np.zeros(3, dtype=np.int64))
            motion.create_dataset("motion_step", data=np.arange(3, dtype=np.int64))
            context = episode.create_group("context_t")
            context.create_dataset("reset_root_pose_delta", data=np.zeros((3, 6)))
            context.create_dataset("reset_root_velocity_delta", data=np.zeros((3, 6)))
            context.create_dataset("reset_joint_pos_delta", data=np.zeros((3, 29)))
            context.create_dataset("reset_joint_vel_delta", data=np.zeros((3, 29)))

        schema = {
            "schema_version": "sonic_minimal_sa_v2",
            "dimensions": {"state": 93, "goal": 63, "action": 29},
            "joint_names": [f"joint_{index}" for index in range(29)],
            "default_joint_pos": {"scope": "global", "values": [0.0] * 29},
            "default_joint_vel": {"scope": "global", "values": [0.0] * 29},
            "action_scale": {"scope": "global", "values": [1.0] * 29},
            "action_offset": {"scope": "global", "values": [0.0] * 29},
            "action_clip": None,
            "wrapper_action_clip": 20.0,
            "wrapper_action_transform_enabled": False,
            "action_term_type": "JointPositionAction",
            "control_dt": 0.02,
            "sim_dt": 0.005,
            "motion_id_to_key": {"0": "motion_0"},
            "motion_collection": {
                "eval_motion_repeat": 1,
                "eval_require_full_batch": True,
                "variant_offset": 0,
                "randomization_profile": "startup",
                "env_to_variant": [0],
                "env_to_motion_slot": [0],
            },
            "runtime_physics": {
                "body_com": {
                    "scope": "per_environment",
                    "values": [[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]],
                },
                "material_properties": {
                    "scope": "per_environment",
                    "values": [[[1.0, 1.0, 0.0]]],
                },
            },
            "termination_terms": [
                "time_out",
                "anchor_pos",
                "anchor_ori_full",
                "ee_body_pos",
                "foot_pos_xyz",
            ],
        }
        if not legacy_timeout_schema:
            schema["termination_term_mapping"] = {
                "motion_time_out": "time_out",
                "anchor_pos": "anchor_pos",
                "anchor_ori_full": "anchor_ori_full",
                "ee_body_pos": "ee_body_pos",
                "foot_pos_xyz": "foot_pos_xyz",
            }
        (run_dir / "manifests" / "state_action_schema.json").write_text(
            json.dumps(schema), encoding="utf-8"
        )

    def _run_verifier(self, run_dir: Path) -> int:
        argv = [
            "verify_state_action.py",
            "--run-dir",
            str(run_dir),
            "--expected-motion-count",
            "1",
            "--expected-variants-per-motion",
            "1",
            "--randomization-profile",
            "startup",
        ]
        with mock.patch.object(sys, "argv", argv):
            return verify_state_action.main()

    def test_accepts_aligned_dataset(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_fixture(run_dir)
            self.assertEqual(self._run_verifier(run_dir), 0)

    def test_rejects_shifted_previous_action(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_fixture(run_dir, corrupt_previous_action=True)
            self.assertEqual(self._run_verifier(run_dir), 1)

    def test_accepts_legacy_timeout_schema_without_rewriting_hdf5(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_fixture(run_dir, legacy_timeout_schema=True)
            self.assertEqual(self._run_verifier(run_dir), 0)
            summary = json.loads(
                (run_dir / "manifests" / "collection_summary.json").read_text()
            )
            recovery = summary["legacy_timeout_recovery"]
            self.assertTrue(recovery["applied"])
            self.assertEqual(recovery["episode_count"], 1)
            self.assertEqual(recovery["unrecorded_runtime_terms"], ["foot_pos_xyz"])


if __name__ == "__main__":
    main()
