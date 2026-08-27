from __future__ import annotations

from pathlib import Path
import unittest


class ActionMaskReplayConfigTest(unittest.TestCase):
    def test_source_and_replay_force_the_same_flat_terrain(self) -> None:
        script = (Path(__file__).resolve().parent / "sonic_repro.sh").read_text(
            encoding="utf-8"
        )
        source_start = script.index("phase_capture_action_mask_source()")
        replay_start = script.index("phase_replay_action_mask()")
        render_start = script.index("phase_render_action_mask()")

        source_phase = script[source_start:replay_start]
        replay_phase = script[replay_start:render_start]
        override = "++manager_env.config.terrain_type=plane"

        self.assertEqual(source_phase.count(override), 1)
        self.assertEqual(replay_phase.count(override), 1)

    def test_replay_runs_each_scenario_as_one_isolated_environment(self) -> None:
        script = (Path(__file__).resolve().parent / "sonic_repro.sh").read_text(
            encoding="utf-8"
        )
        replay_start = script.index("phase_replay_action_mask()")
        render_start = script.index("phase_render_action_mask()")
        replay_phase = script[replay_start:render_start]

        self.assertIn("prepare_action_replay_slices.py", replay_phase)
        self.assertIn("for ((scenario_index = 0;", replay_phase)
        self.assertIn("++num_envs=1", replay_phase)
        self.assertIn("++manager_env.commands.motion.eval_motion_repeat=1", replay_phase)
        self.assertIn(
            "++manager_env.recorders.trajectory.environment_id_offset=$scenario_index",
            replay_phase,
        )

    def test_recorder_exports_physics_v4_superset(self) -> None:
        recorder = (Path(__file__).resolve().parent / "action_replay_recorder.py").read_text(
            encoding="utf-8"
        )
        for field in (
            '"physics_state_v3"',
            '"action_target_canonical"',
            "nominal_default_joint_pos=",
            "joint_robot_information=",
            "joint_actuator_type_names=",
            "global_robot_information=",
            "dynamics_context=",
        ):
            self.assertIn(field, recorder)


if __name__ == "__main__":
    unittest.main()
