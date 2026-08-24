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


if __name__ == "__main__":
    unittest.main()
