from pathlib import Path
import unittest


class StateActionIsolationTests(unittest.TestCase):
    def test_action_evaluator_does_not_depend_on_state_video_modules(self) -> None:
        root = Path(__file__).resolve().parents[1]
        action_source = (root / "src/cvae_sa/action_mask_eval.py").read_text(encoding="utf-8")
        self.assertNotIn("state_mask_eval", action_source)
        self.assertNotIn("state_masks", action_source)

    def test_shell_entrypoints_have_disjoint_environment_namespaces(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "cvae_repro.sh").read_text(encoding="utf-8")
        action_start = script.index("validate_action_mask_replay()")
        state_start = script.index("validate_state_mask_video()")
        action_body = script[action_start:state_start]
        state_body = script[state_start:script.index("ORIGINAL_ARGS=")]
        self.assertNotIn("CVAE_STATE_", action_body)
        self.assertNotIn("completed_actions.npz", state_body)
        self.assertNotIn("cvae_action_mask_replay.ok", state_body)


if __name__ == "__main__":
    unittest.main()

