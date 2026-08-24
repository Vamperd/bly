from __future__ import annotations

import unittest

import numpy as np

from cvae_sa.schema import raw_action_to_relative, resolve_parameter


class SchemaTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

