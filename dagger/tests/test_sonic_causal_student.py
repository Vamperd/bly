import tempfile
import unittest
from pathlib import Path

import torch

from dagger.adapters.sonic_causal_student import SonicCausalStudentAdapter


class SonicCausalStudentContractTest(unittest.TestCase):
    def test_legacy_65_token_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(
                {
                    "format_version": "sonic_65_token_hierarchical_standard_cvae_checkpoint_v2",
                    "model_state_dict": {},
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "SONIC causal CVAE"):
                SonicCausalStudentAdapter(
                    checkpoint=path,
                    device=torch.device("cpu"),
                    action_contract=None,
                )


if __name__ == "__main__":
    unittest.main()
