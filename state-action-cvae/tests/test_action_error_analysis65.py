from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cvae_sa.action_error_analysis65 import analyze


class ActionErrorAnalysisTests(unittest.TestCase):
    def test_reports_physical_and_percentage_action_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            (dataset / "manifests").mkdir(parents=True)
            (dataset / "data").mkdir()
            names = [f"joint_{i}" for i in range(29)]
            (dataset / "manifests" / "dataset_manifest.json").write_text(
                json.dumps({"joint_names": names}), encoding="utf-8"
            )
            mean = np.zeros(29, dtype=np.float32)
            std = np.full(29, 2.0, dtype=np.float32)
            np.savez(dataset / "data" / "normalization.npz", action_mean=mean, action_std=std)

            evaluation = root / "evaluation"
            traces = evaluation / "all_predictions"
            traces.mkdir(parents=True)
            (evaluation.parent / "data").mkdir()
            np.savez(evaluation.parent / "data" / "normalization.npz", action_mean=mean, action_std=std)
            target = np.zeros((2, 29), dtype=float)
            target[0, 0] = 1.0
            prediction = target.copy()
            prediction[0, 0] += 0.2
            trace = {
                "identity": {"fixture_index": 0},
                "domains": {"action": {
                    "physical_prediction": prediction.tolist(),
                    "physical_target": target.tolist(),
                    "mask": [[True] + [False] * 28, [False] * 29],
                    "valid": [True, True],
                }},
            }
            (traces / "fixture_0.json").write_text(json.dumps(trace), encoding="utf-8")
            output = root / "output"
            summary = analyze(evaluation, dataset, output)

            self.assertEqual(summary["action_dimension"], 29)
            joint = summary["per_joint"][0]
            self.assertAlmostEqual(joint["splits"]["full"]["mae_rad"], 0.1)
            self.assertAlmostEqual(joint["splits"]["full"]["rmse_over_dataset_std_pct"], 7.071067811865475)
            self.assertEqual(joint["splits"]["masked"]["count"], 1)
            self.assertTrue((output / "action_per_joint.csv").is_file())
            self.assertTrue((output / "action_joint_percentages.svg").is_file())
            self.assertTrue((output / "action_time_joint_abs_error.svg").is_file())


if __name__ == "__main__":
    unittest.main()
