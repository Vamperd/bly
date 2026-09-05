from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from cvae_sa.posterior_capacity import FIXED_MASK_NAMES, MaskBankDataset
from cvae_sa.posterior_capacity_tail import (
    EXPECTED_PARAMETERS,
    FORMAT_VERSION,
    MARKER_NAME,
    PARTIAL_MASK_NAMES,
    classify_tail_diagnostic,
    distribution,
    evaluate_tail,
    state_feature_labels,
    validate_f4a_checkpoint,
    validate_source_reproduction,
    write_tail_artifacts,
)


class _ListDataset(torch.utils.data.Dataset[dict[str, object]]):
    def __init__(self, items: list[dict[str, object]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.items[index]


class _OffsetModel(torch.nn.Module):
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        state_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del state_mask, action_mask
        state = batch["physical_state"].clone()
        state[..., :68] += 0.02
        contact = batch["physical_state"][..., 68:70]
        logits = contact * 20.0 - 10.0
        return SimpleNamespace(
            physical_state=state,
            action=batch["action"] + 0.02,
            state_contact_logits=logits,
        )


def _mask_summary(value: float) -> dict[str, object]:
    return {
        "state_rmse": distribution([value]),
        "action_rmse": distribution([value]),
        "combined_rmse": distribution([value]),
        "max_abs": distribution([value]),
    }


class PosteriorCapacityTailTest(unittest.TestCase):
    def test_distribution_and_feature_labels_are_explicit(self) -> None:
        values = distribution([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(values["count"], 4)
        self.assertAlmostEqual(values["p50"], 2.5)
        self.assertEqual(distribution([])["p95"], None)
        labels = state_feature_labels([f"joint_{index}" for index in range(29)])
        self.assertEqual(len(labels), 70)
        self.assertEqual(labels[0], "joint_pos_canonical[joint_0]")
        self.assertEqual(labels[29], "joint_vel[joint_0]")
        self.assertEqual(labels[58], "base_lin_vel_robot[x]")
        self.assertEqual(labels[67], "base_height")
        self.assertEqual(labels[68:], ["foot_contact[left]", "foot_contact[right]"])

    def test_f4a_checkpoint_contract_rejects_non_f4d_inputs(self) -> None:
        checkpoint = {
            "format_version": "sonic_posterior_capacity_checkpoint_v1",
            "dataset_manifest_sha256": "dataset-hash",
            "step": 100_000,
            "parameter_count": EXPECTED_PARAMETERS,
            "config": {
                "seed": 20260830,
                "data": {
                    "motion_count": 4,
                    "window_transitions": 128,
                    "max_windows": None,
                },
                "model": {"kind": "physics_posterior_transformer"},
                "training": {
                    "mask_phase": "fixed",
                    "acceptance_gate": "progression",
                    "kl_beta": 0.0,
                },
            },
        }
        result = validate_f4a_checkpoint(checkpoint, "dataset-hash")
        self.assertEqual(result["step"], 100_000)
        altered = copy.deepcopy(checkpoint)
        altered["config"]["data"]["motion_count"] = 32
        with self.assertRaisesRegex(ValueError, "motion_count"):
            validate_f4a_checkpoint(altered, "dataset-hash")
        altered = copy.deepcopy(checkpoint)
        altered["config"]["training"]["kl_beta"] = 0.001
        with self.assertRaisesRegex(ValueError, "kl_disabled"):
            validate_f4a_checkpoint(altered, "dataset-hash")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_f4a_checkpoint(checkpoint, "other-hash")

    def test_tail_classifier_distinguishes_tail_from_broad_failure(self) -> None:
        summaries = {name: _mask_summary(0.005) for name in FIXED_MASK_NAMES}
        summaries["full_both"] = _mask_summary(0.03)
        tail = classify_tail_diagnostic(
            summaries,
            element_exceed_fraction=0.005,
            fixture_exceed_fraction=0.5,
        )
        self.assertEqual(tail["classification"], "tail_objective_mismatch")
        self.assertTrue(tail["global_latent_bottleneck_suspected"])
        broad = copy.deepcopy(summaries)
        broad[PARTIAL_MASK_NAMES[0]] = _mask_summary(0.02)
        classified = classify_tail_diagnostic(
            broad,
            element_exceed_fraction=0.2,
            fixture_exceed_fraction=1.0,
        )
        self.assertEqual(classified["classification"], "broad_reconstruction_failure")
        self.assertTrue(classified["broad_failures"])

    def test_evaluator_records_every_fixture_feature_and_maximum_identity(self) -> None:
        state = torch.zeros(4, 70)
        state[..., 68:70] = torch.tensor([0.0, 1.0])
        item = {
            "physical_state": state,
            "action": torch.zeros(3, 29),
            "valid_state": torch.ones(4, dtype=torch.bool),
            "valid_action": torch.ones(3, dtype=torch.bool),
            "motion_key": "motion-a",
            "variant_id": 0,
            "episode_ref": "/source::episode-a",
            "window_start": 0,
            "source_window_index": 0,
        }
        fixture_data = MaskBankDataset(_ListDataset([item]), len(FIXED_MASK_NAMES))
        result = evaluate_tail(
            _OffsetModel(),
            torch.utils.data.DataLoader(fixture_data, batch_size=10, shuffle=False),
            torch.device("cpu"),
            seed=20260830,
            state_mean=torch.zeros(70).numpy(),
            state_std=torch.ones(70).numpy(),
            action_mean=torch.zeros(29).numpy(),
            action_std=torch.ones(29).numpy(),
            joint_names=[f"joint_{index}" for index in range(29)],
        )
        self.assertEqual(result["global"]["fixture_count"], len(FIXED_MASK_NAMES))
        self.assertAlmostEqual(result["global"]["global_state_rmse"], 0.02, places=6)
        self.assertAlmostEqual(result["global"]["global_action_rmse"], 0.02, places=6)
        self.assertAlmostEqual(result["global"]["worst_max_abs"], 0.02, places=6)
        self.assertEqual(result["global"]["contact_accuracy"], 1.0)
        self.assertEqual(len(result["fixtures"]), len(FIXED_MASK_NAMES))
        self.assertEqual(len(result["features"]), 97)
        self.assertEqual(len(result["windows"]), 1)
        self.assertEqual(result["windows"][0]["fixture_count"], len(FIXED_MASK_NAMES))
        self.assertEqual(len(result["top_worst_windows"]), 1)
        self.assertAlmostEqual(
            result["top_worst_windows"][0]["fixture_max_abs"]["max"], 0.02, places=6
        )
        maximum = result["top_worst_fixtures"][0]["maximum_error"]
        self.assertIn(maximum["domain"], {"state", "action"})
        self.assertIn("feature_name", maximum)
        self.assertIn("absolute_error_physical", maximum)
        self.assertEqual(result["tail_assessment"]["classification"], "broad_reconstruction_failure")

        source = {
            "mask_fixture_count": len(FIXED_MASK_NAMES),
            "best_metrics": {
                "optimizer_step": 7,
                "worst_state_rmse": result["global"]["worst_state_fixture_rmse"],
                "worst_action_rmse": result["global"]["worst_action_fixture_rmse"],
                "continuous_max_abs": result["global"]["worst_max_abs"],
                "contact_accuracy": 1.0,
                "reconstruction_loss": {
                    "state": result["global"]["global_state_rmse"] ** 2,
                    "action": result["global"]["global_action_rmse"] ** 2,
                },
            },
        }
        reproduction = validate_source_reproduction(
            result["global"], source, {"step": 7}
        )
        self.assertTrue(reproduction["passed"])

        artifact_result = {
            "format_version": FORMAT_VERSION,
            "execution_pass": True,
            **result,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            artifacts = write_tail_artifacts(artifact_result, output)
            self.assertTrue((output / "markers" / MARKER_NAME).is_file())
            self.assertEqual(list((output / "checkpoints").iterdir()), [])
            self.assertEqual(
                len((output / "data/posterior_tail_fixture_metrics.jsonl").read_text().splitlines()),
                len(FIXED_MASK_NAMES),
            )
            self.assertEqual(
                len((output / "data/posterior_tail_window_metrics.jsonl").read_text().splitlines()),
                1,
            )
            manifest = json.loads(
                (output / "manifests/posterior_tail_diagnostic.json").read_text()
            )
            self.assertTrue(manifest["execution_pass"])
            svg = Path(artifacts["plot"]).read_text(encoding="utf-8")
            self.assertIn("Value (log10 scale)", svg)
            self.assertIn("progression threshold 1e-2", svg)


if __name__ == "__main__":
    unittest.main()
