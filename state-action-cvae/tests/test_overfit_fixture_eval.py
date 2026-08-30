from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from cvae_sa.masking import MaskBatch
from cvae_sa.overfit_fixture_eval import (
    accumulate_metric_contributions,
    classify_fixture_result,
    diagnose_overfit_fixture,
    fixture_sha256,
    mask_batches_equal,
    parse_checkpoint_kinds,
    sample_mask_sha256,
    sample_metric_contributions,
    validate_checkpoint_fixture_identity,
    write_fixture_artifacts,
)


class OverfitFixtureEvalTest(unittest.TestCase):
    def _batch_and_masks(self) -> tuple[dict, MaskBatch]:
        batch = {
            "physical_state": torch.arange(16, dtype=torch.float32).reshape(2, 4, 2),
            "action": torch.arange(18, dtype=torch.float32).reshape(2, 3, 3),
            "episode_ref": ["run::episode_a", "run::episode_b"],
            "window_start": torch.tensor([0, 64]),
        }
        state_mask = torch.zeros(2, 4, 2, dtype=torch.bool)
        state_mask[:, 1:3] = True
        action_mask = torch.zeros(2, 3, 3, dtype=torch.bool)
        action_mask[:, 1] = True
        transition = torch.zeros(2, 3, dtype=torch.bool)
        transition[:, 0:2] = True
        masks = MaskBatch(
            state_mask.clone(),
            torch.empty(2, 4, 0, dtype=torch.bool),
            action_mask.clone(),
            state_mask.clone(),
            torch.empty(2, 4, 0, dtype=torch.bool),
            action_mask.clone(),
            0,
            "forward_rollout",
            "mixed",
            True,
            transition.clone(),
            transition.clone(),
            transition.clone(),
            torch.zeros(2, dtype=torch.long),
            torch.full((2,), 2, dtype=torch.long),
        )
        return batch, masks

    def test_checkpoint_kind_parser_is_strict(self) -> None:
        self.assertEqual(parse_checkpoint_kinds("best,last"), ("best", "last"))
        self.assertEqual(parse_checkpoint_kinds("last"), ("last",))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse_checkpoint_kinds("best,parent")
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_checkpoint_kinds("best,best")
        self.assertEqual(
            classify_fixture_result(True, False),
            "exact_memory_pass_unseen_mask_fail",
        )

    def test_fixture_hash_is_sample_identity_stable_and_order_independent(self) -> None:
        batch, masks = self._batch_and_masks()
        first_key, first_hash = sample_mask_sha256(batch, masks, 0)
        second_key, second_hash = sample_mask_sha256(batch, masks, 1)
        reversed_batch = {
            "physical_state": batch["physical_state"].flip(0),
            "action": batch["action"].flip(0),
            "episode_ref": list(reversed(batch["episode_ref"])),
            "window_start": batch["window_start"].flip(0),
        }
        reversed_masks = MaskBatch(
            *(getattr(masks, name).flip(0) for name in (
                "state_input", "previous_input", "action_input",
                "state_loss", "previous_loss", "action_loss",
            )),
            masks.task_id,
            masks.task_name,
            masks.completion_name,
            masks.causal,
            *(getattr(masks, name).flip(0) for name in (
                "forward_transition", "inverse_transition",
                "history_action_transition", "rollout_start", "rollout_horizon",
            )),
        )
        reversed_key, reversed_hash = sample_mask_sha256(
            reversed_batch, reversed_masks, 1
        )
        self.assertEqual((first_key, first_hash), (reversed_key, reversed_hash))
        self.assertEqual(
            fixture_sha256({first_key: first_hash, second_key: second_hash}),
            fixture_sha256({second_key: second_hash, first_key: first_hash}),
        )
        self.assertTrue(mask_batches_equal(masks, masks))
        altered_masks = MaskBatch(**{
            name: getattr(masks, name) for name in masks.__dataclass_fields__
        })
        altered_masks.state_input = masks.state_input.clone()
        altered_masks.state_input[0, 0, 0] = ~altered_masks.state_input[0, 0, 0]
        self.assertFalse(mask_batches_equal(masks, altered_masks))

    def test_exact_metric_routing_uses_each_task_specific_head(self) -> None:
        batch, masks = self._batch_and_masks()
        target_delta = batch["physical_state"][:, 1:] - batch["physical_state"][:, :-1]
        output = SimpleNamespace(
            forward_delta=target_delta + 1.0,
            rollout_state=batch["physical_state"][:, 1:3] + 1.0,
            inverse_action=batch["action"] + 1.0,
            history_action=batch["action"] + 1.0,
            physical_state=batch["physical_state"] + 1.0,
            action=batch["action"] + 1.0,
        )
        expected = {
            "forward_rollout": {
                "forward_one_normalized_rmse",
                "forward_rollout_8_normalized_rmse",
            },
            "inverse": {"action_inverse_local_normalized_rmse"},
            "history_action": {"history_action_normalized_rmse"},
            "arbitrary_state": {"arbitrary_state_normalized_rmse"},
            "arbitrary_action": {"action_completion_macro_normalized_rmse"},
        }
        for task, keys in expected.items():
            contributions = sample_metric_contributions(
                task, output, batch, masks, 0
            )
            self.assertEqual(set(contributions), keys)
            for squared, count in contributions.values():
                self.assertGreater(count, 0)
                self.assertAlmostEqual((squared / count) ** 0.5, 1.0)

    def test_zero_target_action_window_is_retained_but_not_aggregated(self) -> None:
        batch, masks = self._batch_and_masks()
        masks.action_loss[0] = False
        output = SimpleNamespace(action=batch["action"] + 1.0)
        batch_contributions: dict[str, list[tuple[float, int]]] = {}
        window_rmses: dict[str, list[float]] = {}
        squared_errors: dict[str, float] = {}
        element_counts: dict[str, int] = {}
        targetless_counts: dict[str, int] = {}

        empty = sample_metric_contributions(
            "arbitrary_action", output, batch, masks, 0
        )
        record, targetless = accumulate_metric_contributions(
            empty,
            batch_contributions,
            window_rmses,
            squared_errors,
            element_counts,
            targetless_counts,
        )
        metric = "action_completion_macro_normalized_rmse"
        self.assertEqual(record, {})
        self.assertEqual(targetless, [metric])
        self.assertEqual(targetless_counts[metric], 1)
        self.assertEqual(window_rmses[metric], [])

        observed = sample_metric_contributions(
            "arbitrary_action", output, batch, masks, 1
        )
        record, targetless = accumulate_metric_contributions(
            observed,
            batch_contributions,
            window_rmses,
            squared_errors,
            element_counts,
            targetless_counts,
        )
        self.assertEqual(targetless, [])
        self.assertAlmostEqual(record[metric]["normalized_rmse"], 1.0)
        self.assertEqual(record[metric]["target_available"], True)
        self.assertEqual(len(window_rmses[metric]), 1)
        self.assertGreater(element_counts[metric], 0)

    def test_best_and_last_must_cover_the_same_complete_fixture(self) -> None:
        results = {
            kind: {
                "exact_training_fixture": {
                    "fixture_sha256": "same",
                    "window_count": 1248,
                }
            }
            for kind in ("best", "last")
        }
        self.assertEqual(
            validate_checkpoint_fixture_identity(results, 1248), "same"
        )
        results["last"]["exact_training_fixture"]["fixture_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            validate_checkpoint_fixture_identity(results, 1248)
        results["last"]["exact_training_fixture"]["fixture_sha256"] = "same"
        results["last"]["exact_training_fixture"]["window_count"] = 1247
        with self.assertRaisesRegex(ValueError, "every fixed window"):
            validate_checkpoint_fixture_identity(results, 1248)

    def test_quality_failure_still_writes_execution_marker_and_artifacts(self) -> None:
        checkpoint = {
            "exact_training_fixture": {"exact_score": 4.0, "exact_pass": False},
            "unseen_mask_diagnostic": {"unseen_score": 5.0, "unseen_pass": False},
        }
        result = {
            "execution_pass": True,
            "checkpoint_kinds": ["best", "last"],
            "quality_summary": {
                "best_all_compact_gate_tasks_exact_pass": False,
                "last_all_compact_gate_tasks_exact_pass": False,
            },
            "runs": [{
                "task_mode": "arbitrary_action",
                "checkpoints": {"best": checkpoint, "last": checkpoint},
            }],
            "interpretations": [
                {
                    "task_mode": "arbitrary_action",
                    "checkpoint_kind": kind,
                    "classification": "exact_and_unseen_both_fail",
                    "objective_metric_gap_flag": False,
                }
                for kind in ("best", "last")
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_fixture_artifacts(
                result,
                output,
                [{"window_key": "run::episode|0", "checkpoint_kind": "best"}],
            )
            self.assertTrue(
                (output / "markers/cvae_overfit_fixture_diagnostic.ok").is_file()
            )
            self.assertTrue((output / "videos/exact_vs_unseen.svg").is_file())
            self.assertTrue((output / "videos/best_vs_last.svg").is_file())
            saved = json.loads(
                (output / "manifests/fixture_diagnostic.json").read_text()
            )
            self.assertFalse(
                saved["quality_summary"]["last_all_compact_gate_tasks_exact_pass"]
            )

    def test_integrity_failure_never_writes_execution_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            output = root / "output"
            dataset.mkdir()
            with self.assertRaises(FileNotFoundError):
                diagnose_overfit_fixture(dataset, [], output)
            self.assertFalse(
                (output / "markers/cvae_overfit_fixture_diagnostic.ok").exists()
            )


if __name__ == "__main__":
    unittest.main()
