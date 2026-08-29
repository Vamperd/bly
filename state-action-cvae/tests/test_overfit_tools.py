from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cvae_sa.overfit_analysis import _checkpoint_loss_config, nearest_target_dispersions
from cvae_sa.overfit_report import summarize_overfit_runs
from cvae_sa.overfit_single_report import summarize_single_task_runs
from cvae_sa.overfit_visualization import (
    read_metrics,
    write_latent_comparison_svg,
    write_training_svg,
)
from cvae_sa.util import load_config


class OverfitToolsTest(unittest.TestCase):
    def test_analysis_reads_loss_settings_from_checkpoint_training_config(self) -> None:
        project = Path(__file__).resolve().parents[1]
        config = load_config(
            project / "configs/default.json",
            project / "configs/overfit_32_compact_capacity.json",
        )
        self.assertIs(_checkpoint_loss_config(config), config["training"])
        with self.assertRaisesRegex(ValueError, "missing the training loss settings"):
            _checkpoint_loss_config({"model": config["model"]})

    def test_empirical_ambiguity_uses_unique_episode_frame_records(self) -> None:
        episodes = []
        for episode_index in range(4):
            states = np.zeros((25, 70), dtype=np.float32)
            states[:, 0] = np.linspace(0.0, 1.0, 25)
            actions = np.zeros((24, 29), dtype=np.float32)
            actions[:, 0] = np.linspace(0.0, 0.5, 24) + episode_index * 0.01
            episodes.append({
                "episode_index": episode_index,
                "states": states,
                "actions": actions,
            })
        result = nearest_target_dispersions(
            episodes, [f"joint_{index}" for index in range(29)],
            (1, 4), max_samples=64, seed=7,
        )
        self.assertEqual(len(result), 6)
        self.assertTrue(all(item["sample_count"] > 0 for item in result))
        self.assertTrue(all(np.isfinite(
            item["target_disagreement_normalized_rmse_p50"]
        ) for item in result))
        self.assertTrue(all(
            item["status"] == "empirical_ambiguity_indicator" for item in result
        ))

    def test_overfit_configs_declare_phase_latent_protocol(self) -> None:
        project = Path(__file__).resolve().parents[1]
        default = project / "configs/default.json"
        capacity_names = (
            "overfit_32_compact_capacity.json",
            "overfit_32_reference_capacity.json",
            "overfit_32_joint_id_capacity.json",
            "overfit_32_smoke.json",
            "overfit_32_lean_capacity.json",
            "overfit_32_lean_single_forward_rollout.json",
            "overfit_32_lean_single_inverse.json",
            "overfit_32_lean_single_history_action.json",
            "overfit_32_lean_single_arbitrary_state.json",
            "overfit_32_lean_single_arbitrary_action.json",
        )
        full_names = (
            "overfit_32_compact_full.json",
            "overfit_32_reference_full.json",
            "overfit_32_joint_id_full.json",
            "overfit_32_no_aux_full.json",
            "overfit_32_lean_full.json",
        )
        for name in capacity_names:
            training = load_config(default, project / "configs" / name)["training"]
            self.assertEqual(training["overfit_gate_latent_mode"], "posterior_mean")
            self.assertEqual(training["overfit_diagnostic_latent_modes"], ["prior_mean"])
        for name in full_names:
            training = load_config(default, project / "configs" / name)["training"]
            self.assertEqual(training["overfit_gate_latent_mode"], "prior_mean")
            self.assertEqual(training["overfit_diagnostic_latent_modes"], [])

    def test_config_extends_and_incomplete_metrics_svg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default = root / "default.json"
            base = root / "base.json"
            child = root / "child.json"
            default.write_text(json.dumps({"a": {"x": 1, "y": 2}}))
            base.write_text(json.dumps({"a": {"x": 3}}))
            child.write_text(json.dumps({"extends": "base.json", "a": {"y": 4}}))
            self.assertEqual(load_config(default, child), {"a": {"x": 3, "y": 4}})
            metrics = root / "metrics.jsonl"
            metrics.write_text(
                json.dumps({
                    "phase": "train", "optimizer_step": 1, "total": 1.0,
                    "learning_rate": 1e-4, "gradient_norm": 2.0, "effective_epoch": 0.5,
                }) + "\n" + json.dumps({
                    "phase": "validation", "optimizer_step": 1,
                    "gate_latent_mode": "posterior_mean",
                    "arbitrary_state_normalized_rmse": 0.1,
                    "latent_diagnostics": {
                        "prior_mean": {"arbitrary_state_normalized_rmse": 0.8}
                    },
                }) + "\n{unfinished"
            )
            output = root / "training.svg"
            write_training_svg(metrics, output)
            latent_output = root / "latent.svg"
            write_latent_comparison_svg(metrics, latent_output)
            self.assertEqual(len(read_metrics(metrics)), 2)
            self.assertIn("<svg", output.read_text(encoding="utf-8"))
            latent_svg = latent_output.read_text(encoding="utf-8")
            self.assertIn("posterior_mean gate", latent_svg)
            self.assertIn("prior_mean diagnostic", latent_svg)

    def test_suite_requires_two_of_three_compact_seed_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            for profile, seed, phases in (
                ("compact", 20260828, ("capacity", "full")),
                ("compact", 20260829, ("capacity", "full")),
                ("compact", 20260830, ("capacity", "full")),
                ("reference", 20260828, ("capacity", "full")),
            ):
                for phase in phases:
                    run = root / f"{profile}_{seed}_{phase}"
                    (run / "manifests").mkdir(parents=True)
                    value = {
                        "memorization_benchmark": True,
                        "model_profile": profile,
                        "overfit_phase": phase,
                        "seed": seed,
                        "parameter_count": 1,
                        "best_overfit_score": 0.8,
                        "passed": not (profile == "compact" and seed == 20260830),
                        "gate_latent_mode": (
                            "posterior_mean" if phase == "capacity" else "prior_mean"
                        ),
                        "diagnostic_latent_modes": (
                            ["prior_mean"] if phase == "capacity" else []
                        ),
                    }
                    (run / "manifests/training_summary.json").write_text(json.dumps(value))
                    runs.append(run)
            output = root / "suite"
            result = summarize_overfit_runs(runs, output)
            self.assertTrue(result["passed"])
            self.assertEqual(result["compact_passed_seed_count"], 2)
            self.assertTrue((output / "markers/cvae_overfit_suite.ok").is_file())
            self.assertTrue((output / "videos/overfit_comparison.svg").is_file())

    def test_failed_suite_does_not_write_success_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "compact_capacity"
            (run / "manifests").mkdir(parents=True)
            (run / "manifests/training_summary.json").write_text(json.dumps({
                "memorization_benchmark": True,
                "model_profile": "compact",
                "overfit_phase": "capacity",
                "seed": 20260828,
                "parameter_count": 1,
                "best_overfit_score": 1.1,
                "passed": False,
                "gate_latent_mode": "posterior_mean",
                "diagnostic_latent_modes": ["prior_mean"],
            }))
            output = root / "suite"
            result = summarize_overfit_runs([run], output)
            self.assertFalse(result["passed"])
            self.assertFalse((output / "markers/cvae_overfit_suite.ok").exists())

    def test_suite_rejects_legacy_mixed_latent_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "legacy_capacity"
            (run / "manifests").mkdir(parents=True)
            (run / "manifests/training_summary.json").write_text(json.dumps({
                "memorization_benchmark": True,
                "model_profile": "compact",
                "overfit_phase": "capacity",
                "seed": 20260828,
                "parameter_count": 1,
                "best_overfit_score": 15.0,
                "passed": False,
            }))
            with self.assertRaisesRegex(ValueError, "invalid latent protocol"):
                summarize_overfit_runs([run], root / "suite")

    def test_single_task_suite_excludes_unconditioned_inverse_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            for task in (
                "forward_rollout", "inverse", "history_action",
                "arbitrary_state", "arbitrary_action",
            ):
                run = root / task
                (run / "manifests").mkdir(parents=True)
                value = {
                    "memorization_benchmark": True,
                    "model_profile": f"compact_single_{task}",
                    "overfit_phase": "capacity",
                    "task_mode": task,
                    "fixed_training_masks": True,
                    "gate_latent_mode": "posterior_mean",
                    "minimum_optimizer_steps": 20_000,
                    "maximum_optimizer_steps": 20_000,
                    "optimizer_steps": 20_000,
                    "seed": 20260828,
                    "parameter_count": 15_065_048,
                    "samples_seen": 1_280_000,
                    "samples_seen_per_task": {task: 1_280_000},
                    "best_overfit_score": 0.8 if task not in {"inverse", "history_action"} else 3.0,
                    "passed": task not in {"inverse", "history_action"},
                }
                (run / "manifests/training_summary.json").write_text(json.dumps(value))
                runs.append(run)
            output = root / "single_suite"
            result = summarize_single_task_runs(runs, output)
            self.assertTrue(result["passed"])
            self.assertTrue((output / "markers/cvae_overfit_single_suite.ok").is_file())
            self.assertTrue((output / "videos/parameter_efficiency.svg").is_file())

    def test_lean_single_task_suite_can_pass_alone_and_all_tasks_are_gated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            tasks = (
                "forward_rollout", "inverse", "history_action",
                "arbitrary_state", "arbitrary_action",
            )
            for task in tasks:
                run = root / task
                (run / "manifests").mkdir(parents=True)
                value = {
                    "memorization_benchmark": True,
                    "model_profile": f"lean_single_{task}",
                    "task_mode": task,
                    "fixed_training_masks": True,
                    "gate_latent_mode": "posterior_mean",
                    "minimum_optimizer_steps": 20_000,
                    "maximum_optimizer_steps": 20_000,
                    "optimizer_steps": 20_000,
                    "seed": 20260828,
                    "parameter_count": 6_204_665,
                    "samples_seen": 1_280_000,
                    "samples_seen_per_task": {task: 1_280_000},
                    "best_overfit_score": 0.8,
                    "passed": True,
                }
                (run / "manifests/training_summary.json").write_text(json.dumps(value))
                runs.append(run)
            result = summarize_single_task_runs(runs, root / "passing")
            self.assertTrue(result["passed"])
            failed = json.loads(
                (runs[1] / "manifests/training_summary.json").read_text()
            )
            failed["passed"] = False
            (runs[1] / "manifests/training_summary.json").write_text(json.dumps(failed))
            result = summarize_single_task_runs(runs, root / "failing")
            self.assertFalse(result["passed"])
            self.assertFalse(
                (root / "failing/markers/cvae_overfit_single_suite.ok").exists()
            )


if __name__ == "__main__":
    unittest.main()
