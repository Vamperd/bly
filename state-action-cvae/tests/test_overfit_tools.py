from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cvae_sa.overfit_report import summarize_overfit_runs
from cvae_sa.overfit_visualization import (
    read_metrics,
    write_latent_comparison_svg,
    write_training_svg,
)
from cvae_sa.util import load_config


class OverfitToolsTest(unittest.TestCase):
    def test_overfit_configs_declare_phase_latent_protocol(self) -> None:
        project = Path(__file__).resolve().parents[1]
        default = project / "configs/default.json"
        capacity_names = (
            "overfit_32_compact_capacity.json",
            "overfit_32_reference_capacity.json",
            "overfit_32_joint_id_capacity.json",
            "overfit_32_smoke.json",
        )
        full_names = (
            "overfit_32_compact_full.json",
            "overfit_32_reference_full.json",
            "overfit_32_joint_id_full.json",
            "overfit_32_no_aux_full.json",
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


if __name__ == "__main__":
    unittest.main()
