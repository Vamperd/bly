from __future__ import annotations

import copy
import json
from pathlib import Path
import signal
import tempfile
import zipfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from test_posterior_hierarchical_standard_cvae import batch, config
from cvae_sa.cvae_protocol import Fixtures, RecoverableSampler, capture_rng, restore_rng, lr_factor, quality_warnings
from cvae_sa.cvae_diagnostics import Diagnostics, evaluate, ensemble_scores, epsilon_for, route_output, stats, ablations
from cvae_sa.cvae_training import run_experiment, append, load_rows
from cvae_sa.models import build_model
from cvae_sa.posterior_t64_protocol import make_physical_masks

ROOT = Path(__file__).resolve().parents[2]


class FakeDataset:
    def __init__(self):
        with torch.random.fork_rng():
            torch.manual_seed(222)
            values = batch(3)
        self.samples = [{key: value[i] for key, value in values.items()} for i in range(3)]

    def __getitem__(self, i):
        return self.samples[i]

    def __len__(self):
        return len(self.samples)

    def close(self):
        pass


def normalization():
    return {"state": (np.zeros(70), np.ones(70)*2), "action": (np.zeros(29), np.ones(29)*3)}


class ProtocolTests(unittest.TestCase):
    def test_mask_invariant_to_batch_order_size(self):
        fixture = Fixtures(FakeDataset(), [0, 1, 2], expand=True)
        a = default_collate([fixture[5], fixture[10]])
        b = default_collate([fixture[10], fixture[5]])
        aa, ab, an = make_physical_masks(a, 19)
        ba, bb, bn = make_physical_masks(b, 19)
        one = make_physical_masks(default_collate([fixture[5]]), 19)
        self.assertTrue(torch.equal(aa[0], ba[1]))
        self.assertTrue(torch.equal(ab[0], bb[1]))
        self.assertTrue(torch.equal(aa[0], one[0][0]))
        self.assertEqual(an[0], bn[1])
        names = make_physical_masks(default_collate([fixture[i] for i in range(8)]), 19)[2]
        self.assertEqual(len(set(names)), 8)

    def test_sampler_resume_exact_permutation_cursor(self):
        fixture = Fixtures(FakeDataset(), [0, 1, 2], expand=True)
        original = RecoverableSampler(len(fixture), 5, 4)
        original.next(fixture)
        state = copy.deepcopy(original.state_dict())
        resumed = RecoverableSampler(len(fixture), 5, 999)
        resumed.load_state_dict(state)
        for _ in range(20):
            a, b = original.next(fixture), resumed.next(fixture)
            self.assertTrue(torch.equal(a["fixture_index"], b["fixture_index"]))
            self.assertTrue(torch.equal(a["sample_ordinal"], b["sample_ordinal"]))

    def test_metrics_empty_partition_argmax_and_physical(self):
        fixture = Fixtures(FakeDataset(), [0], expand=False)
        value = default_collate([fixture[0]])
        pred = value["physical_state"].clone()
        pred[0, 64, 35] += 4
        action = value["action"].clone()
        action[0, 0, 2] -= 2
        output = SimpleNamespace(physical_state=pred, action=action, state_contact_logits=torch.zeros(1, 65, 2))
        sm, am = torch.zeros_like(pred, dtype=torch.bool), torch.zeros_like(action, dtype=torch.bool)
        sm[0, 64, 35] = True
        d = Diagnostics(normalization())
        d.add(value, output, sm, am, ["state_gap"])
        result = d.finish()
        self.assertEqual(result["partitions"]["masked"]["state"]["mse"], 16)
        self.assertEqual(result["partitions"]["visible"]["state"]["mse"], 0)
        self.assertIsNone(result["partitions"]["masked"]["action"]["mse"])
        self.assertEqual(result["top_elements"]["state"][0]["physical_error"], 8)
        self.assertEqual(result["top_elements"]["state"][0]["relative_frame"], 64)
        self.assertEqual(result["top_elements"]["state"][0]["chunk"], 15)
        self.assertEqual(result["top_elements"]["state"][0]["feature_index"], 35)
        self.assertEqual(result["top_elements"]["action"][0]["abs_error"], 2)
        self.assertEqual(len(result["features"]["state"]), 68)

    def test_evaluation_preserves_rng_and_prior_hidden_isolation(self):
        fixture = Fixtures(FakeDataset(), [0, 1], expand=True)
        loader = DataLoader(fixture, batch_size=8)
        model = build_model(config()).train()
        before = capture_rng()
        metrics, _ = evaluate(model, loader, torch.device("cpu"), normalization(), route="standard_normal", seed=3, samples=2)
        self.assertTrue(torch.equal(before["torch"], torch.get_rng_state()))
        self.assertTrue(model.training)
        self.assertGreaterEqual(metrics["energy_score"], 0)
        b = default_collate([fixture[0], fixture[1]])
        sm, am, _ = make_physical_masks(b, 3)
        epsilon = epsilon_for(model, b, 0, 3, "cpu")
        model.eval()
        for route in ("B", "standard_normal"):
            original = route_output(model, b, sm, am, route, epsilon)
            changed = dict(b)
            changed["physical_state"] = b["physical_state"].masked_fill(sm, 1000)
            changed["action"] = b["action"].masked_fill(am, -1000)
            actual = route_output(model, changed, sm, am, route, epsilon)
            self.assertTrue(torch.equal(original.physical_state, actual.physical_state))
            self.assertTrue(torch.equal(original.action, actual.action))

    def test_schedule_last_actual_update(self):
        self.assertEqual(lr_factor(59999, 60000, 2000, .01, "cosine"), .01)
        self.assertEqual(lr_factor(0, 60000, 2000, .01, "cosine"), 1/2000)
        self.assertEqual(lr_factor(1999, 60000, 2000, .01, "cosine"), 1.)
        self.assertTrue(quality_warnings([{"selection_score": 1.}]*5))

    def test_ablation_donors_are_different_windows_and_heldout_overlap_explicit(self):
        fixtures = Fixtures(FakeDataset(), [0, 1], expand=True)
        value = default_collate([fixtures[0], fixtures[8]])
        sm, am, _ = make_physical_masks(value, 7)
        model = build_model(config()).eval()
        for stage in ("A", "B", "C"):
            result = ablations(model, value, sm, am, stage, torch.tensor([1, 0]))
            self.assertEqual(len(result["rows"]), 9)
            self.assertNotEqual(result["recipients"][0]["stable_window_id"], result["recipients"][1]["stable_window_id"])
        metrics, _ = evaluate(model, DataLoader(fixtures, batch_size=8), torch.device("cpu"), normalization(),
            route="B", seed=700008, held_out=True, reference_mask_seed=7)
        self.assertGreaterEqual(metrics["heldout_coordinate_overlap"]["seen"], 2)  # full_action is necessarily identical
        self.assertIn("new", metrics["coordinate_groups"])

    def train(self, folder, stage="B", steps=6, **kwargs):
        dataset_run = folder / "dataset"
        (dataset_run / "data").mkdir(parents=True, exist_ok=True)
        (dataset_run / "manifests").mkdir(exist_ok=True)
        (dataset_run / "manifests/dataset_manifest.json").write_text("{}")
        (dataset_run / "manifests/episodes.jsonl").write_text("{}\n")
        np.savez(dataset_run / "data/normalization.npz", physical_state_mean=np.zeros(70), physical_state_std=np.ones(70),
                 action_mean=np.zeros(29), action_std=np.ones(29))
        dataset = FakeDataset()
        configuration = {"model": config(), "initialization_seed": 33, "training_mask_seed": 42,
            "training": {"micro_batch": 5, "warmup_steps": 1, "validation_interval": 3, "log_interval": 2, "checkpoint_interval": 2}}
        run = kwargs.pop("run", folder / "run")
        with patch("cvae_sa.cvae_training.make_dataset", return_value=(dataset, [0, 1, 2])):
            return run_experiment(dataset_run, run, None, configuration, stage=stage, max_steps_override=steps, eval_samples=2, **kwargs)

    def test_three_stage_training_checkpoint_and_exact_resume(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temporary:
            root = Path(temporary)
            for stage in ("A", "B", "C"):
                result = self.train(root / stage, stage)
                self.assertTrue(result["execution_pass"])
                self.assertNotIn("records", result)
                checkpoint = torch.load(root / stage / "run/checkpoints/last.pt", weights_only=False)
                self.assertEqual(checkpoint["optimizer_step"], 6)
                self.assertAlmostEqual(checkpoint["optimizer"]["param_groups"][0]["lr"], 1e-6)
            interrupted_root = root / "interrupted"
            def interrupt_after_2(path, row):
                append(path, row)
                if row.get("phase") == "train" and row["optimizer_step"] == 2:
                    signal.raise_signal(signal.SIGTERM)
            with patch("cvae_sa.cvae_training.append", side_effect=interrupt_after_2):
                result = self.train(interrupted_root)
            self.assertTrue(result["interrupted"])
            self.train(interrupted_root, resume_run=interrupted_root / "run")
            a = torch.load(root / "B/run/checkpoints/last.pt", weights_only=False)
            b = torch.load(interrupted_root / "run/checkpoints/last.pt", weights_only=False)
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], atol=0, rtol=0)
            log_a = [r["sample_identity_sha256"] for r in load_rows(root / "B/run/logs/metrics.jsonl") if r.get("phase") == "train"]
            log_b = [r["sample_identity_sha256"] for r in load_rows(interrupted_root / "run/logs/metrics.jsonl") if r.get("phase") == "train"]
            self.assertEqual(log_a, log_b)
            with patch("cvae_sa.cvae_training.append", side_effect=interrupt_after_2):
                result = self.train(root / "interrupted_C", stage="C")
            self.assertTrue(result["interrupted"])
            self.train(root / "interrupted_C", stage="C", resume_run=root / "interrupted_C/run")
            a_c = torch.load(root / "C/run/checkpoints/last.pt", weights_only=False)
            b_c = torch.load(root / "interrupted_C/run/checkpoints/last.pt", weights_only=False)
            for name in a_c["model"]:
                torch.testing.assert_close(a_c["model"][name], b_c["model"][name], atol=1e-6, rtol=1e-5)
            continued = self.train(root / "A", stage="A", steps=2, run=root / "continuation",
                continue_checkpoint=root / "A/run/checkpoints/last.pt", additional_steps=2)
            self.assertEqual(continued["cumulative_step"], 8)
            continued_checkpoint = torch.load(root / "continuation/checkpoints/last.pt", weights_only=False)
            self.assertEqual(max(int(s["step"]) for s in continued_checkpoint["optimizer"]["state"].values()), 8)
            dynamic = self.train(root / "B", steps=2, run=root / "dynamic", mask_mode="dynamic",
                init_checkpoint=root / "B/run/checkpoints/best.pt")
            self.assertEqual(dynamic["training_contract"]["mask_mode"], "dynamic")
            self.assertEqual(dynamic["cumulative_step"], 2)
            self.assertTrue((root / "dynamic/checkpoints/best_fixed.pt").is_file())
            self.assertTrue((root / "dynamic/checkpoints/best_heldout.pt").is_file())
            self.assertIsNotNone(dynamic["best_heldout_step"])
            from cvae_sa.cvae_tools import evaluate_checkpoint, export_report, monitor
            with patch("cvae_sa.cvae_tools.make_dataset", return_value=(FakeDataset(), [0, 1, 2])):
                diagnostic = evaluate_checkpoint(SimpleNamespace(checkpoint=root / "A/run/checkpoints/best.pt",
                    output_run=root / "readonly", dataset_run=root / "A/dataset", config=None, route="A",
                    micro_batch=2, samples=2, mask_seed=42, export_all=False))
            self.assertTrue(diagnostic["execution_pass"])
            package = export_report(root / "readonly")
            with zipfile.ZipFile(package) as archive:
                self.assertIn("data/normalization.npz", archive.namelist())
                self.assertFalse(any(name.endswith('.pt') for name in archive.namelist()))

    def test_optimizer_exception_keeps_preceding_durable_checkpoint(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temporary:
            root = Path(temporary)
            real_step = torch.optim.AdamW.step
            def fail_after_partial_update(optimizer, *args, **kwargs):
                real_step(optimizer, *args, **kwargs)
                raise RuntimeError("simulated torn optimizer update")
            with patch.object(torch.optim.AdamW, "step", fail_after_partial_update):
                with self.assertRaisesRegex(RuntimeError, "torn optimizer"):
                    self.train(root, stage="A", steps=2)
            checkpoint = torch.load(root / "run/checkpoints/last.pt", weights_only=False)
            self.assertEqual(checkpoint["optimizer_step"], 0)
            self.assertEqual(checkpoint["sampler"]["exposures"], 0)
            progress = json.loads((root / "run/manifests/progress.json").read_text())
            self.assertEqual(progress["status"], "failed")


if __name__ == "__main__":
    unittest.main()
