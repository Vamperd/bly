import json
import tempfile
import shutil
import os
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

try:
    import h5py  # noqa: F401
except ImportError:  # Windows static environments intentionally lack HDF5.
    h5py = None

from test_posterior_hierarchical_standard_cvae import config as small_model_config
from cvae_sa.cvae_protocol import Fixtures
from cvae_sa.cvae_training import merge_batches, run_experiment, valid_loss_masks
from cvae_sa.models import build_model
from cvae_sa.posterior_t64_protocol import make_physical_masks, make_reference_masks
from cvae_sa.util import atomic_write_json, file_sha256

if h5py is not None:
    from cvae_sa.dataset import StateActionWindowDataset
    from cvae_sa.long_horizon_dataset import build_long_horizon_dataset
    from cvae_sa.physics_indexer import build_physics_index
    from cvae_sa.reference_anchor_dataset import ReferenceAnchorDataset, build_reference_mixed_dataset
    from tests.fixtures import write_physics_collection_run

ROOT = Path(__file__).resolve().parents[2]
TABLE_STEPS = 12
MOTION_STEP_OFFSET = 3  # tests/fixtures.py writes motion_step = arange(steps) + 3


def write_reference_table(root: Path, physics: Path) -> Path:
    run = root / "reference_table"
    for folder in ("data", "manifests", "markers"):
        (run / folder).mkdir(parents=True)
    states = np.zeros((TABLE_STEPS, 70), dtype=np.float32)
    states[:, :68] = np.arange(TABLE_STEPS, dtype=np.float32)[:, None] / 10 + np.arange(68) / 1000
    states[:, 68:] = 1.0
    table = run / "data/reference_states.npz"
    np.savez(table, states=states, motion_keys=np.asarray(["motion_r"]), motion_offsets=np.asarray([0]),
             num_steps=np.asarray([TABLE_STEPS]), nominal_joint_pos=np.zeros(29, np.float32),
             joint_ids=np.arange(29))
    atomic_write_json(run / "manifests/reference_table_manifest.json", {
        "schema_version": "cvae_reference_table_v1", "table": "data/reference_states.npz",
        "table_sha256": file_sha256(table),
        "physics_dataset_manifest_sha256": file_sha256(physics / "manifests/dataset_manifest.json"),
        "alignment": {"pass": True}})
    (run / "markers/cvae_reference_table.ok").write_text("PASS\n")
    return run


@unittest.skipUnless(h5py is not None, "requires h5py")
class ReferenceAnchorDatasetTests(unittest.TestCase):
    def test_migration_preserves_bytes_identities_and_samples(self):
        from cvae_sa.long_horizon_dataset import LongHorizonWindowDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old"
            old.mkdir()
            mixed = self.build(old, 0.25)
            original = ReferenceAnchorDataset(mixed, "train")
            physical = LongHorizonWindowDataset(mixed, "train")
            try:
                expected, expected_physical = original[0], physical[0]
            finally:
                original.close()
                physical.close()
            new = root / "migrated"
            shutil.copytree(old, new)
            manifest_bytes = (new / "mixed/manifests/reference_mixed_manifest.json").read_bytes()
            # The old IO location really is unavailable: this cannot pass by fallback.
            old.rename(root / "original_unavailable")
            with patch.dict(os.environ, {"CVAE_DATASET_PATH_MAP": json.dumps({str(old): str(new)})}):
                for cls, baseline in ((ReferenceAnchorDataset, expected), (LongHorizonWindowDataset, expected_physical)):
                    dataset = cls(old / "mixed", "train")
                    try:
                        actual = dataset[0]
                        self.assertEqual(actual["episode_ref"], baseline["episode_ref"])
                        torch.testing.assert_close(actual["physical_state"], baseline["physical_state"])
                        torch.testing.assert_close(actual["action"], baseline["action"])
                    finally:
                        dataset.close()
            self.assertEqual((new / "mixed/manifests/reference_mixed_manifest.json").read_bytes(), manifest_bytes)

    def build(self, root: Path, probability: float, name: str = "mixed") -> Path:
        physics, long_run = root / "physics", root / "long"
        if not physics.exists():
            sources = (
                write_physics_collection_run(root / "sources", "r0", "motion_r", "Locomotion", (0, 1, 2, 3), reference=True),
                write_physics_collection_run(root / "sources", "r1", "motion_r", "Locomotion", (4, 5, 6, 7), reference=True),
            )
            build_physics_index(sources, physics, expected_motions=1, expected_episodes=8, split_counts=(1, 0, 0), seed=13)
            build_long_horizon_dataset(physics, long_run)
            write_reference_table(root, physics)
        output = root / name
        build_reference_mixed_dataset(long_run, root / "reference_table", output, anchor_stride=4,
                                      history_truncation_probability=probability, seed=5)
        return output

    def test_layout_targets_and_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mixed = self.build(root, 0.0)
            manifest = json.loads((mixed / "manifests/reference_mixed_manifest.json").read_text())
            self.assertEqual(manifest["anchor_count"], 16)  # 8 episodes x anchors at t=0,4
            self.assertEqual(manifest["truncated_count"], 0)
            self.assertEqual(manifest["train_motion_keys"], ["motion_r"])
            dataset = ReferenceAnchorDataset(mixed, "train")
            source = StateActionWindowDataset(root / "physics", "train", 64, 64, max_episodes=None, random_crop=False)
            try:
                by_step = {}
                for index, row in enumerate(dataset.anchors):
                    by_step.setdefault(row["anchor_step"], (index, row))
                index, row = by_step[4]
                self.assertEqual(row["history_steps"], 4)
                sample = dataset[index]
                window = source.get_window(dataset._episode_indices[row["episode_ref"]], 0)
                # History states 0..4 land in slots 28..32; slot 32 is the current State.
                torch.testing.assert_close(sample["physical_state"][28:33], window["physical_state"][0:5])
                torch.testing.assert_close(sample["action"][28:34], window["action"][0:6])
                self.assertFalse(sample["valid_state"][:28].any())
                self.assertTrue(sample["valid_state"][28:].all())
                self.assertTrue(sample["valid_action"][28:].all())
                # Episode has 6 transitions: real targets only for States 5,6 and Actions 4,5.
                self.assertEqual(sample["target_contact_valid"].nonzero().flatten().tolist(), list(range(28, 35)))
                self.assertEqual(sample["target_action_valid"].nonzero().flatten().tolist(), list(range(28, 34)))
                frames = np.clip(4 + MOTION_STEP_OFFSET + np.arange(1, 33), 0, TABLE_STEPS - 1)
                table = np.load(mixed / "data/reference_states.npz")["states"]
                expected = (table[frames, :68] - source.state_mean[:68]) / source.state_std[:68]
                torch.testing.assert_close(sample["physical_state"][33:, :68], torch.from_numpy(expected.astype(np.float32)))
                torch.testing.assert_close(sample["physical_state"][33:35, 68:], window["physical_state"][5:7, 68:])
                self.assertEqual(float(sample["physical_state"][35:, 68:].abs().sum()), 0.0)

                index0, row0 = by_step[0]
                self.assertEqual(row0["history_steps"], 0)
                first = dataset[index0]
                torch.testing.assert_close(first["physical_state"][32], source.get_window(
                    dataset._episode_indices[row0["episode_ref"]], 0)["physical_state"][0])
                self.assertFalse(first["valid_state"][:32].any())
                self.assertFalse(first["valid_action"][:32].any())

                fixtures = Fixtures(dataset, [index, index0], expand=False)
                self.assertEqual(len({row["stable_window_id"] for row in fixtures.manifest()}), 2)
                batch = default_collate([fixtures[0], fixtures[1]])
                sm, am, lsm, lam = make_reference_masks(batch)
                self.assertFalse((sm & ~batch["valid_state"][..., None]).any())
                self.assertFalse((am & ~batch["valid_action"][..., None]).any())
                self.assertFalse(sm[:, :33].any() or sm[:, :, :68].any() or am[:, :32].any())
                self.assertTrue(sm[0, 33:, 68:].all() and am[0, 32:].all())
                self.assertFalse(lsm[..., :68].any())
                self.assertEqual(lsm[0, :, 68].nonzero().flatten().tolist(), [33, 34])
                self.assertEqual(lam[0, :, 0].nonzero().flatten().tolist(), [32, 33])

                physical = Fixtures(source, [0], expand=True)
                physical_batch = default_collate([physical[0]])
                psm, pam, _ = make_physical_masks(physical_batch, 7)
                plsm, plam = valid_loss_masks(physical_batch)
                merged = merge_batches(physical_batch, batch)
                self.assertNotIn("mask_slot", merged)
                self.assertEqual(len(merged["physical_state"]), 3)
                model = build_model(small_model_config())
                model.set_training_stage("C")
                output = model(merged, torch.cat((psm, sm)), torch.cat((pam, am)), stage="C")
                self.assertTrue(torch.isfinite(output.action).all())
                from cvae_sa.posterior_hierarchical_standard_cvae import weighted_reconstruction_loss
                loss = weighted_reconstruction_loss(output, merged, torch.cat((plsm, lsm)), torch.cat((plam, lam)),
                                                    masked_weight=1., full_weight=0.)
                self.assertTrue(torch.isfinite(loss["total"]))
            finally:
                dataset.close()
                source.close()

    def test_truncation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = self.build(root, 1.0, "a"), self.build(root, 1.0, "b")
            rows = lambda run: [json.loads(line) for line in (run / "manifests/anchor_windows.jsonl").read_text().splitlines()]
            self.assertEqual(rows(first), rows(second))
            self.assertTrue(all(row["history_truncated"] and 0 <= row["history_steps"] <= row["natural_history_steps"]
                                for row in rows(first)))

    def test_mixed_c_training_and_cross_dataset_init(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            root = Path(directory)
            mixed = self.build(root, 0.25)
            data = {"long_horizon_manifest": "manifests/long_horizon_windows.jsonl", "num_workers": 0}
            training = {"micro_batch": 4, "eval_micro_batch": 8, "warmup_steps": 1, "validation_interval": 3,
                        "log_interval": 1, "checkpoint_interval": 3}
            base = {"model": small_model_config(), "initialization_seed": 3, "training_mask_seed": 4,
                    "data": data, "training": training}
            physical = run_experiment(root / "long", root / "physical_c", None, base, stage="C",
                                      max_steps_override=3, eval_samples=1)
            self.assertTrue(physical["execution_pass"])
            mixed_config = {**base, "data": {**data, "reference_fraction": 0.5, "anchor_stride": 4,
                                             "history_truncation_probability": 0.25}}
            with self.assertRaises(ValueError):
                run_experiment(mixed, root / "rejected", None, mixed_config, stage="C", max_steps_override=3,
                               eval_samples=1, init_checkpoint=root / "physical_c/checkpoints/best.pt")
            result = run_experiment(mixed, root / "mixed_c", None, mixed_config, stage="C", max_steps_override=3,
                                    eval_samples=1, init_checkpoint=root / "physical_c/checkpoints/best.pt",
                                    allow_cross_dataset_init=True)
            self.assertTrue(result["execution_pass"])
            checkpoint = torch.load(root / "mixed_c/checkpoints/last.pt", weights_only=False)
            self.assertEqual(checkpoint["train_motion_keys"], ["motion_r"])
            self.assertEqual(checkpoint["reference_sampler"]["batch_size"], 2)
            self.assertEqual(checkpoint["sampler"]["batch_size"], 2)
            self.assertEqual(checkpoint["training_contract"]["objective"], "physical_full_plus_reference_target_only_v1")
            self.assertIn("cross_dataset_init", checkpoint["source_checkpoint"])
            evaluations = [json.loads(line) for line in (root / "mixed_c/logs/evaluations.jsonl").read_text().splitlines()]
            self.assertIsNotNone(evaluations[-1]["reference_anchor"]["all"]["slot32_raw_rmse"])
            train_rows = [json.loads(line) for line in (root / "mixed_c/logs/metrics.jsonl").read_text().splitlines()
                          if '"phase": "train"' in line]
            self.assertTrue(all(row["batch_size"] == 4 and row["mask_names"][2:] == ["reference_anchor"] * 2
                                for row in train_rows))


if __name__ == "__main__":
    unittest.main()
