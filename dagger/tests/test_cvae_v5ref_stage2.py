import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

try:
    from omegaconf import OmegaConf

    from dagger import cli
except ImportError:  # the Windows edit env has no omegaconf; these run on the Ubuntu venv
    OmegaConf = cli = None

from dagger.adapters.cvae_student import (
    REFERENCE_CACHE_KEYS,
    SOURCE_CACHE_KEYS,
    CvaeStudentAdapter,
    _cache_samples,
)
from dagger.core.cvae_loop import fall_step_histogram

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _sample(index: int, reference: bool) -> dict:
    row = {
        "physical_state": torch.zeros(65, 70),
        "action": torch.zeros(64, 29),
        "valid_state": torch.ones(65, dtype=torch.bool),
        "valid_action": torch.ones(64, dtype=torch.bool),
        "motion_key": f"motion_{index % 4}",
        "variant_id": 0,
        "window_start": index,
        "episode_ref": f"ep_{index}",
        "extra_only_here": index,
    }
    if reference:
        row["target_action_valid"] = torch.ones(64, dtype=torch.bool)
        row["target_contact_valid"] = torch.ones(65, dtype=torch.bool)
    return row


class FallStepHistogramTest(unittest.TestCase):
    def test_buckets(self):
        counts = fall_step_histogram([0, 31, 32, 100, 191, 192, 500])
        self.assertEqual(
            counts,
            {"[0,32)": 2, "[32,64)": 1, "[64,128)": 1, "[128,192)": 1, "[192,inf)": 2},
        )


class CacheSamplesTest(unittest.TestCase):
    def test_spread_covers_whole_dataset(self):
        rows = [_sample(i, False) for i in range(100)]
        cache = _cache_samples(rows, SOURCE_CACHE_KEYS, 5, spread=True, label="t")
        self.assertEqual([row["window_start"] for row in cache], [0, 25, 50, 74, 99])
        self.assertNotIn("extra_only_here", cache[0])

    def test_prefix_without_spread(self):
        rows = [_sample(i, False) for i in range(10)]
        cache = _cache_samples(rows, SOURCE_CACHE_KEYS, 3, spread=False, label="t")
        self.assertEqual([row["window_start"] for row in cache], [0, 1, 2])


class ReferenceMixedBatchTest(unittest.TestCase):
    def _adapter(self):
        adapter = CvaeStudentAdapter.__new__(CvaeStudentAdapter)
        adapter.source_dataset_kind = "reference_mixed"
        adapter.batch_size = 8
        adapter.reference_fraction = 0.5
        adapter.mask_seed = 20261016
        adapter.source_mask_mode = "fixed"
        adapter.global_step = 0
        physical = [_sample(i, False) for i in range(20)]
        reference = [_sample(i, True) for i in range(20)]
        adapter._source_cache = _cache_samples(physical, SOURCE_CACHE_KEYS, 20, spread=True, label="p")
        adapter._reference_cache = _cache_samples(reference, REFERENCE_CACHE_KEYS, 20, spread=True, label="r")
        adapter._reference_dataset = reference
        adapter._load_source_dataset = lambda: physical
        return adapter

    def test_split_and_masks(self):
        adapter = self._adapter()
        batch = adapter._source_batch(torch.Generator().manual_seed(0))
        self.assertEqual(batch["physical"]["action"].shape[0], 4)
        self.assertEqual(batch["reference"]["action"].shape[0], 4)
        merged, sm, am, lsm, lam = adapter._reference_mixed_inputs(batch["physical"], batch["reference"])
        self.assertEqual(merged["physical_state"].shape, (8, 65, 70))
        self.assertNotIn("target_action_valid", merged)
        self.assertEqual(sm.shape, (8, 65, 70))
        self.assertEqual(am.shape, (8, 64, 29))
        self.assertTrue(lsm[:4].all() and lam[:4].all())
        self.assertFalse(lam[4:, :32].any())
        self.assertTrue(lam[4:, 32:].all())
        self.assertTrue(lsm[4:, 33:, 68:].all())
        self.assertFalse(lsm[4:, :, :68].any())

    def test_fraction_must_leave_both_halves(self):
        adapter = self._adapter()
        adapter.reference_fraction = 1.0
        with self.assertRaises(ValueError):
            adapter._source_batch(torch.Generator().manual_seed(0))


@unittest.skipIf(cli is None, "needs omegaconf")
class MotionGateTest(unittest.TestCase):
    def test_motion_augmentation_flag(self):
        for value, expected in (("true", True), ("1", True), (False, False), ("off", False)):
            config = OmegaConf.create({"dagger": {"disable_motion_augmentation": value}})
            self.assertEqual(cli._motion_augmentation_disabled(config), expected)
        release = OmegaConf.create(
            {"manager_env": {"commands": {"motion": {flag: True for flag in cli.MOTION_AUGMENTATION_FLAGS}}}}
        )
        cli._disable_motion_augmentation(release)
        self.assertFalse(any(release.manager_env.commands.motion[f] for f in cli.MOTION_AUGMENTATION_FLAGS))

    def test_c_motion_coverage(self):
        manifest = {"motion_keys": ["a", "b"], "pilot_motion_keys": ["a"]}
        cli._check_c_motion_coverage(SimpleNamespace(train_motion_keys=["a", "b", "c"]), manifest)
        cli._check_c_motion_coverage(SimpleNamespace(train_motion_keys=[]), manifest)
        with self.assertRaises(ValueError):
            cli._check_c_motion_coverage(SimpleNamespace(train_motion_keys=["a"]), manifest)


@unittest.skipIf(cli is None, "needs omegaconf")
class RawCRenderTest(unittest.TestCase):
    RAW_C = {
        "model_signature": {"architecture_version": "65-token-hierarchical-standard-cvae-v1"},
        "train_motion_keys": ["m1", "m2"],
    }
    DAGGER = {
        "format_version": "dagger_cvae_student_checkpoint_v1",
        "model_signature": {"architecture_version": "65-token-hierarchical-standard-cvae-v1"},
        "train_motion_keys": ["m1", "m2"],
        "motion_keys": ["m1"],
        "motion_coverage": {"m3": 1},
        "future_reference": True,
        "relabel_steps": 8,
        "relabel_gamma": 0.9,
        "execute_steps": 4,
    }

    def test_detection_and_motions(self):
        self.assertTrue(cli._is_raw_c_checkpoint(self.RAW_C))
        self.assertFalse(cli._is_raw_c_checkpoint(self.DAGGER))
        self.assertEqual(cli._cvae_checkpoint_motions(self.RAW_C), {"m1", "m2"})
        self.assertEqual(cli._cvae_checkpoint_motions(self.DAGGER), {"m1", "m3"})

    def test_rollout_settings(self):
        config = OmegaConf.create({"dagger": {"render_c_future_reference": "true", "render_c_execute_steps": 4}})
        self.assertEqual(
            cli._cvae_render_rollout(config, self.RAW_C, True),
            {"future_reference": True, "relabel_steps": 4, "relabel_gamma": 1.0, "execute_steps": 4},
        )
        self.assertEqual(
            cli._cvae_render_rollout(config, self.DAGGER, False),
            {"future_reference": True, "relabel_steps": 8, "relabel_gamma": 0.9, "execute_steps": 4},
        )


class V5ConfigTest(unittest.TestCase):
    def test_manifest_and_config(self):
        manifest = json.loads((CONFIGS / "cvae_kl2_motion_manifest_v5.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["motion_keys"]), 32)
        self.assertTrue(set(manifest["pilot_motion_keys"]) <= set(manifest["motion_keys"]))
        self.assertFalse(manifest["source_mix_exact"])
        config = yaml.safe_load((CONFIGS / "cvae_kl2_pilot_v5ref_12h.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["mode"], "cvae_kl2_pilot_v5ref_12h")
        self.assertTrue(config["dagger"]["disable_motion_augmentation"])
        self.assertEqual(config["student"]["source_dataset_kind"], "reference_mixed")
        self.assertEqual(float(config["student"]["source_causal_fraction"]), 0.0)


if __name__ == "__main__":
    unittest.main()
