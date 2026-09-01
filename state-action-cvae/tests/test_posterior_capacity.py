from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import torch

from cvae_sa.models import PosteriorCapacityTransformerCVAE, build_model, parameter_count
from cvae_sa.posterior_capacity import (
    DeterministicWindowSubset,
    FIXED_MASK_NAMES,
    MaskBankDataset,
    evaluate_exact,
    make_fixture_masks,
    optimizer_step_limit,
    _score_gate,
    reconstruction_loss,
    selected_window_identities,
    validate_initial_checkpoint,
    validation_fixture_seed,
    validate_motion_prefix,
)


def model_config() -> dict[str, object]:
    return {
        "d_model": 32,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "heads": 4,
        "ffn_dim": 64,
        "latent_dim": 16,
        "dropout": 0.0,
        "state_dim": 70,
    }


def batch(batch_size: int = 2, transitions: int = 3) -> dict[str, object]:
    state = torch.randn(batch_size, transitions + 1, 70)
    state[..., 68:70] = torch.randint(0, 2, state[..., 68:70].shape).float()
    return {
        "physical_state": state,
        "action": torch.randn(batch_size, transitions, 29),
        "valid_state": torch.ones(batch_size, transitions + 1, dtype=torch.bool),
        "valid_action": torch.ones(batch_size, transitions, dtype=torch.bool),
        "motion_key": [f"motion-{index}" for index in range(batch_size)],
        "variant_id": torch.arange(batch_size),
        "window_start": torch.zeros(batch_size, dtype=torch.long),
        "window_index": torch.arange(batch_size),
        "mask_slot": torch.arange(batch_size),
    }


class _ListDataset(torch.utils.data.Dataset[dict[str, object]]):
    def __init__(self, items: list[dict[str, object]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.items[index]


class PosteriorCapacityTest(unittest.TestCase):
    def test_reference_25m_configuration_has_exact_parameter_count(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs/posterior_capacity_reference_25m.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["model"]["state_dim"] = 70
        self.assertEqual(parameter_count(build_model(config["model"])), 25_453_411)

    def test_global_latent_and_full_mask_shapes(self) -> None:
        torch.manual_seed(1)
        value = batch()
        state_mask = torch.ones_like(value["physical_state"], dtype=torch.bool)
        action_mask = torch.ones_like(value["action"], dtype=torch.bool)
        output = PosteriorCapacityTransformerCVAE(model_config())(
            value, state_mask, action_mask
        )
        self.assertEqual(tuple(output.posterior_mean.shape), (2, 16))
        self.assertEqual(tuple(output.latent.shape), (2, 16))
        self.assertEqual(tuple(output.physical_state.shape), (2, 4, 70))
        self.assertEqual(tuple(output.action.shape), (2, 3, 29))
        self.assertTrue(torch.isfinite(output.physical_state).all())

    def test_decoder_cannot_read_masked_truth_when_latent_is_fixed(self) -> None:
        torch.manual_seed(2)
        value = batch()
        state_mask = torch.ones_like(value["physical_state"], dtype=torch.bool)
        action_mask = torch.ones_like(value["action"], dtype=torch.bool)
        model = PosteriorCapacityTransformerCVAE(model_config()).eval()
        first = model(value, state_mask, action_mask)
        changed = dict(value)
        changed["physical_state"] = value["physical_state"] + 100.0
        changed["action"] = value["action"] - 100.0
        second = model(
            changed, state_mask, action_mask,
            latent_override=first.posterior_mean,
        )
        fixed = model(
            value, state_mask, action_mask,
            latent_override=first.posterior_mean,
        )
        self.assertTrue(torch.equal(fixed.physical_state, second.physical_state))
        self.assertTrue(torch.equal(fixed.action, second.action))
        self.assertFalse(torch.equal(first.posterior_mean, second.posterior_mean))

    def test_fixed_mask_bank_is_reproducible_and_nonempty(self) -> None:
        value = batch(batch_size=len(FIXED_MASK_NAMES))
        value["mask_slot"] = torch.arange(len(FIXED_MASK_NAMES))
        first_state, first_action, names = make_fixture_masks(value, 123)
        second_state, second_action, second_names = make_fixture_masks(value, 123)
        self.assertEqual(names, list(FIXED_MASK_NAMES))
        self.assertEqual(names, second_names)
        self.assertTrue(torch.equal(first_state, second_state))
        self.assertTrue(torch.equal(first_action, second_action))
        self.assertTrue(bool((first_state.flatten(1).any(1) | first_action.flatten(1).any(1)).all()))
        self.assertTrue(bool(first_state[2].all()))
        self.assertTrue(bool(first_action[2].all()))

    def test_fixed_validation_reuses_training_fixture_seed(self) -> None:
        value = batch(batch_size=len(FIXED_MASK_NAMES))
        value["mask_slot"] = torch.arange(len(FIXED_MASK_NAMES))
        training_state, training_action, _ = make_fixture_masks(value, 123)
        fixed_seed = validation_fixture_seed(123, "fixed")
        validation_state, validation_action, _ = make_fixture_masks(value, fixed_seed)
        self.assertEqual(fixed_seed, 123)
        self.assertTrue(torch.equal(training_state, validation_state))
        self.assertTrue(torch.equal(training_action, validation_action))
        self.assertEqual(validation_fixture_seed(123, "generalization"), 700_124)

    def test_optimizer_step_override_is_positive_and_smoke_stays_short(self) -> None:
        self.assertEqual(optimizer_step_limit({"max_optimizer_steps": 80_000}, False), 80_000)
        self.assertEqual(optimizer_step_limit({"max_optimizer_steps": 80_000}, True), 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            optimizer_step_limit({"max_optimizer_steps": 0}, False)

    def test_progression_gate_is_independent_from_exact_gate(self) -> None:
        values = {
            "worst_state": 0.003641,
            "worst_action": 0.002903,
            "worst_max": 0.017851,
            "contact_accuracy": 1.0,
            "zero_ratio": 331.97,
        }
        exact = _score_gate(
            **values,
            thresholds={
                "state_rmse": 1e-4, "action_rmse": 1e-4,
                "continuous_max_abs": 1e-3, "latent_ratio": 10.0,
            },
        )
        progression = _score_gate(
            **values,
            thresholds={
                "state_rmse": 1e-2, "action_rmse": 1e-2,
                "continuous_max_abs": 1e-2, "latent_ratio": 10.0,
            },
        )
        self.assertFalse(exact["passed"])
        self.assertFalse(progression["passed"])
        self.assertAlmostEqual(progression["score"], 1.7851)

        values["worst_max"] = 0.006158
        progression = _score_gate(
            **values,
            thresholds={
                "state_rmse": 1e-2, "action_rmse": 1e-2,
                "continuous_max_abs": 1e-2, "latent_ratio": 10.0,
            },
        )
        self.assertTrue(progression["passed"])

    def test_scale_warm_start_allows_only_compatible_expansion(self) -> None:
        source_config = {
            "model": {"kind": "physics_posterior_transformer", **model_config()},
            "data": {"motion_count": 1, "window_transitions": 128, "max_windows": None},
            "training": {"mask_phase": "fixed", "acceptance_gate": "progression"},
        }
        checkpoint = {
            "format_version": "sonic_posterior_capacity_checkpoint_v1",
            "step": 123,
            "dataset_manifest_sha256": "dataset-hash",
            "config": source_config,
        }
        target = copy.deepcopy(source_config)
        target["data"]["motion_count"] = 32
        metadata = validate_initial_checkpoint(
            checkpoint,
            dataset_hash="dataset-hash",
            target_config=target,
            target_phase="fixed",
            acceptance_gate="progression",
            allow_scale_expansion=True,
        )
        self.assertEqual(metadata["source_step"], 123)
        self.assertTrue(metadata["model_only"])
        self.assertFalse(metadata["optimizer_scheduler_rng_restored"])
        generalization = copy.deepcopy(source_config)
        generalization["training"]["mask_phase"] = "generalization"
        validate_initial_checkpoint(
            checkpoint,
            dataset_hash="dataset-hash",
            target_config=generalization,
            target_phase="generalization",
            acceptance_gate="progression",
            allow_scale_expansion=False,
        )
        generalization["data"]["motion_count"] = 2
        with self.assertRaisesRegex(ValueError, "must keep"):
            validate_initial_checkpoint(
                checkpoint,
                dataset_hash="dataset-hash",
                target_config=generalization,
                target_phase="generalization",
                acceptance_gate="progression",
                allow_scale_expansion=False,
            )
        target["data"]["window_transitions"] = 64
        with self.assertRaisesRegex(ValueError, "only expand"):
            validate_initial_checkpoint(
                checkpoint,
                dataset_hash="dataset-hash",
                target_config=target,
                target_phase="fixed",
                acceptance_gate="progression",
                allow_scale_expansion=True,
            )

    def test_mask_bank_repeats_every_window_by_slot(self) -> None:
        base = _ListDataset([{"value": 0}, {"value": 1}])
        bank = MaskBankDataset(base, 3)
        self.assertEqual(len(bank), 6)
        self.assertEqual(
            [(bank[index]["value"], bank[index]["mask_slot"]) for index in range(6)],
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        )

    def test_deterministic_window_subset_keeps_first_window_identity(self) -> None:
        base = _ListDataset([
            {"value": 10, "motion_key": "first"},
            {"value": 20, "motion_key": "second"},
        ])
        subset = DeterministicWindowSubset(base, 1)
        self.assertEqual(len(subset), 1)
        self.assertEqual(subset.indices, (0,))
        self.assertEqual(subset[0]["value"], 10)
        self.assertEqual(subset[0]["motion_key"], "first")
        self.assertEqual(subset[0]["source_window_index"], 0)
        self.assertEqual(len(DeterministicWindowSubset(base, None)), 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            DeterministicWindowSubset(base, 0)

    def test_selected_window_identity_is_manifest_ready(self) -> None:
        fake = type("FakeDataset", (), {})()
        fake.refs = [type("Ref", (), {"episode_index": 0, "fixed_start": 16})()]
        fake.episodes = [{
            "motion_key": "motion-a",
            "variant_id": 3,
            "episode": "episode_0003",
            "source_run": "/source/run",
        }]
        self.assertEqual(selected_window_identities(fake, (0,)), [{
            "source_window_index": 0,
            "motion_key": "motion-a",
            "variant_id": 3,
            "episode": "episode_0003",
            "episode_ref": "/source/run::episode_0003",
            "window_start": 16,
        }])

    def test_motion_prefix_requires_eight_complete_variants(self) -> None:
        fake = type("FakeDataset", (), {})()
        fake.episodes = [
            {"motion_key": "motion-a", "variant_id": variant}
            for variant in range(8)
        ]
        self.assertEqual(validate_motion_prefix(fake, 1), ["motion-a"])
        fake.episodes[-1]["variant_id"] = 6
        with self.assertRaisesRegex(ValueError, "variants 0..7"):
            validate_motion_prefix(fake, 1)

    def test_exact_evaluator_reports_every_mask_and_latent_ablation(self) -> None:
        torch.manual_seed(5)
        items = []
        for index in range(2):
            value = batch(batch_size=1, transitions=2)
            items.append({
                "physical_state": value["physical_state"][0],
                "action": value["action"][0],
                "valid_state": value["valid_state"][0],
                "valid_action": value["valid_action"][0],
                "motion_key": f"motion-{index}",
                "variant_id": index,
                "window_start": 0,
            })
        fixture_dataset = MaskBankDataset(_ListDataset(items), len(FIXED_MASK_NAMES))
        loader = torch.utils.data.DataLoader(
            fixture_dataset,
            batch_size=20,
            shuffle=False,
        )
        model = PosteriorCapacityTransformerCVAE(model_config()).eval()
        metrics = evaluate_exact(
            model,
            loader,
            torch.device("cpu"),
            seed=123,
            held_out=False,
            thresholds={
                "state_rmse": 1e-4,
                "action_rmse": 1e-4,
                "continuous_max_abs": 1e-3,
                "latent_ratio": 10.0,
            },
        )
        self.assertEqual(set(metrics["cases"]), set(FIXED_MASK_NAMES))
        self.assertIn("zero_ratio", metrics["latent_dependence"])
        self.assertEqual(metrics["acceptance_gate"], "exact")
        self.assertEqual(metrics["score"], metrics["exact_gate"]["score"])
        self.assertIn("progression_gate", metrics)
        reconstruction = metrics["reconstruction_loss"]
        self.assertGreater(reconstruction["counts"]["state"], 0)
        self.assertGreater(reconstruction["counts"]["action"], 0)
        self.assertGreater(reconstruction["counts"]["contact"], 0)
        self.assertAlmostEqual(
            reconstruction["total"],
            (reconstruction["state"] + reconstruction["action"] + reconstruction["contact"]) / 3,
        )
        differently_batched = evaluate_exact(
            model,
            torch.utils.data.DataLoader(fixture_dataset, batch_size=3, shuffle=False),
            torch.device("cpu"),
            seed=123,
            held_out=False,
            thresholds={
                "state_rmse": 1e-4,
                "action_rmse": 1e-4,
                "continuous_max_abs": 1e-3,
                "latent_ratio": 10.0,
            },
        )["reconstruction_loss"]
        for key in ("total", "state", "action", "contact"):
            self.assertAlmostEqual(reconstruction[key], differently_batched[key], places=6)
        self.assertFalse(metrics["passed"])

    def test_two_short_sequences_can_reduce_posterior_reconstruction_loss(self) -> None:
        torch.manual_seed(7)
        value = batch(batch_size=2, transitions=2)
        state_mask = torch.ones_like(value["physical_state"], dtype=torch.bool)
        action_mask = torch.ones_like(value["action"], dtype=torch.bool)
        model = PosteriorCapacityTransformerCVAE(model_config())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        initial = None
        final = None
        for _ in range(120):
            output = model(value, state_mask, action_mask)
            loss = reconstruction_loss(output, value, state_mask, action_mask).total
            initial = float(loss.detach()) if initial is None else initial
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final = float(loss.detach())
        assert initial is not None and final is not None
        self.assertLess(final, initial * 0.10)
        self.assertLess(final, 0.005)


if __name__ == "__main__":
    unittest.main()
