from __future__ import annotations

import unittest

import torch

from cvae_sa.models import PosteriorCapacityTransformerCVAE
from cvae_sa.posterior_capacity import (
    DeterministicWindowSubset,
    FIXED_MASK_NAMES,
    MaskBankDataset,
    evaluate_exact,
    make_fixture_masks,
    optimizer_step_limit,
    reconstruction_loss,
    selected_window_identities,
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
        loader = torch.utils.data.DataLoader(
            MaskBankDataset(_ListDataset(items), len(FIXED_MASK_NAMES)),
            batch_size=20,
            shuffle=False,
        )
        metrics = evaluate_exact(
            PosteriorCapacityTransformerCVAE(model_config()).eval(),
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
