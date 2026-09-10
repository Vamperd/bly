from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

import torch

from cvae_sa.models import (
    HierarchicalConditionalPriorTransformer,
    HierarchicalPosteriorTransformer,
    build_model,
)
from cvae_sa.posterior_complete_token_protocol import (
    RANDOM_TOKEN_MASK_NAMES,
    TRAINING_MIXTURE,
    make_dynamic_random_token_masks,
    make_heldout_random_token_masks,
    validate_complete_token_masks,
)
from cvae_sa.posterior_hierarchical_conditional_prior import (
    _source_truth_metrics,
    decoder_adaptation_decision,
    evaluate_full_both_conditional_prior,
    evaluate_teacher_latent_dependence,
    evaluate_teacher_preservation,
    full_and_masked_reconstruction_loss,
    initialize_from_h50_a,
    latent_distillation_loss,
    phase_weights,
    validate_checkpoint,
)
from cvae_sa.posterior_t64_protocol import make_autoencode_masks, make_physical_masks
from cvae_sa.util import load_config


def small_config() -> dict[str, object]:
    return {
        "kind": "physics_hierarchical_conditional_prior_transformer",
        "profile": "test",
        "d_model": 16,
        "posterior_encoder_layers": 1,
        "condition_encoder_layers": 1,
        "conditional_prior_encoder_layers": 1,
        "decoder_layers": 1,
        "heads": 4,
        "ffn_dim": 32,
        "global_latent_dim": 8,
        "local_latent_dim": 4,
        "local_chunks": 16,
        "chunk_transitions": 4,
        "max_state_steps": 65,
        "dropout": 0.0,
        "state_dim": 70,
    }


def sample_batch(batch_size: int = 4) -> dict[str, object]:
    state = torch.randn(batch_size, 65, 70)
    state[..., 68:70] = torch.randint(0, 2, (batch_size, 65, 2)).float()
    return {
        "physical_state": state,
        "action": torch.randn(batch_size, 64, 29),
        "valid_state": torch.ones(batch_size, 65, dtype=torch.bool),
        "valid_action": torch.ones(batch_size, 64, dtype=torch.bool),
        "motion_key": [f"motion-{index % 2}" for index in range(batch_size)],
        "variant_id": torch.arange(batch_size),
        "window_start": torch.zeros(batch_size, dtype=torch.long),
        "window_index": torch.arange(batch_size),
        "mask_slot": torch.arange(batch_size),
    }


def initialized_model() -> HierarchicalConditionalPriorTransformer:
    config = small_config()
    source_config = dict(config)
    source_config["kind"] = "physics_hierarchical_posterior_transformer"
    source = HierarchicalPosteriorTransformer(source_config)
    model = HierarchicalConditionalPriorTransformer(config)
    initialize_from_h50_a(model, {"model": source.state_dict()})
    return model


class HierarchicalConditionalPriorTest(unittest.TestCase):
    def test_reference_parameter_counts_and_config(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/posterior_hierarchical_conditional_prior_h50.json"
        )
        config["model"]["state_dim"] = 70
        with torch.device("meta"):
            model = build_model(config["model"])
        self.assertIsInstance(model, HierarchicalConditionalPriorTransformer)
        base = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if not model.is_conditional_prior_parameter(name)
        )
        prior = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if model.is_conditional_prior_parameter(name)
        )
        self.assertEqual(base, 51_005_283)
        self.assertEqual(prior, 14_779_456)
        self.assertEqual(base + prior, 65_784_739)
        self.assertEqual(config["training"]["random_training_mixture"], TRAINING_MIXTURE)
        del model
        gc.collect()

    def test_initialization_copies_posterior_and_zeroes_mask_columns(self) -> None:
        model = initialized_model()
        self.assertTrue(torch.equal(
            model.conditional_prior_state_input.weight[:, :70],
            model.state_input.weight[:, :70],
        ))
        self.assertEqual(
            int(torch.count_nonzero(model.conditional_prior_state_input.weight[:, 70:])), 0
        )
        self.assertEqual(
            int(torch.count_nonzero(model.conditional_prior_action_input.weight[:, 29:])), 0
        )
        for target, source in zip(
            model.conditional_prior_encoder.state_dict().values(),
            model.posterior_encoder.state_dict().values(),
            strict=True,
        ):
            self.assertTrue(torch.equal(target, source))

    def test_decoder_only_interface_matches_original_full_both_path(self) -> None:
        torch.manual_seed(1)
        model = initialized_model()
        value = sample_batch(2)
        full_state, full_action, _ = make_autoencode_masks(value)
        global_latent, local_latents = model.encode_posterior(
            value, full_state, full_action
        )
        original = model.decode_from_hierarchical_latent(
            value, full_state, full_action, global_latent, local_latents
        )
        isolated = model.decode_from_canonical_latents(
            global_latent,
            local_latents,
            valid_state=value["valid_state"],
            valid_action=value["valid_action"],
        )
        self.assertTrue(torch.equal(original.physical_state, isolated.physical_state))
        self.assertTrue(torch.equal(original.action, isolated.action))
        self.assertTrue(torch.equal(original.state_contact_logits, isolated.state_contact_logits))

    def test_masked_truth_can_only_affect_prior_through_visible_values(self) -> None:
        torch.manual_seed(2)
        model = initialized_model().eval()
        value = sample_batch(2)
        state_mask, action_mask, _ = make_physical_masks(value, 123)
        first = model(value, state_mask, action_mask)
        changed = dict(value)
        changed["physical_state"] = value["physical_state"].masked_fill(
            state_mask, 99_999.0
        )
        changed["action"] = value["action"].masked_fill(action_mask, -99_999.0)
        second = model(changed, state_mask, action_mask)
        self.assertTrue(torch.equal(first.global_latent, second.global_latent))
        self.assertTrue(torch.equal(first.local_latents, second.local_latents))
        self.assertTrue(torch.equal(first.physical_state, second.physical_state))
        self.assertTrue(torch.equal(first.action, second.action))

    def test_complete_token_contract_rejects_partial_features(self) -> None:
        model = initialized_model()
        value = sample_batch(2)
        state_mask, action_mask, _ = make_physical_masks(value, 456)
        partial = state_mask.clone()
        partial[0, 0, 0] = True
        partial[0, 0, 1] = False
        with self.assertRaisesRegex(ValueError, "complete State-token"):
            model(value, partial, action_mask)

    def test_training_phase_allowlists(self) -> None:
        model = initialized_model()
        prior = model.set_training_phase("prior")
        self.assertEqual(prior["trainable"], prior["prior"])
        self.assertTrue(all(
            not parameter.requires_grad
            for name, parameter in model.named_parameters()
            if not model.is_conditional_prior_parameter(name)
        ))
        interface = model.set_training_phase("decoder_interface")
        self.assertEqual(
            interface["trainable"],
            interface["prior"] + interface["decoder_interface"],
        )
        full = model.set_training_phase("decoder_full")
        self.assertGreater(full["trainable"], interface["trainable"])
        self.assertTrue(all(
            not parameter.requires_grad
            for name, parameter in model.named_parameters()
            if name.startswith("posterior_encoder.")
        ))

    def test_loss_schedule_and_two_part_reconstruction(self) -> None:
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs/posterior_hierarchical_conditional_prior_h50.json"
        )
        phases = config["training"]["phases"]
        self.assertEqual(phase_weights(1, phases), ("P0", 10.0, 1.0))
        middle = phase_weights(17_500, phases)
        self.assertEqual(middle[0], "P1")
        self.assertAlmostEqual(middle[1], 6.0)
        self.assertAlmostEqual(middle[2], 3.0)
        self.assertEqual(phase_weights(30_000, phases), ("P2", 1.0, 10.0))
        model = initialized_model()
        value = sample_batch(2)
        state_mask, action_mask, _ = make_physical_masks(value, 789)
        output = model(value, state_mask, action_mask)
        losses = full_and_masked_reconstruction_loss(
            output, value, state_mask, action_mask
        )
        self.assertTrue(torch.allclose(
            losses["total"], 0.5 * losses["full"]["total"] + 0.5 * losses["masked"]["total"]
        ))

    def test_latent_loss_standardization(self) -> None:
        predicted_global = torch.ones(2, 4)
        teacher_global = torch.zeros(2, 4)
        predicted_local = torch.ones(2, 3, 2)
        teacher_local = torch.zeros(2, 3, 2)
        losses = latent_distillation_loss(
            predicted_global, predicted_local, teacher_global, teacher_local,
            torch.full((4,), 2.0), torch.full((3, 2), 4.0),
        )
        self.assertAlmostEqual(float(losses["global"]), 0.25)
        self.assertAlmostEqual(float(losses["local"]), 0.0625)
        self.assertAlmostEqual(float(losses["total"]), 0.15625)

    def test_random_mask_protocol_is_reproducible_and_seed_isolated(self) -> None:
        value = sample_batch(len(RANDOM_TOKEN_MASK_NAMES))
        value["mask_slot"] = torch.arange(len(RANDOM_TOKEN_MASK_NAMES))
        first = make_heldout_random_token_masks(value, 20260841)
        second = make_heldout_random_token_masks(value, 20260841)
        training = make_dynamic_random_token_masks(value, 20260840, 17)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertEqual(first[2], list(RANDOM_TOKEN_MASK_NAMES))
        validate_complete_token_masks(value, training[0], training[1])
        self.assertFalse(torch.equal(first[0], training[0]) and torch.equal(first[1], training[1]))

    def test_two_window_multiple_mask_cpu_distillation_reduces_loss(self) -> None:
        torch.manual_seed(4)
        model = initialized_model()
        base = sample_batch(2)
        repeated: dict[str, object] = {}
        for name, value in base.items():
            if isinstance(value, torch.Tensor):
                repeated[name] = value.repeat_interleave(4, dim=0)
            elif isinstance(value, list):
                repeated[name] = [item for item in value for _ in range(4)]
            else:
                repeated[name] = value
        repeated["mask_slot"] = torch.tensor([0, 1, 2, 3] * 2)
        state_mask, action_mask, _ = make_heldout_random_token_masks(
            repeated, 20260841
        )
        with torch.no_grad():
            teacher_global, teacher_local = model.encode_posterior(
                repeated, state_mask, action_mask
            )
        model.set_training_phase("prior")
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )

        def objective() -> torch.Tensor:
            predicted_global, predicted_local = model.encode_conditional_prior(
                repeated, state_mask, action_mask
            )
            return (
                torch.square(predicted_global - teacher_global).mean()
                + torch.square(predicted_local - teacher_local).mean()
            )

        initial = float(objective().detach())
        for _ in range(50):
            loss = objective()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        final = float(objective().detach())
        self.assertLess(final, initial * 0.5)

    def test_decoder_adaptation_decision_table(self) -> None:
        self.assertEqual(
            decoder_adaptation_decision(False, False),
            "STOP_CONDITIONAL_PRIOR_LATENT_ALIGNMENT_FAILED",
        )
        self.assertIn("KL_THREE_PATHS", decoder_adaptation_decision(True, True))
        self.assertEqual(
            decoder_adaptation_decision(True, False),
            "RUN_CONTROLLED_DECODER_ADAPTATION_D1",
        )
        self.assertEqual(
            decoder_adaptation_decision(
                True, False, d1_scores=[1.4, 1.3, 1.2], d1_initial_score=2.0
            ),
            "RUN_CONTROLLED_DECODER_ADAPTATION_D2",
        )
        self.assertIn(
            "NOT_PROMISING",
            decoder_adaptation_decision(
                True, False, d1_scores=[1.9, 1.8, 1.7], d1_initial_score=2.0
            ),
        )
        self.assertIn(
            "NOT_PROMISING",
            decoder_adaptation_decision(
                True,
                False,
                d1_scores=[1.2, 1.3, 1.7],
                d1_initial_score=2.0,
            ),
        )

    def test_full_both_is_diagnostic_and_teacher_donors_are_measured(self) -> None:
        model = initialized_model().eval()
        value = sample_batch(2)
        state_mask, action_mask, _ = make_autoencode_masks(value)
        global_latent, local_latents = model.encode_posterior(
            value, state_mask, action_mask
        )
        decoded = model.decode_from_canonical_latents(
            global_latent,
            local_latents,
            valid_state=value["valid_state"],
            valid_action=value["valid_action"],
        )
        cache = {
            "global_latent": global_latent.detach(),
            "local_latents": local_latents.detach(),
            "physical_state": decoded.physical_state.detach(),
            "action": decoded.action.detach(),
            "state_contact_logits": decoded.state_contact_logits.detach(),
        }
        diagnostic = evaluate_full_both_conditional_prior(
            model, [value], torch.device("cpu")
        )
        self.assertIn("excluded from every PASS", diagnostic["gate_role"])
        dependence = evaluate_teacher_latent_dependence(
            model,
            [value],
            cache,
            torch.device("cpu"),
            {"cross_window": [1, 0], "cross_motion": [1, 0]},
        )
        self.assertEqual(
            set(dependence["main_ratios"]),
            {"zero", "cross_window", "cross_motion"},
        )
        cache["source_truth_metrics"] = _source_truth_metrics(
            model, [value], cache, torch.device("cpu")
        )
        preservation = evaluate_teacher_preservation(
            model,
            [value],
            cache,
            torch.device("cpu"),
            {
                "global_state_rmse": 100.0,
                "global_action_rmse": 100.0,
                "worst_mask_state_rmse": 100.0,
                "worst_mask_action_rmse": 100.0,
                "continuous_p99_abs": 100.0,
                "latent_ratio": 0.0,
            },
            {
                "source_metric_multiplier": 1.05,
                "functional_state_rmse": 1e-8,
                "functional_action_rmse": 1e-8,
                "functional_p99_abs": 1e-8,
            },
            {"cross_window": [1, 0], "cross_motion": [1, 0]},
        )
        self.assertEqual(
            preservation["metrics"]["functional_contact_agreement"], 1.0
        )
        self.assertLess(
            preservation["metrics"]["functional_state_rmse"], 1e-8
        )

    def test_checkpoint_readback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            torch.save({
                "format_version": "sonic_h50_cpd_checkpoint_v1",
                "mode": "train",
                "dataset_manifest_sha256": "dataset",
                "selected_windows_sha256": "windows",
                "source_checkpoint_sha256": "source",
                "teacher_cache_sha256": "teacher",
                "model": {}, "optimizer": {}, "scheduler": {},
            }, path)
            result = validate_checkpoint(path, {
                "mode": "train", "dataset": "dataset", "windows": "windows",
                "source": "source", "teacher_cache": "teacher",
            })
            self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
