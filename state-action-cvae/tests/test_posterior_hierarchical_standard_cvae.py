from __future__ import annotations

import gc
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

from cvae_sa.models import (
    HierarchicalPosteriorTransformer,
    HierarchicalStandardCVAETransformer,
    build_model,
    parameter_count,
)
from cvae_sa.posterior_complete_token_protocol import (
    PHYSICAL_RANDOM_MASK_NAMES,
    make_dynamic_physical_random_masks,
    make_heldout_physical_random_masks,
    validate_physically_inferable_token_masks,
)
from cvae_sa.posterior_hierarchical_standard_cvae import (
    _PathView,
    _parameter_groups,
    alignment_weight,
    evaluate_distribution_statistics,
    evaluate_prior_output_diversity,
    hierarchical_kl,
    initialize_from_h50_a,
    mean_alignment_loss,
    weighted_reconstruction_loss,
)
from cvae_sa.posterior_t64_protocol import make_physical_masks
from cvae_sa.util import load_config


def small_config() -> dict[str, object]:
    return {
        "kind": "physics_hierarchical_standard_cvae_transformer",
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


def initialized_model() -> HierarchicalStandardCVAETransformer:
    config = small_config()
    source_config = dict(config)
    source_config["kind"] = "physics_hierarchical_posterior_transformer"
    source = HierarchicalPosteriorTransformer(source_config)
    model = HierarchicalStandardCVAETransformer(config)
    initialize_from_h50_a(model, {"model": source.state_dict()})
    return model


class HierarchicalStandardCVAETest(unittest.TestCase):
    def test_reference_parameter_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/posterior_hierarchical_standard_cvae_h50.json"
        )
        config["model"]["state_dim"] = 70
        with torch.device("meta"):
            model = build_model(config["model"])
        self.assertIsInstance(model, HierarchicalStandardCVAETransformer)
        self.assertEqual(parameter_count(model), 66_129_571)
        logvar = sum(
            parameter.numel() for name, parameter in model.named_parameters()
            if model.is_logvar_parameter(name)
        )
        self.assertEqual(logvar, 344_832)
        del model
        gc.collect()

    def test_h50_initialization_and_pre_kl_freeze(self) -> None:
        model = initialized_model()
        self.assertEqual(
            int(torch.count_nonzero(model.conditional_prior_state_input.weight[:, 70:])), 0
        )
        self.assertEqual(
            int(torch.count_nonzero(model.conditional_prior_action_input.weight[:, 29:])), 0
        )
        self.assertTrue(torch.all(model.posterior_global_logvar_head.bias == -4.0))
        counts = model.set_standard_training_phase("mean")
        self.assertTrue(all(
            not parameter.requires_grad
            for name, parameter in model.named_parameters()
            if model.is_logvar_parameter(name)
        ))
        self.assertEqual(counts["trainable"], counts["mean"])
        kl_counts = model.set_standard_training_phase("kl")
        self.assertEqual(
            kl_counts["trainable"], kl_counts["mean"] + kl_counts["logvar"]
        )

    def test_prior_and_decoder_condition_do_not_read_masked_truth(self) -> None:
        torch.manual_seed(2)
        model = initialized_model().eval()
        value = sample_batch(2)
        state_mask, action_mask, _ = make_physical_masks(value, 456)
        prior = model.encode_conditional_prior_distribution(value, state_mask, action_mask)
        first = model.decode_from_conditioned_latents(
            value, state_mask, action_mask, prior.global_mean, prior.local_mean
        )
        changed = dict(value)
        changed["physical_state"] = value["physical_state"].masked_fill(state_mask, 99_999.0)
        changed["action"] = value["action"].masked_fill(action_mask, -99_999.0)
        changed_prior = model.encode_conditional_prior_distribution(
            changed, state_mask, action_mask
        )
        second = model.decode_from_conditioned_latents(
            changed, state_mask, action_mask,
            changed_prior.global_mean, changed_prior.local_mean,
        )
        for left, right in (
            (prior.global_mean, changed_prior.global_mean),
            (prior.local_mean, changed_prior.local_mean),
            (first.physical_state, second.physical_state),
            (first.action, second.action),
        ):
            self.assertTrue(torch.equal(left, right))

    def test_decoder_uses_visible_condition_with_latent_held_fixed(self) -> None:
        torch.manual_seed(21)
        model = initialized_model().eval()
        value = sample_batch(2)
        state_mask, action_mask, _ = make_physical_masks(value, 654)
        prior = model.encode_conditional_prior_distribution(value, state_mask, action_mask)
        first = model.decode_from_conditioned_latents(
            value, state_mask, action_mask, prior.global_mean, prior.local_mean
        )
        changed = dict(value)
        changed_state = value["physical_state"].clone()
        visible_state = ~state_mask
        changed_state[visible_state] += 3.0
        changed["physical_state"] = changed_state
        second = model.decode_from_conditioned_latents(
            changed, state_mask, action_mask, prior.global_mean, prior.local_mean
        )
        self.assertFalse(torch.equal(first.physical_state, second.physical_state))
        self.assertFalse(torch.equal(first.action, second.action))

    def test_posterior_reads_truth_and_knows_current_mask(self) -> None:
        torch.manual_seed(3)
        model = initialized_model().eval()
        value = sample_batch(2)
        state_mask, action_mask, _ = make_physical_masks(value, 789)
        first = model.encode_posterior_distribution(value, state_mask, action_mask)
        changed = dict(value)
        changed["physical_state"] = value["physical_state"].masked_fill(state_mask, 17.0)
        changed["action"] = value["action"].masked_fill(action_mask, -19.0)
        second = model.encode_posterior_distribution(changed, state_mask, action_mask)
        self.assertFalse(torch.equal(first.global_mean, second.global_mean))
        alternate_state = torch.zeros_like(state_mask)
        alternate_action = action_mask.clone()
        alternate_action[:, :2] = True
        alternate = model.encode_posterior_distribution(
            value, alternate_state, alternate_action
        )
        self.assertFalse(torch.equal(first.global_mean, alternate.global_mean))

    def test_deployment_is_prior_only_and_one_latent_per_decoder_call(self) -> None:
        torch.manual_seed(4)
        model = initialized_model().eval()
        value = sample_batch(2)
        state_mask, action_mask, _ = make_physical_masks(value, 111)
        model.encode_posterior_distribution = Mock(
            side_effect=AssertionError("deployment must not call posterior")
        )
        output = model.infer_from_conditional_prior(
            value, state_mask, action_mask, sample=False
        )
        self.assertIsNone(output.posterior)
        self.assertEqual(output.latent_source, "prior_mean")
        self.assertIsNotNone(output.prior)
        assert output.prior is not None
        self.assertTrue(torch.equal(output.global_latent, output.prior.global_mean))
        self.assertTrue(torch.equal(output.local_latents, output.prior.local_mean))

    def test_posterior_forward_does_not_evaluate_or_fuse_prior(self) -> None:
        torch.manual_seed(41)
        model = initialized_model().eval()
        value = sample_batch(2)
        state_mask, action_mask, _ = make_physical_masks(value, 112)
        model.encode_conditional_prior_distribution = Mock(
            side_effect=AssertionError("posterior path must not evaluate prior")
        )
        output = model(
            value, state_mask, action_mask, latent_source="posterior_mean"
        )
        self.assertIsNotNone(output.posterior)
        self.assertIsNone(output.prior)
        self.assertEqual(output.latent_source, "posterior_mean")

    def test_reconstruction_alignment_and_kl_formulas(self) -> None:
        torch.manual_seed(5)
        model = initialized_model().eval()
        value = sample_batch(2)
        state_mask, action_mask, _ = make_physical_masks(value, 222)
        output = model(value, state_mask, action_mask, latent_source="prior_mean")
        losses = weighted_reconstruction_loss(
            output, value, state_mask, action_mask
        )
        self.assertTrue(torch.allclose(
            losses["total"],
            0.75 * losses["masked"]["total"] + 0.25 * losses["full"]["total"],
        ))
        posterior = model.encode_posterior_distribution(value, state_mask, action_mask)
        prior = model.encode_conditional_prior_distribution(value, state_mask, action_mask)
        alignment = mean_alignment_loss(
            posterior, prior,
            torch.ones_like(posterior.global_mean[0]),
            torch.ones_like(posterior.local_mean[0]),
        )
        alignment["total"].backward()
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.posterior_encoder.parameters()
        ))
        self.assertIsNotNone(model.conditional_prior_global_head.weight.grad)
        identical_kl = hierarchical_kl(posterior, posterior)
        self.assertAlmostEqual(float(identical_kl["total"]), 0.0, places=6)

    def test_loss_schedule_and_optimizer_cover_full_model(self) -> None:
        schedule = [
            {"start": 0, "end": 10, "start_value": 1.0, "end_value": 1.0},
            {"start": 10, "end": 30, "start_value": 1.0, "end_value": 0.1},
        ]
        self.assertEqual(alignment_weight(5, schedule), 1.0)
        self.assertAlmostEqual(alignment_weight(20, schedule), 0.55)
        model = initialized_model()
        optimizer, _, contract = _parameter_groups(model, "fixed", {
            "encoder_learning_rate": 1e-4,
            "decoder_learning_rate": 3e-5,
            "minimum_learning_rate": 1e-6,
            "warmup_steps": 10,
        }, 100)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(
            contract["phase_counts"]["trainable"],
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        )

    def test_physical_random_masks_are_complete_reproducible_and_separated(self) -> None:
        value = sample_batch(len(PHYSICAL_RANDOM_MASK_NAMES))
        value["mask_slot"] = torch.arange(len(PHYSICAL_RANDOM_MASK_NAMES))
        first = make_heldout_physical_random_masks(value, 20260841)
        second = make_heldout_physical_random_masks(value, 20260841)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertEqual(first[2], list(PHYSICAL_RANDOM_MASK_NAMES))
        validate_physically_inferable_token_masks(value, first[0], first[1])
        for slot in range(8, len(PHYSICAL_RANDOM_MASK_NAMES)):
            active = first[1][slot].any(dim=-1) | first[0][slot, 1:].any(dim=-1)
            active_indices = torch.nonzero(active).flatten().tolist()
            runs = 1 + sum(
                current > previous + 1
                for previous, current in zip(active_indices, active_indices[1:])
            )
            self.assertEqual(runs, 2 if slot < 12 else 3)
        dynamic = make_dynamic_physical_random_masks(value, 20260840, 7, 3)
        validate_physically_inferable_token_masks(value, dynamic[0], dynamic[1])
        self.assertFalse(
            torch.equal(first[0], dynamic[0]) and torch.equal(first[1], dynamic[1])
        )

    def test_shared_epsilon_is_identical_for_q_and_p_paths(self) -> None:
        torch.manual_seed(6)
        model = initialized_model().eval()
        value = sample_batch(2)
        state_mask, action_mask, _ = make_physical_masks(value, 333)
        posterior = model.encode_posterior_distribution(value, state_mask, action_mask)
        prior = model.encode_conditional_prior_distribution(value, state_mask, action_mask)
        q_view = _PathView(model, "posterior_sample", sample_index=3)
        p_view = _PathView(model, "prior_sample", sample_index=3)
        q_epsilon = q_view._epsilon(value, posterior)
        p_epsilon = p_view._epsilon(value, prior)
        self.assertTrue(torch.equal(q_epsilon[0], p_epsilon[0]))
        self.assertTrue(torch.equal(q_epsilon[1], p_epsilon[1]))

    def test_kl_statistics_and_prior_output_diversity_are_finite(self) -> None:
        torch.manual_seed(61)
        model = initialized_model().eval()
        value = sample_batch(2)
        maker = lambda batch: make_physical_masks(batch, 333)
        statistics = evaluate_distribution_statistics(
            model, [value], maker, torch.device("cpu")
        )
        diversity = evaluate_prior_output_diversity(
            model, [value], maker, torch.device("cpu"),
            count=2, sample_seed=20260841,
        )
        self.assertTrue(statistics["finite"])
        self.assertTrue(diversity["finite"])
        self.assertEqual(diversity["sample_count"], 2)
        self.assertGreaterEqual(
            diversity["masked_continuous_mean_standard_deviation"], 0.0
        )

    def test_two_window_cpu_joint_mean_training_reduces_objective(self) -> None:
        torch.manual_seed(7)
        model = initialized_model()
        value = sample_batch(2)
        state_mask, action_mask, _ = make_physical_masks(value, 444)
        model.set_standard_training_phase("mean")
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=2e-3,
        )
        global_std = torch.ones(model.global_latent_dim)
        local_std = torch.ones(model.local_chunks, model.local_latent_dim)

        def objective() -> torch.Tensor:
            posterior = model.encode_posterior_distribution(value, state_mask, action_mask)
            prior = model.encode_conditional_prior_distribution(value, state_mask, action_mask)
            posterior_output = model.decode_from_conditioned_latents(
                value, state_mask, action_mask,
                posterior.global_mean, posterior.local_mean,
            )
            prior_output = model.decode_from_conditioned_latents(
                value, state_mask, action_mask, prior.global_mean, prior.local_mean,
            )
            q_loss = weighted_reconstruction_loss(
                posterior_output, value, state_mask, action_mask
            )["total"]
            p_loss = weighted_reconstruction_loss(
                prior_output, value, state_mask, action_mask
            )["total"]
            align = mean_alignment_loss(
                posterior, prior, global_std, local_std
            )["total"]
            return p_loss + 0.25 * q_loss + align

        initial = float(objective().detach())
        for _ in range(20):
            loss = objective()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        final = float(objective().detach())
        self.assertLess(final, initial * 0.85)


if __name__ == "__main__":
    unittest.main()
