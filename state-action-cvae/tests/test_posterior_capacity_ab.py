from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from cvae_sa.models import PosteriorCapacityTransformerCVAE, build_model, parameter_count
from cvae_sa.posterior_capacity_ab import (
    CHECKPOINT_FORMAT,
    COMPARISON_FORMAT,
    COMPARISON_MARKER,
    EXPECTED_C_PARAMETERS,
    EXECUTION_MARKER,
    FORMAT_VERSION,
    PROGRESSION_MARKER,
    SMOKE_MARKER,
    _residual_ratio,
    _write_failure_manifest,
    ab_reconstruction_objective,
    batch_sample_identities,
    compare_candidate,
    donor_index_maps,
    evaluate_full_objective,
    identity_sha256,
    initial_decision,
    latent_gate_diagnostics,
    load_source_model_weights,
    replication_decision,
    render_training_plots,
    run_ab_comparison,
    tail_mixed_domain_loss,
    validate_f4c_trigger,
    validate_saved_checkpoint,
    validate_step0,
)


class _ZeroModel(torch.nn.Module):
    def forward(self, batch, state_mask, action_mask):
        del state_mask, action_mask
        return SimpleNamespace(
            physical_state=torch.zeros_like(batch["physical_state"]),
            action=torch.zeros_like(batch["action"]),
            state_contact_logits=torch.zeros_like(batch["physical_state"][..., 68:70]),
        )


class _IdentityDecoderLayer(torch.nn.Module):
    def forward(self, value, valid_tokens, times, causal):
        del valid_tokens, times, causal
        return value


def _output(batch: dict[str, torch.Tensor]) -> SimpleNamespace:
    state = torch.zeros_like(batch["physical_state"], requires_grad=True)
    action = torch.zeros_like(batch["action"], requires_grad=True)
    contact = torch.zeros_like(batch["physical_state"][..., 68:70], requires_grad=True)
    return SimpleNamespace(
        physical_state=state,
        action=action,
        state_contact_logits=contact,
    )


def _batch() -> dict[str, torch.Tensor]:
    state = torch.ones(2, 2, 70)
    state[..., 68:70] = 0.0
    return {
        "physical_state": state,
        "action": torch.ones(2, 1, 29),
    }


def _posterior_model_config(*, gates: bool, decoder_layers: int = 2) -> dict[str, object]:
    return {
        "kind": "physics_posterior_transformer",
        "d_model": 32,
        "encoder_layers": 1,
        "decoder_layers": decoder_layers,
        "heads": 4,
        "ffn_dim": 64,
        "latent_dim": 16,
        "dropout": 0.0,
        "state_dim": 70,
        "decoder_layer_latent_gates": gates,
    }


def _posterior_batch(batch_size: int = 2, transitions: int = 3) -> dict[str, object]:
    state = torch.randn(batch_size, transitions + 1, 70)
    state[..., 68:70] = torch.randint(0, 2, state[..., 68:70].shape).float()
    return {
        "physical_state": state,
        "action": torch.randn(batch_size, transitions, 29),
        "valid_state": torch.ones(batch_size, transitions + 1, dtype=torch.bool),
        "valid_action": torch.ones(batch_size, transitions, dtype=torch.bool),
    }


def _evaluation(
    step: int,
    *,
    exceed: float,
    maximum: float,
    progression: bool = False,
    scale: float = 1.0,
) -> dict[str, object]:
    return {
        "optimizer_step": step,
        "exact": {
            "progression_gate": {"passed": progression, "score": maximum / 0.01},
            "worst_state_rmse": 0.009 * scale,
            "worst_action_rmse": 0.008 * scale,
            "continuous_max_abs": maximum,
            "contact_accuracy": 1.0,
            "latent_dependence": {"zero_ratio": 12.0},
        },
        "tail_global": {
            "global_state_rmse": 0.006 * scale,
            "global_action_rmse": 0.005 * scale,
            "threshold_exceed_fraction": exceed,
        },
    }


def _summary(
    arm: str,
    *,
    exceed: float,
    maximum: float,
    progression: bool = False,
    scale: float = 1.0,
) -> dict[str, object]:
    evaluations = [
        _evaluation(0, exceed=0.21, maximum=9.64, scale=1.0),
        *[
            _evaluation(
                step,
                exceed=exceed,
                maximum=maximum,
                progression=progression,
                scale=scale,
            )
            for step in (8000, 9000, 10000)
        ],
    ]
    is_c = arm == "C"
    source = {
        "checkpoint_sha256": "checkpoint",
        "f4a_manifest_sha256": "f4a",
    }
    if is_c:
        source["f4c_trigger_comparison"] = {
            "decision": "IMPLEMENT_F4C",
            "checks": {"authorized": True},
        }
    return {
        "execution_pass": True,
        "smoke": False,
        "arm": arm,
        "fixture_seed": 20260830,
        "optimizer_seed": 20260830,
        "dataset_manifest_sha256": "dataset",
        "source": source,
        "data_contract": {
            "fixture_bitmap_sha256": "fixture",
            "selected_windows_sha256": "windows",
            "fixed_velocity_cases_sha256": "curves",
            "identity_contract_sha256": "identity-contract",
            "window_count": 80,
            "fixture_count": 800,
        },
        "model_contract": {
            "parameter_count": EXPECTED_C_PARAMETERS if is_c else 25_453_411,
            "f4c_layer_gates_enabled": is_c,
            "f4c_layer_gate_parameter_count": 3072 if is_c else 0,
        },
        "training_contract": {
            "training_identity_sha256_by_step": {
                str(step): f"sample-{step}" for step in range(1, 10001)
            }
        },
        "evaluations": evaluations,
    }


class PosteriorCapacityABTest(unittest.TestCase):
    def test_fixed_ab_config_has_exact_model_and_optimizer_contract(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs/posterior_capacity_ab.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["model"]["state_dim"] = 70
        self.assertEqual(parameter_count(build_model(config["model"])), 25_453_411)
        gated = copy.deepcopy(config["model"])
        gated["decoder_layer_latent_gates"] = True
        self.assertEqual(parameter_count(build_model(gated)), EXPECTED_C_PARAMETERS)
        self.assertEqual(config["training"]["max_optimizer_steps"], 10_000)
        self.assertEqual(config["training"]["posterior_path"], "mean")
        self.assertEqual(config["training"]["kl_beta"], 0.0)
        self.assertEqual(config["training"]["weight_decay"], 0.0)

    def test_f4c_zero_gate_migration_is_exact_and_rejects_other_missing_keys(self) -> None:
        torch.manual_seed(17)
        baseline = PosteriorCapacityTransformerCVAE(
            _posterior_model_config(gates=False)
        ).eval()
        gated = PosteriorCapacityTransformerCVAE(
            _posterior_model_config(gates=True)
        ).eval()
        migration = load_source_model_weights(gated, baseline.state_dict(), "C")
        self.assertEqual(len(migration["missing_keys"]), 2)
        self.assertEqual(migration["unexpected_keys"], [])
        self.assertEqual(
            latent_gate_diagnostics(gated)["nonzero_parameter_count"], 0
        )

        value = _posterior_batch()
        state_mask = torch.ones_like(value["physical_state"], dtype=torch.bool)
        action_mask = torch.ones_like(value["action"], dtype=torch.bool)
        baseline_output = baseline(value, state_mask, action_mask)
        gated_output = gated(value, state_mask, action_mask)
        self.assertTrue(torch.equal(baseline_output.physical_state, gated_output.physical_state))
        self.assertTrue(torch.equal(baseline_output.action, gated_output.action))

        incomplete = dict(baseline.state_dict())
        incomplete.pop("state_input.weight")
        with self.assertRaisesRegex(ValueError, "source migration mismatch"):
            load_source_model_weights(gated, incomplete, "C")

    def test_f4c_checkpoint_readback_requires_all_finite_gate_tensors(self) -> None:
        gate_state = {
            f"decoder_layer_latent_gates.{index}": torch.full((384,), index + 1.0)
            for index in range(8)
        }
        payload = {
            "format_version": CHECKPOINT_FORMAT,
            "optimizer_step": 2,
            "model": {"dummy": torch.zeros(1), **gate_state},
            "optimizer": {"state": {}},
            "scheduler": {"last_epoch": 2},
            "resolved_config": {
                "model": {"decoder_layer_latent_gates": True},
            },
            "dataset_manifest_sha256": "dataset",
            "source_checkpoint_sha256": "source",
            "fixture_bitmap_sha256": "fixtures",
            "parameter_count": EXPECTED_C_PARAMETERS,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            torch.save(payload, checkpoint_path)
            result = validate_saved_checkpoint(
                checkpoint_path,
                expected_step=2,
                dataset_hash="dataset",
                source_checkpoint_hash="source",
                fixture_hash="fixtures",
                expected_parameters=EXPECTED_C_PARAMETERS,
                f4c_layer_gates_enabled=True,
            )
            self.assertTrue(result["passed"])

            broken = copy.deepcopy(payload)
            del broken["model"]["decoder_layer_latent_gates.7"]
            torch.save(broken, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "saved-checkpoint readback failed"):
                validate_saved_checkpoint(
                    checkpoint_path,
                    expected_step=2,
                    dataset_hash="dataset",
                    source_checkpoint_hash="source",
                    fixture_hash="fixtures",
                    expected_parameters=EXPECTED_C_PARAMETERS,
                    f4c_layer_gates_enabled=True,
                )

    def test_f4c_layer_injection_skips_latent_token_and_padding(self) -> None:
        model = PosteriorCapacityTransformerCVAE(
            _posterior_model_config(gates=True)
        )
        model.decoder.layers = torch.nn.ModuleList(
            [_IdentityDecoderLayer(), _IdentityDecoderLayer()]
        )
        model.decoder.norm = torch.nn.Identity()
        for gate in model.decoder_layer_latent_gates:
            gate.data.fill_(1.0)
        value = torch.zeros(1, 4, 32)
        valid = torch.tensor([[True, True, False, True]])
        condition = torch.arange(32, dtype=torch.float32).unsqueeze(0)
        decoded = model._decode_tokens(
            value, valid, torch.arange(4), condition
        )
        self.assertTrue(torch.equal(decoded[:, 0], value[:, 0]))
        self.assertTrue(torch.equal(decoded[:, 2], value[:, 2]))
        self.assertTrue(torch.equal(decoded[:, 1], 2.0 * condition))
        self.assertTrue(torch.equal(decoded[:, 3], 2.0 * condition))

    def test_f4c_gates_receive_gradients_and_preserve_truth_isolation(self) -> None:
        torch.manual_seed(23)
        model = PosteriorCapacityTransformerCVAE(
            _posterior_model_config(gates=True)
        ).eval()
        value = _posterior_batch()
        state_mask = torch.ones_like(value["physical_state"], dtype=torch.bool)
        action_mask = torch.ones_like(value["action"], dtype=torch.bool)
        first = model(value, state_mask, action_mask)
        objective = ab_reconstruction_objective(
            first, value, state_mask, action_mask, "C"
        )
        objective.optimization_total.backward()
        diagnostic = latent_gate_diagnostics(model, include_gradients=True)
        self.assertTrue(diagnostic["all_gradients_present"])
        self.assertGreater(diagnostic["total_gradient_l2_norm_before_clip"], 0.0)
        torch.optim.SGD(model.parameters(), lr=1e-3).step()
        self.assertGreater(
            latent_gate_diagnostics(model)["nonzero_parameter_count"], 0
        )

        fixed_latent = first.posterior_mean.detach()
        changed = dict(value)
        changed["physical_state"] = value["physical_state"] + 100.0
        changed["action"] = value["action"] - 100.0
        fixed = model(value, state_mask, action_mask, latent_override=fixed_latent)
        changed_output = model(
            changed, state_mask, action_mask, latent_override=fixed_latent
        )
        self.assertTrue(torch.equal(fixed.physical_state, changed_output.physical_state))
        self.assertTrue(torch.equal(fixed.action, changed_output.action))

    def test_f4c_requires_a_hash_verified_triggering_comparison(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repository) as temporary:
            root = Path(temporary)
            inputs = {}
            for arm in ("A", "B"):
                run = root / arm
                (run / "manifests").mkdir(parents=True)
                summary = _summary(arm, exceed=0.1, maximum=0.02)
                summary["format_version"] = FORMAT_VERSION
                path = run / "manifests/posterior_ab_summary.json"
                path.write_text(json.dumps(summary), encoding="utf-8")
                inputs[arm] = {
                    "run": str(run.resolve()),
                    "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            comparison = root / "comparison"
            (comparison / "manifests").mkdir(parents=True)
            (comparison / "markers").mkdir(parents=True)
            manifest = {
                "format_version": COMPARISON_FORMAT,
                "execution_pass": True,
                "comparison_phase": "initial",
                "inputs": inputs,
                "decision": {
                    "decision": "IMPLEMENT_F4C",
                    "implement_f4c": True,
                    "candidate_comparisons": {"B": {"pairing": {"passed": True}}},
                },
            }
            (comparison / "manifests/posterior_ab_comparison.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (comparison / f"markers/{COMPARISON_MARKER}").write_text(
                "PASS decision=IMPLEMENT_F4C\n", encoding="utf-8"
            )
            validated = validate_f4c_trigger(
                comparison,
                dataset_hash="dataset",
                source_checkpoint_hash="checkpoint",
                optimizer_seed=20260830,
            )
            self.assertEqual(validated["decision"], "IMPLEMENT_F4C")
            with self.assertRaisesRegex(ValueError, "explicit triggering"):
                validate_f4c_trigger(
                    None,
                    dataset_hash="dataset",
                    source_checkpoint_hash="checkpoint",
                    optimizer_seed=20260830,
                )

    def test_arm_a_is_exact_old_objective_and_backpropagates(self) -> None:
        batch = _batch()
        output = _output(batch)
        state_mask = torch.ones_like(batch["physical_state"], dtype=torch.bool)
        action_mask = torch.ones_like(batch["action"], dtype=torch.bool)
        objective = ab_reconstruction_objective(
            output, batch, state_mask, action_mask, "A"
        )
        self.assertIs(objective.optimization_total, objective.raw.total)
        objective.optimization_total.backward()
        self.assertIsNotNone(output.physical_state.grad)
        self.assertIsNotNone(output.action.grad)
        self.assertIsNotNone(output.state_contact_logits.grad)

    def test_arm_b_matches_fixture_weighted_top_twenty_percent_math(self) -> None:
        squared = torch.tensor([[[1.0, 4.0, 100.0]], [[1.0, 1.0, 1.0]]])
        mask = torch.tensor([[[True, True, False]], [[True, True, True]]])
        # Fixture 0: 0.5*2.5 + 0.5*4 = 3.25, weighted by 2.
        # Fixture 1: both mean and top-1 are 1, weighted by 3.
        expected = (2 * 3.25 + 3 * 1.0) / 5
        observed = tail_mixed_domain_loss(squared, mask)
        assert observed is not None
        self.assertAlmostEqual(float(observed), expected)

        uniform = tail_mixed_domain_loss(
            torch.full((1, 1, 7), 3.0), torch.ones(1, 1, 7, dtype=torch.bool)
        )
        single = tail_mixed_domain_loss(
            torch.tensor([[[5.0, 999.0]]]), torch.tensor([[[True, False]]])
        )
        self.assertEqual(float(uniform), 3.0)
        self.assertEqual(float(single), 5.0)
        self.assertIsNone(tail_mixed_domain_loss(
            torch.ones(1, 1, 2), torch.zeros(1, 1, 2, dtype=torch.bool)
        ))

    def test_arm_b_keeps_state_action_separate_and_handles_empty_domains(self) -> None:
        batch = _batch()
        output = _output(batch)
        state_mask = torch.zeros_like(batch["physical_state"], dtype=torch.bool)
        action_mask = torch.ones_like(batch["action"], dtype=torch.bool)
        objective = ab_reconstruction_objective(
            output, batch, state_mask, action_mask, "B"
        )
        self.assertEqual(float(objective.optimization_state), 0.0)
        self.assertGreater(float(objective.optimization_action), 0.0)
        arm_c = ab_reconstruction_objective(output, batch, state_mask, action_mask, "C")
        self.assertIs(arm_c.optimization_total, arm_c.raw.total)
        with self.assertRaisesRegex(ValueError, "arm A, B, or C"):
            ab_reconstruction_objective(output, batch, state_mask, action_mask, "D")
        with self.assertRaisesRegex(ValueError, "no targets"):
            ab_reconstruction_objective(
                output,
                batch,
                torch.zeros_like(state_mask),
                torch.zeros_like(action_mask),
                "B",
            )

    def test_full_evaluation_aggregates_raw_and_optimized_components(self) -> None:
        items = []
        for slot in (0, 1):
            state = torch.ones(2, 70)
            state[..., 68:70] = 0.0
            items.append({
                "physical_state": state,
                "action": torch.ones(1, 29),
                "valid_state": torch.ones(2, dtype=torch.bool),
                "valid_action": torch.ones(1, dtype=torch.bool),
                "motion_key": "motion",
                "variant_id": 0,
                "episode_ref": "episode",
                "window_start": 0,
                "source_window_index": 0,
                "window_index": 0,
                "mask_slot": slot,
            })
        result = evaluate_full_objective(
            _ZeroModel(),
            DataLoader(items, batch_size=2, shuffle=False),
            torch.device("cpu"),
            arm="B",
            tail_fraction=0.2,
            tail_mix=0.5,
        )
        self.assertAlmostEqual(result["raw_reconstruction"]["state"], 1.0)
        self.assertAlmostEqual(result["raw_reconstruction"]["action"], 1.0)
        self.assertAlmostEqual(result["optimization_objective"]["state"], 1.0)
        self.assertAlmostEqual(result["optimization_objective"]["action"], 1.0)
        self.assertEqual(result["raw_reconstruction"]["counts"]["state"], 2 * 68)
        self.assertEqual(result["raw_reconstruction"]["counts"]["action"], 29)

    def test_sample_identity_hash_is_stable_and_includes_mask(self) -> None:
        batch = {
            "motion_key": ["m0", "m1"],
            "variant_id": torch.tensor([0, 1]),
            "episode_ref": ["e0", "e1"],
            "window_start": torch.tensor([0, 128]),
            "source_window_index": torch.tensor([4, 7]),
            "window_index": torch.tensor([0, 1]),
            "mask_slot": torch.tensor([0, 2]),
        }
        identities = batch_sample_identities(batch)
        self.assertEqual(identities[1]["mask_name"], "full_both")
        self.assertEqual(identity_sha256(identities), identity_sha256(copy.deepcopy(identities)))
        changed = copy.deepcopy(identities)
        changed[0]["mask_slot"] = 1
        self.assertNotEqual(identity_sha256(identities), identity_sha256(changed))

    def test_donor_maps_are_real_cross_window_and_cross_motion(self) -> None:
        identities = [
            {
                "motion_key": motion,
                "variant_id": index,
                "episode_ref": f"{motion}-{index}",
                "window_start": 0,
                "source_window_index": index + group * 3,
            }
            for group, motion in enumerate(("a", "b"))
            for index in range(3)
        ]
        maps = donor_index_maps(identities)
        self.assertTrue(all(index != donor for index, donor in enumerate(maps["cross_window"])))
        self.assertTrue(all(
            identities[index]["motion_key"] != identities[donor]["motion_key"]
            for index, donor in enumerate(maps["cross_motion"])
        ))

    def test_step0_reproduction_uses_strict_metrics(self) -> None:
        exact = {
            "worst_state_rmse": 0.2,
            "worst_action_rmse": 0.3,
            "continuous_max_abs": 4.0,
            "contact_accuracy": 1.0,
        }
        tail = {"global": {
            "fixture_count": 800,
            "global_state_rmse": 0.1,
            "global_action_rmse": 0.2,
        }}
        source = {"mask_fixture_count": 800, "best_metrics": {
            "optimizer_step": 10,
            "worst_state_rmse": 0.2,
            "worst_action_rmse": 0.3,
            "continuous_max_abs": 4.0,
            "contact_accuracy": 1.0,
            "reconstruction_loss": {"state": 0.01, "action": 0.04},
        }}
        self.assertTrue(validate_step0(exact, tail, source, {"step": 10})["passed"])
        exact["continuous_max_abs"] = 4.1
        with self.assertRaisesRegex(ValueError, "step0"):
            validate_step0(exact, tail, source, {"step": 10})

    def test_decision_prefers_a_pass_then_b_strong_and_only_then_c(self) -> None:
        passing_a = _summary("A", exceed=0.1, maximum=0.009, progression=True)
        arm_b = _summary("B", exceed=0.02, maximum=0.004)
        self.assertEqual(initial_decision(passing_a, arm_b)["decision"], "REPLICATE_A_ONLY")

        failing_a = _summary("A", exceed=0.1, maximum=0.02)
        strong_b = _summary("B", exceed=0.04, maximum=0.009)
        comparison = compare_candidate(failing_a, strong_b)
        self.assertTrue(comparison["strong_improvement"])
        self.assertEqual(initial_decision(failing_a, strong_b)["decision"], "REPLICATE_A_AND_B")

        weak_b = _summary("B", exceed=0.09, maximum=0.019)
        self.assertEqual(initial_decision(failing_a, weak_b)["decision"], "IMPLEMENT_F4C")
        mismatched_seed = copy.deepcopy(strong_b)
        mismatched_seed["optimizer_seed"] = 20260831
        with self.assertRaisesRegex(ValueError, "optimizer_seed"):
            compare_candidate(failing_a, mismatched_seed)
        non_finite = copy.deepcopy(strong_b)
        non_finite["evaluations"][1]["exact"]["continuous_max_abs"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare_candidate(failing_a, non_finite)

    def test_c_selection_requires_guarded_twenty_percent_gain_and_ties_choose_b(self) -> None:
        arm_a = _summary("A", exceed=0.1, maximum=0.02)
        arm_b = _summary("B", exceed=0.08, maximum=0.016)
        arm_c = _summary("C", exceed=0.08, maximum=0.016)
        decision = initial_decision(arm_a, arm_b, arm_c)
        self.assertEqual(decision["decision"], "REPLICATE_A_AND_B")
        self.assertEqual(decision["winner"], "B")

        bad_b = _summary("B", exceed=0.09, maximum=0.019)
        bad_c = _summary("C", exceed=0.07, maximum=0.014, scale=1.2)
        stopped = initial_decision(arm_a, bad_b, bad_c)
        self.assertEqual(stopped["decision"], "STOP_LOSS_LATENT_SEED_SEARCH")

    def test_residual_zero_denominator_does_not_claim_improvement(self) -> None:
        self.assertEqual(_residual_ratio(0.0, 0.0), 1.0)
        self.assertEqual(_residual_ratio(0.5, 0.0), float("inf"))

    def test_main_residual_is_ratio_of_three_step_medians(self) -> None:
        arm_a = _summary("A", exceed=0.1, maximum=0.02)
        arm_b = _summary("B", exceed=0.04, maximum=0.01)
        for index, (a_p, b_p, a_m, b_m) in enumerate((
            (0.1, 0.04, 0.02, 0.01),
            (0.2, 0.10, 0.04, 0.02),
            (0.3, 0.25, 0.06, 0.05),
        ), start=1):
            arm_a["evaluations"][index]["tail_global"]["threshold_exceed_fraction"] = a_p
            arm_b["evaluations"][index]["tail_global"]["threshold_exceed_fraction"] = b_p
            arm_a["evaluations"][index]["exact"]["continuous_max_abs"] = a_m
            arm_b["evaluations"][index]["exact"]["continuous_max_abs"] = b_m
        comparison = compare_candidate(arm_a, arm_b)
        self.assertAlmostEqual(comparison["residuals"]["R_p"], 0.5)
        self.assertAlmostEqual(comparison["residuals"]["R_a"], 0.5)

    def test_replication_decision_requires_second_seed_and_quality_pass(self) -> None:
        passing_first_a = _summary("A", exceed=0.02, maximum=0.009, progression=True)
        unused_b = _summary("B", exceed=0.02, maximum=0.009)
        a_only_initial = {"decision": initial_decision(passing_first_a, unused_b)}
        passing_second_a = copy.deepcopy(passing_first_a)
        passing_second_a["optimizer_seed"] = 20260831
        a_only = replication_decision(passing_second_a, None, a_only_initial)
        self.assertEqual(a_only["decision"], "FOUR_MOTION_PASS_SELECT_A")

        first_a = _summary("A", exceed=0.1, maximum=0.02)
        first_b = _summary("B", exceed=0.04, maximum=0.009)
        initial = {"decision": initial_decision(first_a, first_b)}
        second_a = _summary("A", exceed=0.1, maximum=0.02)
        second_b = _summary("B", exceed=0.04, maximum=0.009, progression=True)
        second_a["optimizer_seed"] = 20260831
        second_b["optimizer_seed"] = 20260831
        result = replication_decision(second_a, second_b, initial)
        self.assertEqual(result["decision"], "FOUR_MOTION_PASS_SELECT_B")
        self.assertEqual(result["next_stage"], "NEW_32_MOTION_FIXED_RUN")

        second_b["evaluations"][1]["tail_global"]["threshold_exceed_fraction"] = 0.2
        second_b["evaluations"][2]["tail_global"]["threshold_exceed_fraction"] = 0.2
        second_b["evaluations"][3]["tail_global"]["threshold_exceed_fraction"] = 0.2
        second_b["evaluations"][1]["exact"]["progression_gate"]["passed"] = False
        second_b["evaluations"][2]["exact"]["progression_gate"]["passed"] = False
        second_b["evaluations"][3]["exact"]["progression_gate"]["passed"] = False
        stopped = replication_decision(second_a, second_b, initial)
        self.assertEqual(stopped["decision"], "STOP_AT_FOUR_MOTIONS_UNSTABLE_INTERVENTION")

    def test_execution_and_quality_markers_are_distinct(self) -> None:
        self.assertEqual(len({SMOKE_MARKER, EXECUTION_MARKER, PROGRESSION_MARKER, COMPARISON_MARKER}), 4)

    def test_engineering_failure_manifest_is_not_an_execution_marker(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repository) as temporary:
            run = Path(temporary)
            _write_failure_manifest(run, RuntimeError("boom"))
            failure = json.loads(
                (run / "manifests/posterior_ab_failure.json").read_text(encoding="utf-8")
            )
            self.assertFalse(failure["execution_pass"])
            self.assertEqual(failure["error"], "boom")
            self.assertFalse((run / f"markers/{EXECUTION_MARKER}").exists())

    def test_comparison_writes_execution_marker_and_structured_decision(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repository) as temporary:
            root = Path(temporary)
            runs = {}
            for arm, summary in (
                ("A", _summary("A", exceed=0.1, maximum=0.02)),
                ("B", _summary("B", exceed=0.04, maximum=0.009)),
            ):
                run = root / arm
                (run / "manifests").mkdir(parents=True)
                summary["format_version"] = FORMAT_VERSION
                (run / "manifests/posterior_ab_summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )
                runs[arm] = run
            output = root / "comparison"
            result = run_ab_comparison(
                output_run=output, run_a=runs["A"], run_b=runs["B"]
            )
            self.assertEqual(result["decision"]["decision"], "REPLICATE_A_AND_B")
            self.assertTrue((output / f"markers/{COMPARISON_MARKER}").is_file())
            manifest = json.loads(
                (output / "manifests/posterior_ab_comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(manifest["execution_pass"])
            self.assertEqual(manifest["comparison_phase"], "initial")

    def test_ab_plots_label_log_axis_and_quality_thresholds(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        records = [
            {
                "phase": "train",
                "optimizer_step": 1,
                "raw_reconstruction": {"total": 0.0},
                "optimization_objective": {"total": 0.1},
            },
            {
                "phase": "evaluation",
                "optimizer_step": 0,
                "exact": {
                    "reconstruction_loss": {"total": 0.2},
                    "worst_state_rmse": 0.1,
                    "worst_action_rmse": 0.1,
                    "continuous_max_abs": 0.2,
                },
                "full_objective": {"optimization_objective": {"total": 0.3}},
                "tail_global": {"threshold_exceed_fraction": 0.4},
            },
        ]
        with tempfile.TemporaryDirectory(dir=repository) as temporary:
            paths = render_training_plots(Path(temporary), records)
            gate = Path(paths["gates"]).read_text(encoding="utf-8")
            training = Path(paths["training"]).read_text(encoding="utf-8")
            self.assertIn("Value (log10 scale)", gate)
            self.assertIn("progression RMSE/max = 1e-2", gate)
            self.assertIn("exact RMSE = 1e-4", gate)
            self.assertIn("clip non-positive values to 1e-12", training)


if __name__ == "__main__":
    unittest.main()
