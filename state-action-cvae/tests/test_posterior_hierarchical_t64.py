from __future__ import annotations

import gc
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from cvae_sa.models import HierarchicalPosteriorTransformer, build_model, parameter_count
from cvae_sa.posterior_direct_output import (
    DirectWindowOutput,
    assert_output_isolated,
    direct_output_next_step,
)
from cvae_sa.posterior_hierarchical_t64 import (
    configure_optimizer,
    hierarchical_next_step,
    validate_h50_authorization,
)
from cvae_sa.posterior_t64_protocol import (
    PHYSICAL_MASK_NAMES,
    evaluate,
    evaluate_latent_dependence,
    make_autoencode_masks,
    make_physical_masks,
    reconstruction_loss,
    render_plots,
)
from cvae_sa.util import load_config


def small_config() -> dict[str, object]:
    return {
        "kind": "physics_hierarchical_posterior_transformer",
        "profile": "test",
        "d_model": 32,
        "posterior_encoder_layers": 1,
        "condition_encoder_layers": 1,
        "decoder_layers": 2,
        "heads": 4,
        "ffn_dim": 64,
        "global_latent_dim": 16,
        "local_latent_dim": 8,
        "local_chunks": 16,
        "chunk_transitions": 4,
        "max_state_steps": 65,
        "dropout": 0.0,
        "state_dim": 70,
    }


def batch(batch_size: int = 2) -> dict[str, object]:
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


class ItemDataset(Dataset[dict[str, object]]):
    def __init__(self, items: list[dict[str, object]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.items[index]


class HierarchicalPosteriorT64Test(unittest.TestCase):
    def test_h38_h50_reference_parameter_counts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        reference = json.loads((root / "configs/posterior_hierarchical_t64_reference.json").read_text())
        for profile, config_name in (("H38", "posterior_hierarchical_t64_h38.json"), ("H50", "posterior_hierarchical_t64_h50.json")):
            config = load_config(root / "configs" / config_name)
            config["model"]["state_dim"] = 70
            with torch.device("meta"):
                model = build_model(config["model"])
            self.assertEqual(parameter_count(model), reference[profile]["parameter_count"])
            low, high = reference[profile]["parameter_count_range"]
            self.assertLessEqual(low, parameter_count(model))
            self.assertLessEqual(parameter_count(model), high)
            del model
            gc.collect()

    def test_shapes_chunk_boundaries_and_canonical_mask_invariance(self) -> None:
        torch.manual_seed(10)
        model = HierarchicalPosteriorTransformer(small_config()).eval()
        value = batch()
        state_a, action_a, _ = make_physical_masks(value, 123)
        state_b, action_b, _ = make_autoencode_masks(value)
        first_global, first_local = model.encode_posterior(value, state_a, action_a)
        second_global, second_local = model.encode_posterior(value, state_b, action_b)
        self.assertTrue(torch.equal(first_global, second_global))
        self.assertTrue(torch.equal(first_local, second_local))
        self.assertEqual(tuple(first_global.shape), (2, 16))
        self.assertEqual(tuple(first_local.shape), (2, 16, 8))
        output = model(value, state_a, action_a)
        self.assertEqual(tuple(output.physical_state.shape), (2, 65, 70))
        self.assertEqual(tuple(output.action.shape), (2, 64, 29))
        ids = model.local_chunk_ids(torch.arange(65))
        self.assertTrue(torch.equal(ids[:4], torch.zeros(4, dtype=torch.long)))
        self.assertTrue(torch.equal(ids[4:8], torch.ones(4, dtype=torch.long)))
        self.assertTrue(torch.equal(ids[60:65], torch.full((5,), 15, dtype=torch.long)))

    def test_decoder_truth_isolation_cross_attention_and_film_gradients(self) -> None:
        torch.manual_seed(11)
        model = HierarchicalPosteriorTransformer(small_config()).eval()
        value = batch()
        state_mask, action_mask, _ = make_autoencode_masks(value)
        global_latent, local_latents = model.encode_posterior(value, state_mask, action_mask)
        first = model.decode_from_hierarchical_latent(
            value, state_mask, action_mask, global_latent, local_latents
        )
        changed = dict(value)
        changed["physical_state"] = value["physical_state"] + 1000
        changed["action"] = value["action"] - 1000
        second = model.decode_from_hierarchical_latent(
            changed, state_mask, action_mask, global_latent, local_latents
        )
        self.assertTrue(torch.equal(first.physical_state, second.physical_state))
        self.assertTrue(torch.equal(first.action, second.action))
        model.train()
        output = model(value, state_mask, action_mask)
        loss = reconstruction_loss(output, value, state_mask, action_mask)["total"]
        loss.backward()
        self.assertGreater(float(model.film_projection.weight.grad.abs().sum()), 0.0)
        for layer in model.decoder.layers:
            self.assertIsNotNone(layer.cross_attention.query.weight.grad)
            self.assertGreater(float(layer.cross_attention.query.weight.grad.abs().sum()), 0.0)

    def test_physical_mask_bank_semantics_reproducibility_and_heldout_separation(self) -> None:
        value = batch(len(PHYSICAL_MASK_NAMES))
        value["mask_slot"] = torch.arange(len(PHYSICAL_MASK_NAMES))
        state, action, names = make_physical_masks(value, 456)
        again_state, again_action, again_names = make_physical_masks(value, 456)
        held_state, held_action, held_names = make_physical_masks(value, 789, held_out=True)
        self.assertEqual(names, list(PHYSICAL_MASK_NAMES))
        self.assertEqual(names, again_names)
        self.assertEqual(names, held_names)
        self.assertTrue(torch.equal(state, again_state))
        self.assertTrue(torch.equal(action, again_action))
        self.assertFalse(torch.equal(state, held_state) and torch.equal(action, held_action))
        self.assertTrue(bool(state[2, 0].logical_not().all()))
        self.assertTrue(bool(state[2, 1:].all()))
        self.assertTrue(bool(action[5].all()))
        for slot in (6, 7):
            action_times = action[slot].any(dim=-1)
            indices = torch.nonzero(action_times).flatten()
            self.assertFalse(bool(state[slot, indices[0]].any()))
            self.assertFalse(bool(state[slot, indices[-1] + 1].any()))

    def test_direct_output_is_shared_across_masks_and_can_pass_evaluator(self) -> None:
        torch.manual_seed(12)
        source = batch(2)
        model = DirectWindowOutput(2)
        with torch.no_grad():
            model.state_continuous.copy_(source["physical_state"][..., :68])
            model.action.copy_(source["action"])
            contacts = source["physical_state"][..., 68:70]
            model.state_contact_logits.copy_(torch.where(contacts > 0.5, 20.0, -20.0))
        duplicated: list[dict[str, object]] = []
        for window in range(2):
            for slot in range(len(PHYSICAL_MASK_NAMES)):
                duplicated.append({
                    "physical_state": source["physical_state"][window],
                    "action": source["action"][window],
                    "valid_state": source["valid_state"][window],
                    "valid_action": source["valid_action"][window],
                    "motion_key": source["motion_key"][window],
                    "variant_id": source["variant_id"][window],
                    "window_start": source["window_start"][window],
                    "window_index": torch.tensor(window),
                    "mask_slot": torch.tensor(slot),
                })
        loader = DataLoader(ItemDataset(duplicated), batch_size=4, shuffle=False)
        base_loader = DataLoader(ItemDataset(duplicated[:1]), batch_size=1)
        thresholds_fit = {
            "global_state_rmse": 1e-2, "global_action_rmse": 1e-2,
            "worst_mask_state_rmse": 2e-2, "worst_mask_action_rmse": 2e-2,
            "continuous_p99_abs": 5e-2, "contact_accuracy": 1.0,
        }
        thresholds_strict = {
            "worst_state_rmse": 1e-2, "worst_action_rmse": 1e-2,
            "continuous_max_abs": 1e-2, "contact_accuracy": 1.0,
        }
        metrics = evaluate(
            model, loader, base_loader, torch.device("cpu"),
            lambda value: make_physical_masks(value, 456),
            fit_thresholds=thresholds_fit, strict_thresholds=thresholds_strict,
            exact_thresholds={**thresholds_strict, "worst_state_rmse": 1e-4, "worst_action_rmse": 1e-4, "continuous_max_abs": 1e-3},
            state_std=torch.ones(70), action_std=torch.ones(29), latent_diagnostics=False,
        )
        self.assertTrue(metrics["fit_gate"]["passed"])
        value = batch(2)
        value["window_index"] = torch.tensor([1, 1])
        state_mask, action_mask, _ = make_physical_masks(value, 456)
        output = model(value, state_mask, action_mask)
        self.assertTrue(torch.equal(output.physical_state[0], output.physical_state[1]))
        self.assertTrue(torch.equal(output.action[0], output.action[1]))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 2 * (65 * 70 + 64 * 29))

    def test_optimizer_partition_and_plot_contract(self) -> None:
        model = HierarchicalPosteriorTransformer(small_config())
        training = {
            "slow_learning_rate": 1e-4, "fast_learning_rate": 3e-4,
            "minimum_learning_rate": 1e-6, "warmup_steps": 1,
        }
        optimizer, scheduler, contract = configure_optimizer(model, training, 10)
        self.assertTrue(contract["all_parameters_covered_once"])
        self.assertEqual([group["name"] for group in optimizer.param_groups], [
            "encoders_self_attention_ffn", "latent_cross_film_query_output"
        ])
        self.assertEqual(len(scheduler.lr_lambdas), 2)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            paths = render_plots(run, [], None)
            self.assertEqual(len(paths), 5)
            for name, path in paths.items():
                text = Path(path).read_text(encoding="utf-8")
                self.assertIn("Value (log10 scale)", text)
                if name in {"training_curves", "gate_curves"}:
                    self.assertIn("Optimizer step", text)
            self.assertIn("Continuous feature index", Path(paths["feature_error"]).read_text())
            self.assertIn("Mask family index", Path(paths["mask_breakdown"]).read_text())

    def test_padding_empty_chunks_and_whole_or_partial_donor_diagnostics(self) -> None:
        torch.manual_seed(13)
        model = HierarchicalPosteriorTransformer(small_config()).eval()
        value = batch(2)
        value["valid_state"][1, 10:] = False
        value["valid_action"][1, 9:] = False
        state_mask, action_mask, _ = make_autoencode_masks(value)
        global_latent, local_latents = model.encode_posterior(value, state_mask, action_mask)
        self.assertTrue(torch.isfinite(global_latent).all())
        self.assertTrue(torch.isfinite(local_latents).all())
        items = []
        for index in range(2):
            items.append({
                key: (entry[index] if isinstance(entry, torch.Tensor) else entry[index])
                for key, entry in value.items()
            })
        diagnostics = evaluate_latent_dependence(
            model, DataLoader(ItemDataset(items), batch_size=2), torch.device("cpu")
        )
        self.assertEqual(set(diagnostics["main_ratios"]), {"zero", "cross_window", "cross_motion"})
        self.assertIn("cross_window_global", diagnostics["ratios_to_correct"])
        self.assertIn("cross_window_local", diagnostics["ratios_to_correct"])
        self.assertIn("cross_motion_global", diagnostics["ratios_to_correct"])
        self.assertIn("cross_motion_local", diagnostics["ratios_to_correct"])

    def test_fixed_decision_table(self) -> None:
        self.assertEqual(direct_output_next_step(False), "INVESTIGATE_LOSS_MASK_EVALUATOR")
        self.assertEqual(direct_output_next_step(True), "RUN_H38_ENGINEERING_SMOKE")
        self.assertEqual(
            hierarchical_next_step("H38", "autoencode", False),
            "RUN_SINGLE_H50_AUTOENCODE_REPLICATION",
        )
        self.assertEqual(hierarchical_next_step("H50", "autoencode", False), "STOP_MODEL_SCALING")
        self.assertEqual(
            hierarchical_next_step("H38", "fixed", False),
            "STOP_AND_DIAGNOSE_CONDITION_FUSION",
        )
        self.assertEqual(
            hierarchical_next_step("H38", "random", False),
            "STOP_AND_DIAGNOSE_RANDOM_MASK_COVERAGE",
        )
        self.assertEqual(
            hierarchical_next_step("H38", "random", True),
            "FREEZE_KL0_BASELINE_THEN_IMPLEMENT_KL_THREE_PATHS",
        )

    def test_output_run_cannot_overlap_protected_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protected = root / "source"
            protected.mkdir()
            assert_output_isolated(root / "new-run", [protected])
            with self.assertRaisesRegex(ValueError, "protected source"):
                assert_output_isolated(protected / "child", [protected])

    def test_h50_requires_a_formal_failed_h38_autoencode_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "formal failed H38-A"):
            validate_h50_authorization(Path("dataset"), None)


if __name__ == "__main__":
    unittest.main()
