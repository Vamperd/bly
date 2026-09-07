from __future__ import annotations

import gc
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from cvae_sa.models import (
    PosteriorCapacityDecodedOutput,
    PosteriorCapacityTransformerCVAE,
    build_model,
)
from cvae_sa.posterior_capacity import FIXED_MASK_NAMES, reconstruction_loss
from cvae_sa.posterior_capacity_autodecoder import (
    _assert_encoder_isolated,
    _encoder_call_counters,
)
from cvae_sa.posterior_capacity_latent_topology import (
    ARM_CODE_PARAMETERS,
    ARM_SHAPES,
    ARM_TOTAL_PARAMETERS,
    COMPARISON_MARKER,
    COMPARISON_STEPS,
    EXECUTION_MARKER,
    FORMAT_VERSION,
    INITIALIZATION_SEED,
    QUALITY_MARKER,
    SMOKE_MARKER,
    TRAINING_SEED,
    WindowLatentTopologyAutoDecoder,
    _comparison_identity,
    _optimizer,
    _parameter_count,
    _render_plots,
    comparison_decision,
    configure_trainable_parameters,
    initialize_topology,
    run_comparison,
    validate_config_contract,
    validate_saved_checkpoint,
)


def _small_config(width: int = 32) -> dict[str, object]:
    return {
        "kind": "physics_posterior_transformer",
        "d_model": width,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "heads": 4,
        "ffn_dim": width * 2,
        "latent_dim": 256,
        "dropout": 0.0,
        "state_dim": 70,
        "decoder_layer_latent_gates": False,
    }


def _batch(batch_size: int = 2, transitions: int = 3) -> dict[str, object]:
    state = torch.randn(batch_size, transitions + 1, 70)
    state[..., 68:70] = torch.randint(0, 2, state[..., 68:70].shape).float()
    return {
        "physical_state": state,
        "action": torch.randn(batch_size, transitions, 29),
        "valid_state": torch.ones(batch_size, transitions + 1, dtype=torch.bool),
        "valid_action": torch.ones(batch_size, transitions, dtype=torch.bool),
        "window_index": torch.arange(batch_size),
        "source_window_index": torch.arange(batch_size),
    }


class _CaptureDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.latent_projection = nn.Linear(256, 384)
        self.last_prefix: torch.Tensor | None = None
        self.last_time: torch.Tensor | None = None

    def decode_from_latent_topology(
        self,
        batch,
        state_mask,
        action_mask,
        *,
        prefix_tokens=None,
        time_conditions=None,
    ):
        del state_mask, action_mask
        self.last_prefix = prefix_tokens
        self.last_time = time_conditions
        source = prefix_tokens if prefix_tokens is not None else time_conditions
        signal = source.mean(dim=(1, 2))
        state = signal[:, None, None].expand_as(batch["physical_state"]).clone()
        action = signal[:, None, None].expand_as(batch["action"]).clone()
        logits = torch.zeros_like(state[..., 68:70])
        state[..., 68:70] = torch.sigmoid(logits)
        return PosteriorCapacityDecodedOutput(state, action, logits)


def _wrapped(arm: str) -> WindowLatentTopologyAutoDecoder:
    codes, tensors, _ = initialize_topology(arm)
    return WindowLatentTopologyAutoDecoder(_CaptureDecoder(), arm, codes, tensors)


def _comparison_evaluation(step: int, passed: bool = True) -> dict[str, object]:
    value = 0.5 if passed else 2.0
    return {
        "optimizer_step": step,
        "quality_gate": {
            "passed": passed,
            "score": value,
            "global_state_rmse": 0.008,
            "global_action_rmse": 0.007,
        },
        "exact": {
            "worst_state_rmse": 0.005,
            "worst_action_rmse": 0.006,
            "continuous_max_abs": 0.007,
            "contact_accuracy": 1.0,
        },
        "tail_global": {"threshold_exceed_fraction": 0.02},
        "topology_progression_gate": {
            "zero_ratio": 20.0,
            "cross_window_ratio": 19.0,
            "cross_motion_ratio": 18.0,
        },
    }


def _comparison_summary(arm: str, passed: bool = True) -> dict[str, object]:
    return {
        "format_version": FORMAT_VERSION,
        "arm": arm,
        "smoke": False,
        "execution_pass": True,
        "quality_pass": passed,
        "dataset_manifest_sha256": "dataset",
        "source": {
            "checkpoint_sha256": "source",
            "f4e_baseline": {
                "summary_sha256": "f4e",
                "state_global_rmse": 0.008951,
                "action_global_rmse": 0.007856,
            },
        },
        "data_contract": {
            "fixture_bitmap_sha256": "fixture",
            "selected_windows_sha256": "windows",
            "selected_motion_keys": ["m0", "m1", "m2", "m3"],
        },
        "model_contract": {
            "code_shape": list(ARM_SHAPES[arm]),
            "code_parameter_count": ARM_CODE_PARAMETERS[arm],
            "code_scalars_per_window": ARM_CODE_PARAMETERS[arm] // 80,
            "total_parameter_count": ARM_TOTAL_PARAMETERS[arm],
        },
        "initialization": {
            "initialization_seed": INITIALIZATION_SEED,
            "distribution": "independent normal mean=0 std=0.02; projection bias=zeros",
        },
        "training_contract": {
            "training_seed": TRAINING_SEED,
            "fixture_seed": 20260830,
            "objective": "State MSE + Action MSE + contact BCE; equal mean of present components",
            "mask_phase": "fixed",
            "optimizer": "AdamW",
            "optimizer_betas": [0.9, 0.999],
            "optimizer_eps": 1e-8,
            "weight_decay": 0.0,
            "topology_learning_rate": 3e-4,
            "topology_minimum_learning_rate": 1e-5,
            "decoder_learning_rate": 3e-5,
            "decoder_minimum_learning_rate": 1e-6,
            "warmup_steps": 250,
            "gradient_clip": 1.0,
            "micro_batch": 4,
            "gradient_accumulation": 16,
            "effective_batch": 64,
            "precision": "FP32",
            "maximum_optimizer_steps": 15000,
            "validation_interval": 1000,
            "no_early_stop": True,
            "completed_optimizer_steps": 15000,
            "comparison_steps": list(COMPARISON_STEPS),
            "training_identity_sha256_by_step": {str(step): "same" for step in range(1, 15001)},
        },
        "evaluations": [
            _comparison_evaluation(step, passed)
            for step in (0, *range(1000, 15001, 1000))
        ],
    }


class PosteriorCapacityLatentTopologyTest(unittest.TestCase):
    def test_fixed_config_and_exact_parameter_counts(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs/posterior_capacity_latent_topology.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        source = {"config": {"model": {
            key: config["model"][key]
            for key in (
                "kind", "d_model", "encoder_layers", "decoder_layers", "heads",
                "ffn_dim", "latent_dim", "dropout",
            )
        }}}
        for arm in ("G8", "T129"):
            self.assertTrue(all(validate_config_contract(config, source, arm).values()))
        config["model"]["state_dim"] = 70
        base = build_model(config["model"])
        for arm in ("G8", "T129"):
            codes, tensors, _ = initialize_topology(arm)
            wrapped = WindowLatentTopologyAutoDecoder(base, arm, codes, tensors)
            self.assertEqual(_parameter_count(wrapped), ARM_TOTAL_PARAMETERS[arm])
            trainable = configure_trainable_parameters(wrapped)
            optimizer, scheduler = _optimizer(wrapped, trainable, config, max_steps=2)
            self.assertEqual([group["name"] for group in optimizer.param_groups], ["topology", "decoder_side"])
            self.assertEqual(len(scheduler.lr_lambdas), 2)
            del optimizer, scheduler
            del wrapped
            gc.collect()
        self.assertLess(
            abs(ARM_TOTAL_PARAMETERS["G8"] / ARM_TOTAL_PARAMETERS["T129"] - 1.0),
            0.0002,
        )

    def test_initialization_is_deterministic_and_code_budgets_are_equal(self) -> None:
        for arm in ("G8", "T129"):
            first, first_tensors, first_manifest = initialize_topology(arm)
            second, second_tensors, second_manifest = initialize_topology(arm)
            self.assertTrue(torch.equal(first, second))
            self.assertEqual(first_manifest["initialization_sha256"], second_manifest["initialization_sha256"])
            self.assertEqual(set(first_tensors), set(second_tensors))
        self.assertEqual(ARM_CODE_PARAMETERS["G8"], 163_840)
        self.assertEqual(ARM_CODE_PARAMETERS["T129"], 165_120)
        self.assertLess(abs(2048 / 2064 - 1.0), 0.01)

    def test_decoder_topology_api_shapes_alignment_and_truth_isolation(self) -> None:
        torch.manual_seed(8)
        model = PosteriorCapacityTransformerCVAE(_small_config()).eval()
        batch = _batch()
        state_mask = torch.ones_like(batch["physical_state"], dtype=torch.bool)
        action_mask = torch.ones_like(batch["action"], dtype=torch.bool)
        prefix = torch.randn(2, 8, 32)
        time = torch.randn(2, 4, 32)
        prefix_result = model.decode_from_latent_topology(
            batch, state_mask, action_mask, prefix_tokens=prefix
        )
        time_result = model.decode_from_latent_topology(
            batch, state_mask, action_mask, time_conditions=time
        )
        self.assertEqual(tuple(prefix_result.physical_state.shape), (2, 4, 70))
        self.assertEqual(tuple(time_result.action.shape), (2, 3, 29))
        altered = dict(batch)
        altered["physical_state"] = batch["physical_state"] + 1000
        altered["action"] = batch["action"] - 1000
        hidden = model.decode_from_latent_topology(
            altered, state_mask, action_mask, time_conditions=time
        )
        self.assertTrue(torch.equal(time_result.physical_state, hidden.physical_state))
        self.assertTrue(torch.equal(time_result.action, hidden.action))
        with self.assertRaises(ValueError):
            model.decode_from_latent_topology(batch, state_mask, action_mask)
        with self.assertRaises(ValueError):
            model.decode_from_latent_topology(
                batch, state_mask, action_mask, prefix_tokens=prefix, time_conditions=time
            )

    def test_g8_slots_and_t129_time_codes_are_distinct_and_shared_across_masks(self) -> None:
        batch = _batch(transitions=128)
        batch["window_index"] = torch.tensor([5, 5])
        state_mask = torch.zeros_like(batch["physical_state"], dtype=torch.bool)
        action_mask = torch.zeros_like(batch["action"], dtype=torch.bool)
        state_mask[0, 0] = True
        action_mask[1, 0] = True
        g8 = _wrapped("G8")
        g8_output = g8(batch, state_mask, action_mask)
        self.assertTrue(torch.equal(g8_output.posterior_mean[0], g8_output.posterior_mean[1]))
        self.assertEqual(tuple(g8.base_model.last_prefix.shape), (2, 8, 384))
        self.assertFalse(torch.equal(g8.base_model.last_prefix[:, 0], g8.base_model.last_prefix[:, 1]))
        t129 = _wrapped("T129")
        with torch.no_grad():
            t129.time_projection.weight.zero_()
            t129.time_projection.bias.zero_()
            t129.time_projection.weight[0, 0] = 1.0
            t129.window_codes[5, :, 0] = torch.arange(129)
        t_output = t129(batch, state_mask, action_mask)
        self.assertTrue(torch.equal(t_output.posterior_mean[0], t_output.posterior_mean[1]))
        self.assertEqual(tuple(t129.base_model.last_time.shape), (2, 129, 384))
        self.assertTrue(torch.equal(t129.base_model.last_time[0, :, 0], torch.arange(129).float()))

    def test_wrong_partial_donor_shape_is_rejected(self) -> None:
        model = _wrapped("G8")
        batch = _batch()
        state_mask = torch.ones_like(batch["physical_state"], dtype=torch.bool)
        action_mask = torch.ones_like(batch["action"], dtype=torch.bool)
        with self.assertRaises(ValueError):
            model(
                batch,
                state_mask,
                action_mask,
                latent_override=torch.zeros(2, 1, 256),
            )

    def test_trainable_allowlist_freezes_all_encoder_and_old_latent_parameters(self) -> None:
        base = PosteriorCapacityTransformerCVAE(_small_config(width=384))
        codes, tensors, _ = initialize_topology("T129")
        model = WindowLatentTopologyAutoDecoder(base, "T129", codes, tensors)
        contract = configure_trainable_parameters(model)
        trainable = contract["topology_names"] + contract["decoder_names"]
        self.assertTrue(any(name.startswith("time_projection.") for name in trainable))
        self.assertTrue(any(name.startswith("base_model.decoder.") for name in trainable))
        self.assertFalse(any(name.startswith("base_model.encoder") for name in trainable))
        self.assertFalse(any(name.startswith("base_model.posterior") for name in trainable))
        self.assertFalse(any(name.startswith("base_model.prior") for name in trainable))
        self.assertFalse(dict(model.named_parameters())["base_model.decoder_latent_token"].requires_grad)
        self.assertFalse(dict(model.named_parameters())["base_model.latent_projection.weight"].requires_grad)

        g8_base = PosteriorCapacityTransformerCVAE(_small_config(width=384))
        g8_codes, g8_tensors, _ = initialize_topology("G8")
        g8 = WindowLatentTopologyAutoDecoder(g8_base, "G8", g8_codes, g8_tensors)
        configure_trainable_parameters(g8)
        counts, handles = _encoder_call_counters(g8_base)
        batch = _batch()
        state_mask = torch.ones_like(batch["physical_state"], dtype=torch.bool)
        action_mask = torch.ones_like(batch["action"], dtype=torch.bool)
        reconstruction_loss(
            g8(batch, state_mask, action_mask), batch, state_mask, action_mask
        ).total.backward()
        isolation = _assert_encoder_isolated(g8, counts)
        for handle in handles:
            handle.remove()
        self.assertTrue(isolation["passed"])

    def test_two_window_two_mask_cpu_fit_decreases_loss_for_both_topologies(self) -> None:
        batch = {
            "physical_state": torch.zeros(4, 129, 70),
            "action": torch.zeros(4, 128, 29),
            "window_index": torch.tensor([0, 0, 1, 1]),
            "source_window_index": torch.tensor([0, 0, 1, 1]),
        }
        target = torch.tensor([1.0, 1.0, 2.0, 2.0])
        batch["physical_state"][..., :68] = target[:, None, None]
        batch["action"][:] = target[:, None, None]
        state_mask = torch.zeros_like(batch["physical_state"], dtype=torch.bool)
        state_mask[..., :68] = True
        action_mask = torch.ones_like(batch["action"], dtype=torch.bool)
        for arm in ("G8", "T129"):
            model = _wrapped(arm)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            initial = float(
                reconstruction_loss(
                    model(batch, state_mask, action_mask), batch, state_mask, action_mask
                ).total
            )
            for _ in range(80):
                optimizer.zero_grad()
                loss = reconstruction_loss(
                    model(batch, state_mask, action_mask), batch, state_mask, action_mask
                ).total
                loss.backward()
                optimizer.step()
            final = float(
                reconstruction_loss(
                    model(batch, state_mask, action_mask), batch, state_mask, action_mask
                ).total
            )
            self.assertLess(final, initial * 0.02, arm)

    def test_checkpoint_contract_covers_arm_hashes_and_parameter_counts(self) -> None:
        payload = {
            "format_version": "sonic_posterior_latent_topology_checkpoint_v1",
            "arm": "G8",
            "optimizer_step": 2,
            "model": {
                "window_codes": torch.zeros(ARM_SHAPES["G8"]),
                "slot_embedding": torch.zeros(8, 384),
                "base_model.latent_projection.weight": torch.zeros(384, 256),
                "base_model.latent_projection.bias": torch.zeros(384),
            },
            "optimizer": {"state": {}},
            "scheduler": {"last_epoch": 2},
            "dataset_manifest_sha256": "dataset",
            "source_checkpoint_sha256": "source",
            "f4e_summary_sha256": "f4e",
            "fixture_bitmap_sha256": "fixture",
            "initialization_sha256": "init",
            "code_parameter_count": ARM_CODE_PARAMETERS["G8"],
            "total_parameter_count": ARM_TOTAL_PARAMETERS["G8"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save(payload, path)
            result = validate_saved_checkpoint(
                path,
                arm="G8",
                expected_step=2,
                dataset_hash="dataset",
                source_hash="source",
                f4e_hash="f4e",
                fixture_hash="fixture",
                initialization_hash="init",
            )
        self.assertTrue(result["passed"])

    def test_comparison_decision_has_all_four_fixed_branches(self) -> None:
        self.assertEqual(comparison_decision(True, False)[0], "G8_PASS_SINGLE_GLOBAL_TOKEN_BROADCAST_BOTTLENECK")
        self.assertEqual(comparison_decision(False, True)[0], "T129_PASS_TEMPORAL_PLACEMENT_REQUIRED")
        self.assertEqual(comparison_decision(True, True)[0], "BOTH_PASS_PREFER_G8_GLOBAL_TOPOLOGY")
        self.assertEqual(comparison_decision(False, False)[0], "BOTH_FAIL_LATENT_TOPOLOGY_INSUFFICIENT")

    def test_comparison_validates_pairing_and_writes_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {}
            for arm in ("G8", "T129"):
                run = root / arm
                (run / "manifests").mkdir(parents=True)
                (run / "markers").mkdir()
                (run / "checkpoints").mkdir()
                (run / "manifests/posterior_latent_topology_summary.json").write_text(
                    json.dumps(_comparison_summary(arm)), encoding="utf-8"
                )
                (run / f"markers/{EXECUTION_MARKER}").write_text("PASS\n", encoding="utf-8")
                (run / f"markers/{QUALITY_MARKER}").write_text("PASS\n", encoding="utf-8")
                (run / "checkpoints/best_progression.pt").write_bytes(b"best")
                (run / "checkpoints/last.pt").write_bytes(b"last")
                runs[arm] = run
            output = root / "comparison"
            result = run_comparison(
                run_g8=runs["G8"], run_t129=runs["T129"], output_run=output
            )
            self.assertEqual(result["decision"], "BOTH_PASS_PREFER_G8_GLOBAL_TOPOLOGY")
            self.assertTrue((output / f"markers/{COMPARISON_MARKER}").is_file())
            self.assertLess(result["budget_comparison"]["relative_code_scalar_difference"], 0.01)
            broken = _comparison_summary("T129")
            broken["training_contract"]["training_identity_sha256_by_step"][1] = "different"
            self.assertFalse(_comparison_identity(_comparison_summary("G8"), broken)["training_identity"])

    def test_marker_names_are_independent(self) -> None:
        self.assertEqual(
            {SMOKE_MARKER, EXECUTION_MARKER, QUALITY_MARKER, COMPARISON_MARKER},
            {
                "cvae_posterior_latent_topology_smoke.ok",
                "cvae_posterior_latent_topology_execution.ok",
                "cvae_posterior_latent_topology_progression.ok",
                "cvae_posterior_latent_topology_comparison.ok",
            },
        )

    def test_plots_label_log_axes_masks_optimizer_and_code_dependencies(self) -> None:
        train = {
            "phase": "train",
            "optimizer_step": 1,
            "raw_reconstruction": {"total": 0.2},
            "learning_rates": {"topology": 3e-4, "decoder_side": 3e-5},
            "gradient_norm_before_clip": 0.5,
        }
        evaluation = {
            "phase": "evaluation",
            "optimizer_step": 1,
            "exact": {
                "reconstruction_loss": {
                    "total": 0.1,
                    "state": 0.1,
                    "action": 0.1,
                    "contact": 0.1,
                },
                "worst_state_rmse": 0.1,
                "worst_action_rmse": 0.1,
                "continuous_max_abs": 0.2,
                "cases": {
                    name: {
                        "worst_state_rmse": 0.1,
                        "worst_action_rmse": 0.1,
                        "continuous_max_abs": 0.2,
                    }
                    for name in FIXED_MASK_NAMES
                },
            },
            "tail_global": {"threshold_exceed_fraction": 0.3},
            "quality_gate": {
                "global_state_rmse": 0.08,
                "global_action_rmse": 0.07,
            },
            "topology_progression_gate": {
                "zero_ratio": 20.0,
                "cross_window_ratio": 19.0,
                "cross_motion_ratio": 18.0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plots").mkdir()
            paths = _render_plots(root, [train, evaluation], evaluation, "G8")
            combined = "\n".join(
                Path(path).read_text(encoding="utf-8") for path in paths.values()
            )
        self.assertIn("Value (log10 scale)", combined)
        self.assertIn("Topology learning rate", combined)
        self.assertIn("Cross-motion ratio", combined)
        self.assertIn("full_both", combined)


if __name__ == "__main__":
    unittest.main()
