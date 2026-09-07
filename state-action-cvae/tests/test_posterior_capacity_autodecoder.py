from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cvae_sa.models import (
    PosteriorCapacityDecodedOutput,
    PosteriorCapacityTransformerCVAE,
    build_model,
    parameter_count,
)
from cvae_sa.posterior_capacity import FIXED_MASK_NAMES, reconstruction_loss
from cvae_sa.posterior_capacity_ab import run_ab_comparison
from cvae_sa.posterior_capacity_autodecoder import (
    CHECKPOINT_FORMAT,
    E1_MARKER,
    E2_MARKER,
    EXECUTION_MARKER,
    EXPECTED_CODE_PARAMETERS,
    EXPECTED_TOTAL_PARAMETERS,
    EXPECTED_TRIGGER_RUN_NAME,
    SMOKE_MARKER,
    WindowCodeAutoDecoder,
    _all_parameter_count,
    _autodecoder_gate,
    _categorical_svg,
    configure_trainable_parameters,
    initialize_window_codes,
    render_plots,
    validate_saved_checkpoint,
    validate_stop_trigger,
)


def _config() -> dict[str, object]:
    return {
        "kind": "physics_posterior_transformer",
        "d_model": 32,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "heads": 4,
        "ffn_dim": 64,
        "latent_dim": 256,
        "dropout": 0.0,
        "state_dim": 70,
        "decoder_layer_latent_gates": False,
    }


def _batch(batch_size: int = 2, transitions: int = 2) -> dict[str, torch.Tensor]:
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


class _CentroidDataset(Dataset[dict[str, object]]):
    def __len__(self) -> int:
        return 800

    def __getitem__(self, index: int) -> dict[str, object]:
        window, slot = divmod(index, 10)
        state = torch.zeros(2, 70)
        return {
            "physical_state": state,
            "action": torch.zeros(1, 29),
            "valid_state": torch.ones(2, dtype=torch.bool),
            "valid_action": torch.ones(1, dtype=torch.bool),
            "motion_key": f"motion-{window // 20}",
            "variant_id": window % 8,
            "window_start": window,
            "window_index": window,
            "source_window_index": window,
            "mask_slot": slot,
        }


class _CentroidEncoder(nn.Module):
    def forward(self, batch, state_mask, action_mask):
        del state_mask, action_mask
        value = (batch["window_index"] + batch["mask_slot"]).float()[:, None]
        mean = value.expand(-1, 256)
        state = batch["physical_state"]
        action = batch["action"]
        return SimpleNamespace(
            posterior_mean=mean,
            posterior_logvar=torch.zeros_like(mean),
            prior_mean=torch.zeros_like(mean),
            prior_logvar=torch.zeros_like(mean),
            physical_state=state,
            action=action,
            state_contact_logits=torch.zeros_like(state[..., 68:70]),
        )


class _ScalarDecoder(nn.Module):
    def decode_from_global_latent(self, batch, state_mask, action_mask, latent):
        del state_mask, action_mask
        value = latent[:, :1]
        state = value[:, None].expand(-1, batch["physical_state"].shape[1], 70).clone()
        logits = torch.full_like(state[..., 68:70], -10.0)
        state[..., 68:70] = torch.sigmoid(logits)
        action = value[:, None].expand(-1, batch["action"].shape[1], 29)
        return PosteriorCapacityDecodedOutput(state, action, logits)


def _evaluation(step: int, exceed: float, maximum: float) -> dict[str, object]:
    return {
        "optimizer_step": step,
        "exact": {
            "progression_gate": {"passed": False, "score": maximum / 0.01},
            "worst_state_rmse": 0.009,
            "worst_action_rmse": 0.008,
            "continuous_max_abs": maximum,
            "contact_accuracy": 1.0,
            "latent_dependence": {"zero_ratio": 12.0},
        },
        "tail_global": {
            "global_state_rmse": 0.006,
            "global_action_rmse": 0.005,
            "threshold_exceed_fraction": exceed,
        },
    }


def _summary(arm: str, exceed: float, maximum: float) -> dict[str, object]:
    is_c = arm == "C"
    source: dict[str, object] = {
        "checkpoint_sha256": "source",
        "f4a_manifest_sha256": "f4a",
    }
    if is_c:
        source["f4c_trigger_comparison"] = {
            "decision": "IMPLEMENT_F4C",
            "checks": {"authorized": True},
        }
    return {
        "format_version": "sonic_posterior_ab_summary_v1",
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
            "identity_contract_sha256": "identity",
            "window_count": 80,
            "fixture_count": 800,
        },
        "model_contract": {
            "parameter_count": 25_456_483 if is_c else 25_453_411,
            "f4c_layer_gates_enabled": is_c,
            "f4c_layer_gate_parameter_count": 3072 if is_c else 0,
        },
        "training_contract": {
            "training_identity_sha256_by_step": {"1": "same"},
        },
        "evaluations": [
            _evaluation(0, 0.21, 9.64),
            *[_evaluation(step, exceed, maximum) for step in (8000, 9000, 10000)],
        ],
    }


class PosteriorCapacityAutoDecoderTest(unittest.TestCase):
    def test_fixed_config_has_exact_parameter_and_two_stage_contract(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs/posterior_capacity_autodecoder.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["model"]["state_dim"] = 70
        base = build_model(config["model"])
        self.assertEqual(parameter_count(base), 25_453_411)
        wrapped = WindowCodeAutoDecoder(base, torch.zeros(80, 256))
        self.assertEqual(_all_parameter_count(wrapped), EXPECTED_TOTAL_PARAMETERS)
        self.assertEqual(config["training"]["stage_e1"]["max_optimizer_steps"], 5000)
        self.assertEqual(config["training"]["stage_e2"]["max_optimizer_steps"], 15000)

    def test_decoder_only_api_matches_forward_override_and_hides_masked_truth(self) -> None:
        torch.manual_seed(7)
        model = PosteriorCapacityTransformerCVAE(_config()).eval()
        batch = _batch()
        state_mask = torch.ones_like(batch["physical_state"], dtype=torch.bool)
        action_mask = torch.ones_like(batch["action"], dtype=torch.bool)
        latent = torch.randn(2, 256)
        forward = model(batch, state_mask, action_mask, latent_override=latent)
        decoded = model.decode_from_global_latent(batch, state_mask, action_mask, latent)
        self.assertTrue(torch.equal(forward.physical_state, decoded.physical_state))
        self.assertTrue(torch.equal(forward.action, decoded.action))
        altered = dict(batch)
        altered["physical_state"] = batch["physical_state"] + 1000.0
        altered["action"] = batch["action"] - 1000.0
        hidden = model.decode_from_global_latent(altered, state_mask, action_mask, latent)
        self.assertTrue(torch.equal(decoded.physical_state, hidden.physical_state))
        self.assertTrue(torch.equal(decoded.action, hidden.action))

    def test_window_code_is_shared_across_masks_and_decoder_never_calls_encoder(self) -> None:
        base = PosteriorCapacityTransformerCVAE(_config()).eval()
        model = WindowCodeAutoDecoder(base, torch.randn(80, 256)).eval()
        batch = _batch()
        batch["window_index"] = torch.tensor([3, 3])
        state_mask = torch.zeros_like(batch["physical_state"], dtype=torch.bool)
        action_mask = torch.zeros_like(batch["action"], dtype=torch.bool)
        state_mask[0, 0] = True
        action_mask[1, 0] = True
        counts = {"encoder": 0, "posterior": 0, "prior": 0}
        handles = [
            base.encoder.register_forward_pre_hook(lambda *_: counts.__setitem__("encoder", counts["encoder"] + 1)),
            base.posterior.register_forward_pre_hook(lambda *_: counts.__setitem__("posterior", counts["posterior"] + 1)),
            base.prior.register_forward_pre_hook(lambda *_: counts.__setitem__("prior", counts["prior"] + 1)),
        ]
        output = model(batch, state_mask, action_mask)
        for handle in handles:
            handle.remove()
        self.assertTrue(torch.equal(output.posterior_mean[0], output.posterior_mean[1]))
        self.assertEqual(counts, {"encoder": 0, "posterior": 0, "prior": 0})

    def test_centroid_is_deterministic_mean_of_ten_mask_conditioned_codes(self) -> None:
        loader = DataLoader(_CentroidDataset(), batch_size=40, shuffle=False)
        first, manifest, tensors = initialize_window_codes(
            _CentroidEncoder(), loader, torch.device("cpu")
        )
        second, second_manifest, _ = initialize_window_codes(
            _CentroidEncoder(), loader, torch.device("cpu")
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[:, 0], torch.arange(80).float() + 4.5))
        self.assertEqual(manifest["centroid_sha256"], second_manifest["centroid_sha256"])
        self.assertEqual(tuple(tensors["posterior_mean_by_fixture"].shape), (800, 256))

    def test_trainable_allowlists_freeze_every_encoder_parameter(self) -> None:
        model = WindowCodeAutoDecoder(
            PosteriorCapacityTransformerCVAE(_config()), torch.zeros(80, 256)
        )
        e1 = configure_trainable_parameters(model, "E1")
        self.assertEqual(e1["trainable_names"], ["window_codes.weight"])
        self.assertEqual(e1["trainable_parameter_count"], EXPECTED_CODE_PARAMETERS)
        e2 = configure_trainable_parameters(model, "E2")
        self.assertTrue(any(name.startswith("base_model.decoder.") for name in e2["trainable_names"]))
        self.assertFalse(any(name.startswith("base_model.encoder") for name in e2["trainable_names"]))
        self.assertFalse(any(name.startswith("base_model.posterior") for name in e2["trainable_names"]))
        self.assertFalse(any(name.startswith("base_model.prior") for name in e2["trainable_names"]))
        batch = _batch()
        state_mask = torch.ones_like(batch["physical_state"], dtype=torch.bool)
        action_mask = torch.ones_like(batch["action"], dtype=torch.bool)
        reconstruction_loss(
            model(batch, state_mask, action_mask), batch, state_mask, action_mask
        ).total.backward()
        self.assertTrue(all(
            parameter.grad is None
            for name, parameter in model.named_parameters()
            if name.startswith(("base_model.encoder", "base_model.posterior", "base_model.prior"))
        ))

    def test_two_window_two_mask_code_only_synthetic_fit_decreases_loss(self) -> None:
        model = WindowCodeAutoDecoder(_ScalarDecoder(), torch.zeros(80, 256))
        batch = {
            "physical_state": torch.zeros(4, 2, 70),
            "action": torch.zeros(4, 1, 29),
            "window_index": torch.tensor([0, 0, 1, 1]),
        }
        targets = torch.tensor([1.0, 1.0, 2.0, 2.0])
        batch["physical_state"][..., :68] = targets[:, None, None]
        batch["action"][:] = targets[:, None, None]
        state_mask = torch.ones_like(batch["physical_state"], dtype=torch.bool)
        action_mask = torch.ones_like(batch["action"], dtype=torch.bool)
        optimizer = torch.optim.Adam([model.window_codes.weight], lr=0.2)
        initial = float(reconstruction_loss(model(batch, state_mask, action_mask), batch, state_mask, action_mask).total)
        for _ in range(80):
            optimizer.zero_grad()
            loss = reconstruction_loss(model(batch, state_mask, action_mask), batch, state_mask, action_mask).total
            loss.backward()
            optimizer.step()
        final = float(reconstruction_loss(model(batch, state_mask, action_mask), batch, state_mask, action_mask).total)
        self.assertLess(final, initial * 0.01)
        self.assertTrue(torch.equal(model.window_codes.weight[0], model.window_codes.weight[0]))

    def test_gate_requires_zero_cross_window_and_cross_motion_dependence(self) -> None:
        evaluation = {
            "exact": {
                "progression_gate": {
                    "threshold_ratios": {
                        "state_rmse": 0.5,
                        "action_rmse": 0.4,
                        "continuous_max_abs": 0.8,
                        "contact_accuracy": 1.0,
                        "zero_latent_dependence": 0.5,
                    },
                    "thresholds": {},
                },
                "latent_dependence": {"zero_ratio": 20.0},
            },
            "latent_donors": {"metrics": {
                "cross_window": {"continuous_ratio_to_correct": 12.0},
                "cross_motion": {"continuous_ratio_to_correct": 11.0},
            }},
            "zero_code_dependence": {"metrics": {
                "zero": {"continuous_ratio_to_correct": 20.0},
            }},
        }
        self.assertTrue(_autodecoder_gate(evaluation)["passed"])
        evaluation["latent_donors"]["metrics"]["cross_motion"]["continuous_ratio_to_correct"] = 9.0
        self.assertFalse(_autodecoder_gate(evaluation)["passed"])

    def test_final_stop_trigger_is_recomputed_from_all_three_arms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {}
            for arm in ("A", "B", "C"):
                run = root / arm
                (run / "manifests").mkdir(parents=True)
                summary = _summary(arm, 0.20, 0.13)
                path = run / "manifests/posterior_ab_summary.json"
                path.write_text(json.dumps(summary), encoding="utf-8")
                runs[arm] = run
            comparison = root / EXPECTED_TRIGGER_RUN_NAME
            result = run_ab_comparison(
                output_run=comparison,
                run_a=runs["A"], run_b=runs["B"], run_c=runs["C"],
            )
            self.assertEqual(result["decision"]["decision"], "STOP_LOSS_LATENT_SEED_SEARCH")
            validated = validate_stop_trigger(
                comparison, dataset_hash="dataset", source_checkpoint_hash="source"
            )
            self.assertTrue(all(validated["checks"].values()))

    def test_checkpoint_contract_and_marker_names(self) -> None:
        payload = {
            "format_version": CHECKPOINT_FORMAT,
            "stage": "E1",
            "optimizer_step": 2,
            "model": {"window_codes.weight": torch.zeros(80, 256)},
            "optimizer": {"state": {}},
            "scheduler": {"last_epoch": 2},
            "dataset_manifest_sha256": "dataset",
            "source_checkpoint_sha256": "source",
            "fixture_bitmap_sha256": "fixture",
            "code_initialization_sha256": "codes",
            "base_parameter_count": 25_453_411,
            "code_parameter_count": EXPECTED_CODE_PARAMETERS,
            "total_parameter_count": EXPECTED_TOTAL_PARAMETERS,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save(payload, path)
            result = validate_saved_checkpoint(
                path, stage="E1", expected_step=2, dataset_hash="dataset",
                source_hash="source", fixture_hash="fixture",
                code_initialization_hash="codes",
            )
        self.assertTrue(result["passed"])
        self.assertEqual(
            {SMOKE_MARKER, EXECUTION_MARKER, E1_MARKER, E2_MARKER},
            {
                "cvae_posterior_autodecoder_smoke.ok",
                "cvae_posterior_autodecoder_execution.ok",
                "cvae_posterior_autodecoder_code_only.ok",
                "cvae_posterior_autodecoder_coadapt.ok",
            },
        )

    def test_mask_svg_has_log_axis_and_all_ten_masks(self) -> None:
        svg = _categorical_svg("diagnostic", [
            {"name": name, "state": 0.1, "action": 0.2, "max_abs": 0.3}
            for name in FIXED_MASK_NAMES
        ])
        self.assertIn("Value (log10 scale)", svg)
        self.assertIn("clip non-positive values to 1e-12", svg)
        for name in FIXED_MASK_NAMES:
            self.assertIn(name, svg)

    def test_training_plots_record_components_lr_grad_and_code_gates(self) -> None:
        train = {
            "phase": "train",
            "optimizer_step": 1,
            "raw_reconstruction": {"total": 0.2},
            "learning_rates": {"window_codes": 3e-4},
            "gradient_norm_before_clip": 0.5,
        }
        evaluation = {
            "phase": "evaluation",
            "optimizer_step": 1,
            "exact": {
                "reconstruction_loss": {
                    "total": 0.1, "state": 0.1, "action": 0.1, "contact": 0.1,
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
            "autodecoder_progression_gate": {
                "zero_ratio": 20.0,
                "cross_window_ratio": 20.0,
                "cross_motion_ratio": 20.0,
            },
        }
        code_manifest = {
            "per_window": [{
                "window_index": index,
                "rms_distance_to_centroid": 0.1,
                "max_abs_distance_to_centroid": 0.2,
            } for index in range(80)]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plots").mkdir()
            paths = render_plots(root, [train, evaluation], code_manifest, evaluation)
            training = Path(paths["training_curves"]).read_text(encoding="utf-8")
            gates = Path(paths["gate_curves"]).read_text(encoding="utf-8")
            codes = Path(paths["code_statistics"]).read_text(encoding="utf-8")
        self.assertIn("Evaluation contact BCE", training)
        self.assertIn("Peak learning rate", training)
        self.assertIn("Gradient norm", training)
        self.assertIn("10 / cross-motion ratio", gates)
        self.assertIn("Window index", codes)


if __name__ == "__main__":
    unittest.main()
