from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from cvae_sa.models import build_model, parameter_count
from cvae_sa.trainer import (
    action_finetune_parameter_groups,
    load_weight_only_initialization,
    overfit_gate,
    overfit_latent_protocol,
    validate_action_finetune,
    validate_overfit_suite,
    warmup_cosine_factor,
)


MODEL_CONFIG = {
    "kind": "physics_transformer",
    "d_model": 32,
    "encoder_layers": 1,
    "decoder_layers": 1,
    "heads": 4,
    "ffn_dim": 64,
    "latent_dim": 8,
    "joint_width": 16,
    "dropout": 0.0,
    "context_mode": "hidden",
    "state_dim": 70,
    "include_previous_action": False,
    "robot_info_dim": 293,
    "dynamics_context_dim": 648,
    "auxiliary_dim": 35,
    "joint_robot_info_dim": 11,
    "global_robot_info_dim": 9,
    "actuator_type_count": 1,
    "token_layout": "interleaved",
}


class _FixedMasker:
    def __init__(self) -> None:
        self.optimizer_step = 0

    def set_step(self, step: int) -> None:
        self.optimizer_step = step

    def generate(self, batch: dict[str, torch.Tensor], **_: object) -> SimpleNamespace:
        batch_size = batch["action"].shape[0]
        return SimpleNamespace(
            forward_transition=torch.ones(batch_size, 128, dtype=torch.bool),
            inverse_transition=torch.ones(batch_size, 128, dtype=torch.bool),
            history_action_transition=torch.ones(batch_size, 128, dtype=torch.bool),
            state_loss=torch.ones(batch_size, 129, 70, dtype=torch.bool),
            action_loss=torch.ones(batch_size, 128, 29, dtype=torch.bool),
            rollout_start=torch.zeros(batch_size, dtype=torch.long),
            rollout_horizon=torch.full((batch_size,), 8, dtype=torch.long),
        )


class _LatentSpyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[int, bool, bool]] = []

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        masks: SimpleNamespace,
        sample_from_prior: bool = False,
        deterministic: bool = False,
    ) -> SimpleNamespace:
        self.calls.append((id(masks), sample_from_prior, deterministic))
        value = 2.0 if sample_from_prior else 0.0
        return SimpleNamespace(
            forward_delta=torch.full((1, 128, 70), value),
            rollout_state=torch.full((1, 8, 70), value),
            inverse_action=torch.full((1, 128, 29), value),
            history_action=torch.full((1, 128, 29), value),
            action=torch.full((1, 128, 29), value),
            physical_state=torch.full((1, 129, 70), value),
        )


class ActionFinetuneTest(unittest.TestCase):
    def test_compact_parameter_count_and_overfit_gate(self) -> None:
        compact = dict(
            MODEL_CONFIG,
            d_model=320,
            encoder_layers=4,
            decoder_layers=6,
            heads=8,
            ffn_dim=1280,
            latent_dim=80,
            joint_width=128,
            actuator_type_count=2,
            robot_conditioning="full",
        )
        self.assertEqual(parameter_count(build_model(compact)), 15_065_048)
        passing = {
            "forward_one_normalized_rmse": 0.05,
            "action_inverse_local_normalized_rmse": 0.05,
            "arbitrary_state_normalized_rmse": 0.05,
            "action_completion_macro_normalized_rmse": 0.05,
            "history_action_normalized_rmse": 0.08,
            "forward_rollout_8_normalized_rmse": 0.10,
        }
        self.assertTrue(overfit_gate(passing)["overfit_pass"])
        passing["history_action_normalized_rmse"] = 0.081
        self.assertFalse(overfit_gate(passing)["overfit_pass"])

    def test_warmup_cosine_is_monotonic_after_warmup_and_has_floor(self) -> None:
        values = [warmup_cosine_factor(step, 500, 20_000, 0.01) for step in range(500, 20_001)]
        self.assertTrue(all(left >= right for left, right in zip(values, values[1:])))
        self.assertAlmostEqual(values[0], 1.0)
        self.assertAlmostEqual(values[-1], 0.01)

    def test_parameter_groups_are_disjoint_and_use_declared_rates(self) -> None:
        model = build_model(MODEL_CONFIG)
        rates = {"action": 5e-5, "shared": 2e-5, "state": 1e-5}
        groups, summary = action_finetune_parameter_groups(model, rates)
        identifiers = [id(parameter) for group in groups for parameter in group["params"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(len(identifiers), len(list(model.parameters())))
        self.assertEqual({group["group_name"] for group in groups}, set(rates))
        self.assertEqual(
            sum(item["parameter_count"] for item in summary.values()),
            sum(parameter.numel() for parameter in model.parameters()),
        )

    def test_weight_only_initialization_is_strict_and_hash_gated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "parent.pt"
            parent = build_model(MODEL_CONFIG)
            torch.save(
                {
                    "format_version": "sonic_state_action_cvae_checkpoint_v2",
                    "step": 120000,
                    "best_score": 0.3,
                    "model": parent.state_dict(),
                    "config": {"model": MODEL_CONFIG},
                    "dataset_manifest_sha256": "dataset-hash",
                },
                checkpoint_path,
            )
            child = build_model(MODEL_CONFIG)
            provenance = load_weight_only_initialization(
                child, checkpoint_path, "dataset-hash", MODEL_CONFIG
            )
            self.assertEqual(provenance["initialization"], "weights_only")
            self.assertTrue(provenance["optimizer_reinitialized"])
            for parent_value, child_value in zip(parent.parameters(), child.parameters()):
                self.assertTrue(torch.equal(parent_value, child_value))
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                load_weight_only_initialization(
                    child, checkpoint_path, "different-hash", MODEL_CONFIG
                )

    def test_overfit_latent_protocol_is_phase_strict(self) -> None:
        capacity = {
            "overfit_gate_latent_mode": "posterior_mean",
            "overfit_diagnostic_latent_modes": ["prior_mean"],
        }
        full = {
            "overfit_gate_latent_mode": "prior_mean",
            "overfit_diagnostic_latent_modes": [],
        }
        self.assertEqual(
            overfit_latent_protocol(capacity, "capacity"),
            ("posterior_mean", ("prior_mean",)),
        )
        self.assertEqual(overfit_latent_protocol(full, "full"), ("prior_mean", ()))
        with self.assertRaisesRegex(ValueError, "capacity overfit requires"):
            overfit_latent_protocol(full, "capacity")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            overfit_latent_protocol({
                "overfit_gate_latent_mode": "unknown",
                "overfit_diagnostic_latent_modes": [],
            }, "capacity")

    def test_capacity_gate_and_prior_diagnostic_share_masks_but_not_scores(self) -> None:
        model = _LatentSpyModel()
        masker = _FixedMasker()
        batch = {
            "physical_state": torch.zeros(1, 129, 70),
            "action": torch.zeros(1, 128, 29),
        }
        result = validate_overfit_suite(
            model,
            [batch] * 12,
            masker,
            torch.device("cpu"),
            "bf16",
            12,
            "posterior_mean",
            ("prior_mean",),
        )
        self.assertEqual(result["gate_latent_mode"], "posterior_mean")
        self.assertEqual(result["arbitrary_state_normalized_rmse"], 0.0)
        self.assertEqual(
            result["latent_diagnostics"]["prior_mean"][
                "arbitrary_state_normalized_rmse"
            ],
            2.0,
        )
        self.assertTrue(overfit_gate(result)["overfit_pass"])
        self.assertEqual(len(model.calls), 24)
        for posterior, prior in zip(model.calls[0::2], model.calls[1::2]):
            self.assertEqual(posterior[0], prior[0])
            self.assertEqual(posterior[1:], (False, True))
            self.assertEqual(prior[1:], (True, True))

    def test_full_gate_uses_prior_mean(self) -> None:
        model = _LatentSpyModel()
        masker = _FixedMasker()
        batch = {
            "physical_state": torch.zeros(1, 129, 70),
            "action": torch.zeros(1, 128, 29),
        }
        result = validate_overfit_suite(
            model,
            [batch] * 12,
            masker,
            torch.device("cpu"),
            "bf16",
            12,
            "prior_mean",
            (),
        )
        self.assertEqual(result["gate_latent_mode"], "prior_mean")
        self.assertEqual(result["latent_diagnostics"], {})
        self.assertFalse(overfit_gate(result)["overfit_pass"])
        self.assertTrue(all(call[1:] == (True, True) for call in model.calls))

    def test_action_finetune_keeps_prior_mean_validation(self) -> None:
        model = _LatentSpyModel()
        masker = _FixedMasker()
        batch = {
            "physical_state": torch.zeros(1, 129, 70),
            "action": torch.zeros(1, 128, 29),
        }
        result = validate_action_finetune(
            model, [batch] * 12, masker, torch.device("cpu"), "bf16", 12
        )
        self.assertEqual(result["arbitrary_state_normalized_rmse"], 2.0)
        self.assertTrue(all(call[1:] == (True, True) for call in model.calls))


if __name__ == "__main__":
    unittest.main()
