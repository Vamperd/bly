from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from cvae_sa.models import build_model
from cvae_sa.trainer import (
    action_finetune_parameter_groups,
    load_weight_only_initialization,
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


class ActionFinetuneTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
