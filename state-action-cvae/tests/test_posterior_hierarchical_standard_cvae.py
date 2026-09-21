from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from cvae_sa.models import HierarchicalStandardCVAETransformer, build_model
from cvae_sa.posterior_hierarchical_standard_cvae import (
    _new_checkpoint,
    hierarchical_kl,
    load_checkpoint,
    validate_checkpoint,
)


def config() -> dict[str, object]:
    return {
        "kind": "physics_hierarchical_standard_cvae_transformer",
        "architecture_version": "65-token-hierarchical-standard-cvae-v1",
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
        "state_dim": 70,
        "state_input_dim": 99,
        "condition_input_dim": 198,
        "dropout": 0.0,
    }


def batch(size: int = 2) -> dict[str, torch.Tensor | list[str]]:
    state = torch.randn(size, 65, 70)
    state[..., 68:] = torch.randint(0, 2, (size, 65, 2)).float()
    return {
        "physical_state": state,
        "action": torch.randn(size, 64, 29),
        "valid_state": torch.ones(size, 65, dtype=torch.bool),
        "valid_action": torch.ones(size, 64, dtype=torch.bool),
        "motion_key": [f"motion-{i}" for i in range(size)],
    }


class HierarchicalStandardCVAETest(unittest.TestCase):
    def test_builds_new_signature_and_input_output_shapes(self) -> None:
        model = build_model(config())
        self.assertIsInstance(model, HierarchicalStandardCVAETransformer)
        value = batch()
        state_mask = torch.zeros(2, 65, 70, dtype=torch.bool)
        action_mask = torch.zeros(2, 64, 29, dtype=torch.bool)
        posterior = model.encode_posterior_distribution(value, state_mask, action_mask)
        condition = model.encode_condition(value, state_mask, action_mask)
        self.assertEqual(tuple(posterior.global_mean.shape), (2, 16))
        self.assertEqual(tuple(posterior.local_mean.shape), (2, 16, 8))
        self.assertEqual(tuple(condition.memory.shape), (2, 65, 32))
        self.assertEqual(tuple(condition.local_latents.shape), (2, 16, 8))
        output = model(value, state_mask, action_mask, stage="B")
        self.assertEqual(tuple(output.physical_state.shape), (2, 65, 70))
        self.assertEqual(tuple(output.action.shape), (2, 64, 29))
        fused = model.fuse_latents(
            (output.global_latent, output.local_latents), output.condition
        )
        self.assertEqual(tuple(fused.condition_memory.shape), (2, 65, 32))
        self.assertEqual(tuple(fused.film_input.shape), (2, 65, 32))
        self.assertEqual(fused.condition_memory.shape[1] + 1 + fused.memory_local.shape[1], 82)

    def test_posterior_ignores_mask_and_condition_blocks_masked_truth(self) -> None:
        torch.manual_seed(3)
        model = HierarchicalStandardCVAETransformer(config()).eval()
        value = batch()
        state_mask = torch.zeros(2, 65, 70, dtype=torch.bool)
        action_mask = torch.zeros(2, 64, 29, dtype=torch.bool)
        state_mask[:, 10, :] = True
        action_mask[:, 5, :] = True
        first = model.encode_posterior_distribution(value, state_mask, action_mask)
        second = model.encode_posterior_distribution(value, ~state_mask, ~action_mask)
        self.assertTrue(torch.equal(first.global_mean, second.global_mean))
        self.assertTrue(torch.equal(first.local_mean, second.local_mean))
        condition = model.encode_condition(value, state_mask, action_mask)
        changed = dict(value)
        changed["physical_state"] = value["physical_state"].masked_fill(state_mask, 999.0)
        changed["action"] = value["action"].masked_fill(action_mask, -999.0)
        changed_condition = model.encode_condition(changed, state_mask, action_mask)
        self.assertTrue(torch.equal(condition.memory, changed_condition.memory))
        self.assertTrue(torch.equal(condition.global_latent, changed_condition.global_latent))

    def test_terminal_and_hard_chunk_contract(self) -> None:
        model = HierarchicalStandardCVAETransformer(config())
        value = batch(1)
        encoded = model.encode_condition(value, torch.zeros(1, 65, 70, dtype=torch.bool), torch.zeros(1, 64, 29, dtype=torch.bool))
        self.assertEqual(encoded.local_latents.shape[1], 16)
        self.assertTrue(torch.equal(model.local_chunk_ids(torch.arange(60, 65)), torch.full((5,), 15)))
        output = model(value, stage="A")
        self.assertEqual(output.action.shape[1], 64)

    def test_stage_routing_and_finite_backward(self) -> None:
        value = batch()
        for stage in ("A", "B", "C"):
            model = HierarchicalStandardCVAETransformer(config())
            model.set_training_stage(stage)
            state_mask = torch.zeros(2, 65, 70, dtype=torch.bool)
            action_mask = torch.zeros(2, 64, 29, dtype=torch.bool)
            output = model(value, state_mask, action_mask, stage=stage)
            loss = output.physical_state.square().mean() + output.action.square().mean()
            if output.posterior is not None:
                loss = loss + hierarchical_kl(output.posterior)["total"]
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            if stage == "A":
                self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if n.startswith("condition_")))
            if stage == "B":
                self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if n.startswith("posterior_")))

    def test_inference_does_not_call_posterior(self) -> None:
        model = HierarchicalStandardCVAETransformer(config()).eval()
        value = batch()
        state_mask = torch.zeros(2, 65, 70, dtype=torch.bool)
        action_mask = torch.zeros(2, 64, 29, dtype=torch.bool)
        model.encode_posterior_distribution = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("posterior called"))  # type: ignore[method-assign]
        output = model.infer_from_condition(value, state_mask, action_mask)
        self.assertIsNone(output.posterior)
        self.assertEqual(output.latent_source, "standard_normal")

    def test_checkpoint_readback_and_legacy_rejection(self) -> None:
        model = HierarchicalStandardCVAETransformer(config())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "new.pt"
            torch.save(_new_checkpoint(model, optimizer, None, {"model": model.config}, stage="C", step=1), path)
            self.assertTrue(validate_checkpoint(path, {"stage": "C"})["passed"])
            load_checkpoint(model, path)
            legacy = Path(temporary) / "legacy.pt"
            torch.save({"format_version": "sonic_h50_standard_cvae_checkpoint_v1", "model": {}}, legacy)
            with self.assertRaisesRegex(ValueError, "architecture signature mismatch"):
                load_checkpoint(model, legacy)


if __name__ == "__main__":
    unittest.main()
