from __future__ import annotations

import unittest

import torch

from cvae_sa.losses import compute_loss
from cvae_sa.masking import MaskGenerator
from cvae_sa.models import build_model


def batch(batch_size: int = 2, transitions: int = 8) -> dict[str, torch.Tensor]:
    return {
        "physical_state": torch.randn(batch_size, transitions + 1, 64),
        "previous_action": torch.randn(batch_size, transitions + 1, 29),
        "action": torch.randn(batch_size, transitions, 29),
        "action_before_window": torch.randn(batch_size, 29),
        "action_scale": torch.ones(batch_size, 29),
        "robot_information": torch.empty(batch_size, 0),
        "joint_robot_information": torch.empty(batch_size, 29, 0),
        "joint_actuator_type": torch.zeros(batch_size, 29, dtype=torch.long),
        "global_robot_information": torch.empty(batch_size, 0),
        "dynamics_context": torch.empty(batch_size, 0),
        "auxiliary_transition": torch.empty(batch_size, transitions, 0),
        "valid_state": torch.ones(batch_size, transitions + 1, dtype=torch.bool),
        "valid_action": torch.ones(batch_size, transitions, dtype=torch.bool),
        "progress": torch.linspace(0, 1, transitions + 1).repeat(batch_size, 1),
    }


def physics_batch(batch_size: int = 2, transitions: int = 8) -> dict[str, torch.Tensor]:
    value = batch(batch_size, transitions)
    value["physical_state"] = torch.randn(batch_size, transitions + 1, 70)
    value["physical_state"][..., 64:67] = torch.nn.functional.normalize(
        value["physical_state"][..., 64:67], dim=-1
    )
    value["physical_state"][..., 68:70] = torch.randint(
        0, 2, (batch_size, transitions + 1, 2)
    ).float()
    value["previous_action"] = torch.empty(batch_size, transitions + 1, 0)
    value["robot_information"] = torch.randn(batch_size, 293)
    value["joint_robot_information"] = torch.randn(batch_size, 29, 11)
    value["joint_actuator_type"] = torch.zeros(batch_size, 29, dtype=torch.long)
    value["global_robot_information"] = torch.randn(batch_size, 9)
    value["dynamics_context"] = torch.randn(batch_size, 648)
    value["auxiliary_transition"] = torch.randn(batch_size, transitions, 35)
    return value


MASK_CONFIG = {
    "task_probabilities": [0.35, 0.25, 0.40],
    "completion_probabilities": [0.40, 0.30, 0.30],
    "element_fraction": [0.20, 0.20],
    "step_count": [2, 2],
    "feature_fraction": [0.20, 0.20],
}


PHYSICS_MASK_CONFIG = {
    **MASK_CONFIG,
    "strategy": "physics_bidirectional_v1",
    "relation_probabilities": [0.40, 0.35, 0.25],
    "forward_subprobabilities": [0.25, 0.50, 0.25],
    "calibration_steps": [2, 4],
    "physics_step_count": [1, 4],
    "physics_element_fraction": [0.10, 0.50],
    "physics_feature_fraction": [0.10, 0.50],
    "structured_overlay_max_fraction": 0.0,
    "rollout_start_step": 0,
}


MODEL_CONFIG = {
    "kind": "transformer",
    "d_model": 32,
    "encoder_layers": 1,
    "decoder_layers": 1,
    "heads": 4,
    "ffn_dim": 64,
    "latent_dim": 8,
    "dropout": 0.0,
    "tcn_channels": 32,
    "tcn_encoder_layers": 1,
    "tcn_decoder_layers": 1,
    "tcn_kernel_size": 3,
}


class MaskingModelTest(unittest.TestCase):
    def test_physics_rollout_curriculum_uses_fixed_horizons(self) -> None:
        value = physics_batch(transitions=16)
        config = dict(PHYSICS_MASK_CONFIG, rollout_start_step=20)
        masker = MaskGenerator(config)
        early = masker.generate(value, force_task="forward_rollout")
        self.assertTrue(torch.equal(early.rollout_horizon, torch.zeros_like(
            early.rollout_horizon
        )))
        masker.set_step(20)
        validation = masker.generate(value, force_task="forward_rollout")
        self.assertTrue(bool((validation.rollout_horizon == 4).all()))

    def test_action_mask_hides_duplicate_without_double_loss(self) -> None:
        value = batch()
        masks = MaskGenerator(MASK_CONFIG).generate(value, force_task="inverse")
        self.assertTrue(torch.equal(masks.previous_input[:, 1:], masks.action_input))
        self.assertFalse(bool(masks.previous_loss.any()))

    def test_transformer_and_tcn_shapes_are_finite(self) -> None:
        value = batch()
        masks = MaskGenerator(MASK_CONFIG).generate(
            value, force_task="completion", force_completion="feature"
        )
        for kind in ("transformer", "tcn"):
            config = dict(MODEL_CONFIG, kind=kind)
            model = build_model(config)
            output = model(value, masks, deterministic=True)
            self.assertEqual(tuple(output.physical_state.shape), (2, 9, 64))
            self.assertEqual(tuple(output.previous_action.shape), (2, 9, 29))
            self.assertEqual(tuple(output.action.shape), (2, 8, 29))
            self.assertEqual(tuple(output.forward_delta.shape), (2, 8, 64))
            self.assertTrue(torch.isfinite(output.physical_state).all())

    def test_physics_v3_interleaved_models_have_no_previous_action_branch(self) -> None:
        value = physics_batch()
        masks = MaskGenerator(MASK_CONFIG).generate(
            value, force_task="completion", force_completion="step"
        )
        self.assertEqual(masks.previous_input.shape[-1], 0)
        for kind in ("transformer", "tcn"):
            config = dict(
                MODEL_CONFIG,
                kind=kind,
                state_dim=70,
                include_previous_action=False,
                robot_info_dim=293,
                auxiliary_dim=35,
                token_layout="interleaved",
            )
            output = build_model(config)(value, masks, deterministic=True)
            self.assertEqual(tuple(output.physical_state.shape), (2, 9, 70))
            self.assertEqual(tuple(output.previous_action.shape), (2, 9, 0))
            self.assertEqual(tuple(output.action.shape), (2, 8, 29))
            self.assertEqual(tuple(output.forward_delta.shape), (2, 8, 70))
            self.assertEqual(tuple(output.auxiliary_transition.shape), (2, 8, 35))
            self.assertTrue(torch.isfinite(output.action).all())
            loss = compute_loss(
                output,
                value,
                masks,
                {
                    "free_bits": 0.0,
                    "forward_weight": 2.0,
                    "inverse_weight": 1.0,
                    "gravity_weight": 0.1,
                    "auxiliary_weight": 0.1,
                },
                kl_beta=0.0,
            )
            self.assertTrue(torch.isfinite(loss.total))

    def test_physics_v3_explicit_context_path(self) -> None:
        value = physics_batch()
        masks = MaskGenerator(MASK_CONFIG).generate(value, force_task="forward")
        config = dict(
            MODEL_CONFIG,
            state_dim=70,
            include_previous_action=False,
            robot_info_dim=293,
            dynamics_context_dim=648,
            auxiliary_dim=35,
            context_mode="explicit",
            token_layout="interleaved",
        )
        output = build_model(config)(value, masks, deterministic=True)
        self.assertEqual(tuple(output.action.shape), (2, 8, 29))
        self.assertTrue(torch.isfinite(output.forward_delta).all())

    def test_physics_forward_prior_depends_on_visible_sequence(self) -> None:
        value = physics_batch(batch_size=1)
        masks = MaskGenerator(MASK_CONFIG).generate(value, force_task="forward")
        config = dict(
            MODEL_CONFIG,
            state_dim=70,
            include_previous_action=False,
            robot_info_dim=293,
            dynamics_context_dim=648,
            auxiliary_dim=35,
            context_mode="hidden",
            token_layout="interleaved",
        )
        model = build_model(config)
        first = model(value, masks, sample_from_prior=True, deterministic=True)
        changed = dict(value)
        changed["action"] = value["action"] + 2.0
        second = model(changed, masks, sample_from_prior=True, deterministic=True)
        self.assertFalse(torch.allclose(first.prior_mean, second.prior_mean))

    def test_physics_transformer_bidirectional_heads_and_masks(self) -> None:
        value = physics_batch()
        masker = MaskGenerator(PHYSICS_MASK_CONFIG)
        config = dict(
            MODEL_CONFIG,
            kind="physics_transformer",
            state_dim=70,
            include_previous_action=False,
            joint_robot_info_dim=11,
            global_robot_info_dim=9,
            actuator_type_count=1,
            dynamics_context_dim=648,
            auxiliary_dim=35,
            context_mode="hidden",
            joint_width=16,
        )
        model = build_model(config)
        for task in ("forward_one", "inverse", "history_action", "arbitrary"):
            masks = masker.generate(value, force_task=task)
            if task == "forward_one":
                transition = masks.forward_transition[:, :, None].expand(
                    -1, -1, masks.state_input.shape[-1]
                )
                self.assertFalse(bool(masks.state_input[:, :-1].masked_select(transition).any()))
                self.assertFalse(bool(masks.action_input.masked_select(
                    masks.forward_transition[:, :, None].expand_as(masks.action_input)
                ).any()))
            if task == "history_action":
                self.assertTrue(bool(masks.state_input[:, 1:].any()))
                for index in range(masks.state_loss.shape[0]):
                    target = torch.nonzero(
                        masks.history_action_transition[index], as_tuple=False
                    ).flatten()
                    self.assertFalse(bool(
                        masks.state_loss[index, int(target.min().item()) + 1 :].any()
                    ))
            output = model(value, masks, deterministic=True)
            self.assertEqual(tuple(output.physical_state.shape), (2, 9, 70))
            self.assertEqual(tuple(output.action.shape), (2, 8, 29))
            self.assertEqual(tuple(output.inverse_action.shape), (2, 8, 29))
            self.assertEqual(tuple(output.history_action.shape), (2, 8, 29))
            self.assertEqual(output.rollout_state.shape[0], 2)
            self.assertEqual(output.rollout_state.shape[-1], 70)
            self.assertIn(output.rollout_state.shape[1], (0, 8))
            loss = compute_loss(
                output,
                value,
                masks,
                {
                    "free_bits": 0.0,
                    "forward_weight": 1.5,
                    "inverse_weight": 1.0,
                    "history_action_weight": 1.0,
                    "rollout_weight": 1.0,
                    "cycle_weight": 0.1,
                    "gravity_weight": 0.1,
                    "auxiliary_weight": 0.1,
                },
                kl_beta=0.0,
            )
            self.assertTrue(torch.isfinite(loss.total))

    def test_physics_relation_heads_do_not_read_masked_targets(self) -> None:
        torch.manual_seed(17)
        value = physics_batch(batch_size=1)
        masker = MaskGenerator(PHYSICS_MASK_CONFIG)
        config = dict(
            MODEL_CONFIG,
            kind="physics_transformer",
            state_dim=70,
            include_previous_action=False,
            joint_robot_info_dim=11,
            global_robot_info_dim=9,
            actuator_type_count=1,
            dynamics_context_dim=648,
            auxiliary_dim=35,
            context_mode="hidden",
            joint_width=16,
        )
        model = build_model(config).eval()

        forward_masks = masker.generate(value, force_task="forward_one")
        # Relation heads must remain target-blind even during the default
        # posterior reconstruction pass used for training.
        first = model(value, forward_masks, deterministic=True)
        changed = dict(value)
        changed_state = value["physical_state"].clone()
        changed_state[forward_masks.state_input] += 100.0
        changed["physical_state"] = changed_state
        second = model(changed, forward_masks, deterministic=True)
        target = forward_masks.forward_transition[:, :, None].expand_as(first.forward_delta)
        self.assertTrue(torch.allclose(
            first.forward_delta.masked_select(target),
            second.forward_delta.masked_select(target), atol=1e-6,
        ))

        inverse_masks = masker.generate(value, force_task="inverse")
        first = model(value, inverse_masks, deterministic=True)
        changed = dict(value)
        changed_action = value["action"].clone()
        changed_action[inverse_masks.action_input] += 100.0
        changed["action"] = changed_action
        second = model(changed, inverse_masks, deterministic=True)
        target = inverse_masks.inverse_transition[:, :, None].expand_as(first.inverse_action)
        self.assertTrue(torch.allclose(
            first.inverse_action.masked_select(target),
            second.inverse_action.masked_select(target), atol=1e-6,
        ))

        history_masks = masker.generate(value, force_task="history_action")
        self.assertTrue(history_masks.causal)
        self.assertTrue(bool(history_masks.state_input[:, 1:].any()))
        first = model(value, history_masks, deterministic=True)
        changed = dict(value)
        changed_state = value["physical_state"].clone()
        # Every future State hidden by the history task is deliberately
        # corrupted. Causal history-Action predictions must be unchanged.
        changed_state[history_masks.state_input] += 100.0
        changed["physical_state"] = changed_state
        second = model(changed, history_masks, deterministic=True)
        target = history_masks.history_action_transition[:, :, None].expand_as(
            first.history_action
        )
        self.assertTrue(torch.allclose(
            first.history_action.masked_select(target),
            second.history_action.masked_select(target), atol=1e-6,
        ))

    def test_hidden_physics_model_excludes_raw_mapping_and_randomized_context(self) -> None:
        torch.manual_seed(23)
        value = physics_batch(batch_size=1)
        masker = MaskGenerator(PHYSICS_MASK_CONFIG)
        masks = masker.generate(value, force_task="forward_one")
        config = dict(
            MODEL_CONFIG,
            kind="physics_transformer",
            state_dim=70,
            include_previous_action=False,
            joint_robot_info_dim=11,
            global_robot_info_dim=9,
            actuator_type_count=1,
            dynamics_context_dim=648,
            auxiliary_dim=35,
            context_mode="hidden",
            joint_width=16,
        )
        model = build_model(config).eval()
        first = model(value, masks, sample_from_prior=True, deterministic=True)
        changed = dict(value)
        changed["action_scale"] = value["action_scale"] + 100.0
        changed["dynamics_context"] = value["dynamics_context"] + 100.0
        second = model(changed, masks, sample_from_prior=True, deterministic=True)
        self.assertTrue(torch.allclose(first.forward_delta, second.forward_delta, atol=1e-6))

    def test_tiny_batch_can_overfit_inverse_mapping(self) -> None:
        torch.manual_seed(3)
        value = batch(batch_size=2, transitions=4)
        masks = MaskGenerator(MASK_CONFIG).generate(value, force_task="inverse")
        model = build_model(MODEL_CONFIG)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        training = {
            "free_bits": 0.0,
            "forward_weight": 2.0,
            "inverse_weight": 1.0,
            "gravity_weight": 0.1,
        }
        losses = []
        for _ in range(30):
            output = model(value, masks, deterministic=True)
            loss = compute_loss(output, value, masks, training, kl_beta=0.0).total
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(sum(losses[-5:]), sum(losses[:5]))


if __name__ == "__main__":
    unittest.main()
