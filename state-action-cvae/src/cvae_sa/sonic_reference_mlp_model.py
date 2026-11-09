"""Pure-MLP replacement for the 10-history SONIC reference CVAE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .sonic_causal_model import ACTION_DIM, HISTORY_STEPS, REFERENCE_DIM, REFERENCE_FRAMES, STATE_DIM, SonicGaussianLatent

ARCHITECTURE_VERSION = "sonic-10history-reference-mlp-cvae-v1"
STATE_HISTORY_DIM = HISTORY_STEPS * STATE_DIM
ACTION_HISTORY_DIM = HISTORY_STEPS * ACTION_DIM
REFERENCE_HISTORY_DIM = REFERENCE_FRAMES * REFERENCE_DIM


def _mlp(input_dim: int, widths: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = int(input_dim)
    for width in widths:
        layers.extend((nn.Linear(current, int(width)), nn.LayerNorm(int(width)), nn.GELU()))
        current = int(width)
    layers.append(nn.Linear(current, int(output_dim)))
    return nn.Sequential(*layers)


@dataclass
class SonicReferenceMlpOutput:
    action: torch.Tensor
    posterior: SonicGaussianLatent | None
    global_latent: torch.Tensor
    latent_source: str


class SonicReferenceMLPCVAE(nn.Module):
    """One-action CVAE with the exact 10-history/10-reference contract."""

    ARCHITECTURE_VERSION = ARCHITECTURE_VERSION
    HISTORY_STEPS = HISTORY_STEPS
    REFERENCE_FRAMES = REFERENCE_FRAMES
    STATE_DIM = STATE_DIM
    ACTION_DIM = ACTION_DIM
    REFERENCE_DIM = REFERENCE_DIM
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.global_latent_dim = int(config.get("global_latent_dim", 256))
        self.encoder_widths = [int(v) for v in config.get("encoder_widths", [2560, 1792, 896, 512])]
        self.decoder_widths = [int(v) for v in config.get("decoder_widths", [2048, 1536, 768])]
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.reference_dim = REFERENCE_DIM
        self.history_steps = HISTORY_STEPS
        self.reference_frames = REFERENCE_FRAMES
        # Values, masks/validity, offsets, and target marker.
        self.condition_dim = (
            STATE_HISTORY_DIM + ACTION_HISTORY_DIM + REFERENCE_HISTORY_DIM
            + HISTORY_STEPS + HISTORY_STEPS + HISTORY_STEPS * 2
            + REFERENCE_FRAMES + 2 * ACTION_DIM
        )
        self.posterior_condition_dim = self.condition_dim + ACTION_DIM
        self.posterior_encoder = _mlp(self.posterior_condition_dim, self.encoder_widths, self.encoder_widths[-1])
        self.condition_encoder = _mlp(self.condition_dim, self.encoder_widths, self.encoder_widths[-1])
        self.posterior_mean = nn.Linear(self.encoder_widths[-1], self.global_latent_dim)
        self.posterior_logvar = nn.Linear(self.encoder_widths[-1], self.global_latent_dim)
        self.decoder = _mlp(self.condition_dim + self.global_latent_dim, self.decoder_widths, ACTION_DIM)
        nn.init.zeros_(self.posterior_logvar.weight)
        nn.init.constant_(self.posterior_logvar.bias, -4.0)

    @classmethod
    def architecture_signature(cls, config: dict[str, Any], parameter_count: int | None = None) -> dict[str, Any]:
        signature = {
            "architecture_version": ARCHITECTURE_VERSION,
            "history_steps": HISTORY_STEPS,
            "reference_frames": REFERENCE_FRAMES,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "reference_dim": REFERENCE_DIM,
            "global_latent_dim": int(config.get("global_latent_dim", 256)),
            "encoder_widths": [int(v) for v in config.get("encoder_widths", [2560, 1792, 896, 512])],
            "decoder_widths": [int(v) for v in config.get("decoder_widths", [2048, 1536, 768])],
            "condition_dim": 0,
            "latent_contract": "q_posterior_plus_fixed_standard_normal_condition_decoder",
        }
        condition_dim = (
            STATE_HISTORY_DIM + ACTION_HISTORY_DIM + REFERENCE_HISTORY_DIM
            + HISTORY_STEPS + HISTORY_STEPS + HISTORY_STEPS * 2
            + REFERENCE_FRAMES + 2 * ACTION_DIM
        )
        signature["condition_dim"] = condition_dim
        if parameter_count is not None:
            signature["parameter_count"] = int(parameter_count)
        return signature

    @staticmethod
    def _check_batch(batch: dict[str, torch.Tensor]) -> None:
        required = ("state_history", "action_history", "reference_future", "target_action")
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"reference MLP batch is missing {missing}")
        expected = {
            "state_history": (HISTORY_STEPS, STATE_DIM),
            "action_history": (HISTORY_STEPS, ACTION_DIM),
            "reference_future": (REFERENCE_FRAMES, REFERENCE_DIM),
            "target_action": (ACTION_DIM,),
        }
        for key, shape in expected.items():
            if tuple(batch[key].shape[1:]) != shape:
                raise ValueError(f"{key} must have shape [B,{','.join(map(str, shape))}]")

    def _condition_vector(self, batch: dict[str, torch.Tensor], *, include_target: bool) -> torch.Tensor:
        self._check_batch(batch)
        state = batch["state_history"].float()
        action = batch["action_history"].float()
        reference = batch["reference_future"].float()
        device = state.device
        state_valid = batch.get("state_valid", torch.ones(state.shape[:2], device=device, dtype=torch.bool)).bool()
        action_valid = batch.get("action_valid", torch.ones(action.shape[:2], device=device, dtype=torch.bool)).bool()
        history_validity = torch.stack((state_valid, action_valid), dim=-1).to(state.dtype)
        state_value = state * state_valid[..., None].to(state.dtype)
        action_value = action * action_valid[..., None].to(action.dtype)
        offsets = batch.get("reference_time_offsets")
        if offsets is None:
            offsets = torch.arange(REFERENCE_FRAMES, device=device, dtype=state.dtype)[None].expand(state.shape[0], -1) * 0.1
        offsets = offsets.to(state.dtype)
        target_marker = torch.ones(state.shape[0], ACTION_DIM, device=device, dtype=state.dtype)
        target_value = batch["target_action"].float() if include_target else torch.zeros_like(batch["target_action"].float())
        return torch.cat((
            state_value.flatten(1), action_value.flatten(1), reference.flatten(1),
            state_valid.to(state.dtype), action_valid.to(action.dtype), history_validity.flatten(1),
            offsets.flatten(1), target_marker, target_value if include_target else torch.zeros_like(target_value),
        ), dim=1)

    @staticmethod
    def reparameterize(distribution: SonicGaussianLatent, epsilon: torch.Tensor | None = None) -> torch.Tensor:
        if epsilon is None:
            epsilon = torch.randn_like(distribution.mean)
        return distribution.mean.float() + torch.exp(0.5 * distribution.logvar.float()) * epsilon.float()

    def encode_posterior_distribution(self, batch: dict[str, torch.Tensor]) -> SonicGaussianLatent:
        condition = self._condition_vector(batch, include_target=False)
        posterior = torch.cat((condition, batch["target_action"].float()), dim=1)
        hidden = self.posterior_encoder(posterior)
        return SonicGaussianLatent(
            mean=self.posterior_mean(hidden),
            logvar=self.posterior_logvar(hidden).clamp(-8.0, 4.0),
        )

    def encode_condition(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.condition_encoder(self._condition_vector(batch, include_target=False))

    def _decode(self, condition: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat((condition, latent), dim=1))

    def forward(self, batch: dict[str, torch.Tensor], *, stage: str = "C", epsilon: torch.Tensor | None = None) -> SonicReferenceMlpOutput:
        stage = str(stage).upper()
        condition = self._condition_vector(batch, include_target=False)
        if stage == "A":
            posterior = self.encode_posterior_distribution(batch)
            latent = posterior.mean
            source = "posterior_mean"
        elif stage == "C":
            posterior = self.encode_posterior_distribution(batch)
            latent = self.reparameterize(posterior, epsilon)
            source = "posterior_sample"
        elif stage in {"INFER", "STANDARD_NORMAL"}:
            posterior = None
            latent = epsilon if epsilon is not None else torch.randn(batch["state_history"].shape[0], self.global_latent_dim, device=batch["state_history"].device)
            source = "standard_normal"
        else:
            raise ValueError("stage must be A, C, INFER or STANDARD_NORMAL")
        action = self._decode(condition, latent)
        return SonicReferenceMlpOutput(action=action, posterior=posterior, global_latent=latent, latent_source=source)

    def infer_from_condition(self, batch: dict[str, torch.Tensor], *, epsilon: torch.Tensor | None = None, sample: bool = True) -> SonicReferenceMlpOutput:
        if not sample:
            epsilon = torch.zeros(batch["state_history"].shape[0], self.global_latent_dim, device=batch["state_history"].device)
        return self.forward(batch, stage="INFER", epsilon=epsilon)

    def architecture_instance_signature(self) -> dict[str, Any]:
        return self.architecture_signature(self.config, sum(p.numel() for p in self.parameters()))
