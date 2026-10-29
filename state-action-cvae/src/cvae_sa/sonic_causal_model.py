"""SONIC-like single-step conditional CVAE.

This module is intentionally independent from the legacy 65-token model.  It
reuses the repository's Rotary Transformer, hierarchical decoder and gated
fusion primitives, but has an explicit 10-history/10-reference/1-query
contract and emits one canonical Action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .constants import ACTION_DIM
from .models import GatedResidualMLP, HierarchicalDecoderStack, TransformerStack


HISTORY_STEPS = 10
REFERENCE_FRAMES = 10
REFERENCE_DIM = 64
STATE_DIM = 70


@dataclass
class SonicGaussianLatent:
    mean: torch.Tensor
    logvar: torch.Tensor


@dataclass
class SonicCondition:
    memory: torch.Tensor
    pooled: torch.Tensor


@dataclass
class SonicCausalOutput:
    action: torch.Tensor
    posterior: SonicGaussianLatent | None
    global_latent: torch.Tensor
    latent_source: str
    condition: SonicCondition | None = None


class SonicCausalCVAE(nn.Module):
    """Single-action CVAE using 10 history and 10 SONIC reference frames."""

    ARCHITECTURE_VERSION = "sonic-10history-reference-cvae-v1"
    TOKEN_COUNT = 21
    HISTORY_STEPS = HISTORY_STEPS
    REFERENCE_FRAMES = REFERENCE_FRAMES
    STATE_DIM = STATE_DIM
    ACTION_DIM = ACTION_DIM
    REFERENCE_DIM = REFERENCE_DIM

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.width = int(config.get("d_model", 448))
        self.heads = int(config.get("heads", 8))
        self.ffn_dim = int(config.get("ffn_dim", self.width * 4))
        self.posterior_layers = int(config.get("posterior_encoder_layers", 6))
        self.condition_layers = int(config.get("condition_encoder_layers", 6))
        self.decoder_layers = int(config.get("decoder_layers", 8))
        self.global_latent_dim = int(config.get("global_latent_dim", 256))
        self.dropout = float(config.get("dropout", 0.0))
        if self.width % self.heads:
            raise ValueError("d_model must be divisible by heads")
        self.config.update(
            {
                "architecture_version": self.ARCHITECTURE_VERSION,
                "d_model": self.width,
                "heads": self.heads,
                "ffn_dim": self.ffn_dim,
                "posterior_encoder_layers": self.posterior_layers,
                "condition_encoder_layers": self.condition_layers,
                "decoder_layers": self.decoder_layers,
                "global_latent_dim": self.global_latent_dim,
                "history_steps": HISTORY_STEPS,
                "reference_frames": REFERENCE_FRAMES,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "reference_dim": REFERENCE_DIM,
            }
        )

        # History is State[70] + previous Action[29].  The reference token is
        # padded to the same 99-D payload so the three streams share one
        # Transformer width after independent projections.
        self.history_input = nn.Linear(2 * (STATE_DIM + ACTION_DIM), self.width)
        self.reference_input = nn.Linear(2 * (STATE_DIM + ACTION_DIM), self.width)
        self.query_input = nn.Linear(2 * (STATE_DIM + ACTION_DIM), self.width)
        self.condition_history_input = nn.Linear(
            2 * (STATE_DIM + ACTION_DIM), self.width
        )
        self.condition_reference_input = nn.Linear(
            2 * (STATE_DIM + ACTION_DIM), self.width
        )
        self.condition_query_input = nn.Linear(
            2 * (STATE_DIM + ACTION_DIM), self.width
        )
        self.time_embedding = nn.Embedding(self.TOKEN_COUNT, self.width)
        self.modality_embedding = nn.Embedding(3, self.width)
        self.direction_embedding = nn.Embedding(2, self.width)
        self.reference_offset_projection = nn.Linear(1, self.width)
        self.reference_frame_embedding = nn.Embedding(REFERENCE_FRAMES, self.width)
        self.posterior_cls = nn.Parameter(torch.zeros(1, 1, self.width))
        self.condition_cls = nn.Parameter(torch.zeros(1, 1, self.width))
        self.posterior_encoder = TransformerStack(
            self.posterior_layers,
            self.width,
            self.heads,
            self.ffn_dim,
            self.dropout,
        )
        self.condition_encoder = TransformerStack(
            self.condition_layers,
            self.width,
            self.heads,
            self.ffn_dim,
            self.dropout,
        )
        self.posterior_mean = nn.Linear(self.width, self.global_latent_dim)
        self.posterior_logvar = nn.Linear(self.width, self.global_latent_dim)
        self.condition_latent = nn.Linear(self.width, self.global_latent_dim)
        self.latent_fusion = GatedResidualMLP(
            self.global_latent_dim * 2,
            self.width,
            self.width * 2,
        )
        self.latent_projection = nn.Linear(self.width, self.width)
        self.memory_projection = nn.Linear(self.width, self.width)
        self.query_base = nn.Parameter(torch.zeros(1, 1, self.width))
        self.decoder_type_embedding = nn.Parameter(torch.zeros(1, 1, self.width))
        self.decoder = HierarchicalDecoderStack(
            self.decoder_layers,
            self.width,
            self.heads,
            self.ffn_dim,
            self.dropout,
        )
        self.film_heads = nn.ModuleList(
            nn.Linear(self.width, 2 * self.width)
            for _ in range(self.decoder_layers)
        )
        self.action_output = nn.Linear(self.width, ACTION_DIM)
        for head in self.film_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for parameter in (self.posterior_cls, self.condition_cls, self.query_base):
            nn.init.normal_(parameter, std=0.02)
        nn.init.zeros_(self.posterior_logvar.weight)
        nn.init.constant_(self.posterior_logvar.bias, -4.0)

    @classmethod
    def architecture_signature(
        cls, config: dict[str, Any], parameter_count: int | None = None
    ) -> dict[str, Any]:
        signature = {
            "architecture_version": cls.ARCHITECTURE_VERSION,
            "history_steps": HISTORY_STEPS,
            "reference_frames": REFERENCE_FRAMES,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "reference_dim": REFERENCE_DIM,
            "d_model": int(config.get("d_model", 448)),
            "heads": int(config.get("heads", 8)),
            "ffn_dim": int(config.get("ffn_dim", 1792)),
            "posterior_encoder_layers": int(config.get("posterior_encoder_layers", 6)),
            "condition_encoder_layers": int(config.get("condition_encoder_layers", 6)),
            "decoder_layers": int(config.get("decoder_layers", 8)),
            "global_latent_dim": int(config.get("global_latent_dim", 256)),
        }
        if parameter_count is not None:
            signature["parameter_count"] = int(parameter_count)
        return signature

    @staticmethod
    def _check_batch(batch: dict[str, torch.Tensor]) -> None:
        required = ("state_history", "action_history", "reference_future", "target_action")
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"Sonic causal batch is missing {missing}")
        if tuple(batch["state_history"].shape[1:]) != (HISTORY_STEPS, STATE_DIM):
            raise ValueError("state_history must have shape [B,10,70]")
        if tuple(batch["action_history"].shape[1:]) != (HISTORY_STEPS, ACTION_DIM):
            raise ValueError("action_history must have shape [B,10,29]")
        if tuple(batch["reference_future"].shape[1:]) != (REFERENCE_FRAMES, REFERENCE_DIM):
            raise ValueError("reference_future must have shape [B,10,64]")
        if tuple(batch["target_action"].shape[1:]) != (ACTION_DIM,):
            raise ValueError("target_action must have shape [B,29]")
        for key in ("state_valid", "action_valid"):
            if key in batch and tuple(batch[key].shape[1:]) != (HISTORY_STEPS,):
                raise ValueError(f"{key} must have shape [B,10]")
        if "history_validity" in batch and tuple(batch["history_validity"].shape[1:]) != (HISTORY_STEPS, 2):
            raise ValueError("history_validity must have shape [B,10,2]")
        if "reference_time_offsets" in batch and tuple(batch["reference_time_offsets"].shape[1:]) != (REFERENCE_FRAMES,):
            raise ValueError("reference_time_offsets must have shape [B,10]")

    def _streams(
        self, batch: dict[str, torch.Tensor], *, target_visible: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check_batch(batch)
        state = batch["state_history"].float()
        action = batch["action_history"].float()
        reference = batch["reference_future"].float()
        target = batch["target_action"].float()
        batch_size = state.shape[0]
        state_valid = batch.get(
            "state_valid",
            torch.ones(batch_size, HISTORY_STEPS, device=state.device, dtype=torch.bool),
        ).bool()
        action_valid = batch.get(
            "action_valid",
            torch.ones(batch_size, HISTORY_STEPS, device=state.device, dtype=torch.bool),
        ).bool()
        state_mask = (~state_valid)[..., None].expand(-1, -1, STATE_DIM)
        action_mask = (~action_valid)[..., None].expand(-1, -1, ACTION_DIM)
        history_payload = torch.cat(
            (
                state.masked_fill(state_mask, 0.0),
                action.masked_fill(action_mask, 0.0),
            ),
            dim=-1,
        )
        reference_payload = torch.zeros(
            batch_size,
            REFERENCE_FRAMES,
            STATE_DIM + ACTION_DIM,
            device=state.device,
            dtype=state.dtype,
        )
        reference_payload[..., :REFERENCE_DIM] = reference
        query_payload = torch.zeros(
            batch_size,
            1,
            STATE_DIM + ACTION_DIM,
            device=state.device,
            dtype=state.dtype,
        )
        query_payload[..., STATE_DIM:] = target[:, None] if target_visible else 0.0
        values = torch.cat((history_payload, reference_payload, query_payload), dim=1)
        masks = torch.zeros_like(values)
        masks[:, :HISTORY_STEPS, :STATE_DIM] = state_mask.to(values.dtype)
        masks[:, :HISTORY_STEPS, STATE_DIM:] = action_mask.to(values.dtype)
        if not target_visible:
            masks[:, -1, STATE_DIM:] = 1.0
        return values, masks, reference

    def _encode_tokens(
        self,
        batch: dict[str, torch.Tensor],
        *,
        target_visible: bool,
        condition: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values, masks, reference = self._streams(batch, target_visible=target_visible)
        value_encoder = (
            self.condition_history_input
            if condition
            else self.history_input
        )
        reference_encoder = (
            self.condition_reference_input
            if condition
            else self.reference_input
        )
        query_encoder = self.condition_query_input if condition else self.query_input
        history = value_encoder(torch.cat((values[:, :10], masks[:, :10]), dim=-1))
        reference_tokens = reference_encoder(
            torch.cat((values[:, 10:20], masks[:, 10:20]), dim=-1)
        )
        query = query_encoder(torch.cat((values[:, 20:], masks[:, 20:]), dim=-1))
        tokens = torch.cat((history, reference_tokens, query), dim=1)
        positions = torch.arange(self.TOKEN_COUNT, device=tokens.device)
        tokens = tokens + self.time_embedding(positions)[None]
        tokens[:, :10] += self.modality_embedding.weight[0]
        tokens[:, 10:20] += self.modality_embedding.weight[1]
        tokens[:, 20:] += self.modality_embedding.weight[2]
        tokens[:, :10] += self.direction_embedding.weight[0]
        tokens[:, 10:20] += self.direction_embedding.weight[1]
        tokens[:, 20:] += self.direction_embedding.weight[1]
        offsets = batch.get(
            "reference_time_offsets",
            torch.arange(REFERENCE_FRAMES, device=tokens.device, dtype=tokens.dtype)[None]
            .expand(tokens.shape[0], -1)
            * 0.1,
        )
        tokens[:, 10:20] += self.reference_offset_projection(
            offsets.to(tokens.dtype).unsqueeze(-1)
        )
        tokens[:, 10:20] += self.reference_frame_embedding.weight[None]
        valid = torch.ones(
            tokens.shape[0], self.TOKEN_COUNT, device=tokens.device, dtype=torch.bool
        )
        return tokens, valid, positions

    def encode_posterior_distribution(
        self, batch: dict[str, torch.Tensor]
    ) -> SonicGaussianLatent:
        tokens, valid, times = self._encode_tokens(
            batch, target_visible=True, condition=False
        )
        cls = self.posterior_cls.expand(tokens.shape[0], -1, -1)
        encoded = self.posterior_encoder(
            torch.cat((cls, tokens), dim=1),
            torch.cat((torch.ones_like(valid[:, :1]), valid), dim=1),
            torch.cat((times.new_tensor([-1]), times)),
            False,
        )
        pooled = encoded[:, 0]
        return SonicGaussianLatent(
            mean=self.posterior_mean(pooled),
            logvar=self.posterior_logvar(pooled).clamp(-8.0, 4.0),
        )

    def encode_condition(self, batch: dict[str, torch.Tensor]) -> SonicCondition:
        tokens, valid, times = self._encode_tokens(
            batch, target_visible=False, condition=True
        )
        cls = self.condition_cls.expand(tokens.shape[0], -1, -1)
        encoded = self.condition_encoder(
            torch.cat((cls, tokens), dim=1),
            torch.cat((torch.ones_like(valid[:, :1]), valid), dim=1),
            torch.cat((times.new_tensor([-1]), times)),
            False,
        )
        return SonicCondition(memory=encoded[:, 1:], pooled=encoded[:, 0])

    @staticmethod
    def reparameterize(
        distribution: SonicGaussianLatent,
        epsilon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean = distribution.mean.float()
        logvar = distribution.logvar.float()
        if epsilon is None:
            epsilon = torch.randn_like(mean)
        return mean + torch.exp(0.5 * logvar) * epsilon.float()

    def _decode(self, condition: SonicCondition, latent: torch.Tensor) -> torch.Tensor:
        latent_context = self.latent_fusion(
            torch.cat((latent, self.condition_latent(condition.pooled)), dim=-1)
        )
        memory = torch.cat(
            (self.memory_projection(condition.memory), latent_context[:, None]), dim=1
        )
        memory_valid = torch.ones(
            memory.shape[:2], device=memory.device, dtype=torch.bool
        )
        query = self.query_base + self.decoder_type_embedding
        query = query.expand(latent.shape[0], -1, -1)
        query = query + self.latent_projection(latent_context)[:, None]
        query_valid = torch.ones(
            latent.shape[0], 1, device=latent.device, dtype=torch.bool
        )
        times = torch.zeros(1, device=latent.device, dtype=torch.long)
        film = torch.stack(
            [head(latent_context) for head in self.film_heads], dim=1
        )
        scale, shift = film.chunk(2, dim=-1)
        decoded = self.decoder(
            query,
            query_valid,
            times,
            memory,
            memory_valid,
            scale,
            shift,
        )
        return self.action_output(decoded[:, 0])

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        stage: str = "C",
        epsilon: torch.Tensor | None = None,
    ) -> SonicCausalOutput:
        stage = str(stage).upper()
        if stage == "A":
            posterior = self.encode_posterior_distribution(batch)
            condition = self.encode_condition(batch)
            latent = posterior.mean
            source = "posterior_mean"
        elif stage == "B":
            posterior = None
            condition = self.encode_condition(batch)
            latent = torch.zeros(
                batch["state_history"].shape[0],
                self.global_latent_dim,
                device=batch["state_history"].device,
            )
            source = "condition_mean"
        elif stage == "C":
            posterior = self.encode_posterior_distribution(batch)
            condition = self.encode_condition(batch)
            latent = self.reparameterize(posterior, epsilon)
            source = "posterior_sample"
        elif stage in {"INFER", "STANDARD_NORMAL"}:
            posterior = None
            condition = self.encode_condition(batch)
            latent = (
                epsilon
                if epsilon is not None
                else torch.randn(
                    batch["state_history"].shape[0],
                    self.global_latent_dim,
                    device=batch["state_history"].device,
                )
            )
            source = "standard_normal"
        else:
            raise ValueError("stage must be A, B, C, INFER or STANDARD_NORMAL")
        action = self._decode(condition, latent)
        return SonicCausalOutput(action, posterior, latent, source, condition)

    def infer_from_condition(
        self,
        batch: dict[str, torch.Tensor],
        *,
        epsilon: torch.Tensor | None = None,
        sample: bool = True,
    ) -> SonicCausalOutput:
        if not sample:
            epsilon = torch.zeros(
                batch["state_history"].shape[0],
                self.global_latent_dim,
                device=batch["state_history"].device,
            )
        return self.forward(batch, stage="INFER", epsilon=epsilon)


def build_sonic_causal_model(config: dict[str, Any]) -> SonicCausalCVAE:
    return SonicCausalCVAE(config)

