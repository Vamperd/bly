from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .constants import ACTION_DIM, PHYSICAL_STATE_DIM
from .masking import MaskBatch, masked_inputs


@dataclass
class ModelOutput:
    physical_state: torch.Tensor
    previous_action: torch.Tensor
    action: torch.Tensor
    forward_delta: torch.Tensor
    auxiliary_transition: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_logvar: torch.Tensor
    prior_mean: torch.Tensor
    prior_logvar: torch.Tensor
    latent: torch.Tensor
    state_contact_logits: torch.Tensor | None = None
    forward_contact_logits: torch.Tensor | None = None
    inverse_action: torch.Tensor | None = None
    history_action: torch.Tensor | None = None
    rollout_state: torch.Tensor | None = None
    cycle_state: torch.Tensor | None = None
    cycle_action: torch.Tensor | None = None
    inverse_action_log_scale: torch.Tensor | None = None


class MLPTokenizer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def _rotate_half_pair(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even, odd = value[..., 0::2], value[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class RotarySelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % heads or (d_model // heads) % 2:
            raise ValueError("d_model/heads must be an even integer for RoPE")
        self.heads = heads
        self.head_dim = d_model // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        value: torch.Tensor,
        valid_tokens: torch.Tensor,
        times: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        batch, length, width = value.shape
        qkv = self.qkv(value).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, values = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        values = values.transpose(1, 2)
        positions = times.clamp_min(0).to(value.dtype)
        inv_freq = torch.exp(
            torch.arange(0, self.head_dim, 2, device=value.device, dtype=value.dtype)
            * (-math.log(10000.0) / self.head_dim)
        )
        angles = positions[:, None] * inv_freq[None, :]
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]
        query = _rotate_half_pair(query, cos, sin)
        key = _rotate_half_pair(key, cos, sin)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~valid_tokens[:, None, None, :], -torch.inf)
        if causal:
            query_time = times[:, None]
            key_time = times[None, :]
            blocked = (
                (query_time >= 0)
                & (key_time >= 0)
                & (key_time > query_time)
            )
            scores = scores.masked_fill(blocked[None, None], -torch.inf)
        attention = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        attention = self.dropout(attention)
        result = torch.matmul(attention, values).transpose(1, 2).reshape(batch, length, width)
        return self.output(result)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = RotarySelfAttention(d_model, heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        value: torch.Tensor,
        valid_tokens: torch.Tensor,
        times: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        value = value + self.attention(self.norm1(value), valid_tokens, times, causal)
        return value + self.feed_forward(self.norm2(value))


class TransformerStack(nn.Module):
    def __init__(
        self, layers: int, d_model: int, heads: int, ffn_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, heads, ffn_dim, dropout) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        value: torch.Tensor,
        valid_tokens: torch.Tensor,
        times: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value, valid_tokens, times, causal)
        return self.norm(value)


class TransformerCVAE(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        width = int(config["d_model"])
        latent = int(config["latent_dim"])
        dropout = float(config["dropout"])
        self.width = width
        self.latent_dim = latent
        self.state_dim = int(config.get("state_dim", PHYSICAL_STATE_DIM))
        self.include_previous_action = bool(config.get("include_previous_action", True))
        self.robot_info_dim = int(config.get("robot_info_dim", 0))
        self.dynamics_context_dim = int(config.get("dynamics_context_dim", 0))
        self.auxiliary_dim = int(config.get("auxiliary_dim", 0))
        self.context_mode = str(config.get("context_mode", "hidden"))
        self.token_layout = str(config.get("token_layout", "grouped"))
        self.state_tokenizer = MLPTokenizer(self.state_dim * 2 + 1, width)
        self.previous_tokenizer = (
            MLPTokenizer(ACTION_DIM * 2 + 1, width)
            if self.include_previous_action
            else None
        )
        self.action_tokenizer = MLPTokenizer(ACTION_DIM * 2 + 1, width)
        condition_dim = self.robot_info_dim if self.robot_info_dim else ACTION_DIM
        if self.context_mode == "explicit":
            condition_dim += self.dynamics_context_dim
        self.scale_tokenizer = MLPTokenizer(condition_dim, width)
        self.task_base = nn.Parameter(torch.zeros(1, 1, width))
        self.task_embedding = nn.Embedding(3, width)
        self.type_embedding = nn.Embedding(5, width)
        self.encoder = TransformerStack(
            int(config["encoder_layers"]),
            width,
            int(config["heads"]),
            int(config["ffn_dim"]),
            dropout,
        )
        self.decoder = TransformerStack(
            int(config["decoder_layers"]),
            width,
            int(config["heads"]),
            int(config["ffn_dim"]),
            dropout,
        )
        self.posterior = nn.Linear(width, latent * 2)
        self.prior = nn.Linear(width, latent * 2)
        self.latent_projection = nn.Linear(latent, width)
        self.state_head = nn.Linear(width, self.state_dim)
        self.previous_head = (
            nn.Linear(width, ACTION_DIM) if self.include_previous_action else None
        )
        self.action_head = nn.Linear(width, ACTION_DIM)
        self.forward_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, self.state_dim)
        )
        self.auxiliary_head = (
            nn.Sequential(
                nn.Linear(width, width), nn.GELU(), nn.Linear(width, self.auxiliary_dim)
            )
            if self.auxiliary_dim
            else None
        )

    def _tokenize(
        self, batch: dict[str, torch.Tensor], masks: MaskBatch, full: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[Any, Any, Any]]:
        if self.token_layout == "interleaved":
            return self._tokenize_interleaved(batch, masks, full)
        inputs = masked_inputs(batch, masks, full=full)
        state = inputs["physical_state"]
        previous = inputs["previous_action"]
        action = inputs["action"]
        batch_size, state_steps, _ = state.shape
        action_padded = F.pad(action, (0, 0, 0, 1))
        action_mask = F.pad(masks.action_input, (0, 0, 0, 1))
        action_progress = torch.cat(
            (batch["progress"][:, 1:], torch.zeros_like(batch["progress"][:, :1])), dim=1
        )
        state_token = self.state_tokenizer(
            torch.cat((state, masks.state_input.to(state.dtype), batch["progress"][..., None]), -1)
        )
        previous_token = self.previous_tokenizer(
            torch.cat(
                (previous, masks.previous_input.to(previous.dtype), batch["progress"][..., None]),
                -1,
            )
        )
        action_token = self.action_tokenizer(
            torch.cat((action_padded, action_mask.to(action.dtype), action_progress[..., None]), -1)
        )
        task_ids = torch.full(
            (batch_size,), masks.task_id, dtype=torch.long, device=state.device
        )
        task_token = self.task_base.expand(batch_size, -1, -1) + self.task_embedding(
            task_ids
        )[:, None]
        scale_token = self.scale_tokenizer(batch["action_scale"])[:, None]
        tokens = torch.cat((task_token, scale_token, state_token, previous_token, action_token), 1)
        type_ids = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=state.device),
                torch.ones(1, dtype=torch.long, device=state.device),
                torch.full((state_steps,), 2, dtype=torch.long, device=state.device),
                torch.full((state_steps,), 3, dtype=torch.long, device=state.device),
                torch.full((state_steps,), 4, dtype=torch.long, device=state.device),
            )
        )
        tokens = tokens + self.type_embedding(type_ids)[None]
        valid_action = F.pad(batch["valid_action"], (0, 1), value=False)
        valid = torch.cat(
            (
                torch.ones((batch_size, 2), dtype=torch.bool, device=state.device),
                batch["valid_state"],
                batch["valid_state"],
                valid_action,
            ),
            1,
        )
        times = torch.cat(
            (
                torch.full((2,), -1, dtype=torch.long, device=state.device),
                torch.arange(state_steps, device=state.device),
                torch.arange(state_steps, device=state.device),
                torch.arange(1, state_steps + 1, device=state.device),
            )
        )
        state_slice = slice(2, 2 + state_steps)
        previous_slice = slice(state_slice.stop, state_slice.stop + state_steps)
        action_slice = slice(previous_slice.stop, previous_slice.stop + state_steps)
        return tokens, valid, times, (state_slice, previous_slice, action_slice)

    def _tokenize_interleaved(
        self, batch: dict[str, torch.Tensor], masks: MaskBatch, full: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[Any, Any, Any]]:
        inputs = masked_inputs(batch, masks, full=full)
        state = inputs["physical_state"]
        action = inputs["action"]
        batch_size, state_steps, _ = state.shape
        action_padded = F.pad(action, (0, 0, 0, 1))
        action_mask = F.pad(masks.action_input, (0, 0, 0, 1))
        state_token = self.state_tokenizer(
            torch.cat(
                (state, masks.state_input.to(state.dtype), batch["progress"][..., None]), -1
            )
        )
        action_progress = torch.cat(
            (batch["progress"][:, 1:], torch.zeros_like(batch["progress"][:, :1])), dim=1
        )
        action_token = self.action_tokenizer(
            torch.cat((action_padded, action_mask.to(action.dtype), action_progress[..., None]), -1)
        )
        task_ids = torch.full(
            (batch_size,), masks.task_id, dtype=torch.long, device=state.device
        )
        task_token = self.task_base.expand(batch_size, -1, -1) + self.task_embedding(task_ids)[:, None]
        condition = batch["robot_information"] if self.robot_info_dim else batch["action_scale"]
        if self.context_mode == "explicit":
            condition = torch.cat((condition, batch["dynamics_context"]), dim=-1)
        robot_token = self.scale_tokenizer(condition)[:, None]
        interleaved = torch.stack((state_token, action_token), dim=2).flatten(1, 2)[:, :-1]
        tokens = torch.cat((task_token, robot_token, interleaved), dim=1)
        pair_types = torch.stack(
            (
                torch.full((state_steps,), 2, dtype=torch.long, device=state.device),
                torch.full((state_steps,), 4, dtype=torch.long, device=state.device),
            ),
            dim=1,
        ).flatten()[:-1]
        type_ids = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=state.device),
                torch.ones(1, dtype=torch.long, device=state.device),
                pair_types,
            )
        )
        tokens = tokens + self.type_embedding(type_ids)[None]
        valid_action = F.pad(batch["valid_action"], (0, 1), value=False)
        valid_interleaved = torch.stack(
            (batch["valid_state"], valid_action), dim=2
        ).flatten(1, 2)[:, :-1]
        valid = torch.cat(
            (
                torch.ones((batch_size, 2), dtype=torch.bool, device=state.device),
                valid_interleaved,
            ),
            dim=1,
        )
        state_times = torch.arange(state_steps, device=state.device)
        action_times = state_times + 1
        pair_times = torch.stack((state_times, action_times), dim=1).flatten()[:-1]
        times = torch.cat(
            (torch.full((2,), -1, dtype=torch.long, device=state.device), pair_times)
        )
        state_indices = 2 + torch.arange(0, 2 * state_steps - 1, 2, device=state.device)
        action_indices = 2 + torch.arange(1, 2 * state_steps - 1, 2, device=state.device)
        return tokens, valid, times, (state_indices, None, action_indices)

    @staticmethod
    def _distribution(parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = parameters.chunk(2, dim=-1)
        return mean, logvar.clamp(-12.0, 8.0)

    @staticmethod
    def _sample(mean: torch.Tensor, logvar: torch.Tensor, deterministic: bool) -> torch.Tensor:
        return mean if deterministic else mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    def _latent_summary(
        self, encoded: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        if self.token_layout != "interleaved":
            return encoded[:, 0]
        # In a causal forward task the time=-1 task token cannot attend to any
        # sequence token. Pool valid S/A tokens so p(z|visible) and q(z|full)
        # remain conditional on the actual trajectory rather than a constant.
        data_valid = valid[:, 2:].to(encoded.dtype)
        return (
            encoded[:, 2:] * data_valid[:, :, None]
        ).sum(dim=1) / data_valid.sum(dim=1, keepdim=True).clamp_min(1.0)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        masks: MaskBatch,
        sample_from_prior: bool = False,
        deterministic: bool = False,
    ) -> ModelOutput:
        visible, valid, times, slices = self._tokenize(batch, masks, full=False)
        full, _, _, _ = self._tokenize(batch, masks, full=True)
        prior_encoded = self.encoder(visible, valid, times, masks.causal)
        posterior_encoded = self.encoder(full, valid, times, masks.causal)
        prior_summary = self._latent_summary(prior_encoded, valid)
        posterior_summary = self._latent_summary(posterior_encoded, valid)
        prior_mean, prior_logvar = self._distribution(self.prior(prior_summary))
        posterior_mean, posterior_logvar = self._distribution(
            self.posterior(posterior_summary)
        )
        latent = self._sample(
            prior_mean if sample_from_prior else posterior_mean,
            prior_logvar if sample_from_prior else posterior_logvar,
            deterministic,
        )
        decoded = self.decoder(
            visible + self.latent_projection(latent)[:, None], valid, times, masks.causal
        )
        state_slice, previous_slice, action_slice = slices
        state_hidden = decoded[:, state_slice]
        previous_hidden = decoded[:, previous_slice] if previous_slice is not None else None
        action_hidden = decoded[:, action_slice]
        previous_output = (
            self.previous_head(previous_hidden)
            if self.previous_head is not None and previous_hidden is not None
            else state_hidden.new_empty(state_hidden.shape[0], state_hidden.shape[1], 0)
        )
        action_output = self.action_head(action_hidden)
        if self.token_layout != "interleaved":
            action_output = action_output[:, :-1]
        auxiliary_output = (
            self.auxiliary_head(state_hidden[:, 1:])
            if self.auxiliary_head is not None
            else state_hidden.new_empty(state_hidden.shape[0], state_hidden.shape[1] - 1, 0)
        )
        return ModelOutput(
            self.state_head(state_hidden),
            previous_output,
            action_output,
            self.forward_head(state_hidden[:, 1:]),
            auxiliary_output,
            posterior_mean,
            posterior_logvar,
            prior_mean,
            prior_logvar,
            latent,
        )


class PhysicsTransformerCVAE(nn.Module):
    """Joint-aware CVAE for the unique Physics v3 S/A sequence.

    Temporal relation heads are deliberately formed from S_t and A_t (or from
    S_t and S_{t+1} for inverse dynamics).  They never use the target token that
    they are supposed to predict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.width = int(config["d_model"])
        self.latent_dim = int(config["latent_dim"])
        self.auxiliary_dim = int(config.get("auxiliary_dim", 35))
        self.context_mode = str(config.get("context_mode", "hidden"))
        self.robot_conditioning = str(config.get("robot_conditioning", "full"))
        if self.robot_conditioning not in {"full", "joint_id_only"}:
            raise ValueError(
                f"unsupported robot_conditioning {self.robot_conditioning!r}"
            )
        self.dynamics_context_dim = int(config.get("dynamics_context_dim", 0))
        joint_width = int(config.get("joint_width", 128))
        dropout = float(config["dropout"])
        actuator_types = int(config.get("actuator_type_count", 1))

        self.joint_id = nn.Embedding(ACTION_DIM, joint_width)
        self.actuator_type = (
            nn.Embedding(max(1, actuator_types), joint_width)
            if self.robot_conditioning == "full" else None
        )
        self.robot_joint = (
            MLPTokenizer(int(config.get("joint_robot_info_dim", 11)), joint_width)
            if self.robot_conditioning == "full" else None
        )
        robot_layer = nn.TransformerEncoderLayer(
            d_model=joint_width,
            nhead=4,
            dim_feedforward=joint_width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.robot_encoder = nn.TransformerEncoder(robot_layer, num_layers=2)
        self.global_robot = (
            MLPTokenizer(int(config.get("global_robot_info_dim", 9)), self.width)
            if self.robot_conditioning == "full" else None
        )
        self.robot_pool = nn.Linear(joint_width, self.width)
        self.oracle_context = (
            MLPTokenizer(self.dynamics_context_dim, self.width)
            if self.context_mode in {"oracle", "explicit"} and self.dynamics_context_dim
            else None
        )

        self.state_joint = MLPTokenizer(joint_width + 4, joint_width)
        self.action_joint = MLPTokenizer(joint_width + 2, joint_width)
        self.state_joint_pool = nn.Linear(joint_width, self.width)
        self.action_joint_pool = nn.Linear(joint_width, self.width)
        self.state_base = MLPTokenizer(12 * 2 + 1, self.width)
        self.state_fusion = MLPTokenizer(self.width * 2, self.width)
        self.action_fusion = MLPTokenizer(self.width + 1, self.width)
        self.before_fusion = MLPTokenizer(self.width, self.width)
        self.type_embedding = nn.Embedding(3, self.width)
        self.mask_group_embedding = nn.Embedding(3, self.width)

        self.encoder = TransformerStack(
            int(config["encoder_layers"]), self.width, int(config["heads"]),
            int(config["ffn_dim"]), dropout
        )
        self.decoder = TransformerStack(
            int(config["decoder_layers"]), self.width, int(config["heads"]),
            int(config["ffn_dim"]), dropout
        )
        self.prior = nn.Linear(self.width, self.latent_dim * 2)
        self.posterior = nn.Linear(self.width, self.latent_dim * 2)
        self.latent_projection = nn.Linear(self.latent_dim, self.width)

        self.state_joint_decoder = MLPTokenizer(self.width + joint_width, joint_width)
        self.state_joint_output = nn.Linear(joint_width, 2)
        self.state_base_output = nn.Linear(self.width, 10)
        self.state_contact_output = nn.Linear(self.width, 2)
        self.action_joint_decoder = MLPTokenizer(self.width + joint_width, joint_width)
        self.action_joint_output = nn.Linear(joint_width, 1)

        relation_width = self.width * 2
        self.forward_relation = MLPTokenizer(relation_width, self.width)
        self.inverse_relation = MLPTokenizer(relation_width, self.width)
        self.history_relation = MLPTokenizer(self.width, self.width)
        self.relation_prior = nn.Linear(self.width, self.latent_dim * 2)
        self.forward_continuous = nn.Linear(self.width, 68)
        self.forward_contact = nn.Linear(self.width, 2)
        self.auxiliary_head = (
            nn.Linear(self.width, self.auxiliary_dim) if self.auxiliary_dim else None
        )

    @staticmethod
    def _distribution(parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = parameters.chunk(2, dim=-1)
        return mean, logvar.clamp(-12.0, 8.0)

    @staticmethod
    def _sample(mean: torch.Tensor, logvar: torch.Tensor, deterministic: bool) -> torch.Tensor:
        return mean if deterministic else mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    def _local_relation_condition(
        self, relation: torch.Tensor, robot_summary: torch.Tensor
    ) -> torch.Tensor:
        """Infer target-safe local dynamics context from a permitted relation input."""
        mean, _ = self._distribution(self.relation_prior(relation))
        while robot_summary.ndim < relation.ndim:
            robot_summary = robot_summary[:, None]
        return robot_summary + self.latent_projection(mean)

    def _robot_memory(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        joint = batch["joint_robot_information"]
        ids = torch.arange(ACTION_DIM, device=joint.device)
        memory = self.joint_id(ids)[None].expand(joint.shape[0], -1, -1)
        if self.robot_conditioning == "full":
            assert self.robot_joint is not None
            assert self.actuator_type is not None
            actuator = batch["joint_actuator_type"].long().clamp(
                0, self.actuator_type.num_embeddings - 1
            )
            memory = memory + self.robot_joint(joint) + self.actuator_type(actuator)
        memory = self.robot_encoder(memory)
        summary = self.robot_pool(memory.mean(dim=1))
        if self.global_robot is not None:
            summary = summary + self.global_robot(batch["global_robot_information"])
        if self.oracle_context is not None:
            summary = summary + self.oracle_context(batch["dynamics_context"])
        return memory, summary

    def _state_embed(
        self,
        state: torch.Tensor,
        mask: torch.Tensor,
        progress: torch.Tensor,
        robot_memory: torch.Tensor,
    ) -> torch.Tensor:
        qdq = torch.stack((state[..., :29], state[..., 29:58]), dim=-1)
        qdq_mask = torch.stack((mask[..., :29], mask[..., 29:58]), dim=-1).to(state.dtype)
        memory = robot_memory[:, None].expand(-1, state.shape[1], -1, -1)
        joint = self.state_joint(torch.cat((qdq, qdq_mask, memory), dim=-1)).mean(dim=2)
        base = self.state_base(
            torch.cat(
                (state[..., 58:], mask[..., 58:].to(state.dtype), progress[..., None]), dim=-1
            )
        )
        return self.state_fusion(torch.cat((self.state_joint_pool(joint), base), dim=-1))

    def _action_embed(
        self,
        action: torch.Tensor,
        mask: torch.Tensor,
        progress: torch.Tensor,
        robot_memory: torch.Tensor,
    ) -> torch.Tensor:
        memory = robot_memory[:, None].expand(-1, action.shape[1], -1, -1)
        joint = self.action_joint(
            torch.cat((action[..., None], mask.to(action.dtype)[..., None], memory), dim=-1)
        ).mean(dim=2)
        return self.action_fusion(
            torch.cat((self.action_joint_pool(joint), progress[..., None]), dim=-1)
        )

    def _tokenize(
        self,
        batch: dict[str, torch.Tensor],
        masks: MaskBatch,
        full: bool,
        robot_memory: torch.Tensor,
        robot_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = masked_inputs(batch, masks, full=full)
        state = inputs["physical_state"]
        action = inputs["action"]
        state_mask = torch.zeros_like(masks.state_input) if full else masks.state_input
        action_mask = torch.zeros_like(masks.action_input) if full else masks.action_input
        state_token = self._state_embed(state, state_mask, batch["progress"], robot_memory)
        action_progress = batch["progress"][:, :-1]
        action_token = self._action_embed(action, action_mask, action_progress, robot_memory)
        before_mask = torch.zeros_like(batch["action_before_window"], dtype=torch.bool)
        before = self._action_embed(
            batch["action_before_window"][:, None],
            before_mask[:, None],
            torch.zeros_like(batch["progress"][:, :1]),
            robot_memory,
        )
        interleaved = torch.stack((state_token[:, :-1], action_token), dim=2).flatten(1, 2)
        tokens = torch.cat((self.before_fusion(before), interleaved, state_token[:, -1:]), dim=1)
        type_ids = torch.empty(tokens.shape[1], dtype=torch.long, device=tokens.device)
        type_ids[0] = 0
        type_ids[1::2] = 1
        type_ids[2::2] = 2
        tokens = tokens + self.type_embedding(type_ids)[None]
        tokens = tokens + robot_summary[:, None] + self.mask_group_embedding(
            torch.full((tokens.shape[0],), masks.task_id, device=tokens.device, dtype=torch.long)
        )[:, None]
        valid_interleaved = torch.stack(
            (batch["valid_state"][:, :-1], batch["valid_action"]), dim=2
        ).flatten(1, 2)
        valid = torch.cat(
            (
                torch.ones((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device),
                valid_interleaved,
                batch["valid_state"][:, -1:],
            ), dim=1
        )
        times = torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.long)
        state_indices = torch.arange(1, tokens.shape[1], 2, device=tokens.device)
        action_indices = torch.arange(2, tokens.shape[1] - 1, 2, device=tokens.device)
        return tokens, valid, times, state_indices, action_indices

    def _decode_state(
        self, hidden: torch.Tensor, robot_memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = robot_memory[:, None].expand(-1, hidden.shape[1], -1, -1)
        query = hidden[:, :, None].expand(-1, -1, ACTION_DIM, -1)
        joint = self.state_joint_output(
            self.state_joint_decoder(torch.cat((query, memory), dim=-1))
        )
        continuous_base_raw = self.state_base_output(hidden)
        continuous_base = torch.cat(
            (
                continuous_base_raw[..., :6],
                F.normalize(continuous_base_raw[..., 6:9], dim=-1, eps=1e-6),
                continuous_base_raw[..., 9:],
            ), dim=-1
        )
        contact_logits = self.state_contact_output(hidden)
        state = torch.cat(
            (joint[..., 0], joint[..., 1], continuous_base, torch.sigmoid(contact_logits)), dim=-1
        )
        return state, contact_logits

    def _decode_action(self, hidden: torch.Tensor, robot_memory: torch.Tensor) -> torch.Tensor:
        memory = robot_memory[:, None].expand(-1, hidden.shape[1], -1, -1)
        query = hidden[:, :, None].expand(-1, -1, ACTION_DIM, -1)
        return self.action_joint_output(
            self.action_joint_decoder(torch.cat((query, memory), dim=-1))
        ).squeeze(-1)

    def _forward_values(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        progress: torch.Tensor,
        robot_memory: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_hidden = self._state_embed(
            state[:, None], torch.zeros_like(state[:, None], dtype=torch.bool),
            progress[:, None], robot_memory
        )[:, 0]
        action_hidden = self._action_embed(
            action[:, None], torch.zeros_like(action[:, None], dtype=torch.bool),
            progress[:, None], robot_memory
        )[:, 0]
        relation = self.forward_relation(torch.cat((state_hidden, action_hidden), dim=-1)) + condition
        raw_delta = self.forward_continuous(relation)
        contact_logits = self.forward_contact(relation)
        continuous_next = state[:, :68] + raw_delta
        continuous_next = torch.cat(
            (
                continuous_next[:, :64],
                F.normalize(continuous_next[:, 64:67], dim=-1, eps=1e-6),
                continuous_next[:, 67:68],
            ), dim=-1
        )
        delta = continuous_next - state[:, :68]
        next_state = torch.cat(
            (continuous_next, torch.sigmoid(contact_logits)), dim=-1
        )
        return next_state, delta, contact_logits

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        masks: MaskBatch,
        sample_from_prior: bool = False,
        deterministic: bool = False,
    ) -> ModelOutput:
        robot_memory, robot_summary = self._robot_memory(batch)
        visible, valid, times, state_indices, action_indices = self._tokenize(
            batch, masks, False, robot_memory, robot_summary
        )
        full, _, _, _, _ = self._tokenize(batch, masks, True, robot_memory, robot_summary)
        prior_encoded = self.encoder(visible, valid, times, masks.causal)
        posterior_encoded = self.encoder(full, valid, times, masks.causal)
        weights = valid.to(visible.dtype)
        prior_pool = (prior_encoded * weights[:, :, None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
        posterior_pool = (posterior_encoded * weights[:, :, None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
        prior_mean, prior_logvar = self._distribution(self.prior(prior_pool))
        posterior_mean, posterior_logvar = self._distribution(self.posterior(posterior_pool))
        mean = prior_mean if sample_from_prior else posterior_mean
        logvar = prior_logvar if sample_from_prior else posterior_logvar
        latent = self._sample(mean, logvar, deterministic)
        condition = robot_summary + self.latent_projection(latent)
        decoded = self.decoder(visible + condition[:, None], valid, times, masks.causal)
        state_hidden = decoded[:, state_indices]
        action_hidden = decoded[:, action_indices]
        relation_state_hidden = prior_encoded[:, state_indices]
        relation_action_hidden = prior_encoded[:, action_indices]
        state_output, state_contact_logits = self._decode_state(state_hidden, robot_memory)
        action_output = self._decode_action(action_hidden, robot_memory)

        forward_relation = self.forward_relation(
            torch.cat((relation_state_hidden[:, :-1], relation_action_hidden), dim=-1)
        )
        forward_condition = self._local_relation_condition(
            forward_relation, robot_summary
        )
        transition = forward_relation + forward_condition
        raw_continuous_delta = self.forward_continuous(transition)
        current_continuous = batch["physical_state"][:, :-1, :68]
        predicted_continuous = current_continuous + raw_continuous_delta
        predicted_continuous = torch.cat(
            (
                predicted_continuous[..., :64],
                F.normalize(predicted_continuous[..., 64:67], dim=-1, eps=1e-6),
                predicted_continuous[..., 67:68],
            ), dim=-1
        )
        continuous_delta = predicted_continuous - current_continuous
        forward_contact_logits = self.forward_contact(transition)
        contact_delta = torch.sigmoid(forward_contact_logits) - batch["physical_state"][:, :-1, 68:]
        forward_delta = torch.cat((continuous_delta, contact_delta), dim=-1)
        auxiliary = (
            self.auxiliary_head(transition)
            if self.auxiliary_head is not None
            else transition.new_empty(transition.shape[0], transition.shape[1], 0)
        )

        visible_inputs = masked_inputs(batch, masks, full=False)
        direct_state_hidden = self._state_embed(
            visible_inputs["physical_state"], masks.state_input,
            batch["progress"], robot_memory
        )
        inverse_relation = self.inverse_relation(
            torch.cat((direct_state_hidden[:, :-1], direct_state_hidden[:, 1:]), dim=-1)
        )
        inverse_condition = self._local_relation_condition(
            inverse_relation, robot_summary
        )
        inverse_hidden = inverse_relation + inverse_condition
        inverse_action = self._decode_action(inverse_hidden, robot_memory)
        history_relation = self.history_relation(relation_state_hidden[:, :-1])
        history_condition = self._local_relation_condition(
            history_relation, robot_summary
        )
        history_action = self._decode_action(
            history_relation + history_condition, robot_memory
        )

        cycle_state = None
        if masks.inverse_transition is not None and bool(masks.inverse_transition.any()):
            cycle_next, _, _ = self._forward_values(
                batch["physical_state"][:, :-1].reshape(-1, 70),
                inverse_action.reshape(-1, ACTION_DIM),
                batch["progress"][:, :-1].reshape(-1),
                robot_memory[:, None].expand(-1, inverse_action.shape[1], -1, -1).reshape(
                    -1, ACTION_DIM, robot_memory.shape[-1]
                ),
                inverse_condition.reshape(-1, self.width),
            )
            cycle_state = cycle_next.reshape(batch["physical_state"].shape[0], -1, 70)
        cycle_action = None
        if masks.forward_transition is not None and bool(masks.forward_transition.any()):
            predicted_next = batch["physical_state"][:, :-1] + forward_delta
            cycle_state_hidden = self._state_embed(
                predicted_next,
                torch.zeros_like(predicted_next, dtype=torch.bool),
                batch["progress"][:, 1:],
                robot_memory,
            )
            cycle_inverse_relation = self.inverse_relation(
                torch.cat((direct_state_hidden[:, :-1], cycle_state_hidden), dim=-1)
            )
            cycle_inverse_hidden = cycle_inverse_relation + self._local_relation_condition(
                cycle_inverse_relation, robot_summary
            )
            cycle_action = self._decode_action(cycle_inverse_hidden, robot_memory)

        if masks.rollout_horizon is not None and bool((masks.rollout_horizon > 0).any()):
            max_rollout = 8
            current = torch.stack(
                [batch["physical_state"][i, int(masks.rollout_start[i])] for i in range(batch["physical_state"].shape[0])]
            )
            rollout_values = []
            for offset in range(max_rollout):
                action_at = torch.stack(
                    [
                        batch["action"][i, min(int(masks.rollout_start[i]) + offset, batch["action"].shape[1] - 1)]
                        for i in range(batch["action"].shape[0])
                    ]
                )
                progress_at = torch.stack(
                    [
                        batch["progress"][i, min(int(masks.rollout_start[i]) + offset, batch["progress"].shape[1] - 1)]
                        for i in range(batch["progress"].shape[0])
                    ]
                )
                condition_at = torch.stack(
                    [
                        forward_condition[
                            i,
                            min(
                                int(masks.rollout_start[i]) + offset,
                                forward_condition.shape[1] - 1,
                            ),
                        ]
                        for i in range(forward_condition.shape[0])
                    ]
                )
                current, _, _ = self._forward_values(
                    current, action_at, progress_at, robot_memory, condition_at
                )
                rollout_values.append(current)
            rollout_state = torch.stack(rollout_values, dim=1)
        else:
            rollout_state = state_output.new_empty(state_output.shape[0], 0, 70)
        return ModelOutput(
            state_output,
            state_output.new_empty(state_output.shape[0], state_output.shape[1], 0),
            action_output,
            forward_delta,
            auxiliary,
            posterior_mean,
            posterior_logvar,
            prior_mean,
            prior_logvar,
            latent,
            state_contact_logits,
            forward_contact_logits,
            inverse_action,
            history_action,
            rollout_state,
            cycle_state,
            cycle_action,
        )


class JointAttentionPool(nn.Module):
    """Learn a joint-sensitive summary instead of averaging 29 joint tokens."""

    def __init__(self, width: int, heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(value.shape[0], -1, -1)
        pooled, _ = self.attention(query, value, value, need_weights=False)
        return self.norm(pooled[:, 0])


class LeanSplitPhysicsCVAE(nn.Module):
    """Deployable split model: causal dynamics, reference Action, bidirectional completion."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.width = int(config.get("d_model", 192))
        self.latent_dim = int(config.get("latent_dim", 64))
        self.auxiliary_dim = int(config.get("auxiliary_dim", 35))
        self.history_steps = int(config.get("history_steps", 10))
        if self.history_steps < 1:
            raise ValueError("history_steps must be positive")
        self.context_mode = str(config.get("context_mode", "hidden"))
        if self.context_mode != "hidden":
            raise ValueError("LeanSplit forbids oracle/explicit dynamics context")
        self.reference_conditioning = str(config.get("reference_conditioning", "off"))
        if self.reference_conditioning not in {"off", "required"}:
            raise ValueError("reference_conditioning must be off or required")
        if bool(config.get("forward_reference_conditioning", False)):
            raise ValueError("LeanSplit forward dynamics cannot consume reference")
        self.action_queue_conditioning = bool(
            config.get("action_queue_conditioning", True)
        )
        self.causal_dynamics_embedding = bool(
            config.get("causal_dynamics_embedding", True)
        )
        self.robot_conditioning = str(config.get("robot_conditioning", "full"))
        if self.robot_conditioning not in {"full", "joint_id_only"}:
            raise ValueError(f"unsupported robot_conditioning {self.robot_conditioning!r}")

        dropout = float(config.get("dropout", 0.0))
        heads = int(config.get("heads", 6))
        ffn = int(config.get("ffn_dim", self.width * 4))
        joint_width = int(config.get("joint_width", 96))
        actuator_types = int(config.get("actuator_type_count", 1))
        self.joint_id = nn.Embedding(ACTION_DIM, joint_width)
        self.actuator_type = (
            nn.Embedding(max(1, actuator_types), joint_width)
            if self.robot_conditioning == "full" else None
        )
        self.robot_joint = (
            MLPTokenizer(int(config.get("joint_robot_info_dim", 11)), joint_width)
            if self.robot_conditioning == "full" else None
        )
        robot_layer = nn.TransformerEncoderLayer(
            joint_width, 4, joint_width * 4, dropout, "gelu",
            batch_first=True, norm_first=True,
        )
        self.robot_encoder = nn.TransformerEncoder(robot_layer, num_layers=2)
        self.robot_pool = nn.Linear(joint_width, self.width)
        self.global_robot = (
            MLPTokenizer(int(config.get("global_robot_info_dim", 9)), self.width)
            if self.robot_conditioning == "full" else None
        )

        self.state_joint = MLPTokenizer(4, joint_width)
        self.action_joint = MLPTokenizer(2, joint_width)
        spatial_layer = nn.TransformerEncoderLayer(
            joint_width, 4, joint_width * 4, dropout, "gelu",
            batch_first=True, norm_first=True,
        )
        self.joint_spatial = nn.TransformerEncoder(
            spatial_layer, num_layers=int(config.get("joint_spatial_layers", 2))
        )
        self.state_pool = JointAttentionPool(joint_width)
        self.action_pool = JointAttentionPool(joint_width)
        self.state_base = MLPTokenizer(25, self.width)
        self.state_fusion = MLPTokenizer(self.width + joint_width, self.width)
        self.action_fusion = MLPTokenizer(self.width + joint_width + 1, self.width)
        self.transition_fusion = MLPTokenizer(self.width * 2, self.width)
        self.before_fusion = MLPTokenizer(self.width, self.width)
        self.type_embedding = nn.Embedding(3, self.width)
        self.task_embedding = nn.Embedding(3, self.width)

        self.completion_encoder = TransformerStack(
            int(config.get("completion_encoder_layers", 2)),
            self.width, heads, ffn, dropout,
        )
        self.completion_decoder = TransformerStack(
            int(config.get("completion_decoder_layers", 2)),
            self.width, heads, ffn, dropout,
        )
        self.history_encoder = TransformerStack(
            int(config.get("history_layers", 4)),
            self.width, heads, ffn, dropout,
        )
        self.dynamics_identifier = MLPTokenizer(self.width, self.width)
        self.reference_input = MLPTokenizer(int(config.get("reference_dim", 64)), self.width)
        self.reference_position = nn.Embedding(int(config.get("reference_frames", 10)), self.width)
        self.reference_encoder = TransformerStack(
            int(config.get("reference_layers", 2)),
            self.width, heads, ffn, dropout,
        )
        self.prior = nn.Linear(self.width, self.latent_dim * 2)
        self.posterior = nn.Linear(self.width, self.latent_dim * 2)
        self.latent_projection = nn.Linear(self.latent_dim, self.width)

        self.state_joint_decoder = MLPTokenizer(self.width + joint_width, joint_width)
        self.state_joint_output = nn.Linear(joint_width, 2)
        self.state_base_output = nn.Linear(self.width, 10)
        self.state_contact_output = nn.Linear(self.width, 2)
        self.action_joint_decoder = MLPTokenizer(self.width + joint_width, joint_width)
        self.action_joint_output = nn.Linear(joint_width, 1)
        self.inverse_log_scale_output = nn.Linear(joint_width, 1)
        self.forward_relation = MLPTokenizer(self.width * 3, self.width)
        self.inverse_relation = MLPTokenizer(self.width * 4, self.width)
        self.history_relation = MLPTokenizer(self.width * 2, self.width)
        self.forward_continuous = nn.Linear(self.width, 68)
        self.forward_contact = nn.Linear(self.width, 2)
        self.auxiliary_head = (
            nn.Linear(self.width, self.auxiliary_dim) if self.auxiliary_dim else None
        )

    @staticmethod
    def _distribution(parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = parameters.chunk(2, dim=-1)
        return mean, logvar.clamp(-12.0, 8.0)

    def _robot_memory(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        joint = batch["joint_robot_information"]
        ids = torch.arange(ACTION_DIM, device=joint.device)
        memory = self.joint_id(ids)[None].expand(joint.shape[0], -1, -1)
        if self.robot_conditioning == "full":
            assert self.robot_joint is not None and self.actuator_type is not None
            actuator = batch["joint_actuator_type"].long().clamp(
                0, self.actuator_type.num_embeddings - 1
            )
            memory = memory + self.robot_joint(joint) + self.actuator_type(actuator)
        memory = self.robot_encoder(memory)
        summary = self.robot_pool(memory.mean(dim=1))
        if self.global_robot is not None:
            summary = summary + self.global_robot(batch["global_robot_information"])
        return memory, summary

    def _state_embed(
        self,
        state: torch.Tensor,
        mask: torch.Tensor,
        progress: torch.Tensor,
        robot_memory: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps = state.shape[:2]
        qdq = torch.stack((state[..., :29], state[..., 29:58]), dim=-1)
        qdq_mask = torch.stack((mask[..., :29], mask[..., 29:58]), dim=-1).to(state.dtype)
        memory = robot_memory[:, None].expand(-1, steps, -1, -1)
        joint = self.state_joint(torch.cat((qdq, qdq_mask), dim=-1)) + memory
        joint = self.joint_spatial(joint.reshape(batch * steps, ACTION_DIM, -1))
        pooled = self.state_pool(joint).reshape(batch, steps, -1)
        base = self.state_base(
            torch.cat((state[..., 58:], mask[..., 58:].to(state.dtype), progress[..., None]), dim=-1)
        )
        return self.state_fusion(torch.cat((pooled, base), dim=-1))

    def _action_embed(
        self,
        action: torch.Tensor,
        mask: torch.Tensor,
        progress: torch.Tensor,
        robot_memory: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps = action.shape[:2]
        memory = robot_memory[:, None].expand(-1, steps, -1, -1)
        joint = self.action_joint(
            torch.cat((action[..., None], mask.to(action.dtype)[..., None]), dim=-1)
        ) + memory
        joint = self.joint_spatial(joint.reshape(batch * steps, ACTION_DIM, -1))
        pooled = self.action_pool(joint).reshape(batch, steps, -1)
        summary = self.robot_pool(robot_memory.mean(dim=1))[:, None].expand(-1, steps, -1)
        return self.action_fusion(torch.cat((pooled, summary, progress[..., None]), dim=-1))

    def _tokenize(
        self,
        batch: dict[str, torch.Tensor],
        masks: MaskBatch,
        full: bool,
        robot_memory: torch.Tensor,
        robot_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = masked_inputs(batch, masks, full=full)
        state_mask = torch.zeros_like(masks.state_input) if full else masks.state_input
        action_mask = torch.zeros_like(masks.action_input) if full else masks.action_input
        state = self._state_embed(
            inputs["physical_state"], state_mask, batch["progress"], robot_memory
        )
        action = self._action_embed(
            inputs["action"], action_mask, batch["progress"][:, :-1], robot_memory
        )
        before = self._action_embed(
            batch["action_before_window"][:, None],
            torch.zeros_like(batch["action_before_window"][:, None], dtype=torch.bool),
            torch.zeros_like(batch["progress"][:, :1]),
            robot_memory,
        )
        interleaved = torch.stack((state[:, :-1], action), dim=2).flatten(1, 2)
        tokens = torch.cat((self.before_fusion(before), interleaved, state[:, -1:]), dim=1)
        type_ids = torch.empty(tokens.shape[1], dtype=torch.long, device=tokens.device)
        type_ids[0] = 0
        type_ids[1::2] = 1
        type_ids[2::2] = 2
        tokens = tokens + self.type_embedding(type_ids)[None] + robot_summary[:, None]
        tokens = tokens + self.task_embedding(
            torch.full((tokens.shape[0],), masks.task_id, dtype=torch.long, device=tokens.device)
        )[:, None]
        valid_interleaved = torch.stack(
            (batch["valid_state"][:, :-1], batch["valid_action"]), dim=2
        ).flatten(1, 2)
        valid = torch.cat((
            torch.ones(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device),
            valid_interleaved,
            batch["valid_state"][:, -1:],
        ), dim=1)
        times = torch.arange(tokens.shape[1], device=tokens.device)
        state_indices = torch.arange(1, tokens.shape[1], 2, device=tokens.device)
        action_indices = torch.arange(2, tokens.shape[1] - 1, 2, device=tokens.device)
        return tokens, valid, times, state_indices, action_indices

    def _reference_context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        action = batch["action"]
        if self.reference_conditioning == "off":
            return action.new_zeros(action.shape[0], action.shape[1], self.width)
        if "reference_future" not in batch or "reference_available" not in batch:
            raise ValueError("reference_conditioning=required but reference tensors are missing")
        available = batch["reference_available"].bool()
        if not bool(available.all()):
            raise ValueError("reference_conditioning=required but this dataset has no runtime reference")
        reference = batch["reference_future"]
        if reference.shape[1:3] != (action.shape[1], 10) or reference.shape[-1] != 64:
            raise ValueError(f"unexpected reference tensor shape {tuple(reference.shape)}")
        batch_size, steps, frames = reference.shape[:3]
        token = self.reference_input(reference.reshape(batch_size * steps, frames, 64))
        positions = torch.arange(frames, device=reference.device)
        token = token + self.reference_position(positions)[None]
        valid = torch.ones(batch_size * steps, frames, dtype=torch.bool, device=reference.device)
        encoded = self.reference_encoder(token, valid, positions, False)
        return encoded.mean(dim=1).reshape(batch_size, steps, self.width)

    def _history_context(
        self, transition: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        batch, steps, width = transition.shape
        padding = self.history_steps - 1
        padded = F.pad(transition, (0, 0, padding, 0))
        valid_padded = F.pad(valid, (padding, 0), value=False)
        windows = torch.stack(
            [padded[:, index : index + self.history_steps] for index in range(steps)], dim=1
        )
        valid_windows = torch.stack(
            [valid_padded[:, index : index + self.history_steps] for index in range(steps)], dim=1
        )
        flat = windows.reshape(batch * steps, self.history_steps, width)
        flat_valid = valid_windows.reshape(batch * steps, self.history_steps)
        times = torch.arange(self.history_steps, device=transition.device)
        encoded = self.history_encoder(flat, flat_valid, times, False)
        return encoded[:, -1].reshape(batch, steps, width)

    def _history_from_sequences(
        self, sequences: list[list[torch.Tensor]], prototype: torch.Tensor
    ) -> torch.Tensor:
        batch = len(sequences)
        tokens = prototype.new_zeros(batch, self.history_steps, self.width)
        valid = torch.zeros(
            batch, self.history_steps, dtype=torch.bool, device=prototype.device
        )
        nonempty = torch.zeros(batch, dtype=torch.bool, device=prototype.device)
        for index, sequence in enumerate(sequences):
            selected = sequence[-self.history_steps :]
            if selected:
                count = len(selected)
                tokens[index, -count:] = torch.stack(selected)
                valid[index, -count:] = True
                nonempty[index] = True
        valid[~nonempty, -1] = True
        times = torch.arange(self.history_steps, device=prototype.device)
        encoded = self.history_encoder(tokens, valid, times, False)[:, -1]
        return torch.where(nonempty[:, None], encoded, torch.zeros_like(encoded))

    def _decode_state(
        self, hidden: torch.Tensor, robot_memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = robot_memory[:, None].expand(-1, hidden.shape[1], -1, -1)
        query = hidden[:, :, None].expand(-1, -1, ACTION_DIM, -1)
        joint = self.state_joint_output(
            self.state_joint_decoder(torch.cat((query, memory), dim=-1))
        )
        base_raw = self.state_base_output(hidden)
        base = torch.cat((
            base_raw[..., :6],
            F.normalize(base_raw[..., 6:9], dim=-1, eps=1e-6),
            base_raw[..., 9:],
        ), dim=-1)
        contact_logits = self.state_contact_output(hidden)
        state = torch.cat((joint[..., 0], joint[..., 1], base, torch.sigmoid(contact_logits)), dim=-1)
        return state, contact_logits

    def _decode_action(self, hidden: torch.Tensor, robot_memory: torch.Tensor) -> torch.Tensor:
        memory = robot_memory[:, None].expand(-1, hidden.shape[1], -1, -1)
        query = hidden[:, :, None].expand(-1, -1, ACTION_DIM, -1)
        return self.action_joint_output(
            self.action_joint_decoder(torch.cat((query, memory), dim=-1))
        ).squeeze(-1)

    def _decode_inverse_distribution(
        self, hidden: torch.Tensor, robot_memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = robot_memory[:, None].expand(-1, hidden.shape[1], -1, -1)
        query = hidden[:, :, None].expand(-1, -1, ACTION_DIM, -1)
        decoded = self.action_joint_decoder(torch.cat((query, memory), dim=-1))
        mean = self.action_joint_output(decoded).squeeze(-1)
        log_scale = self.inverse_log_scale_output(decoded).squeeze(-1).clamp(-5.0, 2.0)
        return mean, log_scale

    def _predict_next(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        progress: torch.Tensor,
        robot_memory: torch.Tensor,
        history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_hidden = self._state_embed(
            state[:, None], torch.zeros_like(state[:, None], dtype=torch.bool),
            progress[:, None], robot_memory,
        )[:, 0]
        action_hidden = self._action_embed(
            action[:, None], torch.zeros_like(action[:, None], dtype=torch.bool),
            progress[:, None], robot_memory,
        )[:, 0]
        relation = self.forward_relation(torch.cat((state_hidden, action_hidden, history), dim=-1))
        raw = self.forward_continuous(relation)
        continuous = state[:, :68] + raw
        continuous = torch.cat((
            continuous[:, :64],
            F.normalize(continuous[:, 64:67], dim=-1, eps=1e-6),
            continuous[:, 67:68],
        ), dim=-1)
        contact_logits = self.forward_contact(relation)
        next_state = torch.cat((continuous, torch.sigmoid(contact_logits)), dim=-1)
        return next_state, continuous - state[:, :68], contact_logits, relation

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        masks: MaskBatch,
        sample_from_prior: bool = False,
        deterministic: bool = False,
    ) -> ModelOutput:
        robot_memory, robot_summary = self._robot_memory(batch)
        visible, valid, times, state_indices, action_indices = self._tokenize(
            batch, masks, False, robot_memory, robot_summary
        )
        full, _, _, _, _ = self._tokenize(
            batch, masks, True, robot_memory, robot_summary
        )
        prior_encoded = self.completion_encoder(visible, valid, times, masks.causal)
        posterior_encoded = self.completion_encoder(full, valid, times, masks.causal)
        weights = valid.to(visible.dtype)
        prior_pool = (prior_encoded * weights[:, :, None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
        posterior_pool = (posterior_encoded * weights[:, :, None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
        prior_mean, prior_logvar = self._distribution(self.prior(prior_pool))
        posterior_mean, posterior_logvar = self._distribution(self.posterior(posterior_pool))
        mean, logvar = (
            (prior_mean, prior_logvar) if sample_from_prior
            else (posterior_mean, posterior_logvar)
        )
        latent = mean if deterministic else mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        decoded = self.completion_decoder(
            visible + (robot_summary + self.latent_projection(latent))[:, None],
            valid, times, masks.causal,
        )
        reference = self._reference_context(batch)
        state_output, state_contact_logits = self._decode_state(
            decoded[:, state_indices], robot_memory
        )
        action_output = self._decode_action(
            decoded[:, action_indices] + reference, robot_memory
        )

        inputs = masked_inputs(batch, masks, full=False)
        state_direct = self._state_embed(
            inputs["physical_state"], masks.state_input, batch["progress"], robot_memory
        )
        action_direct = self._action_embed(
            inputs["action"], masks.action_input, batch["progress"][:, :-1], robot_memory
        )
        queued_action = (
            action_direct if self.action_queue_conditioning
            else torch.zeros_like(action_direct)
        )
        transition_token = self.transition_fusion(
            torch.cat((state_direct[:, :-1], queued_action), dim=-1)
        ) + robot_summary[:, None]
        history = self._history_context(transition_token, batch["valid_action"].bool())
        identified = torch.cat((
            torch.zeros_like(history[:, :1]),
            self.dynamics_identifier(history[:, :-1]),
        ), dim=1)
        causal_context = (
            history + identified if self.causal_dynamics_embedding else history
        )

        flat_count = batch["physical_state"].shape[0] * batch["action"].shape[1]
        _, continuous_delta, forward_contact_logits, transition = self._predict_next(
            inputs["physical_state"][:, :-1].reshape(flat_count, 70),
            inputs["action"].reshape(flat_count, ACTION_DIM),
            batch["progress"][:, :-1].reshape(flat_count),
            robot_memory[:, None].expand(-1, batch["action"].shape[1], -1, -1).reshape(
                flat_count, ACTION_DIM, robot_memory.shape[-1]
            ),
            causal_context.reshape(flat_count, self.width),
        )
        continuous_delta = continuous_delta.reshape(batch["action"].shape[0], -1, 68)
        forward_contact_logits = forward_contact_logits.reshape(
            batch["action"].shape[0], -1, 2
        )
        contact_delta = torch.sigmoid(forward_contact_logits) - batch["physical_state"][:, :-1, 68:]
        forward_delta = torch.cat((continuous_delta, contact_delta), dim=-1)
        transition = transition.reshape(batch["action"].shape[0], -1, self.width)
        auxiliary = (
            self.auxiliary_head(transition)
            if self.auxiliary_head is not None
            else transition.new_empty(transition.shape[0], transition.shape[1], 0)
        )

        inverse_hidden = self.inverse_relation(torch.cat((
            state_direct[:, :-1], state_direct[:, 1:], causal_context, reference,
        ), dim=-1))
        inverse_action, inverse_action_log_scale = self._decode_inverse_distribution(
            inverse_hidden, robot_memory
        )
        history_action = self._decode_action(
            self.history_relation(torch.cat((causal_context, reference), dim=-1)), robot_memory
        )

        cycle_state = None
        if masks.inverse_transition is not None and bool(masks.inverse_transition.any()):
            cycle_next, _, _, _ = self._predict_next(
                batch["physical_state"][:, :-1].reshape(flat_count, 70),
                inverse_action.reshape(flat_count, ACTION_DIM),
                batch["progress"][:, :-1].reshape(flat_count),
                robot_memory[:, None].expand(
                    -1, inverse_action.shape[1], -1, -1
                ).reshape(flat_count, ACTION_DIM, robot_memory.shape[-1]),
                causal_context.reshape(flat_count, self.width),
            )
            cycle_state = cycle_next.reshape(batch["physical_state"].shape[0], -1, 70)
        cycle_action = None
        if masks.forward_transition is not None and bool(masks.forward_transition.any()):
            predicted_next = batch["physical_state"][:, :-1] + forward_delta
            predicted_hidden = self._state_embed(
                predicted_next,
                torch.zeros_like(predicted_next, dtype=torch.bool),
                batch["progress"][:, 1:],
                robot_memory,
            )
            cycle_hidden = self.inverse_relation(torch.cat((
                state_direct[:, :-1], predicted_hidden, causal_context, reference,
            ), dim=-1))
            cycle_action = self._decode_action(cycle_hidden, robot_memory)

        if masks.rollout_horizon is not None and bool((masks.rollout_horizon > 0).any()):
            current = torch.stack([
                batch["physical_state"][index, int(masks.rollout_start[index])]
                for index in range(batch["physical_state"].shape[0])
            ])
            rollout_histories = [
                [token for token in transition_token[index, : int(masks.rollout_start[index])]]
                for index in range(transition_token.shape[0])
            ]
            rollout = []
            for offset in range(8):
                action_at = torch.stack([
                    batch["action"][index, min(int(masks.rollout_start[index]) + offset, batch["action"].shape[1] - 1)]
                    for index in range(batch["action"].shape[0])
                ])
                progress_at = torch.stack([
                    batch["progress"][index, min(int(masks.rollout_start[index]) + offset, batch["action"].shape[1] - 1)]
                    for index in range(batch["action"].shape[0])
                ])
                previous_history = self._history_from_sequences(
                    rollout_histories, current
                )
                current_state_hidden = self._state_embed(
                    current[:, None],
                    torch.zeros_like(current[:, None], dtype=torch.bool),
                    progress_at[:, None],
                    robot_memory,
                )[:, 0]
                current_action_hidden = self._action_embed(
                    action_at[:, None],
                    torch.zeros_like(action_at[:, None], dtype=torch.bool),
                    progress_at[:, None],
                    robot_memory,
                )[:, 0]
                queued_current_action = (
                    current_action_hidden if self.action_queue_conditioning
                    else torch.zeros_like(current_action_hidden)
                )
                current_token = self.transition_fusion(torch.cat((
                    current_state_hidden, queued_current_action,
                ), dim=-1)) + robot_summary
                for index in range(current_token.shape[0]):
                    rollout_histories[index].append(current_token[index])
                history_at = self._history_from_sequences(
                    rollout_histories, current
                )
                if self.causal_dynamics_embedding:
                    history_at = history_at + self.dynamics_identifier(previous_history)
                current, _, _, _ = self._predict_next(
                    current, action_at, progress_at, robot_memory, history_at
                )
                rollout.append(current)
            rollout_state = torch.stack(rollout, dim=1)
        else:
            rollout_state = state_output.new_empty(state_output.shape[0], 0, 70)
        return ModelOutput(
            state_output,
            state_output.new_empty(state_output.shape[0], state_output.shape[1], 0),
            action_output,
            forward_delta,
            auxiliary,
            posterior_mean,
            posterior_logvar,
            prior_mean,
            prior_logvar,
            latent,
            state_contact_logits,
            forward_contact_logits,
            inverse_action,
            history_action,
            rollout_state,
            cycle_state,
            cycle_action,
            inverse_action_log_scale,
        )


class TemporalResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.norm1 = nn.GroupNorm(1, width)
        self.conv1 = nn.Conv1d(width, width, kernel_size, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, width)
        self.conv2 = nn.Conv1d(width, width, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def _convolve(self, value: torch.Tensor, convolution: nn.Conv1d, causal: bool) -> torch.Tensor:
        padding = self.dilation * (self.kernel_size - 1)
        if causal:
            value = F.pad(value, (padding, 0))
        else:
            left = padding // 2
            value = F.pad(value, (left, padding - left))
        return convolution(value)

    def forward(
        self, value: torch.Tensor, causal: bool, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = F.gelu(self.norm1(value))
        hidden = self._convolve(hidden, self.conv1, causal)
        if valid is not None:
            hidden = hidden * valid
        hidden = self.dropout(hidden)
        hidden = F.gelu(self.norm2(hidden))
        hidden = self._convolve(hidden, self.conv2, causal)
        result = value + self.dropout(hidden)
        return result if valid is None else result * valid


class TemporalStack(nn.Module):
    def __init__(self, width: int, layers: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TemporalResidualBlock(width, 2 ** (index % 5), kernel, dropout)
            for index in range(layers)
        )

    def forward(
        self, value: torch.Tensor, causal: bool, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value, causal, valid)
        return value


class TCNCVAE(nn.Module):
    """Parameter-matched temporal convolution baseline with the same public outputs."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        width = int(config["tcn_channels"])
        latent = int(config["latent_dim"])
        dropout = float(config["dropout"])
        self.latent_dim = latent
        self.state_dim = int(config.get("state_dim", PHYSICAL_STATE_DIM))
        self.include_previous_action = bool(config.get("include_previous_action", True))
        self.robot_info_dim = int(config.get("robot_info_dim", 0))
        self.dynamics_context_dim = int(config.get("dynamics_context_dim", 0))
        self.auxiliary_dim = int(config.get("auxiliary_dim", 0))
        self.context_mode = str(config.get("context_mode", "hidden"))
        self.token_layout = str(config.get("token_layout", "grouped"))
        self.state_tokenizer = MLPTokenizer(self.state_dim * 2 + 1, width)
        self.previous_tokenizer = (
            MLPTokenizer(ACTION_DIM * 2 + 1, width)
            if self.include_previous_action
            else None
        )
        self.action_tokenizer = MLPTokenizer(ACTION_DIM * 2 + 1, width)
        condition_dim = self.robot_info_dim if self.robot_info_dim else ACTION_DIM
        if self.context_mode == "explicit":
            condition_dim += self.dynamics_context_dim
        self.scale_tokenizer = MLPTokenizer(condition_dim, width)
        self.task_embedding = nn.Embedding(3, width)
        self.encoder = TemporalStack(
            width,
            int(config["tcn_encoder_layers"]),
            int(config["tcn_kernel_size"]),
            dropout,
        )
        self.decoder = TemporalStack(
            width,
            int(config["tcn_decoder_layers"]),
            int(config["tcn_kernel_size"]),
            dropout,
        )
        self.prior = nn.Linear(width, latent * 2)
        self.posterior = nn.Linear(width, latent * 2)
        self.latent_projection = nn.Linear(latent, width)
        self.state_head = nn.Conv1d(width, self.state_dim, 1)
        self.previous_head = (
            nn.Conv1d(width, ACTION_DIM, 1) if self.include_previous_action else None
        )
        self.action_head = nn.Conv1d(width, ACTION_DIM, 1)
        self.forward_head = nn.Conv1d(width, self.state_dim, 1)
        self.auxiliary_head = (
            nn.Conv1d(width, self.auxiliary_dim, 1) if self.auxiliary_dim else None
        )

    def _tokens(
        self, batch: dict[str, torch.Tensor], masks: MaskBatch, full: bool
    ) -> torch.Tensor:
        inputs = masked_inputs(batch, masks, full=full)
        action = F.pad(inputs["action"], (0, 0, 0, 1))
        action_mask = F.pad(masks.action_input, (0, 0, 0, 1))
        state = self.state_tokenizer(
            torch.cat(
                (
                    inputs["physical_state"],
                    masks.state_input.to(inputs["physical_state"].dtype),
                    batch["progress"][..., None],
                ),
                -1,
            )
        )
        previous = None
        if self.previous_tokenizer is not None:
            previous = self.previous_tokenizer(
                torch.cat(
                    (
                        inputs["previous_action"],
                        masks.previous_input.to(inputs["previous_action"].dtype),
                        batch["progress"][..., None],
                    ),
                    -1,
                )
            )
        action_progress = torch.cat(
            (batch["progress"][:, 1:], torch.zeros_like(batch["progress"][:, :1])), dim=1
        )
        action = self.action_tokenizer(
            torch.cat((action, action_mask.to(action.dtype), action_progress[..., None]), -1)
        )
        task_ids = torch.full(
            (state.shape[0],), masks.task_id, dtype=torch.long, device=state.device
        )
        condition_input = (
            batch["robot_information"] if self.robot_info_dim else batch["action_scale"]
        )
        if self.context_mode == "explicit":
            condition_input = torch.cat(
                (condition_input, batch["dynamics_context"]), dim=-1
            )
        condition = self.scale_tokenizer(condition_input) + self.task_embedding(task_ids)
        if self.token_layout == "interleaved":
            tokens = torch.stack((state, action), dim=2).flatten(1, 2)[:, :-1]
            return (tokens + condition[:, None]).transpose(1, 2)
        if previous is None:
            raise RuntimeError("grouped TCN requires previous Action tokens")
        return (state + previous + action + condition[:, None]).transpose(1, 2)

    @staticmethod
    def _params(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = value.chunk(2, dim=-1)
        return mean, logvar.clamp(-12.0, 8.0)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        masks: MaskBatch,
        sample_from_prior: bool = False,
        deterministic: bool = False,
    ) -> ModelOutput:
        visible_tokens = self._tokens(batch, masks, False)
        full_tokens = self._tokens(batch, masks, True)
        if self.token_layout == "interleaved":
            valid_action = F.pad(batch["valid_action"], (0, 1), value=False)
            valid_values = torch.stack(
                (batch["valid_state"], valid_action), dim=2
            ).flatten(1, 2)[:, :-1]
        else:
            valid_values = batch["valid_state"]
        valid = valid_values.to(visible_tokens.dtype)[:, None]
        visible = self.encoder(visible_tokens, masks.causal, valid)
        full = self.encoder(full_tokens, masks.causal, valid)
        prior_pool = (visible * valid).sum(-1) / valid.sum(-1).clamp_min(1.0)
        posterior_pool = (full * valid).sum(-1) / valid.sum(-1).clamp_min(1.0)
        prior_mean, prior_logvar = self._params(self.prior(prior_pool))
        posterior_mean, posterior_logvar = self._params(self.posterior(posterior_pool))
        mean = prior_mean if sample_from_prior else posterior_mean
        logvar = prior_logvar if sample_from_prior else posterior_logvar
        latent = mean if deterministic else mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        decoded = self.decoder(
            visible + self.latent_projection(latent)[:, :, None], masks.causal, valid
        )
        if self.token_layout == "interleaved":
            state_hidden = decoded[:, :, 0::2]
            action_hidden = decoded[:, :, 1::2]
            state = self.state_head(state_hidden).transpose(1, 2)
            previous = state.new_empty(state.shape[0], state.shape[1], 0)
            action = self.action_head(action_hidden).transpose(1, 2)
            delta = self.forward_head(state_hidden[:, :, 1:]).transpose(1, 2)
            auxiliary = (
                self.auxiliary_head(state_hidden[:, :, 1:]).transpose(1, 2)
                if self.auxiliary_head is not None
                else state.new_empty(state.shape[0], state.shape[1] - 1, 0)
            )
        else:
            state = self.state_head(decoded).transpose(1, 2)
            if self.previous_head is None:
                raise RuntimeError("grouped TCN requires previous Action head")
            previous = self.previous_head(decoded).transpose(1, 2)
            action = self.action_head(decoded).transpose(1, 2)[:, :-1]
            delta = self.forward_head(decoded).transpose(1, 2)[:, :-1]
            auxiliary = state.new_empty(state.shape[0], state.shape[1] - 1, 0)
        return ModelOutput(
            state,
            previous,
            action,
            delta,
            auxiliary,
            posterior_mean,
            posterior_logvar,
            prior_mean,
            prior_logvar,
            latent,
        )


def build_model(config: dict[str, Any]) -> nn.Module:
    kind = str(config.get("kind", "transformer"))
    if kind == "transformer":
        return TransformerCVAE(config)
    if kind == "physics_transformer":
        return PhysicsTransformerCVAE(config)
    if kind == "physics_lean_split":
        return LeanSplitPhysicsCVAE(config)
    if kind == "tcn":
        return TCNCVAE(config)
    raise ValueError(f"unsupported model kind {kind!r}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
