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
    if kind == "tcn":
        return TCNCVAE(config)
    raise ValueError(f"unsupported model kind {kind!r}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
