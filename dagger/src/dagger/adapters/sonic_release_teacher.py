"""SONIC release teacher adapter; SONIC remains an external dependency."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


def _load_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("policy_state_dict", "actor_model_state_dict", "state_dict"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
        if all(isinstance(key, str) for key in checkpoint):
            return checkpoint
    raise ValueError("teacher checkpoint does not contain policy_state_dict")


class SonicReleaseTeacher:
    def __init__(self, *, config: Any, env: Any, checkpoint: str | Path, device: torch.device):
        from gear_sonic.trl.utils.common import custom_instantiate, materialize_lazy_params

        self.device = device
        self.env = env
        self.motion = env.command_manager.get_term("motion")
        self._validate_g1()
        print("[dagger] Teacher: instantiating actor", flush=True)
        self.model = custom_instantiate(
            config.algo.config.actor,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=getattr(config.algo.config, "module_dim", {}),
            _resolve=False,
        ).to(device)
        print("[dagger] Teacher: materializing lazy params", flush=True)
        materialize_lazy_params(self.model, env.env)
        print("[dagger] Teacher: lazy params materialized", flush=True)
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"teacher checkpoint is missing: {checkpoint_path}")
        print(f"[dagger] Teacher: hashing checkpoint {checkpoint_path}", flush=True)
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        print("[dagger] Teacher: loading checkpoint tensors", flush=True)
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print("[dagger] Teacher: checkpoint tensors loaded", flush=True)
        state_dict = _load_state_dict(state)
        model_state = self.model.state_dict()
        if "std" in model_state and "log_std" in state_dict and "std" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["std"] = torch.exp(state_dict.pop("log_std"))
        elif "log_std" in model_state and "std" in state_dict and "log_std" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["log_std"] = torch.log(state_dict.pop("std"))
        print("[dagger] Teacher: loading state dict", flush=True)
        self.model.load_state_dict(state_dict, strict=True)
        print("[dagger] Teacher: state dict loaded", flush=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.action_dim = int(getattr(self.model, "num_actions", getattr(env.config.robot, "actions_dim", 29)))
        if self.action_dim != int(env.env.action_space.shape[-1]):
            raise ValueError(
                f"teacher action dimension {self.action_dim} does not match environment "
                f"dimension {env.env.action_space.shape[-1]}"
            )
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("SONIC teacher parameters were not fully frozen")

    def _validate_g1(self) -> None:
        g1_index = getattr(self.motion, "g1_encoder_index", None)
        encoder_index = getattr(self.motion, "encoder_index", None)
        if g1_index is None or encoder_index is None:
            raise RuntimeError("SONIC motion command does not expose encoder_index")
        if not torch.all(encoder_index[:, g1_index] == 1):
            raise RuntimeError("SONIC teacher environments are not all assigned to G1")
        if not torch.all(encoder_index.sum(dim=-1) == 1):
            raise RuntimeError("SONIC teacher encoder selection is not one-hot")

    def reset(self, num_envs: int) -> None:
        if hasattr(self.model, "init_rollout"):
            self.model.init_rollout()
        if int(num_envs) != self.env.num_envs:
            raise ValueError("teacher reset num_envs does not match environment")

    def act(self, observation: dict[str, torch.Tensor], done: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action = self.model.act_inference(obs_dict=observation, cur_dones=done)
        if action.ndim != 2:
            raise ValueError(f"SONIC teacher action must be [B,A], found {tuple(action.shape)}")
        if not torch.isfinite(action).all():
            raise FloatingPointError("SONIC teacher produced non-finite Action")
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"SONIC teacher action dimension changed: {tuple(action.shape)}")
        return action

