"""Load the existing SONIC environment without adding DAgger Hydra groups."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


PHYSICS_STATE_DIM = 70
ACTION_DIM = 29
DEFAULT_REFERENCE_TERMS = (
    "command_multi_future_nonflat",
    "motion_anchor_ori_b_mf_nonflat",
)


class SonicDaggerEnv:
    """Thin observation adapter around the unchanged SONIC ManagerEnvWrapper."""

    def __init__(self, env: Any, *, reference_terms: list[str] | tuple[str, ...] | None = None) -> None:
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.config = env.config
        self.reference_terms = tuple(reference_terms or DEFAULT_REFERENCE_TERMS)
        if not self.reference_terms:
            raise ValueError("at least one G1 reference term is required")
        self._reference_slices = self._build_reference_slices()

    def _build_reference_slices(self) -> dict[str, tuple[int, int]]:
        names = list(self.config.obs.group_obs_names["tokenizer"])
        dims = self.config.obs.group_obs_dims["tokenizer"]
        offset = 0
        slices: dict[str, tuple[int, int]] = {}
        for name in names:
            shape = tuple(dims[name])
            width = 1
            for value in shape:
                width *= int(value)
            slices[name] = (offset, offset + width)
            offset += width
        missing = [name for name in self.reference_terms if name not in slices]
        if missing:
            available = sorted(slices)
            raise KeyError(
                f"SONIC tokenizer is missing G1 reference terms: {missing}; "
                f"available terms are: {available}"
            )
        return slices

    def _augment(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokenizer = observation.get("tokenizer")
        if tokenizer is None or tokenizer.ndim != 2:
            raise ValueError("SONIC tokenizer observation must be a flattened [B,D] tensor")
        selected = [
            tokenizer[:, self._reference_slices[name][0] : self._reference_slices[name][1]]
            for name in self.reference_terms
        ]
        result = dict(observation)
        result["student_reference_features"] = torch.cat(selected, dim=-1)
        return result

    def reset(self):
        return self._augment(self.env.reset())

    def step(self, actions):
        observation, reward, done, info = self.env.step(actions)
        return self._augment(observation), reward, done, info

    def physics_state(self) -> torch.Tensor:
        """Read the released SONIC recorder's canonical 70-D physics State.

        The recorder implementation is imported read-only from SONIC.  Keeping
        this conversion in the external adapter avoids adding a DAgger
        observation group to SONIC's Hydra tree.
        """

        from gear_sonic.envs.manager_env.mdp.recorders import _physics_state

        actual_env = getattr(self.env, "env", self.env)
        fields = _physics_state(actual_env)
        value = torch.cat(
            (
                fields["joint_pos_canonical"],
                fields["joint_vel"],
                fields["base_lin_vel_robot"],
                fields["base_ang_vel_robot"],
                fields["gravity_robot"],
                fields["base_height"],
                fields["foot_contact"].to(dtype=torch.float32),
            ),
            dim=-1,
        )
        if tuple(value.shape) != (self.num_envs, PHYSICS_STATE_DIM):
            raise RuntimeError(f"SONIC physics State must be [B,70], found {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise FloatingPointError("SONIC physics State contains non-finite values")
        return value.detach()

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    @property
    def command_manager(self):
        manager = getattr(self.env, "command_manager", None)
        if manager is not None:
            return manager
        return self.env.env.command_manager

    @property
    def motion_term(self):
        return self.command_manager.get_term("motion")

    @property
    def motion_keys(self) -> list[str]:
        motion_lib = getattr(self.motion_term, "motion_lib", None)
        keys = getattr(motion_lib, "_motion_data_keys", None)
        if keys is None:
            return []
        return [value.decode() if isinstance(value, bytes) else str(value) for value in keys]

    @property
    def action_low(self) -> torch.Tensor:
        value = torch.as_tensor(self.env.action_space.low, device=self.device, dtype=torch.float32)
        return value.reshape(1, -1)

    @property
    def action_high(self) -> torch.Tensor:
        value = torch.as_tensor(self.env.action_space.high, device=self.device, dtype=torch.float32)
        return value.reshape(1, -1)


class SonicActionContract:
    """Convert between SONIC raw policy Actions and CVAE canonical Actions."""

    def __init__(self, env: SonicDaggerEnv) -> None:
        actual_env = getattr(env.env, "env", env.env)
        meta_action_dim = env.config.get("meta_action_dim", None) if hasattr(env.config, "get") else getattr(env.config, "meta_action_dim", None)
        if meta_action_dim is None:
            raw_config = getattr(env.env, "config", None)
            meta_action_dim = raw_config.get("meta_action_dim", None) if hasattr(raw_config, "get") else None
        if meta_action_dim is not None:
            raise ValueError("CVAE DAgger requires direct 29-D environment actions")
        wrapper = getattr(actual_env, "wrapper", None)
        if getattr(wrapper, "action_transform_module", None) is not None:
            raise ValueError("CVAE DAgger does not support wrapper action transforms")
        from gear_sonic.envs.manager_env.mdp.recorders import (
            _action_term,
            _physics_nominal_joint_pos,
        )

        term = _action_term(actual_env)
        self.term = term
        self.device = env.device
        self.num_envs = env.num_envs
        action_space = env.env.action_space
        self.raw_low = torch.as_tensor(action_space.low, device=self.device, dtype=torch.float32).reshape(1, -1)
        self.raw_high = torch.as_tensor(action_space.high, device=self.device, dtype=torch.float32).reshape(1, -1)
        if tuple(self.raw_low.shape) != (1, ACTION_DIM) or tuple(self.raw_high.shape) != (1, ACTION_DIM):
            raise ValueError("SONIC direct action space must have shape [29]")
        self.scale = torch.as_tensor(getattr(term, "_scale", 1.0), device=self.device, dtype=torch.float32)
        self.offset = torch.as_tensor(getattr(term, "_offset", 0.0), device=self.device, dtype=torch.float32)
        clip = getattr(term, "_clip", None)
        self.clip = None if clip is None else torch.as_tensor(clip, device=self.device, dtype=torch.float32)
        self.nominal = _physics_nominal_joint_pos(actual_env, getattr(term, "_joint_ids")).to(self.device)
        def expand_param(name: str, value: torch.Tensor) -> torch.Tensor:
            if value.ndim == 0:
                value = value.reshape(1, 1).expand(self.num_envs, ACTION_DIM)
            elif value.ndim == 1:
                if value.numel() != ACTION_DIM:
                    raise ValueError(f"SONIC {name} must contain 29 values")
                value = value[None].expand(self.num_envs, -1)
            if tuple(value.shape) != (self.num_envs, ACTION_DIM):
                raise ValueError(f"SONIC {name} must be [B,29], found {tuple(value.shape)}")
            return value

        self.scale = expand_param("scale", self.scale)
        self.offset = expand_param("offset", self.offset)
        self.nominal = expand_param("nominal", self.nominal)
        if torch.any(self.scale == 0):
            raise ValueError("SONIC action scale contains zero and cannot be inverted")
        self.scale = self.scale.reshape(self.num_envs, ACTION_DIM)
        self.offset = self.offset.reshape(self.num_envs, ACTION_DIM)
        self.nominal = self.nominal.reshape(self.num_envs, ACTION_DIM)
        if self.clip is not None and self.clip.ndim == 2:
            self.clip = self.clip[None].expand(self.num_envs, -1, -1)
        if self.clip is not None and tuple(self.clip.shape) != (self.num_envs, ACTION_DIM, 2):
            raise ValueError(f"SONIC action clip must be [B,29,2], found {tuple(self.clip.shape)}")
        self.action_dim = ACTION_DIM

    def raw_to_processed(self, raw: torch.Tensor) -> torch.Tensor:
        raw = raw.to(self.device, dtype=torch.float32)
        if tuple(raw.shape) != (self.num_envs, ACTION_DIM):
            raise ValueError(f"raw Action must be [{self.num_envs},29]")
        processed = raw * self.scale + self.offset
        if self.clip is not None:
            processed = torch.maximum(processed, self.clip[..., 0])
            processed = torch.minimum(processed, self.clip[..., 1])
        return processed

    def clip_raw(self, raw: torch.Tensor) -> torch.Tensor:
        raw = raw.to(self.device, dtype=torch.float32)
        return torch.maximum(torch.minimum(raw, self.raw_high), self.raw_low)

    def raw_to_canonical(self, raw: torch.Tensor) -> torch.Tensor:
        return self.raw_to_processed(raw) - self.nominal

    def canonical_to_raw(self, canonical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        canonical = canonical.to(self.device, dtype=torch.float32)
        if tuple(canonical.shape) != (self.num_envs, ACTION_DIM):
            raise ValueError(f"canonical Action must be [{self.num_envs},29]")
        desired = canonical + self.nominal
        raw = (desired - self.offset) / self.scale
        if hasattr(self.term, "_raw_action_clip") and self.term._raw_action_clip is not None:
            raw_clip = torch.as_tensor(self.term._raw_action_clip, device=self.device, dtype=torch.float32)
            raw = torch.maximum(raw, raw_clip[..., 0])
            raw = torch.minimum(raw, raw_clip[..., 1])
        raw = self.clip_raw(raw)
        achieved = self.raw_to_canonical(raw)
        return raw, achieved

    def round_trip_error(self, canonical: torch.Tensor) -> torch.Tensor:
        raw, achieved = self.canonical_to_raw(canonical)
        del raw
        return (achieved - canonical).abs().amax(dim=-1)


def compose_release_config(
    sonic_repo_root: str | Path,
    *,
    num_envs: int,
    seed: int,
    motion_file: str | Path | None = None,
    smpl_motion_file: str | Path = "zeros",
    motion_keys: list[str] | tuple[str, ...] | None = None,
    max_unique_motions: int | None = 1,
    sample_unique_motions: bool | None = None,
    observation_corruption: bool = False,
    events: str = "tracking/level0_4",
    terminations: str = "tracking/base_adaptive_strict_ori_foot_xyz",
):
    """Compose only the existing SONIC release experiment configuration."""

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    sonic_root = Path(sonic_repo_root).expanduser().resolve()
    config_dir = sonic_root / "gear_sonic" / "config"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        config = compose(
            config_name="base",
            overrides=[
                "+exp=manager/universal_token/all_modes/sonic_release",
                f"num_envs={int(num_envs)}",
                f"seed={int(seed)}",
                "headless=True",
                "manager_env.commands.motion.encoder_sample_probs={g1:1.0,teleop:0.0,smpl:0.0}",
                "+manager_env.commands.motion.motion_lib_cfg.deterministic_motion_order=True",
                f"manager_env/terminations={terminations}",
                f"manager_env/events={events}",
            ],
        )
    OmegaConf.set_struct(config, False)

    motion_path = (
        sonic_root / "data" / "motion_lib_bones_seed" / "robot_filtered"
        if motion_file is None
        else Path(str(motion_file)).expanduser()
    )
    if not motion_path.is_absolute():
        motion_path = sonic_root / motion_path
    motion_path = motion_path.resolve()
    if not motion_path.exists():
        raise FileNotFoundError(f"SONIC motion_file does not exist: {motion_path}")

    smpl_path = Path(str(smpl_motion_file)).expanduser()
    if str(smpl_motion_file) not in {"zeros", "dummy"}:
        if not smpl_path.is_absolute():
            smpl_path = sonic_root / smpl_path
        smpl_path = smpl_path.resolve()

    keys = [str(value) for value in (motion_keys or [])]
    if keys:
        if max_unique_motions is None or int(max_unique_motions) > len(keys):
            max_unique_motions = len(keys)
        if int(max_unique_motions) < len(keys):
            raise ValueError(
                f"max_unique_motions={max_unique_motions} would drop requested motion keys; "
                f"requested {len(keys)}"
            )
        if sample_unique_motions and int(num_envs) > len(keys):
            raise ValueError(
                f"sample_unique_motions requires num_envs <= requested motions "
                f"({int(num_envs)} > {len(keys)})"
            )

    motion_cfg = config.manager_env.commands.motion
    motion_cfg.motion_lib_cfg.motion_file = str(motion_path)
    motion_cfg.motion_lib_cfg.smpl_motion_file = "zeros" if str(smpl_motion_file) in {"zeros", "dummy"} else str(smpl_path)
    motion_cfg.motion_lib_cfg.asset.assetRoot = str(
        sonic_root / "gear_sonic" / "data" / "assets" / "robot_description" / "mjcf"
    )
    if keys:
        motion_cfg.motion_lib_cfg.filter_motion_keys = keys
    if max_unique_motions is not None:
        motion_cfg.motion_lib_cfg.max_unique_motions = int(max_unique_motions)
    if sample_unique_motions is not None:
        motion_cfg.sample_unique_motions = bool(sample_unique_motions)
    config.manager_env.observations.policy.enable_corruption = bool(observation_corruption)
    config.manager_env.observations.tokenizer.enable_corruption = bool(observation_corruption)
    return config


def prepare_env_dimensions(env: Any) -> None:
    """Populate the dimension maps expected by existing SONIC modules."""

    from gear_sonic.utils.obs_utils import get_group_term_obs_shape

    env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space[
        "policy"
    ].shape[-1]
    env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space[
        "critic"
    ].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space[
        "policy"
    ].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space[
        "critic"
    ].shape[-1]
    example_obs = env.reset(flatten_dict_obs=False)
    for key in env.env.observation_space:
        if key in ("policy", "critic"):
            continue
        dims, names, total = get_group_term_obs_shape(example_obs, key)
        env.config["obs"]["group_obs_dims"][key] = dims
        env.config["obs"]["group_obs_names"][key] = names
        env.config["obs"]["obs_dims"][key] = total
        env.config["robot"]["algo_obs_dim_dict"][key] = total
    if env.config.get("meta_action_dim", None) is not None:
        raise ValueError("DAgger v1 requires direct SONIC environment actions")
    env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]


def create_release_env(
    config: Any,
    *,
    device: str,
    args_cli: Any,
    reference_terms: list[str] | tuple[str, ...] | None = None,
) -> SonicDaggerEnv:
    print("[dagger] importing SONIC create_manager_env", flush=True)
    from gear_sonic.train_agent_trl import create_manager_env

    print("[dagger] calling create_manager_env", flush=True)
    raw_env = create_manager_env(config, device, args_cli)
    print("[dagger] create_manager_env returned", flush=True)
    prepare_env_dimensions(raw_env)
    print("[dagger] prepare_env_dimensions returned", flush=True)
    wrapped = SonicDaggerEnv(raw_env, reference_terms=reference_terms)
    print("[dagger] SonicDaggerEnv wrapped", flush=True)
    return wrapped

