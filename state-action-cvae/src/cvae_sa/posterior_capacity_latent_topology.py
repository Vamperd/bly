from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import PosteriorCapacityOutput, build_model, parameter_count
from .posterior_capacity import (
    DeterministicWindowSubset,
    FIXED_MASK_NAMES,
    MaskBankDataset,
    _device_batch,
    evaluate_exact,
    make_fixture_masks,
    reconstruction_loss,
    selected_window_identities,
    validate_motion_prefix,
)
from .posterior_capacity_ab import (
    _append_record,
    _infinite,
    _learning_rate_multiplier,
    _load_control_dt,
    _load_joint_names,
    _svg_series,
    batch_sample_identities,
    fixture_bitmap_sha256,
    identity_sha256,
    run_full_evaluation,
    validate_step0,
)
from .posterior_capacity_autodecoder import (
    EXPECTED_FIXTURES,
    EXPECTED_WINDOWS,
    WindowCodeAutoDecoder,
    _assert_encoder_isolated,
    _autodecoder_gate,
    _categorical_svg,
    _encoder_call_counters,
    evaluate_zero_code_dependence,
    validate_contract as validate_f4e_source_contract,
)
from .posterior_capacity_tail import EXPECTED_MOTIONS, EXPECTED_PARAMETERS, EXPECTED_WINDOW, evaluate_tail
from .util import (
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    canonical_json_bytes,
    file_sha256,
    load_json,
    seed_everything,
)


FORMAT_VERSION = "sonic_posterior_latent_topology_summary_v1"
CHECKPOINT_FORMAT = "sonic_posterior_latent_topology_checkpoint_v1"
COMPARISON_FORMAT = "sonic_posterior_latent_topology_comparison_v1"
FAILURE_FORMAT = "sonic_posterior_latent_topology_failure_v1"
ARMS = ("G8", "T129")
FIXTURE_SEED = 20260830
INITIALIZATION_SEED = 20260832
TRAINING_SEED = 20260831
COMPARISON_STEPS = (13000, 14000, 15000)
EXPECTED_F4E_RUN_NAME = "cvae_posterior_capacity_autodecoder_f4e_20260907_120414"
EXPECTED_F4E_ASSESSMENT = "E1_E2_FAIL_GLOBAL_CODE_DECODER_CAPACITY_UNPROVEN"
SMOKE_MARKER = "cvae_posterior_latent_topology_smoke.ok"
EXECUTION_MARKER = "cvae_posterior_latent_topology_execution.ok"
QUALITY_MARKER = "cvae_posterior_latent_topology_progression.ok"
COMPARISON_MARKER = "cvae_posterior_latent_topology_comparison.ok"
ARM_SHAPES = {
    "G8": (EXPECTED_WINDOWS, 8, 256),
    "T129": (EXPECTED_WINDOWS, EXPECTED_WINDOW + 1, 16),
}
ARM_CODE_PARAMETERS = {arm: math.prod(shape) for arm, shape in ARM_SHAPES.items()}
ARM_TOPOLOGY_PARAMETERS = {"G8": 8 * 384, "T129": 16 * 384 + 384}
ARM_TOTAL_PARAMETERS = {
    arm: EXPECTED_PARAMETERS + ARM_CODE_PARAMETERS[arm] + ARM_TOPOLOGY_PARAMETERS[arm]
    for arm in ARMS
}


def validate_config_contract(
    config: dict[str, Any], checkpoint: dict[str, Any], arm: str
) -> dict[str, bool]:
    """Reject any silent drift from the approved equal-budget F4F contract."""
    expected = {
        "format_version": "sonic_posterior_latent_topology_config_v1",
        "fixture_seed": FIXTURE_SEED,
        "initialization_seed": INITIALIZATION_SEED,
        "training_seed": TRAINING_SEED,
        "data.motion_count": EXPECTED_MOTIONS,
        "data.window_transitions": EXPECTED_WINDOW,
        "data.max_windows": None,
        "data.num_workers": 4,
        "model.kind": "physics_posterior_transformer",
        "model.d_model": 384,
        "model.encoder_layers": 6,
        "model.decoder_layers": 8,
        "model.heads": 8,
        "model.ffn_dim": 1536,
        "model.latent_dim": 256,
        "model.dropout": 0.0,
        "model.decoder_layer_latent_gates": False,
        "model.base_parameter_count": EXPECTED_PARAMETERS,
        "training.objective": "A",
        "training.mask_phase": "fixed",
        "training.precision": "FP32",
        "training.no_early_stop": True,
        "training.micro_batch": 4,
        "training.gradient_accumulation": 16,
        "training.max_optimizer_steps": 15000,
        "training.validation_interval": 1000,
        "training.warmup_steps": 250,
        "training.gradient_clip": 1.0,
        "training.weight_decay": 0.0,
        "training.code_learning_rate": 3e-4,
        "training.code_minimum_learning_rate": 1e-5,
        "training.decoder_learning_rate": 3e-5,
        "training.decoder_minimum_learning_rate": 1e-6,
        "training.kl_beta": 0.0,
        "training.free_bits": 0.0,
        "training.f4e_global_rmse_guard_multiplier": 1.1,
        "training.comparison_steps": list(COMPARISON_STEPS),
        "training.thresholds.state_rmse": 1e-4,
        "training.thresholds.action_rmse": 1e-4,
        "training.thresholds.continuous_max_abs": 1e-3,
        "training.thresholds.latent_ratio": 10.0,
        "training.progression_thresholds.state_rmse": 1e-2,
        "training.progression_thresholds.action_rmse": 1e-2,
        "training.progression_thresholds.continuous_max_abs": 1e-2,
        "training.progression_thresholds.latent_ratio": 10.0,
    }

    def value_at(path: str) -> Any:
        value: Any = config
        for key in path.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        return value

    checks = {path: value_at(path) == value for path, value in expected.items()}
    arm_record = config.get("model", {}).get("arms", {}).get(arm, {})
    checks.update(
        {
            "arm.code_shape": arm_record.get("code_shape") == list(ARM_SHAPES[arm]),
            "arm.code_parameter_count": int(arm_record.get("code_parameter_count", -1))
            == ARM_CODE_PARAMETERS[arm],
            "arm.topology_parameter_count": int(
                arm_record.get("topology_parameter_count", -1)
            )
            == ARM_TOPOLOGY_PARAMETERS[arm],
            "arm.total_parameter_count": int(arm_record.get("total_parameter_count", -1))
            == ARM_TOTAL_PARAMETERS[arm],
        }
    )
    source_model = checkpoint.get("config", {}).get("model", {})
    for key in (
        "kind",
        "d_model",
        "encoder_layers",
        "decoder_layers",
        "heads",
        "ffn_dim",
        "latent_dim",
        "dropout",
    ):
        checks[f"source_model.{key}"] = config.get("model", {}).get(key) == source_model.get(key)
    checks["source_model.decoder_layer_latent_gates"] = not bool(
        source_model.get("decoder_layer_latent_gates", False)
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"F4F fixed config contract mismatch: {failed}")
    return checks


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class WindowLatentTopologyAutoDecoder(nn.Module):
    """Mask-invariant per-window codes with one of the two F4F placements."""

    def __init__(
        self,
        base_model: nn.Module,
        arm: str,
        initial_codes: torch.Tensor,
        topology_initialization: dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        arm = arm.upper()
        if arm not in ARMS:
            raise ValueError(f"F4F arm must be one of {ARMS}")
        if tuple(initial_codes.shape) != ARM_SHAPES[arm]:
            raise ValueError(f"F4F {arm} code shape is incorrect")
        self.base_model = base_model
        self.arm = arm
        self.window_codes = nn.Parameter(initial_codes.clone())
        if arm == "G8":
            slots = topology_initialization.get("slot_embedding")
            if not isinstance(slots, torch.Tensor) or tuple(slots.shape) != (8, 384):
                raise ValueError("F4F G8 slot embedding must have shape [8, 384]")
            self.slot_embedding = nn.Parameter(slots.clone())
            self.time_projection = None
        else:
            weight = topology_initialization.get("time_projection_weight")
            bias = topology_initialization.get("time_projection_bias")
            if (
                not isinstance(weight, torch.Tensor)
                or tuple(weight.shape) != (384, 16)
                or not isinstance(bias, torch.Tensor)
                or tuple(bias.shape) != (384,)
            ):
                raise ValueError("F4F T129 projection initialization is malformed")
            self.slot_embedding = None
            self.time_projection = nn.Linear(16, 384)
            with torch.no_grad():
                self.time_projection.weight.copy_(weight)
                self.time_projection.bias.copy_(bias)

    @staticmethod
    def code_indices(batch: dict[str, Any]) -> torch.Tensor:
        return WindowCodeAutoDecoder.code_indices(batch)

    def codes(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.window_codes[self.code_indices(batch)]

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        state_mask: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        latent_override: torch.Tensor | None = None,
        use_prior: bool = False,
    ) -> PosteriorCapacityOutput:
        if use_prior:
            raise ValueError("F4F has no conditional-prior path")
        code = self.codes(batch)
        latent = code if latent_override is None else latent_override
        if latent.shape != code.shape:
            raise ValueError(f"F4F {self.arm} latent override shape is incorrect")
        if self.arm == "G8":
            assert self.slot_embedding is not None
            prefix = self.base_model.latent_projection(latent) + self.slot_embedding[None]
            decoded = self.base_model.decode_from_latent_topology(
                batch,
                state_mask,
                action_mask,
                prefix_tokens=prefix,
            )
        else:
            assert self.time_projection is not None
            decoded = self.base_model.decode_from_latent_topology(
                batch,
                state_mask,
                action_mask,
                time_conditions=self.time_projection(latent),
            )
        zeros = torch.zeros_like(code)
        return PosteriorCapacityOutput(
            physical_state=decoded.physical_state,
            action=decoded.action,
            state_contact_logits=decoded.state_contact_logits,
            posterior_mean=code,
            posterior_logvar=zeros,
            prior_mean=zeros,
            prior_logvar=zeros,
            latent=latent,
        )


def initialize_topology(
    arm: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    arm = arm.upper()
    if arm not in ARMS:
        raise ValueError(f"F4F arm must be one of {ARMS}")
    code_generator = torch.Generator().manual_seed(INITIALIZATION_SEED)
    topology_generator = torch.Generator().manual_seed(INITIALIZATION_SEED + 1)
    codes = torch.randn(ARM_SHAPES[arm], generator=code_generator) * 0.02
    if arm == "G8":
        tensors = {
            "slot_embedding": torch.randn((8, 384), generator=topology_generator) * 0.02,
        }
    else:
        tensors = {
            "time_projection_weight": (
                torch.randn((384, 16), generator=topology_generator) * 0.02
            ),
            "time_projection_bias": torch.zeros(384),
        }
    tensor_hashes = {name: _tensor_sha256(value) for name, value in tensors.items()}
    manifest = {
        "format_version": "sonic_posterior_latent_topology_initialization_v1",
        "arm": arm,
        "initialization_seed": INITIALIZATION_SEED,
        "topology_subseed": INITIALIZATION_SEED + 1,
        "distribution": "independent normal mean=0 std=0.02; projection bias=zeros",
        "code_shape": list(codes.shape),
        "code_parameter_count": codes.numel(),
        "code_scalars_per_window": codes[0].numel(),
        "code_sha256": _tensor_sha256(codes),
        "code_rms": float(torch.sqrt(torch.square(codes).mean())),
        "code_max_abs": float(codes.abs().max()),
        "topology_tensor_sha256": tensor_hashes,
        "initialization_sha256": hashlib.sha256(
            canonical_json_bytes({"code": _tensor_sha256(codes), "topology": tensor_hashes})
        ).hexdigest(),
    }
    return codes, tensors, manifest


def _load_f4e_authorization(
    *,
    dataset_run: Path,
    source_checkpoint: Path,
    f4e_run: Path,
    output_run: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    f4e_run = f4e_run.expanduser().resolve()
    resolved_output = output_run.expanduser().resolve()
    if resolved_output == f4e_run or resolved_output.is_relative_to(f4e_run):
        raise ValueError("F4F output must be isolated from the protected F4E run")
    if f4e_run.name != EXPECTED_F4E_RUN_NAME:
        raise ValueError(f"F4F requires formal F4E run {EXPECTED_F4E_RUN_NAME}")
    summary_path = f4e_run / "manifests/posterior_autodecoder_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"F4E summary is missing: {summary_path}")
    if not (f4e_run / "markers/cvae_posterior_autodecoder_execution.ok").is_file():
        raise ValueError("F4E execution marker is missing")
    failure_text = (f4e_run / "markers/cvae.failed").read_text(encoding="utf-8").strip()
    if failure_text != "QUALITY_FAIL execution_complete=true E1=false E2=false":
        raise ValueError("F4E quality-failure marker does not authorize F4F")
    f4e = load_json(summary_path)
    stages = f4e.get("training_contract", {}).get("stages", {})
    checks = {
        "execution_pass": bool(f4e.get("execution_pass")),
        "quality_fail": not bool(f4e.get("quality_pass")),
        "formal": not bool(f4e.get("smoke")),
        "assessment": f4e.get("root_cause_assessment") == EXPECTED_F4E_ASSESSMENT,
        "steps": int(f4e.get("training_contract", {}).get("completed_optimizer_steps", -1)) == 20000,
        "e1_failed": stages.get("E1", {}).get("quality_pass") is False,
        "e2_failed": stages.get("E2", {}).get("quality_pass") is False,
        "source_path": Path(str(f4e.get("source", {}).get("checkpoint", ""))).resolve()
        == source_checkpoint.expanduser().resolve(),
        "source_hash": f4e.get("source", {}).get("checkpoint_sha256")
        == file_sha256(source_checkpoint),
        "dataset_path": Path(str(f4e.get("dataset_run", ""))).resolve()
        == dataset_run.expanduser().resolve(),
        "dataset_hash": f4e.get("dataset_manifest_sha256")
        == file_sha256(dataset_run / "manifests/dataset_manifest.json"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"F4F F4E authorization failed: {failed}")
    f4a_run = Path(str(f4e.get("source", {}).get("f4a_run", "")))
    trigger_record = f4e.get("source", {}).get("trigger_comparison", {})
    trigger_run = Path(
        str(trigger_record.get("run", ""))
        if isinstance(trigger_record, dict)
        else str(trigger_record)
    )
    f4e_config = load_json(
        Path(__file__).resolve().parents[2] / "configs/posterior_capacity_autodecoder.json"
    )
    checkpoint, source_summary, f4a_manifest, _ = validate_f4e_source_contract(
        dataset_run=dataset_run,
        source_checkpoint=source_checkpoint,
        f4a_run=f4a_run,
        trigger_run=trigger_run,
        output_run=output_run,
        config=f4e_config,
    )
    validate_config_contract(config, checkpoint, str(config.get("topology_arm", "")))
    best = f4e.get("best_evaluation", {})
    reconstruction = best.get("exact", {}).get("reconstruction_loss", {})
    baseline = {
        "run": str(f4e_run),
        "summary": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "best_optimizer_step": int(f4e.get("best_optimizer_step", -1)),
        "state_global_rmse": math.sqrt(float(reconstruction.get("state", math.inf))),
        "action_global_rmse": math.sqrt(float(reconstruction.get("action", math.inf))),
        "worst_state_rmse": float(best.get("exact", {}).get("worst_state_rmse", math.inf)),
        "worst_action_rmse": float(best.get("exact", {}).get("worst_action_rmse", math.inf)),
        "continuous_max_abs": float(best.get("exact", {}).get("continuous_max_abs", math.inf)),
        "threshold_exceed_fraction": float(
            best.get("tail_global", {}).get("threshold_exceed_fraction", math.inf)
        ),
        "authorization_checks": checks,
    }
    if not all(math.isfinite(float(value)) for key, value in baseline.items() if key.endswith(("rmse", "abs", "fraction"))):
        raise ValueError("F4E baseline metrics are incomplete")
    return checkpoint, source_summary, f4a_manifest, baseline


def configure_trainable_parameters(
    model: WindowLatentTopologyAutoDecoder,
) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    topology_prefixes = ["window_codes"]
    if model.arm == "G8":
        topology_prefixes += ["slot_embedding", "base_model.latent_projection."]
    else:
        topology_prefixes += ["time_projection."]
    decoder_prefixes = [
        "base_model.state_input.",
        "base_model.action_input.",
        "base_model.type_embedding.",
        "base_model.decoder.",
        "base_model.state_continuous_output.",
        "base_model.state_contact_output.",
        "base_model.action_output.",
    ]
    topology_names: list[str] = []
    decoder_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in topology_prefixes):
            parameter.requires_grad_(True)
            topology_names.append(name)
        elif any(name.startswith(prefix) for prefix in decoder_prefixes):
            parameter.requires_grad_(True)
            decoder_names.append(name)
        else:
            frozen_names.append(name)
    if not topology_names or not decoder_names:
        raise ValueError("F4F trainable parameter groups may not be empty")
    trainable_names = topology_names + decoder_names
    if any(
        name.startswith(
            ("base_model.encoder_cls", "base_model.encoder.", "base_model.posterior.", "base_model.prior.")
        )
        for name in trainable_names
    ):
        raise ValueError("F4F trainable allowlist violated encoder isolation")
    named = dict(model.named_parameters())
    return {
        "arm": model.arm,
        "topology_names": topology_names,
        "decoder_names": decoder_names,
        "frozen_names": frozen_names,
        "topology_parameter_count": sum(named[name].numel() for name in topology_names),
        "decoder_parameter_count": sum(named[name].numel() for name in decoder_names),
        "trainable_parameter_count": sum(named[name].numel() for name in trainable_names),
        "sha256": hashlib.sha256(canonical_json_bytes(trainable_names)).hexdigest(),
    }


def _optimizer(
    model: WindowLatentTopologyAutoDecoder,
    contract: dict[str, Any],
    config: dict[str, Any],
    max_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    named = dict(model.named_parameters())
    training = config["training"]
    topology_lr = float(training["code_learning_rate"])
    decoder_lr = float(training["decoder_learning_rate"])
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [named[name] for name in contract["topology_names"]],
                "lr": topology_lr,
                "name": "topology",
            },
            {
                "params": [named[name] for name in contract["decoder_names"]],
                "lr": decoder_lr,
                "name": "decoder_side",
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    warmup = int(training["warmup_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [
            lambda step: _learning_rate_multiplier(
                step,
                warmup_steps=warmup,
                max_steps=max_steps,
                minimum_ratio=float(training["code_minimum_learning_rate"]) / topology_lr,
            ),
            lambda step: _learning_rate_multiplier(
                step,
                warmup_steps=warmup,
                max_steps=max_steps,
                minimum_ratio=float(training["decoder_minimum_learning_rate"]) / decoder_lr,
            ),
        ],
    )
    return optimizer, scheduler


def _training_loader(
    fixture_data: MaskBankDataset,
    *,
    micro_batch: int,
    workers: int,
    device: torch.device,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        fixture_data,
        batch_size=micro_batch,
        shuffle=True,
        num_workers=workers,
        generator=torch.Generator().manual_seed(TRAINING_SEED),
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def _quality_gate(evaluation: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    topology = evaluation["topology_progression_gate"]
    reconstruction = evaluation["exact"]["reconstruction_loss"]
    state_global = math.sqrt(float(reconstruction["state"]))
    action_global = math.sqrt(float(reconstruction["action"]))
    state_limit = float(baseline["state_global_rmse"]) * 1.10
    action_limit = float(baseline["action_global_rmse"]) * 1.10
    ratios = {
        **{key: float(value) for key, value in topology["threshold_ratios"].items()},
        "global_state_guard": state_global / state_limit,
        "global_action_guard": action_global / action_limit,
    }
    score = max(ratios.values())
    return {
        "passed": bool(math.isfinite(score) and score <= 1.0),
        "score": score,
        "threshold_ratios": ratios,
        "global_state_rmse": state_global,
        "global_action_rmse": action_global,
        "global_state_limit": state_limit,
        "global_action_limit": action_limit,
    }


@torch.no_grad()
def evaluate_topology(
    *,
    model: WindowLatentTopologyAutoDecoder,
    validation_loader: DataLoader[dict[str, Any]],
    base_loader: DataLoader[dict[str, Any]],
    selected_base: DeterministicWindowSubset,
    device: torch.device,
    config: dict[str, Any],
    joint_names: list[str],
    fixed_cases: list[dict[str, Any]],
    output_run: Path,
    step: int,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    evaluation = run_full_evaluation(
        model=model,
        validation_loader=validation_loader,
        base_loader=base_loader,
        selected_base=selected_base,
        device=device,
        config=config,
        joint_names=joint_names,
        fixed_cases=fixed_cases,
        output_run=output_run,
        step=step,
    )
    zero = evaluate_zero_code_dependence(model, base_loader, device)
    zero_path = output_run / f"data/step_{step:05d}_zero_code_dependence.json"
    atomic_write_json(zero_path, zero)
    evaluation["zero_code_dependence"] = {**zero, "artifact": str(zero_path)}
    evaluation["arm"] = model.arm
    evaluation["topology_progression_gate"] = _autodecoder_gate(evaluation)
    evaluation["quality_gate"] = _quality_gate(evaluation, baseline)
    evaluation["evaluation_scope"] = (
        f"F4F {model.arm} identity-conditioned topology on the same 80 windows and 800 fixed Masks"
    )
    return evaluation


def _render_plots(
    output_run: Path,
    records: list[dict[str, Any]],
    best: dict[str, Any],
    arm: str,
) -> dict[str, str]:
    train = [row for row in records if row["phase"] == "train"]
    evaluations = [row for row in records if row["phase"] == "evaluation"]
    if len(train) > 2000:
        stride = max(1, len(train) // 2000)
        train_plot = train[::stride]
    else:
        train_plot = train

    def ema() -> list[tuple[float, float]]:
        result: list[tuple[float, float]] = []
        running: float | None = None
        for row in train:
            value = float(row["raw_reconstruction"]["total"])
            running = value if running is None else 0.05 * value + 0.95 * running
            result.append((row["optimizer_step"], running))
        return result

    training_path = output_run / "plots/training_curves.svg"
    gate_path = output_run / "plots/gate_curves.svg"
    optimizer_path = output_run / "plots/optimizer_curves.svg"
    mask_path = output_run / "plots/mask_breakdown.svg"
    dependence_path = output_run / "plots/code_dependence.svg"
    atomic_write_text(
        training_path,
        _svg_series(
            f"F4F {arm} training and full fixed-fixture evaluation",
            [
                ("Train batch raw", [(r["optimizer_step"], r["raw_reconstruction"]["total"]) for r in train_plot], "#94a3b8"),
                ("Train EMA α=0.05", ema(), "#2563eb"),
                ("Full fixed-fixture evaluation", [(r["optimizer_step"], r["exact"]["reconstruction_loss"]["total"]) for r in evaluations], "#dc2626"),
                ("Evaluation State MSE", [(r["optimizer_step"], r["exact"]["reconstruction_loss"]["state"]) for r in evaluations], "#0284c7"),
                ("Evaluation Action MSE", [(r["optimizer_step"], r["exact"]["reconstruction_loss"]["action"]) for r in evaluations], "#059669"),
                ("Evaluation contact BCE", [(r["optimizer_step"], r["exact"]["reconstruction_loss"]["contact"]) for r in evaluations], "#c2410c"),
            ],
            y_label="Value (log10 scale)",
            log_y=True,
        ),
    )
    atomic_write_text(
        optimizer_path,
        _svg_series(
            f"F4F {arm} optimizer diagnostics",
            [
                ("Topology learning rate", [(r["optimizer_step"], r["learning_rates"]["topology"]) for r in train], "#7c3aed"),
                ("Decoder learning rate", [(r["optimizer_step"], r["learning_rates"]["decoder_side"]) for r in train], "#a855f7"),
                ("Gradient norm before clip", [(r["optimizer_step"], r["gradient_norm_before_clip"]) for r in train], "#111827"),
            ],
            y_label="Value (log10 scale)",
            log_y=True,
            horizontal_lines=[("gradient clip = 1", 1.0, "#b91c1c")],
        ),
    )
    atomic_write_text(
        gate_path,
        _svg_series(
            f"F4F {arm} progression metrics",
            [
                ("Worst State RMSE", [(r["optimizer_step"], r["exact"]["worst_state_rmse"]) for r in evaluations], "#2563eb"),
                ("Worst Action RMSE", [(r["optimizer_step"], r["exact"]["worst_action_rmse"]) for r in evaluations], "#059669"),
                ("Continuous max abs", [(r["optimizer_step"], r["exact"]["continuous_max_abs"]) for r in evaluations], "#dc2626"),
                ("Element exceed fraction", [(r["optimizer_step"], r["tail_global"]["threshold_exceed_fraction"]) for r in evaluations], "#7c3aed"),
                ("Global State RMSE", [(r["optimizer_step"], r["quality_gate"]["global_state_rmse"]) for r in evaluations], "#0ea5e9"),
                ("Global Action RMSE", [(r["optimizer_step"], r["quality_gate"]["global_action_rmse"]) for r in evaluations], "#10b981"),
            ],
            y_label="Value (log10 scale)",
            log_y=True,
            horizontal_lines=[
                ("progression RMSE/max = 1e-2", 1e-2, "#b91c1c"),
                ("exact max = 1e-3", 1e-3, "#d97706"),
                ("exact RMSE = 1e-4", 1e-4, "#059669"),
            ],
        ),
    )
    atomic_write_text(
        dependence_path,
        _svg_series(
            f"F4F {arm} full-both code dependence",
            [
                ("Zero code ratio", [(r["optimizer_step"], r["topology_progression_gate"]["zero_ratio"]) for r in evaluations], "#0f766e"),
                ("Cross-window ratio", [(r["optimizer_step"], r["topology_progression_gate"]["cross_window_ratio"]) for r in evaluations], "#7c3aed"),
                ("Cross-motion ratio", [(r["optimizer_step"], r["topology_progression_gate"]["cross_motion_ratio"]) for r in evaluations], "#c2410c"),
            ],
            y_label="Ratio (log10 scale)",
            log_y=True,
            horizontal_lines=[("required ratio = 10", 10.0, "#b91c1c")],
        ),
    )
    rows = []
    for name in FIXED_MASK_NAMES:
        case = best["exact"]["cases"][name]
        rows.append(
            {
                "name": name,
                "state": float(case["worst_state_rmse"]) / 1e-2,
                "action": float(case["worst_action_rmse"]) / 1e-2,
                "max_abs": float(case["continuous_max_abs"]) / 1e-2,
            }
        )
    atomic_write_text(mask_path, _categorical_svg(f"F4F {arm} best Mask gate ratios", rows))
    return {
        "training_curves": str(training_path),
        "optimizer_curves": str(optimizer_path),
        "gate_curves": str(gate_path),
        "mask_breakdown": str(mask_path),
        "code_dependence": str(dependence_path),
    }


def _checkpoint_payload(
    *,
    model: WindowLatentTopologyAutoDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    step: int,
    best_score: float,
    dataset_hash: str,
    source_hash: str,
    f4e_hash: str,
    fixture_hash: str,
    initialization_hash: str,
    trainable_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT,
        "arm": model.arm,
        "optimizer_step": int(step),
        "best_progression_score": float(best_score),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "resolved_config": config,
        "dataset_manifest_sha256": dataset_hash,
        "source_checkpoint_sha256": source_hash,
        "f4e_summary_sha256": f4e_hash,
        "fixture_bitmap_sha256": fixture_hash,
        "initialization_sha256": initialization_hash,
        "code_shape": list(ARM_SHAPES[model.arm]),
        "code_parameter_count": ARM_CODE_PARAMETERS[model.arm],
        "total_parameter_count": _parameter_count(model),
        "trainable_contract": trainable_contract,
    }


def validate_saved_checkpoint(
    path: Path,
    *,
    arm: str,
    expected_step: int,
    dataset_hash: str,
    source_hash: str,
    f4e_hash: str,
    fixture_hash: str,
    initialization_hash: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model", {})
    code = model_state.get("window_codes")
    if arm == "G8":
        topology_shapes = {
            "slot_embedding": (8, 384),
            "base_model.latent_projection.weight": (384, 256),
            "base_model.latent_projection.bias": (384,),
        }
    else:
        topology_shapes = {
            "time_projection.weight": (384, 16),
            "time_projection.bias": (384,),
        }
    topology_present = all(
        isinstance(model_state.get(name), torch.Tensor)
        and tuple(model_state[name].shape) == shape
        for name, shape in topology_shapes.items()
    )
    finite_state = bool(model_state) and all(
        not isinstance(value, torch.Tensor) or bool(torch.isfinite(value).all())
        for value in model_state.values()
    )
    checks = {
        "format_version": checkpoint.get("format_version") == CHECKPOINT_FORMAT,
        "arm": checkpoint.get("arm") == arm,
        "optimizer_step": int(checkpoint.get("optimizer_step", -1)) == expected_step,
        "dataset_hash": checkpoint.get("dataset_manifest_sha256") == dataset_hash,
        "source_hash": checkpoint.get("source_checkpoint_sha256") == source_hash,
        "f4e_hash": checkpoint.get("f4e_summary_sha256") == f4e_hash,
        "fixture_hash": checkpoint.get("fixture_bitmap_sha256") == fixture_hash,
        "initialization_hash": checkpoint.get("initialization_sha256") == initialization_hash,
        "code_shape": isinstance(code, torch.Tensor) and tuple(code.shape) == ARM_SHAPES[arm],
        "topology_state": topology_present,
        "finite_model_state": finite_state,
        "code_parameters": int(checkpoint.get("code_parameter_count", -1)) == ARM_CODE_PARAMETERS[arm],
        "total_parameters": int(checkpoint.get("total_parameter_count", -1)) == ARM_TOTAL_PARAMETERS[arm],
        "model_state": bool(checkpoint.get("model")),
        "optimizer_state": bool(checkpoint.get("optimizer")),
        "scheduler_state": bool(checkpoint.get("scheduler")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"F4F checkpoint readback failed: {failed}")
    return {"passed": True, "checks": checks, "sha256": file_sha256(path)}


def _last_three_quality(evaluations: list[dict[str, Any]]) -> bool:
    by_step = {int(row["optimizer_step"]): row for row in evaluations}
    return all(
        step in by_step and bool(by_step[step]["quality_gate"]["passed"])
        for step in COMPARISON_STEPS
    )


def run_experiment(
    *,
    dataset_run: Path,
    source_checkpoint: Path,
    f4e_run: Path,
    output_run: Path,
    config: dict[str, Any],
    arm: str,
    smoke: bool,
) -> dict[str, Any]:
    from .dataset import StateActionWindowDataset

    arm = arm.upper()
    if arm not in ARMS:
        raise ValueError(f"F4F arm must be one of {ARMS}")
    dataset_run = dataset_run.expanduser().resolve()
    source_checkpoint = source_checkpoint.expanduser().resolve()
    f4e_run = f4e_run.expanduser().resolve()
    output_run = output_run.expanduser().resolve()
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "plots", "videos"):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    resolved = copy.deepcopy(config)
    resolved["arm"] = "A"
    resolved["topology_arm"] = arm
    resolved["training"]["tail_fraction"] = 0.2
    resolved["training"]["tail_mix"] = 0.5
    checkpoint, source_summary, f4a_manifest, baseline = _load_f4e_authorization(
        dataset_run=dataset_run,
        source_checkpoint=source_checkpoint,
        f4e_run=f4e_run,
        output_run=output_run,
        config=resolved,
    )
    dataset_hash = file_sha256(dataset_run / "manifests/dataset_manifest.json")
    source_hash = file_sha256(source_checkpoint)
    f4e_hash = str(baseline["summary_sha256"])
    base = StateActionWindowDataset(
        dataset_run,
        "train",
        EXPECTED_WINDOW,
        EXPECTED_WINDOW,
        max_episodes=EXPECTED_MOTIONS * 8,
        random_crop=False,
    )
    try:
        selected_motions = validate_motion_prefix(base, EXPECTED_MOTIONS)
        selected_base = DeterministicWindowSubset(base, None)
        selected_windows = selected_window_identities(base, selected_base.indices)
        fixture_data = MaskBankDataset(selected_base, len(FIXED_MASK_NAMES))
        if len(selected_base) != EXPECTED_WINDOWS or len(fixture_data) != EXPECTED_FIXTURES:
            raise ValueError("F4F requires exactly 80 windows and 800 fixtures")
        if selected_motions != list(source_summary.get("selected_motion_keys", [])):
            raise ValueError("F4F selected motions differ from F4D")
        if selected_windows != list(source_summary.get("selected_windows", [])):
            raise ValueError("F4F selected windows differ from F4D")
        resolved["model"]["state_dim"] = base.state_dim
        resolved["control_dt"] = _load_control_dt(base)
        base_model = build_model(resolved["model"])
        if parameter_count(base_model) != EXPECTED_PARAMETERS:
            raise ValueError("F4F base model parameter count mismatch")
        base_model.load_state_dict(checkpoint["model"], strict=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model.to(device)
        workers = 0 if smoke else int(resolved["data"].get("num_workers", 4))
        micro_batch = int(resolved["training"]["micro_batch"])
        validation_loader = DataLoader(
            fixture_data,
            batch_size=micro_batch,
            shuffle=False,
            num_workers=workers,
            generator=torch.Generator().manual_seed(FIXTURE_SEED + 1),
            drop_last=False,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )
        base_loader = DataLoader(
            selected_base,
            batch_size=micro_batch,
            shuffle=False,
            num_workers=workers,
            generator=torch.Generator().manual_seed(FIXTURE_SEED + 2),
            drop_last=False,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )
        fixture_hash = fixture_bitmap_sha256(validation_loader, FIXTURE_SEED)
        selected_hash = identity_sha256(selected_windows)
        f4e_data = load_json(Path(str(baseline["summary"]))).get("data_contract", {})
        if fixture_hash != f4e_data.get("fixture_bitmap_sha256"):
            raise ValueError("F4F fixture bitmap differs from F4E")
        if selected_hash != f4e_data.get("selected_windows_sha256"):
            raise ValueError("F4F selected windows differ from F4E")
        identity_path = output_run / "manifests/posterior_latent_topology_identity.json"
        atomic_write_json(
            identity_path,
            {
                "fixture_seed": FIXTURE_SEED,
                "selected_motion_keys": selected_motions,
                "window_transitions": EXPECTED_WINDOW,
                "selected_windows": selected_windows,
                "mask_names": list(FIXED_MASK_NAMES),
                "fixture_bitmap_sha256": fixture_hash,
                "selected_windows_sha256": selected_hash,
                "shared_window_code": True,
                "per_fixture_code": False,
            },
        )
        joint_names = _load_joint_names(base)
        source_exact = evaluate_exact(
            base_model,
            validation_loader,
            device,
            FIXTURE_SEED,
            False,
            {key: float(value) for key, value in resolved["training"]["thresholds"].items()},
            {key: float(value) for key, value in resolved["training"]["progression_thresholds"].items()},
            "progression",
        )
        source_tail = evaluate_tail(
            base_model,
            validation_loader,
            device,
            seed=FIXTURE_SEED,
            state_mean=selected_base.base.state_mean,
            state_std=selected_base.base.state_std,
            action_mean=selected_base.base.action_mean,
            action_std=selected_base.base.action_std,
            joint_names=joint_names,
        )
        source_reproduction = validate_step0(
            source_exact, {"global": source_tail["global"]}, source_summary, checkpoint
        )
        initial_codes, topology_tensors, initialization = initialize_topology(arm)
        model = WindowLatentTopologyAutoDecoder(
            base_model, arm, initial_codes, topology_tensors
        ).to(device)
        if _parameter_count(model) != ARM_TOTAL_PARAMETERS[arm]:
            raise ValueError(f"F4F {arm} total parameter count mismatch")
        initialization_path = output_run / "manifests/posterior_latent_topology_initialization.json"
        tensor_path = output_run / "data/posterior_latent_topology_initialization.pt"
        atomic_torch_save(
            tensor_path,
            {"window_codes": initial_codes, **topology_tensors},
        )
        initialization["tensor_artifact"] = str(tensor_path)
        initialization["tensor_artifact_sha256"] = file_sha256(tensor_path)
        atomic_write_json(initialization_path, initialization)
        f4e_summary = load_json(Path(str(baseline["summary"])))
        f4e_fixed_path = f4e_run / "manifests/fixed_velocity_cases.json"
        if not f4e_fixed_path.is_file():
            raise FileNotFoundError("F4F requires the formal F4E fixed velocity cases")
        f4e_fixed = load_json(f4e_fixed_path)
        fixed_cases = list(f4e_fixed.get("cases", []))
        fixed_case_hash = identity_sha256(fixed_cases)
        if len(fixed_cases) != 5:
            raise ValueError("F4F requires the five F4E/F4A difficult curve cases")
        if fixed_case_hash != f4e_fixed.get("sha256"):
            raise ValueError("F4F F4E fixed velocity cases hash is malformed")
        if fixed_case_hash != f4e_summary.get("data_contract", {}).get(
            "fixed_velocity_cases_sha256"
        ):
            raise ValueError("F4F fixed velocity cases differ from the F4E baseline")
        atomic_write_json(
            output_run / "manifests/fixed_velocity_cases.json",
            {"sha256": fixed_case_hash, "cases": fixed_cases},
        )
        trainable_contract = configure_trainable_parameters(model)
        encoder_counts, handles = _encoder_call_counters(base_model)
        records: list[dict[str, Any]] = []
        evaluations: list[dict[str, Any]] = []
        metrics_path = output_run / "logs/metrics.jsonl"
        step0 = evaluate_topology(
            model=model,
            validation_loader=validation_loader,
            base_loader=base_loader,
            selected_base=selected_base,
            device=device,
            config=resolved,
            joint_names=joint_names,
            fixed_cases=fixed_cases,
            output_run=output_run,
            step=0,
            baseline=baseline,
        )
        step0["training_context"] = {"learning_rates": None, "gradient_norm_before_clip": None}
        records.append(step0)
        evaluations.append(step0)
        _append_record(metrics_path, step0)
        best = step0
        best_score = float(step0["quality_gate"]["score"])
        max_steps = 2 if smoke else int(resolved["training"]["max_optimizer_steps"])
        validation_interval = 2 if smoke else int(resolved["training"]["validation_interval"])
        accumulation = int(resolved["training"]["gradient_accumulation"])
        gradient_clip = float(resolved["training"]["gradient_clip"])
        seed_everything(TRAINING_SEED)
        train_loader = _training_loader(
            fixture_data,
            micro_batch=micro_batch,
            workers=workers,
            device=device,
        )
        stream = _infinite(train_loader)
        optimizer, scheduler = _optimizer(model, trainable_contract, resolved, max_steps)
        atomic_torch_save(
            output_run / "checkpoints/best_progression.pt",
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=resolved,
                step=0,
                best_score=best_score,
                dataset_hash=dataset_hash,
                source_hash=source_hash,
                f4e_hash=f4e_hash,
                fixture_hash=fixture_hash,
                initialization_hash=initialization["initialization_sha256"],
                trainable_contract=trainable_contract,
            ),
        )
        optimizer.zero_grad(set_to_none=True)
        model.train()
        started = time.monotonic()
        for step in range(1, max_steps + 1):
            step_started = time.monotonic()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            sums = {"total": 0.0, "state": 0.0, "action": 0.0, "contact": 0.0}
            sampled: list[dict[str, Any]] = []
            for _ in range(accumulation):
                cpu_batch = next(stream)
                sampled.extend(batch_sample_identities(cpu_batch))
                state_mask, action_mask, _ = make_fixture_masks(cpu_batch, FIXTURE_SEED)
                batch = _device_batch(cpu_batch, device)
                state_mask = state_mask.to(device)
                action_mask = action_mask.to(device)
                output = model(batch, state_mask, action_mask)
                loss = reconstruction_loss(output, batch, state_mask, action_mask)
                if not bool(torch.isfinite(loss.total)):
                    raise FloatingPointError("F4F reconstruction loss became non-finite")
                (loss.total / accumulation).backward()
                for name in sums:
                    sums[name] += float(getattr(loss, name).detach().cpu()) / accumulation
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, gradient_clip)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("F4F gradient norm became non-finite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            train_record = {
                "phase": "train",
                "arm": arm,
                "optimizer_step": step,
                "sample_identity_sha256": identity_sha256(sampled),
                "sample_count": len(sampled),
                "raw_reconstruction": dict(sums),
                "optimization_objective": dict(sums),
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "gradient_clip_threshold": gradient_clip,
                "gradient_was_clipped": float(gradient_norm.detach().cpu()) > gradient_clip,
                "learning_rates": {
                    str(group.get("name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
                "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                "step_seconds": time.monotonic() - step_started,
            }
            records.append(train_record)
            _append_record(metrics_path, train_record)
            if step % validation_interval:
                continue
            evaluation = evaluate_topology(
                model=model,
                validation_loader=validation_loader,
                base_loader=base_loader,
                selected_base=selected_base,
                device=device,
                config=resolved,
                joint_names=joint_names,
                fixed_cases=fixed_cases,
                output_run=output_run,
                step=step,
                baseline=baseline,
            )
            evaluation["training_context"] = {
                "learning_rates": train_record["learning_rates"],
                "gradient_norm_before_clip": train_record["gradient_norm_before_clip"],
                "gradient_was_clipped": train_record["gradient_was_clipped"],
            }
            records.append(evaluation)
            evaluations.append(evaluation)
            _append_record(metrics_path, evaluation)
            score = float(evaluation["quality_gate"]["score"])
            if score < best_score:
                best_score = score
                best = evaluation
                atomic_torch_save(
                    output_run / "checkpoints/best_progression.pt",
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=resolved,
                        step=step,
                        best_score=best_score,
                        dataset_hash=dataset_hash,
                        source_hash=source_hash,
                        f4e_hash=f4e_hash,
                        fixture_hash=fixture_hash,
                        initialization_hash=initialization["initialization_sha256"],
                        trainable_contract=trainable_contract,
                    ),
                )
            atomic_torch_save(
                output_run / "checkpoints/last.pt",
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=resolved,
                    step=step,
                    best_score=best_score,
                    dataset_hash=dataset_hash,
                    source_hash=source_hash,
                    f4e_hash=f4e_hash,
                    fixture_hash=fixture_hash,
                    initialization_hash=initialization["initialization_sha256"],
                    trainable_contract=trainable_contract,
                ),
            )
            _render_plots(output_run, records, best, arm)
            model.train()
        isolation = _assert_encoder_isolated(model, encoder_counts)
        for handle in handles:
            handle.remove()
        quality_pass = bool(not smoke and _last_three_quality(evaluations))
        last_checkpoint = validate_saved_checkpoint(
            output_run / "checkpoints/last.pt",
            arm=arm,
            expected_step=max_steps,
            dataset_hash=dataset_hash,
            source_hash=source_hash,
            f4e_hash=f4e_hash,
            fixture_hash=fixture_hash,
            initialization_hash=initialization["initialization_sha256"],
        )
        best_checkpoint = validate_saved_checkpoint(
            output_run / "checkpoints/best_progression.pt",
            arm=arm,
            expected_step=int(best["optimizer_step"]),
            dataset_hash=dataset_hash,
            source_hash=source_hash,
            f4e_hash=f4e_hash,
            fixture_hash=fixture_hash,
            initialization_hash=initialization["initialization_sha256"],
        )
        plots = _render_plots(output_run, records, best, arm)
        summary = {
            "format_version": FORMAT_VERSION,
            "scope": "seen 80-window identity-conditioned topology capacity only",
            "execution_pass": True,
            "quality_pass": quality_pass,
            "smoke": bool(smoke),
            "arm": arm,
            "dataset_run": str(dataset_run),
            "dataset_manifest_sha256": dataset_hash,
            "source": {
                "checkpoint": str(source_checkpoint),
                "checkpoint_sha256": source_hash,
                "source_reproduction": source_reproduction,
                "f4e_baseline": baseline,
            },
            "data_contract": {
                "motion_count": EXPECTED_MOTIONS,
                "selected_motion_keys": selected_motions,
                "window_transitions": EXPECTED_WINDOW,
                "window_count": len(selected_base),
                "fixture_count": len(fixture_data),
                "fixture_bitmap_sha256": fixture_hash,
                "selected_windows_sha256": selected_hash,
                "identity_contract_sha256": file_sha256(identity_path),
                "fixed_velocity_cases_sha256": fixed_case_hash,
            },
            "model_contract": {
                "base_parameter_count": EXPECTED_PARAMETERS,
                "code_shape": list(ARM_SHAPES[arm]),
                "code_parameter_count": ARM_CODE_PARAMETERS[arm],
                "code_scalars_per_window": math.prod(ARM_SHAPES[arm][1:]),
                "topology_parameter_count": ARM_TOPOLOGY_PARAMETERS[arm],
                "total_parameter_count": ARM_TOTAL_PARAMETERS[arm],
                "shared_code_per_window": True,
                "per_fixture_code": False,
                "decoder_path_bypasses_encoders": True,
                "kl_beta": 0.0,
                "dropout": 0.0,
                "weight_decay": 0.0,
            },
            "initialization": {**initialization, "manifest": str(initialization_path)},
            "training_contract": {
                "training_seed": TRAINING_SEED,
                "fixture_seed": FIXTURE_SEED,
                "objective": "State MSE + Action MSE + contact BCE; equal mean of present components",
                "mask_phase": "fixed",
                "optimizer": "AdamW",
                "optimizer_betas": [0.9, 0.999],
                "optimizer_eps": 1e-8,
                "weight_decay": 0.0,
                "topology_learning_rate": float(resolved["training"]["code_learning_rate"]),
                "topology_minimum_learning_rate": float(
                    resolved["training"]["code_minimum_learning_rate"]
                ),
                "decoder_learning_rate": float(resolved["training"]["decoder_learning_rate"]),
                "decoder_minimum_learning_rate": float(
                    resolved["training"]["decoder_minimum_learning_rate"]
                ),
                "warmup_steps": int(resolved["training"]["warmup_steps"]),
                "gradient_clip": gradient_clip,
                "micro_batch": micro_batch,
                "gradient_accumulation": accumulation,
                "effective_batch": micro_batch * accumulation,
                "precision": "FP32",
                "completed_optimizer_steps": max_steps,
                "maximum_optimizer_steps": 15000,
                "validation_interval": validation_interval,
                "no_early_stop": True,
                "comparison_steps": list(COMPARISON_STEPS),
                "trainable_contract": trainable_contract,
                "encoder_isolation": isolation,
                "training_identity_sha256_by_step": {
                    str(row["optimizer_step"]): row["sample_identity_sha256"]
                    for row in records
                    if row["phase"] == "train"
                },
                "mean_train_step_seconds": statistics.fmean(
                    float(row["step_seconds"]) for row in records if row["phase"] == "train"
                ),
            },
            "evaluations": evaluations,
            "best_optimizer_step": int(best["optimizer_step"]),
            "best_progression_score": best_score,
            "best_evaluation": best,
            "last_three_evaluations": evaluations[-3:],
            "checkpoint_readback": {
                "best_progression": best_checkpoint,
                "last": last_checkpoint,
            },
            "plots": plots,
            "unique_next_step": (
                "REVIEW_SMOKE_AND_RUN_THE_OTHER_SMOKE"
                if smoke
                else "RUN_THE_PAIRED_ARM_OR_EXPLICIT_COMPARISON"
            ),
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_write_json(output_run / "manifests/posterior_latent_topology_summary.json", summary)
        if smoke:
            atomic_write_text(
                output_run / f"markers/{SMOKE_MARKER}",
                f"PASS execution_complete=true arm={arm} steps=2\n",
            )
        else:
            atomic_write_text(
                output_run / f"markers/{EXECUTION_MARKER}",
                f"PASS execution_complete=true arm={arm} steps=15000\n",
            )
            if quality_pass:
                atomic_write_text(
                    output_run / f"markers/{QUALITY_MARKER}",
                    f"PASS arm={arm} last_three_progression_and_global_guard=true\n",
                )
            else:
                atomic_write_text(
                    output_run / "markers/cvae.failed",
                    f"QUALITY_FAIL execution_complete=true arm={arm} last_three=false\n",
                )
        return summary
    finally:
        base.close()


def _load_formal_summary(run: Path, expected_arm: str) -> tuple[Path, dict[str, Any]]:
    run = run.expanduser().resolve()
    path = run / "manifests/posterior_latent_topology_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"F4F summary is missing: {path}")
    summary = load_json(path)
    recomputed_quality = _last_three_quality(list(summary.get("evaluations", [])))
    training = summary.get("training_contract", {})
    model = summary.get("model_contract", {})
    evaluation_steps = {
        int(row.get("optimizer_step", -1)) for row in summary.get("evaluations", [])
    }
    checks = {
        "format": summary.get("format_version") == FORMAT_VERSION,
        "arm": summary.get("arm") == expected_arm,
        "formal": not bool(summary.get("smoke")),
        "execution": bool(summary.get("execution_pass")),
        "steps": int(summary.get("training_contract", {}).get("completed_optimizer_steps", -1)) == 15000,
        "marker": (run / f"markers/{EXECUTION_MARKER}").is_file(),
        "quality_recomputed": bool(summary.get("quality_pass")) == recomputed_quality,
        "quality_marker": (run / f"markers/{QUALITY_MARKER}").is_file()
        == recomputed_quality,
        "failure_marker": (run / "markers/cvae.failed").is_file()
        == (not recomputed_quality),
        "best_checkpoint": (run / "checkpoints/best_progression.pt").is_file(),
        "last_checkpoint": (run / "checkpoints/last.pt").is_file(),
        "comparison_points": set(COMPARISON_STEPS).issubset(evaluation_steps),
        "code_shape": model.get("code_shape") == list(ARM_SHAPES[expected_arm]),
        "code_parameters": int(model.get("code_parameter_count", -1))
        == ARM_CODE_PARAMETERS[expected_arm],
        "total_parameters": int(model.get("total_parameter_count", -1))
        == ARM_TOTAL_PARAMETERS[expected_arm],
        "training_seed": int(training.get("training_seed", -1)) == TRAINING_SEED,
        "fixture_seed": int(training.get("fixture_seed", -1)) == FIXTURE_SEED,
        "objective": training.get("objective")
        == "State MSE + Action MSE + contact BCE; equal mean of present components",
        "mask_phase": training.get("mask_phase") == "fixed",
        "optimizer": training.get("optimizer") == "AdamW",
        "optimizer_betas": training.get("optimizer_betas") == [0.9, 0.999],
        "optimizer_eps": float(training.get("optimizer_eps", math.nan)) == 1e-8,
        "weight_decay": float(training.get("weight_decay", math.nan)) == 0.0,
        "topology_lr": float(training.get("topology_learning_rate", math.nan)) == 3e-4,
        "topology_min_lr": float(
            training.get("topology_minimum_learning_rate", math.nan)
        )
        == 1e-5,
        "decoder_lr": float(training.get("decoder_learning_rate", math.nan)) == 3e-5,
        "decoder_min_lr": float(
            training.get("decoder_minimum_learning_rate", math.nan)
        )
        == 1e-6,
        "warmup": int(training.get("warmup_steps", -1)) == 250,
        "gradient_clip": float(training.get("gradient_clip", math.nan)) == 1.0,
        "batch": int(training.get("micro_batch", -1)) == 4
        and int(training.get("gradient_accumulation", -1)) == 16
        and int(training.get("effective_batch", -1)) == 64,
        "precision": training.get("precision") == "FP32",
        "no_early_stop": training.get("no_early_stop") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"F4F {expected_arm} formal summary failed: {failed}")
    return path, summary


def _comparison_identity(g8: dict[str, Any], t129: dict[str, Any]) -> dict[str, bool]:
    common_training_keys = (
        "training_seed",
        "fixture_seed",
        "objective",
        "mask_phase",
        "optimizer",
        "optimizer_betas",
        "optimizer_eps",
        "weight_decay",
        "topology_learning_rate",
        "topology_minimum_learning_rate",
        "decoder_learning_rate",
        "decoder_minimum_learning_rate",
        "warmup_steps",
        "gradient_clip",
        "micro_batch",
        "gradient_accumulation",
        "effective_batch",
        "precision",
        "maximum_optimizer_steps",
        "validation_interval",
        "no_early_stop",
        "comparison_steps",
    )
    g8_training = g8.get("training_contract", {})
    t129_training = t129.get("training_contract", {})
    return {
        "dataset_hash": g8.get("dataset_manifest_sha256") == t129.get("dataset_manifest_sha256"),
        "source_hash": g8.get("source", {}).get("checkpoint_sha256")
        == t129.get("source", {}).get("checkpoint_sha256"),
        "f4e_hash": g8.get("source", {}).get("f4e_baseline", {}).get("summary_sha256")
        == t129.get("source", {}).get("f4e_baseline", {}).get("summary_sha256"),
        "fixture_hash": g8.get("data_contract", {}).get("fixture_bitmap_sha256")
        == t129.get("data_contract", {}).get("fixture_bitmap_sha256"),
        "window_hash": g8.get("data_contract", {}).get("selected_windows_sha256")
        == t129.get("data_contract", {}).get("selected_windows_sha256"),
        "motion_keys": g8.get("data_contract", {}).get("selected_motion_keys")
        == t129.get("data_contract", {}).get("selected_motion_keys"),
        "training_seed": g8.get("training_contract", {}).get("training_seed")
        == t129.get("training_contract", {}).get("training_seed") == TRAINING_SEED,
        "training_identity": g8.get("training_contract", {}).get("training_identity_sha256_by_step")
        == t129.get("training_contract", {}).get("training_identity_sha256_by_step"),
        "optimizer_contract": {
            key: g8_training.get(key) for key in common_training_keys
        }
        == {key: t129_training.get(key) for key in common_training_keys},
        "comparison_steps": g8.get("training_contract", {}).get("comparison_steps")
        == t129.get("training_contract", {}).get("comparison_steps") == list(COMPARISON_STEPS),
        "initialization_seed": g8.get("initialization", {}).get("initialization_seed")
        == t129.get("initialization", {}).get("initialization_seed") == INITIALIZATION_SEED,
        "initialization_distribution": g8.get("initialization", {}).get("distribution")
        == t129.get("initialization", {}).get("distribution"),
        "f4e_global_guard": g8.get("source", {}).get("f4e_baseline", {}).get(
            "state_global_rmse"
        )
        == t129.get("source", {}).get("f4e_baseline", {}).get("state_global_rmse")
        and g8.get("source", {}).get("f4e_baseline", {}).get("action_global_rmse")
        == t129.get("source", {}).get("f4e_baseline", {}).get("action_global_rmse"),
        "g8_shape": g8.get("model_contract", {}).get("code_shape")
        == list(ARM_SHAPES["G8"]),
        "t129_shape": t129.get("model_contract", {}).get("code_shape")
        == list(ARM_SHAPES["T129"]),
        "g8_parameters": int(g8.get("model_contract", {}).get("total_parameter_count", -1))
        == ARM_TOTAL_PARAMETERS["G8"],
        "t129_parameters": int(
            t129.get("model_contract", {}).get("total_parameter_count", -1)
        )
        == ARM_TOTAL_PARAMETERS["T129"],
        "code_budget": abs(
            int(g8.get("model_contract", {}).get("code_scalars_per_window", -1000))
            / int(t129.get("model_contract", {}).get("code_scalars_per_window", 1))
            - 1.0
        ) <= 0.01,
    }


def comparison_decision(g8_pass: bool, t129_pass: bool) -> tuple[str, str]:
    if g8_pass and t129_pass:
        return (
            "BOTH_PASS_PREFER_G8_GLOBAL_TOPOLOGY",
            "DESIGN_POSTERIOR_ENCODER_OUTPUTTING_EIGHT_GLOBAL_LATENT_TOKENS",
        )
    if g8_pass:
        return (
            "G8_PASS_SINGLE_GLOBAL_TOKEN_BROADCAST_BOTTLENECK",
            "DESIGN_POSTERIOR_ENCODER_OUTPUTTING_EIGHT_GLOBAL_LATENT_TOKENS",
        )
    if t129_pass:
        return (
            "T129_PASS_TEMPORAL_PLACEMENT_REQUIRED",
            "DESIGN_HIERARCHICAL_GLOBAL_PLUS_TEMPORAL_CVAE",
        )
    return (
        "BOTH_FAIL_LATENT_TOPOLOGY_INSUFFICIENT",
        "RUN_DIRECT_OUTPUT_MEMORY_CEILING_FOR_DECODER_AND_OBJECTIVE",
    )


def _comparison_plot(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    series: list[tuple[str, list[tuple[float, float]], str]] = []
    colors = {"G8": "#2563eb", "T129": "#dc2626"}
    for arm, summary in summaries.items():
        series.append(
            (
                f"{arm} quality score",
                [
                    (row["optimizer_step"], row["quality_gate"]["score"])
                    for row in summary["evaluations"]
                ],
                colors[arm],
            )
        )
    atomic_write_text(
        path,
        _svg_series(
            "F4F equal-code-budget latent topology comparison",
            series,
            y_label="Gate score (log10 scale)",
            log_y=True,
            horizontal_lines=[("PASS = 1", 1.0, "#059669")],
        ),
    )


def run_comparison(*, run_g8: Path, run_t129: Path, output_run: Path) -> dict[str, Any]:
    output_run = output_run.expanduser().resolve()
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "plots", "videos"):
        (output_run / child).mkdir(parents=True, exist_ok=True)
    g8_path, g8 = _load_formal_summary(run_g8, "G8")
    t129_path, t129 = _load_formal_summary(run_t129, "T129")
    checks = _comparison_identity(g8, t129)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"F4F paired identity failed: {failed}")
    g8_pass = bool(g8.get("quality_pass"))
    t129_pass = bool(t129.get("quality_pass"))
    decision, next_step = comparison_decision(g8_pass, t129_pass)
    plot_path = output_run / "plots/latent_topology_comparison.svg"
    _comparison_plot(plot_path, {"G8": g8, "T129": t129})

    def final_metrics(summary: dict[str, Any]) -> list[dict[str, Any]]:
        result = []
        by_step = {int(row["optimizer_step"]): row for row in summary["evaluations"]}
        for step in COMPARISON_STEPS:
            row = by_step[step]
            result.append(
                {
                    "optimizer_step": step,
                    "passed": bool(row["quality_gate"]["passed"]),
                    "score": float(row["quality_gate"]["score"]),
                    "worst_state_rmse": float(row["exact"]["worst_state_rmse"]),
                    "worst_action_rmse": float(row["exact"]["worst_action_rmse"]),
                    "continuous_max_abs": float(row["exact"]["continuous_max_abs"]),
                    "contact_accuracy": float(row["exact"]["contact_accuracy"]),
                    "threshold_exceed_fraction": float(row["tail_global"]["threshold_exceed_fraction"]),
                    "global_state_rmse": float(row["quality_gate"]["global_state_rmse"]),
                    "global_action_rmse": float(row["quality_gate"]["global_action_rmse"]),
                    "zero_ratio": float(row["topology_progression_gate"]["zero_ratio"]),
                    "cross_window_ratio": float(row["topology_progression_gate"]["cross_window_ratio"]),
                    "cross_motion_ratio": float(row["topology_progression_gate"]["cross_motion_ratio"]),
                }
            )
        return result

    summary = {
        "format_version": COMPARISON_FORMAT,
        "execution_pass": True,
        "identity_checks": checks,
        "runs": {
            "G8": {"run": str(run_g8.resolve()), "summary_sha256": file_sha256(g8_path)},
            "T129": {"run": str(run_t129.resolve()), "summary_sha256": file_sha256(t129_path)},
        },
        "quality_pass": {"G8": g8_pass, "T129": t129_pass},
        "comparison_steps": list(COMPARISON_STEPS),
        "last_three": {"G8": final_metrics(g8), "T129": final_metrics(t129)},
        "last_three_median": {
            arm: {
                key: statistics.median(float(row[key]) for row in rows)
                for key in (
                    "score",
                    "worst_state_rmse",
                    "worst_action_rmse",
                    "continuous_max_abs",
                    "threshold_exceed_fraction",
                    "global_state_rmse",
                    "global_action_rmse",
                    "zero_ratio",
                    "cross_window_ratio",
                    "cross_motion_ratio",
                )
            }
            for arm, rows in {
                "G8": final_metrics(g8),
                "T129": final_metrics(t129),
            }.items()
        },
        "budget_comparison": {
            "G8_code_scalars_per_window": math.prod(ARM_SHAPES["G8"][1:]),
            "T129_code_scalars_per_window": math.prod(ARM_SHAPES["T129"][1:]),
            "relative_code_scalar_difference": abs(
                math.prod(ARM_SHAPES["G8"][1:])
                / math.prod(ARM_SHAPES["T129"][1:])
                - 1.0
            ),
            "total_parameter_relative_difference": abs(
                ARM_TOTAL_PARAMETERS["G8"] / ARM_TOTAL_PARAMETERS["T129"] - 1.0
            ),
        },
        "decision": decision,
        "unique_next_step": next_step,
        "plot": str(plot_path),
        "scope": "paired F4F auto-decoder topology capacity; no posterior/prior claim",
    }
    atomic_write_json(output_run / "manifests/posterior_latent_topology_comparison.json", summary)
    atomic_write_text(
        output_run / f"markers/{COMPARISON_MARKER}",
        f"PASS decision={decision}\n",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F4F equal-budget latent topology diagnostic")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--dataset-run", type=Path, required=True)
    train.add_argument("--source-checkpoint", type=Path, required=True)
    train.add_argument("--f4e-run", type=Path, required=True)
    train.add_argument("--output-run", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--arm", choices=ARMS, required=True)
    train.add_argument("--smoke", action="store_true")
    compare = subparsers.add_parser("compare")
    compare.add_argument("--run-g8", type=Path, required=True)
    compare.add_argument("--run-t129", type=Path, required=True)
    compare.add_argument("--output-run", type=Path, required=True)
    return parser.parse_args()


def _write_failure(output_run: Path, error: BaseException) -> None:
    output_run = output_run.expanduser().resolve()
    atomic_write_json(
        output_run / "manifests/posterior_latent_topology_failure.json",
        {
            "format_version": FAILURE_FORMAT,
            "execution_pass": False,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )


def main() -> int:
    args = parse_args()
    try:
        if args.command == "compare":
            summary = run_comparison(
                run_g8=args.run_g8,
                run_t129=args.run_t129,
                output_run=args.output_run,
            )
            print("Posterior F4F comparison: PASS")
            print(json.dumps({key: summary[key] for key in ("decision", "unique_next_step")}, indent=2))
        else:
            summary = run_experiment(
                dataset_run=args.dataset_run,
                source_checkpoint=args.source_checkpoint,
                f4e_run=args.f4e_run,
                output_run=args.output_run,
                config=load_json(args.config),
                arm=args.arm,
                smoke=args.smoke,
            )
            print("Posterior F4F latent topology: PASS (execution complete)")
            print(
                json.dumps(
                    {
                        "output_run": str(args.output_run.expanduser().resolve()),
                        "arm": summary["arm"],
                        "smoke": summary["smoke"],
                        "quality_pass": summary["quality_pass"],
                        "completed_optimizer_steps": summary["training_contract"]["completed_optimizer_steps"],
                        "best_progression_score": summary["best_progression_score"],
                    },
                    indent=2,
                )
            )
        return 0
    except BaseException as error:
        _write_failure(args.output_run, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
