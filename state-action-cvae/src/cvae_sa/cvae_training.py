"""Protocol-v2 trainer. Architecture and full reconstruction objective unchanged."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from .cvae_protocol import (CHECKPOINT, LEGACY_CHECKPOINT, PROTOCOL, Fixtures, RecoverableSampler,
    capture_rng, digest, durable_save, isolated_rng, lr_factor, quality_warnings, restore_rng, run_lock)
from .cvae_diagnostics import ablations, batch_ids, epsilon_for, evaluate, read_normalization, route_output
from .models import build_model
from .posterior_direct_output import assert_output_isolated
from .posterior_t64_protocol import make_physical_masks, _svg
from .util import atomic_write_json, atomic_write_text, file_sha256, seed_everything


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def load_rows(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A killed writer can leave the final line incomplete, never an interior one.
                if stream.read().strip():
                    raise ValueError(f"corrupt interior JSONL record: {path}")
    return rows


def make_dataset(dataset_run, config, smoke=False):
    from .dataset import StateActionWindowDataset
    data = config.get("data", {})
    if int(data.get("window_transitions", 64)) != 64 or data.get("num_workers", 0) != 0:
        raise ValueError("v2 requires T64 and synchronous num_workers=0")
    dataset = StateActionWindowDataset(dataset_run, "train", 64, int(data.get("stride", 64)),
        max_episodes=data.get("max_episodes", 256), random_crop=False)
    limit = 2 if smoke else data.get("max_windows")
    indices = list(range(len(dataset) if limit is None else min(len(dataset), int(limit))))
    if not indices:
        raise ValueError("no selected windows")
    return dataset, indices


def data_identity(dataset_run, windows):
    return {"dataset_run": str(dataset_run), "dataset_manifest_sha256": file_sha256(dataset_run / "manifests/dataset_manifest.json"),
        "episodes_index_sha256": file_sha256(dataset_run / "manifests/episodes.jsonl"),
        "normalization_sha256": file_sha256(dataset_run / "data/normalization.npz"),
        "selected_windows_sha256": digest(windows), "selected_window_count": len(windows), "window": 64}


def check_source_identity(checkpoint, identity, *, exact=False, allow_unknown=False):
    old = checkpoint.get("dataset_identity") or {}
    missing = []
    for key in ("dataset_manifest_sha256", "episodes_index_sha256", "normalization_sha256", "selected_windows_sha256", "selected_window_count"):
        if old.get(key) is None:
            missing.append(key)
        elif old[key] != identity[key]:
            raise ValueError(f"source dataset identity mismatch: {key}")
    if missing and (exact or not allow_unknown):
        raise ValueError("legacy source identity unknown: " + ", ".join(missing) + "; read-only evaluation is allowed, training needs --allow-legacy-identity")
    return {"verified_fields": sorted(set(identity) & set(old)), "unknown_fields": missing,
            "exact_identity_verified": not missing}


def source_info(checkpoint_path, checkpoint):
    return {"path": str(checkpoint_path), "sha256": file_sha256(checkpoint_path),
            "step": checkpoint.get("optimizer_step"), "cumulative_step": checkpoint.get("cumulative_step", checkpoint.get("optimizer_step", 0)),
            "protocol_version": checkpoint.get("protocol_version", "legacy_batch_position_dependent_mask" if checkpoint.get("stage") != "A" else "legacy_A_v1")}


def provenance(config, contract, identity, source):
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    paths = list(Path(__file__).parent.glob("cvae_*.py")) + [Path(__file__).with_name(name) for name in
        ("models.py", "dataset.py", "util.py", "posterior_t64_protocol.py", "posterior_hierarchical_standard_cvae.py")]
    paths.append(root / "state-action-cvae/cvae_repro.sh")
    return {"protocol_version": PROTOCOL, "config": config, "resolved_training_contract": contract,
        "argv": sys.argv, "git_commit": commit, "source_hashes": {str(p.relative_to(root)): file_sha256(p) for p in paths},
        "dataset_identity": identity, "source_checkpoint": source, "torch_version": torch.__version__,
        "python_version": sys.version, "pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat()}


def mask_bank(fixtures, seed, held_out, path):
    rows = []
    loader = DataLoader(fixtures, batch_size=32, shuffle=False, generator=torch.Generator().manual_seed(0))
    for batch in loader:
        sm, am, names = make_physical_masks(batch, seed, held_out=held_out)
        for i, identity in enumerate(batch_ids(batch)):
            rows.append({**identity, "name": names[i], "seed": seed, "held_out": held_out,
                         "state_hidden_indices": sm[i].flatten().nonzero().flatten().tolist(),
                         "action_hidden_indices": am[i].flatten().nonzero().flatten().tolist()})
    atomic_write_json(path, {"protocol_version": PROTOCOL, "sha256": digest(rows), "fixtures": rows})
    return digest(rows)


def plots(run, rows):
    palette = ["#08519c", "#d95f0e", "#238b45", "#756bb1"]
    sampled_training = []
    with (run / "logs/metrics.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("phase") == "train" and (row["optimizer_step"] == 1 or row["optimizer_step"] % 50 == 0):
                sampled_training.append(row)
    _svg(run / "plots/train_loss.svg", "Training objective", "one point / 50 updates; full records in metrics.jsonl",
         [(key, [(r["optimizer_step"], r[key]) for r in sampled_training], color) for key, color in
          zip(("loss", "reconstruction_state", "reconstruction_action", "reconstruction_contact"), palette)])
    _svg(run / "plots/learning_rate.svg", "Actual update learning rate", "LR used, not next-step LR",
         [("LR", [(r["optimizer_step"], r["learning_rate"]) for r in sampled_training], palette[0])])
    for name, keys in (("training_curves", ("total_loss", "state_rmse", "action_rmse", "selection_score")),
                       ("tail_curves", ("state_max_abs", "action_max_abs", "state_abs_p99", "action_abs_p99"))):
        _svg(run / f"plots/{name}.svg", name, "Protocol v2; normalized, log10 scale",
            [(key, [(row["optimizer_step"], row[key]) for row in rows if row.get(key) is not None], color) for key, color in zip(keys, palette)])
    _svg(run / "plots/masked_visible.svg", "masked / visible", "State and Action MSE; target counts in JSON",
         [(f"{part}-{d}", [(r["optimizer_step"], r["partitions"][part][d]["mse"]) for r in rows
             if r["partitions"][part][d]["mse"] is not None], palette[i]) for i, (part, d) in
             enumerate((('masked', 'state'), ('masked', 'action'), ('visible', 'state'), ('visible', 'action')))])
    families = sorted({f for row in rows for f in row["mask_families"]})
    _svg(run / "plots/mask_families.svg", "Mask family masked RMSE", "macro across nonempty domains",
        [(family, [(r["optimizer_step"], float(np.mean([v["micro_rmse"] for v in r["mask_families"][family]["masked"].values() if v["count"]])))
                    for r in rows if family in r["mask_families"] and any(v["count"] for v in r["mask_families"][family]["masked"].values())], palette[i % 4]) for i, family in enumerate(families)])
    heldout_rows = [(r["optimizer_step"], r["heldout_mask"]) for r in rows if r.get("heldout_mask")]
    if heldout_rows:
        _svg(run / "plots/heldout_curves.svg", "Held-out mask diagnostics", "held-out full reconstruction and selection metrics",
             [("heldout_selection", [(step, report["selection_score"]) for step, report in heldout_rows], palette[0]),
              ("heldout_state_rmse", [(step, report["state_rmse"]) for step, report in heldout_rows], palette[1]),
              ("heldout_action_rmse", [(step, report["action_rmse"]) for step, report in heldout_rows], palette[2])])
    if rows and "standard_normal" in rows[-1].get("routes", {}):
        _svg(run / "plots/c_routes.svg", "C routes", "Deployment energy and posterior reconstruction are different scores",
            [(route, [(r["optimizer_step"], r["routes"][route]["selection_score"]) for r in rows if route in r.get("routes", {})], palette[i])
             for i, route in enumerate(("posterior_mean", "posterior_sample", "standard_normal", "zero"))])


def strict_readback(path, model, optimizer, scheduler, probe, stage, seed):
    """Strict model/optimizer/scheduler loading plus deterministic forward comparison."""
    from .posterior_hierarchical_standard_cvae import load_checkpoint
    with isolated_rng():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        other = build_model(model.config).to(next(model.parameters()).device)
        load_checkpoint(other, path)
        other.set_training_stage(stage)
        check_optimizer = torch.optim.AdamW([p for p in other.parameters() if p.requires_grad])
        check_optimizer.load_state_dict(saved["optimizer"])
        check_scheduler = torch.optim.lr_scheduler.LambdaLR(check_optimizer, lambda _: 1.)
        check_scheduler.load_state_dict(saved["scheduler"])
        if saved["scheduler"]["last_epoch"] != saved["optimizer_step"]:
            raise ValueError("checkpoint scheduler/optimizer step mismatch")
        if saved.get("sampler") is None:
            raise ValueError("new checkpoint missing sampler")
        if saved["source_step"] + saved["optimizer_step"] != saved["cumulative_step"]:
            raise ValueError("checkpoint local/source/cumulative step mismatch")
        sampler = RecoverableSampler(saved["sampler"]["size"], saved["sampler"]["batch_size"], 0)
        sampler.load_state_dict(saved["sampler"])
        sm, am, _ = make_physical_masks(probe, seed)
        device = next(model.parameters()).device
        b, sm, am = to_device(probe, device), sm.to(device), am.to(device)
        route = stage if stage in {"A", "B"} else "posterior_mean"
        other.eval()
        with torch.no_grad():
            actual = route_output(other, b, sm, am, route)
        expected = saved["readback_probe"]
        for key in ("physical_state", "action"):
            torch.testing.assert_close(getattr(actual, key).cpu(), expected[key], rtol=1e-5, atol=1e-6)
        if stage == "C" and saved.get("readback_deployment_probe") is not None:
            deployment_epsilon = epsilon_for(other, probe, 0, seed, device)
            with torch.no_grad():
                deployment = route_output(other, b, sm, am, "standard_normal", deployment_epsilon)
            for key in ("physical_state", "action"):
                torch.testing.assert_close(
                    getattr(deployment, key).cpu(),
                    saved["readback_deployment_probe"][key],
                    rtol=1e-5,
                    atol=1e-6,
                )
        for state in check_optimizer.state.values():
            if "step" in state and int(state["step"]) > saved["cumulative_step"]:
                raise ValueError("optimizer state step exceeds completed cumulative step")
        for parameter, state in check_optimizer.state.items():
            for name in ("exp_avg", "exp_avg_sq"):
                if name in state and (state[name].shape != parameter.shape or not bool(torch.isfinite(state[name]).all())):
                    raise ValueError(f"optimizer state invalid: {name}")
        return {"passed": True, "strict_model": True, "optimizer_loaded": True, "scheduler_step": saved["optimizer_step"],
                "forward_comparison": True, "sha256": file_sha256(path)}


def run_experiment(dataset_run, output_run, source_run, config, *, stage, init_run=None, smoke=False,
    kl_beta_override=None, max_steps_override=None, micro_batch_override=None, learning_rate_override=None,
    lr_schedule=None, warmup_steps_override=None, min_lr_ratio_override=None, validation_interval_override=None,
    checkpoint_interval_override=None, log_interval_override=None, resume_run=None, mask_mode="fixed",
    init_checkpoint=None, continue_checkpoint=None, additional_steps=None, allow_legacy_identity=False,
    eval_samples=8, beta_warmup_steps=None):
    from .posterior_hierarchical_standard_cvae import _stage_key, hierarchical_kl, load_checkpoint, weighted_reconstruction_loss
    code, stage_name = _stage_key(stage)
    if source_run is not None:
        raise ValueError("source-run is obsolete; use explicit --init-checkpoint or --continue-checkpoint")
    if mask_mode not in {"fixed", "dynamic"} or eval_samples < 1:
        raise ValueError("invalid mask mode or eval samples")
    if code == "A" and mask_mode != "fixed":
        raise ValueError("A has no condition masks")
    if code != "C" and kl_beta_override is not None:
        raise ValueError("KL override only applies to C")
    if sum(x is not None for x in (init_run, init_checkpoint, resume_run, continue_checkpoint)) > 1:
        raise ValueError("resume, continue, and model-only init are mutually exclusive")
    if code == "C" and (init_run or init_checkpoint):
        raise ValueError("C starts independently from random initialization; use resume only for its own run")
    if (additional_steps is None) != (continue_checkpoint is None):
        raise ValueError("--continue-checkpoint and --additional-steps must be specified together")
    if additional_steps is not None and max_steps_override is not None and int(additional_steps) != int(max_steps_override):
        raise ValueError("max-steps conflicts with additional-steps; use the latter for continuation")
    dataset_run, run = Path(dataset_run).resolve(), Path(output_run).resolve()
    assert_output_isolated(run, [dataset_run])
    if dataset_run.is_relative_to(run):
        raise ValueError("output run must not be an ancestor of the dataset")
    if not resume_run and ((run / "logs/metrics.jsonl").exists() or (run / "manifests/progress.json").exists()):
        raise ValueError("new experiments require a new run directory")
    if resume_run and Path(resume_run).resolve() != run:
        raise ValueError("resume must use the same run directory")
    if resume_run and (run / "manifests/progress.json").exists():
        if json.loads((run / "manifests/progress.json").read_text())["status"] == "completed":
            raise ValueError("completed experiments require continue in a new run")
    for folder in ("logs", "manifests", "data", "checkpoints", "markers", "plots", "evaluations"):
        (run / folder).mkdir(parents=True, exist_ok=True)
    print(f"RUN_DIR={run}\nPID={os.getpid()} protocol={PROTOCOL} stage={code}", flush=True)
    with run_lock(run):
        if not resume_run:
            atomic_write_json(run / "manifests/progress.json", {"status": "running", "phase": "preflight",
                "run_dir": str(run), "pid": os.getpid(), "stage": code, "protocol_version": PROTOCOL,
                "optimizer_step": 0, "updated_at": datetime.now(timezone.utc).isoformat()})
        data = None
        try:
            data, indices = make_dataset(dataset_run, config, smoke)
            return _train(data, indices, dataset_run, run, config, code, stage_name,
            init_run=init_run, smoke=smoke, kl_beta_override=kl_beta_override, max_steps_override=max_steps_override,
            micro_batch_override=micro_batch_override, learning_rate_override=learning_rate_override, lr_schedule=lr_schedule,
            warmup_steps_override=warmup_steps_override, min_lr_ratio_override=min_lr_ratio_override,
            validation_interval_override=validation_interval_override, checkpoint_interval_override=checkpoint_interval_override,
            log_interval_override=log_interval_override, resume_run=resume_run, mask_mode=mask_mode,
            init_checkpoint=init_checkpoint, continue_checkpoint=continue_checkpoint, additional_steps=additional_steps,
            allow_legacy_identity=allow_legacy_identity, eval_samples=eval_samples, beta_warmup_steps=beta_warmup_steps)
        except BaseException as error:
            if not resume_run:
                progress_path = run / "manifests/progress.json"
                previous = json.loads(progress_path.read_text(encoding="utf-8"))
                if previous.get("status") != "failed":
                    atomic_write_json(progress_path, {**previous, "status": "failed", "error": repr(error),
                        "updated_at": datetime.now(timezone.utc).isoformat()})
            raise
        finally:
            if data is not None:
                data.close()


def _train(data, indices, dataset_run, run, config, code, stage_name, **options):
    from .posterior_hierarchical_standard_cvae import hierarchical_kl, load_checkpoint, weighted_reconstruction_loss
    cfg = config.get("training", {})
    def option(key, default):
        value = options.get(key + "_override")
        return cfg.get(key, default) if value is None else value
    requested_max = options.get("additional_steps")
    if requested_max is None:
        requested_max = options.get("max_steps_override")
    maximum = int(requested_max if requested_max is not None else cfg.get(stage_name, {}).get("max_optimizer_steps", 60000))
    if options["smoke"]:
        maximum = min(maximum, 2)
    warmup = int(option("warmup_steps", 2000))
    if options["smoke"]:
        warmup = min(warmup, maximum - 1)
    batch_size, rate = int(option("micro_batch", 32)), float(option("learning_rate", 1e-4))
    floor, schedule = float(option("min_lr_ratio", .01)), options["lr_schedule"] or cfg.get("lr_schedule", "cosine")
    if maximum < 1 or batch_size < 1 or not np.isfinite(rate) or rate <= 0 or schedule not in {"constant", "linear", "cosine"}:
        raise ValueError("invalid training budget or learning rate")
    lr_factor(0, maximum, warmup, floor, schedule)
    intervals = {k: int(option(k, v)) for k, v in (("validation_interval", 1000), ("checkpoint_interval", 500), ("log_interval", 50))}
    if min(intervals.values()) < 1:
        raise ValueError("intervals must be positive")
    seed, mask_seed = int(config.get("initialization_seed", 20260921)), int(config.get("training_mask_seed", 20260920))
    beta = float(options["kl_beta_override"] if options["kl_beta_override"] is not None else cfg.get("kl", {}).get("beta", .001))
    beta_warmup = int(options["beta_warmup_steps"] if options["beta_warmup_steps"] is not None else cfg.get("kl", {}).get("beta_warmup_steps", 10000))
    if not np.isfinite(beta) or beta < 0 or beta_warmup < 1:
        raise ValueError("invalid KL schedule")
    fixtures = Fixtures(data, indices, expand=code != "A" and options["mask_mode"] == "fixed")
    eval_fixtures = Fixtures(data, indices, expand=code != "A")
    windows = fixtures.manifest()
    identity = data_identity(dataset_run, windows)
    normalization = read_normalization(dataset_run / "data/normalization.npz")
    selection = {"A": "full_reconstruction_loss", "B": "equal_family_masked_domain_mean_MSE", "C": "standard_normal_masked_energy_score"}[code]
    contract = {"protocol_version": PROTOCOL, "stage": code, "mask_mode": options["mask_mode"], "max_steps": maximum,
        "micro_batch": batch_size, "learning_rate": rate, "warmup_steps": warmup, "min_lr_ratio": floor,
        "lr_schedule": schedule, **intervals, "initialization_seed": seed, "training_mask_seed": mask_seed,
        "eval_samples": options["eval_samples"], "kl_beta": beta if code == "C" else 0., "beta_warmup_steps": beta_warmup,
        "objective": "full_state_continuous_action_contact_equal_mean_v1", "selection": selection,
        "fixture_count": len(fixtures), "selected_window_count": len(windows), "quality_rule_version": "warnings-v1"}
    seed_everything(seed)
    model = build_model(config["model"])
    counts = model.set_training_stage(code)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=rate, weight_decay=0.)
    sampler = RecoverableSampler(len(fixtures), batch_size, seed)
    effective_config = json.loads(json.dumps(config))
    effective_config["model"] = model.config
    effective_config["data"] = {**config.get("data", {}), "window_transitions": 64, "max_windows": len(indices), "num_workers": 0}
    effective_config["training"] = {**cfg, **contract, stage_name: {**cfg.get(stage_name, {}), "max_optimizer_steps": maximum}}
    current_provenance = provenance(effective_config, contract, identity, None)
    start, source_step, source, resumed = 0, 0, None, None
    path = options["init_checkpoint"] or options["continue_checkpoint"]
    if options["init_run"]:
        # Kept as explicitly model-only legacy convenience, without fallback to last.
        path = Path(options["init_run"]) / "checkpoints/best.pt"
    if options["resume_run"]:
        path = run / "checkpoints/last.pt"
    if path:
        path = Path(path).resolve()
        if not options["resume_run"]:
            assert_output_isolated(run, [path.parent.parent])
        checkpoint = load_checkpoint(model, path)
        source = source_info(path, checkpoint)
        source["identity_check"] = check_source_identity(checkpoint, identity, exact=bool(options["resume_run"]), allow_unknown=options["allow_legacy_identity"])
        if options["resume_run"]:
            if checkpoint.get("format_version") != CHECKPOINT or checkpoint.get("training_contract") != contract:
                raise ValueError("exact resume requires v2 and identical training contract")
            if checkpoint.get("source_hashes") != current_provenance["source_hashes"]:
                raise ValueError("exact resume source hashes differ; use a new controlled initialization/continuation run")
            if checkpoint.get("stage") != code:
                raise ValueError("resume stage mismatch")
            resumed = checkpoint
            start = int(checkpoint["optimizer_step"])
            source_step = int(checkpoint["source_step"])
            source = checkpoint.get("source_checkpoint")
            optimizer.load_state_dict(checkpoint["optimizer"])
            sampler.load_state_dict(checkpoint["sampler"])
            if start > maximum:
                raise ValueError("checkpoint step exceeds requested maximum")
        elif options["continue_checkpoint"]:
            source_progress = path.parent.parent / "manifests/progress.json"
            if not source_progress.exists() or json.loads(source_progress.read_text())["status"] != "completed":
                raise ValueError("continuation requires a completed source run")
            if checkpoint.get("stage") != code:
                raise ValueError("continuation must preserve training stage")
            old_contract = checkpoint.get("training_contract") or {}
            if checkpoint["format_version"] == CHECKPOINT and any(old_contract.get(k) != contract[k] for k in
                ("mask_mode", "micro_batch", "initialization_seed", "training_mask_seed", "objective", "kl_beta", "beta_warmup_steps")):
                raise ValueError("continuation must preserve sampling/objective; use model-only init for a changed experiment")
            optimizer.load_state_dict(checkpoint["optimizer"])
            source_step = int(checkpoint.get("cumulative_step", checkpoint["optimizer_step"]))
            if checkpoint.get("sampler"):
                sampler.load_state_dict(checkpoint["sampler"])
                source["sampling"] = "preserved"
            else:
                source["sampling"] = "explicit_legacy_sampling_restart_not_exact_resume"
        elif code == "B" and options["mask_mode"] == "dynamic":
            if checkpoint.get("stage") != "B" or checkpoint.get("training_contract", {}).get("mask_mode") != "fixed":
                raise ValueError("B-dynamic requires revised B-fixed source weights")
    elif code == "B" and options["mask_mode"] == "dynamic":
        raise ValueError("B-dynamic requires --init-checkpoint from revised B-fixed")
    # A continuation has a new schedule segment, but preserves AdamW moments.
    if not resumed:
        for group in optimizer.param_groups:
            group["lr"], group["initial_lr"] = rate, rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: lr_factor(i, maximum, warmup, floor, schedule))
    if resumed:
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
    mode = "resume" if resumed else ("continue" if options["continue_checkpoint"] else "init" if path else "random")
    loader = DataLoader(eval_fixtures, batch_size=batch_size, shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(seed + 1))
    probe = default_collate([eval_fixtures[i] for i in range(min(2, len(eval_fixtures)))])
    if not resumed:
        shutil.copy2(dataset_run / "data/normalization.npz", run / "data/normalization.npz")
        atomic_write_json(run / "manifests/selected_windows.json", windows)
        atomic_write_json(run / "manifests/effective_config.json", effective_config)
        atomic_write_json(run / "manifests/requested_config.json", config)
        atomic_write_json(run / "manifests/training_contract.json", contract)
        atomic_write_json(run / "manifests/provenance.json", {**current_provenance, "source_checkpoint": source, "initialization_mode": mode})
        atomic_write_json(run / "manifests/model_signature.json", model.architecture_signature(model.config, sum(p.numel() for p in model.parameters())))
        if code != "A":
            with isolated_rng():
                mask_bank(eval_fixtures, mask_seed, False, run / "data/fixed_mask_bank.json")
                mask_bank(eval_fixtures, mask_seed + 700001, True, run / "data/heldout_mask_bank.json")
    metrics_path = run / "logs/metrics.jsonl"
    eval_rows = load_rows(run / "logs/evaluations.jsonl")
    best = resumed.get("best_metrics") if resumed else None
    best_step = resumed.get("best_optimizer_step") if resumed else None
    # B has two selection domains. ``best.pt`` remains the fixed-bank winner
    # for compatibility; dynamic-mask runs additionally persist the best
    # held-out checkpoint and its metrics.
    best_heldout = resumed.get("best_heldout_metrics") if resumed else None
    best_heldout_step = resumed.get("best_heldout_step") if resumed else None
    best_reconstruction = resumed.get("best_reconstruction_loss", float("inf")) if resumed else float("inf")
    best_reconstruction_step = resumed.get("best_reconstruction_step") if resumed else None
    last_step, committed, evaluation_count = start, True, sum(row["optimizer_step"] <= start for row in eval_rows)
    durable_step = start if resumed else None
    grad_seen = set(resumed.get("gradient_seen", []) if resumed else [])
    gradient_observations = {}
    started_at = time.perf_counter()
    stop_requested = []
    def progress(status="running", phase="train", **extra):
        atomic_write_json(run / "manifests/progress.json", {"status": status, "phase": phase, "pid": os.getpid(),
            "run_dir": str(run), "protocol_version": PROTOCOL, "stage": code, "mask_mode": options["mask_mode"],
            "optimizer_step": last_step, "source_step": source_step, "cumulative_step": source_step + last_step,
            "durable_checkpoint_step": durable_step,
            "max_steps": maximum, "sample_exposures": sampler.exposures, "best_step": best_step,
            "best_metrics": best, "best_heldout_step": best_heldout_step,
            "best_heldout_metrics": best_heldout, "quality_pass": None,
            "updated_at": datetime.now(timezone.utc).isoformat(), **extra})
    def checkpoint_payload(step):
        with isolated_rng():
            was_training = model.training
            model.eval()
            sm, am, _ = make_physical_masks(probe, mask_seed)
            with torch.no_grad():
                output = route_output(model, to_device(probe, device), sm.to(device), am.to(device), code if code != "C" else "posterior_mean")
                deployment_output = None
                if code == "C":
                    deployment_epsilon = epsilon_for(model, probe, 0, mask_seed, device)
                    deployment_output = route_output(
                        model, to_device(probe, device), sm.to(device), am.to(device),
                        "standard_normal", deployment_epsilon
                    )
            model.train(was_training)
        return {"format_version": CHECKPOINT, "protocol_version": PROTOCOL, "architecture_version": model.ARCHITECTURE_VERSION,
            "model_signature": model.architecture_signature(model.config, sum(p.numel() for p in model.parameters())),
            "parameter_count": sum(p.numel() for p in model.parameters()), "model_config": model.config, "config": effective_config,
            "source_hashes": current_provenance["source_hashes"],
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "optimizer_step": step, "source_step": source_step, "cumulative_step": source_step + step, "stage": code,
            "training_contract": contract, "dataset_identity": identity, "sampler": sampler.state_dict(), "rng_state": capture_rng(),
            "source_checkpoint": source, "best_metrics": best, "best_optimizer_step": best_step,
            "best_heldout_metrics": best_heldout, "best_heldout_step": best_heldout_step,
            "best_reconstruction_loss": best_reconstruction, "best_reconstruction_step": best_reconstruction_step,
            "gradient_seen": sorted(grad_seen),
            "readback_probe": {key: getattr(output, key).cpu() for key in ("physical_state", "action")},
            "readback_deployment_probe": (
                {key: getattr(deployment_output, key).cpu() for key in ("physical_state", "action")}
                if deployment_output is not None else None
            )}
    def save(name="last.pt"):
        nonlocal durable_step
        if not committed:
            raise RuntimeError("refusing checkpoint from incomplete optimizer update")
        durable_save(run / "checkpoints" / name, checkpoint_payload(last_step))
        if name == "last.pt":
            durable_step = last_step
    def audit(step, warnings):
        readback = strict_readback(run / "checkpoints/last.pt", model, optimizer, scheduler, probe, code, mask_seed)
        missing = sorted({name.split('.')[0] for name, p in model.named_parameters() if p.requires_grad} - grad_seen)
        report = {"protocol_version": PROTOCOL, "optimizer_step": step, "verified_facts": {
            "dataset_identity": identity, "stage": code, "posterior_executed": code in {"A", "C"},
            "condition_executed": code in {"B", "C"}, "checkpoint_readback": readback,
            "gradient_seen_groups": sorted(grad_seen), "recent_gradient_observations": gradient_observations},
            "quality_warnings": warnings, "quality_pass": None, "candidate_causes": [],
            "missing_evidence": (["training gradients not yet observed"] if step == 0 else missing),
            "unique_next_step": "CONTINUE_CURRENT_BUDGET_AND_REVIEW_STAGE_SEPARATELY"}
        if step >= 5 and missing:
            # Fallback parameters can legitimately be unused with all chunks valid.
            unexpected = [name for name in missing if "fallback" not in name and name != "empty_local"]
            if unexpected:
                raise RuntimeError(f"active module gradient never observed: {unexpected}")
        atomic_write_json(run / f"manifests/audit_{step:09d}.json", report)
        return report
    def full_eval():
        nonlocal best, best_step, best_heldout, best_heldout_step
        nonlocal best_reconstruction, best_reconstruction_step, evaluation_count
        routes = [code] if code != "C" else ["posterior_mean", "posterior_sample", "standard_normal", "zero"]
        reports = {}
        for route in routes:
            progress(phase="evaluation", route=route, evaluation_batches=0)
            reports[route], _ = evaluate(model, loader, device, normalization, route=route, seed=mask_seed,
                samples=options["eval_samples"], output_dir=run / f"evaluations/step_{last_step:09d}/fixed/{route}",
                heartbeat=lambda n: progress(phase="evaluation", route=route, evaluation_batches=n))
        primary = reports[code if code != "C" else "standard_normal"]
        row = {**primary, "optimizer_step": last_step, "cumulative_step": source_step + last_step,
               "phase": "evaluation", "routes": reports}
        if code != "A":
            held, _ = evaluate(model, loader, device, normalization, route=code if code == "B" else "standard_normal",
                seed=mask_seed + 700001, held_out=True, samples=options["eval_samples"],
                reference_mask_seed=mask_seed,
                output_dir=run / f"evaluations/step_{last_step:09d}/heldout",
                heartbeat=lambda n: progress(phase="evaluation", route="heldout", evaluation_batches=n))
            row["heldout_mask"] = held
        eval_rows.append(row)
        if code != "A":
            missing_families = set(("state_gap_4", "state_gap_16", "state_rollout", "action_gap_4", "action_gap_16", "full_action", "joint_gap_2", "joint_gap_8")) - set(primary["mask_families"])
            if missing_families:
                raise RuntimeError(f"evaluation target coverage missing families: {sorted(missing_families)}")
        append(run / "logs/evaluations.jsonl", row)
        append(metrics_path, {k: v for k, v in row.items() if k not in {"routes", "partitions", "mask_families", "heldout_mask"}})
        improved = best is None or row["selection_score"] < best["selection_score"]
        if improved:
            best, best_step = {k: row[k] for k in ("selection_score", "total_loss", "state_rmse", "action_rmse", "max_abs", "optimizer_step")}, last_step
        heldout_improved = False
        if code == "B" and row.get("heldout_mask") is not None:
            heldout = row["heldout_mask"]
            heldout_score = heldout.get("selection_score")
            heldout_improved = heldout_score is not None and (
                best_heldout is None or heldout_score < best_heldout["selection_score"]
            )
            if heldout_improved:
                best_heldout = {k: heldout[k] for k in (
                    "selection_score", "total_loss", "state_rmse", "action_rmse", "max_abs"
                )}
                best_heldout["optimizer_step"] = last_step
                best_heldout_step = last_step
        reconstruction_score = reports.get("posterior_mean", primary)["total_loss"]
        reconstruction_improved = reconstruction_score < best_reconstruction
        if reconstruction_improved:
            best_reconstruction = reconstruction_score
            best_reconstruction_step = last_step
        # Commit last first. If interrupted between file replacements, its model
        # reconstructs a missing/stale best for this same step on resume.
        save()
        if improved:
            save("best.pt")
            if code == "B":
                # Explicitly name the fixed-bank selection policy while
                # preserving the historical best.pt alias.
                save("best_fixed.pt")
        if heldout_improved:
            save("best_heldout.pt")
        if reconstruction_improved:
            save("best_reconstruction.pt")
        evaluation_count += 1
        warnings = quality_warnings(eval_rows)
        if evaluation_count in {1, 2} or evaluation_count % 5 == 0 or last_step == maximum:
            audit(last_step, warnings)
        if evaluation_count % 5 == 0 and len(indices) > 1:
            with isolated_rng():
                manifest = eval_fixtures.manifest()
                selected_indices = []
                seen_motions = set()
                fixture_width = 8 if code != "A" else 1
                # Prefer one window per motion so the donor swap is genuinely
                # cross-motion on multi-motion runs.
                for window_position, row_manifest in enumerate(manifest):
                    if row_manifest["motion_key"] in seen_motions:
                        continue
                    seen_motions.add(row_manifest["motion_key"])
                    selected_indices.append(window_position * fixture_width)
                    if len(selected_indices) >= 4:
                        break
                # Tiny smoke/single-motion runs still get four deterministic
                # windows where available, but are recorded as same-motion.
                if len(selected_indices) < min(4, len(manifest)):
                    for window_position in np.linspace(0, len(manifest) - 1,
                                                        num=min(4, len(manifest)), dtype=int).tolist():
                        fixture_index = window_position * fixture_width
                        if fixture_index not in selected_indices:
                            selected_indices.append(fixture_index)
                        if len(selected_indices) >= 4:
                            break
                selected = [eval_fixtures[i] for i in selected_indices]
                cpu = default_collate(selected)
                sm, am, _ = make_physical_masks(cpu, mask_seed)
                donors = torch.roll(torch.arange(len(selected), device=device), 1)
                was_training = model.training
                model.eval()
                result = ablations(model, to_device(cpu, device), sm.to(device), am.to(device), code, donors)
                model.train(was_training)
                identities = batch_ids(cpu)
                result["selection"] = {
                    "strategy": "one_window_per_motion_then_evenly_spaced",
                    "selected_fixture_indices": selected_indices,
                    "selected_identities": identities,
                    "cross_motion_donor_count": sum(
                        identities[i]["motion_key"] != identities[int(donors[i])]["motion_key"]
                        for i in range(len(selected))
                    ),
                }
                atomic_write_json(run / f"evaluations/step_{last_step:09d}/ablations.json", result)
        plots(run, eval_rows)
        progress(quality_warnings=warnings)
        print(f"[stage={code}] eval step={last_step}/{maximum} selection={row['selection_score']:.6g} full={row['total_loss']:.6g} state={row['state_rmse']:.6g} action={row['action_rmse']:.6g} max={row['max_abs']:.6g} best={best_step} warnings={warnings}", flush=True)
    previous_handlers = {}
    encoder_calls = {"posterior": 0, "condition": 0}
    def count_call(name):
        def hook(module, args):
            encoder_calls[name] += 1
        return hook
    route_hooks = [model.posterior_encoder.register_forward_pre_hook(count_call("posterior")),
                   model.condition_encoder.register_forward_pre_hook(count_call("condition"))]
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, lambda number, frame: stop_requested.append(number))
    try:
        if resumed:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            for stale_name in ("cvae.interrupted", "cvae.failed"):
                stale = run / "markers" / stale_name
                if stale.exists():
                    archive = run / "markers/history" / f"{stamp}_{stale_name}"
                    archive.parent.mkdir(exist_ok=True)
                    stale.replace(archive)
            # Roll back only this unfinished run's records beyond its durable checkpoint.
            for log in (metrics_path, run / "logs/evaluations.jsonl"):
                rows = load_rows(log)
                atomic_write_text(log, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r.get("optimizer_step", 0) <= start))
            eval_rows[:] = [r for r in eval_rows if r["optimizer_step"] <= start]
            if best is None:
                eval_rows.clear()  # step-0 evaluation may have logged before its checkpoint commit
            for filename, expected_step in (
                ("best.pt", best_step),
                ("best_fixed.pt", best_step if code == "B" else None),
                ("best_heldout.pt", best_heldout_step if code == "B" else None),
                ("best_reconstruction.pt", best_reconstruction_step),
            ):
                target = run / "checkpoints" / filename
                if expected_step is None and not target.exists():
                    continue
                actual_step = torch.load(target, map_location="cpu", weights_only=False).get("optimizer_step") if target.exists() else None
                if actual_step != expected_step:
                    if expected_step == start:
                        durable_save(target, resumed)
                    elif expected_step is not None:
                        raise ValueError(f"{filename} inconsistent with resume checkpoint")
            restore_rng(resumed["rng_state"])
        elif options["continue_checkpoint"] and checkpoint.get("format_version") == CHECKPOINT:
            restore_rng(checkpoint["rng_state"])
        if start == 0 and not resumed:
            save()
            full_eval()
        elif best is None or not eval_rows or (start == maximum and eval_rows[-1]["optimizer_step"] != maximum):
            full_eval()
        progress()
        model.train()
        for step in range(start + 1, maximum + 1):
            if stop_requested:
                break
            committed = False
            started = time.perf_counter()
            cpu = sampler.next(fixtures)
            batch = to_device(cpu, device)
            encoder_calls.update(posterior=0, condition=0)
            if code == "A":
                output = model(batch, stage="A")
            else:
                dynamic = source_step + step if options["mask_mode"] == "dynamic" else None
                sm, am, names = make_physical_masks(cpu, mask_seed, dynamic_step=dynamic)
                output = model(batch, sm.to(device), am.to(device), stage=code)
            expected_calls = {"posterior": int(code in {"A", "C"}), "condition": int(code in {"B", "C"})}
            if encoder_calls != expected_calls:
                raise RuntimeError(f"stage route violation: {encoder_calls} expected {expected_calls}")
            vs = batch["valid_state"][..., None].expand_as(batch["physical_state"])
            va = batch["valid_action"][..., None].expand_as(batch["action"])
            reconstruction = weighted_reconstruction_loss(output, batch, vs, va, masked_weight=0., full_weight=1.)
            kl = hierarchical_kl(output.posterior) if code == "C" else {k: reconstruction["total"] * 0 for k in ("global", "local", "total")}
            actual_beta = beta * min((source_step + step) / beta_warmup, 1.) if code == "C" else 0.
            loss = reconstruction["total"] + actual_beta * kl["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss before update {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            group_norms = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    group = name.split('.')[0]
                    squared = parameter.grad.detach().square().sum()
                    group_norms[group] = group_norms.get(group, 0.) + squared
            norm_values = torch.stack(list(group_norms.values())).sqrt().detach().cpu().tolist()
            gradient_observations = dict(zip(group_norms, norm_values))
            grad_seen.update(group for group, norm in gradient_observations.items() if norm > 0)
            before = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            used_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            # Validate updated parameters before marking the boundary committed.
            if not bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()):
                raise FloatingPointError("non-finite parameters after update; preserving preceding durable checkpoint")
            last_step, committed = step, True
            row = {"phase": "train", "optimizer_step": step, "source_step": source_step, "cumulative_step": source_step + step,
                "loss": float(loss.detach()), "reconstruction": float(reconstruction["total"].detach()),
                **{f"reconstruction_{k}": float(reconstruction[k].detach()) for k in ("state", "action", "contact")},
                "kl_global": float(kl["global"].detach()), "kl_local": float(kl["local"].detach()), "kl": float(kl["total"].detach()),
                "kl_beta": actual_beta, "weighted_kl_global": actual_beta * .5 * float(kl["global"].detach()),
                "weighted_kl_local": actual_beta * .5 * float(kl["local"].detach()), "learning_rate": used_lr,
                "next_learning_rate": optimizer.param_groups[0]["lr"], "gradient_norm_before_clip": float(before),
                "gradient_norm_after_clip": float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters if p.grad is not None]))),
                "batch_size": len(cpu["physical_state"]), "sample_identity_sha256": digest(batch_ids(cpu)),
                "sample_exposures": sampler.exposures, "sampler_epoch": sampler.epoch, "step_seconds": time.perf_counter() - started,
                "elapsed_seconds": time.perf_counter() - started_at, "updated_at": datetime.now(timezone.utc).isoformat()}
            row["encoder_calls"] = dict(encoder_calls)
            if code != "A":
                row["mask_sha256"] = hashlib.sha256(sm.numpy().tobytes() + am.numpy().tobytes()).hexdigest()
                row["mask_names"] = names
            append(metrics_path, row)
            if step == 1 or step % intervals["log_interval"] == 0:
                progress()
                print(f"[stage={code}] step={step}/{maximum} loss={row['loss']:.6g} lr_used={used_lr:.4g} grad={float(before):.4g} exposures={sampler.exposures}", flush=True)
            if stop_requested:
                break
            if step % intervals["validation_interval"] == 0 or step == maximum:
                full_eval()
            elif step % intervals["checkpoint_interval"] == 0:
                save()
                progress()
        if stop_requested:
            save()
            progress("interrupted")
            atomic_write_text(run / "markers/cvae.interrupted", "INTERRUPTED AT COMMITTED UPDATE BOUNDARY\n")
            return {"interrupted": True, "stage": code, "model_contract": {"actual_parameter_count": sum(p.numel() for p in model.parameters())}}
        save()
        readback = strict_readback(run / "checkpoints/last.pt", model, optimizer, scheduler, probe, code, mask_seed)
        best_readback = strict_readback(run / "checkpoints/best.pt", model, optimizer, scheduler, probe, code, mask_seed)
        best_fixed_readback = (
            strict_readback(run / "checkpoints/best_fixed.pt", model, optimizer, scheduler, probe, code, mask_seed)
            if code == "B" else None
        )
        best_heldout_readback = (
            strict_readback(run / "checkpoints/best_heldout.pt", model, optimizer, scheduler, probe, code, mask_seed)
            if code == "B" and best_heldout_step is not None else None
        )
        summary = {"format_version": "sonic_65_token_hierarchical_standard_cvae_summary_v2", "protocol_version": PROTOCOL,
            "architecture_version": model.ARCHITECTURE_VERSION, "stage": code, "execution_pass": True, "quality_pass": None,
            "smoke": options["smoke"], "completed_optimizer_steps": last_step, "cumulative_step": source_step + last_step,
            "source_checkpoint": source, "training_contract": contract, "dataset_identity": identity,
            "model_contract": {**model.config, "actual_parameter_count": sum(p.numel() for p in model.parameters())},
            "stage_parameter_counts": counts, "best_step": best_step, "best_metrics": best,
            "best_heldout_step": best_heldout_step, "best_heldout_metrics": best_heldout,
            "checkpoint_readback": readback, "best_checkpoint_readback": best_readback,
            "best_fixed_checkpoint_readback": best_fixed_readback,
            "best_heldout_checkpoint_readback": best_heldout_readback,
            "evaluation_count": evaluation_count, "quality_warnings": quality_warnings(eval_rows),
            "artifacts": {"metrics": "logs/metrics.jsonl", "evaluations": "logs/evaluations.jsonl", "diagnostics": "evaluations/", "audits": "manifests/audit_*.json"},
            "unique_next_step": "REVIEW_REPORT_BEFORE_ANY_NEXT_STAGE"}
        atomic_write_json(run / "manifests/standard_cvae_summary.json", summary)
        marker = "cvae_posterior_standard_cvae_smoke.ok" if options["smoke"] else "cvae_posterior_standard_cvae_execution.ok"
        atomic_write_text(run / "markers" / marker, "ENGINEERING PASS; QUALITY NOT ASSESSED\n")
        progress("completed", quality_warnings=summary["quality_warnings"])
        return summary
    except BaseException as error:
        # Never overwrite last.pt after an incomplete/failed optimizer update.
        progress("failed", error=f"{type(error).__name__}: {error}", safe_checkpoint_preserved=True)
        atomic_write_json(run / "manifests/failure.json", {"error": repr(error), "last_complete_step": last_step,
            "update_boundary_complete": committed, "durable_checkpoint": str(run / "checkpoints/last.pt")})
        raise
    finally:
        for hook in route_hooks:
            hook.remove()
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
