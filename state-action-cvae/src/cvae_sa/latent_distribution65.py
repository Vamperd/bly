"""Read-only posterior latent distribution diagnostics for the 65-token C model.

This module does not decode, train, or modify a checkpoint.  It evaluates the
posterior encoder on complete State--Action windows and reports the seventeen
latent groups used by the hierarchical standard CVAE: one 256-dimensional
global group and sixteen 128-dimensional local groups.  The report keeps
posterior means, posterior standard deviations, sampled aggregate statistics,
and per-dimension KL values separate so a small scalar KL cannot hide a
collapsed or over-specialised chunk.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cvae_protocol import Fixtures
from .cvae_training import check_source_identity, data_identity, make_dataset
from .models import build_model
from .posterior_hierarchical_standard_cvae import load_checkpoint
from .replay65 import checkpoint_config
from .util import atomic_write_json, atomic_write_text, file_sha256


VERSION = "65-token-latent-distribution-v1"
GROUP_NAMES = ("global",) + tuple(f"local_{index:02d}" for index in range(16))
RUNS_ROOT = Path("/home/helloworld/bly/runs")


def _finite_array(name: str, value: np.ndarray, *, ndim: int | None = None) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def kl_per_dimension(mean: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    """Return q(z|x)||N(0,I) KL per window and latent dimension."""
    mean = _finite_array("posterior mean", mean)
    logvar = _finite_array("posterior logvar", logvar)
    if mean.shape != logvar.shape:
        raise ValueError(f"mean/logvar shape mismatch: {mean.shape} != {logvar.shape}")
    return 0.5 * (np.exp(logvar.astype(np.float64)) + mean.astype(np.float64) ** 2 - 1.0 - logvar)


def _scalar_stats(value: np.ndarray) -> dict[str, float | int]:
    flat = _finite_array("statistics input", value).reshape(-1).astype(np.float64)
    if flat.size == 0:
        raise ValueError("cannot summarize an empty latent group")
    return {
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "mean_abs": float(np.abs(flat).mean()),
        "q01": float(np.quantile(flat, 0.01)),
        "q05": float(np.quantile(flat, 0.05)),
        "q50": float(np.quantile(flat, 0.50)),
        "q95": float(np.quantile(flat, 0.95)),
        "q99": float(np.quantile(flat, 0.99)),
    }


def _group_summary(name: str, mean: np.ndarray, logvar: np.ndarray, samples: np.ndarray) -> dict:
    """Summarize one global/local group with explicit per-dimension diagnostics."""
    mean = _finite_array(f"{name} mean", mean, ndim=2)
    logvar = _finite_array(f"{name} logvar", logvar, ndim=2)
    samples = _finite_array(f"{name} samples", samples)
    if mean.shape != logvar.shape or mean.shape[0] != samples.shape[-2] or mean.shape[1] != samples.shape[-1]:
        raise ValueError(
            f"{name} group shapes disagree: mean={mean.shape}, logvar={logvar.shape}, samples={samples.shape}"
        )
    kl = kl_per_dimension(mean, logvar)
    sigma = np.exp(0.5 * logvar.astype(np.float64)).astype(np.float32)
    sample_flat = samples.reshape(-1, mean.shape[1])
    per_dim = {
        "kl_mean": kl.mean(axis=0).tolist(),
        "kl_p95": np.quantile(kl, 0.95, axis=0).tolist(),
        "posterior_mean_mean": mean.mean(axis=0).tolist(),
        "posterior_mean_std_across_windows": mean.std(axis=0).tolist(),
        "posterior_sigma_mean": sigma.mean(axis=0).tolist(),
        "posterior_sigma_std": sigma.std(axis=0).tolist(),
        "sample_mean": sample_flat.mean(axis=0).tolist(),
        "sample_std": sample_flat.std(axis=0).tolist(),
    }
    kl_mean = float(kl.mean())
    sample_stats = _scalar_stats(sample_flat)
    sample_std_gap = abs(sample_stats["std"] - 1.0)
    sample_mean_gap = abs(sample_stats["mean"])
    mean_variance = mean.astype(np.float64).var(axis=0)
    kl_dim_mean = kl.mean(axis=0)
    return {
        "group": name,
        "window_count": int(mean.shape[0]),
        "latent_dim": int(mean.shape[1]),
        "kl_mean": kl_mean,
        "kl_median": float(np.median(kl)),
        "kl_p95": float(np.quantile(kl, 0.95)),
        "kl_max": float(kl.max()),
        "kl_dim_fraction_gt_1e-4": float(np.mean(kl_dim_mean > 1e-4)),
        "kl_dim_fraction_gt_1e-2": float(np.mean(kl_dim_mean > 1e-2)),
        "kl_dim_fraction_gt_1e-1": float(np.mean(kl_dim_mean > 1e-1)),
        "posterior_mean": _scalar_stats(mean),
        "posterior_sigma": _scalar_stats(sigma),
        "posterior_sample": sample_stats,
        "standard_normal_gap": {
            "sample_mean_abs_gap": float(sample_mean_gap),
            "sample_std_abs_gap": float(sample_std_gap),
            "sample_mean_z_score_like": float(sample_mean_gap / max(sample_stats["std"], 1e-12)),
        },
        "active_mean_variance_fraction_gt_1e-2": float(np.mean(mean_variance > 1e-2)),
        "active_mean_variance_fraction_gt_1e-1": float(np.mean(mean_variance > 1e-1)),
        "per_dimension": per_dim,
    }


def _config_from_checkpoint(checkpoint: dict) -> dict:
    config = checkpoint_config(checkpoint)
    model_config = dict(config.get("model", {}))
    if model_config.get("kind") != "physics_hierarchical_standard_cvae_transformer":
        raise ValueError("latent distribution diagnostic requires the 65-token standard CVAE checkpoint")
    config["model"] = model_config
    config.setdefault("data", {"window_transitions": 64, "stride": 64, "max_episodes": 256, "num_workers": 0})
    return config


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _write_plots(run: Path, group_summaries: list[dict], arrays: dict[str, np.ndarray], *, seed: int) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - depends on optional Ubuntu plotting runtime
        atomic_write_text(run / "plots/README.txt", f"matplotlib unavailable: {type(error).__name__}: {error}\n")
        return []

    plots = run / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    names = [row["group"] for row in group_summaries]
    x = np.arange(len(names))
    sample_mean = np.array([row["posterior_sample"]["mean"] for row in group_summaries])
    sample_std = np.array([row["posterior_sample"]["std"] for row in group_summaries])
    kl = np.array([row["kl_mean"] for row in group_summaries])
    paths: list[str] = []

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
    axes[0].bar(x, sample_mean, color=["#1f77b4"] + ["#2ca02c"] * 16)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_ylabel("posterior sample mean")
    axes[0].set_xticks(x, names, rotation=45, ha="right")
    axes[1].bar(x, sample_std, color=["#ff7f0e"] + ["#9467bd"] * 16)
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--", label="N(0,1) std")
    axes[1].set_ylabel("posterior sample std")
    axes[1].set_xticks(x, names, rotation=45, ha="right")
    axes[1].legend()
    axes[1].set_xlabel("latent group")
    path = plots / "latent_group_mean_std.svg"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path.relative_to(run)))

    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    ax.bar(x, kl, color=["#d62728"] + ["#17becf"] * 16)
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylabel("mean KL per latent dimension")
    ax.set_xlabel("q(z|complete State, Action) vs N(0,1)")
    path = plots / "latent_group_kl.svg"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path.relative_to(run)))

    fig, axes = plt.subplots(5, 4, figsize=(16, 14), sharex=True, sharey=True, constrained_layout=True)
    axis_list = axes.reshape(-1)
    normal_x = np.linspace(-4.0, 4.0, 241)
    normal_y = np.exp(-0.5 * normal_x**2) / np.sqrt(2.0 * np.pi)
    rng = np.random.default_rng(seed)
    for index, name in enumerate(names):
        values = arrays[f"sample_{name}"].reshape(-1)
        if values.size > 20000:
            values = rng.choice(values, size=20000, replace=False)
        axis = axis_list[index]
        axis.hist(np.clip(values, -5.0, 5.0), bins=40, density=True, alpha=0.65, color="#4c78a8")
        axis.plot(normal_x, normal_y, color="black", linewidth=1)
        axis.set_title(name)
        axis.grid(alpha=0.2)
    for axis in axis_list[len(names):]:
        axis.axis("off")
    fig.suptitle("posterior samples vs standard normal; each panel is one latent group")
    path = plots / "latent_group_distributions.svg"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path.relative_to(run)))

    for key, title, filename in (
        ("global_kl_mean", "global per-dimension mean KL", "global_dimension_kl.svg"),
        ("local_kl_mean", "local per-chunk per-dimension mean KL", "local_dimension_kl.svg"),
        ("global_mean_variance", "global posterior-mean variance across windows", "global_mean_variance.svg"),
        ("local_mean_variance", "local posterior-mean variance across windows", "local_mean_variance.svg"),
        ("global_sigma_mean", "global posterior sigma", "global_sigma.svg"),
        ("local_sigma_mean", "local posterior sigma", "local_sigma.svg"),
    ):
        value = arrays[key]
        if value.ndim == 1:
            value = value[None, :]
        fig, ax = plt.subplots(figsize=(14, max(2.0, 0.45 * value.shape[0])), constrained_layout=True)
        image = ax.imshow(value, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("latent dimension")
        ax.set_ylabel("local chunk" if value.shape[0] > 1 else "global")
        fig.colorbar(image, ax=ax)
        path = plots / filename
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path.relative_to(run)))
    return paths


def _archive(run: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in run.rglob("*"):
            if path.is_file() and path != output:
                archive.write(path, path.relative_to(run))


def analyze(args: argparse.Namespace) -> dict:
    run = Path(args.output_run).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_run = Path(args.dataset_run).expanduser().resolve()
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise FileNotFoundError(f"checkpoint missing or empty: {checkpoint_path}")
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"output run is non-empty; choose a new run: {run}")
    if not run.is_relative_to(RUNS_ROOT):
        raise ValueError(f"output must be under {RUNS_ROOT}: {run}")
    run.mkdir(parents=True, exist_ok=True)
    for name in ("manifests", "data", "plots", "logs", "markers"):
        (run / name).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _config_from_checkpoint(checkpoint)
    dataset, all_indices = make_dataset(dataset_run, config)
    indices = all_indices[: int(args.max_windows)] if args.max_windows is not None else all_indices
    if not indices:
        dataset.close()
        raise ValueError("no windows selected")
    fixtures = Fixtures(dataset, indices, expand=False)
    identity = data_identity(dataset_run, fixtures.manifest())
    identity_check = check_source_identity(
        checkpoint,
        identity,
        allow_unknown=True,
        allow_manifest_mismatch=bool(args.allow_recovered_dataset),
        allow_identity_mismatch=bool(args.allow_recovered_identity),
    )
    device = _device(args.device)
    model = build_model(config["model"]).to(device)
    loaded = load_checkpoint(model, checkpoint_path)
    model.eval()

    global_means: list[np.ndarray] = []
    global_logvars: list[np.ndarray] = []
    local_means: list[np.ndarray] = []
    local_logvars: list[np.ndarray] = []
    global_samples: list[np.ndarray] = []
    local_samples: list[np.ndarray] = []
    loader = DataLoader(fixtures, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = _to_device(cpu_batch, device)
            posterior = model.encode_posterior_distribution(batch)
            gm = posterior.global_mean.float().cpu()
            glv = posterior.global_logvar.float().cpu()
            lm = posterior.local_mean.float().cpu()
            llv = posterior.local_logvar.float().cpu()
            std_g = torch.exp(0.5 * glv)
            std_l = torch.exp(0.5 * llv)
            eps_g = torch.randn((int(args.sample_draws), gm.shape[0], gm.shape[1]), generator=generator)
            eps_l = torch.randn((int(args.sample_draws), lm.shape[0], lm.shape[1], lm.shape[2]), generator=generator)
            global_means.append(gm.numpy())
            global_logvars.append(glv.numpy())
            local_means.append(lm.numpy())
            local_logvars.append(llv.numpy())
            global_samples.append((gm[None] + std_g[None] * eps_g).numpy())
            local_samples.append((lm[None] + std_l[None] * eps_l).numpy())
    dataset.close()

    arrays = {
        "global_mean": np.concatenate(global_means, axis=0),
        "global_logvar": np.concatenate(global_logvars, axis=0),
        "local_mean": np.concatenate(local_means, axis=0),
        "local_logvar": np.concatenate(local_logvars, axis=0),
        "sample_global": np.concatenate(global_samples, axis=1),
        "sample_local": np.concatenate(local_samples, axis=1),
    }
    arrays.update({
        "sample_global": arrays["sample_global"].astype(np.float32),
        "sample_local": arrays["sample_local"].astype(np.float32),
    })
    arrays["global_kl_mean"] = kl_per_dimension(arrays["global_mean"], arrays["global_logvar"]).mean(axis=0).astype(np.float32)
    arrays["local_kl_mean"] = kl_per_dimension(arrays["local_mean"], arrays["local_logvar"]).mean(axis=0).astype(np.float32)
    arrays["global_mean_variance"] = arrays["global_mean"].var(axis=0).astype(np.float32)
    arrays["local_mean_variance"] = arrays["local_mean"].var(axis=0).astype(np.float32)
    arrays["global_sigma_mean"] = np.exp(0.5 * arrays["global_logvar"]).mean(axis=0).astype(np.float32)
    arrays["local_sigma_mean"] = np.exp(0.5 * arrays["local_logvar"]).mean(axis=0).astype(np.float32)
    np.savez_compressed(run / "data/latent_arrays.npz", **arrays)

    group_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "global": (arrays["global_mean"], arrays["global_logvar"], arrays["sample_global"]),
    }
    for index in range(16):
        group_arrays[f"local_{index:02d}"] = (
            arrays["local_mean"][:, index], arrays["local_logvar"][:, index], arrays["sample_local"][:, :, index],
        )
    summaries = [_group_summary(name, *group_arrays[name]) for name in GROUP_NAMES]
    plots = _write_plots(
        run,
        summaries,
        {**arrays, **{f"sample_{name}": group_arrays[name][2] for name in GROUP_NAMES}},
        seed=int(args.seed),
    )
    manifest = {
        "format_version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"path": str(checkpoint_path), "sha256": file_sha256(checkpoint_path), "stage": loaded.get("stage")},
        "dataset_run": str(dataset_run),
        "window_count": len(indices),
        "window_indices": [int(index) for index in indices],
        "sample_seed": int(args.seed),
        "sample_draws": int(args.sample_draws),
        "device": str(device),
        "recovered_dataset": bool(args.allow_recovered_dataset),
        "recovered_identity_override": bool(args.allow_recovered_identity),
        "identity_check": identity_check,
        "groups": list(GROUP_NAMES),
        "plots": plots,
    }
    atomic_write_json(run / "manifests/latent_distribution.json", manifest)
    atomic_write_json(run / "manifests/group_stats.json", {"groups": summaries})
    atomic_write_json(run / "manifests/source_identity.json", {"identity": identity, "identity_check": identity_check})
    atomic_write_text(run / "markers/latent_distribution_diagnostic.ok", "READ-ONLY POSTERIOR LATENT DISTRIBUTION DIAGNOSTIC\n")
    report_path = run / "latent_distribution_report.zip"
    _archive(run, report_path)
    result = {"run": str(run), "report": str(report_path), "window_count": len(indices), "groups": list(GROUP_NAMES), "identity_check": identity_check}
    print(json.dumps(result, ensure_ascii=False))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only 17-group posterior latent distribution diagnostic")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-run", required=True)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-draws", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-recovered-dataset", action="store_true")
    parser.add_argument("--allow-recovered-identity", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_windows is not None and args.max_windows <= 0:
        raise ValueError("--max-windows must be positive")
    if args.batch_size <= 0 or args.sample_draws <= 0:
        raise ValueError("--batch-size and --sample-draws must be positive")
    analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
