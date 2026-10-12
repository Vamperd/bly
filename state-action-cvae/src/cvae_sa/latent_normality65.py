"""Per-dimension normality diagnostics for the 65-token posterior latent report.

This is a read-only postprocessor for ``latent_distribution65`` output.  It
does not load or modify a checkpoint, train a model, or decode predictions.
For every global/local latent dimension it reports both shape-normality and
closeness to N(0,1), then writes heatmaps and histograms for the worst
dimensions.  The thresholds used for labels are diagnostic heuristics, not a
training-quality gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from .latent_distribution65 import GROUP_NAMES, _finite_array
from .util import atomic_write_json, atomic_write_text, file_sha256


VERSION = "65-token-latent-normality-v1"
RUNS_ROOT = Path("/home/helloworld/bly/runs")


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import ndtr
        return np.asarray(ndtr(value), dtype=np.float64)
    except Exception:  # pragma: no cover - fallback for minimal Ubuntu envs
        erf = np.vectorize(math.erf, otypes=[np.float64])
        return 0.5 * (1.0 + erf(np.asarray(value, dtype=np.float64) / math.sqrt(2.0)))


def _ks_statistic(values: np.ndarray) -> np.ndarray:
    """Vectorized one-sample KS statistic against N(0,1), per column."""
    values = _finite_array("normality values", values, ndim=2).astype(np.float64)
    standardized = np.sort(values, axis=0)
    n = standardized.shape[0]
    cdf = _normal_cdf(standardized)
    ranks = np.arange(1, n + 1, dtype=np.float64)[:, None]
    return np.maximum((ranks / n - cdf).max(axis=0), (cdf - (ranks - 1.0) / n).max(axis=0))


def dimension_statistics(values: np.ndarray) -> dict[str, np.ndarray | list[bool]]:
    """Return shape and standard-normal diagnostics for columns of ``values``."""
    values = _finite_array("normality values", values, ndim=2).astype(np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    safe_std = np.maximum(std, 1e-12)
    standardized = (values - mean[None, :]) / safe_std[None, :]
    skew = np.mean(standardized ** 3, axis=0)
    excess_kurtosis = np.mean(standardized ** 4, axis=0) - 3.0
    q01, q05, q50, q95, q99 = np.quantile(values, (0.01, 0.05, 0.50, 0.95, 0.99), axis=0)
    normal_q01, normal_q05, normal_q50, normal_q95, normal_q99 = (-2.3263479, -1.6448536, 0.0, 1.6448536, 2.3263479)
    shape_ks = _ks_statistic(standardized)
    standard_ks = _ks_statistic(values)
    q_gap = np.maximum(np.abs(q01 - normal_q01), np.abs(q99 - normal_q99))
    shape_normal = (shape_ks <= 0.05) & (np.abs(skew) <= 0.20) & (np.abs(excess_kurtosis) <= 0.50)
    near_standard = shape_normal & (np.abs(mean) <= 0.05) & (np.abs(std - 1.0) <= 0.05) & (q_gap <= 0.25)
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "mean_gap": np.abs(mean).astype(np.float32),
        "std_gap": np.abs(std - 1.0).astype(np.float32),
        "skew": skew.astype(np.float32),
        "excess_kurtosis": excess_kurtosis.astype(np.float32),
        "q01": q01.astype(np.float32),
        "q05": q05.astype(np.float32),
        "q50": q50.astype(np.float32),
        "q95": q95.astype(np.float32),
        "q99": q99.astype(np.float32),
        "q_tail_gap": q_gap.astype(np.float32),
        "shape_ks": shape_ks.astype(np.float32),
        "standard_normal_ks": standard_ks.astype(np.float32),
        "shape_approximately_normal": shape_normal.tolist(),
        "near_standard_normal": near_standard.tolist(),
    }


def _as_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _stats_json(stats: dict[str, np.ndarray | list[bool]]) -> dict:
    return {key: _as_json(value) for key, value in stats.items()}


def _groups_from_arrays(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    groups: dict[str, dict[str, np.ndarray]] = {
        "global": {
            "sample": arrays["sample_global"].reshape(-1, arrays["sample_global"].shape[-1]),
            "posterior_mean": arrays["global_mean"],
        }
    }
    for index in range(16):
        groups[f"local_{index:02d}"] = {
            "sample": arrays["sample_local"][:, :, index, :].reshape(-1, arrays["sample_local"].shape[-1]),
            "posterior_mean": arrays["local_mean"][:, index, :],
        }
    return groups


def _plot_heatmap(ax, value: np.ndarray, title: str, *, cmap: str = "viridis", vline: float | None = None):
    image = ax.imshow(value[None, :] if value.ndim == 1 else value, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("latent dimension" if value.ndim == 1 else "local dimension")
    if vline is not None:
        ax.axvline(vline, color="white", linewidth=0.5)
    return image


def _write_plots(run: Path, group_data: dict[str, dict], arrays: dict[str, dict[str, np.ndarray]], top_k: int) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - optional plotting dependency
        atomic_write_text(run / "plots/README.txt", f"matplotlib unavailable: {type(error).__name__}: {error}\n")
        return []

    paths: list[str] = []
    plots = run / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    metrics = ("standard_normal_ks", "mean_gap", "std_gap", "excess_kurtosis")
    titles = ("KS vs N(0,1)", "|sample mean|", "|sample std - 1|", "excess kurtosis")
    for group in GROUP_NAMES:
        stats = arrays[group]
        fig, axes = plt.subplots(4, 1, figsize=(16, 10), constrained_layout=True)
        for ax, metric, title in zip(axes, metrics, titles):
            value = np.asarray(stats[metric])
            image = _plot_heatmap(ax, value, f"{group}: {title}", cmap="magma" if metric == "standard_normal_ks" else "viridis")
            fig.colorbar(image, ax=ax, pad=0.01)
        path = plots / f"{group}_per_dimension_normality.svg"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path.relative_to(run)))

    entries = []
    for group in GROUP_NAMES:
        score = np.asarray(arrays[group]["standard_normal_ks"])
        for index, value in enumerate(score):
            entries.append((float(value), group, index))
    entries.sort(reverse=True)
    normal_x = np.linspace(-4.0, 4.0, 241)
    normal_y = np.exp(-0.5 * normal_x ** 2) / np.sqrt(2.0 * np.pi)
    fig, axes = plt.subplots(6, 4, figsize=(16, 20), constrained_layout=True)
    for ax, (_, group, index) in zip(axes.reshape(-1), entries[:top_k]):
        values = group_data[group]["sample"][:, index]
        ax.hist(np.clip(values, -5.0, 5.0), bins=40, density=True, alpha=0.65, color="#4c78a8")
        ax.plot(normal_x, normal_y, color="black", linewidth=1)
        ax.set_title(f"{group}[{index}] KS={arrays[group]['standard_normal_ks'][index]:.3f}")
        ax.grid(alpha=0.2)
    for ax in axes.reshape(-1)[len(entries[:top_k]):]:
        ax.axis("off")
    fig.suptitle("worst latent dimensions: posterior samples vs N(0,1)")
    path = plots / "worst_dimensions_histograms.svg"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path.relative_to(run)))
    return paths


def analyze(args: argparse.Namespace) -> dict:
    source = Path(args.latent_report).expanduser().resolve()
    run = Path(args.output_run).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("--latent-report must be an unpacked latent_distribution65 output directory")
    source_npz = source / "data/latent_arrays.npz"
    source_manifest = source / "manifests/latent_distribution.json"
    if not source_npz.is_file() or not source_manifest.is_file():
        raise FileNotFoundError("latent report must contain data/latent_arrays.npz and manifests/latent_distribution.json")
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"output run is non-empty; choose a new run: {run}")
    if not run.is_relative_to(RUNS_ROOT):
        raise ValueError(f"output must be under {RUNS_ROOT}")
    run.mkdir(parents=True, exist_ok=True)
    for name in ("manifests", "data", "plots", "markers"):
        (run / name).mkdir(parents=True, exist_ok=True)

    arrays = {key: value for key, value in np.load(source_npz).items()}
    groups = _groups_from_arrays(arrays)
    sample_stats = {name: dimension_statistics(value["sample"]) for name, value in groups.items()}
    mean_stats = {name: dimension_statistics(value["posterior_mean"]) for name, value in groups.items()}

    compact_arrays: dict[str, np.ndarray] = {}
    group_summaries = []
    for name in GROUP_NAMES:
        stats = sample_stats[name]
        mean_shape = mean_stats[name]
        compact_arrays.update({f"{name}_sample_{key}": np.asarray(value) for key, value in stats.items() if isinstance(value, np.ndarray)})
        compact_arrays.update({f"{name}_mean_{key}": np.asarray(value) for key, value in mean_shape.items() if isinstance(value, np.ndarray)})
        group_summaries.append({
            "group": name,
            "sample_count": int(groups[name]["sample"].shape[0]),
            "latent_dim": int(groups[name]["sample"].shape[1]),
            "sample_shape_normal_fraction": float(np.mean(stats["shape_approximately_normal"])),
            "sample_near_standard_normal_fraction": float(np.mean(stats["near_standard_normal"])),
            "sample_max_standard_normal_ks": float(np.max(stats["standard_normal_ks"])),
            "sample_mean_standard_normal_ks": float(np.mean(stats["standard_normal_ks"])),
            "posterior_mean_shape_normal_fraction": float(np.mean(mean_shape["shape_approximately_normal"])),
            "sample_per_dimension": _stats_json(stats),
            "posterior_mean_per_dimension": _stats_json(mean_shape),
        })
    np.savez_compressed(run / "data/normality_arrays.npz", **compact_arrays)
    plots = _write_plots(run, groups, sample_stats, int(args.top_k))
    source_info = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest = {
        "format_version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(source),
        "source_report_sha256": file_sha256(source_npz),
        "checkpoint": source_info.get("checkpoint"),
        "dataset_run": source_info.get("dataset_run"),
        "window_count": source_info.get("window_count"),
        "sample_draws": source_info.get("sample_draws"),
        "groups": list(GROUP_NAMES),
        "thresholds": {
            "shape_ks_max": 0.05,
            "abs_skew_max": 0.20,
            "abs_excess_kurtosis_max": 0.50,
            "abs_mean_gap_max": 0.05,
            "abs_std_gap_max": 0.05,
            "tail_quantile_gap_max": 0.25,
            "note": "heuristic labels for diagnosis; not a model-quality gate",
        },
        "plots": plots,
    }
    atomic_write_json(run / "manifests/normality_summary.json", {"groups": group_summaries})
    atomic_write_json(run / "manifests/normality_manifest.json", manifest)
    atomic_write_text(run / "markers/latent_normality_diagnostic.ok", "READ-ONLY PER-DIMENSION LATENT NORMALITY DIAGNOSTIC\n")
    report = run / "latent_normality_report.zip"
    with zipfile.ZipFile(report, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in run.rglob("*"):
            if path.is_file() and path != report:
                archive.write(path, path.relative_to(run))
    result = {"run": str(run), "report": str(report), "groups": list(GROUP_NAMES)}
    print(json.dumps(result, ensure_ascii=False))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only per-dimension normality diagnostic for a latent_distribution65 report")
    parser.add_argument("--latent-report", required=True)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--top-k", type=int, default=24)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_k <= 0 or args.top_k > 24:
        raise ValueError("--top-k must be in [1,24]")
    analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
