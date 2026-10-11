"""Per-joint Action error analysis for the 65-token CVAE.

The evaluator stores Action in physical units in each exported fixture trace.
This module aggregates those traces without touching the source dataset or a
checkpoint.  It reports normalized-space, physical-space, target-relative and
symmetric percentage errors separately because Action targets can cross zero.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .constants import ACTION_DIM
from .util import atomic_write_json, atomic_write_text, file_sha256


EPSILON = 1.0e-8
DEFAULT_NEAR_ZERO_FRACTION = 0.05
JOINT_ERROR_FORMAT = "sonic_65_token_action_joint_error_analysis_v1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite {name}")
    return result


def _stats(
    normalized_error: np.ndarray,
    physical_error: np.ndarray,
    physical_prediction: np.ndarray,
    physical_target: np.ndarray,
    action_std: float,
    *,
    near_zero_threshold: float,
) -> dict[str, Any]:
    normalized_error = _finite(normalized_error, "normalized error").reshape(-1)
    physical_error = _finite(physical_error, "physical error").reshape(-1)
    physical_prediction = _finite(physical_prediction, "physical prediction").reshape(-1)
    physical_target = _finite(physical_target, "physical target").reshape(-1)
    count = int(physical_error.size)
    if count == 0:
        return {
            "count": 0,
            "unit": "rad",
            "normalized_unit": "dimensionless",
            "target_relative_unit": "percent",
            "near_zero_threshold_rad": float(near_zero_threshold),
            "near_zero_excluded_count": 0,
            "near_zero_excluded_fraction": None,
            "mape_coverage": None,
            **{key: None for key in (
                "signed_bias_rad", "mae_rad", "rmse_rad", "p50_abs_rad", "p90_abs_rad",
                "p95_abs_rad", "p99_abs_rad", "max_abs_rad", "normalized_mae",
                "normalized_rmse", "rmse_over_dataset_std_pct", "mae_over_mean_abs_target_pct",
                "mape_pct", "smape_pct", "target_mean_abs_rad", "target_rms_rad",
            )},
        }
    abs_physical = np.abs(physical_error)
    abs_normalized = np.abs(normalized_error)
    abs_target = np.abs(physical_target)
    eligible = abs_target > float(near_zero_threshold)
    mape_values = 100.0 * abs_physical[eligible] / np.maximum(abs_target[eligible], EPSILON)
    smape_values = 200.0 * abs_physical / np.maximum(
        np.abs(physical_prediction) + abs_target, EPSILON
    )
    target_mean_abs = float(np.mean(abs_target))
    normalized_rmse = float(np.sqrt(np.mean(np.square(normalized_error))))
    physical_rmse = float(np.sqrt(np.mean(np.square(physical_error))))
    return {
        "count": count,
        "unit": "rad",
        "normalized_unit": "dimensionless",
        "target_relative_unit": "percent",
        "near_zero_threshold_rad": float(near_zero_threshold),
        "near_zero_excluded_count": int((~eligible).sum()),
        "near_zero_excluded_fraction": float(np.mean(~eligible)),
        "mape_coverage": float(np.mean(eligible)),
        "signed_bias_rad": float(np.mean(physical_error)),
        "mae_rad": float(np.mean(abs_physical)),
        "rmse_rad": physical_rmse,
        "p50_abs_rad": float(np.quantile(abs_physical, 0.50)),
        "p90_abs_rad": float(np.quantile(abs_physical, 0.90)),
        "p95_abs_rad": float(np.quantile(abs_physical, 0.95)),
        "p99_abs_rad": float(np.quantile(abs_physical, 0.99)),
        "max_abs_rad": float(np.max(abs_physical)),
        "normalized_mae": float(np.mean(abs_normalized)),
        "normalized_rmse": normalized_rmse,
        "rmse_over_dataset_std_pct": float(normalized_rmse * 100.0),
        "mae_over_mean_abs_target_pct": float(
            100.0 * np.mean(abs_physical) / max(target_mean_abs, EPSILON)
        ),
        "mape_pct": float(np.mean(mape_values)) if mape_values.size else None,
        "smape_pct": float(np.mean(smape_values)),
        "target_mean_abs_rad": target_mean_abs,
        "target_rms_rad": float(np.sqrt(np.mean(np.square(physical_target)))),
    }


def _svg_escape(value: Any) -> str:
    return (str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def _bar_svg(path: Path, title: str, ylabel: str, labels: list[str], series: list[tuple[str, list[float], str]], *, suffix: str = "") -> None:
    width, height = 1280, 620
    left, right, top, bottom = 90, 30, 70, 170
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [float(v) for _, rows, _ in series for v in rows if np.isfinite(v)]
    ymax = max(values + [1.0]) * 1.15
    if ymax <= 0:
        ymax = 1.0
    group_w = plot_w / max(len(labels), 1)
    bar_w = min(24.0, group_w / max(len(series), 1) * 0.72)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="30" font-size="20">{_svg_escape(title)}</text>',
        f'<text x="24" y="52" font-size="12" fill="#555">{_svg_escape(ylabel)}{_svg_escape(suffix)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for tick in range(6):
        value = ymax * tick / 5.0
        y = top + plot_h - plot_h * value / ymax
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="10">{value:.3g}</text>')
    for index, label in enumerate(labels):
        x_center = left + group_w * (index + 0.5)
        lines.append(f'<text x="{x_center:.2f}" y="{top + plot_h + 18}" text-anchor="middle" font-size="9" transform="rotate(45 {x_center:.2f} {top + plot_h + 18})">{_svg_escape(label)}</text>')
        for series_index, (_, rows, color) in enumerate(series):
            value = rows[index] if index < len(rows) and np.isfinite(rows[index]) else 0.0
            x = x_center + (series_index - (len(series) - 1) / 2.0) * bar_w
            bar_h = plot_h * max(value, 0.0) / ymax
            y = top + plot_h - bar_h
            lines.append(f'<rect x="{x - bar_w / 2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{color}"/>')
    for index, (name, _, color) in enumerate(series):
        x = width - 260 + (index % 2) * 130
        y = 28 + (index // 2) * 18
        lines.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{x + 17}" y="{y}" font-size="10" fill="{color}">{_svg_escape(name)}</text>')
    lines.append('</svg>')
    atomic_write_text(path, "\n".join(lines) + "\n")


def _heatmap_svg(path: Path, title: str, matrix: np.ndarray, x_labels: list[str], y_labels: list[str], *, unit: str) -> None:
    matrix = np.asarray(matrix, dtype=np.float64)
    height, width = 640, 1280
    left, top, right, bottom = 170, 70, 40, 130
    cell_w = (width - left - right) / max(matrix.shape[1], 1)
    cell_h = (height - top - bottom) / max(matrix.shape[0], 1)
    finite = matrix[np.isfinite(matrix)]
    scale = max(float(np.quantile(finite, 0.99)) if finite.size else 1.0, EPSILON)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="30" font-size="20">{_svg_escape(title)}</text>',
        f'<text x="24" y="52" font-size="12" fill="#555">absolute Action error ({_svg_escape(unit)}), color clipped at p99={scale:.5g}</text>',
    ]
    for row in range(matrix.shape[0]):
        y = top + row * cell_h
        lines.append(f'<text x="{left - 8}" y="{y + cell_h * .7:.2f}" text-anchor="end" font-size="9">{_svg_escape(y_labels[row])}</text>')
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ratio = min(max(float(value) / scale if np.isfinite(value) else 0.0, 0.0), 1.0)
            red = int(255 * ratio)
            blue = int(255 * (1.0 - ratio))
            color = f'rgb({red},80,{blue})'
            x = left + col * cell_w
            lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w + .5:.2f}" height="{cell_h + .5:.2f}" fill="{color}"/>')
    for col, label in enumerate(x_labels):
        x = left + (col + .5) * cell_w
        lines.append(f'<text x="{x:.2f}" y="{top + matrix.shape[0] * cell_h + 18}" text-anchor="middle" font-size="8" transform="rotate(45 {x:.2f} {top + matrix.shape[0] * cell_h + 18})">{_svg_escape(label)}</text>')
    lines.append('</svg>')
    atomic_write_text(path, "\n".join(lines) + "\n")


def analyze(evaluation_dir: Path, dataset_run: Path, output_dir: Path, *, near_zero_fraction: float = DEFAULT_NEAR_ZERO_FRACTION) -> dict[str, Any]:
    evaluation_dir = evaluation_dir.expanduser().resolve()
    dataset_run = dataset_run.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output directory: {output_dir}")
    trace_dir = evaluation_dir / "all_predictions"
    traces = sorted(trace_dir.glob("fixture_*.json"))
    if not traces:
        raise FileNotFoundError(f"no exported fixture traces under {trace_dir}; rerun evaluation with --export-all")
    normalization_path = evaluation_dir.parent.parent / "data" / "normalization.npz"
    if not normalization_path.is_file():
        normalization_path = dataset_run / "data" / "normalization.npz"
    with np.load(normalization_path, allow_pickle=False) as values:
        action_mean = _finite(values["action_mean"], "action mean")
        action_std = _finite(values["action_std"], "action std")
    if action_mean.shape != (ACTION_DIM,) or action_std.shape != (ACTION_DIM,) or (action_std <= 0).any():
        raise ValueError("Action normalization must have shape [29] and positive standard deviations")
    manifest = _json(dataset_run / "manifests" / "dataset_manifest.json")
    joint_names = [str(value) for value in manifest.get("joint_names", [])]
    if len(joint_names) != ACTION_DIM:
        joint_names = [f"action_{index}" for index in range(ACTION_DIM)]
    near_zero = np.maximum(np.abs(action_std) * float(near_zero_fraction), 1.0e-4)
    errors: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    for trace_path in traces:
        trace = _json(trace_path)
        action = trace.get("domains", {}).get("action")
        if not isinstance(action, dict):
            raise ValueError(f"missing Action domain in {trace_path}")
        prediction = _finite(np.asarray(action["physical_prediction"]), "physical prediction")
        target = _finite(np.asarray(action["physical_target"]), "physical target")
        mask = np.asarray(action["mask"], dtype=bool)
        valid = np.asarray(action["valid"], dtype=bool)
        if prediction.ndim != 2 or prediction.shape[1] != ACTION_DIM or target.shape != prediction.shape:
            raise ValueError(f"unexpected Action trace shape in {trace_path}: {prediction.shape}")
        if mask.shape != prediction.shape or valid.shape != (prediction.shape[0],):
            raise ValueError(f"unexpected Action mask/valid shape in {trace_path}")
        errors.append(prediction - target)
        predictions.append(prediction)
        targets.append(target)
        masks.append(mask)
        valids.append(valid)
    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    error = np.concatenate(errors, axis=0)
    mask = np.concatenate(masks, axis=0)
    valid = np.concatenate(valids, axis=0)
    valid_2d = np.broadcast_to(valid[:, None], error.shape)
    splits = {
        "full": valid_2d,
        "masked": valid_2d & mask,
        "visible": valid_2d & ~mask,
    }
    normalized_error = error / action_std[None, :]
    per_joint: list[dict[str, Any]] = []
    for joint in range(ACTION_DIM):
        row: dict[str, Any] = {"index": joint, "name": joint_names[joint], "unit": "rad", "normalization_std_rad": float(action_std[joint]), "normalization_mean_rad": float(action_mean[joint]), "splits": {}}
        for split, take in splits.items():
            row["splits"][split] = _stats(
                normalized_error[:, joint][take[:, joint]],
                error[:, joint][take[:, joint]],
                prediction[:, joint][take[:, joint]],
                target[:, joint][take[:, joint]],
                float(action_std[joint]),
                near_zero_threshold=float(near_zero[joint]),
            )
        per_joint.append(row)
    time_sum = np.zeros((64, ACTION_DIM), dtype=np.float64)
    time_counts = np.zeros((64, ACTION_DIM), dtype=np.int64)
    for trace_path in traces:
        action = _json(trace_path)["domains"]["action"]
        p = np.asarray(action["physical_prediction"], dtype=np.float64)
        t = np.asarray(action["physical_target"], dtype=np.float64)
        v = np.asarray(action["valid"], dtype=bool)
        n = min(p.shape[0], 64)
        time_sum[:n] += np.where(v[:n, None], np.abs(p[:n] - t[:n]), 0.0)
        time_counts[:n] += v[:n, None].astype(np.int64)
    time_abs = np.divide(time_sum, np.maximum(time_counts, 1), where=time_counts > 0, out=np.full_like(time_sum, np.nan))
    split_summary = {split: _stats(
        normalized_error[take], error[take], prediction[take], target[take], float(np.mean(action_std)), near_zero_threshold=float(np.mean(near_zero))
    ) for split, take in splits.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_for_csv = []
    for row in per_joint:
        full = row["splits"]["full"]
        masked = row["splits"]["masked"]
        rows_for_csv.append({
            "index": row["index"], "joint": row["name"], "unit": row["unit"],
            "full_count": full["count"], "full_mae_rad": full["mae_rad"], "full_rmse_rad": full["rmse_rad"],
            "full_p95_abs_rad": full["p95_abs_rad"], "full_p99_abs_rad": full["p99_abs_rad"],
            "full_bias_rad": full["signed_bias_rad"], "full_rmse_over_dataset_std_pct": full["rmse_over_dataset_std_pct"],
            "full_mae_over_mean_abs_target_pct": full["mae_over_mean_abs_target_pct"], "full_mape_pct": full["mape_pct"], "full_smape_pct": full["smape_pct"],
            "masked_count": masked["count"], "masked_mae_rad": masked["mae_rad"], "masked_rmse_rad": masked["rmse_rad"],
            "masked_p95_abs_rad": masked["p95_abs_rad"], "masked_rmse_over_dataset_std_pct": masked["rmse_over_dataset_std_pct"],
            "masked_mae_over_mean_abs_target_pct": masked["mae_over_mean_abs_target_pct"], "masked_mape_pct": masked["mape_pct"], "masked_smape_pct": masked["smape_pct"],
        })
    with (output_dir / "action_per_joint.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_for_csv[0]))
        writer.writeheader()
        writer.writerows(rows_for_csv)
    summary = {
        "format_version": JOINT_ERROR_FORMAT,
        "evaluation_dir": str(evaluation_dir),
        "dataset_run": str(dataset_run),
        "trace_count": len(traces),
        "action_dimension": ACTION_DIM,
        "joint_names": joint_names,
        "units": {"action_target": "rad", "physical_error": "rad", "normalized_error": "dimensionless", "percentage": "%"},
        "normalization": {"mean_rad": action_mean.tolist(), "std_rad": action_std.tolist()},
        "percentage_definitions": {
            "rmse_over_dataset_std_pct": "100 * normalized RMSE = 100 * physical RMSE / dataset action std; scale-relative, not target-relative",
            "mae_over_mean_abs_target_pct": "100 * mean(abs(pred-target)) / mean(abs(target)); stable aggregate target-relative error",
            "mape_pct": "mean(100*abs(error)/abs(target)) after excluding abs(target) <= 0.05*action_std; coverage is reported",
            "smape_pct": "mean(200*abs(error)/(abs(pred)+abs(target))); symmetric and defined near zero",
        },
        "near_zero_fraction_of_dataset_std": float(near_zero_fraction),
        "split_summary": split_summary,
        "per_joint": per_joint,
        "time_mean_abs_rad": time_abs.tolist(),
        "time_valid_counts": time_counts.tolist(),
        "source_hashes": {str(path.relative_to(evaluation_dir)): file_sha256(path) for path in traces},
    }
    atomic_write_json(output_dir / "action_error_summary.json", summary)
    labels = joint_names
    chart_partition = "masked" if any(row["splits"]["masked"]["count"] for row in per_joint) else "full"
    summary["chart_partition"] = chart_partition
    atomic_write_json(output_dir / "action_error_summary.json", summary)
    _bar_svg(output_dir / "action_joint_percentages.svg", f"C Action error by joint ({chart_partition})", "Percent; lower is better", labels, [
        ("RMSE / dataset std", [row["splits"][chart_partition]["rmse_over_dataset_std_pct"] or 0.0 for row in per_joint], "#3182bd"),
        ("MAE / mean target", [row["splits"][chart_partition]["mae_over_mean_abs_target_pct"] or 0.0 for row in per_joint], "#e6550d"),
        ("sMAPE", [row["splits"][chart_partition]["smape_pct"] or 0.0 for row in per_joint], "#31a354"),
    ], suffix="; canonical joint target offset and error are in rad")
    _bar_svg(output_dir / "action_joint_physical.svg", f"C Action physical error by joint ({chart_partition})", "rad; lower is better", labels, [
        ("MAE (rad)", [row["splits"][chart_partition]["mae_rad"] or 0.0 for row in per_joint], "#756bb1"),
        ("p95 abs (rad)", [row["splits"][chart_partition]["p95_abs_rad"] or 0.0 for row in per_joint], "#de2d26"),
    ], suffix="; signed bias is available in action_per_joint.csv")
    _heatmap_svg(output_dir / "action_time_joint_abs_error.svg", "C Action absolute error over time and joint", time_abs, labels, [f"t={i}" for i in range(time_abs.shape[0])], unit="rad")
    atomic_write_text(output_dir / "README.txt", "\n".join([
        "Action error analysis for a 65-token CVAE evaluation trace.",
        "Action is the canonical processed joint target offset q_target - q_nominal, unit rad.",
        "normalized_error = physical_error / action_std, dimensionless.",
        "rmse_over_dataset_std_pct is scale-relative; it is not a physical percentage of the target.",
        "mae_over_mean_abs_target_pct is the recommended stable target-relative percentage.",
        "mape_pct excludes near-zero targets and reports mape_coverage; smape_pct remains defined near zero.",
        "full, masked and visible partitions are all retained in action_error_summary.json.",
    ]) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True, help="evaluation route directory containing all_predictions/")
    parser.add_argument("--dataset-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--near-zero-fraction", type=float, default=DEFAULT_NEAR_ZERO_FRACTION)
    args = parser.parse_args(argv)
    if not 0.0 <= args.near_zero_fraction <= 1.0:
        parser.error("--near-zero-fraction must be in [0, 1]")
    result = analyze(args.evaluation_dir, args.dataset_run, args.output_dir, near_zero_fraction=args.near_zero_fraction)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "trace_count": result["trace_count"], "action_dimension": result["action_dimension"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
