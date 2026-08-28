from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Any, Iterable

from .util import atomic_write_text


COLORS = ("#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2")


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _ema(values: list[float], alpha: float = 0.08) -> list[float]:
    output: list[float] = []
    current: float | None = None
    for value in values:
        current = value if current is None else alpha * value + (1.0 - alpha) * current
        output.append(current)
    return output


def _finite_series(records: Iterable[dict[str, Any]], key: str) -> list[tuple[float, float]]:
    values = []
    for record in records:
        try:
            x = float(record["optimizer_step"])
            y = float(record[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            values.append((x, y))
    return values


def _panel(
    parts: list[str], x: float, y: float, width: float, height: float,
    title: str, series: list[tuple[str, list[tuple[float, float]], str]],
    log_y: bool = False, horizontal: list[tuple[float, str]] | None = None,
) -> None:
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#ffffff" stroke="#cbd5e1"/>')
    parts.append(f'<text x="{x + 12}" y="{y + 22}" font-size="15" font-weight="600">{escape(title)}</text>')
    all_points = [point for _, values, _ in series for point in values]
    if not all_points:
        parts.append(f'<text x="{x + 12}" y="{y + 48}" fill="#64748b">no data</text>')
        return
    x_min, x_max = min(point[0] for point in all_points), max(point[0] for point in all_points)
    transformed = []
    for _, values, _ in series:
        transformed.extend(math.log10(max(point[1], 1e-12)) if log_y else point[1] for point in values)
    if horizontal:
        transformed.extend(math.log10(max(value, 1e-12)) if log_y else value for value, _ in horizontal)
    y_min, y_max = min(transformed), max(transformed)
    if y_max <= y_min:
        y_max = y_min + 1.0
    left, top, plot_w, plot_h = x + 48, y + 35, width - 64, height - 68
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#94a3b8"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#94a3b8"/>')

    def sx(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1.0) * plot_w

    def sy(value: float) -> float:
        adjusted = math.log10(max(value, 1e-12)) if log_y else value
        return top + (y_max - adjusted) / (y_max - y_min) * plot_h

    if horizontal:
        for value, label in horizontal:
            yy = sy(value)
            parts.append(f'<line x1="{left}" y1="{yy}" x2="{left + plot_w}" y2="{yy}" stroke="#94a3b8" stroke-dasharray="5 4"/>')
            parts.append(f'<text x="{left + plot_w - 4}" y="{yy - 4}" text-anchor="end" font-size="10" fill="#64748b">{escape(label)}</text>')
    legend_x = left
    for label, values, color in series:
        if not values:
            continue
        points = " ".join(f"{sx(px):.2f},{sy(py):.2f}" for px, py in values)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.7"/>')
        parts.append(f'<rect x="{legend_x}" y="{y + height - 23}" width="10" height="3" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 14}" y="{y + height - 18}" font-size="10">{escape(label)}</text>')
        legend_x += min(130, 24 + len(label) * 6)
    parts.append(f'<text x="{left}" y="{y + height - 5}" font-size="10" fill="#64748b">step {int(x_min)}–{int(x_max)}</text>')


def write_training_svg(
    metrics_path: Path, output_path: Path, thresholds: dict[str, float] | None = None
) -> None:
    records = read_metrics(metrics_path)
    train = [record for record in records if record.get("phase") == "train"]
    validation = [record for record in records if record.get("phase") == "validation"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">',
        '<rect width="1400" height="900" fill="#f8fafc"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#0f172a}</style>',
        '<text x="32" y="34" font-size="22" font-weight="700">32-motion overfit training</text>',
    ]
    loss_series = []
    for index, key in enumerate(("total", "masked", "forward", "inverse", "history_action", "rollout")):
        raw = _finite_series(train, key)
        if raw:
            smoothed = _ema([value for _, value in raw])
            loss_series.append((f"{key} EMA", [(raw[i][0], smoothed[i]) for i in range(len(raw))], COLORS[index]))
    metric_keys = (
        "forward_one_normalized_rmse",
        "action_inverse_local_normalized_rmse",
        "arbitrary_state_normalized_rmse",
        "action_completion_macro_normalized_rmse",
        "history_action_normalized_rmse",
        "forward_rollout_8_normalized_rmse",
    )
    metric_series = [
        (key.replace("_normalized_rmse", ""), _finite_series(validation, key), COLORS[index])
        for index, key in enumerate(metric_keys)
    ]
    threshold_lines = []
    if thresholds:
        for key, value in thresholds.items():
            threshold_lines.append((float(value), key.replace("_normalized_rmse", "")))
    lr_series = [
        ("learning rate", _finite_series(train, "learning_rate"), COLORS[0]),
        ("gradient norm", _finite_series(train, "gradient_norm"), COLORS[1]),
    ]
    progress_series = [
        ("effective epoch", _finite_series(train, "effective_epoch"), COLORS[2]),
        ("overfit score", _finite_series(validation, "overfit_score"), COLORS[4]),
    ]
    _panel(parts, 30, 55, 660, 390, "Loss components (log scale)", loss_series, log_y=True)
    _panel(parts, 710, 55, 660, 390, "Fixed-suite normalized RMSE", metric_series, horizontal=threshold_lines)
    _panel(parts, 30, 470, 660, 390, "Learning rate and gradient norm", lr_series, log_y=True)
    _panel(parts, 710, 470, 660, 390, "Dataset repetition and gate", progress_series, horizontal=[(1.0, "pass boundary")])
    parts.append("</svg>")
    atomic_write_text(output_path, "".join(parts))


def write_sensitivity_svg(values: dict[str, float], output_path: Path) -> None:
    width = 1100
    row_height = 34
    height = 80 + row_height * max(len(values), 1)
    maximum = max([1.0, *[float(value) for value in values.values() if math.isfinite(float(value))]])
    baseline_x = 380 + 650 / maximum
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#0f172a}</style>',
        '<text x="24" y="32" font-size="20" font-weight="700">Input occlusion sensitivity (RMSE ratio)</text>',
        f'<line x1="{baseline_x:.1f}" y1="50" x2="{baseline_x:.1f}" y2="{height}" stroke="#dc2626" stroke-dasharray="5 4"/>',
    ]
    for index, (name, value) in enumerate(sorted(values.items())):
        y = 58 + index * row_height
        bar = max(0.0, min(float(value) / maximum, 1.0)) * 650
        color = "#dc2626" if value > 1.05 else "#2563eb"
        parts.append(f'<text x="24" y="{y + 18}" font-size="13">{escape(name)}</text>')
        parts.append(f'<rect x="380" y="{y + 4}" width="{bar:.1f}" height="20" fill="{color}" opacity="0.82"/>')
        parts.append(f'<text x="{390 + bar:.1f}" y="{y + 19}" font-size="12">{value:.3f}×</text>')
    parts.append("</svg>")
    atomic_write_text(output_path, "".join(parts))


def write_comparison_svg(records: list[dict[str, Any]], output_path: Path) -> None:
    width = 1200
    height = 100 + max(len(records), 1) * 42
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#0f172a}</style>',
        '<text x="24" y="32" font-size="20" font-weight="700">Overfit suite comparison</text>',
        '<text x="24" y="62" font-size="12" fill="#64748b">model / phase / seed · worst threshold ratio (≤1 passes)</text>',
        '<line x1="850" y1="70" x2="850" y2="100%" stroke="#dc2626" stroke-dasharray="5 4"/>',
    ]
    for index, record in enumerate(records):
        y = 76 + index * 42
        label = f"{record.get('model_profile')} / {record.get('overfit_phase')} / {record.get('seed')}"
        score = float(record.get("best_overfit_score", math.inf))
        finite_score = score if math.isfinite(score) else 2.0
        bar = min(finite_score, 2.0) / 2.0 * 760
        color = "#059669" if record.get("passed") else "#dc2626"
        parts.append(f'<text x="24" y="{y + 20}" font-size="13">{escape(label)}</text>')
        parts.append(f'<rect x="470" y="{y + 5}" width="{bar:.1f}" height="22" fill="{color}" opacity="0.82"/>')
        parts.append(f'<text x="{480 + bar:.1f}" y="{y + 21}" font-size="12">{score:.3f}</text>')
    parts.append("</svg>")
    atomic_write_text(output_path, "".join(parts))
