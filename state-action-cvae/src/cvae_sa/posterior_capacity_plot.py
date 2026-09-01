from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .util import atomic_write_text


LOG_FLOOR = 1e-12
RAW_POINT_LIMIT = 2000
EMA_ALPHA = 0.05
COLORS = (
    "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
    "#0891b2", "#db2777", "#4b5563",
)


def read_posterior_metrics(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise ValueError(f"invalid metrics JSON at line {index + 1}: {path}")
        if isinstance(value, dict):
            records.append(value)
    return records


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _points(records: Iterable[dict[str, Any]], *keys: str) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for record in records:
        current: Any = record
        for key in keys:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        step = _finite(record.get("optimizer_step"))
        value = _finite(current)
        if step is not None and value is not None:
            result.append((step, value))
    return result


def _ema(points: list[tuple[float, float]], alpha: float = EMA_ALPHA) -> list[tuple[float, float]]:
    smoothed: list[tuple[float, float]] = []
    value: float | None = None
    for step, sample in points:
        value = sample if value is None else alpha * sample + (1.0 - alpha) * value
        smoothed.append((step, value))
    return smoothed


def _downsample(points: list[tuple[float, float]], limit: int = RAW_POINT_LIMIT) -> list[tuple[float, float]]:
    if len(points) <= limit:
        return points
    selected = {
        min(len(points) - 1, round(index * (len(points) - 1) / (limit - 1)))
        for index in range(limit)
    }
    return [points[index] for index in sorted(selected)]


def _series(
    label: str,
    points: list[tuple[float, float]],
    color: str,
    *,
    width: float = 2.2,
    opacity: float = 1.0,
    dash: str | None = None,
    markers: bool = False,
) -> dict[str, Any]:
    return {
        "label": label,
        "points": points,
        "color": color,
        "width": width,
        "opacity": opacity,
        "dash": dash,
        "markers": markers,
    }


def _text(x: float, y: float, value: str, **attributes: Any) -> str:
    attrs = " ".join(f'{key.replace("_", "-")}="{html.escape(str(item))}"' for key, item in attributes.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {attrs}>{html.escape(value)}</text>'


def _log_ticks(low: float, high: float) -> list[tuple[float, str]]:
    low_exp = math.floor(low)
    high_exp = math.ceil(high)
    step = max(1, math.ceil((high_exp - low_exp) / 8))
    return [(float(exp), f"10^{exp}") for exp in range(low_exp, high_exp + 1, step)]


def _linear_ticks(low: float, high: float, count: int = 5) -> list[tuple[float, str]]:
    if math.isclose(low, high):
        high = low + 1.0
    return [
        (low + (high - low) * index / count, f"{low + (high - low) * index / count:.3g}")
        for index in range(count + 1)
    ]


def _line_panel(
    parts: list[str],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    series: list[dict[str, Any]],
    y_label: str,
    log_y: bool,
    references: list[dict[str, Any]] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> None:
    references = references or []
    all_points = [point for item in series for point in item["points"]]
    x_values = [point[0] for point in all_points]
    if references and not x_values:
        x_values = [0.0, 1.0]
    x_low, x_high = (min(x_values), max(x_values)) if x_values else (0.0, 1.0)
    if math.isclose(x_low, x_high):
        x_low, x_high = 0.0, max(1.0, x_high)
    raw_y = [point[1] for point in all_points] + [float(item["value"]) for item in references]
    if y_limits is not None:
        y_low, y_high = y_limits
    elif raw_y:
        if log_y:
            transformed = [math.log10(max(value, LOG_FLOOR)) for value in raw_y]
            y_low, y_high = min(transformed), max(transformed)
            padding = max(0.25, (y_high - y_low) * 0.08)
            y_low, y_high = y_low - padding, y_high + padding
        else:
            y_low, y_high = min(raw_y), max(raw_y)
            padding = max(1e-6, (y_high - y_low) * 0.08)
            y_low, y_high = y_low - padding, y_high + padding
    else:
        y_low, y_high = (-12.0, 0.0) if log_y else (0.0, 1.0)
    if math.isclose(y_low, y_high):
        y_low, y_high = y_low - 0.5, y_high + 0.5

    left, right, top, bottom = x + 88, x + width - 24, y + 64, y + height - 58
    chart_width, chart_height = right - left, bottom - top
    map_x = lambda value: left + (value - x_low) / (x_high - x_low) * chart_width
    transform_y = lambda value: math.log10(max(value, LOG_FLOOR)) if log_y else value
    map_y = lambda value: bottom - (transform_y(value) - y_low) / (y_high - y_low) * chart_height

    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" fill="#ffffff" stroke="#d1d5db"/>')
    parts.append(_text(x + 18, y + 26, title, fill="#111827", font_size="16", font_weight="600"))
    y_ticks = _log_ticks(y_low, y_high) if log_y else _linear_ticks(y_low, y_high)
    for value, label in y_ticks:
        if not y_low - 1e-9 <= value <= y_high + 1e-9:
            continue
        py = bottom - (value - y_low) / (y_high - y_low) * chart_height
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{right}" y2="{py:.1f}" stroke="#e5e7eb"/>')
        parts.append(_text(left - 10, py + 4, label, fill="#4b5563", font_size="11", text_anchor="end"))
    for index in range(6):
        value = x_low + (x_high - x_low) * index / 5
        px = map_x(value)
        parts.append(f'<line x1="{px:.1f}" y1="{bottom}" x2="{px:.1f}" y2="{bottom + 5}" stroke="#374151"/>')
        parts.append(_text(px, bottom + 20, f"{value:.0f}", fill="#4b5563", font_size="11", text_anchor="middle"))
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#374151"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#374151"/>')
    parts.append(_text((left + right) / 2, y + height - 12, "Optimizer step", fill="#374151", font_size="12", text_anchor="middle"))
    parts.append(
        f'<text x="{x + 18:.1f}" y="{(top + bottom) / 2:.1f}" fill="#374151" font-size="12" text-anchor="middle" '
        f'transform="rotate(-90 {x + 18:.1f} {(top + bottom) / 2:.1f})">{html.escape(y_label)}</text>'
    )
    for reference in references:
        py = map_y(float(reference["value"]))
        parts.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{right}" y2="{py:.1f}" '
            f'stroke="{reference.get("color", "#6b7280")}" stroke-width="1.5" stroke-dasharray="6 5"/>'
        )
        parts.append(_text(right - 4, py - 5, str(reference["label"]), fill=reference.get("color", "#6b7280"), font_size="10", text_anchor="end"))
    for item in series:
        points = item["points"]
        if not points:
            continue
        coordinates = " ".join(f"{map_x(px):.1f},{map_y(py):.1f}" for px, py in points)
        dash = f' stroke-dasharray="{item["dash"]}"' if item["dash"] else ""
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{item["color"]}" '
            f'stroke-width="{item["width"]}" opacity="{item["opacity"]}"{dash}/>'
        )
        if item["markers"]:
            for px, py in points:
                parts.append(f'<circle cx="{map_x(px):.1f}" cy="{map_y(py):.1f}" r="3" fill="{item["color"]}"/>')
    legend_x, legend_y = left, y + 46
    for index, item in enumerate([item for item in series if item["points"]]):
        lx = legend_x + (index % 4) * max(145, chart_width / 4)
        ly = legend_y + (index // 4) * 15
        parts.append(f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{lx + 18:.1f}" y2="{ly:.1f}" stroke="{item["color"]}" stroke-width="3"/>')
        parts.append(_text(lx + 23, ly + 4, item["label"], fill="#374151", font_size="10"))


def _document(width: int, height: int, title: str, subtitle: str) -> tuple[list[str], int]:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f3f4f6"/>',
        _text(28, 34, title, fill="#111827", font_size="22", font_weight="700"),
        _text(28, 57, subtitle, fill="#4b5563", font_size="12"),
    ]
    return parts, 74


def _write_training_curves(run_dir: Path, train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> None:
    scope = validation[-1].get("evaluation_scope", "no full evaluation recorded") if validation else "no full evaluation recorded"
    evaluation_label = (
        "Held-out-mask evaluation"
        if "held-out-mask" in str(scope)
        else "Full fixed-fixture evaluation"
    )
    parts, top = _document(1440, 1040, "Posterior capacity training curves", str(scope))
    total = _points(train, "total")
    total_series = [
        _series("Train batch raw", _downsample(total), COLORS[0], width=1.0, opacity=0.22),
        _series(f"Train EMA alpha={EMA_ALPHA}", _ema(total), COLORS[0], width=2.5),
        _series(evaluation_label, _points(validation, "reconstruction_loss", "total"), COLORS[1], markers=True),
    ]
    _line_panel(parts, x=20, y=top, width=690, height=440, title="Total reconstruction objective", series=total_series, y_label="Value (log10 scale)", log_y=True)
    component_series: list[dict[str, Any]] = []
    component_evaluation_label = (
        "Held-out eval" if "held-out-mask" in str(scope) else "Fixed-fixture eval"
    )
    for index, key in enumerate(("state", "action", "contact")):
        component_series.append(_series(f"Train {key} EMA", _ema(_points(train, key)), COLORS[index], width=2.2))
        component_series.append(_series(f"{component_evaluation_label} {key}", _points(validation, "reconstruction_loss", key), COLORS[index], dash="5 4", markers=True))
    _line_panel(parts, x=730, y=top, width=690, height=440, title="Reconstruction components", series=component_series, y_label="Value (log10 scale)", log_y=True)
    _line_panel(parts, x=20, y=top + 460, width=690, height=440, title="Learning rate", series=[_series("Learning rate", _points(train, "learning_rate"), COLORS[4])], y_label="Learning rate (log10 scale)", log_y=True)
    _line_panel(parts, x=730, y=top + 460, width=690, height=440, title="Gradient norm before clipping", series=[_series("Gradient norm", _points(train, "gradient_norm"), COLORS[5])], y_label="Gradient norm (log10 scale)", log_y=True)
    parts.append(_text(28, 1018, "Log panels clip non-positive values to 1e-12 for display only. Raw metrics.jsonl values are unchanged.", fill="#4b5563", font_size="11"))
    parts.append("</svg>")
    atomic_write_text(run_dir / "plots/training_curves.svg", "\n".join(parts) + "\n")


def _write_gate_curves(run_dir: Path, validation: list[dict[str, Any]]) -> None:
    scope = validation[-1].get("evaluation_scope", "no full evaluation recorded") if validation else "no full evaluation recorded"
    parts, top = _document(1440, 1040, "Posterior capacity gate curves", str(scope))
    latest = validation[-1] if validation else {}
    exact = latest.get("exact_gate", {}).get("thresholds", {})
    progression = latest.get("progression_gate", {}).get("thresholds", {})
    error_series = [
        _series("State RMSE", _points(validation, "worst_state_rmse"), COLORS[0], markers=True),
        _series("Action RMSE", _points(validation, "worst_action_rmse"), COLORS[1], markers=True),
        _series("Max abs", _points(validation, "continuous_max_abs"), COLORS[2], markers=True),
    ]
    references = []
    for label, values, color in (("exact State/Action", exact, "#7c3aed"), ("progression continuous", progression, "#d97706")):
        value = values.get("state_rmse")
        if value is not None:
            references.append({"label": f"{label} {float(value):.0e}", "value": float(value), "color": color})
    if exact.get("continuous_max_abs") is not None:
        references.append({"label": f"exact max {float(exact['continuous_max_abs']):.0e}", "value": float(exact["continuous_max_abs"]), "color": "#9333ea"})
    _line_panel(parts, x=20, y=top, width=690, height=440, title="Worst continuous errors", series=error_series, y_label="Value (log10 scale)", log_y=True, references=references)
    _line_panel(parts, x=730, y=top, width=690, height=440, title="Contact classification accuracy", series=[_series("Contact accuracy", _points(validation, "contact_accuracy"), COLORS[2], markers=True)], y_label="Accuracy", log_y=False, references=[{"label": "required 1.0", "value": 1.0, "color": "#dc2626"}], y_limits=(0.0, 1.02))
    latent_threshold = float(progression.get("latent_ratio", exact.get("latent_ratio", 10.0)))
    _line_panel(parts, x=20, y=top + 460, width=690, height=440, title="Zero-latent dependence", series=[_series("Zero/correct RMSE ratio", _points(validation, "latent_dependence", "zero_ratio"), COLORS[4], markers=True)], y_label="RMSE ratio (log10 scale)", log_y=True, references=[{"label": f"minimum {latent_threshold:g}", "value": latent_threshold, "color": "#dc2626"}])
    _line_panel(parts, x=730, y=top + 460, width=690, height=440, title="Active gate score", series=[_series("Gate score", _points(validation, "score"), COLORS[5], markers=True)], y_label="Score (log10 scale)", log_y=True, references=[{"label": "PASS <= 1", "value": 1.0, "color": "#dc2626"}])
    parts.append(_text(28, 1018, "Exact and progression are recorded together; only the configured acceptance gate controls stopping and markers.", fill="#4b5563", font_size="11"))
    parts.append("</svg>")
    atomic_write_text(run_dir / "plots/gate_curves.svg", "\n".join(parts) + "\n")


def _write_mask_breakdown(run_dir: Path, validation: list[dict[str, Any]]) -> None:
    best = min(validation, key=lambda item: float(item.get("score", math.inf))) if validation else {}
    scope = best.get("evaluation_scope", "no full evaluation recorded")
    step = best.get("optimizer_step", "N/A")
    gate_name = str(best.get("acceptance_gate", "exact"))
    thresholds = best.get(f"{gate_name}_gate", {}).get("thresholds", {})
    cases = best.get("cases", {})
    labels = list(cases)
    metrics = (
        ("State RMSE / threshold", "worst_state_rmse", "state_rmse", COLORS[0]),
        ("Action RMSE / threshold", "worst_action_rmse", "action_rmse", COLORS[1]),
        ("Max abs / threshold", "continuous_max_abs", "continuous_max_abs", COLORS[2]),
        ("Contact gate ratio", "contact_accuracy", None, COLORS[3]),
    )
    all_ratios = []
    for item in cases.values():
        for _, value_key, threshold_key, _ in metrics:
            raw = float(item.get(value_key, 0.0))
            all_ratios.append(
                2.0 - raw
                if threshold_key is None
                else raw / max(float(thresholds.get(threshold_key, 1.0)), LOG_FLOOR)
            )
    parts, top = _document(1440, 820, "Best checkpoint mask breakdown", f"best active-gate step {step}; {scope}")
    left, right, chart_top, bottom = 100.0, 1410.0, top + 56.0, 650.0
    low = -4.0
    largest_ratio = max(max(all_ratios or [1.0]), LOG_FLOOR)
    high = float(max(2, math.ceil(math.log10(largest_ratio)) + 1))
    map_y = lambda value: bottom - (math.log10(max(value, LOG_FLOOR)) - low) / (high - low) * (bottom - chart_top)
    for exponent in range(int(low), int(high) + 1):
        py = bottom - (exponent - low) / (high - low) * (bottom - chart_top)
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{right}" y2="{py:.1f}" stroke="#e5e7eb"/>')
        parts.append(_text(left - 10, py + 4, f"10^{exponent}", fill="#4b5563", font_size="11", text_anchor="end"))
    pass_y = map_y(1.0)
    parts.append(f'<line x1="{left}" y1="{pass_y:.1f}" x2="{right}" y2="{pass_y:.1f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="7 5"/>')
    parts.append(_text(right - 4, pass_y - 6, "PASS <= 1", fill="#dc2626", font_size="11", text_anchor="end"))
    group_width = (right - left) / max(len(labels), 1)
    bar_width = group_width * 0.17
    for group, label in enumerate(labels):
        center = left + (group + 0.5) * group_width
        item = cases[label]
        for offset, (_, value_key, threshold_key, color) in enumerate(metrics):
            raw = float(item.get(value_key, 0.0))
            ratio = 2.0 - raw if threshold_key is None else raw / max(float(thresholds.get(threshold_key, 1.0)), LOG_FLOOR)
            px = center + (offset - 1.5) * bar_width
            py = max(chart_top, min(bottom, map_y(ratio)))
            parts.append(f'<rect x="{px - bar_width * 0.42:.1f}" y="{py:.1f}" width="{bar_width * 0.84:.1f}" height="{bottom - py:.1f}" fill="{color}" opacity="0.86"/>')
        parts.append(
            f'<text x="{center:.1f}" y="{bottom + 18:.1f}" fill="#374151" font-size="10" text-anchor="end" '
            f'transform="rotate(-35 {center:.1f} {bottom + 18:.1f})">{html.escape(label)}</text>'
        )
    parts.append(f'<line x1="{left}" y1="{chart_top}" x2="{left}" y2="{bottom}" stroke="#374151"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#374151"/>')
    parts.append(f'<text x="30" y="{(chart_top + bottom) / 2:.1f}" fill="#374151" font-size="12" text-anchor="middle" transform="rotate(-90 30 {(chart_top + bottom) / 2:.1f})">Gate ratio (log10 scale)</text>')
    for index, (label, _, _, color) in enumerate(metrics):
        lx = 105 + index * 290
        parts.append(f'<rect x="{lx}" y="{top + 20}" width="14" height="10" fill="{color}"/>')
        parts.append(_text(lx + 20, top + 30, label, fill="#374151", font_size="11"))
    latent = best.get("latent_dependence", {})
    parts.append(_text(100, 754, f"Global latent dependence: zero ratio={float(latent.get('zero_ratio', 0.0)):.6g}, swapped ratio={float(latent.get('swapped_ratio', 0.0)):.6g}", fill="#374151", font_size="12"))
    parts.append(_text(100, 778, "Ratios <= 1 pass. Log values are explicitly labelled; display floor is 1e-12 and does not alter metrics.jsonl.", fill="#4b5563", font_size="11"))
    parts.append("</svg>")
    atomic_write_text(run_dir / "plots/mask_breakdown.svg", "\n".join(parts) + "\n")


def render_posterior_capacity_plots(run_dir: Path) -> tuple[Path, Path, Path]:
    run_dir = run_dir.expanduser().resolve()
    metrics_path = run_dir / "logs/metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"posterior metrics are missing: {metrics_path}")
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    records = read_posterior_metrics(metrics_path)
    train = [record for record in records if record.get("phase") == "train"]
    validation = [record for record in records if record.get("phase") == "validation"]
    summary_path = run_dir / "manifests/posterior_capacity_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        default_scope = (
            "held-out-mask evaluation on seen training windows"
            if summary.get("mask_phase") == "generalization"
            else "full fixed-fixture evaluation on the same training windows and masks"
        )
        validation = [
            record
            if record.get("evaluation_scope")
            else {**record, "evaluation_scope": default_scope}
            for record in validation
        ]
    _write_training_curves(run_dir, train, validation)
    _write_gate_curves(run_dir, validation)
    _write_mask_breakdown(run_dir, validation)
    return (
        run_dir / "plots/training_curves.svg",
        run_dir / "plots/gate_curves.svg",
        run_dir / "plots/mask_breakdown.svg",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate posterior-capacity SVG plots")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    for path in render_posterior_capacity_plots(parse_args().run_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
