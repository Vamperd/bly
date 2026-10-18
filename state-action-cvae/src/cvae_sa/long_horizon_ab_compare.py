"""Compare A and recursive B on one sealed 192-step replay window.

The comparison intentionally keeps only compact JSON/SVG outputs.  It reports
physical State and canonical Action errors against both the recorded source and
the deterministic original-Action replay, including per-frame curves and the
worst joint/feature coordinates.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from .replay65 import _load_replay, file_sha256, verify_worker
from .util import atomic_write_json, atomic_write_text, load_json


VERSION = "sonic_long_horizon_ab_compare_v1"
HORIZON = 192
STATE_CONTINUOUS = 68
ACTION_DIM = 29
STATE_NAMES = (
    [f"joint_pos_{i}" for i in range(29)]
    + [f"joint_vel_{i}" for i in range(29)]
    + ["base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z"]
    + ["base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"]
    + ["gravity_x", "gravity_y", "gravity_z", "base_height"]
)
STATE_ALL_NAMES = STATE_NAMES + ["foot_contact_left", "foot_contact_right"]
ACTION_NAMES = [f"action_{i}" for i in range(ACTION_DIM)]
COLORS = {"A": "#4285f4", "recursive_B": "#f2992e"}


def _finite(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} has shape={value.shape}, expected={shape}, finite required")
    return value


def _window(run: Path) -> tuple[Path, dict[str, Any]]:
    root = load_json(run / "manifests" / "replay65.json")
    windows = root.get("windows", [])
    if len(windows) != 1:
        raise ValueError(f"comparison requires one replay window: {run}")
    window = run / windows[0]
    return window, load_json(window / "manifests" / "replay65.json")


def _entry(meta: dict[str, Any], route: str) -> dict[str, Any]:
    if route == "A":
        entries = [item for item in meta.get("entries", []) if item.get("action_model_replay")]
        if len(entries) != 1:
            raise ValueError("A run must contain exactly one model Action replay")
        return entries[0]
    for item in meta.get("entries", []):
        if item.get("mask") == "recursive_state_feedback_b" and item.get("action_model_replay"):
            return item
    raise ValueError("recursive B run has no recursive_state_feedback_b Action replay")


def _processed_to_canonical(path: Path, nominal: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as values:
        if "processed_action" not in values:
            raise ValueError(f"missing processed_action: {path}")
        processed = np.asarray(values["processed_action"], dtype=np.float32)
    return _finite("canonical Action", processed - nominal[None], (HORIZON, ACTION_DIM))


def _source_action(path: Path, nominal: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as values:
        if "action_target_canonical" in values:
            action = np.asarray(values["action_target_canonical"], dtype=np.float32)
        elif "processed_action" in values:
            action = np.asarray(values["processed_action"], dtype=np.float32) - nominal[None]
        else:
            raise ValueError(f"source replay has no canonical or processed Action: {path}")
    return _finite("source canonical Action", action, (HORIZON, ACTION_DIM))


def _joint_names(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as values:
        if "joint_names" in values:
            names = [str(value) for value in values["joint_names"].tolist()]
            if len(names) == ACTION_DIM:
                return names
    return [f"joint_{index}" for index in range(ACTION_DIM)]


def _load_route(run_path: Path, route: str) -> dict[str, Any]:
    run = run_path.expanduser().resolve()
    window, meta = _window(run)
    entry = _entry(meta, route)
    source_path = window / "data" / "source.replay.npz"
    baseline_child = window / "simulations" / "baseline"
    model_child = window / "simulations" / str(entry["mask"])
    verify_worker(baseline_child)
    verify_worker(model_child)
    source = _load_replay(source_path)
    with np.load(window / "data" / "exact_initialization.npz", allow_pickle=False) as values:
        nominal = np.asarray(values["nominal_default_joint_pos"], dtype=np.float32)
    original_path = baseline_child / "data" / "replay" / "000000.replay.npz"
    model_path = model_child / "data" / "replay" / "000000.replay.npz"
    original = _load_replay(original_path)
    model = _load_replay(model_path)
    for name, replay in (("source", source), ("original", original), (route, model)):
        _finite(f"{name} State", replay["physical_state"], (HORIZON + 1, 70))
        _finite(f"{name} joint position", replay["joint_pos"], (HORIZON + 1, 29))
    return {
        "run": str(run),
        "window": str(window),
        "meta": meta,
        "entry": entry,
        "source_hash": file_sha256(source_path),
        "source": source,
        "original": original,
        "model": model,
        "source_action": _source_action(source_path, nominal),
        "original_action": _processed_to_canonical(original_path, nominal),
        "model_action": _processed_to_canonical(model_path, nominal),
        "joint_names": _joint_names(source_path),
    }


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    absolute = np.abs(values)
    return {
        "rmse": float(np.sqrt(np.mean(values * values))),
        "mae": float(np.mean(absolute)),
        "p95": float(np.quantile(absolute, 0.95)),
        "max_abs": float(np.max(absolute)),
    }


def _domain_metrics(diff: np.ndarray, names: list[str]) -> dict[str, Any]:
    diff = np.asarray(diff, dtype=np.float32)
    per_frame = np.sqrt(np.mean(np.square(diff), axis=1))
    flattened = np.abs(diff).reshape(-1)
    worst = int(np.argmax(flattened))
    frame, feature = np.unravel_index(worst, diff.shape)
    return {
        "overall": _stats(diff),
        "per_frame_rmse": [float(value) for value in per_frame],
        "worst": {
            "frame": int(frame),
            "feature_index": int(feature),
            "feature_name": names[feature],
            "signed_error": float(diff[frame, feature]),
            "absolute_error": float(abs(diff[frame, feature])),
        },
    }


def _state_metrics(candidate: np.ndarray, reference: np.ndarray, joint_names: list[str]) -> dict[str, Any]:
    all_diff = candidate - reference
    continuous = candidate[:, :STATE_CONTINUOUS] - reference[:, :STATE_CONTINUOUS]
    result = _domain_metrics(all_diff, STATE_ALL_NAMES)
    result["continuous"] = _domain_metrics(continuous, STATE_NAMES)
    contact = candidate[:, 68:70] >= 0.5
    truth_contact = reference[:, 68:70] >= 0.5
    result["contact_error_per_frame"] = [float(value) for value in np.mean(contact != truth_contact, axis=1)]
    result["contact_disagreement_rate"] = float(np.mean(contact != truth_contact))
    feature = int(result["continuous"]["worst"]["feature_index"])
    if feature < 29:
        result["continuous"]["worst"]["joint_name"] = joint_names[feature]
    elif feature < 58:
        result["continuous"]["worst"]["joint_name"] = joint_names[feature - 29]
    else:
        result["continuous"]["worst"]["joint_name"] = "non_joint_state"
    result["worst_joint"] = result["continuous"]["worst"]
    return result


def _action_metrics(candidate: np.ndarray, reference: np.ndarray, joint_names: list[str]) -> dict[str, Any]:
    return _domain_metrics(candidate - reference, joint_names)


def _svg_curve(path: Path, series: dict[str, dict[str, list[float]]], title: str,
               annotations: dict[str, list[str]]) -> None:
    width, height = 1600, 1050
    left, right, top, bottom = 230, 430, 70, 70
    panel_height = (height - top - bottom) / 4
    plot_width = width - left - right
    panels = (
        ("state_source", "State RMSE vs source"),
        ("action_source", "Action RMSE vs source"),
        ("state_original", "State RMSE vs original replay"),
        ("action_original", "Action RMSE vs original replay"),
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width // 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
    ]
    for index, (key, label) in enumerate(panels):
        y0 = top + index * panel_height + 8
        y1 = top + (index + 1) * panel_height - 30
        peak = max(1.0e-12, max(max(data[key]) for data in series.values()) * 1.1)
        panel_frames = max(len(data[key]) for data in series.values())
        parts.append(f'<rect x="{left}" y="{y0:.1f}" width="{plot_width}" height="{y1-y0:.1f}" fill="none" stroke="#999"/>')
        parts.append(f'<text x="{left+12}" y="{y0+21:.1f}" font-family="sans-serif" font-size="15" font-weight="bold">{html.escape(label)}</text>')
        parts.append(f'<text x="{left-14}" y="{y0+13:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{peak:.3g}</text>')
        parts.append(f'<text x="{left-14}" y="{y1+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">0</text>')
        for boundary in (32, 64, 96, 128, 160):
            if boundary < panel_frames:
                x = left + boundary * plot_width / max(panel_frames - 1, 1)
                parts.append(f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}" stroke="#b8b8b8" stroke-dasharray="5 5"/>')
                parts.append(f'<text x="{x+3:.1f}" y="{y0+36:.1f}" font-family="sans-serif" font-size="10" fill="#666">{boundary}</text>')
        for name, data in series.items():
            color = COLORS[name]
            points = " ".join(
                f"{left + frame*plot_width/max(panel_frames-1,1):.1f},{y1-value/peak*(y1-y0):.1f}"
                for frame, value in enumerate(data[key])
            )
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        parts.append(f'<text x="{left + plot_width/2:.1f}" y="{y1+20:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">transition (50 Hz)</text>')
    info_x = width - right + 20
    parts.append(f'<text x="{info_x}" y="{top+5}" font-family="sans-serif" font-size="15" font-weight="bold">Legend / largest errors</text>')
    for index, (name, color) in enumerate(COLORS.items()):
        y = top + 35 + index * 125
        parts.append(f'<rect x="{info_x}" y="{y}" width="16" height="12" fill="{color}"/>')
        parts.append(f'<text x="{info_x+25}" y="{y+11}" font-family="sans-serif" font-size="14" font-weight="bold">{html.escape(name)}</text>')
        for line_index, line in enumerate(annotations.get(name, [])):
            parts.append(
                f'<text x="{info_x}" y="{y+30+line_index*17}" '
                f'font-family="sans-serif" font-size="11">{html.escape(line)}</text>'
            )
    parts.append("</svg>")
    atomic_write_text(path, "\n".join(parts) + "\n")


def compare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_run.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output run must be new and empty: {output}")
    for folder in ("manifests", "plots", "logs"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    route_a = _load_route(args.a_run, "A")
    route_b = _load_route(args.b_run, "recursive_B")
    for key in ("source_hash",):
        if route_a[key] != route_b[key]:
            raise ValueError(f"A/B source mismatch for {key}")
    a_meta, b_meta = route_a["meta"], route_b["meta"]
    if a_meta["window"].get("stable_window_id") != b_meta["window"].get("stable_window_id"):
        raise ValueError("A/B stable window identity mismatch")
    source_state = route_a["source"]["physical_state"]
    original_state = route_a["original"]["physical_state"]
    joint_names = route_a["joint_names"]
    if joint_names != route_b["joint_names"]:
        raise ValueError("A/B joint-name order mismatch")
    results: dict[str, Any] = {}
    curves: dict[str, dict[str, list[float]]] = {}
    annotations: dict[str, list[str]] = {}
    for name, route in (("A", route_a), ("recursive_B", route_b)):
        candidate_state = route["model"]["physical_state"]
        candidate_action = route["model_action"]
        state_source = _state_metrics(candidate_state, source_state, joint_names)
        action_source = _action_metrics(candidate_action, route["source_action"], joint_names)
        state_original = _state_metrics(candidate_state, original_state, joint_names)
        action_original = _action_metrics(candidate_action, route["original_action"], joint_names)
        results[name] = {
            "run": route["run"],
            "state_vs_source": state_source,
            "action_vs_source": action_source,
            "state_vs_original": state_original,
            "action_vs_original": action_original,
        }
        curves[name] = {
            "state_source": state_source["per_frame_rmse"],
            "action_source": action_source["per_frame_rmse"],
            "state_original": state_original["per_frame_rmse"],
            "action_original": action_original["per_frame_rmse"],
        }
        annotations[name] = [
            "source max State: " + state_source["worst_joint"]["joint_name"],
            "source max Action: " + action_source["worst"]["feature_name"],
            "Origin max State: " + state_original["worst_joint"]["joint_name"],
            "Origin max Action: " + action_original["worst"]["feature_name"],
        ]
    meta = {
        "version": VERSION,
        "horizon_steps": HORIZON,
        "fps": 50,
        "source_hash": route_a["source_hash"],
        "window": a_meta["window"],
        "joint_names": joint_names,
        "a_run": route_a["run"],
        "b_run": route_b["run"],
        "routes": results,
        "curves": curves,
        "interpretation": {
            "state": "physical State RMSE over all 70 dimensions; continuous 68-dim State and contact disagreement are also reported",
            "action": "canonical Action RMSE over 29 joints",
            "references": "source is recorded State/Action; original is deterministic Origin Action simulator replay",
        },
    }
    atomic_write_json(output / "manifests" / "ab_error_metrics.json", meta)
    _svg_curve(output / "plots" / "ab_error_curves.svg", curves,
               f"A vs recursive B | {a_meta['window'].get('motion_key', 'motion')} | 192 transitions",
               annotations)
    for name, result in results.items():
        print(
            f"{name}: state_rmse_source={result['state_vs_source']['overall']['rmse']:.6g} "
            f"action_rmse_source={result['action_vs_source']['overall']['rmse']:.6g} "
            f"state_worst_joint={result['state_vs_source']['worst_joint']['joint_name']} "
            f"action_worst_joint={result['action_vs_source']['worst']['feature_name']} "
            f"state_rmse_original={result['state_vs_original']['overall']['rmse']:.6g} "
            f"action_rmse_original={result['action_vs_original']['overall']['rmse']:.6g} "
            f"state_worst_joint_original={result['state_vs_original']['worst_joint']['joint_name']} "
            f"action_worst_joint_original={result['action_vs_original']['worst']['feature_name']}",
            flush=True,
        )
    print(json.dumps({"output_run": str(output), "metrics": str(output / "manifests/ab_error_metrics.json"),
                      "curves": str(output / "plots/ab_error_curves.svg")}, ensure_ascii=False, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-run", type=Path, required=True)
    parser.add_argument("--b-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    args = parser.parse_args(argv)
    compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
