from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .overfit_visualization import write_comparison_svg
from .util import atomic_write_json, atomic_write_text


def _summary(run: Path) -> dict[str, Any]:
    resolved = run.expanduser().resolve()
    path = resolved / "manifests/training_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("memorization_benchmark") is not True:
        raise ValueError(f"not an overfit training summary: {path}")
    return {**value, "run_dir": str(resolved)}


def summarize_overfit_runs(run_paths: Iterable[Path], output_run: Path) -> dict[str, Any]:
    output = output_run.expanduser().resolve()
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "videos"):
        (output / child).mkdir(parents=True, exist_ok=True)
    records = [_summary(path) for path in run_paths]
    pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (str(record.get("model_profile")), int(record.get("seed")))
        phase = str(record.get("overfit_phase"))
        if phase in pairs[key]:
            raise ValueError(f"duplicate overfit run for {key}/{phase}")
        pairs[key][phase] = record
    compact_pairs = {
        seed: phases for (profile, seed), phases in pairs.items() if profile == "compact"
    }
    compact_complete = {
        seed: bool(
            {"capacity", "full"}.issubset(phases)
            and phases["capacity"].get("passed") is True
            and phases["full"].get("passed") is True
        )
        for seed, phases in compact_pairs.items()
    }
    required_compact_seeds = {20260828, 20260829, 20260830}
    required_seed_results = {
        seed: compact_complete.get(seed, False) for seed in sorted(required_compact_seeds)
    }
    reference = pairs.get(("reference", 20260828), {})
    reference_complete = {"capacity", "full"}.issubset(reference)
    baseline = pairs.get(("compact", 20260828), {}).get("full")
    metric_names = (
        "forward_one_normalized_rmse",
        "action_inverse_local_normalized_rmse",
        "arbitrary_state_normalized_rmse",
        "action_completion_macro_normalized_rmse",
        "history_action_normalized_rmse",
        "forward_rollout_8_normalized_rmse",
    )
    ablations: dict[str, Any] = {}
    if baseline is not None:
        baseline_metrics = baseline.get("best_validation") or {}
        for profile in ("compact_joint_id_only", "compact_no_aux"):
            candidate = pairs.get((profile, 20260828), {}).get("full")
            if candidate is None:
                continue
            candidate_metrics = candidate.get("best_validation") or {}
            ratios = {
                name: float(candidate_metrics.get(name, float("inf")))
                / max(float(baseline_metrics.get(name, 0.0)), 1e-12)
                for name in metric_names
            }
            maximum = max(ratios.values())
            ablations[profile] = {
                "metric_ratios_to_compact_full": ratios,
                "maximum_metric_ratio": maximum,
                "within_five_percent": bool(candidate.get("passed") and maximum <= 1.05),
                "interpretation": "32-motion memorization only; not a deployment necessity claim",
            }
    passed = (
        required_compact_seeds.issubset(compact_complete)
        and sum(required_seed_results.values()) >= 2
        and reference_complete
    )
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "memorization_benchmark": True,
        "generalization_claim_allowed": False,
        "compact_seed_pairs": compact_complete,
        "compact_complete_seed_count": sum(
            seed in compact_complete for seed in required_compact_seeds
        ),
        "compact_passed_seed_count": sum(required_seed_results.values()),
        "required_compact_seed_count": 3,
        "required_compact_passed_seed_count": 2,
        "reference_seed_20260828_complete": reference_complete,
        "ablations": ablations,
        "runs": records,
    }
    atomic_write_json(output / "manifests/overfit_suite_summary.json", result)
    lines = [
        "# 32-motion 过拟合实验汇总",
        "",
        "> 本报告仅用于训练集记忆能力诊断，不允许作为泛化或部署性能结论。",
        "",
        f"- Suite：{'PASS' if passed else 'FAIL'}",
        f"- 紧凑模型完整 seed：{sum(seed in compact_complete for seed in required_compact_seeds)}/3",
        f"- 紧凑模型两阶段通过：{sum(required_seed_results.values())}/3（要求至少 2）",
        f"- 原模型 seed 20260828 对照完整：{reference_complete}",
        "",
        "| 模型 | 阶段 | Seed | 参数量 | 最差阈值比 | 结果 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for record in sorted(
        records,
        key=lambda item: (
            str(item.get("model_profile")), int(item.get("seed")),
            str(item.get("overfit_phase")),
        ),
    ):
        lines.append(
            "| {model_profile} | {overfit_phase} | {seed} | {parameter_count} | "
            "{best_overfit_score:.4f} | {status} |".format(
                **record,
                status="PASS" if record.get("passed") else "FAIL",
            )
        )
    if ablations:
        lines.extend(["", "## 旁路消融（仅限32-motion记忆任务）", ""])
        for profile, value in sorted(ablations.items()):
            lines.append(
                f"- `{profile}` 最差指标比值 `{value['maximum_metric_ratio']:.4f}`；"
                f"是否在5%以内：`{value['within_five_percent']}`。"
            )
    atomic_write_text(output / "manifests/overfit_report.md", "\n".join(lines) + "\n")
    write_comparison_svg(records, output / "videos/overfit_comparison.svg")
    if passed:
        atomic_write_text(output / "markers/cvae_overfit_suite.ok", "PASS\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize 32-motion overfit runs")
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_overfit_runs(args.run, args.output_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise RuntimeError("overfit suite did not satisfy the 2-of-3 compact seed gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
