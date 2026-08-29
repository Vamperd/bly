from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .overfit_visualization import write_parameter_efficiency_svg
from .util import atomic_write_json, atomic_write_text


TASKS = {
    "forward_rollout", "inverse", "history_action", "arbitrary_state", "arbitrary_action"
}
COMPACT_SUITE_GATE_TASKS = {"forward_rollout", "arbitrary_state", "arbitrary_action"}


def _summary(run: Path) -> dict[str, Any]:
    resolved = run.expanduser().resolve()
    path = resolved / "manifests/training_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    task = str(value.get("task_mode"))
    if value.get("memorization_benchmark") is not True or task not in TASKS:
        raise ValueError(f"not a fixed single-task memorization run: {path}")
    if value.get("fixed_training_masks") is not True:
        raise ValueError(f"single-task run did not use fixed training masks: {path}")
    if value.get("gate_latent_mode") != "posterior_mean":
        raise ValueError(f"single-task capacity gate must use posterior_mean: {path}")
    if (
        int(value.get("minimum_optimizer_steps", -1)) != 20_000
        or int(value.get("maximum_optimizer_steps", -1)) != 20_000
        or int(value.get("optimizer_steps", -1)) != 20_000
    ):
        raise ValueError(f"single-task run did not execute exactly 20k updates: {path}")
    return {**value, "run_dir": str(resolved)}


def summarize_single_task_runs(
    run_paths: Iterable[Path], output_run: Path
) -> dict[str, Any]:
    output = output_run.expanduser().resolve()
    for child in ("data", "manifests", "markers", "logs", "checkpoints", "videos"):
        (output / child).mkdir(parents=True, exist_ok=True)
    records = [_summary(path) for path in run_paths]
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        profile = str(record.get("model_profile", ""))
        family = "lean" if profile.startswith("lean_") else "compact" if profile.startswith("compact_") else profile
        key = (family, int(record["seed"]))
        task = str(record["task_mode"])
        if task in grouped[key]:
            raise ValueError(f"duplicate single-task result for {key}/{task}")
        grouped[key][task] = record
    groups: list[dict[str, Any]] = []
    suite_pass_by_family: dict[str, bool] = {}
    for (family, seed), tasks in sorted(grouped.items()):
        complete = set(tasks) == TASKS
        samples = {
            task: int(record.get("samples_seen_per_task", {}).get(task, -1))
            for task, record in tasks.items()
        }
        same_samples = len(set(samples.values())) == 1 and min(samples.values(), default=-1) > 0
        gate_tasks = COMPACT_SUITE_GATE_TASKS if family == "compact" else TASKS
        passed = complete and same_samples and all(
            tasks[task].get("passed") is True for task in gate_tasks
        )
        suite_pass_by_family[family] = suite_pass_by_family.get(family, True) and passed
        groups.append({
            "model_family": family,
            "seed": seed,
            "complete_five_tasks": complete,
            "same_samples_seen_per_task": same_samples,
            "samples_seen_per_task": samples,
            "suite_gate_tasks": sorted(gate_tasks),
            "passed": passed,
        })
    unsupported_families = sorted(set(suite_pass_by_family) - {"compact", "lean"})
    if unsupported_families:
        raise ValueError(f"unsupported single-task model families: {unsupported_families}")
    samples_by_task = {
        task: {
            int(record.get("samples_seen_per_task", {}).get(task, -1))
            for record in records if record.get("task_mode") == task
        }
        for task in TASKS
    }
    cross_model_same_samples = all(
        len(values) == 1 and min(values, default=-1) > 0
        for values in samples_by_task.values()
    )
    suite_pass = bool(
        groups and all(suite_pass_by_family.values()) and cross_model_same_samples
    )
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": suite_pass,
        "memorization_benchmark": True,
        "generalization_claim_allowed": False,
        "protocol": "fixed windows, fixed masks, fixed task, equal samples_seen_per_task",
        "deterministic_inverse_history_without_reference_are_diagnostic_only": True,
        "passed_by_family": suite_pass_by_family,
        "cross_model_same_samples_seen_per_task": cross_model_same_samples,
        "groups": groups,
        "runs": records,
    }
    atomic_write_json(output / "manifests/single_task_suite_summary.json", result)
    lines = [
        "# 32-motion 固定单任务容量报告", "",
        "> 本报告只回答训练样本记忆和任务冲突问题，不代表泛化或部署性能。", "",
        f"- Suite：{'PASS' if suite_pass else 'FAIL'}", "",
        f"- 跨模型每任务 samples_seen 一致：{'是' if cross_model_same_samples else '否'}", "",
        "| 模型 | 任务 | Seed | 参数量 | samples seen | 最差阈值比 | inverse 95%覆盖 | 结果 | 门禁角色 |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for record in sorted(records, key=lambda item: (
        str(item.get("model_profile")), str(item.get("task_mode")), int(item.get("seed"))
    )):
        task = str(record["task_mode"])
        diagnostic = (
            str(record.get("model_profile", "")).startswith("compact_")
            and task in {"inverse", "history_action"}
        )
        coverage_value = float(
            (record.get("best_validation") or {}).get(
                "inverse_probability_95_coverage", math.nan
            )
        )
        lines.append(
            "| {model} | {task} | {seed} | {parameters} | {samples} | {score:.4f} | {coverage} | {status} | {role} |".format(
                model=record.get("model_profile"), task=task, seed=record.get("seed"),
                parameters=record.get("parameter_count"), samples=record.get("samples_seen"),
                score=float(record.get("best_overfit_score", float("inf"))),
                coverage=f"{coverage_value:.3f}" if math.isfinite(coverage_value) else "—",
                status="PASS" if record.get("passed") else "FAIL",
                role="无 reference 诊断" if diagnostic else "suite 门禁",
            )
        )
    lines.extend((
        "", "## 解释规则", "",
        "- 单任务通过而联合任务失败：优先解释为共享主干梯度竞争。",
        "- 单任务失败且近邻目标分散度高：优先检查任务经验歧义和缺失控制意图。",
        "- 单任务失败但近邻目标分散度低：优先检查编码、关系头和优化路径。",
    ))
    atomic_write_text(output / "manifests/single_task_report_zh.md", "\n".join(lines) + "\n")
    write_parameter_efficiency_svg(records, output / "videos/parameter_efficiency.svg")
    if suite_pass:
        atomic_write_text(output / "markers/cvae_overfit_single_suite.ok", "PASS\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize fixed single-task overfit runs")
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_single_task_runs(args.run, args.output_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
