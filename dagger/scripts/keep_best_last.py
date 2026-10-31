"""Keep best_survival.pt / best_rmse.pt / last.pt (+ best.pt by --by) of a sonic_causal or MLP DAgger run.

round_XX.pt files are deleted only with --delete.
"""

import argparse
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("--by", choices=["rmse", "survival"], default="rmse", help="which best is also copied to best.pt")
parser.add_argument("--delete", action="store_true")
args = parser.parse_args()

ckpt_dir = args.run_dir / "checkpoints"
manifests = args.run_dir / "manifests"
metrics_path = next(
    (manifests / name for name in ("sonic_causal_metrics.partial.json", "dagger_mlp_metrics.partial.json")
     if (manifests / name).exists()),
    None,
)
if metrics_path is None:
    raise SystemExit(f"no metrics json in {manifests}")
metrics = json.loads(metrics_path.read_text())
rounds = [m for m in metrics if (ckpt_dir / f"round_{m['round']:02d}.pt").exists()]
if not rounds:
    raise SystemExit(f"no round_XX.pt with metrics in {ckpt_dir}")


def by_survival(m):
    s = m["student_only"]
    return (-s["survival_ratio"], s["termination_rate"], s["teacher_action_rmse"])


def by_rmse(m):
    s = m["student_only"]
    return (s["teacher_action_rmse"], -s["survival_ratio"])


def show(m):
    s = m["student_only"]
    return (f"round={m['round']} beta={m['beta']:.2f} rmse={s['teacher_action_rmse']:.4f} "
            f"term={s['termination_rate']:.4f} survival={s['survival_ratio']:.3f}")


best_survival = min(rounds, key=by_survival)
best_rmse = min(rounds, key=by_rmse)
last = max(rounds, key=lambda m: m["round"])
for title, key in (("survival", by_survival), ("rmse", by_rmse)):
    print(f"top5 by {title}:")
    for m in sorted(rounds, key=key)[:5]:
        print("  " + show(m))

kept = {
    "best_survival.pt": best_survival,
    "best_rmse.pt": best_rmse,
    "last.pt": last,
    "best.pt": best_survival if args.by == "survival" else best_rmse,
}
for name, m in kept.items():
    shutil.copy2(ckpt_dir / f"round_{m['round']:02d}.pt", ckpt_dir / name)
    print(f"{name:<17} <- {show(m)}")
(ckpt_dir / "kept.json").write_text(json.dumps(
    {"metrics_file": metrics_path.name, "best_metric": args.by,
     **{name: {"round": m["round"], **m["student_only"]} for name, m in kept.items()}},
    indent=2,
))

rest = sorted(ckpt_dir.glob("round_*.pt"))
if args.delete:
    for path in rest:
        path.unlink()
    print(f"deleted {len(rest)} round checkpoints")
else:
    print(f"dry run: {len(rest)} round_*.pt would be deleted; rerun with --delete")
