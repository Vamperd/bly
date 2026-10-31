"""Keep best.pt / last.pt of a sonic_causal or MLP DAgger run; delete round_XX.pt only with --delete."""

import argparse
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("--by", choices=["rmse", "survival"], default="rmse")
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


def rank(m):
    s = m["student_only"]
    if args.by == "survival":
        return (-s["survival_ratio"], s["termination_rate"], s["teacher_action_rmse"])
    return (s["teacher_action_rmse"],)


ordered = sorted(rounds, key=rank)
best = ordered[0]
last = max(rounds, key=lambda m: m["round"])
for m in ordered[:5]:
    s = m["student_only"]
    print(f"round={m['round']} beta={m['beta']:.2f} rmse={s['teacher_action_rmse']:.4f} "
          f"term={s['termination_rate']:.4f} survival={s['survival_ratio']:.2f}")
print(f"best=round_{best['round']:02d} (by {args.by}) last=round_{last['round']:02d}")

for name, m in (("best.pt", best), ("last.pt", last)):
    shutil.copy2(ckpt_dir / f"round_{m['round']:02d}.pt", ckpt_dir / name)
(ckpt_dir / "kept.json").write_text(json.dumps({"best_round": best["round"], "last_round": last["round"], "best_metric": args.by}, indent=2))

rest = sorted(ckpt_dir.glob("round_*.pt"))
if args.delete:
    for path in rest:
        path.unlink()
    print(f"deleted {len(rest)} round checkpoints")
else:
    print(f"dry run: {len(rest)} round_*.pt would be deleted; rerun with --delete")
