"""Keep best.pt / last.pt of a sonic_causal DAgger run; delete round_XX.pt only with --delete."""

import argparse
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("--delete", action="store_true")
args = parser.parse_args()

ckpt_dir = args.run_dir / "checkpoints"
metrics = json.loads((args.run_dir / "manifests" / "sonic_causal_metrics.partial.json").read_text())
rounds = [m for m in metrics if (ckpt_dir / f"round_{m['round']:02d}.pt").exists()]
if not rounds:
    raise SystemExit(f"no round_XX.pt with metrics in {ckpt_dir}")
best = min(rounds, key=lambda m: m["student_only"]["teacher_action_rmse"])
last = max(rounds, key=lambda m: m["round"])

for m in sorted(rounds, key=lambda m: m["student_only"]["teacher_action_rmse"])[:5]:
    s = m["student_only"]
    print(f"round={m['round']} beta={m['beta']:.2f} rmse={s['teacher_action_rmse']:.4f} "
          f"term={s['termination_rate']:.4f} survival={s['survival_ratio']:.2f}")
print(f"best=round_{best['round']:02d} last=round_{last['round']:02d}")

for name, m in (("best.pt", best), ("last.pt", last)):
    shutil.copy2(ckpt_dir / f"round_{m['round']:02d}.pt", ckpt_dir / name)
(ckpt_dir / "kept.json").write_text(json.dumps({"best_round": best["round"], "last_round": last["round"], "best_metric": "teacher_action_rmse"}, indent=2))

rest = sorted(ckpt_dir.glob("round_*.pt"))
if args.delete:
    for path in rest:
        path.unlink()
    print(f"deleted {len(rest)} round checkpoints")
else:
    print(f"dry run: {len(rest)} round_*.pt would be deleted; rerun with --delete")
