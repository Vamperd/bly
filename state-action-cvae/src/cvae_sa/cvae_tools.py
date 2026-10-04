"""Independent evaluate / monitor / export-report commands (no training side effects)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import torch
from torch.utils.data import DataLoader

from .cvae_diagnostics import evaluate, read_normalization
from .cvae_protocol import CHECKPOINT, PROTOCOL, Fixtures, materialize_normalized_window_cache
from .cvae_training import (_autocast_context, _make_eval_route, check_source_identity,
                            data_identity, make_dataset, source_info, provenance)
from .models import build_model
from .posterior_direct_output import assert_output_isolated
from .posterior_hierarchical_standard_cvae import load_checkpoint
from .util import atomic_write_json, file_sha256, load_config


def evaluate_checkpoint(args):
    source_path, run, dataset_run = args.checkpoint.resolve(), args.output_run.resolve(), args.dataset_run.resolve()
    assert_output_isolated(run, [source_path.parent.parent, dataset_run])
    if run.exists() and any(run.iterdir()):
        raise ValueError("read-only evaluation requires a new empty output run")
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    config = load_config(args.config) if args.config else checkpoint.get("config")
    if config is None:
        config = {"model": checkpoint.get("model_config", checkpoint.get("model_signature", {})),
                  "data": {"window_transitions": 64, "stride": 64, "max_episodes": 256, "num_workers": 0}}
    model = build_model(config["model"])
    load_checkpoint(model, source_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = str(config.get("training", {}).get("precision", "FP32")).upper()
    if precision == "BF16":
        _autocast_context(device, precision)
    model.to(device=device, dtype=torch.float32)
    dataset, indices = make_dataset(dataset_run, config)
    try:
        base_fixtures = Fixtures(dataset, indices, expand=False)
        windows = base_fixtures.manifest()
        identity = data_identity(dataset_run, windows)
        check = check_source_identity(checkpoint, identity, allow_unknown=True)
        normalization = read_normalization(dataset_run / "data/normalization.npz")
        (run / "data").mkdir(parents=True, exist_ok=True)
        data_cfg = config.get("data", {})
        cache_mode = str(data_cfg.get("cache_mode", "none")).lower()
        cached_dataset, cache_info = dataset, {"requested": False, "enabled": False, "reason": "disabled"}
        if cache_mode in {"auto", "on"}:
            cached_dataset, cache_info = materialize_normalized_window_cache(
                dataset, indices, cache_path=run / "data/normalized_windows.pt",
                cache_identity={"protocol_version": PROTOCOL, "dataset_identity": identity,
                                "selected_window_ids": [row["stable_window_id"] for row in windows]},
                max_bytes=int(data_cfg.get("cache_max_bytes", 0)),
            )
        fixtures = Fixtures(cached_dataset, indices, expand=args.route != "A")
        shutil.copy2(dataset_run / "data/normalization.npz", run / "data/normalization.npz")
        atomic_write_json(run / "manifests/selected_windows.json", windows)
        runtime_contract = checkpoint.get("runtime_contract", {})
        atomic_write_json(run / "manifests/provenance.json", provenance(config, {
            "route": args.route, "samples": args.samples, "mask_seed": args.mask_seed,
            "runtime_contract": runtime_contract,
        }, identity, source_info(source_path, checkpoint)))
        pin_memory = bool(data_cfg.get("pin_memory", False)) and device.type == "cuda"
        non_blocking = bool(data_cfg.get("non_blocking_transfer", False)) and pin_memory
        loader = DataLoader(fixtures, batch_size=args.micro_batch, shuffle=False, pin_memory=pin_memory,
                            generator=torch.Generator().manual_seed(0))
        routes = ("posterior_mean", "posterior_sample", "standard_normal", "zero") if args.route == "C" else (args.route,)
        compile_cfg = config.get("compile", {})
        compiled_routes = {}
        compile_infos = []
        for route in routes:
            compiled, eager, info = _make_eval_route(model, route, device, precision, compile_cfg)
            compiled_routes[route] = compiled
            compile_infos.append(info)
        reports = {}
        for route in routes:
            print(f"RUN_DIR={run} route={route} checkpoint={source_path}", flush=True)
            reports[route], _ = evaluate(model, loader, device, normalization, route=route, seed=args.mask_seed,
                samples=args.samples, output_dir=run / "evaluations" / route, export_all=args.export_all,
                heartbeat=lambda n: print(f"[{route}] evaluation batch {n}/{len(loader)}", flush=True) if n % 10 == 0 else None,
                route_callable=compiled_routes[route], non_blocking=non_blocking,
                precision=precision, compile_info=compile_infos[routes.index(route)])
        summary = {"protocol_version": PROTOCOL, "execution_pass": True, "quality_pass": None,
            "source_checkpoint": source_info(source_path, checkpoint), "identity_check": check,
            "historical_reassessment_not_original_fixture_replay": checkpoint.get("format_version") != CHECKPOINT and args.route != "A",
            "dataset_identity": identity, "reports": reports, "runtime_contract": runtime_contract,
            "cache_info": cache_info, "compile_info": compile_infos,
            "data_transfer": {"pin_memory": pin_memory, "non_blocking": non_blocking},
            "unique_next_step": "REVIEW_TAIL_COORDINATES_AND_TRACES"}
        atomic_write_json(run / "manifests/diagnostic_summary.json", summary)
        return summary
    finally:
        dataset.close()


def monitor(run, interval=10., once=False):
    run = run.resolve()
    position, remainder = 0, ""
    while True:
        progress_path = run / "manifests/progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"status": "not_started"}
        updated = progress.get("updated_at")
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(updated)).total_seconds() if updated else None
        pid = progress.get("pid")
        process = "unknown"
        if pid and Path(f"/proc/{pid}/cmdline").exists():
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            process = "alive_matching_run" if str(run) in command else "pid_reused_or_run_mismatch"
        elif os.name != "nt" and pid:
            process = "not_running"
        print(json.dumps({"run": str(run), "pid_state": process, "heartbeat_age_seconds": age, **progress}, ensure_ascii=False, indent=2), flush=True)
        path = run / "logs/metrics.jsonl"
        if path.exists():
            if path.stat().st_size < position:
                position, remainder = 0, ""
            with path.open(encoding="utf-8") as stream:
                stream.seek(position)
                text = remainder + stream.read()
                position = stream.tell()
            lines = text.split("\n")
            remainder = lines.pop()
            # Read incrementally, keep recent train/eval information in a usable display.
            for line in lines[-10:]:
                if line.strip():
                    print(line, flush=True)
        for name in ("last.pt", "best.pt", "best_reconstruction.pt"):
            path = run / "checkpoints" / name
            if path.exists():
                print(f"{name}: bytes={path.stat().st_size} mtime={datetime.fromtimestamp(path.stat().st_mtime).isoformat()}", flush=True)
        if once or progress.get("status") in {"completed", "failed", "interrupted"}:
            return
        time.sleep(interval)


def export_report(run, output=None):
    run = run.resolve()
    output = output.resolve() if output else run / "diagnostic_report.zip"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite report: {output}")
    files = []
    for folder in ("manifests", "logs", "plots", "evaluations"):
        for path in (run / folder).rglob("*"):
            if path.is_file() and not path.is_symlink() and "all_predictions" not in path.parts and path.suffix in {".json", ".jsonl", ".svg", ".txt", ".log"}:
                files.append(path)
    norm = run / "data/normalization.npz"
    if norm.is_file():
        files.append(norm)
    hashes = {str(p.relative_to(run)): file_sha256(p) for p in files}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(run)))
        archive.writestr("report_index.json", json.dumps({"protocol_version": PROTOCOL, "run": str(run), "hashes": hashes,
            "excluded": ["HDF5", "checkpoints", "all_predictions"], "snapshot_warning": "export after completion for a consistent snapshot"}, indent=2))
    print(f"REPORT={output}\nBYTES={output.stat().st_size}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--dataset-run", type=Path, required=True)
    ev.add_argument("--output-run", type=Path, required=True)
    ev.add_argument("--config", type=Path)
    ev.add_argument("--route", choices=("A", "B", "C", "posterior_mean", "posterior_sample", "standard_normal", "zero"), required=True)
    ev.add_argument("--micro-batch", type=int, default=32)
    ev.add_argument("--mask-seed", type=int, default=20260920)
    ev.add_argument("--samples", type=int, default=8)
    ev.add_argument("--export-all", action="store_true")
    mon = sub.add_parser("monitor")
    mon.add_argument("--run", type=Path, required=True)
    mon.add_argument("--interval", type=float, default=10.)
    mon.add_argument("--once", action="store_true")
    report = sub.add_parser("export-report")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        if args.micro_batch < 1 or args.samples < 1:
            parser.error("positive micro-batch and samples required")
        evaluate_checkpoint(args)
    elif args.command == "monitor":
        if args.interval <= 0:
            parser.error("positive interval required")
        monitor(args.run, args.interval, args.once)
    else:
        export_report(args.run, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
