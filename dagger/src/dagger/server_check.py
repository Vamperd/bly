"""Read-only server preflight. Never installs packages or starts Isaac Sim."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="store_true", help="Read physical/reference samples and verify migrated source hashes")
    parser.add_argument("--cuda", action="store_true", help="Run a small tensor operation on each visible GPU")
    args = parser.parse_args()
    failed = []

    def report(name, passed, detail):
        print(json.dumps({"check": name, "pass": bool(passed), "detail": detail}, ensure_ascii=False), flush=True)
        if not passed:
            failed.append(name)

    print(f"python={sys.executable} version={sys.version}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    for module in ("torch", "numpy", "h5py", "omegaconf", "hydra", "yaml", "filelock", "isaaclab", "gear_sonic"):
        try:
            spec = importlib.util.find_spec(module)
            report(f"module:{module}", spec is not None, None if spec is None else spec.origin)
        except Exception as error:
            report(f"module:{module}", False, str(error))
    try:
        report("isaacsim-distribution", True, importlib.metadata.version("isaacsim"))
    except importlib.metadata.PackageNotFoundError:
        # Standalone installations can provide Isaac through their own launcher.
        print('isaacsim-distribution=unknown (standalone installation must be checked with its launcher)', flush=True)
    for variable in ("SONIC_REPO_ROOT", "CVAE_STUDENT_CHECKPOINT", "SONIC_TEACHER_CHECKPOINT",
                     "CVAE_KL2_DATASET_RUN", "CVAE_KL2_MOTION_FILE", "DAGGER_MOTION_MANIFEST", "CVAE_MODEL_CONFIG"):
        value = os.environ.get(variable, "")
        report(variable, bool(value) and Path(value).exists(), value)
    try:
        import torch
        print(f"torch={torch.__version__} built_cuda={torch.version.cuda}", flush=True)
        report("cuda_available", torch.cuda.is_available(), torch.cuda.device_count())
        if args.cuda:
            selected = bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
            report("gpu-selection", selected, "set CUDA_VISIBLE_DEVICES to your allocated GPUs before probing")
            for index in range(torch.cuda.device_count() if selected else 0):
                device = f"cuda:{index}"
                value = torch.ones((32, 32), device=device) @ torch.ones((32, 32), device=device)
                report(device, value[0, 0].item() == 32, torch.cuda.get_device_name(index))
    except Exception as error:
        report("torch-runtime", False, str(error))
    if args.data:
        try:
            from cvae_sa.long_horizon_dataset import LongHorizonWindowDataset
            from cvae_sa.reference_anchor_dataset import ReferenceAnchorDataset
            from cvae_sa.path_mapping import dataset_path_mapping, resolve_dataset_path
            from cvae_sa.util import file_sha256

            root = Path(os.environ["CVAE_KL2_DATASET_RUN"])
            print(f"dataset_path_mapping={dataset_path_mapping()}", flush=True)
            manifest = json.loads((root / "manifests/reference_mixed_manifest.json").read_text())
            for field, relative in (("normalization_sha256", "data/normalization.npz"),
                                    ("anchor_windows_sha256", "manifests/anchor_windows.jsonl"),
                                    ("reference_table_sha256", "data/reference_states.npz")):
                report(field, file_sha256(root / relative) == manifest[field], relative)
            source_hashes = json.loads((root / "manifests/source_hashes.json").read_text())
            for category in ("hdf5", "schema"):
                for item in source_hashes[category]:
                    path = resolve_dataset_path(item["path"])
                    report(f"source:{category}", file_sha256(path) == item["sha256"], str(path))
            for cls in (LongHorizonWindowDataset, ReferenceAnchorDataset):
                dataset = cls(root, "train")
                try:
                    sample = dataset[0]
                    report(cls.__name__, tuple(sample["physical_state"].shape) == (65, 70)
                           and tuple(sample["action"].shape) == (64, 29),
                           {"samples": len(dataset), "episode_ref": sample["episode_ref"]})
                finally:
                    dataset.close()
            import torch
            checkpoint = torch.load(os.environ["CVAE_STUDENT_CHECKPOINT"], map_location="cpu", weights_only=False)
            norm = (checkpoint.get("dataset_identity") or {}).get("normalization_sha256")
            report("checkpoint-normalization", norm == manifest["normalization_sha256"], norm)
            motions = json.loads(Path(os.environ["DAGGER_MOTION_MANIFEST"]).read_text())
            keys = set(checkpoint.get("train_motion_keys") or [])
            report("checkpoint-motion-coverage", bool(keys) and set(motions["motion_keys"]) <= keys, len(keys))
        except Exception as error:
            report("dataset-runtime", False, f"{type(error).__name__}: {error}")
    print(json.dumps({"preflight_pass": not failed, "failed": failed,
                      "isaac_simulator_smoke": "not_run"}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
