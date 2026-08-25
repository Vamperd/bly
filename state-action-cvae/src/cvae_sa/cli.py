from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="SONIC State-Action CVAE command line")
    parser.add_argument(
        "command",
        choices=(
            "build-index", "build-physics-index", "smoke-train", "train", "evaluate", "sample"
        ),
    )
    args, remainder = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remainder]
    if args.command == "build-index":
        from .indexer import main as command_main
    elif args.command == "build-physics-index":
        from .physics_indexer import main as command_main
    elif args.command in {"smoke-train", "train"}:
        from .trainer import main as command_main

        if args.command == "smoke-train" and "--smoke" not in sys.argv:
            sys.argv.append("--smoke")
    elif args.command == "evaluate":
        from .evaluator import main as command_main
    else:
        from .sampler import main as command_main
    return command_main()


if __name__ == "__main__":
    raise SystemExit(main())
