#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from litekv.config import load_config  # noqa: E402
from litekv.experiment import run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or validate the LiteKV demo config.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "experiments" / "configs" / "default.yaml"),
        help="Path to a LiteKV config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the resolved settings without running experiments.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    print(json.dumps(config.as_dict(), indent=2, sort_keys=True))

    if not args.dry_run:
        artifacts = run_experiment(config)
        print(
            "Wrote {} rows to {} and {}.".format(
                len(artifacts.rows),
                artifacts.csv_path,
                artifacts.json_path,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
