"""Command-line entry point for the O-level mathematics score project.

    python main.py validate                               # check the data file against the schema
    python main.py train                                  # full run (configs/config.yaml), ~10 min
    python main.py --config configs/fast.yaml train       # quick check that everything runs (~20 s)
    python main.py predict --input students.csv --output scored.csv   # score students with the latest model
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.artifacts import latest_run_dir
from src.config import ConfigError, load_config
from src.data import load_raw
from src.logging_utils import setup_logging
from src.predict import score
from src.schema import SchemaError, validate_raw
from src.train import run_training

logger = logging.getLogger("main")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="O-level mathematics score prediction")
    p.add_argument("--config", default="configs/config.yaml", help="YAML config profile")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="Validate a student CSV against the schema")
    v.add_argument("--input", help="CSV to validate (default: data.path from the config)")
    v.add_argument("--no-target", action="store_true", help="File has no final_test column (students to score)")

    t = sub.add_parser("train", help="Train, compare and select models; write artifacts")
    t.add_argument("--run-id", help="Name of the run folder (default: timestamp)")

    s = sub.add_parser("predict", help="Score students with a trained model")
    s.add_argument("--input", required=True, help="CSV of students (same columns as the training file)")
    s.add_argument("--output", required=True, help="Where to write the scored CSV")
    s.add_argument("--run-dir", help="Run folder to use (default: artifacts/LATEST)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    try:
        if args.command == "validate":
            report = validate_raw(load_raw(args.input or cfg["data"]["path"]), require_target=not args.no_target)
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.ok else 1
        if args.command == "train":
            run_training(cfg, args.run_id)
            return 0
        if args.command == "predict":
            run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(cfg["artifacts"]["dir"])
            scored = score(load_raw(args.input), run_dir, cfg["data"]["valid_age_range"])
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            scored.to_csv(args.output, index=False)
            logger.info("Wrote %s", args.output)
            return 0
    except (SchemaError, FileNotFoundError, KeyError) as e:
        logger.error("%s: %s", type(e).__name__, e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
