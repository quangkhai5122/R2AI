"""Evaluate a G3A offline submission with competition-shaped metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.g3a.evaluate import evaluate_submission
from vifinqa.utils.io import setup_stdout


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--bundle-dir", default="data/g3a_v1")
    parser.add_argument(
        "--config", default="configs/g3a_evaluation_gate_v1.json"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow-pending-hard",
        action="store_true",
        help="Only for draft authoring; not valid for candidate promotion.",
    )
    args = parser.parse_args()
    report = evaluate_submission(
        args.submission,
        args.bundle_dir,
        args.config,
        output_path=args.out,
        require_hard_approved=not args.allow_pending_hard,
    )
    print(json.dumps({
        "integrity": report["integrity"],
        "metrics": report["metrics"],
        "weight_scenarios": report["weight_scenarios"],
        "report": args.out,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
