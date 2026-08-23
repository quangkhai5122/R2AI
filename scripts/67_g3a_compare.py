"""Compare two G3A reports using the unknown-weight promotion policy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.g3a.evaluate import compare_reports
from vifinqa.utils.io import setup_stdout


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--config", default="configs/g3a_evaluation_gate_v1.json"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = compare_reports(
        args.baseline,
        args.candidate,
        args.config,
        output_path=args.out,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
