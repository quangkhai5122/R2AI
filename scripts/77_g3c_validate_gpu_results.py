"""Strictly import-check a downloaded Kaggle G3C result directory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.g3c.validate import validate_gpu_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidate-freeze")
    parser.add_argument(
        "--allow-smoke", action="store_true",
        help="accept fake/limited output for engineering checks only",
    )
    args = parser.parse_args()
    report = validate_gpu_results(
        payload_dir=args.payload,
        result_dir=args.results,
        output_path=args.out,
        require_scientific=not args.allow_smoke,
        candidate_freeze_path=args.candidate_freeze,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
