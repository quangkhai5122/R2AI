"""Freeze an exact G3B promotion candidate before reading locked results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.g3b.freeze import create_candidate_freeze
from vifinqa.utils.io import setup_stdout


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--corpus-dir", default="data/g3b_v1")
    parser.add_argument(
        "--config", default="configs/g3b_evaluation_v1.json"
    )
    parser.add_argument("--submission")
    parser.add_argument("--typed-predictions")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    freeze = create_candidate_freeze(
        args.candidate_name,
        args.corpus_dir,
        args.config,
        submission=args.submission,
        typed_predictions=args.typed_predictions,
        output_path=args.out,
    )
    print(json.dumps(freeze, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
