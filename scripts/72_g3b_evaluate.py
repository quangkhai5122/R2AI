"""Run G3B oracle-evidence or end-to-end diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.g3b.evaluate import evaluate_g3b
from vifinqa.utils.io import setup_stdout


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-mode", required=True, choices=("dev", "promotion")
    )
    parser.add_argument(
        "--evidence-mode",
        required=True,
        choices=("oracle_evidence", "end_to_end"),
    )
    parser.add_argument("--corpus-dir", default="data/g3b_v1")
    parser.add_argument(
        "--extension-dir", default="data/g3a_extension_v1"
    )
    parser.add_argument(
        "--config", default="configs/g3b_evaluation_v1.json"
    )
    parser.add_argument("--submission")
    parser.add_argument("--typed-predictions")
    parser.add_argument("--candidate-freeze")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = evaluate_g3b(
        args.corpus_dir,
        args.extension_dir,
        args.config,
        policy_mode=args.policy_mode,
        evidence_mode=args.evidence_mode,
        submission=args.submission,
        typed_predictions=args.typed_predictions,
        candidate_freeze=args.candidate_freeze,
        output_path=args.out,
    )
    print(json.dumps({
        "integrity": report["integrity"],
        "metrics": report["metrics"],
        "report": args.out,
        "retrieval_interpretation": report[
            "retrieval_interpretation"
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
