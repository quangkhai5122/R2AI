"""Build or validate the G3A same-corpus/new-question bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.g3a.builder import build_bundle, validate_bundle
from vifinqa.utils.io import setup_stdout


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--store-dir", default="artifacts/store")
    build.add_argument(
        "--public-questions",
        default="data/ViFinQA/questions/questions.jsonl",
    )
    build.add_argument(
        "--config", default="configs/g3a_evaluation_gate_v1.json"
    )
    build.add_argument("--out-dir", default="data/g3a_v1")
    build.add_argument(
        "--reviews", default="data/g3a_v1/g3a_hard_reviews.jsonl"
    )

    validate = sub.add_parser("validate")
    validate.add_argument("--bundle-dir", default="data/g3a_v1")
    validate.add_argument(
        "--allow-pending-hard",
        action="store_true",
        help="Only for authoring; promotion evaluation requires approved hard gold.",
    )

    args = parser.parse_args()
    if args.command == "build":
        review_path = Path(args.reviews)
        output = build_bundle(
            args.store_dir,
            args.public_questions,
            args.config,
            args.out_dir,
            review_path=review_path if review_path.exists() else None,
        )
        summary = {
            "bundle_fingerprint_sha256": output[
                "bundle_fingerprint_sha256"
            ],
            "counts": output["counts"],
            "leakage_guard": output["leakage_guard"],
        }
    else:
        summary = validate_bundle(
            args.bundle_dir,
            require_hard_approved=not args.allow_pending_hard,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
