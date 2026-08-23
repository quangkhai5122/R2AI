"""Build or validate the G3A extension and G3B evaluation corpus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.g3b.builder import build_corpus, validate_corpus
from vifinqa.utils.io import setup_stdout


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--store-dir", default="artifacts/store")
    build.add_argument(
        "--public-questions",
        default="data/ViFinQA/questions/questions.jsonl",
    )
    build.add_argument(
        "--config", default="configs/g3b_evaluation_v1.json"
    )
    build.add_argument(
        "--extension-dir", default="data/g3a_extension_v1"
    )
    build.add_argument("--corpus-dir", default="data/g3b_v1")
    build.add_argument(
        "--reviews", default="data/g3b_v1/g3b_reviews.jsonl"
    )
    validate = commands.add_parser("validate")
    validate.add_argument(
        "--config", default="configs/g3b_evaluation_v1.json"
    )
    validate.add_argument(
        "--extension-dir", default="data/g3a_extension_v1"
    )
    validate.add_argument("--corpus-dir", default="data/g3b_v1")
    validate.add_argument(
        "--allow-pending-reviews", action="store_true"
    )
    args = parser.parse_args()
    if args.command == "build":
        review_path = Path(args.reviews)
        result = build_corpus(
            args.store_dir,
            args.public_questions,
            args.config,
            args.extension_dir,
            args.corpus_dir,
            review_path=review_path if review_path.exists() else None,
        )
        output = {
            "g3a_extension_fingerprint_sha256": result[
                "g3a_extension"
            ]["fingerprint_sha256"],
            "g3b_fingerprint_sha256": result[
                "g3b"
            ]["fingerprint_sha256"],
            "counts": result["g3b"]["counts"],
            "candidate_pool": result["candidate_pool"],
        }
    else:
        output = validate_corpus(
            args.extension_dir,
            args.corpus_dir,
            args.config,
            require_reviews=not args.allow_pending_reviews,
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
