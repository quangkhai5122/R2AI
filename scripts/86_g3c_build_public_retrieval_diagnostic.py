"""Build or validate the frozen R4/B1-fixed public retrieval diagnostic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vifinqa import config
from vifinqa.g3c_official.public_diagnostic import (
    build_public_retrieval_diagnostic,
    validate_public_retrieval_diagnostic,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--r0",
        default="artifacts/g3c_v1/official_qwen_results/r0_retrieval.jsonl",
    )
    parser.add_argument(
        "--r4",
        default="artifacts/g3c_v1/official_qwen_results/r4_retrieval.jsonl",
    )
    parser.add_argument(
        "--codegen",
        default="artifacts/clean_v1/b1_nf4/codegen_results_nf4.jsonl",
    )
    parser.add_argument(
        "--baseline-results",
        default=(
            "artifacts/clean_v1/b1_nf4/submission_clean_nf4/results.json"
        ),
    )
    parser.add_argument(
        "--baseline-zip",
        default=(
            "artifacts/clean_v1/b1_nf4/submission_clean_nf4/submission.zip"
        ),
    )
    parser.add_argument(
        "--official-result-manifest",
        default=(
            "artifacts/g3c_v1/official_qwen_results/"
            "g3c_official_result_manifest.json"
        ),
    )
    parser.add_argument(
        "--official-freeze",
        default=(
            "artifacts/g3c_v1/official_qwen_results/"
            "g3c_official_artifact_freeze.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "artifacts/g3c_v1/official_submission_b1fixed_r4_v1"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    _common(build)
    build.add_argument("--store-dir", default=str(config.STORE_DIR))
    validate = sub.add_parser("validate")
    _common(validate)
    args = parser.parse_args()
    common = {
        "r0_path": args.r0,
        "r4_path": args.r4,
        "codegen_path": args.codegen,
        "baseline_results_path": args.baseline_results,
        "baseline_zip": args.baseline_zip,
        "official_result_manifest_path": args.official_result_manifest,
        "official_freeze_path": args.official_freeze,
        "output_dir": args.out_dir,
    }
    if args.command == "build":
        manifest, audit = build_public_retrieval_diagnostic(
            store_dir=args.store_dir,
            **common,
        )
        result = {
            "status": manifest["status"],
            "candidate_name": manifest["candidate_name"],
            "submission_sha256": manifest["artifacts"]["submission_zip"][
                "sha256"
            ],
            "manifest_fingerprint": manifest["manifest_fingerprint"],
            "row_delta": audit["row_delta"],
            "zip_delta": audit["zip_delta"],
        }
    else:
        result = validate_public_retrieval_diagnostic(**common)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
