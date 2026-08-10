"""Build and validate the deterministic P2.4 human-gold development set.

Examples:
  python scripts/14_p24_devset.py build
  python scripts/14_p24_devset.py validate-bundle
  python scripts/14_p24_devset.py validate-gold --split tune --gold artifacts/devset_p24/p24_tune_gold.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.devset.p24 import (  # noqa: E402
    DEFAULT_EXPECTED_SOURCE_COUNT,
    DEFAULT_LOCKED_SIZE,
    DEFAULT_SEED,
    DEFAULT_TUNE_SIZE,
    LOCKED_SEAL_NAME,
    build_bundle,
    check_tune_input,
    seal_locked_gold,
    validate_bundle,
    validate_gold_file,
    verify_locked_seal,
)
from vifinqa.devset.evaluate import evaluate_codegen, fill_gold_hashes  # noqa: E402
from vifinqa.utils.io import setup_stdout  # noqa: E402


DEFAULT_QUESTIONS = "data/ViFinQA/questions/questions.jsonl"
DEFAULT_RETRIEVAL = "artifacts/retrieval.jsonl"
DEFAULT_OUT = "artifacts/devset_p24"
DEFAULT_STORE = "artifacts/store"


def _common_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--retrieval", default=DEFAULT_RETRIEVAL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument(
        "--expected-source-count", type=int, default=DEFAULT_EXPECTED_SOURCE_COUNT
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P2.4 deterministic dev-set builder and gold validator"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="sample tune=100 and locked=50 templates")
    _common_source_args(build)
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--tune-size", type=int, default=DEFAULT_TUNE_SIZE)
    build.add_argument("--locked-size", type=int, default=DEFAULT_LOCKED_SIZE)

    bundle = sub.add_parser("validate-bundle", help="verify split/source/hash guards")
    _common_source_args(bundle)

    gold = sub.add_parser("validate-gold", help="validate a labeled or template JSONL")
    gold.add_argument("--out-dir", default=DEFAULT_OUT)
    gold.add_argument("--split", choices=("tune", "locked"), required=True)
    gold.add_argument("--gold", required=True)
    gold.add_argument("--store-dir", default=DEFAULT_STORE)
    gold.add_argument(
        "--allow-template", action="store_true",
        help="identity/schema check only; strict cell/AST/replay validation is default",
    )

    tune = sub.add_parser(
        "check-tune-input", help="fail if a training/tuning JSONL contains locked ids"
    )
    tune.add_argument("--out-dir", default=DEFAULT_OUT)
    tune.add_argument("--input", required=True)

    seal = sub.add_parser("seal-locked", help="strictly validate and hash locked gold")
    seal.add_argument("--out-dir", default=DEFAULT_OUT)
    seal.add_argument("--gold", required=True)
    seal.add_argument("--store-dir", default=DEFAULT_STORE)
    seal.add_argument("--seal", default="")

    verify = sub.add_parser("verify-locked", help="verify an existing locked-gold seal")
    verify.add_argument("--out-dir", default=DEFAULT_OUT)
    verify.add_argument("--gold", required=True)
    verify.add_argument("--store-dir", default=DEFAULT_STORE)
    verify.add_argument("--seal", default="")

    hashes = sub.add_parser(
        "fill-hashes",
        help="write a new draft with canonical evidence/AST hashes; never overwrite",
    )
    hashes.add_argument("--out-dir", default=DEFAULT_OUT)
    hashes.add_argument("--split", choices=("tune", "locked"), required=True)
    hashes.add_argument("--input", required=True)
    hashes.add_argument("--output", required=True)
    hashes.add_argument("--store-dir", default=DEFAULT_STORE)

    evaluate = sub.add_parser(
        "evaluate", help="replay a complete codegen artifact against strict gold"
    )
    evaluate.add_argument("--out-dir", default=DEFAULT_OUT)
    evaluate.add_argument("--split", choices=("tune", "locked"), required=True)
    evaluate.add_argument("--gold", required=True)
    evaluate.add_argument("--codegen", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--store-dir", default=DEFAULT_STORE)
    evaluate.add_argument(
        "--seal", default="",
        help="required for locked evaluation; ignored for tune",
    )
    return parser


def main() -> None:
    setup_stdout()
    args = make_parser().parse_args()
    if args.command == "build":
        result = build_bundle(
            args.questions, args.retrieval, args.out_dir,
            seed=args.seed, tune_size=args.tune_size, locked_size=args.locked_size,
            expected_source_count=args.expected_source_count,
        )
        output = {
            "bundle_fingerprint_sha256": result["bundle_fingerprint_sha256"],
            "source_count": result["source"]["count"],
            "tune_count": result["splits"]["tune"]["count"],
            "locked_count": result["splits"]["locked"]["count"],
        }
    elif args.command == "validate-bundle":
        output = validate_bundle(
            args.out_dir, questions_path=args.questions,
            retrieval_path=args.retrieval,
            expected_source_count=args.expected_source_count,
        )
    elif args.command == "validate-gold":
        output = validate_gold_file(
            args.gold, args.out_dir, args.split,
            store_dir=None if args.allow_template else args.store_dir,
            require_complete=not args.allow_template,
        )
    elif args.command == "check-tune-input":
        output = check_tune_input(args.input, args.out_dir)
    elif args.command == "seal-locked":
        seal_path = args.seal or str(Path(args.out_dir) / LOCKED_SEAL_NAME)
        output = seal_locked_gold(
            args.gold, args.out_dir, seal_path, store_dir=args.store_dir
        )
    elif args.command == "verify-locked":
        seal_path = args.seal or str(Path(args.out_dir) / LOCKED_SEAL_NAME)
        output = verify_locked_seal(
            args.gold, args.out_dir, seal_path, store_dir=args.store_dir
        )
    elif args.command == "fill-hashes":
        output = fill_gold_hashes(
            args.input, args.output, args.out_dir, args.split,
            store_dir=args.store_dir,
        )
    elif args.command == "evaluate":
        output = evaluate_codegen(
            args.codegen, args.gold, args.out_dir, args.split,
            store_dir=args.store_dir, output_path=args.output,
            seal_path=args.seal or None,
        )
        output = {
            "split": output["split"],
            "metrics": output["metrics"],
            "population_weighted": output["population_weighted"],
            "report": args.output,
            "codegen_sha256": output["provenance"]["codegen_sha256"],
            "run_signature_set_sha256": output["provenance"][
                "run_signature_set_sha256"
            ],
        }
    else:  # pragma: no cover - argparse enforces a known command
        raise SystemExit(f"unknown command: {args.command}")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
