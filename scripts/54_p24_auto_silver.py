"""Build or evaluate the automatic P2.4 adjacent-report silver benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.devset.p24_silver import (  # noqa: E402
    DEFAULT_MAX_PER_REPORT_PAIR,
    DEFAULT_SEED,
    DEFAULT_MAX_TICKERS_PER_SPLIT,
    build_auto_silver_bundle,
    evaluate_auto_silver,
)
from vifinqa.utils.io import setup_stdout  # noqa: E402


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automatic P2.4 silver benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build immutable train/tune/locked splits")
    build.add_argument("--store-dir", default="artifacts/store")
    build.add_argument("--out-dir", default="artifacts/p24_silver_auto")
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument(
        "--max-per-report-pair", type=int, default=DEFAULT_MAX_PER_REPORT_PAIR,
    )
    build.add_argument(
        "--max-tickers-per-split", type=int,
        default=DEFAULT_MAX_TICKERS_PER_SPLIT,
    )

    evaluate = sub.add_parser("evaluate", help="evaluate v5.3 resolver on one split")
    evaluate.add_argument("--split", required=True)
    evaluate.add_argument("--store-dir", default="artifacts/store")
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--expect-split-sha256", default="")
    return parser


def main() -> None:
    setup_stdout()
    args = make_parser().parse_args()
    if args.command == "build":
        result = build_auto_silver_bundle(
            args.store_dir, args.out_dir, seed=args.seed,
            max_per_report_pair=args.max_per_report_pair,
            max_tickers_per_split=args.max_tickers_per_split,
        )
        output = {
            "bundle_fingerprint_sha256": result["bundle_fingerprint_sha256"],
            "counts": result["counts"],
            "files": result["files"],
        }
    else:
        result = evaluate_auto_silver(
            args.split, args.store_dir, args.out,
            expected_split_sha256=args.expect_split_sha256,
        )
        output = {"metrics": result["metrics"], "report": args.out}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
