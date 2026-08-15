"""Evaluate complete codegen on final P2.4 tune gold without opening locked."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.devset import evaluate as evaluator  # noqa: E402
from vifinqa.devset.p24 import TUNE_QUESTIONS, validate_gold_records  # noqa: E402
from vifinqa.devset.p24_authoring_ext import P24ForensicTableLoader  # noqa: E402
from vifinqa.utils.io import read_jsonl, setup_stdout  # noqa: E402


def _validate_tune_forensic(
    gold_path, bundle_dir, split, *, store_dir=None, require_complete=True,
    verify_bundle=True,
):
    if split != "tune" or store_dir is None or not require_complete:
        raise ValueError("forensic evaluator supports complete tune gold only")
    questions = read_jsonl(Path(bundle_dir) / TUNE_QUESTIONS)
    return validate_gold_records(
        read_jsonl(gold_path), questions, "tune",
        table_loader=P24ForensicTableLoader(store_dir), require_complete=True,
    )


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codegen", required=True)
    parser.add_argument("--gold", default="artifacts/devset_p24/p24_tune_gold.final.jsonl")
    parser.add_argument("--bundle-dir", default="artifacts/devset_p24")
    parser.add_argument("--store-dir", default="artifacts/store")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # Patch only the tune-gold validation hook. Prediction replay deliberately
    # retains the standard submission table loader.
    evaluator.validate_gold_file = _validate_tune_forensic
    report = evaluator.evaluate_codegen(
        args.codegen, args.gold, args.bundle_dir, "tune",
        store_dir=args.store_dir, output_path=args.output,
        verify_bundle=False,
    )
    print(json.dumps({
        "split": report["split"], "metrics": report["metrics"],
        "population_weighted": report["population_weighted"],
        "report": args.output, "gold_sha256": report["provenance"]["gold_sha256"],
        "codegen_sha256": report["provenance"]["codegen_sha256"],
        "locked_opened": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
