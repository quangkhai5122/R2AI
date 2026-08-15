"""Strictly validate completed P2.4 tune gold with the forensic loader.

The ordinary store serializer intentionally omits numeric OCR cells mistaken
for note codes.  P2.4 exact authoring retains those cells, so completed tune
gold must use the isolated forensic loader for both build and independent
validation.  This script has no path to the locked split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.devset.p24 import canonical_sha256, validate_gold_records  # noqa: E402
from vifinqa.devset.p24_authoring_ext import P24ForensicTableLoader  # noqa: E402
from vifinqa.utils.io import read_jsonl, setup_stdout  # noqa: E402


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="artifacts/devset_p24/p24_tune_gold.verified.jsonl")
    parser.add_argument("--questions", default="artifacts/devset_p24/p24_tune_questions.jsonl")
    parser.add_argument("--store-dir", default="artifacts/store")
    args = parser.parse_args()
    gold, questions = read_jsonl(args.gold), read_jsonl(args.questions)
    summary = validate_gold_records(
        gold, questions, "tune",
        table_loader=P24ForensicTableLoader(args.store_dir),
        require_complete=True,
    )
    print(json.dumps({
        **summary,
        "gold_sha256": canonical_sha256(gold),
        "question_count": len(questions),
        "loader": "p24_forensic_exact_cell_v1",
        "locked_opened": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
