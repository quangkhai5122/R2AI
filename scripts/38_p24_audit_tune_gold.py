"""Validate and write a descriptive QA audit for completed P2.4 tune gold."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.devset.p24 import validate_gold_records  # noqa: E402
from vifinqa.devset.p24_authoring_ext import P24ForensicTableLoader  # noqa: E402
from vifinqa.devset.p24_gold_audit import build_tune_gold_audit  # noqa: E402
from vifinqa.utils.io import read_jsonl, setup_stdout, write_json  # noqa: E402


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="artifacts/devset_p24/p24_tune_gold.verified.jsonl")
    parser.add_argument("--questions", default="artifacts/devset_p24/p24_tune_questions.jsonl")
    parser.add_argument("--specs", default="artifacts/devset_p24/p24_tune_authoring.final.jsonl")
    parser.add_argument("--store-dir", default="artifacts/store")
    parser.add_argument("--output", default="artifacts/devset_p24/p24_tune_gold.audit.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    gold, questions, specs = read_jsonl(args.gold), read_jsonl(args.questions), read_jsonl(args.specs)
    validate_gold_records(gold, questions, "tune",
        table_loader=P24ForensicTableLoader(args.store_dir), require_complete=True)
    audit = build_tune_gold_audit(gold, questions, specs)
    write_json(output, audit)
    print(json.dumps({key: audit[key] for key in (
        "schema_version", "count", "gold_sha256", "locked_opened",
        "output_type_counts", "provenance", "review_flags",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
