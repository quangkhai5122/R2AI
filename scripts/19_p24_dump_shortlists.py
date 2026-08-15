"""Dump reproducible row-level candidates for selected P2.4 tune questions.

This is a read-only forensic helper. It never reads either locked P2.4 file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.codegen.generate import QuestionBundle  # noqa: E402
from vifinqa.extraction.build_store import Store  # noqa: E402
from vifinqa.retrieval.shortlist import render_shortlist  # noqa: E402
from vifinqa.utils.io import read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", default="artifacts/retrieval.jsonl")
    parser.add_argument("--store-dir", default="artifacts/store")
    parser.add_argument("--ids", required=True, help="comma-separated tune IDs")
    parser.add_argument("--table-k", type=int, default=40)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()
    wanted = {int(part) for part in args.ids.split(",") if part.strip()}
    records = {int(row["id"]): row for row in read_jsonl(args.retrieval)}
    store = Store(Path(args.store_dir))
    missing = wanted - set(records)
    if missing:
        raise SystemExit(f"unknown ids: {sorted(missing)}")
    for qid in sorted(wanted):
        rec = records[qid]
        bundle = QuestionBundle(
            rec,
            store,
            k=args.table_k,
            rescue_no_candidates=True,
            rescue_table_k=args.table_k,
            rescue_min_score=args.min_score,
        )
        shortlist = bundle.shortlist(top_n=args.top_n)
        print(json.dumps({
            "id": qid,
            "question": rec["question"],
            "route": rec.get("route", {}),
            "shortlist_trace": bundle.shortlist_trace,
            "shortlist": shortlist,
            "rendered": render_shortlist(shortlist),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
