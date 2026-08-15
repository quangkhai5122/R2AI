"""Read-only exact-cell finder for P2.4 tune authoring."""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.extraction.build_store import Store  # noqa: E402
from vifinqa.utils.viet_text import tokens  # noqa: E402


def norm(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).lower()
    return " ".join(tokens(text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", default="artifacts/store")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--years", default="")
    parser.add_argument("--doc-type", default="")
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    store = Store(Path(args.store_dir))
    cells = store.cells_of(args.ticker.upper()).copy()
    if args.years:
        years = {int(item) for item in args.years.split(",")}
        cells = cells[cells.year.astype(int).isin(years)]
    if args.doc_type:
        cells = cells[cells.doc_type == args.doc_type]
    query = norm(args.query)
    query_terms = set(query.split())
    scored = []
    for row in cells.itertuples():
        label = norm(row.label)
        overlap = len(query_terms & set(label.split())) / max(1, len(query_terms))
        fuzzy = difflib.SequenceMatcher(None, query, label).ratio()
        score = 0.65 * overlap + 0.35 * fuzzy
        scored.append((score, row))
    for score, row in sorted(scored, key=lambda item: item[0], reverse=True)[:args.limit]:
        print(json.dumps({
            "score": round(score, 4), "report_id": row.report_id,
            "table_pos": int(row.table_pos), "row": int(row.row), "col": int(row.col),
            "label": row.label, "code": row.row_code, "col_name": row.col_name,
            "value": float(row.value), "unit_scale": float(row.unit_scale),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
