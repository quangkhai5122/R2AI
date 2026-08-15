"""Batch read-only fuzzy cell finder for P2.4 forensic authoring."""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.extraction.build_store import Store  # noqa: E402
from vifinqa.utils.viet_text import tokens  # noqa: E402


def norm(value: object) -> str:
    return " ".join(tokens(unicodedata.normalize("NFC", str(value or "")).lower()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", default="artifacts/store")
    parser.add_argument("--tickers", required=True)
    parser.add_argument("--years", required=True)
    parser.add_argument("--doc-type", default="consolidated")
    parser.add_argument("--queries", required=True, help="queries separated by ||")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    store = Store(Path(args.store_dir))
    years = [int(x) for x in args.years.split(",")]
    queries = [x.strip() for x in args.queries.split("||") if x.strip()]
    for ticker in [x.strip().upper() for x in args.tickers.split(",") if x.strip()]:
        cells = store.cells_of(ticker).copy()
        cells = cells[cells.doc_type == args.doc_type]
        for year in years:
            pool = cells[cells.year.astype(int) == year]
            for query_raw in queries:
                query = norm(query_raw)
                terms = set(query.split())
                scored = []
                for row in pool.itertuples():
                    label = norm(row.label)
                    overlap = len(terms & set(label.split())) / max(1, len(terms))
                    fuzzy = difflib.SequenceMatcher(None, query, label).ratio()
                    scored.append((0.65 * overlap + 0.35 * fuzzy, row))
                hits = []
                for score, row in sorted(scored, key=lambda item: item[0], reverse=True)[:args.limit]:
                    hits.append({
                        "score": round(score, 4), "report_id": row.report_id,
                        "table_pos": int(row.table_pos), "row": int(row.row),
                        "col": int(row.col), "label": row.label,
                        "code": row.row_code, "col_name": row.col_name,
                        "value": float(row.value), "unit_scale": float(row.unit_scale),
                    })
                print(json.dumps({
                    "ticker": ticker, "year": year, "query": query_raw, "hits": hits,
                }, ensure_ascii=False))


if __name__ == "__main__":
    main()
