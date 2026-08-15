"""Find standardized statement rows by exact row code for P2.4 authoring."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.extraction.build_store import Store  # noqa: E402


def canon_code(value: object) -> str:
    text = str(value or "").strip().lstrip("0")
    if text.endswith(".0"):
        text = text[:-2]
    return text or "0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", default="artifacts/store")
    parser.add_argument("--tickers", required=True)
    parser.add_argument("--years", required=True)
    parser.add_argument("--doc-type", default="consolidated")
    parser.add_argument("--codes", required=True)
    args = parser.parse_args()
    wanted_codes = {canon_code(x) for x in args.codes.split(",")}
    wanted_years = {int(x) for x in args.years.split(",")}
    store = Store(Path(args.store_dir))
    for ticker in [x.strip().upper() for x in args.tickers.split(",") if x.strip()]:
        cells = store.cells_of(ticker)
        cells = cells[(cells.doc_type == args.doc_type) & cells.year.astype(int).isin(wanted_years)]
        hits = cells[cells.row_code.map(canon_code).isin(wanted_codes)]
        for row in hits.sort_values(["year", "report_id", "table_pos", "row", "col"]).itertuples():
            print(json.dumps({
                "ticker": ticker, "year": int(row.year), "report_id": row.report_id,
                "table_pos": int(row.table_pos), "row": int(row.row), "col": int(row.col),
                "label": row.label, "code": row.row_code, "col_name": row.col_name,
                "value": float(row.value), "unit_scale": float(row.unit_scale),
            }, ensure_ascii=False))


if __name__ == "__main__":
    main()
