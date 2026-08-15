"""Print exact P2.4 forensic table cells around requested row ranges."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.devset.p24_authoring_ext import P24ForensicTableLoader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", default="artifacts/store")
    parser.add_argument(
        "--refs", required=True,
        help="comma-separated report_id|table_pos or report_id|table_pos|start|end",
    )
    args = parser.parse_args()
    loader = P24ForensicTableLoader(args.store_dir)
    for raw in args.refs.split(","):
        parts = raw.strip().split("|")
        if len(parts) not in {2, 4}:
            raise SystemExit(f"invalid ref: {raw}")
        report, pos = parts[0], int(parts[1])
        start, end = (0, 10_000) if len(parts) == 2 else (int(parts[2]), int(parts[3]))
        frame = loader(report, pos)
        frame = frame[(frame.row >= start) & (frame.row <= end)]
        print(json.dumps({"report_id": report, "table_pos": pos}, ensure_ascii=False))
        for row in frame.itertuples():
            print(json.dumps({
                "row": int(row.row), "col": int(row.col), "label": str(row.label),
                "code": str(row.code), "col_name": str(row.col_name),
                "value": float(row.value), "unit_scale": float(row.unit_scale),
            }, ensure_ascii=False))


if __name__ == "__main__":
    main()
