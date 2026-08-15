"""Print strict exact standard-statement metric candidates for P2.4 review."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.devset.p24_metrics import METRICS  # noqa: E402
from vifinqa.devset.p24_metrics_v2 import StandardMetricResolverV2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", default="artifacts/store")
    parser.add_argument("--tickers", required=True)
    parser.add_argument("--years", required=True)
    parser.add_argument("--doc-type", default="consolidated")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    resolver = StandardMetricResolverV2(args.store_dir)
    for ticker in args.tickers.split(","):
        for year in [int(x) for x in args.years.split(",")]:
            for metric in args.metrics.split(","):
                if metric not in METRICS:
                    raise SystemExit(f"unknown metric {metric}")
                print(json.dumps({"ticker": ticker, "year": year, "metric": metric,
                    "hits": resolver.candidates(ticker, year, args.doc_type, metric)[:args.limit]},
                    ensure_ascii=False))


if __name__ == "__main__":
    main()
