import json
from pathlib import Path

import pandas as pd
import pytest

from vifinqa.devset.p24_silver import (
    _ticker_split,
    build_auto_silver_bundle,
    evaluate_auto_silver,
)


def _three_split_tickers(seed: int) -> dict[str, str]:
    found = {}
    for index in range(1000):
        ticker = f"T{index:03d}"
        split = _ticker_split(ticker, seed)
        found.setdefault(split, ticker)
        if len(found) == 3:
            return found
    raise AssertionError("could not find tickers for all splits")


def _write_store(root: Path, seed: int) -> Path:
    store = root / "store"
    (store / "tables").mkdir(parents=True)
    (store / "cells").mkdir()
    reports, by_ticker_tables, by_ticker_cells = [], {}, {}
    for offset, ticker in enumerate(_three_split_tickers(seed).values(), 1):
        tables, cells = [], []
        for year in (2023, 2024):
            report_id = f"{ticker}_financial_statements_{year}_consolidated"
            reports.append({"report_id": report_id, "ticker": ticker,
                            "year": year, "doc_type": "consolidated", "n_tables": 1})
            headers = (["Chỉ tiêu", "31/12/2023"] if year == 2023 else
                       ["Chỉ tiêu", "31/12/2023", "31/12/2024"])
            tables.append({
                "report_id": report_id, "ticker": ticker, "year": year,
                "doc_type": "consolidated", "table_pos": 0, "line_no": 1,
                "grid_json": json.dumps([headers, ["Doanh thu thuần", "100"]]),
                "unit_scale": 1_000_000.0, "unit_source": "explicit", "context": "",
            })
            cells.append({
                "report_id": report_id, "table_pos": 0, "row": 1, "col": 1,
                "label": "Doanh thu thuần", "row_code": "10",
                "col_name": "31/12/2023", "value": float(100 + offset),
            })
        by_ticker_tables[ticker] = tables
        by_ticker_cells[ticker] = cells
    pd.DataFrame(reports).to_parquet(store / "reports.parquet", index=False)
    for ticker, tables in by_ticker_tables.items():
        pd.DataFrame(tables).to_parquet(store / "tables" / f"{ticker}.parquet", index=False)
        pd.DataFrame(by_ticker_cells[ticker]).to_parquet(
            store / "cells" / f"{ticker}.parquet", index=False,
        )
    return store


def test_auto_silver_build_is_ticker_disjoint_and_immutable(tmp_path):
    seed = 2453
    store = _write_store(tmp_path, seed)
    out = tmp_path / "silver"
    manifest = build_auto_silver_bundle(
        store, out, seed=seed, max_per_report_pair=2,
    )
    assert manifest["counts"]["total"] == 3
    ticker_sets = [set(manifest["files"][split]["tickers"])
                   for split in ("train", "tune", "locked")]
    assert all(len(values) == 1 for values in ticker_sets)
    assert not (ticker_sets[0] & ticker_sets[1] | ticker_sets[0] & ticker_sets[2]
                | ticker_sets[1] & ticker_sets[2])
    with pytest.raises(FileExistsError):
        build_auto_silver_bundle(store, out, seed=seed, max_per_report_pair=2)


def test_auto_silver_evaluator_replays_without_gold_value_access(tmp_path):
    seed = 2453
    store = _write_store(tmp_path, seed)
    out = tmp_path / "silver"
    manifest = build_auto_silver_bundle(
        store, out, seed=seed, max_per_report_pair=2,
    )
    split = out / "p24_silver_tune.jsonl"
    report = evaluate_auto_silver(
        split, store, out / "eval_tune.json",
        expected_split_sha256=manifest["files"]["tune"]["sha256"],
    )
    assert report["metrics"] == {
        "count": 1,
        "coverage": 1.0,
        "cell_accuracy": 1.0,
        "answer_accuracy": 1.0,
        "accepted_answer_precision": 1.0,
    }
