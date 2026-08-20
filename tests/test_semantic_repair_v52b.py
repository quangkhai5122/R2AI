from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vifinqa.codegen.semantic_repair_v52b import (
    _compute_answer,
    _supported_plan,
    build_multi_operand_repair_overlay,
    parse_multi_operand_expression,
)
from vifinqa.utils.io import read_json, read_jsonl


def _leaf(var: str, label: str, col: int) -> str:
    return (
        f"float({var}.loc[{var}['label'].str.contains('{label}', "
        f"case=False, regex=False, na=False) & ({var}['col'] == {col}), "
        f"'value'].iloc[0])"
    )


def test_parser_accepts_only_matching_simple_operation_shapes():
    a = _leaf("df1", "Doanh thu", 2)
    b = _leaf("df2", "Doanh thu", 2)
    difference = f"round(({a} * 1 - {b} * 1) / 1000, 2)"
    ratio = f"round({a} * 1 / {b} * 1 * 100, 2)"
    growth = f"round(({a} - {b}) / abs({b}) * 100, 2)"
    average = f"round(({a} + {b}) / 2 / 1000, 2)"

    assert parse_multi_operand_expression(difference, "difference", 2)
    assert parse_multi_operand_expression(ratio, "ratio", 2)
    assert parse_multi_operand_expression(growth, "growth_pct", 2)
    assert parse_multi_operand_expression(average, "average", 2)
    assert parse_multi_operand_expression(difference, "ratio", 2) is None
    assert parse_multi_operand_expression(
        f"round(max({a}, {b}), 2)", "difference", 2,
    ) is None


def test_growth_facts_are_proven_same_series_and_ordered_end_base():
    route = {"output_type": "percent"}
    facts = [
        {"ticker": "AAA", "year": 2023, "doc_type": "consolidated",
         "metric": "doanh thu", "role": "value"},
        {"ticker": "AAA", "year": 2024, "doc_type": "consolidated",
         "metric": "doanh thu", "role": "value"},
    ]
    ok, ordered, reason = _supported_plan(
        "growth_pct", route, facts,
        "Tỷ lệ tăng trưởng doanh thu từ năm 2023 đến 2024 là bao nhiêu %?",
    )
    assert ok and reason == "supported"
    assert [fact["year"] for fact in ordered] == [2024, 2023]

    broken = [dict(facts[0]), {**facts[1], "ticker": "BBB"}]
    ok, _ordered, reason = _supported_plan(
        "growth_pct", route, broken,
        "Tỷ lệ tăng trưởng doanh thu từ năm 2023 đến 2024 là bao nhiêu %?",
    )
    assert not ok and reason == "growth_series_invalid"


def test_operation_math_uses_registry_conventions():
    assert _compute_answer("difference", [300, 200], 1) == 100
    assert _compute_answer("growth_pct", [80, 100], 1) == -20
    assert _compute_answer("ratio", [-40, 100], 1) == -40
    assert _compute_answer("average", [1000, 3000], 1000) == 2


def _table(report_id: str, ticker: str, year: int, pos: int,
           period_header: str) -> dict:
    grid = [["Chỉ tiêu", "Thuyết minh", period_header],
            ["Doanh thu thuần", "10", "value"]]
    return {
        "report_id": report_id,
        "ticker": ticker,
        "year": year,
        "doc_type": "consolidated",
        "table_pos": pos,
        "line_no": pos * 10,
        "page": pos,
        "n_rows": 2,
        "n_cols": 3,
        "unit_scale": 1.0,
        "unit_source": "header",
        "context": "Đơn vị tính: VND",
        "grid_json": json.dumps(grid, ensure_ascii=False),
    }


def _cell(report_id: str, ticker: str, year: int, pos: int,
          col: int, col_name: str, value: float) -> dict:
    return {
        "report_id": report_id,
        "ticker": ticker,
        "year": year,
        "doc_type": "consolidated",
        "table_pos": pos,
        "page": pos,
        "row": 1,
        "col": col,
        "label": "Doanh thu thuần",
        "row_code": "10",
        "col_name": col_name,
        "value": value,
        "unit_scale": 1.0,
        "unit_known": True,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, signed_support: bool = True):
    store = tmp_path / "store"
    (store / "tables").mkdir(parents=True)
    (store / "cells").mkdir()
    reports = []
    for ticker in ("AAA", "BBB"):
        for year in (2024, 2025):
            reports.append({
                "report_id": f"{ticker}_financial_statements_{year}_consolidated",
                "ticker": ticker,
                "year": year,
                "doc_type": "consolidated",
                "n_tables": 1,
            })
    pd.DataFrame(reports).to_parquet(store / "reports.parquet", index=False)

    values = {"AAA": 300.0, "BBB": 200.0}
    for ticker in ("AAA", "BBB"):
        current = f"{ticker}_financial_statements_2024_consolidated"
        next_report = f"{ticker}_financial_statements_2025_consolidated"
        support_value = values[ticker]
        if ticker == "BBB" and not signed_support:
            support_value = -support_value
        tables = [
            _table(current, ticker, 2024, 1, "Năm nay"),
            _table(next_report, ticker, 2025, 2, "Năm trước"),
        ]
        cells = [
            _cell(current, ticker, 2024, 1, 1, "Thuyết minh", 10),
            _cell(current, ticker, 2024, 1, 2, "Năm nay", values[ticker]),
            _cell(next_report, ticker, 2025, 2, 2, "Năm trước", support_value),
        ]
        pd.DataFrame(tables).to_parquet(
            store / "tables" / f"{ticker}.parquet", index=False,
        )
        pd.DataFrame(cells).to_parquet(
            store / "cells" / f"{ticker}.parquet", index=False,
        )

    question = "Chênh lệch doanh thu năm 2024 giữa AAA và BBB là bao nhiêu đồng?"
    query = (
        f"round(({_leaf('df1', 'Doanh thu thuần', 1)} * 1 - "
        f"{_leaf('df2', 'Doanh thu thuần', 1)} * 1) / 1, 2)"
    )
    primary = [{
        "id": 1,
        "question": question,
        "answer": 0.0,
        "pandas_query": query,
        "used_vars": [
            {"var": "df1", "report_id":
             "AAA_financial_statements_2024_consolidated", "table_pos": 1},
            {"var": "df2", "report_id":
             "BBB_financial_statements_2024_consolidated", "table_pos": 1},
        ],
        "status": "ok",
        "source": "rule_composite",
        "run_signature": "v52a-control",
        "detail": "",
    }]
    facts = [
        {"ticker": "AAA", "year": 2024, "doc_type": "consolidated",
         "metric": "doanh thu thuan", "role": "value"},
        {"ticker": "BBB", "year": 2024, "doc_type": "consolidated",
         "metric": "doanh thu thuan", "role": "value"},
    ]
    retrieval = [{
        "id": 1,
        "question": question,
        "route": {
            "question": question,
            "output_type": "number",
            "unit_scale": 1,
            "unit_name": "đồng",
            "plan": {"op": "difference", "facts": facts},
        },
    }]
    primary_path = tmp_path / "primary.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    _write_jsonl(primary_path, primary)
    _write_jsonl(retrieval_path, retrieval)
    return primary_path, retrieval_path, store


def test_overlay_repairs_all_operands_only_with_independent_signed_support(tmp_path):
    primary, retrieval, store = _fixture(tmp_path)
    out = tmp_path / "overlay.jsonl"
    audit_path = tmp_path / "overlay.audit.json"
    audit = build_multi_operand_repair_overlay(
        primary,
        retrieval,
        store,
        out,
        audit_path,
        expected_selected_ids={1},
        expected_primary_signature="v52a-control",
    )
    row = read_jsonl(out)[0]
    assert row["answer"] == 100.0
    assert row["source"] == "deterministic_v52b"
    assert "['col'] == 2" in row["pandas_query"]
    operands = row["semantic_repair_v52b_provenance"]["operands"]
    assert [item["silver_support_count"] for item in operands] == [1, 1]
    assert audit["selected_ids"] == [1]
    assert read_json(audit_path)["output"]["sha256"] == audit["output"]["sha256"]


def test_overlay_rejects_when_one_operand_has_only_opposite_sign_support(tmp_path):
    primary, retrieval, store = _fixture(tmp_path, signed_support=False)
    out = tmp_path / "rejected.jsonl"
    with pytest.raises(ValueError, match="selected id guard mismatch"):
        build_multi_operand_repair_overlay(
            primary,
            retrieval,
            store,
            out,
            expected_selected_ids={1},
            expected_primary_signature="v52a-control",
        )
    assert not out.exists()


def test_overlay_refuses_overwrite_before_discovery(tmp_path):
    primary, retrieval, store = _fixture(tmp_path)
    out = tmp_path / "exists.jsonl"
    out.write_text("sentinel", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_multi_operand_repair_overlay(
            primary,
            retrieval,
            store,
            out,
            expected_selected_ids={1},
            expected_primary_signature="v52a-control",
        )
    assert out.read_text(encoding="utf-8") == "sentinel"
