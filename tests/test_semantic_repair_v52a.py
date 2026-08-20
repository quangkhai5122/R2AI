from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vifinqa.codegen.semantic_repair import (
    build_semantic_repair_overlay,
    classify_column_role,
    parse_simple_lookup,
)
from vifinqa.utils.io import read_json, read_jsonl


def _table(report_id: str, pos: int, headers: list[str], rows: list[list[str]],
           *, unit_scale: float = 1.0, unit_source: str = "header",
           context: str = "Đơn vị tính: VND") -> dict:
    grid = [headers, *rows]
    return {
        "report_id": report_id,
        "ticker": "AAA",
        "year": 2024,
        "doc_type": "consolidated",
        "table_pos": pos,
        "line_no": pos * 10,
        "page": pos,
        "n_rows": len(grid),
        "n_cols": len(headers),
        "unit_scale": unit_scale,
        "unit_source": unit_source,
        "context": context,
        "grid_json": json.dumps(grid, ensure_ascii=False),
    }


def _cell(report_id: str, pos: int, row: int, col: int, label: str,
          col_name: str, value: float, code: str, unit_scale: float = 1.0) -> dict:
    return {
        "report_id": report_id,
        "ticker": "AAA",
        "year": 2024,
        "doc_type": "consolidated",
        "table_pos": pos,
        "page": pos,
        "row": row,
        "col": col,
        "label": label,
        "row_code": code,
        "col_name": col_name,
        "value": value,
        "unit_scale": unit_scale,
        "unit_known": True,
    }


def _query(var: str, label: str, col: int, input_scale: float,
           output_scale: float) -> str:
    return (
        f"round(float({var}.loc[{var}['label'].str.contains('{label}', "
        f"case=False, regex=False, na=False) & ({var}['col'] == {col}), "
        f"'value'].iloc[0]) * {input_scale:g} / {output_scale:.1f}, 2)"
    )


def _primary(qid: int, question: str, var: str, pos: int, query: str,
             answer: float) -> dict:
    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "pandas_query": query,
        "used_vars": [{
            "var": var,
            "report_id": "AAA_financial_statements_2024_consolidated",
            "table_pos": pos,
        }],
        "status": "ok",
        "source": "rule",
        "run_signature": "control-signature",
        "detail": "",
    }


def _retrieval(qid: int, question: str, metric: str, unit_scale: float) -> dict:
    return {
        "id": qid,
        "question": question,
        "route": {
            "question": question,
            "tickers": ["AAA"],
            "years": [2024],
            "doc_type": "consolidated",
            "unit_scale": unit_scale,
            "unit_name": "đơn vị",
            "output_type": "number",
            "metric_norm": metric,
            "plan": {
                "op": "lookup",
                "facts": [{
                    "ticker": "AAA",
                    "year": 2024,
                    "doc_type": "consolidated",
                    "metric": metric,
                    "role": "value",
                }],
            },
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    store = tmp_path / "store"
    (store / "tables").mkdir(parents=True)
    (store / "cells").mkdir()
    report_id = "AAA_financial_statements_2024_consolidated"
    pd.DataFrame([{
        "report_id": report_id,
        "ticker": "AAA",
        "year": 2024,
        "doc_type": "consolidated",
        "n_tables": 6,
    }]).to_parquet(store / "reports.parquet", index=False)

    tables = [
        _table(report_id, 1, ["Mã số", "Chỉ tiêu", "Thuyết minh", "Năm nay"],
               [["10", "Doanh thu thuần", "24", "1.000"]]),
        _table(report_id, 2, ["Chỉ tiêu", "Năm nay"],
               [["Doanh thu thuần", "1.000"]]),
        _table(
            report_id, 3, ["Chỉ tiêu", "Số cuối năm"],
            [["Hàng tồn kho", "46.000.000.000"]],
            unit_scale=1e9, unit_source="sticky",
            context="BẢNG CÂN ĐỐI KẾ TOÁN năm 2024 VND",
        ),
        _table(report_id, 4, ["Chỉ tiêu", "Số cuối năm"],
               [["Hàng tồn kho", "46.000.000.000"]]),
        _table(report_id, 5, ["Mã số", "Chỉ tiêu", "Thuyết minh", "Năm nay"],
               [["11", "Giá vốn hàng bán", "25", "(100)"]]),
        _table(report_id, 6, ["Chỉ tiêu", "Năm nay"],
               [["Giá vốn hàng bán", "100"]]),
    ]
    pd.DataFrame(tables).to_parquet(store / "tables" / "AAA.parquet", index=False)

    cells = [
        _cell(report_id, 1, 1, 2, "Doanh thu thuần", "Thuyết minh", 24, "10"),
        _cell(report_id, 1, 1, 3, "Doanh thu thuần", "Năm nay", 1000, "10"),
        _cell(report_id, 2, 1, 1, "Doanh thu thuần", "Năm nay", 1000, ""),
        _cell(report_id, 3, 1, 1, "Hàng tồn kho", "Số cuối năm", 46e9, "140", 1e9),
        _cell(report_id, 4, 1, 1, "Hàng tồn kho", "Số cuối năm", 46e9, "140"),
        _cell(report_id, 5, 1, 2, "Giá vốn hàng bán", "Thuyết minh", 25, "11"),
        _cell(report_id, 5, 1, 3, "Giá vốn hàng bán", "Năm nay", -100, "11"),
        _cell(report_id, 6, 1, 1, "Giá vốn hàng bán", "Năm nay", 100, ""),
    ]
    pd.DataFrame(cells).to_parquet(store / "cells" / "AAA.parquet", index=False)

    questions = [
        "Doanh thu thuần năm 2024 là bao nhiêu trăm đồng?",
        "Hàng tồn kho cuối năm 2024 là bao nhiêu tỷ đồng?",
        "Giá vốn hàng bán năm 2024 là bao nhiêu đồng?",
    ]
    primary = [
        _primary(1, questions[0], "df1", 1, _query("df1", "Doanh thu thuần", 2, 1, 100), 0.24),
        _primary(2, questions[1], "df2", 3, _query("df2", "Hàng tồn kho", 1, 1e9, 1e9), 46e9),
        _primary(3, questions[2], "df3", 5, _query("df3", "Giá vốn hàng bán", 2, 1, 1), 25),
    ]
    retrieval = [
        _retrieval(1, questions[0], "doanh thu thuan", 100),
        _retrieval(2, questions[1], "hang ton kho", 1e9),
        _retrieval(3, questions[2], "gia von hang ban", 1),
    ]
    primary_path = tmp_path / "primary.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    _write_jsonl(primary_path, primary)
    _write_jsonl(retrieval_path, retrieval)
    return primary_path, retrieval_path, store


def test_column_role_maps_target_period_and_excludes_note_code():
    assert classify_column_role("Thuyết minh", 2024, 2024).role == "note"
    assert classify_column_role("STT", 2024, 2024).role == "code"
    assert classify_column_role("Năm nay", 2024, 2024).role == "target_value"
    assert classify_column_role("Năm trước", 2025, 2024).role == "target_value"
    assert classify_column_role("Số đầu năm", 2025, 2024).role == "target_value"
    assert classify_column_role("31/12/2024", 2024, 2024).role == "target_value"
    assert classify_column_role("1/1/2025", 2025, 2024).role == "target_value"
    assert classify_column_role(
        "Năm trước (Trình bày lại - Thuyết minh số 33)", 2025, 2024,
    ).role == "target_value"


def test_simple_lookup_parser_is_fail_closed():
    parsed = parse_simple_lookup(_query("df1", "Doanh thu thuần", 2, 1, 1e9))
    assert parsed is not None
    assert parsed.var == "df1"
    assert parsed.selected_col == 2
    assert parse_simple_lookup("round(float(df1['value'].sum()), 2)") is None


def test_overlay_repairs_note_and_unit_only_with_signed_silver_support(tmp_path):
    primary, retrieval, store = _fixture(tmp_path)
    out = tmp_path / "overlay.jsonl"
    audit_path = tmp_path / "overlay.audit.json"
    audit = build_semantic_repair_overlay(
        primary, retrieval, store, out, audit_path,
        expected_selected_ids={1, 2},
        expected_primary_signature="control-signature",
    )
    rows = {row["id"]: row for row in read_jsonl(out)}

    assert rows[1]["answer"] == 10.0
    assert "['row'] == 1" in rows[1]["pandas_query"]
    assert "['col'] == 3" in rows[1]["pandas_query"]
    assert rows[2]["answer"] == 46.0
    assert "* 1 / 1000000000" in rows[2]["pandas_query"]
    assert rows[1]["source"] == "deterministic_v52a"
    assert rows[2]["semantic_repair_provenance"]["silver_support_count"] == 1

    # Candidate -100 has only opposite-sign +100 evidence and must stay untouched.
    assert rows[3]["answer"] == 25
    assert rows[3]["source"] == "rule"
    assert "semantic_repair_provenance" not in rows[3]
    assert audit["selected_ids"] == [1, 2]
    assert audit["counts"]["unchanged_semantic_rows"] == 1
    assert read_json(audit_path)["output"]["sha256"] == audit["output"]["sha256"]
    assert len({row["run_signature"] for row in rows.values()}) == 1


def test_selected_id_guard_fails_before_writing(tmp_path):
    primary, retrieval, store = _fixture(tmp_path)
    out = tmp_path / "wrong.jsonl"
    with pytest.raises(ValueError, match="selected id guard mismatch"):
        build_semantic_repair_overlay(
            primary, retrieval, store, out,
            expected_selected_ids={1, 2, 3},
            expected_primary_signature="control-signature",
        )
    assert not out.exists()


def test_overlay_refuses_to_overwrite(tmp_path):
    primary, retrieval, store = _fixture(tmp_path)
    out = tmp_path / "exists.jsonl"
    out.write_text("sentinel", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_semantic_repair_overlay(
            primary, retrieval, store, out,
            expected_selected_ids={1, 2},
            expected_primary_signature="control-signature",
        )
    assert out.read_text(encoding="utf-8") == "sentinel"
