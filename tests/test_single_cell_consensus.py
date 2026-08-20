import json
from pathlib import Path

import pandas as pd

from vifinqa.codegen.single_cell_consensus import (
    _direct_exact_kind,
    _strict_metric_label_match,
    discover_single_cell_consensus,
    resolve_single_fact_candidates,
)
from vifinqa.codegen.semantic_repair import _StoreView


def _write_store(root: Path, *, conflict: bool = False) -> Path:
    store = root / "store"
    (store / "tables").mkdir(parents=True)
    (store / "cells").mkdir()
    reports = [
        {
            "report_id": "ABC_financial_statements_2023_consolidated",
            "ticker": "ABC",
            "year": 2023,
            "doc_type": "consolidated",
            "n_tables": 2,
        },
        {
            "report_id": "ABC_financial_statements_2024_consolidated",
            "ticker": "ABC",
            "year": 2024,
            "doc_type": "consolidated",
            "n_tables": 2,
        },
    ]
    pd.DataFrame(reports).to_parquet(store / "reports.parquet", index=False)
    tables, cells = [], []
    values = [(0, 100.0)] + ([(1, 200.0)] if conflict else [])
    for table_pos, value in values:
        tables.append(
            {
                "report_id": reports[0]["report_id"],
                "ticker": "ABC",
                "year": 2023,
                "doc_type": "consolidated",
                "table_pos": table_pos,
                "line_no": table_pos + 1,
                "grid_json": json.dumps(
                    [
                        ["Chỉ tiêu", "Thuyết minh", "31/12/2023"],
                        ["Doanh thu thuần", "10", str(value)],
                    ]
                ),
                "unit_scale": 1.0,
                "unit_source": "explicit",
                "context": "",
            }
        )
        cells.extend(
            [
                {
                    "report_id": reports[0]["report_id"],
                    "table_pos": table_pos,
                    "row": 1,
                    "col": 1,
                    "label": "Doanh thu thuần",
                    "row_code": "10",
                    "col_name": "Thuyết minh",
                    "value": 10.0,
                },
                {
                    "report_id": reports[0]["report_id"],
                    "table_pos": table_pos,
                    "row": 1,
                    "col": 2,
                    "label": "Doanh thu thuần",
                    "row_code": "10",
                    "col_name": "31/12/2023",
                    "value": value,
                },
            ]
        )
        tables.append(
            {
                "report_id": reports[1]["report_id"],
                "ticker": "ABC",
                "year": 2024,
                "doc_type": "consolidated",
                "table_pos": table_pos,
                "line_no": table_pos + 1,
                "grid_json": json.dumps(
                    [
                        ["Chỉ tiêu", "31/12/2023", "31/12/2024"],
                        ["Doanh thu thuần", str(value), str(value + 10)],
                    ]
                ),
                "unit_scale": 1.0,
                "unit_source": "explicit",
                "context": "",
            }
        )
        cells.append(
            {
                "report_id": reports[1]["report_id"],
                "table_pos": table_pos,
                "row": 1,
                "col": 1,
                "label": "Doanh thu thuần",
                "row_code": "10",
                "col_name": "31/12/2023",
                "value": value,
            }
        )
    pd.DataFrame(tables).to_parquet(store / "tables" / "ABC.parquet", index=False)
    pd.DataFrame(cells).to_parquet(store / "cells" / "ABC.parquet", index=False)
    return store


def _route(qid: int) -> dict:
    question = f"Doanh thu thuần ABC năm 2023 là bao nhiêu? {qid}"
    return {
        "id": qid,
        "question": question,
        "route": {
            "question": question,
            "tickers": ["ABC"],
            "years": [2023],
            "doc_type": "consolidated",
            "output_type": "number",
            "unit_scale": 1.0,
            "unit_name": "đồng",
            "metric_norm": "doanh thu thuần",
            "report_ids": ["ABC_financial_statements_2023_consolidated"],
            "plan": {
                "op": "lookup",
                "facts": [
                    {
                        "ticker": "ABC",
                        "year": 2023,
                        "doc_type": "consolidated",
                        "metric": "doanh thu thuần",
                        "role": "value",
                    }
                ],
            },
        },
    }


def test_resolver_accepts_unique_independently_supported_cell(tmp_path):
    store = _write_store(tmp_path)
    route = _route(1)["route"]
    candidates, reason, _detail = resolve_single_fact_candidates(
        view=_StoreView(store),
        route=route,
        fact=route["plan"]["facts"][0],
    )
    assert reason == "accepted"
    assert candidates[0].answer == 100.0
    assert (candidates[0].row, candidates[0].col) == (1, 2)
    assert len(candidates[0].supports) == 1


def test_resolver_rejects_competing_supported_value_clusters(tmp_path):
    store = _write_store(tmp_path, conflict=True)
    route = _route(1)["route"]
    candidates, reason, detail = resolve_single_fact_candidates(
        view=_StoreView(store),
        route=route,
        fact=route["plan"]["facts"][0],
    )
    assert candidates == []
    assert reason == "competing_answer_clusters"
    assert set(detail["clusters"]) == {"100.00", "200.00"}


def test_repair_and_rescue_modes_are_disjoint_and_fail_closed(tmp_path):
    store = _write_store(tmp_path)
    retrieval = [_route(1), _route(2), _route(3)]
    query = (
        "round(float(df1.loc[df1['label'].str.contains('Doanh thu thuần', "
        "case=False, regex=False, na=False) & (df1['col'] == 1), "
        "'value'].iloc[0]) * 1 / 1, 2)"
    )
    repair = {
        "id": 1,
        "question": retrieval[0]["question"],
        "status": "ok",
        "source": "rule",
        "answer": 10.0,
        "pandas_query": query,
        "used_vars": [
            {
                "var": "df1",
                "report_id": "ABC_financial_statements_2023_consolidated",
                "table_pos": 0,
            }
        ],
        "run_signature": "sig",
    }
    rescue = {
        "id": 2,
        "question": retrieval[1]["question"],
        "status": "failed",
        "source": "none",
        "answer": 0.0,
        "pandas_query": "0.0",
        "used_vars": [],
        "run_signature": "sig",
    }
    complex_none = dict(rescue, id=3, question=retrieval[2]["question"])
    retrieval[2]["route"]["plan"]["facts"].append(
        dict(
            retrieval[2]["route"]["plan"]["facts"][0],
            year=2022,
        )
    )

    repairs, repair_audit = discover_single_cell_consensus(
        [repair, rescue, complex_none],
        retrieval,
        store,
        mode="repair",
    )
    rescues, rescue_audit = discover_single_cell_consensus(
        [repair, rescue, complex_none],
        retrieval,
        store,
        mode="rescue",
    )
    assert [(p.qid, p.answer) for p in repairs] == [(1, 100.0)]
    assert [(p.qid, p.answer) for p in rescues] == [(2, 100.0)]
    assert repair_audit["target_ids"] == [1]
    assert rescue_audit["target_ids"] == [2, 3]
    assert rescue_audit["counts"]["not_single_fact"] == 1


def test_metric_identity_gate_rejects_historical_cost_vs_cost_of_services():
    assert _strict_metric_label_match("loi nhuan gop", "Lợi nhuận gộp")
    assert not _strict_metric_label_match(
        "nguyen gia bat dong san dau tu",
        "Giá vốn cho thuê bất động sản đầu tư và cung cấp dịch vụ",
    )


def test_direct_exact_aliases_are_narrow():
    assert (
        _direct_exact_kind(
            metric="ngoai te usd",
            label='- Đô la Mỹ ("USD")',
            col_name="Số cuối năm",
            context="Ngoại tệ các loại",
            output_type="number",
        )
        == "foreign_currency_usd"
    )
    assert (
        _direct_exact_kind(
            metric="so huu cen vinh phuc",
            label="Công ty Cổ phần Cen Vĩnh Phúc",
            col_name="Tỷ lệ lợi ích",
            context="",
            output_type="percent",
        )
        == "ownership_percentage"
    )
    assert not _direct_exact_kind(
        metric="gia goc no xau",
        label="- Nguyên giá",
        col_name="31/12/2025",
        context="Bảng cân đối kế toán",
        output_type="number",
    )
