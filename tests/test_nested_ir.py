import io

import pandas as pd

from vifinqa.codegen.executor import run_code
from vifinqa.codegen.generate import _typed_ir_result
from vifinqa.codegen.nested_ir import try_nested_formula_ir


def _row(label, value, code="", col=1, col_name="2024", unit=1e9, row=1):
    return {"row": row, "label": label, "code": code, "col": col,
            "col_name": col_name, "value": value, "unit_scale": unit}


def _table(var, year, rows):
    return {
        "var": var,
        "report_id": f"AAA_financial_statements_{year}_consolidated",
        "table_pos": 1, "report_year": year,
        "csv_text": pd.DataFrame(rows).to_csv(index=False),
    }


def test_nested_ir_selects_year_by_de_and_projects_interest_coverage():
    tables = [
        _table("df1", 2023, [
            _row("Nợ phải trả", 80, col_name="2023", row=1),
            _row("Vốn chủ sở hữu", 100, col_name="2023", row=2),
            _row("Lợi nhuận trước thuế", 30, col_name="2023", row=3),
            _row("Chi phí lãi vay", 10, col_name="2023", row=4),
        ]),
        _table("df2", 2024, [
            _row("Nợ phải trả", 180, row=1),
            _row("Vốn chủ sở hữu", 100, row=2),
            _row("Lợi nhuận trước thuế", 20, row=3),
            _row("Chi phí lãi vay", 10, row=4),
        ]),
    ]
    route = {
        "question": ("Trong giai đoạn 2023-2024, vào năm AAA có tỷ số D/E "
                     "cao nhất, hệ số khả năng thanh toán lãi vay là bao nhiêu lần?"),
        "tickers": ["AAA"], "years": [2023, 2024],
        "doc_type": "consolidated", "output_type": "ratio",
        "unit_scale": 1.0, "metric_variants": ["d/e"],
        "plan": {"op": "ranking", "facts": []},
    }
    answer = try_nested_formula_ir(route, tables)
    assert answer.ok, answer.detail
    assert answer.answer == 3.0
    assert "selector=debt_equity" in answer.detail
    assert "target=interest_coverage" in answer.detail
    assert "df1" in answer.pandas_query and "df2" in answer.pandas_query
    dfs = {t["var"]: pd.read_csv(io.StringIO(t["csv_text"])) for t in tables}
    replay = run_code(answer.pandas_query, dfs)
    assert replay["status"] == "ok"
    assert replay["value"] == 3.0


def test_typed_ir_fill_pipeline_accepts_nested_selector_after_replay():
    tables = [
        _table("df1", 2023, [
            _row("Nợ phải trả", 80, col_name="2023", row=1),
            _row("Vốn chủ sở hữu", 100, col_name="2023", row=2),
            _row("Lợi nhuận trước thuế", 30, col_name="2023", row=3),
            _row("Chi phí lãi vay", 10, col_name="2023", row=4),
        ]),
        _table("df2", 2024, [
            _row("Nợ phải trả", 180, row=1),
            _row("Vốn chủ sở hữu", 100, row=2),
            _row("Lợi nhuận trước thuế", 20, row=3),
            _row("Chi phí lãi vay", 10, row=4),
        ]),
    ]
    route = {
        "question": ("Trong giai đoạn 2023-2024, vào năm AAA có tỷ số D/E "
                     "cao nhất, hệ số khả năng thanh toán lãi vay là bao nhiêu lần?"),
        "tickers": ["AAA"], "years": [2023, 2024],
        "doc_type": "consolidated", "output_type": "ratio", "unit_scale": 1.0,
        "metric_variants": ["d/e"], "plan": {"op": "ranking", "facts": []},
    }
    dfs = {t["var"]: pd.read_csv(io.StringIO(t["csv_text"])) for t in tables}

    class Bundle:
        id = 1
        question = route["question"]
        run_signature = "test"

        def used_vars(self, code):
            return [t for t in tables if t["var"] in code]

    bundle = Bundle()
    bundle.route = route
    bundle.tables = tables
    bundle.dfs = dfs

    result = _typed_ir_result(bundle)
    assert result is not None
    assert result["source"] == "typed_ir_nested"
    assert result["answer"] == 3.0
    assert result["semantic"]["ok"]
    assert {t["var"] for t in result["used_vars"]} == {"df1", "df2"}
