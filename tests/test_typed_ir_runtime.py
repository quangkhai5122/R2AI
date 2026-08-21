import io

import pandas as pd

from vifinqa.codegen.executor import run_code
from vifinqa.codegen.typed_ir import try_typed_ir_answer


def _table(var, ticker, value):
    frame = pd.DataFrame([{
        "row": 1, "label": "Doanh thu thuần", "code": "10",
        "col": 1, "col_name": "2024", "value": value,
        "unit_scale": 1e9,
    }])
    return {
        "var": var,
        "report_id": f"{ticker}_financial_statements_2024_consolidated",
        "table_pos": 1, "report_year": 2024,
        "csv_text": frame.to_csv(index=False),
    }


def test_runtime_resolve_compile_and_replay_difference():
    tables = [_table("df1", "AAA", 120.0), _table("df2", "BBB", 100.0)]
    route = {
        "question": "Chênh lệch doanh thu thuần AAA so với BBB năm 2024 là bao nhiêu tỷ đồng?",
        "tickers": ["AAA", "BBB"], "years": [2024],
        "doc_type": "consolidated", "output_type": "number",
        "unit_scale": 1e9, "metric_variants": ["doanh thu thuan"],
        "plan": {"op": "difference", "facts": [
            {"ticker": "AAA", "year": 2024, "doc_type": "consolidated",
             "metric": "doanh thu thuan", "role": "value"},
            {"ticker": "BBB", "year": 2024, "doc_type": "consolidated",
             "metric": "doanh thu thuan", "role": "value"},
        ]},
    }
    answer = try_typed_ir_answer(route, tables)
    assert answer.ok, answer.detail
    assert answer.answer == 20.0
    dfs = {t["var"]: pd.read_csv(io.StringIO(t["csv_text"])) for t in tables}
    replay = run_code(answer.pandas_query, dfs)
    assert replay["status"] == "ok"
    assert replay["value"] == 20.0
