import io

import pandas as pd

from vifinqa.codegen.direct_ranking_ir import try_direct_ranking_ir
from vifinqa.codegen.executor import run_code
from vifinqa.codegen.generate import _typed_ir_result
from vifinqa.codegen.generate import _rule_result


def _table(var, year, value):
    frame = pd.DataFrame([{
        "row": 1, "label": "Doanh thu thuần", "code": "10",
        "col": 1, "col_name": str(year), "value": value,
        "unit_scale": 1e9,
    }])
    return {
        "var": var,
        "report_id": f"AAA_financial_statements_{year}_consolidated",
        "table_pos": 1, "report_year": year,
        "csv_text": frame.to_csv(index=False),
    }


def _route(output_type):
    return {
        "question": "Năm nào doanh thu thuần cao nhất trong 2021, 2022, 2023?",
        "tickers": ["AAA"], "years": [2021, 2022, 2023],
        "doc_type": "consolidated", "output_type": output_type,
        "unit_scale": 1e9, "metric_norm": "doanh thu thuan",
        "metric_variants": ["doanh thu thuan"],
        "plan": {"op": "ranking", "facts": [
            {"ticker": "AAA", "year": year, "doc_type": "consolidated",
             "metric": "doanh thu thuan", "role": "value"}
            for year in [2021, 2022, 2023]
        ]},
    }


def test_direct_ranking_projects_year_and_replays_all_evidence():
    tables = [_table("df1", 2021, 100), _table("df2", 2022, 130),
              _table("df3", 2023, 120)]
    answer = try_direct_ranking_ir(_route("year"), tables)
    assert answer.ok, answer.detail
    assert answer.answer == 2022.0
    assert all(var in answer.pandas_query for var in ("df1", "df2", "df3"))
    dfs = {t["var"]: pd.read_csv(io.StringIO(t["csv_text"])) for t in tables}
    assert run_code(answer.pandas_query, dfs)["value"] == 2022.0


def test_direct_ranking_projects_value():
    tables = [_table("df1", 2021, 100), _table("df2", 2022, 130),
              _table("df3", 2023, 120)]
    route = _route("number")
    route["question"] = "Doanh thu thuần cao nhất là bao nhiêu tỷ đồng?"
    answer = try_direct_ranking_ir(route, tables)
    assert answer.ok, answer.detail
    assert answer.answer == 130.0


def test_typed_ir_fill_pipeline_accepts_direct_ranking_after_replay():
    tables = [_table("df1", 2021, 100), _table("df2", 2022, 130),
              _table("df3", 2023, 120)]
    route = _route("year")
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
    assert result["source"] == "typed_ir_ranking"
    assert result["answer"] == 2022.0
    assert result["semantic"]["ok"]
    assert {t["var"] for t in result["used_vars"]} == {"df1", "df2", "df3"}


def test_direct_ranking_rejects_selector_target_and_ratio_questions():
    route = _route("year")
    route["question"] = (
        "Trong các công ty, công ty có số lượng cổ phiếu phổ thông cao nhất "
        "có tổng tài sản thuế thu nhập hoãn lại là bao nhiêu?"
    )
    assert not try_direct_ranking_ir(route, []).ok

    route = _route("year")
    route["question"] = (
        "Năm nào có tỷ trọng giá vốn cho thuê dài hạn trên tổng giá vốn "
        "cao nhất?"
    )
    assert not try_direct_ranking_ir(route, []).ok


def test_single_fact_composite_is_recovered_as_lookup():
    tables = [_table("df1", 2021, 100)]
    route = _route("number")
    route["question"] = "Doanh thu thuần cao nhất của AAA năm 2021 là bao nhiêu tỷ đồng?"
    route["tickers"] = ["AAA"]
    route["years"] = [2021]
    route["plan"] = {"op": "ranking", "facts": [{
        "ticker": "AAA", "year": 2021, "doc_type": "consolidated",
        "metric": "doanh thu thuan", "role": "value",
    }]}
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
    result = _rule_result(bundle)
    assert result is not None
    assert result["source"] == "rule_single_fact"
    assert result["answer"] == 100.0
    assert result["semantic"]["ok"]
