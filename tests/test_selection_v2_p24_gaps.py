from types import SimpleNamespace

import pandas as pd

from vifinqa.codegen.executor import run_code
from vifinqa.codegen.selection_v2 import compile_program


def _c(i, value):
    return SimpleNamespace(
        var=f"df{i}", row=1, col=1, label="Metric", col_name="2024",
        value=float(value), unit_scale=1.0, score=90.0, rescue=False,
        fact_year=2024, fact_slot=f"F{i}", report_id=f"AAA_{i}",
        table_pos=i, code="",
    )


def _frames(candidates):
    return {c.var: pd.DataFrame([{"row": 1, "col": 1, "value": c.value}])
            for c in candidates}


def _fact(i, as_type="number"):
    return {"ref": i, "as": as_type, "role": "value"}


def test_conditional_average_can_divide_ratio_sum_by_count():
    candidates = [_c(1, 2), _c(2, 4), _c(3, 6), _c(4, 3)]
    program = {
        "schema_version": 2, "output_type": "ratio",
        "facts": {f"x{i}": _fact(i) for i in range(1, 5)},
        "bindings": {
            "r1": {"op": "divide", "args": [{"var": "x1"}, {"var": "x2"}]},
            "r2": {"op": "divide", "args": [{"var": "x3"}, {"var": "x4"}]},
            "keep1": {"op": "gt", "args": [{"var": "r1"},
                                               {"literal": 0, "type": "ratio"}]},
            "keep2": {"op": "gt", "args": [{"var": "r2"},
                                               {"literal": 0, "type": "ratio"}]},
        },
        "root": {"op": "divide", "args": [
            {"op": "sum", "args": [{"var": "r1"}, {"var": "r2"}]},
            {"op": "count_true", "args": [{"var": "keep1"}, {"var": "keep2"}]},
        ]},
    }
    compiled = compile_program(program, candidates,
                               {"output_type": "ratio", "unit_scale": 1}, "q")
    assert run_code(compiled.query, _frames(candidates))["value"] == 1.25


def test_percentage_point_change_accepts_two_computed_ratios():
    candidates = [_c(1, 30), _c(2, 100), _c(3, 20), _c(4, 100)]
    program = {
        "schema_version": 2, "output_type": "percentage_point",
        "facts": {f"x{i}": _fact(i) for i in range(1, 5)}, "bindings": {},
        "root": {"op": "percentage_point_change", "args": [
            {"op": "divide", "args": [{"var": "x1"}, {"var": "x2"}]},
            {"op": "divide", "args": [{"var": "x3"}, {"var": "x4"}]},
        ]},
    }
    compiled = compile_program(
        program, candidates, {"output_type": "percentage_point", "unit_scale": 1}, "q",
    )
    assert run_code(compiled.query, _frames(candidates))["value"] == 10.0


def test_projection_when_filters_without_model_supplied_sentinel():
    candidates = [_c(1, -5), _c(2, 100), _c(3, 3), _c(4, 80),
                  _c(5, 8), _c(6, 60)]
    facts = {f"score{i}": _fact(i * 2 - 1) for i in range(1, 4)}
    facts.update({f"out{i}": _fact(i * 2) for i in range(1, 4)})
    items = []
    for i in range(1, 4):
        items.append({
            "when": {"op": "gt", "args": [{"var": f"score{i}"},
                                              {"literal": 0, "type": "number"}]},
            "score": {"var": f"score{i}"}, "result": {"var": f"out{i}"},
        })
    program = {"schema_version": 2, "output_type": "number", "facts": facts,
               "bindings": {}, "root": {"op": "argmin_project", "items": items}}
    compiled = compile_program(program, candidates,
                               {"output_type": "number", "unit_scale": 1}, "q")
    assert run_code(compiled.query, _frames(candidates))["value"] == 80.0
    assert "1e+99" not in compiled.query


def test_increase_and_decrease_percent_use_grounded_question_literal():
    candidate = _c(1, 200)
    for op, expected in (("increase_percent", 220.0), ("decrease_percent", 180.0)):
        program = {
            "schema_version": 2, "output_type": "number",
            "facts": {"base": _fact(1)}, "bindings": {},
            "root": {"op": op, "args": [
                {"var": "base"}, {"literal": 10, "type": "percent"},
            ]},
        }
        compiled = compile_program(
            program, [candidate], {"output_type": "number", "unit_scale": 1},
            "giả sử tăng hoặc giảm 10 phần trăm",
        )
        assert run_code(compiled.query, _frames([candidate]))["value"] == expected


def test_decimal_comma_literal_allows_following_sentence_comma():
    candidate = _c(1, 200)
    program = {
        "schema_version": 2,
        "output_type": "number",
        "facts": {"base": _fact(1)},
        "bindings": {},
        "root": {"op": "divide", "args": [
            {"var": "base"}, {"literal": 1.5, "type": "ratio"},
        ]},
    }
    compiled = compile_program(
        program, [candidate], {"output_type": "number", "unit_scale": 1},
        "lon hon 1,5, doanh nghiep duoc chon",
    )

