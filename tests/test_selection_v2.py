from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from vifinqa.codegen.executor import run_code
from vifinqa.codegen.selection_v2 import (
    IRValidationError,
    compile_program,
    evaluate_samples,
    parse_program,
)


def _candidate(index: int, value: float, *, unit_scale: float = 1.0,
               fact_year: int | None = 2024, scalar_label: str = "Doanh thu",
               var: str | None = None, row: int | None = None, col: int = 1,
               score: float = 80.0, fact_slot: str = "F1"):
    return SimpleNamespace(
        var=var or f"df{index}", row=row or index, col=col,
        label=scalar_label, col_name=str(fact_year or "Năm nay"),
        value=value, unit_scale=unit_scale, score=score, rescue=False,
        fact_year=fact_year, report_year=fact_year, fact_slot=fact_slot,
        fact_role="value", fact_metric=scalar_label, ticker="AAA",
        report_id=f"AAA_{fact_year}_{index}", table_pos=index, code="",
    )


def _dfs(candidates):
    out = {}
    for c in candidates:
        out[c.var] = pd.DataFrame([
            {"row": c.row, "col": c.col, "value": c.value,
             "unit_scale": c.unit_scale, "label": c.label,
             "col_name": c.col_name},
        ])
    return out


def _program(output_type, facts, root, bindings=None):
    return {"schema_version": 2, "output_type": output_type,
            "facts": facts, "bindings": bindings or {}, "root": root}


def _fact(ref, as_type="money", role="value"):
    return {"ref": ref, "as": as_type, "role": role}


def test_parse_nested_json_from_fence():
    raw = "before\n```json\n" + json.dumps(_program(
        "ratio", {"a": _fact(1)}, {"var": "a"},
    )) + "\n```\nafter"
    parsed = parse_program(raw)
    assert parsed["schema_version"] == 2
    assert parsed["root"] == {"var": "a"}


def test_ratio_to_percent_is_typed_scaled_and_replayable():
    candidates = [_candidate(1, 9, unit_scale=1e6),
                  _candidate(2, 100, unit_scale=1e6)]
    program = _program(
        "percent",
        {"profit": _fact(1, role="numerator"),
         "revenue": _fact(2, role="denominator")},
        {"op": "divide", "args": [{"var": "profit"}, {"var": "revenue"}]},
    )
    compiled = compile_program(program, candidates,
                               {"output_type": "percent", "unit_scale": 1},
                               "Tỷ trọng là bao nhiêu phần trăm?")
    result = run_code(compiled.query, _dfs(candidates))
    assert result["status"] == "ok"
    assert result["value"] == 9.0
    assert compiled.inferred_type == "ratio"
    assert compiled.referenced_indices == (1, 2)


def test_named_nested_bindings_are_compiled_once_with_lambdas():
    candidates = [_candidate(1, 120), _candidate(2, 100),
                  _candidate(3, 60), _candidate(4, 50)]
    program = _program(
        "percentage_point",
        {"p1": _fact(1), "r1": _fact(2),
         "p0": _fact(3), "r0": _fact(4)},
        {"op": "percentage_point_change",
         "args": [{"var": "m1"}, {"var": "m0"}]},
        bindings={
            "m1": {"op": "multiply", "args": [
                {"op": "divide", "args": [{"var": "p1"}, {"var": "r1"}]},
                {"literal": 100, "type": "ratio"},
            ]},
            "m0": {"op": "multiply", "args": [
                {"op": "divide", "args": [{"var": "p0"}, {"var": "r0"}]},
                {"literal": 100, "type": "ratio"},
            ]},
        },
    )
    # Ratio-valued bindings are valid inputs: the compiler converts their
    # difference to percentage points exactly once.
    compiled = compile_program(
        program, candidates,
                        {"output_type": "percentage_point", "unit_scale": 1},
                        "chênh lệch 100 điểm phần trăm")
    assert run_code(compiled.query, _dfs(candidates))["value"] == 0.0
    assert compiled.used_binding_names == ("m1", "m0")
    assert compiled.query.count("lambda _v2_") == 6


def test_percentage_point_from_percent_cells():
    candidates = [
        _candidate(1, 25, scalar_label="Tỷ lệ nợ xấu (%)"),
        _candidate(2, 20, scalar_label="Tỷ lệ nợ xấu (%)"),
    ]
    program = _program(
        "percentage_point",
        {"now": _fact(1, "percent"), "before": _fact(2, "percent")},
        {"op": "percentage_point_change",
         "args": [{"var": "now"}, {"var": "before"}]},
    )
    compiled = compile_program(program, candidates,
                               {"output_type": "percentage_point", "unit_scale": 1},
                               "chênh lệch bao nhiêu điểm phần trăm")
    assert run_code(compiled.query, _dfs(candidates))["value"] == 5.0


def test_count_true_uses_typed_comparisons_and_is_grounded():
    candidates = [_candidate(1, 5), _candidate(2, -3), _candidate(3, 7)]
    facts = {f"c{i}": _fact(i, role="filter") for i in range(1, 4)}
    tests = [
        {"op": "gt", "args": [{"var": f"c{i}"},
                                 {"literal": 0, "type": "number"}]}
        for i in range(1, 4)
    ]
    program = _program("count", facts, {"op": "count_true", "args": tests})
    compiled = compile_program(program, candidates,
                               {"output_type": "count", "unit_scale": 1},
                               "Có bao nhiêu công ty có CFO dương?")
    assert run_code(compiled.query, _dfs(candidates))["value"] == 2.0


def test_argmax_projects_a_year_grounded_by_each_score():
    candidates = [
        _candidate(1, 10, fact_year=2021, fact_slot="F1"),
        _candidate(2, 30, fact_year=2022, fact_slot="F2"),
        _candidate(3, 20, fact_year=2024, fact_slot="F3"),
    ]
    facts = {"v21": _fact(1), "v22": _fact(2), "v24": _fact(3)}
    root = {"op": "argmax_project", "items": [
        {"score": {"var": "v21"}, "result": {"year": 2021}},
        {"score": {"var": "v22"}, "result": {"year": 2022}},
        {"score": {"var": "v24"}, "result": {"year": 2024}},
    ]}
    compiled = compile_program(
        _program("year", facts, root), candidates,
        {"output_type": "year", "unit_scale": 1, "years": [2021, 2022, 2024]},
        "Năm nào cao nhất trong 2021, 2022, 2024?",
    )
    assert run_code(compiled.query, _dfs(candidates))["value"] == 2022.0


def test_argmin_can_project_a_different_metric():
    candidates = [
        _candidate(1, 2, fact_year=2021), _candidate(2, 9, fact_year=2021),
        _candidate(3, 1, fact_year=2022), _candidate(4, 7, fact_year=2022),
    ]
    facts = {"rank21": _fact(1), "out21": _fact(2),
             "rank22": _fact(3), "out22": _fact(4)}
    root = {"op": "argmin_project", "items": [
        {"score": {"var": "rank21"}, "result": {"var": "out21"}},
        {"score": {"var": "rank22"}, "result": {"var": "out22"}},
    ]}
    compiled = compile_program(
        _program("number", facts, root), candidates,
        {"output_type": "number", "unit_scale": 1}, "metric sau năm thấp nhất",
    )
    assert run_code(compiled.query, _dfs(candidates))["value"] == 7.0


def test_median_filter_and_conditional_average():
    candidates = [_candidate(1, 1), _candidate(2, 2),
                  _candidate(3, 8), _candidate(4, 9)]
    facts = {f"v{i}": _fact(i) for i in range(1, 5)}
    vars_ = [{"var": f"v{i}"} for i in range(1, 5)]
    bindings = {"med": {"op": "median", "args": vars_}}
    conditions = [
        {"op": "gt", "args": [{"var": f"v{i}"}, {"var": "med"}]}
        for i in range(1, 5)
    ]
    root = {"op": "count_true", "args": conditions}
    compiled = compile_program(
        _program("count", facts, root, bindings), candidates,
        {"output_type": "count", "unit_scale": 1}, "bao nhiêu giá trị trên trung vị",
    )
    assert run_code(compiled.query, _dfs(candidates))["value"] == 2.0


def test_number_root_converts_requested_unit_once():
    candidates = [_candidate(1, 250, unit_scale=1e6)]
    compiled = compile_program(
        _program("number", {"amount": _fact(1)}, {"var": "amount"}),
        candidates, {"output_type": "number", "unit_scale": 1e9},
        "bao nhiêu tỷ đồng",
    )
    assert run_code(compiled.query, _dfs(candidates))["value"] == 0.25


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p.update(output_type="ratio"), "does not match route"),
    (lambda p: p["root"].update(ref=1), "candidate refs are allowed only"),
])
def test_fail_closed_schema_and_grounding(mutation, match):
    candidates = [_candidate(1, 1)]
    program = _program("number", {"x": _fact(1)}, {"var": "x"})
    if "ref" in program["root"]:
        del program["root"]["var"]
    mutation(program)
    if "ref" in program["root"]:
        program["root"].pop("var", None)
    with pytest.raises(IRValidationError, match=match):
        compile_program(program, candidates,
                        {"output_type": "number", "unit_scale": 1}, "q")


def test_direct_ref_outside_facts_is_rejected():
    candidates = [_candidate(1, 1)]
    program = _program("number", {"x": _fact(1)}, {"ref": 1})
    with pytest.raises(IRValidationError, match="only inside top-level facts"):
        compile_program(program, candidates,
                        {"output_type": "number", "unit_scale": 1}, "q")


def test_distinct_candidate_indices_aliasing_same_cell_are_rejected():
    candidates = [_candidate(1, 1, var="df1", row=5),
                  _candidate(2, 2, var="df1", row=5)]
    program = _program(
        "number", {"a": _fact(1), "b": _fact(2)},
        {"op": "sum", "args": [{"var": "a"}, {"var": "b"}]},
    )
    with pytest.raises(IRValidationError, match="alias the same stable cell"):
        compile_program(program, candidates,
                        {"output_type": "number", "unit_scale": 1}, "q")


def test_unused_fact_is_rejected():
    candidates = [_candidate(1, 1), _candidate(2, 2)]
    program = _program("number", {"a": _fact(1), "b": _fact(2)}, {"var": "a"})
    with pytest.raises(IRValidationError, match="unused definitions"):
        compile_program(program, candidates,
                        {"output_type": "number", "unit_scale": 1}, "q")


def test_binding_cycle_is_rejected():
    candidates = [_candidate(1, 1)]
    program = _program(
        "number", {"x": _fact(1)}, {"var": "a"},
        {"a": {"op": "add", "args": [{"var": "x"}, {"var": "b"}]},
         "b": {"op": "add", "args": [{"var": "x"}, {"var": "a"}]}},
    )
    with pytest.raises(IRValidationError, match="binding cycle"):
        compile_program(program, candidates,
                        {"output_type": "number", "unit_scale": 1}, "q")


def test_ungrounded_literal_is_rejected_but_question_literal_is_allowed():
    candidates = [_candidate(1, 20)]
    program = _program(
        "count", {"x": _fact(1, "number")},
        {"op": "count_true", "args": [
            {"op": "gt", "args": [{"var": "x"},
                                     {"literal": 17, "type": "number"}]},
        ]},
    )
    with pytest.raises(IRValidationError, match="present in the question"):
        compile_program(program, candidates,
                        {"output_type": "count", "unit_scale": 1}, "cao hơn ngưỡng")
    compiled = compile_program(program, candidates,
                               {"output_type": "count", "unit_scale": 1},
                               "cao hơn ngưỡng 17")
    assert run_code(compiled.query, _dfs(candidates))["value"] == 1.0


def test_evaluate_samples_accepts_first_valid_and_traces_later_sample():
    candidates = [_candidate(1, 5)]
    valid = _program("number", {"x": _fact(1)}, {"var": "x"})

    def execute(query):
        return run_code(query, _dfs(candidates))

    decision, trace = evaluate_samples(
        ["not json", json.dumps(valid), json.dumps(valid)], candidates,
        {"output_type": "number", "unit_scale": 1}, "q", execute,
    )
    assert decision is not None and decision.answer == 5.0
    assert trace["schema_version"] == 2
    assert trace["outcome"] == "accepted"
    assert trace["accepted_attempt"] == 2
    assert trace["rejection_counts"] == {"parse_error": 1}
    assert trace["attempts"][2]["reason_code"] == "not_evaluated_after_acceptance"


def test_evaluate_samples_handles_model_none_and_no_candidates():
    none = {"schema_version": 2, "output_type": "number",
            "facts": {}, "bindings": {}, "root": {"op": "none"}}
    decision, trace = evaluate_samples(
        [json.dumps(none)], [_candidate(1, 1)],
        {"output_type": "number", "unit_scale": 1}, "q", lambda _q: {},
    )
    assert decision is None
    assert trace["rejection_counts"] == {"model_none": 1}

    decision, trace = evaluate_samples(
        [json.dumps(none)], [], {"output_type": "number"}, "q", lambda _q: {},
    )
    assert decision is None and trace["outcome"] == "no_candidates"

