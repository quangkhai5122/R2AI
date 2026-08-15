from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vifinqa.codegen.selection_v2 import IRValidationError, compile_program, evaluate_samples
from vifinqa.extraction.report_parser import detect_unit
from vifinqa.extraction.unit_policy import (
    RESOLUTION_OVERRIDE,
    RESOLUTION_STORED,
    has_terminal_bare_vnd,
    resolve_stored_table_unit,
)
from vifinqa.retrieval.shortlist import Candidate, _attach_metadata


def _program() -> dict:
    return {
        "schema_version": 2,
        "output_type": "number",
        "facts": {"cfo": {"ref": 1, "as": "money", "role": "value"}},
        "bindings": {},
        "root": {"var": "cfo"},
    }


def _compiler_candidate(**overrides):
    values = {
        "var": "df1",
        "row": 16,
        "col": 3,
        "label": "Lưu chuyển tiền thuần từ hoạt động kinh doanh",
        "col_name": "Năm nay",
        "value": 932_670_196_340.0,
        "unit_scale": 1e9,
        "unit_original_scale": 1e9,
        "unit_source": "sticky",
        "unit_effective_source": "sticky",
        "unit_resolution": RESOLUTION_STORED,
        "unit_context_terminal_vnd": True,
        "unit_context_sha256": "a" * 64,
        "score": 90.0,
        "rescue": False,
        "fact_year": 2025,
        "report_year": 2025,
        "fact_slot": "F1",
        "fact_role": "value",
        "fact_metric": "dòng tiền thuần từ hoạt động kinh doanh",
        "ticker": "GEG",
        "report_id": "GEG_financial_statements_2025_consolidated",
        "table_pos": 11,
        "code": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_terminal_bare_vnd_policy_is_narrow():
    statement = "B03-DN/HN cho năm tài chính kết thúc ngày 31 tháng 12 năm 2025 VND"
    assert has_terminal_bare_vnd(statement)
    assert detect_unit(statement) == (1.0, "terminal_vnd")

    for multiplied in (
        "Unit: billion VND",
        "Đơn vị: triệu VND",
        "Đơn vị tính: nghìn VND",
        "Amounts in thousand VND",
    ):
        assert not has_terminal_bare_vnd(multiplied)

    assert detect_unit("Đơn vị tính: tỷ VND") == (1e9, "explicit")


def test_runtime_resolution_overrides_only_sticky_metadata():
    context = "B03-DN/HN CASH FLOW STATEMENT for year ended 2025 VND"
    repaired = resolve_stored_table_unit(1e9, "sticky", context)
    assert repaired.changed
    assert repaired.stored_scale == 1e9
    assert repaired.effective_scale == 1.0
    assert repaired.reason == RESOLUTION_OVERRIDE

    explicit = resolve_stored_table_unit(1e9, "explicit", context)
    assert not explicit.changed
    assert explicit.effective_scale == 1e9
    assert explicit.reason == RESOLUTION_STORED


def test_shortlist_attaches_auditable_effective_unit():
    candidate = Candidate(
        var="df1",
        report_id="GEG_financial_statements_2025_consolidated",
        table_pos=11,
        row=16,
        label="Lưu chuyển tiền thuần từ hoạt động kinh doanh",
        code="",
        col=3,
        col_name="Năm nay",
        value=932_670_196_340.0,
        unit_scale=1e9,
        score=90.0,
        lexical=90.0,
        semantic=0.0,
    )
    tables = [{
        "var": "df1",
        "report_id": candidate.report_id,
        "table_pos": 11,
        "ticker": "GEG",
        "report_year": 2025,
        "unit_scale": 1e9,
        "unit_source": "sticky",
        "context": "B03-DN/HN năm 2025 VND",
    }]
    resolved = _attach_metadata([candidate], tables)[0]
    assert resolved.unit_original_scale == 1e9
    assert resolved.unit_scale == 1.0
    assert resolved.unit_source == "sticky"
    assert resolved.unit_effective_source == "terminal_vnd"
    assert resolved.unit_resolution == RESOLUTION_OVERRIDE
    assert resolved.unit_context_terminal_vnd
    assert len(resolved.unit_context_sha256) == 64


def test_compiler_fails_closed_on_unresolved_sticky_terminal_vnd():
    with pytest.raises(IRValidationError, match="keeps sticky"):
        compile_program(
            _program(),
            [_compiler_candidate()],
            {"output_type": "number", "unit_scale": 1.0},
            "dòng tiền thuần từ hoạt động kinh doanh",
        )


def test_compiler_accepts_and_records_valid_terminal_vnd_override():
    candidate = _compiler_candidate(
        unit_scale=1.0,
        unit_effective_source="terminal_vnd",
        unit_resolution=RESOLUTION_OVERRIDE,
    )
    compiled = compile_program(
        _program(),
        [candidate],
        {"output_type": "number", "unit_scale": 1.0},
        "dòng tiền thuần từ hoạt động kinh doanh",
    )
    assert len(compiled.unit_provenance) == 1
    unit = compiled.unit_provenance[0]
    assert unit.candidate_index == 1
    assert unit.stored_scale == 1e9
    assert unit.effective_scale == 1.0
    assert unit.resolution == RESOLUTION_OVERRIDE
    assert "* 1)" in compiled.query


def test_evaluation_trace_exposes_unit_provenance():
    candidate = _compiler_candidate(
        unit_scale=1.0,
        unit_effective_source="terminal_vnd",
        unit_resolution=RESOLUTION_OVERRIDE,
    )
    decision, trace = evaluate_samples(
        [json.dumps(_program())],
        [candidate],
        {"output_type": "number", "unit_scale": 1.0},
        "dòng tiền thuần từ hoạt động kinh doanh",
        lambda _query: {"status": "ok", "value": candidate.value},
    )
    assert decision is not None
    provenance = trace["attempts"][0]["program"]["unit_provenance"]
    assert provenance == [{
        "candidate_index": 1, "stored_scale": 1e9, "effective_scale": 1.0,
        "stored_source": "sticky", "effective_source": "terminal_vnd",
        "resolution": RESOLUTION_OVERRIDE, "terminal_bare_vnd": True,
        "context_sha256": "a" * 64,
    }]
